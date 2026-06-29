#!/usr/bin/env python3
"""
I-JEPA self-supervised encoder training.
Replaces teacher-student distillation with JEPA prediction in universal space.

Fixed version: handles latent [B,L,H] shape, ModelWrapper args, U pooling.
"""

import sys, os, argparse, math, torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, "/heterogeneous-latent-mas-main")
from models import ModelWrapper
from methods.vision_latent_mas_codec_new import LatentToUniversalEncoder
from train_vision_latent_mas_codec_new import (
    _load_anchor_texts, _load_mm_processor_or_tokenizer, _infer_special_token_ids,
    _make_dummy_image, _get_tokenizer_like, _infer_hidden_size, _resolve_dummy_image_specs,
    _parse_model_list,
)

class _DummyArgs:
    latent_space_realign = 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="")
    p.add_argument("--agent_model_names", type=str, default="")
    p.add_argument("--vision_codec_path", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--latent_steps", type=int, default=1024)
    p.add_argument("--vision_codec_dim", type=int, default=256)
    p.add_argument("--vision_codec_tokens", type=int, default=1024)
    p.add_argument("--vision_codec_img_tokens", type=int, default=256)
    p.add_argument("--vision_codec_heads", type=int, default=8)
    p.add_argument("--vision_codec_layers", type=int, default=4)
    p.add_argument("--vision_codec_dropout", type=float, default=0.)
    p.add_argument("--vision_codec_train_steps", type=int, default=200)
    p.add_argument("--vision_codec_train_batch_size", type=int, default=8)
    p.add_argument("--vision_codec_train_lr", type=float, default=5e-4)
    p.add_argument("--vision_codec_rollout_noise_std", type=float, default=0.05)
    p.add_argument("--ijepa_pred_loss_w", type=float, default=1.0)
    p.add_argument("--ijepa_ac_loss_w", type=float, default=0.1)
    p.add_argument("--vision_codec_log_every", type=int, default=10)
    p.add_argument("--vision_codec_checkpoint_every", type=int, default=0)
    p.add_argument("--vision_codec_anchor_texts_path", type=str, default="")
    cfg = p.parse_args()

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    model_names = _parse_model_list(cfg.agent_model_names) if cfg.agent_model_names else \
                  ([] if not cfg.model_name else [cfg.model_name])
    if not model_names:
        raise ValueError("Provide --model_name or --agent_model_names")
    name = model_names[0]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Model: {name}")

    wrapper = ModelWrapper(name, device, use_vllm=False, args=_DummyArgs())
    processor = _load_mm_processor_or_tokenizer(name, wrapper)
    special_ids = _infer_special_token_ids(processor)
    H = _infer_hidden_size(wrapper)
    D = int(cfg.vision_codec_dim)
    K = int(cfg.vision_codec_tokens)

    wrapper.model.eval()
    for p in wrapper.model.parameters():
        p.requires_grad_(False)

    enc = LatentToUniversalEncoder(
        h_in=H, d_univ=D, k_univ=K,
        n_heads=int(cfg.vision_codec_heads), n_layers=int(cfg.vision_codec_layers),
        dropout=float(cfg.vision_codec_dropout),
    ).to(device)
    enc.train()

    predictor = nn.Sequential(
        nn.Linear(D, 1024), nn.ReLU(),
        nn.Linear(1024, 1024), nn.ReLU(),
        nn.Linear(1024, D),
    ).to(device)
    predictor.train()

    opt = torch.optim.AdamW(list(enc.parameters()) + list(predictor.parameters()),
                            lr=float(cfg.vision_codec_train_lr))

    anchor_texts = _load_anchor_texts(cfg.vision_codec_anchor_texts_path)
    print(f"Loaded {len(anchor_texts)} anchor texts")

    steps = int(cfg.vision_codec_train_steps)
    bs = int(cfg.vision_codec_train_batch_size)
    latent_steps = int(cfg.latent_steps)
    noise_std = float(cfg.vision_codec_rollout_noise_std)
    pred_w = float(cfg.ijepa_pred_loss_w)
    ac_w = float(cfg.ijepa_ac_loss_w)
    log_every = int(cfg.vision_codec_log_every)
    ckpt_every = int(cfg.vision_codec_checkpoint_every)

    _step = 0
    pbar = tqdm(total=steps, desc="[I-JEPA]")

    while _step < steps:
        batch = [anchor_texts[(_step * bs + i) % len(anchor_texts)] for i in range(bs)]
        _step += 1

        msgs = [[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Message:\n{t}\n\nAcknowledge."},
        ] for t in batch]
        _, ids, mask, _ = wrapper.prepare_chat_batch(msgs, add_generation_prompt=True)
        ids, mask = ids.to(device), mask.to(device)

        with torch.no_grad():
            _, lat_a = wrapper.generate_latent_batch(ids, mask, latent_steps=latent_steps, return_latent_embeds=True)
            # lat_a: [B, latent_steps, H]
            noise = torch.randn_like(lat_a) * noise_std * lat_a.std(dim=-1, keepdim=True)
            lat_b = lat_a + noise

        U_a = enc(lat_a.float().to(device))  # [B, latent_steps, D]
        with torch.no_grad():
            U_b = enc(lat_b.float().to(device))

        # Pool over latent steps: [B, D]
        U_a_pooled = U_a.mean(dim=1)
        U_b_pooled = U_b.mean(dim=1)

        pred = predictor(U_a_pooled)
        loss_pred = F.mse_loss(pred, U_b_pooled.detach())

        # Anti-collapse loss
        U_2d = U_a.reshape(-1, D)
        std = U_2d.std(dim=0, unbiased=False)
        loss_var = F.relu(1.0 - std).mean()
        Uc = U_2d - U_2d.mean(dim=0, keepdim=True)
        cov = (Uc.T @ Uc) / (U_2d.shape[0] - 1)
        mask_cov = torch.eye(D, device=device, dtype=torch.bool)
        loss_cov = cov[~mask_cov].pow(2).sum() / D

        loss = pred_w * loss_pred + ac_w * (loss_var + loss_cov)

        if not torch.isfinite(loss):
            print(f"[I-JEPA] NaN at step {_step}, skip", flush=True)
            opt.zero_grad(set_to_none=True)
            pbar.update(1)
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # Save intermediate checkpoint
        if ckpt_every > 0 and _step % ckpt_every == 0:
            base = cfg.vision_codec_path.replace(".pt", f"_step{_step}.pt")
            torch.save({
                "encoders": {name: {k: v.detach().float().cpu() for k, v in enc.state_dict().items()}},
                "_steps": _step,
                "comment": "I-JEPA encoder intermediate",
            }, base)
            print(f"[I-JEPA] checkpoint saved: {base}", flush=True)

        if _step % log_every == 0 or _step == steps:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            print(f"[I-JEPA] step={_step}/{steps} loss={loss.item():.4f} pred={loss_pred.item():.4f} var={loss_var.item():.4f} cov={loss_cov.item():.4f}", flush=True)
        pbar.update(1)

    pbar.close()

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "encoders": {name: {k: v.detach().float().cpu() for k, v in enc.state_dict().items()}},
        "_steps": _step,
        "comment": "I-JEPA encoder only (no decoder)",
    }, cfg.vision_codec_path)
    print(f"Saved: {cfg.vision_codec_path}")


if __name__ == "__main__":
    main()
