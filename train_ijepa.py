#!/usr/bin/env python3
"""
I-JEPA self-supervised variant of Vision Wormhole codec training.
Replaces teacher-student distillation with JEPA prediction in universal space.

Key difference from original:
  Instead of: teacher(text) → KL(student_logits, teacher_logits)
  JEPA does:  anchor + noise_a → U_a → predictor → pred_U_b
              anchor + noise_b → U_b (stop_grad) ─── MSE(pred_U_b, U_b) ┘

Ref: I-JEPA (Assran et al., CVPR 2023); VICReg (Bardes et al., ICLR 2022)
"""

import sys, os, argparse, math, random, copy
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, "/heterogeneous-latent-mas-main")
from train_vision_latent_mas_codec_new import (
    _SpecialTokenIds, _parse_model_list, _load_anchor_texts,
    _load_mm_processor_or_tokenizer, _infer_special_token_ids, _make_dummy_image,
    _resolve_dummy_image_specs, _get_tokenizer_like, _infer_hidden_size,
    _extract_dummy_image_tokens, _ddp_rank, _ddp_barrier, _ddp_all_true,
    _derive_partial_ckpt_path, _is_internvl_wrapper, _is_minicpm_wrapper,
    _get_text_backbone, _build_mm_user_content,
)
from models import ModelWrapper
from methods.vision_latent_mas_codec_new import LatentToUniversalEncoder


def _anti_collapse_loss(U: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if U.dim() == 3:
        U = U.reshape(-1, U.shape[-1])
    D = U.shape[-1]
    std = U.std(dim=0, unbiased=False)
    loss_var = F.relu(1.0 - std).mean()
    U_centered = U - U.mean(dim=0, keepdim=True)
    cov = (U_centered.T @ U_centered) / (U.shape[0] - 1)
    mask = torch.eye(D, device=U.device, dtype=torch.bool)
    loss_cov = cov[~mask].pow(2).sum() / D
    return loss_var, loss_cov


class JEPAPredictor(nn.Module):
    """Lightweight MLP predictor: universal space → universal space."""
    def __init__(self, dim: int, hidden_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        if U.dim() == 3:
            B, K, D = U.shape
            U = U.reshape(-1, D)
            out = self.net(U)
            return out.reshape(B, K, D)
        return self.net(U)


def _train_one_model_ijepa(
    *,
    wrapper: ModelWrapper,
    processor,
    special_ids,
    dummy_imgs: List[Any],
    anchor_texts: List[str],
    cfg: argparse.Namespace,
) -> Tuple[Dict[str, torch.Tensor], int]:
    """
    I-JEPA training: encode two noise views of same anchor → predict one from the other.
    No text teacher, no KL, no stats loss. Only encoder is trained (no decoder needed).
    """
    device = wrapper.model.device
    tok = _get_tokenizer_like(processor) or getattr(wrapper, "tokenizer", None)

    codec_dtype = torch.float32
    if _ddp_rank() == 0:
        print(f"[codec-train-ijepa] codec_dtype={str(codec_dtype)}", flush=True)
        print(f"[codec-train-ijepa] pred_loss_w={float(cfg.ijepa_pred_loss_w)} ac_loss_w={float(cfg.ijepa_ac_loss_w)}", flush=True)

    wrapper.model.eval()
    for p in wrapper.model.parameters():
        p.requires_grad_(False)

    H = _infer_hidden_size(wrapper)
    D = int(cfg.vision_codec_dim)
    K = int(cfg.vision_codec_tokens)

    enc = LatentToUniversalEncoder(
        h_in=H, d_univ=D, k_univ=K,
        n_heads=int(cfg.vision_codec_heads), n_layers=int(cfg.vision_codec_layers),
        dropout=float(cfg.vision_codec_dropout),
    ).to(device=device, dtype=codec_dtype)

    predictor = JEPAPredictor(dim=D).to(device=device, dtype=codec_dtype)

    enc.train(); predictor.train()
    opt = torch.optim.AdamW(list(enc.parameters()) + list(predictor.parameters()),
                            lr=float(cfg.vision_codec_train_lr))

    if not dummy_imgs:
        raise RuntimeError("dummy_imgs must be non-empty")
    dummy_img_primary = dummy_imgs[0]
    dummy_tokens = _extract_dummy_image_tokens(wrapper, processor, special_ids, dummy_imgs)

    steps = int(cfg.vision_codec_train_steps)
    bs = int(cfg.vision_codec_train_batch_size)
    latent_steps = int(cfg.latent_steps)
    noise_std = float(cfg.vision_codec_rollout_noise_std) if float(cfg.vision_codec_rollout_noise_std) > 0 else 0.05
    pred_w = float(cfg.ijepa_pred_loss_w)
    ac_w = float(cfg.ijepa_ac_loss_w)
    log_every = int(cfg.vision_codec_log_every)

    _step = 0
    pbar = tqdm(total=steps, desc=f"[codec-train-ijepa] {wrapper.model_name}", position=0, leave=True)

    while _step < steps:
        batch_texts = []
        for i in range(bs):
            idx = (_step * bs + i) % len(anchor_texts)
            batch_texts.append(anchor_texts[idx])
        _step += 1

        # Build chat messages
        msgs = [[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Message:\n{t}\n\nAcknowledge."},
        ] for t in batch_texts]

        _, ids, mask, _ = wrapper.prepare_chat_batch(msgs, add_generation_prompt=True)
        ids = ids.to(device); mask = mask.to(device)

        with torch.no_grad():
            # View A: with noise
            _, lat_a = wrapper.generate_latent_batch(ids, attention_mask=mask,
                                                      latent_steps=latent_steps, return_latent_embeds=True)
            # View B: with different noise seed (add noise to latents)
            lat_b = lat_a.clone()
            for _lat in [lat_b]:
                _lat.add_(torch.randn_like(_lat) * noise_std * _lat.std(dim=-1, keepdim=True))

        // lat: [B, L, H]
        lat_flat_a = lat_a.to(device=device, dtype=codec_dtype)
        lat_flat_b = lat_b.to(device=device, dtype=codec_dtype)
        lat_a_flat = lat_a.reshape(B * R, L_data, H_dim).to(device=device, dtype=codec_dtype)
        lat_b_flat = lat_b.reshape(B * R, L_data, H_dim).to(device=device, dtype=codec_dtype)

        U_a = torch.nan_to_num(enc(lat_a_flat), nan=0.0, posinf=1e4, neginf=-1e4)
        with torch.no_grad():
            U_b = torch.nan_to_num(enc(lat_b_flat), nan=0.0, posinf=1e4, neginf=-1e4)

        pred_U_b = predictor(U_a)

        loss_pred = F.mse_loss(pred_U_b, U_b) if pred_w > 0 else torch.tensor(0.0, device=device)
        loss_var, loss_cov = _anti_collapse_loss(U_a) if ac_w > 0 else (torch.tensor(0.0, device=device), torch.tensor(0.0, device=device))
        loss_ac = loss_var + loss_cov
        loss = pred_w * loss_pred + ac_w * loss_ac

        if not torch.isfinite(loss):
            print(f"[ijepa] Non-finite loss at step {_step}. Skipping.", flush=True)
            opt.zero_grad(set_to_none=True)
            pbar.update(1)
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if _ddp_rank() == 0 and (_step % log_every == 0 or _step == steps):
            pbar.set_postfix(loss=float(loss.detach().cpu()))
            print(
                f"[codec-train-ijepa][{wrapper.model_name}] step={_step}/{steps} "
                f"loss={float(loss.detach().cpu()):.6f} pred={float(loss_pred.detach().cpu()):.6f} "
                f"ac_var={float(loss_var.detach().cpu() if isinstance(loss_var, torch.Tensor) else loss_var):.6f} "
                f"ac_cov={float(loss_cov.detach().cpu() if isinstance(loss_cov, torch.Tensor) else loss_cov):.6f}",
                flush=True,
            )
        pbar.update(1)

    pbar.close()

    enc_sd = enc.state_dict()
    return {k: v.detach().float().cpu() for k, v in enc_sd.items()}, dummy_tokens.shape[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="")
    p.add_argument("--agent_model_names", type=str, default="")
    p.add_argument("--vision_codec_path", type=str, required=True)
    p.add_argument("--vision_codec_partial_ckpt_path", type=str, default="")
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
    p.add_argument("--vision_codec_dummy_image_count", type=int, default=1)
    p.add_argument("--vision_codec_dummy_image_size", type=int, default=224)
    p.add_argument("--vision_codec_anchor_texts_path", type=str, default="")
    p.add_argument("--vision_codec_skip_alignment_if_single", type=int, default=0)
    p.add_argument("--vision_codec_save_per_model", type=int, default=0)
    p.add_argument("--vision_codec_ref_idx", type=int, default=0)
    p.add_argument("--vision_codec_gate_init_bias", type=float, default=-4.0)
    cfg = p.parse_args()

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    model_names = _parse_model_list(cfg.agent_model_names) if cfg.agent_model_names else \
                  ([] if not cfg.model_name else [cfg.model_name])
    if not model_names:
        raise ValueError("Provide --model_name or --agent_model_names")
    model_names = model_names[:1]  # I-JEPA: single model only

    anchor_texts = _load_anchor_texts(cfg.vision_codec_anchor_texts_path)
    print(f"Loaded {len(anchor_texts)} anchor texts.", flush=True)

    dummy_specs = _resolve_dummy_image_specs(model_names, cfg)
    encoders = {}
    dummy_token_lens = {}

    name = model_names[0]
    print(f"\n{'='*30}\nTraining I-JEPA codec for: {name}\n{'='*30}", flush=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class _DA: latent_space_realign = 0
    wrapper = ModelWrapper(name, device, use_vllm=False, args=_DA())
    processor = _load_mm_processor_or_tokenizer(name, wrapper)
    special_ids = _infer_special_token_ids(processor)
    spec = dummy_specs[name]
    dummy_imgs = [_make_dummy_image(spec.size) for _ in range(spec.count)]

    enc_sd, dlen = _train_one_model_ijepa(
        wrapper=wrapper, processor=processor, special_ids=special_ids,
        dummy_imgs=dummy_imgs, anchor_texts=anchor_texts, cfg=cfg,
    )
    encoders[name] = enc_sd
    dummy_token_lens[name] = dlen

    os.makedirs(os.path.dirname(cfg.vision_codec_path) or ".", exist_ok=True)
    torch.save({"encoders": encoders, "dummy_token_lens": dummy_token_lens}, cfg.vision_codec_path)
    print(f"[ijepa] Saved checkpoint: {cfg.vision_codec_path}", flush=True)


if __name__ == "__main__":
    main()
