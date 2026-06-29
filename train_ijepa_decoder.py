#!/usr/bin/env python3
"""
Stage 2: I-JEPA decoder training.
Freeze I-JEPA encoder, train decoder via teacher-student distillation.
Same training loop as M0 but encoder is pretrained & frozen.

Usage:
  python train_ijepa_decoder.py \
    --model_name /heterogeneous-latent-mas-main/model_download/Qwen3-VL-2B-Thinking \
    --ijepa_encoder_path checkpoints/codec_qwen3_ijepa.pt \
    --vision_codec_path checkpoints/codec_qwen3_ijepa_full.pt \
    --vision_codec_anchor_texts_path data/vision_codec_anchor_text/mixed_cose_ocr_prm800k_small.jsonl \
    ... (same params as M0)
"""
import sys, os, argparse, torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, "/heterogeneous-latent-mas-main")
from models import ModelWrapper
from methods.vision_latent_mas_codec_new import LatentToUniversalEncoder, UniversalToVisionDecoder
from train_vision_latent_mas_codec_new import (
    _SpecialTokenIds, _load_anchor_texts, _load_mm_processor_or_tokenizer,
    _infer_special_token_ids, _make_dummy_image, _resolve_dummy_image_specs,
    _get_tokenizer_like, _infer_hidden_size, _extract_dummy_image_tokens,
    _build_mm_user_content, _processor_encode_multimodal, _maybe_to_device,
    _find_image_positions, _resample_tokens, _compute_kl_loss,
    _ddp_rank, _ddp_barrier, _ddp_all_true, _ddp_is_active, _derive_partial_ckpt_path,
    _parse_model_list, _get_text_backbone,
)
import numpy as np

class _DA: latent_space_realign = 0


def _train_decoder(
    wrapper, processor, special_ids, dummy_imgs, anchor_texts, cfg, enc_sd, enc_name
):
    device = wrapper.model.device
    tok = _get_tokenizer_like(processor) or getattr(wrapper, "tokenizer", None)
    if tok is None:
        raise RuntimeError("No tokenizer")

    if _ddp_rank() == 0:
        print(f"[ijepa-decoder] Loading I-JEPA encoder from checkpoint", flush=True)

    H = _infer_hidden_size(wrapper)
    D = int(cfg.vision_codec_dim)
    K = int(cfg.vision_codec_tokens)

    enc = LatentToUniversalEncoder(
        h_in=H, d_univ=D, k_univ=K,
        n_heads=int(cfg.vision_codec_heads), n_layers=int(cfg.vision_codec_layers),
        dropout=float(cfg.vision_codec_dropout),
    ).to(device=device, dtype=torch.float32)
    enc.load_state_dict(enc_sd, strict=False)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    if _ddp_rank() == 0:
        print(f"[ijepa-decoder] Encoder frozen ({sum(p.numel() for p in enc.parameters())} params)", flush=True)

    dec = UniversalToVisionDecoder(
        d_univ=D, h_out=H, k_img=int(cfg.vision_codec_img_tokens),
        n_heads=int(cfg.vision_codec_heads), n_layers=int(cfg.vision_codec_layers),
        dropout=float(cfg.vision_codec_dropout),
        gate_init_bias=float(cfg.vision_codec_gate_init_bias),
    ).to(device=device, dtype=torch.float32)
    dec.train()
    if _ddp_rank() == 0:
        print(f"[ijepa-decoder] Decoder initialized ({sum(p.numel() for p in dec.parameters())} params)", flush=True)

    opt = torch.optim.AdamW(dec.parameters(), lr=float(cfg.vision_codec_train_lr))

    wrapper.model.eval()
    for p in wrapper.model.parameters():
        p.requires_grad_(False)
    text_model = _get_text_backbone(wrapper)

    dummy_img_primary = dummy_imgs[0]
    dummy_tokens = _extract_dummy_image_tokens(wrapper, processor, special_ids, dummy_imgs)
    dummy_rms = dummy_tokens.pow(2).mean().sqrt().clamp_min(1e-6)

    steps = int(cfg.vision_codec_train_steps)
    bs = int(cfg.vision_codec_train_batch_size)
    latent_steps = int(cfg.latent_steps)
    kl_mode = str(getattr(cfg, "vision_codec_kl_mode", "auto")).strip().lower()
    kl_topk = int(getattr(cfg, "vision_codec_kl_topk", 0))
    kl_logit_clip = float(cfg.vision_codec_kl_logit_clip)
    inj_clip = float(cfg.vision_codec_inj_clip)
    l_mse = float(cfg.vision_codec_loss_mse)
    l_kl = float(cfg.vision_codec_loss_kl)
    l_stats = float(cfg.vision_codec_loss_stats)
    log_every = int(getattr(cfg, "vision_codec_log_every", 10))

    _step = 0
    pbar = tqdm(total=steps, desc=f"[ijepa-decoder] {wrapper.model_name}", position=0, leave=True)

    while _step < steps:
        batch_texts = [anchor_texts[(_step * bs + i) % len(anchor_texts)] for i in range(bs)]
        _step += 1

        teacher_msgs = [[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Message:\n{t}\n\nAcknowledge."},
        ] for t in batch_texts]
        _, t_ids, t_mask, _ = wrapper.prepare_chat_batch(teacher_msgs, add_generation_prompt=True)
        t_ids = t_ids.to(device); t_mask = t_mask.to(device)

        with torch.no_grad():
            out_t = text_model(input_ids=t_ids, attention_mask=t_mask,
                               use_cache=True, output_hidden_states=True, return_dict=True)
            last_pos = (t_mask.sum(dim=1) - 1).long()
            teacher_h = out_t.hidden_states[-1][torch.arange(bs, device=device), last_pos, :]
            teacher_logits = out_t.logits[torch.arange(bs, device=device), last_pos, :]

            _, lat = wrapper.generate_latent_batch(t_ids, t_mask, latent_steps=latent_steps, return_latent_embeds=True)

        # lat: [B, L, H] or [B, R, L, H] depending on mc_rollouts
        if lat.dim() == 3:
            lat_flat = lat.reshape(-1, lat.shape[-1]).unsqueeze(1)  # [B*L, 1, H]
            B_eff, R_val = lat.shape[0], 1
        else:
            B_eff, R_val = lat.shape[0], lat.shape[1]
            lat_flat = lat.reshape(B_eff * R_val, lat.shape[2], lat.shape[3])
        lat_flat = lat_flat.to(device=device, dtype=torch.float32)
        U_flat = torch.nan_to_num(enc(lat_flat), nan=0.0, posinf=1e4, neginf=-1e4)
        delta_flat, gate_flat = dec(U_flat)
        delta_flat = torch.nan_to_num(delta_flat, nan=0.0, posinf=1e4, neginf=-1e4)
        gate_flat = torch.nan_to_num(gate_flat, nan=0.0, posinf=1.0, neginf=0.0)
        delta = delta_flat.reshape(B_eff, R_val, delta_flat.shape[1], delta_flat.shape[-1])
        gate = gate_flat.reshape(B_eff, R_val, 1, 1)
        inj = (gate * delta).mean(dim=1)
        inj = torch.nan_to_num(inj, nan=0.0, posinf=1e4, neginf=-1e4)
        if inj_clip > 0:
            inj = inj.clamp(min=-inj_clip, max=inj_clip)

        student_msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": _build_mm_user_content(
                num_images=len(dummy_imgs), text="Message:\n\nAcknowledge.")},
        ]
        if hasattr(tok, "apply_chat_template"):
            chat = tok.apply_chat_template(student_msgs, tokenize=False, add_generation_prompt=True)
        else:
            chat = "You are a helpful assistant.\n\n[IMAGE]\nMessage:\n\nAcknowledge."

        with torch.no_grad():
            mm = _processor_encode_multimodal(processor, texts=[chat] * B_eff, dummy_imgs=dummy_imgs)
            mm = _maybe_to_device(mm, device)
            input_ids = mm.get("input_ids", None)
            if input_ids is None:
                raise RuntimeError("processor did not return input_ids")
            emb = wrapper.model.get_input_embeddings()
            emb_dtype = emb.weight.dtype
            base_embeds = emb(input_ids).detach()
            inputs_embeds = base_embeds.clone()
            for b in range(B_eff):
                pos = _find_image_positions(input_ids, special_ids, tokenizer=tok, batch_index=b)
                if not pos:
                    continue
                base_img = _resample_tokens(dummy_tokens, len(pos)).to(dtype=emb_dtype, device=device)
                add = _resample_tokens(inj[b], len(pos)).to(dtype=emb_dtype, device=device)
                inputs_embeds[b, pos, :] = base_img + add

        with torch.no_grad():
            out_s = wrapper.model(inputs_embeds=inputs_embeds, attention_mask=mm["attention_mask"],
                                  output_hidden_states=True, return_dict=True, use_cache=False)
        student_h = out_s.hidden_states[-1][:, -1, :]
        student_logits = out_s.logits[:, -1, :]

        teacher_h = torch.nan_to_num(teacher_h.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        student_h = torch.nan_to_num(student_h.float(), nan=0.0, posinf=1e4, neginf=-1e4)

        loss = torch.zeros((), device=device, dtype=torch.float32)
        loss_mse = torch.zeros((), device=device, dtype=torch.float32)
        loss_kl = torch.zeros((), device=device, dtype=torch.float32)
        loss_stats = torch.zeros((), device=device, dtype=torch.float32)

        if l_mse > 0:
            loss_mse = F.mse_loss(student_h, teacher_h)
            loss = loss + l_mse * loss_mse
        if l_kl > 0:
            loss_kl = _compute_kl_loss(student_logits=student_logits, teacher_logits=teacher_logits,
                                       temp=float(cfg.vision_codec_kl_temp), mode=kl_mode,
                                       logit_clip=kl_logit_clip, topk=kl_topk)
            loss = loss + l_kl * loss_kl
        if l_stats > 0:
            inj_rms = inj.float().pow(2).mean().sqrt().clamp_min(1e-6)
            loss_stats = F.mse_loss(inj_rms, dummy_rms.expand_as(inj_rms))
            loss = loss + l_stats * loss_stats

        if not torch.isfinite(loss):
            print(f"[ijepa-decoder] NaN at step {_step}, skip", flush=True)
            opt.zero_grad(set_to_none=True)
            pbar.update(1)
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if _ddp_rank() == 0 and (_step % log_every == 0 or _step == steps):
            pbar.set_postfix({"loss": f"{float(loss):.4f}"})
            print(f"[ijepa-decoder][{wrapper.model_name}] step={_step}/{steps} "
                  f"loss={float(loss):.6f} mse={float(loss_mse):.6f} "
                  f"kl={float(loss_kl):.6f} stats={float(loss_stats):.6f}", flush=True)
        pbar.update(1)

    pbar.close()
    dec_sd = {k: v.detach().float().cpu() for k, v in dec.state_dict().items()}
    return dec_sd, dummy_tokens.shape[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--ijepa_encoder_path", type=str, required=True)
    p.add_argument("--vision_codec_path", type=str, required=True)
    p.add_argument("--vision_codec_anchor_texts_path", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--latent_steps", type=int, default=1024)
    p.add_argument("--vision_codec_dim", type=int, default=512)
    p.add_argument("--vision_codec_tokens", type=int, default=1024)
    p.add_argument("--vision_codec_img_tokens", type=int, default=256)
    p.add_argument("--vision_codec_heads", type=int, default=8)
    p.add_argument("--vision_codec_layers", type=int, default=6)
    p.add_argument("--vision_codec_dropout", type=float, default=0.10)
    p.add_argument("--vision_codec_gate_init_bias", type=float, default=-4.0)
    p.add_argument("--vision_codec_train_steps", type=int, default=400)
    p.add_argument("--vision_codec_train_batch_size", type=int, default=2)
    p.add_argument("--vision_codec_train_lr", type=float, default=2e-4)
    p.add_argument("--vision_codec_loss_mse", type=float, default=1.0)
    p.add_argument("--vision_codec_loss_kl", type=float, default=0.25)
    p.add_argument("--vision_codec_loss_stats", type=float, default=0.1)
    p.add_argument("--vision_codec_kl_mode", type=str, default="auto")
    p.add_argument("--vision_codec_kl_topk", type=int, default=0)
    p.add_argument("--vision_codec_kl_temp", type=float, default=1.0)
    p.add_argument("--vision_codec_kl_logit_clip", type=float, default=80.0)
    p.add_argument("--vision_codec_latent_clip", type=float, default=50.0)
    p.add_argument("--vision_codec_inj_clip", type=float, default=20.0)
    p.add_argument("--vision_codec_log_every", type=int, default=10)
    p.add_argument("--vision_codec_dummy_image_count", type=int, default=1)
    p.add_argument("--vision_codec_dummy_image_size", type=int, default=224)
    p.add_argument("--vision_codec_skip_alignment_if_single", type=int, default=1)
    p.add_argument("--vision_codec_save_per_model", type=int, default=1)
    cfg = p.parse_args()

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    anchor_texts = _load_anchor_texts(cfg.vision_codec_anchor_texts_path)
    print(f"Loaded {len(anchor_texts)} anchor texts")

    # Load I-JEPA encoder
    print(f"Loading I-JEPA encoder: {cfg.ijepa_encoder_path}")
    ijepa_ckpt = torch.load(cfg.ijepa_encoder_path, map_location="cpu")
    encoders = ijepa_ckpt["encoders"]
    enc_names = list(encoders.keys())
    if not enc_names:
        raise RuntimeError("No encoder in I-JEPA checkpoint")
    enc_name = enc_names[0]
    enc_sd = encoders[enc_name]
    print(f"Encoder: {enc_name} ({len(enc_sd)} params)")

    wrapper = ModelWrapper(cfg.model_name, device, use_vllm=False, args=_DA())
    processor = _load_mm_processor_or_tokenizer(cfg.model_name, wrapper)
    special_ids = _infer_special_token_ids(processor)
    dummy_imgs = [_make_dummy_image(int(cfg.vision_codec_dummy_image_size)) for _ in range(int(cfg.vision_codec_dummy_image_count))]

    dec_sd, dummy_len = _train_decoder(
        wrapper=wrapper, processor=processor, special_ids=special_ids,
        dummy_imgs=dummy_imgs, anchor_texts=anchor_texts, cfg=cfg,
        enc_sd=enc_sd, enc_name=enc_name,
    )

    os.makedirs("checkpoints", exist_ok=True)
    ckpt = {
        "encoders": {enc_name: {k: v.detach().float().cpu() for k, v in enc_sd.items()}},
        "decoders": {enc_name: dec_sd},
        "dummy_token_lens": {enc_name: dummy_len},
        "comment": "I-JEPA encoder (frozen) + decoder (teacher-student trained)",
    }
    torch.save(ckpt, cfg.vision_codec_path)
    print(f"Saved: {cfg.vision_codec_path}")


if __name__ == "__main__":
    main()
