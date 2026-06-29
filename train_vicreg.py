#!/usr/bin/env python3
"""
VICReg anti-collapse variant of Vision Wormhole codec training.
Adds variance + covariance regularization to prevent latent space collapse.

Key change from original train_vision_latent_mas_codec_new.py:
  loss = MSE + KL + Stats + λ_ac * (loss_var + loss_cov)

Usage: same args as original, plus --vision_codec_loss_anti_collapse (default 0.1)
"""

import sys, os, argparse, json, math, random, copy, itertools, inspect
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm

# ── reuse all imports and helpers from original ──
sys.path.insert(0, "/heterogeneous-latent-mas-main")
from train_vision_latent_mas_codec_new import (
    _SpecialTokenIds, _DummyImageSpec, _parse_model_list, _load_anchor_texts,
    _load_mm_processor_or_tokenizer, _infer_special_token_ids, _make_dummy_image,
    _find_image_positions, _resample_tokens, _build_position_ids_from_attention,
    _resolve_dummy_image_specs, _compute_kl_loss, _get_tokenizer_like,
    _infer_hidden_size, _extract_dummy_image_tokens, _build_mm_user_content,
    _processor_encode_multimodal, _maybe_to_device, _ddp_rank, _ddp_barrier,
    _ddp_is_active, _ddp_all_true, _derive_partial_ckpt_path,
    _past_length, _internvl_prepare_multimodal_batch, _minicpm_prepare_multimodal_batch,
    _minicpm_build_inputs_embeds, _minicpm_positions_from_bounds,
    _is_internvl_wrapper, _is_minicpm_wrapper, _get_text_backbone,
)
from models import ModelWrapper
from methods.vision_latent_mas_codec_new import (
    LatentToUniversalEncoder, UniversalToVisionDecoder,
)

# ── VICReg anti-collapse loss ──
def _anti_collapse_loss(U: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    U: [B*K, D] or [B, K, D] — universal tokens from encoder.
    Returns (loss_var, loss_cov).

    Ref: VICReg (Bardes et al., ICLR 2022)
    """
    if U.dim() == 3:
        U = U.reshape(-1, U.shape[-1])
    D = U.shape[-1]

    # Variance regularization: std per dimension >= 1
    std = U.std(dim=0, unbiased=False)
    loss_var = F.relu(1.0 - std).mean()

    # Covariance regularization: off-diagonal of covariance → 0
    U_centered = U - U.mean(dim=0, keepdim=True)
    # [N, D] @ [D, N] → [D, D], but use efficient form
    cov = (U_centered.T @ U_centered) / (U.shape[0] - 1)
    diag = cov.diagonal().clone()
    mask = torch.eye(D, device=U.device, dtype=torch.bool)
    loss_cov = cov[~mask].pow(2).sum() / D

    return loss_var, loss_cov


def _train_one_model_vicreg(
    *,
    wrapper: ModelWrapper,
    processor,
    special_ids,
    dummy_imgs: List[Any],
    anchor_texts: List[str],
    cfg: argparse.Namespace,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int]:
    """
    Identical to _train_one_model, except:
    - Adds anti-collapse loss on encoder output U_flat
    - Uses cfg.vision_codec_loss_anti_collapse weight
    """
    device = wrapper.model.device
    tok = _get_tokenizer_like(processor) or getattr(wrapper, "tokenizer", None)
    if tok is None:
        raise RuntimeError("processor has no tokenizer")

    codec_bf16_requested = int(getattr(cfg, "codec_uses_bf16", 0)) == 1
    codec_dtype = torch.bfloat16 if (codec_bf16_requested and device.type == "cuda") else torch.float32
    avoid_full_logits_in_student = int(getattr(cfg, "avoid_full_logits_in_student", 0)) == 1
    if _ddp_rank() == 0:
        print(f"[codec-train-vicreg] codec_dtype={str(codec_dtype)} avoid_full_logits_in_student={int(avoid_full_logits_in_student)}", flush=True)
        print(f"[codec-train-vicreg] anti_collapse_weight={float(cfg.vision_codec_loss_anti_collapse)}", flush=True)

    wrapper.model.eval()
    for p in wrapper.model.parameters():
        p.requires_grad_(False)

    H = _infer_hidden_size(wrapper)
    enc = LatentToUniversalEncoder(
        h_in=H, d_univ=int(cfg.vision_codec_dim), k_univ=int(cfg.vision_codec_tokens),
        n_heads=int(cfg.vision_codec_heads), n_layers=int(cfg.vision_codec_layers),
        dropout=float(cfg.vision_codec_dropout),
    ).to(device=device, dtype=codec_dtype)
    dec = UniversalToVisionDecoder(
        d_univ=int(cfg.vision_codec_dim), h_out=H, k_img=int(cfg.vision_codec_img_tokens),
        n_heads=int(cfg.vision_codec_heads), n_layers=int(cfg.vision_codec_layers),
        dropout=float(cfg.vision_codec_dropout), gate_init_bias=float(cfg.vision_codec_gate_init_bias),
    ).to(device=device, dtype=codec_dtype)

    enc.train(); dec.train()
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=float(cfg.vision_codec_train_lr))

    if not dummy_imgs:
        raise RuntimeError("dummy_imgs must be non-empty")
    dummy_img_primary = dummy_imgs[0]
    dummy_tokens = _extract_dummy_image_tokens(wrapper, processor, special_ids, dummy_imgs)
    dummy_rms = dummy_tokens.pow(2).mean().sqrt().clamp_min(1e-6)

    steps = int(cfg.vision_codec_train_steps)
    bs = int(cfg.vision_codec_train_batch_size)
    mc_rollouts = int(cfg.vision_codec_mc_rollouts)
    latent_steps = int(cfg.latent_steps)
    kl_mode = str(cfg.vision_codec_kl_mode)
    kl_topk = int(cfg.vision_codec_kl_topk)
    kl_logit_clip = float(cfg.vision_codec_kl_logit_clip)
    loss_mse_w = float(cfg.vision_codec_loss_mse)
    loss_kl_w = float(cfg.vision_codec_loss_kl)
    loss_stats_w = float(cfg.vision_codec_loss_stats)
    loss_ac_w = float(cfg.vision_codec_loss_anti_collapse)
    inj_clip = float(cfg.vision_codec_inj_clip)
    log_every = int(cfg.vision_codec_log_every)

    loss_ids = list(range(bs * steps))
    _step = 0
    pbar = tqdm(total=steps, desc=f"[codec-train-vicreg] {wrapper.model_name}", position=0, leave=True)

    # For teacher H cache
    teacher_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    while _step < steps:
        batch_texts = []
        for i in range(bs):
            idx = loss_ids[_step * bs + i] % len(anchor_texts)
            batch_texts.append(anchor_texts[idx])
        _step += 1

        texts_for_teacher = batch_texts  # same anchor text for teacher

        # ── Teacher forward ──
        teacher_msgs = [[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Message:\n{t}\n\nAcknowledge."},
        ] for t in texts_for_teacher]
        ids_list, mask_list = [], []
        for msgs in teacher_msgs:
            _, ids, mask, _ = wrapper.prepare_chat_batch([msgs], add_generation_prompt=True)
            ids_list.append(ids); mask_list.append(mask)
        max_len_t = max(ids.shape[-1] for ids in ids_list)
        teacher_ids = torch.cat([
            F.pad(ids, (0, max_len_t - ids.shape[-1]), value=0) for ids in ids_list
        ], dim=0).to(device)
        teacher_mask = torch.cat([
            F.pad(mask, (0, max_len_t - mask.shape[-1]), value=0) for mask in mask_list
        ], dim=0).to(device)

        with torch.no_grad():
            out_t = wrapper.model(input_ids=teacher_ids, attention_mask=teacher_mask,
                                  output_hidden_states=True, return_dict=True, use_cache=False)
        teacher_h = out_t.hidden_states[-1][:, -1, :]
        teacher_logits = out_t.logits[:, -1, :]

        # ── Latent rollout + codec ──
        with torch.no_grad():
            _, lat = wrapper.generate_latent_batch(teacher_ids, attention_mask=teacher_mask,
                                                    latent_steps=latent_steps, return_latent_embeds=True)
        B, R, L, H_dim = lat.shape
        lat_flat = lat.reshape(B * R, L, H_dim).to(device=device, dtype=codec_dtype)

        U_flat = torch.nan_to_num(enc(lat_flat), nan=0.0, posinf=1e4, neginf=-1e4)
        delta_flat, gate_flat = dec(U_flat)
        delta_flat = torch.nan_to_num(delta_flat, nan=0.0, posinf=1e4, neginf=-1e4)
        gate_flat = torch.nan_to_num(gate_flat, nan=0.0, posinf=1.0, neginf=0.0)
        delta = delta_flat.reshape(B, R, delta_flat.shape[1], H_dim)
        gate = gate_flat.reshape(B, R, 1, 1)
        inj = (gate * delta).mean(dim=1)
        inj = torch.nan_to_num(inj, nan=0.0, posinf=1e4, neginf=-1e4)
        if inj_clip > 0:
            inj = inj.clamp(min=-inj_clip, max=inj_clip)

        # ── Student forward ──
        student_msgs = [[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": _build_mm_user_content(
                num_images=len(dummy_imgs),
                text="Message:\n\nAcknowledge.",
            )},
        ] for _ in range(B)]
        _, s_ids, s_mask, _ = wrapper.prepare_chat_batch(student_msgs, add_generation_prompt=True)
        s_ids = s_ids.to(device); s_mask = s_mask.to(device)

        with torch.no_grad():
            mm = _processor_encode_multimodal(processor,
                texts=[_build_mm_user_content(num_images=len(dummy_imgs), text="Message:\n\nAcknowledge.")] * B,
                dummy_imgs=dummy_imgs)
            mm = _maybe_to_device(mm, device)
            input_ids = mm.get("input_ids", None)
            emb = wrapper.model.get_input_embeddings()
            emb_dtype = emb.weight.dtype
            base_embeds = emb(input_ids).detach()
            inputs_embeds = base_embeds.clone()
            for b in range(B):
                pos = _find_image_positions(input_ids, special_ids, tokenizer=tok, batch_index=b)
                if not pos:
                    continue
                base_img = _resample_tokens(dummy_tokens, len(pos)).to(dtype=emb_dtype, device=device)
                add = _resample_tokens(inj[b], len(pos)).to(dtype=emb_dtype, device=device)
                inputs_embeds[b, pos, :] = base_img + add

        student_fwd = {"inputs_embeds": inputs_embeds, "attention_mask": mm["attention_mask"]}
        with torch.no_grad():
            out_s = wrapper.model(**student_fwd, output_hidden_states=True, return_dict=True, use_cache=False)
        student_h = out_s.hidden_states[-1][:, -1, :]
        student_logits = out_s.logits[:, -1, :]

        teacher_h = torch.nan_to_num(teacher_h.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        student_h = torch.nan_to_num(student_h.float(), nan=0.0, posinf=1e4, neginf=-1e4)

        # ── Loss computation ──
        loss = torch.zeros((), device=device, dtype=torch.float32)
        loss_mse = torch.zeros((), device=device, dtype=torch.float32)
        loss_kl = torch.zeros((), device=device, dtype=torch.float32)
        loss_stats = torch.zeros((), device=device, dtype=torch.float32)
        loss_ac = torch.zeros((), device=device, dtype=torch.float32)

        if loss_mse_w > 0:
            loss_mse = F.mse_loss(student_h, teacher_h)
            loss = loss + loss_mse_w * loss_mse
        if loss_kl_w > 0:
            loss_kl = _compute_kl_loss(student_logits=student_logits, teacher_logits=teacher_logits,
                                       temp=float(cfg.vision_codec_kl_temp), mode=kl_mode,
                                       logit_clip=kl_logit_clip, topk=kl_topk)
            loss = loss + loss_kl_w * loss_kl
        if loss_stats_w > 0:
            inj_rms = inj.float().pow(2).mean().sqrt().clamp_min(1e-6)
            loss_stats = F.mse_loss(inj_rms, dummy_rms.expand_as(inj_rms))
            loss = loss + loss_stats_w * loss_stats

        # ★ VICReg anti-collapse ★
        if loss_ac_w > 0:
            loss_var, loss_cov = _anti_collapse_loss(U_flat)
            loss_ac = loss_var + loss_cov
            loss = loss + loss_ac_w * loss_ac

        if not torch.isfinite(loss):
            print(f"[vicreg] Non-finite loss at step {_step}. Skipping.", flush=True)
            opt.zero_grad(set_to_none=True)
            pbar.update(1)
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if _ddp_rank() == 0 and (_step % log_every == 0 or _step == steps):
            pbar.set_postfix(loss=float(loss.detach().cpu()))
            print(
                f"[codec-train-vicreg][{wrapper.model_name}] step={_step}/{steps} "
                f"loss={float(loss.detach().cpu()):.6f} mse={float(loss_mse.detach().cpu()):.6f} "
                f"kl={float(loss_kl.detach().cpu()):.6f} stats={float(loss_stats.detach().cpu()):.6f} "
                f"ac={float(loss_ac.detach().cpu() if isinstance(loss_ac, torch.Tensor) else loss_ac):.6f}",
                flush=True,
            )
        pbar.update(1)

    pbar.close()

    enc_sd = enc.state_dict() if not _ddp_is_active() else enc.module.state_dict()
    dec_sd = dec.state_dict() if not _ddp_is_active() else dec.module.state_dict()
    return {k: v.detach().float().cpu() for k, v in enc_sd.items()}, \
           {k: v.detach().float().cpu() for k, v in dec_sd.items()}, \
           dummy_tokens.shape[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="")
    p.add_argument("--agent_model_names", type=str, default="")
    p.add_argument("--vision_codec_path", type=str, required=True)
    p.add_argument("--vision_codec_partial_ckpt_path", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--latent_steps", type=int, default=1024)
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--vision_codec_dim", type=int, default=256)
    p.add_argument("--vision_codec_tokens", type=int, default=1024)
    p.add_argument("--vision_codec_img_tokens", type=int, default=256)
    p.add_argument("--vision_codec_heads", type=int, default=8)
    p.add_argument("--vision_codec_layers", type=int, default=4)
    p.add_argument("--vision_codec_dropout", type=float, default=0.)
    p.add_argument("--vision_codec_gate_init_bias", type=float, default=-4.0)
    p.add_argument("--vision_codec_train_steps", type=int, default=200)
    p.add_argument("--vision_codec_train_batch_size", type=int, default=8)
    p.add_argument("--vision_codec_train_lr", type=float, default=5e-4)
    p.add_argument("--vision_codec_loss_mse", type=float, default=1.0)
    p.add_argument("--vision_codec_loss_kl", type=float, default=0.25)
    p.add_argument("--vision_codec_loss_stats", type=float, default=0.1)
    p.add_argument("--vision_codec_loss_anti_collapse", type=float, default=0.1)
    p.add_argument("--vision_codec_kl_mode", type=str, default="auto")
    p.add_argument("--vision_codec_kl_topk", type=int, default=0)
    p.add_argument("--vision_codec_kl_temp", type=float, default=1.0)
    p.add_argument("--vision_codec_kl_logit_clip", type=float, default=80.0)
    p.add_argument("--vision_codec_latent_clip", type=float, default=50.0)
    p.add_argument("--vision_codec_inj_clip", type=float, default=20.0)
    p.add_argument("--vision_codec_mc_rollouts", type=int, default=1)
    p.add_argument("--vision_codec_rollout_mode", type=str, default="latent")
    p.add_argument("--vision_codec_rollout_temperature", type=float, default=1.0)
    p.add_argument("--vision_codec_rollout_top_p", type=float, default=1.0)
    p.add_argument("--vision_codec_rollout_noise_std", type=float, default=0.0)
    p.add_argument("--vision_codec_log_every", type=int, default=10)
    p.add_argument("--vision_codec_dummy_image_count", type=int, default=1)
    p.add_argument("--vision_codec_dummy_image_size", type=int, default=224)
    p.add_argument("--vision_codec_anchor_texts_path", type=str, default="")
    p.add_argument("--vision_codec_skip_alignment_if_single", type=int, default=0)
    p.add_argument("--vision_codec_save_per_model", type=int, default=0)
    p.add_argument("--vision_codec_ref_idx", type=int, default=0)
    p.add_argument("--vision_codec_ridge", type=float, default=1e-3)
    p.add_argument("--codec_uses_bf16", type=int, default=0)
    p.add_argument("--avoid_full_logits_in_student", type=int, default=0)
    cfg = p.parse_args()

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    model_names = _parse_model_list(cfg.agent_model_names) if cfg.agent_model_names else \
                  ([] if not cfg.model_name else [cfg.model_name])
    if not model_names:
        raise ValueError("Provide --model_name or --agent_model_names")

    anchor_texts = _load_anchor_texts(cfg.vision_codec_anchor_texts_path)
    print(f"Loaded {len(anchor_texts)} anchor texts for training.", flush=True)

    dummy_specs = _resolve_dummy_image_specs(model_names, cfg)
    encoders, decoders, specials, dummy_token_lens = {}, {}, {}, {}

    for mi, name in enumerate(model_names):
        print(f"\n{'='*30}\nTraining VICReg codec for model[{mi}]: {name}\n{'='*30}", flush=True)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        wrapper = ModelWrapper(name, device, use_vllm=False)
        processor = _load_mm_processor_or_tokenizer(name, wrapper)
        special_ids = _infer_special_token_ids(processor)
        spec = dummy_specs[name]
        dummy_imgs = [_make_dummy_image(spec.size) for _ in range(spec.count)]

        enc_sd, dec_sd, dlen = _train_one_model_vicreg(
            wrapper=wrapper, processor=processor, special_ids=special_ids,
            dummy_imgs=dummy_imgs, anchor_texts=anchor_texts, cfg=cfg,
        )
        encoders[name] = enc_sd; decoders[name] = dec_sd
        specials[name] = special_ids; dummy_token_lens[name] = dlen

        partial_path = cfg.vision_codec_partial_ckpt_path or _derive_partial_ckpt_path(cfg.vision_codec_path)
        if partial_path and int(cfg.vision_codec_save_per_model) == 1:
            os.makedirs(os.path.dirname(partial_path) or ".", exist_ok=True)
            torch.save({
                "encoders": encoders, "decoders": decoders,
                "specials": {k: v._asdict() for k, v in specials.items()},
                "dummy_token_lens": dummy_token_lens,
            }, partial_path)
            print(f"[vicreg] Saved partial checkpoint: {partial_path}", flush=True)

    # Final save
    os.makedirs(os.path.dirname(cfg.vision_codec_path) or ".", exist_ok=True)
    torch.save({
        "encoders": encoders, "decoders": decoders,
        "specials": {k: v._asdict() for k, v in specials.items()},
        "dummy_token_lens": dummy_token_lens,
    }, cfg.vision_codec_path)
    print(f"[vicreg] Saved final checkpoint: {cfg.vision_codec_path}", flush=True)


if __name__ == "__main__":
    main()
