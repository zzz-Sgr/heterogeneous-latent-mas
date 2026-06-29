#!/usr/bin/env python3
"""
Checkpoint analysis: SVD + Effective Rank + RankMe + Cosine Similarity.
Compare original / VICReg / I-JEPA codecs to quantify latent space collapse.

Usage:
  python analyze_checkpoints.py \
    --checkpoints checkpoints/codec_qwen3_original.pt checkpoints/codec_qwen3_vicreg.pt checkpoints/codec_qwen3_ijepa.pt \
    --model_name /heterogeneous-latent-mas-main/model_download/Qwen3-VL-2B-Thinking \
    --anchor_texts_path data/vision_codec_anchor_text/mixed_cose_ocr_prm800k_small.jsonl \
    --num_samples 30
"""

import sys, os, argparse, json, math
from typing import Any, Dict, List, Optional, Sequence, Tuple
from pathlib import Path
import torch, torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, "/heterogeneous-latent-mas-main")
from models import ModelWrapper
from methods.vision_latent_mas_codec_new import LatentToUniversalEncoder
from train_vision_latent_mas_codec_new import (
    _load_anchor_texts, _infer_hidden_size,
    _load_mm_processor_or_tokenizer, _infer_special_token_ids,
)


def compute_metrics(U: torch.Tensor) -> Dict[str, float]:
    """
    U: [Batch*K, D] universal tokens from encoder.
    Returns effective_rank, rankme, top5_ratio, cos_sim_mean, cos_sim_std, dead_dims.
    """
    if U.dim() == 3:
        U = U.reshape(-1, U.shape[-1])
    U = U.float()
    N, D = U.shape

    # SVD
    try:
        _, S, _ = torch.svd(U, some=False)
    except:
        _, S, _ = np.linalg.svd(U.cpu().numpy(), full_matrices=False)
        S = torch.from_numpy(S).float()

    # Effective Rank
    sv_norm = S / (S.sum() + 1e-12)
    entropy = -torch.sum(sv_norm * torch.log(sv_norm + 1e-12))
    effective_rank = float(torch.exp(entropy))

    # RankMe
    rankme_val = float(torch.exp(entropy))  # same formula

    # Top-5 ratio
    top5_ratio = float(S[:5].sum() / (S.sum() + 1e-12))

    # Cosine similarity between tokens
    U_norm = F.normalize(U, dim=-1)
    cos_sim_mat = U_norm @ U_norm.T
    mask = ~torch.eye(N, dtype=torch.bool, device=U.device)
    off_diag = cos_sim_mat[mask]
    cos_sim_mean = float(off_diag.mean())
    cos_sim_std = float(off_diag.std())

    # Dead dimensions: std < 0.01
    dim_std = U.std(dim=0)
    dead_dims = int((dim_std < 0.01).sum().item())

    return {
        "effective_rank": round(effective_rank, 2),
        "rankme": round(rankme_val, 2),
        "top5_ratio": round(top5_ratio, 4),
        "cos_sim_mean": round(cos_sim_mean, 4),
        "cos_sim_std": round(cos_sim_std, 4),
        "dead_dims": dead_dims,
        "total_dims": D,
        "singular_values_top10": [round(float(s), 4) for s in S[:10].tolist()],
        "singular_values_decay": round(float(S[0] / (S[-1] + 1e-12)), 2) if len(S) > 1 else 0,
    }


def extract_U_ref(
    wrapper: ModelWrapper,
    enc: LatentToUniversalEncoder,
    anchor_texts: List[str],
    processor,
    special_ids,
    dummy_imgs,
    latent_steps: int,
    num_samples: int,
    device: torch.device,
) -> torch.Tensor:
    """Extract universal tokens for given anchor texts."""
    enc.eval()
    all_U = []

    for i in tqdm(range(0, min(num_samples, len(anchor_texts))), desc="Extracting U_ref"):
        t = anchor_texts[i]
        msgs = [[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Message:\n{t}\n\nAcknowledge."},
        ]]
        _, ids, mask, _ = wrapper.prepare_chat_batch(msgs, add_generation_prompt=True)
        ids = ids.to(device); mask = mask.to(device)

        with torch.no_grad():
            _, lat = wrapper.generate_latent_batch(ids, attention_mask=mask,
                latent_steps=latent_steps, return_latent_embeds=True)
        # lat: [B, latent_steps, H]
        lat_flat = lat.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            U = enc(lat_flat).detach().float().cpu()  # [B*R, K_univ, D]
        all_U.append(U)

    return torch.cat(all_U, dim=0) if all_U else torch.empty(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", type=str, nargs="+", required=True, help="Paths to .pt checkpoints")
    p.add_argument("--labels", type=str, nargs="+", default=None, help="Labels for each checkpoint")
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--anchor_texts_path", type=str, required=True)
    p.add_argument("--latent_steps", type=int, default=64, help="Shorter for extraction speed")
    p.add_argument("--num_samples", type=int, default=30)
    p.add_argument("--vision_codec_dim", type=int, default=512)
    p.add_argument("--vision_codec_tokens", type=int, default=1024)
    p.add_argument("--vision_codec_heads", type=int, default=8)
    p.add_argument("--vision_codec_layers", type=int, default=6)
    p.add_argument("--vision_codec_img_tokens", type=int, default=256)
    p.add_argument("--vision_codec_dropout", type=float, default=0.10)
    p.add_argument("--output_dir", type=str, default="analysis_output")
    p.add_argument("--extract_only", type=int, default=0, help="If 1, save U tensors and exit")
    cfg = p.parse_args()

    os.makedirs(cfg.output_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading model: {cfg.model_name}")

    # ModelWrapper needs args namespace; latent_space_realign not needed for analysis
    class _DummyArgs:
        latent_space_realign = 0
    wrapper = ModelWrapper(cfg.model_name, device, use_vllm=False, args=_DummyArgs())
    processor = _load_mm_processor_or_tokenizer(cfg.model_name, wrapper)
    special_ids = _infer_special_token_ids(processor)
    H = _infer_hidden_size(wrapper)

    dummy_imgs = []
    from train_vision_latent_mas_codec_new import _make_dummy_image
    dummy_imgs = [_make_dummy_image(224)]

    anchor_texts = _load_anchor_texts(cfg.anchor_texts_path)
    print(f"Loaded {len(anchor_texts)} anchor texts. Using {cfg.num_samples} for analysis.")

    labels = cfg.labels if cfg.labels else [os.path.basename(p).replace(".pt", "") for p in cfg.checkpoints]

    all_results = {}

    for ckpt_path, label in zip(cfg.checkpoints, labels):
        if not os.path.exists(ckpt_path):
            print(f"[warn] Checkpoint not found: {ckpt_path}. Skipping.")
            continue

        print(f"\n{'='*50}\nAnalyzing: {label}\n{'='*50}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        encoders = ckpt.get("encoders", {})
        if not encoders:
            print(f"[warn] No encoders in checkpoint. Keys: {list(ckpt.keys())}")
            continue

        # Get encoder state dict (first model in checkpoint)
        enc_name = list(encoders.keys())[0]
        enc_sd = encoders[enc_name]

        # Determine dim from state dict
        sample_key = [k for k in enc_sd.keys() if "weight" in k and len(enc_sd[k].shape) >= 2]
        if sample_key:
            d_univ = enc_sd[sample_key[0]].shape[0]
        else:
            d_univ = cfg.vision_codec_dim

        enc = LatentToUniversalEncoder(
            h_in=H, d_univ=d_univ,
            k_univ=cfg.vision_codec_tokens,
            n_heads=cfg.vision_codec_heads,
            n_layers=cfg.vision_codec_layers,
            dropout=cfg.vision_codec_dropout,
        )
        enc.load_state_dict(enc_sd, strict=False)
        enc = enc.to(device=device, dtype=torch.float32)

        # Extract U_ref
        U_all = extract_U_ref(
            wrapper=wrapper, enc=enc, anchor_texts=anchor_texts,
            processor=processor, special_ids=special_ids, dummy_imgs=dummy_imgs,
            latent_steps=cfg.latent_steps, num_samples=cfg.num_samples, device=device,
        )

        print(f"U_all shape: {U_all.shape}")

        if cfg.extract_only:
            torch.save(U_all, os.path.join(cfg.output_dir, f"U_{label}.pt"))
            print(f"Saved U tensor to {cfg.output_dir}/U_{label}.pt")
            continue

        # Compute metrics
        metrics = compute_metrics(U_all)
        all_results[label] = metrics

        print(f"  Effective Rank: {metrics['effective_rank']} / {metrics['total_dims']}")
        print(f"  RankMe:         {metrics['rankme']}")
        print(f"  Top-5 Ratio:    {metrics['top5_ratio']}")
        print(f"  CosSim Mean:    {metrics['cos_sim_mean']}")
        print(f"  Dead Dims:      {metrics['dead_dims']} / {metrics['total_dims']}")
        print(f"  SVD decay:      {metrics['singular_values_decay']}")

    if cfg.extract_only:
        return

    # Print comparison table
    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")
    header = f"{'Metric':<25s}"
    for label in all_results:
        header += f" {label:<20s}"
    print(header)
    print("-" * len(header))

    rows = ["effective_rank", "rankme", "top5_ratio", "cos_sim_mean", "dead_dims"]
    row_names = {"effective_rank": "Effective Rank", "rankme": "RankMe", "top5_ratio": "Top-5 Ratio",
                 "cos_sim_mean": "CosSim Mean", "dead_dims": "Dead Dims"}
    for row in rows:
        line = f"{row_names[row]:<25s}"
        for label in all_results:
            val = all_results[label][row]
            line += f" {str(val):<20s}"
        print(line)

    # Save JSON
    out_path = os.path.join(cfg.output_dir, "analysis_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    # Also save all singular values for plotting later
    sv_data = {}
    for label, metrics in all_results.items():
        sv_data[label] = metrics["singular_values_top10"]
    sv_path = os.path.join(cfg.output_dir, "singular_values.json")
    with open(sv_path, "w") as f:
        json.dump(sv_data, f, indent=2)
    print(f"SVD data saved to: {sv_path}")


if __name__ == "__main__":
    main()
