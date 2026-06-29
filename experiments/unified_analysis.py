#!/usr/bin/env python3
"""
统一维度坍塌分析 — 一次前向，全部指标
==========================================
一次 VLM forward 得到 U ∈ R^{N×D}，然后从 U 算:
  1. 逐维激活方差 (sorted)        2. 维度生命分档
  3. Top-K 主导维编号              4. 两 checkpoint 维度重叠率
  5. Effective Rank                6. RankMe
  7. Token CosSim                  8. Dead Dims
  9. SV Gini                       +  Intrinsic Dimension (TwoNN)
"""
import torch, numpy as np, json, sys, os, warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────
ROOT = "/heterogeneous-latent-mas-main"
MODEL_NAME = "model_download/Qwen3-VL-2B-Thinking"
ANCHOR_PATH = "data/vision_codec_anchor_text/mixed_cose_ocr_prm800k.jsonl"
N_TEXTS = 20          # 20条 text → ~20k tokens，SVD足够稳定
LATENT_STEPS = 64     # VLM latent token 步数
DEVICE = "cuda:0"

CHECKPOINTS = [
    # I-JEPA v2 训练过程 (10个中间checkpoint)
    ("checkpoints/codec_qwen3_ijepa_full_data_enc_v2_step200.pt",   "IJEPA_v2_0200"),
    ("checkpoints/codec_qwen3_ijepa_full_data_enc_v2_step400.pt",   "IJEPA_v2_0400"),
    ("checkpoints/codec_qwen3_ijepa_full_data_enc_v2_step600.pt",   "IJEPA_v2_0600"),
    ("checkpoints/codec_qwen3_ijepa_full_data_enc_v2_step800.pt",   "IJEPA_v2_0800"),
    ("checkpoints/codec_qwen3_ijepa_full_data_enc_v2_step1000.pt",  "IJEPA_v2_1000"),
    ("checkpoints/codec_qwen3_ijepa_full_data_enc_v2_step1200.pt",  "IJEPA_v2_1200"),
    ("checkpoints/codec_qwen3_ijepa_full_data_enc_v2_step1400.pt",  "IJEPA_v2_1400"),
    ("checkpoints/codec_qwen3_ijepa_full_data_enc_v2_step1600.pt",  "IJEPA_v2_1600"),
    ("checkpoints/codec_qwen3_ijepa_full_data_enc_v2_step1800.pt",  "IJEPA_v2_1800"),
    ("checkpoints/codec_qwen3_ijepa_full_data_enc_v2_step2000.pt",  "IJEPA_v2_2000"),
    # Original v2
    ("checkpoints/codec_qwen3_original_full_v2_step200.pt",         "Original_v2_0200"),
    ("checkpoints/codec_qwen3_original_full_v2_step400.pt",         "Original_v2_0400"),
    # Baseline (旧)
    ("checkpoints/codec_qwen3_original.pt",                         "Original_small"),
    ("checkpoints/codec_qwen3_original_full.pt",                    "Original_full"),
    ("checkpoints/codec_qwen3_ijepa.pt",                            "IJEPA_small"),
    ("checkpoints/codec_qwen3_ijepa_full.pt",                       "IJEPA_old_full"),
    ("checkpoints/codec_qwen3_vicreg_05.pt",                        "VICReg_small"),
    ("checkpoints/codec_qwen3_vicreg_05_full.pt",                   "VICReg_full"),
]


def load_encoder(ckpt_path):
    """加载 encoder，兼容多种 checkpoint 格式"""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    encoders = ckpt.get("encoders", {})
    steps = ckpt.get("_steps", "?")
    comment = ckpt.get("comment", "")
    if not encoders:
        return None, steps, comment
    enc_sd = list(encoders.values())[0]
    return enc_sd, steps, comment


def compute_all_metrics(U: torch.Tensor, label: str):
    """从 U [N, D] 算出全部指标"""
    U = U.float()
    N, D = U.shape
    r = {"label": label, "N": N, "D": D}

    # ── 1-2. Per-dim variance + life bins ──
    dim_var = U.var(dim=0).numpy()          # [D]
    sorted_idx = np.argsort(dim_var)[::-1]
    sorted_var = dim_var[sorted_idx]
    max_var = dim_var.max()
    life_score = dim_var / (max_var + 1e-12)

    r["dim_var_top30"] = sorted_var[:30].tolist()
    r["dim_var_top30_idx"] = sorted_idx[:30].tolist()
    r["life_strong"] = int((life_score > 0.10).sum())
    r["life_moderate"] = int((life_score > 0.01).sum())
    r["life_weak"] = int((life_score > 0.001).sum())
    r["dead_dims"] = int((dim_var < 1e-8).sum())
    r["dead_dims_std001"] = int((U.std(dim=0) < 0.01).sum().item())

    # ── 3. Top-K dims (already in dim_var_top30_idx) ──

    # ── 4. Cumulative variance ──
    cumsum = np.cumsum(sorted_var) / (sorted_var.sum() + 1e-12)
    for thr in [0.5, 0.8, 0.9, 0.95, 0.99]:
        n_idx = int(np.searchsorted(cumsum, thr) + 1)
        r[f"dims_for_{int(thr*100)}pct"] = min(n_idx, D)

    # ── 5. Effective Rank ──
    _, S, _ = torch.svd(U)
    S_np = S.cpu().numpy()
    sv = S_np / (S_np.sum() + 1e-12)
    entropy = -np.sum(sv * np.log(sv + 1e-12))
    r["effective_rank"] = float(np.exp(entropy))

    # ── 6. RankMe ──
    s2 = S_np ** 2
    p = s2 / (s2.sum() + 1e-12)
    rm_ent = -np.sum(p * np.log(p + 1e-12))
    r["rankme"] = float(np.exp(rm_ent))

    # SmoothRankMe
    for tau in [0.5, 1.0, 2.0]:
        st = S_np ** tau
        pt = st / (st.sum() + 1e-12)
        h = -np.sum(pt * np.log(pt + 1e-12))
        r[f"smooth_rankme_tau{tau}"] = float(np.exp(h))

    # ── 7. Token CosSim (采样避免 N×N 矩阵 OOM) ──
    U_norm = U / (U.norm(dim=1, keepdim=True) + 1e-12)
    n_sample = min(3000, N)
    idx = torch.randperm(N)[:n_sample]
    U_sample = U_norm[idx]
    cs_sample = U_sample @ U_sample.T
    mask = ~torch.eye(n_sample, dtype=bool)
    off_diag_sample = cs_sample[mask]
    r["cos_sim_mean"] = float(off_diag_sample.mean())
    r["cos_sim_std"] = float(off_diag_sample.std())
    for q in [10, 25, 50, 75, 90]:
        r[f"cos_sim_p{q}"] = float(np.quantile(off_diag_sample.numpy(), q / 100))

    # ── 8. Dead Dims (already above as dead_dims_std001) ──

    # ── 9. SV Gini ──
    sv_sorted = np.sort(sv)
    n_sv = len(sv_sorted)
    gini = (2 * np.sum(np.arange(1, n_sv + 1) * sv_sorted) - (n_sv + 1) * sv_sorted.sum()) \
           / (n_sv * sv_sorted.sum() + 1e-12)
    r["sv_gini"] = float(gini)

    # ── Spectral metrics ──
    r["participation_ratio"] = float(S_np.sum() ** 2 / (S_np ** 2).sum())
    r["spectral_gap_max"] = float(np.max(S_np[:-1] / (S_np[1:] + 1e-12)))
    r["s2_to_s1_ratio"] = float(S_np[1] / (S_np[0] + 1e-12))

    # Top-K SV dominance
    for k in [1, 3, 5, 10]:
        r[f"top{k}_sv_ratio"] = float(sv[:k].sum())

    # ── Power-law alpha ──
    xx = np.arange(1, min(21, len(S_np)))
    yy = np.log(S_np[:len(xx)] + 1e-12)
    slope, _ = np.polyfit(np.log(xx), yy, 1)
    r["power_law_alpha"] = float(-slope)

    # ── Intrinsic Dimension (TwoNN) ──
    try:
        from sklearn.neighbors import NearestNeighbors
        U_np = U.cpu().numpy()
        nn = NearestNeighbors(n_neighbors=3).fit(U_np)
        dists, _ = nn.kneighbors(U_np)
        mu = dists[:, 2] / (dists[:, 1] + 1e-12)
        r["intrinsic_dim_twonn"] = float(N / np.sum(np.log(mu)))
    except Exception:
        r["intrinsic_dim_twonn"] = -1

    # ── LiDO eff dim ──
    r["lido_eff_dim"] = float((S_np ** 2).sum() / (S_np ** 2).max())

    # ── Collapse Score ──
    er_norm = min(1.0, r["effective_rank"] / D)
    r["collapse_score"] = float(np.clip(
        0.3 * (1.0 - er_norm) + 0.3 * r["cos_sim_mean"] +
        0.2 * (r["dead_dims_std001"] / D) +
        0.2 * max(0, 1 - r["s2_to_s1_ratio"] * 5), 0, 1))

    r["singular_values_top20"] = S_np[:20].tolist()

    return r, U


def compute_dim_overlap(label_a, idx_a, label_b, idx_b, top_k=50):
    """两个 checkpoint 的 Top-K 维度重叠"""
    set_a = set(idx_a[:top_k].tolist() if isinstance(idx_a, np.ndarray) else idx_a[:top_k])
    set_b = set(idx_b[:top_k].tolist() if isinstance(idx_b, np.ndarray) else idx_b[:top_k])
    inter = set_a & set_b
    return {
        "pair": f"{label_a} vs {label_b}",
        "top_k": top_k,
        "overlap_count": len(inter),
        "overlap_ratio": len(inter) / top_k,
        "overlap_dims": sorted(inter),
    }


def main():
    sys.path.insert(0, ROOT)
    from models import ModelWrapper
    from methods.vision_latent_mas_codec_new import LatentToUniversalEncoder
    from train_vision_latent_mas_codec_new import _load_anchor_texts, _infer_hidden_size

    device = torch.device(DEVICE)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading VLM...", flush=True)

    class DA: latent_space_realign = 0
    w = ModelWrapper(f"{ROOT}/{MODEL_NAME}", device, use_vllm=False, args=DA())
    H = _infer_hidden_size(w)
    texts = _load_anchor_texts(f"{ROOT}/{ANCHOR_PATH}")[:N_TEXTS]
    print(f"[{datetime.now().strftime('%H:%M:%S')}] VLM hidden={H}, texts={len(texts)}", flush=True)

    all_metrics = {}
    all_Us = {}       # 保留 U 用于后续重叠分析
    all_dim_idx = {}   # 保留 sorted_idx 用于重叠分析

    total = len(CHECKPOINTS)
    for i, (ckpt_rel, label) in enumerate(CHECKPOINTS):
        ckpt_path = f"{ROOT}/{ckpt_rel}"
        try:
            if not os.path.exists(ckpt_path):
                print(f"[{i+1}/{total}] SKIP {label}: file not found", flush=True)
                continue

            enc_sd, steps, comment = load_encoder(ckpt_path)
            if enc_sd is None:
                print(f"[{i+1}/{total}] SKIP {label}: no encoder", flush=True)
                continue

            enc = LatentToUniversalEncoder(
                h_in=H, d_univ=512, k_univ=1024,
                n_heads=8, n_layers=6, dropout=0.10
            ).to(device)
            enc.load_state_dict(enc_sd, strict=False)
            enc.eval()

            all_U = []
            with torch.no_grad():
                for j, t in enumerate(texts):
                    msgs = [[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"Message:\n{t}\n\nAcknowledge."}
                    ]]
                    _, ids, mask, _ = w.prepare_chat_batch(msgs, add_generation_prompt=True)
                    ids, mask = ids.to(device), mask.to(device)
                    _, lat = w.generate_latent_batch(
                        ids, mask, latent_steps=LATENT_STEPS, return_latent_embeds=True
                    )
                    U = enc(lat.float().to(device)).detach().cpu()
                    all_U.append(U)
                    if (j + 1) % 10 == 0:
                        print(f"  [{i+1}/{total}] {label} text {j+1}/{len(texts)}", flush=True)

            U_all = torch.cat(all_U, dim=0).reshape(-1, 512)
            metrics, _ = compute_all_metrics(U_all, label)
            metrics["_steps"] = steps
            metrics["_comment"] = comment
            all_metrics[label] = metrics

            # 保留用于重叠分析
            all_Us[label] = U_all
            dim_var = U_all.var(dim=0).numpy()
            all_dim_idx[label] = np.argsort(dim_var)[::-1]

            status = "🔴" if metrics["effective_rank"] < 10 else \
                     ("🟡" if metrics["effective_rank"] < 50 else "🟢")
            print(f"[{i+1}/{total}] DONE {label:22s} {status} EffR={metrics['effective_rank']:>7.1f}  "
                  f"RankMe={metrics['rankme']:>7.1f}  CosSim={metrics['cos_sim_mean']:.4f}  "
                  f"Dead={metrics['dead_dims_std001']:>3d}  St={steps}",
                  flush=True)

            # 每完成一个就保存中间结果
            out_dir = f"{ROOT}/analysis_output"
            os.makedirs(out_dir, exist_ok=True)
            with open(f"{out_dir}/unified_analysis_partial.json", "w") as f:
                json.dump({"metrics": all_metrics, "n_done": len(all_metrics)}, f, indent=2)

        except Exception as e:
            print(f"[{i+1}/{total}] ERROR {label}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue

    # ── Pairwise overlap ──
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Computing pairwise dim overlaps...", flush=True)
    overlaps = []
    # I-JEPA v2 相邻 step
    ijepa_keys = sorted([k for k in all_dim_idx if k.startswith("IJEPA_v2_")])
    for i in range(len(ijepa_keys) - 1):
        overlaps.append(compute_dim_overlap(
            ijepa_keys[i], all_dim_idx[ijepa_keys[i]],
            ijepa_keys[i+1], all_dim_idx[ijepa_keys[i+1]], top_k=50))

    # I-JEPA final vs Original
    target_pairs = [
        ("IJEPA_v2_2000", "Original_v2_0400"),
        ("IJEPA_v2_2000", "Original_full"),
        ("IJEPA_v2_2000", "Original_small"),
        ("IJEPA_v2_2000", "IJEPA_small"),
        ("IJEPA_v2_2000", "IJEPA_old_full"),
        ("Original_v2_0400", "Original_full"),
    ]
    for a, b in target_pairs:
        if a in all_dim_idx and b in all_dim_idx:
            overlaps.append(compute_dim_overlap(
                a, all_dim_idx[a], b, all_dim_idx[b], top_k=50))

    # ── 汇总对比表 ──
    print(f"\n{'=' * 110}")
    print(f"{'Label':<24s} {'EffR':>6s} {'RkMe':>6s} {'σ₂/σ₁':>8s} {'Gini':>6s} {'CosSim':>8s} {'Dead':>5s} {'LiDO':>6s} {'CSS':>6s} 判定")
    print("-" * 110)
    for label in sorted(all_metrics.keys()):
        m = all_metrics[label]
        css = m["collapse_score"]
        s = '🔴' if css > 0.7 else ('🟡' if css > 0.4 else '🟢')
        print(f"{label:<24s} {m['effective_rank']:>6.1f} {m['rankme']:>6.1f} "
              f"{m['s2_to_s1_ratio']:>8.4f} {m['sv_gini']:>6.4f} {m['cos_sim_mean']:>8.4f} "
              f"{m['dead_dims_std001']:>5d} {m['lido_eff_dim']:>6.1f} {css:>6.3f}  {s}")

    print(f"\n{'=' * 90}")
    print("维度重叠分析 (Top-50)")
    print("-" * 90)
    for ov in overlaps:
        print(f"  {ov['pair']:<42s} overlap={ov['overlap_count']:>2d}/{ov['top_k']} ({ov['overlap_ratio']:.1%})")

    # ── Save ──
    out_dir = f"{ROOT}/analysis_output"
    os.makedirs(out_dir, exist_ok=True)

    # Full metrics JSON
    out = {
        "timestamp": datetime.now().isoformat(),
        "n_texts": N_TEXTS,
        "latent_steps": LATENT_STEPS,
        "metrics": all_metrics,
        "pairwise_overlaps": overlaps,
    }
    out_path = f"{out_dir}/unified_analysis.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

    # 精简版 CSV 表格
    csv_path = f"{out_dir}/unified_metrics_table.csv"
    keys = ["label", "effective_rank", "rankme", "cos_sim_mean", "s2_to_s1_ratio",
            "sv_gini", "dead_dims_std001", "lido_eff_dim", "collapse_score",
            "intrinsic_dim_twonn", "power_law_alpha", "life_strong", "life_moderate"]
    with open(csv_path, "w") as f:
        f.write(",".join(keys) + "\n")
        for label in sorted(all_metrics.keys()):
            m = all_metrics[label]
            f.write(",".join(str(m.get(k, "")) for k in keys) + "\n")
    print(f"Saved: {csv_path}")

    print("\n✓ ALL DONE")


if __name__ == "__main__":
    main()
