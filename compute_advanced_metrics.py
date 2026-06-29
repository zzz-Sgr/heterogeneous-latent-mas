#!/usr/bin/env python3
"""Compute Smooth RankMe, NLA Rank, Spectral Gap, SV Gini on all checkpoints."""
import json, math
import numpy as np

with open("/heterogeneous-latent-mas-main/analysis_output/analysis_results.json") as f:
    data = json.load(f)

labels = ["original", "vicreg_01", "vicreg_05", "ijepa"]
names  = ["Original", "VICReg 0.1", "VICReg 0.5", "I-JEPA"]

print("=" * 90)
print(f"{'Metric':<32s} {'Original':>13s} {'VICReg 0.1':>13s} {'VICReg 0.5':>13s} {'I-JEPA':>13s}")
print("=" * 90)

# 1) Smooth RankMe for different tau
for tau in [0.5, 1.0, 2.0]:
    vals = []
    for lab in labels:
        sv = np.array(data[lab]["singular_values_top10"])
        sv_tau = sv ** tau
        p = sv_tau / (sv_tau.sum() + 1e-12)
        entropy = -np.sum(p * np.log(p + 1e-12))
        vals.append(np.exp(entropy))
    line = "SmoothRankMe(tau={:.1f})".format(tau).ljust(32)
    line += " {:>13.2f}" * len(vals)
    print(line.format(*vals))

# 2) NLA Stable Rank
nla_vals = []
for lab in labels:
    sv = np.array(data[lab]["singular_values_top10"])
    # Estimate tail with exponential decay
    decay = sv[-2] / max(sv[-3], 1e-12) if sv[-3] > 0 else 1.0
    tail = sv[-1] * decay / (1.0 - decay) if decay < 1.0 else 0.0
    total_fro = sv[0]**2 + np.sum(sv[1:]**2) + max(tail, 0)
    nla = total_fro / (sv[0]**2 + 1e-12)
    nla_vals.append(nla)
line = "NLA Stable Rank (approx)".ljust(32)
for v in nla_vals:
    line += " {:>13.2f}".format(v)
print(line)

# 3) Spectral Gap - max ratio between consecutive SVs
gap_vals = []
gap_pos  = []
for lab in labels:
    sv = np.array(data[lab]["singular_values_top10"])
    gaps = [sv[i] / max(sv[i+1], 1e-12) for i in range(len(sv)-1)]
    max_gap = max(gaps)
    max_idx = gaps.index(max_gap) + 1
    gap_vals.append(max_gap)
    gap_pos.append(max_idx)
line = "Spectral Gap max(sigma_k/sigma_k+1)".ljust(32)
for v in gap_vals:
    line += " {:>13.1f}".format(v)
print(line)
line = "  at position k".ljust(32)
for p in gap_pos:
    line += " {:>13d}".format(p)
print(line)

# 4) Energy beyond sigma_1 (%)
energy_vals = []
for lab in labels:
    sv = np.array(data[lab]["singular_values_top10"])
    beyond = (sv.sum() - sv[0]) / (sv.sum() + 1e-12)
    energy_vals.append(beyond * 100)
line = "Energy beyond sigma_1 (%)".ljust(32)
for v in energy_vals:
    line += " {:>12.1f}%".format(v)
print(line)

# 5) SV Gini coefficient (0=equal, 1=one dominant)
gini_vals = []
for lab in labels:
    sv = sorted(data[lab]["singular_values_top10"])
    n = len(sv)
    gini = (2 * np.sum((np.arange(1, n+1)) * sv)) / (n * np.sum(sv)) - (n + 1) / n
    gini_vals.append(gini)
line = "SV Gini (0=equal, 1=dominated)".ljust(32)
for v in gini_vals:
    line += " {:>13.4f}".format(v)
print(line)

# 6) sigma_2 / sigma_1 ratio
s2s1_vals = []
for lab in labels:
    sv = np.array(data[lab]["singular_values_top10"])
    s2s1_vals.append(sv[1] / max(sv[0], 1e-12))
line = "sigma_2 / sigma_1 ratio".ljust(32)
for v in s2s1_vals:
    line += " {:>13.4f}".format(v)
print(line)

print("=" * 75)
print()
print("Ideal ranges:")
print("  SmoothRankMe:  > 10")
print("  NLA Rank:      > 10")
print("  Spectral Gap:  < 5x")
print("  Energy > s1:   > 30%")
print("  SV Gini:       < 0.5")
print("  s2/s1 ratio:   > 0.3")
