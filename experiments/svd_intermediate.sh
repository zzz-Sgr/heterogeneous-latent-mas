#!/bin/bash
# SVD all intermediate checkpoints from segmented training
# Shows Eff Rank trend over training steps
set -euo pipefail
cd /heterogeneous-latent-mas-main
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/hy-tmp/huggingface_cache

log() { echo "[$(date +%H:%M)] $*"; }

svd_one() {
    local ckpt=$1 label=$2
    python3 << PYEOF
import torch, sys, os
sys.path.insert(0, "/heterogeneous-latent-mas-main")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from models import ModelWrapper
from methods.vision_latent_mas_codec_new import LatentToUniversalEncoder
from train_vision_latent_mas_codec_new import _load_anchor_texts, _infer_hidden_size
import torch.nn.functional as F

device = torch.device("cuda:0")
class _DA: latent_space_realign = 0

ckpt = torch.load("$ckpt", map_location="cpu")
encoders = ckpt.get("encoders", {})
if not encoders: print("$label: no encoder"); exit()
enc_sd = list(encoders.values())[0]

w = ModelWrapper("/heterogeneous-latent-mas-main/model_download/Qwen3-VL-2B-Thinking", device, use_vllm=False, args=_DA())
H = _infer_hidden_size(w)

enc = LatentToUniversalEncoder(h_in=H,d_univ=512,k_univ=1024,n_heads=8,n_layers=6,dropout=0.10).to(device)
enc.load_state_dict(enc_sd, strict=False); enc.eval()

texts = _load_anchor_texts("data/vision_codec_anchor_text/mixed_cose_ocr_prm800k.jsonl")[:20]
all_U = []
for t in texts:
    msgs = [[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":f"Message:\n{t}\n\nAcknowledge."}]]
    _, ids, mask, _ = w.prepare_chat_batch(msgs, add_generation_prompt=True)
    ids, mask = ids.to(device), mask.to(device)
    with torch.no_grad():
        _, lat = w.generate_latent_batch(ids, mask, latent_steps=64, return_latent_embeds=True)
    lat_flat = lat.to(device=device, dtype=torch.float32)
    with torch.no_grad(): U = enc(lat_flat).detach().cpu()
    all_U.append(U)

U_all = torch.cat(all_U, dim=0).reshape(-1, 512)
try:
    _, S, _ = torch.svd(U_all)
    sv = S / (S.sum() + 1e-12)
    entropy = -torch.sum(sv * torch.log(sv + 1e-12))
    er = float(torch.exp(entropy))
    U_norm = F.normalize(U_all, dim=-1)
    cos = float((U_norm[:100] @ U_norm[100:200].T).mean())
    dead = int((U_all.std(dim=0) < 0.01).sum().item())
    steps = ckpt.get("_steps", "?")
    print(f"$label: steps={steps}, eff_rank={er:.2f}, cos_sim={cos:.4f}, dead={dead}")
except Exception as e:
    print(f"$label: SVD error: {e}")
PYEOF
}

for prefix in seg_original seg_ijepa_enc; do
    log "=== SVD: $prefix ==="
    for step in 150 300 450 600 750; do
        CKPT="checkpoints/${prefix}_step${step}.pt"
        if [ -f "$CKPT" ]; then
            svd_one "$CKPT" "${prefix}_${step}"
        else
            log "${prefix}_${step}: not found, skip"
        fi
    done
done

log "=== Final original vs ijepa ==="
if [ -f "checkpoints/seg_original.pt" ]; then
    svd_one "checkpoints/seg_original.pt" "original_final"
fi
if [ -f "checkpoints/seg_ijepa_dec.pt" ]; then
    svd_one "checkpoints/seg_ijepa_dec.pt" "ijepa_dec_final"
fi

log "SVD ALL DONE"
