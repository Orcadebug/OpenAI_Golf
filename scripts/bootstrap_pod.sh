#!/bin/bash
set -euo pipefail

# === CONFIGURATION (injected by caller via environment) ===
: "${HF_TOKEN:?HF_TOKEN is required}"
: "${GITHUB_TOKEN:?GITHUB_TOKEN is required}"
: "${GITHUB_REPO:?GITHUB_REPO is required}"

echo "=== [1/5] GPU CHECK ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "import torch; print('GPUs:', torch.cuda.device_count(), '| PyTorch:', torch.__version__)"

echo "=== [2/5] CLONE REPO ==="
cd /workspace
if [ -d "parameter-golf" ]; then
    echo "Repo already exists, pulling latest..."
    cd parameter-golf && git pull && cd ..
else
    git clone "https://${GITHUB_TOKEN}@github.com/${GITHUB_REPO}.git" parameter-golf
fi
cd parameter-golf

echo "=== [3/5] INSTALL PACKAGES ==="
pip install --quiet kernels sentencepiece brotli 2>&1 | tail -5
python3 -c "from flash_attn_interface import flash_attn_func; print('FlashAttention3: OK')" \
    || echo "WARNING: FlashAttention3 not available, will fall back to SDPA"

echo "=== [4/5] DOWNLOAD FINEWEB DATASET ==="
HF_TOKEN="${HF_TOKEN}" \
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
echo "Train shards: $(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l)"
echo "Val shards:   $(ls data/datasets/fineweb10B_sp1024/fineweb_val_*.bin 2>/dev/null | wc -l)"

echo "=== [5/5] PREFLIGHT CHECKS ==="
python3 -m py_compile train_gpt.py && echo "train_gpt.py: syntax OK"
CODE_BYTES=$(wc -c < train_gpt.py)
echo "train_gpt.py size: ${CODE_BYTES} bytes (budget: ~90000)"
if [ "$CODE_BYTES" -gt 100000 ]; then
    echo "WARNING: Code is over 100KB — apply SUBMISSION_ROADMAP trims before submitting"
fi

echo "=== BOOTSTRAP COMPLETE ==="
