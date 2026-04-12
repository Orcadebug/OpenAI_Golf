#!/bin/bash
set -euo pipefail

SEED="${1:-1337}"
LOG_FILE="/workspace/parameter-golf/train_seed${SEED}.log"

cd /workspace/parameter-golf

echo "=== STARTING TRAINING SEED=${SEED} at $(date) ===" | tee "$LOG_FILE"

NUM_LAYERS=11 \
BIGRAM_VOCAB_SIZE=3072 \
BIGRAM_DIM=112 \
XSA_LAST_N=11 \
AVG_MODE=ema \
EMA_DECAY=0.997 \
ROPE_DIMS=16 \
LN_SCALE=1 \
LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 \
VE_DIM=128 \
VE_LAYERS=9,10 \
TTT_ENABLED=0 \
GPTQ_ENABLED=1 \
GPTQ_CALIB_SEQS=64 \
GPTQ_CALIB_LEN=2048 \
GPTQ_TEMPERATURE=0.8 \
MUON_WD=0.04 \
ADAM_WD=0.04 \
MATRIX_LR=0.025 \
SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=4000 \
ITERATIONS=9000 \
MAX_WALLCLOCK_SECONDS=600 \
EVAL_STRIDE=64 \
TRAIN_BATCH_TOKENS=786432 \
VAL_BATCH_SIZE=524288 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
RUN_ID="sota_attempt_seed${SEED}" \
SEED="${SEED}" \
    torchrun --standalone --nproc_per_node=8 train_gpt.py \
    2>&1 | tee -a "$LOG_FILE"

echo "=== TRAINING COMPLETE SEED=${SEED} at $(date) ===" | tee -a "$LOG_FILE"

# Parse and report BPB
FINAL_BPB=$(grep "final_int6_sliding_window_exact" "$LOG_FILE" \
    | grep -oP "val_bpb:\K[0-9.]+" | tail -1 || true)
if [ -z "$FINAL_BPB" ]; then
    FINAL_BPB=$(grep "final_int6_roundtrip_exact" "$LOG_FILE" \
        | grep -oP "val_bpb:\K[0-9.]+" | tail -1 || true)
fi

TOTAL_BYTES=$(grep "Total submission size" "$LOG_FILE" \
    | grep -oP "[0-9]+ bytes" | grep -oP "[0-9]+" | tail -1 || true)

echo "==============================" | tee -a "$LOG_FILE"
echo "SEED:        ${SEED}" | tee -a "$LOG_FILE"
echo "FINAL_BPB:   ${FINAL_BPB:-N/A}" | tee -a "$LOG_FILE"
echo "TOTAL_BYTES: ${TOTAL_BYTES:-N/A}" | tee -a "$LOG_FILE"
if [ -n "$TOTAL_BYTES" ] && [ "$TOTAL_BYTES" -gt 16000000 ]; then
    echo "WARNING: Artifact OVER budget (${TOTAL_BYTES} > 16,000,000)" | tee -a "$LOG_FILE"
else
    echo "Artifact: within 16MB budget" | tee -a "$LOG_FILE"
fi
echo "==============================" | tee -a "$LOG_FILE"
