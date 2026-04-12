# GPTQ-lite + Rotated Quant + EMA + LeakyReLU² + TTT

**val_bpb: WIP** (single seed: ~1.20 @ step 4000) | **~15.9 MB** (est.) | 8xH100 SXM

## Results (8xH100 80GB SXM)

| Seed | Steps | step_avg | Mid-train (step 4000) | Final (est.) | Status |
|------|-------|----------|----------------------|--------------|--------|
| 1337 | 6500+ | ~90ms | **1.2036** | TBD | ✅ Completed (pod exited ~1228s) |
| 42 | — | — | — | — | ⏳ Pending |
| 2025 | — | — | — | — | ⏳ Pending |
| **Mean** | — | — | — | **~1.20** (est.) | **Awaiting 3-seed run** |

## Key Innovations

### 1. GPTQ-lite with FWHT Rotation

Standard per-row int6 quantization is enhanced with:
- **FWHT rotation** (Fast Walsh-Hadamard Transform) applied before quantization to decorrelate weight columns, reducing quantization error
- **Hessian-aware GPTQ**: Calibration data is generated via autoregressive sampling, Hessian matrices are collected per-layer, and GPTQ block-wise quantization minimizes reconstruction error weighted by input statistics
- Best-of-5 clipping percentile search for optimal scale factors

### 2. Architecture (PR #414 stack)

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV) |
| MLP | 3x with **LeakyReLU(0.5)²** |
| BigramHash | 2048 |
| XSA | Last 4 layers |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/sqrt(layer+1) |
| VE128 | Layers 9-10 |
| Weight avg | EMA(0.997) |
| Quantization | **GPTQ-lite int6 + FWHT rotation + lzma** |
| Optimizer | Parameter Banking + Parallel Muon |

### 3. Legal TTT Protocol

Score-first TTT following PR #461:
1. Val tokens split into non-overlapping 32K-token chunks
2. For each chunk: SCORE under `torch.inference_mode()`, then TRAIN on already-scored tokens
3. SGD(lr=0.002, momentum=0.9), 3 epochs, cosine LR decay, grad clip 1.0
4. Last chunk scored but never trained on

## Run Command

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
AVG_MODE=ema EMA_DECAY=0.997 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
GPTQ_ENABLED=1 GPTQ_CALIB_SEQS=32 GPTQ_CALIB_LEN=256 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Differences from SOTA (1.1194 BPB)

1. **GPTQ-lite quantization** with Hessian-aware block-wise quantization (SOTA uses simple per-row int6)
2. **FWHT rotation** before quantization to reduce quantization error
3. **BigramHash(2048)** vs SOTA's 1536
4. **Post-reduce-scatter gradient clipping** (clips on averaged gradients, arguably more correct)
5. **Late QAT** at warmdown threshold 0.15

## Credits

- **LeakyReLU² activation**: PR #493 by @parinzee, PR #518 by @sofiabod
- **Optimizer (Parameter Banking + Parallel Muon)**: PR #399 by @abaybektursun
- **TTT recipe**: PR #461 by @Christopher-Lee-McClendon
- **Base model**: PR #414 by @signalrush
- **GPTQ quantization**: Adapted from Frantar et al. (2022)
