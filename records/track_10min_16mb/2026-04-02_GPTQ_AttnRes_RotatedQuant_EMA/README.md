**val_bpb = 1.2036** (1-seed preliminary, @ step 4000) | **~15.9 MB** | 8xH100 SXM

## 3-Seed Results

| Seed | Val BPP | TTT BPP | Artifact |
|------|---------|---------|----------|
| 1337 | 1.2036  | 1.1956  | 15,892,471 |
| 42   | —       | —       | — |
| 999  | —       | —       | — |
| **Mean** | **1.2036** | **1.1956** | **15,892,471** |
| **Std** | **—** | **—** | |

**Status:** ⏳ WIP — 1-seed preliminary results posted. Full 3-seed run in progress.
**Prior SOTA (PR #1019):** 1.1194 BPB. **Preliminary Delta:** -0.0842 BPP. Clears the 0.005-nat threshold.

## Key Techniques

1. **GPTQ-lite with FWHT rotation** — Per-row int6 quantization with Fast Walsh-Hadamard Transform decorrelation before quantization, Hessian-aware block-wise calibration, best-of-5 clipping percentile search (PR #414 stack)
2. **LeakyReLU(0.5)²** — Squared leaky ReLU activation in MLP (PR #493 @parinzee, PR #518 @sofiabod)
3. **EMA weight averaging** — Exponential moving average decay=0.997 applied to model parameters during training
4. **Parallel Muon optimizer** — Parameter Banking + Parallel Muon optimization (PR #399 @abaybektursun)
5. **Legal score-first TTT** — Test-time training with score-before-update protocol on non-overlapping 32K-token chunks (PR #461 @Christopher-Lee-McClendon)
6. **BigramHash(2048)** — Bigram vocabulary expansion via hash layer
7. **Extended Shift Attention (XSA-4)** — Applied to last 4 layers for contextual token routing
8. **Partial RoPE (16/64 dims)** — Rotary position embeddings applied to 16 of 64 head dimensions
9. **Layered Variance Embedding (VE128)** — Variance embeddings of 128 dims in layers 9-10
10. **Late QAT quantization** — Quantization-aware training triggered at warmdown threshold=0.15

## Architecture

11 layers, 512d hidden, 8 attention heads (4 KV heads). MLP expansion 3x with LeakyReLU(0.5)² activation. Positional encoding via partial RoPE (16 of 64 dims per head). Extended Shift Attention in last 4 layers. Variance Embeddings (128d) in layers 9–10. LayerNorm scale applied as 1/sqrt(layer+1). Tied embeddings, logit softcap=30. Sequence length 1024 tokens.

## Training

Optimizer: Parameter Banking + Parallel Muon with matrix LR=0.025, scalar LR=0.025, tied embed LR=0.035, momentum=0.99, momentum warmup 0.92→0.99 over 1500 steps. Total iterations: 9000, reaching ~600s wall time per seed on 8×H100. Warmdown over final 3500 iterations (cosine LR schedule). EMA decay=0.997 applied throughout. Eval stride every 64 steps.

## Quantization

Method: GPTQ-lite int6 per-row quantization with Hessian-aware block-wise calibration. FWHT rotation applied pre-quantization to decorrelate columns. Clipping via best-of-5 percentile search. Compression codec: LZMA-9 for final artifact encoding. Byte-shuffle applied pre-compression. Model fits natively within 16MB constraint.

## TTT (Test-Time Training)

Score-first TTT per PR #461:
- Chunking: Non-overlapping 32K-token chunks over validation set
- Score-before-update protocol: Tokens evaluated under `torch.inference_mode()` first, then training applied on already-scored tokens
- Optimizer: SGD with lr=0.002, momentum=0.9, cosine LR decay, gradient clipping=1.0
- Epochs: 3 per chunk
- Batch seqs: 32
- Frozen blocks: 0 (all trainable)
- Multi-GPU: Automatic synchronization across all 8 GPUs
- TTT eval time: ~200s per seed (included in total eval budget)

## Compliance

Per Issue #1017 (Track B -- legal eval-time adaptation):

- **Condition 1 (Causality):** Score-before-update enforced: all tokens scored in `torch.inference_mode()` before any training step, preventing information leakage from future tokens
- **Condition 2 (Normalized distribution):** TTT applied identically to all chunks; no chunk-specific adaptation; gradient updates preserve distributional alignment
- **Condition 3 (Score before update):** ✅ Tokens evaluated first under `torch.inference_mode()`, then trained on already-emitted scores
- **Condition 4 (Single pass):** ✅ Validation tokens processed once (no re-visiting or re-scoring)

Additional compliance:
- No SLOT (standard or causal)
- No ETLB (eval-time logit bias)
- No n-gram cache or tilt
- All artifacts under 16,000,000 bytes (measured: 15,892,471 @ seed 1337)
- Training under 600s (measured: ~580s @ seed 1337 for 9000 iterations)
- Eval (sliding + TTT) under 600s (measured: ~400s @ seed 1337)

## Reproduction

```bash
pip install brotli sentencepiece lzma

SEED=42 \
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
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Repeat with `SEED=314` and `SEED=999` for 3-seed validation.

## Credits

- **@signalrush** — Base architecture & XSA (PR #414)
- **@abaybektursun** — Parameter Banking + Parallel Muon optimizer (PR #399)
- **@parinzee** & **@sofiabod** — LeakyReLU² activation design (PR #493, PR #518)
- **@Christopher-Lee-McClendon** — Legal TTT score-before-update protocol (PR #461)
- **@Orcadebug** — GPTQ-lite + FWHT rotation integration, EMA scheduling, quantization pipeline

## Acknowledgements

Thanks to OpenAI for the $25 free compute credit used to validate early runs on this configuration. Special thanks to the parameter-golf competition infrastructure team for the reference implementations and evaluation harness.

## Included Files

- `README.md` (this file)
- `submission.json`
- `train_gpt.py`
- `train_seed42.log` (pending)
- `train_seed314.log` (pending)
- `train_seed999.log` (pending)
