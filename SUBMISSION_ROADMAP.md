# Parameter Golf Submission Roadmap

**Target**: Beat the current SOTA of **1.1147 BPB** (PR #1019, merged March 30, 2026)

**Your starting point**: 1.3726 BPB on Colab (5 shards, 3000 steps, no TTT)

**Expected on 8xH100**: ~1.12-1.13 BPB

---

## Competition Guardrails

These rules apply to every future submission change in this fork.

- Apply your own techniques first. Borrowed ideas should be used to strengthen your stack, not replace it by default.
- Do not copy code from OpenAI records, leaderboard repos, or other submissions. Reuse ideas, hyperparameters, architecture patterns, evaluation recipes, and quantization strategies only after re-implementing them locally.
- Every claimed technique must map to a real code path, default setting, or clearly documented opt-in flag in `train_gpt.py`.
- README claims must be truthful and split into one of three states: active by default, implemented but optional, or planned / not yet implemented.
- Do not train on validation data. Test-time training must remain score-first and only update on tokens that have already been scored.
- Do not hide external assets, downloads, or extra compute in evaluation. The submission must remain self-contained and reproducible under the challenge rules.
- The final artifact budget is decimal 16,000,000 bytes including code bytes plus compressed model bytes.

## Borrow Strategy, Not Code

Use top submissions and OpenAI records as idea references, not as implementation sources.

1. Read the record README and identify the idea, not the code.
2. Re-implement the idea in this fork using your own structure and naming.
3. Verify the feature is actually active in logs, defaults, or export-time behavior.
4. Only then update README claims or submission notes.

Allowed borrowing:
- run configuration choices
- architecture and quantization ideas
- calibration and evaluation strategies
- artifact-compression choices

Not allowed:
- copying record code blocks or functions
- copying a submission folder and renaming it
- claiming a technique is present when it is only planned

## Technique Application Checklist

Before promoting any technique into a submission recipe, confirm all of the following:

- The technique has a named implementation site in `train_gpt.py`.
- The default value matches the intended submission recipe, or the exact env flag is documented.
- There is a concrete signal that proves it ran: startup config, train log, eval log, or export log.
- The README classifies it correctly as default, optional, or planned.
- The technique does not violate validation, artifact, or evaluation rules.
- If the technique was inspired by another submission, the implementation is local and independently written.

Current high-priority items that must be verified before claiming a new record-style stack:
- XSA scope: current code still defaults to last 4 layers, not all 11.
- BigramHash defaults: current code still defaults to `2048 x 128`, not `3072 x 112`.
- WARMDOWN: current code still defaults to `3500`, not `4000`.
- GPTQ calibration scale: current code still defaults to `32 x 256`, not a larger record-style calibration pass.
- LZMA preset: current code still uses `preset=6`, not `preset=9`.
- Any pruning or reordering claim must stay out of README unless the code path is actually present.

---

## Current Status

| Metric | Your Code | New SOTA (#1019) | Gap |
|--------|-----------|------------------|-----|
| Architecture | 11L/512d | Same | ✓ |
| Activation | LeakyReLU(0.5)² | Same | ✓ |
| XSA | Last 4 layers | **All 11 layers** | ✗ |
| BigramHash | 2048 (default) | **3072 × 112** | ✗ |
| RoPE | Partial 16/64 | Same | ✓ |
| LN Scale | 1/√(layer+1) | Same | ✓ |
| VE128 | Layers 9-10 | Same | ✓ |
| TTT | Disabled by default, optional | **Disabled** | ✓ |
| Quantization | GPTQ int6 + Hessian capture + FWHT rotation | Full Hessian GPTQ int6 | Partial |
| GPTQ calibration | AR self-gen `32 x 256` | Larger record-style calibration | ✗ |
| LZMA | preset=6 | **preset=9** | ✗ |
| WARMDOWN_ITERS | 3500 | **4000** | ✗ |
| Claims policy | Mixed defaults vs options in docs | Truthful status split | ✗ |
| Code size | 98KB | ~89KB | Blocker (over 16MB) |
| **Artifact** | **16.05MB (OVER)** | **15.91MB (OK)** | **BLOCKER** |

---

## Required Changes (In Order)

### 1. ✅ TRIM CODE (CRITICAL — Required for submission)

Your code is 98KB (over budget). Target: 89KB.

**What to remove:**
- [ ] Flash attention fallback try/except (lines 26-38): ~15 lines, 0.3KB
- [ ] ATTN_RES hyperparameter and all associated code (already disabled, can remove entirely):
  - Line 148: delete `attn_res` parameter
  - Lines 919-923: delete `attn_res_queries` creation in GPT.__init__
  - Lines 964-1004: simplify `_run_blocks` to remove attn_res logic
  - Line 482: remove from CONTROL_TENSOR_NAME_PATTERNS
  - Estimated: ~80 lines, 2KB

- [ ] Dead imports: `uuid` (line 11, only used for default RUN_ID)
  - Lines 77: inline string instead of uuid.uuid4()
  - Estimated: ~2 lines, 0.1KB

- [ ] Verbose comments and docstrings: ~100 lines, 2KB

- [ ] Dead code paths (test whether these are actually used):
  - `_INT8_CLIP_Q` and `quantize_float_tensor` (lines 485-501): used only if GPTQ disabled
  - Keep for safety, but check if needed

**Expected reduction: ~9-15KB** → Should get you to ~83-89KB

**How to trim:**
```bash
# Count current size
wc -c train_gpt.py  # current: 98745
# After trimming target: ~89000
```

### 2. ✅ DISABLE ATTN_RES (already done, but verify)

Check line 148 is set to `"0"`:
```python
attn_res = bool(int(os.environ.get("ATTN_RES", "0")))  # ✓ Already fixed
```

### 3. ✅ DISABLE TTT BY DEFAULT

The new SOTA found TTT is **neutral or negative** on this stack. Disable it by default:

**Line 140**: Change from:
```python
ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "0")))  # Current
```
to:
```python
ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "0")))  # Already "0" ✓
```

**No change needed** — it's already disabled by default. Good!

### 4. ✅ CHANGE XSA TO ALL LAYERS

**Line 130**: Change from:
```python
xsa_last_n = int(os.environ.get("XSA_LAST_N", 4))
```
to:
```python
xsa_last_n = int(os.environ.get("XSA_LAST_N", 11))  # All 11 layers
```

Why: Allows cross-position information mixing from layer 0 with zero parameter cost. Gives ~0.003 BPB improvement.

### 5. ✅ UPDATE BIGRAM TO 3072×112

**Line 128**: Change from:
```python
bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 2048))
```
to:
```python
bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 3072))
```

**Line 129**: Change from:
```python
bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))
```
to:
```python
bigram_dim = int(os.environ.get("BIGRAM_DIM", 112))
```

Why: Larger bigram vocabulary (3072 vs 1536 before) with smaller embedding dimension (112 vs 128) fits better in artifact. Estimated +0.002 BPB.

### 6. ✅ INCREASE LZMA COMPRESSION

**Line 2076**: Change from:
```python
quant_blob = lzma.compress(quant_raw, preset=6)
```
to:
```python
quant_blob = lzma.compress(quant_raw, preset=9)
```

Why: Better compression ratio (slower, but only done once at eval end). Saves ~100-200KB artifact.

### 7. ✅ INCREASE WARMDOWN

**Line 83**: Change from:
```python
warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3500))
```
to:
```python
warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 4000))
```

Why: Longer warmdown period gives more time for LR decay. Marginal improvement.

### 8. ✅ INCREASE GPTQ CALIBRATION FOR RECORD-STYLE RUNS

For the target 8xH100 recipe, use a larger self-generated calibration pass:

```bash
GPTQ_CALIB_SEQS=64
GPTQ_CALIB_LEN=2048
```

Why: your current GPTQ path is real, but the default calibration pass is still much smaller than stronger record-style GPTQ recipes. Keep the calibration self-generated to stay within the challenge rules.

---

## Updated Run Command

Replace your current run command with this (for 8xH100):

```bash
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
SEED=1337 \
RUN_ID=sota_matching_1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For Colab (1 GPU, reduced batch):
```bash
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
MUON_WD=0.04 \
ADAM_WD=0.04 \
MATRIX_LR=0.025 \
SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 \
WARMDOWN_ITERS=4000 \
ITERATIONS=3000 \
MAX_WALLCLOCK_SECONDS=1800 \
EVAL_STRIDE=64 \
TRAIN_BATCH_TOKENS=32768 \
VAL_BATCH_SIZE=32768 \
SEED=1337 \
RUN_ID=colab_test_3000step \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

---

## Expected Improvements

From baseline (your current Colab run: 1.3726 BPB) to 8xH100 with these changes:

| Change | Estimated Δ BPB | Cumulative |
|--------|-----------------|------------|
| Baseline (5 shards, 3000 steps) | — | 1.3726 |
| → Full 80 shards + 9000 steps | -0.18 | 1.1926 |
| + XSA all layers | -0.003 | 1.1896 |
| + BigramHash 3072×112 | -0.002 | 1.1876 |
| + WARMDOWN 4000 | -0.001 | 1.1866 |
| + LZMA preset=9 | (compression, not loss) | 1.1866 |
| **+ GPTQ Full Hessian (your rotation)** | **-0.003 to -0.005** | **1.1816 - 1.1836** |
| **Target to beat SOTA** | **<= 1.1097** | Need new technique |

### Realistic Outcome

Your submission with these changes will likely score **~1.118-1.121 BPB** — not quite beating the SOTA (1.1147), but close enough to be a strong **non-record submission**.

### To Actually Beat the Record

You'd need one of:
1. **Better TTT recipe** (current SOTA dropped it, but maybe you can make it work): +0.005-0.010 BPB
2. **AR self-gen calibration** (like SOTA uses): already built in, but your GPTQ rotation is different
3. **Stride-16 eval** instead of stride-64: +0.005-0.010 BPB but 4x longer eval time
4. **New architecture idea** (ternary, binary, depth recurrence, etc.): +0.010+ BPB but massive R&D

---

## Checklist: Changes to Make

### Code Changes

- [ ] 1. Trim code to ~89KB (remove attn_res, flash fallback, dead code)
- [ ] 2. Line 128: `BIGRAM_VOCAB_SIZE` default from 2048 → 3072
- [ ] 3. Line 129: `BIGRAM_DIM` default from 128 → 112
- [ ] 4. Line 130: `XSA_LAST_N` default from 4 → 11
- [ ] 5. Line 83: `WARMDOWN_ITERS` default from 3500 → 4000
- [ ] 6. Line 2076: LZMA preset from 6 → 9
- [ ] 7. Increase GPTQ calibration target for record-style runs (`GPTQ_CALIB_SEQS`, `GPTQ_CALIB_LEN`)
- [ ] 8. Verify line 140: `TTT_ENABLED` default is "0"
- [ ] 9. Verify line 148: `ATTN_RES` default is "0"
- [ ] 10. Update `README.md` claims so defaults, options, and planned ideas are not mixed

### Testing on Colab (before GPU rental)

- [ ] 1. Upload modified train_gpt.py
- [ ] 2. Download 5 shards + validation data
- [ ] 3. Run smoke test (50 steps): verify loss decreases
- [ ] 4. Run 3000-step Colab test with updated params
- [ ] 5. Check artifact size (should be < 16MB)
- [ ] 6. Record BPB number

### GPU Submission (8xH100, ~$40-50)

- [ ] 1. Rent 1xH100 pod on RunPod for smoke test (5 min, $0.50)
- [ ] 2. Rent 8xH100 for 3 full runs (seeds: 1337, 42, 2025)
- [ ] 3. Each run: ~18-19 min (600s train + 120s eval)
- [ ] 4. Total: ~60 min across 3 seeds
- [ ] 5. Save logs
- [ ] 6. Calculate mean and std BPB across 3 seeds
- [ ] 7. Create submission directory: `records/track_10min_16mb/2026-04-02_XSA-all_BigramHash3072_AR-GPTQ/`

### Submission

- [ ] 1. Copy train_gpt.py to submission directory
- [ ] 2. Update submission.json with actual val_bpb and artifact size
- [ ] 3. Update README.md with results, run command, changes from SOTA
- [ ] 4. Include all 3 training logs
- [ ] 5. Create PR to openai/parameter-golf

---

## File Status After Changes

```
train_gpt.py:
  - Before trim: 98,745 bytes (2,163 lines)
  - After trim: ~89,000 bytes (2,050 lines) [target]
  - Syntax: python3 -m py_compile train_gpt.py

Model artifact on 8xH100:
  - Code: ~89KB
  - Model: ~15.9MB (with LZMA preset=9)
  - Total: ~15.98MB ✓ (under 16MB limit)
```

---

## Questions?

- **What if artifact is still over 16MB?** Remove more dead code (GPTQ is optional, fallback to simple int6)
- **What if loss doesn't decrease?** Check data path, tokenizer, hyperparams
- **What if BPB is 1.12?** Still submit as non-record; it's a valid contribution
- **How to get to 1.11?** Need TTT working better OR a novel architecture

Good luck! 🚀
