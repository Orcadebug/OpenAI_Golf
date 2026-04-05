# `train_gpt.py` Fix Summary

Date: 2026-03-31

## What was fixed

- MTP is now a real training-only auxiliary path.
  - The MTP heads are optimized.
  - They are included in optimizer coverage checks.
  - They are still excluded from export, but that is now logged explicitly.

- Averaging is now explicit and unambiguous.
  - `AVG_MODE` selects one mode: `ema`, `swa`, `lawa`, or `none`.
  - Conflicting legacy flags like `EMA_ENABLED=1` and `SWA_ENABLED=1` now fail fast unless `AVG_MODE` is set.
  - Explicit overrides and fallbacks are logged.

- Distributed gradient handling was corrected.
  - Replicated grads are all-reduced before clipping.
  - Muon bank grads are prepared before clipping.
  - Clipping now happens on synchronized gradients instead of local pre-sync grads.

- Export and evaluation logs now match the actual artifact path.
  - The script now logs `int6 + lzma` honestly.
  - Stale `final_int8_zlib_roundtrip_exact` logging was removed.

- Non-FlashAttention fallback for grouped-query attention was fixed.
  - The fallback now uses grouped-query attention correctly when `num_heads != num_kv_heads`.

## Validation performed

- Static validation:
  - `python3 -m py_compile train_gpt.py`
  - AST parse check

- Runtime validation:
  - Created a local `.venv`
  - Installed `torch` and `sentencepiece`
  - Ran a CPU smoke test that verified:
    - `train_gpt.py` imports successfully
    - conflicting averaging flags fail fast
    - MTP heads receive gradients
    - optimizer parameter coverage passes
    - Muon clip/step path runs
  - Smoke test result: `smoke_ok`

## Remaining limits

- No full CUDA training run was executed here.
- No multi-rank distributed runtime test was executed here.
- Historical `records/...` scripts were intentionally left untouched.

## Files changed

- Modified: `train_gpt.py`
- Added: `TRAIN_GPT_FIX_SUMMARY.md`
- Left untouched: `train_gpt_baseline.py`
