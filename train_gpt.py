from __future__ import annotations
import copy
import glob
import io
import lzma
import math
import os
import random
import sys
import time
import zlib
from pathlib import Path
try:
    import brotli
    _BROTLI_AVAILABLE = True
except ImportError:
    brotli = None
    _BROTLI_AVAILABLE = False
try:
    import zstandard
    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
except ImportError:
    def flash_attn_3_func(q, k, v, causal=True):
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=causal,
            enable_gqa=(q.size(1) != k.size(1)),
        )
        return y.transpose(1, 2)

VALID_AVG_MODES = {"ema", "swa", "lawa", "none"}


def _optional_env_flag(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    return bool(int(raw))


def resolve_avg_mode(raw_mode: str, ema_enabled: bool | None, swa_enabled: bool | None, lawa_enabled: bool | None) -> str:
    mode = raw_mode.strip().lower()
    if mode:
        if mode not in VALID_AVG_MODES:
            raise ValueError(f"AVG_MODE must be one of {sorted(VALID_AVG_MODES)}, got {raw_mode!r}")
        return mode
    enabled = [name for name, flag in (("ema", ema_enabled), ("swa", swa_enabled), ("lawa", lawa_enabled)) if flag is True]
    if len(enabled) > 1:
        raise ValueError(
            "EMA_ENABLED, SWA_ENABLED, and LAWA_ENABLED are mutually exclusive when AVG_MODE is unset. "
            f"Got multiple enabled legacy flags: {enabled}"
        )
    if lawa_enabled is True:
        return "lawa"
    if swa_enabled is True:
        return "swa"
    if ema_enabled is True:
        return "ema"
    if ema_enabled is False:
        return "none"
    return "ema"

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", f"run_{time.time_ns()}")
    seed = int(os.environ.get("SEED", 1337))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 4000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 500))
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 4000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 5.25))
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = float(os.environ.get("MLP_MULT", 4.0))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.035))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.025))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.025))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    mtp_num_heads = int(os.environ.get("MTP_NUM_HEADS", 0))
    mtp_loss_weight = float(os.environ.get("MTP_LOSS_WEIGHT", 0.2))
    muon_beta2 = float(os.environ.get("MUON_BETA2", 0.95))
    avg_mode = os.environ.get("AVG_MODE", "")
    ema_enabled = _optional_env_flag("EMA_ENABLED")
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))
    swa_enabled = _optional_env_flag("SWA_ENABLED")
    swa_every = int(os.environ.get("SWA_EVERY", 50))
    lawa_enabled = _optional_env_flag("LAWA_ENABLED")
    lawa_k = int(os.environ.get("LAWA_K", 10))
    lawa_freq = int(os.environ.get("LAWA_FREQ", 100))
    muon_wd = float(os.environ.get("MUON_WD", 0.04))
    adam_wd = float(os.environ.get("ADAM_WD", 0.04))
    qat_enabled = bool(int(os.environ.get("QAT_ENABLED", "0")))
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 2048))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 112))
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 4))
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    dtg_enabled = bool(int(os.environ.get("DTG_ENABLED", "0")))
    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))
    ve_enabled = bool(int(os.environ.get("VE_ENABLED", "1")))
    ve_dim = int(os.environ.get("VE_DIM", 128))
    ve_layers = os.environ.get("VE_LAYERS", "9,10")
    gated_attention = bool(int(os.environ.get("GATED_ATTENTION", "0")))
    value_residual = bool(int(os.environ.get("VALUE_RESIDUAL", "0")))
    recurrent_layer_start = int(os.environ.get("RECURRENT_LAYER_START", 2))
    recurrent_layer_end = int(os.environ.get("RECURRENT_LAYER_END", 4))
    recurrent_extra_passes = int(os.environ.get("RECURRENT_EXTRA_PASSES", 2))
    parallel_residual_start = int(os.environ.get("PARALLEL_RESIDUAL_START", 6))
    attn_residual_enabled = bool(int(os.environ.get("ATTN_RESIDUAL_ENABLED", "1")))
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "1")))
    ttt_lr = float(os.environ.get("TTT_LR", 0.004))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 32768))
    ttt_freeze_blocks = int(os.environ.get("TTT_FREEZE_BLOCKS", 2))
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.9))
    ttt_batch_seqs = int(os.environ.get("TTT_BATCH_SEQS", 32))
    ttt_grad_clip = float(os.environ.get("TTT_GRAD_CLIP", 1.0))
    gptq_enabled = bool(int(os.environ.get("GPTQ_ENABLED", "0")))
    gptq_calib_seqs = int(os.environ.get("GPTQ_CALIB_SEQS", "32"))
    gptq_calib_len = int(os.environ.get("GPTQ_CALIB_LEN", "256"))
    gptq_temperature = float(os.environ.get("GPTQ_TEMPERATURE", "0.8"))
    gptq_block_size = int(os.environ.get("GPTQ_BLOCK_SIZE", "128"))
    gptq_damp = float(os.environ.get("GPTQ_DAMP", "0.01"))
    turbo_rounds = int(os.environ.get("TURBO_ROUNDS", "1"))
    artifact_budget_bytes = int(os.environ.get("ARTIFACT_BUDGET_BYTES", "16000000"))
    brotli_enabled = bool(int(os.environ.get("BROTLI_ENABLED", "1" if _BROTLI_AVAILABLE else "0")))
    compare_compressors = bool(int(os.environ.get("COMPARE_COMPRESSORS", "1")))

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    was_2d = G.ndim == 2
    if was_2d:
        G = G.unsqueeze(0)
    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    if was_2d:
        X = X.squeeze(0)
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                 nesterov=nesterov, weight_decay=weight_decay),
        )
        self._built = False

    def _build(self):
        self._distributed = dist.is_available() and dist.is_initialized()
        self._world_size = dist.get_world_size() if self._distributed else 1
        self._rank = dist.get_rank() if self._distributed else 0
        ws = self._world_size

        self._bank_meta = []
        for group in self.param_groups:
            for p in group["params"]:
                B = p.shape[0]
                padded_B = ((B + ws - 1) // ws) * ws
                shard_B = padded_B // ws
                tail = p.shape[1:]
                dev = p.device
                self._bank_meta.append({
                    'p': p,
                    'B': B,
                    'padded_grad': torch.zeros(padded_B, *tail, device=dev, dtype=torch.bfloat16),
                    'shard': torch.zeros(shard_B, *tail, device=dev, dtype=torch.bfloat16),
                    'shard_mom': torch.zeros(shard_B, *tail, device=dev, dtype=torch.bfloat16),
                    'full_update': torch.zeros(padded_B, *tail, device=dev, dtype=torch.bfloat16),
                    'scale': max(1, p.shape[-2] / p.shape[-1]) ** 0.5,
                })
        self._bank_meta.sort(key=lambda m: -m['p'].numel())
        self._built = True

    def launch_reduce_scatters(self):
        if not self._built:
            self._build()
        if not self._distributed:
            return
        self._rs_futures = []
        for m in self._bank_meta:
            p = m['p']
            if p.grad is None:
                self._rs_futures.append(None)
                continue
            pg = m['padded_grad']
            pg[:m['B']].copy_(p.grad.bfloat16())
            if pg.shape[0] > m['B']:
                pg[m['B']:].zero_()
            fut = dist.reduce_scatter_tensor(m['shard'], pg, op=dist.ReduceOp.AVG, async_op=True)
            self._rs_futures.append(fut)

    def wait_reduce_scatters(self) -> None:
        if not self._built:
            self._build()
        if not (self._distributed and hasattr(self, "_rs_futures")):
            return
        for fut in self._rs_futures:
            if fut is not None:
                fut.wait()

    def grad_sqnorm(self) -> Tensor:
        if not self._built:
            self._build()
        if self._distributed and hasattr(self, "_rs_futures"):
            self.wait_reduce_scatters()
        sq = torch.zeros((), device=self._bank_meta[0]['p'].device, dtype=torch.float64)
        sharded = self._distributed and hasattr(self, "_rs_futures")
        for m in self._bank_meta:
            p = m['p']
            if p.grad is None:
                continue
            g = m['shard'] if sharded else p.grad
            sq += g.float().square().sum()
        return sq

    @torch.no_grad()
    def scale_grads(self, scale: float) -> None:
        if not self._built:
            self._build()
        if self._distributed and hasattr(self, "_rs_futures"):
            self.wait_reduce_scatters()
        sharded = self._distributed and hasattr(self, "_rs_futures")
        for m in self._bank_meta:
            p = m['p']
            if p.grad is None:
                continue
            g = m['shard'] if sharded else p.grad
            g.mul_(scale)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if not self._built:
            self._build()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            wd = group.get("weight_decay", 0.0)

            prev_ag_handle = None
            prev_m = None

            sharded = self._distributed and hasattr(self, '_rs_futures')

            for i, m in enumerate(self._bank_meta):
                p = m['p']
                if p.grad is None:
                    continue

                if prev_ag_handle is not None:
                    prev_ag_handle.wait()
                    pp = prev_m['p']
                    upd = prev_m['full_update'][:prev_m['B']]
                    if wd > 0.0:
                        pp.data.mul_(1.0 - lr * wd)
                    pp.add_(upd.to(dtype=pp.dtype), alpha=-lr * prev_m['scale'])

                if sharded and self._rs_futures[i] is not None:
                    self._rs_futures[i].wait()
                    g = m['shard']
                    buf = m['shard_mom']
                else:
                    g = p.grad.bfloat16()
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]

                buf.mul_(momentum).add_(g)
                if nesterov:
                    update = g.add(buf, alpha=momentum)
                else:
                    update = buf

                update = zeropower_via_newtonschulz5(update, steps=backend_steps)

                if sharded:
                    prev_ag_handle = dist.all_gather_into_tensor(
                        m['full_update'], update, async_op=True)
                    prev_m = m
                else:
                    if wd > 0.0:
                        p.data.mul_(1.0 - lr * wd)
                    p.add_(update.to(dtype=p.dtype), alpha=-lr * m['scale'])

            if prev_ag_handle is not None:
                prev_ag_handle.wait()
                pp = prev_m['p']
                upd = prev_m['full_update'][:prev_m['B']]
                if wd > 0.0:
                    pp.data.mul_(1.0 - lr * wd)
                pp.add_(upd.to(dtype=pp.dtype), alpha=-lr * prev_m['scale'])

            if hasattr(self, '_rs_futures'):
                del self._rs_futures

        return loss


def allreduce_param_grads(params: list[Tensor]) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)


def clip_grad_norm_with_muon(replicated_params: list[Tensor], muon_optimizer: Muon, max_norm: float) -> float:
    if max_norm <= 0.0:
        return 0.0
    device = None
    for p in replicated_params:
        if p.grad is not None:
            device = p.grad.device
            break
    if device is None:
        if not muon_optimizer._built:
            muon_optimizer._build()
        device = muon_optimizer._bank_meta[0]['p'].device

    replicated_sq = torch.zeros((), device=device, dtype=torch.float64)
    for p in replicated_params:
        if p.grad is not None:
            replicated_sq += p.grad.float().square().sum()

    bank_sq = muon_optimizer.grad_sqnorm()
    if dist.is_available() and dist.is_initialized() and muon_optimizer._distributed:
        dist.all_reduce(bank_sq, op=dist.ReduceOp.SUM)

    total_norm = (replicated_sq + bank_sq).sqrt()
    clip_coef = min(float(max_norm / (total_norm.item() + 1e-6)), 1.0)
    if clip_coef < 1.0:
        for p in replicated_params:
            if p.grad is not None:
                p.grad.mul_(clip_coef)
        muon_optimizer.scale_grads(clip_coef)
    return float(total_norm.item())


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )
def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]
def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    eval_seq_len: int | None = None,
) -> tuple[float, float]:
    seq_len = eval_seq_len or args.train_seq_len
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, seq_len={seq_len}"
        )
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear,dtg_gate,ve_layer_scales,ve_shared.scale,attn_gate,vr_lambda",
    ).split(",")
    if pattern
)
FORCE_INT8_NAME_PATTERNS = ("attn_residual_queries",)
_INT8_CLIP_Q = 0.9999984


def should_force_int8(name: str) -> bool:
    return any(pattern in name for pattern in FORCE_INT8_NAME_PATTERNS)


def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), _INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=torch.float16).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), _INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))
class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0
    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0
    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)
class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)
    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)
class CastedLinear(nn.Linear):
    _qat_enabled: bool = False
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            with torch.no_grad():
                w32 = self.weight.float()
                row_max = w32.abs().amax(dim=1)
                scale = (row_max / 31.0).clamp_min(1.0 / 31.0)
                w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -32, 31) * scale[:, None]).to(x.dtype)
            w = w + (w_q - w).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)
def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()
class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, train_seq_len: int = 1024, rope_dims: int = 0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None
    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            rd = self.rope_dims
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                new_base = self.base * (scale ** (rd / (rd - 2)))
                inv_freq = 1.0 / (new_base ** (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd))
            else:
                inv_freq = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)
def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int = 0) -> Tensor:
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

_gptq_capture: dict[str, tuple[Tensor, int]] | None = None


def _accumulate_gptq_input(name: str, x: Tensor) -> None:
    global _gptq_capture
    if _gptq_capture is None:
        return
    x_flat = x.detach().float().reshape(-1, x.shape[-1])
    h_sum = x_flat.T @ x_flat
    state = _gptq_capture.get(name)
    if state is None:
        _gptq_capture[name] = (h_sum, x_flat.shape[0])
    else:
        prev_h, prev_n = state
        prev_h.add_(h_sum)
        _gptq_capture[name] = (prev_h, prev_n + x_flat.shape[0])

class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        gated_attention: bool = False,
        value_residual: bool = False,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = 0  # set by GPT.__init__ for partial RoPE
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=1024)
        self.use_xsa = False  # set by GPT.__init__ for deep layers only
        self.gated_attention = gated_attention
        if gated_attention:
            self.attn_gate = nn.Linear(dim, num_heads, bias=True)
            nn.init.zeros_(self.attn_gate.weight)
            nn.init.constant_(self.attn_gate.bias, 4.0)
        self.value_residual = value_residual
        if value_residual:
            self.vr_lambda = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
    def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)        # [B, T, Hkv, group, D]
        vn = F.normalize(v, dim=-1).unsqueeze(-2)    # [B, T, Hkv, 1, D] -- broadcast ready
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)
    def forward(self, x: Tensor, q_w: Tensor, k_w: Tensor, v_w: Tensor, out_w: Tensor, v_embed: Tensor | None = None, v0: Tensor | None = None, layer_idx: int = -1) -> tuple[Tensor, Tensor | None]:
        bsz, seqlen, dim = x.shape
        if _gptq_capture is not None and layer_idx >= 0:
            _accumulate_gptq_input(f"blocks.{layer_idx}.attn.c_q.weight", x)
            _accumulate_gptq_input(f"blocks.{layer_idx}.attn.c_k.weight", x)
            _accumulate_gptq_input(f"blocks.{layer_idx}.attn.c_v.weight", x)
        q = F.linear(x, q_w.to(x.dtype)).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = F.linear(x, k_w.to(x.dtype)).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = F.linear(x, v_w.to(x.dtype))
        if v_embed is not None:
            v = v + v_embed
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        raw_v = v if self.value_residual else None
        if self.value_residual and v0 is not None:
            lam = self.vr_lambda.to(dtype=v.dtype)
            v = lam[0] * v0 + lam[1] * v
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = flash_attn_3_func(q, k, v, causal=True)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        if self.gated_attention:
            gate = torch.sigmoid(self.attn_gate(x)).unsqueeze(-1)
            y = y * gate
        y = y.reshape(bsz, seqlen, dim)
        if _gptq_capture is not None and layer_idx >= 0:
            _accumulate_gptq_input(f"blocks.{layer_idx}.attn.proj.weight", y)
        return F.linear(y, out_w.to(x.dtype)), raw_v

class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev

class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
    def bigram_hash(self, tokens: Tensor) -> Tensor:
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()
    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)

class ValueEmbedding(nn.Module):
    def __init__(self, vocab_size: int, ve_dim: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ve_dim)
        nn.init.normal_(self.embed.weight, std=0.01)
        self.proj = CastedLinear(ve_dim, model_dim, bias=False) if ve_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(token_ids)
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
    def forward(self, x: Tensor, up_w: Tensor, down_w: Tensor, layer_idx: int = -1) -> Tensor:
        if _gptq_capture is not None and layer_idx >= 0:
            _accumulate_gptq_input(f"blocks.{layer_idx}.mlp.fc.weight", x)
        x = F.leaky_relu(F.linear(x, up_w.to(x.dtype)), negative_slope=0.5)
        x_sq = x.square()
        if _gptq_capture is not None and layer_idx >= 0:
            _accumulate_gptq_input(f"blocks.{layer_idx}.mlp.proj.weight", x_sq)
        return F.linear(x_sq, down_w.to(x.dtype))

class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        layer_idx: int = 0,
        ln_scale: bool = False,
        dtg: bool = False,
        parallel_residual: bool = False,
        gated_attention: bool = False,
        value_residual: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.parallel_residual = parallel_residual
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                                        gated_attention=gated_attention, value_residual=value_residual)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
        if dtg:
            self.dtg_gate = nn.Linear(dim, 1, bias=True)
            nn.init.zeros_(self.dtg_gate.weight)
            nn.init.constant_(self.dtg_gate.bias, 2.0)
        else:
            self.dtg_gate = None
    def forward(self, x: Tensor, x0: Tensor, q_w: Tensor, k_w: Tensor, v_w: Tensor, out_w: Tensor, up_w: Tensor, down_w: Tensor, v_embed: Tensor | None = None, v0: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_input = self.attn_norm(x_in) * self.ln_scale_factor
        attn_out, raw_v = self.attn(attn_input, q_w, k_w, v_w, out_w, v_embed=v_embed, v0=v0, layer_idx=self.layer_idx)
        if self.parallel_residual:
            mlp_out = self.mlp(self.mlp_norm(x_in) * self.ln_scale_factor, up_w, down_w, layer_idx=self.layer_idx)
            x_out = x_in
            x_out = x_out + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
            x_out = x_out + self.mlp_scale.to(dtype=x_in.dtype)[None, None, :] * mlp_out
        else:
            x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
            x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor, up_w, down_w, layer_idx=self.layer_idx)
        if self.dtg_gate is not None:
            gate = torch.sigmoid(self.dtg_gate(x_in.detach()))
            x_out = x_in + gate * (x_out - x_in)
        return x_out, raw_v

class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        mtp_num_heads: int = 0,
        mtp_loss_weight: float = 0.1,
        bigram_vocab_size: int = 0,
        bigram_dim: int = 128,
        xsa_last_n: int = 0,
        rope_dims: int = 0,
        ln_scale: bool = False,
        dtg: bool = False,
        ve_enabled: bool = False,
        ve_dim: int = 128,
        ve_layers: str = "9,10",
        recurrent_layer_start: int = 2,
        recurrent_layer_end: int = 4,
        recurrent_extra_passes: int = 2,
        parallel_residual_start: int = 6,
        attn_residual_enabled: bool = True,
        gated_attention: bool = False,
        value_residual: bool = False,
    ):
        super().__init__()
        self._ve_target_dim = num_kv_heads * (model_dim // num_heads)  # kv_dim for value projection
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.value_residual = value_residual
        self.mtp_num_heads = mtp_num_heads
        self.mtp_loss_weight = mtp_loss_weight
        self.parallel_residual_start = parallel_residual_start
        self.attn_residual_enabled = attn_residual_enabled
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim) if bigram_vocab_size > 0 else None
        self.smear = SmearGate(model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.virtual_schedule = self._build_virtual_schedule(
            num_layers,
            recurrent_layer_start,
            recurrent_layer_end,
            recurrent_extra_passes,
        )
        self.virtual_num_layers = len(self.virtual_schedule)
        if self.attn_residual_enabled:
            self.attn_residual_queries = nn.Parameter(torch.empty(self.virtual_num_layers, model_dim))
        else:
            self.register_parameter("attn_residual_queries", None)
        head_dim = model_dim // num_heads
        kv_dim = num_kv_heads * head_dim
        mlp_dim = int(mlp_mult * model_dim)
        self.num_layers = num_layers
        self.qo_bank = nn.Parameter(torch.empty(2 * num_layers, model_dim, model_dim))
        self.kv_bank = nn.Parameter(torch.empty(2 * num_layers, kv_dim, model_dim))
        self.mlp_up_bank = nn.Parameter(torch.empty(num_layers, mlp_dim, model_dim))
        self.mlp_down_bank = nn.Parameter(torch.empty(num_layers, model_dim, mlp_dim))
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    layer_idx=i,
                    ln_scale=ln_scale,
                    dtg=dtg,
                    parallel_residual=i >= parallel_residual_start,
                    gated_attention=gated_attention,
                    value_residual=value_residual,
                )
                for i in range(num_layers)
            ]
        )
        if rope_dims > 0:
            head_dim = model_dim // num_heads
            for block in self.blocks:
                block.attn.rope_dims = rope_dims
                block.attn.rotary = Rotary(head_dim, base=rope_base, train_seq_len=1024, rope_dims=rope_dims)
        self.ve_layer_indices = [int(x) for x in ve_layers.split(",") if x.strip()] if ve_enabled else []
        kv_dim_ve = self._ve_target_dim
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(vocab_size, ve_dim, kv_dim_ve)
            self.ve_layer_scales = nn.ParameterList(
                [nn.Parameter(torch.ones(1, dtype=torch.float32)) for _ in self.ve_layer_indices]
            )
        else:
            self.ve_shared = None
            self.ve_layer_scales = nn.ParameterList()
        self.value_embeds = nn.ModuleList()  # keep empty for compat
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self.mtp_heads = nn.ModuleList(
            [CastedLinear(model_dim, vocab_size, bias=False) for _ in range(mtp_num_heads)]
        )
        for head in self.mtp_heads:
            head._zero_init = True
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True
        self._init_weights()
    @staticmethod
    def _build_virtual_schedule(
        num_layers: int,
        recurrent_layer_start: int,
        recurrent_layer_end: int,
        recurrent_extra_passes: int,
    ) -> tuple[int, ...]:
        if recurrent_extra_passes <= 0:
            return tuple(range(num_layers))
        start = max(0, recurrent_layer_start)
        end = min(num_layers - 1, recurrent_layer_end)
        if start > end:
            return tuple(range(num_layers))
        schedule: list[int] = []
        for layer_idx in range(num_layers):
            schedule.append(layer_idx)
            if layer_idx == end:
                for _ in range(recurrent_extra_passes):
                    schedule.extend(range(start, end + 1))
        return tuple(schedule)
    def _attn_residual_mix(self, history: list[Tensor], history_summaries: list[Tensor], virtual_idx: int) -> Tensor:
        if not history:
            raise ValueError("attn residual history cannot be empty")
        if not self.attn_residual_enabled or len(history) == 1:
            return history[-1]
        query = F.rms_norm(
            self.attn_residual_queries[virtual_idx].to(dtype=history_summaries[-1].dtype),
            (history_summaries[-1].size(-1),),
        )
        scores = []
        for summary in history_summaries:
            score = (F.rms_norm(summary, (summary.size(-1),)) * query[None, :]).sum(dim=-1)
            scores.append(score)
        score_tensor = torch.stack(scores, dim=1) / math.sqrt(history_summaries[-1].size(-1))
        weights = F.softmax(score_tensor, dim=1).to(dtype=history[-1].dtype)
        mixed = history[0] * weights[:, 0].view(-1, 1, 1)
        for hist_idx in range(1, len(history)):
            mixed = mixed + history[hist_idx] * weights[:, hist_idx].view(-1, 1, 1)
        return mixed
    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        if self.attn_residual_queries is not None:
            nn.init.normal_(self.attn_residual_queries, mean=0.0, std=self.tied_embed_init_std)
        n = self.num_layers
        proj_scale = 1.0 / math.sqrt(2 * n)
        for i in range(n):
            nn.init.orthogonal_(self.qo_bank.data[i], gain=1.0)
            nn.init.zeros_(self.qo_bank.data[n + i])
            nn.init.orthogonal_(self.kv_bank.data[i], gain=1.0)
            nn.init.orthogonal_(self.kv_bank.data[n + i], gain=1.0)
            nn.init.orthogonal_(self.mlp_up_bank.data[i], gain=1.0)
            nn.init.zeros_(self.mlp_down_bank.data[i])
            self.qo_bank.data[n + i].mul_(proj_scale)
            self.mlp_down_bank.data[i].mul_(proj_scale)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)
    def _get_ve(self, layer_idx: int, input_ids: Tensor, ve_cache: dict | None = None) -> Tensor | None:
        if self.ve_shared is None or layer_idx not in self.ve_layer_indices:
            return None
        if ve_cache is not None and 've' not in ve_cache:
            ve_cache['ve'] = self.ve_shared(input_ids)
        ve_base = ve_cache['ve'] if ve_cache is not None else self.ve_shared(input_ids)
        ve_idx = self.ve_layer_indices.index(layer_idx)
        return ve_base * self.ve_layer_scales[ve_idx].to(dtype=ve_base.dtype)
    def _run_blocks(self, input_ids: Tensor) -> Tensor:
        n = self.num_layers
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        v0 = None
        history = [x]
        history_summaries = [x.mean(dim=1)]
        skip_states: list[Tensor | None] = [None] * self.num_skip_weights
        ve_cache: dict = {}
        for virtual_idx, block_idx in enumerate(self.virtual_schedule):
            x_in = self._attn_residual_mix(history, history_summaries, virtual_idx)
            if block_idx >= self.num_encoder_layers:
                decoder_idx = block_idx - self.num_encoder_layers
                if decoder_idx < self.num_skip_weights:
                    skip_slot = self.num_skip_weights - 1 - decoder_idx
                    skip_state = skip_states[skip_slot]
                    if skip_state is not None:
                        x_in = x_in + self.skip_weights[decoder_idx].to(dtype=x_in.dtype)[None, None, :] * skip_state
            ve = self._get_ve(block_idx, input_ids, ve_cache)
            x, raw_v = self.blocks[block_idx](x_in, x0,
                self.qo_bank[block_idx], self.kv_bank[block_idx], self.kv_bank[n + block_idx],
                self.qo_bank[n + block_idx], self.mlp_up_bank[block_idx], self.mlp_down_bank[block_idx],
                v_embed=ve, v0=v0)
            if v0 is None and raw_v is not None:
                v0 = raw_v
            if block_idx < self.num_skip_weights:
                skip_states[block_idx] = x
            history.append(x)
            history_summaries.append(x.mean(dim=1))
        return self.final_norm(x)
    def _logits(self, x: Tensor) -> Tensor:
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self._run_blocks(input_ids)
        logits = self._logits(x.reshape(-1, x.size(-1)))
        main_loss = F.cross_entropy(logits.float(), target_ids.reshape(-1), reduction="mean")
        if self.training and self.mtp_num_heads > 0 and self.mtp_loss_weight > 0.0:
            _, seqlen, dim = x.shape
            mtp_loss_sum = x.new_zeros(())
            mtp_loss_count = 0
            for k, mtp_head in enumerate(self.mtp_heads):
                valid_t = seqlen - (k + 1)
                if valid_t <= 0:
                    continue
                mtp_hidden = x[:, :valid_t, :].reshape(-1, dim)
                mtp_logits_proj = mtp_head(mtp_hidden)
                mtp_logits = self.logit_softcap * torch.tanh(mtp_logits_proj / self.logit_softcap)
                mtp_loss_sum = mtp_loss_sum + F.cross_entropy(mtp_logits.float(), target_ids[:, k + 1:].reshape(-1), reduction="mean")
                mtp_loss_count += 1
            if mtp_loss_count > 0:
                main_loss = main_loss + self.mtp_loss_weight * (mtp_loss_sum / mtp_loss_count)
        return main_loss
    def forward_logits(self, input_ids: Tensor) -> Tensor:
        return self._logits(self._run_blocks(input_ids))


def instantiate_model(
    args: Hyperparameters,
    device: torch.device,
    *,
    mtp_num_heads: int | None = None,
    mtp_loss_weight: float | None = None,
    bigram_vocab_size: int | None = None,
    ve_enabled: bool | None = None,
) -> GPT:
    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        mtp_num_heads=args.mtp_num_heads if mtp_num_heads is None else mtp_num_heads,
        mtp_loss_weight=args.mtp_loss_weight if mtp_loss_weight is None else mtp_loss_weight,
        bigram_vocab_size=args.bigram_vocab_size if bigram_vocab_size is None else bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        dtg=args.dtg_enabled,
        ve_enabled=args.ve_enabled if ve_enabled is None else ve_enabled,
        ve_dim=args.ve_dim,
        ve_layers=args.ve_layers,
        recurrent_layer_start=args.recurrent_layer_start,
        recurrent_layer_end=args.recurrent_layer_end,
        recurrent_extra_passes=args.recurrent_extra_passes,
        parallel_residual_start=args.parallel_residual_start,
        attn_residual_enabled=args.attn_residual_enabled,
        gated_attention=args.gated_attention,
        value_residual=args.value_residual,
    ).to(device).bfloat16()
    model.qo_bank.data = model.qo_bank.data.float()
    model.kv_bank.data = model.kv_bank.data.float()
    model.mlp_up_bank.data = model.mlp_up_bank.data.float()
    model.mlp_down_bank.data = model.mlp_down_bank.data.float()
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(model)
    return model

def eval_val_sliding(
    args: Hyperparameters,
    base_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    batch_seqs: int = 32,
    eval_seq_len: int | None = None,
) -> tuple[float, float]:
    seq_len = eval_seq_len or args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    base_model.eval()
    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled_logits(x_batch)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum / token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    base_model.train()
    return val_loss, bits_per_token * tokens_per_byte


def eval_val_sliding_ttt(
    args: Hyperparameters, base_model: nn.Module, rank: int, world_size: int,
    device: torch.device, val_tokens: Tensor, base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor, is_boundary_token_lut: Tensor,
    stride: int, batch_seqs: int = 32, log0=print,
) -> tuple[float, float]:
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    ttt_chunk = args.ttt_chunk_tokens

    # Pre-compute all window starts
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= stride or ws == 0]

    # Assign each window to a chunk based on the first token it scores
    num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk
    chunk_windows: list[list[int]] = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        end = min(ws + seq_len, total_tokens)
        wlen = end - ws
        s = 0 if ws == 0 else max(wlen - stride, 0)
        scored_start = ws + s
        ci = min(scored_start // ttt_chunk, num_chunks - 1)
        chunk_windows[ci].append(ws)

    log0(f"ttt_sliding:start chunks={num_chunks} chunk_tokens={ttt_chunk} "
         f"total_windows={len(window_starts)} stride={stride} "
         f"ttt_lr={args.ttt_lr} ttt_epochs={args.ttt_epochs} "
         f"freeze_blocks={args.ttt_freeze_blocks}")

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    # Freeze first N blocks
    frozen_block_ids = set(range(min(args.ttt_freeze_blocks, len(base_model.blocks))))
    ttt_params = []
    for name, p in base_model.named_parameters():
        freeze = False
        for bi in frozen_block_ids:
            if f"blocks.{bi}." in name:
                freeze = True
                break
        if freeze:
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)
            ttt_params.append(p)

    log0(f"ttt_sliding:params unfrozen={sum(p.numel() for p in ttt_params)} "
         f"frozen={sum(p.numel() for p in base_model.parameters() if not p.requires_grad)}")

    optimizer = torch.optim.SGD(ttt_params, lr=args.ttt_lr, momentum=args.ttt_momentum)
    t0 = time.perf_counter()

    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows:
            continue
        chunk_start = ci * ttt_chunk
        chunk_end = min((ci + 1) * ttt_chunk, total_tokens)

        # --- Phase 1: SCORE this chunk's windows (inference_mode) ---
        my_s = (len(windows) * rank) // world_size
        my_e = (len(windows) * (rank + 1)) // world_size
        my_windows = windows[my_s:my_e]

        base_model.eval()
        with torch.inference_mode():
            for bi in range(0, len(my_windows), batch_seqs):
                batch_ws = my_windows[bi:bi + batch_seqs]
                bsz = len(batch_ws)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                wlens: list[int] = []
                for i, ws in enumerate(batch_ws):
                    end = min(ws + seq_len, total_tokens)
                    wlen = end - ws
                    wlens.append(wlen)
                    chunk_tok = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = chunk_tok[:-1]
                    y_batch[i, :wlen] = chunk_tok[1:]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = base_model.forward_logits(x_batch)
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y_batch.reshape(-1), reduction="none",
                ).reshape(bsz, seq_len)
                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s = 0 if ws == 0 else max(wlen - stride, 0)
                    scored_nll = nll[i, s:wlen].to(torch.float64)
                    loss_sum += scored_nll.sum()
                    token_count += float(wlen - s)
                    tgt, prev = y_batch[i, s:wlen], x_batch[i, s:wlen]
                    tb = base_bytes_lut[tgt].to(torch.float64)
                    tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                    byte_count += tb.sum()

        # --- Phase 2: TRAIN on this chunk (already scored = legal) ---
        is_last_chunk = (ci == num_chunks - 1)
        if not is_last_chunk and args.ttt_epochs > 0:
            base_model.train()
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs > 0:
                cos_lr = args.ttt_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
                for pg in optimizer.param_groups:
                    pg['lr'] = cos_lr
                my_seq_s = (chunk_seqs * rank) // world_size
                my_seq_e = (chunk_seqs * (rank + 1)) // world_size
                my_chunk_seqs = my_seq_e - my_seq_s
                for _ep in range(args.ttt_epochs):
                    for bs in range(0, my_chunk_seqs, args.ttt_batch_seqs):
                        be = min(bs + args.ttt_batch_seqs, my_chunk_seqs)
                        actual_bs = my_seq_s + bs
                        start_tok = chunk_start + actual_bs * seq_len
                        end_tok = chunk_start + (my_seq_s + be) * seq_len + 1
                        if end_tok > val_tokens.numel():
                            continue
                        local = val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
                        x = local[:-1].reshape(-1, seq_len)
                        y = local[1:].reshape(-1, seq_len)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            loss = base_model(x, y)
                        loss.backward()
                        if world_size > 1:
                            for p in ttt_params:
                                if p.grad is not None:
                                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                        torch.nn.utils.clip_grad_norm_(ttt_params, args.ttt_grad_clip)
                        optimizer.step()

        if rank == 0 and (ci % 10 == 0 or ci == num_chunks - 1):
            elapsed = time.perf_counter() - t0
            rl = loss_sum.item() / max(token_count.item(), 1)
            rbpb = rl / math.log(2.0) * (token_count.item() / max(byte_count.item(), 1)) if token_count.item() > 0 else 0.0
            log0(f"  ttt_chunk [{ci+1}/{num_chunks}] bpb={rbpb:.6f} time={elapsed:.1f}s")

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())

    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.eval()

    log0(f"ttt_sliding:done val_loss={val_loss:.6f} val_bpb={val_bpb:.6f} "
         f"elapsed={time.perf_counter() - t0:.1f}s")
    return val_loss, val_bpb


@torch.no_grad()
def generate_calibration(model: nn.Module, vocab_size: int, num_seqs: int = 32,
                         seq_len: int = 256, temperature: float = 0.8,
                         device: torch.device = torch.device("cuda")) -> Tensor:
    was_training = model.training
    model.eval()
    tokens = torch.zeros(num_seqs, seq_len, dtype=torch.long, device=device)
    prompt_len = min(seq_len, 2)
    if prompt_len > 0:
        tokens[:, :prompt_len] = torch.randint(1, vocab_size, (num_seqs, prompt_len), device=device)
    for t in range(prompt_len, seq_len):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model.forward_logits(tokens[:, :t])
        next_logits = logits[:, -1, :].float() / temperature
        probs = F.softmax(next_logits, dim=-1)
        tokens[:, t] = torch.multinomial(probs, 1).squeeze(-1)
    model.train(was_training)
    return tokens


def collect_hessians(model: nn.Module, calib_tokens: Tensor, device: torch.device,
                     batch_size: int = 8) -> dict[str, Tensor]:
    global _gptq_capture
    hessians: dict[str, Tensor] = {}
    was_training = model.training
    _gptq_capture = {}
    model.eval()
    try:
        with torch.no_grad():
            for i in range(0, calib_tokens.shape[0], batch_size):
                batch = calib_tokens[i:i + batch_size].to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    model.forward_logits(batch)
        for name, (h_sum, n_samples) in _gptq_capture.items():
            hessians[name] = h_sum / max(n_samples, 1)
    finally:
        _gptq_capture = None
        model.train(was_training)
    return hessians


_sign_cache: dict[tuple[int, int], Tensor] = {}
_turbo_rounds = 1


def _det_signs(n: int, seed: int = 42) -> Tensor:
    key = (n, seed)
    if key not in _sign_cache:
        g = torch.Generator().manual_seed(seed)
        _sign_cache[key] = (torch.randint(0, 2, (n,), generator=g) * 2 - 1).float()
    return _sign_cache[key]


def _fwht(x: Tensor) -> Tensor:
    n = x.shape[-1]
    h = 1
    while h < n:
        x = x.view(*x.shape[:-1], -1, 2 * h)
        a, b = x[..., :h], x[..., h:]
        x = torch.cat([a + b, a - b], dim=-1).view(*x.shape[:-2], -1)
        h *= 2
    return x * (n ** -0.5)

def _fwht_cols(t32: Tensor, cols: int, p2: int) -> Tensor:
    if p2 == cols: return _fwht(t32)
    return _fwht(t32.view(t32.shape[0], -1, p2)).view(t32.shape)

def _rotate_cols(t: Tensor) -> Tensor:
    if t.ndim != 2: return t
    cols = t.shape[1]
    p2 = 1
    while p2 * 2 <= cols and cols % (p2 * 2) == 0: p2 *= 2
    t32 = t.float()
    for r in range(_turbo_rounds):
        d1 = _det_signs(cols, seed=42 + 2 * r).to(t.device)
        d2 = _det_signs(cols, seed=43 + 2 * r).to(t.device)
        t32 = _fwht_cols(t32 * d1, cols, p2) * d2
    return t32

def _unrotate_cols(t: Tensor) -> Tensor:
    if t.ndim != 2: return t
    cols = t.shape[1]
    p2 = 1
    while p2 * 2 <= cols and cols % (p2 * 2) == 0: p2 *= 2
    t32 = t.float()
    for r in range(_turbo_rounds - 1, -1, -1):
        d1 = _det_signs(cols, seed=42 + 2 * r).to(t.device)
        d2 = _det_signs(cols, seed=43 + 2 * r).to(t.device)
        t32 = _fwht_cols(t32 * d2, cols, p2) * d1
    return t32


def _rotate_rows(t: Tensor) -> Tensor:
    if t.ndim != 2:
        return t
    return _rotate_cols(t.T).T


def _rotate_hessian(h: Tensor) -> Tensor:
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError(f"expected square Hessian, got shape={tuple(h.shape)}")
    h_rot = _rotate_rows(_rotate_cols(h.float()))
    return 0.5 * (h_rot + h_rot.T)


def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"
def quantize_int6_per_row(t: Tensor, clip_range: int = 31) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        best_q, best_s, best_err = None, None, float('inf')
        for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
            if pct < 1.0:
                row_clip = torch.quantile(t32.abs(), pct, dim=1)
            else:
                row_clip = t32.abs().amax(dim=1)
            s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
            q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)
            recon = q.float() * s.float()[:, None]
            err = (t32 - recon).pow(2).mean().item()
            if err < best_err:
                best_q, best_s, best_err = q, s, err
        return best_q, best_s
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8)
    return q, scale

def gptq_quantize_int6(
    W: Tensor,
    H: Tensor,
    clip_range: int = 31,
    block_size: int = 128,
    damp: float = 0.01,
) -> tuple[Tensor, Tensor, str]:
    W = W.float().clone()
    H = H.float().clone()
    _, n_cols = W.shape
    if H.ndim != 2 or H.shape[0] != H.shape[1] or H.shape[0] != n_cols:
        q, s = quantize_int6_per_row(W, clip_range)
        return q, s, "bad_hessian"
    block_size = max(1, min(int(block_size), n_cols))
    diag_mean = H.diagonal().mean().abs().clamp_min(1e-8)
    H.diagonal().add_(diag_mean * max(float(damp), 0.0))
    try:
        L = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(L)
    except Exception:
        q, s = quantize_int6_per_row(W, clip_range)
        return q, s, "cholesky_fallback"
    best_q, best_s, best_err = None, None, float('inf')
    for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
        if pct < 1.0:
            row_clip = torch.quantile(W.abs(), pct, dim=1)
        else:
            row_clip = W.abs().amax(dim=1)
        s_candidate = (row_clip / clip_range).clamp_min(1.0 / clip_range)
        W_work = W.clone()
        q_candidate = torch.zeros_like(W, dtype=torch.int8)
        recon_candidate = torch.zeros_like(W)
        for col_start in range(0, n_cols, block_size):
            col_end = min(col_start + block_size, n_cols)
            H_inv_block = H_inv[col_start:col_end, col_start:col_end]
            err_block = torch.zeros(W.shape[0], col_end - col_start, dtype=W.dtype, device=W.device)
            for j in range(col_end - col_start):
                col_idx = col_start + j
                w_col = W_work[:, col_idx]
                q_col = torch.clamp(torch.round(w_col / s_candidate), -clip_range, clip_range)
                recon_col = q_col * s_candidate
                q_candidate[:, col_idx] = q_col.to(torch.int8)
                recon_candidate[:, col_idx] = recon_col
                err_col = (w_col - recon_col) / H_inv_block[j, j].abs().clamp_min(1e-8)
                err_block[:, j] = err_col
                if j + 1 < col_end - col_start:
                    W_work[:, col_idx + 1:col_end] -= err_col.unsqueeze(1) * H_inv_block[j, j + 1:].unsqueeze(0)
            if col_end < n_cols:
                W_work[:, col_end:] -= err_block @ H_inv[col_start:col_end, col_end:]
        err = (W - recon_candidate).pow(2).mean().item()
        if err < best_err:
            best_q, best_s, best_err = q_candidate, s_candidate.to(torch.float16), err
    return best_q, best_s, "gptq"

def mixed_quantize_int6_gptq(
    state_dict: dict[str, Tensor],
    int6_cats: set[str],
    hessians: dict[str, Tensor],
    block_size: int = 128,
    damp: float = 0.01,
) -> tuple[dict[str, Tensor], dict[str, object], dict[str, int]]:
    result: dict[str, Tensor] = {}
    meta: dict[str, object] = {}
    stats = {
        "gptq_layers": 0,
        "fallback_missing_hessian": 0,
        "fallback_bad_hessian": 0,
        "fallback_cholesky": 0,
    }
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)
        if not t.is_floating_point():
            result[name] = t
            meta[name] = "passthrough"
            continue
        if t.numel() <= 65536 and not should_force_int8(name):
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue
        if cat in int6_cats and t.ndim >= 1:
            t_rot = _rotate_cols(t)
            H = hessians.get(name)
            used_gptq = False
            if H is None:
                q, s = quantize_int6_per_row(t_rot)
                stats["fallback_missing_hessian"] += 1
            else:
                H_cpu = H.detach().cpu()
                if H_cpu.ndim != 2 or H_cpu.shape[0] != H_cpu.shape[1] or H_cpu.shape[0] != t_rot.shape[1]:
                    q, s = quantize_int6_per_row(t_rot)
                    stats["fallback_bad_hessian"] += 1
                else:
                    q, s, mode = gptq_quantize_int6(
                        t_rot,
                        _rotate_hessian(H_cpu),
                        block_size=block_size,
                        damp=damp,
                    )
                    if mode == "gptq":
                        used_gptq = True
                        stats["gptq_layers"] += 1
                    elif mode == "bad_hessian":
                        stats["fallback_bad_hessian"] += 1
                    else:
                        stats["fallback_cholesky"] += 1
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int6", "rotated": True, "gptq": used_gptq}
        else:
            q, s = quantize_float_tensor(_rotate_cols(t))
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8", "rotated": True}
    return result, meta, stats

def _unbank_state_dict(sd: dict[str, Tensor], num_layers: int) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    n = num_layers
    for name, tensor in sd.items():
        if name == "qo_bank":
            for i in range(n):
                out[f"blocks.{i}.attn.c_q.weight"] = tensor[i]
                out[f"blocks.{i}.attn.proj.weight"] = tensor[n + i]
        elif name == "kv_bank":
            for i in range(n):
                out[f"blocks.{i}.attn.c_k.weight"] = tensor[i]
                out[f"blocks.{i}.attn.c_v.weight"] = tensor[n + i]
        elif name == "mlp_up_bank":
            for i in range(n):
                out[f"blocks.{i}.mlp.fc.weight"] = tensor[i]
        elif name == "mlp_down_bank":
            for i in range(n):
                out[f"blocks.{i}.mlp.proj.weight"] = tensor[i]
        else:
            out[name] = tensor
    return out

def _rebank_state_dict(sd: dict[str, Tensor], num_layers: int, template_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    n = num_layers
    qo_slices = [None] * (2 * n)
    kv_slices = [None] * (2 * n)
    up_slices = [None] * n
    down_slices = [None] * n
    consumed = set()
    for i in range(n):
        qk = f"blocks.{i}.attn.c_q.weight"
        if qk in sd:
            qo_slices[i] = sd[qk]
            consumed.add(qk)
        ok = f"blocks.{i}.attn.proj.weight"
        if ok in sd:
            qo_slices[n + i] = sd[ok]
            consumed.add(ok)
        kk = f"blocks.{i}.attn.c_k.weight"
        if kk in sd:
            kv_slices[i] = sd[kk]
            consumed.add(kk)
        vk = f"blocks.{i}.attn.c_v.weight"
        if vk in sd:
            kv_slices[n + i] = sd[vk]
            consumed.add(vk)
        fk = f"blocks.{i}.mlp.fc.weight"
        if fk in sd:
            up_slices[i] = sd[fk]
            consumed.add(fk)
        dk = f"blocks.{i}.mlp.proj.weight"
        if dk in sd:
            down_slices[i] = sd[dk]
            consumed.add(dk)
    out["qo_bank"] = torch.stack(qo_slices).to(dtype=template_sd["qo_bank"].dtype)
    out["kv_bank"] = torch.stack(kv_slices).to(dtype=template_sd["kv_bank"].dtype)
    out["mlp_up_bank"] = torch.stack(up_slices).to(dtype=template_sd["mlp_up_bank"].dtype)
    out["mlp_down_bank"] = torch.stack(down_slices).to(dtype=template_sd["mlp_down_bank"].dtype)
    for name, tensor in sd.items():
        if name not in consumed:
            out[name] = tensor
    return out

def mixed_quantize_int6(state_dict: dict[str, Tensor], int6_cats: set[str]):
    result: dict[str, Tensor] = {}
    meta: dict[str, object] = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)
        if not t.is_floating_point():
            result[name] = t
            meta[name] = "passthrough"
            continue
        if t.numel() <= 65536 and not should_force_int8(name):
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue
        if cat in int6_cats and t.ndim >= 1:
            q, s = quantize_int6_per_row(_rotate_cols(t))
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int6", "rotated": True}
        else:
            q, s = quantize_float_tensor(_rotate_cols(t))
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8", "rotated": True}
    return result, meta
def dequantize_mixed_int6(result: dict[str, Tensor], meta: dict[str, object],
                          template_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        if info in ("passthrough", "passthrough_ctrl", "passthrough_fp16"):
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        q, s = result[name + ".q"], result[name + ".scale"]
        rotated = isinstance(info, dict) and info.get("rotated", False)
        if s.ndim > 0:
            recon = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1))))
        else:
            recon = (q.float() * float(s.item()))
        if rotated:
            recon = _unrotate_cols(recon)
        out[name] = recon.to(orig_dtype)
    return out


def filter_export_state_dict(sd: dict[str, Tensor], *, include_bigram: bool, include_ve: bool) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, tensor in sd.items():
        if not include_bigram and name.startswith("bigram."):
            continue
        if not include_ve and (name.startswith("ve_shared.") or name.startswith("ve_layer_scales.")):
            continue
        out[name] = tensor
    return out


def compress_artifact(raw: bytes, args: Hyperparameters) -> tuple[str, bytes, dict[str, int]]:
    candidates: list[tuple[str, bytes]] = [("lzma", lzma.compress(raw, preset=9))]
    if args.brotli_enabled:
        if not _BROTLI_AVAILABLE:
            raise ImportError("BROTLI_ENABLED=1 requires the `brotli` package to be installed")
        brotli_blob = brotli.compress(raw, quality=11)
        candidates.append(("brotli", brotli_blob))
    sizes = {name: len(blob) for name, blob in candidates}
    if args.compare_compressors and len(candidates) > 1:
        codec, blob = min(candidates, key=lambda item: len(item[1]))
    else:
        codec, blob = candidates[0]
    return codec, blob, sizes


def decompress_artifact(blob: bytes, codec: str) -> bytes:
    if codec == "lzma":
        return lzma.decompress(blob)
    if codec == "brotli":
        if not _BROTLI_AVAILABLE:
            raise ImportError("Cannot decompress Brotli artifact without the `brotli` package")
        return brotli.decompress(blob)
    raise ValueError(f"Unsupported artifact codec: {codec}")

def main() -> None:
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    resolved_avg_mode = resolve_avg_mode(args.avg_mode, args.ema_enabled, args.swa_enabled, args.lawa_enabled)
    mtp_export_mode = "aux_train_only"
    global _turbo_rounds; _turbo_rounds = max(int(args.turbo_rounds), 1)
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)
    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)
    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)
    log0(code, console=False)
    log0(f"Python {sys.version} PyTorch {torch.__version__}", console=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    effective_eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_seq_len = max(args.train_seq_len, effective_eval_seq_len)
    val_tokens = load_validation_tokens(args.val_files, val_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")
    legacy_avg_flags = [
        name for name, flag in (
            ("ema", args.ema_enabled),
            ("swa", args.swa_enabled),
            ("lawa", args.lawa_enabled),
        )
        if flag is True
    ]
    if args.avg_mode.strip():
        if legacy_avg_flags:
            log0(f"avg_mode:explicit_override avg_mode={resolved_avg_mode} legacy_flags={legacy_avg_flags}")
    CastedLinear._qat_enabled = args.qat_enabled
    base_model = instantiate_model(args, device)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model = compiled_model

    matrix_params = [
        base_model.qo_bank, base_model.kv_bank,
        base_model.mlp_up_bank, base_model.mlp_down_bank,
    ]
    block_named_params = list(base_model.blocks.named_parameters())
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    scalar_params.append(base_model.smear.gate)
    if base_model.bigram is not None:
        scalar_params.append(base_model.bigram.scale)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_params = [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]
    if base_model.bigram is not None:
        tok_params.append({"params": [base_model.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.bigram.proj is not None:
            scalar_params.append(base_model.bigram.proj.weight)
    if base_model.ve_shared is not None:
        tok_params.append({"params": [base_model.ve_shared.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.ve_shared.proj is not None:
            scalar_params.append(base_model.ve_shared.proj.weight)
        scalar_params.append(base_model.ve_shared.scale)
        for s in base_model.ve_layer_scales:
            scalar_params.append(s)
    optimizer_tok = torch.optim.AdamW(
        tok_params,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=True,
    )
    tok_bucket_params = [p for group in tok_params for p in group["params"]]
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.muon_wd,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=True,
    )
    aux_head_params: list[Tensor] = []
    aux_head_groups = []
    if base_model.lm_head is not None:
        aux_head_groups.append({"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr})
        aux_head_params.append(base_model.lm_head.weight)
    if args.mtp_num_heads > 0:
        mtp_head_params = list(base_model.mtp_heads.parameters())
        if mtp_head_params:
            aux_head_groups.append({"params": mtp_head_params, "lr": args.head_lr, "base_lr": args.head_lr})
            aux_head_params.extend(mtp_head_params)
    replicated_params = list(tok_bucket_params)
    replicated_params.extend(scalar_params)
    optimizer_heads = None
    if aux_head_groups:
        optimizer_heads = torch.optim.Adam(
            aux_head_groups,
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        replicated_params.extend(aux_head_params)
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if optimizer_heads is not None:
        optimizers.append(optimizer_heads)
    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params} avg_mode:{resolved_avg_mode} seed:{args.seed}")
    log0(f"world_size:{world_size} grad_accum:{grad_accum_steps} iterations:{args.iterations}")
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            if distributed:
                for p in base_model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    from collections import deque
    collect_ema = resolved_avg_mode == "ema"
    collect_swa = resolved_avg_mode == "swa"
    collect_lawa = resolved_avg_mode == "lawa"
    swa_state: dict[str, Tensor] | None = None
    swa_count = 0
    lawa_queue: deque[dict[str, Tensor]] = deque(maxlen=args.lawa_k if collect_lawa else 1)
    ema_state = (
        {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}
        if collect_ema else None
    )
    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        if args.late_qat_threshold > 0 and scale < args.late_qat_threshold and not CastedLinear._qat_enabled:
            CastedLinear._qat_enabled = True
            log0(f"late_qat:enabled step:{step} scale:{scale:.4f}")
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps
        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale
        optimizer_muon.launch_reduce_scatters()
        allreduce_param_grads(replicated_params)
        if args.grad_clip_norm > 0:
            clip_grad_norm_with_muon(replicated_params, optimizer_muon, args.grad_clip_norm)
        optimizer_tok.step()
        optimizer_scalar.step()
        if optimizer_heads is not None:
            optimizer_heads.step()
        optimizer_muon.step()
        zero_grad_all()
        if ema_state is not None:
            with torch.no_grad():
                for name, t in base_model.state_dict().items():
                    ema_state[name].mul_(args.ema_decay).add_(t.detach().float(), alpha=1.0 - args.ema_decay)
        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if collect_swa and scale < 0.2 and step % args.swa_every == 0:
            if swa_state is None:
                swa_state = {name: t.detach().cpu().clone() for name, t in base_model.state_dict().items()}
                swa_count = 1
                log0(f"swa:start step:{step}")
            else:
                for name, t in base_model.state_dict().items():
                    swa_state[name] += t.detach().cpu()
                swa_count += 1
        if collect_lawa and step % args.lawa_freq == 0:
            lawa_queue.append({name: t.detach().cpu().clone() for name, t in base_model.state_dict().items()})
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step
    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )
    applied_avg_mode = resolved_avg_mode
    if resolved_avg_mode == "lawa" and len(lawa_queue) > 1:
        log0(f"lawa:applying LAWA averaging k={len(lawa_queue)}")
        current_state = base_model.state_dict()
        avg_state = {name: torch.zeros(t.shape, dtype=torch.float32, device='cpu') for name, t in current_state.items()}
        for snap in lawa_queue:
            for name in avg_state:
                avg_state[name] += snap[name].float()
        for name in avg_state:
            avg_state[name] /= len(lawa_queue)
            avg_state[name] = avg_state[name].to(dtype=current_state[name].dtype)
        base_model.load_state_dict(avg_state, strict=True)
    elif resolved_avg_mode == "swa" and swa_state is not None and swa_count > 0:
        log0(f"swa:applying SWA averaging count={swa_count}")
        current_state = base_model.state_dict()
        avg_state = {
            name: (tensor / swa_count).to(dtype=current_state[name].dtype)
            for name, tensor in swa_state.items()
        }
        base_model.load_state_dict(avg_state, strict=True)
    elif resolved_avg_mode == "ema" and ema_state is not None:
        log0(f"ema:applying EMA weights decay={args.ema_decay:.6f}")
        current_state = base_model.state_dict()
        avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}
        base_model.load_state_dict(avg_state, strict=True)
    else:
        applied_avg_mode = "none"
        if resolved_avg_mode == "lawa":
            log0("lawa:fallback none reason=insufficient_snapshots")
        elif resolved_avg_mode == "swa":
            log0("swa:fallback none reason=no_swa_snapshots")
        elif resolved_avg_mode == "ema":
            log0("ema:fallback none reason=ema_state_missing")
        else:
            log0("avg_mode:none")
    torch.cuda.synchronize()
    t_diag = time.perf_counter()
    diag_val_loss, diag_val_bpb = eval_val(
        args, compiled_model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"DIAGNOSTIC post_avg mode:{applied_avg_mode} val_loss:{diag_val_loss:.4f} val_bpb:{diag_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_diag):.0f}ms"
    )
    full_state_dict = base_model.state_dict()
    export_sd = {k: v for k, v in full_state_dict.items() if "mtp_heads" not in k}
    excluded_mtp = sum(int(t.numel()) for k, t in full_state_dict.items() if "mtp_heads" in k)
    log0(f"mtp_export_mode:{mtp_export_mode} excluded_params:{excluded_mtp}")
    if master_process:
        torch.save(export_sd, "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
    code_bytes = len(code.encode("utf-8"))
    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
    full_unbanked_sd = _unbank_state_dict(sd_cpu, args.num_layers)
    hessians = None
    if args.gptq_enabled:
        log0("gptq:generating calibration data...")
        t_gptq = time.perf_counter()
        calib_tokens = generate_calibration(
            base_model, args.vocab_size, num_seqs=args.gptq_calib_seqs,
            seq_len=args.gptq_calib_len, temperature=args.gptq_temperature,
            device=device,
        )
        log0(f"gptq:generated {calib_tokens.shape[0]}x{calib_tokens.shape[1]} tokens "
             f"in {1000*(time.perf_counter()-t_gptq):.0f}ms")
        hessians = collect_hessians(base_model, calib_tokens, device)
        gptq_targets = {
            name for name, tensor in full_unbanked_sd.items()
            if tensor.is_floating_point() and tensor.ndim == 2 and _classify_param(name) in {"mlp", "attn"}
        }
        log0(f"gptq:collected {len(hessians)} hessians "
             f"in {1000*(time.perf_counter()-t_gptq):.0f}ms")
        log0(f"gptq:hessian_targets collected={len(gptq_targets & set(hessians))}/{len(gptq_targets)}")
    export_candidates: list[tuple[bool, bool, str]] = []
    seen_candidates: set[tuple[bool, bool]] = set()
    for include_bigram, include_ve, label in [
        (args.bigram_vocab_size > 0, args.ve_enabled, "full"),
        (False, args.ve_enabled, "drop_bigram"),
        (False, False, "drop_bigram_ve"),
    ]:
        key = (include_bigram, include_ve)
        if key not in seen_candidates:
            export_candidates.append((include_bigram, include_ve, label))
            seen_candidates.add(key)
    selected_export: dict[str, object] | None = None
    for include_bigram, include_ve, label in export_candidates:
        filtered_sd = filter_export_state_dict(sd_cpu, include_bigram=include_bigram, include_ve=include_ve)
        unbanked_sd = _unbank_state_dict(filtered_sd, args.num_layers)
        if args.gptq_enabled:
            quant_result, quant_meta, gptq_stats = mixed_quantize_int6_gptq(
                unbanked_sd, {"mlp", "attn"}, hessians or {},
                block_size=args.gptq_block_size,
                damp=args.gptq_damp,
            )
            log0(
                f"gptq:layer_usage candidate:{label} "
                f"used={gptq_stats['gptq_layers']} "
                f"missing_hessian={gptq_stats['fallback_missing_hessian']} "
                f"bad_hessian={gptq_stats['fallback_bad_hessian']} "
                f"cholesky_fallback={gptq_stats['fallback_cholesky']}"
            )
        else:
            quant_result, quant_meta = mixed_quantize_int6(unbanked_sd, {"mlp", "attn"})
        quant_buf = io.BytesIO()
        torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
        quant_raw = quant_buf.getvalue()
        codec, quant_blob, codec_sizes = compress_artifact(quant_raw, args)
        total_bytes = code_bytes + len(quant_blob)
        log0(
            f"artifact_candidate label:{label} include_bigram:{int(include_bigram)} include_ve:{int(include_ve)} "
            f"codec:{codec} model_bytes:{len(quant_blob)} total_bytes:{total_bytes} "
            f"lzma_bytes:{codec_sizes.get('lzma', -1)} brotli_bytes:{codec_sizes.get('brotli', -1)}"
        )
        selected_export = {
            "label": label,
            "include_bigram": include_bigram,
            "include_ve": include_ve,
            "filtered_sd": filtered_sd,
            "unbanked_sd": unbanked_sd,
            "codec": codec,
            "codec_sizes": codec_sizes,
            "quant_blob": quant_blob,
            "total_bytes": total_bytes,
        }
        if total_bytes <= args.artifact_budget_bytes:
            break
    if selected_export is None:
        raise RuntimeError("Failed to build any export candidate")
    if args.gptq_enabled:
        log0(f"gptq:quantization complete in {1000*(time.perf_counter()-t_gptq):.0f}ms")
    selected_codec = str(selected_export["codec"])
    quant_blob = selected_export["quant_blob"]
    artifact_ext = "ptbr" if selected_codec == "brotli" else "ptz"
    artifact_path = f"final_model.int6.{artifact_ext}"
    if master_process:
        with open(artifact_path, "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = len(quant_blob)
        log0(
            f"artifact_format:mixed_int6_{selected_codec} "
            f"budget_label:{selected_export['label']} "
            f"include_bigram:{int(bool(selected_export['include_bigram']))} "
            f"include_ve:{int(bool(selected_export['include_ve']))}"
        )
        log0(f"Serialized model int6+{selected_codec}: {quant_file_bytes} bytes")
        log0(f"Total submission size int6+{selected_codec}: {quant_file_bytes + code_bytes} bytes")
        if selected_export["total_bytes"] > args.artifact_budget_bytes:
            log0(
                f"artifact_budget:exceeded total_bytes:{selected_export['total_bytes']} "
                f"budget:{args.artifact_budget_bytes} using_last_candidate:{selected_export['label']}"
            )
    if distributed:
        dist.barrier()
    with open(artifact_path, "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(
        io.BytesIO(decompress_artifact(quant_blob_disk, selected_codec)),
        map_location="cpu",
    )
    selected_unbanked_sd = selected_export["unbanked_sd"]
    selected_filtered_sd = selected_export["filtered_sd"]
    deq_unbanked = dequantize_mixed_int6(quant_state["w"], quant_state["m"], selected_unbanked_sd)
    deq_state = _rebank_state_dict(deq_unbanked, args.num_layers, selected_filtered_sd)
    eval_model = instantiate_model(
        args,
        device,
        mtp_num_heads=0,
        mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size if selected_export["include_bigram"] else 0,
        ve_enabled=bool(selected_export["include_ve"]),
    )
    eval_model.load_state_dict(deq_state, strict=True)
    compiled_eval = torch.compile(eval_model, dynamic=False, fullgraph=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args, compiled_eval, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        eval_seq_len=effective_eval_seq_len,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int6_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int6_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")
    sw_seq_len = effective_eval_seq_len
    if args.eval_stride > 0 and args.eval_stride < sw_seq_len:
        torch.cuda.synchronize()
        t_slide = time.perf_counter()
        sw_val_loss, sw_val_bpb = eval_val_sliding(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride,
            eval_seq_len=sw_seq_len,
        )
        torch.cuda.synchronize()
        log0(
            f"final_int6_sliding_window val_loss:{sw_val_loss:.4f} val_bpb:{sw_val_bpb:.4f} "
            f"stride:{args.eval_stride} eval_time:{1000.0 * (time.perf_counter() - t_slide):.0f}ms"
        )
        log0(f"final_int6_sliding_window_exact val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}")
        log0(f"final_int6_{selected_codec}_roundtrip_exact val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}")
    # Legal score-first TTT (PR #461 recipe)
    if args.ttt_enabled:
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        ttt_loss, ttt_bpb = eval_val_sliding_ttt(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride, log0=log0,
        )
        torch.cuda.synchronize()
        log0(f"legal_ttt val_loss:{ttt_loss:.4f} val_bpb:{ttt_bpb:.4f} "
             f"eval_time:{1000.0 * (time.perf_counter() - t_ttt):.0f}ms")
        log0(f"legal_ttt_exact val_loss:{ttt_loss:.8f} val_bpb:{ttt_bpb:.8f}")
    if distributed:
        dist.destroy_process_group()
if __name__ == "__main__":
    main()
