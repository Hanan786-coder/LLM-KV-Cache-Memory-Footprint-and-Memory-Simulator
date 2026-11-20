"""Scaling curves: how latency and throughput change with batch, context and prompt length.

All curves come from the analytic steps in ``perf.py`` (no simulation), and each
point says whether it fits in GPU memory (Part 1's budget), so a curve shows both
where the GPU runs out of bandwidth and where it runs out of memory.

* ``batch_curve``   -- decode at a fixed context, growing batch. Weights are read
  once per step whatever the batch, so weight bytes per token fall as 1/batch and
  throughput rises until KV reads or compute take over.
* ``context_curve`` -- decode at a fixed batch, growing context: KV reads grow
  linearly, so step latency grows and tokens/s falls.
* ``prefill_curve`` -- one prompt of growing length: time to first token, and the
  point where prefill crosses from memory- to compute-bound.
"""

from __future__ import annotations

from . import budget as bud
from .perf import PerfModel, StepCost

DEFAULT_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
DEFAULT_CONTEXTS = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
DEFAULT_PROMPTS = (1, 4, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
CURVES = ("batch", "context", "prompt")


def _fits(perf: PerfModel, batch: int, context: int, prefill_chunk: int, reserved: int) -> bool:
    scenario = bud.Scenario(perf.model, perf.gpu, context_len=context, batch=batch,
                            weight_precision=perf.weight_precision, kv_precision=perf.kv_precision,
                            prefill_chunk=prefill_chunk, reserved_bytes=reserved)
    return bud.fits(scenario)


def _point(cost: StepCost, batch: int, context: int, fits: bool) -> dict:
    tokens = max(cost.logits, 1)
    return {
        "batch": batch,
        "context": context,
        "latency_ms": cost.latency_s * 1e3,
        "tokens_per_s": cost.tokens_per_s,
        "tokens_per_s_per_seq": cost.tokens_per_s / batch,
        "bytes_weights_per_token": cost.weight_bytes / tokens,
        "bytes_kv_per_token": (cost.kv_read_bytes + cost.kv_write_bytes) / tokens,
        "intensity": cost.intensity,
        "bound": cost.bound,
        "fits": fits,
    }


def batch_curve(perf: PerfModel, batches=DEFAULT_BATCHES, context: int = 1024,
                prefill_chunk: int = bud.DEFAULT_PREFILL_CHUNK,
                reserved_bytes: int = bud.DEFAULT_RESERVED_BYTES) -> list[dict]:
    """Decode step at ``context`` tokens for each batch size."""
    return [_point(perf.decode(b, context), b, context,
                   _fits(perf, b, context + 1, prefill_chunk, reserved_bytes)) for b in batches]


def context_curve(perf: PerfModel, contexts=DEFAULT_CONTEXTS, batch: int = 1,
                  prefill_chunk: int = bud.DEFAULT_PREFILL_CHUNK,
                  reserved_bytes: int = bud.DEFAULT_RESERVED_BYTES) -> list[dict]:
    """Decode step with ``batch`` sequences for each context length."""
    return [_point(perf.decode(batch, c), batch, c,
                   _fits(perf, batch, c + 1, prefill_chunk, reserved_bytes)) for c in contexts]


def prefill_curve(perf: PerfModel, prompts=DEFAULT_PROMPTS,
                  prefill_chunk: int = bud.DEFAULT_PREFILL_CHUNK,
                  reserved_bytes: int = bud.DEFAULT_RESERVED_BYTES) -> list[dict]:
    """Time to first token of a single prompt of each length (one unchunked pass)."""
    rows = []
    for p in prompts:
        cost = perf.prefill(p)
        rows.append({
            "prompt": p,
            "ttft_ms": cost.latency_s * 1e3,
            "prompt_tokens_per_s": p / cost.latency_s if cost.latency_s else 0.0,
            "flops": cost.flops,
            "hbm_bytes": cost.hbm_bytes,
            "intensity": cost.intensity,
            "bound": cost.bound,
            "fits": _fits(perf, 1, p, max(prefill_chunk, p), reserved_bytes),
        })
    return rows


def prefill_compute_bound_prompt(perf: PerfModel, limit: int = 1 << 20) -> int | None:
    """Shortest prompt whose prefill is compute-bound (None up to ``limit``)."""
    if perf.prefill(limit).bound == "memory":
        return None
    lo, hi = 0, limit
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if perf.prefill(mid).bound == "compute":
            hi = mid
        else:
            lo = mid
    return hi


def best_batch(rows: list[dict]) -> dict | None:
    """The fitting point of a batch curve with the highest throughput."""
    fitting = [r for r in rows if r["fits"]]
    return max(fitting, key=lambda r: r["tokens_per_s"]) if fitting else None
