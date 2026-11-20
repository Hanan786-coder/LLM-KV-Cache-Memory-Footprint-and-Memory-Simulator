"""Full VRAM budget, OOM detection and max-batch / max-context solvers.

    VRAM used = weights + KV-cache + activation workspace + reserved

* weights    -- see ``weights.weight_bytes``
* KV-cache   -- ``batch x context x kv_bytes_per_token``
* activations-- peak transient tensors of one transformer layer during a forward
                pass. Layers run one after another, so only one layer's
                activations are alive at a time. Assumes fused/flash attention,
                so the context x context score matrix is never materialized.
                Prefill is chunked: at most ``prefill_chunk`` tokens (summed over
                the batch) go through the model in one pass, like vLLM's
                ``max_num_batched_tokens``.
* reserved   -- CUDA context, cuBLAS workspace and allocator slack.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .config import GIB, GPUConfig, ModelConfig
from .footprint import kv_bytes_per_token, kv_cache_bytes
from .precision import FP16, Precision
from .weights import weight_bytes

DEFAULT_PREFILL_CHUNK = 2048
DEFAULT_RESERVED_BYTES = 1 * GIB
LOGITS_BYTES = 4  # logits are computed in FP32
SOLVER_LIMIT = 1 << 24


@dataclass(frozen=True)
class Scenario:
    """One deployment: a model on a GPU serving ``batch`` sequences of ``context_len`` tokens."""

    model: ModelConfig
    gpu: GPUConfig
    context_len: int
    batch: int = 1
    weight_precision: Precision = FP16
    kv_precision: Precision = FP16
    act_precision: Precision = FP16
    prefill_chunk: int = DEFAULT_PREFILL_CHUNK
    reserved_bytes: int = DEFAULT_RESERVED_BYTES

    def __post_init__(self) -> None:
        if self.context_len < 0 or self.batch < 0:
            raise ValueError("context_len and batch must be >= 0")
        if self.prefill_chunk <= 0 or self.reserved_bytes < 0:
            raise ValueError("prefill_chunk must be > 0 and reserved_bytes >= 0")

    def with_(self, **changes) -> "Scenario":
        return replace(self, **changes)


@dataclass(frozen=True)
class MemoryBudget:
    capacity: int
    weights: int
    kv_cache: int
    activations: int
    reserved: int

    @property
    def total(self) -> int:
        return self.weights + self.kv_cache + self.activations + self.reserved

    @property
    def free(self) -> int:
        """Bytes left on the GPU; negative means out of memory."""
        return self.capacity - self.total

    @property
    def fits(self) -> bool:
        return self.free >= 0

    @property
    def utilization(self) -> float:
        return self.total / self.capacity

    def rows(self) -> list[dict]:
        parts = [("weights", self.weights), ("kv_cache", self.kv_cache),
                 ("activations", self.activations), ("reserved", self.reserved),
                 ("total", self.total), ("gpu_capacity", self.capacity), ("free", self.free)]
        return [{"component": name, "bytes": value, "share_of_gpu": value / self.capacity}
                for name, value in parts]


def tokens_in_flight(s: Scenario) -> int:
    """Largest number of tokens processed in a single forward pass."""
    if s.batch == 0:
        return 0
    all_tokens = s.batch * max(s.context_len, 1)
    return min(all_tokens, max(s.prefill_chunk, s.batch))  # decode needs >= 1 token per sequence


def activation_bytes_per_token(model: ModelConfig, act_precision: Precision = FP16) -> int:
    """Peak live activation values per token inside one transformer layer."""
    qkv = (model.num_heads + 2 * model.num_kv_heads) * model.head_dim
    mlp_hidden = (2 if model.mlp_type == "swiglu" else 1) * model.intermediate_size
    # residual stream, normed input, attention output, MLP output + QKV + MLP hidden
    values = 4 * model.hidden_size + qkv + mlp_hidden
    return act_precision.bytes_for(values)


def activation_bytes(s: Scenario) -> int:
    per_token = activation_bytes_per_token(s.model, s.act_precision)
    logits = s.batch * s.model.vocab_size * LOGITS_BYTES  # last position of each sequence
    return tokens_in_flight(s) * per_token + logits


def memory_budget(s: Scenario) -> MemoryBudget:
    return MemoryBudget(
        capacity=s.gpu.memory_bytes,
        weights=weight_bytes(s.model, s.weight_precision),
        kv_cache=kv_cache_bytes(s.model, s.kv_precision, s.context_len, s.batch),
        activations=activation_bytes(s),
        reserved=s.reserved_bytes,
    )


def fits(s: Scenario) -> bool:
    return memory_budget(s).fits


def _largest_fitting(predicate, limit: int) -> int:
    """Largest n in [0, limit] with predicate(n) true, for a monotone predicate."""
    if not predicate(1):
        return 0
    lo, hi = 1, 2
    while hi <= limit and predicate(hi):  # exponential search for an upper bound
        lo, hi = hi, hi * 2
    hi = min(hi, limit + 1)
    while hi - lo > 1:  # invariant: predicate(lo) true, predicate(hi) false or hi > limit
        mid = (lo + hi) // 2
        if predicate(mid):
            lo = mid
        else:
            hi = mid
    return lo


def max_batch(s: Scenario, limit: int = SOLVER_LIMIT) -> int:
    """Most sequences of ``s.context_len`` tokens that fit before OOM (0 = none)."""
    return _largest_fitting(lambda b: fits(s.with_(batch=b)), limit)


def max_context(s: Scenario, limit: int = SOLVER_LIMIT) -> int:
    """Longest context each of ``s.batch`` sequences can reach before OOM (0 = none)."""
    return _largest_fitting(lambda c: fits(s.with_(context_len=c)), limit)


def kv_token_capacity(s: Scenario) -> int:
    """KV-cache tokens (summed over all sequences) that fit after everything else.

    This is the size of the KV memory pool that the Part 2 allocators manage.
    """
    budget = memory_budget(s.with_(context_len=0))
    return max(budget.free, 0) // kv_bytes_per_token(s.model, s.kv_precision)


def oom_report(s: Scenario) -> dict:
    """Whether the scenario fits, and where its limits are."""
    budget = memory_budget(s)
    static = budget.weights + budget.reserved
    return {
        "fits": budget.fits,
        "used_bytes": budget.total,
        "free_bytes": budget.free,
        "weights_fit": static <= budget.capacity,
        "max_batch_at_context": max_batch(s),
        "max_context_at_batch": max_context(s),
        "kv_token_capacity": kv_token_capacity(s),
        "model_max_context": s.model.max_context,
    }
