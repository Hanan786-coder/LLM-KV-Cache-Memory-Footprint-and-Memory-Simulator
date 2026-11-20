"""The KV memory pool: the GPU memory left for KV-cache after everything else.

    pool_bytes  = GPU capacity - weights - activations(one full prefill chunk) - reserved
    pool_tokens = pool_bytes // kv_bytes_per_token
    num_blocks  = pool_tokens // block_size

The activations are those of the worst single forward pass the scheduler can run:
a full prefill chunk plus the logits of every running sequence. Part 1's
``kv_token_capacity`` reserves activations for one token only, so it reports a
few hundred more token slots (458 for LLaMA-7B in FP16).
"""

from __future__ import annotations

from dataclasses import dataclass

from .budget import Scenario, memory_budget
from .config import ConfigError
from .footprint import kv_bytes_per_token

DEFAULT_MAX_NUM_SEQS = 256  # vLLM's default limit on running sequences


def _check_int(label: str, value, low: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < low:
        raise ConfigError(f"{label} must be an integer >= {low}, got {value!r}")


@dataclass(frozen=True)
class KVPool:
    """KV memory measured in token slots. One slot holds one token's K and V in every layer."""

    bytes_per_token: int
    total_tokens: int

    def __post_init__(self) -> None:
        _check_int("bytes_per_token", self.bytes_per_token, 1)
        _check_int("total_tokens", self.total_tokens, 0)

    @property
    def total_bytes(self) -> int:
        return self.total_tokens * self.bytes_per_token

    def num_blocks(self, block_size: int) -> int:
        """Whole blocks of ``block_size`` tokens that fit; leftover slots are unusable."""
        _check_int("block_size", block_size, 1)
        return self.total_tokens // block_size

    def block_bytes(self, block_size: int) -> int:
        _check_int("block_size", block_size, 1)
        return block_size * self.bytes_per_token


def kv_pool_bytes(s: Scenario, max_num_seqs: int = DEFAULT_MAX_NUM_SEQS) -> int:
    """GPU bytes left for KV-cache; negative when the weights alone do not fit."""
    _check_int("max_num_seqs", max_num_seqs, 1)
    b = memory_budget(s.with_(context_len=s.prefill_chunk, batch=max_num_seqs))
    return b.capacity - b.weights - b.activations - b.reserved  # b.kv_cache is ignored


def pool_from_scenario(
    s: Scenario,
    max_num_seqs: int = DEFAULT_MAX_NUM_SEQS,
    pool_tokens: int | None = None,
    pool_bytes: int | None = None,
) -> KVPool:
    """The KV pool for a deployment. ``pool_tokens`` or ``pool_bytes`` override the size."""
    per_token = kv_bytes_per_token(s.model, s.kv_precision)
    if pool_tokens is not None and pool_bytes is not None:
        raise ConfigError("give pool_tokens or pool_bytes, not both")
    if pool_tokens is not None:
        _check_int("pool_tokens", pool_tokens, 1)
        return KVPool(per_token, pool_tokens)
    if pool_bytes is not None:
        _check_int("pool_bytes", pool_bytes, per_token)
        return KVPool(per_token, pool_bytes // per_token)
    free = kv_pool_bytes(s, max_num_seqs)
    if free < per_token:
        raise ConfigError(
            f"no GPU memory left for KV-cache ({s.model.name} on {s.gpu.name}, weights "
            f"{s.weight_precision.name}); try --weight-dtype int8/int4 or a bigger GPU")
    return KVPool(per_token, free // per_token)
