"""What to do when the paged allocator runs out of blocks mid-generation.

Eviction is all-or-nothing per request (vLLM paper, Sec. 4.5): the victim gives
up every block it holds. Which request becomes the victim:

    lifo     -- the latest-arrived running request (vLLM's choice; least work lost)
    fifo     -- the earliest-arrived running request (the classic replacement order)
    largest  -- the request holding the most blocks (frees the most per eviction)

What happens to it (``preempt_mode``):

    recompute -- its blocks are freed and it goes back to the front of the queue;
                 when re-admitted it prefills everything it knew again
    swap      -- its blocks are copied to CPU swap space and copied back later;
                 falls back to recompute when swap space is full

Contiguous allocators never preempt: they reserve the whole sequence up front.
"""

from __future__ import annotations

from typing import Callable

from .config import ConfigError

VICTIM_POLICIES = ("lifo", "fifo", "largest")
PREEMPT_MODES = ("recompute", "swap")


def choose_victim(policy: str, candidates: list, held_tokens: Callable[[int], int]):
    """Pick the request to evict from ``candidates`` (objects with ``.id`` and ``.key``).

    ``key`` is (arrival_step, id), so ties are always broken the same way.
    """
    if not candidates:
        raise ValueError("no running request to evict")
    if policy == "lifo":
        return max(candidates, key=lambda s: s.key)
    if policy == "fifo":
        return min(candidates, key=lambda s: s.key)
    if policy == "largest":
        return max(candidates, key=lambda s: (held_tokens(s.id), s.key))  # ties: newest
    raise ConfigError(f"unknown eviction policy {policy!r}; choose from {', '.join(VICTIM_POLICIES)}")


class SwapSpace:
    """CPU memory that holds swapped-out KV blocks, counted in token slots."""

    def __init__(self, capacity_tokens: int) -> None:
        if capacity_tokens < 0:
            raise ConfigError("swap space cannot be negative")
        self.capacity_tokens = capacity_tokens
        self.used_tokens = 0

    def can_hold(self, tokens: int) -> bool:
        return self.used_tokens + tokens <= self.capacity_tokens

    def store(self, tokens: int) -> None:
        if not self.can_hold(tokens):
            raise RuntimeError("swap space is full")
        self.used_tokens += tokens

    def release(self, tokens: int) -> None:
        if tokens > self.used_tokens:
            raise RuntimeError("releasing more swap than is in use")
        self.used_tokens -= tokens
