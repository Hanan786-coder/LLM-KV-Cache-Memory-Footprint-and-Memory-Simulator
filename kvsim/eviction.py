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
CACHE_POLICIES = ("lru", "fifo", "opt")


def choose_victim(policy: str, candidates: list, held_tokens: Callable[[object], int]):
    """Pick the running request to evict from ``candidates`` (objects with a ``.key``).

    ``key`` is (arrival_step, id), so ties are always broken the same way.
    ``held_tokens(candidate)`` gives how many slots a candidate holds.
    """
    if not candidates:
        raise ValueError("no running request to evict")
    if policy == "lifo":
        return max(candidates, key=lambda s: s.key)
    if policy == "fifo":
        return min(candidates, key=lambda s: s.key)
    if policy == "largest":
        return max(candidates, key=lambda s: (held_tokens(s), s.key))  # ties: newest
    raise ConfigError(f"unknown eviction policy {policy!r}; choose from {', '.join(VICTIM_POLICIES)}")


def choose_cached_victim(policy: str, entries: list):
    """Pick the idle session cache to drop (sessions.py), like a page-replacement policy.

    Each entry has ``session_id``, ``resident_since`` (when it entered the cache),
    ``last_used`` (when its last turn finished) and ``next_arrival``.

        lru  -- least recently used
        fifo -- in the cache the longest
        opt  -- Belady: needed again furthest in the future
    """
    if not entries:
        raise ValueError("no cached session to evict")
    if policy == "lru":
        return min(entries, key=lambda e: (e.last_used, e.session_id))
    if policy == "fifo":
        return min(entries, key=lambda e: (e.resident_since, e.session_id))
    if policy == "opt":
        return max(entries, key=lambda e: (e.next_arrival, e.session_id))
    raise ConfigError(f"unknown cache policy {policy!r}; choose from {', '.join(CACHE_POLICIES)}")


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
