"""KV-cache allocators: how the pool's token slots are handed out to requests.

Every allocator manages one ``KVPool`` and keeps, for each request it serves,
how many token slots the request has claimed (``num_tokens``) and how many it
holds (``held_tokens``). Held minus claimed is slack the request cannot use yet.

* ``PagedAllocator`` -- vLLM's PagedAttention: fixed-size blocks of ``block_size``
  tokens, a block table per request and a free block list.
* ``ContiguousAllocator`` -- one reservation per request in a single address
  space, placed first-fit or best-fit; sized like the vLLM paper's Orca baselines.

Requests are identified by an integer ``seq_id``. All sizes are in token slots;
multiply by ``pool.bytes_per_token`` for bytes.
"""

from __future__ import annotations

import bisect
import math
from abc import ABC, abstractmethod

from .config import ConfigError
from .kvpool import KVPool
from .workload import Request

BLOCK_TABLE_ENTRY_BYTES = 4  # one int32 block id per entry


class AllocationError(RuntimeError):
    """An allocator call that is not allowed right now (check ``can_*`` first)."""


def _check_window(window: int | None, sink_tokens: int) -> None:
    if window is not None and (not isinstance(window, int) or isinstance(window, bool) or window < 1):
        raise ConfigError(f"window must be a positive number of tokens, got {window!r}")
    if not isinstance(sink_tokens, int) or isinstance(sink_tokens, bool) or sink_tokens < 0:
        raise ConfigError(f"sink_tokens must be an integer >= 0, got {sink_tokens!r}")
    if sink_tokens and window is None:
        raise ConfigError("sink_tokens only make sense with a sliding window")


class KVAllocator(ABC):
    """What the scheduler needs from any allocator."""

    name: str
    pool: KVPool

    @property
    @abstractmethod
    def capacity_tokens(self) -> int:
        """Token slots this allocator can hand out (allocated + free)."""

    # --- admission and growth ---
    @abstractmethod
    def can_admit(self, seq_id: int, n_tokens: int, request: Request | None = None,
                  reserve_watermark: bool = True) -> bool:
        """Whether a new request can claim its first ``n_tokens`` slots now.

        ``reserve_watermark=False`` lets the paged allocator use its last free blocks
        (for swap-ins, and when nothing else is running).
        """

    @abstractmethod
    def admit(self, seq_id: int, n_tokens: int, request: Request | None = None,
              reserve_watermark: bool = True) -> None:
        """Claim slots for a new request's first ``n_tokens`` tokens (its prompt)."""

    @abstractmethod
    def can_append(self, seq_id: int, n: int = 1) -> bool:
        """Whether a request can claim ``n`` more slots now."""

    @abstractmethod
    def append(self, seq_id: int, n: int = 1) -> None:
        """Claim ``n`` more slots for a request."""

    @abstractmethod
    def free(self, seq_id: int) -> int:
        """Release everything a request holds; returns the slots freed."""

    # --- accounting, in token slots ---
    @abstractmethod
    def num_tokens(self, seq_id: int) -> int:
        """Slots the request has claimed."""

    @abstractmethod
    def held_tokens(self, seq_id: int) -> int:
        """Slots the request holds (claimed plus unusable slack)."""

    @abstractmethod
    def allocated_tokens(self) -> int:
        """Slots held by all requests together."""

    @abstractmethod
    def largest_free_run(self) -> int:
        """Biggest single allocation possible right now."""

    @abstractmethod
    def check_invariants(self) -> None:
        """Raise AssertionError if the internal state is inconsistent (used by tests)."""

    @abstractmethod
    def __contains__(self, seq_id: int) -> bool:
        """Whether the request currently holds memory here."""

    def free_tokens(self) -> int:
        return self.capacity_tokens - self.allocated_tokens()


class PagedAllocator(KVAllocator):
    """Fixed-size blocks, a block table per request and a LIFO free list.

    Any free block can serve any request, so there is no external fragmentation.
    The only waste is the unfilled end of each request's last block.

    Sliding window (``window`` W, ``sink_tokens`` S): a request keeps token
    positions ``[0, S)`` (the attention sinks) and its latest W positions. Logical
    block ``i`` covers positions ``[i*B, (i+1)*B)`` and is kept while it overlaps
    either range; ``append`` frees blocks the window has moved past.
    """

    def __init__(self, pool: KVPool, block_size: int = 16, watermark: float = 0.01,
                 window: int | None = None, sink_tokens: int = 0) -> None:
        _check_window(window, sink_tokens)
        self.window = window
        self.sink_tokens = sink_tokens
        self.sink_blocks = math.ceil(sink_tokens / block_size) if block_size > 0 else 0
        self.pool = pool
        self.block_size = block_size
        self.num_blocks = pool.num_blocks(block_size)  # validates block_size
        if self.num_blocks == 0:
            raise ConfigError(f"the KV pool ({pool.total_tokens} token slots) is smaller "
                              f"than one block of {block_size} tokens")
        if not 0 <= watermark < 1:
            raise ConfigError(f"watermark must be in [0, 1), got {watermark!r}")
        self.watermark_blocks = int(watermark * self.num_blocks)  # as vLLM computes it
        self.name = f"paged-{block_size}"
        self._free: list[int] = list(range(self.num_blocks - 1, -1, -1))  # pop() gives block 0 first
        self._ref_count = [0] * self.num_blocks  # always 0 or 1 until prefix sharing (stretch)
        self._tables: dict[int, dict[int, int]] = {}  # seq_id -> {logical block: physical block}
        self._end: dict[int, int] = {}  # positions written or claimed so far
        self._start: dict[int, int] = {}  # first kept position after the sinks

    # --- helpers ---
    def blocks_for(self, n_tokens: int) -> int:
        return math.ceil(n_tokens / self.block_size)

    def _window_start(self, end: int, start: int | None = None) -> int:
        start = self.sink_tokens if start is None else start
        return start if self.window is None else max(start, end - self.window)

    def _live_blocks(self, start: int, end: int) -> list[int]:
        """Logical blocks kept when positions [0, S) and [start, end) are live."""
        n_end = self.blocks_for(end)
        sinks = min(self.sink_blocks, n_end)
        return list(range(sinks)) + list(range(max(start // self.block_size, sinks), n_end))

    def _live_count(self, start: int, end: int) -> int:
        n_end = self.blocks_for(end)
        sinks = min(self.sink_blocks, n_end)
        return sinks + max(0, n_end - max(start // self.block_size, sinks))

    def max_blocks(self, positions: int) -> int:
        """Most blocks a request ever holds while growing to ``positions`` tokens."""
        full = self.blocks_for(positions)
        if self.window is None:
            return full
        return min(full, self.sink_blocks + self.blocks_for(self.window) + 1)

    def _table(self, seq_id: int) -> dict[int, int]:
        try:
            return self._tables[seq_id]
        except KeyError:
            raise AllocationError(f"request {seq_id} holds no blocks") from None

    def _take(self) -> int:
        block = self._free.pop()
        self._ref_count[block] = 1
        return block

    def _release(self, block: int) -> None:
        self._ref_count[block] = 0
        self._free.append(block)

    @staticmethod
    def _check_count(n: int) -> None:
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise AllocationError(f"token count must be a positive integer, got {n!r}")

    # --- block-level view ---
    @property
    def capacity_tokens(self) -> int:
        return self.num_blocks * self.block_size

    @property
    def blocks_free(self) -> int:
        return len(self._free)

    @property
    def blocks_used(self) -> int:
        return self.num_blocks - len(self._free)

    @property
    def block_table_bytes(self) -> int:
        return BLOCK_TABLE_ENTRY_BYTES * self.blocks_used

    def block_table(self, seq_id: int) -> list[int]:
        """Physical blocks the request holds, in logical order."""
        return list(self._table(seq_id).values())

    # --- admission and growth ---
    def can_admit(self, seq_id: int, n_tokens: int, request: Request | None = None,
                  reserve_watermark: bool = True) -> bool:
        self._check_count(n_tokens)
        if seq_id in self._tables:
            return False
        # keep a watermark of free blocks so a new request does not force an
        # eviction on its very next step; growth (append) ignores it
        keep = self.watermark_blocks if reserve_watermark else 0
        need = self._live_count(self._window_start(n_tokens), n_tokens)
        return len(self._free) - need >= keep

    def admit(self, seq_id: int, n_tokens: int, request: Request | None = None,
              reserve_watermark: bool = True) -> None:
        """Claim the prompt's slots; with a window only the sinks and the last W are kept."""
        if seq_id in self._tables:
            raise AllocationError(f"request {seq_id} is already admitted")
        if not self.can_admit(seq_id, n_tokens, request, reserve_watermark):
            raise AllocationError(f"not enough free blocks to admit request {seq_id}")
        start = self._window_start(n_tokens)
        self._tables[seq_id] = {i: self._take() for i in self._live_blocks(start, n_tokens)}
        self._start[seq_id], self._end[seq_id] = start, n_tokens

    def _net_new_blocks(self, seq_id: int, n: int) -> int:
        table = self._table(seq_id)  # raises AllocationError for an unknown request
        end = self._end[seq_id] + n
        return self._live_count(self._window_start(end, self._start[seq_id]), end) - len(table)

    def can_append(self, seq_id: int, n: int = 1) -> bool:
        self._check_count(n)
        return self._net_new_blocks(seq_id, n) <= len(self._free)

    def append(self, seq_id: int, n: int = 1) -> None:
        """Claim ``n`` more positions; with a window, first free blocks it has moved past."""
        if not self.can_append(seq_id, n):
            raise AllocationError(f"no free block for request {seq_id}")
        table, old_start, old_end = self._tables[seq_id], self._start[seq_id], self._end[seq_id]
        end = old_end + n
        start = self._window_start(end, old_start)
        b, sinks = self.block_size, self.sink_blocks
        for i in range(max(old_start // b, sinks), max(start // b, sinks)):  # now behind the window
            if i in table:
                self._release(table.pop(i))
        for i in self._live_blocks(start, end):
            if i not in table:
                table[i] = self._take()
        self._start[seq_id], self._end[seq_id] = start, end

    def free(self, seq_id: int) -> int:
        table = self._table(seq_id)
        for block in reversed(list(table.values())):  # its first block is reused first
            self._release(block)
        del self._tables[seq_id], self._start[seq_id], self._end[seq_id]
        return len(table) * self.block_size

    # --- accounting ---
    def num_tokens(self, seq_id: int) -> int:
        """Positions the request keeps: the sinks plus its window (all of them without one)."""
        self._table(seq_id)
        start, end, s = self._start[seq_id], self._end[seq_id], self.sink_tokens
        return min(end, s) + max(0, end - max(start, s))

    def held_tokens(self, seq_id: int) -> int:
        return len(self._table(seq_id)) * self.block_size

    def allocated_tokens(self) -> int:
        return self.blocks_used * self.block_size

    def largest_free_run(self) -> int:
        return self.free_tokens()

    def __contains__(self, seq_id: int) -> bool:
        return seq_id in self._tables

    def check_invariants(self) -> None:
        held = [block for table in self._tables.values() for block in table.values()]
        everything = held + self._free
        assert sorted(everything) == list(range(self.num_blocks)), "a block is lost or listed twice"
        assert all(self._ref_count[b] == 1 for b in held), "held block with ref count != 1"
        assert all(self._ref_count[b] == 0 for b in self._free), "free block with ref count != 0"
        assert self._tables.keys() == self._end.keys() == self._start.keys(), "per-request state disagrees"
        for seq_id, table in self._tables.items():
            start, end = self._start[seq_id], self._end[seq_id]
            assert end >= 1 and list(table) == self._live_blocks(start, end), (
                f"request {seq_id}: blocks {sorted(table)} do not match positions {start}..{end}")
            if self.window is not None:
                assert self.num_tokens(seq_id) <= self.sink_tokens + self.window
        assert self.allocated_tokens() + self.free_tokens() == self.capacity_tokens


# --- contiguous allocation ----------------------------------------------------

RESERVATIONS = ("max", "request", "pow2", "oracle")
FITS = ("first", "best")


def next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


def reservation_tokens(request: Request, strategy: str, max_model_len: int) -> int:
    """Slots a contiguous allocator reserves for ``request`` (vLLM paper, Sec. 6.1).

    max     -- the model's maximum length (Orca Max, FasterTransformer)
    request -- prompt + the client's max_new_tokens
    pow2    -- prompt + true output length rounded up to a power of two (Orca Pow2)
    oracle  -- exactly the peak, prompt + output - 1 (Orca Oracle)
    """
    if strategy == "max":
        size = max_model_len
    elif strategy == "request":
        size = request.prompt_len + request.new_token_limit(max_model_len)
    elif strategy == "pow2":
        size = request.prompt_len + next_power_of_two(request.output_len)
    elif strategy == "oracle":
        size = request.peak_kv_tokens
    else:
        raise ConfigError(f"unknown reservation {strategy!r}; choose from {', '.join(RESERVATIONS)}")
    return min(size, max_model_len)


class ContiguousAllocator(KVAllocator):
    """One region per request in a single address space of token slots.

    The region is reserved at admission and never grows, so a running request can
    always append. Freed regions become holes that merge with their neighbours. A
    request can be blocked even when enough memory is free in total, because no
    single hole is big enough: external fragmentation.
    """

    def __init__(self, pool: KVPool, max_model_len: int, reservation: str = "max",
                 fit: str = "first", window: int | None = None, sink_tokens: int = 0) -> None:
        _check_window(window, sink_tokens)
        # with a window the region is a ring buffer of sinks + window slots
        self.ring_tokens = None if window is None else sink_tokens + window
        if reservation not in RESERVATIONS:
            raise ConfigError(f"unknown reservation {reservation!r}; choose from {', '.join(RESERVATIONS)}")
        if fit not in FITS:
            raise ConfigError(f"unknown fit {fit!r}; choose from {', '.join(FITS)}")
        if not isinstance(max_model_len, int) or isinstance(max_model_len, bool) or max_model_len < 2:
            raise ConfigError(f"max_model_len must be an integer >= 2, got {max_model_len!r}")
        if pool.total_tokens == 0:
            raise ConfigError("the KV pool has no token slots")
        self.pool = pool
        self.max_model_len = max_model_len
        self.reservation = reservation
        self.fit = fit
        self.name = f"contig-{reservation}" + ("-best" if fit == "best" else "")
        self._hole_starts: list[int] = [0]  # sorted addresses of free regions
        self._hole_sizes: dict[int, int] = {0: pool.total_tokens}
        self._regions: dict[int, tuple[int, int]] = {}  # seq_id -> (start, size)
        self._tokens: dict[int, int] = {}
        self._allocated = 0

    @property
    def capacity_tokens(self) -> int:
        return self.pool.total_tokens

    def reservation_for(self, request: Request) -> int:
        size = reservation_tokens(request, self.reservation, self.max_model_len)
        return size if self.ring_tokens is None else min(size, self.ring_tokens)

    def holes(self) -> list[tuple[int, int]]:
        """Free regions as (start, size), in address order."""
        return [(s, self._hole_sizes[s]) for s in self._hole_starts]

    def region(self, seq_id: int) -> tuple[int, int]:
        try:
            return self._regions[seq_id]
        except KeyError:
            raise AllocationError(f"request {seq_id} holds no region") from None

    def _find_hole(self, size: int) -> int | None:
        fitting = [s for s in self._hole_starts if self._hole_sizes[s] >= size]
        if not fitting:
            return None
        if self.fit == "first":
            return fitting[0]
        return min(fitting, key=lambda s: (self._hole_sizes[s], s))  # smallest, then lowest address

    def _size_for(self, n_tokens: int, request: Request | None) -> int:
        PagedAllocator._check_count(n_tokens)
        if request is None:
            raise AllocationError("the contiguous allocator needs the request to size its reservation")
        size = self.reservation_for(request)
        if n_tokens > size and self.ring_tokens is None:  # a ring buffer keeps only the last ones
            raise AllocationError(f"request {request.id}: {n_tokens} tokens do not fit "
                                  f"its {size}-slot reservation")
        return size

    # --- admission and growth ---
    def can_admit(self, seq_id: int, n_tokens: int, request: Request | None = None,
                  reserve_watermark: bool = True) -> bool:  # no watermark here
        size = self._size_for(n_tokens, request)
        return seq_id not in self._regions and self._find_hole(size) is not None

    def admit(self, seq_id: int, n_tokens: int, request: Request | None = None,
              reserve_watermark: bool = True) -> None:
        if seq_id in self._regions:
            raise AllocationError(f"request {seq_id} is already admitted")
        size = self._size_for(n_tokens, request)
        start = self._find_hole(size)
        if start is None:
            raise AllocationError(f"no hole of {size} slots for request {seq_id}")
        hole = self._hole_sizes.pop(start)
        self._hole_starts.remove(start)
        if hole > size:  # the rest of the hole stays free
            bisect.insort(self._hole_starts, start + size)
            self._hole_sizes[start + size] = hole - size
        self._regions[seq_id] = (start, size)
        self._tokens[seq_id] = min(n_tokens, size)
        self._allocated += size

    def can_append(self, seq_id: int, n: int = 1) -> bool:
        PagedAllocator._check_count(n)
        size = self.region(seq_id)[1]
        return self.ring_tokens is not None or self._tokens[seq_id] + n <= size

    def append(self, seq_id: int, n: int = 1) -> None:
        if not self.can_append(seq_id, n):
            raise AllocationError(f"request {seq_id} has filled its reservation")
        size = self._regions[seq_id][1]
        self._tokens[seq_id] = min(self._tokens[seq_id] + n, size)  # a ring overwrites the oldest

    def free(self, seq_id: int) -> int:
        start, size = self.region(seq_id)
        del self._regions[seq_id], self._tokens[seq_id]
        self._allocated -= size
        freed = size
        i = bisect.bisect_left(self._hole_starts, start)
        if i < len(self._hole_starts) and self._hole_starts[i] == start + size:  # merge with next hole
            size += self._hole_sizes.pop(self._hole_starts.pop(i))
        if i > 0:
            prev = self._hole_starts[i - 1]
            if prev + self._hole_sizes[prev] == start:  # merge with previous hole
                self._hole_sizes[prev] += size
                return freed
        self._hole_starts.insert(i, start)
        self._hole_sizes[start] = size
        return freed

    # --- accounting ---
    def num_tokens(self, seq_id: int) -> int:
        self.region(seq_id)
        return self._tokens[seq_id]

    def held_tokens(self, seq_id: int) -> int:
        return self.region(seq_id)[1]

    def allocated_tokens(self) -> int:
        return self._allocated

    def largest_free_run(self) -> int:
        return max(self._hole_sizes.values(), default=0)

    def __contains__(self, seq_id: int) -> bool:
        return seq_id in self._regions

    def check_invariants(self) -> None:
        holes = self.holes()
        assert self._hole_starts == sorted(self._hole_sizes), "hole index out of sync"
        position = 0
        for start, size in sorted(holes + list(self._regions.values())):
            assert size > 0 and start == position, f"gap or overlap at slot {position}"
            position += size
        assert position == self.capacity_tokens, "address space not fully covered"
        for (s1, n1), (s2, _) in zip(holes, holes[1:]):
            assert s1 + n1 < s2, f"holes at {s1} and {s2} should have been merged"
        assert self._allocated == sum(size for _, size in self._regions.values())
        for seq_id, (_, size) in self._regions.items():
            assert 1 <= self._tokens[seq_id] <= size, f"request {seq_id} overflows its region"
