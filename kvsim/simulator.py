"""Step-by-step simulation of an LLM server's KV-cache (Part 2).

One step is one forward pass. Each step the scheduler admits waiting requests
(strict FCFS) and then runs either a prefill step (prompt tokens of newly
admitted requests, at most ``max_batched_tokens``) or a decode step (one token
for every running request). Prefill has priority, so every step is purely one
phase, which lets Part 3 label it compute- or memory-bound.

Request timeline (``P`` prompt tokens, ``O`` output tokens): prefill writes P KV
entries and emits output token 1; each decode step writes the KV of the token it
feeds back in and emits the next one. So a request takes 1 prefill + (O - 1)
decode steps and its cache peaks at P + O - 1 entries.

Every recorded step reports the KV memory split into used, reserved for later,
internal fragmentation, external fragmentation and free (see ``StepStats``).
"""

from __future__ import annotations

import bisect
from collections import deque
from dataclasses import asdict, dataclass, fields
from enum import Enum

from .allocator import ContiguousAllocator, KVAllocator, PagedAllocator, _check_window
from .config import GIB, ConfigError
from .eviction import (
    CACHE_POLICIES, PREEMPT_MODES, VICTIM_POLICIES, SwapSpace, choose_cached_victim, choose_victim,
)
from .sessions import Session, validate_sessions
from .kvpool import DEFAULT_MAX_NUM_SEQS, KVPool
from .workload import Request, percentile, validate_requests

ALLOCATORS = ("paged", "contiguous")
BATCHING = ("continuous", "static")


class SimulationError(RuntimeError):
    """The simulation cannot continue (a bug, or a feature not built yet)."""


@dataclass(frozen=True)
class SimConfig:
    allocator: str = "paged"  # "paged" | "contiguous"
    block_size: int = 16  # paged
    watermark: float = 0.01  # paged: share of blocks kept free when admitting
    reservation: str = "max"  # contiguous: max | request | pow2 | oracle
    fit: str = "first"  # contiguous: first | best
    batching: str = "continuous"  # continuous | static
    max_num_seqs: int = DEFAULT_MAX_NUM_SEQS  # most requests running at once
    max_batched_tokens: int = 2048  # prompt tokens per prefill step (Part 1's prefill_chunk)
    max_model_len: int = 2048  # longest prompt + output; use the model's max_context
    window: int | None = None  # sliding window: keep only the latest W tokens
    sink_tokens: int = 0  # with a window, also keep the first S tokens (attention sinks)
    evict_policy: str = "lifo"  # lifo | fifo | largest
    preempt_mode: str = "recompute"  # recompute | swap
    swap_bytes: int = 4 * GIB  # CPU swap space (vLLM's default swap_space)
    cache_policy: str = "lru"  # sessions: which idle session cache to drop (lru | fifo | opt)
    max_steps: int = 10_000_000  # safety stop
    check_invariants: bool = False  # verify allocator and accounting after every step (tests)

    def __post_init__(self) -> None:
        for name, allowed in (("allocator", ALLOCATORS), ("batching", BATCHING),
                              ("evict_policy", VICTIM_POLICIES), ("preempt_mode", PREEMPT_MODES),
                              ("cache_policy", CACHE_POLICIES)):
            if getattr(self, name) not in allowed:
                raise ConfigError(f"unknown {name} {getattr(self, name)!r}; choose from {', '.join(allowed)}")
        if not isinstance(self.swap_bytes, int) or self.swap_bytes < 0:
            raise ConfigError(f"swap_bytes must be a non-negative integer, got {self.swap_bytes!r}")
        for name, low in (("block_size", 1), ("max_num_seqs", 1), ("max_batched_tokens", 1),
                          ("max_model_len", 2), ("max_steps", 1)):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < low:
                raise ConfigError(f"{name} must be an integer >= {low}, got {value!r}")
        _check_window(self.window, self.sink_tokens)

    @property
    def kv_cap(self) -> int | None:
        """Most KV entries one request keeps (sinks + window), or None without a window."""
        return None if self.window is None else self.sink_tokens + self.window


def make_allocator(config: SimConfig, pool: KVPool) -> KVAllocator:
    if config.allocator == "paged":
        return PagedAllocator(pool, config.block_size, config.watermark, config.window, config.sink_tokens)
    return ContiguousAllocator(pool, config.max_model_len, config.reservation, config.fit,
                               config.window, config.sink_tokens)


class Status(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    SWAPPED = "swapped"
    FINISHED = "finished"
    REJECTED = "rejected"


@dataclass(eq=False)
class Sequence:
    """Runtime state of one request."""

    request: Request
    kv_cap: int | None = None  # sliding window: most KV entries kept (sinks + window)
    status: Status = Status.WAITING
    prefill_target: int = 0  # tokens to prefill in the current admission
    prefill_done: int = 0
    kv_len: int = 0  # KV entries written
    generated: int = 0  # output tokens emitted
    admitted_step: int | None = None  # first admission
    first_token_step: int | None = None
    finish_step: int | None = None
    preemptions: int = 0
    recompute_tokens: int = 0
    swap_bytes: int = 0  # bytes moved over PCIe, both directions
    swapped_tokens: int = 0  # slots it occupies in swap space while SWAPPED
    peak_kv_tokens: int = 0
    peak_held_tokens: int = 0
    resident_since: int | None = None  # sessions: when this session's cache was first built

    @property
    def id(self) -> int:
        return self.request.id

    @property
    def alloc_key(self):
        """Key of its KV memory in the allocator; all turns of a session share one."""
        sid = self.request.session_id
        return self.request.id if sid is None else ("session", sid)

    @property
    def key(self) -> tuple[int, int]:
        """Priority: earlier arrival first, then lower id."""
        return (self.request.arrival_step, self.request.id)

    @property
    def prefill_remaining(self) -> int:
        return self.prefill_target - self.prefill_done

    @property
    def final_tokens(self) -> int:
        """Most KV entries this request will hold at once."""
        return self._capped(self.request.peak_kv_tokens)

    def _capped(self, tokens: int) -> int:
        return tokens if self.kv_cap is None else min(tokens, self.kv_cap)

    def store(self, n: int) -> None:
        """Write ``n`` KV entries; with a window the oldest beyond the cap are dropped."""
        self.kv_len = self._capped(self.kv_len + n)


@dataclass
class StepStats:
    """One row of the per-step trace. The field names are the contract with Part 3."""

    step: int
    phase: str  # "prefill" | "decode"
    batch_size: int  # requests processed in this forward pass
    running: int
    waiting: int
    swapped: int
    cached: int  # idle session caches kept between turns (sessions only)
    prefill_tokens: int
    output_tokens: int
    kv_tokens: int  # token states stored after the step
    kv_used_bytes: int
    kv_read_bytes: int  # KV bytes attention reads in this step
    kv_write_bytes: int
    attention_pairs: int  # query-key pairs computed (for attention FLOPs)
    kv_allocated_bytes: int
    reserved_future_bytes: int  # held, will be filled later
    internal_frag_bytes: int  # held, will never be filled
    external_frag_bytes: int  # free, but not part of the largest free run
    free_bytes: int
    pool_bytes: int
    blocks_used: int | None  # paged only
    blocks_free: int | None
    block_table_bytes: int | None
    admitted: int
    finished: int
    preempted: int
    overflow_events: int
    admission_stall: int  # 1 if the head of the queue was blocked by memory
    frag_stall: int  # 1 if it was blocked although enough memory was free in total
    swap_out_bytes: int
    swap_in_bytes: int
    swap_used_bytes: int
    recompute_tokens: int


STEP_COLUMNS = tuple(f.name for f in fields(StepStats))
REQUEST_COLUMNS = (
    "id", "arrival_step", "prompt_len", "output_len", "status", "admitted_step",
    "first_token_step", "finish_step", "queue_steps", "ttft_steps", "e2e_steps", "preemptions",
    "recompute_tokens", "swap_bytes", "peak_kv_tokens", "peak_held_tokens",
)


class SimResult:
    def __init__(self, config: SimConfig, allocator: KVAllocator, steps: list[StepStats],
                 sequences: list[Sequence], counters: dict | None = None) -> None:
        self.config = config
        self.counters = counters or {}  # run totals that have no per-step column
        self.allocator_name = allocator.name
        self.bytes_per_token = allocator.pool.bytes_per_token
        self.capacity_tokens = allocator.capacity_tokens
        self.steps = steps
        self.sequences = sequences

    def decode_steps(self) -> list[StepStats]:
        return [s for s in self.steps if s.phase == "decode"]

    def step_rows(self) -> list[dict]:
        return [asdict(s) for s in self.steps]

    def request_rows(self) -> list[dict]:
        rows = []
        for seq in self.sequences:
            r = seq.request

            def since_arrival(step, extra=0):
                return None if step is None else step - r.arrival_step + extra

            rows.append({
                "id": r.id, "arrival_step": r.arrival_step, "prompt_len": r.prompt_len,
                "output_len": r.output_len, "status": seq.status.value,
                "admitted_step": seq.admitted_step, "first_token_step": seq.first_token_step,
                "finish_step": seq.finish_step,
                "queue_steps": since_arrival(seq.admitted_step),  # steps waited before admission
                "ttft_steps": since_arrival(seq.first_token_step, 1),  # arrival step counts as 1
                "e2e_steps": since_arrival(seq.finish_step, 1),
                "preemptions": seq.preemptions, "recompute_tokens": seq.recompute_tokens,
                "swap_bytes": seq.swap_bytes, "peak_kv_tokens": seq.peak_kv_tokens,
                "peak_held_tokens": seq.peak_held_tokens,
            })
        return rows

    def summary(self) -> dict:
        steps, n = self.steps, len(self.steps)
        decode = self.decode_steps()
        done = [r for r in self.request_rows() if r["status"] == "finished"]
        pool = self.capacity_tokens * self.bytes_per_token

        def share(column):
            return sum(getattr(s, column) for s in steps) / (n * pool) if n else 0.0

        occupied = sum(s.kv_used_bytes + s.reserved_future_bytes + s.internal_frag_bytes
                       + s.external_frag_bytes for s in steps)
        output_tokens = sum(s.output_tokens for s in steps)

        def pct(column, p):
            return percentile([r[column] for r in done], p) if done else None

        return {
            "allocator": self.allocator_name,
            "requests": len(self.sequences),
            "finished": len(done),
            "rejected": sum(seq.status is Status.REJECTED for seq in self.sequences),
            "steps": n,
            "prefill_steps": n - len(decode),
            "decode_steps": len(decode),
            "output_tokens": output_tokens,
            "tokens_per_step": output_tokens / n if n else 0.0,
            "avg_decode_batch": sum(s.batch_size for s in decode) / len(decode) if decode else 0.0,
            "peak_running": max((s.running for s in steps), default=0),
            "share_token_states": sum(s.kv_used_bytes for s in steps) / occupied if occupied else 0.0,
            "share_used": share("kv_used_bytes"),
            "share_reserved": share("reserved_future_bytes"),
            "share_internal_frag": share("internal_frag_bytes"),
            "share_external_frag": share("external_frag_bytes"),
            "peak_kv_used_bytes": max((s.kv_used_bytes for s in steps), default=0),
            "preemptions": sum(s.preempted for s in steps),
            "overflow_events": sum(s.overflow_events for s in steps),
            "admission_stalls": sum(s.admission_stall for s in steps),
            "frag_stalls": sum(s.frag_stall for s in steps),
            "recompute_tokens": sum(s.recompute_tokens for s in steps),
            "swap_out_bytes": sum(s.swap_out_bytes for s in steps),
            "swap_in_bytes": sum(s.swap_in_bytes for s in steps),
            "swap_fallbacks": self.counters.get("swap_fallbacks", 0),
            "queue_p50": pct("queue_steps", 50), "queue_p95": pct("queue_steps", 95),
            "ttft_p50": pct("ttft_steps", 50), "ttft_p95": pct("ttft_steps", 95),
            "e2e_p50": pct("e2e_steps", 50), "e2e_p95": pct("e2e_steps", 95),
            **self._session_summary(),
        }

    def _session_summary(self) -> dict:
        hits, misses = self.counters.get("cache_hits", 0), self.counters.get("cache_misses", 0)
        return {
            "sessions": self.counters.get("sessions", 0),
            "cache_hits": hits,
            "cache_misses": misses,
            "cache_hit_rate": hits / (hits + misses) if hits + misses else None,
            "cache_evictions": self.counters.get("cache_evictions", 0),
            "saved_prefill_tokens": self.counters.get("saved_prefill_tokens", 0),
        }


class _StepCounters:
    def __init__(self) -> None:
        self.batch_size = self.prefill_tokens = self.output_tokens = 0
        self.kv_read = self.kv_write = self.attention_pairs = 0
        self.admitted = self.finished = self.preempted = self.overflow_events = 0
        self.admission_stall = self.frag_stall = 0
        self.swap_out = self.swap_in = self.recompute_tokens = 0


@dataclass
class CachedSession:
    """An idle session whose KV-cache is kept between turns."""

    session_id: int
    alloc_key: object
    kv_len: int
    resident_since: int  # when this cache was first built (for FIFO)
    last_used: int  # when its last turn finished (for LRU)
    next_arrival: int  # when its next turn arrives (for OPT)


class Simulator:
    def __init__(self, requests: list[Request], pool: KVPool, config: SimConfig,
                 sessions: list[Session] | None = None) -> None:
        self.config = config
        self.sessions = {s.id: s for s in sessions or []}
        if self.sessions:
            if config.allocator != "paged" or config.window is not None:
                raise ConfigError("multi-turn sessions need the paged allocator without a window")
            validate_sessions(list(self.sessions.values()), config.max_model_len)
        self._next_id = max((r.id for r in requests), default=-1) + 1
        requests = list(requests) + [self._turn_request(s, 0, s.start_step) for s in self.sessions.values()]
        validate_requests(requests, config.max_model_len)
        self.alloc = make_allocator(config, pool)
        self.sequences = [Sequence(r, kv_cap=config.kv_cap)
                          for r in sorted(requests, key=lambda r: (r.arrival_step, r.id))]
        self.pending = deque(self.sequences)  # not arrived yet
        self.cache: dict[int, CachedSession] = {}  # idle session caches, by session id
        self.counters = {"swap_fallbacks": 0, "sessions": len(self.sessions), "cache_hits": 0,
                         "cache_misses": 0, "cache_evictions": 0, "saved_prefill_tokens": 0}
        self.waiting: deque[Sequence] = deque()  # arrived, FCFS
        self.running: list[Sequence] = []  # hold GPU memory, kept in priority order
        self.swapped: list[Sequence] = []  # KV in CPU swap space, kept in priority order
        self.swap = SwapSpace(config.swap_bytes // pool.bytes_per_token)
        self.steps: list[StepStats] = []
        self.now = 0

    def _turn_request(self, session: Session, k: int, arrival_step: int) -> Request:
        request = session.request(k, self._next_id, arrival_step)
        self._next_id += 1
        return request

    # --- main loop ---
    def run(self) -> SimResult:
        while self.pending or self.waiting or self.running or self.swapped:
            if len(self.steps) >= self.config.max_steps:
                raise SimulationError(f"stopped after {self.config.max_steps} steps")
            self._arrivals()
            if not (self.waiting or self.running or self.swapped):  # idle: jump to the next arrival
                if self.pending:  # (empty if the last arrival was just rejected)
                    self.now = self.pending[0].request.arrival_step
                continue
            self._step()
            self.now += 1
        return SimResult(self.config, self.alloc, self.steps, self.sequences, self.counters)

    @staticmethod
    def _insert(queue: list | deque, seq: Sequence) -> None:
        """Insert keeping ``queue`` in priority order."""
        keys = [s.key for s in queue]
        queue.insert(bisect.bisect(keys, seq.key), seq)

    def _arrivals(self) -> None:
        while self.pending and self.pending[0].request.arrival_step <= self.now:
            seq = self.pending.popleft()
            if self._fits_alone(seq):
                self.waiting.append(seq)
            else:
                seq.status = Status.REJECTED  # a session ends here: drop its kept cache
                entry = self.cache.pop(seq.request.session_id, None)
                if entry is not None:
                    self.alloc.free(entry.alloc_key)

    def _evict_cached(self, keep: int | None = None) -> bool:
        """Drop one idle session cache (not session ``keep``); False if there is none."""
        entries = [e for sid, e in self.cache.items() if sid != keep]
        if not entries:
            return False
        victim = choose_cached_victim(self.config.cache_policy, entries)
        self.alloc.free(victim.alloc_key)
        del self.cache[victim.session_id]
        self.counters["cache_evictions"] += 1
        return True

    def _fits_alone(self, seq: Sequence) -> bool:
        """Could this request run to completion in an otherwise empty pool?"""
        a = self.alloc
        if isinstance(a, PagedAllocator):  # the watermark is waived when nothing else runs
            return a.max_blocks(seq.request.peak_kv_tokens) <= a.num_blocks
        return a.reservation_for(seq.request) <= a.capacity_tokens

    def _step(self) -> None:
        c = _StepCounters()
        self._swap_in(c)
        self._admit(c)
        prefilling = [s for s in self.running if s.prefill_remaining]
        if prefilling:
            phase = "prefill"
            self._prefill(prefilling, c)
        else:
            phase = "decode"
            if not self.running:
                raise SimulationError(f"step {self.now}: nothing can run")
            self._decode(c)
        for seq in self.running:
            seq.peak_kv_tokens = max(seq.peak_kv_tokens, seq.kv_len)
            seq.peak_held_tokens = max(seq.peak_held_tokens, self.alloc.held_tokens(seq.alloc_key))
        self._finish(c)
        self._record(phase, c)

    # --- scheduling ---
    def _swap_in(self, c: _StepCounters) -> None:
        """Bring swapped requests back, oldest first, while blocks are free."""
        while self.swapped and len(self.running) < self.config.max_num_seqs:
            seq = self.swapped[0]
            # room for its cache plus the next token, so it is not evicted again at once
            while (not self.alloc.can_admit(seq.alloc_key, seq.kv_len + 1, reserve_watermark=False)
                   and self._evict_cached()):
                pass  # idle session caches go first
            if not self.alloc.can_admit(seq.alloc_key, seq.kv_len + 1, reserve_watermark=False):
                break
            self.swapped.pop(0)
            self.alloc.admit(seq.alloc_key, seq.kv_len, reserve_watermark=False)
            self.swap.release(seq.swapped_tokens)
            c.swap_in += seq.swapped_tokens
            seq.swap_bytes += seq.swapped_tokens * self.alloc.pool.bytes_per_token
            seq.swapped_tokens = 0
            seq.status = Status.RUNNING
            self._insert(self.running, seq)

    def _admit(self, c: _StepCounters) -> None:
        if self.swapped:  # preempted requests go before new ones
            return
        if self.config.batching == "static" and self.running:
            return
        budget = self.config.max_batched_tokens - sum(s.prefill_remaining for s in self.running)
        while self.waiting and budget > 0 and len(self.running) < self.config.max_num_seqs:
            seq = self.waiting[0]
            sid = seq.request.session_id
            hit = self.cache.get(sid) if sid is not None and not seq.generated else None
            if hit is not None:
                # cache hit: the session's KV from earlier turns is still here, so only the
                # new tokens are prefilled (the last answer token and the new message)
                n = seq.request.prompt_len - hit.kv_len
                while not self.alloc.can_append(seq.alloc_key, n) and self._evict_cached(keep=sid):
                    pass
                fits = self.alloc.can_append(seq.alloc_key, n)
            else:
                # a new request prefills its prompt; a recomputed one prefills everything it
                # knew: prompt + generated tokens (P + g), which rebuilds its cache
                n = seq.request.prompt_len + seq.generated
                if seq.generated and seq.kv_cap is not None:  # with a window only the kept
                    n = min(n, seq.kv_cap + 1)  # tokens are rebuilt (a simplification)
                watermark = bool(self.running)  # waived when nothing else is running
                while (not self.alloc.can_admit(seq.alloc_key, n, seq.request, reserve_watermark=watermark)
                       and self._evict_cached()):
                    pass  # idle session caches go before anyone waits
                fits = self.alloc.can_admit(seq.alloc_key, n, seq.request, reserve_watermark=watermark)
            if not fits:
                c.admission_stall = 1
                if isinstance(self.alloc, ContiguousAllocator):
                    c.frag_stall = int(self.alloc.free_tokens() >= self.alloc.reservation_for(seq.request))
                break
            self.waiting.popleft()
            if hit is not None:
                self.alloc.append(seq.alloc_key, n)
                del self.cache[sid]
                seq.kv_len, seq.resident_since = hit.kv_len, hit.resident_since
                self.counters["cache_hits"] += 1
                self.counters["saved_prefill_tokens"] += hit.kv_len
            else:
                self.alloc.admit(seq.alloc_key, n, seq.request, reserve_watermark=watermark)
                if sid is not None and not seq.generated:
                    seq.resident_since = self.now
                    if seq.request.turn > 0:
                        self.counters["cache_misses"] += 1
            seq.status = Status.RUNNING
            seq.prefill_target, seq.prefill_done = n, 0
            if seq.admitted_step is None:
                seq.admitted_step = self.now
            self._insert(self.running, seq)
            budget -= min(n, budget)
            c.admitted += 1

    def _prefill(self, prefilling: list[Sequence], c: _StepCounters) -> None:
        budget = self.config.max_batched_tokens
        for seq in prefilling:  # already in priority order
            if budget == 0:
                break
            chunk = min(seq.prefill_remaining, budget)
            c.kv_read += seq.kv_len  # earlier chunks are read back
            c.kv_write += chunk
            c.attention_pairs += chunk * seq.kv_len + chunk * (chunk + 1) // 2
            seq.store(chunk)
            seq.prefill_done += chunk
            budget -= chunk
            c.batch_size += 1
            c.prefill_tokens += chunk
            if not seq.prefill_remaining:  # prefill emits a token
                seq.generated += 1
                c.output_tokens += 1
                if seq.first_token_step is None:
                    seq.first_token_step = self.now

    def _decode(self, c: _StepCounters) -> None:
        decoded = []
        for seq in list(self.running):
            if seq.status is not Status.RUNNING:
                continue  # preempted earlier in this step
            overflowed = False
            while not self.alloc.can_append(seq.alloc_key):
                if not overflowed:
                    c.overflow_events += 1
                    overflowed = True
                if self._evict_cached():  # idle session caches go before running requests
                    continue
                victim = self._choose_victim(seq)
                self._preempt(victim, c)
                if victim is seq:
                    break
            else:
                self.alloc.append(seq.alloc_key)  # the slot is claimed now
                decoded.append(seq)
        for seq in decoded:
            if seq.status is Status.RUNNING:
                seq.store(1)
                seq.generated += 1
                c.kv_write += 1
                c.kv_read += seq.kv_len  # attention reads the whole cache, new entry included
                c.attention_pairs += seq.kv_len
                c.output_tokens += 1
                c.batch_size += 1

    def _choose_victim(self, needy: Sequence) -> Sequence:
        return choose_victim(self.config.evict_policy, self.running,
                             lambda s: self.alloc.held_tokens(s.alloc_key))

    def _preempt(self, victim: Sequence, c: _StepCounters) -> None:
        """Evict all of ``victim``'s blocks: swap them out, or drop them to recompute later."""
        held = self.alloc.held_tokens(victim.alloc_key)
        self.alloc.free(victim.alloc_key)  # also drops a slot it may have claimed this step
        self.running.remove(victim)
        victim.preemptions += 1
        c.preempted += 1
        if self.config.preempt_mode == "swap":
            if self.swap.can_hold(held):
                self.swap.store(held)
                victim.swapped_tokens = held
                victim.swap_bytes += held * self.alloc.pool.bytes_per_token
                victim.status = Status.SWAPPED
                c.swap_out += held
                self._insert(self.swapped, victim)
                return
            self.counters["swap_fallbacks"] += 1  # swap space is full: recompute instead
        victim.recompute_tokens += victim.kv_len
        c.recompute_tokens += victim.kv_len
        victim.kv_len = 0
        victim.prefill_target = victim.prefill_done = 0
        victim.status = Status.WAITING
        self._insert(self.waiting, victim)  # front of the queue: it arrived before the rest

    def _finish(self, c: _StepCounters) -> None:
        still_running = []
        for seq in self.running:
            if seq.generated >= seq.request.output_len:
                seq.status = Status.FINISHED
                seq.finish_step = self.now
                c.finished += 1
                if not self._next_turn(seq):
                    self.alloc.free(seq.alloc_key)
            else:
                still_running.append(seq)
        self.running = still_running

    def _next_turn(self, seq: Sequence) -> bool:
        """If ``seq`` was a session turn with more to come, schedule the next turn and
        keep this session's KV-cache while the user thinks. Returns True if kept."""
        sid, k = seq.request.session_id, seq.request.turn
        if sid is None or k + 1 >= len(self.sessions[sid].turns):
            return False
        session = self.sessions[sid]
        arrival = self.now + session.turns[k + 1].think_steps
        nxt = Sequence(self._turn_request(session, k + 1, arrival), kv_cap=self.config.kv_cap)
        self.sequences.append(nxt)
        self._insert(self.pending, nxt)
        self.cache[sid] = CachedSession(sid, seq.alloc_key, seq.kv_len, seq.resident_since or self.now,
                                        last_used=self.now, next_arrival=arrival)
        return True

    # --- accounting ---
    def _record(self, phase: str, c: _StepCounters) -> None:
        a, bpt = self.alloc, self.alloc.pool.bytes_per_token
        used = reserved = internal = 0
        for seq in self.running:
            held = a.held_tokens(seq.alloc_key)
            fill = min(held, seq.final_tokens)
            used += seq.kv_len
            reserved += fill - seq.kv_len
            internal += held - fill
        for entry in self.cache.values():  # idle session caches: kept tokens, slack never filled
            used += entry.kv_len
            internal += a.held_tokens(entry.alloc_key) - entry.kv_len
        allocated = a.allocated_tokens()
        free = a.capacity_tokens - allocated
        external = free - a.largest_free_run()
        paged = isinstance(a, PagedAllocator)
        if self.config.check_invariants:
            a.check_invariants()
            assert used + reserved + internal == allocated, "memory breakdown does not add up"
            assert all(s.kv_len <= a.num_tokens(s.alloc_key) for s in self.running)
        self.steps.append(StepStats(
            step=self.now, phase=phase, batch_size=c.batch_size, running=len(self.running),
            waiting=len(self.waiting), swapped=len(self.swapped), cached=len(self.cache),
            prefill_tokens=c.prefill_tokens,
            output_tokens=c.output_tokens, kv_tokens=used, kv_used_bytes=used * bpt,
            kv_read_bytes=c.kv_read * bpt, kv_write_bytes=c.kv_write * bpt,
            attention_pairs=c.attention_pairs, kv_allocated_bytes=allocated * bpt,
            reserved_future_bytes=reserved * bpt, internal_frag_bytes=internal * bpt,
            external_frag_bytes=external * bpt, free_bytes=free * bpt,
            pool_bytes=a.capacity_tokens * bpt,
            blocks_used=a.blocks_used if paged else None,
            blocks_free=a.blocks_free if paged else None,
            block_table_bytes=a.block_table_bytes if paged else None,
            admitted=c.admitted, finished=c.finished, preempted=c.preempted,
            overflow_events=c.overflow_events, admission_stall=c.admission_stall,
            frag_stall=c.frag_stall, swap_out_bytes=c.swap_out * bpt, swap_in_bytes=c.swap_in * bpt,
            swap_used_bytes=self.swap.used_tokens * bpt, recompute_tokens=c.recompute_tokens,
        ))


def simulate(requests: list[Request], pool: KVPool, config: SimConfig | None = None,
             sessions: list[Session] | None = None) -> SimResult:
    """Run the whole workload and return the per-step and per-request traces.

    ``sessions`` adds multi-turn chats (``sessions.py``): each turn is created when
    the previous one finishes, and the session's cache is kept in between.
    """
    return Simulator(requests, pool, config or SimConfig(), sessions).run()
