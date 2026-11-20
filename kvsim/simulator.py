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

from .allocator import ContiguousAllocator, KVAllocator, PagedAllocator
from .config import ConfigError
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
    max_steps: int = 10_000_000  # safety stop
    check_invariants: bool = False  # verify allocator and accounting after every step (tests)

    def __post_init__(self) -> None:
        if self.allocator not in ALLOCATORS:
            raise ConfigError(f"unknown allocator {self.allocator!r}; choose from {', '.join(ALLOCATORS)}")
        if self.batching not in BATCHING:
            raise ConfigError(f"unknown batching {self.batching!r}; choose from {', '.join(BATCHING)}")
        for name, low in (("block_size", 1), ("max_num_seqs", 1), ("max_batched_tokens", 1),
                          ("max_model_len", 2), ("max_steps", 1)):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < low:
                raise ConfigError(f"{name} must be an integer >= {low}, got {value!r}")


def make_allocator(config: SimConfig, pool: KVPool) -> KVAllocator:
    if config.allocator == "paged":
        return PagedAllocator(pool, config.block_size, config.watermark)
    return ContiguousAllocator(pool, config.max_model_len, config.reservation, config.fit)


class Status(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    REJECTED = "rejected"


@dataclass(eq=False)
class Sequence:
    """Runtime state of one request."""

    request: Request
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
    swap_bytes: int = 0
    peak_kv_tokens: int = 0
    peak_held_tokens: int = 0

    @property
    def id(self) -> int:
        return self.request.id

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
        return self.request.peak_kv_tokens


@dataclass
class StepStats:
    """One row of the per-step trace. The field names are the contract with Part 3."""

    step: int
    phase: str  # "prefill" | "decode"
    batch_size: int  # requests processed in this forward pass
    running: int
    waiting: int
    swapped: int
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
                 sequences: list[Sequence]) -> None:
        self.config = config
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
            "queue_p50": pct("queue_steps", 50), "queue_p95": pct("queue_steps", 95),
            "ttft_p50": pct("ttft_steps", 50), "ttft_p95": pct("ttft_steps", 95),
            "e2e_p50": pct("e2e_steps", 50), "e2e_p95": pct("e2e_steps", 95),
        }


class _StepCounters:
    def __init__(self) -> None:
        self.batch_size = self.prefill_tokens = self.output_tokens = 0
        self.kv_read = self.kv_write = self.attention_pairs = 0
        self.admitted = self.finished = self.preempted = self.overflow_events = 0
        self.admission_stall = self.frag_stall = 0
        self.swap_out = self.swap_in = self.recompute_tokens = 0


class Simulator:
    def __init__(self, requests: list[Request], pool: KVPool, config: SimConfig) -> None:
        validate_requests(requests, config.max_model_len)
        self.config = config
        self.alloc = make_allocator(config, pool)
        self.sequences = [Sequence(r) for r in sorted(requests, key=lambda r: (r.arrival_step, r.id))]
        self.pending = deque(self.sequences)  # not arrived yet
        self.waiting: deque[Sequence] = deque()  # arrived, FCFS
        self.running: list[Sequence] = []  # hold GPU memory, kept in priority order
        self.steps: list[StepStats] = []
        self.now = 0

    # --- main loop ---
    def run(self) -> SimResult:
        while self.pending or self.waiting or self.running:
            if len(self.steps) >= self.config.max_steps:
                raise SimulationError(f"stopped after {self.config.max_steps} steps")
            self._arrivals()
            if not (self.waiting or self.running):  # idle: jump to the next arrival
                self.now = self.pending[0].request.arrival_step
                continue
            self._step()
            self.now += 1
        return SimResult(self.config, self.alloc, self.steps, self.sequences)

    def _arrivals(self) -> None:
        while self.pending and self.pending[0].request.arrival_step <= self.now:
            seq = self.pending.popleft()
            if self._fits_alone(seq):
                self.waiting.append(seq)
            else:
                seq.status = Status.REJECTED

    def _fits_alone(self, seq: Sequence) -> bool:
        """Could this request run to completion in an otherwise empty pool?"""
        a = self.alloc
        if isinstance(a, PagedAllocator):
            usable = a.num_blocks - a.watermark_blocks  # admission keeps the watermark free
            return (a.blocks_for(seq.request.prompt_len) <= usable
                    and a.blocks_for(seq.final_tokens) <= a.num_blocks)
        return a.reservation_for(seq.request) <= a.capacity_tokens

    def _step(self) -> None:
        c = _StepCounters()
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
            seq.peak_held_tokens = max(seq.peak_held_tokens, self.alloc.held_tokens(seq.id))
        self._finish(c)
        self._record(phase, c)

    # --- scheduling ---
    def _admit(self, c: _StepCounters) -> None:
        if self.config.batching == "static" and self.running:
            return
        budget = self.config.max_batched_tokens - sum(s.prefill_remaining for s in self.running)
        while self.waiting and budget > 0 and len(self.running) < self.config.max_num_seqs:
            seq = self.waiting[0]
            n = seq.request.prompt_len
            if not self.alloc.can_admit(seq.id, n, seq.request):
                c.admission_stall = 1
                if isinstance(self.alloc, ContiguousAllocator):
                    c.frag_stall = int(self.alloc.free_tokens() >= self.alloc.reservation_for(seq.request))
                break
            self.waiting.popleft()
            self.alloc.admit(seq.id, n, seq.request)
            seq.status = Status.RUNNING
            seq.prefill_target, seq.prefill_done = n, 0
            if seq.admitted_step is None:
                seq.admitted_step = self.now
            keys = [s.key for s in self.running]
            self.running.insert(bisect.bisect(keys, seq.key), seq)
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
            seq.kv_len += chunk
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
            while not self.alloc.can_append(seq.id):
                if not overflowed:
                    c.overflow_events += 1
                    overflowed = True
                victim = self._choose_victim(seq)
                self._preempt(victim, c)
                if victim is seq:
                    break
            else:
                self.alloc.append(seq.id)  # the slot is claimed now
                decoded.append(seq)
        for seq in decoded:
            if seq.status is Status.RUNNING:
                seq.kv_len += 1
                seq.generated += 1
                c.kv_write += 1
                c.kv_read += seq.kv_len  # attention reads the whole cache, new entry included
                c.attention_pairs += seq.kv_len
                c.output_tokens += 1
                c.batch_size += 1

    def _choose_victim(self, needy: Sequence) -> Sequence:
        raise SimulationError(
            f"step {self.now}: request {needy.id} needs a new block but the pool is full; "
            "preemption is not implemented yet (milestone M5)")

    def _preempt(self, victim: Sequence, c: _StepCounters) -> None:
        raise SimulationError("preemption is not implemented yet (milestone M5)")

    def _finish(self, c: _StepCounters) -> None:
        still_running = []
        for seq in self.running:
            if seq.generated >= seq.request.output_len:
                self.alloc.free(seq.id)
                seq.status = Status.FINISHED
                seq.finish_step = self.now
                c.finished += 1
            else:
                still_running.append(seq)
        self.running = still_running

    # --- accounting ---
    def _record(self, phase: str, c: _StepCounters) -> None:
        a, bpt = self.alloc, self.alloc.pool.bytes_per_token
        used = reserved = internal = 0
        for seq in self.running:
            held = a.held_tokens(seq.id)
            fill = min(held, seq.final_tokens)
            used += seq.kv_len
            reserved += fill - seq.kv_len
            internal += held - fill
        allocated = a.allocated_tokens()
        free = a.capacity_tokens - allocated
        external = free - a.largest_free_run()
        paged = isinstance(a, PagedAllocator)
        if self.config.check_invariants:
            a.check_invariants()
            assert used + reserved + internal == allocated, "memory breakdown does not add up"
            assert all(s.kv_len <= a.num_tokens(s.id) for s in self.running)
        self.steps.append(StepStats(
            step=self.now, phase=phase, batch_size=c.batch_size, running=len(self.running),
            waiting=len(self.waiting), swapped=0, prefill_tokens=c.prefill_tokens,
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
            swap_used_bytes=0, recompute_tokens=c.recompute_tokens,
        ))


def simulate(requests: list[Request], pool: KVPool, config: SimConfig | None = None) -> SimResult:
    """Run the whole workload and return the per-step and per-request traces."""
    return Simulator(requests, pool, config or SimConfig()).run()
