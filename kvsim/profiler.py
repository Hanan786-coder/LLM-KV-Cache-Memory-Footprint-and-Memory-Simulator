"""Turn a Part 2 step trace into wall-clock time, throughput and request latency.

Every recorded step is priced with ``PerfModel.step`` and the step times are
added up into a timeline. Request events (arrival, first token, finish) are
given in steps by the simulator and are mapped onto that timeline:

* a request's arrival time is the start of the first step at or after its
  ``arrival_step`` (an idle server costs no time, so arrival gaps are not
  counted),
* TTFT = end of its ``first_token_step`` - arrival time,
* E2E  = end of its ``finish_step``      - arrival time,
* TPOT = (E2E - TTFT) / (output tokens - 1), the time per output token after the first.

The input can be a live ``SimResult`` or the CSVs written by
``kvsim simulate --steps-csv/--requests-csv``.
"""

from __future__ import annotations

import bisect
import csv
from dataclasses import dataclass
from pathlib import Path

from .config import ConfigError
from .perf import PerfModel, StepCost
from .workload import percentile

#: step-trace columns the profiler needs (a subset of simulator.STEP_COLUMNS)
NEEDED_STEP_COLUMNS = ("step", "phase", "batch_size", "prefill_tokens", "output_tokens",
                       "kv_used_bytes", "kv_read_bytes", "kv_write_bytes", "attention_pairs",
                       "swap_out_bytes", "swap_in_bytes")
NEEDED_REQUEST_COLUMNS = ("id", "arrival_step", "output_len", "status", "first_token_step",
                          "finish_step")

TIMED_STEP_COLUMNS = ("step", "phase", "batch_size", "tokens", "output_tokens", "kv_used_bytes",
                      "weight_bytes", "kv_read_bytes", "kv_write_bytes", "hbm_bytes", "swap_bytes",
                      "flops", "intensity", "bound", "t_memory_ms", "t_compute_ms", "t_swap_ms",
                      "latency_ms", "start_ms", "end_ms", "tokens_per_s")
TIMED_REQUEST_COLUMNS = ("id", "status", "prompt_len", "output_len", "arrival_ms",
                         "first_token_ms", "finish_ms", "ttft_ms", "tpot_ms", "e2e_ms",
                         "decode_tokens_per_s")


def _value(step, name: str) -> int:
    raw = step.get(name) if isinstance(step, dict) else getattr(step, name)
    if raw in (None, ""):
        return 0
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise ConfigError(f"step column {name!r} must be an integer, got {raw!r}") from None


def _phase(step) -> str:
    return step["phase"] if isinstance(step, dict) else step.phase


def cost_of_step(perf: PerfModel, step) -> StepCost:
    """Price one row of the Part 2 step trace (a ``StepStats`` or a CSV dict)."""
    phase = _phase(step)
    tokens = _value(step, "prefill_tokens") if phase == "prefill" else _value(step, "batch_size")
    return perf.step(
        phase, tokens=tokens, logits=_value(step, "output_tokens"),
        kv_read_bytes=_value(step, "kv_read_bytes"), kv_write_bytes=_value(step, "kv_write_bytes"),
        attention_pairs=_value(step, "attention_pairs"),
        swap_bytes=_value(step, "swap_out_bytes") + _value(step, "swap_in_bytes"),
    )


@dataclass(frozen=True)
class TimedStep:
    step: int
    batch_size: int
    kv_used_bytes: int
    cost: StepCost
    start_s: float
    end_s: float

    def row(self) -> dict:
        c = self.cost
        return {
            "step": self.step, "phase": c.phase, "batch_size": self.batch_size, "tokens": c.tokens,
            "output_tokens": c.logits, "kv_used_bytes": self.kv_used_bytes,
            "weight_bytes": c.weight_bytes, "kv_read_bytes": c.kv_read_bytes,
            "kv_write_bytes": c.kv_write_bytes, "hbm_bytes": c.hbm_bytes, "swap_bytes": c.swap_bytes,
            "flops": c.flops, "intensity": c.intensity, "bound": c.bound,
            "t_memory_ms": c.t_memory_s * 1e3, "t_compute_ms": c.t_compute_s * 1e3,
            "t_swap_ms": c.t_swap_s * 1e3, "latency_ms": c.latency_s * 1e3,
            "start_ms": self.start_s * 1e3, "end_ms": self.end_s * 1e3,
            "tokens_per_s": c.tokens_per_s,
        }


class Profile:
    """A priced step trace plus the requests it served."""

    def __init__(self, perf: PerfModel, steps: list[TimedStep], requests: list[dict] | None = None):
        self.perf = perf
        self.steps = steps
        self._indexes = [s.step for s in steps]
        self.requests = [self._time_request(r) for r in (requests or [])]

    # --- time lookups ---
    def _start_at(self, step_index: int) -> float | None:
        """Start time of the first recorded step at or after ``step_index``."""
        i = bisect.bisect_left(self._indexes, step_index)
        return self.steps[i].start_s if i < len(self.steps) else None

    def _end_of(self, step_index) -> float | None:
        if step_index in (None, ""):
            return None
        i = bisect.bisect_left(self._indexes, int(step_index))
        if i == len(self.steps) or self.steps[i].step != int(step_index):
            raise ConfigError(f"request refers to step {step_index}, which is not in the step trace")
        return self.steps[i].end_s

    def _time_request(self, r: dict) -> dict:
        arrival = self._start_at(int(r["arrival_step"]))
        first, finish = self._end_of(r.get("first_token_step")), self._end_of(r.get("finish_step"))
        output = int(r["output_len"])

        def since(t):
            return None if t is None or arrival is None else (t - arrival) * 1e3

        ttft, e2e = since(first), since(finish)
        tpot = (e2e - ttft) / (output - 1) if e2e is not None and ttft is not None and output > 1 else None
        decode_s = (finish - first) if finish is not None and first is not None else None
        return {
            "id": int(r["id"]), "status": r["status"], "prompt_len": int(r.get("prompt_len") or 0),
            "output_len": output, "arrival_ms": None if arrival is None else arrival * 1e3,
            "first_token_ms": None if first is None else first * 1e3,
            "finish_ms": None if finish is None else finish * 1e3,
            "ttft_ms": ttft, "tpot_ms": tpot, "e2e_ms": e2e,
            "decode_tokens_per_s": (output - 1) / decode_s if decode_s else None,
        }

    # --- outputs ---
    def step_rows(self) -> list[dict]:
        return [s.row() for s in self.steps]

    def request_rows(self) -> list[dict]:
        return list(self.requests)

    @property
    def total_s(self) -> float:
        return self.steps[-1].end_s if self.steps else 0.0

    def phase_rows(self) -> list[dict]:
        """Time, traffic and work split by phase."""
        rows = []
        total = self.total_s
        for phase in ("prefill", "decode"):
            costs = [s.cost for s in self.steps if s.cost.phase == phase]
            t = sum(c.latency_s for c in costs)
            hbm, flops = sum(c.hbm_bytes for c in costs), sum(c.flops for c in costs)
            rows.append({
                "phase": phase, "steps": len(costs), "time_s": t,
                "share_time": t / total if total else 0.0,
                "share_memory_bound": (sum(c.latency_s for c in costs if c.bound == "memory") / t
                                       if t else 0.0),
                "avg_step_ms": t / len(costs) * 1e3 if costs else 0.0,
                "output_tokens": sum(c.logits for c in costs),
                "hbm_bytes": hbm, "flops": flops,
                "avg_intensity": flops / hbm if hbm else 0.0,
            })
        return rows

    def summary(self) -> dict:
        costs = [s.cost for s in self.steps]
        total = self.total_s
        decode = [c for c in costs if c.phase == "decode"]
        decode_ms = [c.latency_s * 1e3 for c in decode]
        decode_s = sum(c.latency_s for c in decode)
        hbm = sum(c.hbm_bytes for c in costs)
        flops = sum(c.flops for c in costs)
        weights = sum(c.weight_bytes for c in costs)
        output = sum(c.logits for c in costs)
        busy = sum(max(c.t_memory_s, c.t_compute_s) for c in costs)
        done = [r for r in self.requests if r["status"] == "finished"]

        def pct(key, p):
            values = [r[key] for r in done if r[key] is not None]
            return percentile(values, p) if values else None

        hw = self.perf.hw
        return {
            "steps": len(costs),
            "total_time_s": total,
            "output_tokens": output,
            "throughput_tok_s": output / total if total else 0.0,
            "decode_tok_s": sum(c.logits for c in decode) / decode_s if decode_s else 0.0,
            "decode_step_p50_ms": percentile(decode_ms, 50) if decode_ms else None,
            "decode_step_p95_ms": percentile(decode_ms, 95) if decode_ms else None,
            "share_time_prefill": sum(c.latency_s for c in costs if c.phase == "prefill") / total
            if total else 0.0,
            "share_time_memory_bound": sum(c.latency_s for c in costs if c.bound == "memory") / total
            if total else 0.0,
            "share_time_swap": sum(c.t_swap_s for c in costs) / total if total else 0.0,
            "hbm_bytes": hbm,
            "share_bytes_weights": weights / hbm if hbm else 0.0,
            "share_bytes_kv": (hbm - weights) / hbm if hbm else 0.0,
            "avg_intensity": flops / hbm if hbm else 0.0,
            "achieved_gbs": hbm / total / 1e9 if total else 0.0,
            "share_peak_bandwidth": hbm / busy / hw.gpu.bandwidth_bytes_per_s if busy else 0.0,
            "achieved_tflops": flops / total / 1e12 if total else 0.0,
            "share_peak_compute": flops / busy / (hw.gpu.peak_fp16_tflops * 1e12) if busy else 0.0,
            "finished": len(done),
            "ttft_p50_ms": pct("ttft_ms", 50), "ttft_p95_ms": pct("ttft_ms", 95),
            "tpot_p50_ms": pct("tpot_ms", 50), "tpot_p95_ms": pct("tpot_ms", 95),
            "e2e_p50_ms": pct("e2e_ms", 50), "e2e_p95_ms": pct("e2e_ms", 95),
        }


def profile_steps(perf: PerfModel, steps, requests: list[dict] | None = None) -> Profile:
    """Price every step in order and lay them end to end on one timeline."""
    timed, now, last = [], 0.0, None
    for step in steps:
        index = _value(step, "step")
        if last is not None and index <= last:
            raise ConfigError(f"step trace is not in increasing step order at step {index}")
        last = index
        cost = cost_of_step(perf, step)
        timed.append(TimedStep(index, _value(step, "batch_size"), _value(step, "kv_used_bytes"),
                               cost, now, now + cost.latency_s))
        now += cost.latency_s
    return Profile(perf, timed, requests)


def profile_result(perf: PerfModel, result) -> Profile:
    """Profile a live ``simulator.SimResult``."""
    return profile_steps(perf, result.steps, result.request_rows())


def _read_csv(path: str | Path, needed: tuple[str, ...], kind: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in needed if c not in (reader.fieldnames or [])]
        if missing:
            raise ConfigError(f"{path}: not a {kind} CSV, missing columns {missing}")
        return list(reader)


def load_steps_csv(path: str | Path) -> list[dict]:
    """Rows of a ``kvsim simulate --steps-csv`` file."""
    return _read_csv(path, NEEDED_STEP_COLUMNS, "step trace")


def load_requests_csv(path: str | Path) -> list[dict]:
    """Rows of a ``kvsim simulate --requests-csv`` file."""
    return _read_csv(path, NEEDED_REQUEST_COLUMNS, "request trace")
