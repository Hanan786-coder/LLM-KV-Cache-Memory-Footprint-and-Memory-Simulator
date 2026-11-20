"""Workloads for the Part 2 simulator: requests, length distributions and trace files.

A workload is a list of ``Request``s. Each one says when it reaches the server
(``arrival_step``), how long its prompt is, and how many tokens it will really
generate (``output_len``). Only the simulator knows ``output_len``; the scheduler
and allocators see the prompt and ``max_new_tokens``, as a real server would.

Workloads come from the random generator (``generate_workload``) or from a trace
CSV file (``load_trace`` / ``save_trace``)::

    id,arrival_step,prompt_len,output_len,max_new_tokens
    0,0,512,128,
    1,0,40,900,1024

Lengths are drawn from ``LengthSpec`` distributions written as short strings:
``N`` or ``fixed:N``, ``uniform:LO:HI`` and ``lognormal:MEAN:SIGMA[:LO:HI]``.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

from .config import ConfigError

TRACE_COLUMNS = ("id", "arrival_step", "prompt_len", "output_len", "max_new_tokens")
REQUIRED_TRACE_COLUMNS = TRACE_COLUMNS[:4]
LENGTH_KINDS = ("fixed", "uniform", "lognormal")
SPEC_HELP = "N, fixed:N, uniform:LO:HI or lognormal:MEAN:SIGMA[:LO:HI]"


def _check_int(label: str, value, low: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < low:
        raise ConfigError(f"{label} must be an integer >= {low}, got {value!r}")


# --- requests -----------------------------------------------------------------

@dataclass(frozen=True)
class Request:
    """One request. Lengths are in tokens; time is in scheduler steps."""

    id: int
    arrival_step: int
    prompt_len: int  # P
    output_len: int  # O: tokens it will really generate (known to the simulator only)
    max_new_tokens: int | None = None  # the client's limit; None = up to the model's max length

    def __post_init__(self) -> None:
        _check_int("request id", self.id, 0)
        for name, low in (("arrival_step", 0), ("prompt_len", 1), ("output_len", 1)):
            _check_int(f"request {self.id}: {name}", getattr(self, name), low)
        if self.max_new_tokens is not None:
            _check_int(f"request {self.id}: max_new_tokens", self.max_new_tokens, 1)
            if self.output_len > self.max_new_tokens:
                raise ConfigError(
                    f"request {self.id}: output_len {self.output_len} is more than "
                    f"max_new_tokens {self.max_new_tokens}")

    @property
    def total_len(self) -> int:
        """Prompt plus output tokens; this must fit in the model's maximum length."""
        return self.prompt_len + self.output_len

    @property
    def peak_kv_tokens(self) -> int:
        """Most KV entries the request ever holds: the last token's KV is never written."""
        return self.total_len - 1

    def new_token_limit(self, max_model_len: int) -> int:
        """Most tokens a server would let this request generate."""
        room = max_model_len - self.prompt_len
        return room if self.max_new_tokens is None else min(room, self.max_new_tokens)


def validate_requests(requests, max_model_len: int) -> None:
    """Raise ConfigError if ids repeat or a request is longer than the model allows."""
    _check_int("max_model_len", max_model_len, 2)
    seen: set[int] = set()
    for r in requests:
        if r.id in seen:
            raise ConfigError(f"request {r.id}: duplicate id")
        seen.add(r.id)
        if r.total_len > max_model_len:
            raise ConfigError(
                f"request {r.id}: prompt_len + output_len = {r.total_len} is more than "
                f"max_model_len {max_model_len}")


# --- length distributions -----------------------------------------------------

@dataclass(frozen=True)
class LengthSpec:
    """A distribution of token lengths. Build one from a string with ``parse_length_spec``."""

    kind: str  # "fixed" | "uniform" | "lognormal"
    low: int  # smallest length (fixed: the length itself)
    high: int | None  # largest length; None = no upper clip (lognormal only)
    mean: float | None = None  # lognormal: mean before clipping
    sigma: float | None = None  # lognormal: standard deviation of the underlying normal

    def __post_init__(self) -> None:
        if self.kind not in LENGTH_KINDS:
            raise ConfigError(f"length kind must be one of {', '.join(LENGTH_KINDS)}, got {self.kind!r}")
        _check_int(f"{self.kind} lower bound", self.low, 1)
        if self.high is not None:
            _check_int(f"{self.kind} upper bound", self.high, self.low)
        elif self.kind != "lognormal":
            raise ConfigError(f"{self.kind} lengths need an upper bound")
        if self.kind == "fixed" and self.high != self.low:
            raise ConfigError("a fixed length has a single value")
        if self.kind == "lognormal":
            for name in ("mean", "sigma"):
                value = getattr(self, name)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ConfigError(f"lognormal {name} must be a finite number, got {value!r}")
            if self.mean <= 0 or self.sigma < 0:
                raise ConfigError("lognormal needs MEAN > 0 and SIGMA >= 0")

    def sample(self, rng: random.Random) -> int:
        if self.kind == "fixed":
            return self.low
        if self.kind == "uniform":
            return rng.randint(self.low, self.high)
        mu = math.log(self.mean) - self.sigma**2 / 2  # so that the mean is self.mean
        value = max(round(rng.lognormvariate(mu, self.sigma)), self.low)
        return value if self.high is None else min(value, self.high)

    @property
    def nominal_mean(self) -> float:
        """Mean before clipping: the log-normal's MEAN, or the middle of a range."""
        return self.mean if self.kind == "lognormal" else (self.low + self.high) / 2

    def __str__(self) -> str:
        if self.kind == "fixed":
            return f"fixed:{self.low}"
        if self.kind == "uniform":
            return f"uniform:{self.low}:{self.high}"
        clip = "" if self.high is None else f":{self.low}:{self.high}"
        return f"lognormal:{self.mean:g}:{self.sigma:g}{clip}"


_SPEC_ARITY = {"fixed": (1,), "uniform": (2,), "lognormal": (2, 4)}


def parse_length_spec(spec: str | int | LengthSpec) -> LengthSpec:
    """Parse ``N``, ``fixed:N``, ``uniform:LO:HI`` or ``lognormal:MEAN:SIGMA[:LO:HI]``."""
    if isinstance(spec, LengthSpec):
        return spec
    kind, *args = (part.strip() for part in str(spec).strip().lower().split(":"))
    if not args:  # a bare number
        kind, args = "fixed", [kind]
    if kind not in _SPEC_ARITY or len(args) not in _SPEC_ARITY[kind]:
        raise ConfigError(f"bad length spec {spec!r}; use {SPEC_HELP}")
    try:
        if kind == "lognormal":
            low, high = (int(args[2]), int(args[3])) if len(args) == 4 else (1, None)
            return LengthSpec(kind, low, high, float(args[0]), float(args[1]))
        return LengthSpec(kind, int(args[0]), int(args[-1]))  # fixed: low == high
    except ConfigError as exc:  # before ValueError: ConfigError is a ValueError
        raise ConfigError(f"bad length spec {spec!r}: {exc}") from None
    except ValueError:
        raise ConfigError(f"bad length spec {spec!r}: lengths must be whole numbers; "
                          f"use {SPEC_HELP}") from None


@dataclass(frozen=True)
class Preset:
    description: str
    prompt: LengthSpec
    output: LengthSpec


# Our own choices. "chat" has 8.4x longer prompts and 5.8x longer outputs than "short",
# the ratio the vLLM paper reports between ShareGPT and Alpaca.
PRESETS: dict[str, Preset] = {
    "short": Preset("instruction following (Alpaca-like)",
                    parse_length_spec("lognormal:20:0.8:4:256"),
                    parse_length_spec("lognormal:60:0.8:4:512")),
    "chat": Preset("chatbot conversations (ShareGPT-like)",
                   parse_length_spec("lognormal:168:1.0:4:1536"),
                   parse_length_spec("lognormal:348:1.0:4:1536")),
    "long-prompt": Preset("summarisation / RAG",
                          parse_length_spec("uniform:1024:1792"),
                          parse_length_spec("uniform:32:256")),
    "long-output": Preset("code or story generation",
                          parse_length_spec("uniform:16:128"),
                          parse_length_spec("uniform:512:1536")),
}
DEFAULT_PRESET = "chat"


# --- random generator ---------------------------------------------------------

@dataclass
class GeneratedWorkload:
    requests: list[Request]
    clipped: int  # requests whose drawn lengths were cut to fit max_model_len / max_new_tokens


def generate_workload(
    n: int,
    *,
    preset: str | None = None,
    prompt: str | int | LengthSpec | None = None,
    output: str | int | LengthSpec | None = None,
    max_model_len: int,
    max_new_tokens: int | None = None,
    arrival_rate: float | None = None,
    seed: int = 0,
) -> GeneratedWorkload:
    """Draw ``n`` random requests.

    ``preset`` picks the prompt and output distributions (default ``chat``). ``prompt``
    or ``output`` replace the preset's distribution for that length. Without
    ``arrival_rate`` every request arrives at step 0; with it, arrivals follow a
    Poisson process with that many requests per step on average.

    Lengths are clipped so that ``P + O <= max_model_len`` and ``O <= max_new_tokens``.
    Prompt lengths, output lengths and arrival gaps come from three separate random
    streams, so changing one setting (say the arrival rate) leaves the others unchanged.
    """
    _check_int("number of requests", n, 1)
    _check_int("max_model_len", max_model_len, 2)
    if max_new_tokens is not None:
        _check_int("max_new_tokens", max_new_tokens, 1)
    if arrival_rate is not None and not (math.isfinite(arrival_rate) and arrival_rate > 0):
        raise ConfigError(f"arrival_rate must be a positive number of requests per step, "
                          f"got {arrival_rate!r}")
    name = DEFAULT_PRESET if preset is None else preset.lower()
    if name not in PRESETS:
        raise ConfigError(f"unknown preset {preset!r}; choose from {', '.join(PRESETS)}")
    prompt_spec = PRESETS[name].prompt if prompt is None else parse_length_spec(prompt)
    output_spec = PRESETS[name].output if output is None else parse_length_spec(output)

    prompt_rng = random.Random(f"{seed}:prompt")
    output_rng = random.Random(f"{seed}:output")
    arrival_rng = random.Random(f"{seed}:arrival")
    requests: list[Request] = []
    clipped = 0
    clock = 0.0
    for i in range(n):
        drawn_p, drawn_o = prompt_spec.sample(prompt_rng), output_spec.sample(output_rng)
        if arrival_rate is not None:
            clock += arrival_rng.expovariate(arrival_rate)
        p = min(drawn_p, max_model_len - 1)  # leave room for at least one output token
        limit = max_model_len - p
        if max_new_tokens is not None:
            limit = min(limit, max_new_tokens)
        o = min(drawn_o, limit)
        clipped += (p, o) != (drawn_p, drawn_o)
        requests.append(Request(i, math.floor(clock), p, o, max_new_tokens))
    return GeneratedWorkload(requests, clipped)


def generate_requests(n: int, **kwargs) -> list[Request]:
    """Same as ``generate_workload`` but returns only the requests."""
    return generate_workload(n, **kwargs).requests


# --- trace files --------------------------------------------------------------

def save_trace(requests, path: str | Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(TRACE_COLUMNS)
        for r in requests:
            writer.writerow([r.id, r.arrival_step, r.prompt_len, r.output_len,
                             "" if r.max_new_tokens is None else r.max_new_tokens])


def _parse_int(text: str | None, where: str, column: str) -> int:
    text = (text or "").strip()
    if not text:
        raise ConfigError(f"{where}: missing {column}")
    try:
        return int(text)
    except ValueError:
        raise ConfigError(f"{where}: {column} must be a whole number, got {text!r}") from None


def _row_to_request(row: dict, path: Path, line: int) -> Request:
    if None in row:  # csv.DictReader files extra values under the key None
        raise ConfigError(f"{path}, line {line}: more values than columns")
    request_id = _parse_int(row.get("id"), f"{path}, line {line}", "id")
    where = f"{path}, request id {request_id}"
    values = {c: _parse_int(row.get(c), where, c) for c in ("arrival_step", "prompt_len", "output_len")}
    limit = (row.get("max_new_tokens") or "").strip()
    values["max_new_tokens"] = _parse_int(limit, where, "max_new_tokens") if limit else None
    try:
        return Request(request_id, **values)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from None


def load_trace(path: str | Path, max_model_len: int) -> list[Request]:
    """Read a trace CSV, check every row, and return the requests in arrival order."""
    path = Path(path)
    try:
        f = open(path, newline="", encoding="utf-8-sig")  # -sig: tolerate Excel's byte-order mark
    except OSError as exc:
        raise ConfigError(f"cannot read trace {path}: {exc.strerror}") from None
    with f:
        reader = csv.DictReader(f)
        header = [name.strip() for name in reader.fieldnames or []]
        if not header:
            raise ConfigError(f"{path}: empty trace file")
        unknown = sorted(set(header) - set(TRACE_COLUMNS))
        missing = [c for c in REQUIRED_TRACE_COLUMNS if c not in header]
        if unknown or missing or len(set(header)) != len(header):
            problem = (f"unknown column(s) {unknown}" if unknown
                       else f"missing column(s) {missing}" if missing else "repeated column names")
            raise ConfigError(f"{path}: {problem}; expected {', '.join(TRACE_COLUMNS)}")
        reader.fieldnames = header
        requests = [_row_to_request(row, path, line) for line, row in enumerate(reader, start=2)]
    if not requests:
        raise ConfigError(f"{path}: the trace has no requests")
    try:
        validate_requests(requests, max_model_len)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from None
    return sorted(requests, key=lambda r: (r.arrival_step, r.id))


# --- statistics ---------------------------------------------------------------

def percentile(values, p: float):
    """Nearest-rank percentile: the smallest value with at least ``p``% of values at or below it."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of no values")
    if not 0 <= p <= 100:
        raise ValueError(f"p must be between 0 and 100, got {p}")
    rank = max(1, math.ceil(p * len(ordered) / 100))
    return ordered[rank - 1]


def workload_stats(requests) -> list[dict]:
    """min / mean / p50 / p95 / max of prompt, output and total length, and of arrival step."""
    requests = list(requests)
    if not requests:
        return []
    columns = {
        "prompt_len": [r.prompt_len for r in requests],
        "output_len": [r.output_len for r in requests],
        "total_len": [r.total_len for r in requests],
        "arrival_step": [r.arrival_step for r in requests],
    }
    return [{"quantity": name, "min": min(v), "mean": sum(v) / len(v),
             "p50": percentile(v, 50), "p95": percentile(v, 95), "max": max(v)}
            for name, v in columns.items()]
