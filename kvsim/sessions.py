"""Multi-turn chat sessions: the workload that makes LRU meaningful.

A session is a conversation of several turns. Turn ``k`` sends the whole history
(every earlier prompt and answer) plus a new message, so its prompt is

    prompt_len(k) = sum(new_prompt + output over turns < k) + new_prompt(k)

After a turn finishes, the user thinks for ``think_steps`` steps before sending
the next one. Meanwhile the server can keep the turn's KV-cache. If it is still
there when the next turn arrives (a cache hit), only the new tokens are
prefilled; if it was evicted to make room (a miss), the whole history is
prefilled again. Which idle cache to evict is the page-replacement question:
LRU, FIFO, or Belady's optimal policy (OPT), which the simulator can compute
exactly because an idle session's next arrival is already known.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .config import ConfigError
from .workload import LengthSpec, Request, _check_int, parse_length_spec


@dataclass(frozen=True)
class Turn:
    new_prompt_len: int  # tokens the user adds in this turn
    output_len: int  # tokens the model answers with
    think_steps: int = 1  # steps between the previous turn finishing and this one arriving

    def __post_init__(self) -> None:
        _check_int("turn new_prompt_len", self.new_prompt_len, 1)
        _check_int("turn output_len", self.output_len, 1)
        _check_int("turn think_steps", self.think_steps, 0)


@dataclass(frozen=True)
class Session:
    id: int
    start_step: int  # arrival of the first turn
    turns: tuple[Turn, ...]

    def __post_init__(self) -> None:
        _check_int("session id", self.id, 0)
        _check_int(f"session {self.id}: start_step", self.start_step, 0)
        if not self.turns:
            raise ConfigError(f"session {self.id} has no turns")

    def history_len(self, k: int) -> int:
        """Tokens of all turns before turn ``k``."""
        return sum(t.new_prompt_len + t.output_len for t in self.turns[:k])

    def prompt_len(self, k: int) -> int:
        return self.history_len(k) + self.turns[k].new_prompt_len

    def total_len(self) -> int:
        return self.history_len(len(self.turns))

    def request(self, k: int, request_id: int, arrival_step: int) -> Request:
        return Request(request_id, arrival_step, self.prompt_len(k), self.turns[k].output_len,
                       session_id=self.id, turn=k)


def validate_sessions(sessions: list[Session], max_model_len: int) -> None:
    seen: set[int] = set()
    for s in sessions:
        if s.id in seen:
            raise ConfigError(f"session {s.id}: duplicate id")
        seen.add(s.id)
        if s.total_len() > max_model_len:
            raise ConfigError(f"session {s.id}: {s.total_len()} tokens is more than "
                              f"max_model_len {max_model_len}")


def generate_sessions(
    n: int,
    *,
    turns: str | int | LengthSpec = "uniform:2:6",
    prompt: str | int | LengthSpec = "lognormal:60:0.8:4:512",
    output: str | int | LengthSpec = "lognormal:120:0.8:4:512",
    think: str | int | LengthSpec = "uniform:1:200",
    max_model_len: int,
    arrival_rate: float | None = None,
    seed: int = 0,
) -> list[Session]:
    """Draw ``n`` random sessions.

    Each length comes from its own random stream. A session stops early once its
    next turn would no longer fit in ``max_model_len``; its first turn is clipped
    to fit, like ``generate_workload``. Without ``arrival_rate`` all sessions start
    at step 0; with it, session starts follow a Poisson process.
    """
    _check_int("number of sessions", n, 1)
    _check_int("max_model_len", max_model_len, 2)
    if arrival_rate is not None and not (math.isfinite(arrival_rate) and arrival_rate > 0):
        raise ConfigError(f"arrival_rate must be a positive number, got {arrival_rate!r}")
    specs = {name: parse_length_spec(spec) for name, spec in
             (("turns", turns), ("prompt", prompt), ("output", output), ("think", think))}
    rngs = {name: random.Random(f"{seed}:session-{name}") for name in (*specs, "arrival")}
    sessions, clock = [], 0.0
    for sid in range(n):
        if arrival_rate is not None:
            clock += rngs["arrival"].expovariate(arrival_rate)
        wanted = specs["turns"].sample(rngs["turns"])
        session_turns, used = [], 0
        for k in range(wanted):
            new = specs["prompt"].sample(rngs["prompt"])
            out = specs["output"].sample(rngs["output"])
            pause = specs["think"].sample(rngs["think"])
            if k == 0:  # always keep the first turn, clipped to fit
                new = min(new, max_model_len - 1)
                out = min(out, max_model_len - new)
            elif used + new + out > max_model_len:
                break
            session_turns.append(Turn(new, out, pause))
            used += new + out
        sessions.append(Session(sid, math.floor(clock), tuple(session_turns)))
    return sessions
