"""Part 2 command-line commands: workload, simulate, compare, blocksweep and sessions.

Registered from ``cli.build_parser`` with one call to ``register`` so that each
part keeps its commands in its own module.
"""

from __future__ import annotations

import argparse

from . import budget as bud
from .config import GIB, ConfigError, load_gpu, load_model
from .kvpool import DEFAULT_MAX_NUM_SEQS, KVPool, pool_from_scenario
from .precision import PRECISIONS, get_precision
from .report import _fmt_cell, fmt_bytes, print_table, write_csv
from .simulator import (
    ALLOCATORS, BATCHING, REQUEST_COLUMNS, STEP_COLUMNS, SimConfig, SimResult, simulate,
)
from .allocator import FITS, RESERVATIONS
from .eviction import CACHE_POLICIES, PREEMPT_MODES, VICTIM_POLICIES
from .sessions import generate_sessions
from .workload import PRESETS, generate_workload, load_trace, save_trace, workload_stats

DEFAULT_CONFIGS = "contig-max,contig-pow2,contig-oracle,paged-16"
DEFAULT_BLOCK_SIZES = "1,2,4,8,16,32,64,128,256"
TIMELINE_COLUMNS = ("step", "phase", "batch_size", "running", "waiting", "kv_used_bytes",
                    "kv_allocated_bytes", "internal_frag_bytes", "external_frag_bytes",
                    "free_bytes", "preempted")
COMPARE_COLUMNS = ("allocator", "finished", "rejected", "steps", "avg_decode_batch", "peak_running",
                   "share_token_states", "share_internal_frag", "share_external_frag",
                   "preemptions", "frag_stalls", "e2e_p50", "e2e_p95")


def _int_list(text: str) -> list[int]:
    try:
        values = [int(v) for v in text.split(",") if v.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {text!r}")
    if not values or any(v < 1 for v in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


# --- building a run from the flags -------------------------------------------

def _max_model_len(args, model=None) -> int:
    if args.max_model_len is not None:
        return args.max_model_len
    if model is not None and model.max_context:
        return model.max_context
    if model is None:
        return 2048
    raise ConfigError(f"{model.name} has no max_context; pass --max-model-len")


def _requests(args, max_model_len: int):
    """The workload from --trace or the generator; returns (requests, one-line description)."""
    if args.trace:
        requests = load_trace(args.trace, max_model_len)
        return requests, f"{len(requests)} requests from {args.trace}"
    w = generate_workload(args.num_requests, preset=args.preset, prompt=args.prompt_len,
                          output=args.output_len, max_model_len=max_model_len,
                          max_new_tokens=args.max_new_tokens, arrival_rate=args.arrival_rate,
                          seed=args.seed)
    arrivals = "all at step 0" if args.arrival_rate is None else f"{args.arrival_rate} per step"
    return w.requests, (f"{len(w.requests)} '{args.preset}' requests, seed {args.seed}, arrivals "
                        f"{arrivals}, {w.clipped} clipped to {max_model_len} tokens")


def _scenario(args, model, gpu) -> bud.Scenario:
    return bud.Scenario(
        model=model, gpu=gpu, context_len=0, batch=1,
        weight_precision=get_precision(args.weight_dtype, args.weight_group_size),
        kv_precision=get_precision(args.kv_dtype, args.kv_group_size),
        prefill_chunk=args.prefill_chunk, reserved_bytes=int(args.reserved_gib * GIB),
    )


def _pool(args, scenario) -> KVPool:
    pool_bytes = None if args.pool_gib is None else int(args.pool_gib * GIB)
    return pool_from_scenario(scenario, args.max_num_seqs, args.pool_tokens, pool_bytes)


def _config(args, max_model_len: int, **changes) -> SimConfig:
    settings = dict(
        allocator=args.allocator, block_size=args.block_size, watermark=args.watermark,
        reservation=args.reservation, fit=args.fit, batching=args.batching,
        max_num_seqs=args.max_num_seqs, max_model_len=max_model_len,
        max_batched_tokens=args.max_batched_tokens or args.prefill_chunk,
        window=args.window, sink_tokens=args.sink_tokens, evict_policy=args.evict_policy,
        preempt_mode=args.preempt_mode, swap_bytes=int(args.swap_gib * GIB),
    )
    settings.update(changes)
    return SimConfig(**settings)


def parse_config_label(label: str) -> dict:
    """``paged-<B>`` or ``contig-<reservation>[-best]`` -> SimConfig changes."""
    parts = label.strip().lower().split("-")
    if len(parts) == 2 and parts[0] == "paged" and parts[1].isdigit() and int(parts[1]) > 0:
        return {"allocator": "paged", "block_size": int(parts[1])}
    if parts[0] == "contig" and len(parts) in (2, 3) and parts[1] in RESERVATIONS:
        if len(parts) == 2 or parts[2] in FITS:
            return {"allocator": "contiguous", "reservation": parts[1],
                    "fit": parts[2] if len(parts) == 3 else "first"}
    raise ConfigError(f"bad configuration {label!r}; use paged-<block size> or "
                      f"contig-<{'|'.join(RESERVATIONS)}>[-best]")


def _setup(args):
    model, gpu = load_model(args.model), load_gpu(args.gpu)
    max_model_len = _max_model_len(args, model)
    scenario = _scenario(args, model, gpu)
    pool = _pool(args, scenario)
    requests, described = _requests(args, max_model_len)
    print(f"{model.name} ({model.attention_type}) on {gpu.name} | weights "
          f"{scenario.weight_precision.name}, KV {scenario.kv_precision.name}")
    print(f"KV pool: {fmt_bytes(pool.total_bytes)} = {pool.total_tokens:,} token slots "
          f"of {fmt_bytes(pool.bytes_per_token)}")
    print(f"Workload: {described}")
    return model, pool, requests, max_model_len


def _write_traces(args, result: SimResult) -> None:
    if args.steps_csv:
        write_csv(result.step_rows(), args.steps_csv)
        print(f"wrote {len(result.steps)} steps to {args.steps_csv}")
    if args.requests_csv:
        write_csv(result.request_rows(), args.requests_csv)
        print(f"wrote {len(result.sequences)} requests to {args.requests_csv}")


def _timeline(result: SimResult, n: int) -> list[dict]:
    steps = result.steps
    if n <= 0 or not steps:
        return []
    if len(steps) <= n:
        picked = steps
    else:
        picked = [steps[round(i * (len(steps) - 1) / (n - 1))] for i in range(n)] if n > 1 else steps[:1]
    return [{c: getattr(s, c) for c in TIMELINE_COLUMNS} for s in picked]


# --- commands -----------------------------------------------------------------

def cmd_workload(args) -> None:
    model = load_model(args.model) if args.model else None
    max_model_len = _max_model_len(args, model)
    requests, described = _requests(args, max_model_len)
    print_table(workload_stats(requests), f"Workload: {described}")
    if args.csv:
        save_trace(requests, args.csv)
        print(f"\nwrote {len(requests)} requests to {args.csv}")


def cmd_simulate(args) -> None:
    _, pool, requests, max_model_len = _setup(args)
    result = simulate(requests, pool, _config(args, max_model_len))
    summary = result.summary()
    print()
    print_table([{"metric": k, "value": _fmt_cell(k, v)} for k, v in summary.items()],
                f"Summary ({summary['allocator']})")
    timeline = _timeline(result, args.timeline)
    if timeline:
        print()
        print_table(timeline, f"Timeline ({len(timeline)} of {len(result.steps)} steps)")
    if args.csv:
        write_csv([summary], args.csv)
        print(f"\nwrote the summary to {args.csv}")
    _write_traces(args, result)


def _run_many(args, labelled_changes, columns, title) -> None:
    _, pool, requests, max_model_len = _setup(args)
    rows, full = [], []
    for extra, changes in labelled_changes:
        summary = simulate(requests, pool, _config(args, max_model_len, **changes)).summary()
        full.append({**extra, **summary})
        rows.append({**extra, **{c: summary[c] for c in columns}})
    print()
    print_table(rows, title)
    if args.csv:
        write_csv(full, args.csv)
        print(f"\nwrote {len(full)} rows to {args.csv}")


def cmd_compare(args) -> None:
    configs = [(label, parse_config_label(label)) for label in args.configs.split(",") if label.strip()]
    _run_many(args, [({}, changes) for _, changes in configs], COMPARE_COLUMNS,
              "Same workload, different allocators")


def cmd_blocksweep(args) -> None:
    sizes = args.block_sizes
    columns = ("steps", "avg_decode_batch", "share_token_states", "share_internal_frag",
               "preemptions", "recompute_tokens", "e2e_p95")
    _run_many(args, [({"block_size": b}, {"allocator": "paged", "block_size": b}) for b in sizes],
              columns, "Paged allocator by block size")


def cmd_sessions(args) -> None:
    policies = [p.strip().lower() for p in args.policies.split(",") if p.strip()]
    unknown = [p for p in policies if p not in CACHE_POLICIES]
    if unknown or not policies:
        raise ConfigError(f"unknown cache policies {unknown}; choose from {', '.join(CACHE_POLICIES)}")
    model, gpu = load_model(args.model), load_gpu(args.gpu)
    max_model_len = _max_model_len(args, model)
    scenario = _scenario(args, model, gpu)
    pool = _pool(args, scenario)
    lengths = {k: v for k, v in (("prompt", args.prompt_len), ("output", args.output_len)) if v}
    sessions = generate_sessions(args.num_sessions, turns=args.turns, think=args.think,
                                 max_model_len=max_model_len, arrival_rate=args.arrival_rate,
                                 seed=args.seed, **lengths)
    turns = sum(len(s.turns) for s in sessions)
    print(f"{model.name} on {gpu.name} | KV pool {fmt_bytes(pool.total_bytes)} = "
          f"{pool.total_tokens:,} token slots")
    print(f"Workload: {len(sessions)} sessions, {turns} turns, seed {args.seed}")
    rows, full = [], []
    for policy in policies:
        summary = simulate([], pool, _config(args, max_model_len, cache_policy=policy),
                           sessions=sessions).summary()
        full.append({"cache_policy": policy, **summary})
        rows.append({"cache_policy": policy, **{c: summary[c] for c in (
            "steps", "cache_hits", "cache_misses", "cache_hit_rate", "cache_evictions",
            "saved_prefill_tokens", "preemptions", "e2e_p95")}})
    print()
    print_table(rows, "Multi-turn sessions: which idle cache to drop")
    if args.csv:
        write_csv(full, args.csv)
        print(f"\nwrote {len(full)} rows to {args.csv}")


# --- parser -------------------------------------------------------------------

def _workload_parent() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("workload")
    g.add_argument("--trace", metavar="FILE", help="replay a trace CSV instead of generating requests")
    g.add_argument("--preset", default="chat", choices=PRESETS, help="length distributions (default chat)")
    g.add_argument("--num-requests", type=int, default=200, help="requests to generate (default 200)")
    g.add_argument("--prompt-len", metavar="SPEC", help="e.g. fixed:128, uniform:16:512, lognormal:168:1.0")
    g.add_argument("--output-len", metavar="SPEC", help="output length distribution (same syntax)")
    g.add_argument("--max-new-tokens", type=int, help="client limit on output tokens")
    g.add_argument("--arrival-rate", type=float, help="Poisson arrivals per step (default: all at step 0)")
    g.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    g.add_argument("--max-model-len", type=int, help="longest prompt + output (default: the model's max_context)")
    return p


def _sim_parent() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("model", help="model key (see `list`) or path to a .json profile")
    p.add_argument("--kv-dtype", default="fp16", choices=PRECISIONS, help="KV-cache precision")
    p.add_argument("--kv-group-size", type=int, help="quantization group size for INT KV")
    return p


def _engine_parent() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("KV pool")
    size = g.add_mutually_exclusive_group()
    size.add_argument("--pool-tokens", type=int, help="override the pool size in token slots")
    size.add_argument("--pool-gib", type=float, help="override the pool size in GiB")
    g = p.add_argument_group("allocator")
    g.add_argument("--allocator", default="paged", choices=ALLOCATORS)
    g.add_argument("--block-size", type=int, default=16, help="paged: tokens per block (default 16)")
    g.add_argument("--watermark", type=float, default=0.01, help="paged: share of blocks kept free at admission")
    g.add_argument("--reservation", default="max", choices=RESERVATIONS, help="contiguous: reservation size")
    g.add_argument("--fit", default="first", choices=FITS, help="contiguous: hole placement")
    g = p.add_argument_group("scheduler")
    g.add_argument("--batching", default="continuous", choices=BATCHING)
    g.add_argument("--max-num-seqs", type=int, default=DEFAULT_MAX_NUM_SEQS, help="most requests running at once")
    g.add_argument("--max-batched-tokens", type=int, help="prompt tokens per prefill step (default --prefill-chunk)")
    g = p.add_argument_group("eviction")
    g.add_argument("--evict-policy", default="lifo", choices=VICTIM_POLICIES)
    g.add_argument("--preempt-mode", default="recompute", choices=PREEMPT_MODES)
    g.add_argument("--swap-gib", type=float, default=4.0, help="CPU swap space (default 4 GiB)")
    g = p.add_argument_group("sliding window")
    g.add_argument("--window", type=int, help="keep only the latest W tokens of each request")
    g.add_argument("--sink-tokens", type=int, default=0, help="with --window, also keep the first S tokens")
    return p


def _trace_parent() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("output")
    g.add_argument("--steps-csv", metavar="FILE", help="write the per-step trace (the Part 3 input)")
    g.add_argument("--requests-csv", metavar="FILE", help="write the per-request trace")
    return p


def register(sub, out: argparse.ArgumentParser, gpu: argparse.ArgumentParser) -> None:
    """Add the Part 2 commands to the ``kvsim`` sub-command parser."""
    workload, sim, engine, traces = _workload_parent(), _sim_parent(), _engine_parent(), _trace_parent()

    p = sub.add_parser("workload", parents=[workload], help="generate a workload and show its statistics")
    p.add_argument("--model", help="model whose max_context limits the lengths")
    p.add_argument("--csv", metavar="FILE", help="save the requests as a trace CSV")
    p.set_defaults(func=cmd_workload)

    p = sub.add_parser("simulate", parents=[sim, gpu, workload, engine, traces, out],
                       help="simulate KV-cache allocation step by step (Part 2)")
    p.add_argument("--timeline", type=int, default=10, help="steps to show in the terminal (default 10)")
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("compare", parents=[sim, gpu, workload, engine, out],
                       help="run several allocators on the same workload")
    p.add_argument("--configs", default=DEFAULT_CONFIGS,
                   help=f"comma-separated paged-<B> / contig-<reservation>[-best] (default {DEFAULT_CONFIGS})")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("blocksweep", parents=[sim, gpu, workload, engine, out],
                       help="paged allocator across block sizes")
    p.add_argument("--block-sizes", type=_int_list, default=_int_list(DEFAULT_BLOCK_SIZES))
    p.set_defaults(func=cmd_blocksweep)

    p = sub.add_parser("sessions", parents=[sim, gpu, engine, out],
                       help="multi-turn chats: keep caches between turns, compare LRU/FIFO/OPT")
    g = p.add_argument_group("sessions")
    g.add_argument("--num-sessions", type=int, default=200, help="conversations (default 200)")
    g.add_argument("--turns", default="uniform:2:6", metavar="SPEC", help="turns per session")
    g.add_argument("--prompt-len", metavar="SPEC", help="new tokens per user message")
    g.add_argument("--output-len", metavar="SPEC", help="tokens per answer")
    g.add_argument("--think", default="uniform:1:200", metavar="SPEC", help="steps between turns")
    g.add_argument("--arrival-rate", type=float, default=0.2, help="new sessions per step (default 0.2)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--max-model-len", type=int, help="longest conversation (default: max_context)")
    g.add_argument("--policies", default="lru,fifo,opt", help="cache policies to compare")
    p.set_defaults(func=cmd_sessions)


__all__ = ["register", "parse_config_label", "STEP_COLUMNS", "REQUEST_COLUMNS"]
