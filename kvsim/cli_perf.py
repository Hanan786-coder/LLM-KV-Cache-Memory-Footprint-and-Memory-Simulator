"""Bandwidth and latency commands: roofline, scaling and profile.

Registered from ``cli.build_parser`` with one call to ``register``.
"""

from __future__ import annotations

import argparse

from . import budget as bud
from .cli_sim import (
    _config, _engine_parent, _setup, _sim_parent, _workload_parent, parse_config_label,
)
from .config import GIB, ConfigError, load_gpu, load_model
from .perf import Hardware, PerfModel
from .plot import line_chart, roofline_chart, save_svg
from .precision import get_precision
from .profiler import load_requests_csv, load_steps_csv, profile_result, profile_steps
from .report import _fmt_cell, print_table, write_csv
from .scaling import (
    CURVES, DEFAULT_BATCHES, DEFAULT_CONTEXTS, DEFAULT_PROMPTS, batch_curve, best_batch,
    context_curve, prefill_compute_bound_prompt, prefill_curve,
)
from .simulator import simulate

STEP_COLUMNS = ("phase", "tokens", "weight_bytes", "kv_read_bytes", "kv_write_bytes", "hbm_bytes",
                "flops", "intensity", "t_memory_ms", "t_compute_ms", "latency_ms", "tokens_per_s",
                "bound")
PROFILE_COMPARE_COLUMNS = ("config", "finished", "steps", "total_time_s", "throughput_tok_s",
                           "decode_step_p50_ms", "share_time_memory_bound", "ttft_p50_ms",
                           "ttft_p95_ms", "tpot_p50_ms", "e2e_p95_ms")
ROOFLINE_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
ROOFLINE_PROMPTS = (16, 64, 256, 1024, 4096)


def _int_list(text: str) -> list[int]:
    try:
        values = [int(v) for v in text.split(",") if v.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {text!r}")
    if not values or any(v < 1 for v in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def _hardware(args, gpu) -> Hardware:
    return Hardware(gpu, bw_efficiency=args.bw_efficiency,
                    compute_efficiency=args.compute_efficiency,
                    step_overhead_s=args.step_overhead_us * 1e-6)


def _perf(args, model, gpu) -> PerfModel:
    return PerfModel(model, _hardware(args, gpu),
                     weight_precision=get_precision(args.weight_dtype, args.weight_group_size),
                     kv_precision=get_precision(args.kv_dtype, args.kv_group_size))


def _step_row(cost) -> dict:
    row = cost.row()
    row.update(t_memory_ms=cost.t_memory_s * 1e3, t_compute_ms=cost.t_compute_s * 1e3,
               latency_ms=cost.latency_s * 1e3)
    return {c: row[c] for c in STEP_COLUMNS}


def _write(rows, path, what="rows") -> None:
    if path and rows:
        write_csv(rows, path)
        print(f"wrote {len(rows)} {what} to {path}")


def _svg(svg: str, path) -> None:
    if path:
        save_svg(svg, path)
        print(f"wrote {path}")


# --- commands -----------------------------------------------------------------

def cmd_roofline(args) -> None:
    model = load_model(args.model)
    perf = _perf(args, model, load_gpu(args.gpu))
    prefill, decode = perf.prefill(args.prompt), perf.decode(args.batch, args.context)
    print(perf.describe())
    rows = [_step_row(prefill), _step_row(decode)]
    print_table(rows, f"\nprefill of {args.prompt} tokens | decode of batch {args.batch} "
                      f"at {args.context} tokens of context")

    scenario = bud.Scenario(model, perf.gpu, context_len=args.context + 1, batch=args.batch,
                            weight_precision=perf.weight_precision, kv_precision=perf.kv_precision,
                            prefill_chunk=args.prefill_chunk, reserved_bytes=int(args.reserved_gib * GIB))
    crossing = perf.decode_compute_bound_batch(args.context)
    max_batch = bud.max_batch(scenario)
    prompt_crossing = prefill_compute_bound_prompt(perf)
    print()
    print(f"decode at batch {args.batch}: {decode.tokens_per_s:,.0f} tok/s in total, "
          f"{decode.latency_s * 1e3:.2f} ms per token for each sequence, {decode.bound}-bound")
    if crossing is None:
        print("decode stays memory-bound at every batch size (KV reads grow with the batch)", end="")
    else:
        print(f"decode turns compute-bound from batch {crossing:,}", end="")
    print(f"; memory fits batch {max_batch:,} at this context")
    if prompt_crossing is None:
        print("prefill stays memory-bound at every prompt length")
    else:
        print(f"prefill turns compute-bound from prompts of {prompt_crossing:,} tokens")
    if not bud.fits(scenario):
        print(f"warning: batch {args.batch} at context {args.context} does not fit in GPU memory")
    _write(rows, args.csv)

    if args.svg:
        points = [("decode", c.intensity, c.flops / c.latency_s)
                  for c in (perf.decode(b, args.context) for b in ROOFLINE_BATCHES)]
        points += [("prefill", c.intensity, c.flops / c.latency_s)
                   for c in (perf.prefill(p) for p in ROOFLINE_PROMPTS)]
        _svg(roofline_chart(perf.hw.flops, perf.hw.bandwidth, points,
                            f"{model.name} on {perf.gpu.name}: decode batch 1-512, prefill 16-4096"),
             args.svg)


def cmd_scaling(args) -> None:
    model = load_model(args.model)
    gpus = [g.strip() for g in args.gpu.split(",") if g.strip()]
    reserved = int(args.reserved_gib * GIB)
    all_rows, series = [], []
    for key in gpus:
        perf = _perf(args, model, load_gpu(key))
        if args.vary == "batch":
            rows = batch_curve(perf, args.values or DEFAULT_BATCHES, args.context,
                               args.prefill_chunk, reserved)
            y = "tokens_per_s"
        elif args.vary == "context":
            rows = context_curve(perf, args.values or DEFAULT_CONTEXTS, args.batch,
                                 args.prefill_chunk, reserved)
            y = "latency_ms"
        else:
            rows = prefill_curve(perf, args.values or DEFAULT_PROMPTS, args.prefill_chunk, reserved)
            y = "ttft_ms"
        x = "prompt" if args.vary == "prompt" else args.vary
        print(perf.describe())
        print_table(rows)
        if args.vary == "batch":
            best = best_batch(rows)
            if best:
                print(f"best fitting batch: {best['batch']:,} at {best['tokens_per_s']:,.0f} tok/s")
        print()
        all_rows += [{"gpu": key, **r} for r in rows]
        series.append((perf.gpu.name, [r[x] for r in rows], [r[y] for r in rows]))
    _write(all_rows, args.csv)
    if args.svg:
        labels = {"batch": ("batch size", "decode tokens/s (all sequences)",
                            f"decode at {args.context} tokens of context"),
                  "context": ("context length (tokens)", "decode step latency (ms)",
                              f"decode at batch {args.batch}"),
                  "prompt": ("prompt length (tokens)", "time to first token (ms)", "prefill")}
        xl, yl, what = labels[args.vary]
        _svg(line_chart(series, f"{model.name}: {what}", xl, yl, logx=True, logy=True), args.svg)


def _load_or_simulate(args, perf_for):
    if args.from_steps:
        model, gpu = load_model(args.model), load_gpu(args.gpu)
        perf = perf_for(model, gpu)
        requests = load_requests_csv(args.from_requests) if args.from_requests else None
        print(f"Profiling {args.from_steps}" + (f" and {args.from_requests}" if requests else ""))
        return perf, profile_steps(perf, load_steps_csv(args.from_steps), requests)
    model, pool, requests, max_model_len = _setup(args)
    perf = perf_for(model, load_gpu(args.gpu))
    return perf, profile_result(perf, simulate(requests, pool, _config(args, max_model_len)))


def cmd_profile(args) -> None:
    if args.from_requests and not args.from_steps:
        raise ConfigError("--from-requests needs --from-steps")
    if args.configs:
        if args.from_steps:
            raise ConfigError("--configs simulates the workload; it cannot be used with --from-steps")
        _profile_many(args)
        return
    perf, prof = _load_or_simulate(args, lambda m, g: _perf(args, m, g))
    print(perf.describe())
    summary = prof.summary()
    print()
    print_table([{"metric": k, "value": _fmt_cell(k, v)} for k, v in summary.items()],
                "Time, throughput and latency")
    print()
    print_table(prof.phase_rows(), "By phase")
    if args.csv:
        write_csv([summary], args.csv)
        print(f"\nwrote the summary to {args.csv}")
    _write(prof.step_rows(), args.steps_csv, "timed steps")
    _write(prof.request_rows(), args.requests_csv, "timed requests")
    if args.svg:
        points = [(s.cost.phase, s.cost.intensity, s.cost.flops / s.cost.latency_s)
                  for s in prof.steps if s.cost.latency_s]
        _svg(roofline_chart(perf.hw.flops, perf.hw.bandwidth, points,
                            f"{perf.model.name} on {perf.gpu.name}: every simulated step"), args.svg)


def _profile_many(args) -> None:
    model, pool, requests, max_model_len = _setup(args)
    perf = _perf(args, model, load_gpu(args.gpu))
    print(perf.describe())
    rows, full = [], []
    for label in (c.strip() for c in args.configs.split(",") if c.strip()):
        result = simulate(requests, pool, _config(args, max_model_len, **parse_config_label(label)))
        summary = {"config": label, **profile_result(perf, result).summary()}
        full.append(summary)
        rows.append({c: summary[c] for c in PROFILE_COMPARE_COLUMNS})
    print()
    print_table(rows, "Same workload, different allocators, in real time")
    if args.csv:
        write_csv(full, args.csv)
        print(f"\nwrote {len(full)} rows to {args.csv}")


# --- parser -------------------------------------------------------------------

def _hardware_parent() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("hardware efficiency (default: the ideal roofline)")
    g.add_argument("--bw-efficiency", type=float, default=1.0,
                   help="share of peak memory bandwidth reached (default 1.0)")
    g.add_argument("--compute-efficiency", type=float, default=1.0,
                   help="share of peak FP16 FLOP/s reached (default 1.0)")
    g.add_argument("--step-overhead-us", type=float, default=0.0,
                   help="fixed time added to every step, in microseconds (default 0)")
    return p


def register(sub, out: argparse.ArgumentParser, gpu: argparse.ArgumentParser) -> None:
    """Add the bandwidth/latency commands to the ``kvsim`` sub-command parser."""
    sim, hw = _sim_parent(), _hardware_parent()

    p = sub.add_parser("roofline", parents=[sim, gpu, hw, out],
                       help="bytes, FLOPs, latency and bound of one prefill and one decode step")
    p.add_argument("--batch", type=int, default=1, help="decode batch size (default 1)")
    p.add_argument("--context", type=int, default=1024, help="tokens already cached (default 1024)")
    p.add_argument("--prompt", type=int, default=512, help="prefill prompt length (default 512)")
    p.add_argument("--svg", metavar="FILE", help="draw the roofline with decode and prefill points")
    p.set_defaults(func=cmd_roofline)

    p = sub.add_parser("scaling", parents=[sim, gpu, hw, out],
                       help="latency and throughput vs batch, context or prompt length")
    p.add_argument("--vary", default="batch", choices=CURVES, help="what to sweep (default batch)")
    p.add_argument("--values", type=_int_list, help="comma-separated values to sweep")
    p.add_argument("--batch", type=int, default=1, help="batch for --vary context (default 1)")
    p.add_argument("--context", type=int, default=1024, help="context for --vary batch (default 1024)")
    p.add_argument("--svg", metavar="FILE", help="draw the curve (one line per GPU)")
    p.set_defaults(func=cmd_scaling)

    p = sub.add_parser("profile", parents=[sim, gpu, _workload_parent(), _engine_parent(), hw, out],
                       help="simulate a workload and time every step (throughput, TTFT, TPOT)")
    g = p.add_argument_group("input / output")
    g.add_argument("--from-steps", metavar="FILE", help="profile a `simulate --steps-csv` file instead")
    g.add_argument("--from-requests", metavar="FILE", help="with --from-steps: its `--requests-csv` file")
    g.add_argument("--configs", help="compare allocators in real time, e.g. paged-16,contig-max")
    g.add_argument("--steps-csv", metavar="FILE", help="write every step with its time")
    g.add_argument("--requests-csv", metavar="FILE", help="write every request's TTFT/TPOT/E2E")
    g.add_argument("--svg", metavar="FILE", help="draw every step on the roofline")
    p.set_defaults(func=cmd_profile)


__all__ = ["register"]
