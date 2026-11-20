"""Command-line interface: ``python -m kvsim <command> ...``."""

from __future__ import annotations

import argparse
import sys

from . import budget as bud
from .config import GIB, ConfigError, list_gpus, list_models, load_gpu, load_model
from .footprint import (
    compare_attention, compare_precision, kv_bytes_per_token, kv_bytes_per_token_per_layer,
    kv_cache_bytes,
)
from .precision import PRECISIONS, get_precision
from .report import fmt_bytes, print_table, write_csv
from .weights import param_breakdown


def _int_list(text: str) -> list[int]:
    try:
        values = [int(v) for v in text.split(",") if v.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {text!r}")
    if not values or any(v < 0 for v in values):
        raise argparse.ArgumentTypeError("values must be non-negative integers")
    return values


def _emit(rows: list[dict], args, title: str | None = None) -> None:
    print_table(rows, title)
    if getattr(args, "csv", None):
        write_csv(rows, args.csv)
        print(f"\nwrote {len(rows)} rows to {args.csv}")


def _kv_precision(args):
    return get_precision(args.kv_dtype, args.kv_group_size)


def _scenario(args) -> bud.Scenario:
    return bud.Scenario(
        model=load_model(args.model),
        gpu=load_gpu(args.gpu),
        context_len=args.context,
        batch=args.batch,
        weight_precision=get_precision(args.weight_dtype, args.weight_group_size),
        kv_precision=_kv_precision(args),
        prefill_chunk=args.prefill_chunk,
        reserved_bytes=int(args.reserved_gib * GIB),
    )


def _describe(s: bud.Scenario) -> str:
    return (f"{s.model.name} ({s.model.attention_type}, {s.model.num_kv_heads} KV heads) on "
            f"{s.gpu.name} | weights {s.weight_precision.name}, KV {s.kv_precision.name}")


# --- commands ---------------------------------------------------------------

def cmd_list(args) -> None:
    models = []
    for key in list_models():
        m = load_model(key)
        models.append({"key": key, "name": m.name, "layers": m.num_layers, "heads": m.num_heads,
                       "kv_heads": m.num_kv_heads, "attention": m.attention_type,
                       "params_b": param_breakdown(m).total / 1e9, "max_context": m.max_context})
    gpus = []
    for key in list_gpus():
        g = load_gpu(key)
        gpus.append({"key": key, "name": g.name, "memory_gib": g.memory_gib,
                     "bandwidth_gbs": g.bandwidth_gbs, "fp16_tflops": g.peak_fp16_tflops,
                     "ridge_flops_per_byte": g.ridge_point})
    print_table(models, "Models")
    print()
    print_table(gpus, "GPUs")
    print("\nPrecisions:", ", ".join(PRECISIONS))


def cmd_kv(args) -> None:
    model, p = load_model(args.model), _kv_precision(args)
    rows = [{
        "model": model.name,
        "attention": model.attention_type,
        "precision": p.name,
        "context": args.context,
        "batch": args.batch,
        "bytes_per_token_per_layer": kv_bytes_per_token_per_layer(model, p),
        "bytes_per_token": kv_bytes_per_token(model, p),
        "bytes_per_sequence": kv_bytes_per_token(model, p) * args.context,
        "kv_cache_bytes": kv_cache_bytes(model, p, args.context, args.batch),
    }]
    _emit(rows, args, "KV-cache = 2 x layers x kv_heads x head_dim x context x batch x bytes")


def cmd_attention(args) -> None:
    model = load_model(args.model)
    rows = compare_attention(model, _kv_precision(args), args.context, args.batch, args.gqa_kv_heads)
    _emit(rows, args, f"{model.name}: KV-cache by attention type "
                      f"({args.context} tokens x {args.batch} seq, {args.kv_dtype})")


def cmd_precision(args) -> None:
    model = load_model(args.model)
    dtypes = [get_precision(d, args.kv_group_size) for d in args.dtypes.split(",")]
    rows = compare_precision(model, dtypes, args.context, args.batch)
    _emit(rows, args, f"{model.name}: KV-cache by precision ({args.context} tokens x {args.batch} seq)")


def cmd_budget(args) -> None:
    s = _scenario(args)
    b = bud.memory_budget(s)
    _emit(b.rows(), args, f"{_describe(s)}\ncontext {s.context_len}, batch {s.batch}")
    if b.fits:
        print(f"\nFITS: {b.utilization:.1%} of VRAM used, {fmt_bytes(b.free)} free")
    else:
        print(f"\nOUT OF MEMORY: needs {fmt_bytes(-b.free)} more than the GPU has")


def cmd_solve(args) -> None:
    s = _scenario(args)
    r = bud.oom_report(s)
    rows = [
        {"question": f"max batch at context {s.context_len}", "answer": r["max_batch_at_context"]},
        {"question": f"max context at batch {s.batch}", "answer": r["max_context_at_batch"]},
        {"question": "KV tokens that fit (all sequences)", "answer": r["kv_token_capacity"]},
        {"question": "model trained context", "answer": r["model_max_context"] or "unknown"},
    ]
    _emit(rows, args, _describe(s))
    if not r["weights_fit"]:
        print("\nThe weights alone do not fit on this GPU; try --weight-dtype int8/int4.")
    elif r["model_max_context"] and r["max_context_at_batch"] > r["model_max_context"]:
        print("\nNote: memory allows more context than the model was trained for.")


def cmd_sweep(args) -> None:
    base = _scenario(args)
    rows = []
    for batch in args.batches:
        for context in args.contexts:
            s = base.with_(batch=batch, context_len=context)
            b = bud.memory_budget(s)
            rows.append({"batch": batch, "context": context, "kv_cache_bytes": b.kv_cache,
                         "weights_bytes": b.weights, "activations_bytes": b.activations,
                         "total_bytes": b.total, "free_bytes": b.free, "fits": b.fits})
    _emit(rows, args, _describe(base))


# --- parser -----------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kvsim", description="LLM KV-cache memory footprint, allocation and bandwidth simulator.")
    sub = parser.add_subparsers(dest="command", required=True)

    out = argparse.ArgumentParser(add_help=False)
    out.add_argument("--csv", metavar="FILE", help="also write the table to a CSV file")

    shape = argparse.ArgumentParser(add_help=False)
    shape.add_argument("model", help="model key (see `list`) or path to a .json profile")
    shape.add_argument("--context", type=int, default=4096, help="tokens per sequence (default 4096)")
    shape.add_argument("--batch", type=int, default=1, help="number of sequences (default 1)")
    shape.add_argument("--kv-dtype", default="fp16", choices=PRECISIONS, help="KV-cache precision")
    shape.add_argument("--kv-group-size", type=int, help="quantization group size for INT KV")

    gpu = argparse.ArgumentParser(add_help=False)
    gpu.add_argument("gpu", help="GPU key (see `list`) or path to a .json profile")
    gpu.add_argument("--weight-dtype", default="fp16", choices=PRECISIONS, help="weight precision")
    gpu.add_argument("--weight-group-size", type=int, help="quantization group size for INT weights")
    gpu.add_argument("--prefill-chunk", type=int, default=bud.DEFAULT_PREFILL_CHUNK,
                     help="max tokens per forward pass (default %(default)s)")
    gpu.add_argument("--reserved-gib", type=float, default=bud.DEFAULT_RESERVED_BYTES / GIB,
                     help="runtime overhead kept free (default %(default)s GiB)")

    sub.add_parser("list", help="show bundled model and GPU profiles").set_defaults(func=cmd_list)
    sub.add_parser("kv", parents=[shape, out], help="KV-cache size for one configuration"
                   ).set_defaults(func=cmd_kv)
    p = sub.add_parser("attention", parents=[shape, out], help="compare MHA vs GQA vs MQA")
    p.add_argument("--gqa-kv-heads", type=int, help="KV heads for the GQA row")
    p.set_defaults(func=cmd_attention)
    p = sub.add_parser("precision", parents=[shape, out], help="compare FP16/INT8/INT4 KV-cache")
    p.add_argument("--dtypes", default="fp32,fp16,fp8,int8,int4", help="comma-separated precisions")
    p.set_defaults(func=cmd_precision)
    sub.add_parser("budget", parents=[shape, gpu, out], help="full VRAM budget on a GPU"
                   ).set_defaults(func=cmd_budget)
    sub.add_parser("solve", parents=[shape, gpu, out], help="max batch / context before OOM"
                   ).set_defaults(func=cmd_solve)
    p = sub.add_parser("sweep", parents=[shape, gpu, out], help="memory over a grid of batch x context")
    p.add_argument("--contexts", type=_int_list, default=[512, 1024, 2048, 4096, 8192])
    p.add_argument("--batches", type=_int_list, default=[1, 8, 32])
    p.set_defaults(func=cmd_sweep)

    from . import cli_sim  # Part 2: workload, simulate, compare, blocksweep
    cli_sim.register(sub, out=out, gpu=gpu)
    from . import cli_perf  # Part 3: roofline, scaling, profile
    cli_perf.register(sub, out=out, gpu=gpu)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (ConfigError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
