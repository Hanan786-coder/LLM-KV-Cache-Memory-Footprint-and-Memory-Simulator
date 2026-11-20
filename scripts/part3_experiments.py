"""Run the bandwidth/roofline experiments (P1-P8) and write docs/part3-results.md.

    python scripts/part3_experiments.py            # writes the results and docs/figures/*.svg
    python scripts/part3_experiments.py --stdout   # prints the results instead (no figures)

Every simulated workload is seeded, so the tables are reproducible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kvsim.budget import Scenario  # noqa: E402
from kvsim.config import GIB, load_gpu, load_model, list_gpus  # noqa: E402
from kvsim.kvpool import pool_from_scenario  # noqa: E402
from kvsim.perf import Hardware, PerfModel  # noqa: E402
from kvsim.plot import line_chart, roofline_chart, save_svg  # noqa: E402
from kvsim.precision import get_precision  # noqa: E402
from kvsim.profiler import profile_result  # noqa: E402
from kvsim.report import _fmt_cell  # noqa: E402
from kvsim.scaling import (  # noqa: E402
    batch_curve, context_curve, prefill_compute_bound_prompt, prefill_curve,
)
from kvsim.simulator import SimConfig, simulate  # noqa: E402
from kvsim.weights import num_params, weight_bytes  # noqa: E402
from kvsim.workload import generate_requests  # noqa: E402

SEED = 1
FIGURES: dict[str, str] = {}  # file name -> SVG text, written by main()


def table(rows: list[dict]) -> str:
    columns = list(rows[0])
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join(_fmt_cell(c, row[c]) for c in columns) + " |")
    return "\n".join(lines)


def figure(name: str, svg: str, caption: str) -> str:
    FIGURES[name] = svg
    return f"![{caption}](figures/{name})"


def perf(model: str, gpu: str, **kw) -> PerfModel:
    return PerfModel(load_model(model), load_gpu(gpu), **kw)


def p1_validation() -> str:
    model = load_model("llama-7b")
    rows = []
    for key in list_gpus():
        gpu = load_gpu(key)
        step = PerfModel(model, gpu).decode(1, 128)
        expected = gpu.bandwidth_bytes_per_s / weight_bytes(model)
        rows.append({"gpu": gpu.name, "bandwidth_gbs": gpu.bandwidth_gbs,
                     "bandwidth_over_weights_tok_s": expected, "model_tok_s": step.tokens_per_s,
                     "ratio": step.tokens_per_s / expected, "flops_over_2x_params":
                     step.flops / (2 * num_params(model)), "intensity": step.intensity})
    ridges = [load_gpu(k).ridge_point for k in list_gpus()]
    return ("## P1. Checks against known numbers\n\nLLaMA-7B in FP16 (12.55 GiB of weights), one "
            "sequence, 128 tokens of context. The usual rule of thumb is that batch-1 decode "
            "runs at memory bandwidth / model size, and that a token costs about 2 FLOPs per "
            "parameter.\n\n" + table(rows) + "\n\nThe model lands within 2% of the rule of thumb "
            "on every GPU (the small gap is the KV-cache read and the embedding table, which "
            "is not streamed). One FLOP per byte is far below every ridge point "
            f"({min(ridges):.0f}-{max(ridges):.0f} FLOPs/byte), so batch-1 decode uses under 1% "
            "of the GPU's compute.")


def p2_roofline() -> str:
    p = perf("llama-7b", "a100-80gb")
    rows, points = [], []
    for label, cost in ([(f"decode b={b}", p.decode(b, 1024)) for b in (1, 16, 128)]
                        + [(f"prefill {n}", p.prefill(n)) for n in (64, 512, 2048)]):
        rows.append({"step": label, "hbm_bytes": cost.hbm_bytes, "flops": cost.flops,
                     "intensity": cost.intensity, "t_memory_ms": cost.t_memory_s * 1e3,
                     "t_compute_ms": cost.t_compute_s * 1e3, "latency_ms": cost.latency_s * 1e3,
                     "bound": cost.bound})
    for b in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        c = p.decode(b, 1024)
        points.append(("decode, 1k context", c.intensity, c.flops / c.latency_s))
        c = p.decode(b, 0)
        points.append(("decode, no context", c.intensity, c.flops / c.latency_s))
    for n in (16, 64, 256, 1024, 4096):
        c = p.prefill(n)
        points.append(("prefill", c.intensity, c.flops / c.latency_s))
    fig = figure("roofline-llama7b-a100.svg",
                 roofline_chart(p.hw.flops, p.hw.bandwidth, points, "LLaMA-7B on A100 80GB"),
                 "Roofline of LLaMA-7B on A100 80GB")
    return ("## P2. Prefill against decode on the roofline\n\nLLaMA-7B, FP16, A100 80GB (ridge "
            "point 153 FLOPs/byte).\n\n" + table(rows) + "\n\n" + fig + "\n\nPrefill reuses "
            "every weight for each prompt token, so a prompt of a few hundred tokens or more "
            "sits right of the ridge and is compute-bound (a 64-token prompt is still "
            "memory-bound). Decode does one token per sequence, so without KV traffic its "
            "intensity would equal the batch size. With 1,024 tokens of context, MHA decode never reaches the ridge: "
            "each sequence's KV read (512 MiB at 1k tokens) grows with the batch, so attention "
            "keeps the step near 1 FLOP per KV byte.")


def p3_batch() -> str:
    parts, series = [], []
    for gpu in ("a100-80gb", "h100-sxm"):
        p = perf("llama3-8b", gpu)
        rows = batch_curve(p, (1, 4, 16, 64, 256, 512), context=1024)
        series.append((p.gpu.name, [r["batch"] for r in rows], [r["tokens_per_s"] for r in rows]))
        if gpu == "a100-80gb":
            parts.append(table([{k: r[k] for k in ("batch", "latency_ms", "tokens_per_s",
                                                   "tokens_per_s_per_seq", "bytes_weights_per_token",
                                                   "bytes_kv_per_token", "intensity", "bound", "fits")}
                                for r in rows]))
    fig = figure("batch-llama3-8b.svg", line_chart(series, "LLaMA-3-8B decode at 1k context",
                                                   "batch size", "tokens/s (all sequences)",
                                                   logx=True, logy=True), "Throughput by batch")
    return ("## P3. Bigger batches reuse the weights\n\nLLaMA-3-8B (GQA, 128 KiB of KV per "
            "token), FP16, 1,024 tokens of context, A100 80GB.\n\n" + parts[0] + "\n\n" + fig +
            "\n\nWeights are read once per step, so the bytes spent on weights per token fall "
            "as 1/batch while the KV bytes per token stay the same. Going from batch 1 to 64 "
            "makes each step only 1.6x slower but gives 41x the tokens. Past that, KV reads "
            "take over and the gains flatten; batch 512 no longer fits in 80 GB.")


def p4_context() -> str:
    series, rows = [], []
    for key in ("llama-7b", "llama3-8b"):
        p = perf(key, "a100-80gb")
        curve = context_curve(p, (128, 512, 2048, 8192, 32768), batch=16)
        series.append((f"{p.model.name} ({p.model.attention_type})",
                       [r["context"] for r in curve], [r["latency_ms"] for r in curve]))
        rows += [{"model": p.model.name, **{k: r[k] for k in ("context", "latency_ms", "tokens_per_s",
                                                                "bytes_kv_per_token", "fits")}}
                 for r in curve]
    fig = figure("context-mha-gqa.svg", line_chart(series, "Decode step at batch 16 on A100 80GB",
                                                   "context length (tokens)", "step latency (ms)",
                                                   logx=True, logy=True), "Latency by context")
    return ("## P4. Long contexts: MHA against GQA\n\nDecode at batch 16 on an A100 80GB.\n\n" +
            table(rows) + "\n\n" + fig + "\n\nThe cache read grows linearly with context. "
            "LLaMA-7B's MHA cache is 4x larger per token than LLaMA-3-8B's GQA-8 cache, so "
            "LLaMA-7B at 2,048 tokens is about as slow as LLaMA-3-8B at 8,192. Neither fits 16 "
            "sequences of 32k tokens in 80 GB. (LLaMA-7B was trained for 2,048 tokens; its "
            "longer rows only show the trend.)")


def p5_prefill() -> str:
    rows, series = [], []
    for key in ("rtx-4090", "a100-80gb", "h100-sxm"):
        p = perf("llama-7b", key)
        curve = prefill_curve(p, (16, 64, 256, 1024, 4096))
        series.append((p.gpu.name, [r["prompt"] for r in curve], [r["ttft_ms"] for r in curve]))
        by = {r["prompt"]: r["ttft_ms"] for r in curve}
        rows.append({"gpu": p.gpu.name, "ridge_flops_per_byte": p.hw.ridge_point,
                     "compute_bound_from_prompt": prefill_compute_bound_prompt(p),
                     "ttft_64_ms": by[64], "ttft_1024_ms": by[1024], "ttft_4096_ms": by[4096]})
    fig = figure("ttft-llama7b.svg", line_chart(series, "LLaMA-7B time to first token",
                                                "prompt length (tokens)", "TTFT (ms)",
                                                logx=True, logy=True), "TTFT by prompt length")
    return ("## P5. Time to first token\n\nOne LLaMA-7B prompt, FP16, prefilled in one pass.\n\n"
            + table(rows) + "\n\n" + fig + "\n\nShort prompts cost the same as one decode step "
            "(the weights have to be read anyway). Once the prompt is longer than about the "
            "ridge point, prefill is compute-bound and TTFT grows linearly, then a little faster "
            "as causal attention adds P^2 work.")


def p6_precision() -> str:
    rows = []
    for w, kv in (("fp16", "fp16"), ("int8", "fp16"), ("int4", "fp16"), ("int4", "int8"), ("int4", "int4")):
        p = perf("llama-7b", "rtx-4090", weight_precision=get_precision(w), kv_precision=get_precision(kv))
        one, many = p.decode(1, 1024), p.decode(32, 1024)
        rows.append({"weights": w, "kv": kv, "batch1_tok_s": one.tokens_per_s,
                     "batch32_tok_s": many.tokens_per_s,
                     "share_kv_traffic_batch32": (many.kv_read_bytes + many.kv_write_bytes) / many.hbm_bytes,
                     "batch32_fits": [r["fits"] for r in batch_curve(p, (32,), 1024)][0]})
    return ("## P6. Quantization speeds up decode\n\nLLaMA-7B on an RTX 4090 at 1,024 tokens of "
            "context. Decode is memory-bound, so fewer bytes means proportionally more speed.\n\n"
            + table(rows) + "\n\nINT4 weights make batch-1 decode "
            f"{rows[2]['batch1_tok_s'] / rows[0]['batch1_tok_s']:.1f}x faster (not 4x: the group "
            "scales, the FP16 embeddings and LM head, and the FP16 KV-cache still have to be "
            "read). At batch 32 most of the traffic is the KV-cache, so INT4 weights alone give "
            f"only {rows[2]['batch32_tok_s'] / rows[0]['batch32_tok_s']:.1f}x; adding an INT4 "
            f"KV-cache gives {rows[4]['batch32_tok_s'] / rows[2]['batch32_tok_s']:.1f}x more. In "
            "FP16, batch 32 at 1k tokens does not even fit in 24 GB.")


def p7_workload() -> str:
    model, gpu = load_model("llama-7b"), load_gpu("rtx-4090")
    pool = pool_from_scenario(Scenario(model, gpu, context_len=0))
    requests = generate_requests(500, preset="chat", max_model_len=model.max_context, seed=SEED)
    p = PerfModel(model, gpu)
    rows, points = [], []
    for label, cfg in (("contig-max", {"allocator": "contiguous", "reservation": "max"}),
                       ("contig-oracle", {"allocator": "contiguous", "reservation": "oracle"}),
                       ("paged-16", {"allocator": "paged", "block_size": 16})):
        result = simulate(requests, pool, SimConfig(max_model_len=model.max_context, **cfg))
        prof = profile_result(p, result)
        s = prof.summary()
        rows.append({"allocator": label, "steps": s["steps"], "total_time_s": s["total_time_s"],
                     "throughput_tok_s": s["throughput_tok_s"],
                     "decode_step_p50_ms": s["decode_step_p50_ms"],
                     "share_time_memory_bound": s["share_time_memory_bound"],
                     "ttft_p50_ms": s["ttft_p50_ms"], "tpot_p50_ms": s["tpot_p50_ms"],
                     "e2e_p95_ms": s["e2e_p95_ms"]})
        if label == "paged-16":
            points = [(st.cost.phase, st.cost.intensity, st.cost.flops / st.cost.latency_s)
                      for st in prof.steps[::5] if st.cost.latency_s]
    fig = figure("profile-paged-4090.svg",
                 roofline_chart(p.hw.flops, p.hw.bandwidth, points,
                                "500 chat requests, paged-16, LLaMA-7B on RTX 4090"),
                 "Every fifth simulated step on the roofline")
    speed = rows[-1]["throughput_tok_s"] / rows[0]["throughput_tok_s"]
    fewer = rows[0]["steps"] / rows[-1]["steps"]
    return ("## P7. A served workload in real time\n\nThe Part 2 workload (500 `chat` requests "
            f"at step 0, seed {SEED}) on LLaMA-7B / RTX 4090, with every step timed by the "
            "roofline model.\n\n" + table(rows) + "\n\n" + fig + f"\n\nPaged allocation needs "
            f"{fewer:.1f}x fewer steps than contig-max but gives {speed:.1f}x the throughput, "
            "not the same factor: its decode steps run bigger batches, which read more KV and "
            "take longer. Each request therefore waits longer per output token "
            f"({rows[-1]['tpot_p50_ms']:.0f} against {rows[0]['tpot_p50_ms']:.0f} ms), but far "
            "less for its first token, and the whole workload finishes much sooner. Step counts "
            "alone overstate the gain; timing the steps gives the real one. Almost all of the "
            "time is memory-bound decode.")


def p8_efficiency() -> str:
    model, gpu = load_model("llama-7b"), load_gpu("rtx-4090")
    pool = pool_from_scenario(Scenario(model, gpu, context_len=0))
    requests = generate_requests(200, preset="chat", max_model_len=model.max_context, seed=SEED)
    result = simulate(requests, pool, SimConfig(max_model_len=model.max_context))
    rows = []
    for label, hw in (("ideal roofline", Hardware(gpu)),
                      ("80% bandwidth, 60% compute", Hardware(gpu, 0.8, 0.6)),
                      ("same + 2 ms per step", Hardware(gpu, 0.8, 0.6, 2e-3))):
        s = profile_result(PerfModel(model, hw), result).summary()
        rows.append({"hardware": label, "total_time_s": s["total_time_s"],
                     "throughput_tok_s": s["throughput_tok_s"], "tpot_p50_ms": s["tpot_p50_ms"],
                     "ttft_p50_ms": s["ttft_p50_ms"]})
    return ("## P8. Ideal against realistic hardware\n\nThe default model is the ideal roofline: "
            "an upper bound on speed. Real kernels reach about 70-90% of peak bandwidth and "
            "50-70% of peak FLOP/s, and every step pays a few ms of launch and scheduling "
            "overhead. LLaMA-7B / RTX 4090, 200 `chat` requests, paged-16.\n\n" + table(rows) +
            "\n\nUse `--bw-efficiency`, `--compute-efficiency` and `--step-overhead-us` to "
            "calibrate against a measured system.")


def build() -> str:
    sections = [p1_validation(), p2_roofline(), p3_batch(), p4_context(), p5_prefill(),
                p6_precision(), p7_workload(), p8_efficiency()]
    head = ("# Bandwidth, latency and roofline results\n\nGenerated by "
            "`python scripts/part3_experiments.py`. All numbers come from the analytic roofline "
            "model in `kvsim/perf.py` (ideal hardware unless stated) and, for P7-P8, from "
            "seeded Part 2 simulations. The model is described in "
            "[part3-bandwidth-roofline.md](part3-bandwidth-roofline.md).")
    return "\n\n".join([head, *sections]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stdout", action="store_true", help="print instead of writing files")
    args = parser.parse_args()
    text = build()
    if args.stdout:
        print(text)
        return
    docs = ROOT / "docs"
    (docs / "figures").mkdir(exist_ok=True)
    for name, svg in FIGURES.items():
        save_svg(svg, docs / "figures" / name)
    (docs / "part3-results.md").write_text(text, encoding="utf-8")
    print(f"wrote {docs / 'part3-results.md'} and {len(FIGURES)} figures")


if __name__ == "__main__":
    main()
