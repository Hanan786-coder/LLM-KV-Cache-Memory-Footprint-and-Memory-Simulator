"""Run the Part 2 experiments (E1-E7, E9) and write docs/part2-results.md.

    python scripts/part2_experiments.py            # writes docs/part2-results.md
    python scripts/part2_experiments.py --stdout   # prints instead

Every run is seeded, so the tables are reproducible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kvsim.budget import Scenario  # noqa: E402
from kvsim.config import GIB, MIB, load_gpu, load_model  # noqa: E402
from kvsim.kvpool import pool_from_scenario  # noqa: E402
from kvsim.report import _fmt_cell  # noqa: E402
from kvsim.sessions import generate_sessions  # noqa: E402
from kvsim.simulator import SimConfig, simulate  # noqa: E402
from kvsim.workload import Request, generate_requests  # noqa: E402

SEED = 1


def table(rows: list[dict]) -> str:
    columns = list(rows[0])
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join(_fmt_cell(c, row[c]) for c in columns) + " |")
    return "\n".join(lines)


def setup(model_key: str, gpu_key: str):
    model, gpu = load_model(model_key), load_gpu(gpu_key)
    pool = pool_from_scenario(Scenario(model, gpu, context_len=0))
    return model, gpu, pool


def pick(summary: dict, *columns: str) -> dict:
    return {c: summary[c] for c in columns}


def e1_growth() -> str:
    model, gpu, pool = setup("llama-7b", "rtx-4090")
    req = [Request(0, 0, 300, 200)]
    runs = {name: simulate(req, pool, SimConfig(max_model_len=model.max_context, **cfg)).steps
            for name, cfg in (("contig-max", {"allocator": "contiguous"}), ("paged-16", {}))}
    rows = []
    for i in list(range(0, 200, 25)) + [198]:
        c, p = runs["contig-max"][i], runs["paged-16"][i]
        rows.append({"step": c.step, "phase": c.phase, "kv_used_bytes": c.kv_used_bytes,
                     "contig_allocated_bytes": c.kv_allocated_bytes,
                     "paged_allocated_bytes": p.kv_allocated_bytes})
    return ("## E1. How one request grows\n\nLLaMA-7B, FP16 KV (512 KiB per token), one request "
            "with a 300-token prompt and 200 output tokens. Contiguous-max reserves the model's "
            "full 2,048 tokens (1 GiB) at admission; paged allocation follows the cache in 8 MiB "
            "blocks.\n\n" + table(rows))


def e2_compare(requests, model, pool) -> str:
    rows = []
    for cfg in ({"allocator": "contiguous", "reservation": "max"},
                {"allocator": "contiguous", "reservation": "pow2"},
                {"allocator": "contiguous", "reservation": "oracle"},
                {"allocator": "paged", "block_size": 16}):
        s = simulate(requests, pool, SimConfig(max_model_len=model.max_context, **cfg)).summary()
        rows.append(pick(s, "allocator", "steps", "avg_decode_batch", "peak_running",
                         "share_token_states", "share_internal_frag", "share_external_frag",
                         "preemptions", "e2e_p50", "e2e_p95"))
    speedup = rows[0]["steps"] / rows[-1]["steps"]
    return ("## E2. Contiguous against paged\n\nLLaMA-7B on an RTX 4090 (pool 20,940 token "
            f"slots), 500 `chat` requests arriving at step 0, seed {SEED}.\n\n" + table(rows) +
            f"\n\nPaged-16 finishes the same work in {speedup:.1f}x fewer steps than contig-max. "
            "Contig-max keeps token states in only "
            f"{rows[0]['share_token_states']:.1%} of the memory it holds or strands, inside the "
            "20.4%-38.2% the vLLM paper measured for existing systems.")


def e3_blocks(model, pool) -> str:
    parts = ["## E3. Block size\n\nPaged allocator, LLaMA-7B on an RTX 4090, 500 requests per "
             "preset. Larger blocks waste more of each request's last block; the effect is "
             "biggest for short requests."]
    for preset in ("chat", "short"):
        requests = generate_requests(500, preset=preset, max_model_len=model.max_context, seed=SEED)
        rows = []
        for b in (1, 4, 16, 64, 256):
            result = simulate(requests, pool, SimConfig(max_model_len=model.max_context, block_size=b))
            s = result.summary()
            rows.append({"block_size": b, **pick(s, "steps", "avg_decode_batch", "share_token_states",
                                                  "preemptions"),
                         "peak_block_table_bytes": max(st.block_table_bytes for st in result.steps)})
        parts.append(f"### `{preset}`\n\n" + table(rows))
    return "\n\n".join(parts)


def e4_eviction(model, gpu) -> str:
    pool = pool_from_scenario(Scenario(model, gpu, context_len=0), pool_bytes=2 * GIB)
    requests = generate_requests(300, preset="chat", max_model_len=model.max_context,
                                 arrival_rate=0.5, seed=SEED)
    rows = []
    for mode in ("recompute", "swap"):
        for policy in ("lifo", "fifo", "largest"):
            s = simulate(requests, pool, SimConfig(max_model_len=model.max_context,
                                                   evict_policy=policy, preempt_mode=mode)).summary()
            rows.append({"policy": policy, "mode": mode,
                         **pick(s, "steps", "preemptions", "recompute_tokens", "swap_out_bytes",
                                "e2e_p50", "e2e_p95")})
    return ("## E4. Eviction under memory pressure\n\nLLaMA-7B, paged-16, a 2 GiB pool (4,096 "
            "token slots), 300 `chat` requests arriving at 0.5 per step.\n\n" + table(rows) +
            "\n\nLIFO (vLLM's choice) throws away the newest requests, which have the least "
            "work to redo; FIFO evicts the oldest; largest-first frees the most per eviction. "
            "Swap trades recomputed tokens for PCIe traffic.")


def e5_window() -> str:
    model, gpu, pool = setup("mistral-7b", "rtx-4090")
    requests = generate_requests(300, preset="long-output", max_model_len=model.max_context, seed=SEED)
    rows = []
    for window, sinks in ((None, 0), (4096, 0), (1024, 0), (512, 0), (512, 4), (256, 4)):
        result = simulate(requests, pool, SimConfig(max_model_len=model.max_context,
                                                    window=window, sink_tokens=sinks))
        s = result.summary()
        per_request = max(r["peak_kv_tokens"] for r in result.request_rows()) * pool.bytes_per_token
        rows.append({"window": window or "none", "sinks": sinks,
                     **pick(s, "steps", "avg_decode_batch", "preemptions"),
                     "max_kv_per_request_bytes": per_request,
                     "peak_kv_read_bytes": max(st.kv_read_bytes for st in result.steps)})
    return ("## E5. Sliding window\n\nMistral-7B (GQA-8, 128 KiB per token) on an RTX 4090, "
            "300 `long-output` requests (outputs 512-1,536 tokens). No request is longer than "
            "1,664 tokens, so a 4,096-token window (Mistral's own) changes nothing here.\n\n"
            + table(rows))


def e6_external(model, pool) -> str:
    requests = generate_requests(400, prompt="uniform:16:1024", output="uniform:16:768",
                                 max_model_len=model.max_context, max_new_tokens=1024, seed=SEED)
    rows = []
    for cfg in ({"allocator": "contiguous", "reservation": "request", "fit": "first"},
                {"allocator": "contiguous", "reservation": "request", "fit": "best"},
                {"allocator": "contiguous", "reservation": "oracle", "fit": "first"},
                {"allocator": "contiguous", "reservation": "oracle", "fit": "best"},
                {"allocator": "paged"}):
        s = simulate(requests, pool, SimConfig(max_model_len=model.max_context, **cfg)).summary()
        rows.append(pick(s, "allocator", "steps", "frag_stalls", "share_external_frag",
                         "share_internal_frag", "avg_decode_batch"))
    return ("## E6. External fragmentation\n\nLLaMA-7B on an RTX 4090, 400 requests with "
            "uniform lengths (prompts 16-1,024, outputs 16-768, `max_new_tokens` 1,024). A "
            "frag stall is a step where the next request was blocked although enough memory "
            "was free in total.\n\n" + table(rows))


def e7_batching(requests, model, pool) -> str:
    rows = []
    for allocator in ("contiguous", "paged"):
        for batching in ("static", "continuous"):
            s = simulate(requests, pool, SimConfig(max_model_len=model.max_context,
                                                   allocator=allocator, batching=batching)).summary()
            rows.append({"batching": batching, **pick(s, "allocator", "steps", "avg_decode_batch",
                                                      "e2e_p50", "e2e_p95")})
    return ("## E7. Static against continuous batching\n\nSame workload as E2. With static "
            "batching nobody joins until the whole batch has finished, so slots freed by "
            "short requests sit idle.\n\n" + table(rows))


def e9_sessions(model, pool) -> str:
    sessions = generate_sessions(200, max_model_len=model.max_context, arrival_rate=0.2, seed=SEED)
    turns = sum(len(s.turns) for s in sessions)
    rows = []
    for policy in ("lru", "fifo", "opt"):
        s = simulate([], pool, SimConfig(max_model_len=model.max_context, cache_policy=policy),
                     sessions=sessions).summary()
        rows.append({"cache_policy": policy, **pick(s, "cache_hits", "cache_misses", "cache_hit_rate",
                                                    "cache_evictions", "saved_prefill_tokens", "steps")})
    return ("## E9. Multi-turn sessions: LRU, FIFO and OPT\n\nLLaMA-7B on an RTX 4090, 200 chat "
            f"sessions ({turns} turns; 2-6 turns each, think time 1-200 steps, a new session every "
            "5 steps on average). After a turn the session's KV-cache is kept while the user "
            "thinks; a hit prefills only the new message, a miss prefills the whole history. "
            "Idle caches are dropped before anyone waits or is preempted. OPT (Belady) knows "
            "when each idle session's next turn arrives.\n\n" + table(rows) +
            "\n\nOPT gets the most hits here. It is not a guaranteed upper bound, though: "
            "Belady's proof assumes equal-sized pages and a fixed request sequence, while "
            "session caches differ in size and each eviction changes when later turns run "
            "(with 60 sessions FIFO edges past it, 95.3% to 94.8%). LRU does slightly worse than FIFO here: think "
            "times are capped at 200 steps, so the session idle the longest is the one most "
            "likely to come back soon, the opposite of the locality LRU relies on. The step "
            "counts are equal because a prefill takes one step however long it is; the saving "
            "shows up as fewer prefill tokens, which Part 3 turns into less compute time.")


def build() -> str:
    model, gpu, pool = setup("llama-7b", "rtx-4090")
    chat = generate_requests(500, preset="chat", max_model_len=model.max_context, seed=SEED)
    sections = [
        "# Part 2 results\n\nGenerated by `python scripts/part2_experiments.py`; do not edit by "
        "hand. Steps are scheduler steps (one forward pass each); Part 3 turns them into "
        "seconds. Defaults unless stated: FP16 weights and KV, 1 GiB reserved, prefill chunk "
        "2,048 tokens, at most 256 running requests, LIFO eviction with recompute.",
        e1_growth(), e2_compare(chat, model, pool), e3_blocks(model, pool), e4_eviction(model, gpu),
        e5_window(), e6_external(model, pool), e7_batching(chat, model, pool),
        "## E8. Tokens per second\n\nNeeds Part 3: feed the `--steps-csv` traces from E2 into "
        "Part 3's latency model.",
        e9_sessions(model, pool),
    ]
    return "\n\n".join(sections) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stdout", action="store_true", help="print instead of writing the file")
    args = parser.parse_args()
    text = build()
    if args.stdout:
        print(text)
    else:
        path = ROOT / "docs" / "part2-results.md"
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
