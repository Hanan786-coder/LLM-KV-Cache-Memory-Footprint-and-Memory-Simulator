import tempfile
import unittest
from pathlib import Path

from kvsim.budget import Scenario
from kvsim.config import ConfigError, load_gpu, load_model
from kvsim.kvpool import KVPool, pool_from_scenario
from kvsim.perf import Hardware, PerfModel
from kvsim.profiler import (
    TIMED_REQUEST_COLUMNS, TIMED_STEP_COLUMNS, cost_of_step, load_requests_csv, load_steps_csv,
    profile_result, profile_steps,
)
from kvsim.report import write_csv
from kvsim.simulator import SimConfig, simulate
from kvsim.workload import Request, generate_requests


def setup(gpu="rtx-4090", **hw):
    model, g = load_model("llama-7b"), load_gpu(gpu)
    return model, g, PerfModel(model, Hardware(g, **hw))


class TestSingleRequest(unittest.TestCase):
    """One request alone: its timeline can be checked by hand."""

    def setUp(self):
        self.model, gpu, self.perf = setup()
        pool = pool_from_scenario(Scenario(self.model, gpu, context_len=0))
        self.result = simulate([Request(0, 0, prompt_len=100, output_len=11)], pool,
                               SimConfig(max_model_len=2048))
        self.profile = profile_result(self.perf, self.result)

    def test_steps_match_the_analytic_model(self):
        steps = self.profile.steps
        self.assertEqual(len(steps), 11)  # 1 prefill + 10 decode
        self.assertEqual(steps[0].cost, self.perf.prefill(100))
        for i, s in enumerate(steps[1:]):
            self.assertEqual(s.cost, self.perf.decode(1, 100 + i))

    def test_steps_are_laid_end_to_end(self):
        steps = self.profile.steps
        self.assertEqual(steps[0].start_s, 0.0)
        for a, b in zip(steps, steps[1:]):
            self.assertEqual(a.end_s, b.start_s)
        self.assertAlmostEqual(self.profile.total_s, sum(s.cost.latency_s for s in steps))

    def test_request_latency(self):
        (r,) = self.profile.request_rows()
        steps = self.profile.steps
        self.assertAlmostEqual(r["ttft_ms"], steps[0].cost.latency_s * 1e3)
        self.assertAlmostEqual(r["e2e_ms"], self.profile.total_s * 1e3)
        decode_ms = sum(s.cost.latency_s for s in steps[1:]) * 1e3
        self.assertAlmostEqual(r["tpot_ms"], decode_ms / 10)
        self.assertAlmostEqual(r["decode_tokens_per_s"], 10 / (decode_ms / 1e3))

    def test_summary(self):
        s = self.profile.summary()
        self.assertEqual((s["steps"], s["output_tokens"], s["finished"]), (11, 11, 1))
        self.assertAlmostEqual(s["throughput_tok_s"], 11 / self.profile.total_s)
        self.assertGreater(s["share_time_memory_bound"], 0.8)  # decode dominates
        self.assertAlmostEqual(s["share_bytes_weights"] + s["share_bytes_kv"], 1.0)
        self.assertAlmostEqual(s["share_peak_bandwidth"], 1.0, delta=0.05)  # memory-bound at the roof

    def test_phase_rows_add_up(self):
        rows = self.profile.phase_rows()
        self.assertEqual([r["steps"] for r in rows], [1, 10])
        self.assertAlmostEqual(sum(r["share_time"] for r in rows), 1.0)
        self.assertEqual(rows[1]["share_memory_bound"], 1.0)


class TestRealWorkload(unittest.TestCase):
    def setUp(self):
        self.model, gpu, self.perf = setup()
        self.pool = pool_from_scenario(Scenario(self.model, gpu, context_len=0))
        self.requests = generate_requests(60, preset="chat", max_model_len=2048, seed=3)

    def test_paged_finishes_sooner_than_contiguous(self):
        cfg = SimConfig(max_model_len=2048)
        paged = profile_result(self.perf, simulate(self.requests, self.pool, cfg)).summary()
        contig = profile_result(self.perf, simulate(
            self.requests, self.pool, SimConfig(max_model_len=2048, allocator="contiguous"))).summary()
        self.assertLess(paged["total_time_s"], contig["total_time_s"])
        self.assertGreater(paged["throughput_tok_s"], contig["throughput_tok_s"])

    def test_bigger_batches_amortize_the_weights(self):
        small = profile_result(self.perf, simulate(
            self.requests, self.pool, SimConfig(max_model_len=2048, max_num_seqs=4))).summary()
        big = profile_result(self.perf, simulate(
            self.requests, self.pool, SimConfig(max_model_len=2048))).summary()
        self.assertGreater(big["throughput_tok_s"], 2 * small["throughput_tok_s"])
        self.assertLess(big["share_bytes_weights"], small["share_bytes_weights"])

    def test_swap_time_is_charged(self):
        tiny = KVPool(self.pool.bytes_per_token, 3000)
        result = simulate(self.requests, tiny, SimConfig(max_model_len=2048, preempt_mode="swap"))
        summary = profile_result(self.perf, result).summary()
        self.assertGreater(result.summary()["swap_out_bytes"], 0)
        self.assertGreater(summary["share_time_swap"], 0)

    def test_csv_round_trip_gives_the_same_profile(self):
        result = simulate(self.requests, self.pool, SimConfig(max_model_len=2048))
        live = profile_result(self.perf, result)
        with tempfile.TemporaryDirectory() as d:
            steps, reqs = Path(d) / "steps.csv", Path(d) / "reqs.csv"
            write_csv(result.step_rows(), steps)
            write_csv(result.request_rows(), reqs)
            offline = profile_steps(self.perf, load_steps_csv(steps), load_requests_csv(reqs))
        self.assertEqual(live.summary(), offline.summary())
        self.assertEqual(tuple(offline.step_rows()[0]), TIMED_STEP_COLUMNS)
        self.assertEqual(tuple(offline.request_rows()[0]), TIMED_REQUEST_COLUMNS)

    def test_slower_hardware_takes_longer(self):
        result = simulate(self.requests, self.pool, SimConfig(max_model_len=2048))
        ideal = profile_result(self.perf, result).summary()
        _, _, real = setup(bw_efficiency=0.8, compute_efficiency=0.6, step_overhead_s=1e-3)
        slow = profile_result(real, result).summary()
        self.assertGreater(slow["total_time_s"], ideal["total_time_s"] / 0.8)


class TestInputs(unittest.TestCase):
    def setUp(self):
        _, _, self.perf = setup()

    def test_csv_strings_are_parsed(self):
        row = {"step": "0", "phase": "decode", "batch_size": "2", "prefill_tokens": "0",
               "output_tokens": "2", "kv_read_bytes": "100", "kv_write_bytes": "10",
               "attention_pairs": "4", "swap_out_bytes": "", "swap_in_bytes": "0"}
        cost = cost_of_step(self.perf, row)
        self.assertEqual((cost.tokens, cost.kv_read_bytes, cost.swap_bytes), (2, 100, 0))

    def test_out_of_order_steps_are_rejected(self):
        row = {"phase": "decode", "batch_size": 1, "output_tokens": 1}
        with self.assertRaisesRegex(ConfigError, "increasing"):
            profile_steps(self.perf, [{**row, "step": 5}, {**row, "step": 5}])

    def test_wrong_csv_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.csv"
            path.write_text("id,arrival_step\n1,0\n", encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "missing columns"):
                load_steps_csv(path)

    def test_request_pointing_at_a_missing_step(self):
        row = {"step": 0, "phase": "decode", "batch_size": 1, "output_tokens": 1}
        bad = {"id": 1, "arrival_step": 0, "output_len": 3, "status": "finished",
               "first_token_step": 7, "finish_step": 9}
        with self.assertRaisesRegex(ConfigError, "not in the step trace"):
            profile_steps(self.perf, [row], [bad])

    def test_empty_trace(self):
        summary = profile_steps(self.perf, []).summary()
        self.assertEqual((summary["steps"], summary["total_time_s"]), (0, 0.0))


if __name__ == "__main__":
    unittest.main()
