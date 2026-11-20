import unittest

from kvsim.config import load_gpu, load_model
from kvsim.perf import PerfModel
from kvsim.scaling import (
    batch_curve, best_batch, context_curve, prefill_compute_bound_prompt, prefill_curve,
)


class TestCurves(unittest.TestCase):
    def setUp(self):
        self.perf = PerfModel(load_model("llama3-8b"), load_gpu("a100-80gb"))

    def test_throughput_rises_with_batch_while_latency_rises_slowly(self):
        rows = batch_curve(self.perf, [1, 8, 64], context=1024)
        tps = [r["tokens_per_s"] for r in rows]
        self.assertEqual(tps, sorted(tps))
        self.assertGreater(tps[1], 7 * tps[0])  # 8x the batch, barely slower steps
        self.assertLess(rows[1]["latency_ms"], 1.1 * rows[0]["latency_ms"])

    def test_weight_bytes_per_token_fall_as_one_over_batch(self):
        one, many = batch_curve(self.perf, [1, 100], context=0)
        self.assertAlmostEqual(one["bytes_weights_per_token"] / many["bytes_weights_per_token"],
                               100, delta=1)

    def test_kv_bytes_per_token_do_not_depend_on_batch(self):
        rows = batch_curve(self.perf, [1, 32], context=2048)
        self.assertEqual(rows[0]["bytes_kv_per_token"], rows[1]["bytes_kv_per_token"])

    def test_huge_batches_do_not_fit(self):
        rows = batch_curve(self.perf, [1, 4096], context=4096)
        self.assertEqual([r["fits"] for r in rows], [True, False])
        self.assertEqual(best_batch(rows)["batch"], 1)
        self.assertIsNone(best_batch([dict(rows[1])]))

    def test_longer_context_means_slower_decode(self):
        rows = context_curve(self.perf, [128, 4096, 32768], batch=16)
        latency = [r["latency_ms"] for r in rows]
        self.assertEqual(latency, sorted(latency))
        self.assertFalse(rows[-1]["fits"])

    def test_short_prompts_are_memory_bound_long_ones_compute_bound(self):
        rows = prefill_curve(self.perf, [1, 4096])
        self.assertEqual([r["bound"] for r in rows], ["memory", "compute"])
        crossing = prefill_compute_bound_prompt(self.perf)
        # a prompt of P tokens does ~P FLOPs per weight byte in FP16, so it crosses near the ridge
        self.assertAlmostEqual(crossing, self.perf.hw.ridge_point, delta=20)
        self.assertEqual(prefill_curve(self.perf, [crossing])[0]["bound"], "compute")
        self.assertEqual(prefill_curve(self.perf, [crossing - 1])[0]["bound"], "memory")

    def test_ttft_grows_with_the_prompt_once_compute_bound(self):
        a, b = prefill_curve(self.perf, [1024, 2048])
        self.assertAlmostEqual(b["ttft_ms"] / a["ttft_ms"], 2.0, delta=0.1)


if __name__ == "__main__":
    unittest.main()
