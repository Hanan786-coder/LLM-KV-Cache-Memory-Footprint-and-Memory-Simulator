import unittest

from kvsim.config import ConfigError, load_gpu, load_model
from kvsim.footprint import kv_bytes_per_token
from kvsim.perf import Hardware, PerfModel
from kvsim.precision import get_precision
from kvsim.weights import num_params, param_breakdown, weight_bytes


class TestHardware(unittest.TestCase):
    def test_ridge_point_is_peak_over_bandwidth(self):
        hw = Hardware(load_gpu("a100-80gb"))
        self.assertAlmostEqual(hw.ridge_point, 312e12 / 2039e9)

    def test_efficiencies_scale_the_peaks(self):
        hw = Hardware(load_gpu("a100-80gb"), bw_efficiency=0.5, compute_efficiency=0.25)
        self.assertAlmostEqual(hw.bandwidth, 2039e9 * 0.5)
        self.assertAlmostEqual(hw.flops, 312e12 * 0.25)

    def test_bad_settings_are_rejected(self):
        gpu = load_gpu("a100-80gb")
        for kw in ({"bw_efficiency": 0}, {"compute_efficiency": 1.5}, {"step_overhead_s": -1}):
            with self.assertRaises(ConfigError):
                Hardware(gpu, **kw)


class TestPublishedFigures(unittest.TestCase):
    """The checks from the work split: known numbers the model must reproduce."""

    def setUp(self):
        self.model = load_model("llama-7b")
        self.gpu = load_gpu("a100-80gb")
        self.perf = PerfModel(self.model, self.gpu)

    def test_a100_decode_speed_is_bandwidth_over_model_size(self):
        step = self.perf.decode(batch=1, context=128)
        expected = self.gpu.bandwidth_bytes_per_s / weight_bytes(self.model)  # ~151 tok/s
        self.assertAlmostEqual(step.tokens_per_s / expected, 1.0, delta=0.03)

    def test_flops_per_token_are_about_twice_the_parameters(self):
        step = self.perf.decode(batch=1, context=128)
        self.assertAlmostEqual(step.flops / (2 * num_params(self.model)), 1.0, delta=0.02)

    def test_prefill_is_compute_bound_and_decode_memory_bound(self):
        self.assertEqual(self.perf.prefill(512).bound, "compute")
        self.assertEqual(self.perf.decode(1, 512).bound, "memory")

    def test_batch_one_fp16_decode_does_about_one_flop_per_byte(self):
        self.assertAlmostEqual(self.perf.decode(1, 0).intensity, 1.0, delta=0.01)

    def test_decode_turns_compute_bound_near_the_ridge_point(self):
        # without KV traffic, intensity ~= batch, so the crossing sits at the ridge (153)
        batch = self.perf.decode_compute_bound_batch(context=0)
        self.assertAlmostEqual(batch, self.perf.hw.ridge_point, delta=5)

    def test_mha_long_context_decode_never_becomes_compute_bound(self):
        # KV reads grow with the batch, so attention keeps intensity near 1 FLOP/byte
        self.assertIsNone(self.perf.decode_compute_bound_batch(context=2048))

    def test_int4_weights_decode_about_three_times_faster(self):
        int4 = PerfModel(self.model, self.gpu, weight_precision=get_precision("int4"))
        speedup = int4.decode(1, 128).tokens_per_s / self.perf.decode(1, 128).tokens_per_s
        self.assertTrue(2.8 < speedup < 3.6, speedup)

    def test_h100_decodes_faster_than_a100(self):
        h100 = PerfModel(self.model, load_gpu("h100-sxm"))
        ratio = h100.decode(1, 128).tokens_per_s / self.perf.decode(1, 128).tokens_per_s
        self.assertAlmostEqual(ratio, 3350 / 2039, delta=0.02)


class TestStepAccounting(unittest.TestCase):
    def setUp(self):
        self.model = load_model("llama-7b")
        self.perf = PerfModel(self.model, load_gpu("rtx-4090"))
        self.bpt = kv_bytes_per_token(self.model)

    def test_decode_reads_the_whole_cache_and_writes_one_entry(self):
        step = self.perf.decode(batch=4, context=100)
        self.assertEqual(step.kv_read_bytes, 4 * 101 * self.bpt)
        self.assertEqual(step.kv_write_bytes, 4 * self.bpt)
        self.assertEqual(step.attention_flops, 4 * 101 * 4 * 32 * 32 * 128)

    def test_weights_are_shared_by_the_batch(self):
        one, many = self.perf.decode(1, 10), self.perf.decode(64, 10)
        params = param_breakdown(self.model)
        row = 2 * self.model.hidden_size
        self.assertEqual(many.weight_bytes - one.weight_bytes, 63 * row)  # only embedding rows
        self.assertLess(many.weight_bytes, weight_bytes(self.model) + 64 * row)
        self.assertGreater(one.weight_bytes, 2 * (params.total - params.embedding))

    def test_prefill_counts_causal_pairs(self):
        step = self.perf.prefill(prompt=10, batch=2)
        self.assertEqual(step.tokens, 20)
        self.assertEqual(step.logits, 2)
        self.assertEqual(step.attention_flops, self.perf.attention_flops(2 * 55))
        self.assertEqual(step.kv_write_bytes, 20 * self.bpt)

    def test_no_lm_head_when_nothing_is_emitted(self):
        with_head = self.perf.step("prefill", 100, 1, 0, 0, 0)
        without = self.perf.step("prefill", 100, 0, 0, 0, 0)
        self.assertEqual(without.lm_head_flops, 0)
        self.assertGreater(with_head.weight_bytes, without.weight_bytes)

    def test_latency_is_the_larger_roof_plus_swap_and_overhead(self):
        perf = PerfModel(self.model, Hardware(load_gpu("rtx-4090"), step_overhead_s=1e-3))
        step = perf.step("decode", 8, 8, 10**8, 10**6, 1000, swap_bytes=32 * 10**9)
        self.assertAlmostEqual(step.t_swap_s, 1.0)  # 32 GB over 32 GB/s PCIe
        self.assertAlmostEqual(step.latency_s, max(step.t_memory_s, step.t_compute_s) + 1.0 + 1e-3)

    def test_empty_step_costs_nothing(self):
        step = self.perf.step("decode", 0, 0, 0, 0, 0)
        self.assertEqual((step.hbm_bytes, step.latency_s, step.intensity), (0, 0.0, 0.0))

    def test_tied_embeddings_read_the_table_once(self):
        tied = PerfModel(self.model.__class__(**{**self.model.to_dict(), "tie_embeddings": True}),
                         load_gpu("rtx-4090"))
        head = 2 * self.model.vocab_size * self.model.hidden_size
        self.assertEqual(tied.decode(4, 0).weight_bytes, tied.weight_read_bytes(4, 0) - 4 * 2 * 4096 + head)

    def test_bad_inputs(self):
        with self.assertRaises(ConfigError):
            self.perf.step("train", 1, 1, 0, 0, 0)
        with self.assertRaises(ValueError):
            self.perf.decode(0, 10)
        with self.assertRaises(ValueError):
            self.perf.step("decode", 1, 1, -1, 0, 0)

    def test_row_has_derived_columns(self):
        row = self.perf.decode(2, 5).row()
        for key in ("hbm_bytes", "flops", "intensity", "bound", "tokens_per_s", "latency_s"):
            self.assertIn(key, row)


if __name__ == "__main__":
    unittest.main()
