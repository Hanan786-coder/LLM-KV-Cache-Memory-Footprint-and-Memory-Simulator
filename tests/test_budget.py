import unittest

from kvsim.budget import (
    Scenario, activation_bytes, fits, kv_token_capacity, max_batch, max_context,
    memory_budget, oom_report, tokens_in_flight,
)
from kvsim.config import load_gpu, load_model
from kvsim.footprint import kv_bytes_per_token
from kvsim.precision import FP16, INT4, INT8
from kvsim.weights import num_params, param_breakdown, weight_bytes

GIB = 1024**3


class TestWeights(unittest.TestCase):
    def assertClose(self, actual, expected, rel=0.005):
        self.assertLess(abs(actual - expected) / expected, rel, f"{actual:,} vs {expected:,}")

    def test_param_counts_match_published_sizes(self):
        self.assertClose(num_params(load_model("llama-7b")), 6.738e9)
        self.assertClose(num_params(load_model("llama-13b")), 13.016e9)
        self.assertClose(num_params(load_model("llama2-70b")), 68.977e9)
        self.assertClose(num_params(load_model("llama3-8b")), 8.030e9)
        self.assertClose(num_params(load_model("mistral-7b")), 7.242e9)

    def test_fp16_weights_are_two_bytes_per_param(self):
        model = load_model("llama-7b")
        self.assertEqual(weight_bytes(model, FP16), 2 * num_params(model))

    def test_int4_keeps_embeddings_in_fp16(self):
        model = load_model("llama-7b")
        p = param_breakdown(model)
        approx = p.linear * INT4.bytes_per_value + 2 * p.other
        self.assertClose(weight_bytes(model, INT4), approx, rel=1e-6)
        self.assertLess(weight_bytes(model, INT4, keep_embeddings_fp16=False), weight_bytes(model, INT4))


class TestBudget(unittest.TestCase):
    def setUp(self):
        self.s = Scenario(load_model("llama-7b"), load_gpu("rtx-4090"), context_len=2048, batch=1)

    def test_components_add_up(self):
        b = memory_budget(self.s)
        self.assertEqual(b.kv_cache, 1 * GIB)  # 2048 tokens x 0.5 MiB
        self.assertEqual(b.total, b.weights + b.kv_cache + b.activations + b.reserved)
        self.assertEqual(b.capacity, 24 * GIB)
        self.assertTrue(b.fits)

    def test_prefill_is_chunked(self):
        self.assertEqual(tokens_in_flight(self.s), 2048)
        self.assertEqual(tokens_in_flight(self.s.with_(batch=8)), 2048)
        self.assertEqual(tokens_in_flight(self.s.with_(batch=4096, prefill_chunk=512)), 4096)
        self.assertGreater(activation_bytes(self.s.with_(batch=8)), activation_bytes(self.s))

    def test_70b_fp16_does_not_fit_on_one_a100(self):
        s = Scenario(load_model("llama2-70b"), load_gpu("a100-80gb"), context_len=4096)
        self.assertFalse(fits(s))
        self.assertEqual(max_batch(s), 0)
        self.assertFalse(oom_report(s)["weights_fit"])
        self.assertTrue(fits(s.with_(weight_precision=INT4)))


class TestSolvers(unittest.TestCase):
    def setUp(self):
        self.s = Scenario(load_model("llama-7b"), load_gpu("a100-80gb"), context_len=4096, batch=1)

    def test_max_batch_is_the_oom_boundary(self):
        b = max_batch(self.s)
        self.assertGreater(b, 1)
        self.assertTrue(fits(self.s.with_(batch=b)))
        self.assertFalse(fits(self.s.with_(batch=b + 1)))

    def test_max_context_is_the_oom_boundary(self):
        s = self.s.with_(batch=16)
        c = max_context(s)
        self.assertTrue(fits(s.with_(context_len=c)))
        self.assertFalse(fits(s.with_(context_len=c + 1)))

    def test_cheaper_kv_allows_bigger_batches(self):
        fp16 = max_batch(self.s)
        int8 = max_batch(self.s.with_(kv_precision=INT8))
        int4 = max_batch(self.s.with_(kv_precision=INT4))
        self.assertLess(fp16, int8)
        self.assertLess(int8, int4)
        gqa = max_batch(self.s.with_(model=self.s.model.with_kv_heads(8)))
        self.assertGreater(gqa, 3 * fp16)

    def test_limit_caps_the_search(self):
        tiny = self.s.with_(context_len=1)
        self.assertEqual(max_batch(tiny, limit=100), 100)

    def test_kv_token_capacity(self):
        tokens = kv_token_capacity(self.s)
        per_token = kv_bytes_per_token(self.s.model, FP16)
        b = memory_budget(self.s.with_(context_len=0))
        self.assertEqual(tokens, b.free // per_token)


if __name__ == "__main__":
    unittest.main()
