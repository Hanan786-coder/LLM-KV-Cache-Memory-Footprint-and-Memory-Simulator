import unittest

from kvsim.config import ConfigError, load_model
from kvsim.footprint import (
    compare_attention, compare_precision, default_gqa_kv_heads, kv_bytes_per_token,
    kv_bytes_per_token_per_layer, kv_cache_bytes,
)
from kvsim.precision import FP16, FP32, INT4, INT8, get_precision

MIB, GIB = 1024**2, 1024**3


class TestPrecision(unittest.TestCase):
    def test_plain_formats_have_no_metadata(self):
        self.assertEqual(FP16.bytes_for(100), 200)
        self.assertEqual(FP32.bytes_per_value, 4)

    def test_int4_metadata(self):
        # 65 values: ceil(65 * 4 / 8) = 33 data bytes + 2 groups x 4 bytes
        self.assertEqual(INT4.bytes_for(65), 41)
        self.assertAlmostEqual(INT4.bytes_per_value, 0.5625)
        self.assertAlmostEqual(INT4.effective_bits, 4.5)

    def test_group_size_override(self):
        p = get_precision("int4", group_size=128)
        self.assertEqual(p.group_size, 128)
        self.assertAlmostEqual(p.bytes_per_value, 0.5 + 4 / 128)
        self.assertEqual(get_precision("fp16", group_size=32), FP16)  # ignored for float

    def test_unknown(self):
        with self.assertRaises(ConfigError):
            get_precision("int3")


class TestKVFormula(unittest.TestCase):
    def setUp(self):
        self.llama7b = load_model("llama-7b")

    def test_llama7b_published_figure(self):
        # 2 x 32 layers x 32 heads x 128 x 2 bytes = 0.5 MiB per token
        self.assertEqual(kv_bytes_per_token(self.llama7b, FP16), 512 * 1024)
        self.assertEqual(kv_bytes_per_token_per_layer(self.llama7b, FP16), 16 * 1024)
        self.assertEqual(kv_cache_bytes(self.llama7b, FP16, 4096, 1), 2 * GIB)

    def test_linear_in_batch_and_context(self):
        one = kv_cache_bytes(self.llama7b, FP16, 1000, 1)
        self.assertEqual(kv_cache_bytes(self.llama7b, FP16, 1000, 8), 8 * one)
        self.assertEqual(kv_cache_bytes(self.llama7b, FP16, 3000, 1), 3 * one)
        self.assertEqual(kv_cache_bytes(self.llama7b, FP16, 0, 4), 0)

    def test_llama2_70b_gqa(self):
        # 2 x 80 x 8 KV heads x 128 x 2 = 320 KiB per token
        self.assertEqual(kv_bytes_per_token(load_model("llama2-70b"), FP16), 320 * 1024)

    def test_quantized_kv(self):
        # per head vector: 64 data bytes + 2 groups x 4 = 72 bytes vs 256 in FP16
        self.assertEqual(kv_bytes_per_token(self.llama7b, INT4), 2 * 32 * 32 * 72)
        self.assertEqual(kv_bytes_per_token(self.llama7b, INT8), 2 * 32 * 32 * 130)


class TestComparisons(unittest.TestCase):
    def test_attention_variants(self):
        rows = {r["attention"]: r for r in compare_attention(load_model("llama-7b"))}
        self.assertEqual(rows["MHA"]["reduction_vs_mha"], 1)
        self.assertEqual(rows["GQA"]["kv_heads"], 8)
        self.assertEqual(rows["GQA"]["reduction_vs_mha"], 4)
        self.assertEqual(rows["MQA"]["reduction_vs_mha"], 32)
        self.assertTrue(rows["MHA"]["is_model_default"])

    def test_gqa_model_uses_its_own_grouping(self):
        rows = {r["attention"]: r for r in compare_attention(load_model("llama2-70b"))}
        self.assertEqual(rows["GQA"]["kv_heads"], 8)
        self.assertTrue(rows["GQA"]["is_model_default"])
        self.assertEqual(rows["GQA"]["reduction_vs_mha"], 8)

    def test_invalid_gqa_grouping(self):
        with self.assertRaises(ConfigError):
            compare_attention(load_model("llama-7b"), gqa_kv_heads=3)

    def test_default_gqa_kv_heads(self):
        self.assertEqual(default_gqa_kv_heads(32), 8)
        self.assertEqual(default_gqa_kv_heads(40), 10)
        self.assertEqual(default_gqa_kv_heads(4), 1)

    def test_int4_is_not_exactly_4x(self):
        rows = {r["precision"]: r for r in compare_precision(load_model("llama-7b"))}
        self.assertEqual(rows["fp16"]["reduction_vs_fp16"], 1)
        self.assertEqual(rows["fp32"]["reduction_vs_fp16"], 0.5)
        self.assertAlmostEqual(rows["int4"]["reduction_vs_fp16"], 256 / 72)
        self.assertLess(rows["int4"]["reduction_vs_fp16"], rows["int4"]["ideal_reduction_vs_fp16"])
        self.assertAlmostEqual(rows["int4"]["effective_bits"], 4.5)


if __name__ == "__main__":
    unittest.main()
