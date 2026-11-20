import json
import tempfile
import unittest
from pathlib import Path

from kvsim.config import (
    ConfigError, GPUConfig, ModelConfig, list_gpus, list_models, load_gpu, load_model,
)


class TestProfiles(unittest.TestCase):
    def test_every_bundled_profile_loads(self):
        self.assertIn("llama-7b", list_models())
        self.assertIn("a100-80gb", list_gpus())
        for key in list_models():
            load_model(key)
        for key in list_gpus():
            load_gpu(key)

    def test_lookup_is_case_insensitive(self):
        self.assertEqual(load_model("LLaMA-7B").name, "LLaMA-7B")

    def test_load_from_path(self):
        data = {"name": "tiny", "num_layers": 2, "hidden_size": 64, "num_heads": 4,
                "num_kv_heads": 2, "intermediate_size": 128, "vocab_size": 100}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "tiny.json"
            path.write_text(json.dumps(data))
            model = load_model(str(path))
        self.assertEqual(model.head_dim, 16)
        self.assertEqual(model.attention_type, "GQA")

    def test_unknown_profile_lists_options(self):
        with self.assertRaisesRegex(ConfigError, "llama-7b"):
            load_model("gpt-9000")


class TestModelConfig(unittest.TestCase):
    def make(self, **kw):
        base = dict(name="m", num_layers=4, hidden_size=512, num_heads=8, num_kv_heads=8,
                    intermediate_size=1024, vocab_size=1000)
        base.update(kw)
        return ModelConfig(**base)

    def test_attention_type(self):
        self.assertEqual(self.make().attention_type, "MHA")
        self.assertEqual(self.make(num_kv_heads=2).attention_type, "GQA")
        self.assertEqual(self.make(num_kv_heads=1).attention_type, "MQA")
        self.assertEqual(self.make(num_kv_heads=2).queries_per_kv_head, 4)

    def test_with_kv_heads(self):
        self.assertEqual(self.make().with_kv_heads(1).attention_type, "MQA")

    def test_rejects_bad_values(self):
        with self.assertRaises(ConfigError):
            self.make(num_kv_heads=3)  # 8 heads not divisible by 3
        with self.assertRaises(ConfigError):
            self.make(num_layers=0)
        with self.assertRaises(ConfigError):
            self.make(mlp_type="relu")


class TestGPUConfig(unittest.TestCase):
    def test_units(self):
        gpu = load_gpu("a100-80gb")
        self.assertEqual(gpu.memory_bytes, 80 * 1024**3)
        self.assertAlmostEqual(gpu.bandwidth_bytes_per_s, 2.039e12)
        self.assertAlmostEqual(gpu.ridge_point, 312e12 / 2.039e12)

    def test_rejects_bad_values(self):
        with self.assertRaises(ConfigError):
            GPUConfig(name="x", memory_gib=0, bandwidth_gbs=1, peak_fp16_tflops=1)


if __name__ == "__main__":
    unittest.main()
