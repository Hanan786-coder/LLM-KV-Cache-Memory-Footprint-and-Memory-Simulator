import unittest

from kvsim.budget import Scenario, kv_token_capacity
from kvsim.config import GIB, MIB, ConfigError, load_gpu, load_model
from kvsim.kvpool import KVPool, kv_pool_bytes, pool_from_scenario
from kvsim.precision import INT4


class TestKVPool(unittest.TestCase):
    def setUp(self):
        self.s = Scenario(load_model("llama-7b"), load_gpu("rtx-4090"), context_len=0)

    def test_llama7b_on_rtx4090(self):
        pool = pool_from_scenario(self.s)
        self.assertEqual(pool.bytes_per_token, 512 * 1024)
        self.assertEqual(pool.total_tokens, 20_940)
        self.assertEqual(pool.num_blocks(16), 1_308)
        self.assertEqual(pool.block_bytes(16), 8 * MIB)

    def test_reserves_a_full_prefill_chunk_unlike_part1(self):
        # Part 1 keeps activations for one token; we keep them for 2048 (about 229 MiB more)
        pool = pool_from_scenario(self.s)
        self.assertEqual(kv_token_capacity(self.s) - pool.total_tokens, 458)

    def test_more_running_sequences_leave_less_room(self):
        self.assertGreater(kv_pool_bytes(self.s, max_num_seqs=1), kv_pool_bytes(self.s, max_num_seqs=256))

    def test_overrides(self):
        self.assertEqual(pool_from_scenario(self.s, pool_tokens=100).total_tokens, 100)
        self.assertEqual(pool_from_scenario(self.s, pool_bytes=1 * GIB).total_tokens, 2048)
        with self.assertRaises(ConfigError):
            pool_from_scenario(self.s, pool_tokens=100, pool_bytes=GIB)
        with self.assertRaises(ConfigError):
            pool_from_scenario(self.s, pool_tokens=0)

    def test_weights_that_do_not_fit(self):
        s = Scenario(load_model("llama2-70b"), load_gpu("a100-80gb"), context_len=0)
        with self.assertRaisesRegex(ConfigError, "int8/int4"):
            pool_from_scenario(s)
        self.assertEqual(pool_from_scenario(s.with_(weight_precision=INT4)).bytes_per_token, 320 * 1024)

    def test_validation(self):
        with self.assertRaises(ConfigError):
            KVPool(0, 10)
        with self.assertRaises(ConfigError):
            KVPool(1, -1)
        with self.assertRaises(ConfigError):
            KVPool(1, 10).num_blocks(0)
        self.assertEqual(KVPool(2, 30).num_blocks(4), 7)  # 2 leftover slots are unusable
        self.assertEqual(KVPool(2, 30).total_bytes, 60)


if __name__ == "__main__":
    unittest.main()
