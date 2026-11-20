import random
import unittest

from kvsim.allocator import ContiguousAllocator, PagedAllocator
from kvsim.config import ConfigError
from kvsim.kvpool import KVPool
from kvsim.simulator import SimConfig, Status, simulate
from kvsim.workload import Request, generate_requests


def run(requests, pool_tokens, **config):
    config = {"check_invariants": True, "watermark": 0.0, "max_model_len": 256, **config}
    return simulate(requests, KVPool(1, pool_tokens), SimConfig(**config))


class TestPagedWindow(unittest.TestCase):
    def test_cache_stops_growing_at_the_window(self):
        result = run([Request(0, 0, 4, 30)], 64, block_size=4, window=8)
        rows = result.step_rows()
        self.assertEqual(max(r["kv_tokens"] for r in rows), 8)
        self.assertLessEqual(max(r["blocks_used"] for r in rows), 2 + 1)  # ceil(W/B) + 1
        self.assertEqual(max(r["kv_read_bytes"] for r in rows), 8)  # reads level off too
        self.assertEqual(sum(r["output_tokens"] for r in rows), 30)
        self.assertEqual(result.request_rows()[0]["peak_kv_tokens"], 8)

    def test_sink_blocks_are_kept(self):
        a = PagedAllocator(KVPool(1, 64), block_size=4, watermark=0.0, window=8, sink_tokens=2)
        a.admit(0, 3)
        first = a.block_table(0)[0]
        for _ in range(40):
            a.append(0)
            a.check_invariants()
            self.assertEqual(a.block_table(0)[0], first)  # block 0 holds the sinks
            self.assertLessEqual(a.num_tokens(0), 2 + 8)
            self.assertLessEqual(a.blocks_used, 1 + 2 + 1)  # ceil(S/B) + ceil(W/B) + 1
        self.assertEqual(a.num_tokens(0), 10)

    def test_long_prompt_keeps_only_the_window(self):
        result = run([Request(0, 0, 40, 3)], 64, block_size=4, window=8)
        first = result.step_rows()[0]
        self.assertEqual((first["prefill_tokens"], first["kv_write_bytes"]), (40, 40))
        self.assertEqual(first["kv_tokens"], 8)
        self.assertLessEqual(first["blocks_used"], 3)

    def test_window_lets_more_requests_run(self):
        requests = generate_requests(30, prompt="fixed:16", output="fixed:200", max_model_len=256)
        # 96 blocks: windowed requests hold at most ceil(32/16) + 1 = 3 blocks each (90 in
        # total), full ones grow to ceil(215/16) = 14 blocks each
        full = run(requests, 1536, block_size=16).summary()
        windowed = run(requests, 1536, block_size=16, window=32).summary()
        self.assertGreater(windowed["avg_decode_batch"], full["avg_decode_batch"])
        self.assertEqual(windowed["preemptions"], 0)
        self.assertGreater(full["preemptions"], 0)

    def test_recompute_rebuilds_only_the_kept_tokens(self):
        requests = [Request(i, 0, 2, 10) for i in range(3)]  # each keeps 4, 3 x 4 > 10 slots
        result = run(requests, 10, block_size=1, window=4)
        s = result.summary()
        self.assertGreater(s["preemptions"], 0)
        self.assertEqual(s["finished"], 3)
        self.assertEqual(s["output_tokens"], 30)
        later_prefills = [r["prefill_tokens"] for r in result.step_rows()[1:] if r["phase"] == "prefill"]
        self.assertTrue(later_prefills and all(p <= 2 * (4 + 1) for p in later_prefills))
        self.assertLessEqual(s["recompute_tokens"], 4 * s["preemptions"])

    def test_invariants_under_random_use(self):
        rng = random.Random(5)
        a = PagedAllocator(KVPool(1, 512), block_size=8, watermark=0.0, window=20, sink_tokens=3)
        live = []
        for i in range(3000):
            if rng.random() < 0.3:
                n = rng.randint(1, 60)
                if a.can_admit(i, n):
                    a.admit(i, n)
                    live.append(i)
            elif rng.random() < 0.85 and live:
                seq = rng.choice(live)
                if a.can_append(seq):
                    a.append(seq)
            elif live:
                a.free(live.pop(rng.randrange(len(live))))
            a.check_invariants()


class TestContiguousWindow(unittest.TestCase):
    def test_reservation_is_a_ring_of_sinks_plus_window(self):
        result = run([Request(0, 0, 4, 60)], 512, allocator="contiguous", reservation="max",
                     window=8, sink_tokens=2)
        row = result.request_rows()[0]
        self.assertEqual((row["peak_held_tokens"], row["peak_kv_tokens"]), (10, 10))
        self.assertEqual(row["status"], "finished")

    def test_short_requests_keep_their_smaller_reservation(self):
        a = ContiguousAllocator(KVPool(1, 100), 256, "oracle", window=50)
        a.admit(0, 5, Request(0, 0, 5, 6))  # oracle: 10 slots, under the 50-slot ring
        self.assertEqual(a.held_tokens(0), 10)


class TestConfig(unittest.TestCase):
    def test_bad_window_settings(self):
        for bad in ({"window": 0}, {"sink_tokens": 4}, {"window": 8, "sink_tokens": -1}):
            with self.assertRaises(ConfigError, msg=bad):
                SimConfig(**bad)
        self.assertEqual(SimConfig(window=4096, sink_tokens=4).kv_cap, 4100)
        self.assertIsNone(SimConfig().kv_cap)

    def test_every_request_finishes_with_a_window(self):
        requests = generate_requests(40, prompt="uniform:4:120", output="uniform:4:120",
                                     max_model_len=256, arrival_rate=1.0, seed=8)
        for allocator in ("paged", "contiguous"):
            result = run(requests, 600, allocator=allocator, block_size=8, window=24, sink_tokens=4)
            self.assertTrue(all(s.status is Status.FINISHED for s in result.sequences), allocator)
            self.assertLessEqual(max(r["kv_tokens"] for r in result.step_rows()), 40 * 28)


if __name__ == "__main__":
    unittest.main()
