import unittest

from kvsim.config import ConfigError
from kvsim.eviction import SwapSpace, choose_victim
from kvsim.kvpool import KVPool
from kvsim.simulator import SimConfig, Status, simulate
from kvsim.workload import Request, generate_requests


def run(requests, pool_tokens, **config):
    config = {"check_invariants": True, "watermark": 0.0, "block_size": 1, "max_model_len": 64, **config}
    return simulate(requests, KVPool(1, pool_tokens), SimConfig(**config))


# A 12-slot pool with 1-token blocks. After three steps both requests together hold
# all 12 slots and both still need to grow, so exactly one must be evicted.
OLD_BIG = [Request(0, 0, 6, 4), Request(1, 0, 2, 6)]  # older request holds more
OLD_SMALL = [Request(0, 0, 2, 6), Request(1, 0, 6, 4)]  # older request holds less


def evicted(result):
    return [s.id for s in result.sequences if s.preemptions]


class TestVictimPolicies(unittest.TestCase):
    def test_who_gets_evicted(self):
        expected = {  # (older is bigger, older is smaller)
            "lifo": ([1], [1]),  # always the newest
            "fifo": ([0], [0]),  # always the oldest
            "largest": ([0], [1]),  # whoever holds more
        }
        for policy, (big_first, small_first) in expected.items():
            self.assertEqual(evicted(run(OLD_BIG, 12, evict_policy=policy)), big_first, policy)
            self.assertEqual(evicted(run(OLD_SMALL, 12, evict_policy=policy)), small_first, policy)

    def test_overflow_is_counted_once_per_request_and_step(self):
        s = run(OLD_BIG, 12).summary()
        self.assertEqual((s["overflow_events"], s["preemptions"]), (1, 1))

    def test_choose_victim_breaks_ties_the_same_way(self):
        class Seq:
            def __init__(self, id, arrival):
                self.id, self.key = id, (arrival, id)

        seqs = [Seq(3, 0), Seq(1, 0), Seq(2, 5)]
        held = {1: 8, 2: 8, 3: 4}.get
        self.assertEqual(choose_victim("lifo", seqs, held).id, 2)
        self.assertEqual(choose_victim("fifo", seqs, held).id, 1)
        self.assertEqual(choose_victim("largest", seqs, held).id, 2)  # tie on size: newest
        with self.assertRaises(ConfigError):
            choose_victim("random", seqs, held)


class TestRecompute(unittest.TestCase):
    def test_recompute_rebuilds_the_cache_and_still_emits_every_token(self):
        result = run(OLD_BIG, 12, evict_policy="lifo")
        victim = result.sequences[1]  # P = 2; evicted after 3 tokens, holding 2 + 3 - 1 = 4 KV
        self.assertEqual((victim.recompute_tokens, victim.status), (4, Status.FINISHED))
        rows = result.step_rows()
        self.assertIn(5, [r["prefill_tokens"] for r in rows])  # re-prefill of P + 3 tokens
        self.assertEqual(sum(r["output_tokens"] for r in rows), 4 + 6)
        self.assertEqual(sum(r["recompute_tokens"] for r in rows), 4)
        self.assertEqual(result.request_rows()[1]["preemptions"], 1)


class TestSwap(unittest.TestCase):
    def test_swap_moves_blocks_instead_of_recomputing(self):
        result = run(OLD_BIG, 12, evict_policy="lifo", preempt_mode="swap")
        s = result.summary()
        self.assertEqual((s["recompute_tokens"], s["swap_fallbacks"]), (0, 0))
        self.assertEqual(s["swap_out_bytes"], 4)  # the victim held 4 one-token blocks
        self.assertEqual(s["swap_in_bytes"], 4)
        self.assertEqual(result.request_rows()[1]["swap_bytes"], 8)  # out and back
        self.assertEqual(s["output_tokens"], 10)
        self.assertTrue(any(r["swapped"] and r["swap_used_bytes"] == 4 for r in result.step_rows()))
        self.assertEqual(result.steps[-1].swap_used_bytes, 0)
        # after swapping back in it decodes on: no second prefill
        self.assertEqual(sum(r["prefill_tokens"] for r in result.step_rows()), 6 + 2)

    def test_full_swap_space_falls_back_to_recompute(self):
        s = run(OLD_BIG, 12, evict_policy="lifo", preempt_mode="swap", swap_bytes=3).summary()
        self.assertEqual((s["swap_fallbacks"], s["swap_out_bytes"], s["recompute_tokens"]), (1, 0, 4))

    def test_swap_space(self):
        space = SwapSpace(10)
        space.store(6)
        self.assertFalse(space.can_hold(5))
        space.release(6)
        self.assertEqual(space.used_tokens, 0)
        with self.assertRaises(RuntimeError):
            space.release(1)


class TestUnderPressure(unittest.TestCase):
    def test_every_policy_and_mode_finishes_everything(self):
        requests = generate_requests(60, prompt="uniform:4:40", output="uniform:4:60",
                                     max_model_len=128, arrival_rate=2.0, seed=11)
        for policy in ("lifo", "fifo", "largest"):
            for mode in ("recompute", "swap"):
                result = run(requests, 256, block_size=4, max_model_len=128,
                             evict_policy=policy, preempt_mode=mode, swap_bytes=64)
                s = result.summary()
                label = f"{policy}/{mode}"
                self.assertEqual(s["finished"], 60, label)
                self.assertGreater(s["preemptions"], 0, label)
                self.assertEqual(s["output_tokens"], sum(r.output_len for r in requests), label)
                self.assertEqual(result.steps[-1].swap_used_bytes, 0, label)

    def test_lifo_loses_less_work_than_fifo(self):
        requests = generate_requests(80, preset="chat", max_model_len=1024, seed=12)
        lifo = run(requests, 4096, block_size=16, max_model_len=1024).summary()
        fifo = run(requests, 4096, block_size=16, max_model_len=1024, evict_policy="fifo").summary()
        self.assertLess(lifo["recompute_tokens"], fifo["recompute_tokens"])

    def test_contiguous_never_preempts(self):
        requests = generate_requests(60, prompt="uniform:4:40", output="uniform:4:60",
                                     max_model_len=128, seed=11)
        s = run(requests, 512, allocator="contiguous", max_model_len=128).summary()
        self.assertEqual((s["preemptions"], s["finished"]), (0, 60))

    def test_bad_config(self):
        for bad in ({"evict_policy": "random"}, {"preempt_mode": "drop"}, {"swap_bytes": -1}):
            with self.assertRaises(ConfigError):
                SimConfig(**bad)


if __name__ == "__main__":
    unittest.main()
