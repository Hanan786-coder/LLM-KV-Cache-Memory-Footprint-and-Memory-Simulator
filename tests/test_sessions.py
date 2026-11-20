import unittest
from types import SimpleNamespace

from kvsim.config import ConfigError
from kvsim.eviction import choose_cached_victim
from kvsim.kvpool import KVPool
from kvsim.sessions import Session, Turn, generate_sessions, validate_sessions
from kvsim.simulator import SimConfig, Simulator, Status, simulate


def run(sessions, pool_tokens, **config):
    config = {"check_invariants": True, "watermark": 0.0, "block_size": 4, "max_model_len": 256, **config}
    return simulate([], KVPool(1, pool_tokens), SimConfig(**config), sessions=sessions)


class TestSession(unittest.TestCase):
    def test_prompt_includes_the_history(self):
        s = Session(0, 0, (Turn(10, 5), Turn(6, 4), Turn(3, 2)))
        self.assertEqual([s.prompt_len(k) for k in range(3)], [10, 10 + 5 + 6, 10 + 5 + 6 + 4 + 3])
        self.assertEqual(s.total_len(), 30)
        r = s.request(1, request_id=7, arrival_step=9)
        self.assertEqual((r.id, r.arrival_step, r.prompt_len, r.session_id, r.turn), (7, 9, 21, 0, 1))

    def test_validation(self):
        with self.assertRaises(ConfigError):
            Session(0, 0, ())
        with self.assertRaises(ConfigError):
            Turn(0, 5)
        with self.assertRaises(ConfigError):
            validate_sessions([Session(0, 0, (Turn(100, 100),))], max_model_len=150)
        with self.assertRaises(ConfigError):
            validate_sessions([Session(0, 0, (Turn(1, 1),)), Session(0, 3, (Turn(1, 1),))], 150)

    def test_generator(self):
        a = generate_sessions(50, turns="uniform:2:5", max_model_len=512, seed=3)
        self.assertEqual(a, generate_sessions(50, turns="uniform:2:5", max_model_len=512, seed=3))
        validate_sessions(a, 512)
        self.assertTrue(all(1 <= len(s.turns) <= 5 and s.start_step == 0 for s in a))
        timed = generate_sessions(50, max_model_len=512, arrival_rate=0.5, seed=3)
        starts = [s.start_step for s in timed]
        self.assertEqual(starts, sorted(starts))
        self.assertGreater(starts[-1], 0)


class TestKeptCache(unittest.TestCase):
    def test_hit_prefills_only_the_new_tokens(self):
        # turn 0: P = 10, O = 5 -> finishes at step 4 holding 14 KV entries
        # turn 1 arrives 3 steps later with prompt 10 + 5 + 6 = 21; only 7 are new
        session = Session(0, 0, (Turn(10, 5), Turn(6, 4, think_steps=3)))
        result = run([session], 512)
        rows = {r["step"]: r for r in result.step_rows()}
        self.assertEqual(list(rows), [0, 1, 2, 3, 4, 7, 8, 9, 10])  # steps 5-6 are idle
        self.assertEqual((rows[4]["cached"], rows[4]["kv_tokens"]), (1, 14))  # kept while thinking
        self.assertEqual((rows[7]["prefill_tokens"], rows[7]["kv_read_bytes"]), (7, 14))
        s = result.summary()
        self.assertEqual((s["cache_hits"], s["cache_misses"], s["saved_prefill_tokens"]), (1, 0, 14))
        self.assertEqual(s["cache_hit_rate"], 1.0)
        self.assertEqual(result.steps[-1].kv_allocated_bytes, 0)  # the last turn frees it

    def test_evicted_cache_means_a_miss(self):
        # 8 blocks of 4. A keeps 3 blocks after its first turn; B needs 6 at step 5,
        # so A's idle cache is dropped and A's second turn prefills everything again
        a = Session(0, 0, (Turn(8, 2), Turn(4, 2, think_steps=20)))
        b = Session(1, 5, (Turn(24, 2),))
        result = run([a, b], 32, max_model_len=64)
        rows = {r["step"]: r for r in result.step_rows()}
        self.assertEqual(rows[21]["prefill_tokens"], 8 + 2 + 4)
        s = result.summary()
        self.assertEqual((s["cache_hits"], s["cache_misses"], s["cache_evictions"]), (0, 1, 1))
        self.assertEqual(s["preemptions"], 0)  # idle caches go before running requests
        self.assertTrue(all(seq.status is Status.FINISHED for seq in result.sequences))

    def test_rejected_turn_ends_the_session_and_frees_its_cache(self):
        session = Session(0, 0, (Turn(8, 2), Turn(60, 2, think_steps=2)))  # turn 1 needs 71 slots
        sim = Simulator([], KVPool(1, 64), SimConfig(block_size=4, watermark=0.0, max_model_len=128),
                        sessions=[session])
        result = sim.run()
        self.assertEqual([s.status for s in result.sequences], [Status.FINISHED, Status.REJECTED])
        # no step is recorded after the rejection, so check the allocator itself
        self.assertEqual((sim.alloc.allocated_tokens(), sim.cache), (0, {}))


class TestReplacementPolicies(unittest.TestCase):
    def test_lru_fifo_and_opt_pick_different_victims(self):
        entries = [
            SimpleNamespace(session_id=0, resident_since=0, last_used=10, next_arrival=50),
            SimpleNamespace(session_id=1, resident_since=5, last_used=6, next_arrival=20),
            SimpleNamespace(session_id=2, resident_since=8, last_used=12, next_arrival=200),
        ]
        self.assertEqual(choose_cached_victim("lru", entries).session_id, 1)  # used longest ago
        self.assertEqual(choose_cached_victim("fifo", entries).session_id, 0)  # cached first
        self.assertEqual(choose_cached_victim("opt", entries).session_id, 2)  # needed last
        with self.assertRaises(ConfigError):
            choose_cached_victim("random", entries)

    def test_every_policy_finishes_every_turn(self):
        sessions = generate_sessions(40, turns="uniform:2:5", prompt="uniform:8:40",
                                     output="uniform:8:60", think="uniform:1:40",
                                     max_model_len=512, arrival_rate=0.5, seed=3)
        turns = sum(len(s.turns) for s in sessions)
        rates = {}
        for policy in ("lru", "fifo", "opt"):
            result = run(sessions, 1024, block_size=8, max_model_len=512, cache_policy=policy)
            s = result.summary()
            self.assertEqual((s["requests"], s["finished"]), (turns, turns), policy)
            self.assertEqual(s["cache_hits"] + s["cache_misses"], turns - len(sessions), policy)
            self.assertGreater(s["cache_evictions"], 0, policy)
            rates[policy] = s["cache_hit_rate"]
        # true for this seeded workload, not in general: caches differ in size and evictions
        # change later timing, so Belady's optimality proof does not apply
        self.assertGreaterEqual(rates["opt"], max(rates["lru"], rates["fifo"]))

    def test_sessions_need_the_paged_allocator_without_a_window(self):
        session = [Session(0, 0, (Turn(4, 4),))]
        for bad in ({"allocator": "contiguous"}, {"window": 64}):
            with self.assertRaises(ConfigError, msg=bad):
                run(session, 512, **bad)
        with self.assertRaises(ConfigError):
            SimConfig(cache_policy="random")


if __name__ == "__main__":
    unittest.main()
