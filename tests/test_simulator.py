import unittest

from kvsim.config import ConfigError
from kvsim.kvpool import KVPool
from kvsim.simulator import STEP_COLUMNS, SimConfig, Status, simulate
from kvsim.workload import Request, generate_requests


def run(requests, pool_tokens, **config):
    config = {"check_invariants": True, "watermark": 0.0, **config}
    return simulate(requests, KVPool(1, pool_tokens), SimConfig(**config))


def small_workload(n=40, seed=0, max_model_len=1024):
    return generate_requests(n, prompt="uniform:10:50", output="uniform:10:50",
                             max_model_len=max_model_len, seed=seed)


class TestTimeline(unittest.TestCase):
    def test_one_request_step_by_step(self):
        # P = 4, O = 3, blocks of 4: section 6 of the design doc
        result = run([Request(0, 0, 4, 3)], 16, block_size=4)
        rows = result.step_rows()
        self.assertEqual([r["phase"] for r in rows], ["prefill", "decode", "decode"])
        self.assertEqual([r["output_tokens"] for r in rows], [1, 1, 1])
        self.assertEqual([r["kv_write_bytes"] for r in rows], [4, 1, 1])
        self.assertEqual([r["kv_read_bytes"] for r in rows], [0, 5, 6])  # cache incl. the new entry
        self.assertEqual([r["attention_pairs"] for r in rows], [10, 5, 6])
        self.assertEqual([r["kv_tokens"] for r in rows], [4, 5, 0])  # freed when it finishes
        self.assertEqual([r["blocks_used"] for r in rows], [1, 2, 0])
        self.assertEqual(rows[2]["finished"], 1)
        req = result.request_rows()[0]
        self.assertEqual((req["ttft_steps"], req["e2e_steps"], req["queue_steps"]), (1, 3, 0))
        self.assertEqual((req["peak_kv_tokens"], req["peak_held_tokens"]), (6, 8))  # P + O - 1

    def test_single_output_token_finishes_in_prefill(self):
        result = run([Request(0, 0, 7, 1)], 16, block_size=4)
        self.assertEqual([r["phase"] for r in result.step_rows()], ["prefill"])
        self.assertEqual(result.sequences[0].status, Status.FINISHED)

    def test_long_prompt_is_prefilled_in_chunks(self):
        result = run([Request(0, 0, 10, 2)], 32, block_size=4, max_batched_tokens=4)
        rows = result.step_rows()
        self.assertEqual([r["prefill_tokens"] for r in rows], [4, 4, 2, 0])
        self.assertEqual([r["kv_read_bytes"] for r in rows[:3]], [0, 4, 8])  # earlier chunks
        self.assertEqual(result.sequences[0].first_token_step, 2)

    def test_idle_gaps_are_skipped(self):
        result = run([Request(0, 0, 4, 2), Request(1, 100, 4, 2)], 32, block_size=4)
        self.assertEqual([r["step"] for r in result.step_rows()], [0, 1, 100, 101])


class TestConservation(unittest.TestCase):
    def test_every_request_finishes_and_memory_comes_back(self):
        requests = small_workload()
        for config in ({"allocator": "paged", "block_size": 16},
                       {"allocator": "contiguous", "reservation": "max"},
                       {"allocator": "contiguous", "reservation": "oracle", "fit": "best"}):
            result = run(requests, 8192, max_model_len=1024, **config)
            self.assertTrue(all(s.status is Status.FINISHED for s in result.sequences), config)
            self.assertEqual(sum(r["output_tokens"] for r in result.step_rows()),
                             sum(r.output_len for r in requests))
            last = result.steps[-1]
            self.assertEqual((last.kv_allocated_bytes, last.running), (0, 0))
            for row in result.step_rows():
                self.assertEqual(row["kv_used_bytes"] + row["reserved_future_bytes"]
                                 + row["internal_frag_bytes"] + row["free_bytes"], row["pool_bytes"])

    def test_oracle_reservation_has_no_internal_fragmentation(self):
        result = run(small_workload(), 8192, allocator="contiguous", reservation="oracle", max_model_len=1024)
        self.assertEqual(max(s.internal_frag_bytes for s in result.steps), 0)

    def test_same_input_gives_the_same_trace(self):
        a = run(small_workload(), 8192, max_model_len=1024).step_rows()
        b = run(small_workload(), 8192, max_model_len=1024).step_rows()
        self.assertEqual(a, b)


class TestAllocatorsCompared(unittest.TestCase):
    def test_contiguous_max_limits_concurrency(self):
        # 4096 slots / 1024 reserved per request = at most 4 at once
        result = run(small_workload(), 4096, allocator="contiguous", reservation="max", max_model_len=1024)
        self.assertEqual(max(s.running for s in result.steps), 4)
        self.assertGreater(result.summary()["admission_stalls"], 0)

    def test_paged_runs_more_requests_and_wastes_less(self):
        requests = small_workload()
        paged = run(requests, 8192, max_model_len=1024).summary()
        contig = run(requests, 8192, allocator="contiguous", max_model_len=1024).summary()
        self.assertGreater(paged["peak_running"], contig["peak_running"])
        self.assertGreater(paged["avg_decode_batch"], contig["avg_decode_batch"])
        self.assertLess(paged["steps"], contig["steps"])
        # requests here are only 20-100 tokens, so a half-empty 16-token block is a big
        # share of each one (86%); smaller blocks waste less
        self.assertGreater(paged["share_token_states"], 0.8)
        self.assertLess(contig["share_token_states"], 0.2)
        small_blocks = run(requests, 8192, max_model_len=1024, block_size=4).summary()
        self.assertGreater(small_blocks["share_token_states"], paged["share_token_states"])

    def test_chat_workload_against_the_vllm_paper(self):
        # the paper: existing systems keep token states in only 20.4%-38.2% of KV memory,
        # vLLM wastes almost none
        requests = generate_requests(200, preset="chat", max_model_len=2048, seed=1)
        paged = run(requests, 200_000, max_model_len=2048).summary()
        contig = run(requests, 200_000, allocator="contiguous", max_model_len=2048).summary()
        self.assertGreater(paged["share_token_states"], 0.9)
        self.assertLess(contig["share_token_states"], 0.5)

    def test_external_fragmentation_shows_up_as_frag_stalls(self):
        # varied reservations leave holes between running requests
        requests = generate_requests(200, prompt="uniform:1:200", output="uniform:1:400",
                                     max_model_len=1024, seed=3)
        contig = run(requests, 2048, allocator="contiguous", reservation="oracle", max_model_len=1024)
        self.assertGreater(contig.summary()["frag_stalls"], 0)
        self.assertGreater(max(s.external_frag_bytes for s in contig.steps), 0)


class TestScheduler(unittest.TestCase):
    def test_static_batching_admits_only_when_empty(self):
        result = run(small_workload(12), 4096, allocator="contiguous", max_model_len=1024,
                     batching="static")
        rows = result.step_rows()
        for prev, row in zip(rows, rows[1:]):
            if row["admitted"]:
                self.assertEqual(prev["running"], 0)
        continuous = run(small_workload(12), 4096, allocator="contiguous", max_model_len=1024)
        self.assertGreater(len(rows), len(continuous.steps))

    def test_max_num_seqs(self):
        result = run(small_workload(), 8192, max_model_len=1024, max_num_seqs=5)
        self.assertEqual(max(s.running for s in result.steps), 5)

    def test_request_that_can_never_fit_is_rejected(self):
        requests = [Request(0, 0, 4, 4), Request(1, 0, 60, 10), Request(2, 0, 4, 4)]
        result = run(requests, 32, block_size=4, max_model_len=128)
        self.assertEqual([s.status for s in result.sequences],
                         [Status.FINISHED, Status.REJECTED, Status.FINISHED])
        self.assertEqual(result.summary()["rejected"], 1)

    def test_last_arrival_rejected_while_idle(self):
        # regression: the idle jump looked for a next arrival in an empty queue
        result = run([Request(0, 0, 4, 2), Request(1, 50, 60, 10)], 32, block_size=4, max_model_len=128)
        self.assertEqual([s.status for s in result.sequences], [Status.FINISHED, Status.REJECTED])

    def test_bad_config(self):
        for bad in ({"allocator": "buddy"}, {"batching": "dynamic"}, {"block_size": 0},
                    {"max_model_len": 1}):
            with self.assertRaises(ConfigError):
                SimConfig(**bad)

    def test_trace_columns(self):
        result = run([Request(0, 0, 4, 3)], 16, block_size=4)
        self.assertEqual(tuple(result.step_rows()[0]), STEP_COLUMNS)
        summary = result.summary()
        self.assertEqual((summary["finished"], summary["output_tokens"]), (1, 3))


if __name__ == "__main__":
    unittest.main()
