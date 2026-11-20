import csv
import random
import tempfile
import unittest
from pathlib import Path

from kvsim.config import ConfigError
from kvsim.workload import (
    DEFAULT_PRESET, PRESETS, TRACE_COLUMNS, LengthSpec, Request, generate_requests,
    generate_workload, load_trace, parse_length_spec, percentile, save_trace,
    validate_requests, workload_stats,
)


class TestRequest(unittest.TestCase):
    def test_lengths(self):
        r = Request(7, 3, prompt_len=4, output_len=3)
        self.assertEqual(r.total_len, 7)
        self.assertEqual(r.peak_kv_tokens, 6)  # P + O - 1: the last token's KV is never written

    def test_new_token_limit(self):
        self.assertEqual(Request(0, 0, 100, 10).new_token_limit(2048), 1948)
        self.assertEqual(Request(0, 0, 100, 10, max_new_tokens=64).new_token_limit(2048), 64)
        self.assertEqual(Request(0, 0, 2000, 10, max_new_tokens=64).new_token_limit(2048), 48)

    def test_invalid_values(self):
        for change in ({"prompt_len": 0}, {"output_len": 0}, {"arrival_step": -1}, {"id": -1},
                       {"prompt_len": 1.5}, {"output_len": True}, {"max_new_tokens": 0},
                       {"output_len": 9, "max_new_tokens": 8}):
            args = {"id": 1, "arrival_step": 0, "prompt_len": 4, "output_len": 2, **change}
            with self.assertRaises(ConfigError, msg=change):
                Request(**args)

    def test_validate_requests(self):
        ok = [Request(0, 0, 10, 4), Request(1, 0, 4, 12)]  # 14 and 16 tokens
        validate_requests(ok, max_model_len=16)
        with self.assertRaisesRegex(ConfigError, "request 1"):
            validate_requests(ok, max_model_len=15)
        with self.assertRaisesRegex(ConfigError, "duplicate"):
            validate_requests([Request(0, 0, 1, 1), Request(0, 1, 1, 1)], 16)


class TestLengthSpec(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_length_spec("fixed:128"), LengthSpec("fixed", 128, 128))
        self.assertEqual(parse_length_spec("128"), LengthSpec("fixed", 128, 128))
        self.assertEqual(parse_length_spec(128), LengthSpec("fixed", 128, 128))
        self.assertEqual(parse_length_spec(" Uniform:16:512 "), LengthSpec("uniform", 16, 512))
        self.assertEqual(parse_length_spec("lognormal:168:1.0"),
                         LengthSpec("lognormal", 1, None, 168.0, 1.0))
        self.assertEqual(parse_length_spec("lognormal:168:1:4:1536"),
                         LengthSpec("lognormal", 4, 1536, 168.0, 1.0))

    def test_str_gives_the_spec_back(self):
        for text in ("fixed:128", "uniform:16:512", "lognormal:168:1", "lognormal:20:0.8:4:256"):
            self.assertEqual(str(parse_length_spec(text)), text)

    def test_bad_specs(self):
        for text in ("", "abc", "fixed", "fixed:0", "fixed:1.5", "uniform:5", "uniform:9:3",
                     "uniform:a:b", "lognormal:100", "lognormal:-5:1", "lognormal:100:-1",
                     "lognormal:nan:1", "lognormal:100:1:4", "lognormal:100:1:50:10", "normal:3:1"):
            with self.assertRaises(ConfigError, msg=text):
                parse_length_spec(text)

    def test_samples_stay_in_bounds(self):
        rng = random.Random(1)
        for text, low, high in (("fixed:7", 7, 7), ("uniform:16:512", 16, 512),
                                ("lognormal:168:1.0:4:1536", 4, 1536)):
            spec = parse_length_spec(text)
            values = [spec.sample(rng) for _ in range(2000)]
            self.assertTrue(all(isinstance(v, int) and low <= v <= high for v in values), text)

    def test_lognormal_has_the_requested_mean(self):
        spec, rng = parse_length_spec("lognormal:100:0.5"), random.Random(2)
        values = [spec.sample(rng) for _ in range(20000)]
        self.assertAlmostEqual(sum(values) / len(values) / 100, 1, delta=0.03)

    def test_nominal_mean(self):
        self.assertEqual(parse_length_spec("uniform:10:20").nominal_mean, 15)
        self.assertEqual(parse_length_spec("fixed:9").nominal_mean, 9)
        self.assertEqual(parse_length_spec("lognormal:168:1:4:1536").nominal_mean, 168)


class TestPresets(unittest.TestCase):
    def test_chat_vs_short_matches_the_paper_ratios(self):
        chat, short = PRESETS["chat"], PRESETS["short"]  # ShareGPT vs Alpaca: 8.4x, 5.8x
        self.assertAlmostEqual(chat.prompt.nominal_mean / short.prompt.nominal_mean, 8.4)
        self.assertAlmostEqual(chat.output.nominal_mean / short.output.nominal_mean, 5.8)

    def test_every_preset_generates_valid_requests(self):
        for name in PRESETS:
            requests = generate_requests(300, preset=name, max_model_len=2048, seed=3)
            self.assertEqual(len(requests), 300)
            validate_requests(requests, 2048)

    def test_default_preset_is_chat(self):
        self.assertEqual(DEFAULT_PRESET, "chat")
        self.assertEqual(generate_requests(50, max_model_len=2048),
                         generate_requests(50, preset="chat", max_model_len=2048))


class TestGenerator(unittest.TestCase):
    @staticmethod
    def gen(**changes):
        return generate_requests(200, **{"max_model_len": 2048, "seed": 1, **changes})

    @staticmethod
    def lengths(requests):
        return [(r.prompt_len, r.output_len) for r in requests]

    def test_same_seed_gives_the_same_requests(self):
        self.assertEqual(self.gen(), self.gen())
        self.assertNotEqual(self.gen(), self.gen(seed=2))

    def test_random_streams_are_independent(self):
        base = self.gen()
        self.assertEqual(self.lengths(self.gen(arrival_rate=0.5)), self.lengths(base))
        self.assertEqual([r.prompt_len for r in self.gen(output="fixed:10")],
                         [r.prompt_len for r in base])

    def test_offline_batch_arrives_at_step_zero(self):
        requests = self.gen()
        self.assertTrue(all(r.arrival_step == 0 for r in requests))
        self.assertEqual([r.id for r in requests], list(range(200)))

    def test_poisson_arrivals(self):
        requests = generate_requests(4000, max_model_len=2048, arrival_rate=0.5, seed=4)
        steps = [r.arrival_step for r in requests]
        self.assertEqual(steps, sorted(steps))
        self.assertAlmostEqual(steps[-1] / 8000, 1, delta=0.05)  # 4000 requests at 0.5 per step

    def test_clipping_to_the_model_length(self):
        w = generate_workload(500, preset="chat", max_model_len=256, seed=5)
        self.assertGreater(w.clipped, 0)
        self.assertTrue(all(r.total_len <= 256 and r.output_len >= 1 for r in w.requests))
        unclipped = generate_workload(50, prompt="fixed:10", output="fixed:10", max_model_len=2048)
        self.assertEqual(unclipped.clipped, 0)

    def test_max_new_tokens(self):
        w = generate_workload(300, max_model_len=2048, max_new_tokens=64, seed=6)
        self.assertTrue(all(r.output_len <= 64 and r.max_new_tokens == 64 for r in w.requests))
        self.assertGreater(w.clipped, 0)

    def test_bad_arguments(self):
        for change in ({"preset": "nope"}, {"arrival_rate": 0}, {"arrival_rate": -1.0},
                       {"max_new_tokens": 0}, {"max_model_len": 1}, {"prompt": "uniform:9:3"}):
            with self.assertRaises(ConfigError, msg=change):
                generate_workload(10, **{"max_model_len": 2048, **change})
        with self.assertRaises(ConfigError):
            generate_workload(0, max_model_len=2048)


class TestTrace(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def write(self, text, encoding="utf-8"):
        path = self.dir / "trace.csv"
        path.write_text(text, encoding=encoding)
        return path

    def test_round_trip(self):
        requests = generate_requests(50, max_model_len=2048, arrival_rate=0.3, seed=7)
        requests.append(Request(50, 999, 10, 5, max_new_tokens=32))
        path = self.dir / "saved.csv"
        save_trace(requests, path)
        self.assertEqual(load_trace(path, 2048), requests)
        with open(path, newline="", encoding="utf-8") as f:
            self.assertEqual(next(csv.reader(f)), list(TRACE_COLUMNS))

    def test_hand_written_trace(self):
        path = self.write("id,arrival_step,prompt_len,output_len,max_new_tokens\n"
                          "2,5,1800,64,\n0,0,512,128,\n1,0,40,900,1024\n")
        requests = load_trace(path, 2048)
        self.assertEqual([r.id for r in requests], [0, 1, 2])  # by arrival step, then id
        self.assertEqual(requests[1].max_new_tokens, 1024)
        self.assertIsNone(requests[0].max_new_tokens)

    def test_spreadsheet_quirks(self):
        # no max_new_tokens column, a byte-order mark, and spaces after the commas
        path = self.write("id, arrival_step, prompt_len, output_len\n0, 0, 8, 4\n", encoding="utf-8-sig")
        self.assertEqual(load_trace(path, 2048), [Request(0, 0, 8, 4)])

    def test_errors_say_what_is_wrong(self):
        header = "id,arrival_step,prompt_len,output_len\n"
        cases = {
            "": "empty",
            header: "no requests",
            "id,arrival,prompt_len,output_len\n0,0,1,1\n": "unknown column",
            "id,arrival_step,prompt_len\n0,0,1\n": "missing column",
            header + "0,0,1,1\n1,0,abc,1\n": "request id 1.*prompt_len",
            header + "3,0,1,\n": "request id 3.*missing output_len",
            header + "x,0,1,1\n": "line 2.*id",
            header + "0,0,1,1,9\n": "more values",
            header + "0,0,2000,100\n": "request 0.*max_model_len",
            header + "0,0,1,1\n0,1,1,1\n": "duplicate",
        }
        for text, pattern in cases.items():
            with self.assertRaisesRegex(ConfigError, pattern, msg=text):
                load_trace(self.write(text), 2048)

    def test_missing_file(self):
        with self.assertRaisesRegex(ConfigError, "cannot read"):
            load_trace(self.dir / "nope.csv", 2048)


class TestStats(unittest.TestCase):
    def test_percentile_is_nearest_rank(self):
        values = [40, 10, 30, 20]
        self.assertEqual(percentile(values, 50), 20)
        self.assertEqual(percentile(values, 95), 40)
        self.assertEqual(percentile(values, 0), 10)
        self.assertEqual(percentile(values, 100), 40)
        self.assertEqual(percentile(range(1, 21), 95), 19)
        with self.assertRaises(ValueError):
            percentile([], 50)
        with self.assertRaises(ValueError):
            percentile(values, 101)

    def test_workload_stats(self):
        rows = [(0, 10, 1), (0, 20, 2), (1, 30, 3), (5, 40, 4)]
        requests = [Request(i, a, p, o) for i, (a, p, o) in enumerate(rows)]
        stats = {row["quantity"]: row for row in workload_stats(requests)}
        self.assertEqual(stats["prompt_len"]["mean"], 25)
        self.assertEqual(stats["prompt_len"]["p50"], 20)
        self.assertEqual(stats["total_len"]["max"], 44)
        self.assertEqual(stats["arrival_step"]["max"], 5)
        self.assertEqual(workload_stats([]), [])


if __name__ == "__main__":
    unittest.main()
