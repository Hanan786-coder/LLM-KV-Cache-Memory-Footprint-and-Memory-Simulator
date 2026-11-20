import csv
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from kvsim.cli import main
from kvsim.cli_sim import parse_config_label
from kvsim.config import ConfigError
from kvsim.simulator import REQUEST_COLUMNS, STEP_COLUMNS
from kvsim.workload import TRACE_COLUMNS

# a small, fast workload on a real model/GPU pair
SMALL = ["llama-7b", "rtx-4090", "--num-requests", "30", "--prompt-len", "uniform:16:128",
         "--output-len", "uniform:16:128", "--seed", "1"]


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue(), err.getvalue()


def header(path):
    with open(path, newline="", encoding="utf-8") as f:
        return tuple(next(csv.reader(f)))


class TestCommands(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def test_workload_writes_a_trace(self):
        path = self.dir / "trace.csv"
        code, out, _ = run("workload", "--preset", "short", "--num-requests", "50", "--csv", str(path))
        self.assertEqual(code, 0)
        self.assertIn("prompt_len", out)
        self.assertEqual(header(path), TRACE_COLUMNS)

    def test_simulate_writes_the_part3_trace(self):
        steps, reqs, summary = self.dir / "steps.csv", self.dir / "reqs.csv", self.dir / "summary.csv"
        code, out, _ = run("simulate", *SMALL, "--steps-csv", str(steps), "--requests-csv", str(reqs),
                           "--csv", str(summary))
        self.assertEqual(code, 0)
        self.assertIn("KV pool: 10.22 GiB = 20,940 token slots", out)
        self.assertIn("Timeline", out)
        self.assertEqual(header(steps), STEP_COLUMNS)  # the contract with Part 3
        self.assertEqual(header(reqs), REQUEST_COLUMNS)
        with open(summary, newline="", encoding="utf-8") as f:
            row = next(csv.DictReader(f))
        self.assertEqual((row["allocator"], row["finished"]), ("paged-16", "30"))

    def test_simulate_replays_a_trace(self):
        path = self.dir / "trace.csv"
        run("workload", "--num-requests", "20", "--seed", "4", "--csv", str(path))
        code, out, _ = run("simulate", "llama-7b", "a100-80gb", "--trace", str(path),
                           "--allocator", "contiguous", "--reservation", "pow2", "--fit", "best")
        self.assertEqual(code, 0)
        self.assertIn("contig-pow2-best", out)

    def test_compare_has_one_row_per_configuration(self):
        path = self.dir / "compare.csv"
        code, out, _ = run("compare", *SMALL, "--configs", "contig-max,paged-16,paged-4", "--csv", str(path))
        self.assertEqual(code, 0)
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([r["allocator"] for r in rows], ["contig-max", "paged-16", "paged-4"])

    def test_blocksweep(self):
        code, out, _ = run("blocksweep", *SMALL, "--block-sizes", "8,32")
        self.assertEqual(code, 0)
        self.assertIn("Paged allocator by block size", out)

    def test_window_eviction_and_pool_flags(self):
        code, out, _ = run("simulate", *SMALL, "--pool-tokens", "2000", "--window", "64",
                           "--sink-tokens", "4", "--evict-policy", "fifo", "--preempt-mode", "swap")
        self.assertEqual(code, 0)
        self.assertIn("2,000 token slots", out)

    def test_errors_exit_with_code_2(self):
        for argv in (["simulate", *SMALL, "--block-size", "0"],
                     ["simulate", *SMALL, "--sink-tokens", "4"],
                     ["compare", *SMALL, "--configs", "paged-x"],
                     ["simulate", "llama2-70b", "a100-80gb"],  # FP16 70B weights do not fit
                     ["simulate", *SMALL, "--trace", str(self.dir / "missing.csv")]):
            code, _, err = run(*argv)
            self.assertEqual(code, 2, argv)
            self.assertIn("error:", err)

    def test_argparse_rejects_unknown_choices(self):
        with self.assertRaises(SystemExit):
            run("simulate", *SMALL, "--allocator", "buddy")


class TestConfigLabels(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(parse_config_label("paged-16"), {"allocator": "paged", "block_size": 16})
        self.assertEqual(parse_config_label("contig-pow2-best"),
                         {"allocator": "contiguous", "reservation": "pow2", "fit": "best"})
        self.assertEqual(parse_config_label("Contig-max")["fit"], "first")
        for bad in ("paged", "paged-0", "contig-all", "contig-max-worst", "buddy-16"):
            with self.assertRaises(ConfigError, msg=bad):
                parse_config_label(bad)


if __name__ == "__main__":
    unittest.main()
