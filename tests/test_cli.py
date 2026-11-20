import csv
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from kvsim.cli import main
from kvsim.report import fmt_bytes


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue(), err.getvalue()


class TestCLI(unittest.TestCase):
    def test_every_command_runs(self):
        for argv in (["list"], ["kv", "llama-7b"], ["attention", "llama-7b"],
                     ["precision", "llama-7b"], ["budget", "llama-7b", "rtx-4090"],
                     ["solve", "llama-7b", "a100-80gb"],
                     ["sweep", "llama3-8b", "h100-sxm", "--batches", "1,4", "--contexts", "1024"]):
            code, out, _ = run(*argv)
            self.assertEqual(code, 0, argv)
            self.assertTrue(out.strip(), argv)

    def test_kv_reports_half_mib_per_token(self):
        _, out, _ = run("kv", "llama-7b", "--context", "4096")
        self.assertIn("512.00 KiB", out)
        self.assertIn("2.00 GiB", out)

    def test_oom_is_reported(self):
        _, out, _ = run("budget", "llama2-70b", "a100-80gb")
        self.assertIn("OUT OF MEMORY", out)

    def test_csv_export_has_raw_numbers(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sweep.csv"
            run("sweep", "llama-7b", "a100-80gb", "--batches", "1,2", "--contexts", "4096",
                "--csv", str(path))
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)
        self.assertEqual(int(rows[1]["kv_cache_bytes"]), 4 * 1024**3)

    def test_bad_profile_is_an_error_not_a_crash(self):
        code, _, err = run("kv", "not-a-model")
        self.assertEqual(code, 2)
        self.assertIn("unknown model", err)


class TestFormatting(unittest.TestCase):
    def test_fmt_bytes(self):
        self.assertEqual(fmt_bytes(512), "512 B")
        self.assertEqual(fmt_bytes(512 * 1024), "512.00 KiB")
        self.assertEqual(fmt_bytes(-2 * 1024**3), "-2.00 GiB")


if __name__ == "__main__":
    unittest.main()
