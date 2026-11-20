import csv
import io
import tempfile
import unittest
import xml.etree.ElementTree as ET
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from kvsim.cli import main
from kvsim.profiler import TIMED_REQUEST_COLUMNS, TIMED_STEP_COLUMNS

SMALL = ["--num-requests", "20", "--prompt-len", "uniform:16:128", "--output-len",
         "uniform:16:64", "--seed", "1"]


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue(), err.getvalue()


def rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestPerfCommands(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def test_roofline(self):
        table, svg = self.dir / "r.csv", self.dir / "r.svg"
        code, out, _ = run("roofline", "llama-7b", "a100-80gb", "--batch", "8", "--csv", str(table),
                           "--svg", str(svg))
        self.assertEqual(code, 0)
        self.assertIn("ridge point 153 FLOPs/byte", out)
        self.assertIn("decode stays memory-bound", out)
        self.assertIn("prefill turns compute-bound from prompts of", out)
        self.assertEqual([r["bound"] for r in rows(table)], ["compute", "memory"])
        ET.parse(svg)

    def test_roofline_gqa_decode_crosses_the_ridge_at_short_context(self):
        code, out, _ = run("roofline", "llama2-70b", "h100-sxm", "--weight-dtype", "int4",
                           "--context", "16")
        self.assertEqual(code, 0)
        self.assertIn("decode turns compute-bound from batch", out)

    def test_roofline_warns_when_it_does_not_fit(self):
        _, out, _ = run("roofline", "llama-7b", "rtx-4090", "--batch", "512", "--context", "2048")
        self.assertIn("does not fit", out)

    def test_scaling_over_two_gpus(self):
        table, svg = self.dir / "s.csv", self.dir / "s.svg"
        code, out, _ = run("scaling", "llama3-8b", "a100-80gb,h100-sxm", "--values", "1,8,64",
                           "--csv", str(table), "--svg", str(svg))
        self.assertEqual(code, 0)
        self.assertIn("best fitting batch", out)
        data = rows(table)
        self.assertEqual([r["gpu"] for r in data], ["a100-80gb"] * 3 + ["h100-sxm"] * 3)
        self.assertEqual(len(ET.parse(svg).getroot().findall("{http://www.w3.org/2000/svg}polyline")), 2)

    def test_scaling_context_and_prompt(self):
        for vary, column in (("context", "latency_ms"), ("prompt", "ttft_ms")):
            code, out, _ = run("scaling", "mistral-7b", "rtx-4090", "--vary", vary, "--values", "128,1024")
            self.assertEqual(code, 0)
            self.assertIn(column, out)

    def test_profile_writes_timed_traces(self):
        steps, reqs, summary = self.dir / "t.csv", self.dir / "q.csv", self.dir / "s.csv"
        code, out, _ = run("profile", "llama-7b", "rtx-4090", *SMALL, "--steps-csv", str(steps),
                           "--requests-csv", str(reqs), "--csv", str(summary), "--svg",
                           str(self.dir / "p.svg"))
        self.assertEqual(code, 0)
        self.assertIn("throughput_tok_s", out)
        self.assertIn("By phase", out)
        self.assertEqual(tuple(rows(steps)[0]), TIMED_STEP_COLUMNS)
        self.assertEqual(tuple(rows(reqs)[0]), TIMED_REQUEST_COLUMNS)
        self.assertEqual(rows(summary)[0]["finished"], "20")

    def test_profile_from_simulate_csvs_matches_a_live_run(self):
        steps, reqs = self.dir / "steps.csv", self.dir / "reqs.csv"
        run("simulate", "llama-7b", "rtx-4090", *SMALL, "--steps-csv", str(steps),
            "--requests-csv", str(reqs))
        live, offline = self.dir / "live.csv", self.dir / "off.csv"
        run("profile", "llama-7b", "rtx-4090", *SMALL, "--csv", str(live))
        code, out, _ = run("profile", "llama-7b", "rtx-4090", "--from-steps", str(steps),
                           "--from-requests", str(reqs), "--csv", str(offline))
        self.assertEqual(code, 0)
        self.assertIn("Profiling", out)
        self.assertEqual(rows(live), rows(offline))

    def test_profile_compares_configs(self):
        code, out, _ = run("profile", "llama-7b", "rtx-4090", *SMALL, "--configs", "paged-16,contig-max")
        self.assertEqual(code, 0)
        self.assertIn("contig-max", out)
        self.assertIn("in real time", out)

    def test_efficiency_flags(self):
        _, ideal, _ = run("roofline", "llama-7b", "a100-80gb")
        code, real, _ = run("roofline", "llama-7b", "a100-80gb", "--bw-efficiency", "0.5")
        self.assertEqual(code, 0)
        self.assertIn("at 50% bandwidth", real)
        self.assertIn("ridge point 306", real)
        self.assertNotEqual(ideal, real)

    def test_errors(self):
        code, _, err = run("roofline", "llama-7b", "a100-80gb", "--bw-efficiency", "2")
        self.assertEqual(code, 2)
        self.assertIn("bw_efficiency", err)
        code, _, err = run("profile", "llama-7b", "rtx-4090", "--from-requests", "x.csv")
        self.assertEqual((code, "--from-steps" in err), (2, True))


if __name__ == "__main__":
    unittest.main()
