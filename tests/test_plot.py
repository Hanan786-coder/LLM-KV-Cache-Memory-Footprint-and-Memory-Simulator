import unittest
import xml.etree.ElementTree as ET

from kvsim.plot import Axis, line_chart, roofline_chart

NS = "{http://www.w3.org/2000/svg}"


class TestAxis(unittest.TestCase):
    def test_log_axis_snaps_to_decades(self):
        a = Axis([3, 700], log=True, lo_px=0, hi_px=300)
        self.assertEqual((a.lo, a.hi), (1, 1000))
        self.assertAlmostEqual(a(10), 100)
        self.assertEqual(a.ticks(), [1, 10, 100, 1000])
        short = Axis([3, 70], log=True, lo_px=0, hi_px=300)  # few decades: 2 and 5 added
        self.assertEqual(short.ticks(), [1, 2, 5, 10, 20, 50, 100])

    def test_linear_axis_starts_at_zero_and_has_round_ticks(self):
        a = Axis([3, 47], log=False, lo_px=0, hi_px=100)
        self.assertEqual(a.lo, 0)
        self.assertEqual(a.ticks(), [0, 10, 20, 30, 40, 50])

    def test_degenerate_values(self):
        self.assertLess(Axis([], log=True, lo_px=0, hi_px=1).lo, Axis([], log=True, lo_px=0, hi_px=1).hi)
        self.assertEqual(Axis([5, 5], log=False, lo_px=0, hi_px=1).lo, 0)


class TestCharts(unittest.TestCase):
    def test_line_chart_is_valid_svg_with_one_line_per_series(self):
        svg = line_chart([("a", [1, 2, 4], [1, 4, 16]), ("b & c", [1, 2], [2, 3])],
                         "Title <x>", "x", "y", logx=True)
        root = ET.fromstring(svg)
        self.assertEqual(len(root.findall(f"{NS}polyline")), 2)
        self.assertIn("b &amp; c", svg)

    def test_points_that_cannot_go_on_a_log_axis_are_skipped(self):
        svg = line_chart([("a", [0, 1, 10], [5, None, 6])], "t", "x", "y", logx=True)
        self.assertEqual(len(ET.fromstring(svg).findall(f"{NS}circle")), 1)

    def test_roofline_has_the_roof_ridge_and_points(self):
        svg = roofline_chart(312e12, 2039e9, [("decode", 1.0, 2e12), ("prefill", 500, 300e12),
                                              ("decode", 0, 1)], "Roofline")
        root = ET.fromstring(svg)
        self.assertEqual(len(root.findall(f"{NS}circle")), 2)  # the zero point is dropped
        self.assertIn("ridge 153", svg)


if __name__ == "__main__":
    unittest.main()
