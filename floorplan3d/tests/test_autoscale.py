"""Tests for automatic scale from dimension strings (api/autoscale.py)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "blender_addon" / "api"))
import autoscale  # type: ignore


class TestParseDimension(unittest.TestCase):
    def test_imperial_notations(self):
        for text, ft, inch in [("17'9\"", 17, 9), ("24'5\"", 24, 5), ("12'", 12, 0), ("9\"", 0, 9),
                               ("17′9″", 17, 9), ("12'-6\"", 12, 6), ("10' 0\"", 10, 0), ("6'2", 6, 2)]:
            with self.subTest(text=text):
                self.assertAlmostEqual(autoscale.parse_dimension(text), ft * 0.3048 + inch * 0.0254, places=4)

    def test_metric_notations(self):
        self.assertAlmostEqual(autoscale.parse_dimension("4.20"), 4.2)
        self.assertAlmostEqual(autoscale.parse_dimension("4,20 m"), 4.2)
        self.assertAlmostEqual(autoscale.parse_dimension("420 cm"), 4.2)
        self.assertAlmostEqual(autoscale.parse_dimension("3500mm"), 3.5)

    def test_non_lengths_rejected(self):
        for text in ("1560", "[AREA: 1560]", "KITCHEN", "13x15", "1st floor", "", "99'0\"" * 3):
            with self.subTest(text=text):
                self.assertIsNone(autoscale.parse_dimension(text))


def _rooms_px_to_m(ppm):
    # Two rooms: 500 px wide x 400 px tall, side by side, in "metres" at ppm.
    def m(px):
        return px / ppm
    return [
        {"label": "a", "polygon": [[m(0), m(0)], [m(500), m(0)], [m(500), m(400)], [m(0), m(400)]]},
        {"label": "b", "polygon": [[m(500), m(0)], [m(900), m(0)], [m(900), m(400)], [m(500), m(400)]]},
    ]


class TestEstimate(unittest.TestCase):
    def test_recovers_true_scale_from_annotations(self):
        # Truth: 500 px == 16'5" (5.0 m) -> 100 px/m; user guessed 50.
        texts = [
            ("16'5\"", [200, 405, 300, 425]),   # under room a's bottom edge (500 px)
            ("13'1\"", [650, 405, 750, 425]),   # under room b's bottom edge (400 px = 4.0 m)
            ("13'1\"", [905, 150, 925, 250]),   # right of room b's right edge (vertical text box)
            ("KITCHEN", [200, 200, 300, 220]),  # not a dimension
        ]
        ppm, rep = autoscale.estimate_pixels_per_meter(_rooms_px_to_m(50), 50, texts)
        self.assertIsNotNone(ppm)
        self.assertAlmostEqual(ppm, 100, delta=1.0)
        self.assertEqual(rep["parsed"], 3)
        self.assertEqual(rep["used"], 3)

    def test_too_few_or_disagreeing_pairs_gives_none(self):
        texts = [("16'5\"", [200, 405, 300, 425]), ("3'0\"", [650, 405, 750, 425])]
        ppm, rep = autoscale.estimate_pixels_per_meter(_rooms_px_to_m(50), 50, texts)
        self.assertIsNone(ppm)

    def test_text_far_from_any_edge_is_ignored(self):
        texts = [("16'5\"", [200, 2000, 300, 2020])] * 3
        ppm, rep = autoscale.estimate_pixels_per_meter(_rooms_px_to_m(50), 50, texts)
        self.assertIsNone(ppm)
        self.assertEqual(rep["used"], 0)


class TestRescale(unittest.TestCase):
    def test_rescale_all_lengths_and_scale_field(self):
        plan = {"scale": {"pixels_per_meter": 50}, "walls": [{"start": [0, 0], "end": [10, 0], "thickness": 0.2}],
                "doors": [{"position": [5, 0], "width": 1.0, "wall_index": 0}],
                "windows": [{"position": [2, 0], "width": 2.0, "wall_index": 0, "sill_height": 1.0}],
                "rooms": [{"label": "a", "polygon": [[0, 0], [10, 0], [10, 8], [0, 8]], "area": 80.0}]}
        out = autoscale.rescale_plan(plan, 0.5)
        self.assertEqual(out["walls"][0]["end"], [5.0, 0.0])
        self.assertAlmostEqual(out["walls"][0]["thickness"], 0.1)
        self.assertEqual(out["doors"][0]["position"], [2.5, 0.0])
        self.assertAlmostEqual(out["windows"][0]["sill_height"], 0.5)
        self.assertAlmostEqual(out["rooms"][0]["area"], 20.0)
        self.assertAlmostEqual(out["scale"]["pixels_per_meter"], 100.0)
        self.assertEqual(plan["walls"][0]["end"], [10, 0])  # input untouched

    def test_apply_auto_scale_end_to_end(self):
        plan = {"scale": {"pixels_per_meter": 50}, "walls": [], "doors": [], "windows": [], "rooms": _rooms_px_to_m(50)}
        texts = [("16'5\"", [200, 405, 300, 425]), ("13'1\"", [650, 405, 750, 425]), ("13'1\"", [905, 150, 925, 250])]
        out, rep = autoscale.apply_auto_scale(plan, 50, texts)
        self.assertTrue(rep["applied"])
        # room a is now 5.0 m wide
        self.assertAlmostEqual(out["rooms"][0]["polygon"][1][0], 5.0, delta=0.05)
        self.assertAlmostEqual(out["scale"]["pixels_per_meter"], 100, delta=1)


if __name__ == "__main__":
    unittest.main()


class TestFootprint(unittest.TestCase):
    def test_footprint_in_pixels(self):
        plan = {"rooms": _rooms_px_to_m(50), "walls": []}
        self.assertEqual(autoscale.footprint_px(plan, 50), [0.0, 0.0, 900.0, 400.0])
        self.assertIsNone(autoscale.footprint_px({"rooms": [], "walls": []}, 50))
