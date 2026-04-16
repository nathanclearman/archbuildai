"""Tests for the held-out eval harness.

Covers the metric primitives (polygon IoU, room matching, per-sample
scoring, aggregation) and the end-to-end loop with the two
dependency-free predictors (copy = perfect, null = floor). The CV and
VLM predictors are exercised by separate integration tests gated on
opencv / weights being present, not this unit test.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

from dataset import Sample  # type: ignore
from evaluate import (  # type: ignore
    SampleMetrics,
    aggregate,
    copy_predictor,
    evaluate_sample,
    match_rooms,
    null_predictor,
    polygon_iou,
    run_eval,
)


REAL_MLS_ROOT = Path(__file__).parent / "fixtures" / "real_mls"


UNIT_SQUARE = [[0, 0], [1, 0], [1, 1], [0, 1]]


class PolygonIoUTest(unittest.TestCase):
    def test_identical_polygons_hit_one(self):
        iou = polygon_iou(UNIT_SQUARE, UNIT_SQUARE)
        self.assertGreater(iou, 0.99)

    def test_disjoint_polygons_are_zero(self):
        other = [[10, 10], [11, 10], [11, 11], [10, 11]]
        self.assertEqual(polygon_iou(UNIT_SQUARE, other), 0.0)

    def test_half_overlap_is_one_third(self):
        # Two unit squares overlapping by half: intersection = 0.5, union = 1.5,
        # IoU = 1/3. Rasterization should hit that to within grid resolution.
        right = [[0.5, 0], [1.5, 0], [1.5, 1], [0.5, 1]]
        iou = polygon_iou(UNIT_SQUARE, right)
        self.assertAlmostEqual(iou, 1 / 3, delta=0.02)

    def test_degenerate_polygon_returns_zero(self):
        self.assertEqual(polygon_iou([[0, 0], [1, 0]], UNIT_SQUARE), 0.0)


class RoomMatchingTest(unittest.TestCase):
    def test_greedy_picks_best_iou_per_pair(self):
        # Two pred rooms, two GT rooms; pred[0] overlaps gt[0] perfectly,
        # pred[1] overlaps gt[1] perfectly. Matching should recover both.
        pred = [
            {"label": "a", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]},
            {"label": "b", "polygon": [[2, 0], [3, 0], [3, 1], [2, 1]]},
        ]
        gt = [
            {"label": "a", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]},
            {"label": "b", "polygon": [[2, 0], [3, 0], [3, 1], [2, 1]]},
        ]
        matches = match_rooms(pred, gt)
        self.assertEqual(len(matches), 2)
        self.assertEqual({(i, j) for i, j, _ in matches}, {(0, 0), (1, 1)})

    def test_empty_inputs_return_empty(self):
        self.assertEqual(match_rooms([], [{"label": "a", "polygon": UNIT_SQUARE}]), [])
        self.assertEqual(match_rooms([{"label": "a", "polygon": UNIT_SQUARE}], []), [])


class EvaluateSampleTest(unittest.TestCase):
    def _gt(self):
        return {
            "scale": {"pixels_per_meter": 50},
            "walls": [
                {"start": [0, 0], "end": [4, 0], "thickness": 0.15},
                {"start": [4, 0], "end": [4, 3], "thickness": 0.15},
            ],
            "doors": [{"position": [2, 0], "width": 0.9, "type": "hinged", "wall_index": 0}],
            "windows": [],
            "rooms": [{"label": "living_room", "polygon": UNIT_SQUARE, "area": 1.0}],
        }

    def test_identity_prediction_is_perfect(self):
        gt = self._gt()
        m = evaluate_sample("x", gt, gt)
        self.assertTrue(m.valid)
        self.assertEqual(m.wall_count_pred, m.wall_count_gt)
        self.assertAlmostEqual(m.wall_length_ratio, 1.0, places=2)
        self.assertGreater(m.mean_room_iou, 0.99)
        self.assertEqual(m.room_label_accuracy, 1.0)

    def test_none_prediction_is_invalid(self):
        gt = self._gt()
        m = evaluate_sample("x", None, gt)
        self.assertFalse(m.valid)
        self.assertEqual(m.wall_count_pred, 0)
        self.assertEqual(m.wall_count_gt, 2)
        self.assertEqual(m.mean_room_iou, 0.0)

    def test_wrong_label_drops_label_accuracy_not_iou(self):
        gt = self._gt()
        pred = json.loads(json.dumps(gt))
        pred["rooms"][0]["label"] = "bedroom"
        m = evaluate_sample("x", pred, gt)
        self.assertGreater(m.mean_room_iou, 0.99)
        self.assertEqual(m.room_label_accuracy, 0.0)


class AggregateTest(unittest.TestCase):
    def test_empty_input_returns_zero_row(self):
        agg = aggregate([])
        self.assertEqual(agg["n"], 0)
        self.assertEqual(agg["parse_rate"], 0.0)

    def test_parse_rate_reflects_valid_fraction(self):
        valid = SampleMetrics(
            slug="a", valid=True,
            wall_count_pred=1, wall_count_gt=1,
            door_count_pred=0, door_count_gt=0,
            window_count_pred=0, window_count_gt=0,
            room_count_pred=1, room_count_gt=1,
            wall_length_ratio=1.0, mean_room_iou=1.0,
            room_label_accuracy=1.0, matched_rooms=1,
        )
        invalid = SampleMetrics(
            slug="b", valid=False,
            wall_count_pred=0, wall_count_gt=2,
            door_count_pred=0, door_count_gt=0,
            window_count_pred=0, window_count_gt=0,
            room_count_pred=0, room_count_gt=1,
            wall_length_ratio=0.0, mean_room_iou=0.0,
            room_label_accuracy=0.0, matched_rooms=0,
        )
        agg = aggregate([valid, invalid])
        self.assertEqual(agg["n"], 2)
        self.assertEqual(agg["parse_rate"], 0.5)


class RunEvalTest(unittest.TestCase):
    """End-to-end loop against the shipped real_mls fixtures."""

    def setUp(self):
        # Silence the RealMLSLoader's missing-label warnings for the
        # fixtures that are intentionally malformed.
        import contextlib
        import io as _io
        self._buf = _io.StringIO()
        self._ctx = contextlib.redirect_stdout(self._buf)
        self._ctx.__enter__()

    def tearDown(self):
        self._ctx.__exit__(None, None, None)

    def _samples(self):
        from dataset import build_eval_set  # type: ignore
        return build_eval_set(real_mls_root=REAL_MLS_ROOT)

    def test_copy_predictor_hits_perfect_aggregate(self):
        samples = self._samples()
        self.assertGreaterEqual(len(samples), 2)
        gt_by_image = {str(s.image_path): json.loads(s.target_json) for s in samples}
        per_sample, agg = run_eval(samples, copy_predictor(gt_by_image))
        self.assertEqual(agg["parse_rate"], 1.0)
        self.assertEqual(agg["mean_wall_count_abs_err"], 0.0)
        self.assertEqual(agg["mean_door_count_abs_err"], 0.0)
        self.assertAlmostEqual(agg["mean_wall_length_ratio"], 1.0, places=2)
        self.assertGreater(agg["mean_room_iou"], 0.99)
        self.assertEqual(agg["mean_room_label_accuracy"], 1.0)

    def test_null_predictor_is_zero_iou_but_still_parses(self):
        samples = self._samples()
        per_sample, agg = run_eval(samples, null_predictor())
        self.assertEqual(agg["parse_rate"], 1.0)
        self.assertEqual(agg["mean_room_iou"], 0.0)
        self.assertEqual(agg["mean_room_label_accuracy"], 0.0)
        # Null predictor has 0 walls, so count error = mean(|0 - gt|).
        self.assertGreater(agg["mean_wall_count_abs_err"], 0.0)


if __name__ == "__main__":
    unittest.main()
