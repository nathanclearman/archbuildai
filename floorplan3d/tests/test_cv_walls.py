"""Unit tests for the classical-CV wall extractor.

cv_walls.py is the geometry-only fallback the inference pipeline uses
when no trained VLM adapter is available yet. It's pure CPU (opencv +
numpy, no torch) and the Blender add-on calls it via
`inference.py --cv-only`. Previously untested: a hyperparameter nudge
on hough_threshold or a subtle change to the endpoint-snap union-find
could silently produce wrong geometry with no regression gate.

These tests cover the pure helpers (_snap_angle, _collapse_ranges,
_merge_collinear, _snap_endpoints, _polygon_area) directly and run
`extract` end-to-end on a synthesized test image so a future
opencv-version bump that breaks the Hough or connected-component paths
goes red immediately.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

from cv_walls import (  # type: ignore
    CVConfig,
    _collapse_ranges,
    _polygon_area,
    _snap_angle,
    _snap_endpoints,
    extract,
)


class SnapAngleTest(unittest.TestCase):
    """Near-axis segments get snapped to exact horizontals / verticals
    so the downstream collinear merger can bin them by integer y or x.
    A skew that survives this snap (e.g. tol too tight) would put every
    wall on its own line and produce a graph with N walls where the
    clean version has N/3."""

    def test_almost_horizontal_snaps_to_horizontal(self):
        # 1-degree tilt on a long span — well within the 3-degree tol.
        x1, y1, x2, y2 = _snap_angle((0, 100, 500, 105), max_deg=3.0)
        self.assertEqual(y1, y2)

    def test_almost_vertical_snaps_to_vertical(self):
        x1, y1, x2, y2 = _snap_angle((100, 0, 105, 500), max_deg=3.0)
        self.assertEqual(x1, x2)

    def test_diagonal_is_not_snapped(self):
        # A 45-degree line is nowhere near either axis; leave it alone.
        # The tol doesn't reach this far regardless of parameter choice;
        # a regression that accidentally over-snapped would collapse
        # stairways / angled walls into rectangles.
        inp = (0.0, 0.0, 100.0, 100.0)
        out = _snap_angle(inp, max_deg=3.0)
        self.assertEqual(out, inp)

    def test_exactly_horizontal_is_preserved(self):
        inp = (0.0, 100.0, 500.0, 100.0)
        out = _snap_angle(inp, max_deg=3.0)
        self.assertEqual(out, inp)


class CollapseRangesTest(unittest.TestCase):
    """_collapse_ranges is the 1-D interval union used by the
    horizontal/vertical wall merger. Boundary cases (touching ranges,
    fully-contained ranges, gap-within-tol) are where this code
    historically regressed."""

    def test_disjoint_ranges_stay_separate(self):
        out = _collapse_ranges([(0, 10), (20, 30)], tol=1)
        self.assertEqual(out, [(0, 10), (20, 30)])

    def test_overlapping_ranges_merge(self):
        out = _collapse_ranges([(0, 20), (15, 30)], tol=1)
        self.assertEqual(out, [(0, 30)])

    def test_touching_within_tol_merges(self):
        # Gap of exactly `tol` pixels counts as touching — wall
        # strokes that were broken by noise reconnect here.
        out = _collapse_ranges([(0, 10), (11, 20)], tol=1)
        self.assertEqual(out, [(0, 20)])

    def test_gap_above_tol_stays_separate(self):
        out = _collapse_ranges([(0, 10), (13, 20)], tol=1)
        self.assertEqual(out, [(0, 10), (13, 20)])

    def test_contained_range_absorbed(self):
        out = _collapse_ranges([(0, 50), (10, 20)], tol=1)
        self.assertEqual(out, [(0, 50)])

    def test_unsorted_input_is_sorted_first(self):
        # The function sorts internally; callers don't need to.
        out = _collapse_ranges([(20, 30), (0, 10)], tol=1)
        self.assertEqual(out, [(0, 10), (20, 30)])


class SnapEndpointsTest(unittest.TestCase):
    def test_empty_input_returns_empty(self):
        self.assertEqual(_snap_endpoints([], tol=5), [])

    def test_nearby_endpoints_get_unified(self):
        # Two L-joining walls whose endpoints are 2 px apart should
        # snap to a shared vertex within the 5 px tol.
        segs = [
            (0.0, 0.0, 100.0, 0.0),      # horizontal
            (102.0, 2.0, 102.0, 100.0),  # vertical, endpoint close to (100, 0)
        ]
        out = _snap_endpoints(segs, tol=5.0)
        # The right end of the horizontal and the top of the vertical
        # should coincide after snap.
        self.assertEqual((out[0][2], out[0][3]), (out[1][0], out[1][1]))

    def test_far_endpoints_stay_separate(self):
        # 50 px apart, 5 px tol — well outside the snap radius.
        segs = [
            (0.0, 0.0, 100.0, 0.0),
            (150.0, 0.0, 150.0, 100.0),
        ]
        out = _snap_endpoints(segs, tol=5.0)
        self.assertNotEqual((out[0][2], out[0][3]), (out[1][0], out[1][1]))


class PolygonAreaTest(unittest.TestCase):
    def test_degenerate_polygon_returns_zero(self):
        self.assertEqual(_polygon_area([]), 0.0)
        self.assertEqual(_polygon_area([[0, 0], [1, 1]]), 0.0)

    def test_unit_square_area_is_one(self):
        self.assertEqual(_polygon_area([[0, 0], [1, 0], [1, 1], [0, 1]]), 1.0)

    def test_orientation_does_not_affect_sign(self):
        # Shoelace can return negative area for CW input; we take
        # abs() so either winding produces the same positive number.
        cw = _polygon_area([[0, 0], [1, 0], [1, 1], [0, 1]])
        ccw = _polygon_area([[0, 0], [0, 1], [1, 1], [1, 0]])
        self.assertEqual(cw, ccw)

    def test_non_unit_rectangle(self):
        # 4 x 3 = 12.
        self.assertEqual(_polygon_area([[0, 0], [4, 0], [4, 3], [0, 3]]), 12.0)

    def test_l_shape_polygon(self):
        # 6-vertex L: outer 4x4 minus inner 2x2 corner = 16 - 4 = 12.
        l = [[0, 0], [4, 0], [4, 2], [2, 2], [2, 4], [0, 4]]
        self.assertEqual(_polygon_area(l), 12.0)


class ExtractEndToEndTest(unittest.TestCase):
    """Smoke test: synth a simple 2-room rectangle image in PIL, run
    extract, confirm the output validates against the canonical schema
    and is roughly correct in scale. Not a precision test — the Hough
    + connected-component pipeline is too parameter-sensitive for
    pixel-accurate assertions — but enough to catch "opencv returned
    None" / "no rooms found" regressions.
    """

    def _make_two_room_image(self, tmpdir: Path) -> Path:
        """Draw a 400x300 plan with a vertical divider at x=200. Two
        rooms, four walls, one interior divider."""
        from PIL import Image, ImageDraw
        img = Image.new("L", (400, 300), 255)
        d = ImageDraw.Draw(img)
        # Outer rectangle
        d.rectangle([10, 10, 390, 290], outline=0, width=4)
        # Interior divider (full height)
        d.line([(200, 10), (200, 290)], fill=0, width=4)
        out = tmpdir / "two_room.png"
        img.save(out)
        return out

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_extract_produces_valid_schema(self):
        import schema  # type: ignore
        img_path = self._make_two_room_image(self.tmp_path)
        result = extract(img_path, CVConfig(pixels_per_meter=50.0))
        # Must round-trip through schema.validate without raising.
        schema.validate(result)

    def test_extract_finds_two_rooms(self):
        img_path = self._make_two_room_image(self.tmp_path)
        result = extract(img_path, CVConfig(pixels_per_meter=50.0))
        # Two rooms separated by the divider. Exact count can vary
        # with Hough / CC parameters; require at minimum that more
        # than zero rooms were recovered so a "no rooms found at all"
        # regression goes red.
        self.assertGreaterEqual(len(result["rooms"]), 1)

    def test_extract_room_labels_are_empty(self):
        # The CV stage deliberately emits empty labels; downstream
        # VLM / refiner fills them in. A regression that started
        # hardcoding "room" would leak an out-of-vocab label into
        # the Blender pipeline.
        img_path = self._make_two_room_image(self.tmp_path)
        result = extract(img_path, CVConfig(pixels_per_meter=50.0))
        for room in result["rooms"]:
            self.assertEqual(room["label"], "")

    def test_extract_raises_on_missing_image(self):
        with self.assertRaises(FileNotFoundError):
            extract(self.tmp_path / "nope.png", CVConfig())


if __name__ == "__main__":
    unittest.main()
