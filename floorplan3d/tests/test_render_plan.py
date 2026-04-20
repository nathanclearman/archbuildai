"""Tests for the top-down eyeball-eval renderer."""

import io
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))

from render_plan import _bounds, render  # type: ignore  # noqa: E402


MINIMAL_PLAN = {
    "scale": {"pixels_per_meter": 50},
    "walls": [
        {"start": [0.0, 0.0], "end": [4.0, 0.0], "thickness": 0.15},
        {"start": [4.0, 0.0], "end": [4.0, 3.0], "thickness": 0.15},
        {"start": [4.0, 3.0], "end": [0.0, 3.0], "thickness": 0.15},
        {"start": [0.0, 3.0], "end": [0.0, 0.0], "thickness": 0.15},
    ],
    "doors": [{"position": [2.0, 0.0], "width": 0.9, "type": "hinged", "wall_index": 0}],
    "windows": [{"position": [0.0, 1.5], "width": 1.2, "wall_index": 3}],
    "rooms": [{"label": "bedroom",
               "polygon": [[0, 0], [4, 0], [4, 3], [0, 3]],
               "area": 12.0}],
}


class BoundsTest(unittest.TestCase):
    def test_returns_extents_across_walls_and_rooms(self):
        lo_x, lo_y, hi_x, hi_y = _bounds(MINIMAL_PLAN)
        self.assertEqual(lo_x, 0.0)
        self.assertEqual(lo_y, 0.0)
        self.assertEqual(hi_x, 4.0)
        self.assertEqual(hi_y, 3.0)

    def test_empty_plan_returns_nonzero_default(self):
        lo_x, lo_y, hi_x, hi_y = _bounds({})
        # Must be non-degenerate so the renderer doesn't divide by zero
        # when building the canvas.
        self.assertLess(lo_x, hi_x)
        self.assertLess(lo_y, hi_y)

    def test_room_only_plan_reports_room_extents(self):
        plan = {"rooms": [{"label": "r",
                           "polygon": [[1, 1], [5, 1], [5, 4], [1, 4]]}]}
        lo_x, lo_y, hi_x, hi_y = _bounds(plan)
        self.assertEqual((lo_x, lo_y, hi_x, hi_y), (1, 1, 5, 4))


class RenderTest(unittest.TestCase):
    def test_produces_rgba_image_with_padding(self):
        img = render(MINIMAL_PLAN, padding_px=20)
        # 4m wide * 50 ppm + 2*20 padding = 240; 3m tall * 50 + 40 = 190.
        self.assertEqual(img.size, (240, 190))
        self.assertEqual(img.mode, "RGBA")

    def test_saves_as_png_without_crashing(self):
        img = render(MINIMAL_PLAN)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        self.assertGreater(buf.tell(), 0)

    def test_handles_plan_with_no_rooms(self):
        plan = dict(MINIMAL_PLAN)
        plan["rooms"] = []
        img = render(plan)
        self.assertGreater(img.size[0], 0)
        self.assertGreater(img.size[1], 0)


if __name__ == "__main__":
    unittest.main()
