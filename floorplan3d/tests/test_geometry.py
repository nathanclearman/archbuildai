"""
Tests for the geometry generation module.

These tests validate the JSON-to-geometry pipeline without requiring Blender.
They test the pure computation functions and validate data structures.

To test the full Blender pipeline, run inside Blender's Python:
    blender --background --python tests/test_geometry_blender.py
"""

import json
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


# Mock bpy and related modules so tests can run outside Blender
sys.modules["bpy"] = MagicMock()
sys.modules["bmesh"] = MagicMock()
sys.modules["mathutils"] = MagicMock()

# Now we can import the geometry helpers that don't need real bpy
TESTS_DIR = Path(__file__).parent
SAMPLE_DIR = TESTS_DIR / "sample_plans"
PROJECT_DIR = TESTS_DIR.parent


class TestFloorPlanDataValidation(unittest.TestCase):
    """Test that sample floor plan data is valid."""

    def _load_sample(self, name):
        with open(SAMPLE_DIR / name, 'r') as f:
            return json.load(f)

    def test_simple_apartment_structure(self):
        data = self._load_sample("simple_apartment.json")

        self.assertIn("scale", data)
        self.assertIn("walls", data)
        self.assertIn("doors", data)
        self.assertIn("windows", data)
        self.assertIn("rooms", data)

        self.assertEqual(len(data["walls"]), 5)
        self.assertEqual(len(data["doors"]), 2)
        self.assertEqual(len(data["windows"]), 2)
        self.assertEqual(len(data["rooms"]), 2)

    def test_wall_data_format(self):
        data = self._load_sample("simple_apartment.json")

        for wall in data["walls"]:
            self.assertIn("start", wall)
            self.assertIn("end", wall)
            self.assertIn("thickness", wall)
            self.assertEqual(len(wall["start"]), 2)
            self.assertEqual(len(wall["end"]), 2)
            self.assertGreater(wall["thickness"], 0)

    def test_door_data_format(self):
        data = self._load_sample("simple_apartment.json")

        for door in data["doors"]:
            self.assertIn("position", door)
            self.assertIn("width", door)
            self.assertIn("wall_index", door)
            self.assertEqual(len(door["position"]), 2)
            self.assertGreater(door["width"], 0)
            self.assertLess(door["wall_index"], len(data["walls"]))

    def test_window_data_format(self):
        data = self._load_sample("simple_apartment.json")

        for window in data["windows"]:
            self.assertIn("position", window)
            self.assertIn("width", window)
            self.assertIn("wall_index", window)
            self.assertEqual(len(window["position"]), 2)
            self.assertGreater(window["width"], 0)
            self.assertLess(window["wall_index"], len(data["walls"]))

    def test_room_data_format(self):
        data = self._load_sample("simple_apartment.json")

        for room in data["rooms"]:
            self.assertIn("label", room)
            self.assertIn("polygon", room)
            self.assertGreaterEqual(len(room["polygon"]), 3)
            for point in room["polygon"]:
                self.assertEqual(len(point), 2)

    def test_studio_loads(self):
        data = self._load_sample("studio.json")
        self.assertEqual(len(data["walls"]), 6)
        self.assertEqual(len(data["rooms"]), 3)

    def test_wall_lengths_positive(self):
        data = self._load_sample("simple_apartment.json")

        for wall in data["walls"]:
            dx = wall["end"][0] - wall["start"][0]
            dy = wall["end"][1] - wall["start"][1]
            length = math.sqrt(dx * dx + dy * dy)
            self.assertGreater(length, 0, f"Wall has zero length: {wall}")


class TestWallGeometryHelpers(unittest.TestCase):
    """Test wall geometry computation functions."""

    def test_wall_direction_horizontal(self):
        # Import after mocking
        start = [0, 0]
        end = [5, 0]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx * dx + dy * dy)
        direction = (dx / length, dy / length)

        self.assertAlmostEqual(direction[0], 1.0)
        self.assertAlmostEqual(direction[1], 0.0)
        self.assertAlmostEqual(length, 5.0)

    def test_wall_direction_vertical(self):
        start = [3, 0]
        end = [3, 4]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx * dx + dy * dy)
        direction = (dx / length, dy / length)

        self.assertAlmostEqual(direction[0], 0.0)
        self.assertAlmostEqual(direction[1], 1.0)
        self.assertAlmostEqual(length, 4.0)

    def test_perpendicular(self):
        # Perpendicular of (1, 0) should be (0, 1)
        direction = (1.0, 0.0)
        perp = (-direction[1], direction[0])
        self.assertAlmostEqual(perp[0], 0.0)
        self.assertAlmostEqual(perp[1], 1.0)

    def test_polygon_area(self):
        """Test shoelace formula for polygon area."""
        # Simple 3x4 rectangle
        polygon = [[0, 0], [3, 0], [3, 4], [0, 4]]
        n = len(polygon)
        area = 0
        for i in range(n):
            j = (i + 1) % n
            area += polygon[i][0] * polygon[j][1]
            area -= polygon[j][0] * polygon[i][1]
        area = abs(area) / 2.0
        self.assertAlmostEqual(area, 12.0)


class TestMockModelOutput(unittest.TestCase):
    """Test the mock model output for development."""

    def test_mock_output_valid(self):
        # Manually load the mock data (can't import local_model due to bpy dep)
        mock_data = {
            "scale": {"pixels_per_meter": 50},
            "walls": [
                {"start": [0, 0], "end": [6, 0], "thickness": 0.15},
                {"start": [6, 0], "end": [6, 4], "thickness": 0.15},
                {"start": [6, 4], "end": [0, 4], "thickness": 0.15},
                {"start": [0, 4], "end": [0, 0], "thickness": 0.15},
                {"start": [3, 0], "end": [3, 4], "thickness": 0.1},
            ],
            "doors": [
                {"position": [3, 2], "width": 0.9, "type": "hinged", "wall_index": 4},
            ],
            "windows": [
                {"position": [4.5, 4], "width": 1.2, "wall_index": 2},
            ],
            "rooms": [
                {"label": "living_room", "polygon": [[3, 0], [6, 0], [6, 4], [3, 4]], "area": 12.0},
                {"label": "bedroom", "polygon": [[0, 0], [3, 0], [3, 4], [0, 4]], "area": 12.0},
            ],
        }

        # Validate structure
        self.assertIn("walls", mock_data)
        self.assertIn("doors", mock_data)
        self.assertIn("windows", mock_data)
        self.assertIn("rooms", mock_data)

        # Validate all door wall_indices are valid
        for door in mock_data["doors"]:
            self.assertLess(door["wall_index"], len(mock_data["walls"]))

        # Validate all window wall_indices are valid
        for window in mock_data["windows"]:
            self.assertLess(window["wall_index"], len(mock_data["walls"]))


class TestSignedPolygonArea(unittest.TestCase):
    """Shoelace signed-area helper used by _ensure_ccw. Positive means
    CCW in a standard math frame (y-up); Blender's world XY is the
    same frame so a CCW polygon gets a +Z face normal."""

    def setUp(self):
        # Lazy import after the bpy mocks at the top of this file are
        # in place — geometry.py imports bpy at module scope.
        sys.path.insert(0, str(PROJECT_DIR / "blender_addon"))
        import geometry  # type: ignore
        self.geometry = geometry

    def test_ccw_unit_square_has_positive_area(self):
        ccw = [[0, 0], [1, 0], [1, 1], [0, 1]]
        self.assertGreater(self.geometry._signed_polygon_area(ccw), 0)

    def test_cw_unit_square_has_negative_area(self):
        cw = [[0, 0], [0, 1], [1, 1], [1, 0]]
        self.assertLess(self.geometry._signed_polygon_area(cw), 0)

    def test_degenerate_polygon_has_zero_area(self):
        self.assertEqual(self.geometry._signed_polygon_area([]), 0.0)
        self.assertEqual(self.geometry._signed_polygon_area([[0, 0], [1, 1]]), 0.0)

    def test_magnitude_matches_unsigned_area(self):
        # 4x3 rect: signed area ±12 depending on winding; magnitude 12.
        ccw = [[0, 0], [4, 0], [4, 3], [0, 3]]
        cw = [[0, 0], [0, 3], [4, 3], [4, 0]]
        self.assertAlmostEqual(self.geometry._signed_polygon_area(ccw), 12.0)
        self.assertAlmostEqual(self.geometry._signed_polygon_area(cw), -12.0)


class TestEnsureCcw(unittest.TestCase):
    """Regression guard for the augmentation-flip winding bug: CW
    polygons from `flip_x=True` synth samples produced invisible floors
    and wrong-facing ceilings. _ensure_ccw normalizes the winding; this
    class pins the contract."""

    def setUp(self):
        sys.path.insert(0, str(PROJECT_DIR / "blender_addon"))
        import geometry  # type: ignore
        self.geometry = geometry

    def test_ccw_polygon_is_unchanged(self):
        ccw = [[0, 0], [1, 0], [1, 1], [0, 1]]
        self.assertEqual(self.geometry._ensure_ccw(ccw), ccw)

    def test_cw_polygon_is_reversed_to_ccw(self):
        cw = [[0, 0], [0, 1], [1, 1], [1, 0]]
        out = self.geometry._ensure_ccw(cw)
        # Output must be CCW (positive signed area).
        self.assertGreater(self.geometry._signed_polygon_area(out), 0)
        # Reversal, not a mutation of the input list.
        self.assertEqual(cw, [[0, 0], [0, 1], [1, 1], [1, 0]])

    def test_l_shape_cw_gets_reversed(self):
        # A 6-vertex L in CW order, the shape colonial_compartmentalized
        # and studio_apartment produce after the plan_to_schema merge
        # plus a horizontal-flip aug.
        cw_l = [[0, 0], [0, 4], [2, 4], [2, 2], [4, 2], [4, 0]]
        self.assertLess(self.geometry._signed_polygon_area(cw_l), 0)
        out = self.geometry._ensure_ccw(cw_l)
        self.assertGreater(self.geometry._signed_polygon_area(out), 0)


class TestWallIndexGuard(unittest.TestCase):
    """Doors and windows accept the schema's `wall_index=-1` sentinel
    (meaning "not attached to a wall") but the geometry layer must
    skip those, not silently route them through `walls[-1]` and cut
    the last wall in the list at a random position.
    """

    def test_negative_wall_index_skips_door(self):
        # Build a plan with one valid wall and one door that points at
        # wall_index=-1 (schema "no wall"). Running generate_door_
        # openings should produce 0 doors, not 1 cut through walls[-1].
        sys.path.insert(0, str(PROJECT_DIR / "blender_addon"))
        # geometry imports trigger the bpy mock; patch the bits that
        # would error on a mock call before we exercise the guard.
        import geometry  # type: ignore

        plan = {
            "walls": [
                {"start": [0, 0], "end": [10, 0], "thickness": 0.15},
            ],
            "doors": [
                {"position": [5, 0], "width": 0.9, "wall_index": -1},
            ],
            "windows": [],
            "rooms": [],
        }
        collection = MagicMock()
        # bpy.data.objects.get returns None → the modifier-apply branch
        # is skipped, so the guard is the only thing gating `count`.
        with patch.object(geometry.bpy.data.objects, "get", return_value=None):
            count = geometry.generate_door_openings(plan, collection, wall_height=2.7)
        self.assertEqual(count, 0)

    def test_negative_wall_index_skips_window(self):
        sys.path.insert(0, str(PROJECT_DIR / "blender_addon"))
        import geometry  # type: ignore

        plan = {
            "walls": [
                {"start": [0, 0], "end": [10, 0], "thickness": 0.15},
            ],
            "doors": [],
            "windows": [
                {"position": [5, 0], "width": 1.2, "wall_index": -1},
            ],
            "rooms": [],
        }
        collection = MagicMock()
        with patch.object(geometry.bpy.data.objects, "get", return_value=None):
            count = geometry.generate_window_openings(plan, collection, wall_height=2.7)
        self.assertEqual(count, 0)

    def test_out_of_range_wall_index_still_skipped(self):
        # Regression guard: the pre-fix check `wall_idx >= len(walls)`
        # also needs to hold. Don't let a refactor that rewrites the
        # guard drop the upper bound.
        sys.path.insert(0, str(PROJECT_DIR / "blender_addon"))
        import geometry  # type: ignore

        plan = {
            "walls": [
                {"start": [0, 0], "end": [10, 0], "thickness": 0.15},
            ],
            "doors": [
                {"position": [5, 0], "width": 0.9, "wall_index": 99},
            ],
            "windows": [],
            "rooms": [],
        }
        collection = MagicMock()
        with patch.object(geometry.bpy.data.objects, "get", return_value=None):
            count = geometry.generate_door_openings(plan, collection, wall_height=2.7)
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
