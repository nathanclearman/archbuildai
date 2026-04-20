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


class TestClampAlongWall(unittest.TestCase):
    """Regression guard for GEO-3. A door/window `position` that projects
    to a `dist_along` past either end of the wall must be clamped so the
    cutter fits inside [0, wall_length]. Without clamp, the boolean
    succeeds but cuts nothing — visibly silent failure."""

    def setUp(self):
        sys.path.insert(0, str(PROJECT_DIR / "blender_addon"))
        import geometry  # type: ignore
        self.geometry = geometry

    def test_in_range_passes_through(self):
        # door at 2.5m into a 5m wall, width 0.9 → half_w 0.45. Well inside.
        self.assertAlmostEqual(
            self.geometry._clamp_along_wall(2.5, 5.0, 0.45), 2.5
        )

    def test_overshoot_past_end_clamps_to_hi(self):
        # Projected 7m into a 5m wall, door half-width 0.45 → clamp to 4.55.
        self.assertAlmostEqual(
            self.geometry._clamp_along_wall(7.0, 5.0, 0.45), 4.55
        )

    def test_undershoot_before_start_clamps_to_lo(self):
        # Projected -1m → clamp to 0.45 (so cutter's leading edge is at 0).
        self.assertAlmostEqual(
            self.geometry._clamp_along_wall(-1.0, 5.0, 0.45), 0.45
        )

    def test_degenerate_wall_shorter_than_door_collapses(self):
        # 0.5m wall, 0.9m door → half_w 0.45, hi = max(0.45, 0.05) = 0.45,
        # range collapses to [0.45, 0.45]. Cutter will overshoot but still
        # at least intersect the wall — caller may pick to skip via
        # `dist_along > wall_length` check if stricter behaviour needed.
        self.assertAlmostEqual(
            self.geometry._clamp_along_wall(2.0, 0.5, 0.45), 0.45
        )


class TestProjectPositionToWall(unittest.TestCase):
    """Both the canonical `[x, y]` position form and the legacy scalar
    distance-along-wall form must be supported. The schema serializer
    always emits [x, y] but older sample fixtures use the scalar form.
    """

    def setUp(self):
        sys.path.insert(0, str(PROJECT_DIR / "blender_addon"))
        import geometry  # type: ignore
        self.geometry = geometry
        # The helper uses `mathutils.Vector`; the top-of-file mock makes
        # `Vector(p) - Vector(q)` return a MagicMock. To test the real
        # arithmetic, patch in a vector stand-in that supports subtraction
        # and `.dot`. A namedtuple-ish shim keeps the test readable.
        self._install_real_vector()

    def _install_real_vector(self):
        import geometry as geom  # type: ignore

        class V:
            def __init__(self, p):
                self.x, self.y = float(p[0]), float(p[1])

            def __sub__(self, o):
                return V((self.x - o.x, self.y - o.y))

            def dot(self, o):
                return self.x * o.x + self.y * o.y

        self._saved_vector = geom.Vector
        geom.Vector = V

    def tearDown(self):
        import geometry as geom  # type: ignore
        geom.Vector = self._saved_vector

    def test_xy_position_on_wall_returns_projection(self):
        # Wall from (0,0) along +x, position (3, 0) → dist 3.
        start = (0.0, 0.0)
        direction = self.geometry.Vector((1.0, 0.0))
        self.assertAlmostEqual(
            self.geometry._project_position_to_wall(start, direction, [3.0, 0.0]),
            3.0,
        )

    def test_xy_position_off_wall_projects(self):
        # Position (3, 2) on a +x wall projects to 3. Caller is expected
        # to clamp this; this helper only does the projection.
        start = (0.0, 0.0)
        direction = self.geometry.Vector((1.0, 0.0))
        self.assertAlmostEqual(
            self.geometry._project_position_to_wall(start, direction, [3.0, 2.0]),
            3.0,
        )

    def test_scalar_position_passes_through(self):
        start = (0.0, 0.0)
        direction = self.geometry.Vector((1.0, 0.0))
        self.assertAlmostEqual(
            self.geometry._project_position_to_wall(start, direction, 2.5),
            2.5,
        )

    def test_xy_position_on_offset_wall(self):
        # Wall from (5, 5) along +y, door at (5, 8) → dist 3 along wall.
        start = (5.0, 5.0)
        direction = self.geometry.Vector((0.0, 1.0))
        self.assertAlmostEqual(
            self.geometry._project_position_to_wall(start, direction, [5.0, 8.0]),
            3.0,
        )


class TestLabelSizeForPolygon(unittest.TestCase):
    """GEO-8: font size must scale with the room bbox so a 0.3m hardcode
    doesn't make labels illegible in a great room or overflow a
    hallway."""

    def setUp(self):
        sys.path.insert(0, str(PROJECT_DIR / "blender_addon"))
        import geometry  # type: ignore
        self.geometry = geometry

    def test_medium_room_returns_scaled_size(self):
        # 4x4 bedroom → minor 4m × ratio 0.12 = 0.48m. In-range, no clamp.
        poly = [[0, 0], [4, 0], [4, 4], [0, 4]]
        self.assertAlmostEqual(
            self.geometry._label_size_for_polygon(poly), 0.48
        )

    def test_huge_room_clamps_to_max(self):
        # 20x20 great room → 20 * 0.12 = 2.4m, clamped to max_size=1.0.
        poly = [[0, 0], [20, 0], [20, 20], [0, 20]]
        self.assertAlmostEqual(
            self.geometry._label_size_for_polygon(poly), 1.0
        )

    def test_tiny_room_clamps_to_min(self):
        # 1m hallway → 0.12m, clamped to min_size=0.2.
        poly = [[0, 0], [1, 0], [1, 5], [0, 5]]
        self.assertAlmostEqual(
            self.geometry._label_size_for_polygon(poly), 0.2
        )

    def test_nonrect_polygon_uses_bbox_minor_axis(self):
        # L-shape: bbox is 4 × 4, minor = 4 → 0.48m.
        poly = [[0, 0], [4, 0], [4, 2], [2, 2], [2, 4], [0, 4]]
        self.assertAlmostEqual(
            self.geometry._label_size_for_polygon(poly), 0.48
        )

    def test_empty_polygon_returns_min_size(self):
        self.assertEqual(self.geometry._label_size_for_polygon([]), 0.2)

    def test_custom_clamp_override(self):
        poly = [[0, 0], [4, 0], [4, 4], [0, 4]]
        # Override ratio so we can verify clamp + ratio independently.
        self.assertAlmostEqual(
            self.geometry._label_size_for_polygon(poly, ratio=0.5),
            1.0,  # 4*0.5 = 2.0, clamped to max_size=1.0
        )


class TestApplyCutterBoolean(unittest.TestCase):
    """GEO-4/5/6 regression guards. The helper must:
      - use `temp_override` (context override, not view_layer mutation)
      - remove the cutter on success
      - remove modifier AND cutter on modifier_apply failure (don't leak
        orphaned cutters, don't leave a corrupt modifier stack for the
        next cutter on the same wall)
    """

    def setUp(self):
        sys.path.insert(0, str(PROJECT_DIR / "blender_addon"))
        import geometry  # type: ignore
        self.geometry = geometry

    def _make_wall_and_cutter(self):
        """Build a pair of MagicMocks that look like wall + cutter objects
        the same way the generate_door_openings path creates them."""
        wall_obj = MagicMock(name="wall_obj")
        modifier = MagicMock(name="modifier")
        modifier.name = "Door_0"
        wall_obj.modifiers.new.return_value = modifier
        cutter_obj = MagicMock(name="cutter_obj")
        return wall_obj, cutter_obj, modifier

    def test_success_path_removes_cutter_returns_true(self):
        wall_obj, cutter_obj, modifier = self._make_wall_and_cutter()

        with patch.object(self.geometry.bpy.ops.object, "modifier_apply") as apply_op:
            apply_op.return_value = None
            # temp_override returns a context manager — the default
            # MagicMock supports __enter__/__exit__, so no extra patch.
            result = self.geometry._apply_cutter_boolean(wall_obj, cutter_obj, "Door_0")

        self.assertTrue(result)
        wall_obj.modifiers.new.assert_called_once_with(name="Door_0", type='BOOLEAN')
        apply_op.assert_called_once()
        self.geometry.bpy.data.objects.remove.assert_called_with(
            cutter_obj, do_unlink=True
        )

    def test_failure_path_rolls_back_modifier_and_cutter(self):
        wall_obj, cutter_obj, modifier = self._make_wall_and_cutter()

        # Reset the remove mock so we can assert on this call specifically.
        self.geometry.bpy.data.objects.remove.reset_mock()

        with patch.object(self.geometry.bpy.ops.object, "modifier_apply") as apply_op:
            apply_op.side_effect = RuntimeError("boolean solver failed")
            result = self.geometry._apply_cutter_boolean(wall_obj, cutter_obj, "Door_0")

        self.assertFalse(result)
        wall_obj.modifiers.remove.assert_called_once_with(modifier)
        # Cutter is still removed so the scene isn't cluttered with
        # orphaned cutter boxes referencing a removed modifier.
        self.geometry.bpy.data.objects.remove.assert_called_with(
            cutter_obj, do_unlink=True
        )

    def test_uses_temp_override_not_view_layer_mutation(self):
        """Guard against a refactor that drops `temp_override` and reverts
        to `bpy.context.view_layer.objects.active = wall_obj`. The
        mutation idiom is what GEO-4 is about: it leaks scene state."""
        wall_obj, cutter_obj, _ = self._make_wall_and_cutter()

        with patch.object(
            self.geometry.bpy.context, "temp_override"
        ) as temp_override:
            with patch.object(self.geometry.bpy.ops.object, "modifier_apply"):
                self.geometry._apply_cutter_boolean(wall_obj, cutter_obj, "Door_0")

        temp_override.assert_called_once()
        # The call should pass active_object — the operator key that
        # modifier_apply reads. `object=` is also accepted; assert at
        # least one of the expected kwargs appears.
        kwargs = temp_override.call_args.kwargs
        self.assertTrue(
            "active_object" in kwargs or "object" in kwargs,
            f"temp_override called without object context: {kwargs}",
        )


if __name__ == "__main__":
    unittest.main()
