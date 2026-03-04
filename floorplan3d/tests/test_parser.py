"""
Tests for model output validation and post-processing.

Tests the inference post-processing logic (snap endpoints, polygon area,
nearest wall detection) without requiring a trained model.
"""

import math
import unittest


class TestSnapEndpoints(unittest.TestCase):
    """Test wall endpoint snapping logic."""

    @staticmethod
    def snap_endpoints(walls, threshold):
        """Standalone implementation for testing (mirrors inference.py)."""
        if len(walls) < 2:
            return walls

        for i in range(len(walls)):
            for j in range(i + 1, len(walls)):
                for end_a in ("start", "end"):
                    for end_b in ("start", "end"):
                        pa = walls[i][end_a]
                        pb = walls[j][end_b]
                        dist = ((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5
                        if dist < threshold and dist > 0:
                            mid = [
                                round((pa[0] + pb[0]) / 2, 3),
                                round((pa[1] + pb[1]) / 2, 3),
                            ]
                            walls[i][end_a] = mid
                            walls[j][end_b] = mid

        return walls

    def test_snaps_close_endpoints(self):
        walls = [
            {"start": [0, 0], "end": [5.01, 0], "thickness": 0.15},
            {"start": [5.0, 0], "end": [5.0, 4], "thickness": 0.15},
        ]
        snapped = self.snap_endpoints(walls, threshold=0.1)

        # Both endpoints should now match
        self.assertEqual(snapped[0]["end"], snapped[1]["start"])

    def test_does_not_snap_distant_endpoints(self):
        walls = [
            {"start": [0, 0], "end": [3, 0], "thickness": 0.15},
            {"start": [5, 0], "end": [5, 4], "thickness": 0.15},
        ]
        snapped = self.snap_endpoints(walls, threshold=0.1)

        # Endpoints should remain unchanged
        self.assertEqual(snapped[0]["end"], [3, 0])
        self.assertEqual(snapped[1]["start"], [5, 0])

    def test_single_wall_unchanged(self):
        walls = [{"start": [0, 0], "end": [5, 0], "thickness": 0.15}]
        snapped = self.snap_endpoints(walls, threshold=0.1)
        self.assertEqual(len(snapped), 1)
        self.assertEqual(snapped[0]["start"], [0, 0])


class TestPointToSegmentDistance(unittest.TestCase):
    """Test point-to-line-segment distance calculation."""

    @staticmethod
    def point_to_segment_distance(px, py, sx, sy, ex, ey):
        dx = ex - sx
        dy = ey - sy
        length_sq = dx * dx + dy * dy
        if length_sq < 1e-10:
            return ((px - sx) ** 2 + (py - sy) ** 2) ** 0.5

        t = max(0, min(1, ((px - sx) * dx + (py - sy) * dy) / length_sq))
        proj_x = sx + t * dx
        proj_y = sy + t * dy
        return ((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5

    def test_point_on_segment(self):
        dist = self.point_to_segment_distance(2.5, 0, 0, 0, 5, 0)
        self.assertAlmostEqual(dist, 0.0)

    def test_point_above_segment(self):
        dist = self.point_to_segment_distance(2.5, 1, 0, 0, 5, 0)
        self.assertAlmostEqual(dist, 1.0)

    def test_point_beyond_endpoint(self):
        dist = self.point_to_segment_distance(7, 0, 0, 0, 5, 0)
        self.assertAlmostEqual(dist, 2.0)

    def test_point_before_start(self):
        dist = self.point_to_segment_distance(-3, 0, 0, 0, 5, 0)
        self.assertAlmostEqual(dist, 3.0)

    def test_degenerate_segment(self):
        dist = self.point_to_segment_distance(3, 4, 0, 0, 0, 0)
        self.assertAlmostEqual(dist, 5.0)


class TestPolygonArea(unittest.TestCase):
    """Test polygon area calculation (shoelace formula)."""

    @staticmethod
    def polygon_area(polygon):
        n = len(polygon)
        if n < 3:
            return 0
        area = 0
        for i in range(n):
            j = (i + 1) % n
            area += polygon[i][0] * polygon[j][1]
            area -= polygon[j][0] * polygon[i][1]
        return abs(area) / 2.0

    def test_rectangle(self):
        polygon = [[0, 0], [4, 0], [4, 3], [0, 3]]
        self.assertAlmostEqual(self.polygon_area(polygon), 12.0)

    def test_triangle(self):
        polygon = [[0, 0], [4, 0], [0, 3]]
        self.assertAlmostEqual(self.polygon_area(polygon), 6.0)

    def test_unit_square(self):
        polygon = [[0, 0], [1, 0], [1, 1], [0, 1]]
        self.assertAlmostEqual(self.polygon_area(polygon), 1.0)

    def test_reversed_winding(self):
        """Area should be the same regardless of winding order."""
        polygon = [[0, 3], [4, 3], [4, 0], [0, 0]]
        self.assertAlmostEqual(self.polygon_area(polygon), 12.0)

    def test_too_few_points(self):
        self.assertEqual(self.polygon_area([[0, 0], [1, 1]]), 0)
        self.assertEqual(self.polygon_area([]), 0)


class TestNearestWallFinding(unittest.TestCase):
    """Test finding the nearest wall to a point."""

    @staticmethod
    def point_to_segment_distance(px, py, sx, sy, ex, ey):
        dx = ex - sx
        dy = ey - sy
        length_sq = dx * dx + dy * dy
        if length_sq < 1e-10:
            return ((px - sx) ** 2 + (py - sy) ** 2) ** 0.5
        t = max(0, min(1, ((px - sx) * dx + (py - sy) * dy) / length_sq))
        proj_x = sx + t * dx
        proj_y = sy + t * dy
        return ((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5

    def find_nearest_wall(self, x, y, walls):
        if not walls:
            return 0
        min_dist = float("inf")
        nearest_idx = 0
        for i, wall in enumerate(walls):
            sx, sy = wall["start"]
            ex, ey = wall["end"]
            dist = self.point_to_segment_distance(x, y, sx, sy, ex, ey)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        return nearest_idx

    def test_point_on_wall(self):
        walls = [
            {"start": [0, 0], "end": [5, 0]},
            {"start": [5, 0], "end": [5, 4]},
        ]
        self.assertEqual(self.find_nearest_wall(2.5, 0, walls), 0)

    def test_point_near_second_wall(self):
        walls = [
            {"start": [0, 0], "end": [5, 0]},
            {"start": [5, 0], "end": [5, 4]},
        ]
        self.assertEqual(self.find_nearest_wall(5, 2, walls), 1)

    def test_empty_walls(self):
        self.assertEqual(self.find_nearest_wall(0, 0, []), 0)


if __name__ == "__main__":
    unittest.main()
