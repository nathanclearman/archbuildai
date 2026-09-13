"""Tests for the CV post-pass that makes the outside shape match (api/cleanup.py)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "blender_addon" / "api"))
import cleanup  # type: ignore


def _plan():
    # One 10 x 8 building made of two rooms; walls: 4 real exterior walls,
    # one dimension-line wall 3 m outside, one wall that overhangs the
    # building by 4 m, and one tiny sliver room.
    return {
        "scale": {"pixels_per_meter": 50},
        "rooms": [
            {"label": "living_room", "polygon": [[0, 0], [6, 0], [6, 8], [0, 8]]},
            {"label": "bedroom", "polygon": [[6, 0], [10, 0], [10, 8], [6, 8]]},
            {"label": "room", "polygon": [[0, 8], [10, 8], [10, 8.1], [0, 8.1]]},  # sliver
        ],
        "walls": [
            {"start": [0, 0], "end": [10, 0], "thickness": 0.15},        # 0 real
            {"start": [10, 0], "end": [10, 8], "thickness": 0.15},       # 1 real
            {"start": [10, 8], "end": [0, 8], "thickness": 0.15},        # 2 real
            {"start": [0, 8], "end": [0, 0], "thickness": 0.15},         # 3 real
            {"start": [-3, 0], "end": [-3, 8], "thickness": 0.15},       # 4 dimension line, outside
            {"start": [0, 4], "end": [-4, 4], "thickness": 0.15},        # 5 leader: entirely outside (except endpoint)
            {"start": [-4, 8], "end": [10, 8], "thickness": 0.15},       # 6 overhangs 4 m past the corner
        ],
        "doors": [
            {"position": [5, 0], "width": 0.9, "wall_index": 0},
            {"position": [-3, 4], "width": 0.9, "wall_index": 4},        # on the dropped wall, outside
            {"position": [3, 8], "width": 0.9, "wall_index": 6},         # on the trimmed wall
        ],
        "windows": [
            {"position": [10, 4], "width": 1.2, "wall_index": 1},
        ],
    }


class TestClipWalls(unittest.TestCase):
    def test_outside_walls_dropped_and_overhang_trimmed(self):
        out, rep = cleanup.clip_walls_to_footprint(_plan(), margin=0.5, step=0.25)
        starts = [tuple(w["start"]) for w in out["walls"]]
        self.assertEqual(rep["dropped"], 2)
        self.assertEqual(rep["trimmed"], 1)
        self.assertEqual(len(out["walls"]), 5)
        self.assertNotIn((-3, 0), starts)
        overhang = out["walls"][-1]
        # cut back to the corner (within the margin, which covers wall thickness)
        self.assertAlmostEqual(overhang["start"][0], 0.0, delta=0.6)
        self.assertEqual(overhang["end"], [10, 8])

    def test_door_indices_follow_kept_walls(self):
        out, _ = cleanup.clip_walls_to_footprint(_plan(), margin=0.5, step=0.25)
        doors = out["doors"]
        self.assertEqual(len(doors), 2)                       # the outside door is gone
        self.assertEqual(doors[0]["wall_index"], 0)
        self.assertEqual(doors[1]["wall_index"], 4)           # old 6 -> new 4
        self.assertEqual(out["windows"][0]["wall_index"], 1)

    def test_input_not_mutated(self):
        plan = _plan()
        cleanup.clip_walls_to_footprint(plan, margin=0.5, step=0.25)
        self.assertEqual(len(plan["walls"]), 7)
        self.assertEqual(plan["doors"][2]["wall_index"], 6)

    def test_no_rooms_leaves_walls_alone(self):
        plan = _plan()
        plan["rooms"] = []
        out, rep = cleanup.clip_walls_to_footprint(plan)
        self.assertEqual(len(out["walls"]), 7)


class TestFillExterior(unittest.TestCase):
    def test_missing_left_wall_is_added_and_covered_edges_are_not(self):
        plan = _plan()
        plan["rooms"] = plan["rooms"][:2]
        plan["walls"] = [w for i, w in enumerate(plan["walls"]) if i in (0, 1, 2)]  # no left wall (x=0)
        out, added = cleanup.fill_missing_exterior_walls(plan, margin=0.5)
        self.assertEqual(added, 1)
        w = out["walls"][-1]
        self.assertEqual(w.get("source"), "footprint_fill")
        self.assertAlmostEqual(w["start"][0], 0.0, delta=0.6)
        self.assertAlmostEqual(w["end"][0], 0.0, delta=0.6)
        self.assertGreater(abs(w["end"][1] - w["start"][1]), 6.0)   # spans (most of) the 8 m side

    def test_closed_outline_adds_nothing(self):
        plan = _plan()
        plan["rooms"] = plan["rooms"][:2]
        plan["walls"] = plan["walls"][:4]
        out, added = cleanup.fill_missing_exterior_walls(plan, margin=0.5)
        self.assertEqual(added, 0)
        self.assertEqual(len(out["walls"]), 4)

    def test_clean_cv_plan_closes_the_building(self):
        plan = _plan()
        del plan["walls"][3]  # drop the real left wall; keep the outside junk
        out, rep = cleanup.clean_cv_plan(plan)
        self.assertEqual(rep["exterior_walls_added"], 1)


class TestSlivers(unittest.TestCase):
    def test_sliver_room_dropped(self):
        rooms, n = cleanup.drop_sliver_rooms(_plan()["rooms"])
        self.assertEqual(n, 1)
        self.assertEqual([r["label"] for r in rooms], ["living_room", "bedroom"])

    def test_clean_cv_plan_reports_both(self):
        out, rep = cleanup.clean_cv_plan(_plan())
        self.assertEqual(rep["sliver_rooms_dropped"], 1)
        self.assertEqual(rep["dropped"], 2)
        self.assertEqual(len(out["rooms"]), 2)


if __name__ == "__main__":
    unittest.main()
