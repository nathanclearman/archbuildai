"""Unit tests for the pure functions in synthesize.py.

Focuses on the helpers a senior would want pinned down with explicit
contracts: wall graph construction, opening snapping, swing-direction
math, augmentation transforms, and fixture placement. Template plans
themselves get a coverage / overlap / door-resolution sweep across many
seeds rather than per-template assertions, since template internals
intentionally vary.
"""

from __future__ import annotations

import math
import random
import sys
import unittest
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

import schema  # type: ignore
import synthesize  # type: ignore
from synthesize import (  # type: ignore
    Plan,
    Room,
    STYLES,
    TEMPLATES,
    _apply_augmentation,
    _build_walls,
    _fixture_rect,
    _pick_swing_target,
    _snap_door_to_wall,
    _swing_vector_into_room,
    _wall_unit_vector,
)


# ---------- _build_walls ----------

class BuildWallsTest(unittest.TestCase):
    def _two_room_plan(self):
        # Two rooms sharing a vertical edge at x=3.
        rooms = [
            Room("a", (0, 0, 3, 4)),
            Room("b", (3, 0, 3, 4)),
        ]
        return Plan(rooms=rooms, footprint=(6, 4))

    def test_no_overlapping_collinear_walls(self):
        plan = self._two_room_plan()
        walls = _build_walls(plan)
        # Collect collinear pairs and verify none overlap by more than eps.
        for a, b in combinations(walls, 2):
            sa, ea = a["start"], a["end"]
            sb, eb = b["start"], b["end"]
            horiz = abs(sa[1] - ea[1]) < 0.01 and abs(sb[1] - eb[1]) < 0.01 and abs(sa[1] - sb[1]) < 0.01
            vert = abs(sa[0] - ea[0]) < 0.01 and abs(sb[0] - eb[0]) < 0.01 and abs(sa[0] - sb[0]) < 0.01
            if horiz:
                lo1, hi1 = sorted([sa[0], ea[0]])
                lo2, hi2 = sorted([sb[0], eb[0]])
                self.assertFalse(lo1 < hi2 - 0.05 and lo2 < hi1 - 0.05,
                                 f"collinear horiz overlap: {a} {b}")
            elif vert:
                lo1, hi1 = sorted([sa[1], ea[1]])
                lo2, hi2 = sorted([sb[1], eb[1]])
                self.assertFalse(lo1 < hi2 - 0.05 and lo2 < hi1 - 0.05,
                                 f"collinear vert overlap: {a} {b}")

    def test_shared_edge_emitted_once(self):
        plan = self._two_room_plan()
        walls = _build_walls(plan)
        # The shared edge at x=3 from y=0 to y=4 should appear once.
        shared = [w for w in walls
                  if w["start"][0] == 3 and w["end"][0] == 3]
        self.assertEqual(len(shared), 1)
        s, e = shared[0]["start"], shared[0]["end"]
        self.assertEqual({s[1], e[1]}, {0, 4})

    def test_t_junction_subdivides_long_edge(self):
        # Room A's south edge spans [0, 6]; rooms B and C together tile the
        # north side at [0, 2] and [2, 6]. The naive "one wall per edge"
        # builder would emit three walls; the union+split builder must
        # produce two walls that tile [0, 6] without the redundant span.
        rooms = [
            Room("a", (0, 0, 6, 3)),
            Room("b", (0, 3, 2, 2)),
            Room("c", (2, 3, 4, 2)),
        ]
        plan = Plan(rooms=rooms, footprint=(6, 5))
        walls = _build_walls(plan)
        at_y3 = [w for w in walls if w["start"][1] == 3 and w["end"][1] == 3]
        # Exactly two walls along y=3: [0,2] and [2,6].
        self.assertEqual(len(at_y3), 2)
        spans = sorted([(w["start"][0], w["end"][0]) for w in at_y3])
        self.assertEqual(spans, [(0, 2), (2, 6)])


# ---------- _snap_door_to_wall ----------

class SnapDoorTest(unittest.TestCase):
    def test_snaps_to_longest_eligible_wall(self):
        # Two collinear walls on y=0: a short [0, 0.7] and a long [0, 5]
        # — both contain the probe, but only the long one can host a
        # 0.9 m door without overhang. The snapper must pick the long one.
        walls = [
            {"start": [0, 0], "end": [0.7, 0], "thickness": 0.15},
            {"start": [0, 0], "end": [5, 0], "thickness": 0.15},
        ]
        result = _snap_door_to_wall((0.4, 0), 0.9, walls)
        self.assertIsNotNone(result)
        idx, (px, py) = result
        self.assertEqual(idx, 1, "expected the long wall, not the short one")
        self.assertGreaterEqual(px, 0.45 - 0.01)
        self.assertLessEqual(px, 5 - 0.45 + 0.01)
        self.assertEqual(py, 0)

    def test_clamps_door_within_wall_endpoints(self):
        # A 0.9 m door declared at x=0.1 on a 1.5 m wall must shift right
        # so the full door fits inside the wall.
        walls = [{"start": [0, 0], "end": [1.5, 0], "thickness": 0.15}]
        result = _snap_door_to_wall((0.1, 0), 0.9, walls)
        self.assertIsNotNone(result)
        _, (px, _) = result
        self.assertGreaterEqual(px, 0.45 - 0.01)
        self.assertLessEqual(px, 1.5 - 0.45 + 0.01)

    def test_returns_none_when_no_wall_fits(self):
        walls = [{"start": [0, 0], "end": [0.5, 0], "thickness": 0.15}]
        result = _snap_door_to_wall((0.25, 0), 0.9, walls)
        self.assertIsNone(result)


# ---------- swing direction ----------

class SwingDirectionTest(unittest.TestCase):
    def test_circulation_loses_to_non_circulation(self):
        hall = Room("hallway", (0, 0, 1, 5))
        bath = Room("bathroom", (1, 0, 3, 5))
        self.assertIs(_pick_swing_target(hall, bath), bath)
        self.assertIs(_pick_swing_target(bath, hall), bath)

    def test_smaller_room_wins_among_non_circulation(self):
        big = Room("master_bedroom", (0, 0, 5, 5))
        small = Room("en_suite", (5, 0, 2, 2))
        self.assertIs(_pick_swing_target(big, small), small)

    def test_swing_vector_points_into_room(self):
        # Door on the shared horizontal edge at y=3 between two rooms.
        # The target rect is the south room — swing vector should point south.
        wall = {"start": [0, 3], "end": [4, 3], "thickness": 0.15}
        target = (0, 3, 4, 4)  # south of y=3
        vec = _swing_vector_into_room((2, 3), wall, target)
        self.assertAlmostEqual(vec[0], 0.0, places=5)
        self.assertGreater(vec[1], 0.5)  # positive y = south


# ---------- _apply_augmentation ----------

class AugmentationTest(unittest.TestCase):
    def setUp(self):
        rng = random.Random(0)
        plan = synthesize.ranch_open_concept(rng)
        self.base = synthesize.plan_to_schema(plan, rng)

    def test_identity_when_rot_zero_and_no_flip(self):
        # generate_one short-circuits this case; the helper itself returns
        # the input unchanged for safety.
        out = _apply_augmentation(self.base, 0, False)
        self.assertIs(out, self.base)

    def test_preserves_coverage_under_all_orientations(self):
        for rot_k in range(4):
            for flip in (False, True):
                if rot_k == 0 and not flip:
                    continue
                aug = _apply_augmentation(self.base, rot_k, flip)
                # Same number of walls/doors/windows/rooms.
                for key in ("walls", "doors", "windows", "rooms"):
                    self.assertEqual(len(aug[key]), len(self.base[key]),
                                     f"key={key} rot={rot_k} flip={flip}")

    def test_rotates_swing_into_vector(self):
        # A door's swing vector should rotate with the plan, otherwise
        # the arc would render swinging out of the wrong side.
        for d in self.base["doors"]:
            self.assertIn("swing_into", d)
        rot1 = _apply_augmentation(self.base, 1, False)
        for orig, rotated in zip(self.base["doors"], rot1["doors"]):
            ox, oy = orig["swing_into"]
            rx, ry = rotated["swing_into"]
            # 90 deg CCW (visual): (x, y) -> (y, -x). After our transform
            # the rotation is CW math = CCW visual, so (x, y) -> (y, -x).
            self.assertAlmostEqual(rx, oy, places=5)
            self.assertAlmostEqual(ry, -ox, places=5)

    def test_flip_x_preserves_unit_length_of_swing(self):
        flipped = _apply_augmentation(self.base, 0, True)
        for d in flipped["doors"]:
            sx, sy = d["swing_into"]
            self.assertAlmostEqual(math.hypot(sx, sy), 1.0, places=5)


# ---------- _fixture_rect ----------

class FixtureRectTest(unittest.TestCase):
    def test_corner_anchor_pins_to_corner(self):
        r = _fixture_rect("NW", 0, 0, 5, 5, along_m=1.0, depth_m=1.0, inset=0.1)
        self.assertIsNotNone(r)
        x, y, w, h = r
        self.assertAlmostEqual(x, 0.1)
        self.assertAlmostEqual(y, 0.1)

    def test_wall_center_anchor_centers_along_axis(self):
        # N anchor on a 6 m wide room with 1.0 m fixture along the wall:
        # x should be (6 - 1.0) / 2 = 2.5 (after inset adjustment).
        r = _fixture_rect("N", 0, 0, 6, 4, along_m=1.0, depth_m=0.5, inset=0.0)
        self.assertIsNotNone(r)
        x, y, w, h = r
        self.assertAlmostEqual(x, 2.5)
        self.assertAlmostEqual(y, 0.0)

    def test_returns_none_when_too_big(self):
        self.assertIsNone(_fixture_rect("NW", 0, 0, 1, 1, along_m=2, depth_m=0.5))


# ---------- end-to-end template sanity ----------

class TemplateIntegrityTest(unittest.TestCase):
    def test_every_template_seed_validates(self):
        for fn in TEMPLATES:
            for seed in range(20):
                rng = random.Random(seed)
                plan = fn(rng)
                d = synthesize.plan_to_schema(plan, rng)
                schema.validate(d)
                schema.deserialize(schema.serialize(d))

    def test_no_overlapping_walls_in_any_template(self):
        for fn in TEMPLATES:
            for seed in range(10):
                rng = random.Random(seed)
                plan = fn(rng)
                d = synthesize.plan_to_schema(plan, rng)
                for a, b in combinations(d["walls"], 2):
                    sa, ea = a["start"], a["end"]
                    sb, eb = b["start"], b["end"]
                    horiz = (abs(sa[1] - ea[1]) < 0.01
                             and abs(sb[1] - eb[1]) < 0.01
                             and abs(sa[1] - sb[1]) < 0.01)
                    vert = (abs(sa[0] - ea[0]) < 0.01
                            and abs(sb[0] - eb[0]) < 0.01
                            and abs(sa[0] - sb[0]) < 0.01)
                    if horiz:
                        lo1, hi1 = sorted([sa[0], ea[0]])
                        lo2, hi2 = sorted([sb[0], eb[0]])
                        self.assertFalse(
                            lo1 < hi2 - 0.05 and lo2 < hi1 - 0.05,
                            f"{fn.__name__} seed {seed}: overlap {a} {b}")
                    elif vert:
                        lo1, hi1 = sorted([sa[1], ea[1]])
                        lo2, hi2 = sorted([sb[1], eb[1]])
                        self.assertFalse(
                            lo1 < hi2 - 0.05 and lo2 < hi1 - 0.05,
                            f"{fn.__name__} seed {seed}: overlap {a} {b}")

    def test_doors_fit_their_walls(self):
        for fn in TEMPLATES:
            for seed in range(10):
                rng = random.Random(seed)
                plan = fn(rng)
                d = synthesize.plan_to_schema(plan, rng)
                for door in d["doors"]:
                    w = d["walls"][door["wall_index"]]
                    sx, sy = w["start"]
                    ex, ey = w["end"]
                    wlen = math.hypot(ex - sx, ey - sy)
                    if wlen < 1e-6:
                        continue
                    tx, ty = (ex - sx) / wlen, (ey - sy) / wlen
                    px, py = door["position"]
                    t = (px - sx) * tx + (py - sy) * ty
                    half = door["width"] / 2
                    self.assertGreaterEqual(t - half, -0.05,
                                            f"{fn.__name__} door past start")
                    self.assertLessEqual(t + half, wlen + 0.05,
                                         f"{fn.__name__} door past end")

    def test_swing_into_present_on_every_door(self):
        for fn in TEMPLATES:
            rng = random.Random(0)
            plan = fn(rng)
            d = synthesize.plan_to_schema(plan, rng)
            for door in d["doors"]:
                self.assertIn("swing_into", door, fn.__name__)
                sx, sy = door["swing_into"]
                self.assertAlmostEqual(math.hypot(sx, sy), 1.0, places=4,
                                       msg=f"{fn.__name__} swing not unit")


# ---------- generate_one (orchestration) ----------

class GenerateOneTest(unittest.TestCase):
    def test_augment_off_is_deterministic(self):
        a, _ = synthesize.generate_one(42, augment=False)
        b, _ = synthesize.generate_one(42, augment=False)
        self.assertEqual(a.tobytes(), b.tobytes())

    def test_augment_on_produces_valid_json(self):
        for seed in range(20):
            _, plan = synthesize.generate_one(seed)
            schema.validate(plan)
            schema.deserialize(schema.serialize(plan))


if __name__ == "__main__":
    unittest.main()
