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
    BayWindow,
    Plan,
    Room,
    STYLES,
    TEMPLATES,
    _apply_augmentation,
    _apply_bay_to_walls,
    _bay_corners,
    _build_walls,
    _filter_internal_walls,
    _fixture_rect,
    _maybe_attach_bay,
    _pick_swing_target,
    _room_polygon_with_bays,
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


# ---------- _filter_internal_walls ----------

class FilterInternalWallsTest(unittest.TestCase):
    def test_drops_wall_between_same_label_rects(self):
        # Two "main_room" rectangles stacked vertically: shared edge at
        # y=3 is internal and should be filtered out.
        plan = Plan(rooms=[
            Room("main_room", (0, 0, 4, 3)),
            Room("main_room", (0, 3, 4, 2)),
        ], footprint=(4, 5))
        walls = _build_walls(plan)
        filtered = _filter_internal_walls(walls, plan)
        # Internal wall at y=3 should be gone.
        for w in filtered:
            sx, sy = w["start"]
            ex, ey = w["end"]
            self.assertFalse(
                abs(sy - 3) < 0.01 and abs(ey - 3) < 0.01 and 0 < sx < 4 and 0 < ex < 4,
                f"internal wall not filtered: {w}",
            )

    def test_keeps_wall_between_distinct_labels(self):
        # Same geometry but distinct labels — wall must stay.
        plan = Plan(rooms=[
            Room("kitchen", (0, 0, 4, 3)),
            Room("dining_room", (0, 3, 4, 2)),
        ], footprint=(4, 5))
        walls = _build_walls(plan)
        filtered = _filter_internal_walls(walls, plan)
        on_y3 = [w for w in filtered
                 if abs(w["start"][1] - 3) < 0.01 and abs(w["end"][1] - 3) < 0.01]
        self.assertTrue(on_y3, "shared edge between distinct rooms should survive")

    def test_keeps_exterior_walls(self):
        # Exterior walls have only one neighbour; they must always be kept.
        plan = Plan(rooms=[Room("main_room", (0, 0, 4, 4))], footprint=(4, 4))
        walls = _build_walls(plan)
        filtered = _filter_internal_walls(walls, plan)
        self.assertEqual(len(filtered), 4, "single room should keep all 4 exterior walls")


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

    def test_augment_on_is_seed_deterministic(self):
        # Same seed -> same pixels. Photometric aug must not introduce
        # any RNG source the public seed can't reproduce.
        a, _ = synthesize.generate_one(11, augment=True)
        b, _ = synthesize.generate_one(11, augment=True)
        self.assertEqual(a.tobytes(), b.tobytes())


# ---------- photometric augmentation ----------

class PhotometricAugTest(unittest.TestCase):
    def _white(self, n=64):
        from PIL import Image
        return Image.new("RGB", (n, n), (255, 255, 255))

    def test_paper_tint_shifts_white(self):
        # Every entry in PAPER_TINTS has at least one channel gain < 1,
        # so a pure-white input must come back with a non-pure-white pixel.
        rng = random.Random(0)
        out = synthesize._apply_paper_tint(self._white(), rng)
        px = out.getpixel((0, 0))
        self.assertLess(min(px), 255)
        # ...but stay within a narrow band — tint, not posterize.
        self.assertGreater(min(px), 200)

    def test_jpeg_roundtrip_preserves_shape(self):
        rng = random.Random(0)
        img = self._white(128)
        out = synthesize._apply_jpeg_roundtrip(img, rng)
        self.assertEqual(out.size, img.size)
        self.assertEqual(out.mode, "RGB")

    def test_small_skew_preserves_shape(self):
        rng = random.Random(3)
        img = self._white(128)
        out = synthesize._apply_small_skew(img, rng, bg_color=(255, 255, 255))
        self.assertEqual(out.size, img.size)
        self.assertEqual(out.mode, "RGB")

    def test_grayscale_collapses_chroma(self):
        # A red-only RGB input should come back with R == G == B after
        # the grayscale pass (luminance conversion).
        from PIL import Image
        red = Image.new("RGB", (4, 4), (200, 20, 20))
        out = synthesize._apply_grayscale(red)
        r, g, b = out.getpixel((0, 0))
        self.assertEqual(r, g)
        self.assertEqual(g, b)
        self.assertEqual(out.mode, "RGB")  # not "L" — downstream expects RGB

    def test_augment_image_is_rng_deterministic(self):
        # Same random.Random state should produce byte-identical output.
        img = self._white()
        out1 = synthesize._augment_image(img.copy(), random.Random(7))
        out2 = synthesize._augment_image(img.copy(), random.Random(7))
        self.assertEqual(out1.tobytes(), out2.tobytes())

    def test_augment_image_different_seeds_differ(self):
        # Sanity: a different RNG should usually produce different pixels.
        img = self._white()
        out1 = synthesize._augment_image(img.copy(), random.Random(1))
        out2 = synthesize._augment_image(img.copy(), random.Random(2))
        self.assertNotEqual(out1.tobytes(), out2.tobytes())


# ---------- US dimension callouts ----------

class DimensionFormatTest(unittest.TestCase):
    def test_exact_feet(self):
        # 3.048 m == 10'0" exactly (10 ft = 3.048 m).
        self.assertEqual(synthesize._metric_to_ft_in(3.048), "10'0\"")

    def test_inch_rollover(self):
        # 3.81 m = 12.5 ft -> 12'6". Regression case: make sure we don't
        # ever emit "12'12"" when the inch fraction rounds up to 12.
        out = synthesize._metric_to_ft_in(3.81)
        self.assertEqual(out, "12'6\"")
        # And a length that rounds to exactly 12" -> bumps to next foot.
        out = synthesize._metric_to_ft_in(3.657)  # 11.998 ft
        self.assertFalse("'12\"" in out, f"leaked 12\": {out}")

    def test_zero_is_zero(self):
        self.assertEqual(synthesize._metric_to_ft_in(0.0), "0'0\"")

    def test_negative_clamps_to_zero(self):
        # Shouldn't happen in practice, but don't crash.
        self.assertEqual(synthesize._metric_to_ft_in(-2.0), "0'0\"")


# ---------- overlays ----------

class OverlayTest(unittest.TestCase):
    def setUp(self):
        self.cfg = synthesize.SynthConfig(image_size=300)
        _, self.plan = synthesize.generate_one(5, self.cfg, augment=False)

    def test_title_block_does_not_change_size(self):
        img = synthesize.render(self.plan, self.cfg, title_block="MAIN LEVEL")
        base = synthesize.render(self.plan, self.cfg)
        self.assertEqual(img.size, base.size)
        # Bottom-left corner should differ: the title banner sits there.
        self.assertNotEqual(img.getpixel((10, 290)), base.getpixel((10, 290)))

    def test_watermark_does_not_change_size(self):
        img = synthesize.render(self.plan, self.cfg, watermark="DRAFT")
        base = synthesize.render(self.plan, self.cfg)
        self.assertEqual(img.size, base.size)
        # Most pixels should differ somewhere because the watermark tiles
        # across the canvas.
        self.assertNotEqual(img.tobytes(), base.tobytes())

    def test_render_stays_deterministic(self):
        # Render must be pure given its inputs — overlays included.
        a = synthesize.render(self.plan, self.cfg, show_dimensions=True,
                              title_block="FLOOR PLAN", watermark="DRAFT")
        b = synthesize.render(self.plan, self.cfg, show_dimensions=True,
                              title_block="FLOOR PLAN", watermark="DRAFT")
        self.assertEqual(a.tobytes(), b.tobytes())


# ---------- bay windows ----------

class BayWindowGeometryTest(unittest.TestCase):
    def test_north_bay_corners_protrude_outward(self):
        # Room at (0, 0, 6, 4). A bay on N with center_t=0.5, base=2.4,
        # depth=0.75, top_ratio=0.55 should produce:
        #   base_lo (1.8, 0)
        #   top_lo  (2.34, -0.75)
        #   top_hi  (3.66, -0.75)
        #   base_hi (4.2, 0)
        rect = (0.0, 0.0, 6.0, 4.0)
        bay = BayWindow("great_room", "N", 0.5, 2.4, 0.75, 0.55)
        base_lo, top_lo, top_hi, base_hi = _bay_corners(rect, bay)
        self.assertAlmostEqual(base_lo[0], 1.8, places=3)
        self.assertAlmostEqual(base_lo[1], 0.0, places=3)
        self.assertAlmostEqual(top_lo[0],  2.34, places=3)
        self.assertAlmostEqual(top_lo[1], -0.75, places=3)
        self.assertAlmostEqual(top_hi[0],  3.66, places=3)
        self.assertAlmostEqual(top_hi[1], -0.75, places=3)
        self.assertAlmostEqual(base_hi[0], 4.2, places=3)
        self.assertAlmostEqual(base_hi[1], 0.0, places=3)

    def test_east_bay_protrudes_east(self):
        rect = (0.0, 0.0, 4.0, 6.0)
        bay = BayWindow("living_room", "E", 0.5, 2.4, 0.75, 0.55)
        base_lo, top_lo, top_hi, base_hi = _bay_corners(rect, bay)
        # Base corners stay on x=4 (the east wall).
        self.assertAlmostEqual(base_lo[0], 4.0, places=3)
        self.assertAlmostEqual(base_hi[0], 4.0, places=3)
        # Top corners protrude east (x > 4).
        self.assertGreater(top_lo[0], 4.0)
        self.assertGreater(top_hi[0], 4.0)


class ApplyBayToWallsTest(unittest.TestCase):
    def test_replaces_host_segment_with_three_walls(self):
        # Single rect room 6x4, walls built naturally.
        plan = Plan(rooms=[Room("great_room", (0, 0, 6, 4))], footprint=(6, 4))
        walls = _build_walls(plan)
        n_before = len(walls)
        bay = BayWindow("great_room", "N", 0.5, 2.4, 0.75, 0.55)
        walls_after = _apply_bay_to_walls(walls, (0, 0, 6, 4), bay, thickness=0.15)
        # N wall was one segment spanning [0,6]. It should now be two
        # skirt walls (0..1.8 and 4.2..6) plus three new bay walls —
        # so the total count rises from n_before to n_before+4
        # (one removed, five added).
        self.assertEqual(len(walls_after), n_before + 4)

    def test_does_not_touch_walls_on_other_sides(self):
        plan = Plan(rooms=[Room("great_room", (0, 0, 6, 4))], footprint=(6, 4))
        walls = _build_walls(plan)
        bay = BayWindow("great_room", "N", 0.5, 2.4, 0.75, 0.55)
        walls_after = _apply_bay_to_walls(walls, (0, 0, 6, 4), bay, thickness=0.15)
        # S / E / W walls should survive untouched.
        def present(wanted_start, wanted_end):
            for w in walls_after:
                if (w["start"] == list(wanted_start) and w["end"] == list(wanted_end)) \
                   or (w["start"] == list(wanted_end) and w["end"] == list(wanted_start)):
                    return True
            return False
        self.assertTrue(present((0, 4), (6, 4)))  # south
        self.assertTrue(present((0, 0), (0, 4)))  # west
        self.assertTrue(present((6, 0), (6, 4)))  # east

    def test_top_wall_is_parallel_to_host_side(self):
        rect = (0, 0, 6, 4)
        bay = BayWindow("great_room", "N", 0.5, 2.4, 0.75, 0.55)
        plan = Plan(rooms=[Room("great_room", rect)], footprint=(6, 4))
        walls = _apply_bay_to_walls(_build_walls(plan), rect, bay, thickness=0.15)
        # Top wall on N bay lies at y = -depth and is horizontal.
        tops = [w for w in walls
                if abs(w["start"][1] + 0.75) < 1e-3 and abs(w["end"][1] + 0.75) < 1e-3]
        self.assertEqual(len(tops), 1)


class RoomPolygonWithBaysTest(unittest.TestCase):
    def test_rect_polygon_extended_with_bay_vertices(self):
        rect = (0, 0, 6, 4)
        bay = BayWindow("r", "N", 0.5, 2.4, 0.75, 0.55)
        poly = _room_polygon_with_bays(rect, [bay])
        # 4 original corners + 4 bay vertices = 8.
        self.assertEqual(len(poly), 8)
        # Every bay corner must appear.
        bay_pts = [list(p) for p in _bay_corners(rect, bay)]
        for p in bay_pts:
            self.assertIn([round(p[0], 3), round(p[1], 3)], poly)

    def test_polygon_min_bound_shifts_for_north_bay(self):
        # A north bay makes the polygon extend upward (smaller y).
        rect = (0, 0, 6, 4)
        bay = BayWindow("r", "N", 0.5, 2.4, 0.75, 0.55)
        poly = _room_polygon_with_bays(rect, [bay])
        min_y = min(p[1] for p in poly)
        self.assertAlmostEqual(min_y, -0.75, places=3)


class PlanToSchemaBayTest(unittest.TestCase):
    def test_bay_grows_wall_and_window_counts(self):
        rect = (0, 0, 8, 5)
        bay = BayWindow("great_room", "S", 0.5, 2.6, 0.7, 0.55)
        plan_no_bay = Plan(rooms=[Room("great_room", rect)], footprint=(8, 5))
        plan_bay = Plan(rooms=[Room("great_room", rect)], footprint=(8, 5), bays=[bay])
        rng = random.Random(0)
        d_no = synthesize.plan_to_schema(plan_no_bay, rng)
        rng = random.Random(0)
        d_yes = synthesize.plan_to_schema(plan_bay, rng)
        # More walls (5 - 1 = 4 extra) and at least three more windows
        # (pre-placed bay windows).
        self.assertGreaterEqual(len(d_yes["walls"]) - len(d_no["walls"]), 3)
        self.assertGreaterEqual(len(d_yes["windows"]) - len(d_no["windows"]), 3)

    def test_bay_room_area_grows(self):
        rect = (0, 0, 8, 5)
        bay = BayWindow("great_room", "S", 0.5, 2.6, 0.7, 0.55)
        plan = Plan(rooms=[Room("great_room", rect)], footprint=(8, 5), bays=[bay])
        d = synthesize.plan_to_schema(plan, random.Random(0))
        base_area = 8 * 5
        # Trapezoid: ½ · (base + top) · depth = ½ · (2.6 + 2.6*0.55) · 0.7
        trap_area = 0.5 * (2.6 + 2.6 * 0.55) * 0.7
        self.assertAlmostEqual(d["rooms"][0]["area"], round(base_area + trap_area, 2),
                               places=2)

    def test_plan_to_schema_bay_validates(self):
        rect = (0, 0, 8, 5)
        bay = BayWindow("great_room", "E", 0.5, 2.2, 0.6, 0.55)
        plan = Plan(rooms=[Room("great_room", rect)], footprint=(8, 5), bays=[bay])
        d = synthesize.plan_to_schema(plan, random.Random(0))
        schema.validate(d)
        schema.deserialize(schema.serialize(d))


class MaybeAttachBayTest(unittest.TestCase):
    def test_no_op_when_roll_fails(self):
        # Seeding the rng so rng.random() yields a value >= P_BAY
        # leaves plan.bays empty. Find such a seed.
        for seed in range(200):
            rng = random.Random(seed)
            if rng.random() >= synthesize.P_BAY:
                rng = random.Random(seed)
                plan = synthesize.ranch_open_concept(rng)
                _maybe_attach_bay(plan, rng)
                self.assertEqual(plan.bays, [], f"seed {seed} should not attach")
                return
        self.fail("no failing roll found — P_BAY suspiciously high")

    def test_attached_bay_targets_eligible_room(self):
        # Force a success roll by patching rng. Drive with a seed where
        # the first rng.random() returns < P_BAY.
        for seed in range(200):
            rng = random.Random(seed)
            plan = synthesize.ranch_open_concept(rng)
            _maybe_attach_bay(plan, rng)
            if plan.bays:
                b = plan.bays[0]
                self.assertIn(b.room_label, synthesize._BAY_ELIGIBLE_ROOMS)
                self.assertIn(b.side, ("N", "S", "E", "W"))
                self.assertGreaterEqual(b.base_width, 2.2)
                self.assertLessEqual(b.base_width, 3.0)
                return
        self.fail("no successful roll in 200 seeds — P_BAY suspiciously low")

    def test_generate_one_with_bay_still_validates(self):
        # Sweep enough seeds that at least one hits P_BAY and produces a
        # bay-laden plan; every output must still validate.
        found_bay = False
        for seed in range(60):
            _, plan = synthesize.generate_one(seed)
            schema.validate(plan)
            schema.deserialize(schema.serialize(plan))
            # Rough detection: a bay bumps the polygon vertex count > 4.
            if any(len(r["polygon"]) > 4 for r in plan["rooms"]):
                found_bay = True
        self.assertTrue(found_bay, "no bay ever attached in 60 seeds")


if __name__ == "__main__":
    unittest.main()
