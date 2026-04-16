"""End-to-end test for the CubiCasa5k loader.

We can't ship the 5-GB CubiCasa corpus in the repo, so `fixtures/cubicasa_mini`
is a hand-crafted SVG that reproduces the structural quirks the real dataset
relies on: nested `<g transform="translate(...)">` wrappers around walls /
rooms / openings, mixed `<polygon>` / `<rect>` / `<path>` geometry, CubiCasa
room-class strings. The loader has to compose ancestor transforms and
normalize everything into the canonical schema — a regression there would
silently produce off-by-translate coordinates on the real dataset.

Expected layout (metric, @ 50 px/m):
  2-room plate 4.0 m x 3.0 m
  living_room  rect (0.2, 0.2) .. (1.9, 2.8)
  bedroom      rect (2.1, 0.2) .. (3.8, 2.8)
  door         at  (2.0, 1.5)     (centered on interior divider)
  window       at  (1.2, 0.1)     (on top exterior wall)
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

import schema  # type: ignore
from dataset import (  # type: ignore
    CubiCasaLoader,
    RealMLSLoader,
    _apply,
    _compose,
    _IDENTITY,
    _parse_transform,
    build_eval_set,
    build_training_set,
)


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "cubicasa_mini"
REAL_MLS_ROOT = Path(__file__).parent / "fixtures" / "real_mls"


class CubiCasaLoaderTest(unittest.TestCase):
    def setUp(self):
        samples = list(CubiCasaLoader(FIXTURE_ROOT))
        self.assertEqual(len(samples), 1, "fixture should yield one sample")
        self.sample = samples[0]
        self.plan = json.loads(self.sample.target_json)

    def test_sample_metadata(self):
        self.assertEqual(self.sample.source, "cubicasa5k")
        self.assertTrue(self.sample.image_path.exists())
        self.assertEqual(self.sample.image_path.name, "F1_scaled.png")

    def test_output_validates_against_schema(self):
        schema.validate(self.plan)
        # Round-trip through the canonical serializer too — catches any
        # field the loader emits that the trainer can't accept.
        schema.deserialize(schema.serialize(self.plan))

    def test_wall_count_and_coordinates(self):
        walls = self.plan["walls"]
        self.assertEqual(len(walls), 5)

        def span(w):
            return (tuple(w["start"]), tuple(w["end"]))

        # Sort by (start_x, start_y, end_x, end_y) so the comparison is
        # stable regardless of traversal order.
        got = sorted(span(w) for w in walls)
        want = sorted([
            ((0.0, 2.9), (4.0, 2.9)),   # bottom exterior
            ((0.0, 0.1), (4.0, 0.1)),   # top exterior
            ((0.1, 0.0), (0.1, 3.0)),   # left  (under nested translate)
            ((3.9, 0.0), (3.9, 3.0)),   # right (under nested translate)
            ((2.0, 0.2), (2.0, 2.8)),   # interior divider (under nested translate)
        ])
        for (gs, ge), (ws, we) in zip(got, want):
            self.assertAlmostEqual(gs[0], ws[0], places=2)
            self.assertAlmostEqual(gs[1], ws[1], places=2)
            self.assertAlmostEqual(ge[0], we[0], places=2)
            self.assertAlmostEqual(ge[1], we[1], places=2)

    def test_door_position_uses_composed_transform(self):
        # Without transform composition the door would land at (0.2, 0.2)
        # instead of (2.0, 1.5) — that is the regression this case guards.
        doors = self.plan["doors"]
        self.assertEqual(len(doors), 1)
        self.assertAlmostEqual(doors[0]["position"][0], 2.0, places=2)
        self.assertAlmostEqual(doors[0]["position"][1], 1.5, places=2)

    def test_door_snaps_to_interior_divider(self):
        # The nearest wall to (2.0, 1.5) is the interior divider at x=2.0.
        door = self.plan["doors"][0]
        wall = self.plan["walls"][door["wall_index"]]
        self.assertAlmostEqual(wall["start"][0], 2.0, places=2)
        self.assertAlmostEqual(wall["end"][0], 2.0, places=2)

    def test_window_on_top_wall(self):
        windows = self.plan["windows"]
        self.assertEqual(len(windows), 1)
        self.assertAlmostEqual(windows[0]["position"][0], 1.2, places=2)
        self.assertAlmostEqual(windows[0]["position"][1], 0.1, places=2)

    def test_room_labels_and_polygons(self):
        labels = sorted(r["label"] for r in self.plan["rooms"])
        self.assertEqual(labels, ["bedroom", "living_room"])

        by_label = {r["label"]: r for r in self.plan["rooms"]}
        living = by_label["living_room"]["polygon"]
        # Closed rect should have 4 distinct corners spanning (0.2..1.9, 0.2..2.8)
        xs = sorted({round(p[0], 2) for p in living})
        ys = sorted({round(p[1], 2) for p in living})
        self.assertEqual(xs, [0.2, 1.9])
        self.assertEqual(ys, [0.2, 2.8])

        # Bedroom was emitted as a <path>, exercising the d="M ... L ..." branch.
        bedroom = by_label["bedroom"]["polygon"]
        xs = sorted({round(p[0], 2) for p in bedroom})
        ys = sorted({round(p[1], 2) for p in bedroom})
        self.assertEqual(xs, [2.1, 3.8])
        self.assertEqual(ys, [0.2, 2.8])

    def test_room_areas_are_positive(self):
        for r in self.plan["rooms"]:
            self.assertGreater(r["area"], 0.0)

    def test_scale_preserved(self):
        self.assertEqual(self.plan["scale"]["pixels_per_meter"], 50)


class BuildTrainingSetTest(unittest.TestCase):
    def test_reads_cubicasa_and_shuffles_deterministically(self):
        a = build_training_set(cubicasa_root=FIXTURE_ROOT, shuffle=True, seed=7)
        b = build_training_set(cubicasa_root=FIXTURE_ROOT, shuffle=True, seed=7)
        self.assertEqual([s.image_path for s in a], [s.image_path for s in b])

    def test_missing_root_is_skipped_not_crashed(self):
        # Passing None should produce an empty list without raising.
        self.assertEqual(build_training_set(), [])


# ---------- real-MLS eval set ----------

class RealMLSLoaderTest(unittest.TestCase):
    def setUp(self):
        # Silence the loader's "missing label" / "invalid label" prints so
        # the test output stays clean. We still assert the filtering took
        # effect via the sample list.
        import io as _io
        import contextlib
        self._buf = _io.StringIO()
        self._ctx = contextlib.redirect_stdout(self._buf)
        self._ctx.__enter__()

    def tearDown(self):
        self._ctx.__exit__(None, None, None)

    def test_accepts_dataset_root_or_samples_dir(self):
        root_samples = list(RealMLSLoader(REAL_MLS_ROOT))
        samples_dir = list(RealMLSLoader(REAL_MLS_ROOT / "samples"))
        self.assertEqual(
            [s.image_path.name for s in root_samples],
            [s.image_path.name for s in samples_dir],
        )

    def test_loads_labeled_samples_only(self):
        samples = list(RealMLSLoader(REAL_MLS_ROOT))
        # listing_a (png+json), listing_b (jpg+json) should load.
        # listing_c (no json) and listing_d (invalid json) must be skipped.
        stems = sorted(s.image_path.stem for s in samples)
        self.assertEqual(stems, ["listing_a", "listing_b"])
        for s in samples:
            self.assertEqual(s.source, "real_mls")
            schema.deserialize(s.target_json)  # label validates

    def test_build_eval_set_is_isolated_from_training(self):
        eval_samples = build_eval_set(real_mls_root=REAL_MLS_ROOT)
        self.assertEqual(len(eval_samples), 2)
        # None of the eval samples should leak into build_training_set's
        # output when only cubicasa is passed — verifies eval stays eval.
        train = build_training_set(cubicasa_root=FIXTURE_ROOT, shuffle=False)
        train_paths = {s.image_path for s in train}
        for s in eval_samples:
            self.assertNotIn(s.image_path, train_paths)


# ---------- SVG transform math ----------

class TransformMathTest(unittest.TestCase):
    def test_parse_translate(self):
        m = _parse_transform("translate(5, 10)")
        self.assertAlmostEqual(_apply(m, (0, 0))[0], 5.0)
        self.assertAlmostEqual(_apply(m, (0, 0))[1], 10.0)

    def test_parse_missing_is_identity(self):
        self.assertEqual(_parse_transform(""), _IDENTITY)
        self.assertEqual(_parse_transform("unknown(1 2)"), _IDENTITY)

    def test_compose_is_nested_application(self):
        # translate(1,2) composed with translate(10,20) should map (0,0) -> (11,22).
        outer = _parse_transform("translate(1, 2)")
        inner = _parse_transform("translate(10, 20)")
        combined = _compose(outer, inner)
        self.assertAlmostEqual(_apply(combined, (0, 0))[0], 11.0)
        self.assertAlmostEqual(_apply(combined, (0, 0))[1], 22.0)

    def test_matrix_form(self):
        # matrix(2 0 0 3 5 7) scales x*2, y*3, then translates (5,7).
        m = _parse_transform("matrix(2 0 0 3 5 7)")
        x, y = _apply(m, (1, 1))
        self.assertAlmostEqual(x, 7.0)   # 2*1 + 5
        self.assertAlmostEqual(y, 10.0)  # 3*1 + 7

    def test_scale_single_arg(self):
        m = _parse_transform("scale(2)")
        x, y = _apply(m, (3, 4))
        self.assertAlmostEqual(x, 6.0)
        self.assertAlmostEqual(y, 8.0)

    def test_rotate_90_around_origin(self):
        m = _parse_transform("rotate(90)")
        x, y = _apply(m, (1, 0))
        self.assertAlmostEqual(x, 0.0, places=5)
        self.assertAlmostEqual(y, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
