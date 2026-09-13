"""
Tests for truncated-output salvage in model/inference.py.

A real MLS plan serializes to 3000+ tokens; when the decoding budget cuts
the JSON mid-element the parser must keep every complete element instead
of discarding the whole prediction.
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))
import inference  # type: ignore
from schema import serialize  # type: ignore


def _plan():
    return {
        "scale": {"pixels_per_meter": 50},
        "walls": [
            {"start": [0, 0], "end": [6, 0], "thickness": 0.15},
            {"start": [6, 0], "end": [6, 4], "thickness": 0.15},
            {"start": [6, 4], "end": [0, 4], "thickness": 0.15},
            {"start": [0, 4], "end": [0, 0], "thickness": 0.15},
        ],
        "doors": [{"position": [3, 0], "width": 0.9, "type": "hinged", "wall_index": 0}],
        "windows": [{"position": [3, 4], "width": 1.2, "wall_index": 2}],
        "rooms": [
            {"label": "living_room", "polygon": [[0, 0], [6, 0], [6, 4], [0, 4]], "area": 24.0},
            {"label": "bedroom", "polygon": [[0, 0], [3, 0], [3, 4], [0, 4]], "area": 12.0},
        ],
    }


class TestSalvageTruncatedJson(unittest.TestCase):
    def test_complete_output_is_unchanged(self):
        out = inference._deserialize_with_drift_repair(serialize(_plan()))
        self.assertEqual(len(out["walls"]), 4)
        self.assertEqual(len(out["rooms"]), 2)

    def test_cut_inside_last_room_keeps_earlier_rooms(self):
        text = serialize(_plan())
        cut = text[: text.rfind('"bedroom"') + 12]  # mid-way through the 2nd room
        out = inference._deserialize_with_drift_repair(cut)
        self.assertEqual(len(out["walls"]), 4)
        self.assertEqual(len(out["doors"]), 1)
        self.assertEqual([r["label"] for r in out["rooms"]], ["living_room"])

    def test_cut_inside_walls_keeps_complete_walls_only(self):
        text = serialize(_plan())
        third = text.find('"start"', text.find('"start"', text.find('"start"') + 1) + 1)
        out = inference._deserialize_with_drift_repair(text[: third + 20])
        self.assertEqual(len(out["walls"]), 2)
        self.assertEqual(out["doors"], [])
        self.assertEqual(out["rooms"], [])

    def test_cut_inside_a_string_is_handled(self):
        text = serialize(_plan())
        cut = text[: text.rfind('"living_room"') + 5]  # inside the label string
        out = inference._deserialize_with_drift_repair(cut)
        self.assertEqual(len(out["walls"]), 4)
        self.assertEqual(out["rooms"], [])

    def test_nothing_recoverable_raises(self):
        with self.assertRaises(ValueError):  # JSONDecodeError or SchemaError
            inference._deserialize_with_drift_repair('{"scale": {"pixels_per_meter": 5')

    def test_salvage_helper_closes_brackets(self):
        s = '{"walls":[{"start":[0,0],"end":[1,0]},{"start":[1,0],"end"'
        fixed = inference._salvage_truncated_json(s)
        self.assertEqual(json.loads(fixed), {"walls": [{"start": [0, 0], "end": [1, 0]}]})


if __name__ == "__main__":
    unittest.main()


class TestBaseModelFallback(unittest.TestCase):
    def test_default_base_when_no_train_config(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(inference._resolve_base_model(Path(d)), inference.DEFAULT_BASE_MODEL)
            (Path(d) / "train_config.json").write_text('{"base_model": "Org/Custom"}')
            self.assertEqual(inference._resolve_base_model(Path(d)), "Org/Custom")


class TestBackendHelpers(unittest.TestCase):
    def test_select_backend_env_override(self):
        from unittest.mock import patch
        with patch.dict("os.environ", {"FP3D_VLM_BACKEND": "torch"}):
            self.assertEqual(inference._select_backend("auto"), "torch")
        with patch.dict("os.environ", {"FP3D_VLM_BACKEND": "mlx"}):
            self.assertEqual(inference._select_backend("torch"), "mlx")

    def test_grounded_items_parse_and_rescale(self):
        raw = 'Here: [{"bbox_2d":[10,20,30,40],"text_content":"KITCHEN"},{"bbox_2d":[1,2],"text_content":"x"}]'
        out = inference._parse_grounded_items(raw, 2.0, 0.5)
        self.assertEqual(out, [{"text": "KITCHEN", "bbox_px": [20.0, 10.0, 60.0, 20.0]}])

    def test_crop_text_cleanup(self):
        self.assertEqual(inference._clean_crop_text(' "FULL BATH".\n'), "FULL BATH")
        self.assertEqual(inference._clean_crop_text("None"), "")
        self.assertEqual(inference._clean_crop_text(""), "")


class TestTiledOcr(unittest.TestCase):
    def test_small_image_is_one_tile_and_big_image_is_gridded(self):
        self.assertEqual(inference._tile_boxes(900, 700), [(0, 0, 900, 700)])
        boxes = inference._tile_boxes(2776, 1788)
        self.assertEqual(len(boxes), 4)                      # 3 x 2 at 1 MP; 2 x 2 at the 1.4 MP tile budget
        self.assertEqual(boxes[0][:2], (0, 0))
        self.assertEqual(boxes[-1][2:], (2776, 1788))
        # neighbouring tiles overlap
        self.assertGreater(boxes[0][2], boxes[1][0])

    def test_merge_drops_duplicates_across_tiles(self):
        items = [{"text": "17'9\"", "bbox_px": [100, 100, 200, 130]},
                 {"text": "17'9\"", "bbox_px": [105, 102, 198, 131]},
                 {"text": "17'9\"", "bbox_px": [900, 100, 1000, 130]}]
        self.assertEqual(len(inference._merge_grounded_items(items)), 2)

    def test_tiled_driver_offsets_boxes_into_original_pixels(self):
        import tempfile
        from PIL import Image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            Image.new("RGB", (2776, 1788), "white").save(f.name)
            calls = []

            def fake_generate(pil, prompt, n):
                calls.append(pil.size)
                return '[{"bbox_2d":[10,10,50,20],"text_content":"X"}]'

            out = inference._grounded_ocr_tiled(fake_generate, f.name, "p", 10)
        self.assertEqual(len(calls), 4)
        self.assertEqual(len(out), 4)
        xs = sorted(b["bbox_px"][0] for b in out)
        self.assertGreater(xs[-1], 1200)     # tiles on the right report right-hand pixels
