"""Tests for the YOLO-geometry + VLM-label hybrid glue (api/hybrid.py)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "blender_addon" / "api"))
import hybrid  # type: ignore


class TestNormalizeLabel(unittest.TestCase):
    def test_common_printed_names(self):
        cases = {
            "PRIMARY BEDROOM": "master_bedroom", "W.I.C.": "walk_in_closet",
            "1/2 BATH": "powder_room", "HALF BATH": "powder_room", "Eat-in Kitchen": "kitchen",
            "FULL BATH": "bathroom", "DINING AREA": "dining_room", "FAMILY ROOM": "family_room",
            "HALL": "hallway", "BALCONY": "balcony", "Laundry Storage": "laundry_room",
            "Bedroom 2": "bedroom", "Formal Living": "living_room", "FOYER": "foyer",
        }
        for text, want in cases.items():
            with self.subTest(text=text):
                self.assertEqual(hybrid.normalize_label(text), want)

    def test_unknown_text_is_slugified_not_lost(self):
        self.assertEqual(hybrid.normalize_label("Wine Cellar"), "wine_cellar")

    def test_noise_is_dropped(self):
        for text in ("17'9\"", "[AREA: 1560]", "1st floor", "N", ""):
            with self.subTest(text=text):
                self.assertIsNone(hybrid.normalize_label(text))


class TestAssignLabels(unittest.TestCase):
    def setUp(self):
        # Two 4x4 m rooms side by side, ppm=100 → pixels 0-400 and 400-800.
        self.rooms = [
            {"label": "bedroom_3", "polygon": [[0, 0], [4, 0], [4, 4], [0, 4]]},
            {"label": "bathroom_7", "polygon": [[4, 0], [8, 0], [8, 4], [4, 4]]},
        ]

    def test_text_inside_polygon_relabels_it(self):
        texts = [("KITCHEN", [100, 150, 300, 200]), ("LIVING ROOM", [500, 150, 700, 200])]
        out, rep = hybrid.assign_labels(self.rooms, texts, pixels_per_meter=100)
        self.assertEqual([r["label"] for r in out], ["kitchen", "living_room"])
        self.assertEqual([r["label_source"] for r in out], ["ocr", "ocr"])
        self.assertEqual(rep["labeled_from_ocr"], 2)
        # input not mutated
        self.assertEqual(self.rooms[0]["label"], "bedroom_3")

    def test_room_without_text_gets_neutral_label(self):
        out, rep = hybrid.assign_labels(self.rooms, [("KITCHEN", [100, 150, 300, 200])], 100)
        self.assertEqual(out[1]["label"], "room")
        self.assertEqual(out[1]["label_source"], "none")
        self.assertEqual(rep["kept_yolo_label"], 1)

    def test_unlabeled_none_keeps_yolo_guess(self):
        out, _ = hybrid.assign_labels(self.rooms, [], 100, unlabeled=None)
        self.assertEqual([r["label"] for r in out], ["bedroom_3", "bathroom_7"])

    def test_text_just_outside_snaps_to_nearest_within_threshold(self):
        # Text centre 0.5 m below room 0 (y = 4.5 m) → snaps; 3 m below → dropped.
        out, rep = hybrid.assign_labels(self.rooms, [("FOYER", [100, 440, 300, 460])], 100)
        self.assertEqual(out[0]["label"], "foyer")
        out, rep = hybrid.assign_labels(self.rooms, [("FOYER", [100, 690, 300, 710])], 100)
        self.assertEqual(out[0]["label"], "room")
        self.assertIn("FOYER", rep["dropped_texts"])

    def test_two_texts_in_one_room_keep_the_larger(self):
        texts = [("FORMAL DINING", [100, 150, 200, 170]), ("FORMAL LIVING", [100, 100, 300, 140])]
        out, _ = hybrid.assign_labels(self.rooms, texts, 100)
        self.assertEqual(out[0]["label"], "living_room")

    def test_dimension_strings_are_ignored(self):
        out, rep = hybrid.assign_labels(self.rooms, [("17'9\"", [100, 150, 300, 200])], 100)
        self.assertEqual(out[0]["label"], "room")
        self.assertEqual(rep["dropped_texts"], ["17'9\""])


class TestCropPass(unittest.TestCase):
    def test_fills_only_unlabeled_rooms_and_normalizes(self):
        rooms = [
            {"label": "kitchen", "label_source": "ocr", "polygon": [[0, 0], [4, 0], [4, 4], [0, 4]]},
            {"label": "room", "label_source": "none", "polygon": [[4, 0], [8, 0], [8, 4], [4, 4]]},
            {"label": "room", "label_source": "none", "polygon": [[8, 0], [12, 0], [12, 4], [8, 4]]},
        ]
        asked = []

        def read_crop(box):
            asked.append(box)
            return "W.I.C." if box[0] == 400 else "NONE"

        out, filled = hybrid.fill_unlabeled_by_crop(rooms, read_crop, pixels_per_meter=100)
        self.assertEqual(filled, 1)
        self.assertEqual([r["label"] for r in out], ["kitchen", "walk_in_closet", "room"])
        self.assertEqual(out[1]["label_source"], "crop")
        self.assertEqual(asked, [[400, 0, 800, 400], [800, 0, 1200, 400]])  # labeled room not asked
        self.assertEqual(rooms[1]["label"], "room")  # input untouched

    def test_allowed_set_vets_crop_guesses(self):
        rooms = [
            {"label": "room", "label_source": "none", "polygon": [[0, 0], [4, 0], [4, 4], [0, 4]]},
            {"label": "room", "label_source": "none", "polygon": [[4, 0], [8, 0], [8, 4], [4, 4]]},
        ]
        answers = iter(["BATHROOM", "W.I.C."])
        out, filled = hybrid.fill_unlabeled_by_crop(
            rooms, lambda box: next(answers), 100, allowed={"walk_in_closet", "kitchen"})
        self.assertEqual(filled, 1)
        self.assertEqual([r["label"] for r in out], ["room", "walk_in_closet"])

    def test_crop_reader_failure_is_skipped(self):
        rooms = [{"label": "room", "label_source": "none", "polygon": [[0, 0], [4, 0], [4, 4], [0, 4]]}]

        def boom(box):
            raise RuntimeError("daemon died")

        out, filled = hybrid.fill_unlabeled_by_crop(rooms, boom, 100)
        self.assertEqual(filled, 0)
        self.assertEqual(out[0]["label"], "room")


if __name__ == "__main__":
    unittest.main()
