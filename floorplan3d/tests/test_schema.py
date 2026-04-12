"""Tests for the canonical floor plan schema."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

from schema import (  # type: ignore
    FloorPlan,
    SchemaError,
    deserialize,
    serialize,
    validate,
)


MINIMAL = {
    "scale": {"pixels_per_meter": 50},
    "walls": [{"start": [0, 0], "end": [3, 0], "thickness": 0.15}],
    "doors": [],
    "windows": [],
    "rooms": [
        {"label": "bedroom", "polygon": [[0, 0], [3, 0], [3, 3], [0, 3]], "area": 9.0},
    ],
}


class TestValidate(unittest.TestCase):
    def test_minimal_valid(self):
        validate(MINIMAL)  # no raise

    def test_rejects_bad_wall(self):
        bad = dict(MINIMAL)
        bad["walls"] = [{"start": [0], "end": [1, 1]}]
        with self.assertRaises(SchemaError):
            validate(bad)

    def test_rejects_wall_index_out_of_range(self):
        bad = dict(MINIMAL)
        bad["doors"] = [{"position": [1, 0], "wall_index": 99}]
        with self.assertRaises(SchemaError):
            validate(bad)

    def test_rejects_room_with_too_few_points(self):
        bad = dict(MINIMAL)
        bad["rooms"] = [{"label": "bedroom", "polygon": [[0, 0], [1, 0]]}]
        with self.assertRaises(SchemaError):
            validate(bad)

    def test_rejects_non_dict(self):
        with self.assertRaises(SchemaError):
            validate([])


class TestSerializeRoundtrip(unittest.TestCase):
    def test_roundtrip(self):
        text = serialize(MINIMAL)
        parsed = json.loads(text)
        # Deterministic key order matters for training-target stability.
        self.assertEqual(list(parsed.keys()), ["scale", "walls", "doors", "windows", "rooms"])
        validate(parsed)

    def test_dataclass_input(self):
        fp = FloorPlan.from_dict(MINIMAL)
        text = serialize(fp)
        self.assertIn("bedroom", text)

    def test_rounds_coordinates(self):
        noisy = {
            "scale": {"pixels_per_meter": 50},
            "walls": [{"start": [0.12345, 0.98765], "end": [3.11111, 0.22222], "thickness": 0.151}],
            "doors": [], "windows": [], "rooms": [],
        }
        parsed = json.loads(serialize(noisy, decimals=2))
        self.assertEqual(parsed["walls"][0]["start"], [0.12, 0.99])
        self.assertEqual(parsed["walls"][0]["end"], [3.11, 0.22])


class TestDeserialize(unittest.TestCase):
    def test_bare_json(self):
        out = deserialize(serialize(MINIMAL))
        self.assertEqual(len(out["walls"]), 1)
        self.assertEqual(out["rooms"][0]["label"], "bedroom")

    def test_strips_code_fences(self):
        text = "```json\n" + serialize(MINIMAL) + "\n```"
        out = deserialize(text)
        self.assertEqual(out["rooms"][0]["label"], "bedroom")

    def test_tolerates_surrounding_text(self):
        text = "Sure, here you go: " + serialize(MINIMAL) + " -- done"
        out = deserialize(text)
        self.assertEqual(len(out["walls"]), 1)

    def test_fills_missing_sections(self):
        minimal_walls_only = json.dumps({
            "walls": [{"start": [0, 0], "end": [1, 0], "thickness": 0.15}]
        })
        out = deserialize(minimal_walls_only)
        self.assertEqual(out["doors"], [])
        self.assertEqual(out["rooms"], [])
        self.assertIn("scale", out)

    def test_rejects_non_json(self):
        with self.assertRaises(SchemaError):
            deserialize("not json at all")


if __name__ == "__main__":
    unittest.main()
