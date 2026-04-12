"""
Tests for the Claude vision client (floorplan3d.blender_addon.api.claude_client).

These tests never hit the network. ``_post`` is patched so we can assert
what payload the client would have sent to Anthropic.
"""

import base64
import importlib.util
import os
import struct
import sys
import types
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

# The Blender add-on package imports ``bpy`` at the package level, which is
# only available inside Blender. Load ``schema`` and ``claude_client`` by file
# path under a synthetic package so relative imports (``from . import schema``)
# still resolve without touching ``blender_addon/__init__.py``.
API_DIR = Path(__file__).resolve().parent.parent / "blender_addon" / "api"

_pkg = types.ModuleType("_fp3d_api")
_pkg.__path__ = [str(API_DIR)]
sys.modules["_fp3d_api"] = _pkg


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"_fp3d_api.{name}",
        API_DIR / f"{name}.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"_fp3d_api.{name}"] = module
    spec.loader.exec_module(module)
    return module


fp_schema = _load("schema")
claude_client = _load("claude_client")


def _make_png(path, width=8, height=8):
    """Write a tiny valid PNG to ``path`` so _encode_image has something to read."""
    def chunk(tag, data):
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    raw = b""
    for _ in range(height):
        raw += b"\x00" + (b"\xff\xff\xff" * width)
    idat = zlib.compress(raw)
    png = sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
    path.write_bytes(png)


def _valid_plan():
    return {
        "scale": {"pixels_per_meter": 50},
        "walls": [
            {"start": [0, 0], "end": [4, 0], "thickness": 0.15},
            {"start": [4, 0], "end": [4, 3], "thickness": 0.15},
            {"start": [4, 3], "end": [0, 3], "thickness": 0.15},
            {"start": [0, 3], "end": [0, 0], "thickness": 0.15},
        ],
        "doors": [
            {"position": [2, 0], "width": 0.9, "type": "hinged", "wall_index": 0},
        ],
        "windows": [
            {"position": [2, 3], "width": 1.2, "wall_index": 2},
        ],
        "rooms": [
            {
                "label": "studio",
                "polygon": [[0, 0], [4, 0], [4, 3], [0, 3]],
                "area": 12.0,
            }
        ],
    }


def _tool_use_response(tool_input):
    return {
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_123",
                "name": claude_client.PARSE_FLOOR_PLAN_TOOL["name"],
                "input": tool_input,
            }
        ],
        "stop_reason": "tool_use",
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 300,
            "cache_creation_input_tokens": 800,
            "cache_read_input_tokens": 0,
        },
    }


class TestSchemaValidation(unittest.TestCase):
    def test_valid_plan_has_no_errors(self):
        self.assertEqual(fp_schema.validate(_valid_plan()), [])

    def test_missing_scale_flagged(self):
        plan = _valid_plan()
        del plan["scale"]
        errors = fp_schema.validate(plan)
        self.assertTrue(any("scale" in e for e in errors))

    def test_missing_walls_flagged(self):
        plan = _valid_plan()
        del plan["walls"]
        errors = fp_schema.validate(plan)
        self.assertTrue(any("walls" in e for e in errors))


class TestImageEncoding(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(os.path.abspath(os.path.dirname(__file__))) / "_tmp_claude"
        self.tmpdir.mkdir(exist_ok=True)
        self.png_path = self.tmpdir / "plan.png"
        _make_png(self.png_path)

    def tearDown(self):
        for p in self.tmpdir.iterdir():
            p.unlink()
        self.tmpdir.rmdir()

    def test_encodes_png_as_base64_block(self):
        block = claude_client.ClaudeClient._encode_image(self.png_path)
        self.assertEqual(block["type"], "image")
        self.assertEqual(block["source"]["type"], "base64")
        self.assertEqual(block["source"]["media_type"], "image/png")
        # Round-trip the data and check it starts with the PNG signature.
        raw = base64.b64decode(block["source"]["data"])
        self.assertTrue(raw.startswith(b"\x89PNG"))

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            claude_client.ClaudeClient._encode_image(self.tmpdir / "does_not_exist.png")

    def test_unsupported_format_raises(self):
        bogus = self.tmpdir / "not_an_image.bin"
        bogus.write_bytes(b"\x00" * 32)
        with self.assertRaises(ValueError):
            claude_client.ClaudeClient._encode_image(bogus)


class TestParseFloorPlanFromImage(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(os.path.abspath(os.path.dirname(__file__))) / "_tmp_parse"
        self.tmpdir.mkdir(exist_ok=True)
        self.png_path = self.tmpdir / "plan.png"
        _make_png(self.png_path)

    def tearDown(self):
        for p in self.tmpdir.iterdir():
            p.unlink()
        self.tmpdir.rmdir()

    def test_happy_path_uses_vision_tool_use_thinking_and_cache(self):
        client = claude_client.ClaudeClient(api_key="sk-test")
        captured = {}

        def fake_post(payload):
            captured["payload"] = payload
            return _tool_use_response(_valid_plan())

        with patch.object(client, "_post", side_effect=fake_post):
            result = client.parse_floor_plan_from_image(self.png_path)

        # Output validates.
        self.assertEqual(fp_schema.validate(result), [])

        payload = captured["payload"]

        # 1. Image was sent as a content block.
        user_content = payload["messages"][0]["content"]
        self.assertEqual(user_content[0]["type"], "image")
        self.assertEqual(user_content[0]["source"]["media_type"], "image/png")

        # 2. Tool schema matches the canonical floor plan schema.
        self.assertEqual(len(payload["tools"]), 1)
        self.assertEqual(payload["tools"][0]["name"], "submit_floor_plan")
        self.assertEqual(
            payload["tools"][0]["input_schema"],
            fp_schema.FLOOR_PLAN_SCHEMA,
        )

        # 3. tool_choice forces our specific tool.
        self.assertEqual(payload["tool_choice"]["type"], "tool")
        self.assertEqual(payload["tool_choice"]["name"], "submit_floor_plan")

        # 4. Prompt caching is enabled on system prompt and tool.
        self.assertEqual(
            payload["system"][0]["cache_control"],
            {"type": "ephemeral"},
        )
        self.assertEqual(
            payload["tools"][0]["cache_control"],
            {"type": "ephemeral"},
        )

        # 5. Extended thinking is enabled by default.
        self.assertEqual(payload["thinking"]["type"], "enabled")
        self.assertGreater(payload["thinking"]["budget_tokens"], 0)

        # 6. Model defaults to Opus 4.6.
        self.assertEqual(payload["model"], claude_client.MODEL_OPUS)

        # 7. Usage was tallied.
        self.assertEqual(client.usage["input_tokens"], 1000)
        self.assertEqual(client.usage["output_tokens"], 300)
        self.assertEqual(client.usage["cache_creation_input_tokens"], 800)

    def test_scale_hint_and_notes_reach_user_turn(self):
        client = claude_client.ClaudeClient(api_key="sk-test", use_thinking=False)
        captured = {}

        def fake_post(payload):
            captured["payload"] = payload
            return _tool_use_response(_valid_plan())

        with patch.object(client, "_post", side_effect=fake_post):
            client.parse_floor_plan_from_image(
                self.png_path,
                scale_hint="hallway is 1.2m wide",
                user_notes="hand-drawn sketch",
            )

        text_block = captured["payload"]["messages"][0]["content"][1]
        self.assertEqual(text_block["type"], "text")
        self.assertIn("hallway is 1.2m wide", text_block["text"])
        self.assertIn("hand-drawn sketch", text_block["text"])

    def test_repair_loop_runs_when_first_result_invalid(self):
        client = claude_client.ClaudeClient(api_key="sk-test", use_thinking=False)

        invalid_plan = {"walls": []}  # missing every other required key
        calls = []

        def fake_post(payload):
            calls.append(payload)
            if len(calls) == 1:
                return _tool_use_response(invalid_plan)
            return _tool_use_response(_valid_plan())

        with patch.object(client, "_post", side_effect=fake_post):
            result = client.parse_floor_plan_from_image(self.png_path)

        self.assertEqual(len(calls), 2, "expected one initial call plus one repair")
        self.assertEqual(fp_schema.validate(result), [])

    def test_repair_disabled_raises_on_invalid_output(self):
        client = claude_client.ClaudeClient(api_key="sk-test", use_thinking=False)
        with patch.object(
            client,
            "_post",
            return_value=_tool_use_response({"walls": []}),
        ):
            with self.assertRaises(claude_client.ClaudeAPIError):
                client.parse_floor_plan_from_image(self.png_path, repair=False)

    def test_missing_tool_call_raises(self):
        client = claude_client.ClaudeClient(api_key="sk-test", use_thinking=False)
        text_only = {
            "content": [{"type": "text", "text": "I refuse to use the tool."}],
            "stop_reason": "end_turn",
            "usage": {},
        }
        with patch.object(client, "_post", return_value=text_only):
            with self.assertRaises(claude_client.ClaudeAPIError):
                client.parse_floor_plan_from_image(self.png_path)


class TestRetryBehavior(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(os.path.abspath(os.path.dirname(__file__))) / "_tmp_retry"
        self.tmpdir.mkdir(exist_ok=True)
        self.png_path = self.tmpdir / "plan.png"
        _make_png(self.png_path)

    def tearDown(self):
        for p in self.tmpdir.iterdir():
            p.unlink()
        self.tmpdir.rmdir()

    def test_backoff_is_non_blocking_in_tests(self):
        # Smoke-test the backoff helper: attempt 0 -> 2s, capped at 16s.
        # We patch time.sleep so the suite stays fast.
        with patch.object(claude_client.time, "sleep") as fake_sleep:
            claude_client.ClaudeClient._sleep_backoff(0)
            claude_client.ClaudeClient._sleep_backoff(5)
        waits = [c.args[0] for c in fake_sleep.call_args_list]
        self.assertEqual(waits[0], 2)
        self.assertEqual(waits[1], 16)


class TestRequiresApiKey(unittest.TestCase):
    def test_empty_key_rejected(self):
        with self.assertRaises(ValueError):
            claude_client.ClaudeClient(api_key="")


if __name__ == "__main__":
    unittest.main()
