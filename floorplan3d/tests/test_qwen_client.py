"""
Tests for the Qwen2.5-VL add-on client (blender_addon/api/qwen_client.py).

The client is a thin adapter over local_model.LocalModelClient; these tests
inject a fake backend so nothing spawns a daemon or loads a model.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "blender_addon" / "api"),
)
import qwen_client  # type: ignore
from qwen_client import QwenModelClient  # type: ignore


class _FakeBackend:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def predict(self, image_path):
        self.calls.append(image_path)
        return dict(self.payload)


def _payload():
    return {
        "scale": {"pixels_per_meter": 50},
        "walls": [{"start": [0, 0], "end": [4, 0], "thickness": 0.15}],
        "doors": [
            {"position": [1, 0], "width": 0.9, "wall_index": 0, "confidence": 0.9},
            {"position": [2, 0], "width": 0.9, "wall_index": 0, "confidence": 0.3},
            {"position": [3, 0], "width": 0.9, "wall_index": 0},  # no confidence key
        ],
        "windows": [
            {"position": [1.5, 0], "width": 1.2, "wall_index": 0, "confidence": 0.5},
        ],
        "rooms": [],
    }


class TestQwenModelClient(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        self._tmp.close()
        self.image = self._tmp.name
        # Bypass the on-disk guards for inference.py / weights.
        self._patches = [
            patch.object(qwen_client.local_model, "INFERENCE_SCRIPT"),
            patch.object(qwen_client.local_model, "DEFAULT_WEIGHTS_DIR"),
        ]
        script, weights = (p.start() for p in self._patches)
        script.exists.return_value = True
        weights.__truediv__.return_value.exists.return_value = True

    def tearDown(self):
        for p in self._patches:
            p.stop()
        Path(self.image).unlink(missing_ok=True)
        qwen_client.shutdown()

    def test_missing_image_raises_before_touching_backend(self):
        backend = _FakeBackend(_payload())
        with self.assertRaises(FileNotFoundError):
            QwenModelClient(client=backend).predict("/nonexistent/plan.png")
        self.assertEqual(backend.calls, [])

    def test_predict_forwards_image_and_returns_backend_result(self):
        backend = _FakeBackend(_payload())
        out = QwenModelClient(client=backend).predict(self.image)
        self.assertEqual(backend.calls, [self.image])
        self.assertEqual(len(out["doors"]), 3)
        self.assertEqual(len(out["windows"]), 1)

    def test_conf_threshold_filters_doors_and_windows_but_not_walls(self):
        backend = _FakeBackend(_payload())
        out = QwenModelClient(client=backend).predict(self.image, conf_threshold=0.6)
        # 0.9 kept, 0.3 dropped, missing-confidence treated as 1.0 and kept
        self.assertEqual([d["position"][0] for d in out["doors"]], [1, 3])
        self.assertEqual(out["windows"], [])
        self.assertEqual(len(out["walls"]), 1)

    def test_shared_client_is_created_once_and_shutdown_clears_it(self):
        created = []

        class _Fake:
            def __init__(self):
                created.append(self)

            def close(self):
                self.closed = True

        with patch.object(qwen_client.local_model, "LocalModelClient", _Fake):
            a = qwen_client._get_shared_client()
            b = qwen_client._get_shared_client()
            self.assertIs(a, b)
            self.assertEqual(len(created), 1)
            qwen_client.shutdown()
            self.assertTrue(created[0].closed)
            c = qwen_client._get_shared_client()
            self.assertIsNot(a, c)


if __name__ == "__main__":
    unittest.main()


class TestAutoScale(unittest.TestCase):
    def test_auto_scale_uses_dimension_ocr_and_rescales(self):
        class _Backend:
            def ocr_dimensions(self, image_path):
                return [{"text": "16'5\"", "bbox_px": [200, 405, 300, 425]},
                        {"text": "13'1\"", "bbox_px": [650, 405, 750, 425]},
                        {"text": "13'1\"", "bbox_px": [905, 150, 925, 250]},
                        {"text": "KITCHEN", "bbox_px": [10, 10, 50, 20]}]
        rooms = [{"label": "a", "polygon": [[0, 0], [10, 0], [10, 8], [0, 8]]},
                 {"label": "b", "polygon": [[10, 0], [18, 0], [18, 8], [10, 8]]}]
        plan = {"scale": {"pixels_per_meter": 50}, "walls": [], "doors": [], "windows": [], "rooms": rooms}
        out, rep = qwen_client.auto_scale("/any.png", plan, 50, client=_Backend())
        self.assertTrue(rep["applied"])
        self.assertAlmostEqual(out["scale"]["pixels_per_meter"], 100, delta=1)
        self.assertAlmostEqual(out["rooms"][0]["polygon"][1][0], 5.0, delta=0.05)
