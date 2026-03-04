"""
Local CV model inference wrapper for floor plan parsing.

Handles loading the fine-tuned model and running inference on floor plan images.
Falls back to a mock output when no trained model weights are available.
"""

import json
import os
from pathlib import Path


# Default path for model weights (relative to this file)
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parent.parent.parent / "model" / "weights"


class LocalModelClient:
    """Client for the local floor plan parsing model."""

    def __init__(self, weights_dir=None):
        self.weights_dir = Path(weights_dir) if weights_dir else DEFAULT_WEIGHTS_DIR
        self._model = None

    def _load_model(self):
        """Load the trained model weights."""
        weights_path = self.weights_dir / "floorplan_parser.pt"

        if not weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {weights_path}. "
                "Run the training script first, or use a JSON override in the add-on. "
                "See model/data/README.md for dataset download instructions."
            )

        try:
            from ..._model_impl import load_model
            self._model = load_model(weights_path)
        except ImportError:
            # PyTorch/model dependencies not available in Blender's Python
            # Use subprocess to call the inference script
            self._model = "subprocess"

    def predict(self, image_path):
        """Run inference on a floor plan image.

        Args:
            image_path: Path to the floor plan image file.

        Returns:
            dict: Parsed floor plan data in the standard JSON format.
        """
        image_path = str(image_path)

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        if self._model is None:
            self._load_model()

        if self._model == "subprocess":
            return self._predict_subprocess(image_path)

        return self._predict_direct(image_path)

    def _predict_subprocess(self, image_path):
        """Run inference via subprocess (when model deps aren't in Blender's Python)."""
        import subprocess
        import sys

        inference_script = Path(__file__).resolve().parent.parent.parent / "model" / "inference.py"

        result = subprocess.run(
            [sys.executable, str(inference_script), "--image", image_path, "--output", "json"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Model inference failed: {result.stderr}")

        return json.loads(result.stdout)

    def _predict_direct(self, image_path):
        """Run inference directly (when model is loaded in-process)."""
        # This path is used when PyTorch is available in Blender's Python
        return self._model.predict(image_path)


def get_mock_output():
    """Return a sample floor plan JSON for testing without a trained model.

    This represents a simple one-bedroom apartment.
    """
    return {
        "scale": {"pixels_per_meter": 50},
        "walls": [
            {"start": [0, 0], "end": [6, 0], "thickness": 0.15},
            {"start": [6, 0], "end": [6, 4], "thickness": 0.15},
            {"start": [6, 4], "end": [0, 4], "thickness": 0.15},
            {"start": [0, 4], "end": [0, 0], "thickness": 0.15},
            {"start": [3, 0], "end": [3, 4], "thickness": 0.1},
        ],
        "doors": [
            {"position": [3, 2], "width": 0.9, "type": "hinged", "wall_index": 4},
            {"position": [1, 0], "width": 0.9, "type": "hinged", "wall_index": 0},
        ],
        "windows": [
            {"position": [4.5, 4], "width": 1.2, "wall_index": 2},
            {"position": [1.5, 4], "width": 1.0, "wall_index": 2},
        ],
        "rooms": [
            {
                "label": "living_room",
                "polygon": [[3, 0], [6, 0], [6, 4], [3, 4]],
                "area": 12.0,
            },
            {
                "label": "bedroom",
                "polygon": [[0, 0], [3, 0], [3, 4], [0, 4]],
                "area": 12.0,
            },
        ],
    }


def get_mock_studio():
    """Return a studio apartment mock floor plan."""
    return {
        "scale": {"pixels_per_meter": 50},
        "walls": [
            {"start": [0, 0], "end": [5, 0], "thickness": 0.15},
            {"start": [5, 0], "end": [5, 4], "thickness": 0.15},
            {"start": [5, 4], "end": [0, 4], "thickness": 0.15},
            {"start": [0, 4], "end": [0, 0], "thickness": 0.15},
            {"start": [0, 2.5], "end": [2, 2.5], "thickness": 0.1},
            {"start": [2, 2.5], "end": [2, 4], "thickness": 0.1},
        ],
        "doors": [
            {"position": [2.5, 0], "width": 0.9, "type": "hinged", "wall_index": 0},
            {"position": [2, 3.2], "width": 0.7, "type": "hinged", "wall_index": 5},
        ],
        "windows": [
            {"position": [3.5, 4], "width": 1.5, "wall_index": 2},
            {"position": [5, 2], "width": 1.0, "wall_index": 1},
        ],
        "rooms": [
            {
                "label": "main_room",
                "polygon": [[0, 0], [5, 0], [5, 2.5], [2, 2.5], [0, 2.5]],
                "area": 12.5,
            },
            {
                "label": "kitchenette",
                "polygon": [[2, 2.5], [5, 2.5], [5, 4], [2, 4]],
                "area": 4.5,
            },
            {
                "label": "bathroom",
                "polygon": [[0, 2.5], [2, 2.5], [2, 4], [0, 4]],
                "area": 3.0,
            },
        ],
    }


def get_mock_three_bedroom():
    """Return a three-bedroom house mock floor plan."""
    return {
        "scale": {"pixels_per_meter": 50},
        "walls": [
            # Exterior
            {"start": [0, 0], "end": [10, 0], "thickness": 0.2},
            {"start": [10, 0], "end": [10, 8], "thickness": 0.2},
            {"start": [10, 8], "end": [0, 8], "thickness": 0.2},
            {"start": [0, 8], "end": [0, 0], "thickness": 0.2},
            # Hallway horizontal
            {"start": [0, 4], "end": [6, 4], "thickness": 0.12},
            # Bedroom dividers
            {"start": [3.5, 4], "end": [3.5, 8], "thickness": 0.12},
            {"start": [7, 4], "end": [7, 8], "thickness": 0.12},
            # Kitchen/living divider
            {"start": [6, 0], "end": [6, 4], "thickness": 0.12},
            # Hallway vertical
            {"start": [6, 4], "end": [10, 4], "thickness": 0.12},
        ],
        "doors": [
            {"position": [5, 0], "width": 1.0, "type": "hinged", "wall_index": 0},
            {"position": [1.5, 4], "width": 0.8, "type": "hinged", "wall_index": 4},
            {"position": [5, 4], "width": 0.8, "type": "hinged", "wall_index": 4},
            {"position": [8.5, 4], "width": 0.8, "type": "hinged", "wall_index": 8},
            {"position": [6, 2], "width": 0.9, "type": "hinged", "wall_index": 7},
        ],
        "windows": [
            {"position": [2, 8], "width": 1.5, "wall_index": 2},
            {"position": [5.5, 8], "width": 1.2, "wall_index": 2},
            {"position": [8.5, 8], "width": 1.2, "wall_index": 2},
            {"position": [0, 2], "width": 1.2, "wall_index": 3},
            {"position": [10, 6], "width": 1.2, "wall_index": 1},
            {"position": [8, 0], "width": 1.5, "wall_index": 0},
        ],
        "rooms": [
            {
                "label": "living_room",
                "polygon": [[0, 0], [6, 0], [6, 4], [0, 4]],
                "area": 24.0,
            },
            {
                "label": "kitchen",
                "polygon": [[6, 0], [10, 0], [10, 4], [6, 4]],
                "area": 16.0,
            },
            {
                "label": "bedroom_1",
                "polygon": [[0, 4], [3.5, 4], [3.5, 8], [0, 8]],
                "area": 14.0,
            },
            {
                "label": "bedroom_2",
                "polygon": [[3.5, 4], [7, 4], [7, 8], [3.5, 8]],
                "area": 14.0,
            },
            {
                "label": "bedroom_3",
                "polygon": [[7, 4], [10, 4], [10, 8], [7, 8]],
                "area": 12.0,
            },
        ],
    }
