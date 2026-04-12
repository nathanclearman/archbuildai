"""
Local VLM client for floor plan parsing.

Wraps the fine-tuned Qwen2.5-VL model (see model/ for training) and returns
structured floor plan JSON for the Blender geometry layer to consume.

The model itself runs outside Blender's bundled Python — this client shells
out to model/inference.py so the heavy ML dependencies (torch, transformers,
mlx) don't need to be installed into Blender.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


# Default path for model weights (relative to this file)
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parent.parent.parent / "model" / "weights"
INFERENCE_SCRIPT = Path(__file__).resolve().parent.parent.parent / "model" / "inference.py"


class LocalModelClient:
    """Client for the fine-tuned floor plan VLM."""

    def __init__(self, weights_dir=None, python_bin=None, timeout=120):
        self.weights_dir = Path(weights_dir) if weights_dir else DEFAULT_WEIGHTS_DIR
        self.python_bin = python_bin or sys.executable
        self.timeout = timeout

    def predict(self, image_path):
        """Run inference on a floor plan image.

        Args:
            image_path: Path to the floor plan image file.

        Returns:
            dict: Parsed floor plan data in the standard JSON format
                  (walls, doors, windows, rooms, scale).
        """
        image_path = str(image_path)

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        if not INFERENCE_SCRIPT.exists():
            raise FileNotFoundError(
                f"Inference script not found at {INFERENCE_SCRIPT}. "
                "The VLM inference entry point has not been implemented yet."
            )

        result = subprocess.run(
            [
                self.python_bin,
                str(INFERENCE_SCRIPT),
                "--image", image_path,
                "--weights", str(self.weights_dir),
                "--output", "json",
            ],
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Model inference failed: {result.stderr}")

        return json.loads(result.stdout)
