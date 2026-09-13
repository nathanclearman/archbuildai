"""
Qwen2.5-VL client for floor plan parsing (the "Qwen2.5-VL (trained)" option).

Thin adapter that gives the fine-tuned VLM the same predict() contract as
the CubiCasa / YOLO / Premium Vision clients, so operators.py can dispatch
to any backend uniformly:

    predict(image_path, pixels_per_meter=None, conf_threshold=None) -> dict

All the heavy lifting lives in `local_model.LocalModelClient`, which runs
the repo's floorplan3d/model/inference.py as a persistent daemon in a
Python that has torch/transformers/peft (resolved automatically, or set
FP3D_PYTHON). One daemon is shared across Generate clicks for the whole
Blender session so the ~30-90 s model load is paid once; `shutdown()` is
called from the add-on's unregister() to stop it.
"""

import os
import threading

try:  # inside the Blender add-on package
    from . import local_model, hybrid, autoscale
except ImportError:  # imported standalone (tests put api/ on sys.path)
    import local_model  # type: ignore
    import hybrid  # type: ignore
    import autoscale  # type: ignore


_shared_client = None
_shared_lock = threading.Lock()


def _get_shared_client():
    """Return the session-wide LocalModelClient, creating it on first use.

    Construction resolves the ML interpreter (a few subprocess probes), so
    it is done lazily here rather than at import time — the add-on must
    import instantly even on machines without the ML environment.
    """
    global _shared_client
    with _shared_lock:
        if _shared_client is None:
            _shared_client = local_model.LocalModelClient()
        return _shared_client


def shutdown():
    """Stop the shared inference daemon, if any. Idempotent, never raises."""
    global _shared_client
    with _shared_lock:
        client, _shared_client = _shared_client, None
    if client is not None:
        try:
            client.close()
        except Exception:
            pass


def label_rooms(image_path, plan, pixels_per_meter, client=None):
    """Hybrid backend, label half: replace the room labels in `plan` (YOLO
    geometry, metres = pixels / ppm) with the names printed on the image.

    Returns (new_plan, report). Never mutates `plan`. Raises if the VLM
    daemon is unavailable — the caller decides whether to fall back to the
    YOLO heuristic labels.
    """
    client = client if client is not None else _get_shared_client()
    image_path = str(image_path)
    # Pass 1: grounded OCR over the whole plan → names with positions.
    texts = [(t["text"], t["bbox_px"]) for t in client.ocr_labels(image_path)
             if t.get("text") and t.get("bbox_px")]
    rooms, report = hybrid.assign_labels(plan.get("rooms", []), texts, pixels_per_meter)
    # Pass 2: crop each still-unlabeled polygon and read its name directly.
    rooms, filled = hybrid.fill_unlabeled_by_crop(
        rooms, lambda box: client.ocr_crop(image_path, box), pixels_per_meter,
        allowed=set(report["labels_seen"]))
    report["labeled_from_crop"] = filled
    report["kept_yolo_label"] = max(0, report["kept_yolo_label"] - filled)
    out = dict(plan)
    out["rooms"] = rooms
    return out, report


def auto_scale(image_path, plan, ppm_guess, client=None):
    """Read the dimension strings off the plan and rescale `plan` (metres at
    `ppm_guess`) to the scale they imply. Returns (plan, report); the plan
    is returned unchanged when too few dimensions agree."""
    client = client if client is not None else _get_shared_client()
    texts = [(t["text"], t["bbox_px"]) for t in client.ocr_dimensions(str(image_path))
             if t.get("text") and t.get("bbox_px")]
    return autoscale.apply_auto_scale(plan, ppm_guess, texts)


class QwenModelClient:
    """Client for the fine-tuned Qwen2.5-VL floor plan parser."""

    def __init__(self, client=None):
        # `client` is an injection seam for tests; production uses the
        # shared daemon-backed client.
        self._client = client

    def predict(self, image_path, pixels_per_meter=None, conf_threshold=None):
        """Run inference on a floor plan image.

        Args:
            image_path: Path to the floor plan image.
            pixels_per_meter: Accepted for interface parity. The VLM reads
                scale from the image itself, so this is not forwarded.
            conf_threshold: When set, drop doors/windows whose per-element
                `confidence` is below it. Walls are never filtered — the
                geometry layer needs wall_index references to stay stable.

        Returns:
            dict: Parsed floor plan JSON (walls, doors, windows, rooms, scale).
        """
        image_path = str(image_path)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        if not local_model.INFERENCE_SCRIPT.exists():
            raise FileNotFoundError(
                f"Qwen inference script not found at {local_model.INFERENCE_SCRIPT}. "
                "Set FP3D_QWEN_MODEL_DIR to your floorplan3d/model/ directory."
            )
        if not (local_model.DEFAULT_WEIGHTS_DIR / "train_config.json").exists():
            raise FileNotFoundError(
                f"Trained Qwen weights not found in {local_model.DEFAULT_WEIGHTS_DIR}. "
                "Expected weights/train_config.json + weights/adapter/. "
                "Run the training script first, or set FP3D_QWEN_MODEL_DIR."
            )

        client = self._client if self._client is not None else _get_shared_client()
        data = client.predict(image_path)

        if conf_threshold is not None:
            data["doors"] = [
                d for d in data.get("doors", [])
                if d.get("confidence", 1.0) >= conf_threshold
            ]
            data["windows"] = [
                w for w in data.get("windows", [])
                if w.get("confidence", 1.0) >= conf_threshold
            ]
        return data
