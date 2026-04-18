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


# Probe timeout for `import torch` on a candidate interpreter. Torch's first
# import is dominated by shared-object loading (~1-3s on a warm disk, up to
# 8s on cold NVMe). 15s covers the slow path without letting a hung python
# freeze Blender's UI thread forever.
_PROBE_TIMEOUT_S = 15


def _is_blender_python(executable: str) -> bool:
    """True when `executable` looks like Blender's bundled interpreter.

    Blender ships a Python with no ML deps; shelling out to it from inside
    the add-on silently fails on `import torch`. We match on the file name
    plus the ancestor path so we catch both Linux (`.../blender/3.x/python/bin/python3.10`)
    and macOS (`.../Blender.app/Contents/Resources/.../python3.10`) layouts
    without depending on `sys.executable` containing the string 'blender'
    case-insensitively — that would false-positive on a user who happened
    to unpack Python under `~/Blender-Projects/`.
    """
    p = Path(executable).resolve()
    parts_lower = [part.lower() for part in p.parts]
    return any("blender" in part for part in parts_lower[:-1])


def _probe_python(candidate: str) -> bool:
    """True iff `candidate` can import torch within the probe budget.

    We import rather than check the path because a venv's python may be a
    symlink whose name doesn't carry its installed packages. A silent
    non-zero return (missing module, syntax error, permission denied) all
    collapse to False — caller decides what to do with that.
    """
    try:
        result = subprocess.run(
            [candidate, "-c", "import torch"],
            capture_output=True,
            timeout=_PROBE_TIMEOUT_S,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0


# Candidates tried after FP3D_PYTHON and sys.executable. Covers the two
# interpreters a user is likely to have on PATH with an ML env installed.
# Order matters — `python3` is preferred on modern Ubuntu/macOS; plain
# `python` resolves to 2.x on some older distros and is a fallback.
_FALLBACK_PYTHON_CANDIDATES: tuple[str, ...] = ("python3", "python")


# Memoized resolution keyed on the FP3D_PYTHON env value. The resolver
# spawns up to 3 `import torch` subprocesses (15 s timeout each) on a
# cold miss; without this cache, every `LocalModelClient()` in the
# Blender operator re-probes on every Generate click. Keyed on env so a
# user who sets FP3D_PYTHON mid-session gets a fresh resolution.
# Injected probes (test seams) bypass the cache — see _resolve_python_bin.
_RESOLVED_PYTHON_CACHE: dict[str | None, str] = {}


def _resolve_python_bin(probe=None) -> str:
    """Find a Python interpreter with `torch` installed.

    Resolution order:
      1. `FP3D_PYTHON` env var (explicit user override, no probe — trust it)
      2. `sys.executable` (only when NOT Blender's bundled interpreter)
      3. `python3`, `python` on PATH

    Every candidate past (1) is probed with `_probe_python` so we don't
    hand the user a silent `subprocess returncode=1` when the chosen
    interpreter can't import torch. `probe` is an injection seam so tests
    don't spawn real subprocesses; injecting a probe also bypasses the
    module-level cache so tests get deterministic behaviour.

    `probe=None` resolves to the module-level `_probe_python` AT CALL TIME
    rather than at def-time, so tests can `patch('local_model._probe_python',
    ...)` and see the patch take effect. A default-argument binding
    (`probe=_probe_python`) would capture the original reference when
    this module was first imported and defeat the patch.

    Raises RuntimeError with a message tailored to the actual failure
    mode: Blender-bundled-python diagnosis when `sys.executable` is
    Blender's, plain "torch not found on PATH" otherwise. Accusing a
    user's system python of being Blender's would send them hunting
    for a problem that isn't theirs.
    """
    is_default_probe = probe is None
    if is_default_probe:
        probe = _probe_python

    env = os.environ.get("FP3D_PYTHON")
    if env:
        return env

    # Cache only when using the default probe. Injected probes belong to
    # tests, which must see a fresh resolution every call.
    if is_default_probe and env in _RESOLVED_PYTHON_CACHE:
        return _RESOLVED_PYTHON_CACHE[env]

    blender_python = _is_blender_python(sys.executable)
    candidates: list[str] = []
    if not blender_python:
        candidates.append(sys.executable)
    candidates.extend(_FALLBACK_PYTHON_CANDIDATES)

    for candidate in candidates:
        if probe(candidate):
            if is_default_probe:
                _RESOLVED_PYTHON_CACHE[env] = candidate
            return candidate

    # Tailor the diagnosis so a non-Blender user isn't told their
    # /usr/bin/python3 is Blender's bundled interpreter.
    if blender_python:
        diagnosis = (
            f"sys.executable is {sys.executable!r} — Blender's bundled "
            "interpreter, which does not carry ML dependencies, and "
            "'python3' / 'python' on PATH could not import torch either."
        )
    else:
        diagnosis = (
            f"Tried {sys.executable!r}, 'python3', and 'python'; none "
            "could import torch. Your ML environment may not be "
            "installed, or it may not be on PATH."
        )
    raise RuntimeError(
        "Could not find a Python interpreter with `torch` installed. "
        "Set the FP3D_PYTHON environment variable to the path of a "
        "Python that has the model dependencies (see "
        f"floorplan3d/model/requirements.txt). {diagnosis}"
    )


class LocalModelClient:
    """Client for the fine-tuned floor plan VLM."""

    def __init__(self, weights_dir=None, python_bin=None, timeout=300):
        # Default timeout bumped from 120s to 300s. The first invocation
        # after a fresh Blender launch pays:
        #   Python subprocess start      ~1 s
        #   torch import                 ~1-3 s
        #   transformers import          ~2-5 s
        #   Qwen2.5-VL weights load      20-40 s (disk-bound, worse on
        #                                 external drives)
        #   CUDA / MPS context init      2-10 s
        #   Model generate               5-30 s
        # Total cold-call budget: 30-90 s typically, up to 120 s under
        # swap pressure on an M4 Max loading a 14 GB bf16 model. 300 s
        # covers that without letting a genuinely hung subprocess (stuck
        # on a corrupt weights file, OOM in MPS) freeze the UI forever.
        self.weights_dir = Path(weights_dir) if weights_dir else DEFAULT_WEIGHTS_DIR
        # Resolve lazily-but-eagerly: if caller passed an explicit path,
        # trust it. Otherwise probe for a torch-capable interpreter and
        # cache the result. The previous default (sys.executable) silently
        # routed every Blender-initiated predict() to Blender's bundled
        # Python, which has no torch — predict() then raised with a
        # generic "Model inference failed" instead of a fixable message.
        self.python_bin = python_bin if python_bin else _resolve_python_bin()
        self.timeout = timeout

    def predict(self, image_path, cv_only=False, refine=False):
        """Run inference on a floor plan image.

        Args:
            image_path: Path to the floor plan image file.
            cv_only: If True, skip the VLM and use the OpenCV fallback only.
                     Useful for testing without trained weights.
            refine: If True, run the optional Claude Opus refinement pass.
                    Requires ANTHROPIC_API_KEY in the environment.

        Returns:
            dict: Parsed floor plan data in the canonical JSON schema
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

        cmd = [
            self.python_bin,
            str(INFERENCE_SCRIPT),
            "--image", image_path,
            "--weights", str(self.weights_dir),
            "--output", "json",
        ]
        if cv_only:
            cmd.append("--cv-only")
        if refine:
            cmd.append("--refine")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Model inference failed: {result.stderr}")

        return json.loads(result.stdout)
