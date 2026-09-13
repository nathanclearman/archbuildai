"""
First-run environment setup for the add-on (no repo checkout required).

Everything the add-on needs at runtime can live inside Blender's own
Python: the CV stack (ultralytics, OpenCV, Shapely) for geometry and the
VLM stack (torch, transformers, peft, ...) for the room-label pass. This
module installs them into Blender's user site-packages via pip in a
subprocess, reports what is missing, and downloads the Qwen2.5-VL base
checkpoint into the Hugging Face cache. UI lives in preferences.py.

Pure Python, no bpy — unit-tested outside Blender.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# (pip requirement, import name). Versions pinned to what the pipeline was
# validated with; transformers < 5 because the Qwen2.5-VL processor API is
# what inference.py targets.
CV_PACKAGES: list[tuple[str, str]] = [
    ("ultralytics>=8.3", "ultralytics"),
    ("opencv-python-headless>=4.9", "cv2"),
    ("shapely>=2.0", "shapely"),
    ("pyyaml>=6", "yaml"),
    ("pillow>=10", "PIL"),
    ("numpy>=1.24", "numpy"),
]
VLM_PACKAGES: list[tuple[str, str]] = [
    ("torch>=2.3", "torch"),
    ("transformers>=4.49,<5", "transformers"),
    ("peft>=0.12", "peft"),
    ("accelerate>=0.33", "accelerate"),
    ("qwen-vl-utils>=0.0.10", "qwen_vl_utils"),
    ("huggingface_hub>=0.25", "huggingface_hub"),
    ("pillow>=10", "PIL"),
    ("numpy>=1.24", "numpy"),
]
MLX_VLM_PACKAGES: list[tuple[str, str]] = [
    ("mlx-vlm>=0.7", "mlx_vlm"),
    ("transformers>=4.49,<5", "transformers"),
    ("huggingface_hub>=0.25", "huggingface_hub"),
    ("qwen-vl-utils>=0.0.10", "qwen_vl_utils"),
    ("pillow>=10", "PIL"),
    ("numpy>=1.24", "numpy"),
]
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
BASE_MODEL_SIZE_GB = {"torch": 15, "mlx": 8}


def vlm_packages(backend: str) -> list[tuple[str, str]]:
    """Packages the room-label pass needs for `backend` ('mlx' or 'torch')."""
    return MLX_VLM_PACKAGES if backend == "mlx" else VLM_PACKAGES


def blender_python() -> str:
    """Blender's bundled interpreter (sys.executable inside Blender)."""
    return sys.executable


def missing_packages(python: str, packages: list[tuple[str, str]], timeout: int = 120) -> list[str]:
    """Import-probe `python` for each package; return the pip names missing.

    One subprocess for all imports (a torch import alone takes seconds).
    A failed probe (no such interpreter, timeout) reports everything as
    missing rather than guessing.
    """
    names = [imp for _, imp in packages]
    script = (
        "import importlib,sys\n"
        "missing=[]\n"
        f"for m in {names!r}:\n"
        "    try: importlib.import_module(m)\n"
        "    except Exception: missing.append(m)\n"
        "print('MISSING:'+','.join(missing))\n"
    )
    try:
        r = subprocess.run([python, "-c", script], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return [req for req, _ in packages]
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("MISSING:")), None)
    if r.returncode != 0 or line is None:
        return [req for req, _ in packages]
    missing_imports = set(filter(None, line[len("MISSING:"):].split(",")))
    return [req for req, imp in packages if imp in missing_imports]


def pip_install_command(python: str, requirements: list[str], user_site: bool = True) -> list[str]:
    """The pip invocation used by the installer (kept separate for tests)."""
    cmd = [python, "-m", "pip", "install", "--upgrade", "--no-input"]
    if user_site:
        # Blender's site-packages is inside the app bundle; user site keeps
        # the install out of the application and survives Blender updates
        # of the same Python minor version.
        cmd.append("--user")
    return cmd + requirements


def run_streaming(cmd: list[str], log_cb, env: dict | None = None) -> int:
    """Run `cmd`, feeding each output line to `log_cb(str)`. Returns the exit code."""
    log_cb("$ " + " ".join(cmd))
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env)
    except OSError as e:
        log_cb(f"failed to start: {e}")
        return 127
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            log_cb(line)
    return proc.wait()


def install_packages(python: str, packages: list[tuple[str, str]], log_cb) -> int:
    """pip-install the missing entries of `packages` into `python`'s user site."""
    missing = missing_packages(python, packages)
    if not missing:
        log_cb("all packages already installed")
        return 0
    return run_streaming(pip_install_command(python, missing), log_cb)


def download_base_model(python: str, log_cb, model_id: str = DEFAULT_BASE_MODEL,
                        cache_dir: str | None = None) -> int:
    """Download `model_id` into the Hugging Face cache using `python`
    (which must have huggingface_hub). Streams hub progress to log_cb."""
    script = (
        "import sys\n"
        "from huggingface_hub import snapshot_download\n"
        f"p = snapshot_download({model_id!r}, cache_dir={cache_dir!r}, max_workers=4)\n"
        "print('DOWNLOADED:' + p)\n"
    )
    return run_streaming([python, "-c", script], log_cb)


def bundled_vlm_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "vlm"
