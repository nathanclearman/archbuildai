"""
Local VLM client for floor plan parsing.

Wraps the fine-tuned Qwen2.5-VL model (see model/ for training) and returns
structured floor plan JSON for the Blender geometry layer to consume.

The model itself runs outside Blender's bundled Python — this client shells
out to model/inference.py so the heavy ML dependencies (torch, transformers,
mlx) don't need to be installed into Blender.

VLM requests use a persistent daemon (`inference.py --serve`) so the
~30-90s model-load cost is paid once per Blender session, not per Generate
click. CV-only requests bypass the daemon — they don't touch the model
and starting one for a single OpenCV call would be pure overhead. See
`_ensure_daemon` and `_predict_via_daemon` for the protocol.
"""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path


_ADDON_DIR = Path(__file__).resolve().parent.parent
BUNDLED_VLM_DIR = _ADDON_DIR / "vlm"          # written by package.py from floorplan3d/model/


def _resolve_model_dir() -> Path:
    """Locate the VLM runtime dir (inference.py + optional weights/).

    Order: FP3D_QWEN_MODEL_DIR env var → the repo's model/ dir next to
    this add-on (dev layout, or an install symlinked to the repo) → the
    `vlm/` runtime bundled into the packaged add-on. No home-directory
    guessing: a shipped add-on must work on any machine.
    """
    env = os.environ.get("FP3D_QWEN_MODEL_DIR")
    if env:
        return Path(env).expanduser().resolve()
    sibling = _ADDON_DIR.parent / "model"
    if (sibling / "inference.py").exists():
        return sibling
    return BUNDLED_VLM_DIR


def _resolve_weights_dir(model_dir: Path) -> Path:
    """FP3D_WEIGHTS_DIR env var (set from the add-on preferences) → model_dir/weights."""
    env = os.environ.get("FP3D_WEIGHTS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return model_dir / "weights"


MODEL_DIR = _resolve_model_dir()
DEFAULT_WEIGHTS_DIR = _resolve_weights_dir(MODEL_DIR)
INFERENCE_SCRIPT = MODEL_DIR / "inference.py"


def reconfigure() -> None:
    """Re-read the env overrides (called when the add-on preferences change)
    and forget the memoized interpreter so the next predict re-resolves."""
    global MODEL_DIR, DEFAULT_WEIGHTS_DIR, INFERENCE_SCRIPT
    MODEL_DIR = _resolve_model_dir()
    DEFAULT_WEIGHTS_DIR = _resolve_weights_dir(MODEL_DIR)
    INFERENCE_SCRIPT = MODEL_DIR / "inference.py"
    _RESOLVED_PYTHON_CACHE.clear()


# Probe timeout for `import torch` on a candidate interpreter. Torch's first
# import is dominated by shared-object loading (~1-3s on a warm disk, up to
# 8s on cold NVMe). 15s covers the slow path without letting a hung python
# freeze Blender's UI thread forever.
_PROBE_TIMEOUT_S = 15

# Daemon-startup timeout. inference.py --serve has to import torch +
# transformers + load the ~14 GB bf16 model from disk + initialize CUDA
# before printing "READY" on stderr:
#   torch import                  ~1-3 s
#   transformers import           ~2-5 s
#   Qwen2.5-VL weights load       20-40 s (disk-bound; worse on externals)
#   CUDA / MPS context init       2-10 s
#   adapter merge                 1-3 s
# Total: 30-60 s typical, up to ~120 s under swap pressure or on a cold
# NVMe. 180 s leaves headroom for any of those without letting a stuck
# daemon (corrupt weights, dead disk) hang the client forever.
_DAEMON_READY_TIMEOUT_S = 180

# Decoding budget sent with every request. Real MLS plans need 3000-4000
# tokens; inference.py's own default is the same 6144. Override with
# FP3D_MAX_NEW_TOKENS for unusually large plans.
_DEFAULT_MAX_NEW_TOKENS = 6144


def _max_new_tokens() -> int:
    env = os.environ.get("FP3D_MAX_NEW_TOKENS")
    try:
        return int(env) if env else _DEFAULT_MAX_NEW_TOKENS
    except ValueError:
        return _DEFAULT_MAX_NEW_TOKENS


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


# Mirrors inference.MLX_MODEL_ID / DEFAULT_BASE_MODEL (kept here so the
# add-on can report status without importing the runtime).
TORCH_BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
MLX_BASE_MODEL = "mlx-community/Qwen2.5-VL-7B-Instruct-8bit"


def preferred_backend() -> str:
    """'mlx' on Apple Silicon (fast path), else 'torch'. FP3D_VLM_BACKEND
    env var (torch|mlx) overrides; the daemon applies the same rule."""
    import platform
    env = (os.environ.get("FP3D_VLM_BACKEND") or "auto").lower()
    if env in ("torch", "mlx"):
        return env
    return "mlx" if (sys.platform == "darwin" and platform.machine() == "arm64") else "torch"


def _probe_imports() -> str:
    """Imports an interpreter must satisfy to host the daemon on this backend."""
    if preferred_backend() == "mlx":
        return "import mlx_vlm, transformers"
    return "import torch, transformers, peft"


_PROBE_IMPORTS = _probe_imports()


def _probe_python(candidate: str) -> bool:
    """True iff `candidate` can import the VLM stack within the probe budget.

    Probes torch + transformers + peft together: an interpreter with only
    torch (e.g. a YOLO-era venv, or Blender's own python with torch
    pip-installed) would pass a torch-only probe and then fail inside
    inference.py with a far less actionable error.

    We import rather than check the path because a venv's python may be a
    symlink whose name doesn't carry its installed packages. A silent
    non-zero return (missing module, syntax error, permission denied) all
    collapse to False — caller decides what to do with that.
    """
    try:
        result = subprocess.run(
            [candidate, "-c", _probe_imports()],
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


def _known_venv_pythons() -> list[str]:
    """Interpreters from the project's own virtualenvs that exist on disk.

    Tried last, after the PATH candidates, so they never shadow an
    explicit user setup — but they make the add-on work out of the box
    from Blender (whose PATH usually has no ML python) when the repo's
    venv is present. Covers POSIX (`bin/python`) and Windows
    (`Scripts/python.exe`) layouts.
    """
    repo_root = MODEL_DIR.parent.parent
    roots = [repo_root / ".venv", repo_root / "floorplan_env", MODEL_DIR / "venv"]
    found: list[str] = []
    for root in roots:
        for rel in ("bin/python", "Scripts/python.exe"):
            candidate = root / rel
            if candidate.exists():
                found.append(str(candidate))
    return found


# Memoized resolution keyed on the FP3D_PYTHON env value. The resolver
# spawns up to 3 `import torch` subprocesses (15 s timeout each) on a
# cold miss; without this cache, every `LocalModelClient()` in the
# Blender operator re-probes on every Generate click. Keyed on env so a
# user who sets FP3D_PYTHON mid-session gets a fresh resolution.
# Injected probes (test seams) bypass the cache — see _resolve_python_bin.
_RESOLVED_PYTHON_CACHE: dict[str | None, str] = {}


def _resolve_python_bin(probe=None) -> str:
    """Find a Python interpreter with the VLM stack installed.

    Resolution order:
      1. `FP3D_PYTHON` env var (explicit user override, no probe — trust it)
      2. `sys.executable` (only when NOT Blender's bundled interpreter)
      3. `python3`, `python` on PATH
      4. the project's own venvs (see _known_venv_pythons)
      5. Blender's bundled interpreter (only useful after the add-on
         preferences installed the VLM packages into it)

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
    candidates.extend(_known_venv_pythons())
    if blender_python:
        # Blender's own interpreter goes LAST: it normally has no ML stack,
        # but the add-on's "Install VLM dependencies" button can put one in
        # its user site-packages, and then it is the shipped, no-repo path.
        candidates.append(sys.executable)

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
            "'python3' / 'python' on PATH and the project venvs could not "
            "import torch/transformers/peft either."
        )
    else:
        diagnosis = (
            f"Tried {sys.executable!r}, 'python3', 'python', and the project "
            "venvs; none could import torch/transformers/peft. Your ML "
            "environment may not be installed, or it may not be on PATH."
        )
    raise RuntimeError(
        "Could not find a Python interpreter with the VLM stack "
        "(torch, transformers, peft) installed. Open Edit > Preferences > "
        "Add-ons > FloorPlan3D and click 'Install VLM dependencies', or set "
        "FP3D_PYTHON to a Python that has them (see requirements.txt). "
        f"{diagnosis}"
    )


def hf_cache_dir() -> Path:
    """Hugging Face hub cache, honouring HF_HUB_CACHE / HF_HOME like the hub does."""
    env = os.environ.get("HF_HUB_CACHE")
    if env:
        return Path(env).expanduser()
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def is_base_model_cached(model_id: str) -> bool:
    """True when a complete snapshot of `model_id` is in the hub cache."""
    repo = hf_cache_dir() / ("models--" + model_id.replace("/", "--"))
    snaps = repo / "snapshots"
    if not snaps.is_dir():
        return False
    return any((s / "config.json").exists() for s in snaps.iterdir() if s.is_dir())


def environment_status(probe=None) -> dict:
    """What the VLM path needs and whether it is there. Runs interpreter
    probes (subprocesses, seconds) — call from a background thread, never
    from a UI draw() callback."""
    status = {
        "inference_script": INFERENCE_SCRIPT.exists(),
        "model_dir": str(MODEL_DIR),
        "python": None,
        "python_error": "",
        "base_model_cached": False,
        "adapter": (DEFAULT_WEIGHTS_DIR / "adapter").is_dir(),
    }
    try:
        status["python"] = _resolve_python_bin(probe=probe)
    except RuntimeError as e:
        status["python_error"] = str(e)
    backend = preferred_backend()
    status["backend"] = backend
    if backend == "mlx":
        base = MLX_BASE_MODEL
    else:
        base = TORCH_BASE_MODEL
        cfg = DEFAULT_WEIGHTS_DIR / "train_config.json"
        if cfg.exists():
            try:
                base = json.loads(cfg.read_text()).get("base_model", base)
            except (json.JSONDecodeError, OSError):
                pass
    status["base_model"] = base
    status["base_model_cached"] = is_base_model_cached(base)
    status["ready"] = bool(status["inference_script"] and status["python"] and status["base_model_cached"])
    return status


class LocalModelClient:
    """Client for the fine-tuned floor plan VLM.

    Holds a persistent `inference.py --serve` daemon that VLM requests
    are dispatched to over JSON-line pipes. Spawned lazily on the first
    VLM predict() and reused across all subsequent ones, so a Blender
    session pays the 30-90 s model-load cost once instead of per click.

    Threading: predict() is safe to call from multiple threads — a
    single lock serializes daemon I/O so concurrent requests don't
    interleave bytes on the shared subprocess pipes. The model itself
    is the bottleneck (one decode at a time on the GPU), so the lock
    isn't a real perf cost — it just makes the protocol correct under
    Blender's modal-operator threading.
    """

    def __init__(self, weights_dir=None, python_bin=None, timeout=300,
                 daemon_ready_timeout=_DAEMON_READY_TIMEOUT_S):
        # `timeout` is the per-call ceiling for the one-shot CV-only
        # subprocess.run path; the daemon path doesn't use it (the
        # daemon is long-lived). 300 s covers a slow cold CV-only run
        # without letting a stuck OpenCV call freeze the UI forever.
        #
        # `daemon_ready_timeout` bounds how long _ensure_daemon will
        # wait for the daemon to print READY on stderr before giving
        # up — see _DAEMON_READY_TIMEOUT_S for the budget breakdown.
        self.weights_dir = Path(weights_dir) if weights_dir else DEFAULT_WEIGHTS_DIR
        # Resolve lazily-but-eagerly: if caller passed an explicit path,
        # trust it. Otherwise probe for a torch-capable interpreter and
        # cache the result. The previous default (sys.executable) silently
        # routed every Blender-initiated predict() to Blender's bundled
        # Python, which has no torch — predict() then raised with a
        # generic "Model inference failed" instead of a fixable message.
        self.python_bin = python_bin if python_bin else _resolve_python_bin()
        self.timeout = timeout
        self.daemon_ready_timeout = daemon_ready_timeout
        # Lazy daemon: only spawned on the first non-cv_only predict().
        # cv_only-only clients (eval scripts that skip the VLM) pay
        # zero daemon cost.
        self._daemon: subprocess.Popen | None = None
        # Remember the quantize config the running daemon was started
        # with. If a subsequent predict() asks for a different value
        # we have to restart — quantize affects model loading, not
        # per-request behavior, so a running daemon can't switch.
        self._daemon_quantize: bool = False
        # Serializes daemon I/O. Without it, two concurrent predict()
        # calls could interleave their JSON request lines on stdin and
        # then race to readline() the responses, mismatching results
        # to threads. Cheap lock; the actual generate() is what's slow.
        self._lock = threading.Lock()

    def predict(self, image_path, cv_only=False, refine=False, quantize=False):
        """Run inference on a floor plan image.

        Args:
            image_path: Path to the floor plan image file.
            cv_only: If True, skip the VLM and use the OpenCV fallback only.
                     Routed through a one-shot subprocess (no daemon spinup);
                     starting a daemon for a single OpenCV call would be
                     pure overhead.
            refine: If True, run the optional Claude Opus refinement pass.
                    Requires ANTHROPIC_API_KEY in the environment.
            quantize: If True, load the VLM in 4-bit NF4 so the ~14 GB
                      bfloat16 model fits on 16 GB GPUs (5070 Ti, 4090).
                      Default False — the primary target (M4 Max, 128 GB
                      unified) doesn't need it and quality is ~5-10%
                      better without it. Changing this between calls
                      transparently restarts the daemon (model loading
                      is the only thing quantize affects).

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

        if cv_only:
            # CV path skips the VLM entirely. Keep the old one-shot
            # subprocess path for these — starting a daemon for a
            # single OpenCV call (which runs in well under 1 s and
            # doesn't touch the GPU) would be pure overhead.
            return self._predict_via_one_shot(image_path, cv_only=True,
                                              refine=refine, quantize=quantize)

        return self._predict_via_daemon(image_path, refine=refine,
                                        quantize=quantize)

    def ocr_labels(self, image_path, quantize=False):
        """Read printed room names + pixel boxes off a plan (hybrid backend).

        Routed through the same daemon as predict() — the base model
        answers with the adapter disabled — so it shares the one model
        load per session. Returns [{"text", "bbox_px": [x1, y1, x2, y2]}]
        in original image pixels.
        """
        image_path = str(image_path)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not INFERENCE_SCRIPT.exists():
            raise FileNotFoundError(f"Inference script not found at {INFERENCE_SCRIPT}.")
        return self._request_via_daemon({"op": "ocr_labels", "image": image_path},
                                        quantize=quantize)

    def ocr_dimensions(self, image_path, quantize=False):
        """Dimension strings + pixel boxes off a plan (auto-scale pass).
        Same daemon, same return shape as ocr_labels()."""
        image_path = str(image_path)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not INFERENCE_SCRIPT.exists():
            raise FileNotFoundError(f"Inference script not found at {INFERENCE_SCRIPT}.")
        return self._request_via_daemon({"op": "ocr_dimensions", "image": image_path},
                                        quantize=quantize)

    def ocr_crop(self, image_path, bbox_px, quantize=False):
        """Room name printed inside one pixel box of the plan ("" if none)."""
        image_path = str(image_path)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        return self._request_via_daemon(
            {"op": "ocr_crop", "image": image_path, "bbox_px": [float(v) for v in bbox_px]},
            quantize=quantize)

    def _predict_via_daemon(self, image_path, refine, quantize):
        req = {"image": image_path, "refine": bool(refine),
               "max_new_tokens": _max_new_tokens()}
        return self._request_via_daemon(req, quantize=quantize)

    def _request_via_daemon(self, req, quantize):
        """Send one request to the persistent daemon, return the parsed
        response. Spawns the daemon on first call; respawns if the
        previous daemon died between calls (OOM, crash)."""
        with self._lock:
            self._ensure_daemon_unlocked(quantize=quantize)
            assert self._daemon is not None  # _ensure_daemon raises on failure
            try:
                self._daemon.stdin.write(json.dumps(req) + "\n")
                self._daemon.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                # Daemon died between _ensure_daemon and write. Drain
                # stderr (for diagnosis), drop the dead handle so the
                # next call respawns.
                stderr = self._drain_daemon_stderr_unlocked()
                self._close_daemon_unlocked()
                raise RuntimeError(
                    f"Daemon disconnected during write ({e}). "
                    f"Daemon stderr: {stderr or '(empty)'}"
                )

            line = self._daemon.stdout.readline()
            if not line:
                # EOF on stdout — daemon exited mid-request. Same drain-
                # and-respawn logic as above.
                stderr = self._drain_daemon_stderr_unlocked()
                self._close_daemon_unlocked()
                raise RuntimeError(
                    f"Daemon disconnected during read. "
                    f"Daemon stderr: {stderr or '(empty)'}"
                )
            try:
                response = json.loads(line)
            except json.JSONDecodeError as e:
                # Daemon emitted something other than valid JSON. This
                # is a protocol violation — kill it and raise so the
                # user sees a fixable diagnostic instead of a cryptic
                # downstream KeyError.
                self._close_daemon_unlocked()
                raise RuntimeError(
                    f"Daemon emitted non-JSON line ({e}): {line!r}"
                )

        if not response.get("ok"):
            raise RuntimeError(
                f"Inference failed: {response.get('error', '<no error message>')}"
            )
        return response["result"]

    def _predict_via_one_shot(self, image_path, cv_only, refine, quantize):
        """Original one-shot path. Used for cv_only requests. Kept as
        a per-call subprocess.run because the CV pipeline runs in
        well under 1 s and a daemon would be pure overhead."""
        cmd = [
            self.python_bin,
            str(INFERENCE_SCRIPT),
            "--image", image_path,
            "--weights", str(self.weights_dir),
            "--output", "json",
            "--max-new-tokens", str(_max_new_tokens()),
        ]
        if cv_only:
            cmd.append("--cv-only")
        if refine:
            cmd.append("--refine")
        if quantize:
            cmd.append("--quantize")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Model inference failed: {result.stderr}")

        return json.loads(result.stdout)

    def _ensure_daemon_unlocked(self, quantize):
        """Spawn the daemon if it's not running, or restart it if the
        previous spawn used a different quantize config. Caller holds
        self._lock. Blocks until the daemon writes 'READY' on stderr."""
        # Reap a dead daemon so the respawn path takes over.
        if self._daemon is not None and self._daemon.poll() is not None:
            self._daemon = None
        # Restart if quantize config has changed since last spawn —
        # the running model has the wrong weights for the new request.
        if self._daemon is not None and self._daemon_quantize != quantize:
            self._close_daemon_unlocked()
        if self._daemon is not None:
            return

        cmd = [
            self.python_bin,
            str(INFERENCE_SCRIPT),
            "--serve",
            "--weights", str(self.weights_dir),
        ]
        if quantize:
            cmd.append("--quantize")

        # bufsize=1 = line-buffered (text mode): readline() returns
        # promptly per emitted line instead of waiting for a full
        # block. Critical for the READY handshake and the per-request
        # response framing — without it the daemon's output would sit
        # in the pipe buffer for kilobytes before we saw any of it.
        self._daemon = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._daemon_quantize = quantize

        try:
            self._wait_for_ready_unlocked()
        except BaseException:
            # Failed startup: kill the partial daemon so we don't
            # leak a zombie process. _close_daemon_unlocked is
            # idempotent.
            self._close_daemon_unlocked()
            raise

    def _wait_for_ready_unlocked(self):
        """Block until the daemon writes 'READY' on stderr. Caller
        holds self._lock. Raises RuntimeError if the daemon exits or
        the ready deadline elapses first. Forwards any non-READY
        stderr lines to the parent stderr so user-visible diagnostics
        (model-load progress, warnings) aren't silently swallowed."""
        import time
        deadline = time.monotonic() + self.daemon_ready_timeout
        while True:
            if self._daemon.poll() is not None:
                # Daemon exited before printing READY.
                stderr_remaining = self._drain_daemon_stderr_unlocked()
                raise RuntimeError(
                    f"Daemon exited with code {self._daemon.returncode} "
                    f"before becoming ready. "
                    f"Stderr: {stderr_remaining or '(empty)'}"
                )
            # No portable way to put a true timeout on readline() in
            # stdlib subprocess without a reader thread or select() —
            # both fragile across Windows / macOS / Linux. The poll()
            # check above gives a responsive death-detection path;
            # the deadline check below surfaces a stuck-but-alive
            # daemon eventually.
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"Daemon failed to print READY within "
                    f"{self.daemon_ready_timeout}s. Process is still "
                    "running but may be stuck on model load or CUDA "
                    "init. Kill it manually if this persists."
                )
            line = self._daemon.stderr.readline()
            if not line:
                # EOF before READY — daemon exited.
                stderr_remaining = self._drain_daemon_stderr_unlocked()
                raise RuntimeError(
                    f"Daemon closed stderr before becoming ready. "
                    f"Stderr: {stderr_remaining or '(empty)'}"
                )
            line = line.rstrip("\n")
            if line == "READY":
                return
            # Forward non-READY lines so the user sees model-load
            # progress and warnings in the same place as other
            # Blender-add-on stderr.
            if line:
                print(f"[fp3d-daemon] {line}", file=sys.stderr)

    def _drain_daemon_stderr_unlocked(self):
        """Best-effort read of any buffered stderr from a dying daemon.
        Returns a single string. Safe to call only on a dead or
        about-to-be-killed daemon — blocking until EOF on a live
        daemon would deadlock. Caller holds self._lock."""
        if self._daemon is None or self._daemon.stderr is None:
            return ""
        try:
            return self._daemon.stderr.read() or ""
        except Exception:
            return ""

    def close(self):
        """Shut down the daemon if running. Idempotent; safe to call
        from __del__. Never raises."""
        try:
            with self._lock:
                self._close_daemon_unlocked()
        except Exception:
            pass

    def _close_daemon_unlocked(self):
        """Close the daemon. Caller holds self._lock."""
        if self._daemon is None:
            return
        # Close stdin first — that's the daemon's exit signal (the
        # `for line in sys.stdin` loop in _run_serve ends on EOF).
        # Daemon finishes any in-flight request, exits the loop,
        # returns; wait() then collects the status.
        try:
            if self._daemon.stdin and not self._daemon.stdin.closed:
                self._daemon.stdin.close()
        except Exception:
            pass
        try:
            self._daemon.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Daemon ignored EOF — force-kill. Rare; only happens
            # when generate() is stuck in a kernel call that isn't
            # interruptible (e.g. corrupted CUDA context).
            self._daemon.kill()
            try:
                self._daemon.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        except Exception:
            pass
        self._daemon = None

    def __del__(self):
        # Safety net: if the user forgot to call close() and the
        # client is being GC'd, kill the daemon so we don't leak a
        # zombie process across Blender sessions. __del__ during
        # interpreter shutdown is unreliable, hence the broad except.
        try:
            self.close()
        except Exception:
            pass
