# FloorPlan3D Blender Addon — Session Handoff

## Where you're picking up

The ML pipeline is trained and validated end-to-end outside Blender:
image → Qwen2.5-VL 7B (LoRA) → canonical schema JSON. The addon's
job is to add **Qwen** as a fourth option in the Model dropdown
(currently: CubiCasa / YOLO / Premium Vision) and wire it to the
existing inference script.

End-to-end state:

- ✅ Trained Qwen2.5-VL 7B LoRA adapter on disk
  (`~/Desktop/archbuildai/floorplan3d/model/weights/adapter/`, ~363 MB)
- ✅ `inference.py` smoke-tested on the M4 Max (April 2026 session):
  loaded in ~7s from warm disk, produced 1087 bytes of valid schema JSON
  from a CubiCasa fixture. Schema validation passed.
- ✅ `floorplan3d/blender_addon/geometry.py` in the archbuildai repo
  has been hardened against the four silent-failure modes most likely
  to bite on first-run-in-Blender (see "Reference geometry module" below)
- ❓ Addon installed in Blender but UI doesn't match repo addon — the
  Mac has a locally-modified version with CubiCasa/YOLO/Premium Vision
  model dropdown, Sensitivity slider, Stories counter, "Review Before 3D"
  toggle, "Generate Sample" button. None of that is in the archbuildai
  repo; it lives only on the user's Mac master branch at `a096cdb`,
  unpushed.
- ❓ Qwen option not yet wired into the Model dropdown
- ❓ VLM → Blender round trip never run inside real Blender

## Verified commands (run on the M4 Max, April 2026)

These are the exact commands that worked. Use them as ground truth
for what the Qwen branch of the addon should reproduce.

```bash
cd ~/Desktop/archbuildai
.venv/bin/python floorplan3d/model/inference.py \
  --image floorplan3d/tests/fixtures/cubicasa_mini/sample_0001/F1_scaled.png \
  --weights floorplan3d/model/weights \
  --output json > /tmp/out.json 2> /tmp/out.err
```

Output (first 500 chars of `/tmp/out.json`):

```json
{"scale":{"pixels_per_meter":40},"walls":[{"start":[7.52,0.0],"end":[7.52,3.91],"thickness":0.15},...
```

Validates cleanly against the canonical schema.

## Environment

Confirmed working on the Mac:

| Thing | Value |
|-------|-------|
| Python | 3.14.3 (venv at `~/Desktop/archbuildai/.venv/bin/python`) |
| torch | 2.11.0, MPS available |
| transformers | 4.57.6 |
| peft | 0.19.1 |
| Weights | `floorplan3d/model/weights/` containing `adapter/` + `train_config.json` |
| Test images | `floorplan3d/tests/fixtures/cubicasa_mini/sample_0001/F1_scaled.png` and `floorplan3d/tests/fixtures/real_mls/samples/listing_*.{png,jpg}` |

Cold-start timing on warm disk: ~7 s to load 5 checkpoint shards. Expect
first inference in the addon to take 15–60 s total while the subprocess
spins up. Plan for a timeout of at least 300 s.

## Canonical JSON schema (contract with the geometry layer)

The geometry layer in archbuildai consumes this exact shape. Don't
reshape it in the addon; match it.

```json
{
  "scale": {"pixels_per_meter": 50},
  "walls": [
    {"start": [0.0, 0.0], "end": [4.2, 0.0], "thickness": 0.15}
  ],
  "doors": [
    {"position": [2.1, 0.0], "width": 0.9, "type": "hinged", "wall_index": 0}
  ],
  "windows": [
    {"position": [1.0, 3.5], "width": 1.2, "wall_index": 2}
  ],
  "rooms": [
    {"label": "bedroom",
     "polygon": [[0,0],[4.2,0],[4.2,3.5],[0,3.5]],
     "area": 14.7}
  ]
}
```

- Coordinates are in meters.
- `wall_index = -1` is the "not attached to any wall" sentinel — skip
  those when building geometry.
- `position` is canonical `[x, y]` absolute; older fixtures use a scalar
  distance-along-wall. `_project_position_to_wall` in the repo's
  geometry.py handles both.
- Validator lives at `floorplan3d/model/schema.py::validate`. Call it
  on every model output before passing to the geometry layer — the VLM
  occasionally emits fields with the wrong type under high-temperature
  sampling.

## Wiring Qwen into the Model dropdown

The Model dropdown already has 3 options. Add a 4th and route it to
the same subprocess that worked in the verified command above.

### Minimum wiring sketch

```python
# panels.py — add to the Model EnumProperty
('QWEN', "Qwen 2.5-VL (trained)", "Fine-tuned Qwen2.5-VL 7B LoRA"),

# operators.py — in the Generate handler, after Model selection
if scene.fp3d_model == 'QWEN':
    from .api.qwen_client import QwenClient
    client = QwenClient()
    floor_plan = client.predict(scene.fp3d_image_path)
    build_geometry(floor_plan)  # whatever the addon's geometry entry point is

# api/qwen_client.py — new file
import json, os, subprocess, sys
from pathlib import Path

# Adjust to your repo layout. This assumes the addon and model live
# in the same archbuildai repo; if the addon repo is separate, make
# the model path configurable via a preferences property.
REPO_ROOT = Path(os.environ.get(
    "FP3D_MODEL_ROOT",
    str(Path.home() / "Desktop" / "archbuildai"),
))
INFERENCE_SCRIPT = REPO_ROOT / "floorplan3d" / "model" / "inference.py"
WEIGHTS_DIR = REPO_ROOT / "floorplan3d" / "model" / "weights"


def _resolve_python_bin():
    """Find a Python with torch installed. Blender's bundled Python
    does NOT have ML deps — falling through to it gives a cryptic
    ModuleNotFoundError. FP3D_PYTHON env var is the explicit escape
    hatch. See archbuildai's local_model.py for the full resolver
    with probe-based fallback."""
    env = os.environ.get("FP3D_PYTHON")
    if env:
        return env
    # sys.executable inside Blender IS Blender's bundled python.
    # Don't use it. Hard-require FP3D_PYTHON for Qwen path.
    raise RuntimeError(
        "FP3D_PYTHON env var is required for the Qwen model. "
        "Launch Blender from a terminal with e.g.: "
        "export FP3D_PYTHON=~/Desktop/archbuildai/.venv/bin/python && open -a Blender"
    )


class QwenClient:
    def __init__(self, timeout=300):
        self.python_bin = _resolve_python_bin()
        self.timeout = timeout

    def predict(self, image_path):
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not INFERENCE_SCRIPT.exists():
            raise FileNotFoundError(
                f"Inference script not found: {INFERENCE_SCRIPT}. "
                "Set FP3D_MODEL_ROOT to the archbuildai repo root."
            )
        cmd = [
            self.python_bin, str(INFERENCE_SCRIPT),
            "--image", str(image_path),
            "--weights", str(WEIGHTS_DIR),
            "--output", "json",
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=self.timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Qwen inference failed: {result.stderr.strip()}")
        return json.loads(result.stdout)
```

### Threading

The subprocess takes 15–60 s on a cold first call. Blocking the main
thread freezes Blender's UI. Run the subprocess in a background
thread and poll from a modal operator using `event_timer_add` — see
`archbuildai/floorplan3d/blender_addon/operators.py::FP3D_OT_GenerateModel`
for a working pattern you can copy.

## Reference geometry module (archbuildai)

The hardened geometry layer lives at:

- Repo: `nathanclearman/archbuildai`
- Branch: `claude/blender-floorplan-integration-GW9Yp`
- File: `floorplan3d/blender_addon/geometry.py`
- Head commit: `1da3bc8 Harden Blender geometry for first real in-Blender run`

What it does (given canonical JSON):

1. `generate_walls` — extrudes rectangles to `wall_height`
2. `generate_door_openings` — boolean-DIFFERENCE cutters (safe-applied;
   failures roll back both modifier and cutter)
3. `generate_window_openings` — same, with sill height offset
4. `generate_floors` — CCW-normalized polygons with `+Z` face normal
5. `generate_ceilings` — CCW-normalized, normal flipped to `-Z`
6. `generate_room_labels` — text objects sized proportional to room
   bbox (clamped [0.2m, 1.0m])

Safety features pinned by 39 unit tests (run in CI without Blender
via mocked bpy):

- CCW winding normalization — floors/ceilings face the right way
  regardless of upstream polygon order
- `wall_index = -1` sentinel skipped for doors and windows
- Boolean modifier apply uses `bpy.context.temp_override` not
  `view_layer.objects.active` mutation
- Failed modifier apply rolls back modifier + cutter, so subsequent
  cutters on the same wall don't inherit a broken modifier stack
- Door/window position clamped to stay inside wall span
- Label size proportional to room bbox minor axis

Two choices for the addon:

1. **Copy the geometry module** into the addon repo. Simple, but it
   diverges over time. Update plan: periodically rebase against
   archbuildai's `floorplan3d/blender_addon/geometry.py`.
2. **Symlink or submodule**. Clean, but complicates installation
   (addon.zip won't preserve symlinks across users).

If the addon already has its own geometry code, either:
- Replace it with the archbuildai version (simpler, battle-tested),
  OR
- Keep the addon's version and port just the safety fixes from
  commit `1da3bc8` (GEO-3/4/5/6/8 — see that commit message).

## What's known to NOT work

- The version of the addon currently installed in Blender on the Mac
  has a "Generate Sample" button and model selector that are not
  wired to anything that produced geometry when clicked. User reports
  "not working" — specific error unknown because the console wasn't
  captured.
- The FP3D_PYTHON env var is not being set by the current addon
  launch path. Without it the subprocess falls back to Blender's
  bundled Python, which can't import torch.

## Minimum viable test

Once Qwen is wired:

1. `export FP3D_PYTHON=~/Desktop/archbuildai/.venv/bin/python && open -a Blender`
2. N-panel → pick any image from
   `~/Desktop/archbuildai/floorplan3d/tests/fixtures/cubicasa_mini/sample_0001/F1_scaled.png`
3. Model dropdown → select Qwen
4. Click Generate. UI will freeze 15–60 s on the first call.
5. Expected: walls + floor appear. Doors/windows may or may not cut
   cleanly depending on the chosen image and boolean solver.

Working fallback if Qwen path is broken but you want to validate the
geometry layer in isolation:

```bash
# From terminal, generate JSON:
~/Desktop/archbuildai/.venv/bin/python \
  ~/Desktop/archbuildai/floorplan3d/model/inference.py \
  --image <your-image> \
  --weights ~/Desktop/archbuildai/floorplan3d/model/weights \
  --output json > /tmp/out.json
```

Then use the addon's JSON-override input (if it exists) to load
`/tmp/out.json` directly, bypassing the subprocess layer.

## Gotchas

Ranked by likelihood:

1. **FP3D_PYTHON not set** → subprocess uses Blender's bundled Python
   → `ModuleNotFoundError: torch`. Always launch Blender from a
   terminal that has FP3D_PYTHON exported, or set it via Blender
   preferences.

2. **Weights path** → weights dir layout is
   `weights/adapter/` + `weights/train_config.json`. Pass the parent
   (`weights/`), not the adapter subdir, as `--weights`.

3. **Python 3.14 venv** → `torch_dtype` deprecation warning on
   Python 3.14 is cosmetic; model still loads. The "temperature
   flag ignored" warning is also cosmetic. If you see a REAL error,
   it'll be after those.

4. **Cold load time** → the 7 GB Qwen 7B model takes 20–60 s to load
   from disk on first call. Addon timeout should be ≥ 300 s.

5. **Subprocess stdout must be clean JSON** → if you add print
   statements to inference.py for debugging, make sure they go to
   stderr (`print(..., file=sys.stderr)`), not stdout. The addon
   parses stdout as JSON and will fail on any stray text.

6. **Blender `bpy` not thread-safe** → the subprocess can run in a
   background thread, but all `bpy.data.*` and `bpy.ops.*` calls
   must happen on the main thread. Pass results back via a modal
   operator timer.

7. **Image path spaces** → if paths may contain spaces, `subprocess.run`
   with `shell=False` (default) handles them correctly. Don't rewrite
   to `shell=True`.

## What the user cares about

In priority order:

1. Click Generate in the N-panel with a floor plan image and Qwen
   selected → 3D model appears.
2. It looks roughly like the input plan.
3. Walls + floor + labels at minimum. Doors/windows are nice-to-have
   (boolean solver is the weakest link).
4. Existing Model dropdown still works for CubiCasa / YOLO / Premium
   Vision — don't break them when adding Qwen.

Out of scope for this session:
- Claude refiner integration
- Furniture auto-placement
- Natural language modifications
- Multi-story (Stories counter in UI is there but shouldn't wire to
  Qwen path yet — just keep 1-story behavior)

## Session push policy

- This addon repo: push to whatever feature branch the user
  specifies. Never push to `master` without explicit authorization.
- archbuildai repo (reference geometry): push branch
  `claude/blender-floorplan-integration-GW9Yp`. Do not merge to main.

## Files to read first (in the addon repo)

The current addon is on the user's Mac at
`~/Desktop/archbuildai/floorplan3d/blender_addon/` on branch `master`
(commit `a096cdb`), not pushed. To see what's actually running:

```bash
cd ~/Desktop/archbuildai
cat floorplan3d/blender_addon/panels.py
cat floorplan3d/blender_addon/operators.py
cat floorplan3d/blender_addon/__init__.py
ls floorplan3d/blender_addon/api/
cat floorplan3d/blender_addon/api/*.py
```

That's what needs the Qwen wiring. The archbuildai branch version is
a different UI — use it as a reference for the geometry layer and
the `LocalModelClient` subprocess pattern, not for the panel code.
