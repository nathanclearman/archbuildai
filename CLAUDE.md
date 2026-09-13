# FloorPlan3D — AI-Powered Floor Plan to Blender 3D Model Pipeline

## Project Overview

A Blender add-on that takes 2D floor plan images as input and generates editable 3D architectural models using a hybrid AI pipeline: a locally-run specialized computer vision model for precise floor plan parsing, with optional Claude API integration for higher-level reasoning tasks.

## Goal

Build a Blender-integrated tool that allows users to import a floor plan image and receive a 3D model with extruded walls, door/window openings, floor planes, room labels, and (eventually) auto-placed furniture — all editable within Blender.

## Target Users

- Real estate agents and property marketers (virtual tours, staging)
- Architects and interior designers (rapid early-stage visualization)
- Contractors and renovators (spatial planning from existing floor plans)
- Game/film level and set designers (quick base geometry from sketches)

---

## Architecture

### Two-Model Hybrid Pipeline

```
Floor Plan Image
       │
       ▼
┌─────────────────────────┐
│  LOCAL CV MODEL (core)  │  ← Fast, precise, offline, free
│  Floor plan parsing     │
│  Wall/door/window       │
│  detection & geometry   │
└──────────┬──────────────┘
           │ Structured JSON output
           ▼
┌─────────────────────────┐
│  BLENDER ADD-ON (bpy)   │  ← Geometry generation
│  Wall extrusion         │
│  Door/window cutouts    │
│  Floor/ceiling planes   │
│  Room labeling          │
└──────────┬──────────────┘
           │
           ▼
     3D Blender Model (.blend)

           │ (optional)
           ▼
┌─────────────────────────┐
│  CLAUDE API (optional)  │  ← Smart layer, reasoning
│  Furniture suggestions  │
│  Layout critique        │
│  Natural language edits │
│  Ambiguity resolution   │
└─────────────────────────┘
```

### Why Hybrid

- **Local model** handles the repetitive, precision-critical perception task (parsing walls, doors, windows, dimensions from images). It runs offline, costs nothing per inference, returns results in milliseconds, and will outperform a general-purpose LLM on accuracy because it's trained specifically on floor plan data.
- **Claude API** handles higher-level reasoning: furniture placement suggestions, natural language modification requests ("make the kitchen bigger"), design feedback, interpreting ambiguous or hand-drawn plans where the local model flags low confidence. This is optional and the tool should work fully offline without it.

---

## Technical Stack

### Local Model (Floor Plan Parser)

- **Approach:** Fine-tune an existing vision/segmentation model, do NOT train from scratch.
- **Base model options:**
  - YOLOv8/v9 for object detection (doors, windows, fixtures)
  - SAM (Segment Anything Model) for wall/room segmentation
  - A lightweight vision encoder fine-tuned for structured output
- **Training data:**
  - CubiCasa5k dataset (5,000 annotated floor plans with walls, doors, windows, rooms)
  - ROBIN dataset (annotated architectural floor plans)
  - Supplement with synthetic floor plans if needed
- **Output format:** Structured JSON containing:
  ```json
  {
    "scale": { "pixels_per_meter": 50 },
    "walls": [
      { "start": [0, 0], "end": [4.2, 0], "thickness": 0.15 }
    ],
    "doors": [
      { "position": [2.1, 0], "width": 0.9, "type": "hinged", "wall_index": 0 }
    ],
    "windows": [
      { "position": [1.0, 3.5], "width": 1.2, "wall_index": 2 }
    ],
    "rooms": [
      { "label": "bedroom", "polygon": [[0,0],[4.2,0],[4.2,3.5],[0,3.5]], "area": 14.7 }
    ]
  }
  ```
- **Framework:** PyTorch. Use MLX for inference on Apple Silicon if targeting Mac deployment.

### Available Hardware

- **Apple M4 Max, 128GB unified RAM** — Primary development and inference machine. Can run models up to ~30B parameters comfortably with MLX. Unified memory means no VRAM bottleneck. Best option for running the local model in production alongside Blender.
- **NVIDIA RTX 5070 Ti (16GB VRAM)** — Faster raw training throughput via CUDA. Use for fine-tuning. 16GB VRAM limits model size during training, but the floor plan parser should be well under that.
- **Recommendation:** Train/fine-tune on the 5070 Ti, deploy/infer on the M4 Max.

### Blender Add-on

- **Language:** Python (Blender's `bpy` API)
- **Blender version target:** 4.x+
- **Key `bpy` operations:**
  - `bmesh` for wall mesh creation (extrude rectangles along wall paths)
  - Boolean modifiers or manual geometry for door/window openings
  - Simple planes for floors and ceilings
  - Text objects or custom properties for room labels
  - Material assignment (basic defaults, expandable later)
- **UI:** Blender side panel (N-panel) with:
  - Image file picker for floor plan input
  - Scale/unit configuration
  - Generate button
  - Post-generation adjustment controls (wall height, default materials)
  - Optional: Claude API key field and natural language input box
- **Threading:** API calls (both local model and Claude) must run in background threads or use Blender's modal operator system to prevent UI freezing.

### Claude API Integration (Optional Layer)

- **Model:** claude-sonnet-4-5-20250929 for cost efficiency, claude-opus-4-6 for complex reasoning
- **Use cases:**
  - Furniture auto-placement based on room type and size
  - Natural language model modifications ("add a bathroom next to the master bedroom")
  - Layout quality feedback and suggestions
  - Interpreting ambiguous/low-confidence results from the local parser
  - Style-based material and lighting suggestions
- **Implementation:** Standard REST calls via `requests` library from within the Blender add-on

---

## Development Phases

### Phase 1 — MVP (Core Pipeline)

1. Set up Blender add-on scaffold (panel UI, file picker, generate button)
2. Implement Blender geometry generation from hardcoded JSON (prove the bpy pipeline works)
3. Fine-tune local CV model on CubiCasa5k dataset for wall/door/window detection
4. Connect local model inference to the add-on
5. End-to-end: floor plan image → local model → JSON → 3D Blender model
6. Basic error handling and confidence reporting

**MVP output:** Clean floor plans produce a 3D model with walls (default 2.7m height), door openings, window openings, floor planes, and room labels. User can adjust dimensions after generation.

### Phase 2 — Refinement

- Scale detection (auto-read scale bars and dimension annotations)
- Support for irregular/angled walls
- Improved door/window type recognition (sliding, hinged, casement, etc.)
- User correction tools (click to adjust wall positions, drag to resize rooms)
- Export options (.blend, .fbx, .obj, .glTF)

### Phase 3 — Smart Layer (Claude Integration)

- Claude-powered furniture auto-placement
- Natural language modification interface
- Design style presets and material suggestions
- Ambiguity resolution for complex or hand-drawn plans
- Layout optimization suggestions

### Phase 4 — Polish

- Multi-story support
- Staircase and split-level handling
- Plumbing/electrical fixture placement
- Batch processing (multiple floor plans)
- Marketplace-ready packaging

---

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Local model vs cloud-only | Local model primary, cloud optional | Precision, speed, cost, offline capability |
| Training approach | Fine-tune, not from scratch | CubiCasa5k and ROBIN provide sufficient labeled data |
| Blender integration | Native add-on via bpy | Direct mesh control, no import/export friction |
| Intermediate data format | JSON | Human-readable, debuggable, easy to validate and manually correct |
| Threading | Background threads / modal operators | Prevent Blender UI freezing during inference |

---

## File Structure (Actual)

```
floorplan3d/
├── blender_addon/               # THE Blender add-on (v1.1.0) — single source of truth
│   ├── __init__.py              # bl_info, registration, scene props, user-site shim
│   ├── operators.py             # Generate / sample / adjust / export / Premium AI operators
│   ├── panels.py                # N-panel UI (Model dropdown, stories, stairs, Premium AI)
│   ├── geometry.py              # bpy mesh generation: walls, openings, floors, ceilings,
│   │                            #   stairs, furniture, exterior/roofs, sanitize + dedupe
│   ├── materials.py             # Material assignment
│   ├── preferences.py           # Add-on prefs: env status, pip installer, base-model download, overrides
│   ├── blender_manifest.toml    # Blender 4.2+ extension manifest (package.py builds an extension zip)
│   ├── correction*.py           # Interactive 2D "Review Before 3D" correction mode
│   ├── vlm/                     # (build output, git-ignored) VLM runtime copied from model/ by package.py
│   ├── weights/                 # YOLO detection.pt / segmentation.pt (git-ignored, ~100 MB)
│   └── api/
│       ├── sample_plans.py      # built-in sample plans for "Generate Sample"
│       ├── hybrid.py            # YOLO-geometry + VLM-label glue (pure python)
│       ├── cleanup.py           # CV post-pass: clip walls to footprint, drop slivers, close outline
│       ├── autoscale.py         # pixels-per-metre from OCR'd dimension strings (pure python)
│       ├── setup_env.py         # first-run installer helpers (pip into Blender's user site, HF download)
│       ├── yolo_model.py        # "YOLO" backend (in-process ultralytics)
│       ├── local_model.py       # Qwen2.5-VL daemon client (repo model/, resolves ML venv)
│       ├── qwen_client.py       # "Qwen2.5-VL (trained)" backend adapter over local_model
│       ├── claude_vision_client.py  # "Premium Vision" parser (Claude API)
│       ├── claude_client.py     # Premium furniture / critique / modify / exterior
│       └── local_llm_client.py  # Same features via Ollama / MLX server
├── model/                       # Qwen2.5-VL pipeline: train.py, inference.py, evaluate.py,
│   ├── ...                      #   synthesize.py, dataset.py, schema.py, cv_walls.py
│   └── weights/                 # LoRA adapter + train_config.json (git-ignored)
├── tests/                       # pytest suite (+ test_geometry_blender.py for real Blender)
├── package.py                   # Zip the add-on (--with-weights bundles the .pt files)
└── requirements.txt
papercole/                       # Research paper — local-only, git-ignored (see layout rule)
```

## Blender Add-on: How It Is Installed and Run

- Blender 5.0 / 5.1 / 5.2 on this Mac load the add-on through a symlink:
  `~/Library/Application Support/Blender/<ver>/scripts/addons/floorplan3d -> floorplan3d/blender_addon`.
  Editing the repo edits the installed add-on; reload with F8 / disable+enable. Do not copy
  the add-on into the addons folder again — that recreates the divergence fixed on 2026-09-12
  (backups of the old divergent copies: `~/Library/Application Support/Blender/floorplan3d_backups/`).
- Model dropdown backends (default **Hybrid**): HYBRID = YOLO geometry (walls, openings, room
  polygons; in-process ultralytics in Blender's Python) + room names read off the image by the local
  Qwen2.5-VL base model (grounded OCR over the whole plan, then a per-room crop pass for anything
  unlabeled; `api/hybrid.py` snaps names to polygons). Both CV paths run `api/cleanup.py` first so the
  outside shape matches the plan (walls outside the room-polygon footprint are trimmed/dropped, sliver
  rooms removed, uncovered outline edges get an exterior wall). YOLO = geometry only, heuristic labels.
  QWEN = the fine-tuned adapter end to end (reads labels well, draws a memorized grid layout on real
  plans — see memory `project_qwen_failure_diagnosis`; 7-10 min per plan). CLAUDE_VISION = Premium.
  The CubiCasa backend was removed 2026-09-12 (unusable on MLS-style plans).
- The VLM runs as `floorplan3d/model/inference.py --serve` (ops: extract, ocr_labels, ocr_dimensions,
  ocr_crop; `--backend auto|torch|mlx`). On Apple Silicon the OCR ops run on **MLX** (mlx-vlm,
  `mlx-community/Qwen2.5-VL-7B-Instruct-8bit`, ~8 GB, several× the torch/MPS token rate); the fine-tuned
  adapter (extract op) always runs on torch and is loaded lazily. FP3D_VLM_BACKEND overrides. Hybrid also
  reads the plan's dimension strings and rescales the model (`api/autoscale.py`; Scale Factor becomes the
  starting guess; toggle `fp3d_auto_scale`). It runs in a
  Python with torch+transformers+peft; `api/local_model.py` finds it automatically (FP3D_PYTHON env →
  non-Blender sys.executable → python3/python → repo `.venv`, `floorplan_env`, `model/venv`). One daemon
  per Blender session, stopped on add-on unregister. Decoding budget 6144 tokens (FP3D_MAX_NEW_TOKENS);
  truncated JSON is salvaged element-by-element.
- Test fixtures under `tests/fixtures/` are thumbnails (≤200 px); the CV backends return almost
  nothing on them by design. Use `model/data/mls_qualitative/*.png|jpg` for real predictions.
- Headless checks: `blender --factory-startup --background --python floorplan3d/tests/test_geometry_blender.py`.
- License: the project's own code is MIT (LICENSE, manifest). Ultralytics (AGPL) and the CubiCasa-trained
  YOLO weights (non-commercial) are the only encumbered pieces; see floorplan3d/README.md.
- Shipping: `python floorplan3d/package.py --with-weights` builds an extension zip (manifest at root,
  `vlm/` runtime bundled from `model/`, no adapter). Validate with `blender --command extension validate <zip>`.
  A shipped install has no repo: `api/local_model.py` resolves the runtime to the bundled `vlm/`, and the
  add-on Preferences install the CV and VLM packages into Blender's own Python (user site) and download the
  base model; Blender's interpreter is then the last-resort VLM python. Never add home-directory paths.

---

## Constraints and Gotchas

- **Precision is approximate.** The AI parser will not produce millimeter-accurate output. Always provide user correction tools. Include scale reference support.
- **Complex plans will fail.** Curved walls, split levels, and unusual architectural elements should be flagged as unsupported in MVP rather than producing bad geometry.
- **Hand-drawn plans are hard.** The local model will struggle with sketchy, inconsistent line work. Consider this a Phase 2+ problem or route these to Claude for interpretation.
- **Blender API threading.** `bpy` is not thread-safe. All Blender operations must happen on the main thread. Use background threads only for model inference and API calls, then pass results back via a timer or modal operator.
- **Add-on packaging.** The local model weights will need to be distributed separately or downloaded on first run — they'll be too large to bundle in a `.zip` add-on.

---

## Repo Layout Rule: Add-on vs Paper

- `floorplan3d/` is the product: Blender add-on (`blender_addon/`), ML pipeline (`model/`), tests. Tracked in git.
- `papercole/` is the research paper: drafts, evidence, figures, and the analysis scripts in `papercole/experiments/`. Git-ignored, local-only, never committed.
- New experiment / statistics scripts go in `papercole/experiments/` and import the pipeline via `MODEL_DIR` (see its README). Do not add paper-only scripts to `floorplan3d/model/`.
- `floorplan3d/model/data/` holds datasets and eval sets only; debug outputs there are git-ignored.
