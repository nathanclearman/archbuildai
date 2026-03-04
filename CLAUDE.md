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

## File Structure (Proposed)

```
floorplan3d/
├── blender_addon/
│   ├── __init__.py              # Add-on registration, bl_info
│   ├── operators.py             # Generate, adjust, export operators
│   ├── panels.py                # UI panel definitions
│   ├── geometry.py              # bpy mesh generation from JSON
│   ├── materials.py             # Default material assignment
│   └── api/
│       ├── local_model.py       # Local CV model inference wrapper
│       └── claude_client.py     # Optional Claude API client
├── model/
│   ├── train.py                 # Fine-tuning script
│   ├── inference.py             # Inference entry point
│   ├── config.yaml              # Model and training config
│   └── data/
│       └── README.md            # Dataset download instructions
├── tests/
│   ├── test_geometry.py         # Geometry generation unit tests
│   ├── test_parser.py           # Model output validation tests
│   └── sample_plans/            # Test floor plan images
├── CLAUDE.md                    # This file
├── requirements.txt
└── README.md
```

---

## Constraints and Gotchas

- **Precision is approximate.** The AI parser will not produce millimeter-accurate output. Always provide user correction tools. Include scale reference support.
- **Complex plans will fail.** Curved walls, split levels, and unusual architectural elements should be flagged as unsupported in MVP rather than producing bad geometry.
- **Hand-drawn plans are hard.** The local model will struggle with sketchy, inconsistent line work. Consider this a Phase 2+ problem or route these to Claude for interpretation.
- **Blender API threading.** `bpy` is not thread-safe. All Blender operations must happen on the main thread. Use background threads only for model inference and API calls, then pass results back via a timer or modal operator.
- **Add-on packaging.** The local model weights will need to be distributed separately or downloaded on first run — they'll be too large to bundle in a `.zip` add-on.