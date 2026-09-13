# ArchbuildAI — floor plan image → editable 3D model in Blender

Drop in a floor plan image (MLS listing, CAD export, scan), press **Generate**, get walls,
door and window openings, floors, ceilings and named rooms as editable Blender objects.
Optional Premium features (furniture, exterior, layout critique, natural-language edits)
use the Claude API with your own key.

## Requirements

- Blender 4.2 or newer (tested on 5.1)
- macOS Apple Silicon or a machine with an NVIDIA GPU for the room-label model
  (the Hybrid backend needs the 7B Qwen2.5-VL base model: ~15 GB download, ~16 GB RAM/VRAM)
- Internet on first run (package install + model download)

## Install

1. Build or download the extension zip (`archbuildai-<version>.zip` on the releases page):
   ```bash
   python floorplan3d/package.py --with-weights
   ```
2. Blender: **Edit > Preferences > Get Extensions > (v) > Install from Disk** and pick the zip.
3. Open **Preferences > Add-ons > ArchbuildAI** and click, in order:
   - **Install geometry packages** (YOLO, OpenCV, Shapely into Blender's Python)
   - **Install VLM packages** (torch, transformers, peft into Blender's Python)
   - **Download base model (15 GB)**

   The panel shows what is still missing. Everything installs into Blender's user
   site-packages and the Hugging Face cache; nothing touches the Blender app bundle.
4. In the 3D Viewport press **N** → **ArchbuildAI** tab.

## Use

1. Pick the floor plan image.
2. Leave **Auto** scale on: the printed dimension strings set the scale. **Scale Factor** is only the
   starting guess (a typical 2500 px-wide MLS plan is 40–60) and is used as-is when Auto is off or no
   dimensions are found.
3. Model: **Hybrid (recommended)** — YOLO reads the geometry, Qwen2.5-VL reads the printed
   room names and dimensions. On Apple Silicon the label and scale passes run on MLX (8-bit
   model, ~8 GB): about 2 minutes per plan end to end, most of it the dimension pass.
4. **Generate 3D Model**. With **Review Before 3D** on, you get a 2D correction pass first
   (move wall endpoints, add/delete doors and windows) before the mesh is built.

Backends:

| Model | What it does | Needs |
|---|---|---|
| Hybrid | YOLO geometry + Qwen room names | geometry packages, VLM packages, base model |
| YOLO only | Geometry with heuristic room names | geometry packages |
| Qwen2.5-VL (trained) | The fine-tuned adapter end to end (slow, layout quality limited) | VLM packages, base model, adapter folder set in preferences |
| Premium Vision | Claude reads the plan | API key |

## Troubleshooting

- *"Setup needed"* in the panel → open the add-on preferences; each line says what is missing.
- Hybrid says *labels from YOLO only* → the VLM interpreter or base model is missing; geometry still builds.
- To use your own Python for the VLM (e.g. a venv with CUDA torch), set **VLM Python** in
  the preferences or the `FP3D_PYTHON` environment variable.
- Large plans: the label pass budget is 6144 tokens (`FP3D_MAX_NEW_TOKENS`).

## Licensing

ArchbuildAI's own code is **MIT** (see `LICENSE`). Third-party pieces it uses:

| Component | License | Effect |
|---|---|---|
| Qwen2.5-VL-7B-Instruct (room labels, dimensions) | Apache-2.0 | free for any use |
| mlx-vlm / MLX (Apple Silicon inference) | MIT | free for any use |
| Ultralytics (YOLO geometry backend) | AGPL-3.0 | a distributed build that imports it must be open source under AGPL terms, or needs an Ultralytics commercial license |
| YOLO detection/segmentation weights | trained on CubiCasa5k (CC BY-NC-SA 4.0) | non-commercial use only |

To ship a build with no copyleft or non-commercial strings attached, replace the
YOLO geometry backend (an Apache/MIT detector, or the weights re-trained on
permissively licensed plans). The label and scale passes are already clean.
