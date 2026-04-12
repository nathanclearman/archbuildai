# Datasets

The floor plan parser is a fine-tune of **Qwen2.5-VL** trained to emit
structured floor plan JSON directly from a raster image, following the
FloorplanVLM recipe (arXiv 2602.06507).

## Primary: CubiCasa5k

5,000 annotated floor plan images with walls, doors, windows, and room labels.

```bash
git clone https://github.com/CubiCasa/CubiCasa5k.git cubicasa5k
```

Kalervo, A., Ylioinas, J., Häikiö, M., Karhu, A., & Kannala, J. (2019).
*CubiCasa5K: A Dataset and an Improved Multi-task Model for Floorplan Image Analysis.*

## Supplementary: ResPlan

17,000 vector-graph residential floor plans with connectivity annotations.
Used to expand topological diversity beyond CubiCasa5k. See arXiv 2508.14006.

## Supplementary: CFP Dataset

~1,062 diverse floor plans (villas, malls, complex buildings). Used to stress
non-Manhattan geometry.

## Annotation Target Format

Each training sample converts to a single `(image, json)` pair where the JSON
matches the schema the Blender add-on consumes:

```json
{
  "scale": { "pixels_per_meter": 50 },
  "walls":   [{ "start": [x,y], "end": [x,y], "thickness": t }, ...],
  "doors":   [{ "position": [x,y], "width": w, "type": "hinged", "wall_index": i }, ...],
  "windows": [{ "position": [x,y], "width": w, "wall_index": i }, ...],
  "rooms":   [{ "label": "...", "polygon": [[x,y], ...], "area": a }, ...]
}
```

The training and inference scripts (`model/train.py`, `model/inference.py`)
will be added alongside the Qwen2.5-VL fine-tuning pipeline.
