# Dataset Download Instructions

## CubiCasa5k (Primary Dataset)

CubiCasa5k contains 5,000 annotated floor plan images with walls, doors, windows, and room labels.

### Download

```bash
# Clone the CubiCasa5k repository
git clone https://github.com/CubiCasa/CubiCasa5k.git cubicasa5k

# The dataset structure:
# cubicasa5k/
#   colorful/       # Category: colorful floor plans
#   high_quality/   # Category: high quality scans
#   ...
#   Each sample contains:
#     F1_scaled.png  # Floor plan image
#     model.svg      # SVG annotations
```

### Paper

Kalervo, A., Ylioinas, J., Häikiö, M., Karhu, A., & Kannala, J. (2019).
CubiCasa5K: A Dataset and an Improved Multi-task Model for Floorplan Image Analysis.

## ROBIN Dataset (Supplementary)

ROBIN provides additional annotated architectural floor plans. Use to supplement
CubiCasa5k if more training data is needed.

## Converting to Training Format

After downloading CubiCasa5k, run the dataset converter:

```bash
cd floorplan3d/model
python dataset_converter.py
```

Or use the training script which handles conversion automatically:

```bash
python train.py --config config.yaml
```

## Training

```bash
# Full training pipeline
python train.py

# Detection model only
python train.py --skip-segmentation

# Segmentation model only
python train.py --skip-detection

# Resume from checkpoint
python train.py --resume weights/last.pt

# Export weights only (after training)
python train.py --export-only
```

## Inference

```bash
# Run inference on a floor plan image
python inference.py --image path/to/floorplan.png --output json

# Save to file
python inference.py --image path/to/floorplan.png --output file --out-path result.json
```
