"""
Fine-tuning script for the floor plan parsing model.

Uses YOLOv8 as base architecture, fine-tuned on the CubiCasa5k dataset
for detecting walls, doors, windows, and room boundaries in floor plan images.

Usage:
    python train.py                          # Train with default config
    python train.py --config config.yaml     # Train with custom config
    python train.py --resume weights/last.pt # Resume training
"""

import argparse
import sys
from pathlib import Path

import yaml


def load_config(config_path):
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_cubicasa_dataset(data_dir, config):
    """Convert CubiCasa5k dataset to YOLO format.

    CubiCasa5k provides SVG annotations that need to be converted to
    YOLO bounding box format for training.
    """
    from dataset_converter import CubiCasaConverter

    converter = CubiCasaConverter(
        source_dir=Path(data_dir),
        output_dir=Path(data_dir) / "yolo_format",
        classes=config["model"]["classes"],
        val_split=config["training"]["val_split"],
        test_split=config["training"]["test_split"],
    )
    converter.convert()
    return Path(data_dir) / "yolo_format" / "dataset.yaml"


def train_detection_model(config, dataset_yaml, resume_weights=None):
    """Train the object detection model (doors, windows, fixtures)."""
    from ultralytics import YOLO

    if resume_weights:
        model = YOLO(resume_weights)
    else:
        model = YOLO(config["model"]["base_weights"])

    train_cfg = config["training"]
    aug_cfg = train_cfg.get("augmentation", {})

    results = model.train(
        data=str(dataset_yaml),
        epochs=train_cfg["epochs"],
        batch=train_cfg["batch_size"],
        imgsz=config["model"]["input_size"],
        lr0=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        warmup_epochs=train_cfg["warmup_epochs"],
        patience=train_cfg["patience"],
        device=train_cfg["device"],
        workers=train_cfg["workers"],
        # Augmentation
        hsv_h=aug_cfg.get("hsv_h", 0.015),
        hsv_s=aug_cfg.get("hsv_s", 0.7),
        hsv_v=aug_cfg.get("hsv_v", 0.4),
        degrees=aug_cfg.get("degrees", 0),
        translate=aug_cfg.get("translate", 0.1),
        scale=aug_cfg.get("scale", 0.5),
        flipud=aug_cfg.get("flipud", 0.0),
        fliplr=aug_cfg.get("fliplr", 0.5),
        mosaic=aug_cfg.get("mosaic", 0.0),
        # Output
        project="runs/floorplan",
        name="detection",
        save=True,
        plots=True,
    )

    return results


def train_segmentation_model(config, dataset_yaml, resume_weights=None):
    """Train the segmentation model (walls, room boundaries)."""
    from ultralytics import YOLO

    seg_config = config["model"].get("segmentation", {})

    if resume_weights:
        model = YOLO(resume_weights)
    else:
        model = YOLO(seg_config.get("base_weights", "yolov8m-seg.pt"))

    train_cfg = config["training"]

    results = model.train(
        data=str(dataset_yaml),
        epochs=train_cfg["epochs"],
        batch=train_cfg["batch_size"],
        imgsz=config["model"]["input_size"],
        lr0=train_cfg["learning_rate"],
        patience=train_cfg["patience"],
        device=train_cfg["device"],
        workers=train_cfg["workers"],
        project="runs/floorplan",
        name="segmentation",
        save=True,
        plots=True,
    )

    return results


def export_weights(config):
    """Copy best weights to the standard location for the add-on."""
    import shutil

    weights_dir = Path(__file__).parent / "weights"
    weights_dir.mkdir(exist_ok=True)

    # Copy detection model
    det_best = Path("runs/floorplan/detection/weights/best.pt")
    if det_best.exists():
        shutil.copy2(det_best, weights_dir / "detection.pt")
        print(f"Detection weights saved to {weights_dir / 'detection.pt'}")

    # Copy segmentation model
    seg_best = Path("runs/floorplan/segmentation/weights/best.pt")
    if seg_best.exists():
        shutil.copy2(seg_best, weights_dir / "segmentation.pt")
        print(f"Segmentation weights saved to {weights_dir / 'segmentation.pt'}")

    # Create combined weights marker
    (weights_dir / "floorplan_parser.pt").write_text(
        "combined\ndetection.pt\nsegmentation.pt\n"
    )
    print(f"Model weights ready at {weights_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train FloorPlan3D parsing model")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--resume", default=None, help="Resume from weights file")
    parser.add_argument(
        "--skip-detection", action="store_true",
        help="Skip detection model training",
    )
    parser.add_argument(
        "--skip-segmentation", action="store_true",
        help="Skip segmentation model training",
    )
    parser.add_argument(
        "--export-only", action="store_true",
        help="Only export existing weights (skip training)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.export_only:
        export_weights(config)
        return

    # Prepare dataset
    data_dir = config["training"]["data_dir"]
    if not Path(data_dir).exists():
        print(f"Dataset not found at {data_dir}")
        print("Download CubiCasa5k first. See data/README.md for instructions.")
        sys.exit(1)

    print("Preparing dataset...")
    dataset_yaml = prepare_cubicasa_dataset(data_dir, config)

    # Train detection model
    if not args.skip_detection:
        print("\n=== Training Detection Model ===")
        train_detection_model(config, dataset_yaml, args.resume)

    # Train segmentation model
    if not args.skip_segmentation:
        print("\n=== Training Segmentation Model ===")
        train_segmentation_model(config, dataset_yaml, args.resume)

    # Export weights
    print("\n=== Exporting Weights ===")
    export_weights(config)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
