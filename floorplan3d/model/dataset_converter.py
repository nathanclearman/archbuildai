"""
Dataset converter for CubiCasa5k to YOLO format.

CubiCasa5k provides floor plan images with SVG-based annotations.
This converter transforms them into YOLO-compatible bounding box
and segmentation annotations.
"""

import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


class CubiCasaConverter:
    """Convert CubiCasa5k dataset to YOLO training format."""

    # Mapping from CubiCasa SVG class names to our class IDs
    CLASS_MAP = {
        "Wall": 0,
        "Door": 1,
        "Window": 2,
        "Room": 3,
    }

    def __init__(self, source_dir, output_dir, classes, val_split=0.15, test_split=0.05):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.classes = classes
        self.val_split = val_split
        self.test_split = test_split

    def convert(self):
        """Run the full conversion pipeline."""
        print(f"Converting CubiCasa5k from {self.source_dir} to {self.output_dir}")

        # Create output directory structure
        for split in ("train", "val", "test"):
            (self.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        # Find all floor plan samples
        samples = self._find_samples()
        print(f"Found {len(samples)} samples")

        # Shuffle and split
        random.seed(42)
        random.shuffle(samples)

        n_test = int(len(samples) * self.test_split)
        n_val = int(len(samples) * self.val_split)

        test_samples = samples[:n_test]
        val_samples = samples[n_test:n_test + n_val]
        train_samples = samples[n_test + n_val:]

        print(f"Split: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test")

        # Convert each split
        for split, split_samples in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
            for sample in split_samples:
                self._convert_sample(sample, split)

        # Write dataset.yaml
        self._write_dataset_yaml()

        print("Conversion complete!")

    def _find_samples(self):
        """Find all valid floor plan samples in the CubiCasa5k directory."""
        samples = []

        # CubiCasa5k structure: category/subcategory/image.png + model.svg
        for category_dir in sorted(self.source_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            for sample_dir in sorted(category_dir.iterdir()):
                if not sample_dir.is_dir():
                    continue
                image_path = sample_dir / "F1_scaled.png"
                svg_path = sample_dir / "model.svg"
                if image_path.exists() and svg_path.exists():
                    samples.append({
                        "image": image_path,
                        "annotation": svg_path,
                        "name": f"{category_dir.name}_{sample_dir.name}",
                    })

        return samples

    def _convert_sample(self, sample, split):
        """Convert a single sample to YOLO format."""
        from PIL import Image

        image_path = sample["image"]
        svg_path = sample["annotation"]
        name = sample["name"]

        # Copy image
        dest_image = self.output_dir / "images" / split / f"{name}.png"
        shutil.copy2(image_path, dest_image)

        # Parse SVG annotations and convert to YOLO labels
        img = Image.open(image_path)
        img_w, img_h = img.size

        labels = self._parse_svg_annotations(svg_path, img_w, img_h)

        # Write label file
        label_path = self.output_dir / "labels" / split / f"{name}.txt"
        with open(label_path, 'w') as f:
            for label in labels:
                f.write(label + "\n")

    def _parse_svg_annotations(self, svg_path, img_w, img_h):
        """Parse CubiCasa5k SVG annotations to YOLO format.

        Returns a list of YOLO-format label strings:
        class_id center_x center_y width height (all normalized 0-1)
        """
        labels = []

        try:
            tree = ET.parse(svg_path)
            root = tree.getroot()

            # SVG namespace
            ns = {"svg": "http://www.w3.org/2000/svg"}

            # Find all annotated elements
            for group in root.findall(".//svg:g", ns):
                class_name = group.get("id", "")

                # Map CubiCasa class names to our classes
                cls_id = None
                for key, cid in self.CLASS_MAP.items():
                    if key.lower() in class_name.lower():
                        cls_id = cid
                        break

                if cls_id is None:
                    continue

                # Extract bounding box from child elements
                for elem in group:
                    bbox = self._extract_bbox(elem, ns)
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        # Convert to YOLO format (normalized center + dimensions)
                        cx = (x1 + x2) / 2.0 / img_w
                        cy = (y1 + y2) / 2.0 / img_h
                        w = (x2 - x1) / img_w
                        h = (y2 - y1) / img_h

                        # Clamp values
                        cx = max(0, min(1, cx))
                        cy = max(0, min(1, cy))
                        w = max(0, min(1, w))
                        h = max(0, min(1, h))

                        if w > 0.001 and h > 0.001:
                            labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        except ET.ParseError as e:
            print(f"Warning: Failed to parse {svg_path}: {e}")

        return labels

    def _extract_bbox(self, elem, ns):
        """Extract bounding box from an SVG element."""
        tag = elem.tag.replace(f"{{{ns.get('svg', '')}}}", "")

        if tag == "rect":
            x = float(elem.get("x", 0))
            y = float(elem.get("y", 0))
            w = float(elem.get("width", 0))
            h = float(elem.get("height", 0))
            return (x, y, x + w, y + h)

        elif tag == "line":
            x1 = float(elem.get("x1", 0))
            y1 = float(elem.get("y1", 0))
            x2 = float(elem.get("x2", 0))
            y2 = float(elem.get("y2", 0))
            return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

        elif tag == "polygon" or tag == "polyline":
            points_str = elem.get("points", "")
            if not points_str:
                return None
            points = []
            for pair in points_str.strip().split():
                parts = pair.split(",")
                if len(parts) == 2:
                    points.append((float(parts[0]), float(parts[1])))
            if points:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                return (min(xs), min(ys), max(xs), max(ys))

        return None

    def _write_dataset_yaml(self):
        """Write the YOLO dataset.yaml file."""
        dataset_config = {
            "path": str(self.output_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(self.classes),
            "names": self.classes,
        }

        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)

        print(f"Dataset config written to {yaml_path}")
