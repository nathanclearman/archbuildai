"""
Inference entry point for the floor plan parsing model.

Takes a floor plan image and outputs structured JSON with detected
walls, doors, windows, and rooms.

Usage:
    python inference.py --image plan.png --output json
    python inference.py --image plan.png --output json --weights weights/
    python inference.py --image plan.png --output file --out-path result.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml


def load_config():
    """Load inference configuration."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class FloorPlanParser:
    """Combined detection + segmentation floor plan parser."""

    def __init__(self, weights_dir=None, config=None):
        self.config = config or load_config()
        self.weights_dir = Path(weights_dir) if weights_dir else Path(__file__).parent / "weights"

        self.detection_model = None
        self.segmentation_model = None

        self._load_models()

    def _load_models(self):
        """Load detection and segmentation models."""
        from ultralytics import YOLO

        det_path = self.weights_dir / "detection.pt"
        seg_path = self.weights_dir / "segmentation.pt"

        if det_path.exists():
            self.detection_model = YOLO(str(det_path))
        else:
            raise FileNotFoundError(f"Detection weights not found: {det_path}")

        if seg_path.exists():
            self.segmentation_model = YOLO(str(seg_path))
        else:
            print(f"Warning: Segmentation weights not found: {seg_path}", file=sys.stderr)

    def predict(self, image_path):
        """Run full pipeline on a floor plan image.

        Args:
            image_path: Path to the floor plan image.

        Returns:
            dict: Structured floor plan data.
        """
        image_path = str(image_path)
        inf_config = self.config.get("inference", {})
        out_config = self.config.get("output", {})

        conf_threshold = inf_config.get("confidence_threshold", 0.5)
        iou_threshold = inf_config.get("iou_threshold", 0.45)
        device = inf_config.get("device", "auto")

        if device == "auto":
            device = self._detect_device()

        # Run detection
        det_results = self.detection_model.predict(
            image_path,
            conf=conf_threshold,
            iou=iou_threshold,
            device=device,
            verbose=False,
        )

        # Run segmentation if available
        seg_results = None
        if self.segmentation_model:
            seg_results = self.segmentation_model.predict(
                image_path,
                conf=conf_threshold,
                iou=iou_threshold,
                device=device,
                verbose=False,
            )

        # Convert to structured output
        return self._build_output(det_results, seg_results, image_path, out_config, inf_config)

    def _detect_device(self):
        """Auto-detect the best available device."""
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _build_output(self, det_results, seg_results, image_path, out_config, inf_config):
        """Convert model predictions to the standard JSON format."""
        from PIL import Image

        img = Image.open(image_path)
        img_w, img_h = img.size

        # Estimate scale (pixels per meter) — rough default, will be refined with scale detection
        pixels_per_meter = 50  # Default assumption

        classes = self.config["model"]["classes"]
        snap_threshold = inf_config.get("snap_threshold", 0.1)
        min_wall_length = inf_config.get("min_wall_length", 0.3)

        walls = []
        doors = []
        windows = []
        rooms = []

        # Process detection results
        if det_results and len(det_results) > 0:
            result = det_results[0]
            boxes = result.boxes

            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Convert pixel coords to meters
                x1_m = x1 / pixels_per_meter
                y1_m = y1 / pixels_per_meter
                x2_m = x2 / pixels_per_meter
                y2_m = y2 / pixels_per_meter

                cls_name = classes[cls_id] if cls_id < len(classes) else "unknown"

                if cls_name == "wall":
                    # Determine wall orientation and create wall segment
                    w = x2_m - x1_m
                    h = y2_m - y1_m
                    thickness = out_config.get("default_wall_thickness", 0.15)

                    if w > h:
                        # Horizontal wall
                        mid_y = (y1_m + y2_m) / 2
                        walls.append({
                            "start": [round(x1_m, 3), round(mid_y, 3)],
                            "end": [round(x2_m, 3), round(mid_y, 3)],
                            "thickness": thickness,
                            "confidence": round(conf, 3),
                        })
                    else:
                        # Vertical wall
                        mid_x = (x1_m + x2_m) / 2
                        walls.append({
                            "start": [round(mid_x, 3), round(y1_m, 3)],
                            "end": [round(mid_x, 3), round(y2_m, 3)],
                            "thickness": thickness,
                            "confidence": round(conf, 3),
                        })

                elif cls_name == "door":
                    cx = (x1_m + x2_m) / 2
                    cy = (y1_m + y2_m) / 2
                    width = min(x2_m - x1_m, y2_m - y1_m)
                    # Find nearest wall
                    wall_idx = self._find_nearest_wall(cx, cy, walls)
                    doors.append({
                        "position": [round(cx, 3), round(cy, 3)],
                        "width": round(max(width, 0.7), 3),
                        "type": "hinged",
                        "wall_index": wall_idx,
                        "confidence": round(conf, 3),
                    })

                elif cls_name == "window":
                    cx = (x1_m + x2_m) / 2
                    cy = (y1_m + y2_m) / 2
                    width = min(x2_m - x1_m, y2_m - y1_m)
                    wall_idx = self._find_nearest_wall(cx, cy, walls)
                    windows.append({
                        "position": [round(cx, 3), round(cy, 3)],
                        "width": round(max(width, 0.5), 3),
                        "wall_index": wall_idx,
                        "confidence": round(conf, 3),
                    })

        # Process segmentation results for rooms
        if seg_results and len(seg_results) > 0:
            result = seg_results[0]
            if result.masks is not None:
                for mask_idx, mask in enumerate(result.masks):
                    cls_id = int(result.boxes[mask_idx].cls[0])
                    cls_name = classes[cls_id] if cls_id < len(classes) else "unknown"

                    if cls_name == "room":
                        # Convert mask to polygon
                        polygon = self._mask_to_polygon(
                            mask.xy[0] if mask.xy else [],
                            pixels_per_meter,
                        )
                        if len(polygon) >= 3:
                            area = self._polygon_area(polygon)
                            if area >= inf_config.get("min_room_area", 1.0):
                                rooms.append({
                                    "label": f"room_{mask_idx}",
                                    "polygon": polygon,
                                    "area": round(area, 2),
                                })

        # Snap wall endpoints
        walls = self._snap_endpoints(walls, snap_threshold)

        # Filter short walls
        walls = [w for w in walls if self._wall_length(w) >= min_wall_length]

        return {
            "scale": {"pixels_per_meter": pixels_per_meter},
            "walls": walls,
            "doors": doors,
            "windows": windows,
            "rooms": rooms,
        }

    def _find_nearest_wall(self, x, y, walls):
        """Find the index of the nearest wall to a point."""
        if not walls:
            return 0

        min_dist = float("inf")
        nearest_idx = 0

        for i, wall in enumerate(walls):
            sx, sy = wall["start"]
            ex, ey = wall["end"]
            # Distance from point to line segment
            dist = self._point_to_segment_distance(x, y, sx, sy, ex, ey)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        return nearest_idx

    @staticmethod
    def _point_to_segment_distance(px, py, sx, sy, ex, ey):
        """Calculate distance from point to line segment."""
        dx = ex - sx
        dy = ey - sy
        length_sq = dx * dx + dy * dy
        if length_sq < 1e-10:
            return ((px - sx) ** 2 + (py - sy) ** 2) ** 0.5

        t = max(0, min(1, ((px - sx) * dx + (py - sy) * dy) / length_sq))
        proj_x = sx + t * dx
        proj_y = sy + t * dy
        return ((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5

    @staticmethod
    def _snap_endpoints(walls, threshold):
        """Snap wall endpoints that are close together."""
        if len(walls) < 2:
            return walls

        for i in range(len(walls)):
            for j in range(i + 1, len(walls)):
                for end_a in ("start", "end"):
                    for end_b in ("start", "end"):
                        pa = walls[i][end_a]
                        pb = walls[j][end_b]
                        dist = ((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5
                        if dist < threshold and dist > 0:
                            mid = [
                                round((pa[0] + pb[0]) / 2, 3),
                                round((pa[1] + pb[1]) / 2, 3),
                            ]
                            walls[i][end_a] = mid
                            walls[j][end_b] = mid

        return walls

    @staticmethod
    def _wall_length(wall):
        """Calculate wall length."""
        s = wall["start"]
        e = wall["end"]
        return ((e[0] - s[0]) ** 2 + (e[1] - s[1]) ** 2) ** 0.5

    @staticmethod
    def _mask_to_polygon(xy_points, pixels_per_meter):
        """Convert mask contour points to a simplified polygon in meters."""
        if len(xy_points) < 3:
            return []

        # Simplify polygon (reduce point count)
        points = np.array(xy_points)

        # Use Douglas-Peucker simplification
        from shapely.geometry import Polygon as ShapelyPolygon

        poly = ShapelyPolygon(points)
        simplified = poly.simplify(5.0, preserve_topology=True)

        coords = list(simplified.exterior.coords)[:-1]  # Remove closing point
        return [
            [round(x / pixels_per_meter, 3), round(y / pixels_per_meter, 3)]
            for x, y in coords
        ]

    @staticmethod
    def _polygon_area(polygon):
        """Calculate area of a polygon using the shoelace formula."""
        n = len(polygon)
        if n < 3:
            return 0

        area = 0
        for i in range(n):
            j = (i + 1) % n
            area += polygon[i][0] * polygon[j][1]
            area -= polygon[j][0] * polygon[i][1]
        return abs(area) / 2.0


def main():
    parser = argparse.ArgumentParser(description="Run floor plan inference")
    parser.add_argument("--image", required=True, help="Path to floor plan image")
    parser.add_argument(
        "--output", choices=["json", "file"], default="json",
        help="Output mode: 'json' prints to stdout, 'file' saves to file",
    )
    parser.add_argument("--out-path", default="output.json", help="Output file path")
    parser.add_argument("--weights", default=None, help="Path to weights directory")
    parser.add_argument("--config", default=None, help="Path to config file")
    args = parser.parse_args()

    config = None
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    parser_model = FloorPlanParser(weights_dir=args.weights, config=config)
    result = parser_model.predict(args.image)

    if args.output == "json":
        print(json.dumps(result))
    else:
        with open(args.out_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Output saved to {args.out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
