"""
YOLO-based floor plan inference client.

Uses ultralytics YOLO detection + segmentation models trained on
CubiCasa5k data. Better suited for hand-drawn and non-standard floor plans
compared to the CubiCasa hourglass segmentation model.

Both this client and LocalModelClient (CubiCasa) output the same JSON
schema so geometry.py works identically with either backend.
"""

import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger("fp3d")

# Classes the YOLO models were trained on
YOLO_CLASSES = ["wall", "door", "window", "room"]

# Default inference settings (embedded so we don't depend on config.yaml)
DEFAULT_CONF_THRESHOLD = 0.25  # lowered from 0.5 — floor plan elements are subtle
DEFAULT_IOU_THRESHOLD = 0.5    # raised from 0.45 — avoid merging nearby valid detections
DEFAULT_SNAP_THRESHOLD = 0.1   # meters — snap wall endpoints closer than this
DEFAULT_MIN_WALL_LENGTH = 0.3  # meters — skip wall fragments shorter than this
DEFAULT_MIN_ROOM_AREA = 1.0    # sq meters — skip tiny rooms
DEFAULT_WALL_THICKNESS = 0.15  # meters

# Test-Time Augmentation and multi-scale settings
DEFAULT_TTA_ENABLED = True
DEFAULT_MULTISCALE_SIZES = [640, 1024, 1280]  # run at multiple resolutions
DEFAULT_MULTISCALE_MERGE_IOU = 0.5            # IOU to dedup across scales


class _MergedDetResult:
    """Lightweight wrapper to make merged multi-scale detections
    look like a single ultralytics result for _build_output."""

    def __init__(self, xyxy, conf, cls):
        self.boxes = _MergedBoxes(xyxy, conf, cls)


class _MergedBoxes:
    """Iterable box container matching the ultralytics Boxes interface."""

    def __init__(self, xyxy, conf, cls):
        self._xyxy = xyxy
        self._conf = conf
        self._cls = cls

    def __iter__(self):
        for i in range(len(self._xyxy)):
            yield _SingleBox(self._xyxy[i], self._conf[i], self._cls[i])

    def __len__(self):
        return len(self._xyxy)


class _SingleBox:
    """Single detection box with .xyxy, .conf, .cls attributes."""

    def __init__(self, xyxy, conf, cls):
        self.xyxy = xyxy.unsqueeze(0)
        self.conf = conf.unsqueeze(0)
        self.cls = cls.unsqueeze(0)


def _find_yolo_weights_dir():
    """Find the directory containing YOLO weights (detection.pt, segmentation.pt)."""
    addon_dir = Path(__file__).resolve().parent.parent  # api/ -> addon root
    candidates = [
        addon_dir / "weights",                    # blender_addon/weights/
        addon_dir.parent / "model" / "weights",   # model/weights/ (dev layout)
    ]
    for c in candidates:
        if (c / "detection.pt").exists():
            return c
    return candidates[0]


class YOLOModelClient:
    """Client for the YOLO-based floor plan parsing model.

    Interface matches LocalModelClient so the operator can swap between them.
    """

    def __init__(self, weights_dir=None):
        self.weights_dir = Path(weights_dir) if weights_dir else _find_yolo_weights_dir()
        self._detection_model = None
        self._segmentation_model = None
        self._loaded = False

    # ── public API ────────────────────────────────────────────────────

    def predict(self, image_path, pixels_per_meter=None, conf_threshold=None):
        """Run inference on a floor plan image.

        Args:
            image_path: Path to the floor plan image file.
            pixels_per_meter: Scale factor. If None, defaults to 50.
            conf_threshold: Confidence threshold (0-1). Lower = more detections.

        Returns:
            dict: Parsed floor plan data in the standard JSON format.
        """
        image_path = str(image_path)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        if not self._loaded:
            self._load_models()

        self._conf_threshold = conf_threshold
        return self._run_inference(image_path, pixels_per_meter)

    # ── model loading ─────────────────────────────────────────────────

    @staticmethod
    def _detect_device():
        """Auto-detect the best available device."""
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_models(self):
        """Load YOLO detection and segmentation models."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "YOLO model requires the 'ultralytics' package. "
                "Install it in Blender's Python:\n"
                "  <blender>/python/bin/python -m pip install ultralytics"
            )

        det_path = self.weights_dir / "detection.pt"
        seg_path = self.weights_dir / "segmentation.pt"

        if not det_path.exists():
            raise FileNotFoundError(
                f"YOLO detection weights not found at {det_path}. "
                "Ensure detection.pt is in the weights directory."
            )

        log.info(f"[FP3D] Loading YOLO detection model from {det_path}")
        self._detection_model = YOLO(str(det_path))

        if seg_path.exists():
            log.info(f"[FP3D] Loading YOLO segmentation model from {seg_path}")
            self._segmentation_model = YOLO(str(seg_path))
        else:
            log.warning(
                f"[FP3D] YOLO segmentation weights not found at {seg_path}, "
                "room detection will be unavailable"
            )

        self._loaded = True
        log.info("[FP3D] YOLO models loaded successfully")

    # ── inference pipeline ────────────────────────────────────────────

    @staticmethod
    def _preprocess_image(image_path):
        """Enhance floor plan image for better detection."""
        from PIL import Image, ImageEnhance

        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = ImageEnhance.Contrast(img).enhance(1.4)
        img = ImageEnhance.Sharpness(img).enhance(1.5)

        return img

    def _run_inference(self, image_path, pixels_per_meter=None):
        """Run detection + segmentation and convert to standard JSON.

        Uses test-time augmentation and multi-scale inference to improve
        detection quality without retraining.
        """
        from PIL import Image

        device = self._detect_device()

        conf = self._conf_threshold if self._conf_threshold is not None else DEFAULT_CONF_THRESHOLD

        # Preprocess image for better detection
        enhanced_img = self._preprocess_image(image_path)

        # Multi-scale detection: run at multiple resolutions and merge
        all_det_boxes = []
        for imgsz in DEFAULT_MULTISCALE_SIZES:
            results = self._detection_model.predict(
                enhanced_img,
                conf=conf,
                iou=DEFAULT_IOU_THRESHOLD,
                imgsz=imgsz,
                augment=DEFAULT_TTA_ENABLED,  # ultralytics built-in TTA
                device=device,
                verbose=False,
            )
            if results and len(results) > 0 and results[0].boxes is not None:
                all_det_boxes.append(results[0].boxes)

        # Merge multi-scale detection results
        det_results = self._merge_multiscale_detections(all_det_boxes)

        # Run segmentation (single scale + TTA is sufficient for masks)
        seg_results = None
        if self._segmentation_model:
            seg_results = self._segmentation_model.predict(
                enhanced_img,
                conf=conf,
                iou=DEFAULT_IOU_THRESHOLD,
                augment=DEFAULT_TTA_ENABLED,
                device=device,
                verbose=False,
            )

        img_w, img_h = enhanced_img.size

        if pixels_per_meter is None or pixels_per_meter <= 0:
            pixels_per_meter = 50.0

        return self._build_output(
            det_results, seg_results, pixels_per_meter, img_w, img_h,
        )

    def _merge_multiscale_detections(self, all_boxes):
        """Merge detection boxes from multiple scales using NMS.

        Args:
            all_boxes: List of ultralytics Boxes objects, one per scale.

        Returns:
            List-like object compatible with _build_output (single result).
        """
        import torch

        if not all_boxes:
            return None

        # Concatenate all boxes across scales
        all_xyxy = torch.cat([b.xyxy for b in all_boxes], dim=0)
        all_conf = torch.cat([b.conf for b in all_boxes], dim=0)
        all_cls = torch.cat([b.cls for b in all_boxes], dim=0)

        if len(all_xyxy) == 0:
            return None

        # Per-class NMS to deduplicate across scales
        from torchvision.ops import batched_nms
        keep = batched_nms(
            all_xyxy, all_conf, all_cls.int(),
            iou_threshold=DEFAULT_MULTISCALE_MERGE_IOU,
        )

        # Build a lightweight result wrapper
        return [_MergedDetResult(
            all_xyxy[keep], all_conf[keep], all_cls[keep]
        )]

    def _build_output(self, det_results, seg_results, pixels_per_meter,
                      img_w, img_h):
        """Convert YOLO predictions to the standard FloorPlan3D JSON format."""
        walls = []
        doors = []
        windows = []
        rooms = []

        # ── Process detection results (walls, doors, windows) ─────────
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

                cls_name = (
                    YOLO_CLASSES[cls_id] if cls_id < len(YOLO_CLASSES) else "unknown"
                )

                if cls_name == "wall":
                    w = x2_m - x1_m
                    h = y2_m - y1_m

                    if w > h:
                        # Horizontal wall
                        mid_y = (y1_m + y2_m) / 2
                        walls.append({
                            "start": [round(x1_m, 3), round(mid_y, 3)],
                            "end": [round(x2_m, 3), round(mid_y, 3)],
                            "thickness": DEFAULT_WALL_THICKNESS,
                            "confidence": round(conf, 3),
                        })
                    else:
                        # Vertical wall
                        mid_x = (x1_m + x2_m) / 2
                        walls.append({
                            "start": [round(mid_x, 3), round(y1_m, 3)],
                            "end": [round(mid_x, 3), round(y2_m, 3)],
                            "thickness": DEFAULT_WALL_THICKNESS,
                            "confidence": round(conf, 3),
                        })

                elif cls_name == "door":
                    cx = (x1_m + x2_m) / 2
                    cy = (y1_m + y2_m) / 2
                    width = min(x2_m - x1_m, y2_m - y1_m)
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

        # ── Process segmentation results (rooms) ─────────────────────
        if seg_results and len(seg_results) > 0:
            result = seg_results[0]
            if result.masks is not None:
                for mask_idx, mask in enumerate(result.masks):
                    cls_id = int(result.boxes[mask_idx].cls[0])
                    cls_name = (
                        YOLO_CLASSES[cls_id]
                        if cls_id < len(YOLO_CLASSES)
                        else "unknown"
                    )

                    if cls_name == "room":
                        polygon = self._mask_to_polygon(
                            mask.xy[0] if mask.xy else [],
                            pixels_per_meter,
                        )
                        if len(polygon) >= 3:
                            area = self._polygon_area(polygon)
                            if area >= DEFAULT_MIN_ROOM_AREA:
                                rooms.append({
                                    "label": f"room_{mask_idx}",
                                    "polygon": polygon,
                                    "area": round(area, 2),
                                })

        # ── Post-processing: iterative snap + merge ────────────────────
        for _ in range(3):
            prev_count = len(walls)
            walls = self._snap_endpoints(walls, DEFAULT_SNAP_THRESHOLD)
            walls = self._merge_collinear_walls(walls)
            walls = self._extend_walls_to_intersections(walls)
            if len(walls) == prev_count:
                break
        walls = [w for w in walls if self._wall_length(w) >= max(DEFAULT_MIN_WALL_LENGTH, 0.5)]

        # Fallback: infer rooms from walls if segmentation produced none
        if not rooms and walls:
            rooms, extra_walls = self._infer_rooms_from_walls(walls, DEFAULT_MIN_ROOM_AREA)
            walls = walls + extra_walls

        # Remove zero-length, very short, and duplicate walls
        walls = self._clean_walls(walls)

        log.info(
            f"[FP3D] YOLO output: {len(walls)} walls, {len(doors)} doors, "
            f"{len(windows)} windows, {len(rooms)} rooms"
        )

        return {
            "scale": {"pixels_per_meter": pixels_per_meter},
            "walls": walls,
            "doors": doors,
            "windows": windows,
            "rooms": rooms,
        }

    # ── helper methods ────────────────────────────────────────────────

    @staticmethod
    def _find_nearest_wall(x, y, walls):
        """Find the index of the nearest wall to a point."""
        if not walls:
            return 0
        min_dist = float("inf")
        nearest_idx = 0
        for i, wall in enumerate(walls):
            sx, sy = wall["start"]
            ex, ey = wall["end"]
            dx, dy = ex - sx, ey - sy
            length_sq = dx * dx + dy * dy
            if length_sq < 1e-10:
                dist = ((x - sx) ** 2 + (y - sy) ** 2) ** 0.5
            else:
                t = max(0, min(1, ((x - sx) * dx + (y - sy) * dy) / length_sq))
                proj_x, proj_y = sx + t * dx, sy + t * dy
                dist = ((x - proj_x) ** 2 + (y - proj_y) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        return nearest_idx

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
                        if 0 < dist < threshold:
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
    def _merge_collinear_walls(walls, alignment_tol=0.4, gap_max=1.5,
                               max_result_length=18.0):
        """Merge collinear wall fragments that are nearly aligned.

        Two walls are collinear if:
        - Both horizontal (or both vertical),
        - Their fixed coordinate differs by less than alignment_tol,
        - The gap between their extents is ≤ gap_max,
        - The resulting wall is ≤ max_result_length.
        """
        merged = list(walls)
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(merged):
                j = i + 1
                while j < len(merged):
                    wi = merged[i]
                    wj = merged[j]

                    dxi = abs(wi["end"][0] - wi["start"][0])
                    dyi = abs(wi["end"][1] - wi["start"][1])
                    dxj = abs(wj["end"][0] - wj["start"][0])
                    dyj = abs(wj["end"][1] - wj["start"][1])

                    hi = dxi > dyi + 0.3
                    vi = dyi > dxi + 0.3
                    hj = dxj > dyj + 0.3
                    vj = dyj > dxj + 0.3

                    if hi and hj:
                        yi = (wi["start"][1] + wi["end"][1]) / 2
                        yj = (wj["start"][1] + wj["end"][1]) / 2
                        if abs(yi - yj) <= alignment_tol:
                            xi_min = min(wi["start"][0], wi["end"][0])
                            xi_max = max(wi["start"][0], wi["end"][0])
                            xj_min = min(wj["start"][0], wj["end"][0])
                            xj_max = max(wj["start"][0], wj["end"][0])
                            gap = max(0, max(xi_min, xj_min) - min(xi_max, xj_max))
                            result_len = max(xi_max, xj_max) - min(xi_min, xj_min)
                            if gap <= gap_max and result_len <= max_result_length:
                                avg_y = round((yi + yj) / 2, 3)
                                new_min = round(min(xi_min, xj_min), 3)
                                new_max = round(max(xi_max, xj_max), 3)
                                conf = max(
                                    wi.get("confidence", 0.5),
                                    wj.get("confidence", 0.5),
                                )
                                merged[i] = {
                                    "start": [new_min, avg_y],
                                    "end": [new_max, avg_y],
                                    "thickness": wi.get("thickness", 0.15),
                                    "confidence": round(conf, 3),
                                }
                                merged.pop(j)
                                changed = True
                                continue

                    elif vi and vj:
                        xi = (wi["start"][0] + wi["end"][0]) / 2
                        xj = (wj["start"][0] + wj["end"][0]) / 2
                        if abs(xi - xj) <= alignment_tol:
                            yi_min = min(wi["start"][1], wi["end"][1])
                            yi_max = max(wi["start"][1], wi["end"][1])
                            yj_min = min(wj["start"][1], wj["end"][1])
                            yj_max = max(wj["start"][1], wj["end"][1])
                            gap = max(0, max(yi_min, yj_min) - min(yi_max, yj_max))
                            result_len = max(yi_max, yj_max) - min(yi_min, yj_min)
                            if gap <= gap_max and result_len <= max_result_length:
                                avg_x = round((xi + xj) / 2, 3)
                                new_min = round(min(yi_min, yj_min), 3)
                                new_max = round(max(yi_max, yj_max), 3)
                                conf = max(
                                    wi.get("confidence", 0.5),
                                    wj.get("confidence", 0.5),
                                )
                                merged[i] = {
                                    "start": [avg_x, new_min],
                                    "end": [avg_x, new_max],
                                    "thickness": wi.get("thickness", 0.15),
                                    "confidence": round(conf, 3),
                                }
                                merged.pop(j)
                                changed = True
                                continue
                    j += 1
                i += 1
        return merged

    @staticmethod
    def _extend_walls_to_intersections(walls, max_extend=2.0):
        """Extend wall endpoints to reach nearby perpendicular wall lines.

        Creates proper T-junctions where the model detected walls that
        don't quite reach each other.
        """
        HORIZ_TOL = 0.5

        def is_horizontal(w):
            return abs(w["end"][0] - w["start"][0]) > abs(w["end"][1] - w["start"][1]) + HORIZ_TOL

        def is_vertical(w):
            return abs(w["end"][1] - w["start"][1]) > abs(w["end"][0] - w["start"][0]) + HORIZ_TOL

        walls = [dict(w, start=list(w["start"]), end=list(w["end"])) for w in walls]

        for wi in walls:
            if is_horizontal(wi):
                yi = (wi["start"][1] + wi["end"][1]) / 2
                x_min = min(wi["start"][0], wi["end"][0])
                x_max = max(wi["start"][0], wi["end"][0])

                for wj in walls:
                    if wj is wi or not is_vertical(wj):
                        continue
                    vx = (wj["start"][0] + wj["end"][0]) / 2
                    vy_min = min(wj["start"][1], wj["end"][1])
                    vy_max = max(wj["start"][1], wj["end"][1])

                    if not (vy_min - 0.5 <= yi <= vy_max + 0.5):
                        continue

                    if 0 < x_min - vx <= max_extend:
                        if wi["start"][0] <= wi["end"][0]:
                            wi["start"][0] = round(vx, 3)
                        else:
                            wi["end"][0] = round(vx, 3)

                    if 0 < vx - x_max <= max_extend:
                        if wi["end"][0] >= wi["start"][0]:
                            wi["end"][0] = round(vx, 3)
                        else:
                            wi["start"][0] = round(vx, 3)

            elif is_vertical(wi):
                xi = (wi["start"][0] + wi["end"][0]) / 2
                y_min = min(wi["start"][1], wi["end"][1])
                y_max = max(wi["start"][1], wi["end"][1])

                for wj in walls:
                    if wj is wi or not is_horizontal(wj):
                        continue
                    hy = (wj["start"][1] + wj["end"][1]) / 2
                    hx_min = min(wj["start"][0], wj["end"][0])
                    hx_max = max(wj["start"][0], wj["end"][0])

                    if not (hx_min - 0.5 <= xi <= hx_max + 0.5):
                        continue

                    if 0 < y_min - hy <= max_extend:
                        if wi["start"][1] <= wi["end"][1]:
                            wi["start"][1] = round(hy, 3)
                        else:
                            wi["end"][1] = round(hy, 3)

                    if 0 < hy - y_max <= max_extend:
                        if wi["end"][1] >= wi["start"][1]:
                            wi["end"][1] = round(hy, 3)
                        else:
                            wi["start"][1] = round(hy, 3)

        return walls

    @staticmethod
    def _clean_walls(walls, min_length=0.1, dedup_tol=0.05):
        """Remove zero-length, very short, and duplicate wall segments.

        Args:
            walls: List of wall dicts.
            min_length: Minimum wall length in meters.  Walls shorter than
                this (including zero-length) are dropped.
            dedup_tol: Coordinate tolerance for identifying duplicate walls.
                Two walls whose start/end (in either order) are within this
                distance are considered duplicates; only the first is kept.
        """
        # 1. Filter by minimum length
        cleaned = []
        for w in walls:
            dx = w["end"][0] - w["start"][0]
            dy = w["end"][1] - w["start"][1]
            length = (dx * dx + dy * dy) ** 0.5
            if length >= min_length:
                cleaned.append(w)

        # 2. Remove duplicates (same segment in either direction)
        seen = set()
        deduped = []
        for w in cleaned:
            s = (round(w["start"][0] / dedup_tol) * dedup_tol,
                 round(w["start"][1] / dedup_tol) * dedup_tol)
            e = (round(w["end"][0] / dedup_tol) * dedup_tol,
                 round(w["end"][1] / dedup_tol) * dedup_tol)
            key = (min(s, e), max(s, e))
            if key not in seen:
                seen.add(key)
                deduped.append(w)

        return deduped

    @staticmethod
    def _bridge_wall_gaps(walls, max_bridge=5.0):
        """Extend wall endpoints to connect with nearby perpendicular walls."""
        HORIZ_TOL = 0.5

        def is_horizontal(w):
            dy = abs(w["end"][1] - w["start"][1])
            dx = abs(w["end"][0] - w["start"][0])
            return dx > dy + HORIZ_TOL

        def is_vertical(w):
            dy = abs(w["end"][1] - w["start"][1])
            dx = abs(w["end"][0] - w["start"][0])
            return dy > dx + HORIZ_TOL

        h_walls = [w for w in walls if is_horizontal(w)]
        v_walls = [w for w in walls if is_vertical(w)]
        bridges = []

        for hw in h_walls:
            for endpoint_key in ("start", "end"):
                px, py = hw[endpoint_key]
                best_dist = max_bridge
                best_target = None
                for vw in v_walls:
                    vx = (vw["start"][0] + vw["end"][0]) / 2
                    vy_min = min(vw["start"][1], vw["end"][1])
                    vy_max = max(vw["start"][1], vw["end"][1])
                    if vy_min - 1.0 <= py <= vy_max + 1.0:
                        dist = abs(px - vx)
                        if dist < best_dist and dist > 0.05:
                            best_dist = dist
                            best_target = (vx, py)
                if best_target:
                    bridges.append({
                        "start": [px, py],
                        "end": [best_target[0], best_target[1]],
                        "thickness": 0.15,
                    })

        for vw in v_walls:
            for endpoint_key in ("start", "end"):
                px, py = vw[endpoint_key]
                best_dist = max_bridge
                best_target = None
                for hw in h_walls:
                    hy = (hw["start"][1] + hw["end"][1]) / 2
                    hx_min = min(hw["start"][0], hw["end"][0])
                    hx_max = max(hw["start"][0], hw["end"][0])
                    if hx_min - 1.0 <= px <= hx_max + 1.0:
                        dist = abs(py - hy)
                        if dist < best_dist and dist > 0.05:
                            best_dist = dist
                            best_target = (px, hy)
                if best_target:
                    bridges.append({
                        "start": [px, py],
                        "end": [best_target[0], best_target[1]],
                        "thickness": 0.15,
                    })

        # Bridge nearby dangling endpoints (not connected to another wall)
        snap_tol = 0.3
        all_endpoints = []
        for idx, w in enumerate(walls):
            all_endpoints.append((w["start"][0], w["start"][1], idx))
            all_endpoints.append((w["end"][0], w["end"][1], idx))

        def is_dangling(ep_idx):
            ax, ay, aw = all_endpoints[ep_idx]
            for k, (bx, by, bw) in enumerate(all_endpoints):
                if bw == aw:
                    continue
                if ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 < snap_tol:
                    return False
            return True

        dangling = [i for i in range(len(all_endpoints)) if is_dangling(i)]

        endpoint_bridge_max = 2.0
        connected = set()
        for i in dangling:
            ax, ay, aw = all_endpoints[i]
            best_dist = endpoint_bridge_max
            best_j = None
            for j in dangling:
                if all_endpoints[j][2] == aw:
                    continue
                bx, by, _ = all_endpoints[j]
                dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
                if 0.05 < dist < best_dist:
                    best_dist = dist
                    best_j = j
            if best_j is not None:
                bx, by, _ = all_endpoints[best_j]
                key = (min((ax, ay), (bx, by)), max((ax, ay), (bx, by)))
                if key not in connected:
                    connected.add(key)
                    bridges.append({
                        "start": [round(ax, 3), round(ay, 3)],
                        "end": [round(bx, 3), round(by, 3)],
                        "thickness": 0.15,
                    })

        return list(walls) + bridges

    @staticmethod
    def _trace_perimeter(walls, min_x, max_x, min_y, max_y, edge_tolerance=2.0):
        """Trace the building perimeter by connecting wall endpoints near each edge."""
        all_points = []
        for w in walls:
            all_points.append(tuple(w["start"]))
            all_points.append(tuple(w["end"]))
        unique_points = list(set(all_points))

        max_corner = 5.0
        perimeter = []

        def _make_wall(x1, y1, x2, y2):
            return {"start": [round(x1, 3), round(y1, 3)],
                    "end": [round(x2, 3), round(y2, 3)],
                    "thickness": 0.15}

        def _dist(x1, y1, x2, y2):
            return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

        left_pts = sorted(
            [p for p in unique_points if p[0] - min_x < edge_tolerance],
            key=lambda p: p[1],
        )
        for i in range(len(left_pts) - 1):
            perimeter.append(_make_wall(min_x, left_pts[i][1],
                                        min_x, left_pts[i + 1][1]))

        right_pts = sorted(
            [p for p in unique_points if max_x - p[0] < edge_tolerance],
            key=lambda p: p[1],
        )
        for i in range(len(right_pts) - 1):
            perimeter.append(_make_wall(max_x, right_pts[i][1],
                                        max_x, right_pts[i + 1][1]))

        top_pts = sorted(
            [p for p in unique_points if p[1] - min_y < edge_tolerance],
            key=lambda p: p[0],
        )
        for i in range(len(top_pts) - 1):
            perimeter.append(_make_wall(top_pts[i][0], min_y,
                                        top_pts[i + 1][0], min_y))

        bottom_pts = sorted(
            [p for p in unique_points if max_y - p[1] < edge_tolerance],
            key=lambda p: p[0],
        )
        for i in range(len(bottom_pts) - 1):
            perimeter.append(_make_wall(bottom_pts[i][0], max_y,
                                        bottom_pts[i + 1][0], max_y))

        # Corner connections — only if short enough
        if left_pts and top_pts:
            if _dist(min_x, left_pts[0][1], top_pts[0][0], min_y) <= max_corner:
                perimeter.append(_make_wall(min_x, left_pts[0][1],
                                            top_pts[0][0], min_y))
        if top_pts and right_pts:
            if _dist(top_pts[-1][0], min_y, max_x, right_pts[0][1]) <= max_corner:
                perimeter.append(_make_wall(top_pts[-1][0], min_y,
                                            max_x, right_pts[0][1]))
        if right_pts and bottom_pts:
            if _dist(max_x, right_pts[-1][1], bottom_pts[-1][0], max_y) <= max_corner:
                perimeter.append(_make_wall(max_x, right_pts[-1][1],
                                            bottom_pts[-1][0], max_y))
        if bottom_pts and left_pts:
            if _dist(bottom_pts[0][0], max_y, min_x, left_pts[-1][1]) <= max_corner:
                perimeter.append(_make_wall(bottom_pts[0][0], max_y,
                                            min_x, left_pts[-1][1]))

        return perimeter

    @staticmethod
    def _wall_coverage_ratio(polygon, walls, proximity=1.5):
        """What fraction of a room polygon's perimeter is near a real detected wall.

        Samples multiple points along each edge for accurate coverage.
        """
        total_len = 0.0
        covered_len = 0.0
        n = len(polygon)
        sample_spacing = 0.5

        for i in range(n):
            j = (i + 1) % n
            ex = polygon[j][0] - polygon[i][0]
            ey = polygon[j][1] - polygon[i][1]
            edge_len = (ex * ex + ey * ey) ** 0.5
            if edge_len < 1e-6:
                continue
            total_len += edge_len

            num_samples = max(1, int(edge_len / sample_spacing))
            covered_samples = 0
            for s in range(num_samples):
                t_edge = (s + 0.5) / num_samples
                mx = polygon[i][0] + t_edge * ex
                my = polygon[i][1] + t_edge * ey

                for w in walls:
                    sx, sy = w["start"]
                    wx, wy = w["end"]
                    dx, dy = wx - sx, wy - sy
                    lsq = dx * dx + dy * dy
                    if lsq < 1e-10:
                        dist = ((mx - sx) ** 2 + (my - sy) ** 2) ** 0.5
                    else:
                        t = max(0, min(1, ((mx - sx) * dx + (my - sy) * dy) / lsq))
                        px, py = sx + t * dx, sy + t * dy
                        dist = ((mx - px) ** 2 + (my - py) ** 2) ** 0.5
                    if dist <= proximity:
                        covered_samples += 1
                        break

            covered_len += edge_len * (covered_samples / num_samples)

        return covered_len / total_len if total_len > 0 else 0.0

    @staticmethod
    def _infer_rooms_from_walls(walls, min_room_area):
        """Infer room polygons from wall segments using flood-fill.

        Returns:
            tuple: (rooms_list, extra_walls_list)
        """
        try:
            import cv2
        except ImportError:
            log.warning("[FP3D] cv2 not available, cannot infer rooms from walls")
            return [], []

        if not walls:
            return [], []

        walls_merged = YOLOModelClient._merge_collinear_walls(walls)
        walls_extended = YOLOModelClient._extend_walls_to_intersections(walls_merged)

        all_x = [w["start"][0] for w in walls_extended] + [w["end"][0] for w in walls_extended]
        all_y = [w["start"][1] for w in walls_extended] + [w["end"][1] for w in walls_extended]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        footprint_area = max((max_x - min_x) * (max_y - min_y), 1.0)

        boundary_walls = YOLOModelClient._trace_perimeter(
            walls_extended, min_x, max_x, min_y, max_y)
        walls_bridged = YOLOModelClient._bridge_wall_gaps(walls_extended, max_bridge=5.0)
        all_walls = walls_bridged + boundary_walls

        all_walls = YOLOModelClient._snap_endpoints(all_walls, threshold=0.5)

        pad = 1.0
        rmin_x, rmin_y = min_x - pad, min_y - pad
        rmax_x, rmax_y = max_x + pad, max_y + pad

        resolution = 20
        grid_w = max(int((rmax_x - rmin_x) * resolution) + 1, 1)
        grid_h = max(int((rmax_y - rmin_y) * resolution) + 1, 1)

        if grid_w > 4000 or grid_h > 4000:
            resolution = 10
            grid_w = max(int((rmax_x - rmin_x) * resolution) + 1, 1)
            grid_h = max(int((rmax_y - rmin_y) * resolution) + 1, 1)

        grid = np.zeros((grid_h, grid_w), dtype=np.uint8)

        wall_px = max(int(0.15 * resolution), 2)
        for wall in all_walls:
            sx = int((wall["start"][0] - rmin_x) * resolution)
            sy = int((wall["start"][1] - rmin_y) * resolution)
            ex = int((wall["end"][0] - rmin_x) * resolution)
            ey = int((wall["end"][1] - rmin_y) * resolution)
            cv2.line(grid, (sx, sy), (ex, ey), 255, wall_px)
            cv2.circle(grid, (sx, sy), wall_px, 255, -1)
            cv2.circle(grid, (ex, ey), wall_px, 255, -1)

        k = max(int(0.15 * resolution), 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, kernel)

        border_mask = np.zeros((grid_h + 2, grid_w + 2), dtype=np.uint8)
        cv2.floodFill(grid, border_mask, (0, 0), 128)

        room_mask = (grid == 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_rooms = []
        max_single_room = footprint_area * 0.5

        for contour in contours:
            epsilon = max(resolution * 0.15, 1.5)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) < 3:
                continue
            polygon = []
            for pt in approx:
                mx = round(pt[0][0] / resolution + rmin_x, 3)
                my = round(pt[0][1] / resolution + rmin_y, 3)
                polygon.append([mx, my])
            area = YOLOModelClient._polygon_area(polygon)
            if area < min_room_area:
                continue

            if area > max_single_room:
                sub_rooms = YOLOModelClient._split_room_by_walls(
                    {"polygon": polygon, "area": round(area, 2)},
                    walls_extended, min_room_area)
                for sr in sub_rooms:
                    if sr["area"] > max_single_room:
                        # Blind halve it
                        xs = [p[0] for p in sr["polygon"]]
                        ys = [p[1] for p in sr["polygon"]]
                        w_ = max(xs) - min(xs)
                        h_ = max(ys) - min(ys)
                        if w_ >= h_:
                            mid = (min(xs) + max(xs)) / 2
                            raw_rooms.append({"polygon": [[min(xs), min(ys)], [mid, min(ys)], [mid, max(ys)], [min(xs), max(ys)]],
                                              "area": round((mid - min(xs)) * h_, 2)})
                            raw_rooms.append({"polygon": [[mid, min(ys)], [max(xs), min(ys)], [max(xs), max(ys)], [mid, max(ys)]],
                                              "area": round((max(xs) - mid) * h_, 2)})
                        else:
                            mid = (min(ys) + max(ys)) / 2
                            raw_rooms.append({"polygon": [[min(xs), min(ys)], [max(xs), min(ys)], [max(xs), mid], [min(xs), mid]],
                                              "area": round(w_ * (mid - min(ys)), 2)})
                            raw_rooms.append({"polygon": [[min(xs), mid], [max(xs), mid], [max(xs), max(ys)], [min(xs), max(ys)]],
                                              "area": round(w_ * (max(ys) - mid), 2)})
                    else:
                        raw_rooms.append(sr)
            else:
                raw_rooms.append({"polygon": polygon, "area": round(area, 2)})

        raw_rooms.sort(key=lambda r: r["area"])
        rooms = YOLOModelClient._resolve_room_overlaps(raw_rooms, min_room_area)

        # Filter phantom rooms
        filtered = []
        for r in rooms:
            coverage = YOLOModelClient._wall_coverage_ratio(r["polygon"], walls)
            if coverage >= 0.10:
                r["label"] = "room"
                filtered.append(r)
        rooms = filtered

        rooms.sort(key=lambda r: r["area"], reverse=True)
        rooms = YOLOModelClient._label_rooms(rooms)

        original_set = set()
        for w in walls:
            key = (tuple(w["start"]), tuple(w["end"]))
            original_set.add(key)
            original_set.add((tuple(w["end"]), tuple(w["start"])))

        extra_walls = []
        for w in boundary_walls:
            key = (tuple(w["start"]), tuple(w["end"]))
            if key not in original_set:
                extra_walls.append(w)
        for w in walls_bridged[len(walls_extended):]:
            key = (tuple(w["start"]), tuple(w["end"]))
            if key not in original_set:
                extra_walls.append(w)

        return rooms, extra_walls

    @staticmethod
    def _resolve_room_overlaps(rooms, min_area):
        """Remove overlapping area between rooms using Shapely."""
        try:
            from shapely.geometry import Polygon as ShapelyPolygon
            from shapely.validation import make_valid
        except ImportError:
            return rooms

        claimed = []
        result = []

        for room in rooms:
            poly = ShapelyPolygon(room["polygon"])
            if not poly.is_valid:
                poly = make_valid(poly)
            if poly.is_empty or poly.area < min_area:
                continue

            for claimed_poly, _ in claimed:
                poly = poly.difference(claimed_poly)
                if poly.is_empty:
                    break

            if poly.is_empty or poly.area < min_area:
                continue

            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda g: g.area)

            coords = list(poly.exterior.coords)[:-1]
            new_polygon = [[round(x, 3), round(y, 3)] for x, y in coords]
            new_area = round(poly.area, 2)

            if len(new_polygon) >= 3 and new_area >= min_area:
                result.append({"polygon": new_polygon, "area": new_area})
                claimed.append((poly, result[-1]))

        return result

    @staticmethod
    def _split_room_by_walls(room, walls, min_area):
        """Split an oversized room using interior wall segments."""
        polygon = room.get("polygon", [])
        if len(polygon) < 3:
            return [room]

        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        room_min_x, room_max_x = min(xs), max(xs)
        room_min_y, room_max_y = min(ys), max(ys)
        room_w = room_max_x - room_min_x
        room_h = room_max_y - room_min_y

        h_splits = []
        v_splits = []
        for w in walls:
            sx, sy = w["start"]
            ex, ey = w["end"]
            dx = abs(ex - sx)
            dy = abs(ey - sy)

            if dx > dy + 0.3:
                wy = (sy + ey) / 2
                margin = room_h * 0.15
                if (room_min_y + margin < wy < room_max_y - margin and
                        min(sx, ex) < room_max_x and max(sx, ex) > room_min_x):
                    h_splits.append(wy)
            elif dy > dx + 0.3:
                wx = (sx + ex) / 2
                margin = room_w * 0.15
                if (room_min_x + margin < wx < room_max_x - margin and
                        min(sy, ey) < room_max_y and max(sy, ey) > room_min_y):
                    v_splits.append(wx)

        split_results = []
        if room_w >= room_h and v_splits:
            center = (room_min_x + room_max_x) / 2
            v_splits.sort(key=lambda x: abs(x - center))
            split_results = YOLOModelClient._do_split(
                polygon, "v", v_splits[0], room_min_x, room_max_x, room_min_y, room_max_y)
        elif h_splits:
            center = (room_min_y + room_max_y) / 2
            h_splits.sort(key=lambda y: abs(y - center))
            split_results = YOLOModelClient._do_split(
                polygon, "h", h_splits[0], room_min_x, room_max_x, room_min_y, room_max_y)
        elif v_splits:
            center = (room_min_x + room_max_x) / 2
            v_splits.sort(key=lambda x: abs(x - center))
            split_results = YOLOModelClient._do_split(
                polygon, "v", v_splits[0], room_min_x, room_max_x, room_min_y, room_max_y)

        if not split_results:
            # Blind halving fallback
            if room_w >= room_h:
                mid = (room_min_x + room_max_x) / 2
                split_results = [
                    {"label": "room",
                     "polygon": [[room_min_x, room_min_y], [mid, room_min_y],
                                 [mid, room_max_y], [room_min_x, room_max_y]],
                     "area": round((mid - room_min_x) * room_h, 2)},
                    {"label": "room",
                     "polygon": [[mid, room_min_y], [room_max_x, room_min_y],
                                 [room_max_x, room_max_y], [mid, room_max_y]],
                     "area": round((room_max_x - mid) * room_h, 2)},
                ]
            else:
                mid = (room_min_y + room_max_y) / 2
                split_results = [
                    {"label": "room",
                     "polygon": [[room_min_x, room_min_y], [room_max_x, room_min_y],
                                 [room_max_x, mid], [room_min_x, mid]],
                     "area": round(room_w * (mid - room_min_y), 2)},
                    {"label": "room",
                     "polygon": [[room_min_x, mid], [room_max_x, mid],
                                 [room_max_x, room_max_y], [room_min_x, room_max_y]],
                     "area": round(room_w * (room_max_y - mid), 2)},
                ]

        return [r for r in split_results if r["area"] >= min_area]

    @staticmethod
    def _do_split(polygon, axis, split_val, min_x, max_x, min_y, max_y):
        """Split a polygon along a horizontal or vertical line using Shapely."""
        try:
            from shapely.geometry import Polygon as ShapelyPolygon, box
            from shapely.validation import make_valid
        except ImportError:
            return []

        poly = ShapelyPolygon(polygon)
        if not poly.is_valid:
            poly = make_valid(poly)

        if axis == "h":
            box_a = box(min_x - 1, min_y - 1, max_x + 1, split_val)
            box_b = box(min_x - 1, split_val, max_x + 1, max_y + 1)
        else:
            box_a = box(min_x - 1, min_y - 1, split_val, max_y + 1)
            box_b = box(split_val, min_y - 1, max_x + 1, max_y + 1)

        results = []
        for clip_box in (box_a, box_b):
            piece = poly.intersection(clip_box)
            if piece.is_empty:
                continue
            if piece.geom_type == "MultiPolygon":
                piece = max(piece.geoms, key=lambda g: g.area)
            if piece.geom_type != "Polygon" or piece.area < 0.5:
                continue
            coords = list(piece.exterior.coords)[:-1]
            results.append({
                "label": "room",
                "polygon": [[round(x, 3), round(y, 3)] for x, y in coords],
                "area": round(piece.area, 2),
            })

        return results

    @staticmethod
    def _label_rooms(rooms):
        """Assign room labels using multi-pass heuristics.

        Strategy (in order):
        1. Hallway/corridor — high aspect ratio, narrow dimension < 4 m
        2. Living room — the single largest non-hallway room
        3. Kitchen — one medium room (10-35 sq m), preferring the largest
        4. Bathrooms / WC — very small rooms (< 8 sq m)
        5. Bedrooms — everything else

        Most residential plans have 1 living room, 1 kitchen, 1-3 bedrooms,
        1-2 bathrooms, and optionally a hallway.
        """
        if not rooms:
            return rooms

        n = len(rooms)
        labels = [None] * n

        # Pre-compute geometry for each room
        aspects = []
        narrow_dims = []
        for room in rooms:
            xs = [p[0] for p in room["polygon"]]
            ys = [p[1] for p in room["polygon"]]
            bb_w = max(xs) - min(xs)
            bb_h = max(ys) - min(ys)
            aspect = max(bb_w, bb_h) / max(min(bb_w, bb_h), 0.1)
            narrow = min(bb_w, bb_h)
            aspects.append(aspect)
            narrow_dims.append(narrow)

        # ── Pass 1: hallways (high aspect ratio, narrow) ────────────
        hall_count = 0
        for i, room in enumerate(rooms):
            if aspects[i] > 3.0 and narrow_dims[i] < 4.0:
                hall_count += 1
                labels[i] = "hallway" if hall_count == 1 else f"hallway_{hall_count}"

        # ── Pass 2: living room (largest non-hallway) ───────────────
        remaining = [(i, rooms[i]["area"]) for i in range(n) if labels[i] is None]
        remaining.sort(key=lambda x: x[1], reverse=True)
        if remaining:
            labels[remaining[0][0]] = "living_room"
            remaining = remaining[1:]

        # ── Pass 3: kitchen (one medium room, 10-35 sq m) ───────────
        kitchen_candidates = [
            (i, a) for i, a in remaining
            if 10 <= a <= 35
        ]
        if kitchen_candidates:
            best = max(kitchen_candidates, key=lambda x: x[1])
            labels[best[0]] = "kitchen"

        # ── Pass 4: bathrooms / WC (very small rooms) ───────────────
        bath_count = 0
        bath_labels = ["bathroom", "wc", "utility"]
        remaining = [(i, rooms[i]["area"]) for i in range(n) if labels[i] is None]
        remaining.sort(key=lambda x: x[1])  # smallest first
        for i, area in remaining:
            if area < 8:
                if bath_count < len(bath_labels):
                    labels[i] = bath_labels[bath_count]
                else:
                    labels[i] = f"bathroom_{bath_count}"
                bath_count += 1

        # ── Pass 5: everything else → bedroom ───────────────────────
        bed_count = 0
        for i in range(n):
            if labels[i] is None:
                bed_count += 1
                labels[i] = "bedroom" if bed_count == 1 else f"bedroom_{bed_count}"

        # Apply labels
        for i, room in enumerate(rooms):
            room["label"] = labels[i]

        return rooms

    @staticmethod
    def _mask_to_polygon(xy_points, pixels_per_meter):
        """Convert mask contour points to a simplified polygon in meters."""
        if len(xy_points) < 3:
            return []

        points = np.array(xy_points)

        # Simplify polygon using Douglas-Peucker algorithm
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
