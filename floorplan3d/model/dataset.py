"""
Training-data loader for the floor plan VLM.

Reads CubiCasa5k (and, when available, ResPlan / CFP / our synthetic set),
normalizes every sample to the canonical schema (see schema.py), and emits
(image, target_json_string) pairs suitable for SFT of a vision-language model.

Handles only the dataset-specific parsing. The training loop itself
(tokenization, chat-template formatting, batching) lives in train.py.
"""

from __future__ import annotations

import json
import random
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from schema import FloorPlan, serialize  # type: ignore


# ---------- CubiCasa5k ----------

# CubiCasa5k stores each sample as a directory containing:
#   F1_scaled.png    (floor plan image)
#   F1_original.png  (optional higher-res)
#   model.svg        (SVG annotations: walls, doors, windows, rooms)
# Category folders: colorful/, high_quality/, high_quality_architectural/

_SVG_NS = "{http://www.w3.org/2000/svg}"

# CubiCasa svg class name → our canonical category
CUBI_CLASS_MAP = {
    "Wall": "wall",
    "Railing": "wall",
    "Door": "door",
    "Window": "window",
    "Room": "room",
}

# CubiCasa room-class names → our canonical room label vocabulary.
CUBI_ROOM_LABELS = {
    "LivingRoom": "living_room",
    "Kitchen": "kitchen",
    "Bedroom": "bedroom",
    "Bathroom": "bathroom",
    "Hallway": "hallway",
    "Corridor": "hallway",
    "Entry": "foyer",
    "DiningRoom": "dining_room",
    "Storage": "storage",
    "Garage": "garage",
    "Closet": "closet",
    "Outdoor": "balcony",
    "Balcony": "balcony",
    "Undefined": "room",
}


@dataclass
class Sample:
    image_path: Path
    target_json: str          # serialized FloorPlan, canonical format
    source: str               # dataset name, for weighting/logging


class CubiCasaLoader:
    """Iterate (image, target_json) pairs from a CubiCasa5k directory tree."""

    def __init__(self, root: str | Path, pixels_per_meter: float = 50.0):
        self.root = Path(root)
        self.ppm = pixels_per_meter

    def __iter__(self) -> Iterator[Sample]:
        if not self.root.exists():
            raise FileNotFoundError(f"CubiCasa5k root not found: {self.root}")
        for sample_dir in sorted(self.root.rglob("model.svg")):
            sample_dir = sample_dir.parent
            image = sample_dir / "F1_scaled.png"
            if not image.exists():
                image = sample_dir / "F1_original.png"
            if not image.exists():
                continue
            try:
                fp = self._parse_svg(sample_dir / "model.svg")
            except Exception as e:
                # Skip malformed samples rather than blowing up the epoch.
                print(f"[skip] {sample_dir}: {e}")
                continue
            yield Sample(
                image_path=image,
                target_json=serialize(fp),
                source="cubicasa5k",
            )

    def _parse_svg(self, svg_path: Path) -> FloorPlan:
        tree = ET.parse(svg_path)
        root = tree.getroot()

        walls: list[dict] = []
        doors: list[dict] = []
        windows: list[dict] = []
        rooms: list[dict] = []

        for elem in root.iter():
            cls = elem.attrib.get("class", "")
            kind = next((v for k, v in CUBI_CLASS_MAP.items() if k in cls), None)
            if kind is None:
                continue

            polygon = _parse_polygon(elem)
            if not polygon:
                continue

            if kind == "wall":
                start, end, thickness = _polygon_to_wall(polygon)
                walls.append(
                    {
                        "start": [start[0] / self.ppm, start[1] / self.ppm],
                        "end": [end[0] / self.ppm, end[1] / self.ppm],
                        "thickness": round(thickness / self.ppm, 3),
                    }
                )
            elif kind == "door":
                cx, cy, width = _polygon_bbox_center(polygon)
                doors.append(
                    {
                        "position": [cx / self.ppm, cy / self.ppm],
                        "width": round(width / self.ppm, 2),
                        "type": "hinged",
                        "wall_index": -1,
                    }
                )
            elif kind == "window":
                cx, cy, width = _polygon_bbox_center(polygon)
                windows.append(
                    {
                        "position": [cx / self.ppm, cy / self.ppm],
                        "width": round(width / self.ppm, 2),
                        "wall_index": -1,
                    }
                )
            elif kind == "room":
                label = _room_label_from_class(cls)
                metric_poly = [[p[0] / self.ppm, p[1] / self.ppm] for p in polygon]
                rooms.append(
                    {
                        "label": label,
                        "polygon": metric_poly,
                        "area": _polygon_area(metric_poly),
                    }
                )

        # Associate doors/windows with nearest wall (best-effort, improved in training).
        for d in doors:
            d["wall_index"] = _nearest_wall(d["position"], walls)
        for w in windows:
            w["wall_index"] = _nearest_wall(w["position"], walls)

        return FloorPlan(
            walls=walls,
            doors=doors,
            windows=windows,
            rooms=rooms,
            scale={"pixels_per_meter": int(self.ppm)},
        )


def _parse_polygon(elem) -> list[tuple[float, float]]:
    """Convert a <polygon>, <rect>, or <path> element to a list of points."""
    tag = elem.tag.replace(_SVG_NS, "")
    if tag == "polygon":
        points = elem.attrib.get("points", "")
        return _parse_points_attr(points)
    if tag == "rect":
        x = float(elem.attrib.get("x", 0))
        y = float(elem.attrib.get("y", 0))
        w = float(elem.attrib.get("width", 0))
        h = float(elem.attrib.get("height", 0))
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    if tag == "path":
        d = elem.attrib.get("d", "")
        return _parse_path_d(d)
    return []


def _parse_points_attr(s: str) -> list[tuple[float, float]]:
    tokens = re.split(r"[,\s]+", s.strip())
    pts = []
    for i in range(0, len(tokens) - 1, 2):
        try:
            pts.append((float(tokens[i]), float(tokens[i + 1])))
        except ValueError:
            continue
    return pts


def _parse_path_d(d: str) -> list[tuple[float, float]]:
    """Very permissive SVG path parser — handles M/L absolute coords only."""
    pts = []
    for match in re.finditer(r"[ML]\s*(-?\d+\.?\d*)[ ,](-?\d+\.?\d*)", d):
        pts.append((float(match.group(1)), float(match.group(2))))
    return pts


def _polygon_to_wall(polygon: list[tuple[float, float]]):
    """Collapse a thin polygon (rectangle-ish) to a single wall segment + thickness."""
    if len(polygon) < 2:
        return (0, 0), (0, 0), 0.15
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    if w >= h:
        y = (min(ys) + max(ys)) / 2
        return (min(xs), y), (max(xs), y), h
    else:
        x = (min(xs) + max(xs)) / 2
        return (x, min(ys)), (x, max(ys)), w


def _polygon_bbox_center(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    width = max(max(xs) - min(xs), max(ys) - min(ys))
    return (sum(xs) / len(xs), sum(ys) / len(ys), width)


def _room_label_from_class(class_attr: str) -> str:
    for key, label in CUBI_ROOM_LABELS.items():
        if key in class_attr:
            return label
    return "room"


def _polygon_area(polygon) -> float:
    n = len(polygon)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return round(abs(area) / 2.0, 2)


def _nearest_wall(point, walls) -> int:
    if not walls:
        return -1
    px, py = point
    best_i, best_d = -1, float("inf")
    for i, w in enumerate(walls):
        sx, sy = w["start"]
        ex, ey = w["end"]
        dx, dy = ex - sx, ey - sy
        l2 = dx * dx + dy * dy
        if l2 < 1e-9:
            d = ((px - sx) ** 2 + (py - sy) ** 2) ** 0.5
        else:
            t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / l2))
            cx, cy = sx + t * dx, sy + t * dy
            d = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
        if d < best_d:
            best_d, best_i = d, i
    return best_i


# ---------- composed training set ----------

def build_training_set(
    cubicasa_root: str | Path | None = None,
    synthetic_root: str | Path | None = None,
    shuffle: bool = True,
    seed: int = 0,
) -> list[Sample]:
    """Walk all available datasets and return a combined list of Samples.

    ResPlan and CFP loaders will be added as separate methods when we
    download those corpora — the current bottleneck is CubiCasa5k + synth.
    """
    samples: list[Sample] = []
    if cubicasa_root:
        samples.extend(list(CubiCasaLoader(cubicasa_root)))
    if synthetic_root:
        samples.extend(list(_load_synthetic(synthetic_root)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(samples)
    return samples


def _load_synthetic(root: str | Path) -> Iterator[Sample]:
    """Load samples produced by synthesize.py: each sample is a .png + .json pair."""
    root = Path(root)
    if not root.exists():
        return
    for img in sorted(root.glob("*.png")):
        target = img.with_suffix(".json")
        if not target.exists():
            continue
        yield Sample(
            image_path=img,
            target_json=target.read_text().strip(),
            source="synthetic",
        )
