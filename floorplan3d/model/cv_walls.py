"""
Classical-CV wall extractor.

Runs as the first stage of the hybrid pipeline: extracts a wall graph from
a raster floor plan using thresholding + morphology + Hough lines, then
flood-fills the enclosed regions to produce unlabeled room polygons.

Output conforms to schema.py (FloorPlan). Room labels are left empty —
the VLM / Claude stage labels them. This stage only handles geometry.

Runs entirely on CPU. No ML dependencies.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from schema import FloorPlan, serialize  # type: ignore


@dataclass
class CVConfig:
    pixels_per_meter: float = 50.0
    wall_min_length_px: int = 40
    wall_thickness_m: float = 0.15
    hough_threshold: int = 80
    hough_min_line_length: int = 50
    hough_max_line_gap: int = 10
    angle_snap_deg: float = 3.0           # snap near-axis lines to axis
    endpoint_snap_px: float = 8.0         # merge endpoints within this radius
    min_room_area_px: int = 2500          # skip tiny regions


def extract(image_path: str | Path, cfg: CVConfig | None = None) -> dict:
    """Extract wall graph + raw room polygons from a floor plan raster.

    Returns a dict conforming to schema.FloorPlan. Rooms have empty labels —
    downstream stages fill them in.
    """
    import cv2  # local import: keeps the Blender side lean
    import numpy as np

    cfg = cfg or CVConfig()
    path = str(image_path)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")

    h, w = img.shape

    # 1. Binarize: walls are dark ink on light background.
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 2. Clean up: close small gaps in wall strokes, remove noise.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 3. Hough lines for wall segments.
    lines = cv2.HoughLinesP(
        closed,
        rho=1,
        theta=np.pi / 180,
        threshold=cfg.hough_threshold,
        minLineLength=cfg.hough_min_line_length,
        maxLineGap=cfg.hough_max_line_gap,
    )
    segments: list[tuple[float, float, float, float]] = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length < cfg.wall_min_length_px:
                continue
            segments.append(_snap_angle((x1, y1, x2, y2), cfg.angle_snap_deg))

    segments = _merge_collinear(segments, cfg.endpoint_snap_px)
    segments = _snap_endpoints(segments, cfg.endpoint_snap_px)

    # 4. Rasterize the cleaned wall graph and flood-fill to recover rooms.
    room_mask = _rasterize_walls(h, w, segments)
    room_polys_px = _find_rooms(room_mask, cfg.min_room_area_px)

    # 5. Convert pixels → meters.
    ppm = cfg.pixels_per_meter
    walls = [
        {
            "start": [x1 / ppm, y1 / ppm],
            "end": [x2 / ppm, y2 / ppm],
            "thickness": cfg.wall_thickness_m,
        }
        for (x1, y1, x2, y2) in segments
    ]
    rooms = [
        {
            "label": "",  # filled in downstream
            "polygon": [[x / ppm, y / ppm] for (x, y) in poly],
            "area": _polygon_area([[x / ppm, y / ppm] for (x, y) in poly]),
        }
        for poly in room_polys_px
    ]

    return FloorPlan(
        walls=walls,
        doors=[],
        windows=[],
        rooms=rooms,
        scale={"pixels_per_meter": int(ppm)},
    ).to_dict()


# ---------- helpers ----------

def _snap_angle(seg, max_deg):
    import math
    x1, y1, x2, y2 = seg
    dx, dy = x2 - x1, y2 - y1
    angle = math.degrees(math.atan2(dy, dx))
    # Snap to horizontal or vertical if close enough.
    if abs(angle) < max_deg or abs(abs(angle) - 180) < max_deg:
        y2 = y1  # horizontal
    elif abs(abs(angle) - 90) < max_deg:
        x2 = x1  # vertical
    return (float(x1), float(y1), float(x2), float(y2))


def _snap_endpoints(segments, tol):
    """Merge endpoints within `tol` pixels to a shared averaged point."""
    if not segments:
        return segments
    pts = []
    for (x1, y1, x2, y2) in segments:
        pts.append([x1, y1])
        pts.append([x2, y2])

    # Greedy union-find by proximity.
    parent = list(range(len(pts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if abs(pts[i][0] - pts[j][0]) < tol and abs(pts[i][1] - pts[j][1]) < tol:
                parent[find(i)] = find(j)

    # Average each cluster.
    clusters: dict[int, list[list[float]]] = {}
    for i, p in enumerate(pts):
        clusters.setdefault(find(i), []).append(p)
    averaged = {
        root: [sum(p[0] for p in ps) / len(ps), sum(p[1] for p in ps) / len(ps)]
        for root, ps in clusters.items()
    }

    out = []
    for k, (x1, y1, x2, y2) in enumerate(segments):
        a = averaged[find(2 * k)]
        b = averaged[find(2 * k + 1)]
        out.append((a[0], a[1], b[0], b[1]))
    return out


def _merge_collinear(segments, tol):
    """Merge nearly-collinear overlapping segments into single spans."""
    # Simple pass: group by rounded axis (horizontal → y, vertical → x)
    # and collapse overlapping ranges.
    merged: list[tuple[float, float, float, float]] = []
    horizontals: dict[int, list[tuple[float, float]]] = {}
    verticals: dict[int, list[tuple[float, float]]] = {}
    others: list[tuple[float, float, float, float]] = []

    for (x1, y1, x2, y2) in segments:
        if abs(y1 - y2) < 1:
            key = round((y1 + y2) / 2 / tol) * int(tol)
            horizontals.setdefault(int(key), []).append((min(x1, x2), max(x1, x2)))
        elif abs(x1 - x2) < 1:
            key = round((x1 + x2) / 2 / tol) * int(tol)
            verticals.setdefault(int(key), []).append((min(y1, y2), max(y1, y2)))
        else:
            others.append((x1, y1, x2, y2))

    for y, ranges in horizontals.items():
        for a, b in _collapse_ranges(ranges, tol):
            merged.append((a, float(y), b, float(y)))
    for x, ranges in verticals.items():
        for a, b in _collapse_ranges(ranges, tol):
            merged.append((float(x), a, float(x), b))
    merged.extend(others)
    return merged


def _collapse_ranges(ranges, tol):
    ranges = sorted(ranges)
    out = []
    cur_a, cur_b = ranges[0]
    for a, b in ranges[1:]:
        if a <= cur_b + tol:
            cur_b = max(cur_b, b)
        else:
            out.append((cur_a, cur_b))
            cur_a, cur_b = a, b
    out.append((cur_a, cur_b))
    return out


def _rasterize_walls(h, w, segments):
    import cv2
    import numpy as np
    mask = np.ones((h, w), dtype=np.uint8) * 255
    for (x1, y1, x2, y2) in segments:
        cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 0, thickness=3)
    return mask


def _find_rooms(mask, min_area):
    import cv2
    import numpy as np
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    polys = []
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        # Skip the background (largest component touching image border).
        x, y, w, h = stats[lbl, cv2.CC_STAT_LEFT], stats[lbl, cv2.CC_STAT_TOP], stats[lbl, cv2.CC_STAT_WIDTH], stats[lbl, cv2.CC_STAT_HEIGHT]
        if x == 0 and y == 0 and w == mask.shape[1] and h == mask.shape[0]:
            continue
        region = (labels == lbl).astype(np.uint8) * 255
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        poly = [(float(p[0][0]), float(p[0][1])) for p in approx]
        if len(poly) >= 3:
            polys.append(poly)
    return polys


def _polygon_area(polygon):
    n = len(polygon)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return round(abs(area) / 2.0, 2)


def main():
    parser = argparse.ArgumentParser(description="Extract wall graph from a floor plan raster")
    parser.add_argument("--image", required=True)
    parser.add_argument("--ppm", type=float, default=50.0, help="pixels per meter")
    parser.add_argument("--out", default="-", help="output file or - for stdout")
    args = parser.parse_args()

    result = extract(args.image, CVConfig(pixels_per_meter=args.ppm))
    text = serialize(result)
    if args.out == "-":
        print(text)
    else:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
