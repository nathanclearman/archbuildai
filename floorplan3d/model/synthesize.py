"""
Procedural floor plan generator.

Renders synthetic floor plans with known ground truth to augment the real
training data. Uses only numpy + Pillow so it runs anywhere, no heavy deps.

Each emitted sample is an (image.png, image.json) pair where the JSON
conforms to schema.py. Run this once to produce N samples, then point
dataset.build_training_set(synthetic_root=...) at the output directory.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from schema import FloorPlan, serialize  # type: ignore


ROOM_LABELS = [
    "living_room", "bedroom", "kitchen", "bathroom", "hallway",
    "dining_room", "foyer", "closet", "office", "laundry",
]


@dataclass
class SynthConfig:
    image_size: int = 800
    pixels_per_meter: float = 50.0
    wall_thickness_m: float = 0.15
    min_rooms: int = 3
    max_rooms: int = 8


def generate_one(seed: int, cfg: SynthConfig | None = None):
    """Generate one synthetic plan. Returns (PIL.Image, floor_plan_dict)."""
    from PIL import Image, ImageDraw
    cfg = cfg or SynthConfig()
    rng = random.Random(seed)

    size = cfg.image_size
    ppm = cfg.pixels_per_meter

    # 1. Pick a building footprint (in meters).
    width_m = rng.uniform(8, 14)
    height_m = rng.uniform(6, 12)

    # 2. Recursively split the footprint into rooms.
    n_rooms = rng.randint(cfg.min_rooms, cfg.max_rooms)
    rects_m = _bsp_split(0, 0, width_m, height_m, n_rooms, rng, min_side=2.0)

    # 3. Convert to walls + rooms in our schema.
    walls_m, rooms_m = _rects_to_schema(rects_m, rng)

    # 4. Sprinkle doors and windows along walls.
    doors_m = _sprinkle_doors(walls_m, rects_m, rng)
    windows_m = _sprinkle_windows(walls_m, width_m, height_m, rng)

    fp = FloorPlan(
        walls=walls_m,
        doors=doors_m,
        windows=windows_m,
        rooms=rooms_m,
        scale={"pixels_per_meter": int(ppm)},
    )

    # 5. Render.
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    # Center the footprint in the image.
    off_x = (size - width_m * ppm) / 2
    off_y = (size - height_m * ppm) / 2

    def to_px(p):
        return (off_x + p[0] * ppm, off_y + p[1] * ppm)

    for r in rooms_m:
        poly = [to_px(p) for p in r["polygon"]]
        fill = (240, 240, 235) if r["label"] == "living_room" else (250, 247, 240)
        draw.polygon(poly, fill=fill, outline=None)

    wall_px = max(2, int(cfg.wall_thickness_m * ppm * 0.8))
    for w in walls_m:
        draw.line([to_px(w["start"]), to_px(w["end"])], fill=(20, 20, 20), width=wall_px)

    door_px = max(3, wall_px + 1)
    for d in doors_m:
        cx, cy = to_px(d["position"])
        half = d["width"] * ppm / 2
        draw.line([(cx - half, cy), (cx + half, cy)], fill=(220, 220, 220), width=door_px)

    for w in windows_m:
        cx, cy = to_px(w["position"])
        half = w["width"] * ppm / 2
        draw.line([(cx - half, cy), (cx + half, cy)], fill=(140, 180, 220), width=door_px)

    return img, fp.to_dict()


def _bsp_split(x, y, w, h, target_rooms, rng, min_side):
    """Binary space partition into `target_rooms` rectangles."""
    rects = [(x, y, w, h)]
    while len(rects) < target_rooms:
        # pick the largest rect to split.
        rects.sort(key=lambda r: r[2] * r[3], reverse=True)
        rx, ry, rw, rh = rects.pop(0)
        # split along the longer dimension if possible.
        if rw >= rh and rw >= 2 * min_side:
            cut = rng.uniform(min_side, rw - min_side)
            rects.append((rx, ry, cut, rh))
            rects.append((rx + cut, ry, rw - cut, rh))
        elif rh >= 2 * min_side:
            cut = rng.uniform(min_side, rh - min_side)
            rects.append((rx, ry, rw, cut))
            rects.append((rx, ry + cut, rw, rh - cut))
        else:
            rects.append((rx, ry, rw, rh))  # can't split further
            break
    return rects


def _rects_to_schema(rects, rng):
    walls = []
    rooms = []
    seen_edges: set[tuple] = set()

    def add_wall(a, b):
        key = tuple(sorted([tuple(round(c, 3) for c in a), tuple(round(c, 3) for c in b)]))
        if key in seen_edges:
            return
        seen_edges.add(key)
        walls.append({"start": list(a), "end": list(b), "thickness": 0.15})

    for (x, y, w, h) in rects:
        poly = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        add_wall(poly[0], poly[1])
        add_wall(poly[1], poly[2])
        add_wall(poly[2], poly[3])
        add_wall(poly[3], poly[0])
        label = rng.choice(ROOM_LABELS)
        rooms.append({"label": label, "polygon": poly, "area": round(w * h, 2)})
    return walls, rooms


def _sprinkle_doors(walls, rects, rng):
    doors = []
    # Put one door between each adjacent pair of rooms (approximate).
    for i, a in enumerate(rects):
        for j, b in enumerate(rects):
            if j <= i:
                continue
            overlap = _rect_edge_overlap(a, b)
            if overlap is None:
                continue
            cx, cy = overlap
            # Find nearest wall to this point.
            wall_idx = _nearest_wall([cx, cy], walls)
            doors.append({
                "position": [round(cx, 2), round(cy, 2)],
                "width": 0.9,
                "type": "hinged",
                "wall_index": wall_idx,
            })
    # Add a front door on an exterior wall.
    if walls:
        exterior = walls[0]
        sx, sy = exterior["start"]
        ex, ey = exterior["end"]
        doors.append({
            "position": [round((sx + ex) / 2, 2), round((sy + ey) / 2, 2)],
            "width": 1.0,
            "type": "hinged",
            "wall_index": 0,
        })
    return doors


def _sprinkle_windows(walls, total_w, total_h, rng):
    """One window per exterior wall segment on average."""
    windows = []
    for i, w in enumerate(walls):
        sx, sy = w["start"]
        ex, ey = w["end"]
        on_boundary = (
            (sx == 0 and ex == 0) or (sy == 0 and ey == 0)
            or (abs(sx - total_w) < 0.01 and abs(ex - total_w) < 0.01)
            or (abs(sy - total_h) < 0.01 and abs(ey - total_h) < 0.01)
        )
        if not on_boundary:
            continue
        if rng.random() > 0.6:
            continue
        cx, cy = (sx + ex) / 2, (sy + ey) / 2
        windows.append({
            "position": [round(cx, 2), round(cy, 2)],
            "width": round(rng.uniform(0.9, 1.6), 2),
            "wall_index": i,
        })
    return windows


def _rect_edge_overlap(a, b):
    """If rects a and b share a wall segment, return its midpoint."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    # vertical shared edge
    if abs(ax + aw - bx) < 0.01:
        y0, y1 = max(ay, by), min(ay + ah, by + bh)
        if y1 - y0 > 1.0:
            return (ax + aw, (y0 + y1) / 2)
    if abs(bx + bw - ax) < 0.01:
        y0, y1 = max(ay, by), min(ay + ah, by + bh)
        if y1 - y0 > 1.0:
            return (ax, (y0 + y1) / 2)
    # horizontal shared edge
    if abs(ay + ah - by) < 0.01:
        x0, x1 = max(ax, bx), min(ax + aw, bx + bw)
        if x1 - x0 > 1.0:
            return ((x0 + x1) / 2, ay + ah)
    if abs(by + bh - ay) < 0.01:
        x0, x1 = max(ax, bx), min(ax + aw, bx + bw)
        if x1 - x0 > 1.0:
            return ((x0 + x1) / 2, ay)
    return None


def _nearest_wall(point, walls):
    if not walls:
        return -1
    px, py = point
    best_i, best_d = 0, float("inf")
    for i, w in enumerate(walls):
        sx, sy = w["start"]
        ex, ey = w["end"]
        cx, cy = (sx + ex) / 2, (sy + ey) / 2
        d = (px - cx) ** 2 + (py - cy) ** 2
        if d < best_d:
            best_d, best_i = d, i
    return best_i


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic floor plans")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=800)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = SynthConfig(image_size=args.size)

    for i in range(args.count):
        img, fp = generate_one(args.seed + i, cfg)
        img.save(out / f"plan_{i:06d}.png")
        (out / f"plan_{i:06d}.json").write_text(serialize(fp))
        if (i + 1) % 100 == 0:
            print(f"[{i + 1}/{args.count}] generated")

    print(f"done: {args.count} samples in {out}")


if __name__ == "__main__":
    main()
