"""
US-style procedural floor plan generator.

Produces synthetic training data with the room conventions a US-market
product needs to recognize: great rooms (merged living + dining + kitchen),
master suites (bedroom + en-suite + walk-in closet as a connected triplet),
attached garages, mudrooms, powder rooms, pantries.

Two templates so far:
  - ranch_open_concept: single-floor rectangular ranch with open great
    room, bedroom wing, attached garage. Most common US SFH layout.
  - colonial_compartmentalized: single-floor compartmentalized layout
    with separate living / dining / kitchen rooms.

Each emitted sample is an (image.png, image.json) pair where the JSON
conforms to schema.py. No ML dependencies — pure numpy + Pillow.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from schema import FloorPlan, serialize  # type: ignore


# US room vocabulary used throughout the project. Keep this list in sync
# with claude_refiner.py's prompt — drift between them = bad eval numbers.
US_ROOM_LABELS = [
    "great_room", "living_room", "family_room", "dining_room", "kitchen",
    "master_bedroom", "bedroom", "en_suite", "bathroom", "powder_room",
    "walk_in_closet", "closet", "foyer", "hallway", "mudroom",
    "laundry_room", "pantry", "garage", "office", "den",
]


@dataclass
class SynthConfig:
    image_size: int = 900
    pixels_per_meter: float = 40.0
    wall_thickness_m: float = 0.15
    draw_labels: bool = True
    draw_fixtures: bool = True


# ---------- data model ----------

@dataclass
class Room:
    label: str
    # axis-aligned rectangle in meters: (x, y, w, h)
    rect: tuple[float, float, float, float]
    # doors connecting this room to neighbors, placed on shared walls
    # as (shared_with_label, position_along_shared_wall_0to1, width)
    doors: list[tuple[str, float, float]] = field(default_factory=list)


@dataclass
class Plan:
    rooms: list[Room]
    footprint: tuple[float, float]  # overall width, height in meters
    exterior_door: tuple[str, float] | None = None  # (room_label, side) where side ∈ {N,S,E,W}


# ---------- templates ----------

def ranch_open_concept(rng: random.Random) -> Plan:
    """Single-floor open-concept ranch, ~1800-2400 sqft.

    Layout (axes: +x east, +y south):
      +------------------------------------------------+
      |   garage   | mud  |        kitchen             |
      |            +------+----------------------------+
      |            |                                   |
      +------------+   great_room (LR + DR + kitchen)  |
      |  bedroom   |                                   |
      +------------+------+------+------+---------+----+
      |  bedroom   | bath | bed3 | hall |  M.BR   | wic|
      |            |      |      |      +---------+    |
      |            |      |      |      | en_suite|    |
      +------------+------+------+------+---------+----+

    Bedroom wing is a single row of full-height columns so every room has
    a clean rectangular polygon with no voids or overlaps. The vertical
    hallway bridges the bed3/bath cluster to the master suite; bed2 (over
    the garage footprint) and bed3 share the bathroom jack-and-jill style
    — the single-rectangle hallway can only be adjacent to two
    bedroom-wing neighbours, so this is the most connected arrangement
    achievable without an L-shaped corridor.
    """
    w = rng.uniform(16, 20)           # overall width (meters)
    h = rng.uniform(11, 14)           # overall height
    garage_w = rng.uniform(5.5, 6.5)
    mud_w = rng.uniform(1.8, 2.6)
    great_w = w - garage_w
    great_h = h * rng.uniform(0.45, 0.55)
    # Garage shares the top-half height with great_room so the two bands
    # line up exactly; the earlier independent garage_h could exceed great_h
    # and spill into the bedroom wing.
    garage_h = great_h

    bed_wing_h = h - great_h
    mbr_w = rng.uniform(4.5, 5.5)
    wic_w = rng.uniform(1.8, 2.5)
    hall_w = rng.uniform(1.2, 1.5)
    bath_w = rng.uniform(2.0, 2.6)
    # bed3_w fills the remaining wing-row width. The bottom row is:
    #   bed2 (garage_w) | bath | bed3 | hall | master | wic  ==  w
    bed3_w = w - garage_w - bath_w - hall_w - mbr_w - wic_w
    if bed3_w < 2.5:
        # Not enough wall budget — trim the master and walk-in to recover.
        deficit = 2.5 - bed3_w
        mbr_w = max(4.0, mbr_w - deficit * 0.7)
        wic_w = max(1.5, wic_w - deficit * 0.3)
        bed3_w = w - garage_w - bath_w - hall_w - mbr_w - wic_w

    rooms: list[Room] = []

    # --- top half: garage | mudroom+laundry | great_room ---
    rooms.append(Room("garage", (0, 0, garage_w, garage_h)))
    rooms.append(Room("mudroom", (garage_w, 0, mud_w, garage_h * 0.5)))
    rooms.append(Room("laundry_room", (garage_w, garage_h * 0.5, mud_w, garage_h * 0.5)))
    rooms.append(Room("great_room", (garage_w + mud_w, 0, great_w - mud_w, great_h)))

    # Powder room carved from great_room's south-west corner.
    # (The great_room polygon still covers this area — that overlap is
    # tracked separately and intentionally untouched here.)
    pow_x = garage_w + mud_w
    pow_w = rng.uniform(1.4, 1.8)
    pow_h = rng.uniform(1.8, 2.2)
    rooms.append(Room("powder_room", (pow_x, great_h - pow_h, pow_w, pow_h)))

    # --- bedroom wing: one full-height row, left to right ---
    y0 = great_h
    x = 0.0

    # bed2 sits over the garage footprint, full wing height.
    rooms.append(Room("bedroom", (x, y0, garage_w, bed_wing_h),
                      doors=[("bathroom", 0.5, 0.9)]))
    x += garage_w

    # Shared bathroom, full wing height (jack-and-jill between bed2 and bed3).
    rooms.append(Room("bathroom", (x, y0, bath_w, bed_wing_h)))
    x += bath_w

    # Third bedroom, full wing height. Accesses both the bathroom and the hall.
    rooms.append(Room("bedroom", (x, y0, bed3_w, bed_wing_h),
                      doors=[("bathroom", 0.5, 0.9), ("hallway", 0.5, 0.9)]))
    x += bed3_w

    # Vertical hallway, full wing height. Bridges bed3 → master suite.
    rooms.append(Room("hallway", (x, y0, hall_w, bed_wing_h)))
    x += hall_w

    # Master suite: master (top 60%) + en-suite (bottom 40%) in the same
    # column, so their polygons tile the column without overlapping.
    mbr_x = x
    rooms.append(Room("master_bedroom", (mbr_x, y0, mbr_w, bed_wing_h * 0.6),
                      doors=[("hallway", 0.5, 0.9),
                             ("en_suite", 0.5, 0.8),
                             ("walk_in_closet", 0.7, 0.7)]))
    rooms.append(Room("en_suite", (mbr_x, y0 + bed_wing_h * 0.6, mbr_w, bed_wing_h * 0.4)))
    x += mbr_w

    # Walk-in closet, full wing height — avoids the old south-edge void
    # that used to sit under the 60%-tall walk-in.
    rooms.append(Room("walk_in_closet", (x, y0, wic_w, bed_wing_h)))

    plan = Plan(rooms=rooms, footprint=(w, h), exterior_door=("great_room", "S"))
    return plan


def colonial_compartmentalized(rng: random.Random) -> Plan:
    """Compartmentalized single-floor plan with separate living / dining /
    kitchen, no merged great room. Closer to traditional US colonial.

    Bands (y increases going south):
      front band (front_h):   dining | foyer | living    — full band height
      middle band (middle_h): kitchen | pantry/laundry | family_room — full
      back band (back_h):     master  | hallway | bathroom/bed | bed2
    """
    w = rng.uniform(12, 15)
    h = rng.uniform(10, 12)

    foyer_w = rng.uniform(1.8, 2.4)
    dining_w = rng.uniform(3.5, 4.2)
    living_w = w - foyer_w - dining_w

    # --- front band: dining | foyer | living (all full front_h) ---
    # Foyer extends the full band height so there is no void below it;
    # the original foyer_h < front_h left a slot that wasn't assigned.
    front_h = rng.uniform(3.8, 4.5)
    rooms: list[Room] = []
    rooms.append(Room("dining_room", (0, 0, dining_w, front_h)))
    rooms.append(Room("foyer", (dining_w, 0, foyer_w, front_h),
                      doors=[("dining_room", 0.5, 1.0), ("living_room", 0.5, 1.0)]))
    rooms.append(Room("living_room", (dining_w + foyer_w, 0, living_w, front_h)))

    # --- middle band: kitchen | (pantry over laundry) | family_room ---
    # This row must tile the full width. The old code placed only kitchen +
    # a small pantry and left the strip right of the pantry (and the slot
    # below it) unassigned, producing the kitchen-row void.
    middle_h = h - front_h - rng.uniform(3.8, 4.5)
    kitchen_w = rng.uniform(3.8, 4.5)
    pantry_w = rng.uniform(1.4, 1.8)
    family_w = w - kitchen_w - pantry_w
    if family_w < 2.5:
        deficit = 2.5 - family_w
        kitchen_w = max(3.0, kitchen_w - deficit)
        family_w = w - kitchen_w - pantry_w

    # Pantry takes the top of the middle column; laundry fills below so the
    # column tiles completely. Clamp pantry_h so both sub-rooms stay usable.
    pantry_h = min(rng.uniform(1.4, 1.8), middle_h * 0.55)
    laundry_h = middle_h - pantry_h

    rooms.append(Room("kitchen", (0, front_h, kitchen_w, middle_h),
                      doors=[("dining_room", 0.5, 1.0), ("pantry", 0.5, 0.8)]))
    rooms.append(Room("pantry", (kitchen_w, front_h, pantry_w, pantry_h)))
    rooms.append(Room("laundry_room", (kitchen_w, front_h + pantry_h,
                                       pantry_w, laundry_h),
                      doors=[("kitchen", 0.5, 0.8)]))
    rooms.append(Room("family_room", (kitchen_w + pantry_w, front_h,
                                       family_w, middle_h),
                      doors=[("living_room", 0.5, 1.2), ("hallway", 0.5, 1.0)]))

    # --- back band: bedroom wing (untouched — bug #3 tracked separately) ---
    y0 = front_h + middle_h
    back_h = h - y0
    mbr_w = rng.uniform(4.2, 5.0)
    bed2_w = rng.uniform(3.2, 3.8)
    bath_w = rng.uniform(2.2, 2.6)
    hall_w = w - mbr_w - bed2_w - bath_w
    if hall_w < 1.2:
        hall_w = 1.2
        bed2_w = w - mbr_w - bath_w - hall_w

    rooms.append(Room("master_bedroom", (0, y0, mbr_w, back_h),
                      doors=[("en_suite", 0.8, 0.8), ("hallway", 0.95, 0.9)]))
    rooms.append(Room("en_suite", (mbr_w - rng.uniform(1.8, 2.2), y0 + back_h - rng.uniform(2.0, 2.4),
                                    rng.uniform(1.8, 2.2), rng.uniform(2.0, 2.4))))
    rooms.append(Room("hallway", (mbr_w, y0, hall_w, back_h)))
    rooms.append(Room("bathroom", (mbr_w + hall_w, y0, bath_w, back_h * 0.5)))
    rooms.append(Room("bedroom", (mbr_w + hall_w, y0 + back_h * 0.5, bath_w, back_h * 0.5)))
    rooms.append(Room("bedroom", (mbr_w + hall_w + bath_w, y0, bed2_w, back_h)))

    return Plan(rooms=rooms, footprint=(w, h), exterior_door=("foyer", "N"))


TEMPLATES: list[Callable[[random.Random], Plan]] = [
    ranch_open_concept,
    colonial_compartmentalized,
]


# ---------- walls + openings from rooms ----------

def plan_to_schema(plan: Plan, rng: random.Random) -> dict:
    """Convert a template-produced Plan into a canonical floor plan dict."""
    walls: list[dict] = []
    seen: dict[tuple, int] = {}

    def add_wall(a, b, thickness=0.15) -> int:
        key = tuple(sorted([tuple(round(c, 3) for c in a), tuple(round(c, 3) for c in b)]))
        if key in seen:
            return seen[key]
        idx = len(walls)
        walls.append({"start": list(a), "end": list(b), "thickness": thickness})
        seen[key] = idx
        return idx

    # 1. One wall per room edge; shared edges deduplicate via `seen`.
    for r in plan.rooms:
        x, y, w, h = r.rect
        add_wall([x, y], [x + w, y])
        add_wall([x + w, y], [x + w, y + h])
        add_wall([x + w, y + h], [x, y + h])
        add_wall([x, y + h], [x, y])

    # 2. Doors: declared room-to-room connections.
    doors: list[dict] = []
    for r in plan.rooms:
        for (other_label, t, width) in r.doors:
            other = _find_room(plan, other_label, after=r)
            if other is None:
                continue
            pos = _shared_edge_point(r.rect, other.rect, t)
            if pos is None:
                continue
            wall_idx = _nearest_wall(pos, walls)
            doors.append({
                "position": [round(pos[0], 2), round(pos[1], 2)],
                "width": round(width, 2),
                "type": "hinged",
                "wall_index": wall_idx,
            })

    # 3. Exterior door on front of house.
    if plan.exterior_door:
        label, side = plan.exterior_door
        room = _find_room(plan, label)
        if room is not None:
            pos = _exterior_edge_midpoint(room.rect, side, plan.footprint)
            wall_idx = _nearest_wall(pos, walls)
            doors.append({
                "position": [round(pos[0], 2), round(pos[1], 2)],
                "width": 1.0,
                "type": "hinged",
                "wall_index": wall_idx,
            })

    # 4. Windows on exterior walls, roughly one per exterior room edge.
    windows: list[dict] = []
    fw, fh = plan.footprint
    for r in plan.rooms:
        if r.label in {"closet", "walk_in_closet", "pantry", "bathroom", "powder_room", "en_suite", "hallway", "foyer", "mudroom", "laundry_room", "garage"}:
            # These rooms typically don't get windows, or only small ones —
            # skip for now to keep windows on visible public rooms.
            if rng.random() > 0.2:
                continue
        for side in ("N", "S", "E", "W"):
            if not _edge_is_exterior(r.rect, side, fw, fh):
                continue
            if rng.random() > 0.55:
                continue
            cx, cy = _edge_midpoint(r.rect, side)
            wall_idx = _nearest_wall((cx, cy), walls)
            windows.append({
                "position": [round(cx, 2), round(cy, 2)],
                "width": round(rng.uniform(0.9, 1.6), 2),
                "wall_index": wall_idx,
            })

    rooms_out = [
        {
            "label": r.label,
            "polygon": [
                [r.rect[0], r.rect[1]],
                [r.rect[0] + r.rect[2], r.rect[1]],
                [r.rect[0] + r.rect[2], r.rect[1] + r.rect[3]],
                [r.rect[0], r.rect[1] + r.rect[3]],
            ],
            "area": round(r.rect[2] * r.rect[3], 2),
        }
        for r in plan.rooms
    ]

    return FloorPlan(
        walls=walls,
        doors=doors,
        windows=windows,
        rooms=rooms_out,
        scale={"pixels_per_meter": 40},
    ).to_dict()


def _find_room(plan: Plan, label: str, after: Room | None = None) -> Room | None:
    """Find a room with the given label. If `after` given, prefers the first
    matching room that comes after it in plan.rooms (so we don't re-match self)."""
    if after is not None:
        seen_self = False
        for r in plan.rooms:
            if r is after:
                seen_self = True
                continue
            if seen_self and r.label == label:
                return r
    for r in plan.rooms:
        if r is after:
            continue
        if r.label == label:
            return r
    return None


def _shared_edge_point(a, b, t):
    """Return a point on the shared edge between two rects, parameterized
    by t ∈ [0, 1] along the edge. Returns None if no shared edge."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    # vertical shared edge
    if abs(ax + aw - bx) < 0.02:
        y0, y1 = max(ay, by), min(ay + ah, by + bh)
        if y1 - y0 > 0.8:
            return (ax + aw, y0 + (y1 - y0) * t)
    if abs(bx + bw - ax) < 0.02:
        y0, y1 = max(ay, by), min(ay + ah, by + bh)
        if y1 - y0 > 0.8:
            return (ax, y0 + (y1 - y0) * t)
    # horizontal shared edge
    if abs(ay + ah - by) < 0.02:
        x0, x1 = max(ax, bx), min(ax + aw, bx + bw)
        if x1 - x0 > 0.8:
            return (x0 + (x1 - x0) * t, ay + ah)
    if abs(by + bh - ay) < 0.02:
        x0, x1 = max(ax, bx), min(ax + aw, bx + bw)
        if x1 - x0 > 0.8:
            return (x0 + (x1 - x0) * t, ay)
    return None


def _edge_midpoint(rect, side):
    x, y, w, h = rect
    if side == "N": return (x + w / 2, y)
    if side == "S": return (x + w / 2, y + h)
    if side == "W": return (x, y + h / 2)
    if side == "E": return (x + w, y + h / 2)
    return (x + w / 2, y + h / 2)


def _edge_is_exterior(rect, side, fw, fh, tol=0.02):
    x, y, w, h = rect
    if side == "N": return abs(y) < tol
    if side == "S": return abs(y + h - fh) < tol
    if side == "W": return abs(x) < tol
    if side == "E": return abs(x + w - fw) < tol
    return False


def _exterior_edge_midpoint(rect, side, footprint):
    x, y, w, h = rect
    fw, fh = footprint
    if side == "N": return (x + w / 2, 0)
    if side == "S": return (x + w / 2, fh)
    if side == "W": return (0, y + h / 2)
    if side == "E": return (fw, y + h / 2)
    return _edge_midpoint(rect, side)


def _nearest_wall(point, walls):
    if not walls:
        return -1
    px, py = point
    best_i, best_d = 0, float("inf")
    for i, w in enumerate(walls):
        sx, sy = w["start"]
        ex, ey = w["end"]
        dx, dy = ex - sx, ey - sy
        l2 = dx * dx + dy * dy
        if l2 < 1e-9:
            d = (px - sx) ** 2 + (py - sy) ** 2
        else:
            t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / l2))
            cx, cy = sx + t * dx, sy + t * dy
            d = (px - cx) ** 2 + (py - cy) ** 2
        if d < best_d:
            best_d, best_i = d, i
    return best_i


# ---------- rendering ----------

# Rough per-label fill color, to vaguely resemble MLS-style colored plans.
ROOM_COLORS = {
    "great_room": (245, 240, 225),
    "living_room": (245, 240, 225),
    "family_room": (245, 240, 225),
    "dining_room": (240, 230, 220),
    "kitchen": (235, 245, 225),
    "master_bedroom": (235, 230, 245),
    "bedroom": (230, 235, 245),
    "en_suite": (225, 235, 240),
    "bathroom": (225, 235, 240),
    "powder_room": (225, 235, 240),
    "walk_in_closet": (240, 232, 230),
    "closet": (240, 232, 230),
    "foyer": (245, 245, 240),
    "hallway": (248, 246, 240),
    "mudroom": (238, 232, 222),
    "laundry_room": (238, 232, 222),
    "pantry": (240, 238, 225),
    "garage": (225, 225, 225),
    "office": (238, 238, 230),
    "den": (238, 238, 230),
}


def render(plan_dict: dict, cfg: SynthConfig):
    from PIL import Image, ImageDraw, ImageFont
    size = cfg.image_size
    ppm = cfg.pixels_per_meter

    # Compute footprint from rooms so centering is accurate.
    xs = [p[0] for r in plan_dict["rooms"] for p in r["polygon"]]
    ys = [p[1] for r in plan_dict["rooms"] for p in r["polygon"]]
    fw = max(xs) - min(xs)
    fh = max(ys) - min(ys)
    off_x = (size - fw * ppm) / 2
    off_y = (size - fh * ppm) / 2

    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    def to_px(p):
        return (off_x + p[0] * ppm, off_y + p[1] * ppm)

    # Fill rooms.
    for r in plan_dict["rooms"]:
        color = ROOM_COLORS.get(r["label"], (245, 245, 245))
        draw.polygon([to_px(p) for p in r["polygon"]], fill=color)

    # Draw walls on top.
    wall_px = max(3, int(cfg.wall_thickness_m * ppm * 0.9))
    for w in plan_dict["walls"]:
        draw.line([to_px(w["start"]), to_px(w["end"])], fill=(25, 25, 25), width=wall_px)

    # Doors: short gap rendered as lighter line.
    door_px = max(4, wall_px + 1)
    for d in plan_dict["doors"]:
        cx, cy = to_px(d["position"])
        half = d["width"] * ppm / 2
        draw.line([(cx - half, cy), (cx + half, cy)], fill=(230, 225, 215), width=door_px)

    # Windows: blue-tinted short segments.
    for win in plan_dict["windows"]:
        cx, cy = to_px(win["position"])
        half = win["width"] * ppm / 2
        draw.line([(cx - half, cy), (cx + half, cy)], fill=(150, 185, 220), width=door_px)

    # Room labels.
    if cfg.draw_labels:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        for r in plan_dict["rooms"]:
            xs2 = [p[0] for p in r["polygon"]]
            ys2 = [p[1] for p in r["polygon"]]
            cx = sum(xs2) / len(xs2)
            cy = sum(ys2) / len(ys2)
            label = r["label"].replace("_", " ").title()
            draw.text(to_px((cx, cy)), label, fill=(60, 60, 60), font=font, anchor="mm")

    return img


# ---------- public API ----------

def generate_one(seed: int, cfg: SynthConfig | None = None):
    cfg = cfg or SynthConfig()
    rng = random.Random(seed)
    template = rng.choice(TEMPLATES)
    plan = template(rng)
    plan_dict = plan_to_schema(plan, rng)
    img = render(plan_dict, cfg)
    return img, plan_dict


def main():
    parser = argparse.ArgumentParser(description="Generate US-style synthetic floor plans")
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=900)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = SynthConfig(image_size=args.size)

    for i in range(args.count):
        img, fp = generate_one(args.seed + i, cfg)
        img.save(out / f"plan_{i:06d}.png")
        (out / f"plan_{i:06d}.json").write_text(serialize(fp))
        if (i + 1) % 200 == 0:
            print(f"[{i + 1}/{args.count}]")

    print(f"done: {args.count} samples in {out}")


if __name__ == "__main__":
    main()
