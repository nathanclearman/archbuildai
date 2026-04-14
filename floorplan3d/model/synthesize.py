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

    # --- top half: garage | (mudroom / laundry / powder_room stacked) | great_room ---
    # Powder room lives inside the service column so its polygon does not
    # overlap the great_room rectangle. The mud/laundry/powder stack tiles
    # the full great_h height exactly.
    mud_h = rng.uniform(1.8, 2.4)
    laundry_h = rng.uniform(1.6, 2.2)
    pow_h = great_h - mud_h - laundry_h
    if pow_h < 1.5:
        mud_h = great_h * 0.4
        laundry_h = great_h * 0.3
        pow_h = great_h - mud_h - laundry_h

    # Garage connects to the mudroom on its east wall — the standard US
    # attached-garage access path into the house. Without this door the
    # garage was sealed in every sample.
    rooms.append(Room("garage", (0, 0, garage_w, garage_h),
                      doors=[("mudroom", 0.5, 0.9)]))
    rooms.append(Room("mudroom", (garage_w, 0, mud_w, mud_h)))
    rooms.append(Room("laundry_room", (garage_w, mud_h, mud_w, laundry_h)))
    rooms.append(Room("powder_room",
                      (garage_w, mud_h + laundry_h, mud_w, pow_h),
                      doors=[("great_room", 0.5, 0.8)]))
    rooms.append(Room("great_room", (garage_w + mud_w, 0, great_w - mud_w, great_h)))

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

    # Front door on great_room's N edge (y=0) — which actually IS the
    # footprint exterior. The previous ("great_room", "S") pointed at the
    # great_room's south edge (y=great_h), which is interior (the bedroom
    # wing is below it), so the door rendered on the en-suite instead.
    plan = Plan(rooms=rooms, footprint=(w, h), exterior_door=("great_room", "N"))
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
    # h bumped: the previous (10, 12) range let middle_h shrink to ~1 m in
    # the worst case, squeezing laundry_h below the 0.8 m shared-edge
    # threshold used by plan_to_schema.
    h = rng.uniform(11, 12)

    foyer_w = rng.uniform(1.8, 2.4)
    dining_w = rng.uniform(3.5, 4.2)
    living_w = w - foyer_w - dining_w

    # --- band heights: front + middle + back = h, each with a viable min ---
    # Middle and back both need >= 3 m so the laundry column and bathroom
    # stack keep real shared-edge width.
    front_h = rng.uniform(3.8, 4.5)
    middle_h = rng.uniform(3.0, min(4.5, h - front_h - 3.5))
    back_h = h - front_h - middle_h

    # --- front band: dining | foyer | living (all full front_h) ---
    rooms: list[Room] = []
    rooms.append(Room("dining_room", (0, 0, dining_w, front_h)))
    rooms.append(Room("foyer", (dining_w, 0, foyer_w, front_h),
                      doors=[("dining_room", 0.5, 1.0), ("living_room", 0.5, 1.0)]))
    rooms.append(Room("living_room", (dining_w + foyer_w, 0, living_w, front_h)))

    # --- middle band: kitchen | (pantry over laundry) | family_room ---
    # Full-width tiling; no voids right of or below the pantry.
    kitchen_w = rng.uniform(3.8, 4.5)
    pantry_w = rng.uniform(1.4, 1.8)
    family_w = w - kitchen_w - pantry_w
    if family_w < 2.5:
        deficit = 2.5 - family_w
        kitchen_w = max(3.0, kitchen_w - deficit)
        family_w = w - kitchen_w - pantry_w

    # Clamp pantry_h so laundry_h stays >= 1.2 m for a robust shared edge
    # with the kitchen (shared-edge threshold inside plan_to_schema is 0.8).
    pantry_h = min(rng.uniform(1.4, 1.8), middle_h - 1.2)
    laundry_h = middle_h - pantry_h

    rooms.append(Room("kitchen", (0, front_h, kitchen_w, middle_h),
                      doors=[("dining_room", 0.5, 1.0), ("pantry", 0.5, 0.8)]))
    rooms.append(Room("pantry", (kitchen_w, front_h, pantry_w, pantry_h)))
    rooms.append(Room("laundry_room", (kitchen_w, front_h + pantry_h,
                                       pantry_w, laundry_h),
                      doors=[("kitchen", 0.5, 0.8)]))
    # Note: family_room <-> hallway is NOT declared as a door here. Their
    # shared edge width depends on w, mbr_w, hall_w, kitchen_w, and pantry_w,
    # and can drop below the 0.8 m threshold. Connectivity is still provided
    # via family_room <-> living_room <-> foyer and master <-> hallway.
    rooms.append(Room("family_room", (kitchen_w + pantry_w, front_h,
                                       family_w, middle_h),
                      doors=[("living_room", 0.5, 1.2)]))

    # --- back band: master suite | hallway | bath over bedroom | bed2 ---
    y0 = front_h + middle_h
    mbr_w = rng.uniform(4.2, 5.0)
    bed2_w = rng.uniform(3.2, 3.8)
    bath_w = rng.uniform(2.2, 2.6)
    hall_w = w - mbr_w - bed2_w - bath_w
    if hall_w < 1.2:
        hall_w = 1.2
        bed2_w = w - mbr_w - bath_w - hall_w

    # Master column tiles cleanly: master_bedroom on top, en-suite and
    # walk-in closet share the bottom sub-row. The old code planted an
    # en_suite rectangle inside the master_bedroom rectangle — both
    # polygons then claimed the same floor area, teaching the model
    # that overlapping labels are legal.
    ens_w = rng.uniform(1.8, 2.2)
    ens_h = min(rng.uniform(2.0, 2.4), back_h * 0.55)
    mbr_h = back_h - ens_h
    wic_w = mbr_w - ens_w

    rooms.append(Room("master_bedroom", (0, y0, mbr_w, mbr_h),
                      doors=[("en_suite", 0.3, 0.8),
                             ("walk_in_closet", 0.7, 0.7),
                             ("hallway", 0.95, 0.9)]))
    rooms.append(Room("en_suite", (0, y0 + mbr_h, ens_w, ens_h)))
    rooms.append(Room("walk_in_closet", (ens_w, y0 + mbr_h, wic_w, ens_h)))

    rooms.append(Room("hallway", (mbr_w, y0, hall_w, back_h)))
    rooms.append(Room("bathroom", (mbr_w + hall_w, y0, bath_w, back_h * 0.5)))
    rooms.append(Room("bedroom", (mbr_w + hall_w, y0 + back_h * 0.5, bath_w, back_h * 0.5)))
    rooms.append(Room("bedroom", (mbr_w + hall_w + bath_w, y0, bed2_w, back_h)))

    return Plan(rooms=rooms, footprint=(w, h), exterior_door=("foyer", "N"))


def split_bedroom_ranch(rng: random.Random) -> Plan:
    """Wide single-floor ranch with the master suite and secondary bedrooms
    on opposite ends of the house, a vertical hallway bridging the
    secondary wing to the central great room, and an attached garage
    directly above the master suite.

    Layout (y south; columns left-to-right: sec_wing | hall | great_room | mbr_col):
      +------+---+------------------+--------+
      | bed2 |   |                  | garage |
      +------+   |                  |        |
      | bath |hall|  great_room     +--------+
      +------+   |                  | master |
      | bed3 |   |                  +--+-----+
      |      |   |                  |en| wic |
      +------+---+------------------+--+-----+

    The master column stacks: garage (top) / master_bedroom (middle) /
    en_suite + walk_in_closet (bottom). The great_room is a single big
    open rectangle (kitchen + dining + living merged) with openings to
    the garage and hall on its sides.
    """
    w = rng.uniform(19, 23)
    h = rng.uniform(12, 14)

    sec_w = rng.uniform(4.2, 5.0)
    hall_w = rng.uniform(1.2, 1.6)
    mbr_col_w = rng.uniform(5.2, 6.2)
    great_w = w - sec_w - hall_w - mbr_col_w
    if great_w < 5.0:
        deficit = 5.0 - great_w
        mbr_col_w = max(4.8, mbr_col_w - deficit * 0.6)
        sec_w = max(4.0, sec_w - deficit * 0.4)
        great_w = w - sec_w - hall_w - mbr_col_w

    # Secondary wing stacked: bed | bath | bed
    sec_bed2_h = rng.uniform(3.8, 4.5)
    sec_bath_h = rng.uniform(2.5, 3.0)
    sec_bed3_h = h - sec_bed2_h - sec_bath_h
    if sec_bed3_h < 3.2:
        scale = (h - 3.2) / (sec_bed2_h + sec_bath_h)
        sec_bed2_h *= scale
        sec_bath_h *= scale
        sec_bed3_h = 3.2

    # Master column sub-heights: garage top, master middle, en-suite+wic bottom
    garage_h = rng.uniform(5.8, 6.4)
    ens_h = min(rng.uniform(2.2, 2.6), (h - garage_h) * 0.45)
    mbr_h = h - garage_h - ens_h

    ens_w = rng.uniform(2.2, 2.6)
    wic_w = mbr_col_w - ens_w

    rooms: list[Room] = []

    # Secondary bedroom wing (left column)
    rooms.append(Room("bedroom", (0, 0, sec_w, sec_bed2_h),
                      doors=[("hallway", 0.5, 0.9)]))
    rooms.append(Room("bathroom", (0, sec_bed2_h, sec_w, sec_bath_h),
                      doors=[("hallway", 0.5, 0.8)]))
    rooms.append(Room("bedroom", (0, sec_bed2_h + sec_bath_h, sec_w, sec_bed3_h),
                      doors=[("hallway", 0.5, 0.9)]))

    # Vertical hallway (full height)
    rooms.append(Room("hallway", (sec_w, 0, hall_w, h),
                      doors=[("great_room", 0.5, 1.2)]))

    # Great room (merged kitchen + dining + living)
    rooms.append(Room("great_room", (sec_w + hall_w, 0, great_w, h)))

    # Master column (right): garage / master / en-suite + wic
    mbr_x = sec_w + hall_w + great_w
    rooms.append(Room("garage", (mbr_x, 0, mbr_col_w, garage_h),
                      doors=[("great_room", 0.5, 0.9)]))
    rooms.append(Room("master_bedroom", (mbr_x, garage_h, mbr_col_w, mbr_h),
                      doors=[("great_room", 0.5, 0.9),
                             ("en_suite", 0.3, 0.8),
                             ("walk_in_closet", 0.7, 0.7)]))
    rooms.append(Room("en_suite", (mbr_x, garage_h + mbr_h, ens_w, ens_h)))
    rooms.append(Room("walk_in_closet", (mbr_x + ens_w, garage_h + mbr_h, wic_w, ens_h)))

    return Plan(rooms=rooms, footprint=(w, h), exterior_door=("great_room", "N"))


def l_shape_ranch(rng: random.Random) -> Plan:
    """Single-floor L-shape ranch. The footprint's NW corner is a cut-out
    (exterior), so the house outline has a concave corner. Bedroom wing
    sits in the top-right (north of the cut-out), the main living band
    spans the full width at the bottom.

    Layout (y south; * marks exterior cut-out):
      +--------+---+---+----+
      |  *     |bed|bth|bed |   (wing at top right, above the cut-out)
      +--------+---+---+----+
      | foyer  |   hallway   |
      +--------+-------------+
      |kit|din| living |mbr  |
      +---+---+--------+-----+
      |            |ens| wic |
      +---+---+--------+-----+
    """
    w = rng.uniform(14, 17)
    h = rng.uniform(12, 14)

    cut_w = rng.uniform(4.5, 5.5)
    cut_h = rng.uniform(3.6, 4.5)

    wing_w = w - cut_w
    wing_h = cut_h

    # Wing columns: bed | bath | bed (three equal-ish sub-columns)
    wing_bath_w = rng.uniform(2.2, 2.5)
    wing_bed_each = (wing_w - wing_bath_w) / 2
    if wing_bed_each < 2.8:
        wing_bath_w = max(1.8, wing_w - 5.6)
        wing_bed_each = (wing_w - wing_bath_w) / 2

    # Below-cut strip heights: hallway sub-strip + bottom band
    hall_h = rng.uniform(1.3, 1.6)
    bot_y = cut_h + hall_h
    bot_h = h - bot_y
    if bot_h < 4.5:
        hall_h = max(1.2, h - cut_h - 4.5)
        bot_y = cut_h + hall_h
        bot_h = h - bot_y

    # Bottom columns: kitchen | dining | living | master
    kit_w = rng.uniform(3.2, 3.8)
    din_w = rng.uniform(2.6, 3.2)
    mbr_w = rng.uniform(4.2, 5.2)
    liv_w = w - kit_w - din_w - mbr_w
    if liv_w < 3.0:
        deficit = 3.0 - liv_w
        mbr_w = max(4.0, mbr_w - deficit * 0.5)
        kit_w = max(3.0, kit_w - deficit * 0.5)
        liv_w = w - kit_w - din_w - mbr_w

    ens_w = rng.uniform(1.8, 2.2)
    ens_h = min(rng.uniform(2.0, 2.4), bot_h * 0.4)
    mbr_h_local = bot_h - ens_h
    wic_w_local = mbr_w - ens_w

    rooms: list[Room] = []

    # Wing rooms (top-right, north of the cut-out)
    rooms.append(Room("bedroom", (cut_w, 0, wing_bed_each, wing_h),
                      doors=[("hallway", 0.2, 0.9)]))
    rooms.append(Room("bathroom", (cut_w + wing_bed_each, 0, wing_bath_w, wing_h),
                      doors=[("hallway", 0.5, 0.8)]))
    rooms.append(Room("bedroom", (cut_w + wing_bed_each + wing_bath_w, 0, wing_bed_each, wing_h),
                      doors=[("hallway", 0.8, 0.9)]))

    # Hallway strip + foyer together span the full width at y=cut_h
    rooms.append(Room("foyer", (0, cut_h, cut_w, hall_h),
                      doors=[("hallway", 0.5, 1.0), ("kitchen", 0.5, 1.0)]))
    rooms.append(Room("hallway", (cut_w, cut_h, w - cut_w, hall_h)))

    # Bottom band: kitchen | dining | living | master column
    rooms.append(Room("kitchen", (0, bot_y, kit_w, bot_h),
                      doors=[("dining_room", 0.5, 1.2)]))
    # Dining's shared edge with the hallway is the sliver formed by the
    # kit_w / cut_w offset — can fall below a usable door width, so no
    # direct dining <-> hallway door is declared. Access is via kitchen
    # and living_room instead.
    rooms.append(Room("dining_room", (kit_w, bot_y, din_w, bot_h),
                      doors=[("living_room", 0.5, 1.2)]))
    rooms.append(Room("living_room", (kit_w + din_w, bot_y, liv_w, bot_h),
                      doors=[("hallway", 0.5, 1.2)]))

    mbr_x = kit_w + din_w + liv_w
    rooms.append(Room("master_bedroom", (mbr_x, bot_y, mbr_w, mbr_h_local),
                      doors=[("living_room", 0.9, 0.9),
                             ("en_suite", 0.3, 0.8),
                             ("walk_in_closet", 0.7, 0.7)]))
    rooms.append(Room("en_suite", (mbr_x, bot_y + mbr_h_local, ens_w, ens_h)))
    rooms.append(Room("walk_in_closet", (mbr_x + ens_w, bot_y + mbr_h_local, wic_w_local, ens_h)))

    # Exterior door on the foyer's N edge — that edge borders the cut-out
    # (exterior), which is the L's concave corner. The shape makes it
    # read as a porch under the wing's south overhang.
    return Plan(rooms=rooms, footprint=(w, h), exterior_door=("foyer", "N"))


def narrow_townhouse(rng: random.Random) -> Plan:
    """Deep narrow single-story house (urban shotgun / rowhouse ground
    plan). One primary column of rooms with a double-loaded corridor
    on the east side that everything opens onto. Five public/private
    rooms stacked — sum-of-minimum heights fits inside h without
    requiring the negative-height salvage path the earlier version had.

    Layout (y south):
      +----------+---+
      |  foyer   | h |
      +----------+ a |
      | living   | l |
      +----------+ l |
      | kitchen  | w |
      +----------+ a |
      | bathroom | y |
      +----------+   |
      | bedroom  |   |
      +----------+---+
    """
    w = rng.uniform(6.8, 8.5)
    h = rng.uniform(14, 17)

    hall_w = rng.uniform(1.2, 1.5)
    rooms_w = w - hall_w

    # Heights sampled so their sum fits h. We sample relative weights and
    # scale them to exactly hit h, which keeps proportions natural across
    # the h range without risking negative heights.
    raw = [
        rng.uniform(1.8, 2.3),   # foyer
        rng.uniform(3.8, 4.8),   # living_room
        rng.uniform(3.0, 3.8),   # kitchen
        rng.uniform(2.0, 2.6),   # bathroom
        rng.uniform(3.0, 3.8),   # bedroom
    ]
    scale = h / sum(raw)
    foyer_h, living_h, kitchen_h, bath_h, bedroom_h = (v * scale for v in raw)

    rooms: list[Room] = []
    y = 0.0
    rooms.append(Room("foyer", (0, y, rooms_w, foyer_h),
                      doors=[("hallway", 0.5, 0.9)]))
    y += foyer_h
    rooms.append(Room("living_room", (0, y, rooms_w, living_h),
                      doors=[("hallway", 0.5, 1.2)]))
    y += living_h
    rooms.append(Room("kitchen", (0, y, rooms_w, kitchen_h),
                      doors=[("hallway", 0.5, 1.0)]))
    y += kitchen_h
    rooms.append(Room("bathroom", (0, y, rooms_w, bath_h),
                      doors=[("hallway", 0.5, 0.8)]))
    y += bath_h
    rooms.append(Room("bedroom", (0, y, rooms_w, bedroom_h),
                      doors=[("hallway", 0.5, 0.9)]))

    # East hallway spans the full depth
    rooms.append(Room("hallway", (rooms_w, 0, hall_w, h)))

    return Plan(rooms=rooms, footprint=(w, h), exterior_door=("foyer", "N"))


TEMPLATES: list[Callable[[random.Random], Plan]] = [
    ranch_open_concept,
    colonial_compartmentalized,
    split_bedroom_ranch,
    l_shape_ranch,
    narrow_townhouse,
]


# ---------- walls + openings from rooms ----------

def plan_to_schema(plan: Plan, rng: random.Random) -> dict:
    """Convert a template-produced Plan into a canonical floor plan dict."""
    walls = _build_walls(plan)

    # 2. Doors: declared room-to-room connections.
    # Each door is snapped onto the actual wall segment it belongs to: the
    # wall graph is now subdivided at T-junctions, so the nearest wall may
    # be shorter than the shared edge. We pick the longest wall segment
    # that lies on the shared edge and clamp the door so it fits entirely
    # within that segment (avoids door_width overhangs past wall endpoints).
    doors: list[dict] = []
    for r in plan.rooms:
        for (other_label, t, width) in r.doors:
            other = _find_room(plan, other_label, after=r)
            if other is None:
                continue
            pos = _shared_edge_point(r.rect, other.rect, t)
            if pos is None:
                continue
            snapped = _snap_door_to_wall(pos, width, walls)
            if snapped is None:
                continue
            wall_idx, (px, py) = snapped
            doors.append({
                "position": [round(px, 2), round(py, 2)],
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
            ext_width = 1.0
            snapped = _snap_door_to_wall(pos, ext_width, walls)
            if snapped is not None:
                wall_idx, (px, py) = snapped
                doors.append({
                    "position": [round(px, 2), round(py, 2)],
                    "width": ext_width,
                    "type": "hinged",
                    "wall_index": wall_idx,
                })

    # 4. Windows on exterior walls. Previously one window per exterior edge,
    # always at edge midpoint — which (a) collided with the exterior door
    # when both picked the same midpoint and (b) produced sparse
    # single-window walls on long public rooms. Now windows are spread
    # along each exterior edge (~one per 3 m), clamped to fit their wall
    # segment, and skipped where they would overlap an existing door.
    windows: list[dict] = []
    fw, fh = plan.footprint
    service_rooms = {
        "closet", "walk_in_closet", "pantry", "bathroom", "powder_room",
        "en_suite", "hallway", "foyer", "mudroom", "laundry_room", "garage",
    }
    door_by_wall: dict[int, list[dict]] = {}
    for d_ in doors:
        door_by_wall.setdefault(d_["wall_index"], []).append(d_)

    def _clashes_with_door(wall_idx: int, wx: float, wy: float, width: float) -> bool:
        for dd in door_by_wall.get(wall_idx, ()):
            dx_, dy_ = dd["position"]
            dist = ((dx_ - wx) ** 2 + (dy_ - wy) ** 2) ** 0.5
            if dist < (dd["width"] + width) / 2 + 0.3:
                return True
        return False

    for r in plan.rooms:
        service = r.label in service_rooms
        x, y, w, h = r.rect
        for side in ("N", "S", "E", "W"):
            if not _edge_is_exterior(r.rect, side, fw, fh):
                continue
            if side == "N":
                p0, p1 = (x, y), (x + w, y)
            elif side == "S":
                p0, p1 = (x, y + h), (x + w, y + h)
            elif side == "W":
                p0, p1 = (x, y), (x, y + h)
            else:
                p0, p1 = (x + w, y), (x + w, y + h)
            edge_len = ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5
            n_candidates = max(1, int(edge_len / 3.0))
            per_window_prob = 0.1 if service else 0.55
            for k in range(n_candidates):
                if rng.random() > per_window_prob:
                    continue
                t = (k + 0.5) / n_candidates
                px = p0[0] + t * (p1[0] - p0[0])
                py = p0[1] + t * (p1[1] - p0[1])
                win_width = round(rng.uniform(0.9, 1.6), 2)
                snapped = _snap_door_to_wall((px, py), win_width, walls)
                if snapped is None:
                    continue
                wall_idx, (wx, wy) = snapped
                if _clashes_with_door(wall_idx, wx, wy, win_width):
                    continue
                windows.append({
                    "position": [round(wx, 2), round(wy, 2)],
                    "width": win_width,
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


def _build_walls(plan: Plan, thickness: float = 0.15, coord_precision: int = 3) -> list[dict]:
    """Generate a non-overlapping wall graph from the room rectangles.

    Every rectangle edge is treated as a 1-D interval on its line (horizontal
    or vertical). Intervals on the same line are unioned, then split at every
    other rectangle's corner that lies inside the union. The output is the
    minimal set of wall segments such that no two walls are collinear and
    overlapping.

    The naive "one wall per room edge with endpoint-pair dedup" approach
    produced 2-3x overcounts wherever one room's edge spanned two adjacent
    rooms on the other side (a T-junction) — each partial edge plus the
    full spanning edge all landed in the wall list as independent entries,
    teaching downstream models to emit redundant walls.
    """
    # Collect {line_coord: [(start, end), ...]} per horizontal / vertical line.
    h_segments: dict[float, list[tuple[float, float]]] = {}
    v_segments: dict[float, list[tuple[float, float]]] = {}
    # Split points on each line (corners of every rectangle).
    h_splits: dict[float, set[float]] = {}
    v_splits: dict[float, set[float]] = {}

    def _round(c: float) -> float:
        return round(c, coord_precision)

    for r in plan.rooms:
        x, y, w, h = r.rect
        x0, x1 = _round(x), _round(x + w)
        y0, y1 = _round(y), _round(y + h)

        h_segments.setdefault(y0, []).append((x0, x1))
        h_segments.setdefault(y1, []).append((x0, x1))
        v_segments.setdefault(x0, []).append((y0, y1))
        v_segments.setdefault(x1, []).append((y0, y1))

        for yy in (y0, y1):
            h_splits.setdefault(yy, set()).update({x0, x1})
        for xx in (x0, x1):
            v_splits.setdefault(xx, set()).update({y0, y1})

    def _union_1d(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if not intervals:
            return []
        segs = sorted((min(a, b), max(a, b)) for a, b in intervals)
        merged: list[list[float]] = [list(segs[0])]
        for s, e in segs[1:]:
            if s <= merged[-1][1] + 1e-6:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        return [(a, b) for a, b in merged]

    def _subdivide(s: float, e: float, splits: list[float],
                   eps: float = 0.05) -> list[float]:
        """Return a monotonically-increasing list of cut points from s to e
        with no two adjacent values within `eps`. The eps filter drops
        sub-segments shorter than 5 cm — narrower than any wall thickness
        and always an artefact of near-coincident rectangle corners
        rather than a real wall."""
        raw = [s] + [p for p in splits if s + eps < p < e - eps] + [e]
        raw.sort()
        out = [raw[0]]
        for p in raw[1:]:
            if p - out[-1] > eps:
                out.append(p)
        if out[-1] < e - eps / 2:
            out.append(e)
        elif out[-1] != e:
            out[-1] = e
        return out

    walls: list[dict] = []

    for y, segs in h_segments.items():
        splits = sorted(h_splits.get(y, set()))
        for s, e in _union_1d(segs):
            pts = _subdivide(s, e, splits)
            for i in range(len(pts) - 1):
                walls.append({
                    "start": [pts[i], y],
                    "end": [pts[i + 1], y],
                    "thickness": thickness,
                })

    for x, segs in v_segments.items():
        splits = sorted(v_splits.get(x, set()))
        for s, e in _union_1d(segs):
            pts = _subdivide(s, e, splits)
            for i in range(len(pts) - 1):
                walls.append({
                    "start": [x, pts[i]],
                    "end": [x, pts[i + 1]],
                    "thickness": thickness,
                })

    return walls


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
    # Use the room's OWN edge midpoint. The old implementation assumed every
    # exterior-door room touched the footprint on the given side and wrote
    # the door at the footprint boundary — so when a template pointed at a
    # room whose edge was actually interior (e.g. ranch great_room, "S"),
    # the door landed on whatever room did touch the footprint south edge
    # (in the ranch case, the master en-suite). Using the room's own edge
    # keeps the door on that room's wall, or reveals the template bug
    # when the room's edge isn't at the footprint.
    return _edge_midpoint(rect, side)


def _snap_door_to_wall(point, door_width, walls, tol=0.02):
    """Find the best wall segment to host a door and clamp the door to fit.

    Picks the longest wall segment whose line contains `point` and is at
    least `door_width` long. Returns (wall_index, (px, py)) with the door
    position clamped so the door extends only within that segment, or
    None if no wall can host it.
    """
    px, py = point
    best: tuple[int, tuple[float, float], float] | None = None  # (idx, pos, length)

    for i, w in enumerate(walls):
        sx, sy = w["start"]
        ex, ey = w["end"]
        is_horizontal = abs(sy - ey) < tol
        is_vertical = abs(sx - ex) < tol
        if is_horizontal:
            # Point must lie on this line within tol and within x-range
            if abs(py - sy) > tol:
                continue
            xlo, xhi = (sx, ex) if sx <= ex else (ex, sx)
            if px < xlo - tol or px > xhi + tol:
                continue
            wall_len = xhi - xlo
            if wall_len < door_width - tol:
                continue
            # Clamp so the door fits
            half = door_width / 2
            new_x = max(xlo + half, min(xhi - half, px))
            if best is None or wall_len > best[2]:
                best = (i, (new_x, sy), wall_len)
        elif is_vertical:
            if abs(px - sx) > tol:
                continue
            ylo, yhi = (sy, ey) if sy <= ey else (ey, sy)
            if py < ylo - tol or py > yhi + tol:
                continue
            wall_len = yhi - ylo
            if wall_len < door_width - tol:
                continue
            half = door_width / 2
            new_y = max(ylo + half, min(yhi - half, py))
            if best is None or wall_len > best[2]:
                best = (i, (sx, new_y), wall_len)

    if best is None:
        return None
    return best[0], best[1]


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

    # Doors and windows: gap lines rendered ALONG their wall's direction.
    # The old code always drew a horizontal gap, so openings on vertical
    # walls showed as bars crossing the wall instead of gaps in it.
    door_px = max(4, wall_px + 1)

    def _wall_dir(wall):
        sx, sy = wall["start"]
        ex, ey = wall["end"]
        length = ((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5
        if length < 1e-6:
            return (1.0, 0.0)
        return ((ex - sx) / length, (ey - sy) / length)

    def _pick_swing_side(door_pos_m, wall, rooms):
        """Return a (dx, dy) unit vector perpendicular to the wall pointing
        into the room the door swings into. Interior side when unambiguous,
        smaller-room side for interior doors between two rooms, caller
        fallback if the point sits outside every room."""
        wdx, wdy = _wall_dir(wall)
        perp_a = (-wdy, wdx)
        perp_b = (wdy, -wdx)
        px_m, py_m = door_pos_m
        eps = 0.3

        def _room_areas_at(pt):
            x_, y_ = pt
            out = []
            for r in rooms:
                xs_ = [p[0] for p in r["polygon"]]
                ys_ = [p[1] for p in r["polygon"]]
                if min(xs_) <= x_ <= max(xs_) and min(ys_) <= y_ <= max(ys_):
                    out.append((max(xs_) - min(xs_)) * (max(ys_) - min(ys_)))
            return out

        a_areas = _room_areas_at((px_m + perp_a[0] * eps, py_m + perp_a[1] * eps))
        b_areas = _room_areas_at((px_m + perp_b[0] * eps, py_m + perp_b[1] * eps))
        if a_areas and not b_areas:
            return perp_a
        if b_areas and not a_areas:
            return perp_b
        if a_areas and b_areas:
            # Both sides interior: door swings into the smaller room (the
            # one that would be awkward to swing out of).
            return perp_a if min(a_areas) <= min(b_areas) else perp_b
        return perp_a

    def _render_door_arc(door, wall):
        """Draw the door as a quarter-arc + leaf line instead of a flat
        gap bar. PIL screen-y increases downward, same as world-y, so
        angles computed on world-space vectors map directly to the
        draw.arc / atan2 convention."""
        import math

        wdx, wdy = _wall_dir(wall)
        swing_x, swing_y = _pick_swing_side(door["position"], wall, plan_dict["rooms"])
        width_m = door["width"]
        half_m = width_m / 2
        px_m, py_m = door["position"]

        # Hinge end of the opening: alternate by wall_index for variety.
        sign = 1 if (door["wall_index"] % 2) == 0 else -1
        hinge_m = (px_m + wdx * half_m * sign, py_m + wdy * half_m * sign)
        free_m = (px_m - wdx * half_m * sign, py_m - wdy * half_m * sign)
        free_open_m = (hinge_m[0] + swing_x * width_m,
                       hinge_m[1] + swing_y * width_m)
        hinge_px = to_px(hinge_m)

        r_px = width_m * ppm
        start_ang = math.atan2(-wdy * sign, -wdx * sign)   # hinge -> closed free-end
        end_ang = math.atan2(swing_y, swing_x)             # hinge -> open free-end

        # Take the shortest 90-ish-degree path.
        delta = end_ang - start_ang
        while delta > math.pi:
            delta -= 2 * math.pi
        while delta < -math.pi:
            delta += 2 * math.pi

        # Sample the arc as a polyline (PIL's arc angle conventions are
        # annoying to reason about; trig we already trust).
        steps = 14
        arc_pts = []
        for i in range(steps + 1):
            ang = start_ang + (i / steps) * delta
            arc_pts.append((hinge_px[0] + r_px * math.cos(ang),
                            hinge_px[1] + r_px * math.sin(ang)))
        draw.line(arc_pts, fill=(55, 55, 55), width=1)
        # Door leaf from the hinge to the opened free-end position.
        draw.line([hinge_px, to_px(free_open_m)], fill=(55, 55, 55), width=2)

    walls = plan_dict["walls"]
    for d in plan_dict["doors"]:
        wall = walls[d["wall_index"]]
        wdx, wdy = _wall_dir(wall)
        cx, cy = to_px(d["position"])
        half = d["width"] * ppm / 2
        # Erase the wall at the door opening with a white gap; the arc +
        # leaf are drawn on top.
        draw.line([(cx - wdx * half, cy - wdy * half),
                   (cx + wdx * half, cy + wdy * half)],
                  fill=(255, 255, 255), width=door_px)
        _render_door_arc(d, wall)

    for win in plan_dict["windows"]:
        wall = walls[win["wall_index"]]
        wdx, wdy = _wall_dir(wall)
        perp_x, perp_y = -wdy, wdx
        cx, cy = to_px(win["position"])
        half = win["width"] * ppm / 2
        # Erase the wall at the window opening.
        draw.line([(cx - wdx * half, cy - wdy * half),
                   (cx + wdx * half, cy + wdy * half)],
                  fill=(255, 255, 255), width=door_px)
        # Two thin parallel lines flanking the wall axis — the classic
        # floor-plan window glyph. Offset is about one wall-thickness.
        offset = max(2.0, (wall_px - 1) / 2)
        for s in (-1.0, 1.0):
            ox = perp_x * offset * s
            oy = perp_y * offset * s
            draw.line([(cx - wdx * half + ox, cy - wdy * half + oy),
                       (cx + wdx * half + ox, cy + wdy * half + oy)],
                      fill=(55, 90, 140), width=1)
        # Thin centerline between the two parallel lines for a sash look.
        draw.line([(cx - wdx * half, cy - wdy * half),
                   (cx + wdx * half, cy + wdy * half)],
                  fill=(120, 160, 200), width=1)

    # Room labels. Each label is wrapped and sized to fit inside its room
    # rectangle — the old single-line draw clipped multi-word labels
    # (e.g. "Master Bedroom", "Walk In Closet") on narrow rooms, which
    # taught the downstream VLM to emit truncated strings.
    if cfg.draw_labels:
        for r in plan_dict["rooms"]:
            xs2 = [p[0] for p in r["polygon"]]
            ys2 = [p[1] for p in r["polygon"]]
            cx = sum(xs2) / len(xs2)
            cy = sum(ys2) / len(ys2)
            label = r["label"].replace("_", " ").title()
            room_w_px = (max(xs2) - min(xs2)) * ppm
            room_h_px = (max(ys2) - min(ys2)) * ppm
            _draw_label_fitted(
                draw, to_px((cx, cy)), label,
                room_w_px=room_w_px, room_h_px=room_h_px,
            )

    return img


# ---------- label fitting ----------

_FONT_CANDIDATES = (
    "DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "Arial.ttf",
)
_FONT_CACHE: dict[int, object] = {}


def _get_font(size: int):
    """Return a scalable font at the requested pixel size, cached."""
    from PIL import ImageFont
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    font = None
    for name in _FONT_CANDIDATES:
        try:
            font = ImageFont.truetype(name, size)
            break
        except (OSError, IOError):
            continue
    if font is None:
        # Last resort — PIL's bundled default. Size may be ignored on old
        # Pillow, but labels still render without clipping on small rooms
        # because the fitter will pick the layout that fits.
        try:
            font = ImageFont.load_default(size=size)
        except TypeError:
            font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def _wrap_words(words: list[str], n_lines: int) -> list[str]:
    """Balanced greedy wrap of `words` into exactly `n_lines` lines."""
    if n_lines <= 1 or len(words) <= 1:
        return [" ".join(words)]
    n_lines = min(n_lines, len(words))
    lines = []
    start = 0
    remaining = len(words)
    for i in range(n_lines):
        count = max(1, round(remaining / (n_lines - i)))
        lines.append(" ".join(words[start:start + count]))
        start += count
        remaining -= count
    return lines


def _measure(draw, font, text: str):
    """Return (width, height, y_offset) for a single-line string."""
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font, anchor="lt")
    return x1 - x0, y1 - y0, y0


def _draw_label_fitted(draw, centroid_px, text: str,
                       room_w_px: float, room_h_px: float,
                       margin_px: int = 6) -> None:
    """Draw `text` at `centroid_px` wrapped and shrunk to fit the room box."""
    max_w = max(room_w_px - 2 * margin_px, 20)
    max_h = max(room_h_px - 2 * margin_px, 12)
    words = text.split()
    line_gap = 2

    # Try font sizes from large to small. For each size, try 1-, 2-, 3-line
    # wraps and pick the first that fits horizontally and vertically.
    for font_size in (18, 15, 13, 11, 9, 8, 7):
        font = _get_font(font_size)
        for n_lines in range(1, min(len(words), 3) + 1):
            lines = _wrap_words(words, n_lines)
            widths, heights = zip(*((_measure(draw, font, ln)[0],
                                     _measure(draw, font, ln)[1]) for ln in lines))
            line_h = max(heights)
            total_w = max(widths)
            total_h = line_h * len(lines) + line_gap * (len(lines) - 1)
            if total_w <= max_w and total_h <= max_h:
                cx, cy = centroid_px
                top_y = cy - total_h / 2 + line_h / 2
                for i, ln in enumerate(lines):
                    draw.text((cx, top_y + i * (line_h + line_gap)), ln,
                              fill=(60, 60, 60), font=font, anchor="mm")
                return

    # Nothing fit cleanly — draw the text at the smallest tried size. The
    # label may still overflow the room on truly tiny rooms, but picking
    # the smallest-size wrap keeps the tail characters visible instead of
    # mid-word-clipped.
    font = _get_font(7)
    lines = _wrap_words(words, min(len(words), 3))
    cx, cy = centroid_px
    _, line_h, _ = _measure(draw, font, lines[0])
    total_h = line_h * len(lines) + line_gap * (len(lines) - 1)
    top_y = cy - total_h / 2 + line_h / 2
    for i, ln in enumerate(lines):
        draw.text((cx, top_y + i * (line_h + line_gap)), ln,
                  fill=(60, 60, 60), font=font, anchor="mm")


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
