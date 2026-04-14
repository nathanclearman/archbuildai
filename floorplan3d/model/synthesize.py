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
import colorsys
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, ImageFont

from schema import FloorPlan, serialize  # type: ignore


# Tunables that used to live as scattered magic numbers in render() and the
# wall builder. Promoting them to named constants both documents the value
# and makes it discoverable when tuning.
DEFAULT_ROOM_COLOR: tuple[int, int, int] = (245, 245, 245)
WALL_DEDUP_EPS_M: float = 0.05      # collapse sub-walls shorter than this
WALL_COORD_PRECISION: int = 3
ARC_POLYLINE_STEPS: int = 14        # quarter-arc rendered as N+1 points
SWING_PROBE_M: float = 0.30         # offset used to detect "interior side"
DOOR_GAP_PADDING_PX: int = 1        # extra px on the door gap vs wall_px
LABEL_LINE_GAP_PX: int = 2
LABEL_FONT_LADDER: tuple[int, ...] = (18, 15, 13, 11, 9, 8, 7)
LABEL_MIN_FONT_PX: int = 7
LABEL_MARGIN_PX: int = 6


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
    """Convert a template-produced Plan into a canonical floor plan dict.

    Each door records `swing_into` — a unit (dx, dy) vector pointing
    into the room the door swings into. We know which two rooms a door
    connects at generation time (the declaration is `(other_label, t,
    width)`), so we can compute swing direction once with full
    information instead of inferring it from polygon containment at
    render time. Exterior doors swing into the room they belong to.
    """
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
            into = _pick_swing_target(r, other)
            doors.append({
                "position": [px, py],
                "width": width,
                "type": "hinged",
                "wall_index": wall_idx,
                "swing_into": _swing_vector_into_room((px, py), walls[wall_idx], into.rect),
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
                    "position": [px, py],
                    "width": ext_width,
                    "type": "hinged",
                    "wall_index": wall_idx,
                    "swing_into": _swing_vector_into_room((px, py), walls[wall_idx], room.rect),
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
                   eps: float = WALL_DEDUP_EPS_M) -> list[float]:
        """Return cut points from `s` to `e` with no two values within `eps`.

        The eps filter drops sub-segments shorter than 5 cm — narrower
        than any wall thickness and always an artefact of near-coincident
        rectangle corners rather than a real wall.

        Endpoint handling: `raw` starts as `[s, ...inner_splits..., e]`.
        After the merge loop, `out[-1]` may be a split that absorbed `e`
        (if the last inner split was within eps of `e`). In that case we
        snap the final element back to `e` so the wall always terminates
        exactly at the union endpoint and downstream code can rely on
        `walls[i]["end"]` matching the parent rect's corner."""
        raw = [s] + [p for p in splits if s + eps < p < e - eps] + [e]
        raw.sort()
        out = [raw[0]]
        for p in raw[1:]:
            if p - out[-1] > eps:
                out.append(p)
        # Always snap the last point exactly to `e`.
        if out[-1] != e:
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


CIRCULATION_LABELS = {"hallway", "foyer", "mudroom"}


def _rect_area(rect: tuple[float, float, float, float]) -> float:
    return rect[2] * rect[3]


def _pick_swing_target(r: "Room", other: "Room") -> "Room":
    """Pick which of two adjacent rooms a door swings INTO.

    Convention follows real US plans: doors between a circulation space
    (hallway / foyer / mudroom) and any other room swing into the other
    room. Between two non-circulation rooms (e.g. master <-> en-suite),
    the door swings into the smaller one."""
    r_circ = r.label in CIRCULATION_LABELS
    o_circ = other.label in CIRCULATION_LABELS
    if r_circ and not o_circ:
        return other
    if o_circ and not r_circ:
        return r
    return other if _rect_area(other.rect) <= _rect_area(r.rect) else r


def _wall_unit_vector(wall: dict) -> tuple[float, float]:
    sx, sy = wall["start"]
    ex, ey = wall["end"]
    length = ((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5
    if length < 1e-6:
        return (1.0, 0.0)
    return ((ex - sx) / length, (ey - sy) / length)


def _swing_vector_into_room(door_pos: tuple[float, float], wall: dict,
                            target_rect: tuple[float, float, float, float]
                            ) -> list[float]:
    """Return a unit perpendicular to the wall that points into `target_rect`.

    Each wall has two perpendicular directions; we pick the one whose
    short probe lands inside the target rectangle. Used at generation
    time so render() never needs to do polygon-containment scans for
    door swings."""
    wdx, wdy = _wall_unit_vector(wall)
    perp_a = (-wdy, wdx)
    perp_b = (wdy, -wdx)
    rx, ry, rw, rh = target_rect
    px, py = door_pos
    for perp in (perp_a, perp_b):
        tx = px + perp[0] * SWING_PROBE_M
        ty = py + perp[1] * SWING_PROBE_M
        if rx <= tx <= rx + rw and ry <= ty <= ry + rh:
            return [perp[0], perp[1]]
    return [perp_a[0], perp_a[1]]


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

# Rough per-label fill color for the default MLS-pastel style. Style
# presets below can override the whole palette for a different look.
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


def _recolor(base: dict, hue_shift: int, sat_mul: float = 1.0,
             lightness_adj: int = 0) -> dict:
    """Produce a new palette by shifting the HSV hue of `base`. Keeps
    each room's relative color relationship but moves the whole palette
    into a different hue family, which lets us reuse ROOM_COLORS for
    warm-tone, grayscale, and blueprint variants without hand-tuning
    every label."""
    out = {}
    for label, (r, g, b) in base.items():
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        h = (h + hue_shift / 360.0) % 1.0
        s = max(0.0, min(1.0, s * sat_mul))
        v = max(0.0, min(1.0, v + lightness_adj / 255.0))
        r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
        out[label] = (int(r2 * 255), int(g2 * 255), int(b2 * 255))
    return out


def _mono(value: int) -> dict:
    """Return a palette where every room fills with the same near-white
    color. Used for architect-style mono renders."""
    c = (value, value, value)
    return {label: c for label in ROOM_COLORS}


STYLES: dict = {
    "mls_pastel": {
        # dict() so callers mutating STYLES["mls_pastel"]["palette"] do not
        # also mutate ROOM_COLORS.
        "palette": dict(ROOM_COLORS),
        "wall": (25, 25, 25),
        "wall_scale": 1.0,
        "bg": (255, 255, 255),
        "arc": (55, 55, 55),
        "window_a": (55, 90, 140),
        "window_b": (120, 160, 200),
        "text": (60, 60, 60),
    },
    "architect_mono": {
        "palette": _mono(252),
        "wall": (10, 10, 10),
        "wall_scale": 1.4,
        "bg": (253, 252, 247),
        "arc": (25, 25, 25),
        "window_a": (40, 40, 40),
        "window_b": (150, 150, 150),
        "text": (20, 20, 20),
    },
    "warm_tones": {
        "palette": _recolor(ROOM_COLORS, hue_shift=-15, sat_mul=1.25,
                            lightness_adj=-4),
        "wall": (45, 30, 20),
        "wall_scale": 1.1,
        "bg": (252, 248, 240),
        "arc": (70, 50, 40),
        "window_a": (90, 100, 130),
        "window_b": (160, 170, 195),
        "text": (70, 50, 35),
    },
    "grayscale": {
        "palette": _recolor(ROOM_COLORS, hue_shift=0, sat_mul=0.0,
                            lightness_adj=-6),
        "wall": (15, 15, 15),
        "wall_scale": 0.9,
        "bg": (255, 255, 255),
        "arc": (40, 40, 40),
        "window_a": (60, 60, 60),
        "window_b": (150, 150, 150),
        "text": (30, 30, 30),
    },
    "blueprint": {
        "palette": {label: (30, 75, 140) for label in ROOM_COLORS},
        "wall": (235, 240, 255),
        "wall_scale": 1.2,
        "bg": (30, 75, 140),
        "arc": (235, 240, 255),
        "window_a": (215, 230, 250),
        "window_b": (255, 255, 255),
        "text": (235, 240, 255),
    },
    "marketing_bold": {
        "palette": _recolor(ROOM_COLORS, hue_shift=0, sat_mul=2.0,
                            lightness_adj=-8),
        "wall": (35, 35, 35),
        "wall_scale": 1.0,
        "bg": (255, 255, 255),
        "arc": (60, 60, 60),
        "window_a": (40, 110, 180),
        "window_b": (110, 170, 220),
        "text": (50, 50, 50),
    },
}

DEFAULT_STYLE = STYLES["mls_pastel"]


# Per-room fixture layouts. Each entry is (kind, anchor, along_m, depth_m):
#   anchor in {NW,NE,SW,SE} — corner codes; fixture extends from that corner
#            toward the room's interior
#   anchor in {N,S,E,W}    — wall centers; fixture centered on that wall
#   along_m  — long dimension (parallel to the anchor's wall)
#   depth_m  — short dimension (perpendicular, into the room)
FIXTURE_LAYOUT: dict[str, list[tuple[str, str, float, float]]] = {
    "bathroom": [
        ("tub", "S", 1.7, 0.75),
        ("toilet", "NW", 0.45, 0.7),
        ("sink", "NE", 0.55, 0.4),
    ],
    "powder_room": [
        ("toilet", "NW", 0.45, 0.7),
        ("sink", "NE", 0.55, 0.4),
    ],
    "en_suite": [
        ("tub", "S", 1.7, 0.75),
        ("toilet", "NW", 0.45, 0.7),
        ("sink", "NE", 0.55, 0.4),
    ],
    "kitchen": [
        ("stove", "N", 0.6, 0.6),
        ("fridge", "NW", 0.7, 0.7),
        ("kitchen_sink", "NE", 0.7, 0.5),
    ],
}


def _fixture_rect(anchor: str, ax: float, ay: float, rw: float, rh: float,
                  along_m: float, depth_m: float,
                  inset: float = 0.05) -> tuple[float, float, float, float] | None:
    """Return (x, y, w, h) of a fixture's bounding rect given its anchor code
    and the room's origin + dimensions. `along_m` x `depth_m` is axis-aligned
    by default; wall anchors rotate to make `along_m` parallel to the wall.
    Returns None if the fixture would not fit inside the room."""
    if anchor == "NW":
        fw, fh = along_m, depth_m
        x = ax + inset
        y = ay + inset
    elif anchor == "NE":
        fw, fh = along_m, depth_m
        x = ax + rw - along_m - inset
        y = ay + inset
    elif anchor == "SW":
        fw, fh = along_m, depth_m
        x = ax + inset
        y = ay + rh - depth_m - inset
    elif anchor == "SE":
        fw, fh = along_m, depth_m
        x = ax + rw - along_m - inset
        y = ay + rh - depth_m - inset
    elif anchor == "N":
        fw, fh = along_m, depth_m
        x = ax + (rw - along_m) / 2
        y = ay + inset
    elif anchor == "S":
        fw, fh = along_m, depth_m
        x = ax + (rw - along_m) / 2
        y = ay + rh - depth_m - inset
    elif anchor == "W":
        fw, fh = depth_m, along_m
        x = ax + inset
        y = ay + (rh - along_m) / 2
    elif anchor == "E":
        fw, fh = depth_m, along_m
        x = ax + rw - depth_m - inset
        y = ay + (rh - along_m) / 2
    else:
        return None
    # Drop the fixture if the room is too small to hold it.
    if fw > rw - 2 * inset or fh > rh - 2 * inset:
        return None
    return (x, y, fw, fh)


def _draw_fixture_glyph(draw, kind: str, rect_px: tuple[float, float, float, float],
                        ink: tuple, fill: tuple) -> None:
    """Draw a single fixture glyph inside its rect. These are intentionally
    schematic — thin lines on the room fill — matching how fixtures appear
    on real MLS / architect floor plans."""
    x, y, w, h = rect_px
    x1, y1 = x + w, y + h
    lw = 1

    if kind == "toilet":
        # Tank along the shorter top edge, seat (oval) below.
        tank_h = h * 0.28
        draw.rectangle([x, y, x1, y + tank_h], outline=ink, fill=fill, width=lw)
        seat_pad = w * 0.08
        draw.ellipse([x + seat_pad, y + tank_h, x1 - seat_pad, y1],
                     outline=ink, fill=fill, width=lw)
    elif kind == "sink":
        # Outer rect + inner oval basin.
        draw.rectangle([x, y, x1, y1], outline=ink, fill=fill, width=lw)
        pad_w, pad_h = w * 0.15, h * 0.18
        draw.ellipse([x + pad_w, y + pad_h, x1 - pad_w, y1 - pad_h],
                     outline=ink, width=lw)
    elif kind == "tub":
        # Outer rect, rounded inner rect for the basin.
        draw.rectangle([x, y, x1, y1], outline=ink, fill=fill, width=lw)
        pad = min(w, h) * 0.15
        try:
            draw.rounded_rectangle([x + pad, y + pad, x1 - pad, y1 - pad],
                                   radius=min(w, h) * 0.2, outline=ink, width=lw)
        except AttributeError:
            # Pillow < 8.2
            draw.rectangle([x + pad, y + pad, x1 - pad, y1 - pad],
                           outline=ink, width=lw)
    elif kind == "stove":
        # Square with four circular burners.
        draw.rectangle([x, y, x1, y1], outline=ink, fill=fill, width=lw)
        r = min(w, h) * 0.15
        for cx_frac in (0.3, 0.7):
            for cy_frac in (0.3, 0.7):
                cx = x + w * cx_frac
                cy = y + h * cy_frac
                draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                             outline=ink, width=lw)
    elif kind == "fridge":
        # Rectangle divided by a horizontal line (freezer / fridge split).
        draw.rectangle([x, y, x1, y1], outline=ink, fill=fill, width=lw)
        split_y = y + h * 0.33
        draw.line([(x, split_y), (x1, split_y)], fill=ink, width=lw)
    elif kind == "kitchen_sink":
        # Rectangle with two basin rectangles.
        draw.rectangle([x, y, x1, y1], outline=ink, fill=fill, width=lw)
        pad = min(w, h) * 0.12
        mid = x + w / 2
        draw.rectangle([x + pad, y + pad, mid - pad / 2, y1 - pad],
                       outline=ink, width=lw)
        draw.rectangle([mid + pad / 2, y + pad, x1 - pad, y1 - pad],
                       outline=ink, width=lw)


def render(plan_dict: dict, cfg: SynthConfig, style: dict | None = None):
    from PIL import Image, ImageDraw, ImageFont
    size = cfg.image_size
    ppm = cfg.pixels_per_meter
    style = style or DEFAULT_STYLE
    palette = style["palette"]
    wall_color = style["wall"]
    bg_color = style["bg"]
    arc_color = style["arc"]
    window_a = style["window_a"]
    window_b = style["window_b"]
    text_color = style["text"]
    wall_scale = style.get("wall_scale", 1.0)

    # Compute footprint from rooms so centering is accurate.
    xs = [p[0] for r in plan_dict["rooms"] for p in r["polygon"]]
    ys = [p[1] for r in plan_dict["rooms"] for p in r["polygon"]]
    fw = max(xs) - min(xs)
    fh = max(ys) - min(ys)
    off_x = (size - fw * ppm) / 2
    off_y = (size - fh * ppm) / 2

    img = Image.new("RGB", (size, size), bg_color)
    draw = ImageDraw.Draw(img)

    def to_px(p):
        return (off_x + p[0] * ppm, off_y + p[1] * ppm)

    # Fill rooms using the current style's palette.
    for r in plan_dict["rooms"]:
        color = palette.get(r["label"], DEFAULT_ROOM_COLOR)
        draw.polygon([to_px(p) for p in r["polygon"]], fill=color)

    # Fixtures (toilet, sink, tub, stove, fridge, kitchen sink) drawn on top
    # of the room fill but UNDER walls + door arcs + labels. Opt-out via
    # cfg.draw_fixtures=False; the cfg flag used to exist and read nothing.
    if cfg.draw_fixtures:
        fixture_ink = style.get("fixture_ink", wall_color)
        for r in plan_dict["rooms"]:
            spec = FIXTURE_LAYOUT.get(r["label"])
            if not spec:
                continue
            xs2 = [p[0] for p in r["polygon"]]
            ys2 = [p[1] for p in r["polygon"]]
            rx, ry = min(xs2), min(ys2)
            rw, rh = max(xs2) - rx, max(ys2) - ry
            room_fill = palette.get(r["label"], DEFAULT_ROOM_COLOR)
            for kind, anchor, along_m, depth_m in spec:
                rect_m = _fixture_rect(anchor, rx, ry, rw, rh, along_m, depth_m)
                if rect_m is None:
                    continue
                fx, fy, fw_, fh_ = rect_m
                rect_px = (
                    to_px((fx, fy))[0], to_px((fx, fy))[1],
                    to_px((fx + fw_, fy + fh_))[0], to_px((fx + fw_, fy + fh_))[1],
                )
                rect_px_xywh = (
                    rect_px[0], rect_px[1],
                    rect_px[2] - rect_px[0], rect_px[3] - rect_px[1],
                )
                _draw_fixture_glyph(draw, kind, rect_px_xywh,
                                    ink=fixture_ink, fill=room_fill)

    # Draw walls on top, using the style's wall color + thickness scale.
    wall_px = max(3, int(cfg.wall_thickness_m * ppm * 0.9 * wall_scale))
    for w in plan_dict["walls"]:
        draw.line([to_px(w["start"]), to_px(w["end"])], fill=wall_color, width=wall_px)

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

    def _render_door_arc(door, wall):
        """Draw the door as a quarter-arc + leaf line. Swing direction is
        read from `door["swing_into"]` (set at plan_to_schema time); the
        old render-time inference scanned every room's polygon per door."""
        wdx, wdy = _wall_dir(wall)
        swing = door.get("swing_into")
        if swing is None:
            # Augmented-but-pre-swing legacy data path; fall back to a
            # consistent perpendicular so rendering still works.
            swing_x, swing_y = -wdy, wdx
        else:
            swing_x, swing_y = swing[0], swing[1]
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
        draw.line(arc_pts, fill=arc_color, width=1)
        # Door leaf from the hinge to the opened free-end position.
        draw.line([hinge_px, to_px(free_open_m)], fill=arc_color, width=2)

    walls = plan_dict["walls"]
    for d in plan_dict["doors"]:
        wall = walls[d["wall_index"]]
        wdx, wdy = _wall_dir(wall)
        cx, cy = to_px(d["position"])
        half = d["width"] * ppm / 2
        # Erase the wall at the door opening with the BG color so the
        # wall disappears under styles with non-white backgrounds too.
        draw.line([(cx - wdx * half, cy - wdy * half),
                   (cx + wdx * half, cy + wdy * half)],
                  fill=bg_color, width=door_px)
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
                  fill=bg_color, width=door_px)
        # Two thin parallel lines flanking the wall axis — the classic
        # floor-plan window glyph. Offset is about one wall-thickness.
        offset = max(2.0, (wall_px - 1) / 2)
        for s in (-1.0, 1.0):
            ox = perp_x * offset * s
            oy = perp_y * offset * s
            draw.line([(cx - wdx * half + ox, cy - wdy * half + oy),
                       (cx + wdx * half + ox, cy + wdy * half + oy)],
                      fill=window_a, width=1)
        # Thin centerline between the two parallel lines for a sash look.
        draw.line([(cx - wdx * half, cy - wdy * half),
                   (cx + wdx * half, cy + wdy * half)],
                  fill=window_b, width=1)

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
                fill=text_color,
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
                       margin_px: int = 6,
                       fill: tuple = (60, 60, 60)) -> None:
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
                              fill=fill, font=font, anchor="mm")
                return

    # Nothing fit cleanly — draw the text at the smallest tried size.
    font = _get_font(7)
    lines = _wrap_words(words, min(len(words), 3))
    cx, cy = centroid_px
    _, line_h, _ = _measure(draw, font, lines[0])
    total_h = line_h * len(lines) + line_gap * (len(lines) - 1)
    top_y = cy - total_h / 2 + line_h / 2
    for i, ln in enumerate(lines):
        draw.text((cx, top_y + i * (line_h + line_gap)), ln,
                  fill=fill, font=font, anchor="mm")


# ---------- public API ----------

def _apply_augmentation(plan_dict: dict, rot_k: int, flip_x: bool) -> dict:
    """Rotate by `rot_k * 90` degrees and optionally horizontally mirror
    the plan, remapping every coordinate so the image and the JSON stay
    in lock-step.

    Coordinate convention: world y increases southward, same as PIL's
    screen-y. The rotation `(x, y) -> (y, w - x)` is mathematically a
    clockwise quarter-turn but renders as a *counter-clockwise* rotation
    in screen space (because the y axis is flipped relative to standard
    math). We keep the screen-CCW visual semantics — that's what callers
    care about — and document the math here so future readers don't
    chase the sign discrepancy.

    Areas are preserved by rotation + reflection, so we copy them
    through unchanged. Rounding is intentionally NOT applied here:
    `schema.serialize` is the single canonical source of precision and
    rounding twice (here + there) creates two places to drift from."""
    rot = rot_k % 4
    if rot == 0 and not flip_x:
        # Caller is responsible for short-circuiting; assert as a guard.
        return plan_dict

    w0 = max(p[0] for r in plan_dict["rooms"] for p in r["polygon"])
    h0 = max(p[1] for r in plan_dict["rooms"] for p in r["polygon"])

    def transform(x: float, y: float) -> tuple[float, float]:
        bx, by = x, y
        w, h = w0, h0
        for _ in range(rot):
            bx, by = by, w - bx
            w, h = h, w
        if flip_x:
            bx = w - bx
        return bx, by

    def rotate_vector(dx: float, dy: float) -> tuple[float, float]:
        """Direction-vector rotation: same per-step formula as `transform`
        but WITHOUT the `w - bx` translation. Then x-flip negates dx."""
        vx, vy = dx, dy
        for _ in range(rot):
            vx, vy = vy, -vx
        if flip_x:
            vx = -vx
        return vx, vy

    return {
        "scale": dict(plan_dict.get("scale", {"pixels_per_meter": 40})),
        "walls": [
            {
                "start": list(transform(*w_["start"])),
                "end": list(transform(*w_["end"])),
                "thickness": w_.get("thickness", 0.15),
            }
            for w_ in plan_dict["walls"]
        ],
        "doors": [
            {
                "position": list(transform(*d["position"])),
                "width": d["width"],
                "type": d.get("type", "hinged"),
                "wall_index": d["wall_index"],
                **({"swing_into": list(rotate_vector(*d["swing_into"]))}
                   if "swing_into" in d else {}),
            }
            for d in plan_dict["doors"]
        ],
        "windows": [
            {
                "position": list(transform(*wn["position"])),
                "width": wn["width"],
                "wall_index": wn["wall_index"],
            }
            for wn in plan_dict["windows"]
        ],
        "rooms": [
            {
                "label": r["label"],
                "polygon": [list(transform(*p)) for p in r["polygon"]],
                "area": r.get("area", 0.0),
            }
            for r in plan_dict["rooms"]
        ],
    }


def generate_one(seed: int, cfg: SynthConfig | None = None,
                 augment: bool = True):
    """Generate a single (image, plan_dict) pair.

    When augment=True (default), the plan is rotated by 0/90/180/270
    degrees and optionally horizontally flipped before rendering. The
    JSON and image stay in sync because the same transform is applied
    to every coordinate.
    """
    cfg = cfg or SynthConfig()
    rng = random.Random(seed)
    template = rng.choice(TEMPLATES)
    plan = template(rng)
    plan_dict = plan_to_schema(plan, rng)
    if augment:
        rot_k = rng.randrange(4)
        flip_x = rng.random() < 0.5
        if rot_k or flip_x:
            plan_dict = _apply_augmentation(plan_dict, rot_k, flip_x)
        style_name = rng.choice(list(STYLES.keys()))
        style = STYLES[style_name]
    else:
        style = DEFAULT_STYLE
    img = render(plan_dict, cfg, style=style)
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
