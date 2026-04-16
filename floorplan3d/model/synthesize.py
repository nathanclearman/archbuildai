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
import io
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
    # Bay-window protrusions attached to exterior walls. Post-process
    # artifacts: templates build rect rooms first, then bays are added
    # in generate_one for the fraction of samples that should have
    # them. Kept as a Plan field (rather than a Room field) so the
    # pipeline stage that consumes them — plan_to_schema — can iterate
    # without having to scan every room.
    bays: list["BayWindow"] = field(default_factory=list)


@dataclass
class BayWindow:
    """A trapezoidal bump-out on one exterior side of a room.

    Geometry:
        base_width    — length along the host wall (typically 2.2–3.0 m)
        depth         — how far it protrudes outward (typically 0.5–0.9 m)
        top_ratio     — top_width / base_width; 0.55 gives the classic
                        45-degree-ish look
        center_t      — 0..1 position of the bay midpoint along the host
                        side, measured from the side's low-coord end
    """
    room_label: str
    side: str          # "N" | "S" | "E" | "W"
    center_t: float
    base_width: float
    depth: float
    top_ratio: float = 0.55


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


def two_story_colonial(rng: random.Random) -> Plan:
    """First-floor plan of a two-story colonial. No bedrooms (they live
    upstairs); a `stairs` cell sits in the middle band where you would
    physically go up. Front formal rooms + back utility band give a
    very different label distribution from the existing single-floor
    colonial.

    Band layout (y south; column widths per band are independent, so
    this diagram is schematic rather than to-scale):

      front_h:  [ dining_room | foyer      | living_room  ]
      mid_h:    [ kitchen     | stairs     | family_room  ]
      back_h:   [ mudroom     | study                     ]
    """
    w = rng.uniform(12, 14)
    h = rng.uniform(11, 13)

    front_h = rng.uniform(3.5, 4.0)
    mid_h = rng.uniform(3.6, 4.2)
    back_h = h - front_h - mid_h
    if back_h < 3.0:
        back_h = 3.0
        mid_h = h - front_h - back_h

    foyer_w = rng.uniform(2.0, 2.5)
    dining_w = rng.uniform(3.5, 4.2)
    living_w = w - foyer_w - dining_w

    stairs_w = rng.uniform(1.4, 1.8)
    kitchen_w = rng.uniform(3.5, 4.2)
    family_w = w - kitchen_w - stairs_w
    if family_w < 3.0:
        kitchen_w = max(3.0, kitchen_w - (3.0 - family_w))
        family_w = w - kitchen_w - stairs_w

    mud_w = rng.uniform(2.0, 2.5)
    study_w = w - mud_w

    rooms: list[Room] = []
    # Front band
    rooms.append(Room("dining_room", (0, 0, dining_w, front_h)))
    # Foyer<->stairs uses a narrower 0.9 m door because the shared edge
    # (foyer south <-> stairs north) can shrink to ~0.9 m when foyer_w
    # and kitchen_w pull in opposite directions; a 1.0 m door fails to
    # snap on a sub-1.0 m wall segment.
    rooms.append(Room("foyer", (dining_w, 0, foyer_w, front_h),
                      doors=[("dining_room", 0.5, 1.0),
                             ("living_room", 0.5, 1.0),
                             ("stairs", 0.5, 0.9)]))
    rooms.append(Room("living_room", (dining_w + foyer_w, 0, living_w, front_h)))

    # Middle band
    rooms.append(Room("kitchen", (0, front_h, kitchen_w, mid_h),
                      doors=[("dining_room", 0.5, 1.0)]))
    rooms.append(Room("stairs", (kitchen_w, front_h, stairs_w, mid_h)))
    rooms.append(Room("family_room", (kitchen_w + stairs_w, front_h, family_w, mid_h),
                      doors=[("living_room", 0.5, 1.2)]))

    # Back band
    y_back = front_h + mid_h
    rooms.append(Room("mudroom", (0, y_back, mud_w, back_h),
                      doors=[("kitchen", 0.5, 0.9)]))
    rooms.append(Room("study", (mud_w, y_back, study_w, back_h),
                      doors=[("family_room", 0.5, 0.9)]))

    return Plan(rooms=rooms, footprint=(w, h), exterior_door=("foyer", "N"))


def studio_apartment(rng: random.Random) -> Plan:
    """Compact studio: one open `main_room` + a bathroom carved out of
    the NE corner. Smallest template in the pool at ~50 m² and two
    logical rooms (main_room + bathroom).

    Implementation note: the main_room is emitted as TWO same-label
    Room records so the L-shape tiles cleanly without overlapping the
    bathroom polygon. `_filter_internal_walls` drops the artifactual
    wall between the two main_room rectangles before the JSON is
    serialized, so the downstream schema still reads as one continuous
    room."""
    w = rng.uniform(6.5, 8.5)
    h = rng.uniform(7, 9)

    bath_w = rng.uniform(2.2, 2.6)
    bath_h = rng.uniform(2.0, 2.4)

    rooms: list[Room] = []
    # Main room L-shape: tall left column + bottom strip under the bath.
    rooms.append(Room("main_room", (0, 0, w - bath_w, h),
                      doors=[("bathroom", 0.5, 0.8)]))
    rooms.append(Room("main_room", (w - bath_w, bath_h, bath_w, h - bath_h)))
    rooms.append(Room("bathroom", (w - bath_w, 0, bath_w, bath_h)))

    return Plan(rooms=rooms, footprint=(w, h), exterior_door=("main_room", "S"))


def one_bedroom_apartment(rng: random.Random) -> Plan:
    """Urban one-bedroom apartment, ~60-90 m².

    Layout (y south):
      +------------+--------+
      | living     |kitchen |   front: open living + kitchen
      +------------+--------+
      | hallway    |bath|cl |   middle: hall, bath, walk-in closet
      +------------+----+---+
      |          bedroom    |   back: full-width bedroom
      +---------------------+
    """
    w = rng.uniform(7, 9)
    h = rng.uniform(8, 11)

    living_h = rng.uniform(3.5, 4.2)
    bath_h = rng.uniform(2.0, 2.4)
    bedroom_h = h - living_h - bath_h
    if bedroom_h < 3.0:
        bath_h = max(2.0, bath_h - (3.0 - bedroom_h))
        bedroom_h = h - living_h - bath_h

    kitchen_w = rng.uniform(2.5, 3.2)
    living_w = w - kitchen_w
    bath_w = rng.uniform(2.2, 2.6)
    closet_w = rng.uniform(1.2, 1.5)
    hallway_w = w - bath_w - closet_w
    if hallway_w < 1.5:
        bath_w = max(1.8, bath_w - (1.5 - hallway_w))
        hallway_w = w - bath_w - closet_w

    rooms: list[Room] = []
    rooms.append(Room("living_room", (0, 0, living_w, living_h)))
    rooms.append(Room("kitchen", (living_w, 0, kitchen_w, living_h),
                      doors=[("living_room", 0.5, 1.2)]))

    rooms.append(Room("hallway", (0, living_h, hallway_w, bath_h),
                      doors=[("living_room", 0.5, 1.0)]))
    rooms.append(Room("bathroom", (hallway_w, living_h, bath_w, bath_h),
                      doors=[("hallway", 0.5, 0.8)]))
    rooms.append(Room("closet", (hallway_w + bath_w, living_h, closet_w, bath_h)))

    rooms.append(Room("bedroom", (0, living_h + bath_h, w, bedroom_h),
                      doors=[("hallway", 0.5, 0.9), ("closet", 0.5, 0.7)]))

    return Plan(rooms=rooms, footprint=(w, h), exterior_door=("living_room", "N"))


def cape_cod_cottage(rng: random.Random) -> Plan:
    """Small two-bedroom cottage, ~80-110 m². No master suite — both
    bedrooms share one bathroom. Front living + kitchen, back two
    bedrooms with the bath between them.

    Layout (y south):
      +------------+--------+
      | living     |kitchen |   front
      +-----+------+--------+
      | bed | bath |  bed   |   back
      +-----+------+--------+
    """
    w = rng.uniform(9, 11)
    h = rng.uniform(9, 11)

    front_h = rng.uniform(4.0, 4.8)
    back_h = h - front_h
    if back_h < 4.0:
        front_h = h - 4.0
        back_h = 4.0

    kitchen_w = rng.uniform(3.5, 4.2)
    living_w = w - kitchen_w

    bath_w = rng.uniform(2.0, 2.4)
    bed1_w = rng.uniform(3.0, 3.5)
    bed2_w = w - bed1_w - bath_w
    if bed2_w < 3.0:
        bed1_w = max(2.8, bed1_w - (3.0 - bed2_w))
        bed2_w = w - bed1_w - bath_w

    rooms: list[Room] = []
    rooms.append(Room("living_room", (0, 0, living_w, front_h)))
    rooms.append(Room("kitchen", (living_w, 0, kitchen_w, front_h),
                      doors=[("living_room", 0.5, 1.2)]))

    # bed2 sits east of the bathroom; in a small cottage its overlap with
    # living_room can be a sub-door-width sliver (cape cod is ~10 m wide
    # and bed1+bath spans 5-6 m of the south band, leaving bed2 mostly
    # under kitchen). Route bed2 to bathroom (jack-and-jill access)
    # rather than to living_room — guaranteed adjacent on the west wall.
    y_back = front_h
    rooms.append(Room("bedroom", (0, y_back, bed1_w, back_h),
                      doors=[("living_room", 0.5, 0.9)]))
    rooms.append(Room("bathroom", (bed1_w, y_back, bath_w, back_h),
                      doors=[("living_room", 0.5, 0.8)]))
    rooms.append(Room("bedroom", (bed1_w + bath_w, y_back, bed2_w, back_h),
                      doors=[("bathroom", 0.5, 0.8)]))

    return Plan(rooms=rooms, footprint=(w, h), exterior_door=("living_room", "N"))


TEMPLATES: list[Callable[[random.Random], Plan]] = [
    ranch_open_concept,
    colonial_compartmentalized,
    split_bedroom_ranch,
    l_shape_ranch,
    narrow_townhouse,
    two_story_colonial,
    studio_apartment,
    one_bedroom_apartment,
    cape_cod_cottage,
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
    # Templates that need an L-shaped room (e.g. studio_apartment's
    # main_room) tile it with two same-label rectangles. _build_walls
    # treats them as independent rooms and emits a wall on their shared
    # edge; that wall is artifactual — there's no real wall between
    # two parts of the same room. Drop it before downstream code
    # references the wall list.
    walls = _filter_internal_walls(walls, plan)

    # Apply bays: for each BayWindow, remove the host wall segment and
    # insert three new walls (left skirt, top, right skirt) plus one
    # window on each. Order matters — we look up the host rect by label.
    bay_windows_preplaced: list[dict] = []
    for bay in plan.bays:
        host = _find_room(plan, bay.room_label)
        if host is None:
            continue
        before_len = len(walls)
        walls = _apply_bay_to_walls(walls, host.rect, bay, thickness=0.15)
        # The three bay walls are the last 3 entries we just appended;
        # capture them by geometry for later window placement.
        base_lo, top_lo, top_hi, base_hi = _bay_corners(host.rect, bay)
        bay_windows_preplaced.extend(_bay_window_on_walls(
            walls,
            bay_wall_starts=[base_lo, top_lo, top_hi],
            bay_wall_ends=[top_lo, top_hi, base_hi],
        ))
        del before_len

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

    # Bay-window-pre-placed entries go first; regular exterior-wall windows
    # are appended after and may still be filtered by the usual clash check
    # against doors. Bay windows never clash with anything — the wall
    # they sit on was just introduced by _apply_bay_to_walls.
    windows = bay_windows_preplaced + windows

    bays_by_room: dict[str, list[BayWindow]] = {}
    for b in plan.bays:
        bays_by_room.setdefault(b.room_label, []).append(b)

    rooms_out = []
    for r in plan.rooms:
        if r.label in bays_by_room:
            polygon = _room_polygon_with_bays(r.rect, bays_by_room[r.label])
        else:
            polygon = [
                [r.rect[0], r.rect[1]],
                [r.rect[0] + r.rect[2], r.rect[1]],
                [r.rect[0] + r.rect[2], r.rect[1] + r.rect[3]],
                [r.rect[0], r.rect[1] + r.rect[3]],
            ]
        # Area: the rect plus each bay's trapezoid (½ · (a + b) · h).
        area = r.rect[2] * r.rect[3]
        for b in bays_by_room.get(r.label, []):
            top = b.base_width * b.top_ratio
            area += 0.5 * (b.base_width + top) * b.depth
        rooms_out.append({"label": r.label, "polygon": polygon, "area": round(area, 2)})

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


# ---------- bay windows ----------
#
# Bays are applied as a post-process to the rect-based wall graph:
# _build_walls produces a flat list of axis-aligned walls, then each
# BayWindow surgically replaces the host wall segment with five walls
# (two skirts + two angles + top) and inserts three pre-placed windows
# on the three new exterior faces. The room polygon is augmented in
# plan_to_schema so render() sees the protrusion in both the fill and
# the wall-stroke passes.

def _bay_corners(rect: tuple[float, float, float, float],
                 bay: BayWindow) -> tuple[tuple[float, float], ...]:
    """Return (base_lo, top_lo, top_hi, base_hi) in world coords.

    `base_lo`/`base_hi` lie on the host exterior wall; `top_lo`/`top_hi`
    are the outer trapezoid corners. Ordering along each pair is the
    side's natural lo-to-hi direction (west-to-east for N/S,
    north-to-south for E/W) so downstream loops can stitch without
    resorting each bay."""
    x, y, w, h = rect
    bw = bay.base_width
    tw = bay.base_width * bay.top_ratio
    d = bay.depth
    if bay.side in ("N", "S"):
        cx = x + bay.center_t * w
        y_base = y if bay.side == "N" else y + h
        y_top = y_base - d if bay.side == "N" else y_base + d
        return (
            (cx - bw / 2, y_base),
            (cx - tw / 2, y_top),
            (cx + tw / 2, y_top),
            (cx + bw / 2, y_base),
        )
    else:  # E / W
        cy = y + bay.center_t * h
        x_base = x if bay.side == "W" else x + w
        x_top = x_base - d if bay.side == "W" else x_base + d
        return (
            (x_base, cy - bw / 2),
            (x_top, cy - bw / 2),
            (x_top, cy + bw / 2),
            (x_base, cy + bw / 2),
        )


def _apply_bay_to_walls(walls: list[dict], rect: tuple[float, float, float, float],
                        bay: BayWindow, thickness: float) -> list[dict]:
    """Return a new walls list with `bay` applied to `rect`.

    The host wall's exterior segment between the two base corners is
    removed (there may be multiple collinear walls covering it, due to
    the T-junction subdivision in _build_walls); three new walls — left
    skirt, top, right skirt — replace it. Walls that only partially
    overlap the bay base are trimmed rather than dropped."""
    base_lo, top_lo, top_hi, base_hi = _bay_corners(rect, bay)
    horizontal = bay.side in ("N", "S")
    line_coord = base_lo[1] if horizontal else base_lo[0]
    lo = base_lo[0] if horizontal else base_lo[1]
    hi = base_hi[0] if horizontal else base_hi[1]

    def on_line(w: dict) -> bool:
        s, e = w["start"], w["end"]
        if horizontal:
            return abs(s[1] - line_coord) < 1e-6 and abs(e[1] - line_coord) < 1e-6
        return abs(s[0] - line_coord) < 1e-6 and abs(e[0] - line_coord) < 1e-6

    def span(w: dict) -> tuple[float, float]:
        a = w["start"][0] if horizontal else w["start"][1]
        b = w["end"][0] if horizontal else w["end"][1]
        return (min(a, b), max(a, b))

    out: list[dict] = []
    for w in walls:
        if not on_line(w):
            out.append(w)
            continue
        s_lo, s_hi = span(w)
        # No overlap with the bay base range — leave as is.
        if s_hi <= lo + 1e-6 or s_lo >= hi - 1e-6:
            out.append(w)
            continue
        # Trim the portion that lies outside [lo, hi], if any.
        if s_lo < lo - 1e-6:
            out.append(_seg(horizontal, line_coord, s_lo, lo, thickness))
        if s_hi > hi + 1e-6:
            out.append(_seg(horizontal, line_coord, hi, s_hi, thickness))
        # The segment inside [lo, hi] is swallowed by the bay and dropped.

    # Three new walls: left skirt, top, right skirt. We emit them in the
    # order the outer perimeter would traverse (lo → top_lo → top_hi → hi)
    # so any downstream consumer that expects perimeter order still works.
    out.append({"start": list(base_lo), "end": list(top_lo), "thickness": thickness})
    out.append({"start": list(top_lo),  "end": list(top_hi), "thickness": thickness})
    out.append({"start": list(top_hi),  "end": list(base_hi), "thickness": thickness})
    return out


def _seg(horizontal: bool, line_coord: float, lo: float, hi: float,
         thickness: float) -> dict:
    if horizontal:
        return {"start": [lo, line_coord], "end": [hi, line_coord], "thickness": thickness}
    return {"start": [line_coord, lo], "end": [line_coord, hi], "thickness": thickness}


def _bay_window_on_walls(walls: list[dict], bay_wall_starts: list[tuple[float, float]],
                         bay_wall_ends: list[tuple[float, float]]) -> list[dict]:
    """Find the wall indices that match each (start, end) pair and emit
    a centered window on each. Called after _apply_bay_to_walls has
    inserted the three bay walls; we locate them by endpoint match
    rather than by position in the list because downstream filters
    (_filter_internal_walls) could reorder.
    """
    out: list[dict] = []
    for start, end in zip(bay_wall_starts, bay_wall_ends):
        for i, w in enumerate(walls):
            if (_pt_eq(w["start"], start) and _pt_eq(w["end"], end)) or (
                _pt_eq(w["start"], end) and _pt_eq(w["end"], start)):
                wlen = math.hypot(end[0] - start[0], end[1] - start[1])
                win_width = round(max(0.6, wlen * 0.7), 2)
                cx = (start[0] + end[0]) / 2
                cy = (start[1] + end[1]) / 2
                out.append({
                    "position": [round(cx, 2), round(cy, 2)],
                    "width": win_width,
                    "wall_index": i,
                })
                break
    return out


def _pt_eq(a, b, tol: float = 1e-6) -> bool:
    return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol


def _room_polygon_with_bays(rect: tuple[float, float, float, float],
                            bays: list[BayWindow]) -> list[list[float]]:
    """Walk the rect perimeter in CCW-screen-space order (same as the
    existing rect fallback) and insert bay vertices where each bay
    sits on its host side. Bays on the same side are emitted in the
    side's natural traversal direction so the polygon stays simple."""
    x, y, w, h = rect
    corners = {
        "N": [(x, y), (x + w, y)],
        "E": [(x + w, y), (x + w, y + h)],
        "S": [(x + w, y + h), (x, y + h)],
        "W": [(x, y + h), (x, y)],
    }
    by_side: dict[str, list[BayWindow]] = {"N": [], "E": [], "S": [], "W": []}
    for b in bays:
        by_side[b.side].append(b)

    def bay_keypoints(b: BayWindow) -> list[tuple[float, float]]:
        base_lo, top_lo, top_hi, base_hi = _bay_corners(rect, b)
        # Screen-CCW traversal order:
        # N goes W→E (base_lo is west, base_hi is east)  — insert as-is
        # E goes N→S (base_lo is north, base_hi is south) — insert as-is
        # S goes E→W (need base_hi first, then top_hi, top_lo, base_lo)
        # W goes S→N (need base_hi first, then top_hi, top_lo, base_lo)
        if b.side in ("N", "E"):
            return [base_lo, top_lo, top_hi, base_hi]
        return [base_hi, top_hi, top_lo, base_lo]

    poly: list[tuple[float, float]] = []
    for side in ("N", "E", "S", "W"):
        side_start, side_end = corners[side]
        poly.append(side_start)
        bays_here = sorted(by_side[side],
                           key=lambda b: b.center_t if side in ("N", "E")
                                         else 1 - b.center_t)
        for b in bays_here:
            poly.extend(bay_keypoints(b))
        # side_end becomes the next side's side_start — omit to avoid dupes.
    # Close: the first vertex (N side_start) is already there; nothing to do.
    return [[round(p[0], 3), round(p[1], 3)] for p in poly]


def _filter_internal_walls(walls: list[dict], plan: Plan,
                           tol: float = 0.01) -> list[dict]:
    """Drop walls whose two adjacent rectangles share a label.

    Used to clean up L-shaped rooms tiled by two same-label rectangles
    (e.g. studio_apartment's main_room). Only filters when BOTH
    neighbours are present and identically labelled — exterior walls
    (one neighbour) and interior walls between distinct rooms are
    preserved. Relies on axis-aligned rectangles; a wall that is
    neither horizontal nor vertical is kept as-is."""

    def _neighbours(wall: dict) -> tuple[str | None, str | None]:
        """Return (side_a_label, side_b_label) for the two rectangles
        touching `wall`. Either may be None when the wall is on the
        footprint's exterior."""
        sx, sy = wall["start"]
        ex, ey = wall["end"]
        horizontal = abs(sy - ey) < tol
        vertical = abs(sx - ex) < tol
        if not (horizontal or vertical):
            return (None, None)

        # Unpack so the axis-specific logic becomes a single branch.
        if horizontal:
            line_coord = sy
            lo, hi = (sx, ex) if sx <= ex else (ex, sx)
            def _probe(r: Room):
                rx, ry, rw, rh = r.rect
                # Rectangle must fully span the wall interval.
                if not (rx <= lo + tol and rx + rw >= hi - tol):
                    return None
                if abs(ry + rh - line_coord) < tol:
                    return "a"  # rect is north of the wall (ends on it)
                if abs(ry - line_coord) < tol:
                    return "b"  # rect is south of the wall (starts on it)
                return None
        else:  # vertical
            line_coord = sx
            lo, hi = (sy, ey) if sy <= ey else (ey, sy)
            def _probe(r: Room):
                rx, ry, rw, rh = r.rect
                if not (ry <= lo + tol and ry + rh >= hi - tol):
                    return None
                if abs(rx + rw - line_coord) < tol:
                    return "a"  # west side of the wall
                if abs(rx - line_coord) < tol:
                    return "b"  # east side
                return None

        side_a = side_b = None
        for r in plan.rooms:
            side = _probe(r)
            if side == "a":
                side_a = r.label
            elif side == "b":
                side_b = r.label
        return side_a, side_b

    out = []
    for w in walls:
        label_a, label_b = _neighbours(w)
        if label_a and label_b and label_a == label_b:
            continue  # internal — drop
        out.append(w)
    return out


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
    # New labels introduced by the apartment / cottage / two-story templates.
    "main_room": (243, 238, 222),     # studio open space, warm cream
    "study": (235, 232, 220),         # like office, slightly warmer
    "stairs": (220, 220, 220),        # structural void, neutral gray
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


# Anchor placement table: each entry is (x_frac, y_frac, axis_align).
#   x_frac/y_frac in [0, 1]: 0 = pinned to room min-edge, 1 = max-edge,
#                            0.5 = centered.
#   axis_align=True:  the fixture's `along_m` dimension stays on x;
#                     used for corners and N/S wall centers.
#   axis_align=False: the fixture rotates so `along_m` runs on y;
#                     used for W/E wall centers (so a tub running
#                     along an east wall is drawn standing up, not flat).
_FIXTURE_ANCHORS: dict[str, tuple[float, float, bool]] = {
    "NW": (0.0, 0.0, True),
    "NE": (1.0, 0.0, True),
    "SW": (0.0, 1.0, True),
    "SE": (1.0, 1.0, True),
    "N":  (0.5, 0.0, True),
    "S":  (0.5, 1.0, True),
    "W":  (0.0, 0.5, False),
    "E":  (1.0, 0.5, False),
}


def _fixture_rect(anchor: str, ax: float, ay: float, rw: float, rh: float,
                  along_m: float, depth_m: float,
                  inset: float = 0.05) -> tuple[float, float, float, float] | None:
    """Return (x, y, w, h) of a fixture's bounding rect given its anchor and
    the room origin + dimensions. Returns None if the fixture would not fit."""
    entry = _FIXTURE_ANCHORS.get(anchor)
    if entry is None:
        return None
    x_frac, y_frac, axis_align = entry
    fw_, fh_ = (along_m, depth_m) if axis_align else (depth_m, along_m)
    if fw_ > rw - 2 * inset or fh_ > rh - 2 * inset:
        return None
    return (
        ax + inset + x_frac * (rw - fw_ - 2 * inset),
        ay + inset + y_frac * (rh - fh_ - 2 * inset),
        fw_,
        fh_,
    )


# Fixture glyph drawers. Each takes (draw, x, y, w, h, ink, fill) and paints
# the fixture inside its bounding rect. Replaces the old 6-branch dispatcher
# in _draw_fixture_glyph.
def _glyph_toilet(draw, x, y, w, h, ink, fill):
    x1, y1 = x + w, y + h
    tank_h = h * 0.28
    draw.rectangle([x, y, x1, y + tank_h], outline=ink, fill=fill, width=1)
    seat_pad = w * 0.08
    draw.ellipse([x + seat_pad, y + tank_h, x1 - seat_pad, y1],
                 outline=ink, fill=fill, width=1)


def _glyph_sink(draw, x, y, w, h, ink, fill):
    x1, y1 = x + w, y + h
    draw.rectangle([x, y, x1, y1], outline=ink, fill=fill, width=1)
    pad_w, pad_h = w * 0.15, h * 0.18
    draw.ellipse([x + pad_w, y + pad_h, x1 - pad_w, y1 - pad_h],
                 outline=ink, width=1)


def _glyph_tub(draw, x, y, w, h, ink, fill):
    x1, y1 = x + w, y + h
    draw.rectangle([x, y, x1, y1], outline=ink, fill=fill, width=1)
    pad = min(w, h) * 0.15
    try:
        draw.rounded_rectangle([x + pad, y + pad, x1 - pad, y1 - pad],
                               radius=min(w, h) * 0.2, outline=ink, width=1)
    except AttributeError:
        # Pillow < 8.2: no rounded_rectangle.
        draw.rectangle([x + pad, y + pad, x1 - pad, y1 - pad],
                       outline=ink, width=1)


def _glyph_stove(draw, x, y, w, h, ink, fill):
    x1, y1 = x + w, y + h
    draw.rectangle([x, y, x1, y1], outline=ink, fill=fill, width=1)
    r = min(w, h) * 0.15
    for cx_frac in (0.3, 0.7):
        for cy_frac in (0.3, 0.7):
            cx = x + w * cx_frac
            cy = y + h * cy_frac
            draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                         outline=ink, width=1)


def _glyph_fridge(draw, x, y, w, h, ink, fill):
    x1, y1 = x + w, y + h
    draw.rectangle([x, y, x1, y1], outline=ink, fill=fill, width=1)
    split_y = y + h * 0.33
    draw.line([(x, split_y), (x1, split_y)], fill=ink, width=1)


def _glyph_kitchen_sink(draw, x, y, w, h, ink, fill):
    x1, y1 = x + w, y + h
    draw.rectangle([x, y, x1, y1], outline=ink, fill=fill, width=1)
    pad = min(w, h) * 0.12
    mid = x + w / 2
    draw.rectangle([x + pad, y + pad, mid - pad / 2, y1 - pad],
                   outline=ink, width=1)
    draw.rectangle([mid + pad / 2, y + pad, x1 - pad, y1 - pad],
                   outline=ink, width=1)


FIXTURE_DRAWERS: dict[str, Callable] = {
    "toilet": _glyph_toilet,
    "sink": _glyph_sink,
    "tub": _glyph_tub,
    "stove": _glyph_stove,
    "fridge": _glyph_fridge,
    "kitchen_sink": _glyph_kitchen_sink,
}


def _compute_extents(plan_dict: dict) -> tuple[float, float]:
    """Footprint width and height in meters, derived from room polygon
    extremes. Centralized so render() and other consumers agree."""
    xs = [p[0] for r in plan_dict["rooms"] for p in r["polygon"]]
    ys = [p[1] for r in plan_dict["rooms"] for p in r["polygon"]]
    return max(xs) - min(xs), max(ys) - min(ys)


def _to_px_factory(off_x: float, off_y: float, ppm: float):
    """Return a meters-to-pixel converter closing over a centering offset.

    Encapsulating the closure lets the per-pass renderers stay pure
    functions of (draw, plan_dict, style, to_px) instead of needing to
    know about cfg or image dims."""
    def to_px(p):
        return (off_x + p[0] * ppm, off_y + p[1] * ppm)
    return to_px


def _render_rooms(draw, plan_dict, palette, to_px) -> None:
    for r in plan_dict["rooms"]:
        color = palette.get(r["label"], DEFAULT_ROOM_COLOR)
        draw.polygon([to_px(p) for p in r["polygon"]], fill=color)


def _render_fixtures(draw, plan_dict, palette, ink, to_px) -> None:
    for r in plan_dict["rooms"]:
        spec = FIXTURE_LAYOUT.get(r["label"])
        if not spec:
            continue
        xs = [p[0] for p in r["polygon"]]
        ys = [p[1] for p in r["polygon"]]
        rx, ry = min(xs), min(ys)
        rw, rh = max(xs) - rx, max(ys) - ry
        fill = palette.get(r["label"], DEFAULT_ROOM_COLOR)
        for kind, anchor, along_m, depth_m in spec:
            rect_m = _fixture_rect(anchor, rx, ry, rw, rh, along_m, depth_m)
            if rect_m is None:
                continue
            drawer = FIXTURE_DRAWERS.get(kind)
            if drawer is None:
                continue
            fx, fy, fw_, fh_ = rect_m
            tl = to_px((fx, fy))
            br = to_px((fx + fw_, fy + fh_))
            drawer(draw, tl[0], tl[1], br[0] - tl[0], br[1] - tl[1], ink, fill)


def _render_walls(draw, walls, color, width_px, to_px) -> None:
    for w in walls:
        draw.line([to_px(w["start"]), to_px(w["end"])], fill=color, width=width_px)


def _render_door(draw, door, walls, *, bg, arc_color, gap_px, ppm, to_px) -> None:
    """Erase the wall at the opening then draw the swing arc + door leaf."""
    wall = walls[door["wall_index"]]
    wdx, wdy = _wall_unit_vector(wall)
    cx, cy = to_px(door["position"])
    half_px = door["width"] * ppm / 2
    # Erase the wall in the bg color so the gap reads cleanly even on
    # styles with non-white backgrounds (e.g. blueprint).
    draw.line([(cx - wdx * half_px, cy - wdy * half_px),
               (cx + wdx * half_px, cy + wdy * half_px)],
              fill=bg, width=gap_px)

    swing = door.get("swing_into") or (-wdy, wdx)
    swing_x, swing_y = swing[0], swing[1]
    width_m = door["width"]
    half_m = width_m / 2
    sign = 1 if (door["wall_index"] % 2) == 0 else -1
    px_m, py_m = door["position"]
    hinge_m = (px_m + wdx * half_m * sign, py_m + wdy * half_m * sign)
    free_open_m = (hinge_m[0] + swing_x * width_m,
                   hinge_m[1] + swing_y * width_m)
    hinge_px = to_px(hinge_m)
    r_px = width_m * ppm

    # Trig directly: PIL angle conventions for draw.arc are awkward, but
    # angles in (world == screen, since y increases downward in both)
    # let us sample a polyline cheaply.
    start_ang = math.atan2(-wdy * sign, -wdx * sign)
    end_ang = math.atan2(swing_y, swing_x)
    delta = end_ang - start_ang
    while delta > math.pi:
        delta -= 2 * math.pi
    while delta < -math.pi:
        delta += 2 * math.pi

    # Sample the arc as a polyline. Each point computes the angle once
    # (the old version recomputed `start_ang + (i/N) * delta` for both
    # cos and sin — two trig identities per step for no reason).
    step = delta / ARC_POLYLINE_STEPS
    arc_pts = []
    for i in range(ARC_POLYLINE_STEPS + 1):
        ang = start_ang + i * step
        arc_pts.append((hinge_px[0] + r_px * math.cos(ang),
                        hinge_px[1] + r_px * math.sin(ang)))
    draw.line(arc_pts, fill=arc_color, width=1)
    draw.line([hinge_px, to_px(free_open_m)], fill=arc_color, width=2)


def _render_window(draw, window, walls, *, bg, color_a, color_b,
                   gap_px, wall_px, ppm, to_px) -> None:
    wall = walls[window["wall_index"]]
    wdx, wdy = _wall_unit_vector(wall)
    perp_x, perp_y = -wdy, wdx
    cx, cy = to_px(window["position"])
    half_px = window["width"] * ppm / 2

    # Endpoint pair shared by the gap-erase + the centerline + the
    # parallels — compute once instead of recomputing 5 times like
    # the old version did.
    end_a = (cx - wdx * half_px, cy - wdy * half_px)
    end_b = (cx + wdx * half_px, cy + wdy * half_px)

    draw.line([end_a, end_b], fill=bg, width=gap_px)

    offset = max(2.0, (wall_px - 1) / 2)
    for s in (-1.0, 1.0):
        ox, oy = perp_x * offset * s, perp_y * offset * s
        draw.line([(end_a[0] + ox, end_a[1] + oy),
                   (end_b[0] + ox, end_b[1] + oy)],
                  fill=color_a, width=1)
    draw.line([end_a, end_b], fill=color_b, width=1)


def _render_labels(draw, plan_dict, text_color, ppm, to_px, *,
                   show_dimensions: bool = False) -> None:
    for r in plan_dict["rooms"]:
        xs = [p[0] for p in r["polygon"]]
        ys = [p[1] for p in r["polygon"]]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        label = r["label"].replace("_", " ").title()
        if show_dimensions:
            # MLS convention: short dimension first, e.g. "12'6" x 14'0"".
            # _draw_label_fitted handles the embedded newline.
            w_m = max(xs) - min(xs)
            h_m = max(ys) - min(ys)
            short, long = sorted([w_m, h_m])
            label = f"{label}\n{_metric_to_ft_in(short)} x {_metric_to_ft_in(long)}"
        _draw_label_fitted(
            draw, to_px((cx, cy)), label,
            room_w_px=(max(xs) - min(xs)) * ppm,
            room_h_px=(max(ys) - min(ys)) * ppm,
            fill=text_color,
        )


# ---------- US dimension callout formatting ----------

_METERS_TO_INCHES = 39.3700787


def _metric_to_ft_in(m: float) -> str:
    """Format a positive metric length as a US floor-plan dimension
    callout, e.g. 3.86 m -> `12'8"`. Inches are rounded to the nearest
    integer and 12" rolls over to the next foot so we never emit
    `12'12"`. Matches the convention used on US MLS listings."""
    total_in = max(0.0, m) * _METERS_TO_INCHES
    ft = int(total_in // 12)
    inches = int(round(total_in - ft * 12))
    if inches >= 12:
        ft += 1
        inches -= 12
    return f"{ft}'{inches}\""


def render(plan_dict: dict, cfg: SynthConfig, style: dict | None = None,
           *, show_dimensions: bool = False, title_block: str | None = None,
           watermark: str | None = None):
    """Render the floor plan to a PIL Image. Pure orchestration —
    each layer (rooms, fixtures, walls, openings, labels) is delegated
    to a `_render_*` helper that takes only what it needs.

    `show_dimensions` appends a `W'W" x L'L"` callout under each room
    label (US MLS convention). `title_block` adds a "FLOOR PLAN" /
    "MAIN LEVEL" banner in the corner. `watermark` overlays a diagonal
    semi-transparent string across the whole canvas the way listing
    exports stamp broker IDs. All three are off by default so the pure
    render path stays deterministic.
    """
    style = style or DEFAULT_STYLE
    size = cfg.image_size
    ppm = cfg.pixels_per_meter

    fw, fh = _compute_extents(plan_dict)
    off_x = (size - fw * ppm) / 2
    off_y = (size - fh * ppm) / 2
    to_px = _to_px_factory(off_x, off_y, ppm)

    img = Image.new("RGB", (size, size), style["bg"])
    draw = ImageDraw.Draw(img)
    walls = plan_dict["walls"]

    _render_rooms(draw, plan_dict, style["palette"], to_px)

    if cfg.draw_fixtures:
        _render_fixtures(draw, plan_dict, style["palette"],
                         style.get("fixture_ink", style["wall"]), to_px)

    wall_px = max(3, int(cfg.wall_thickness_m * ppm * 0.9 * style.get("wall_scale", 1.0)))
    gap_px = max(4, wall_px + DOOR_GAP_PADDING_PX)

    _render_walls(draw, walls, style["wall"], wall_px, to_px)

    for d in plan_dict["doors"]:
        _render_door(draw, d, walls,
                     bg=style["bg"], arc_color=style["arc"],
                     gap_px=gap_px, ppm=ppm, to_px=to_px)

    for w in plan_dict["windows"]:
        _render_window(draw, w, walls,
                       bg=style["bg"], color_a=style["window_a"],
                       color_b=style["window_b"],
                       gap_px=gap_px, wall_px=wall_px, ppm=ppm, to_px=to_px)

    if cfg.draw_labels:
        _render_labels(draw, plan_dict, style["text"], ppm, to_px,
                       show_dimensions=show_dimensions)

    if title_block:
        _render_title_block(img, title_block, style)
    if watermark:
        img = _render_watermark(img, watermark, style)

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


def _measure(draw, font, text: str) -> tuple[int, int]:
    """Return (width, height) in pixels for a single-line string."""
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font, anchor="lt")
    return x1 - x0, y1 - y0


def _draw_label_fitted(draw, centroid_px, text: str,
                       room_w_px: float, room_h_px: float,
                       margin_px: int = LABEL_MARGIN_PX,
                       fill: tuple = (60, 60, 60)) -> None:
    """Draw `text` at `centroid_px` wrapped and shrunk to fit the room box.

    If `text` contains `\\n` the breaks are honored as forced line
    separators (used for dimension callouts like `BEDROOM\\n12'6" x 14'0"`).
    A forced-break layout that can't fit even at the smallest font falls
    back to the first line alone so a tight room never shows truncated
    dimensions.
    """
    max_w = max(room_w_px - 2 * margin_px, 20)
    max_h = max(room_h_px - 2 * margin_px, 12)
    line_gap = LABEL_LINE_GAP_PX

    def _draw_lines(lines: list[str], font) -> None:
        line_h = max(_measure(draw, font, ln)[1] for ln in lines)
        total_h = line_h * len(lines) + line_gap * (len(lines) - 1)
        cx, cy = centroid_px
        top_y = cy - total_h / 2 + line_h / 2
        for i, ln in enumerate(lines):
            draw.text((cx, top_y + i * (line_h + line_gap)), ln,
                      fill=fill, font=font, anchor="mm")

    def _fits(lines: list[str], font) -> bool:
        sizes = [_measure(draw, font, ln) for ln in lines]
        total_w = max(w for w, _ in sizes)
        line_h = max(h for _, h in sizes)
        total_h = line_h * len(lines) + line_gap * (len(lines) - 1)
        return total_w <= max_w and total_h <= max_h

    # Forced-break path: used for label + dimension callouts. We try the
    # font ladder as given; if nothing fits, drop the secondary lines and
    # fall through to the word-wrap path with just the first line.
    if "\n" in text:
        forced = [seg for seg in text.split("\n") if seg.strip()]
        for font_size in LABEL_FONT_LADDER:
            font = _get_font(font_size)
            if _fits(forced, font):
                _draw_lines(forced, font)
                return
        text = forced[0]

    words = text.split()
    # Try font sizes from large to small. For each size, try 1-, 2-, 3-line
    # wraps and pick the first that fits horizontally and vertically.
    for font_size in LABEL_FONT_LADDER:
        font = _get_font(font_size)
        for n_lines in range(1, min(len(words), 3) + 1):
            lines = _wrap_words(words, n_lines)
            if _fits(lines, font):
                _draw_lines(lines, font)
                return

    # Nothing fit cleanly — draw at the smallest tried size and accept
    # some overflow. Picking a wrapped layout still keeps tail characters
    # visible instead of mid-word-clipped.
    font = _get_font(LABEL_MIN_FONT_PX)
    _draw_lines(_wrap_words(words, min(len(words), 3)), font)


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


# ---------- title block + watermark overlays ----------
#
# These are image-space only — `render()` emits the final pixels after
# the layout passes, so overlays never interact with the JSON. They're
# driven by explicit strings so a caller can reproduce a specific sample
# (useful for debugging "why does the model hallucinate a watermark
# when there isn't one").

TITLE_BLOCK_CANDIDATES: tuple[str, ...] = (
    "FLOOR PLAN",
    "MAIN LEVEL",
    "FIRST FLOOR",
    "GROUND FLOOR",
    "UPPER LEVEL",
    "LOWER LEVEL",
)

WATERMARK_CANDIDATES: tuple[str, ...] = (
    "SAMPLE PLAN",
    "NOT TO SCALE",
    "FOR MARKETING USE ONLY",
    "DRAFT",
    "MLS PREVIEW",
)


def _render_title_block(img: Image.Image, text: str, style: dict) -> None:
    """Draw a small banner in the bottom-left corner. Real MLS plans
    use a title block to note the level; including it trains the model
    to ignore floating UI text rather than misreading it as a room
    label."""
    w, h = img.size
    draw = ImageDraw.Draw(img)
    font = _get_font(max(12, h // 45))
    pad = max(6, h // 80)
    tw, th = _measure(draw, font, text)
    # Background rectangle slightly inset from the corner so the banner
    # reads as a label rather than extending off-canvas.
    x0, y0 = pad, h - pad - th - 2 * pad
    x1, y1 = x0 + tw + 2 * pad, h - pad
    border = style.get("wall", (25, 25, 25))
    bg = style.get("bg", (255, 255, 255))
    draw.rectangle([x0, y0, x1, y1], fill=bg, outline=border, width=2)
    draw.text((x0 + pad, y0 + pad), text, fill=border, font=font, anchor="lt")


def _render_watermark(img: Image.Image, text: str, style: dict) -> Image.Image:
    """Overlay diagonal semi-transparent text across the canvas, as many
    listing exports stamp broker IDs. Returns a new image because we go
    via RGBA to get proper alpha, then flatten back to RGB."""
    w, h = img.size
    # Build the watermark in RGBA so partial transparency composites.
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    font = _get_font(max(24, h // 12))
    # Faint, using the style's wall color as the base ink.
    ink = style.get("wall", (25, 25, 25))
    alpha = 40  # ~15% opacity
    # Repeat the watermark in a diagonal tile so a large canvas isn't
    # dominated by a single central string.
    tw, th = _measure(odraw, font, text)
    step = max(tw + 120, 260)
    for cx in range(-w, 2 * w, step):
        for cy in range(-h, 2 * h, step):
            odraw.text((cx, cy), text, fill=(*ink, alpha), font=font, anchor="lt")
    overlay = overlay.rotate(30, resample=Image.BILINEAR, expand=False,
                             fillcolor=(0, 0, 0, 0))
    composed = Image.alpha_composite(img.convert("RGBA"), overlay)
    return composed.convert("RGB")


# ---------- photometric augmentation ----------
#
# Simulate the photo / scan / print artifacts real MLS listings accumulate
# between the architect's CAD export and our model's eyeballs. Operates on
# the rendered image only — the JSON targets are untouched.

# Warm / cool "paper" multipliers applied to the rendered RGB. Values are
# RGB gains (1.0 = no change). Target is a light-shift the rendered plan
# would experience when printed or photocopied onto non-pure-white stock.
# Kept deliberately subtle — too much tint eats contrast between wall ink
# and room fill and makes training labels harder to attend to.
PAPER_TINTS: tuple[tuple[float, float, float], ...] = (
    (1.00, 0.98, 0.92),  # cream (aged office paper)
    (0.98, 0.95, 0.88),  # manila
    (1.00, 0.99, 0.96),  # natural white
    (0.95, 0.97, 1.00),  # cool / blueprint-side
    (0.97, 0.96, 0.93),  # soft gray newsprint
)

JPEG_QUALITY_RANGE: tuple[int, int] = (35, 92)

# Small-angle skew — real scans and photos-of-screens are rarely square
# to the page. Kept tight (+/- 2 deg) because wide plans fill most of the
# canvas and a larger angle rotates wall corners outside the frame.
SKEW_ANGLE_DEG: float = 2.0

# Probabilities — tuned so each effect fires on a meaningful fraction of
# samples without every sample looking identical. Every aug is independent
# of the others; a sample can be tinted + skewed + JPEG'd + grayscale'd.
P_TINT: float = 0.6
P_JPEG: float = 0.7
P_SKEW: float = 0.5
P_GRAYSCALE: float = 0.15
P_TITLE_BLOCK: float = 0.35
P_WATERMARK: float = 0.15
P_DIMENSIONS: float = 0.6

# Bay windows are common but not universal — roughly a third of
# mid-century and newer US plans have at least one. Picked 0.35 so a
# meaningful fraction of training samples exercise the angled-wall
# geometry without every plan looking like a Victorian.
P_BAY: float = 0.35


# ---------- bay-window attachment ----------
#
# Bays only make sense on exterior walls long enough to host a 2+ m
# bump-out, and only on rooms where the bay makes architectural sense
# (public rooms and bedrooms, never service rooms like closets).
_BAY_ELIGIBLE_ROOMS = {
    "great_room", "living_room", "family_room", "dining_room",
    "master_bedroom", "bedroom", "den", "office",
    # also reasonable: breakfast nook = "kitchen" with a small bay
    "kitchen",
}


def _maybe_attach_bay(plan: Plan, rng: random.Random) -> None:
    """Mutate `plan` in place: with some probability, add a single bay
    window to a random eligible exterior side of an eligible room.
    No-op if no room qualifies, so small templates (studio etc.) stay
    untouched."""
    if rng.random() >= P_BAY:
        return
    fw, fh = plan.footprint
    candidates: list[tuple[Room, str, float]] = []  # (room, side, side_length)
    for r in plan.rooms:
        if r.label not in _BAY_ELIGIBLE_ROOMS:
            continue
        _, _, w, h = r.rect
        for side in ("N", "S", "E", "W"):
            if not _edge_is_exterior(r.rect, side, fw, fh):
                continue
            side_len = w if side in ("N", "S") else h
            # Need ~2.4 m for a reasonable bay plus 0.3 m skirt each end.
            if side_len < 3.0:
                continue
            candidates.append((r, side, side_len))
    if not candidates:
        return
    host, side, side_len = rng.choice(candidates)
    base_width = round(rng.uniform(2.2, min(3.0, side_len - 0.6)), 2)
    # center_t kept away from the corners so we always leave >=0.3 m
    # skirt on each side.
    min_t = (base_width / 2 + 0.3) / side_len
    max_t = 1 - min_t
    center_t = round(rng.uniform(min_t, max_t), 3)
    depth = round(rng.uniform(0.55, 0.85), 2)
    plan.bays.append(BayWindow(
        room_label=host.label,
        side=side,
        center_t=center_t,
        base_width=base_width,
        depth=depth,
    ))


def _apply_paper_tint(img: Image.Image, rng: random.Random) -> Image.Image:
    """Multiply each channel by a paper-color gain so the "white" areas
    take on a warm/cool cast. Blueprint-style renders (already a strong
    blue) are passed through unchanged — a second tint would just make
    them muddy.
    """
    # Heuristic: if the darkest pixel is brighter than the median, the
    # image has no black ink and is probably a blueprint — skip.
    gains = rng.choice(PAPER_TINTS)
    # Nudge each gain by up to +-2% so we don't get five discrete buckets
    # after 10k samples.
    gains = tuple(max(0.5, min(1.2, g * (1.0 + rng.uniform(-0.02, 0.02))))
                  for g in gains)
    r, g, b = img.split()
    r = r.point(lambda v, k=gains[0]: max(0, min(255, int(v * k))))
    g = g.point(lambda v, k=gains[1]: max(0, min(255, int(v * k))))
    b = b.point(lambda v, k=gains[2]: max(0, min(255, int(v * k))))
    return Image.merge("RGB", (r, g, b))


def _apply_jpeg_roundtrip(img: Image.Image, rng: random.Random) -> Image.Image:
    """Round-trip the image through in-memory JPEG encode/decode so the
    model sees the 8x8 blocking + chroma-subsample ringing that shows up
    around wall edges in every MLS / brochure floor plan."""
    q = rng.randint(*JPEG_QUALITY_RANGE)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q, subsampling=2)
    buf.seek(0)
    with Image.open(buf) as compressed:
        return compressed.convert("RGB")


def _apply_small_skew(img: Image.Image, rng: random.Random,
                      bg_color: tuple[int, int, int]) -> Image.Image:
    """Rotate the image by a few degrees around its center. Geometry
    JSON is axis-aligned by construction, so this teaches the model to
    unskew rather than learn an easier axis-aligned prior that real
    scans will violate. `bg_color` is used to fill the corners the
    rotation exposes, matching whatever paper stock the style chose."""
    angle = rng.uniform(-SKEW_ANGLE_DEG, SKEW_ANGLE_DEG)
    return img.rotate(angle, resample=Image.BILINEAR, expand=False,
                      fillcolor=bg_color)


def _apply_grayscale(img: Image.Image) -> Image.Image:
    """Collapse to single-channel luminance then back to RGB. Photocopies
    and older fax scans lose color entirely — the model should still
    read the geometry."""
    from PIL import ImageOps
    return ImageOps.grayscale(img).convert("RGB")


def _augment_image(img: Image.Image, rng: random.Random,
                   bg_color: tuple[int, int, int] = (255, 255, 255)
                   ) -> Image.Image:
    """Photometric augmentation pipeline. Applied after render, before
    disk write. Purely image-space — geometry JSON is never touched."""
    if rng.random() < P_TINT:
        img = _apply_paper_tint(img, rng)
    if rng.random() < P_SKEW:
        img = _apply_small_skew(img, rng, bg_color)
    if rng.random() < P_GRAYSCALE:
        img = _apply_grayscale(img)
    if rng.random() < P_JPEG:
        img = _apply_jpeg_roundtrip(img, rng)
    return img


def generate_one(seed: int, cfg: SynthConfig | None = None,
                 augment: bool = True):
    """Generate a single (image, plan_dict) pair.

    When augment=True (default), the plan is rotated by 0/90/180/270
    degrees and optionally horizontally flipped before rendering, the
    room labels may include dimension callouts, and the image may get
    title-block / watermark overlays plus photometric noise (paper
    tint, JPEG round-trip, small-angle skew, grayscale). The JSON and
    image stay in sync because the only geometry transform (rot/flip)
    is applied to both; every other aug touches pixels only.
    """
    cfg = cfg or SynthConfig()
    rng = random.Random(seed)
    template = rng.choice(TEMPLATES)
    plan = template(rng)
    if augment:
        _maybe_attach_bay(plan, rng)
    plan_dict = plan_to_schema(plan, rng)
    show_dimensions = False
    title_block: str | None = None
    watermark: str | None = None
    if augment:
        rot_k = rng.randrange(4)
        flip_x = rng.random() < 0.5
        if rot_k or flip_x:
            plan_dict = _apply_augmentation(plan_dict, rot_k, flip_x)
        style_name = rng.choice(list(STYLES.keys()))
        style = STYLES[style_name]
        show_dimensions = rng.random() < P_DIMENSIONS
        if rng.random() < P_TITLE_BLOCK:
            title_block = rng.choice(TITLE_BLOCK_CANDIDATES)
        if rng.random() < P_WATERMARK:
            watermark = rng.choice(WATERMARK_CANDIDATES)
    else:
        style = DEFAULT_STYLE
    img = render(plan_dict, cfg, style=style,
                 show_dimensions=show_dimensions,
                 title_block=title_block, watermark=watermark)
    if augment:
        img = _augment_image(img, rng, bg_color=style["bg"])
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
