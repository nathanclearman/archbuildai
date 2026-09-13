"""
Hybrid backend glue: YOLO geometry + VLM-read room labels.

The YOLO backend finds walls, openings and room polygons on real plans well
but assigns room labels by heuristics (mostly wrong). The Qwen2.5-VL base
model reads printed text with positions reliably. This module marries the
two: every room-name text box the VLM found is dropped onto the YOLO room
polygon that contains it (or the nearest one within `max_snap_m`), and that
polygon takes the printed name, normalized to the schema vocabulary.

Pure Python, no bpy — unit-tested outside Blender.
"""

from __future__ import annotations

import math
import re

# Printed text → canonical label. Keys are normalized (lowercase, alnum +
# spaces). Anything not listed is slugified as-is so nothing is lost.
LABEL_SYNONYMS: dict[str, str] = {
    "living": "living_room", "living room": "living_room", "living rm": "living_room",
    "formal living": "living_room", "great room": "great_room", "family": "family_room",
    "family room": "family_room", "family rm": "family_room", "rec room": "family_room",
    "dining": "dining_room", "dining room": "dining_room", "dining area": "dining_room",
    "formal dining": "dining_room", "breakfast": "dining_room", "nook": "dining_room",
    "kitchen": "kitchen", "eat in kitchen": "kitchen", "eatin kitchen": "kitchen", "kit": "kitchen",
    "primary bedroom": "master_bedroom", "master bedroom": "master_bedroom", "primary": "master_bedroom",
    "master": "master_bedroom", "primary bed": "master_bedroom", "master bed": "master_bedroom",
    "owners suite": "master_bedroom", "owner s suite": "master_bedroom",
    "bedroom": "bedroom", "bed": "bedroom", "br": "bedroom", "bdrm": "bedroom", "bedrm": "bedroom",
    "guest room": "bedroom", "guest bedroom": "bedroom",
    "bath": "bathroom", "bathroom": "bathroom", "full bath": "bathroom", "full bathroom": "bathroom",
    "ensuite": "en_suite", "en suite": "en_suite", "master bath": "en_suite", "primary bath": "en_suite",
    "half bath": "powder_room", "1 2 bath": "powder_room", "12 bath": "powder_room",
    "powder": "powder_room", "powder room": "powder_room", "wc": "powder_room", "toilet": "powder_room",
    "wic": "walk_in_closet", "w i c": "walk_in_closet", "walk in closet": "walk_in_closet",
    "walk in": "walk_in_closet", "closet": "closet", "cl": "closet", "clo": "closet", "wardrobe": "closet",
    "foyer": "foyer", "entry": "foyer", "entrance": "foyer", "entry hall": "foyer",
    "hall": "hallway", "hallway": "hallway", "corridor": "hallway",
    "mudroom": "mudroom", "mud room": "mudroom",
    "laundry": "laundry_room", "laundry room": "laundry_room", "utility": "laundry_room",
    "laundry storage": "laundry_room", "pantry": "pantry",
    "garage": "garage", "office": "office", "study": "study", "den": "den", "library": "study",
    "stairs": "stairs", "stair": "stairs", "stairway": "stairs", "up": "stairs", "dn": "stairs",
    "balcony": "balcony", "deck": "deck", "patio": "patio", "porch": "porch", "terrace": "balcony",
    "storage": "closet", "mechanical": "utility", "mech": "utility", "furnace": "utility",
}

# Text the OCR pass may return that is not a room name.
_NOISE = re.compile(r"^(\d+['\"]?\s*\d*[\"']?|x|area|\[?area:?.*|1st floor|2nd floor|floor \d|n|s|e|w|up|dn|none|n a|unknown|brh|bhr|ch|clg)$")


def normalize_label(text: str) -> str | None:
    """Map printed text to a canonical snake_case label. None for non-rooms."""
    t = text.lower().replace("&", " and ")
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t or _NOISE.match(t):
        return None
    if t in LABEL_SYNONYMS:  # exact hit first ("1 2 bath" keeps its digits)
        return LABEL_SYNONYMS[t]
    # Strip area / dimension fragments ("living room 13x15")
    t = re.sub(r"\b\d+(\s*x\s*\d+)?\b", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return None
    if t in LABEL_SYNONYMS:
        return LABEL_SYNONYMS[t]
    # Try dropping a numeric suffix like "bedroom 2"
    base = re.sub(r"\s+\d+$", "", t)
    if base in LABEL_SYNONYMS:
        return LABEL_SYNONYMS[base]
    return re.sub(r"\s", "_", t)


def point_in_polygon(pt, poly) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xi:
                inside = not inside
    return inside


def _dist_point_polygon(pt, poly) -> float:
    if point_in_polygon(pt, poly):
        return 0.0
    px, py = pt
    best = math.inf
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        lsq = dx * dx + dy * dy
        t = 0.0 if lsq == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / lsq))
        cx, cy = x1 + t * dx, y1 + t * dy
        best = min(best, math.hypot(px - cx, py - cy))
    return best


def polygon_bbox_px(polygon, pixels_per_meter: float) -> list[float]:
    """Axis-aligned pixel box of a room polygon (metres → original pixels)."""
    xs = [p[0] * pixels_per_meter for p in polygon]
    ys = [p[1] * pixels_per_meter for p in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


def fill_unlabeled_by_crop(rooms: list[dict], read_crop, pixels_per_meter: float,
                           unlabeled: str = "room",
                           allowed: set[str] | None = None) -> tuple[list[dict], int]:
    """Second pass: for every room still unlabeled, ask `read_crop(bbox_px)`
    for the name printed inside its box. Rooms are copied, never mutated.

    `allowed`: when given, a crop answer is accepted only if its normalized
    label is in this set — pass the labels found by the whole-plan OCR
    pass. The crop model tends to guess ("bathroom" for any small room with
    fixtures, "BRH" from window annotations); pass 1 is reliable about
    WHICH names exist, so it vets pass 2, which only resolves WHERE.
    Returns (rooms, number_filled)."""
    out = [dict(r) for r in rooms]
    filled = 0
    for r in out:
        if r.get("label_source") not in (None, "none", "yolo"):
            continue
        poly = r.get("polygon") or []
        if len(poly) < 3:
            continue
        try:
            text = read_crop(polygon_bbox_px(poly, pixels_per_meter))
        except Exception:  # noqa: BLE001 — one bad crop must not sink the plan
            continue
        label = normalize_label(text or "")
        if not label or (allowed is not None and label not in allowed):
            continue
        r["label"] = label
        r["label_source"] = "crop"
        filled += 1
    return out, filled


def assign_labels(rooms: list[dict], texts: list[tuple[str, list[float]]],
                  pixels_per_meter: float, max_snap_m: float = 1.5,
                  unlabeled: str = "room") -> tuple[list[dict], dict]:
    """Relabel `rooms` (YOLO polygons, metres) from OCR `texts`.

    texts: [(printed_text, [x1, y1, x2, y2] in ORIGINAL image pixels)].
    Room coordinates follow the YOLO convention: metres = pixels / ppm,
    origin top-left, no y flip.

    Rooms that receive no printed name are relabeled `unlabeled` ("room")
    rather than keeping YOLO's heuristic guess — those guesses are wrong far
    more often than right, and an honest neutral label is better for the
    downstream Premium features than a confident wrong one. Pass
    unlabeled=None to keep YOLO's labels instead.

    Returns (new_rooms, report). Rooms are copied, never mutated. Each
    output room carries `label_source`: "ocr", "yolo" or "none".
    """
    out = [dict(r) for r in rooms]
    for r in out:
        r["label_source"] = "yolo"
    # Larger text first: a room's main name usually beats a stray sub-label.
    order = sorted(range(len(texts)), key=lambda i: -abs((texts[i][1][2] - texts[i][1][0]) * (texts[i][1][3] - texts[i][1][1])))
    assigned: dict[int, str] = {}
    dropped: list[str] = []
    for i in order:
        raw, box = texts[i]
        label = normalize_label(raw)
        if label is None:
            dropped.append(raw)
            continue
        cx = (box[0] + box[2]) / 2.0 / pixels_per_meter
        cy = (box[1] + box[3]) / 2.0 / pixels_per_meter
        best_j, best_d = None, math.inf
        for j, r in enumerate(out):
            poly = r.get("polygon") or []
            if len(poly) < 3:
                continue
            d = _dist_point_polygon((cx, cy), poly)
            if d < best_d:
                best_j, best_d = j, d
        if best_j is None or best_d > max_snap_m:
            dropped.append(raw)
            continue
        if best_j in assigned:
            # Second text in the same room (e.g. "FORMAL LIVING" + "FORMAL
            # DINING" printed on two lines) — keep the first, larger one.
            continue
        assigned[best_j] = label
        out[best_j]["label"] = label
        out[best_j]["label_source"] = "ocr"
    if unlabeled is not None:
        for j, r in enumerate(out):
            if j not in assigned:
                r["label"] = unlabeled
                r["label_source"] = "none"
    report = {
        "labels_seen": sorted({l for l in (normalize_label(t) for t, _ in texts) if l}),
        "rooms": len(out),
        "labeled_from_ocr": len(assigned),
        "kept_yolo_label": len(out) - len(assigned),
        "ocr_texts": len(texts),
        "dropped_texts": dropped,
    }
    return out, report
