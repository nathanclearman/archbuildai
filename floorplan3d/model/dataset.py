"""
Training-data loader for the floor plan VLM.

Reads CubiCasa5k (and, when available, ResPlan / CFP / our synthetic set),
normalizes every sample to the canonical schema (see schema.py), and emits
(image, target_json_string) pairs suitable for SFT of a vision-language model.

Also exposes a RealMLSLoader for the held-out real-listing eval set under
model/data/real_mls/. Those samples are eval-only and must never be mixed
into the training stream.

Handles only the dataset-specific parsing. The training loop itself
(tokenization, chat-template formatting, batching) lives in train.py.
"""

from __future__ import annotations

import json
import math
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

# CubiCasa svg class name → our canonical category.
#
# Matched token-wise against the element's space-separated class string
# (see walk()): real CubiCasa classes are compound ("Wall External",
# "Door Swing Beside", "Window Regular", "Space Bedroom"), so the first
# token carries the category. "Space" is CubiCasa's room wrapper — the
# corpus does NOT use a literal "Room" class; "Room" is retained only for
# the synthetic/legacy fixture layout. Token matching (not substring)
# matters: a substring test would mis-classify "SpaceDimensionsLabel" as
# a room.
CUBI_CLASS_MAP = {
    "Wall": "wall",
    "Railing": "wall",
    "Door": "door",
    "Window": "window",
    "Room": "room",
    "Space": "room",
}

# CubiCasa room-class token → our canonical room label vocabulary.
#
# Keys are the actual tokens the CubiCasa5k corpus uses inside its
# "Space <X> [<Y>]" room classes — verified against the real SVGs, NOT
# guessed. The corpus does NOT use compound names like "Bathroom" or
# "DiningRoom"; it uses "Bath" (often "Bath Shower") and "Dining". An
# earlier version of this map keyed on the guessed compound names, so
# token matching never fired and ~1000 bathrooms + ~190 dining rooms
# were silently dropped from every CubiCasa target. Matched token-wise
# (see _room_label_from_class) against the space-separated class string.
#
# Every non-None value here MUST appear in synthesize.US_ROOM_LABELS,
# otherwise the training target JSON contains labels the VLM emits but
# that the refiner's in-vocab normalizer and MIN_ROOM_DIMS don't
# recognize — vocabulary drift, the bug Cluster E partially fixed.
#
# Remapping rationale:
#   "Bath"    → "bathroom"  (corpus token; "Bath Shower" matches too)
#   "Dining"  → "dining_room"
#   "Laundry" → "laundry_room" (from "Utility Laundry")
#   "Storage" → "closet"    (US-vocab analog of a small enclosed store)
#   "Den"     → "den"
#   "Outdoor" → None        (terraces/balconies/porches all carry the
#   "Balcony" → None         "Outdoor" token — not interior rooms)
#   "Undefined" / "UserDefined" → None  (genuinely unknown is best
#                           signalled by absence — a generic catch-all
#                           label poisons the vocabulary)
#
# Tokens absent here (Sauna, Attic, Alcove, TechnicalRoom, the generic
# "Space Room") match nothing and fall through to None — the room is
# skipped. Its surrounding walls/doors are still emitted, so the
# boundary shows up geometrically but isn't labelled as a ghost
# category.
CUBI_ROOM_LABELS: dict[str, str | None] = {
    "LivingRoom": "living_room",
    "Kitchen": "kitchen",
    "Bedroom": "bedroom",
    "Bath": "bathroom",
    "Dining": "dining_room",
    "Hallway": "hallway",
    "Corridor": "hallway",
    "Entry": "foyer",
    "Storage": "closet",
    "Closet": "closet",
    "Garage": "garage",
    "Laundry": "laundry_room",
    "Den": "den",
    "Outdoor": None,
    "Balcony": None,
    "Undefined": None,
    "UserDefined": None,
}


@dataclass
class Sample:
    image_path: Path
    target_json: str          # serialized FloorPlan, canonical format
    source: str               # dataset name, for weighting/logging


class CubiCasaLoader:
    """Iterate (image, target_json) pairs from a CubiCasa5k directory tree.

    `split` selects the official CubiCasa5k fold: "train" / "val" / "test"
    read the matching `{split}.txt` in the dataset root and yield ONLY the
    sample dirs listed there. `split=None` (default) walks every sample —
    convenient for fixtures and ad-hoc inspection, but NEVER use None for
    training against this corpus or the official test fold leaks into
    training. `train.py` passes split="train" so test/val stay held out.
    """

    def __init__(self, root: str | Path, pixels_per_meter: float = 50.0,
                 split: str | None = None):
        self.root = Path(root)
        self.ppm = pixels_per_meter
        self.split = split
        self._allowed = self._load_split(split) if split else None

    def _load_split(self, split: str) -> set:
        """Resolve the official split file to a set of sample dirs."""
        split_file = self.root / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(
                f"CubiCasa split file not found: {split_file} "
                f"(expected train.txt/val.txt/test.txt in the dataset root)"
            )
        allowed = set()
        for line in split_file.read_text().splitlines():
            rel = line.strip().strip("/")
            if rel:
                allowed.add((self.root / rel).resolve())
        return allowed

    def __iter__(self) -> Iterator[Sample]:
        if not self.root.exists():
            raise FileNotFoundError(f"CubiCasa5k root not found: {self.root}")
        for sample_dir in sorted(self.root.rglob("model.svg")):
            sample_dir = sample_dir.parent
            if self._allowed is not None and sample_dir.resolve() not in self._allowed:
                continue
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

        # CubiCasa SVGs nest walls/rooms inside <g transform="translate(...)">
        # (and occasionally matrix(...)) groups — coordinates in the child
        # elements are in the innermost local frame. A plain root.iter() loses
        # the parent chain, so we walk the tree ourselves and compose
        # transforms along the way.
        def walk(elem, ctm):
            ctm = _compose(ctm, _parse_transform(elem.attrib.get("transform", "")))
            cls = elem.attrib.get("class", "")
            tokens = cls.split()
            kind = next((v for k, v in CUBI_CLASS_MAP.items() if k in tokens), None)
            polygon = _element_geometry(elem) if kind is not None else []
            if kind is not None and polygon:
                polygon = [_apply(ctm, p) for p in polygon]
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
                    if label is None:
                        # Skip categories that don't map into our US
                        # vocabulary (Outdoor / Balcony / Undefined).
                        # Walls and doors around the skipped polygon
                        # are still emitted — the boundary stays
                        # geometrically present, just unlabeled.
                        pass
                    else:
                        metric_poly = [[p[0] / self.ppm, p[1] / self.ppm] for p in polygon]
                        rooms.append(
                            {
                                "label": label,
                                "polygon": metric_poly,
                                "area": _polygon_area(metric_poly),
                            }
                        )
            for child in elem:
                walk(child, ctm)

        walk(root, _IDENTITY)

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


def _element_geometry(elem) -> list[tuple[float, float]]:
    """Return a classified element's geometry, in the element's own frame.

    Real CubiCasa wraps every wall / door / window / room in a classified
    `<g>` whose actual shape is a *bare* child `<polygon>` / `<rect>` /
    `<path>` (no class) — the classed group itself has no geometry. The
    older synthetic/fixture layout instead puts the geometry directly on
    the classed element. Handle both:

      1. If the element itself is a geometry primitive, use it (fixture).
      2. Otherwise return the first bare child primitive (real corpus).

    Children that carry their own class are skipped: a door / window is
    nested inside its wall's `<g>`, and pulling the door's polygon up
    into the wall would corrupt both. Those nested entities are emitted
    on their own when walk() recurses into them.

    A child primitive may carry its own transform; it is composed in here
    so the returned points are in `elem`'s frame, which the caller then
    maps through the ancestor ctm.
    """
    pts = _parse_polygon(elem)
    if pts:
        return pts
    for child in elem:
        tag = child.tag.replace(_SVG_NS, "")
        if tag not in ("polygon", "rect", "path"):
            continue
        if child.attrib.get("class"):
            continue  # a classified primitive is its own entity, not this one's shape
        pts = _parse_polygon(child)
        if not pts:
            continue
        ct = _parse_transform(child.attrib.get("transform", ""))
        if ct != _IDENTITY:
            pts = [_apply(ct, p) for p in pts]
        return pts
    return []


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


# ---------- SVG transforms ----------
#
# Minimal affine-transform machinery so we can compose the ancestor chain
# of `transform="translate(...)"` / `matrix(...)` / `scale(...)` / `rotate(...)`
# attributes that wrap most CubiCasa elements. Represented as a 6-tuple
# (a, b, c, d, e, f) corresponding to the matrix
#     | a  c  e |
#     | b  d  f |
#     | 0  0  1 |
# which is the same ordering SVG's matrix() uses.

_IDENTITY: tuple[float, float, float, float, float, float] = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def _compose(parent, child):
    """Return parent * child (parent applied after child, as SVG nests)."""
    a1, b1, c1, d1, e1, f1 = parent
    a2, b2, c2, d2, e2, f2 = child
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def _apply(m, p):
    a, b, c, d, e, f = m
    x, y = p
    return (a * x + c * y + e, b * x + d * y + f)


_TRANSFORM_RE = re.compile(r"(translate|matrix|scale|rotate)\s*\(([^)]*)\)")


def _parse_transform(s: str):
    """Parse an SVG transform attribute. Handles the four forms that
    appear in CubiCasa; returns identity for unrecognized input so the
    loader degrades gracefully on novel corpora."""
    if not s:
        return _IDENTITY
    m = _IDENTITY
    for op, args in _TRANSFORM_RE.findall(s):
        nums = [float(x) for x in re.split(r"[,\s]+", args.strip()) if x]
        if op == "translate":
            tx = nums[0] if nums else 0.0
            ty = nums[1] if len(nums) > 1 else 0.0
            step = (1.0, 0.0, 0.0, 1.0, tx, ty)
        elif op == "matrix" and len(nums) == 6:
            step = tuple(nums)  # type: ignore[assignment]
        elif op == "scale":
            sx = nums[0] if nums else 1.0
            sy = nums[1] if len(nums) > 1 else sx
            step = (sx, 0.0, 0.0, sy, 0.0, 0.0)
        elif op == "rotate":
            angle = math.radians(nums[0]) if nums else 0.0
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            if len(nums) == 3:
                cx, cy = nums[1], nums[2]
                # rotate around (cx, cy) = translate(cx,cy) * R * translate(-cx,-cy)
                step = (
                    cos_a, sin_a, -sin_a, cos_a,
                    cx - cos_a * cx + sin_a * cy,
                    cy - sin_a * cx - cos_a * cy,
                )
            else:
                step = (cos_a, sin_a, -sin_a, cos_a, 0.0, 0.0)
        else:
            continue
        m = _compose(m, step)
    return m


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


def _room_label_from_class(class_attr: str) -> str | None:
    """Return the canonical US_ROOM_LABELS value for a CubiCasa svg class
    string, or None if the class matches a None-mapped category (Outdoor,
    Balcony, Undefined — not interior rooms) or doesn't match anything in
    CUBI_ROOM_LABELS at all. Callers must skip the room when None is
    returned — previously this returned a literal "room" string that
    leaked an out-of-vocabulary label into the training target.

    Matched token-wise: CubiCasa room classes are space-separated
    ("Space Bath Shower", "Space Entry Lobby"), so we test whole tokens
    rather than substrings. A substring test would let "Bath" match
    inside an unrelated token and, worse, made the earlier compound-name
    keys ("Bathroom") fail to match the corpus's actual "Bath" token.
    Dict order disambiguates multi-token classes — "Room LivingRoom"
    resolves via the earlier "LivingRoom" key, not a generic fallback.
    """
    tokens = class_attr.split()
    for key, label in CUBI_ROOM_LABELS.items():
        if key in tokens:
            return label
    return None


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
    cubicasa_split: str | None = None,
) -> list[Sample]:
    """Walk all available datasets and return a combined list of Samples.

    Pass `cubicasa_split="train"` to train only on the official train fold
    and keep val/test held out (required for a valid CubiCasa benchmark).
    `cubicasa_split=None` loads every sample — only safe for fixtures.

    ResPlan and CFP loaders will be added as separate methods when we
    download those corpora — the current bottleneck is CubiCasa5k + synth.
    """
    samples: list[Sample] = []
    if cubicasa_root:
        samples.extend(list(CubiCasaLoader(cubicasa_root, split=cubicasa_split)))
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


# ---------- real-MLS eval set ----------

# Image extensions a hand-labelled sample might use. We allow both so
# labelers aren't forced to re-encode vendor PDFs/JPEGs.
_REAL_MLS_IMAGE_EXTS = (".png", ".jpg", ".jpeg")


class RealMLSLoader:
    """Iterate held-out real MLS / listing floor plans.

    Directory layout (see model/data/real_mls/README.md):
        root/
        └── samples/
            ├── {slug}.png     (or .jpg)
            └── {slug}.json    (canonical schema)

    Every returned Sample is validated against `schema.validate` so a
    malformed label fails loud rather than silently producing 0-metric
    rows in eval.
    """

    def __init__(self, root: str | Path):
        # Accept either the dataset root or the samples/ dir directly, so
        # callers don't have to remember which layer the CLI expects.
        root = Path(root)
        if (root / "samples").is_dir():
            root = root / "samples"
        self.root = root

    def __iter__(self) -> Iterator[Sample]:
        if not self.root.exists():
            return
        seen = set()
        for ext in _REAL_MLS_IMAGE_EXTS:
            for img in sorted(self.root.glob(f"*{ext}")):
                if img.stem in seen:
                    continue  # prefer the first extension we hit per stem
                seen.add(img.stem)
                target = img.with_suffix(".json")
                if not target.exists():
                    # A sample without a label is incomplete — skip loudly
                    # so the labeler notices.
                    print(f"[real_mls] missing label for {img.name}")
                    continue
                text = target.read_text().strip()
                # Validate eagerly so eval never runs against a broken row.
                try:
                    from schema import deserialize  # type: ignore
                    deserialize(text)
                except Exception as e:
                    print(f"[real_mls] invalid label for {img.name}: {e}")
                    continue
                yield Sample(image_path=img, target_json=text, source="real_mls")


def build_eval_set(
    real_mls_root: str | Path | None = None,
    cubicasa_root: str | Path | None = None,
    cubicasa_split: str = "test",
) -> list[Sample]:
    """Return the held-out eval set. Kept separate from `build_training_set`
    so it's structurally impossible to accidentally train on eval data.

    `cubicasa_root` scores against the official CubiCasa fold named by
    `cubicasa_split` (default "test"). Ground truth comes from the same
    SVG->JSON converter used in training, so the VLM and any baseline are
    scored in one metric space. Only valid if the trained model held this
    fold out — see `build_training_set(cubicasa_split="train")`.
    """
    samples: list[Sample] = []
    if real_mls_root:
        samples.extend(list(RealMLSLoader(real_mls_root)))
    if cubicasa_root:
        samples.extend(list(CubiCasaLoader(cubicasa_root, split=cubicasa_split)))
    return samples


def _smoke_report(name: str, samples: Iterator[Sample], limit: int | None) -> None:
    """Consume `samples` (lazily) and print parse-rate + element histograms.
    Shared between the CubiCasa and real-MLS CLIs so the two corpora always
    get compared on the same axes."""
    total = empty_rooms = empty_walls = 0
    walls_hist: list[int] = []
    doors_hist: list[int] = []
    windows_hist: list[int] = []
    rooms_hist: list[int] = []
    label_counts: dict[str, int] = {}

    for i, sample in enumerate(samples):
        if limit is not None and i >= limit:
            break
        total += 1
        plan = json.loads(sample.target_json)
        walls_hist.append(len(plan["walls"]))
        doors_hist.append(len(plan["doors"]))
        windows_hist.append(len(plan["windows"]))
        rooms_hist.append(len(plan["rooms"]))
        if not plan["walls"]:
            empty_walls += 1
        if not plan["rooms"]:
            empty_rooms += 1
        for r in plan["rooms"]:
            label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1

    def stats(xs: list[int]) -> str:
        if not xs:
            return "n=0"
        return f"min={min(xs)} med={sorted(xs)[len(xs) // 2]} max={max(xs)} mean={sum(xs)/len(xs):.1f}"

    print(f"[{name}] parsed {total} samples")
    print(f"  walls:   {stats(walls_hist)}  (empty: {empty_walls})")
    print(f"  doors:   {stats(doors_hist)}")
    print(f"  windows: {stats(windows_hist)}")
    print(f"  rooms:   {stats(rooms_hist)}  (empty: {empty_rooms})")
    if label_counts:
        top = sorted(label_counts.items(), key=lambda kv: -kv[1])[:10]
        print("  top room labels:")
        for label, count in top:
            print(f"    {label:20s} {count}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cubicasa", help="path to a CubiCasa5k extraction")
    ap.add_argument("--real-mls", help="path to a real_mls dataset root")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N samples per corpus (default: all)")
    args = ap.parse_args()

    if not args.cubicasa and not args.real_mls:
        ap.error("pass at least one of --cubicasa or --real-mls")

    if args.cubicasa:
        _smoke_report("cubicasa5k", iter(CubiCasaLoader(args.cubicasa)), args.limit)
    if args.real_mls:
        _smoke_report("real_mls", iter(RealMLSLoader(args.real_mls)), args.limit)
