"""
Post-processing for CV (YOLO) floor plans: make the outside shape match.

YOLO reads dimension lines, leader arrows and balcony hatching as walls,
which puts stray wall segments outside the building, and its segmentation
leaves sliver polygons along thick wall bands. Both are fixed here using
the room polygons as the source of truth for the building footprint:

  * a wall (or the part of a wall) further than `margin` from every room
    polygon is outside the building and is trimmed or dropped;
  * rooms that are tiny relative to the plan are dropped as slivers.

Pure Python, no bpy / shapely — unit-tested outside Blender. Distances are
in plan units (metres at the pixels-per-meter the caller chose), but the
thresholds are relative to the plan so a wrong scale guess doesn't break
them.
"""

from __future__ import annotations

import math

try:  # inside the add-on package
    from .hybrid import _dist_point_polygon
except ImportError:  # standalone (tests put api/ on sys.path)
    from hybrid import _dist_point_polygon  # type: ignore


def _polygon_area(poly) -> float:
    n = len(poly)
    if n < 3:
        return 0.0
    a = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def _short_side(poly) -> float:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(max(xs) - min(xs), max(ys) - min(ys))


def _dist_to_rooms(pt, polys) -> float:
    return min((_dist_point_polygon(pt, p) for p in polys), default=math.inf)


def drop_sliver_rooms(rooms: list[dict], area_frac: float = 0.005,
                      side_frac: float = 0.03) -> tuple[list[dict], int]:
    """Drop room polygons that are negligible relative to the whole plan.

    A room is a sliver if its area is under `area_frac` of the total room
    area OR its bounding-box short side is under `side_frac` of the
    footprint's short side. Relative thresholds keep this scale-agnostic.
    """
    polys = [r.get("polygon") or [] for r in rooms]
    valid = [(r, p) for r, p in zip(rooms, polys) if len(p) >= 3]
    if not valid:
        return list(rooms), 0
    total = sum(_polygon_area(p) for _, p in valid)
    xs = [q[0] for _, p in valid for q in p]
    ys = [q[1] for _, p in valid for q in p]
    fp_short = min(max(xs) - min(xs), max(ys) - min(ys))
    kept = []
    dropped = 0
    for r, p in zip(rooms, polys):
        if len(p) >= 3 and (_polygon_area(p) < area_frac * total or _short_side(p) < side_frac * fp_short):
            dropped += 1
            continue
        kept.append(r)
    return kept, dropped


def clip_walls_to_footprint(plan: dict, margin: float | None = None,
                            step: float | None = None) -> tuple[dict, dict]:
    """Trim / drop walls that lie outside the union of the room polygons.

    Each wall is sampled every `step` along its length; a sample is inside
    when it is within `margin` of some room polygon. The wall is cut down
    to its longest inside run (so a real exterior wall that YOLO extended
    along a dimension line keeps its real part) and dropped if no run is
    at least two samples long. Doors/windows are re-indexed; those on a
    dropped wall are kept with wall_index -1 if they sit inside the
    footprint (the geometry layer snaps them to the nearest wall) and
    dropped otherwise.

    Defaults scale with the plan: margin = 4% of the footprint's short
    side, step = margin / 2.
    """
    rooms = [r for r in plan.get("rooms", []) if len(r.get("polygon") or []) >= 3]
    walls = plan.get("walls", [])
    out = dict(plan)
    if not rooms or not walls:
        return out, {"walls_in": len(walls), "walls_out": len(walls), "dropped": 0, "trimmed": 0}
    polys = [r["polygon"] for r in rooms]
    xs = [q[0] for p in polys for q in p]
    ys = [q[1] for p in polys for q in p]
    fp_short = max(1e-6, min(max(xs) - min(xs), max(ys) - min(ys)))
    if margin is None:
        margin = 0.04 * fp_short
    if step is None:
        step = margin / 2.0

    new_walls: list[dict] = []
    old_to_new: dict[int, int] = {}
    dropped = trimmed = 0
    for i, w in enumerate(walls):
        (x1, y1), (x2, y2) = w["start"], w["end"]
        length = math.hypot(x2 - x1, y2 - y1)
        n = max(2, int(math.ceil(length / step)) + 1)
        ts = [k / (n - 1) for k in range(n)]
        inside = [_dist_to_rooms((x1 + t * (x2 - x1), y1 + t * (y2 - y1)), polys) <= margin for t in ts]
        # longest run of consecutive inside samples
        best_len = best_start = 0
        run_start = None
        for k, ok in enumerate(inside + [False]):
            if ok and run_start is None:
                run_start = k
            elif not ok and run_start is not None:
                if k - run_start > best_len:
                    best_len, best_start = k - run_start, run_start
                run_start = None
        if best_len < 2:
            dropped += 1
            continue
        if best_len == n:
            # Entirely inside: keep as-is, however short (interior stubs,
            # closet walls and door jambs are real).
            new_walls.append(dict(w))
        else:
            kept_len = (best_len - 1) * (length / (n - 1))
            # Partially outside and the inside part is a stub shorter than
            # 2x the margin: that's the inside end of a dimension / leader
            # line touching the building — noise, not wall.
            if kept_len < 2.0 * margin:
                dropped += 1
                continue
            trimmed += 1
            ta, tb = ts[best_start], ts[best_start + best_len - 1]
            nw = dict(w)
            nw["start"] = [round(x1 + ta * (x2 - x1), 3), round(y1 + ta * (y2 - y1), 3)]
            nw["end"] = [round(x1 + tb * (x2 - x1), 3), round(y1 + tb * (y2 - y1), 3)]
            new_walls.append(nw)
        old_to_new[i] = len(new_walls) - 1

    def remap(items: list[dict]) -> list[dict]:
        kept = []
        for it in items:
            it = dict(it)
            idx = it.get("wall_index", -1)
            if isinstance(idx, int) and idx in old_to_new:
                it["wall_index"] = old_to_new[idx]
                kept.append(it)
                continue
            pos = it.get("position")
            if isinstance(pos, (list, tuple)) and len(pos) == 2 and _dist_to_rooms(pos, polys) <= margin:
                it["wall_index"] = -1
                kept.append(it)
        return kept

    out["walls"] = new_walls
    out["doors"] = remap(plan.get("doors", []))
    out["windows"] = remap(plan.get("windows", []))
    return out, {"walls_in": len(walls), "walls_out": len(new_walls), "dropped": dropped,
                 "trimmed": trimmed, "margin": round(margin, 3)}


def _dist_point_segment(pt, a, b) -> float:
    px, py = pt
    (x1, y1), (x2, y2) = a, b
    dx, dy = x2 - x1, y2 - y1
    lsq = dx * dx + dy * dy
    t = 0.0 if lsq == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / lsq))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def footprint_boundary_edges(rooms: list[dict], res: float) -> list[tuple[list[float], list[float]]]:
    """Axis-aligned edges of the building outline, from the room polygons.

    Rasterizes the union of the rooms at `res`, takes every cell side that
    separates inside from outside, and merges collinear runs. Good enough
    for the rectilinear plans YOLO produces; a diagonal outline comes back
    as a staircase, which the coverage test below tolerates.
    """
    polys = [r["polygon"] for r in rooms if len(r.get("polygon") or []) >= 3]
    if not polys:
        return []
    xs = [q[0] for p in polys for q in p]
    ys = [q[1] for p in polys for q in p]
    x0, y0 = min(xs) - res, min(ys) - res
    nx = int(math.ceil((max(xs) - x0) / res)) + 2
    ny = int(math.ceil((max(ys) - y0) / res)) + 2
    inside = [[False] * nx for _ in range(ny)]
    for j in range(ny):
        cy = y0 + (j + 0.5) * res
        for i in range(nx):
            cx = x0 + (i + 0.5) * res
            inside[j][i] = any(_dist_point_polygon((cx, cy), p) <= res * 0.5 for p in polys)

    def is_in(i, j):
        return 0 <= i < nx and 0 <= j < ny and inside[j][i]

    # Horizontal boundary runs (top/bottom sides) keyed by y-line, vertical by x-line.
    horiz: dict[int, list[int]] = {}
    vert: dict[int, list[int]] = {}
    for j in range(ny):
        for i in range(nx):
            if not inside[j][i]:
                continue
            if not is_in(i, j - 1):
                horiz.setdefault(j, []).append(i)          # bottom side at y = y0 + j*res
            if not is_in(i, j + 1):
                horiz.setdefault(-(j + 1) - 1, []).append(i)  # top side at y = y0 + (j+1)*res (negative key)
            if not is_in(i - 1, j):
                vert.setdefault(i, []).append(j)
            if not is_in(i + 1, j):
                vert.setdefault(-(i + 1) - 1, []).append(j)
    edges = []
    for key, cells in horiz.items():
        line = key if key >= 0 else -key - 1
        y = y0 + line * res
        cells.sort()
        start = prev = cells[0]
        for c in cells[1:] + [None]:
            if c is not None and c == prev + 1:
                prev = c
                continue
            edges.append(([x0 + start * res, y], [x0 + (prev + 1) * res, y]))
            if c is not None:
                start = prev = c
    for key, cells in vert.items():
        line = key if key >= 0 else -key - 1
        x = x0 + line * res
        cells.sort()
        start = prev = cells[0]
        for c in cells[1:] + [None]:
            if c is not None and c == prev + 1:
                prev = c
                continue
            edges.append(([x, y0 + start * res], [x, y0 + (prev + 1) * res]))
            if c is not None:
                start = prev = c
    return edges


def fill_missing_exterior_walls(plan: dict, margin: float | None = None) -> tuple[dict, int]:
    """Add walls along stretches of the building outline no wall covers.

    YOLO regularly detects the windows on an exterior wall but not the wall
    itself, which leaves the building open on that side. For every outline
    edge (from the room polygons) we sample points and check whether some
    existing wall passes within `margin`; uncovered runs longer than
    2 * margin get a new wall of the plan's typical thickness. Returns
    (plan, walls_added); never mutates the input.
    """
    rooms = plan.get("rooms", [])
    walls = list(plan.get("walls", []))
    polys = [r["polygon"] for r in rooms if len(r.get("polygon") or []) >= 3]
    if not polys:
        return dict(plan), 0
    xs = [q[0] for p in polys for q in p]
    ys = [q[1] for p in polys for q in p]
    fp_short = max(1e-6, min(max(xs) - min(xs), max(ys) - min(ys)))
    if margin is None:
        margin = 0.04 * fp_short
    res = margin
    thick = sorted(w.get("thickness", 0.15) for w in walls)
    thickness = thick[len(thick) // 2] if thick else 0.15
    segs = [(w["start"], w["end"]) for w in walls]
    added = []
    for a, b in footprint_boundary_edges(rooms, res):
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(2, int(math.ceil(length / (margin / 2))) + 1)
        pts = [(a[0] + k / (n - 1) * (b[0] - a[0]), a[1] + k / (n - 1) * (b[1] - a[1])) for k in range(n)]
        covered = [any(_dist_point_segment(pt, s, e) <= margin for s, e in segs) for pt in pts]
        run_start = None
        for k, ok in enumerate(covered + [True]):
            if not ok and run_start is None:
                run_start = k
            elif ok and run_start is not None:
                run_len = (k - run_start - 1) * (length / (n - 1))
                if run_len >= 2.0 * margin:
                    p0, p1 = pts[run_start], pts[k - 1]
                    added.append({"start": [round(p0[0], 3), round(p0[1], 3)],
                                  "end": [round(p1[0], 3), round(p1[1], 3)],
                                  "thickness": thickness, "source": "footprint_fill"})
                run_start = None
    out = dict(plan)
    out["walls"] = walls + added
    return out, len(added)


def clean_cv_plan(plan: dict) -> tuple[dict, dict]:
    """Full CV post-pass: drop sliver rooms, clip walls to the footprint,
    then close any open stretch of the outline with a new exterior wall."""
    rooms, slivers = drop_sliver_rooms(plan.get("rooms", []))
    out = dict(plan)
    out["rooms"] = rooms
    out, rep = clip_walls_to_footprint(out)
    out, added = fill_missing_exterior_walls(out)
    rep["sliver_rooms_dropped"] = slivers
    rep["exterior_walls_added"] = added
    return out, rep
