"""
Automatic scale (pixels per metre) from the dimension strings printed on a plan.

MLS and CAD plans annotate rooms with dimensions ("17'9\"", "24'5\"", "4.20").
Each string sits next to, and parallel with, the edge it measures. Given the
OCR'd dimension texts with pixel boxes and the YOLO room polygons, we pair
every string with the nearest parallel room edge and derive
pixels_per_metre = edge_length_px / value_m. The median over the agreeing
pairs is the plan's scale; the whole plan is then rescaled from the user's
guess to it.

Pure Python, no bpy — unit-tested outside Blender.
"""

from __future__ import annotations

import math
import re
import statistics

FT = 0.3048
IN = 0.0254

# 17'9"  17' 9"  17'-9"  17′9″  17'9  12'  9"   (feet/inches, ASCII or typographic marks)
_FT_IN = re.compile(
    r"""^\s*(?:(?P<ft>\d{1,3})\s*(?:'|′|ft)\s*[- ]?\s*)?(?:(?P<in>\d{1,2}(?:\.\d+)?)\s*(?:"|″|''|in)?)?\s*$"""
)
# 4.20 m / 4,20 / 420 cm / 4.2
_METRIC = re.compile(r"""^\s*(?P<num>\d{1,5}(?:[.,]\d{1,2})?)\s*(?P<unit>m|cm|mm)?\s*$""", re.I)


def parse_dimension(text: str) -> float | None:
    """Return the length in metres encoded by a dimension string, or None.

    Handles imperial feet/inches in the common notations and metric values
    with or without a unit (a bare number with a decimal point is metres).
    Rejects things that are not lengths (areas, "1st floor", room names).
    """
    if not text:
        return None
    t = text.strip().replace("’", "'").replace("”", '"').replace("“", '"').replace("×", "x")
    if "x" in t.lower() and any(c.isdigit() for c in t):
        return None  # "13x15" is two dimensions; ambiguous, skip
    m = _FT_IN.match(t)
    if m and (m.group("ft") or m.group("in")) and ("'" in t or "′" in t or '"' in t or "″" in t or "ft" in t or "in" in t):
        ft = float(m.group("ft") or 0)
        inch = float(m.group("in") or 0)
        if inch >= 12 and not m.group("ft"):
            return None
        val = ft * FT + inch * IN
        return val if 0.2 <= val <= 60 else None
    m = _METRIC.match(t)
    if m:
        num = float(m.group("num").replace(",", "."))
        unit = (m.group("unit") or "").lower()
        if unit == "cm":
            val = num / 100
        elif unit == "mm":
            val = num / 1000
        elif unit == "m" or "." in m.group("num") or "," in m.group("num"):
            val = num
        else:
            return None  # bare integer: could be an area or a label number
        return val if 0.2 <= val <= 60 else None
    return None


def _axis_lines(rooms: list[dict], walls: list[dict], ppm_guess: float):
    """Vertical and horizontal line segments (in original pixels) that a
    dimension line's extension lines can land on: every axis-aligned wall
    and every axis-aligned room-polygon edge.

    Returns (vertical, horizontal): vertical = [(x, y_lo, y_hi)],
    horizontal = [(y, x_lo, x_hi)]. Diagonal segments are ignored.
    """
    segs = []
    for w in walls or []:
        segs.append((w["start"], w["end"]))
    for r in rooms or []:
        poly = r.get("polygon") or []
        for i in range(len(poly)):
            segs.append((poly[i], poly[(i + 1) % len(poly)]))
    vertical, horizontal = [], []
    for (x1, y1), (x2, y2) in segs:
        x1, y1, x2, y2 = x1 * ppm_guess, y1 * ppm_guess, x2 * ppm_guess, y2 * ppm_guess
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if dx < 1e-6 and dy < 1e-6:
            continue
        if dx <= 0.15 * dy:
            vertical.append(((x1 + x2) / 2, min(y1, y2), max(y1, y2)))
        elif dy <= 0.15 * dx:
            horizontal.append(((y1 + y2) / 2, min(x1, x2), max(x1, x2)))
    return vertical, horizontal


def _nearest_line(lines, coord: float, along: float, tol: float, reach: float):
    """Nearest line (by perpendicular coordinate) to `coord` whose extent
    comes within `reach` of `along`. Returns its coordinate or None."""
    best = None
    for c, lo, hi in lines:
        if abs(c - coord) > tol:
            continue
        gap = max(lo - along, along - hi, 0.0)
        if gap > reach:
            continue
        if best is None or abs(c - coord) < abs(best - coord):
            best = c
    return best


def estimate_pixels_per_meter(rooms: list[dict], ppm_guess: float,
                              dimension_texts: list[tuple[str, list[float]]],
                              walls: list[dict] | None = None,
                              min_pairs: int = 3, min_frac: float = 0.15,
                              agreement: float = 0.12) -> tuple[float | None, dict]:
    """Estimate the plan's pixels-per-metre from dimension strings.

    rooms/walls: YOLO output in metres at `ppm_guess` (metres = px / ppm_guess).
    dimension_texts: [(text, [x1, y1, x2, y2] in original image pixels)].

    A dimension string sits at the midpoint of the span it measures, and the
    span runs parallel to the text between two extension lines that end on
    walls. So for a candidate scale p, a string of value v metres centred at
    c predicts perpendicular walls at c ± v·p/2 along the text axis. We
    score every candidate p on a fine log grid by how many strings find a
    wall (or room edge) at BOTH predicted ends, take the best, and return
    the median of the scales implied by the matched wall positions. This
    never depends on any single polygon edge being whole, which YOLO's
    fragmented rooms cannot guarantee.

    Returns (ppm or None, report). None when fewer than max(min_pairs,
    min_frac × parsed) strings are consistent with the winning scale.
    """
    report = {"parsed": 0, "used": 0, "estimates": []}
    polys = [r for r in (rooms or []) if len(r.get("polygon") or []) >= 3]
    if (not polys and not walls) or not dimension_texts:
        return None, report
    xs = [q[0] for r in polys for q in r["polygon"]] + [p[0] for w in (walls or []) for p in (w["start"], w["end"])]
    ys = [q[1] for r in polys for q in r["polygon"]] + [p[1] for w in (walls or []) for p in (w["start"], w["end"])]
    fp_short_px = max(1e-6, min(max(xs) - min(xs), max(ys) - min(ys))) * ppm_guess
    fp_long_px = max(max(xs) - min(xs), max(ys) - min(ys)) * ppm_guess
    tol = 0.025 * fp_short_px                      # wall-position tolerance
    vertical, horizontal = _axis_lines(polys, walls or [], ppm_guess)

    dims = []
    for text, box in dimension_texts:
        metres = parse_dimension(text)
        if metres is None:
            continue
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        horizontal_text = (box[2] - box[0]) >= (box[3] - box[1])
        dims.append((metres, cx, cy, horizontal_text))
    report["parsed"] = len(dims)
    # Closet-sized strings (under 1.5 m) are the ones OCR misplaces and that
    # fit between any two wall fragments; they add noise, not evidence.
    big = [d for d in dims if d[0] >= 1.5]
    if len(big) >= min_pairs:
        dims = big
    report["candidates"] = len(dims)
    if len(dims) < min_pairs:
        return None, report

    def matches(p: float):
        """(count, implied ppm list) for candidate scale p. Matches are
        counted once per (wall pair, value): OCR loops emit the same string
        many times with shifted boxes, and letting each copy vote would let
        one hallucinated repeat outvote ten distinct real dimensions."""
        seen = set()
        implied = []
        for metres, cx, cy, horiz in dims:
            half = metres * p / 2.0
            reach = max(tol * 2, 0.5 * metres * p)
            if horiz:
                a = _nearest_line(vertical, cx - half, cy, tol, reach)
                b = _nearest_line(vertical, cx + half, cy, tol, reach)
            else:
                a = _nearest_line(horizontal, cy - half, cx, tol, reach)
                b = _nearest_line(horizontal, cy + half, cx, tol, reach)
            if a is None or b is None or abs(b - a) <= 2 * tol:
                continue
            key = (horiz, round(min(a, b) / tol), round(max(a, b) / tol), round(metres, 2))
            if key in seen:
                continue
            seen.add(key)
            implied.append(abs(b - a) / metres)
        return len(implied), implied

    # Candidate scales: the largest single dimension can't exceed the plan,
    # the smallest can't be under a few pixels.
    max_m = max(d[0] for d in dims)
    lo_p = max(2.0, 4 * tol / max_m)
    hi_p = max(lo_p * 1.5, fp_long_px / max(0.5, max_m))
    best_count, best_implied = 0, []
    p = lo_p
    while p <= hi_p:
        n, implied = matches(p)
        if n > best_count:
            best_count, best_implied = n, implied
        p *= 1.01
    # OCR over-reports (looped repeats, mis-reads), so demand a floor of
    # agreeing strings rather than a large fraction of a noisy total.
    needed = max(min_pairs, min(8, int(math.ceil(min_frac * len(dims)))))
    if best_count < needed:
        report["estimates"] = [round(e, 1) for e in best_implied]
        return None, report
    med = statistics.median(best_implied)
    agreeing = [e for e in best_implied if abs(e - med) <= agreement * med]
    report["estimates"] = [round(e, 1) for e in best_implied]
    report["used"] = len(agreeing)
    if len(agreeing) < needed:
        return None, report
    ppm = statistics.median(agreeing)
    report["ppm"] = round(ppm, 2)
    return ppm, report


def rescale_plan(plan: dict, factor: float) -> dict:
    """Multiply every length in `plan` by `factor` (areas by factor²). Copy, never mutate."""
    def pt(p):
        return [round(p[0] * factor, 3), round(p[1] * factor, 3)]

    out = dict(plan)
    out["walls"] = [dict(w, start=pt(w["start"]), end=pt(w["end"]),
                         thickness=round(w.get("thickness", 0.15) * factor, 3)) for w in plan.get("walls", [])]
    for key in ("doors", "windows"):
        items = []
        for it in plan.get(key, []):
            it = dict(it)
            if isinstance(it.get("position"), (list, tuple)) and len(it["position"]) == 2:
                it["position"] = pt(it["position"])
            for k in ("width", "height", "sill_height"):
                if isinstance(it.get(k), (int, float)):
                    it[k] = round(it[k] * factor, 3)
            items.append(it)
        out[key] = items
    rooms = []
    for r in plan.get("rooms", []):
        r = dict(r)
        r["polygon"] = [pt(p) for p in r.get("polygon", [])]
        if isinstance(r.get("area"), (int, float)):
            r["area"] = round(r["area"] * factor * factor, 3)
        rooms.append(r)
    out["rooms"] = rooms
    old = float((plan.get("scale") or {}).get("pixels_per_meter", 50.0))
    out["scale"] = dict(plan.get("scale") or {}, pixels_per_meter=round(old / factor, 3))
    return out


def apply_auto_scale(plan: dict, ppm_guess: float,
                     dimension_texts: list[tuple[str, list[float]]]) -> tuple[dict, dict]:
    """Estimate the true scale and rescale `plan` (which is in metres at
    `ppm_guess`). Returns (plan, report); unchanged plan if no estimate."""
    ppm, report = estimate_pixels_per_meter(plan.get("rooms", []), ppm_guess, dimension_texts,
                                            walls=plan.get("walls", []))
    if ppm is None:
        report["applied"] = False
        return plan, report
    factor = ppm_guess / ppm
    report["applied"] = abs(factor - 1.0) > 0.01
    report["factor"] = round(factor, 4)
    return (rescale_plan(plan, factor) if report["applied"] else plan), report
