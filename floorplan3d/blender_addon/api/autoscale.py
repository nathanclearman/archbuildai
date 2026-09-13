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


def _facing_spans(lines, min_overlap_frac: float = 0.5):
    """Pairs of parallel lines that face each other (their extents overlap
    along the line direction), as (coord_lo, coord_hi, along_lo, along_hi).
    A dimension line always measures between two such facing walls."""
    spans = []
    n = len(lines)
    for i in range(n):
        ci, lo_i, hi_i = lines[i]
        for j in range(i + 1, n):
            cj, lo_j, hi_j = lines[j]
            overlap = min(hi_i, hi_j) - max(lo_i, lo_j)
            if overlap <= min_overlap_frac * max(1e-6, min(hi_i - lo_i, hi_j - lo_j)):
                continue
            if abs(ci - cj) < 1e-6:
                continue
            spans.append((min(ci, cj), max(ci, cj), max(lo_i, lo_j), min(hi_i, hi_j)))
    return spans


def estimate_pixels_per_meter(rooms: list[dict], ppm_guess: float,
                              dimension_texts: list[tuple[str, list[float]]],
                              walls: list[dict] | None = None,
                              min_pairs: int = 3, min_frac: float = 0.15,
                              agreement: float = 0.12) -> tuple[float | None, dict]:
    """Estimate the plan's pixels-per-metre from dimension strings.

    rooms/walls: YOLO output in metres at `ppm_guess` (metres = px / ppm_guess).
    dimension_texts: [(text, [x1, y1, x2, y2] in original image pixels)].

    A dimension line measures the distance between two facing walls, and
    its string sits somewhere along that span. For a candidate scale p we
    look, for each string of value v, for a facing wall pair whose pixel
    span matches v·p (within a tolerance) and which reaches the string's
    position; the string's own box only has to lie within the span (padded
    by a quarter, since OCR boxes for rotated text drift). Strong evidence
    = such positioned matches; weak evidence = a value that matches SOME
    facing span anywhere on the plan (used at half weight, for the big
    exterior dimensions whose boxes the OCR mislocates). The best-scoring p
    on a fine log grid wins; the returned scale is the median of the pixel
    spans over the matched values.

    Returns (ppm or None, report). None when fewer than max(min_pairs,
    min(8, min_frac × candidates)) distinct values agree.
    """
    report = {"parsed": 0, "used": 0, "estimates": []}
    polys = [r for r in (rooms or []) if len(r.get("polygon") or []) >= 3]
    if (not polys and not walls) or not dimension_texts:
        return None, report
    xs = [q[0] for r in polys for q in r["polygon"]] + [p[0] for w in (walls or []) for p in (w["start"], w["end"])]
    ys = [q[1] for r in polys for q in r["polygon"]] + [p[1] for w in (walls or []) for p in (w["start"], w["end"])]
    fp_short_px = max(1e-6, min(max(xs) - min(xs), max(ys) - min(ys))) * ppm_guess
    fp_long_px = max(max(xs) - min(xs), max(ys) - min(ys)) * ppm_guess
    tol = 0.03 * fp_short_px
    vertical, horizontal = _axis_lines(polys if not walls else [], walls or [], ppm_guess)
    # horizontal spans (between vertical lines) measure horizontal strings, and vice versa
    h_spans = [sp for sp in _facing_spans(vertical) if sp[1] - sp[0] > 2 * tol]
    v_spans = [sp for sp in _facing_spans(horizontal) if sp[1] - sp[0] > 2 * tol]

    dims = []
    for text, box in dimension_texts:
        metres = parse_dimension(text)
        if metres is None:
            continue
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        horizontal_text = (box[2] - box[0]) >= (box[3] - box[1])
        dims.append((metres, cx, cy, horizontal_text))
    report["parsed"] = len(dims)
    big = [d for d in dims if d[0] >= 1.5]
    if len(big) >= min_pairs:
        dims = big
    report["candidates"] = len(dims)
    if len(dims) < min_pairs:
        return None, report

    def evaluate(p: float):
        """(score, strong implied ppm list, weak implied ppm list) for scale p.
        One vote per string: the single facing span that best fits it."""
        strong = []
        strong_values = set()
        strong_values_list = []
        weak: dict = {}
        for metres, cx, cy, horiz in dims:
            want = metres * p
            best_pos = None      # (midpoint distance, implied)
            for spans, coord, along in ((h_spans, cx, cy), (v_spans, cy, cx)):
                for lo, hi, alo, ahi in spans:
                    span = hi - lo
                    if abs(span - want) > tol:
                        continue
                    inside = lo - tol <= coord <= hi + tol
                    near = alo - 4 * tol <= along <= ahi + 4 * tol
                    if inside and near:
                        d = abs(coord - (lo + hi) / 2)
                        if best_pos is None or d < best_pos[0]:
                            best_pos = (d, span / metres)
                    else:
                        weak.setdefault(round(metres, 2), span / metres)
            if best_pos is not None:
                # OCR loops emit one string many times; cap its votes.
                if sum(1 for v in strong_values_list if v == round(metres, 2)) < 3:
                    strong.append(best_pos[1])
                    strong_values_list.append(round(metres, 2))
                strong_values.add(round(metres, 2))
        weak_only = [e for v, e in weak.items() if v not in strong_values]
        return len(strong) + 0.5 * len(weak_only), strong, weak_only

    max_m = max(d[0] for d in dims)
    lo_p = max(2.0, 4 * tol / max_m)
    hi_p = max(lo_p * 1.5, fp_long_px / max(0.5, max_m))
    best = (0.0, [], [])
    p = lo_p
    while p <= hi_p:
        res = evaluate(p)
        if res[0] > best[0]:
            best = res
        p *= 1.01
    score, strong, weak = best
    needed = max(min_pairs, min(8, int(math.ceil(min_frac * len(dims)))))
    pool = strong if len(strong) >= min_pairs else strong + weak
    report["estimates"] = [round(e, 1) for e in pool]
    report["strong"] = len(strong)
    report["weak"] = len(weak)
    if len(pool) < needed:
        return None, report
    med = statistics.median(pool)
    agreeing = [e for e in pool if abs(e - med) <= agreement * med]
    report["used"] = len(agreeing)
    if len(agreeing) < needed:
        return None, report
    ppm = statistics.median(agreeing)
    report["ppm"] = round(ppm, 2)
    return ppm, report


def footprint_px(plan: dict, ppm_guess: float) -> list[float] | None:
    """Bounding box of the building in image pixels (rooms + walls), or None."""
    pts = [q for r in plan.get("rooms", []) for q in (r.get("polygon") or [])]
    pts += [p for w in plan.get("walls", []) for p in (w["start"], w["end"])]
    if not pts:
        return None
    xs = [p[0] * ppm_guess for p in pts]
    ys = [p[1] * ppm_guess for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


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
