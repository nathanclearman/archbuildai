"""
Held-out eval harness for floor plan predictors.

Takes an iterable of (image, ground_truth_plan) samples — normally the
real_mls eval set from dataset.build_eval_set — runs a predictor on each
image, compares the prediction to the ground-truth plan, and prints a
per-sample and aggregate numeric report.

The predictor is pluggable:
  - copy: identity baseline (returns the GT). Sanity check for the
    metric code itself; every metric must hit its best value.
  - null: returns an empty plan. Lower-bound baseline; shows the "cost
    of predicting nothing" so any trained model can be compared to it.
  - cv:   classical CV extractor (cv_walls.extract). Requires opencv;
    raises ImportError at first-call time if unavailable.
  - vlm:  the fine-tuned Qwen2.5-VL adapter (inference.run_vlm). Requires
    weights on disk; raises at first-call time if missing.

Metrics are deliberately simple and additive, so adding a predictor or
metric later is a local change, not a refactor:
  - count deltas: |pred - gt| for walls / doors / windows / rooms.
    Reported twice in the aggregate — once over all samples (so invalid
    predictions count as "predicted 0") and once over valid samples
    only (so count error isn't compressed against parse failures).
  - total-wall-length ratio (pred / gt), reported as a factor, not a
    squared error, so 0.5 and 2.0 both show up as "off by 2x". None
    when either side is empty (ratio is undefined); excluded from the
    mean rather than dragged to 0.
  - room IoU coverage: sum of matched-pair IoUs divided by the larger
    of |pred_rooms|, |gt_rooms|. A predictor that emits 1 perfect room
    against a 5-room GT plan scores 0.2, not 1.0.
  - room label accuracy over matched rooms.
  - parse rate (prediction validates against schema).

No numpy / shapely / cv2 required — pure Python + PIL. Heavy predictors
import their own deps lazily.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image, ImageChops, ImageDraw

from dataset import Sample, build_eval_set  # type: ignore
from schema import validate  # type: ignore


Plan = dict
Predictor = Callable[[str], Plan]


# ---------- polygon rasterization + IoU ----------

# A polygon IoU that doesn't pull in shapely. We rasterize both polygons
# into a shared grid that covers their combined bounding box, count
# intersection and union pixels, and divide. Resolution is fixed at
# `IOU_GRID` cells across the long side of the bbox — enough precision
# for room-scale geometry (centimeter-ish), cheap to compute.

IOU_GRID = 128


def _bbox(polygon: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _combined_bbox(a: list[list[float]], b: list[list[float]]):
    ax0, ay0, ax1, ay1 = _bbox(a)
    bx0, by0, bx1, by1 = _bbox(b)
    return min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1)


def polygon_iou(a: list[list[float]], b: list[list[float]], grid: int = IOU_GRID) -> float:
    """Intersection-over-union of two metric polygons via rasterization.

    Returns 0.0 when either polygon is degenerate. Correct to roughly
    1/grid on the long side of the combined bbox.
    """
    if len(a) < 3 or len(b) < 3:
        return 0.0

    x0, y0, x1, y1 = _combined_bbox(a, b)
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return 0.0

    # Keep one cell per meter-fraction in the long dimension, so tiny
    # polygons don't collapse to a single pixel and huge ones don't
    # blow up the grid.
    scale = grid / max(w, h)
    gw = max(1, int(round(w * scale)))
    gh = max(1, int(round(h * scale)))

    def rasterize(poly):
        img = Image.new("1", (gw, gh), 0)
        pts = [(int(round((p[0] - x0) * scale)), int(round((p[1] - y0) * scale))) for p in poly]
        ImageDraw.Draw(img).polygon(pts, fill=1)
        return img

    ra, rb = rasterize(a), rasterize(b)

    # Pixel AND / OR via PIL bitmap logic.
    inter = sum(ImageChops.logical_and(ra, rb).get_flattened_data())
    union = sum(ImageChops.logical_or(ra, rb).get_flattened_data())
    if union == 0:
        return 0.0
    return inter / union


# ---------- room matching ----------

def match_rooms(pred: list[dict], gt: list[dict]) -> list[tuple[int, int, float]]:
    """Greedy max-IoU matching between predicted and GT rooms.

    Returns a list of (pred_idx, gt_idx, iou). Unmatched rooms on either
    side are dropped — callers compute recall/precision from the counts.
    Greedy rather than Hungarian because N is small (<20 rooms) and the
    matching quality difference is negligible for this use case.
    """
    if not pred or not gt:
        return []
    pairs: list[tuple[float, int, int]] = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            iou = polygon_iou(p["polygon"], g["polygon"])
            if iou > 0:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)
    used_p: set[int] = set()
    used_g: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        matches.append((i, j, iou))
        used_p.add(i)
        used_g.add(j)
    return matches


# ---------- length / count helpers ----------

def _wall_length(w: dict) -> float:
    sx, sy = w["start"]
    ex, ey = w["end"]
    return math.hypot(ex - sx, ey - sy)


def _total_wall_length(walls: list[dict]) -> float:
    return sum(_wall_length(w) for w in walls)


# ---------- per-sample evaluation ----------

@dataclass
class SampleMetrics:
    slug: str
    valid: bool
    wall_count_pred: int
    wall_count_gt: int
    door_count_pred: int
    door_count_gt: int
    window_count_pred: int
    window_count_gt: int
    room_count_pred: int
    room_count_gt: int
    # None when either pred or gt has zero total wall length: the ratio is
    # undefined there. Aggregate skips Nones rather than averaging them in
    # as 0, which would otherwise conflate "undefined" with "100x off".
    wall_length_ratio: float | None
    # Sum of matched-pair IoUs divided by max(|pred_rooms|, |gt_rooms|), so
    # unmatched rooms on either side drag the score down. 0 when both sides
    # are empty.
    room_iou_coverage: float
    # Fraction of matched rooms where pred label == gt label. 0 when no
    # rooms were matched.
    room_label_accuracy: float
    matched_rooms: int

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def evaluate_sample(slug: str, pred: dict | None, gt: dict) -> SampleMetrics:
    """Score one prediction against one ground-truth plan.

    `pred is None` (predictor raised or produced unparseable output) is
    treated as a valid evaluation data point — it shows up as zeros with
    valid=False, which is exactly the signal eval should surface.
    """
    gt_walls = gt.get("walls", [])
    gt_doors = gt.get("doors", [])
    gt_windows = gt.get("windows", [])
    gt_rooms = gt.get("rooms", [])

    if pred is None:
        return SampleMetrics(
            slug=slug,
            valid=False,
            wall_count_pred=0,
            wall_count_gt=len(gt_walls),
            door_count_pred=0,
            door_count_gt=len(gt_doors),
            window_count_pred=0,
            window_count_gt=len(gt_windows),
            room_count_pred=0,
            room_count_gt=len(gt_rooms),
            wall_length_ratio=None,
            room_iou_coverage=0.0,
            room_label_accuracy=0.0,
            matched_rooms=0,
        )

    p_walls = pred.get("walls", [])
    p_doors = pred.get("doors", [])
    p_windows = pred.get("windows", [])
    p_rooms = pred.get("rooms", [])

    gt_len = _total_wall_length(gt_walls)
    pred_len = _total_wall_length(p_walls)
    # None when either side is zero-length: the ratio is undefined there.
    # A hallucinated wall set with gt_len=0 is caught by the wall-count
    # delta, not by this ratio.
    wall_ratio = (pred_len / gt_len) if (gt_len > 0 and pred_len > 0) else None

    matches = match_rooms(p_rooms, gt_rooms)
    denom = max(len(p_rooms), len(gt_rooms))
    if matches:
        iou_coverage = sum(m[2] for m in matches) / denom
        correct_label = sum(
            1 for i, j, _ in matches if p_rooms[i]["label"] == gt_rooms[j]["label"]
        )
        label_acc = correct_label / len(matches)
    else:
        iou_coverage = 0.0
        label_acc = 0.0

    return SampleMetrics(
        slug=slug,
        valid=True,
        wall_count_pred=len(p_walls),
        wall_count_gt=len(gt_walls),
        door_count_pred=len(p_doors),
        door_count_gt=len(gt_doors),
        window_count_pred=len(p_windows),
        window_count_gt=len(gt_windows),
        room_count_pred=len(p_rooms),
        room_count_gt=len(gt_rooms),
        wall_length_ratio=(round(wall_ratio, 3) if wall_ratio is not None else None),
        room_iou_coverage=round(iou_coverage, 3),
        room_label_accuracy=round(label_acc, 3),
        matched_rooms=len(matches),
    )


# ---------- aggregation ----------

def _mean(values: list[float]) -> float:
    """Mean that returns 0.0 on empty input so the aggregate dict shape
    is stable regardless of sample count / validity. Rounds to 3dp so
    downstream diffing is readable."""
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)


def aggregate(per_sample: list[SampleMetrics]) -> dict:
    """Parse rate + mean metrics across per-sample rows.

    Count errors are reported twice: over all samples (so invalid
    predictions count as predicted-zero) and over valid samples only
    (so the count signal isn't diluted by parse failures). Ratios /
    IoU are reported only over valid samples — they're undefined
    otherwise. `wall_length_ratio` additionally skips samples where
    either pred or gt has no walls (ratio is undefined).
    """
    n = len(per_sample)
    valid = [m for m in per_sample if m.valid]

    def abs_err(field_pred: str, field_gt: str):
        return [abs(getattr(m, field_pred) - getattr(m, field_gt)) for m in per_sample]

    def abs_err_valid(field_pred: str, field_gt: str):
        return [abs(getattr(m, field_pred) - getattr(m, field_gt)) for m in valid]

    def recall(m: SampleMetrics) -> float:
        return (m.matched_rooms / m.room_count_gt) if m.room_count_gt else 0.0

    wall_ratios = [m.wall_length_ratio for m in valid if m.wall_length_ratio is not None]

    return {
        "n": n,
        "parse_rate": round(len(valid) / n, 3) if n else 0.0,
        "mean_wall_count_abs_err": _mean(abs_err("wall_count_pred", "wall_count_gt")),
        "mean_door_count_abs_err": _mean(abs_err("door_count_pred", "door_count_gt")),
        "mean_window_count_abs_err": _mean(abs_err("window_count_pred", "window_count_gt")),
        "mean_room_count_abs_err": _mean(abs_err("room_count_pred", "room_count_gt")),
        "mean_wall_count_abs_err_valid": _mean(abs_err_valid("wall_count_pred", "wall_count_gt")),
        "mean_door_count_abs_err_valid": _mean(abs_err_valid("door_count_pred", "door_count_gt")),
        "mean_window_count_abs_err_valid": _mean(abs_err_valid("window_count_pred", "window_count_gt")),
        "mean_room_count_abs_err_valid": _mean(abs_err_valid("room_count_pred", "room_count_gt")),
        "mean_wall_length_ratio": _mean(wall_ratios),
        "wall_length_ratio_defined_n": len(wall_ratios),
        "mean_room_iou_coverage": _mean([m.room_iou_coverage for m in valid]),
        "mean_room_label_accuracy": _mean([m.room_label_accuracy for m in valid]),
        "mean_matched_room_recall": _mean([recall(m) for m in valid]),
    }


# ---------- predictors ----------

def copy_predictor(samples: Iterable[Sample]) -> Predictor:
    """Sanity-check predictor: return the GT verbatim.

    Keyed by the resolved absolute image path so it's robust to caller
    path-normalization drift. On a cache miss we raise loudly — a silent
    KeyError would get swallowed by run_eval's except and show up as
    valid=False, which would look exactly like a broken model. Making
    it explicit turns "my baseline mysteriously scores zero" into
    "copy_predictor has no GT for <path>".
    """
    gt_by_path: dict[str, dict] = {}
    for s in samples:
        key = str(Path(s.image_path).resolve())
        # Two samples that resolve to the same file can happen legitimately
        # when a dataset is re-exposed under a symlinked or mounted root.
        # Warn and keep the first — raising here would refuse a legitimate
        # config. A real collision where the two samples carry different
        # GT would still be caught at comparison time (matched rooms drop).
        if key in gt_by_path:
            print(
                f"[eval] copy_predictor: two samples resolve to {key}; "
                f"keeping first",
                file=sys.stderr,
            )
            continue
        gt_by_path[key] = json.loads(s.target_json)

    def _predict(image_path: str) -> Plan:
        key = str(Path(image_path).resolve())
        if key not in gt_by_path:
            raise KeyError(f"copy_predictor has no GT for {image_path!r}")
        return gt_by_path[key]
    return _predict


def null_predictor(_samples: Iterable[Sample] | None = None) -> Predictor:
    """Floor-baseline predictor: empty plan. Shows the score of
    'predicting nothing', so any real model is anchored to a known
    worst-case. Takes samples it ignores so all builders share one
    signature — makes the dispatch table trivial."""
    def _predict(image_path: str) -> Plan:
        return {"scale": {"pixels_per_meter": 50}, "walls": [], "doors": [], "windows": [], "rooms": []}
    return _predict


def cv_predictor(_samples: Iterable[Sample] | None = None, ppm: float = 50.0) -> Predictor:
    """Classical-CV predictor (cv_walls.extract). Lazy cv2 import lives
    in cv_walls.extract itself, so constructing the predictor is cheap
    and the real ImportError only surfaces on the first prediction call
    — which is what callers expect when probing whether CV is available."""
    from cv_walls import CVConfig, extract  # type: ignore
    cfg = CVConfig(pixels_per_meter=ppm)

    def _predict(image_path: str) -> Plan:
        return extract(image_path, cfg)
    return _predict


def vlm_predictor(_samples: Iterable[Sample] | None = None,
                  weights_dir: Path | str | None = None) -> Predictor:
    """Fine-tuned VLM predictor. Same lazy-import discipline as cv.
    Default weights dir is resolved relative to this file, not CWD, so
    direct callers don't get surprised by CWD-sensitive path resolution."""
    from inference import run_vlm  # type: ignore
    weights = Path(weights_dir) if weights_dir is not None else Path(__file__).parent / "weights"

    def _predict(image_path: str) -> Plan:
        return run_vlm(image_path, weights)
    return _predict


# Builder signature: (samples, **kwargs) -> Predictor. Each entry is
# self-contained so adding a predictor is one dict line, not a new
# branch in main().
PREDICTOR_BUILDERS: dict[str, Callable[..., Predictor]] = {
    "copy": copy_predictor,
    "null": null_predictor,
    "cv": cv_predictor,
    "vlm": vlm_predictor,
}


# ---------- runner ----------

def run_eval(samples: Iterable[Sample], predict: Predictor) -> tuple[list[SampleMetrics], dict]:
    per_sample: list[SampleMetrics] = []
    for s in samples:
        gt = json.loads(s.target_json)
        try:
            pred = predict(str(s.image_path))
            validate(pred)
        except Exception as e:
            # Any predictor failure (missing deps, malformed output, schema
            # violation) becomes a row with valid=False, so eval always
            # produces a complete report instead of bailing on the first error.
            print(f"[eval] {s.image_path.name}: predictor failed: {e}", file=sys.stderr)
            pred = None
        m = evaluate_sample(s.image_path.stem, pred, gt)
        per_sample.append(m)
    agg = aggregate(per_sample)
    return per_sample, agg


def format_report(per_sample: list[SampleMetrics], agg: dict) -> str:
    lines: list[str] = []
    lines.append("per-sample:")
    if not per_sample:
        lines.append("  (no samples)")
    for m in per_sample:
        ratio = "n/a" if m.wall_length_ratio is None else m.wall_length_ratio
        lines.append(
            f"  {m.slug:20s} valid={m.valid} "
            f"walls={m.wall_count_pred}/{m.wall_count_gt} "
            f"doors={m.door_count_pred}/{m.door_count_gt} "
            f"wins={m.window_count_pred}/{m.window_count_gt} "
            f"rooms={m.room_count_pred}/{m.room_count_gt} "
            f"wall_len_ratio={ratio} "
            f"room_iou_cov={m.room_iou_coverage} "
            f"label_acc={m.room_label_accuracy}"
        )
    lines.append("aggregate:")
    for k, v in agg.items():
        lines.append(f"  {k:36s} {v}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real-mls", required=True, help="path to real_mls dataset root")
    ap.add_argument("--predictor", choices=tuple(PREDICTOR_BUILDERS), default="null")
    ap.add_argument("--weights", default=str(Path(__file__).parent / "weights"),
                    help="VLM weights dir (only used with --predictor vlm)")
    ap.add_argument("--ppm", type=float, default=50.0,
                    help="pixels-per-meter prior for the CV predictor")
    ap.add_argument("--json", action="store_true",
                    help="emit machine-readable JSON instead of the human report")
    args = ap.parse_args()

    samples = build_eval_set(real_mls_root=args.real_mls)
    if not samples:
        print(f"[eval] no samples under {args.real_mls}", file=sys.stderr)
        sys.exit(1)

    build = PREDICTOR_BUILDERS[args.predictor]
    kwargs: dict = {}
    if args.predictor == "cv":
        kwargs["ppm"] = args.ppm
    elif args.predictor == "vlm":
        kwargs["weights_dir"] = args.weights
    predict = build(samples, **kwargs)

    per_sample, agg = run_eval(samples, predict)

    if args.json:
        print(json.dumps({
            "predictor": args.predictor,
            "per_sample": [m.to_dict() for m in per_sample],
            "aggregate": agg,
        }, indent=2))
    else:
        print(f"predictor: {args.predictor}")
        print(format_report(per_sample, agg))


if __name__ == "__main__":
    main()
