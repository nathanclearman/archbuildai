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
    skipped with a warning if unavailable.
  - vlm:  the fine-tuned Qwen2.5-VL adapter (inference.run_vlm). Skipped
    if no weights are on disk.

Metrics are deliberately simple and additive, so adding a predictor or
metric later is a local change, not a refactor:
  - count deltas: |pred - gt| for walls / doors / windows / rooms
  - total-wall-length ratio (pred / gt), reported as a factor, not a
    squared error, so 0.5 and 2.0 both show up as "off by 2x"
  - mean room polygon IoU, via rasterized intersection-over-union with
    greedy max-IoU room matching (no shapely dep)
  - room label accuracy over matched rooms
  - parse rate (prediction validates against schema)

No numpy / shapely / cv2 required — pure Python + PIL. Heavy predictors
import their own deps lazily.
"""

from __future__ import annotations

import argparse
import json
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
    return ((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5


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
    wall_length_ratio: float  # pred / gt, 0 if gt has no walls
    mean_room_iou: float      # over matched rooms, 0 if none matched
    room_label_accuracy: float  # over matched rooms, 0 if none matched
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
            wall_length_ratio=0.0,
            mean_room_iou=0.0,
            room_label_accuracy=0.0,
            matched_rooms=0,
        )

    p_walls = pred.get("walls", [])
    p_doors = pred.get("doors", [])
    p_windows = pred.get("windows", [])
    p_rooms = pred.get("rooms", [])

    gt_len = _total_wall_length(gt_walls)
    pred_len = _total_wall_length(p_walls)
    wall_ratio = (pred_len / gt_len) if gt_len > 0 else 0.0

    matches = match_rooms(p_rooms, gt_rooms)
    if matches:
        mean_iou = sum(m[2] for m in matches) / len(matches)
        correct_label = sum(
            1 for i, j, _ in matches if p_rooms[i]["label"] == gt_rooms[j]["label"]
        )
        label_acc = correct_label / len(matches)
    else:
        mean_iou = 0.0
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
        wall_length_ratio=round(wall_ratio, 3),
        mean_room_iou=round(mean_iou, 3),
        room_label_accuracy=round(label_acc, 3),
        matched_rooms=len(matches),
    )


# ---------- aggregation ----------

def aggregate(per_sample: list[SampleMetrics]) -> dict:
    """Mean / parse-rate across per-sample metrics. Returns zeros on an
    empty input rather than raising, so callers can still print a
    predictable report shape."""
    n = len(per_sample)
    if n == 0:
        return {
            "n": 0,
            "parse_rate": 0.0,
            "mean_wall_count_abs_err": 0.0,
            "mean_door_count_abs_err": 0.0,
            "mean_window_count_abs_err": 0.0,
            "mean_room_count_abs_err": 0.0,
            "mean_wall_length_ratio": 0.0,
            "mean_room_iou": 0.0,
            "mean_room_label_accuracy": 0.0,
            "mean_matched_room_recall": 0.0,
        }

    def mean(f):
        return round(sum(f(m) for m in per_sample) / n, 3)

    valid = [m for m in per_sample if m.valid]
    valid_n = max(len(valid), 1)

    def mean_valid(f):
        return round(sum(f(m) for m in valid) / valid_n, 3)

    def recall(m: SampleMetrics) -> float:
        return (m.matched_rooms / m.room_count_gt) if m.room_count_gt else 0.0

    return {
        "n": n,
        "parse_rate": round(len(valid) / n, 3),
        "mean_wall_count_abs_err": mean(lambda m: abs(m.wall_count_pred - m.wall_count_gt)),
        "mean_door_count_abs_err": mean(lambda m: abs(m.door_count_pred - m.door_count_gt)),
        "mean_window_count_abs_err": mean(lambda m: abs(m.window_count_pred - m.window_count_gt)),
        "mean_room_count_abs_err": mean(lambda m: abs(m.room_count_pred - m.room_count_gt)),
        "mean_wall_length_ratio": mean_valid(lambda m: m.wall_length_ratio),
        "mean_room_iou": mean_valid(lambda m: m.mean_room_iou),
        "mean_room_label_accuracy": mean_valid(lambda m: m.room_label_accuracy),
        "mean_matched_room_recall": mean_valid(recall),
    }


# ---------- predictors ----------

def copy_predictor(gt_by_image: dict[str, dict]) -> Predictor:
    """Sanity-check predictor: return the GT verbatim. Every metric
    should pin to its best value. If it doesn't, the metric code is
    wrong, not the model."""
    def _predict(image_path: str) -> Plan:
        return gt_by_image[image_path]
    return _predict


def null_predictor() -> Predictor:
    """Floor-baseline predictor: empty plan. Shows the score of
    'predicting nothing', so any real model is anchored to a known
    worst-case."""
    def _predict(image_path: str) -> Plan:
        return {"scale": {"pixels_per_meter": 50}, "walls": [], "doors": [], "windows": [], "rooms": []}
    return _predict


def cv_predictor(ppm: float = 50.0) -> Predictor:
    """Classical-CV predictor (cv_walls.extract). Lazy import so the
    harness loads fine when opencv isn't installed; the CLI surfaces the
    ImportError on first use rather than at module load."""
    from cv_walls import CVConfig, extract  # type: ignore
    cfg = CVConfig(pixels_per_meter=ppm)

    def _predict(image_path: str) -> Plan:
        return extract(image_path, cfg)
    return _predict


def vlm_predictor(weights_dir: Path) -> Predictor:
    """Fine-tuned VLM predictor. Same lazy-import discipline as cv."""
    from inference import run_vlm  # type: ignore

    def _predict(image_path: str) -> Plan:
        return run_vlm(image_path, weights_dir)
    return _predict


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
        lines.append(
            f"  {m.slug:20s} valid={m.valid} "
            f"walls={m.wall_count_pred}/{m.wall_count_gt} "
            f"doors={m.door_count_pred}/{m.door_count_gt} "
            f"wins={m.window_count_pred}/{m.window_count_gt} "
            f"rooms={m.room_count_pred}/{m.room_count_gt} "
            f"wall_len_ratio={m.wall_length_ratio} "
            f"room_iou={m.mean_room_iou} "
            f"label_acc={m.room_label_accuracy}"
        )
    lines.append("aggregate:")
    for k, v in agg.items():
        lines.append(f"  {k:32s} {v}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real-mls", required=True, help="path to real_mls dataset root")
    ap.add_argument("--predictor", choices=("copy", "null", "cv", "vlm"), default="null")
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

    if args.predictor == "copy":
        gt_by_image = {str(s.image_path): json.loads(s.target_json) for s in samples}
        predict = copy_predictor(gt_by_image)
    elif args.predictor == "null":
        predict = null_predictor()
    elif args.predictor == "cv":
        predict = cv_predictor(args.ppm)
    else:
        predict = vlm_predictor(Path(args.weights))

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
