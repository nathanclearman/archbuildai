"""
Iterative Claude Vision accuracy test.

Runs the rebuilt ClaudeClient against a floor plan image, scores the result
against a ground-truth JSON, and if the score is below the acceptance
threshold invokes verify_and_repair in a loop — up to --max-iterations times
— feeding the previous attempt back to Opus each round. This is the
"keep working if it isn't accurate" behavior.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python tools/vision_accuracy_test.py \\
        --image path/to/floorplan.png \\
        --ground-truth tests/sample_plans/large_house_ground_truth.json \\
        --max-iterations 4

Exit codes:
    0 — converged (score >= threshold)
    1 — failed to converge within the iteration budget
    2 — configuration / input error
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
API_DIR = REPO_ROOT / "blender_addon" / "api"


def _load_api_modules():
    """Load the client modules without triggering blender_addon/__init__.py
    (which imports bpy and isn't available outside Blender)."""
    pkg = types.ModuleType("_fp3d_api")
    pkg.__path__ = [str(API_DIR)]
    sys.modules["_fp3d_api"] = pkg

    def _load(name):
        spec = importlib.util.spec_from_file_location(
            f"_fp3d_api.{name}", API_DIR / f"{name}.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"_fp3d_api.{name}"] = mod
        spec.loader.exec_module(mod)
        return mod

    return _load("schema"), _load("claude_client")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _bbox(walls):
    xs, ys = [], []
    for w in walls:
        xs.extend([w["start"][0], w["end"][0]])
        ys.extend([w["start"][1], w["end"][1]])
    if not xs:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs), min(ys), max(xs), max(ys))


def _iou_bbox(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _wall_length(w):
    dx = w["end"][0] - w["start"][0]
    dy = w["end"][1] - w["start"][1]
    return (dx * dx + dy * dy) ** 0.5


def score(predicted, ground_truth):
    """Return a 0-1 accuracy score and a breakdown dict.

    Metric mix:
      * Bounding-box IoU of wall extents (50%)
      * Wall-count closeness (20%)
      * Total perimeter length ratio (20%)
      * Room count closeness (10%)
    """
    pw = predicted.get("walls", []) or []
    gw = ground_truth.get("walls", []) or []

    iou = _iou_bbox(_bbox(pw), _bbox(gw))

    pc, gc = len(pw), len(gw)
    count_score = 1.0 - min(1.0, abs(pc - gc) / max(1, gc))

    pl = sum(_wall_length(w) for w in pw)
    gl = sum(_wall_length(w) for w in gw)
    length_score = 1.0 - min(1.0, abs(pl - gl) / max(1e-6, gl))

    pr = len(predicted.get("rooms", []) or [])
    gr = len(ground_truth.get("rooms", []) or [])
    room_score = 1.0 - min(1.0, abs(pr - gr) / max(1, gr))

    total = 0.5 * iou + 0.2 * count_score + 0.2 * length_score + 0.1 * room_score

    return total, {
        "bbox_iou": round(iou, 3),
        "wall_count_score": round(count_score, 3),
        "length_ratio_score": round(length_score, 3),
        "room_count_score": round(room_score, 3),
        "predicted_walls": pc,
        "ground_truth_walls": gc,
        "predicted_perimeter_m": round(pl, 2),
        "ground_truth_perimeter_m": round(gl, 2),
    }


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run(args):
    fp_schema, claude_client = _load_api_modules()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: set ANTHROPIC_API_KEY or pass --api-key", file=sys.stderr)
        return 2

    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"ERROR: image not found: {image_path}", file=sys.stderr)
        return 2

    gt_path = Path(args.ground_truth)
    if not gt_path.is_file():
        print(f"ERROR: ground-truth not found: {gt_path}", file=sys.stderr)
        return 2

    ground_truth = json.loads(gt_path.read_text())

    client = claude_client.ClaudeClient(
        api_key=api_key,
        use_thinking=True,
        thinking_budget=8192,
    )

    print(f"→ Running Claude Vision parse on {image_path.name} ...")
    attempt = client.parse_floor_plan_from_image(
        image_path,
        user_notes=(
            "Trace the exterior perimeter of the main 1st-floor footprint "
            "precisely. Include every jog, notch, and angled wall. The plan "
            "has a triangular balcony on the west side that meets the living "
            "room at an angle."
        ),
    )

    best_score, best_attempt = score(attempt, ground_truth)[0], attempt
    print(f"  iteration 0: score={best_score:.3f}")
    print(f"  usage: {client.usage}")

    iteration = 0
    while best_score < args.threshold and iteration < args.max_iterations:
        iteration += 1
        print(f"→ Score {best_score:.3f} < {args.threshold}, running repair pass {iteration}/{args.max_iterations} ...")
        repaired = client.verify_and_repair(image_path, best_attempt)
        s, breakdown = score(repaired, ground_truth)
        print(f"  iteration {iteration}: score={s:.3f}  details={breakdown}")
        if s > best_score:
            best_score = s
            best_attempt = repaired
        else:
            # No improvement — don't burn more budget.
            print("  repair did not improve score; stopping.")
            break

    out_path = REPO_ROOT / "tools" / "last_vision_attempt.json"
    out_path.write_text(json.dumps(best_attempt, indent=2))
    print(f"→ Final score: {best_score:.3f} (threshold {args.threshold})")
    print(f"→ Final attempt written to {out_path}")
    print(f"→ Total usage: {client.usage}")

    return 0 if best_score >= args.threshold else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True, help="Path to floor plan image")
    p.add_argument(
        "--ground-truth",
        default=str(REPO_ROOT / "tests" / "sample_plans" / "large_house_ground_truth.json"),
        help="Path to ground-truth JSON",
    )
    p.add_argument("--api-key", default=None, help="Anthropic API key (or use $ANTHROPIC_API_KEY)")
    p.add_argument("--threshold", type=float, default=0.75,
                   help="Minimum acceptable score (0-1). Default 0.75.")
    p.add_argument("--max-iterations", type=int, default=4,
                   help="Maximum number of verify_and_repair passes. Default 4.")
    args = p.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
