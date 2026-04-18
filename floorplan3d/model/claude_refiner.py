"""
Claude Opus refiner — optional post-processing stage.

Takes the floor plan JSON emitted by the VLM (or CV fallback) plus the
original image, asks Claude to correct obvious errors, fill in missing
room labels, and normalize label vocabulary. Runs only when an
ANTHROPIC_API_KEY is present.

Designed to be cheap: a single Opus call per image, ~2K input tokens,
~2K output tokens. At current pricing this is a fraction of a cent per
refined plan.
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

# Local schema validation.
sys.path.insert(0, str(Path(__file__).parent))
from schema import deserialize, serialize  # type: ignore
from synthesize import US_ROOM_LABELS  # type: ignore


# Latest Opus generation at knowledge-cutoff. Claude 4.7 is the current
# top-of-line Opus; 4.6 (the previous default) still works but doesn't
# benefit from the reasoning improvements that matter most for the
# refiner's "correct geometric inconsistencies from pixel evidence"
# task. Cost per image is fractional cents either way.
MODEL = "claude-opus-4-7"


# Built once from US_ROOM_LABELS so the refiner prompt and the training
# target vocabulary can never drift. Previously inline-listed, which
# went stale the moment Cluster E added `study`, `stairs`, and
# `main_room` without updating the prompt — the refiner would "fix"
# those labels back to a generic bedroom/office.
_VOCAB_STR = ", ".join(US_ROOM_LABELS)


REFINER_PROMPT = f"""You are a floor plan refiner. You will receive:
  1. A raster floor plan image.
  2. A candidate JSON parse of that floor plan.

Your job: return a corrected version of the JSON that is geometrically
self-consistent and matches the image. Fix only clear errors. Do not
invent walls or rooms not visible in the image.

Specifically:
- Fill in empty or generic room labels ("" or "room") using visible text
  or typical layout cues (kitchen icons, fixture icons).
- Snap wall endpoints that are obviously meant to coincide.
- Correct wall_index references on doors/windows to point at the actual
  wall they lie on.
- Normalize room labels to snake_case from this US-focused vocabulary:
  {_VOCAB_STR}.
  Prefer master_bedroom over bedroom if it is attached to an en_suite
  and/or a walk_in_closet. Prefer great_room when living, dining, and
  kitchen are one contiguous open space.

Respond with ONLY the corrected JSON object. No prose, no code fences.
"""


def refine(image_path: str, candidate: dict) -> dict:
    """Refine a candidate floor plan JSON using Claude Opus. Idempotent on
    failure: returns the candidate unchanged if the API is unreachable or
    the refined output fails schema validation."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return candidate

    try:
        import anthropic
    except ImportError:
        print("[refiner] anthropic SDK not installed, skipping", file=sys.stderr)
        return candidate

    try:
        img_b64 = base64.standard_b64encode(Path(image_path).read_bytes()).decode()
        media_type = _guess_media_type(image_path)

        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=MODEL,
            max_tokens=8192,
            system=REFINER_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": img_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Candidate JSON:\n" + serialize(candidate),
                        },
                    ],
                }
            ],
        )
        text = "".join(
            block.text for block in msg.content if getattr(block, "type", None) == "text"
        )
        # deserialize() runs schema.validate() internally, so a separate
        # validate(refined) afterward was a redundant re-check that
        # would never catch anything deserialize() hadn't already caught.
        return deserialize(text)
    except Exception as e:
        print(f"[refiner] refinement failed, keeping candidate: {e}", file=sys.stderr)
        return candidate


def _guess_media_type(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(suffix, "image/png")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--candidate", required=True, help="path to candidate JSON")
    parser.add_argument("--out", default="-")
    args = parser.parse_args()

    candidate = json.loads(Path(args.candidate).read_text())
    refined = refine(args.image, candidate)
    text = json.dumps(refined)
    if args.out == "-":
        print(text)
    else:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
