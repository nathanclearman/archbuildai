"""
Canonical floor plan JSON schema.

One source of truth for the format that flows between the CV layer, the VLM,
the refiner, and the Blender geometry layer. Includes a validator and the
serializer used as the VLM's training target.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable


class SchemaError(ValueError):
    """Raised when a floor plan dict does not conform to the schema."""


@dataclass
class FloorPlan:
    """A parsed floor plan. Field names match the JSON schema exactly."""

    walls: list[dict] = field(default_factory=list)
    doors: list[dict] = field(default_factory=list)
    windows: list[dict] = field(default_factory=list)
    rooms: list[dict] = field(default_factory=list)
    scale: dict = field(default_factory=lambda: {"pixels_per_meter": 50})

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FloorPlan":
        validate(data)
        return cls(
            walls=list(data.get("walls", [])),
            doors=list(data.get("doors", [])),
            windows=list(data.get("windows", [])),
            rooms=list(data.get("rooms", [])),
            scale=dict(data.get("scale", {"pixels_per_meter": 50})),
        )


# ---------- validation ----------

def _is_point(p: Any) -> bool:
    return (
        isinstance(p, (list, tuple))
        and len(p) == 2
        and all(isinstance(c, (int, float)) for c in p)
    )


def validate(data: dict) -> None:
    """Raise SchemaError if `data` is not a valid floor plan."""
    if not isinstance(data, dict):
        raise SchemaError("floor plan must be a dict")

    for key in ("walls", "doors", "windows", "rooms"):
        if key in data and not isinstance(data[key], list):
            raise SchemaError(f"{key} must be a list")

    n_walls = len(data.get("walls", []))

    for i, w in enumerate(data.get("walls", [])):
        if not _is_point(w.get("start")) or not _is_point(w.get("end")):
            raise SchemaError(f"wall[{i}] missing start/end points")
        if not isinstance(w.get("thickness", 0.15), (int, float)):
            raise SchemaError(f"wall[{i}] thickness must be numeric")

    for i, d in enumerate(data.get("doors", [])):
        if not _is_point(d.get("position")):
            raise SchemaError(f"door[{i}] missing position")
        idx = d.get("wall_index")
        if idx is not None and (not isinstance(idx, int) or idx >= n_walls):
            raise SchemaError(f"door[{i}] wall_index {idx} out of range")

    for i, w in enumerate(data.get("windows", [])):
        if not _is_point(w.get("position")):
            raise SchemaError(f"window[{i}] missing position")
        idx = w.get("wall_index")
        if idx is not None and (not isinstance(idx, int) or idx >= n_walls):
            raise SchemaError(f"window[{i}] wall_index {idx} out of range")

    for i, r in enumerate(data.get("rooms", [])):
        if "label" not in r or not isinstance(r["label"], str):
            raise SchemaError(f"room[{i}] missing string label")
        poly = r.get("polygon", [])
        if not isinstance(poly, list) or len(poly) < 3 or not all(_is_point(p) for p in poly):
            raise SchemaError(f"room[{i}] polygon must be >=3 points")


# ---------- serialization ----------
#
# The VLM is trained to emit this exact compact format. Keeping it explicit
# and deterministic matters: any drift between training-target format and
# generation format shows up as degraded eval.

def _round_point(p: Iterable[float], nd: int) -> list[float]:
    return [round(float(p[0]), nd), round(float(p[1]), nd)]


def serialize(fp: dict | FloorPlan, *, decimals: int = 2) -> str:
    """Serialize a floor plan to the canonical JSON string used as the VLM
    training target. Deterministic key order, fixed precision, compact."""
    d = fp.to_dict() if isinstance(fp, FloorPlan) else dict(fp)
    validate(d)

    out = {
        "scale": {"pixels_per_meter": int(d.get("scale", {}).get("pixels_per_meter", 50))},
        "walls": [
            {
                "start": _round_point(w["start"], decimals),
                "end": _round_point(w["end"], decimals),
                "thickness": round(float(w.get("thickness", 0.15)), 3),
            }
            for w in d.get("walls", [])
        ],
        "doors": [
            {
                "position": _round_point(x["position"], decimals),
                "width": round(float(x.get("width", 0.9)), 2),
                "type": x.get("type", "hinged"),
                "wall_index": int(x["wall_index"]) if x.get("wall_index") is not None else -1,
            }
            for x in d.get("doors", [])
        ],
        "windows": [
            {
                "position": _round_point(x["position"], decimals),
                "width": round(float(x.get("width", 1.0)), 2),
                "wall_index": int(x["wall_index"]) if x.get("wall_index") is not None else -1,
            }
            for x in d.get("windows", [])
        ],
        "rooms": [
            {
                "label": r["label"],
                "polygon": [_round_point(p, decimals) for p in r["polygon"]],
                "area": round(float(r.get("area", 0.0)), 2),
            }
            for r in d.get("rooms", [])
        ],
    }
    return json.dumps(out, separators=(",", ":"))


def deserialize(s: str) -> dict:
    """Parse a VLM-emitted string back into a floor plan dict. Permissive:
    handles minor trailing junk, missing sections, extra whitespace."""
    s = s.strip()
    # Models occasionally wrap output in ```json ... ```; strip fences.
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
    # Find first { ... last matching } to tolerate extra text.
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1:
        raise SchemaError("no JSON object found in model output")
    data = json.loads(s[start : end + 1])
    # Fill in required sections if missing so validator passes.
    for key in ("walls", "doors", "windows", "rooms"):
        data.setdefault(key, [])
    data.setdefault("scale", {"pixels_per_meter": 50})
    validate(data)
    return data
