"""
Canonical floor plan JSON schema.

Single source of truth for the structured output produced by any parser
(local CV model or Claude vision). Used as the Anthropic tool ``input_schema``
so Claude is forced to return a well-typed result, and as a jsonschema
validator for post-processing / repair loops.
"""

FLOOR_PLAN_SCHEMA = {
    "type": "object",
    "required": ["scale", "walls", "doors", "windows", "rooms"],
    "additionalProperties": False,
    "properties": {
        "scale": {
            "type": "object",
            "required": ["pixels_per_meter"],
            "additionalProperties": False,
            "properties": {
                "pixels_per_meter": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": "Image scale in pixels per meter. Infer from "
                                   "dimension annotations, a scale bar, or typical "
                                   "room sizes if not explicit.",
                },
            },
        },
        "walls": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["start", "end", "thickness"],
                "additionalProperties": False,
                "properties": {
                    "start": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "[x, y] in meters",
                    },
                    "end": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "[x, y] in meters",
                    },
                    "thickness": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "description": "Wall thickness in meters, typically 0.1-0.3",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                },
            },
        },
        "doors": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["position", "width", "wall_index"],
                "additionalProperties": False,
                "properties": {
                    "position": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "width": {"type": "number", "exclusiveMinimum": 0},
                    "type": {
                        "type": "string",
                        "enum": ["hinged", "sliding", "folding", "pocket", "double"],
                    },
                    "wall_index": {"type": "integer", "minimum": 0},
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                },
            },
        },
        "windows": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["position", "width", "wall_index"],
                "additionalProperties": False,
                "properties": {
                    "position": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "width": {"type": "number", "exclusiveMinimum": 0},
                    "wall_index": {"type": "integer", "minimum": 0},
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                },
            },
        },
        "rooms": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["label", "polygon"],
                "additionalProperties": False,
                "properties": {
                    "label": {"type": "string"},
                    "polygon": {
                        "type": "array",
                        "minItems": 3,
                        "items": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                    },
                    "area": {"type": "number", "minimum": 0},
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                },
            },
        },
        "notes": {
            "type": "string",
            "description": "Free-form notes about ambiguities or assumptions.",
        },
    },
}


def validate(data):
    """Validate data against FLOOR_PLAN_SCHEMA.

    Returns a list of human-readable error messages (empty if valid).
    Uses jsonschema if available, otherwise a lightweight fallback that
    checks required top-level keys and basic shapes.
    """
    try:
        import jsonschema  # type: ignore

        validator = jsonschema.Draft202012Validator(FLOOR_PLAN_SCHEMA)
        return [
            f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
            for e in validator.iter_errors(data)
        ]
    except ImportError:
        return _fallback_validate(data)


def _fallback_validate(data):
    errors = []
    if not isinstance(data, dict):
        return ["<root>: expected object"]

    for key in ("scale", "walls", "doors", "windows", "rooms"):
        if key not in data:
            errors.append(f"<root>: missing required key '{key}'")

    scale = data.get("scale")
    if isinstance(scale, dict):
        ppm = scale.get("pixels_per_meter")
        if not isinstance(ppm, (int, float)) or ppm <= 0:
            errors.append("scale/pixels_per_meter: must be a positive number")

    for arr_key in ("walls", "doors", "windows", "rooms"):
        arr = data.get(arr_key)
        if arr is not None and not isinstance(arr, list):
            errors.append(f"{arr_key}: must be an array")

    for i, wall in enumerate(data.get("walls", []) or []):
        for field in ("start", "end", "thickness"):
            if field not in wall:
                errors.append(f"walls/{i}: missing '{field}'")

    return errors
