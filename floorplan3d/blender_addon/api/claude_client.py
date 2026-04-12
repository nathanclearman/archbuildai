"""
Claude API client — multimodal (vision) integration for floor plan reasoning.

This is a ground-up rebuild of the Claude integration. Previous versions sent
text-only prompts wrapping already-parsed JSON, which meant Claude never
actually looked at the floor plan image and frequently hallucinated geometry.

This client uses every relevant Claude Opus 4.6 capability:

    * Vision input — the floor plan image is sent as a base64 image block.
    * Tool use — a ``submit_floor_plan`` tool whose ``input_schema`` is the
      project's canonical JSON schema, so Claude must return structured data
      that already matches what the Blender geometry pipeline expects. This
      eliminates fragile regex-based JSON extraction.
    * Extended thinking — enabled for parse calls so Opus can reason about
      wall intersections, scale, and door swings before emitting the tool
      call.
    * Prompt caching — the long system prompt + tool schema are marked with
      ``cache_control`` so repeated calls within a Blender session are cheap.
    * System prompt separation — expert architectural instructions live in a
      dedicated ``system`` block, not stuffed into the user turn.
    * Retries with exponential backoff on transient errors.
    * Post-call jsonschema validation and an optional one-shot repair turn.

Primary entry point: :meth:`ClaudeClient.parse_floor_plan_from_image`.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import time
from pathlib import Path

from . import schema as fp_schema

log = logging.getLogger(__name__)


# Model tiers
MODEL_OPUS = "claude-opus-4-6"      # Complex reasoning (default)
MODEL_SONNET = "claude-sonnet-4-6"  # Cost-efficient alternative

# Anthropic API constants
API_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 8192
DEFAULT_THINKING_BUDGET = 4096
DEFAULT_TIMEOUT = 180  # thinking + vision can take a while

# Supported image formats per Anthropic's vision docs
SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

# Tool definition: Claude is forced to return data that validates against the
# project's canonical floor plan schema.
PARSE_FLOOR_PLAN_TOOL = {
    "name": "submit_floor_plan",
    "description": (
        "Submit the parsed floor plan geometry. Every measurement is in meters. "
        "All wall endpoints must be snapped to shared junctions, every door and "
        "window must reference the wall it sits on via wall_index, and every "
        "room polygon must be closed (first and last vertices equal) and wound "
        "consistently."
    ),
    "input_schema": fp_schema.FLOOR_PLAN_SCHEMA,
}

SYSTEM_PROMPT = """You are an expert architectural draughtsman and computer \
vision specialist. You parse 2D floor plan images into precise, structured \
geometry suitable for driving a 3D modelling pipeline in Blender.

When parsing a floor plan:

1. SCALE. Look for an explicit scale bar, dimension annotations (e.g. "3.2m", \
"10'-6\""), or a north arrow. If none are visible, estimate from typical \
architectural norms (interior doors are ~0.8-0.9m wide, standard rooms are \
2.5-5m across). Report the scale as pixels_per_meter.

2. WALLS. Trace every wall segment. Exterior walls are typically 0.2-0.3m \
thick; interior partitions 0.1-0.15m. Return walls as straight line segments \
with [x, y] endpoints in meters. SNAP endpoints that share a junction so they \
are bit-identical — the downstream mesh generator depends on this.

3. DOORS. Detect every door opening. A door's position is the centre point \
of the opening. width is the clear opening width. wall_index references the \
wall the door is embedded in (0-indexed into the walls array). Infer type \
(hinged, sliding, folding, pocket, double) from the swing arc or symbol.

4. WINDOWS. Same idea as doors but for windows. Windows typically sit on \
exterior walls.

5. ROOMS. For each enclosed region, emit a room with a label (bedroom, \
kitchen, bathroom, living_room, hallway, etc.), a closed polygon of vertices \
walked in order around the boundary, and the area in square meters.

6. CONFIDENCE. For any element you are uncertain about (ambiguous symbol, \
partially occluded line, hand-drawn sketch), include a confidence score \
between 0 and 1. Omit confidence on elements you are sure about.

7. COORDINATE SYSTEM. Use a right-handed 2D system with +X right, +Y up, \
origin at the bottom-left of the drawing. All coordinates in meters.

Think step by step before emitting the tool call. Use the submit_floor_plan \
tool exactly once per response — do not answer in prose."""


class ClaudeAPIError(RuntimeError):
    """Raised when the Claude API returns an error we cannot recover from."""


class ClaudeClient:
    """Multimodal Claude client for floor plan parsing and reasoning.

    Parameters
    ----------
    api_key : str
        Anthropic API key.
    model : str, optional
        Model identifier. Defaults to Opus 4.6.
    use_thinking : bool, optional
        Whether to enable extended thinking for parse-style calls. Default True.
    thinking_budget : int, optional
        Token budget for extended thinking. Default 4096.
    max_retries : int, optional
        Maximum retry attempts for transient errors. Default 4.
    """

    def __init__(
        self,
        api_key,
        model=None,
        *,
        use_thinking=True,
        thinking_budget=DEFAULT_THINKING_BUDGET,
        max_retries=4,
    ):
        if not api_key:
            raise ValueError("Claude API key is required")
        self.api_key = api_key
        self.model = model or MODEL_OPUS
        self.use_thinking = use_thinking
        self.thinking_budget = thinking_budget
        self.max_retries = max_retries
        self.base_url = API_BASE_URL
        # Per-session usage tally; callers can log / display this.
        self.usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_floor_plan_from_image(
        self,
        image_path,
        *,
        scale_hint=None,
        user_notes=None,
        repair=True,
    ):
        """Parse a floor plan image into structured geometry.

        This is the primary entry point. Sends the image + system prompt +
        tool schema to Opus, forces a structured tool call, validates the
        result, and optionally runs one repair pass if validation fails.

        Parameters
        ----------
        image_path : str or Path
            Local filesystem path to the floor plan image.
        scale_hint : str, optional
            Optional hint about the drawing's scale
            (e.g. "the hallway is 1.2m wide").
        user_notes : str, optional
            Extra context (e.g. "this is a hand-drawn sketch").
        repair : bool, optional
            If True, attempt a single repair pass on schema-invalid output.

        Returns
        -------
        dict
            Floor plan data matching FLOOR_PLAN_SCHEMA.
        """
        image_block = self._encode_image(image_path)

        user_text_parts = [
            "Parse this floor plan into the canonical JSON schema using the "
            "submit_floor_plan tool. Think carefully about scale, wall "
            "junctions, and door/window placement before submitting."
        ]
        if scale_hint:
            user_text_parts.append(f"Scale hint: {scale_hint}")
        if user_notes:
            user_text_parts.append(f"Additional context: {user_notes}")

        messages = [
            {
                "role": "user",
                "content": [
                    image_block,
                    {"type": "text", "text": "\n\n".join(user_text_parts)},
                ],
            }
        ]

        result = self._call_with_tool(
            messages=messages,
            tool=PARSE_FLOOR_PLAN_TOOL,
            tool_choice={"type": "tool", "name": PARSE_FLOOR_PLAN_TOOL["name"]},
            use_thinking=self.use_thinking,
        )

        errors = fp_schema.validate(result)
        if errors and repair:
            log.warning("Floor plan failed schema validation, repairing: %s", errors)
            result = self._repair(image_block, result, errors)
            errors = fp_schema.validate(result)

        if errors:
            raise ClaudeAPIError(
                "Claude returned a floor plan that failed schema validation "
                "after repair: " + "; ".join(errors)
            )

        return result

    def verify_and_repair(self, image_path, parsed_plan):
        """Second-pass verification against the original image.

        Feeds both the image and a previously parsed plan (from either the
        local CV model or an earlier Claude call) back to Opus and asks it to
        correct errors.
        """
        image_block = self._encode_image(image_path)
        plan_json = json.dumps(parsed_plan, indent=2)

        messages = [
            {
                "role": "user",
                "content": [
                    image_block,
                    {
                        "type": "text",
                        "text": (
                            "A previous parser produced the following floor "
                            "plan for this image:\n\n"
                            f"```json\n{plan_json}\n```\n\n"
                            "Compare it carefully to the image. Correct any "
                            "errors in wall positions, door/window placement, "
                            "room polygons, or scale. Snap shared endpoints. "
                            "Submit the corrected plan via the "
                            "submit_floor_plan tool."
                        ),
                    },
                ],
            }
        ]

        return self._call_with_tool(
            messages=messages,
            tool=PARSE_FLOOR_PLAN_TOOL,
            tool_choice={"type": "tool", "name": PARSE_FLOOR_PLAN_TOOL["name"]},
            use_thinking=self.use_thinking,
        )

    def resolve_ambiguity(self, image_path, floor_plan_data, confidence_report):
        """Resolve low-confidence detections from the local CV model.

        Unlike the previous version, this one sees the image.
        """
        image_block = self._encode_image(image_path)
        plan_json = json.dumps(floor_plan_data, indent=2)
        report_json = json.dumps(confidence_report, indent=2)

        messages = [
            {
                "role": "user",
                "content": [
                    image_block,
                    {
                        "type": "text",
                        "text": (
                            "The local CV model flagged the following elements "
                            "as low-confidence. Consult the image and correct "
                            "or remove them.\n\n"
                            f"Current plan:\n```json\n{plan_json}\n```\n\n"
                            f"Low-confidence elements:\n```json\n{report_json}\n```\n\n"
                            "Submit the corrected plan via the "
                            "submit_floor_plan tool."
                        ),
                    },
                ],
            }
        ]

        return self._call_with_tool(
            messages=messages,
            tool=PARSE_FLOOR_PLAN_TOOL,
            tool_choice={"type": "tool", "name": PARSE_FLOOR_PLAN_TOOL["name"]},
            use_thinking=self.use_thinking,
        )

    def suggest_furniture(self, rooms_data, image_path=None):
        """Suggest furniture placement. Optionally grounded in the image."""
        content = []
        if image_path:
            content.append(self._encode_image(image_path))
        content.append(
            {
                "type": "text",
                "text": (
                    "You are an interior designer. Based on the rooms below "
                    "(and the image if provided), propose furniture placements. "
                    "Stay within each room polygon. Keep 0.9m door clearance "
                    "and 0.6m walkways. Return JSON only, keyed by room label, "
                    "with items of shape "
                    "{name, position:[x,y], dimensions:[w,d], rotation}.\n\n"
                    f"Rooms:\n{json.dumps(rooms_data, indent=2)}"
                ),
            }
        )

        text = self._call_text([{"role": "user", "content": content}])
        return self._parse_json_response(text, "furniture suggestions")

    def interpret_modification(self, current_plan, request, image_path=None):
        """Apply a natural-language modification to a floor plan."""
        content = []
        if image_path:
            content.append(self._encode_image(image_path))
        content.append(
            {
                "type": "text",
                "text": (
                    "Apply the following modification to this floor plan and "
                    "return the updated plan via the submit_floor_plan tool. "
                    "Preserve structural walls, keep every room reachable, "
                    "snap to 0.1m increments, and update polygon areas.\n\n"
                    f"Current plan:\n```json\n{json.dumps(current_plan, indent=2)}\n```\n\n"
                    f"Modification request: {request}"
                ),
            }
        )

        return self._call_with_tool(
            messages=[{"role": "user", "content": content}],
            tool=PARSE_FLOOR_PLAN_TOOL,
            tool_choice={"type": "tool", "name": PARSE_FLOOR_PLAN_TOOL["name"]},
            use_thinking=self.use_thinking,
        )

    def critique_layout(self, floor_plan_data, image_path=None):
        """Architectural critique. Free-form JSON, no tool use."""
        content = []
        if image_path:
            content.append(self._encode_image(image_path))
        content.append(
            {
                "type": "text",
                "text": (
                    "As a senior architect, review this floor plan for "
                    "livability, circulation, and code compliance. Return JSON "
                    "with keys: score (1-10), strengths (list), issues "
                    "(list of {description, severity: minor|moderate|critical}), "
                    "suggestions (list).\n\n"
                    f"Plan:\n```json\n{json.dumps(floor_plan_data, indent=2)}\n```"
                ),
            }
        )

        text = self._call_text([{"role": "user", "content": content}])
        return self._parse_json_response(text, "layout critique")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_image(image_path):
        """Base64-encode a local image file into an Anthropic image block."""
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Floor plan image not found: {path}")

        mime, _ = mimetypes.guess_type(str(path))
        if mime not in SUPPORTED_IMAGE_TYPES:
            # Fallback: sniff the first few bytes.
            with open(path, "rb") as f:
                head = f.read(12)
            if head.startswith(b"\x89PNG"):
                mime = "image/png"
            elif head[:3] == b"\xff\xd8\xff":
                mime = "image/jpeg"
            elif head[:4] == b"RIFF" and head[8:12] == b"WEBP":
                mime = "image/webp"
            elif head[:6] in (b"GIF87a", b"GIF89a"):
                mime = "image/gif"
            else:
                raise ValueError(
                    f"Unsupported image format for {path}. "
                    f"Supported: {sorted(SUPPORTED_IMAGE_TYPES)}"
                )

        with open(path, "rb") as f:
            data = f.read()

        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime,
                "data": base64.standard_b64encode(data).decode("ascii"),
            },
        }

    def _call_with_tool(self, *, messages, tool, tool_choice, use_thinking):
        """Send a request that forces a specific tool call and return its input.

        Uses prompt caching on the system prompt + tool so repeated calls in
        one session skip re-tokenising the expertise prompt.
        """
        payload = {
            "model": self.model,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "system": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "tools": [
                {
                    **tool,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "tool_choice": tool_choice,
            "messages": messages,
        }

        if use_thinking:
            # Extended thinking: temperature must be 1 when enabled.
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget,
            }
            payload["temperature"] = 1
        else:
            payload["temperature"] = 0

        data = self._post(payload)
        self._accumulate_usage(data.get("usage", {}))

        for block in data.get("content", []):
            if block.get("type") == "tool_use" and block.get("name") == tool["name"]:
                return block.get("input") or {}

        raise ClaudeAPIError(
            "Claude did not emit the expected tool call "
            f"'{tool['name']}'. Stop reason: {data.get('stop_reason')}."
        )

    def _call_text(self, messages, *, max_tokens=DEFAULT_MAX_TOKENS):
        """Plain text request (no forced tool use). Used for critique/furniture."""
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": messages,
            "temperature": 0,
        }
        data = self._post(payload)
        self._accumulate_usage(data.get("usage", {}))

        for block in data.get("content", []):
            if block.get("type") == "text":
                return block.get("text", "")
        raise ClaudeAPIError("Claude returned no text content")

    def _repair(self, image_block, bad_result, errors):
        """One-shot repair pass: show Claude its own invalid output and ask for a fix."""
        messages = [
            {
                "role": "user",
                "content": [
                    image_block,
                    {
                        "type": "text",
                        "text": (
                            "Your previous submit_floor_plan call failed "
                            "schema validation. Here is the invalid output "
                            "and the validator errors. Re-emit a corrected "
                            "call.\n\n"
                            f"Invalid output:\n```json\n{json.dumps(bad_result, indent=2)}\n```\n\n"
                            "Errors:\n- " + "\n- ".join(errors)
                        ),
                    },
                ],
            }
        ]
        return self._call_with_tool(
            messages=messages,
            tool=PARSE_FLOOR_PLAN_TOOL,
            tool_choice={"type": "tool", "name": PARSE_FLOOR_PLAN_TOOL["name"]},
            use_thinking=False,  # repair is mechanical, skip thinking for speed
        )

    def _post(self, payload):
        """POST with exponential backoff retries on transient errors."""
        import requests

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

        last_exc = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/messages",
                    headers=headers,
                    json=payload,
                    timeout=DEFAULT_TIMEOUT,
                )
            except requests.RequestException as e:
                last_exc = e
                self._sleep_backoff(attempt)
                continue

            if response.status_code == 200:
                return response.json()

            # Retry on rate limits and transient server errors
            if response.status_code in (408, 429, 500, 502, 503, 504):
                last_exc = ClaudeAPIError(
                    f"HTTP {response.status_code}: {response.text[:400]}"
                )
                self._sleep_backoff(attempt)
                continue

            # Non-retryable error
            raise ClaudeAPIError(
                f"Claude API error HTTP {response.status_code}: "
                f"{response.text[:400]}"
            )

        raise ClaudeAPIError(
            f"Claude API failed after {self.max_retries} retries: {last_exc}"
        )

    @staticmethod
    def _sleep_backoff(attempt):
        time.sleep(min(2 ** (attempt + 1), 16))

    def _accumulate_usage(self, usage):
        for key in self.usage:
            self.usage[key] += usage.get(key, 0) or 0

    @staticmethod
    def _parse_json_response(response_text, context="response"):
        """Parse JSON from a free-form text response.

        Only used by critique / furniture methods that don't force tool use.
        Tool-use paths bypass this entirely.
        """
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        import re
        match = re.search(r"```(?:json)?\s*\n(.*?)\n```", response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse {context} from Claude response")
