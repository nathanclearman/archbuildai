"""
Claude Vision floor plan parser.

Uses Claude's vision API to analyze floor plan images and produce
structured JSON output (walls, doors, windows, rooms) in the same
format as the CubiCasa and YOLO parsers.

This is the most accurate option for complex, hand-drawn, or
non-standard floor plans where the local CV models struggle.
Requires a Claude API key and incurs API usage costs.
"""

import json
import os
import re


class ClaudeVisionClient:
    """Parse floor plan images using Claude's vision API."""

    def __init__(self, api_key, model=None):
        if not api_key or not api_key.strip():
            raise ValueError("Claude API key is required for vision parsing")
        self.api_key = api_key.strip()
        self.model = model or "claude-sonnet-4-6"
        self.base_url = "https://api.anthropic.com/v1"

    def predict(self, image_path, pixels_per_meter=50.0, conf_threshold=0.5):
        """Analyze a floor plan image and return structured JSON.

        Args:
            image_path: Path to the floor plan image.
            pixels_per_meter: Scale factor (used as hint for Claude).
            conf_threshold: Not used directly (Claude doesn't have
                confidence scores), but included for API compatibility.

        Returns:
            dict: Floor plan data with walls, doors, windows, rooms.
        """
        prompt = self._build_prompt(pixels_per_meter)
        response = self._call_vision_api(prompt, image_path)
        result = self._parse_json_response(response)

        # Validate and fix the result
        result = self._validate_and_fix(result, pixels_per_meter)

        return result

    def _build_prompt(self, pixels_per_meter):
        """Build the floor plan analysis prompt."""
        return (
            "You are an expert architectural floor plan analyzer. Analyze this floor plan image "
            "and extract ALL structural elements with precise coordinates.\n\n"
            "## Coordinate System\n"
            "- Use meters as the unit\n"
            f"- The image scale is approximately {pixels_per_meter} pixels per meter\n"
            "- Origin (0,0) is at the top-left of the image\n"
            "- X increases to the right, Y increases downward\n"
            "- Measure coordinates from the image pixel positions divided by the scale factor\n\n"
            "## What to Extract\n\n"
            "### Walls\n"
            "- Every wall segment as a start [x,y] and end [x,y] point (in meters)\n"
            "- Include both exterior walls (the building perimeter) and interior partition walls\n"
            "- Each wall should be a straight line segment from one corner/junction to the next\n"
            "- Break L-shaped or multi-segment walls into individual segments\n"
            "- Typical wall thickness is 0.15m for interior, 0.25m for exterior\n\n"
            "### Doors\n"
            "- Position [x,y] is the center of the door opening along the wall\n"
            "- Width is the door opening width (typically 0.8-0.9m for standard doors)\n"
            "- wall_index: the index of the wall this door is on (0-based)\n"
            "- Look for door arcs/swings, double doors, sliding doors in the image\n\n"
            "### Windows\n"
            "- Position [x,y] is the center of the window along the wall\n"
            "- Width is the window width (typically 0.6-1.8m)\n"
            "- wall_index: the index of the wall this window is on (0-based)\n"
            "- Look for parallel lines on walls indicating windows\n\n"
            "### Rooms\n"
            "- Read room labels from the image text (e.g., 'MASTER SUITE', 'BEDROOM 2', 'KITCHEN')\n"
            "- For each room, provide a polygon as a list of [x,y] corner points tracing the room boundary\n"
            "- Calculate the area in square meters\n"
            "- Use standardized labels: living_room, kitchen, bedroom, bedroom_2, bedroom_3, "
            "bathroom, master_suite, study, gameroom, dining_room, hallway, utility, wc, "
            "closet, wardrobe, balcony, garage, laundry, pantry, foyer, stairway\n\n"
            "## CRITICAL Rules\n"
            "- Be PRECISE with coordinates — walls must connect at corners/junctions\n"
            "- Every wall endpoint should connect to another wall endpoint (no floating walls)\n"
            "- The building perimeter must form a closed shape\n"
            "- Each door/window MUST reference the correct wall_index that it's on\n"
            "- Read ALL text labels in the image for room names and dimensions\n"
            "- If you see dimension annotations (like 13'0\"x15'4\"), use them to calibrate coordinates\n"
            "- Convert feet/inches to meters (1 foot = 0.3048m)\n\n"
            "## Output Format\n"
            "Return ONLY valid JSON (no markdown, no explanation) in exactly this format:\n"
            "```\n"
            "{\n"
            "  \"scale\": {\"pixels_per_meter\": <number>},\n"
            "  \"walls\": [\n"
            "    {\"start\": [x1, y1], \"end\": [x2, y2], \"thickness\": 0.15}\n"
            "  ],\n"
            "  \"doors\": [\n"
            "    {\"position\": [x, y], \"width\": 0.9, \"type\": \"hinged\", \"wall_index\": 0}\n"
            "  ],\n"
            "  \"windows\": [\n"
            "    {\"position\": [x, y], \"width\": 1.2, \"wall_index\": 0}\n"
            "  ],\n"
            "  \"rooms\": [\n"
            "    {\"label\": \"master_suite\", \"polygon\": [[x1,y1], [x2,y2], ...], \"area\": 25.0}\n"
            "  ]\n"
            "}\n"
            "```\n\n"
            "Return ONLY the JSON object. No other text."
        )

    def _call_vision_api(self, prompt, image_path):
        """Send image + prompt to Claude vision API."""
        try:
            import requests
        except ImportError:
            raise ImportError(
                "The 'requests' package is required for Claude API calls. "
                "Install it in Blender's Python:\n"
                "  <blender>/python/bin/python -m pip install requests"
            )

        import base64

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Floor plan image not found: {image_path}")

        max_size_mb = 20
        file_size = os.path.getsize(image_path)
        if file_size > max_size_mb * 1024 * 1024:
            raise ValueError(
                f"Image too large ({file_size / 1024 / 1024:.0f}MB). "
                f"Max {max_size_mb}MB. Resize or use a lower-res export."
            )

        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        ext = os.path.splitext(image_path)[1].lower()
        media_types = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".gif": "image/gif",
            ".webp": "image/webp",
        }
        media_type = media_types.get(ext, "image/jpeg")

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_data,
                },
            },
            {"type": "text", "text": prompt},
        ]

        payload = {
            "model": self.model,
            "max_tokens": 8192,  # Floor plans need detailed output
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,  # Deterministic for precision
        }

        response = requests.post(
            f"{self.base_url}/messages",
            headers=headers,
            json=payload,
            timeout=180,  # Vision + complex analysis can be slow
        )

        if not response.ok:
            try:
                error_body = response.json()
                error_obj = error_body.get("error", {})
                error_msg = (error_obj.get("message", response.text)
                             if isinstance(error_obj, dict) else str(error_obj))
            except Exception:
                error_msg = response.text
            raise RuntimeError(
                f"Claude API error {response.status_code}: {error_msg}"
            )

        data = response.json()
        content_blocks = data.get("content", [])
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                return block["text"]
        raise RuntimeError("Claude API returned no text content")

    @staticmethod
    def _parse_json_response(response_text):
        """Parse JSON from Claude's response, handling markdown blocks."""
        # Try direct parse
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code blocks
        match = re.search(r"```(?:json)?\s*\n(.*?)\n```", response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding first { to last }
        first_brace = response_text.find("{")
        last_brace = response_text.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            try:
                return json.loads(response_text[first_brace:last_brace + 1])
            except json.JSONDecodeError:
                pass

        raise RuntimeError(
            f"Failed to parse Claude's response as JSON. "
            f"Response starts with: {response_text[:200]}"
        )

    @staticmethod
    def _validate_and_fix(data, pixels_per_meter):
        """Validate and fix the parsed floor plan data.

        Ensures all required fields exist and wall_index references
        are valid.
        """
        # Ensure required top-level keys
        if "scale" not in data:
            data["scale"] = {"pixels_per_meter": pixels_per_meter}
        if "walls" not in data:
            data["walls"] = []
        if "doors" not in data:
            data["doors"] = []
        if "windows" not in data:
            data["windows"] = []
        if "rooms" not in data:
            data["rooms"] = []

        walls = data["walls"]
        num_walls = len(walls)

        # Ensure wall thickness
        for w in walls:
            if "thickness" not in w:
                w["thickness"] = 0.15
            # Add a high confidence since Claude's detections are
            # generally reliable
            w["confidence"] = 0.9

        # Validate door wall_index references
        for d in data["doors"]:
            if "type" not in d:
                d["type"] = "hinged"
            if "width" not in d:
                d["width"] = 0.9
            d["confidence"] = 0.9
            idx = d.get("wall_index", 0)
            if idx >= num_walls:
                # Find nearest wall
                d["wall_index"] = _nearest_wall(
                    d.get("position", [0, 0]), walls)

        # Validate window wall_index references
        for w in data["windows"]:
            if "width" not in w:
                w["width"] = 1.2
            w["confidence"] = 0.9
            idx = w.get("wall_index", 0)
            if idx >= num_walls:
                w["wall_index"] = _nearest_wall(
                    w.get("position", [0, 0]), walls)

        # Validate rooms
        for r in data["rooms"]:
            if "polygon" not in r:
                r["polygon"] = []
            if "area" not in r and r["polygon"]:
                r["area"] = _polygon_area(r["polygon"])
            if "label" not in r:
                r["label"] = "room"

        return data


    def refine_with_vision(self, image_path, local_result, pixels_per_meter=50.0):
        """Refine local model output using Claude Vision.

        Sends both the floor plan image and the local model's JSON output
        to Claude, asking it to identify missing walls, correct room
        boundaries/labels, and flag missed doors/windows.

        Args:
            image_path: Path to the floor plan image.
            local_result: dict from the local CV model (walls, doors, etc.).
            pixels_per_meter: Scale factor.

        Returns:
            dict: Merged floor plan data combining local precision with
                  Claude's structural understanding.
        """
        prompt = self._build_refinement_prompt(local_result, pixels_per_meter)
        response = self._call_vision_api(prompt, image_path)
        claude_result = self._parse_json_response(response)
        claude_result = self._validate_and_fix(claude_result, pixels_per_meter)

        return self._merge_refinement(local_result, claude_result)

    @staticmethod
    def _build_refinement_prompt(local_result, pixels_per_meter):
        """Build prompt for refining local model output."""
        # Summarize local model output concisely
        num_walls = len(local_result.get("walls", []))
        num_doors = len(local_result.get("doors", []))
        num_windows = len(local_result.get("windows", []))
        rooms = local_result.get("rooms", [])
        room_summary = ", ".join(
            f"{r.get('label', 'room')} ({r.get('area', 0):.0f}m²)"
            for r in rooms[:15]
        )

        local_json = json.dumps(local_result, indent=None, separators=(",", ":"))
        # Truncate if very large to stay within token limits
        if len(local_json) > 12000:
            local_json = local_json[:12000] + "...(truncated)"

        return (
            "You are an expert architectural floor plan analyzer. A computer vision model "
            "has already analyzed this floor plan image but produced inaccurate results. "
            "Your job is to provide a CORRECTED and COMPLETE analysis.\n\n"
            f"## What the CV model detected\n"
            f"- {num_walls} walls, {num_doors} doors, {num_windows} windows\n"
            f"- Rooms: {room_summary}\n\n"
            "## Known problems with the CV model output\n"
            "- Missing interior walls (especially partition walls between rooms)\n"
            "- Room boundaries are wrong — rooms merge into each other\n"
            "- Room labels are guessed by area heuristics, not read from the image\n"
            "- May miss doors and windows\n\n"
            "## Your task\n"
            "1. Look at the floor plan image carefully\n"
            "2. Read ALL text labels in the image (room names, dimensions)\n"
            "3. Identify ALL walls, including ones the model missed\n"
            "4. Provide correct room polygons matching the actual room boundaries\n"
            "5. Use the actual room names from the image text as labels\n"
            "6. Identify all doors and windows\n\n"
            "## Coordinate System\n"
            "- Use meters as the unit\n"
            f"- The image scale is approximately {pixels_per_meter} pixels per meter\n"
            "- Origin (0,0) is at the top-left of the image\n"
            "- X increases to the right, Y increases downward\n"
            "- Measure coordinates from the image pixel positions divided by the scale factor\n"
            "- If you see dimension annotations (like 13'0\"x15'4\"), use them to calibrate\n"
            "- Convert feet/inches to meters (1 foot = 0.3048m)\n\n"
            "## CV model output for reference\n"
            f"```json\n{local_json}\n```\n\n"
            "## Output Format\n"
            "Return a COMPLETE corrected JSON (not a diff). Include ALL walls, doors, "
            "windows, and rooms — both the ones the model got right and the ones it missed.\n"
            "Use standardized room labels: living_room, kitchen, bedroom, bedroom_2, "
            "bathroom, master_suite, study, dining_room, hallway, utility, wc, "
            "closet, wardrobe, balcony, garage, laundry, pantry, foyer, stairway, "
            "great_room, mud_room, porch, covered_porch\n\n"
            "Wall thickness: 0.15m interior, 0.25m exterior.\n"
            "Each wall: {\"start\": [x,y], \"end\": [x,y], \"thickness\": 0.15}\n"
            "Each door: {\"position\": [x,y], \"width\": 0.9, \"type\": \"hinged\", \"wall_index\": N}\n"
            "Each window: {\"position\": [x,y], \"width\": 1.2, \"wall_index\": N}\n"
            "Each room: {\"label\": \"name\", \"polygon\": [[x,y],...], \"area\": N}\n\n"
            "Return ONLY the JSON object:\n"
            "{\n"
            "  \"scale\": {\"pixels_per_meter\": <number>},\n"
            "  \"walls\": [...],\n"
            "  \"doors\": [...],\n"
            "  \"windows\": [...],\n"
            "  \"rooms\": [...]\n"
            "}\n\n"
            "Return ONLY valid JSON. No markdown, no explanation."
        )

    @staticmethod
    def _merge_refinement(local_result, claude_result):
        """Merge Claude's refined output with local model output.

        Strategy:
        - Walls: keep local walls, add Claude walls that are far from existing
        - Rooms: replace entirely with Claude's (reads actual labels)
        - Doors/Windows: keep local, add Claude extras that aren't duplicates
        """
        import copy
        merged = copy.deepcopy(local_result)

        local_walls = merged.get("walls", [])
        claude_walls = claude_result.get("walls", [])

        # Add Claude walls that are >0.5m from any existing wall
        for cw in claude_walls:
            cs = cw["start"]
            ce = cw["end"]
            c_mid = [(cs[0] + ce[0]) / 2, (cs[1] + ce[1]) / 2]

            is_duplicate = False
            for lw in local_walls:
                ls = lw["start"]
                le = lw["end"]
                l_mid = [(ls[0] + le[0]) / 2, (ls[1] + le[1]) / 2]
                dist = ((c_mid[0] - l_mid[0]) ** 2 +
                        (c_mid[1] - l_mid[1]) ** 2) ** 0.5
                if dist < 0.5:
                    is_duplicate = True
                    break
            if not is_duplicate:
                local_walls.append(cw)

        merged["walls"] = local_walls

        # Replace rooms entirely with Claude's output (reads actual labels)
        if claude_result.get("rooms"):
            merged["rooms"] = claude_result["rooms"]

        # Add Claude doors/windows that aren't near existing ones
        for key, min_dist in [("doors", 0.3), ("windows", 0.3)]:
            local_items = merged.get(key, [])
            claude_items = claude_result.get(key, [])
            for ci in claude_items:
                cp = ci.get("position", [0, 0])
                is_dup = False
                for li in local_items:
                    lp = li.get("position", [0, 0])
                    d = ((cp[0] - lp[0]) ** 2 +
                         (cp[1] - lp[1]) ** 2) ** 0.5
                    if d < min_dist:
                        is_dup = True
                        break
                if not is_dup:
                    # Reassign wall_index to merged wall list
                    ci["wall_index"] = _nearest_wall(cp, merged["walls"])
                    local_items.append(ci)
            merged[key] = local_items

        # Reassign all door/window wall_index references to the merged
        # wall list (indices may have shifted due to added walls)
        for key in ("doors", "windows"):
            for item in merged.get(key, []):
                pos = item.get("position", [0, 0])
                item["wall_index"] = _nearest_wall(pos, merged["walls"])

        return merged


def _nearest_wall(position, walls):
    """Find the index of the nearest wall to a position."""
    if not walls:
        return 0
    px, py = position[0], position[1]
    best_idx = 0
    best_dist = float("inf")
    for i, w in enumerate(walls):
        sx, sy = w["start"]
        ex, ey = w["end"]
        dx, dy = ex - sx, ey - sy
        lsq = dx * dx + dy * dy
        if lsq < 1e-10:
            dist = ((px - sx) ** 2 + (py - sy) ** 2) ** 0.5
        else:
            t = max(0, min(1, ((px - sx) * dx + (py - sy) * dy) / lsq))
            cx, cy = sx + t * dx, sy + t * dy
            dist = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


def _polygon_area(polygon):
    """Shoelace formula for polygon area."""
    n = len(polygon)
    if n < 3:
        return 0
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return abs(area) / 2.0
