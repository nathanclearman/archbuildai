"""
Local LLM client for AI features via Ollama or MLX.

Provides the same interface as ClaudeClient but connects to a local
OpenAI-compatible endpoint (e.g., mlx_lm.server or Ollama).  This allows
furniture suggestions, layout critique, and natural language modifications
to work offline and without API credits.

Setup (pick one):
  MLX:    mlx_lm.server --model mlx-community/Qwen2.5-32B-Instruct-4bit --port 8080
  Ollama: ollama serve   (API at localhost:11434)
"""

import json
import re


class LocalLLMClient:
    """Client for local LLM inference via OpenAI-compatible API.

    Works with any server that exposes /v1/chat/completions:
    - mlx_lm.server  (default, best perf on Apple Silicon)
    - Ollama
    - llama.cpp server
    - LM Studio
    """

    def __init__(self, model="mlx-community/Qwen2.5-32B-Instruct-4bit",
                 base_url="http://localhost:8080"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    # ── Public methods (same interface as ClaudeClient) ──────────

    def suggest_furniture(self, rooms_data):
        """Suggest furniture placement based on room type and dimensions."""
        prompt = self._build_furniture_prompt(rooms_data)
        response = self._call_api(prompt)
        return self._parse_json_response(response, "furniture suggestions")

    def interpret_modification(self, current_plan, natural_language_request):
        """Interpret a natural language modification request."""
        prompt = self._build_modification_prompt(current_plan, natural_language_request)
        response = self._call_api(prompt)
        return self._parse_json_response(response, "modified plan")

    def resolve_ambiguity(self, floor_plan_data, confidence_report):
        """Resolve ambiguous detections from the local CV model."""
        prompt = self._build_ambiguity_prompt(floor_plan_data, confidence_report)
        response = self._call_api(prompt)
        return self._parse_json_response(response, "corrected plan")

    def critique_layout(self, floor_plan_data):
        """Provide design feedback and layout optimization suggestions."""
        prompt = self._build_critique_prompt(floor_plan_data)
        response = self._call_api(prompt)
        return self._parse_json_response(response, "layout critique")

    def generate_exterior(self, floor_plan_data, style_prompt,
                          reference_image_path=""):
        """Generate exterior shell configuration based on an architectural style.

        Note: reference_image_path is accepted for API compatibility but
        ignored — local LLM models typically lack vision support.
        """
        prompt = self._build_exterior_prompt(floor_plan_data, style_prompt)
        response = self._call_api(prompt, max_tokens=4096)
        return self._parse_json_response(response, "exterior configuration")

    # ── API call ─────────────────────────────────────────────────

    def _call_api(self, prompt, max_tokens=4096):
        """Call a local OpenAI-compatible /v1/chat/completions endpoint."""
        import requests

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "stream": False,
        }

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=300,  # local models can be slow on first load
            )
        except requests.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to local LLM at {self.base_url}. "
                "Is your model server running?\n\n"
                "Start it with one of:\n"
                "  MLX:    mlx_lm.server --model <model> --port 8080\n"
                "  Ollama: ollama serve"
            )
        except requests.Timeout:
            raise RuntimeError(
                "Local LLM request timed out (300s). The model may still "
                "be loading. Try again in a moment."
            )

        if not response.ok:
            try:
                error_body = response.json()
                error_msg = (
                    error_body.get("error", {}).get("message")
                    or error_body.get("error", response.text)
                )
            except Exception:
                error_msg = response.text
            raise RuntimeError(
                f"Local LLM error {response.status_code}: {error_msg}"
            )

        data = response.json()
        return data["choices"][0]["message"]["content"]

    # ── Prompt builders (identical to ClaudeClient) ──────────────

    @staticmethod
    def _build_furniture_prompt(rooms_data):
        rooms_json = json.dumps(rooms_data, indent=2)
        return (
            "You are an expert interior designer and spatial planner. Given the following "
            "room data from a parsed floor plan, suggest appropriate furniture placement.\n\n"
            "Key constraints:\n"
            "- All positions must fall within the room polygon boundaries\n"
            "- Leave clearance for doors (min 0.9m swing radius) and walkways (min 0.6m)\n"
            "- Consider standard furniture dimensions and ergonomic spacing\n"
            "- Group furniture by function (sleeping area, work area, dining area, etc.)\n\n"
            f"Rooms:\n{rooms_json}\n\n"
            "For each room, return a JSON object with furniture items, each containing:\n"
            "- name: furniture type (e.g., 'double_bed', 'desk', 'sofa')\n"
            "- position: [x, y] in meters (center point, within the room polygon)\n"
            "- dimensions: [width, depth] in meters\n"
            "- rotation: degrees (0 = aligned with X axis)\n"
            "- geometry (OPTIONAL): for unusual or style-specific items only, provide a "
            "list of primitive shapes that make up the piece. Each primitive is:\n"
            "  {\"type\":\"box\", \"cx\":0,\"cy\":0,\"cz\":0.4, \"sx\":0.3,\"sy\":0.2,\"sz\":0.4}\n"
            "  or {\"type\":\"cylinder\", \"cx\":0,\"cy\":0,\"cz\":0.3, \"r\":0.15,\"h\":0.6}\n"
            "  Coordinates are in local furniture space (origin=centre, Z=up).\n"
            "  Only use geometry for items that DON'T match common types like bed, sofa, "
            "desk, table, chair, wardrobe, bookshelf, toilet, bathtub, sink, fridge, oven, tv.\n\n"
            "Return valid JSON only, as a dict keyed by room label. "
            "Do NOT include any text before or after the JSON."
        )

    @staticmethod
    def _build_modification_prompt(current_plan, request):
        plan_json = json.dumps(current_plan, indent=2)
        return (
            "You are an expert architectural assistant. Given the current floor plan data and "
            "a natural language modification request, produce the updated floor plan JSON.\n\n"
            "Rules:\n"
            "- Maintain structural integrity (exterior walls, load-bearing walls)\n"
            "- Ensure all rooms remain accessible (doors connect to hallways or adjacent rooms)\n"
            "- Preserve the same JSON schema exactly\n"
            "- Update room areas when polygons change\n"
            "- Snap walls to reasonable increments (0.1m)\n\n"
            f"Current floor plan:\n{plan_json}\n\n"
            f"Modification request: {request}\n\n"
            "Return the complete updated floor plan as valid JSON only, using the same schema. "
            "Do NOT include any text before or after the JSON."
        )

    @staticmethod
    def _build_ambiguity_prompt(floor_plan_data, confidence_report):
        plan_json = json.dumps(floor_plan_data, indent=2)
        report_json = json.dumps(confidence_report, indent=2)
        return (
            "You are an expert at reading architectural floor plans. The computer vision model "
            "detected the following elements but flagged them as low-confidence. Use your "
            "architectural knowledge to determine the most likely correct interpretation.\n\n"
            f"Floor plan data:\n{plan_json}\n\n"
            f"Low-confidence elements:\n{report_json}\n\n"
            "For each ambiguous element, determine:\n"
            "- Is it a real element or a false positive? (remove false positives)\n"
            "- If real, what are the correct properties? (adjust position, size, type)\n\n"
            "Return the corrected floor plan as valid JSON only, using the same schema. "
            "Do NOT include any text before or after the JSON."
        )

    @staticmethod
    def _build_critique_prompt(floor_plan_data):
        plan_json = json.dumps(floor_plan_data, indent=2)
        return (
            "You are a senior architect reviewing a floor plan for livability, "
            "functionality, and code compliance. Analyze this floor plan and provide feedback.\n\n"
            f"Floor plan:\n{plan_json}\n\n"
            "Return a JSON object with:\n"
            "- score: overall rating 1-10\n"
            "- strengths: list of positive aspects\n"
            "- issues: list of problems found (with severity: 'minor', 'moderate', 'critical')\n"
            "- suggestions: list of specific improvement suggestions\n\n"
            "Return valid JSON only. "
            "Do NOT include any text before or after the JSON."
        )

    @staticmethod
    def _build_exterior_prompt(floor_plan_data, style_prompt):
        walls = floor_plan_data.get("walls", [])
        rooms = floor_plan_data.get("rooms", [])
        n_windows = len(floor_plan_data.get("windows", []))
        n_doors = len(floor_plan_data.get("doors", []))
        total_area = sum(r.get("area", 0) for r in rooms)
        room_labels = [r.get("label", "unknown") for r in rooms]
        summary = (
            f"Building has {len(walls)} walls, {len(rooms)} rooms "
            f"(total area ~{total_area:.0f} m²), {n_doors} doors, {n_windows} windows.\n"
            f"Rooms: {', '.join(room_labels)}\n"
        )
        return (
            "You are an expert architect specialising in residential exterior design. "
            "Given the floor plan summary and an architectural style, design a "
            "visually rich exterior shell as a structured JSON config.\n\n"
            f"Floor plan summary:\n{summary}\n"
            f"Requested style: {style_prompt}\n\n"
            "*** HIGHEST PRIORITY — USER REQUESTS ***\n"
            "The user's style prompt above may contain SPECIFIC requests beyond just the "
            "style name. Examples: 'pink italian villa', 'modern with brick walls', "
            "'scandinavian cabin with green roof'. You MUST:\n"
            "1. Parse the style prompt for ANY color, material, or feature requests\n"
            "2. Implement them — they OVERRIDE style defaults\n"
            "3. For WALL colors (pink, yellow, blue, cream, etc.): use the cladding 'color' "
            "field with an appropriate [r,g,b] value. Pick the closest base material "
            "for texture (e.g. white_stucco for stucco-like colors).\n"
            "   Examples: pink → color [0.90, 0.72, 0.70], yellow → color [0.92, 0.85, 0.55], "
            "cream → color [0.95, 0.90, 0.75], blue → color [0.70, 0.78, 0.88]\n"
            "4. For ROOF colors (orange, red, green, blue, etc.): use the roof 'color' "
            "field with an appropriate [r,g,b] value. Pick the closest base material.\n"
            "   Examples: orange → material clay_tile + color [0.85, 0.45, 0.15], "
            "red → material clay_tile + color [0.70, 0.18, 0.12], "
            "green → material metal + color [0.25, 0.45, 0.30]\n\n"
            "CRITICAL: The roof type MUST match the architectural style. Use these guidelines:\n"
            '- Modern / contemporary / minimalist → "modern_flat" (flat roof with parapet walls)\n'
            '- Modern Korean (hanok-inspired) → "hip" with pitch 20-30, large overhang 0.8-1.2, clay_tile\n'
            '- Traditional Korean hanok → "hip" with pitch 25-35, overhang 1.0-1.5, clay_tile\n'
            '- Japanese → "hip" with pitch 25-30, overhang 0.8-1.0, clay_tile or slate\n'
            '- Mediterranean / Italian / Spanish → "hip" with pitch 20-30, overhang 0.5-0.8, clay_tile\n'
            '- Scandinavian / Nordic → "shed" with pitch 10-20, overhang 0.4-0.6, metal\n'
            '- Colonial / American traditional → "gable" with pitch 35-45, overhang 0.3-0.5, shingle\n'
            '- Craftsman / cottage → "gable" with pitch 30-40, overhang 0.5-0.7, shingle or slate\n'
            '- Mid-century modern → "modern_flat" with overhang 0.6-1.0, metal, parapet_height 0.3-0.5. '
            'MUST use raised_volume detail for multi-height massing.\n'
            '- Industrial / loft → "shed" with pitch 5-15, overhang 0.3, metal\n'
            '- Desert / Southwest → "flat" with overhang 0.2-0.4, concrete\n'
            '- Victorian → "gable" with pitch 45-55, overhang 0.4-0.6, slate\n'
            '- Farmhouse → "gable" with pitch 35-45, overhang 0.5, metal or shingle\n'
            '- Danish / hygge → "gable" with pitch 40-50, overhang 0.4, thatch or clay_tile\n'
            '- Brutalist → "modern_flat" with concrete materials throughout\n'
            '- Spanish Colonial → "hip" with pitch 15-25, clay_tile\n'
            '- Tudor → "gable" with pitch 50-60, overhang 0.3, slate or shingle\n'
            '- Art Deco → "modern_flat" with parapet or stepped profile\n'
            '- Cape Cod → "gable" with pitch 40-50, overhang 0.3, shingle\n'
            '- Prairie → "hip" with pitch 10-20, overhang 1.0-1.5, metal or shingle\n'
            '- Colonial → "gable" with pitch 35-45, overhang 0.3, shingle\n'
            '- Cottage → "gable" with pitch 40-55, overhang 0.4, shingle or clay_tile\n\n'
            "Return a JSON object with this exact schema:\n"
            "{\n"
            '  "roof": {\n'
            '    "type": one of "gable", "hip", "flat", "modern_flat", "shed",\n'
            '    "pitch_degrees": 0-60 (use 0 for flat/modern_flat),\n'
            '    "overhang": metres (0.2-1.5),\n'
            '    "parapet_height": metres (only for modern_flat, 0.4-0.8),\n'
            '    "material": one of "clay_tile","slate","metal","shingle","thatch","concrete","zinc","copper",\n'
            '    "color": [r,g,b] (OPTIONAL — for user-specified roof colors. Values 0.0-1.0)\n'
            "  },\n"
            '  "facade": {\n'
            '    "foundation": {"height": 0.2-0.5, "material": "concrete"|"stone"|"brick"},\n'
            '    "cladding": {"material": "white_stucco"|"warm_stucco"|"stone"|"brick"|"light_wood"'
            '|"dark_wood"|"concrete"|"glass"|"charcoal"|"white_concrete"|"corten"|"zinc", '
            '"color": [r,g,b] (OPTIONAL — use for exact wall color. Values 0.0-1.0)},\n'
            '    "trim": {"material": "dark_wood"|"light_wood"|"metal"|"concrete"|"black_metal", "width": metres},\n'
            '    "window_frames": {"material": "dark_wood"|"light_wood"|"metal"|"black_metal"|"charcoal", "depth": metres},\n'
            '    "door_surround": {"material": "stone"|"dark_wood"|"light_wood"|"concrete"|"black_metal", "width": metres}\n'
            "  },\n"
            '  "window_style": {\n'
            '    "sill_height": metres (0.0-0.9; use 0.0-0.3 for floor-to-ceiling modern windows),\n'
            '    "height": metres (1.0-2.6; use 2.0+ for modern/mid-century/contemporary),\n'
            '    "width": metres (REQUIRED, 1.0-3.0; use 1.8+ for wide modern windows — do NOT omit this)\n'
            "  },\n"
            '  "details": [\n'
            "    // CRITICAL: Include 5-10 detail elements. Be CREATIVE and BOLD.\n"
            "    // HIGH-IMPACT ARCHITECTURAL details (USE THESE for modern/MCM styles!):\n"
            '    {"type": "glass_wall", "mullion_spacing": 1.0-2.0, "mullion_width": 0.03-0.05, '
            '"sill_height": 0.0-0.3, "height": 2.0-2.6, "material": "black_metal"},\n'
            '    {"type": "exposed_beams", "beam_width": 0.12-0.20, "beam_height": 0.15-0.25, '
            '"beam_protrusion": 0.3-0.6, "spacing": 0.6-1.0, "material": "..."},\n'
            '    {"type": "columns", "diameter": 0.20-0.35, "shape": "square"|"round", '
            '"count": 2-6, "standoff": 0.0-1.0 (0=flush with wall, max 1.0), "material": "..."},\n'
            '    {"type": "carport", "depth": 4.0-6.0, "width": 5.0-7.0, '
            '"slab_thickness": 0.12-0.20, "column_size": 0.15-0.25, "material": "..."},\n'
            '    {"type": "clerestory_windows", "window_width": 0.6-1.0, "window_height": 0.3-0.5, '
            '"count": 3-8, "spacing": 0.2-0.4, "material": "glass"},\n'
            '    {"type": "fascia_board", "height": 0.15-0.25, "thickness": 0.02-0.04, "material": "..."},\n'
            '    {"type": "raised_volume", "extra_height": 0.8-1.5, "coverage": 0.3-0.6, '
            '"overhang": 0.1-0.3, "material": "..."} // KEY for MCM: creates a taller box volume,\n'
            "    // CLASSIC details:\n"
            '    {"type": "cornice", "projection": 0.08-0.20, "height": 0.10-0.20, "material": "..."},\n'
            '    {"type": "pilasters", "width": 0.15-0.30, "depth": 0.05-0.12, "material": "..."},\n'
            "    // feature_wall is NOT available — do NOT use it\n"
            '    {"type": "balcony", "depth": 1.2-2.0, "width": 2.5-4.0, "railing_height": 1.0, "material": "..."},\n'
            '    {"type": "canopy", "depth": 1.0-2.5, "thickness": 0.08-0.15, "material": "..."},\n'
            '    {"type": "pergola", "depth": 2.0-3.5, "width": 3.0-5.0, "beam_count": 5-8, "material": "..."},\n'
            '    {"type": "louver_screen", "fin_count": 8-15, "fin_depth": 0.3-0.6, "height": metres, "material": "..."},\n'
            '    {"type": "window_sills", "depth": 0.10-0.20, "thickness": 0.04-0.06, "material": "..."},\n'
            '    {"type": "accent_band", "z_position": metres, "height": 0.03-0.08, "projection": 0.02-0.05, "material": "..."},\n'
            '    {"type": "planter_box", "height": 0.4-0.7, "depth": 0.4-0.6, "width": 1.5-3.0, "material": "greenery"},\n'
            '    {"type": "chimney", "width": 0.5-0.8, "depth": 0.3-0.5, "material": "brick"|"stone"}\n'
            "  ]\n"
            "}\n\n"
            "STYLE IDENTITY — these features DEFINE each style. Without them it looks generic:\n"
            "- Mid-century modern: MUST have raised_volume (extra_height 0.8-1.5, coverage 0.3-0.5) "
            "to create interlocking box volumes at different heights — this is the #1 defining feature. "
            "Use dark_wood cladding as primary material, white_stucco or white_concrete for secondary "
            "contrast panels. Add fascia_board (dark_wood) wrapping the roof edges. "
            "modern_flat roof with deep overhang (0.6-1.0m). Large windows with black_metal frames. "
            "Optional: exposed_beams, planter_box, carport.\n"
            "- Modern minimalist: MUST have glass_wall, fascia_board, columns (white_concrete). "
            "No ornamentation. Pure geometry. modern_flat roof.\n"
            "- Craftsman: MUST have exposed_beams (dark_wood), columns (tapered, stone base), "
            "deep gable roof. Natural materials (stone, wood). Covered porch via pergola.\n"
            "- Mediterranean: Hip clay_tile roof (pitch 20-30), warm_stucco walls "
            "(warm yellow/ochre — NOT white_stucco), pilasters (stone) at corners, "
            "cornice, window_sills (stone). Warm palette. "
            "Columns use standoff 0.0-0.3 to stay near walls.\n"
            "- Farmhouse: Gable metal or shingle roof, light_wood cladding "
            "(NOT white_stucco — farmhouses use WOOD siding), exposed_beams, "
            "pergola, columns (light_wood or stone). Warm, natural palette.\n"
            "- Industrial: shed/flat roof with metal, glass_wall, exposed_beams (metal), "
            "columns (black_metal), accent_band (corten). Raw materials.\n"
            "- Scandinavian: shed/gable metal roof, light_wood cladding, large windows, "
            "canopy, planter_box, clean fascia_board.\n"
            "- Victorian: Steep gable, pilasters, cornice, chimney, balcony, accent_band. Ornate.\n"
            "- Brutalist: modern_flat, concrete everything, columns, bold accent_band, canopy.\n\n"
            "CREATIVE GUIDELINES — be INNOVATIVE and use MULTIPLE detail types:\n"
            "- Modern: glass_wall, columns (black_metal), canopy (concrete), louver_screen, "
            "fascia_board, planter_box\n"
            "- Mid-century: raised_volume (extra_height 0.8-1.5, coverage 0.35-0.5, dark_wood), "
            "fascia_board (dark_wood), exposed_beams (dark_wood), planter_box. "
            "Cladding: dark_wood primary + white_stucco accent\n"
            "- Korean/Japanese: cornice (dark_wood), canopy (dark_wood), planter_box, "
            "columns, window_sills, fascia_board\n"
            "- Mediterranean: pilasters (stone), cornice (warm_stucco), window_sills (stone), "
            "pergola (light_wood), chimney (stone), planter_box (terracotta). "
            "Cladding: warm_stucco (NOT white_stucco). "
            "Columns standoff 0.0-0.3 (flush with or very near walls).\n"
            "- Scandinavian: canopy (light_wood), fascia_board, planter_box, "
            "clerestory_windows, columns\n"
            "- Industrial: glass_wall, exposed_beams (metal), accent_band (corten), "
            "columns (black_metal), accent_band (metal)\n\n"
            "WINDOW STYLE GUIDELINES — match window proportions to architectural style:\n"
            "- Modern / contemporary / minimalist: sill 0.1-0.3, height 2.0-2.4, width 1.5-2.5\n"
            "- Mid-century modern: sill 0.0-0.2, height 2.2-2.5, width 2.0-3.0 (floor-to-ceiling glass)\n"
            "- Industrial / loft: sill 0.3, height 2.0-2.4, width 1.5-2.0\n"
            "- Traditional / colonial: sill 0.8-0.9, height 1.2-1.4\n"
            "- Mediterranean: sill 0.7-0.9, height 1.3-1.6, width 1.0-1.4\n"
            "- Scandinavian: sill 0.3-0.5, height 1.6-2.0\n"
            "- Brutalist: sill 0.3, height 1.8-2.2, width 1.5-2.0\n\n"
            "MATERIAL CONTRAST RULE: Do NOT use the same material for everything! "
            "A real building uses at least 3 DISTINCT materials. For example:\n"
            "- MCM: dark_wood cladding + white_stucco contrast panels + black_metal frames + brick accent wall\n"
            "- Modern: white_stucco walls + charcoal trim + glass + black_metal columns\n"
            "- Craftsman: stone foundation + light_wood cladding + dark_wood beams\n"
            "Each detail's material MUST contrast with the cladding material.\n\n"
            "SIGNATURE FEATURE — THIS IS CRITICAL:\n"
            "Every great house has 1-2 unique, style-defining features that make it memorable. "
            "You MUST include 1-2 'signature' details that go BEYOND the standard template — "
            "something creative, exotic, and vital to the style that elevates it from generic "
            "to iconic. Think like a famous architect designing their masterpiece.\n\n"
            "IMPORTANT: The details array must NOT be empty — include at least 5 details "
            "that make the building look like real ICONIC architecture of the requested style. "
            "1-2 of those details should be your SIGNATURE features — bold, creative, style-defining. "
            "For MCM styles, raised_volume + fascia_board + dark_wood cladding are NON-NEGOTIABLE. "
            "Mix structural, functional, and decorative elements. "
            "Available materials: clay_tile, slate, metal, shingle, thatch, concrete, "
            "white_stucco, warm_stucco (Mediterranean yellow/ochre), stone, brick, "
            "dark_wood, light_wood, glass, copper, corten, "
            "charcoal, greenery, black_metal, white_concrete, terracotta, zinc.\n"
            "Return valid JSON only. Do NOT include any text before or after the JSON."
        )

    # ── Response parsing (same as ClaudeClient) ──────────────────

    @staticmethod
    def _parse_json_response(response_text, context="response"):
        """Parse JSON from LLM response, handling markdown code blocks."""
        # Try direct parse first
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

        raise ValueError(f"Could not parse {context} from local LLM response")
