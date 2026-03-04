"""
Optional Claude API client for higher-level reasoning tasks.

Uses Claude Opus 4.6 for complex architectural reasoning:
- Furniture auto-placement based on room type and size
- Natural language modification requests
- Layout quality feedback and suggestions
- Interpreting ambiguous/low-confidence results from the local parser

The core floor plan parsing pipeline works entirely offline without this module.
"""

import json


# Model tiers as described in CLAUDE.md
MODEL_OPUS = "claude-opus-4-6"      # Complex reasoning (default)
MODEL_SONNET = "claude-sonnet-4-6"  # Cost-efficient alternative


class ClaudeClient:
    """Client for Claude API integration (optional smart layer).

    Uses Opus 4.6 by default for complex architectural reasoning tasks.
    Can be switched to Sonnet 4.6 for cost efficiency on simpler queries.
    """

    def __init__(self, api_key, model=None):
        if not api_key:
            raise ValueError("Claude API key is required")
        self.api_key = api_key
        self.model = model or MODEL_OPUS
        self.base_url = "https://api.anthropic.com/v1"

    def suggest_furniture(self, rooms_data):
        """Suggest furniture placement based on room type and dimensions.

        Uses Opus 4.6 for spatial reasoning about furniture arrangements.

        Args:
            rooms_data: List of room dicts from the parsed floor plan.

        Returns:
            dict: Furniture placement suggestions per room.
        """
        prompt = self._build_furniture_prompt(rooms_data)
        response = self._call_api(prompt)
        return self._parse_json_response(response, "furniture suggestions")

    def interpret_modification(self, current_plan, natural_language_request):
        """Interpret a natural language modification request.

        Uses Opus 4.6 for understanding architectural intent
        (e.g., "make the kitchen bigger", "add a bathroom next to the master bedroom").

        Args:
            current_plan: Current floor plan JSON data.
            natural_language_request: User's modification request in plain English.

        Returns:
            dict: Modified floor plan data.
        """
        prompt = self._build_modification_prompt(current_plan, natural_language_request)
        response = self._call_api(prompt)
        return self._parse_json_response(response, "modified plan")

    def resolve_ambiguity(self, floor_plan_data, confidence_report):
        """Resolve ambiguous detections from the local model.

        When the local CV model flags low-confidence detections, Claude
        can apply architectural reasoning to resolve them.

        Args:
            floor_plan_data: Current parsed floor plan with confidence scores.
            confidence_report: Dict of low-confidence elements to resolve.

        Returns:
            dict: Corrected floor plan data.
        """
        prompt = self._build_ambiguity_prompt(floor_plan_data, confidence_report)
        response = self._call_api(prompt)
        return self._parse_json_response(response, "corrected plan")

    def critique_layout(self, floor_plan_data):
        """Provide design feedback and layout optimization suggestions.

        Args:
            floor_plan_data: Parsed floor plan data.

        Returns:
            dict: Critique with suggestions.
        """
        prompt = self._build_critique_prompt(floor_plan_data)
        response = self._call_api(prompt)
        return self._parse_json_response(response, "layout critique")

    def _call_api(self, prompt, max_tokens=4096):
        """Make a request to the Claude API.

        Uses the Messages API with the configured model (default: Opus 4.6).
        """
        import requests

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

        response = requests.post(
            f"{self.base_url}/messages",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()

        data = response.json()
        return data["content"][0]["text"]

    def _build_furniture_prompt(self, rooms_data):
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
            "- rotation: degrees (0 = aligned with X axis)\n\n"
            "Return valid JSON only, as a dict keyed by room label."
        )

    def _build_modification_prompt(self, current_plan, request):
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
            "Return the complete updated floor plan as valid JSON only, using the same schema."
        )

    def _build_ambiguity_prompt(self, floor_plan_data, confidence_report):
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
            "Return the corrected floor plan as valid JSON only, using the same schema."
        )

    def _build_critique_prompt(self, floor_plan_data):
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
            "Return valid JSON only."
        )

    @staticmethod
    def _parse_json_response(response_text, context="response"):
        """Parse JSON from Claude's response, handling markdown code blocks."""
        # Try direct parse first
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code blocks
        import re
        match = re.search(r"```(?:json)?\s*\n(.*?)\n```", response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse {context} from Claude response")
