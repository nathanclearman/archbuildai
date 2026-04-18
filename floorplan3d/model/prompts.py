"""
Chat-prompt constants shared by the train and inference entry points.

Single source of truth for the instruction pair the VLM is both trained
against and generated from. Drift between training-time and inference-
time prompts is a silent correctness bug: the tokenized prefix the model
sees at sampling time no longer matches the one it was supervised on, so
the conditional distribution shifts in ways eval metrics flag only
after a full run. Keeping both sides importing from here makes that
class of bug a compile error instead of a training artefact.

Kept tiny and dep-free so both the Blender add-on subprocess and the
cloud training job can import it without dragging in torch / PIL.
"""

from __future__ import annotations


SYSTEM_PROMPT: str = (
    "You are a floor plan vectorization model. Given a raster floor plan "
    "image, emit a JSON object with keys 'scale', 'walls', 'doors', "
    "'windows', 'rooms' matching the canonical schema. Respond with JSON "
    "only — no prose, no code fences."
)

USER_PROMPT: str = "Vectorize this floor plan."
