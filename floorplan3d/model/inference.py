"""
Inference entry point for the fine-tuned floor plan VLM.

Loads the Qwen2.5-VL base + LoRA adapter from model/weights/, runs the
pipeline (VLM → optional Claude refiner), and prints the canonical floor
plan JSON to stdout. The VLM is the only automatic inference path; the
classical-CV extractor remains available solely as an explicit `--cv-only`
opt-in (debugging / no-GPU use) and is never used as a silent fallback.

This script is the target of LocalModelClient in blender_addon/api/local_model.py.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Shared with train.py. Any divergence between the two is a silent
# training/inference drift bug: the model conditions on this prefix at
# sampling time but was supervised against it at train time, so a change
# here without a retrain produces degraded output that eval picks up
# only after a full run. See prompts.py.
sys.path.insert(0, str(Path(__file__).parent))
from prompts import SYSTEM_PROMPT, USER_PROMPT  # type: ignore  # noqa: E402
from schema import serialize  # type: ignore  # noqa: E402

# Default decoding budget. Real-world MLS plans serialize to ~3000-4000
# tokens (55 walls / 13 rooms measured at 3193 tokens); the old 2048 default
# truncated them mid-JSON and the whole prediction was lost. 6144 covers
# every plan seen so far with headroom; runaway decodes are still bounded.
DEFAULT_MAX_NEW_TOKENS = 6144

# Base checkpoint used when weights_dir has no train_config.json — i.e. the
# add-on shipped without the fine-tuned adapter. The hybrid backend only
# needs the base model (grounded OCR), so this is the common shipped case.
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


def _resolve_base_model(weights_dir: Path) -> str:
    """Base model id from weights_dir/train_config.json, else the default."""
    cfg = weights_dir / "train_config.json"
    if cfg.exists():
        try:
            return json.loads(cfg.read_text())["base_model"]
        except (json.JSONDecodeError, KeyError, OSError):
            pass
    return DEFAULT_BASE_MODEL


def _load_vlm(weights_dir: Path, quantize: bool = False):
    """Load base model + processor + LoRA adapter from `weights_dir`.

    Split out of `run_vlm` so the daemon (`_run_serve`) can pay this 20-90s
    cost once at startup and amortize it across all subsequent inference
    requests. The one-shot `run_vlm` calls this every time and pays it
    on every Blender Generate click.

    Returns (model, processor). Both are stateless after construction —
    safe to share across many `_run_vlm_inference` calls.

    `quantize=True` loads the base model with the same 4-bit NF4 config
    training used, dropping the memory footprint from ~14 GB (bfloat16)
    to ~4 GB at a ~5-10% quality cost. Needed to fit on 16 GB GPUs
    (5070 Ti, 4090) but unnecessary — and net-worse — on the M4 Max's
    128 GB unified memory where full-precision load-and-go beats the
    extra bnb dequantize cost per token. Default off to prefer quality
    on the primary inference target.
    """
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    adapter_dir = weights_dir / "adapter"
    processor_dir = weights_dir / "processor"
    base = _resolve_base_model(weights_dir)

    processor = AutoProcessor.from_pretrained(
        str(processor_dir) if processor_dir.exists() else base,
        trust_remote_code=True,
    )

    # Match the training-time bnb config exactly when quantize=True.
    # Using a different quant layout (e.g. fp4 here vs nf4 in train.py)
    # would silently change the weight distribution the LoRA was
    # calibrated against, degrading output without raising.
    #
    # attn_implementation="sdpa" picks PyTorch's fused scaled-dot-product
    # attention kernel over the transformers "eager" fallback. SDPA is
    # always available (built into torch>=2.0), runs ~1.3x faster on the
    # ~1500-token prompt + ~500-800 token decode regime this model
    # operates in, and is numerically equivalent. flash_attention_2 is
    # ~2x faster on prefill but requires Ampere+ (sm_80+) and the
    # flash-attn package; SDPA is the portable default that also works
    # on M-series via MPS. No accuracy delta either way.
    load_kwargs: dict = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
    }
    if quantize:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base, **load_kwargs)
    if adapter_dir.exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    # Defensive: train.py disables use_cache (incompatible with
    # gradient_checkpointing — see train.py:415) and that flag can ride
    # along on the PEFT adapter save. Re-enable explicitly at inference
    # or every decode step recomputes the full prefill, blowing decode
    # latency by ~10x on a 1500-token prompt. transformers' generate()
    # default is True, but a stale config.use_cache=False from the
    # adapter overrides it silently.
    model.config.use_cache = True
    model.eval()
    return model, processor


def _run_vlm_inference(model, processor, image_path: str,
                       max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> dict:
    """Run inference on a pre-loaded model. Separates per-call cost
    (image preprocessing + generate, ~5-30s) from per-load cost
    (~20-90s) so the daemon path can reuse a single load across many
    images.
    """
    import torch
    from PIL import Image
    # smart_resize: Qwen2.5-VL's canonical aspect-preserving downscale to
    # dims that are multiples of 28 (the vision patch size) with area
    # capped at max_pixels. Used here to pre-compute the exact target
    # dims so we can apply LANCZOS resampling ourselves rather than
    # letting the processor's internal BICUBIC do it.
    from qwen_vl_utils.vision_process import smart_resize

    image = Image.open(image_path).convert("RGB")
    # Pre-resize with LANCZOS to Qwen-aligned dims (multiples of 28, the
    # vision patch size; area capped at max_pixels). Two wins versus the
    # previous `image.thumbnail((1024, 1024))`:
    #   1. LANCZOS preserves the 2-3 px wall lines that PIL's default
    #      BICUBIC smears at heavy downscale ratios — a real-world
    #      accuracy hit on CubiCasa-style line-art input.
    #   2. Computing the target dims ourselves means the processor's
    #      internal smart_resize sees an already-aligned image and the
    #      effective resampling is LANCZOS, not BICUBIC-then-BICUBIC.
    # max_pixels=1024*1024 mirrors train.py — drift here silently shifts
    # the supervised image-token distribution from what the LoRA was
    # trained against. Keep both call sites in sync.
    target_h, target_w = smart_resize(
        image.height, image.width, factor=28, max_pixels=1024 * 1024
    )
    image = image.resize((target_w, target_h), Image.LANCZOS)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    with torch.no_grad():
        # Greedy decoding: do_sample=False. Do NOT set temperature here —
        # temperature only applies to sampling, and transformers >= 4.45
        # raises / warns when you pair do_sample=False with an explicit
        # temperature (the value is a no-op but the config looks
        # contradictory). Passing only do_sample=False is the clean
        # greedy-decode contract.
        # repetition_penalty=1.05 is the anti-loop guard. Greedy decoding
        # on dense JSON output collapses into pivot-point loops on the
        # most complex plans (~25+ walls), emitting near-duplicate wall
        # entries until max_new_tokens. 1.05 is gentle enough not to
        # disturb the structural JSON repetition (`{"start":..."end":...}`
        # template recurs once per wall); anything >=1.10 starts harming
        # the well-behaved samples by skewing legitimate coordinate digit
        # repetition (numbers like 4.18 sharing tokens with 4.16 etc.).
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
        )
    generated = output_ids[0, inputs["input_ids"].shape[1] :]
    if generated.shape[0] >= max_new_tokens:
        # No EOS within budget: the JSON is cut off. The salvage path in
        # _deserialize_with_drift_repair keeps every complete element, so
        # the user gets a partial plan plus this warning instead of nothing.
        print(f"WARNING: generation hit the {max_new_tokens}-token budget; "
              "output is truncated and will be salvaged. Raise "
              "--max-new-tokens / FP3D_MAX_NEW_TOKENS for very large plans.",
              file=sys.stderr, flush=True)
    text_out = processor.tokenizer.decode(generated, skip_special_tokens=True)
    return _deserialize_with_drift_repair(text_out)


OCR_PROMPT = (
    "This is a residential floor plan. Find every ROOM NAME printed on the plan "
    "(e.g. LIVING ROOM, KITCHEN, PRIMARY BEDROOM, BATH, FOYER, HALL, W.I.C., BALCONY). "
    "Ignore dimension strings like 17'9\" and area notes. "
    "Output a JSON list only, one entry per room name, in the form "
    '[{"bbox_2d":[x1,y1,x2,y2],"text_content":"ROOM NAME"}].'
)

DIM_PROMPT = (
    "This is a residential floor plan. Find every DIMENSION STRING printed on the plan: "
    "lengths such as 17'9\", 24'5\", 12'-6\", 4.20, 350 cm. Ignore room names and area notes. "
    "Output a JSON list only, one entry per dimension string, in the form "
    '[{"bbox_2d":[x1,y1,x2,y2],"text_content":"17\'9\\""}].'
)

CROP_PROMPT = (
    "This is a small crop of one room from a residential floor plan. "
    "What room name is printed inside it (e.g. BEDROOM, W.I.C., FULL BATH, HALL)? "
    "Reply with the room name only. If no room name is printed, reply NONE."
)

# MLX build of the base model used on Apple Silicon. 8-bit: ~8 GB download,
# OCR quality indistinguishable from bf16 on printed text, several times the
# torch/MPS token rate.
MLX_MODEL_ID = "mlx-community/Qwen2.5-VL-7B-Instruct-8bit"


# ── shared image prep / parsing (backend-independent) ────────────────────

def _resized_for_vlm(image_path: str):
    """Open + aspect-preserving resize to Qwen's 28-px grid, ≤1 MP.
    Returns (resized PIL image, sx, sy) where (sx, sy) map resized→original px."""
    from PIL import Image
    from qwen_vl_utils.vision_process import smart_resize

    image = Image.open(image_path).convert("RGB")
    W, H = image.size
    target_h, target_w = smart_resize(H, W, factor=28, max_pixels=1024 * 1024)
    resized = image.resize((target_w, target_h), Image.LANCZOS)
    return resized, W / target_w, H / target_h


def _crop_for_vlm(image_path: str, bbox_px: list[float], margin: float = 0.15):
    """Crop one region (with margin) and upscale tiny crops so the text is
    legible to the vision encoder. Returns a PIL image or None if degenerate."""
    from PIL import Image
    from qwen_vl_utils.vision_process import smart_resize

    image = Image.open(image_path).convert("RGB")
    W, H = image.size
    x1, y1, x2, y2 = bbox_px
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    box = (max(0, int(x1 - mx)), max(0, int(y1 - my)), min(W, int(x2 + mx)), min(H, int(y2 + my)))
    if box[2] - box[0] < 8 or box[3] - box[1] < 8:
        return None
    crop = image.crop(box)
    th, tw = smart_resize(max(crop.height, 112), max(crop.width, 112), factor=28,
                          min_pixels=112 * 112, max_pixels=640 * 640)
    return crop.resize((tw, th), Image.LANCZOS)


def _parse_grounded_items(raw: str, sx: float, sy: float) -> list[dict]:
    """Parse Qwen's grounded-OCR JSON list into [{"text", "bbox_px"}] in
    ORIGINAL image pixels (Qwen emits coordinates in the resized frame)."""
    import re
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        salvaged = _salvage_truncated_json("{\"items\":" + m.group(0))
        try:
            items = json.loads(salvaged)["items"] if salvaged else []
        except json.JSONDecodeError:
            items = []
    out = []
    for it in items:
        box = it.get("bbox_2d") if isinstance(it, dict) else None
        label = (it.get("text_content") or it.get("label") or "") if isinstance(it, dict) else ""
        if not box or len(box) != 4 or not str(label).strip():
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in box)
        except (TypeError, ValueError):
            continue
        out.append({"text": str(label).strip(),
                    "bbox_px": [round(x1 * sx, 1), round(y1 * sy, 1), round(x2 * sx, 1), round(y2 * sy, 1)]})
    return out


TILE_MAX_PIXELS = 1400 * 1000   # per-tile budget for the dimension pass


def _tile_boxes(width: int, height: int, max_pixels: int = TILE_MAX_PIXELS,
                overlap: float = 0.12) -> list[tuple[int, int, int, int]]:
    """Split a large image into overlapping tiles of at most `max_pixels`
    each, so small printed text (dimension strings are ~1% of a plan's
    height) survives the vision encoder's 1 MP budget. A single full-image
    tile when it already fits."""
    if width * height <= max_pixels * 1.3:
        return [(0, 0, width, height)]
    # Smallest grid whose tiles fit the budget (2776x1788 -> 2x2 at 1.4 MP).
    nx = ny = 1
    for total in range(1, 64):
        options = [(a, math.ceil(total / a)) for a in range(1, total + 1) if a * math.ceil(total / a) == total]
        fits = [(a, b) for a, b in options if (width / a) * (height / b) <= max_pixels]
        if fits:
            # prefer the squarest tiles
            nx, ny = min(fits, key=lambda ab: abs((width / ab[0]) - (height / ab[1])))
            break
    tw, th = width / nx, height / ny
    ox, oy = tw * overlap, th * overlap
    boxes = []
    for j in range(ny):
        for i in range(nx):
            x1 = int(max(0, i * tw - ox))
            y1 = int(max(0, j * th - oy))
            x2 = int(min(width, (i + 1) * tw + ox))
            y2 = int(min(height, (j + 1) * th + oy))
            boxes.append((x1, y1, x2, y2))
    return boxes


def _merge_grounded_items(items: list[dict], iou_thresh: float = 0.4) -> list[dict]:
    """Drop duplicate detections from overlapping tiles (same text, boxes overlap)."""
    def iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area = lambda r: max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])  # noqa: E731
        union = area(a) + area(b) - inter
        return inter / union if union > 0 else 0.0
    kept: list[dict] = []
    for it in items:
        if any(k["text"].strip().lower() == it["text"].strip().lower() and iou(k["bbox_px"], it["bbox_px"]) >= iou_thresh
               for k in kept):
            continue
        kept.append(it)
    return kept


def _perimeter_strips(width: int, height: int, footprint_px, band_frac: float = 0.16,
                      max_pixels: int = TILE_MAX_PIXELS) -> list[tuple[int, int, int, int]]:
    """Four strips along the building's outer edges (top, bottom, left,
    right), each extended outward to catch the dimension line that runs
    outside the wall. The exterior dimensions are the long, reliable ones
    the scale estimate needs; interior closet dimensions are noise. Strips
    wider than `max_pixels` are split along their long axis."""
    fx1, fy1, fx2, fy2 = footprint_px
    fw, fh = max(1.0, fx2 - fx1), max(1.0, fy2 - fy1)
    bx, by = fw * band_frac, fh * band_frac
    raw = [
        (fx1 - bx, fy1 - by, fx2 + bx, fy1 + by),   # top
        (fx1 - bx, fy2 - by, fx2 + bx, fy2 + by),   # bottom
        (fx1 - bx, fy1 - by, fx1 + bx, fy2 + by),   # left
        (fx2 - bx, fy1 - by, fx2 + bx, fy2 + by),   # right
    ]
    strips = []
    for x1, y1, x2, y2 in raw:
        x1, y1 = int(max(0, x1)), int(max(0, y1))
        x2, y2 = int(min(width, x2)), int(min(height, y2))
        if x2 - x1 < 28 or y2 - y1 < 28:
            continue
        n = max(1, math.ceil((x2 - x1) * (y2 - y1) / max_pixels))
        if (x2 - x1) >= (y2 - y1):
            step = (x2 - x1) / n
            strips += [(int(x1 + i * step), y1, int(min(x2, x1 + (i + 1) * step + 0.1 * step)), y2) for i in range(n)]
        else:
            step = (y2 - y1) / n
            strips += [(x1, int(y1 + i * step), x2, int(min(y2, y1 + (i + 1) * step + 0.1 * step))) for i in range(n)]
    return strips


def _grounded_ocr_regions(generate_fn, image_path: str, prompt: str, max_new_tokens: int,
                          regions: list[tuple[int, int, int, int]] | None = None) -> list[dict]:
    """Run grounded OCR over `regions` of the image (default: a tile grid)
    and merge. `generate_fn(pil, prompt, max_new_tokens) -> str` is the
    backend's single-image call."""
    from PIL import Image
    from qwen_vl_utils.vision_process import smart_resize

    image = Image.open(image_path).convert("RGB")
    W, H = image.size
    out: list[dict] = []
    for (x1, y1, x2, y2) in (regions if regions else _tile_boxes(W, H)):
        tile = image.crop((x1, y1, x2, y2))
        th, tw = smart_resize(tile.height, tile.width, factor=28, max_pixels=TILE_MAX_PIXELS)
        resized = tile.resize((tw, th), Image.LANCZOS)
        sx, sy = tile.width / tw, tile.height / th
        for it in _parse_grounded_items(generate_fn(resized, prompt, max_new_tokens), sx, sy):
            b = it["bbox_px"]
            it["bbox_px"] = [round(b[0] + x1, 1), round(b[1] + y1, 1), round(b[2] + x1, 1), round(b[3] + y1, 1)]
            out.append(it)
    return _merge_grounded_items(out)


def _grounded_ocr_tiled(generate_fn, image_path: str, prompt: str, max_new_tokens: int) -> list[dict]:
    return _grounded_ocr_regions(generate_fn, image_path, prompt, max_new_tokens)


def _loop_detected(text: str, repeats: int = 3) -> bool:
    """True when the streamed JSON has emitted the same item `repeats`
    times — the model has started looping and nothing new will follow."""
    import re
    seen: dict[str, int] = {}
    for m in re.finditer(r'"text_content"\s*:\s*"((?:[^"\\]|\\.)*)"', text):
        seen[m.group(1)] = seen.get(m.group(1), 0) + 1
        if seen[m.group(1)] >= repeats:
            return True
    return False


def _clean_crop_text(raw: str) -> str:
    raw = (raw or "").strip().strip('"\'.')
    if not raw or raw.upper().startswith("NONE"):
        return ""
    return raw.splitlines()[0][:40]


def _select_backend(requested: str = "auto") -> str:
    """'mlx' on Apple Silicon when mlx_vlm imports, else 'torch'.
    FP3D_VLM_BACKEND env var overrides an 'auto' request."""
    import importlib.util
    import os
    import platform
    req = (os.environ.get("FP3D_VLM_BACKEND") or requested or "auto").lower()
    if req in ("torch", "mlx"):
        return req
    if sys.platform == "darwin" and platform.machine() == "arm64" \
            and importlib.util.find_spec("mlx_vlm") is not None:
        return "mlx"
    return "torch"


# ── torch backend ─────────────────────────────────────────────────────────

DIM_MAX_TOKENS = 700   # dense tiles list 25+ strings; 450 truncated them and starved auto-scale
DIM_REPETITION_PENALTY = 1.1


def _torch_generate(model, processor, pil_image, prompt: str, max_new_tokens: int,
                    repetition_penalty: float = 1.0) -> str:
    """One base-model (adapter disabled) generation on a single image."""
    import contextlib
    import torch
    messages = [{"role": "user", "content": [{"type": "image", "image": pil_image},
                                             {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[pil_image], return_tensors="pt").to(model.device)
    no_adapter = model.disable_adapter() if hasattr(model, "disable_adapter") else contextlib.nullcontext()
    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
    if repetition_penalty and repetition_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = repetition_penalty
    with torch.no_grad(), no_adapter:
        output_ids = model.generate(**inputs, **gen_kwargs)
    return processor.tokenizer.decode(output_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def _run_grounded_ocr(model, processor, image_path: str, prompt: str, max_new_tokens: int = 700) -> list[dict]:
    resized, sx, sy = _resized_for_vlm(image_path)
    return _parse_grounded_items(_torch_generate(model, processor, resized, prompt, max_new_tokens), sx, sy)


def _run_label_ocr(model, processor, image_path: str, max_new_tokens: int = 700) -> list[dict]:
    """Room names with positions (hybrid backend, label pass). Base model,
    adapter disabled: its grounded-OCR skill is what we want here."""
    return _run_grounded_ocr(model, processor, image_path, OCR_PROMPT, max_new_tokens)


def _dimension_regions(image_path: str, footprint_px):
    """Perimeter strips when the caller knows the building's pixel bbox
    (fast: only the long exterior dimensions), else a full tile grid."""
    if not footprint_px:
        return None
    from PIL import Image
    W, H = Image.open(image_path).size
    return _perimeter_strips(W, H, footprint_px) or None


def _run_dimension_ocr(model, processor, image_path: str, max_new_tokens: int = DIM_MAX_TOKENS,
                       footprint_px=None) -> list[dict]:
    """Dimension strings with positions (auto-scale pass). Reads the strips
    along the building outline at native resolution — dimension text is
    tiny, and a whole 2500-px plan squeezed into 1 MP loses it. A
    repetition penalty discourages the model from looping on a string."""
    return _grounded_ocr_regions(
        lambda im, pr, n: _torch_generate(model, processor, im, pr, n, DIM_REPETITION_PENALTY),
        image_path, DIM_PROMPT, max_new_tokens, regions=_dimension_regions(image_path, footprint_px))


def _run_crop_ocr(model, processor, image_path: str, bbox_px: list[float],
                  margin: float = 0.15, max_new_tokens: int = 12) -> str:
    """Name of the room printed inside one region (hybrid, second pass)."""
    crop = _crop_for_vlm(image_path, bbox_px, margin)
    if crop is None:
        return ""
    return _clean_crop_text(_torch_generate(model, processor, crop, CROP_PROMPT, max_new_tokens))


class TorchBackend:
    """HF transformers + torch. The only backend that can run the fine-tuned
    adapter (extract op); also serves the OCR ops."""
    name = "torch"

    def __init__(self, weights_dir: Path, quantize: bool = False):
        self.model, self.processor = _load_vlm(weights_dir, quantize=quantize)

    def extract(self, image_path: str, max_new_tokens: int) -> dict:
        return _run_vlm_inference(self.model, self.processor, image_path, max_new_tokens=max_new_tokens)

    def ocr_labels(self, image_path: str) -> list[dict]:
        return _run_label_ocr(self.model, self.processor, image_path)

    def ocr_dimensions(self, image_path: str, footprint_px=None) -> list[dict]:
        return _run_dimension_ocr(self.model, self.processor, image_path, footprint_px=footprint_px)

    def ocr_crop(self, image_path: str, bbox_px: list[float]) -> str:
        return _run_crop_ocr(self.model, self.processor, image_path, bbox_px)


# ── MLX backend (Apple Silicon) ───────────────────────────────────────────

class MlxBackend:
    """mlx-vlm running the 8-bit base model. OCR ops only: the fine-tuned
    PEFT adapter is a torch artifact, so `extract` is delegated by the daemon
    to a lazily-loaded TorchBackend."""
    name = "mlx"

    def __init__(self, model_id: str = MLX_MODEL_ID):
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
        self.model_id = model_id
        self.model, self.processor = load(model_id)
        self.config = load_config(model_id)

    def _generate(self, pil_image, prompt: str, max_tokens: int, repetition_penalty: float = 1.0,
                  stop_on_loop: bool = False) -> str:
        """One base-model generation. With `stop_on_loop`, streams and stops
        as soon as the JSON starts repeating an item (grounded-OCR lists
        otherwise run to the token cap once the model begins to loop)."""
        import os
        import tempfile
        from mlx_vlm import generate, stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template
        chat = apply_chat_template(self.processor, self.config, prompt, num_images=1)
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            pil_image.save(path)
            kwargs = {"max_tokens": max_tokens, "temperature": 0.0}
            if repetition_penalty and repetition_penalty != 1.0:
                kwargs["repetition_penalty"] = repetition_penalty
            if not stop_on_loop:
                out = generate(self.model, self.processor, chat, image=[path], verbose=False, **kwargs)
                return out.text if hasattr(out, "text") else str(out)
            text = ""
            n = 0
            for chunk in stream_generate(self.model, self.processor, chat, image=[path], **kwargs):
                text += chunk.text if hasattr(chunk, "text") else str(chunk)
                n += 1
                if n % 16 == 0 and _loop_detected(text):
                    break
            return text
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def extract(self, image_path: str, max_new_tokens: int) -> dict:
        raise RuntimeError("extract (fine-tuned adapter) is not available on the MLX backend")

    def ocr_labels(self, image_path: str) -> list[dict]:
        resized, sx, sy = _resized_for_vlm(image_path)
        return _parse_grounded_items(self._generate(resized, OCR_PROMPT, 700), sx, sy)

    def ocr_dimensions(self, image_path: str, footprint_px=None) -> list[dict]:
        return _grounded_ocr_regions(
            lambda im, pr, n: self._generate(im, pr, n, DIM_REPETITION_PENALTY, stop_on_loop=True),
            image_path, DIM_PROMPT, DIM_MAX_TOKENS, regions=_dimension_regions(image_path, footprint_px))

    def ocr_crop(self, image_path: str, bbox_px: list[float]) -> str:
        crop = _crop_for_vlm(image_path, bbox_px)
        if crop is None:
            return ""
        return _clean_crop_text(self._generate(crop, CROP_PROMPT, 12))


def run_vlm(image_path: str, weights_dir: Path, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
            quantize: bool = False) -> dict:
    """One-shot: load the VLM and run a single inference. Used by main()
    and kept as the stable entry point for evaluate.py — anything that
    only needs to score one image at a time pays the load cost once,
    here, rather than spinning up a daemon for a single call.

    For multi-image workflows (the Blender add-on, batched eval) use
    `_run_serve` via `python inference.py --serve` so the load cost is
    paid once across all calls.

    See `_load_vlm` for the quantize semantics and `_run_vlm_inference`
    for the max_new_tokens semantics — both apply unchanged here.
    """
    model, processor = _load_vlm(weights_dir, quantize=quantize)
    return _run_vlm_inference(model, processor, image_path,
                              max_new_tokens=max_new_tokens)


def _run_serve(weights_dir: Path, quantize: bool = False, backend: str = "auto") -> int:
    """Daemon mode: load the model once, then loop on stdin requests.

    Protocol (newline-delimited JSON over the subprocess pipes):

        * On startup, daemon writes the literal line "READY" to stderr
          once the model is loaded and ready to accept requests. Until
          that line, the client must NOT send any request — generate()
          on a half-loaded model is undefined behavior.
        * Each stdin line is one JSON object: {"image": "<path>",
          "refine": <bool>, "max_new_tokens": <int|null>}. Unknown keys
          are ignored so the protocol can extend forward-compatibly.
        * Each stdout line is one JSON object: either
          {"ok": true, "result": <floor_plan_dict>} on success or
          {"ok": false, "error": "<message>"} on failure.
        * The daemon does NOT exit on per-request errors — the model is
          still loaded, and the next request is unrelated. Only fatal
          conditions (EOF on stdin, unrecoverable model state) end the
          loop.
        * One blank stdin line is tolerated (skipped) — convenient for
          manual debugging with `cat | python inference.py --serve`.

    Returns the process exit code so main() can sys.exit cleanly.
    """
    be = _select_backend(backend)
    print(f"backend: {be}", file=sys.stderr, flush=True)
    primary = MlxBackend() if be == "mlx" else TorchBackend(weights_dir, quantize=quantize)
    torch_be = primary if isinstance(primary, TorchBackend) else None
    # Single-token sentinel on stderr signals "model loaded, accepting
    # requests". The client (LocalModelClient) blocks on this exact
    # string before sending the first request. Anything else on stderr
    # before READY is treated as a startup error.
    print("READY", file=sys.stderr, flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        response: dict
        try:
            req = json.loads(line)
            image_path = req["image"]
            op = req.get("op") or "extract"
            if op in ("ocr_labels", "ocr_dimensions", "ocr_crop"):
                # Hybrid / auto-scale passes: base model on the fast backend.
                if op == "ocr_crop":
                    result = primary.ocr_crop(image_path, req["bbox_px"])
                elif op == "ocr_dimensions":
                    result = primary.ocr_dimensions(image_path, footprint_px=req.get("footprint_px"))
                else:
                    result = getattr(primary, op)(image_path)
                sys.stdout.write(json.dumps({"ok": True, "result": result}) + "\n")
                sys.stdout.flush()
                continue
            if torch_be is None:
                # The fine-tuned adapter only runs on torch; load it on the
                # first extract request so MLX-only sessions never pay for it.
                print("loading torch backend for the fine-tuned adapter…", file=sys.stderr, flush=True)
                torch_be = TorchBackend(weights_dir, quantize=quantize)
            max_new_tokens = int(req.get("max_new_tokens") or DEFAULT_MAX_NEW_TOKENS)
            result = torch_be.extract(image_path, max_new_tokens)
            if req.get("refine"):
                # Lazy: refine pulls in the Anthropic SDK which we don't
                # want loaded in daemons that never refine. Importing
                # here instead of at the top keeps `--serve` startup
                # zero-cost for the common case.
                sys.path.insert(0, str(Path(__file__).parent))
                from claude_refiner import refine  # type: ignore
                result = refine(image_path, result)
            response = {"ok": True, "result": result}
        except Exception as e:
            # Per-request errors must NOT kill the daemon. A bad path,
            # a corrupt image, a Claude API timeout — none of those
            # invalidate the loaded model, and respawning a 14 GB model
            # for a recoverable error would defeat the whole daemon
            # design. Surface the error in the response and continue.
            response = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        # Single-line JSON keeps the framing trivial: client does one
        # readline() per request and gets exactly one response back.
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
    return 0


def _salvage_truncated_json(s: str) -> str | None:
    """Repair a JSON document whose tail was cut off by the decoding budget.

    Scans the text tracking string state and the open-bracket stack, and
    remembers every position where an element inside a top-level array
    was just completed (depth 2, e.g. one wall / door / room object). The
    result is the text cut at the last such point with the remaining
    brackets closed — i.e. every complete element survives and only the
    partially-emitted trailing one is dropped. Returns None if no
    complete element can be recovered.
    """
    start = s.find("{")
    if start == -1:
        return None
    stack: list[str] = []
    in_str = False
    esc = False
    best: tuple[int, list[str]] | None = None
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c in "{[":
            stack.append("}" if c == "{" else "]")
        elif c in "}]":
            if not stack or stack[-1] != c:
                break  # malformed beyond repair
            stack.pop()
            if len(stack) == 2 and stack[-1] == "]":
                best = (i + 1, list(stack))
    if best is None:
        return None
    cut, remaining = best
    return s[start:cut] + "".join(reversed(remaining))


def _deserialize_with_drift_repair(text_out: str) -> dict:
    """Parse VLM output, clamping out-of-range wall_index references.

    On long structured outputs the model occasionally drifts in its wall
    count, leaving doors/windows pointing at a wall_index that wasn't
    emitted. The strict validator rejects the whole output, which then
    triggers a CV fallback whose 1000+ over-segmented walls are far worse
    than the VLM's mostly-correct output minus a stale reference. Clamp
    bad indices to -1 (the existing "unassigned" sentinel; the geometry
    layer falls back to position-based snapping for those entries).
    """
    from schema import SchemaError, validate, deserialize  # type: ignore  # noqa: E402
    import json as _json
    try:
        return deserialize(text_out)
    except _json.JSONDecodeError:
        salvaged = _salvage_truncated_json(text_out)
        if salvaged is None:
            raise
        print("WARNING: model output was truncated mid-JSON; salvaged the "
              "complete elements and dropped the partial tail.",
              file=sys.stderr, flush=True)
        return _deserialize_with_drift_repair(salvaged)
    except SchemaError as e:
        if "wall_index" not in str(e):
            raise
    # Re-parse with the same permissive logic as deserialize, then clamp.
    s = text_out.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1:
        raise SchemaError("no JSON object found in model output")
    data = _json.loads(s[start : end + 1])
    for key in ("walls", "doors", "windows", "rooms"):
        data.setdefault(key, [])
    data.setdefault("scale", {"pixels_per_meter": 50})
    n_walls = len(data["walls"])
    for x in data["doors"]:
        if isinstance(x.get("wall_index"), int) and x["wall_index"] >= n_walls:
            x["wall_index"] = -1
    for x in data["windows"]:
        if isinstance(x.get("wall_index"), int) and x["wall_index"] >= n_walls:
            x["wall_index"] = -1
    validate(data)
    return data


def run_cv_only(image_path: str, ppm: float = 50.0) -> dict:
    """Geometry-only fallback when no trained VLM is available yet."""
    sys.path.insert(0, str(Path(__file__).parent))
    from cv_walls import extract, CVConfig  # type: ignore
    return extract(image_path, CVConfig(pixels_per_meter=ppm))


def main():
    parser = argparse.ArgumentParser(description="Run floor plan inference")
    # --image is required for one-shot mode but unused (and unwanted)
    # in --serve mode where image paths arrive on stdin per request.
    # Defaulting to None and validating downstream lets argparse stay
    # declarative without a custom subcommand split.
    parser.add_argument("--image", default=None)
    parser.add_argument("--weights", default=str(Path(__file__).parent / "weights"))
    parser.add_argument("--output", choices=["json", "file"], default="json")
    parser.add_argument("--out-path", default="output.json")
    parser.add_argument("--ppm", type=float, default=50.0)
    parser.add_argument("--cv-only", action="store_true",
                        help="Skip the VLM and use only the CV fallback.")
    parser.add_argument("--refine", action="store_true",
                        help="Run the Claude refiner on low-confidence regions.")
    parser.add_argument("--quantize", action="store_true",
                        help="Load the VLM in 4-bit NF4 to fit 16 GB GPUs "
                             "(5070 Ti, 4090). Not needed on Apple Silicon "
                             "unified memory or H100; costs ~5-10%% output "
                             "quality for ~3.5x lower peak VRAM.")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                        help="Decoding budget for the VLM (default %(default)s)")
    parser.add_argument("--ocr-labels", action="store_true",
                        help="Instead of extracting geometry, list the printed room names "
                             "with pixel boxes (hybrid backend's label pass)")
    parser.add_argument("--backend", choices=["auto", "torch", "mlx"], default="auto",
                        help="OCR backend: mlx on Apple Silicon when mlx-vlm is installed, else torch "
                             "(the fine-tuned adapter always runs on torch)")
    parser.add_argument("--serve", action="store_true",
                        help="Daemon mode: load the model once and read "
                             "JSON requests on stdin, write JSON responses "
                             "on stdout. Used by LocalModelClient to amortize "
                             "the ~30-90s cold-start cost across many "
                             "Blender Generate clicks. See _run_serve for "
                             "the protocol.")
    args = parser.parse_args()

    if args.serve:
        # Serve mode skips the one-shot pipeline entirely. cv_only,
        # refine, image, output, out-path, ppm are all per-request
        # concerns in --serve mode and live in the JSON protocol, not
        # on argv. Only --weights and --quantize affect the daemon
        # (load-time config).
        sys.exit(_run_serve(Path(args.weights), quantize=args.quantize, backend=args.backend))

    if not args.image:
        parser.error("--image is required unless --serve is set")

    weights_dir = Path(args.weights)

    if args.cv_only:
        # Explicit opt-in to the classical-CV extractor (debugging / no-GPU
        # use). This is the ONLY way to reach the CV path — the VLM no
        # longer silently falls back to CV on failure.
        result = run_cv_only(args.image, args.ppm)
    else:
        # VLM is the only automatic inference path. Missing weights or a
        # failed generate() raise a clear error instead of degrading to a
        # lower-quality CV result behind the user's back — a corrupt
        # adapter, OOM, or unreadable image should be visible, not masked.
        if not weights_dir.exists() or not (weights_dir / "train_config.json").exists():
            raise SystemExit(
                f"No trained model found at {weights_dir} (expected an "
                f"'adapter' directory and train_config.json). Train the VLM "
                f"first (see model/train.py), point --weights at the weights "
                f"directory, or pass --cv-only to use the classical-CV extractor."
            )
        try:
            if args.ocr_labels:
                be = _select_backend(args.backend)
                backend_obj = MlxBackend() if be == "mlx" else TorchBackend(weights_dir, quantize=args.quantize)
                result = backend_obj.ocr_labels(args.image)
            else:
                result = run_vlm(args.image, weights_dir, max_new_tokens=args.max_new_tokens,
                                 quantize=args.quantize)
        except Exception as e:
            raise SystemExit(
                f"VLM inference failed: {type(e).__name__}: {e}. Pass "
                f"--cv-only to use the classical-CV extractor instead."
            )

    if args.refine:
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from claude_refiner import refine  # type: ignore
            result = refine(args.image, result)
        except Exception as e:
            print(f"[warn] refiner failed, using unrefined output: {e}", file=sys.stderr)

    # Use the canonical serializer, not plain json.dumps, so the output
    # matches the exact key order, rounding precision, and compact
    # format the VLM was trained against. Anything that pattern-matches
    # on the output (the Blender add-on, downstream eval, diffing two
    # runs) would otherwise see spurious changes from dict insertion
    # order or numeric-precision drift.
    text = serialize(result)
    if args.output == "json":
        print(text)
    else:
        Path(args.out_path).write_text(text)
        print(f"wrote {args.out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
