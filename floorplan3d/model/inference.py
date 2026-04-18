"""
Inference entry point for the fine-tuned floor plan VLM.

Loads the Qwen2.5-VL base + LoRA adapter from model/weights/, runs the
hybrid pipeline (optional CV pre-pass → VLM → optional Claude refiner),
and prints the canonical floor plan JSON to stdout.

This script is the target of LocalModelClient in blender_addon/api/local_model.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Shared with train.py. Any divergence between the two is a silent
# training/inference drift bug: the model conditions on this prefix at
# sampling time but was supervised against it at train time, so a change
# here without a retrain produces degraded output that eval picks up
# only after a full run. See prompts.py.
sys.path.insert(0, str(Path(__file__).parent))
from prompts import SYSTEM_PROMPT, USER_PROMPT  # type: ignore  # noqa: E402


def run_vlm(image_path: str, weights_dir: Path, max_new_tokens: int = 4096) -> dict:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent))
    from schema import deserialize  # type: ignore

    adapter_dir = weights_dir / "adapter"
    processor_dir = weights_dir / "processor"
    meta = json.loads((weights_dir / "train_config.json").read_text())
    base = meta["base_model"]

    processor = AutoProcessor.from_pretrained(
        str(processor_dir) if processor_dir.exists() else base,
        trust_remote_code=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_dir.exists():
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    image = Image.open(image_path).convert("RGB")
    if max(image.size) > 1024:
        image.thumbnail((1024, 1024))

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
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    generated = output_ids[0, inputs["input_ids"].shape[1] :]
    text_out = processor.tokenizer.decode(generated, skip_special_tokens=True)
    return deserialize(text_out)


def run_cv_only(image_path: str, ppm: float = 50.0) -> dict:
    """Geometry-only fallback when no trained VLM is available yet."""
    sys.path.insert(0, str(Path(__file__).parent))
    from cv_walls import extract, CVConfig  # type: ignore
    return extract(image_path, CVConfig(pixels_per_meter=ppm))


def main():
    parser = argparse.ArgumentParser(description="Run floor plan inference")
    parser.add_argument("--image", required=True)
    parser.add_argument("--weights", default=str(Path(__file__).parent / "weights"))
    parser.add_argument("--output", choices=["json", "file"], default="json")
    parser.add_argument("--out-path", default="output.json")
    parser.add_argument("--ppm", type=float, default=50.0)
    parser.add_argument("--cv-only", action="store_true",
                        help="Skip the VLM and use only the CV fallback.")
    parser.add_argument("--refine", action="store_true",
                        help="Run the Claude refiner on low-confidence regions.")
    args = parser.parse_args()

    weights_dir = Path(args.weights)
    use_vlm = (
        not args.cv_only
        and weights_dir.exists()
        and (weights_dir / "train_config.json").exists()
    )

    if use_vlm:
        try:
            result = run_vlm(args.image, weights_dir)
        except Exception as e:
            print(f"[warn] VLM failed, falling back to CV: {e}", file=sys.stderr)
            result = run_cv_only(args.image, args.ppm)
    else:
        result = run_cv_only(args.image, args.ppm)

    if args.refine:
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from claude_refiner import refine  # type: ignore
            result = refine(args.image, result)
        except Exception as e:
            print(f"[warn] refiner failed, using unrefined output: {e}", file=sys.stderr)

    text = json.dumps(result)
    if args.output == "json":
        print(text)
    else:
        Path(args.out_path).write_text(text)
        print(f"wrote {args.out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
