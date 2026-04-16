"""
Supervised fine-tune of Qwen2.5-VL for floor plan vectorization.

Uses QLoRA via PEFT to fit on a single H100 80GB with Qwen2.5-VL-7B
(or a 3B profile for dev). Output adapter is merged and saved to
model/weights/ so inference.py can load it.

Usage (inside a GPU env with `pip install -r model/requirements.txt`):
    python model/train.py \
        --base Qwen/Qwen2.5-VL-7B-Instruct \
        --cubicasa data/cubicasa5k \
        --synthetic data/synthetic \
        --out model/weights \
        --epochs 2 --batch-size 1 --grad-accum 16

See runpod_launcher.md for end-to-end cloud instructions.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

# NOTE: heavy imports happen inside main() so `python train.py --help` works
# without torch installed, and so the Blender add-on process never drags
# these in by accident.


SYSTEM_PROMPT = (
    "You are a floor plan vectorization model. Given a raster floor plan "
    "image, emit a JSON object with keys 'scale', 'walls', 'doors', "
    "'windows', 'rooms' matching the canonical schema. Respond with JSON "
    "only — no prose, no code fences."
)

USER_PROMPT = "Vectorize this floor plan."


@dataclass
class TrainConfig:
    base_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    cubicasa_root: str | None = None
    synthetic_root: str | None = None
    resplan_root: str | None = None
    output_dir: str = "model/weights"
    epochs: int = 2
    per_device_batch_size: int = 1
    grad_accum: int = 16
    learning_rate: float = 2e-5
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    max_length: int = 4096
    warmup_ratio: float = 0.03
    save_steps: int = 500
    logging_steps: int = 10
    dataloader_num_workers: int = 2
    seed: int = 0


def build_samples(cfg: TrainConfig):
    """Return a list of (image_path, target_json_string) samples."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from dataset import build_training_set  # type: ignore

    samples = build_training_set(
        cubicasa_root=cfg.cubicasa_root,
        synthetic_root=cfg.synthetic_root,
        shuffle=True,
        seed=cfg.seed,
    )
    if not samples:
        raise RuntimeError(
            "No training samples found. Point --cubicasa at a CubiCasa5k "
            "extraction, or --synthetic at an output of synthesize.py."
        )
    print(f"loaded {len(samples)} samples")
    return samples


def format_conversation(processor, image, target_json, max_length):
    """Format a single sample as a Qwen2.5-VL chat conversation."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
        {"role": "assistant", "content": target_json},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_length,
    )
    # Mask everything except the assistant's reply so loss is only on the target.
    input_ids = inputs["input_ids"][0]
    labels = input_ids.clone()
    # Find the last "<|im_start|>assistant" token boundary — everything before
    # should be -100. This relies on Qwen's chat template; adjust if you swap
    # base models.
    assistant_token_ids = processor.tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
    boundary = _find_subseq(input_ids.tolist(), assistant_token_ids)
    if boundary != -1:
        cutoff = boundary + len(assistant_token_ids)
        labels[:cutoff] = -100
    inputs["labels"] = labels.unsqueeze(0)
    return inputs


def _find_subseq(seq, sub):
    for i in range(len(seq) - len(sub) + 1):
        if seq[i : i + len(sub)] == sub:
            return i
    return -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=TrainConfig.base_model)
    parser.add_argument("--cubicasa")
    parser.add_argument("--synthetic")
    parser.add_argument("--resplan")
    parser.add_argument("--out", default=TrainConfig.output_dir)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.per_device_batch_size)
    parser.add_argument("--grad-accum", type=int, default=TrainConfig.grad_accum)
    parser.add_argument("--lr", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--lora-r", type=int, default=TrainConfig.lora_r)
    parser.add_argument("--max-length", type=int, default=TrainConfig.max_length)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = TrainConfig(
        base_model=args.base,
        cubicasa_root=args.cubicasa,
        synthetic_root=args.synthetic,
        resplan_root=args.resplan,
        output_dir=args.out,
        epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        max_length=args.max_length,
        seed=args.seed,
    )

    # Heavy imports here so --help works without ML deps.
    import torch
    from PIL import Image
    from transformers import (
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
        TrainingArguments,
        Trainer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import Dataset

    samples = build_samples(cfg)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"loading base model: {cfg.base_model}")
    processor = AutoProcessor.from_pretrained(cfg.base_model, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    # gradient_checkpointing saves activations by re-running forward on
    # backward, which is incompatible with the KV cache. Leaving use_cache
    # on triggers a per-step warning from transformers and in some
    # versions a silent correctness bug. Disable explicitly.
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )

    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    class FloorPlanDS(Dataset):
        def __init__(self, samples, processor, max_length):
            self.samples = samples
            self.processor = processor
            self.max_length = max_length

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            s = self.samples[idx]
            image = Image.open(s.image_path).convert("RGB")
            # Downscale very large plans to keep token count tractable.
            if max(image.size) > 1024:
                image.thumbnail((1024, 1024))
            return format_conversation(self.processor, image, s.target_json, self.max_length)

    def data_collator(batch):
        # Processor already produced tensors with a leading batch dim of 1
        # in __getitem__, so a single-example batch is ready to go as-is.
        # Squeezing would drop that dim and crash the model's forward on
        # step 1 — the kind of bug that silently runs on a dry CPU smoke
        # but turns a cloud pod into $5-of-nothing.
        if len(batch) == 1:
            return batch[0]
        keys = batch[0].keys()
        out = {}
        for k in keys:
            vals = [b[k] for b in batch]
            # Text tensors have shape [1, seq_len] from the processor; drop
            # the batch-of-1 so pad_sequence can stack to [B, max_len].
            # pixel_values / image_grid_thw are passed through because Qwen
            # flattens patches across the batch with per-image grids, and
            # pad_sequence would corrupt that layout.
            if k in ("input_ids", "attention_mask", "labels"):
                out[k] = torch.nn.utils.rnn.pad_sequence(
                    [v.squeeze(0) for v in vals], batch_first=True, padding_value=0
                )
            else:
                out[k] = torch.cat(vals, dim=0) if hasattr(vals[0], "shape") else vals
        return out

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        # Overlap image decode + thumbnail with the GPU step. 0 (default)
        # serializes everything on the main process and bottlenecks the
        # H100 on PIL; 2 is enough for a single-GPU VLM job and avoids
        # worker-fork overhead that larger values impose.
        dataloader_num_workers=cfg.dataloader_num_workers,
        report_to="none",
        remove_unused_columns=False,
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=FloorPlanDS(samples, processor, cfg.max_length),
        data_collator=data_collator,
    )

    print("starting training")
    trainer.train(resume_from_checkpoint=args.resume)

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out / "adapter"))
    processor.save_pretrained(str(out / "processor"))
    (out / "train_config.json").write_text(json.dumps(cfg.__dict__, indent=2))
    print(f"done. adapter + processor saved to {out}")


if __name__ == "__main__":
    main()
