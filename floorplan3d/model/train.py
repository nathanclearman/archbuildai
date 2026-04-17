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
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

# NOTE: heavy imports (torch / transformers / peft) happen inside main() so
# `python train.py --help` works without an ML env, and so the Blender
# add-on process never drags them in by accident. PIL is cheap and already
# a hard project dep, so it lives at module scope — FloorPlanDS needs it
# to be importable by multiprocessing workers under `spawn` start method.


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
    # Fraction of the corpus held out for eval. Computed as
    # max(1, int(len(samples) * eval_split)) — never zero for non-empty
    # corpora. 10% is the standard compromise: small enough to keep
    # most samples in train, large enough to stabilise the eval_loss
    # signal for checkpoint selection.
    eval_split: float = 0.10
    # Eval happens every this many optimizer steps. Must equal save_steps
    # when load_best_model_at_end=True, so a single knob covers both.
    eval_steps: int = 500


# Coarse char-per-token upper bound for Qwen's BPE on JSON text. JSON is
# dense in short tokens (punctuation, digit runs) and averages ~2.5-3.2
# chars/token in practice; 4.0 is a conservative ceiling — any target
# whose raw chars exceed max_length * this factor cannot possibly fit
# after tokenization, regardless of prompt overhead. Borderline samples
# (over max_length*2 but under max_length*4) still reach training and
# get a precise token-count check inside format_conversation.
MAX_TARGET_CHARS_PER_TOKEN = 4.0


def build_samples(cfg: TrainConfig):
    """Return a list of (image_path, target_json_string) samples.

    Applies a cheap char-based pre-filter to drop targets that cannot
    fit within cfg.max_length tokens. Exact token counting requires the
    processor and would cost a full tokenization pass over the corpus;
    a conservative char bound catches the obvious outliers (e.g. dense
    CubiCasa plans with hundreds of walls) without the overhead.
    """
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

    kept, dropped, char_budget = _filter_oversized_samples(samples, cfg.max_length)
    if dropped:
        print(
            f"pre-filter: dropped {dropped}/{len(samples)} samples with "
            f"target JSON over {char_budget} chars (~{cfg.max_length} tokens). "
            f"These would have crashed format_conversation mid-training."
        )
    print(f"loaded {len(kept)} samples")
    return kept


def _filter_oversized_samples(samples, max_length):
    """Drop samples whose raw target JSON exceeds the char budget.

    Returns (kept, dropped_count, char_budget). Pure function so it's
    unit-testable without a dataset on disk.
    """
    char_budget = int(max_length * MAX_TARGET_CHARS_PER_TOKEN)
    kept = [s for s in samples if len(s.target_json) <= char_budget]
    return kept, len(samples) - len(kept), char_budget


def _split_eval(samples, eval_split: float, seed: int) -> tuple[list, list]:
    """Deterministic held-out split.

    Shuffles with an explicit `random.Random(seed)` and takes the first
    `int(n * eval_split)` samples as eval. Previously the function
    relied on the shuffle that happens inside `build_training_set`,
    but that's a fragile cross-function coupling — a refactor that
    reordered or removed the upstream shuffle would silently make the
    eval set "the first 10% by load order", which could be all
    CubiCasa or all one template if the loader batches by directory.
    Shuffling here pins the invariant locally.

    Floors the eval half at 1 sample on any non-empty corpus so the
    Trainer always has something to evaluate — a zero-sized
    eval_dataset silently skips eval and trains blind.

    Returns (train, eval). The train order is whatever the shuffle
    produced; the Trainer's DataLoader will reshuffle it each epoch,
    so train-side order inside this function doesn't affect the run.
    """
    if not samples:
        return [], []
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    n_eval = max(1, int(len(shuffled) * eval_split))
    return shuffled[n_eval:], shuffled[:n_eval]


def format_conversation(processor, image, target_json, max_length):
    """Format a single sample as a Qwen2.5-VL chat conversation.

    Raises RuntimeError if the tokenized sequence exceeds `max_length`.
    The previous `truncation=True` silently cut the assistant reply mid-
    JSON, teaching the model to emit malformed output — a training-data
    corruption invisible on the loss curve. Loud failure here lets
    callers filter long samples before training starts; the caller
    build_samples() does a coarse char-based pre-filter so the raise
    only fires on borderline cases.
    """
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
        truncation=False,
    )
    seq_len = inputs["input_ids"].shape[1]
    if seq_len > max_length:
        raise RuntimeError(
            f"Sample exceeds max_length={max_length} tokens (got {seq_len}). "
            f"Filter long samples in build_samples() before training, or "
            f"raise TrainConfig.max_length. Silent truncation would corrupt "
            f"the target JSON and teach the model to emit malformed output."
        )
    # Mask everything except the assistant's reply so loss is only on the target.
    input_ids = inputs["input_ids"][0]
    labels = input_ids.clone()
    assistant_token_ids = processor.tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
    cutoff = _mask_prompt_cutoff(input_ids.tolist(), assistant_token_ids)
    labels[:cutoff] = -100
    inputs["labels"] = labels.unsqueeze(0)
    return inputs


def _find_subseq(seq, sub):
    for i in range(len(seq) - len(sub) + 1):
        if seq[i : i + len(sub)] == sub:
            return i
    return -1


def _mask_prompt_cutoff(input_ids: list[int], assistant_token_ids: list[int]) -> int:
    """Return the label-masking cutoff — the index just past the assistant
    role boundary in `input_ids`. Everything strictly before the returned
    index should be set to -100 so loss is computed only on the assistant
    reply.

    Raises RuntimeError if the boundary isn't found. The previous silent
    fallback (leave all tokens as labels) would teach the model to
    reproduce the system prompt + image tokens as supervision targets,
    producing a mediocre loss curve that looks like a data problem. A
    loud failure at step 0 costs nothing; a silent one costs a training
    run.
    """
    if not assistant_token_ids:
        raise RuntimeError(
            "Empty assistant_token_ids — the tokenizer produced no tokens "
            "for '<|im_start|>assistant'. The chat template or tokenizer "
            "config has changed in an incompatible way."
        )
    boundary = _find_subseq(input_ids, assistant_token_ids)
    if boundary == -1:
        raise RuntimeError(
            "Could not locate the assistant-role boundary token sequence "
            "in the tokenized conversation. The chat template or "
            "processor.tokenizer no longer emits '<|im_start|>assistant' "
            "as a contiguous subsequence — verify the base model and "
            "transformers version. Training with a silent fallback would "
            "mask nothing and leak the prompt into supervision."
        )
    return boundary + len(assistant_token_ids)


class FloorPlanDS:
    """Map-style dataset over (image_path, target_json) samples.

    No `torch.utils.data.Dataset` inheritance — PyTorch's DataLoader
    duck-types on `__len__` + `__getitem__`, and keeping this class at
    module scope (not nested inside main()) matters for pickling under
    the `spawn` multiprocessing start method. A nested class would
    fail with `PicklingError: Can't pickle ...<locals>.FloorPlanDS` the
    moment dataloader_num_workers > 0 on macOS or Windows — a Linux-
    fork-only bug that vanishes in a CI move.
    """

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=TrainConfig.base_model)
    parser.add_argument("--cubicasa")
    parser.add_argument("--synthetic")
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
    from transformers import (
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
        TrainingArguments,
        Trainer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    samples = build_samples(cfg)
    train_samples, eval_samples = _split_eval(samples, cfg.eval_split, cfg.seed)
    print(f"split: {len(train_samples)} train / {len(eval_samples)} eval "
          f"(eval_split={cfg.eval_split:.2f})")

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
            if k == "labels":
                # -100 is cross-entropy's ignore_index — padded positions
                # must NOT contribute to the training loss. Padding with 0
                # (a real token id in Qwen's vocab) would teach the model
                # to predict that token on padding, which is silent
                # corruption at batch_size > 1.
                out[k] = torch.nn.utils.rnn.pad_sequence(
                    [v.squeeze(0) for v in vals], batch_first=True, padding_value=-100
                )
            elif k in ("input_ids", "attention_mask"):
                # input_ids at padded positions are ignored by the forward
                # because attention_mask=0 masks them out, so padding value
                # doesn't affect loss — 0 is conventional.
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
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=3,
        # Eval cadence matches save cadence so load_best_model_at_end
        # can pair each checkpoint with the eval_loss from that step.
        # Using a separate eval_steps would either skip pairings or
        # duplicate evals — HF requires eval_strategy == save_strategy
        # for best-checkpoint tracking to work.
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        # Recent torch flags the default (reentrant) checkpointing path
        # with a per-step deprecation warning under PEFT + bnb. Opting
        # into the non-reentrant variant silences it and keeps behaviour
        # identical for this training loop.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Explicit to match the HF default. Stated loudly because QLoRA
        # on a VLM is known to spike on pathological samples and the
        # default is doing real work — not a free parameter to silently
        # inherit.
        max_grad_norm=1.0,
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
        train_dataset=FloorPlanDS(train_samples, processor, cfg.max_length),
        eval_dataset=FloorPlanDS(eval_samples, processor, cfg.max_length) if eval_samples else None,
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
