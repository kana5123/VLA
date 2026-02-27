#!/usr/bin/env python3
"""LIBERO LoRA Fine-Tuning for Bottleneck VLA Models (OpenVLA, ECoT).

Fine-tunes VLA models on LIBERO demonstration data using LoRA.
Purpose: probe whether position-anchoring bottleneck persists after
task-specific fine-tuning (structural vs adaptive routing question).

Data: LIBERO HDF5 demos with actions already in [-1, 1] range.
Loss: Teacher-forced cross-entropy on 7 discrete action tokens.

Usage:
    python train_libero_lora.py --model openvla-7b --suite libero_spatial \
        --device cuda:0 --max_steps 5000 --lr 5e-4 --lora_r 32
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None

sys.path.insert(0, str(Path(__file__).parent))

import config
from extract_attention import (
    load_model_from_registry,
    call_processor,
    detect_token_boundaries,
)


# ── LIBERO Dataset ──────────────────────────────────────────────────

LIBERO_DATA_DIR = Path("/ceph_data/kana5123/libero_datasets")


class LiberoDataset:
    """Loads LIBERO HDF5 demos as flat list of (image, action, instruction)."""

    def __init__(self, suite_name: str, data_dir: Path = LIBERO_DATA_DIR,
                 image_size: int = 224, max_demos_per_task: int = 50):
        self.image_size = image_size
        self.samples = []  # list of (hdf5_path, demo_key, step_idx, instruction)

        suite_dir = data_dir / suite_name
        hdf5_files = sorted(suite_dir.glob("*.hdf5"))
        print(f"  Found {len(hdf5_files)} task files in {suite_dir}")

        for hdf5_path in hdf5_files:
            # Task name from filename (remove _demo.hdf5)
            task_name = hdf5_path.stem.replace("_demo", "")
            instruction = task_name.replace("_", " ")

            with h5py.File(hdf5_path, "r") as f:
                demo_keys = sorted(
                    [k for k in f["data"].keys() if k.startswith("demo_")],
                    key=lambda x: int(x.split("_")[1])
                )[:max_demos_per_task]

                for dk in demo_keys:
                    n_steps = f[f"data/{dk}/actions"].shape[0]
                    for si in range(n_steps):
                        self.samples.append((str(hdf5_path), dk, si, instruction))

        print(f"  Total samples: {len(self.samples)} "
              f"({len(hdf5_files)} tasks × demos × steps)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hdf5_path, dk, si, instruction = self.samples[idx]
        with h5py.File(hdf5_path, "r") as f:
            img_arr = f[f"data/{dk}/obs/agentview_rgb"][si]  # (128, 128, 3) uint8
            action = f[f"data/{dk}/actions"][si]  # (7,) float64, [-1, 1]

        # Resize to model input size
        img = Image.fromarray(img_arr).resize(
            (self.image_size, self.image_size), Image.LANCZOS
        )
        return {
            "image": img,
            "action": np.array(action, dtype=np.float64),
            "instruction": instruction,
        }


# ── LIBERO Action Tokenizer ─────────────────────────────────────────

class LiberoActionTokenizer:
    """LIBERO-specific tokenizer: actions already [-1,1], no normalization needed."""

    def __init__(self, model):
        self.n_bins = 256
        cfg = model.config
        pad = getattr(cfg, "pad_to_multiple_of", 0)
        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "vocab_size"):
            self.vocab_size = cfg.text_config.vocab_size - pad
        elif hasattr(cfg, "vocab_size"):
            self.vocab_size = cfg.vocab_size - pad
        else:
            raise RuntimeError("Cannot determine vocab_size")

        edges = np.linspace(-1, 1, self.n_bins + 1)
        self.bin_centers = (edges[:-1] + edges[1:]) / 2.0

    def action_to_token_ids(self, action_7d):
        """[-1,1] action → 7 token IDs (vocab_size - 1 - bin_index)."""
        action = np.clip(np.asarray(action_7d, dtype=np.float64), -1.0, 1.0)
        bin_indices = np.digitize(action, self.bin_centers) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        return (self.vocab_size - 1 - bin_indices).tolist()

    def token_ids_to_action(self, token_ids):
        """7 token IDs → [-1,1] action (inverse)."""
        token_ids = np.array(token_ids)
        bin_indices = self.vocab_size - 1 - token_ids
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        return self.bin_centers[bin_indices]


# ── Training Step ────────────────────────────────────────────────────

def train_step(model, model_cfg, processor, sample, device, bounds,
               tokenizer: LiberoActionTokenizer):
    """Teacher-forced CE loss on 7 discrete action tokens.

    Reuses pattern from train_entropy_reg.py forward_with_entropy_reg().
    """
    # Tokenize GT action
    gt_token_ids = tokenizer.action_to_token_ids(sample["action"])

    # Process image + prompt
    prompt = model_cfg.prompt_template.format(instruction=sample["instruction"])
    inputs = call_processor(
        processor, prompt, sample["image"], model_cfg, return_tensors="pt"
    ).to(device)
    if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

    # Teacher-forced input: [prompt+vision] + [7 GT action tokens]
    base_ids = inputs["input_ids"]  # (1, seq_len)
    n_base = base_ids.shape[1]
    gt_suffix = torch.tensor([gt_token_ids], device=device, dtype=base_ids.dtype)
    tf_ids = torch.cat([base_ids, gt_suffix], dim=1)

    fwd_kwargs = {k: v for k, v in inputs.items()}
    fwd_kwargs["input_ids"] = tf_ids
    if "attention_mask" in fwd_kwargs:
        ext = torch.ones(1, 7, device=device, dtype=fwd_kwargs["attention_mask"].dtype)
        fwd_kwargs["attention_mask"] = torch.cat(
            [fwd_kwargs["attention_mask"], ext], dim=1
        )
    fwd_kwargs["use_cache"] = False

    # Forward
    out = model(**fwd_kwargs)

    # CE loss: logits[n_base + d - 1] predicts action token d
    ce_losses = []
    for d in range(7):
        logit_pos = n_base + d - 1
        logits_d = out.logits[0, logit_pos, :]
        target_d = torch.tensor([gt_token_ids[d]], device=device, dtype=torch.long)
        ce_losses.append(F.cross_entropy(logits_d.unsqueeze(0), target_d))

    ce_loss = torch.stack(ce_losses).mean()
    return ce_loss


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LIBERO LoRA Fine-Tuning")
    parser.add_argument("--model", required=True,
                        help="Model name (openvla-7b or ecot-7b)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--suite", default="libero_spatial",
                        help="LIBERO suite name")
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=16,
                        help="Gradient accumulation steps")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Paths ──
    out_dir = Path(args.output_dir) if args.output_dir else \
        config.OUTPUT_DIR / "libero_ft" / args.model / args.suite
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  LIBERO LoRA Fine-Tuning")
    print(f"  Model:     {args.model}")
    print(f"  Suite:     {args.suite}")
    print(f"  Device:    {args.device}")
    print(f"  LoRA:      r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  LR:        {args.lr}")
    print(f"  Steps:     {args.max_steps}")
    print(f"  Grad Acc:  {args.grad_accum}")
    print(f"  Output:    {out_dir}")
    print(f"{'='*60}\n", flush=True)

    # ── Load model ──
    print("Loading model...", flush=True)
    processor, model, model_cfg = load_model_from_registry(args.model, args.device)

    # ── Apply LoRA ──
    print("Applying LoRA...", flush=True)
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── Tokenizer ──
    base_model = model.base_model.model if hasattr(model, "base_model") else model
    tokenizer = LiberoActionTokenizer(base_model)
    print(f"  Action tokenizer: vocab={tokenizer.vocab_size}, bins={tokenizer.n_bins}")

    # ── Token boundaries ──
    dummy_img = Image.new("RGB", (224, 224), (128, 128, 128))
    bounds = detect_token_boundaries(
        processor, model, dummy_img, "pick up the object",
        args.device, model_cfg,
    )
    print(f"  Vision tokens: [{bounds['vision_start']}:{bounds['vision_end']}]")

    # ── Dataset ──
    print(f"\nLoading LIBERO dataset '{args.suite}'...", flush=True)
    dataset = LiberoDataset(args.suite, image_size=224)
    n_samples = len(dataset)
    print(f"  Dataset size: {n_samples} samples")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    # ── Training loop ──
    history = {
        "model": args.model,
        "suite": args.suite,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lr": args.lr,
        "max_steps": args.max_steps,
        "grad_accum": args.grad_accum,
        "steps": [],
    }

    rng = np.random.default_rng(seed=args.seed)
    t_start = time.time()
    accum_loss = 0.0

    print(f"\n{'='*60}")
    print(f"  Starting training ({args.max_steps} steps)")
    print(f"{'='*60}\n", flush=True)

    for step in range(1, args.max_steps + 1):
        # Random sample
        si = int(rng.integers(0, n_samples))
        sample = dataset[si]

        loss = train_step(model, model_cfg, processor, sample, args.device,
                          bounds, tokenizer)

        # Gradient accumulation
        scaled_loss = loss / args.grad_accum
        scaled_loss.backward()
        accum_loss += loss.item()

        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        # Logging
        if step % args.log_every == 0 or step == 1:
            avg_loss = accum_loss / min(step, args.log_every)
            elapsed = time.time() - t_start
            steps_per_sec = step / elapsed
            eta = (args.max_steps - step) / steps_per_sec if steps_per_sec > 0 else 0

            step_info = {
                "step": step,
                "loss": round(avg_loss, 4),
                "elapsed_s": round(elapsed, 1),
            }
            history["steps"].append(step_info)
            accum_loss = 0.0

            print(f"  Step {step:5d}/{args.max_steps} | "
                  f"loss={avg_loss:.4f} | "
                  f"{steps_per_sec:.1f} it/s | "
                  f"ETA {eta/60:.0f}min", flush=True)

        # Save checkpoint
        if step % args.save_every == 0 or step == args.max_steps:
            ckpt_dir = out_dir / f"checkpoint_{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt_dir))
            print(f"  Saved checkpoint: {ckpt_dir}", flush=True)

    # ── Save final adapter ──
    final_dir = out_dir / "lora_adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))

    # Save history
    t_total = time.time() - t_start
    history["total_time_s"] = round(t_total, 1)
    history_path = out_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Time: {t_total/60:.1f}min")
    print(f"  Final adapter: {final_dir}")
    print(f"  History: {history_path}")
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
