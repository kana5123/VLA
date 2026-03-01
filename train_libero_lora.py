#!/usr/bin/env python3
"""LIBERO LoRA Fine-Tuning for Bottleneck VLA Models (OpenVLA, ECoT).

Fine-tunes VLA models on LIBERO demonstration data using LoRA.
Purpose: probe whether position-anchoring bottleneck persists after
task-specific fine-tuning (structural vs adaptive routing question).

Data: LIBERO HDF5 demos with actions already in [-1, 1] range.
Loss: Teacher-forced cross-entropy on 7 discrete action tokens.

Config matched to OpenVLA official fine-tuning:
  - LoRA: r=32, alpha=16, dropout=0.0, init="gaussian"
  - Batch size: 16 per 80GB GPU (single sample processing, grad_accum=16)
  - Steps: 50K optimizer steps (not forward passes)
  - LR: 5e-4 (constant), weight_decay=0.0
  - Image augmentation: random resize-crop

Usage:
    python train_libero_lora.py --model openvla-7b --suite libero_spatial \
        --device cuda:0 --max_steps 50000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None

sys.path.insert(0, str(Path(__file__).parent))

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
                 image_size: int = 224, max_demos_per_task: int = 50,
                 image_aug: bool = False):
        self.image_size = image_size
        self.image_aug = image_aug
        self.samples = []  # list of (hdf5_path, demo_key, step_idx, instruction)

        if image_aug:
            self.aug_transform = T.Compose([
                T.RandomResizedCrop(image_size, scale=(0.8, 1.0),
                                    interpolation=T.InterpolationMode.BICUBIC),
            ])

        suite_dir = data_dir / suite_name
        hdf5_files = sorted(suite_dir.glob("*.hdf5"))
        print(f"  Found {len(hdf5_files)} task files in {suite_dir}")

        for hdf5_path in hdf5_files:
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
              f"({len(hdf5_files)} tasks x demos x steps)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hdf5_path, dk, si, instruction = self.samples[idx]
        with h5py.File(hdf5_path, "r") as f:
            img_arr = f[f"data/{dk}/obs/agentview_rgb"][si]  # (128, 128, 3) uint8
            action = f[f"data/{dk}/actions"][si]  # (7,) float64, [-1, 1]

        img = Image.fromarray(img_arr)

        if self.image_aug:
            img = self.aug_transform(img)
        else:
            img = img.resize((self.image_size, self.image_size), Image.LANCZOS)

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
        """[-1,1] action -> 7 token IDs (vocab_size - 1 - bin_index)."""
        action = np.clip(np.asarray(action_7d, dtype=np.float64), -1.0, 1.0)
        bin_indices = np.digitize(action, self.bin_centers) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        return (self.vocab_size - 1 - bin_indices).tolist()

    def token_ids_to_action(self, token_ids):
        """7 token IDs -> [-1,1] action (inverse)."""
        token_ids = np.array(token_ids)
        bin_indices = self.vocab_size - 1 - token_ids
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        return self.bin_centers[bin_indices]


# ── Training Step (Batched) ──────────────────────────────────────────

def train_step_batch(model, model_cfg, processor, samples, device,
                     vision_offset: int, tokenizer: LiberoActionTokenizer):
    """Teacher-forced CE loss on 7 discrete action tokens, batched.

    Args:
        samples: list of sample dicts, each with 'image', 'action', 'instruction'
        vision_offset: number of vision tokens prepended by the model
                       (e.g., 256 for Prismatic).

    Returns:
        Scalar CE loss (averaged over batch and 7 dims).
    """
    B = len(samples)

    # Process each sample individually then pad + stack
    all_input_ids = []
    all_pixel_values = []
    all_attention_masks = []
    all_gt_token_ids = []
    all_n_text = []  # track per-sample text length for correct logit indexing

    for sample in samples:
        gt_tids = tokenizer.action_to_token_ids(sample["action"])
        all_gt_token_ids.append(gt_tids)

        prompt = model_cfg.prompt_template.format(instruction=sample["instruction"])
        inputs = call_processor(
            processor, prompt, sample["image"], model_cfg, return_tensors="pt"
        )

        # Append action token IDs (teacher forcing)
        base_ids = inputs["input_ids"]  # (1, n_text)
        n_text = base_ids.shape[1]
        all_n_text.append(n_text)
        gt_suffix = torch.tensor([gt_tids], dtype=base_ids.dtype)
        tf_ids = torch.cat([base_ids, gt_suffix], dim=1)  # (1, n_text + 7)
        all_input_ids.append(tf_ids)

        if "pixel_values" in inputs:
            all_pixel_values.append(inputs["pixel_values"])

        if "attention_mask" in inputs:
            am = inputs["attention_mask"]
            ext = torch.ones(1, 7, dtype=am.dtype)
            all_attention_masks.append(torch.cat([am, ext], dim=1))

    # Pad sequences to same length (right-pad with pad_token_id=0)
    max_len = max(ids.shape[1] for ids in all_input_ids)
    padded_ids = []
    padded_masks = []
    for i in range(B):
        ids = all_input_ids[i]
        pad_len = max_len - ids.shape[1]
        if pad_len > 0:
            pad = torch.zeros(1, pad_len, dtype=ids.dtype)
            padded_ids.append(torch.cat([ids, pad], dim=1))
        else:
            padded_ids.append(ids)

        if all_attention_masks:
            am = all_attention_masks[i]
            if pad_len > 0:
                pad_am = torch.zeros(1, pad_len, dtype=am.dtype)
                padded_masks.append(torch.cat([am, pad_am], dim=1))
            else:
                padded_masks.append(am)

    # Stack into batch tensors
    fwd_kwargs = {
        "input_ids": torch.cat(padded_ids, dim=0).to(device),
        "use_cache": False,
    }
    if all_pixel_values:
        pv = torch.cat(all_pixel_values, dim=0).to(device)
        if pv.dtype != model.dtype:
            pv = pv.to(model.dtype)
        fwd_kwargs["pixel_values"] = pv
    if padded_masks:
        fwd_kwargs["attention_mask"] = torch.cat(padded_masks, dim=0).to(device)

    # Forward pass (batched)
    out = model(**fwd_kwargs)

    # CE loss at correct positions in expanded logits:
    #   Logits layout per sample: [0..V-1] vision | [V..V+n_text-1] text | [V+n_text..] action
    #   Since padding is RIGHT-padded, each sample's action tokens are at different positions.
    #   logits[b, V + n_text_b + d - 1, :] predicts action token d for batch item b
    gt_tensor = torch.tensor(all_gt_token_ids, device=device, dtype=torch.long)  # (B, 7)

    ce_losses = []
    for d in range(7):
        logits_list = []
        for b in range(B):
            logit_pos = vision_offset + all_n_text[b] + d - 1
            logits_list.append(out.logits[b, logit_pos, :].unsqueeze(0))
        logits_d = torch.cat(logits_list, dim=0)  # (B, vocab)
        targets_d = gt_tensor[:, d]  # (B,)
        ce_losses.append(F.cross_entropy(logits_d, targets_d))

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
    parser.add_argument("--max_steps", type=int, default=50000,
                        help="Max OPTIMIZER steps (not forward passes)")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size per forward pass (80GB A100 can do 16-24)")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--save_every", type=int, default=5000,
                        help="Save checkpoint every N optimizer steps")
    parser.add_argument("--log_every", type=int, default=100,
                        help="Log every N optimizer steps")
    parser.add_argument("--output_dir", type=str,
                        default="/ceph_data/kana5123/libero_ft_50k")
    parser.add_argument("--image_aug", action="store_true", default=True,
                        help="Enable image augmentation (random resize-crop)")
    parser.add_argument("--no_image_aug", dest="image_aug", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from checkpoint dir (optimizer step)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Paths ──
    out_dir = Path(args.output_dir) / args.model / args.suite
    out_dir.mkdir(parents=True, exist_ok=True)

    effective_batch = args.batch_size * args.grad_accum

    print(f"\n{'='*60}")
    print(f"  LIBERO LoRA Fine-Tuning")
    print(f"  Model:     {args.model}")
    print(f"  Suite:     {args.suite}")
    print(f"  Device:    {args.device}")
    print(f"  LoRA:      r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  LR:        {args.lr}")
    print(f"  Steps:     {args.max_steps} (optimizer steps)")
    print(f"  Batch:     {args.batch_size} x {args.grad_accum} grad_accum = {effective_batch} effective")
    print(f"  Image Aug: {args.image_aug}")
    print(f"  Output:    {out_dir}")
    print(f"{'='*60}\n", flush=True)

    # ── Load model ──
    print("Loading model...", flush=True)
    processor, model, model_cfg = load_model_from_registry(args.model, args.device)

    # ── Apply LoRA (matching OpenVLA official config) ──
    print("Applying LoRA...", flush=True)
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules="all-linear",
        lora_dropout=0.0,          # OpenVLA uses 0.0
        init_lora_weights="gaussian",  # OpenVLA uses "gaussian"
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

    # ── Detect vision offset (critical for correct logit indexing) ──
    dummy_img = Image.new("RGB", (224, 224), (128, 128, 128))
    bounds = detect_token_boundaries(
        processor, model, dummy_img, "pick up the object",
        args.device, model_cfg,
    )
    vision_offset = bounds["num_vision_tokens"]
    print(f"  Text input_ids length: {bounds['num_text_tokens']}")
    print(f"  Full sequence length (after vision expansion): {bounds['total_seq_len']}")
    print(f"  Vision tokens (from seq diff): {vision_offset}, "
          f"Text tokens: {bounds['num_text_tokens']}")
    print(f"  Token boundaries: {bounds}")
    print(f"  Vision tokens: [{bounds['vision_start']}:{bounds['vision_end']}]")

    # ── Dataset ──
    print(f"\nLoading LIBERO dataset '{args.suite}'...", flush=True)
    dataset = LiberoDataset(args.suite, image_size=224, image_aug=args.image_aug)
    n_samples = len(dataset)
    print(f"  Dataset size: {n_samples} samples")

    # ── Optimizer (matching OpenVLA: AdamW, no weight_decay) ──
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.0,  # OpenVLA uses no explicit weight_decay
    )

    # ── Training loop ──
    history = {
        "model": args.model,
        "suite": args.suite,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lr": args.lr,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": effective_batch,
        "image_aug": args.image_aug,
        "vision_offset": vision_offset,
        "lora_dropout": 0.0,
        "init_lora_weights": "gaussian",
        "step_definition": "optimizer_steps",
        "steps": [],
    }

    rng = np.random.default_rng(seed=args.seed)
    t_start = time.time()
    running_loss = 0.0
    n_loss_samples = 0
    opt_step = 0   # optimizer steps (= the real "step" count)
    samples_seen = 0
    start_opt_step = 0

    # Resume from checkpoint
    if args.resume_from:
        resume_dir = Path(args.resume_from)
        if (resume_dir / "optimizer.pt").exists():
            opt_state = torch.load(resume_dir / "optimizer.pt", map_location=args.device)
            optimizer.load_state_dict(opt_state)
            print(f"  Restored optimizer from {resume_dir}")
        if (resume_dir / "training_state.json").exists():
            with open(resume_dir / "training_state.json") as f:
                state = json.load(f)
            start_opt_step = state["opt_step"]
            opt_step = start_opt_step
            samples_seen = state.get("samples_seen", opt_step * effective_batch)
            rng = np.random.default_rng(seed=args.seed + opt_step)
            print(f"  Resuming from optimizer step {start_opt_step}")
        # Load LoRA weights
        from peft import PeftModel
        model.load_state_dict(
            torch.load(resume_dir / "adapter_model.bin", map_location=args.device),
            strict=False
        )
        print(f"  Restored LoRA weights from {resume_dir}")

    total_samples = args.max_steps * effective_batch
    remaining_steps = args.max_steps - start_opt_step

    print(f"\n{'='*60}")
    print(f"  Starting training ({args.max_steps} optimizer steps)")
    print(f"  Samples per opt step: {effective_batch} "
          f"({args.batch_size} batch x {args.grad_accum} accum)")
    print(f"  Total samples: {total_samples:,}")
    if start_opt_step > 0:
        print(f"  Resuming from step {start_opt_step}, {remaining_steps} steps remaining")
    print(f"{'='*60}\n", flush=True)

    optimizer.zero_grad()

    for step_i in range(remaining_steps):
        # Each optimizer step: grad_accum forward passes, each with batch_size samples
        step_loss = 0.0
        for accum_i in range(args.grad_accum):
            # Sample a batch
            batch_indices = rng.integers(0, n_samples, size=args.batch_size)
            batch_samples = [dataset[int(idx)] for idx in batch_indices]

            loss = train_step_batch(model, model_cfg, processor, batch_samples,
                                    args.device, vision_offset, tokenizer)

            scaled_loss = loss / args.grad_accum
            scaled_loss.backward()
            step_loss += loss.item()
            samples_seen += args.batch_size

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        opt_step += 1

        running_loss += step_loss / args.grad_accum
        n_loss_samples += 1

        # Logging (every log_every optimizer steps)
        if opt_step % args.log_every == 0 or opt_step == start_opt_step + 1:
            avg_loss = running_loss / n_loss_samples
            elapsed = time.time() - t_start
            steps_done = opt_step - start_opt_step
            steps_per_sec = steps_done / elapsed if elapsed > 0 else 0
            remaining = args.max_steps - opt_step
            eta = remaining / steps_per_sec if steps_per_sec > 0 else 0

            step_info = {
                "opt_step": opt_step,
                "loss": round(avg_loss, 4),
                "elapsed_s": round(elapsed, 1),
                "steps_per_sec": round(steps_per_sec, 3),
                "samples_seen": samples_seen,
            }
            history["steps"].append(step_info)
            running_loss = 0.0
            n_loss_samples = 0

            print(f"  Step {opt_step:5d}/{args.max_steps} | "
                  f"loss={avg_loss:.4f} | "
                  f"{steps_per_sec:.2f} step/s | "
                  f"ETA {eta/60:.0f}min", flush=True)

        # Save checkpoint
        if opt_step % args.save_every == 0:
            ckpt_dir = out_dir / f"checkpoint_{opt_step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt_dir))
            torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
            with open(ckpt_dir / "training_state.json", "w") as f:
                json.dump({"opt_step": opt_step, "samples_seen": samples_seen}, f)
            print(f"  Saved checkpoint: {ckpt_dir}", flush=True)

    # ── Save final adapter ──
    final_dir = out_dir / "lora_adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    torch.save(optimizer.state_dict(), final_dir / "optimizer.pt")
    with open(final_dir / "training_state.json", "w") as f:
        json.dump({"opt_step": opt_step, "samples_seen": samples_seen}, f)

    # Save history
    t_total = time.time() - t_start
    history["total_time_s"] = round(t_total, 1)
    history["total_samples_seen"] = samples_seen
    history["total_opt_steps"] = opt_step
    history_path = out_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Time: {t_total/60:.1f}min ({t_total/3600:.1f}h)")
    print(f"  Optimizer steps: {opt_step}")
    print(f"  Total samples seen: {samples_seen:,}")
    print(f"  Final adapter: {final_dir}")
    print(f"  History: {history_path}")
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
