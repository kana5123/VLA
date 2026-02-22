"""LoRA baseline training for comparison with Attention Adapter.

Fine-tunes q_proj + v_proj in the last 4 transformer layers using PEFT LoRA.
Parameter count matched to our adapter (~1.05M at rank 16).

Usage:
    # Single GPU
    python lora_train.py --model openvla-7b

    # Multi-GPU
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 lora_train.py

    # Resume from checkpoint
    python lora_train.py --resume outputs/lora_results/openvla-7b/checkpoints/step_5000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model

import config
from adapter_data import ActionTokenizer, create_dataloaders
from extract_attention import load_model_from_registry, get_layers
from model_registry import get_model, list_experiment_models


def forward_lora(
    model,
    processor,
    image,
    instruction: str,
    target_token_ids: list[int],
    device: torch.device,
    model_cfg,
) -> torch.Tensor:
    """Teacher-forced forward pass with LoRA-modified model.

    Returns:
        total_loss: scalar CE loss
    """
    prompt = model_cfg.prompt_template.format(instruction=instruction)
    inputs = processor(prompt, image, return_tensors="pt").to(device)
    if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs.get("pixel_values")

    total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
    }

    for token_idx in range(model_cfg.action_tokens):
        outputs = model(**model_inputs, use_cache=False)
        logits = outputs.logits[:, -1, :]

        target = torch.tensor(
            [target_token_ids[token_idx]], device=device, dtype=torch.long
        )
        loss_i = F.cross_entropy(logits.float(), target)
        total_loss = total_loss + loss_i

        gt_token = torch.tensor(
            [[target_token_ids[token_idx]]], device=device, dtype=torch.long
        )
        input_ids = torch.cat([input_ids, gt_token], dim=-1)
        if attention_mask is not None:
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(1, 1, device=device, dtype=attention_mask.dtype),
            ], dim=-1)
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }

    return total_loss / model_cfg.action_tokens


@torch.no_grad()
def evaluate_lora(
    model, processor, tokenizer, val_loader, device, accelerator, model_cfg,
    max_steps: int = 100,
) -> float:
    """Evaluate LoRA model on validation set."""
    model.eval()
    total_loss = 0.0
    n_steps = 0

    for batch in val_loader:
        for i in range(len(batch["images"])):
            image = batch["images"][i]
            instruction = batch["instructions"][i]
            gt_action = batch["actions"][i]
            if isinstance(gt_action, torch.Tensor):
                gt_action = gt_action.cpu().numpy()
            target_tokens = tokenizer.action_to_token_ids(gt_action)

            prompt = model_cfg.prompt_template.format(instruction=instruction)
            inputs = processor(prompt, image, return_tensors="pt").to(device)
            if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
                inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask")
            pixel_values = inputs.get("pixel_values")

            step_loss = 0.0
            for token_idx in range(model_cfg.action_tokens):
                model_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "pixel_values": pixel_values,
                }
                outputs = model(**model_inputs, use_cache=False)
                logits = outputs.logits[:, -1, :]

                target = torch.tensor(
                    [target_tokens[token_idx]], device=device, dtype=torch.long
                )
                step_loss += F.cross_entropy(logits.float(), target).item()

                next_token = logits.argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones(1, 1, device=device, dtype=attention_mask.dtype),
                    ], dim=-1)

            total_loss += step_loss / model_cfg.action_tokens
            n_steps += 1

        if n_steps >= max_steps:
            break

    loss_tensor = torch.tensor([total_loss], device=device)
    count_tensor = torch.tensor([n_steps], device=device, dtype=torch.float32)
    loss_tensor = accelerator.reduce(loss_tensor, reduction="sum")
    count_tensor = accelerator.reduce(count_tensor, reduction="sum")

    avg_loss = loss_tensor.item() / max(count_tensor.item(), 1)
    model.train()
    return avg_loss


def train():
    parser = argparse.ArgumentParser(description="Train LoRA baseline")
    parser.add_argument("--model", type=str, default="openvla-7b",
                        choices=list_experiment_models())
    parser.add_argument("--num_episodes", type=int, default=config.ADAPTER_NUM_TRAIN_EPISODES)
    parser.add_argument("--batch_size", type=int, default=config.ADAPTER_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LORA_LR)
    parser.add_argument("--lora_r", type=int, default=config.LORA_R)
    parser.add_argument("--lora_alpha", type=int, default=config.LORA_ALPHA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None, help="Path to PEFT checkpoint dir")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device
    set_seed(args.seed)

    is_main = accelerator.is_main_process
    per_gpu_bs = args.batch_size // max(accelerator.num_processes, 1)

    # Load model
    if is_main:
        print(f"Loading {args.model}...")
    processor, model, model_cfg = load_model_from_registry(args.model, device=str(device))

    if model_cfg.action_type != "discrete":
        raise NotImplementedError(f"LoRA training not implemented for {model_cfg.action_type} models")

    # Determine which layers to apply LoRA
    adapter_cfg = model_cfg.get_adapter_config()
    target_layers = adapter_cfg["target_layers"]

    # Apply LoRA to q_proj and v_proj in target layers
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=config.LORA_TARGET_MODULES,
        layers_to_transform=target_layers,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    if is_main:
        model.print_trainable_parameters()
        print(f"\n{'=' * 60}")
        print(f"LoRA Baseline Training")
        print(f"  Model: {args.model}")
        print(f"  LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
        print(f"  Target layers: {target_layers}")
        print(f"  Target modules: {config.LORA_TARGET_MODULES}")
        print(f"  Seed: {args.seed}")
        print(f"  Devices: {accelerator.num_processes} GPU(s)")
        print(f"{'=' * 60}\n")

    # Tokenizer
    tokenizer = ActionTokenizer(model)

    # Optimizer + Scheduler
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=config.ADAPTER_WEIGHT_DECAY,
    )
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=config.LORA_WARMUP_STEPS)
    cosine = CosineAnnealingLR(optimizer, T_max=config.LORA_MAX_STEPS - config.LORA_WARMUP_STEPS, eta_min=config.ADAPTER_MIN_LR)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[config.LORA_WARMUP_STEPS])

    # Data loaders
    if is_main:
        print("Creating data loaders...")
    train_loader, val_loader, _ = create_dataloaders(
        num_episodes=args.num_episodes,
        batch_size=per_gpu_bs,
        source="tfrecord",
        accelerator=accelerator,
        use_object_masks=False,
    )

    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, val_loader,
    )

    # Output dirs
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = config.LORA_RESULTS_DIR / args.model
    ckpt_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

    accelerator.wait_for_everyone()

    # Training loop
    model.train()
    global_step = 0
    best_val_loss = float("inf")
    patience_counter = 0
    log_entries = []
    t_start = time.time()

    while global_step < config.LORA_MAX_STEPS:
        for batch in train_loader:
            if global_step >= config.LORA_MAX_STEPS:
                break

            local_bs = len(batch["images"])
            batch_loss_value = 0.0

            for i in range(local_bs):
                image = batch["images"][i]
                instruction = batch["instructions"][i]
                gt_action = batch["actions"][i]
                if isinstance(gt_action, torch.Tensor):
                    gt_action = gt_action.cpu().numpy()

                target_tokens = tokenizer.action_to_token_ids(gt_action)

                loss_i = forward_lora(
                    model, processor, image, instruction,
                    target_tokens, device, model_cfg,
                )
                sample_loss = loss_i / local_bs

                sync_ctx = accelerator.no_sync(model) if i < local_bs - 1 else __import__("contextlib").nullcontext()
                with sync_ctx:
                    accelerator.backward(sample_loss)

                batch_loss_value += sample_loss.item()

            grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.ADAPTER_GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # Logging
            if is_main and global_step % 50 == 0:
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t_start
                log_entry = {
                    "step": global_step,
                    "loss": batch_loss_value,
                    "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm),
                    "lr": lr,
                    "elapsed_s": elapsed,
                }
                log_entries.append(log_entry)
                print(
                    f"Step {global_step:6d} | "
                    f"Loss {batch_loss_value:.4f} | "
                    f"GradNorm {log_entry['grad_norm']:.4f} | "
                    f"LR {lr:.2e}"
                )

            # Evaluation
            if global_step % config.ADAPTER_EVAL_EVERY == 0:
                val_loss = evaluate_lora(
                    model, processor, tokenizer, val_loader, device, accelerator, model_cfg,
                )
                if is_main:
                    print(f"  >>> Val Loss: {val_loss:.4f} (best: {best_val_loss:.4f})")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        unwrapped = accelerator.unwrap_model(model)
                        unwrapped.save_pretrained(ckpt_dir / "best")
                    else:
                        patience_counter += 1

                should_stop = torch.tensor(
                    [1 if patience_counter >= config.ADAPTER_PATIENCE else 0], device=device,
                )
                should_stop = accelerator.reduce(should_stop, reduction="max")
                if should_stop.item() >= 1:
                    if is_main:
                        print(f"Early stopping at step {global_step}")
                    break

            # Periodic save
            if global_step % config.ADAPTER_SAVE_EVERY == 0 and is_main:
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_pretrained(ckpt_dir / f"step_{global_step}")

        if global_step >= config.LORA_MAX_STEPS:
            break
        stop_flag = torch.tensor([1 if patience_counter >= config.ADAPTER_PATIENCE else 0], device=device)
        stop_flag = accelerator.reduce(stop_flag, reduction="max")
        if stop_flag.item() >= 1:
            break

    # Save final
    if is_main:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(ckpt_dir / "final")

        log_path = log_dir / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(log_entries, f, indent=2)
        print(f"\nLoRA training complete. Logs: {log_path}")


if __name__ == "__main__":
    train()
