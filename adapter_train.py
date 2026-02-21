"""Training loop for the Differentiable Attention Adapter (multi-GPU).

The adapter learns per-head VAR redistribution strengths by backpropagating
through the frozen OpenVLA model's last 4 layers.

Gradient path:
    CE Loss → logits → LM head (frozen) → h_31 → modified_attn → apply_var
    → p_matrix → Adapter MLP (learnable)

Usage:
    # Multi-GPU (4× H100, GPU 0-3)
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 adapter_train.py

    # Single GPU (fallback)
    python adapter_train.py

    # Resume from checkpoint
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 adapter_train.py \\
        --resume outputs/adapter_results/checkpoints/step_5000.pt
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from accelerate import Accelerator
from accelerate.utils import set_seed

import config
from adapter_data import ActionTokenizer, create_dataloaders
from adapter_model import AttentionAdapter, AttentionAdapterV2
from attention_v3 import (
    V3Context,
    install_v3_patch,
    set_v3_context,
    set_var_differentiable,
    uninstall_v3_patch,
)
from extract_attention import detect_token_boundaries, load_model


# ═══════════════════════════════════════════════════════════════════════════
# Forward pass: frozen model + adapter
# ═══════════════════════════════════════════════════════════════════════════

def forward_with_adapter(
    model,
    adapter,
    processor,
    ctx: V3Context,
    image,
    instruction: str,
    target_token_ids: list[int],
    device: torch.device,
    object_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a single sample through the model with adapter-controlled VAR.

    Uses teacher forcing: feeds ground-truth tokens instead of predictions.

    Returns:
        total_loss: scalar CE loss (requires_grad via adapter)
        p_matrix: (1, num_target_layers, num_heads) adapter output
    """
    prompt = config.PROMPT_TEMPLATE.format(instruction=instruction)
    inputs = processor(prompt, image, return_tensors="pt").to(device)
    if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs.get("pixel_values")

    # ── Step 1: Forward through model to capture h_27 (no grad) ──
    captured_hidden = {}

    def capture_hook(module, args, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        captured_hidden["h_last"] = h[:, -1, :].detach()           # (1, hidden_dim)
        captured_hidden["h_vision"] = h[:, :ctx.vision_end, :].detach()  # (1, V, hidden_dim)

    if hasattr(model, "language_model"):
        layers = model.language_model.model.layers
    else:
        layers = model.model.layers

    hook = layers[config.ADAPTER_SOURCE_LAYER].register_forward_hook(capture_hook)

    with torch.no_grad():
        _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            use_cache=False,
        )
    hook.remove()

    # ── Step 2: Adapter produces p_matrix + redistribution_weights ──
    h_last = captured_hidden["h_last"].float()     # (1, 4096)
    h_vision = captured_hidden["h_vision"].float()  # (1, V, 4096)

    # Detect v2 adapter (handles DDP wrapping)
    raw_adapter = adapter.module if hasattr(adapter, 'module') else adapter
    is_v2 = isinstance(raw_adapter, AttentionAdapterV2)

    if is_v2:
        mask_tensor = None
        if object_mask is not None:
            mask_tensor = object_mask.float().to(device)
            if mask_tensor.dim() == 1:
                mask_tensor = mask_tensor.unsqueeze(0)

        p_matrix, redist_raw = adapter(h_last, h_vision, mask_tensor)

        # Blend learned redistribution with proportional
        if redist_raw is not None:
            blend = raw_adapter.blend_alpha
            V = ctx.vision_end
            prop_weights = torch.ones(1, V, device=device, dtype=torch.float32)
            sink_set = set(ctx.var_sink_indices)
            for si in sink_set:
                if si < V:
                    prop_weights[0, si] = 0.0
            prop_weights = prop_weights / prop_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)

            final_redist = blend * redist_raw + (1 - blend) * prop_weights
            ctx.redistribution_weights = final_redist.squeeze(0)  # (V,)
        else:
            ctx.redistribution_weights = None
    else:
        p_matrix = adapter(h_last)
        ctx.redistribution_weights = None

    # Map (4_target_layers, 32_heads) → full (32_layers, 32_heads)
    full_p = torch.zeros(
        config.NUM_LAYERS, config.NUM_HEADS, device=device, dtype=p_matrix.dtype,
    )
    _target_idx = torch.tensor(
        config.ADAPTER_TARGET_LAYERS, device=device,
    ).unsqueeze(1).expand(-1, config.NUM_HEADS)
    full_p = full_p.scatter(0, _target_idx, p_matrix[0])
    ctx.per_head_var_strength = full_p

    # ── Step 3: Teacher-forced autoregressive (grad through layers 28-31) ──
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
    }

    for token_idx in range(config.NUM_ACTION_TOKENS):
        ctx.current_token_idx = token_idx

        outputs = model(**model_inputs, use_cache=False)
        logits = outputs.logits[:, -1, :]  # (1, vocab_size)

        target = torch.tensor(
            [target_token_ids[token_idx]], device=device, dtype=torch.long
        )
        loss_i = F.cross_entropy(logits.float(), target)
        total_loss = total_loss + loss_i

        # Teacher forcing: append ground truth token
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

    total_loss = total_loss / config.NUM_ACTION_TOKENS
    return total_loss, p_matrix


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
    model,
    adapter,
    processor,
    tokenizer: ActionTokenizer,
    ctx: V3Context,
    val_loader,
    device: torch.device,
    accelerator: Accelerator,
    max_steps: int = 100,
) -> float:
    """Evaluate adapter on validation set (distributed across GPUs)."""
    adapter.eval()
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

            # Object mask (v2)
            obj_mask_np = None
            if "object_masks" in batch:
                obj_mask_np = batch["object_masks"][i]

            # Get hidden state from layer 27
            prompt = config.PROMPT_TEMPLATE.format(instruction=instruction)
            inputs = processor(prompt, image, return_tensors="pt").to(device)
            if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
                inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

            captured = {}

            def hook_fn(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                captured["h_last"] = h[:, -1, :]
                captured["h_vision"] = h[:, :ctx.vision_end, :]

            if hasattr(model, "language_model"):
                hook_layer = model.language_model.model.layers[config.ADAPTER_SOURCE_LAYER]
            else:
                hook_layer = model.model.layers[config.ADAPTER_SOURCE_LAYER]

            hook = hook_layer.register_forward_hook(hook_fn)
            model(**{k: v for k, v in inputs.items()}, use_cache=False)
            hook.remove()

            # Adapter prediction
            raw_adapter = accelerator.unwrap_model(adapter)
            is_v2 = isinstance(raw_adapter, AttentionAdapterV2)

            if is_v2:
                mask_tensor = None
                if obj_mask_np is not None:
                    mask_tensor = torch.from_numpy(obj_mask_np).float().unsqueeze(0).to(device)
                p_matrix, redist_raw = raw_adapter(captured["h_last"].float(), captured["h_vision"].float(), mask_tensor)

                if redist_raw is not None:
                    blend = raw_adapter.blend_alpha
                    V = ctx.vision_end
                    prop_weights = torch.ones(1, V, device=device, dtype=torch.float32)
                    for si in ctx.var_sink_indices:
                        if si < V:
                            prop_weights[0, si] = 0.0
                    prop_weights = prop_weights / prop_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                    final_redist = blend * redist_raw + (1 - blend) * prop_weights
                    ctx.redistribution_weights = final_redist.squeeze(0)
                else:
                    ctx.redistribution_weights = None
            else:
                p_matrix = raw_adapter(captured["h_last"].float())
                ctx.redistribution_weights = None

            full_p = torch.zeros(
                config.NUM_LAYERS, config.NUM_HEADS, device=device, dtype=p_matrix.dtype,
            )
            _target_idx = torch.tensor(
                config.ADAPTER_TARGET_LAYERS, device=device,
            ).unsqueeze(1).expand(-1, config.NUM_HEADS)
            full_p = full_p.scatter(0, _target_idx, p_matrix[0])
            ctx.per_head_var_strength = full_p

            # Autoregressive eval (predicted tokens, not teacher forcing)
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask")
            pixel_values = inputs.get("pixel_values")

            step_loss = 0.0
            for token_idx in range(config.NUM_ACTION_TOKENS):
                ctx.current_token_idx = token_idx
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

                # Use model's prediction
                next_token = logits.argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones(1, 1, device=device, dtype=attention_mask.dtype),
                    ], dim=-1)

            total_loss += step_loss / config.NUM_ACTION_TOKENS
            n_steps += 1

        if n_steps >= max_steps:
            break

    # All-reduce across GPUs
    loss_tensor = torch.tensor([total_loss], device=device)
    count_tensor = torch.tensor([n_steps], device=device, dtype=torch.float32)
    loss_tensor = accelerator.reduce(loss_tensor, reduction="sum")
    count_tensor = accelerator.reduce(count_tensor, reduction="sum")

    avg_loss = loss_tensor.item() / max(count_tensor.item(), 1)
    adapter.train()
    return avg_loss


# ═══════════════════════════════════════════════════════════════════════════
# Checkpointing
# ═══════════════════════════════════════════════════════════════════════════

def save_checkpoint(
    adapter,
    optimizer,
    scheduler,
    global_step: int,
    best_val_loss: float,
    patience_counter: int,
    accelerator: Accelerator,
    filename: str,
    checkpoint_dir: Path = None,
):
    """Save adapter checkpoint (main process only)."""
    if not accelerator.is_main_process:
        return

    save_dir = checkpoint_dir or config.ADAPTER_CHECKPOINT_DIR
    path = save_dir / filename
    raw_adapter = accelerator.unwrap_model(adapter)
    is_v2 = isinstance(raw_adapter, AttentionAdapterV2)
    torch.save({
        "adapter_state_dict": raw_adapter.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,
        "config": {
            "lr": config.ADAPTER_LR,
            "num_target_layers": config.ADAPTER_NUM_TARGET_LAYERS,
            "target_layers": config.ADAPTER_TARGET_LAYERS,
            "source_layer": config.ADAPTER_SOURCE_LAYER,
            "l1_lambda": config.ADAPTER_L1_LAMBDA,
            "adapter_version": 2 if is_v2 else 1,
            **({"query_dim": config.ADAPTER_V2_QUERY_DIM,
                "temperature": config.ADAPTER_V2_TEMPERATURE,
                "blend_init": config.ADAPTER_V2_BLEND_INIT,
                "mask_dim": config.ADAPTER_V2_MASK_DIM,
               } if is_v2 else {}),
        },
    }, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(
    path: str,
    adapter,
    optimizer,
    scheduler,
    accelerator: Accelerator,
    device: torch.device,
) -> tuple[int, float, int]:
    """Load checkpoint and return (global_step, best_val_loss, patience_counter)."""
    ckpt = torch.load(path, map_location=device)
    raw_adapter = accelerator.unwrap_model(adapter)
    raw_adapter.load_state_dict(ckpt["adapter_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    step = ckpt["global_step"]
    best = ckpt["best_val_loss"]
    patience = ckpt["patience_counter"]
    if accelerator.is_main_process:
        print(f"Resumed from step {step} (best val: {best:.4f})")
    return step, best, patience


# ═══════════════════════════════════════════════════════════════════════════
# Main training loop
# ═══════════════════════════════════════════════════════════════════════════

def train():
    parser = argparse.ArgumentParser(description="Train Differentiable Attention Adapter")
    parser.add_argument("--num_episodes", type=int, default=config.ADAPTER_NUM_TRAIN_EPISODES)
    parser.add_argument("--batch_size", type=int, default=config.ADAPTER_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.ADAPTER_LR)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--adapter_version", type=int, default=2,
                        choices=[1, 2], help="Adapter version (1=MLP only, 2=object-aware)")
    parser.add_argument("--freeze_blend", action="store_true",
                        help="Freeze blend_alpha at 0 (v2-prop: proportional redistribution only)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory (checkpoints + logs saved here)")
    args = parser.parse_args()

    # ── Initialize accelerator ──
    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device
    set_seed(42)

    is_main = accelerator.is_main_process

    per_gpu_bs = args.batch_size // max(accelerator.num_processes, 1)
    if is_main:
        print(f"\n{'=' * 60}")
        print(f"Differentiable Attention Adapter Training")
        print(f"  Devices: {accelerator.num_processes} GPU(s)")
        print(f"  Episodes: {args.num_episodes}")
        print(f"  Batch size: {args.batch_size} (effective = {per_gpu_bs}/GPU × {accelerator.num_processes} GPUs)")
        print(f"  Per-GPU batch: {per_gpu_bs}")
        print(f"  LR: {args.lr} → {config.ADAPTER_MIN_LR}")
        print(f"  Max steps: {config.ADAPTER_MAX_STEPS}")
        print(f"  Target layers: {config.ADAPTER_TARGET_LAYERS}")
        print(f"{'=' * 60}\n")

    # ── Load frozen model (each GPU gets a copy) ──
    if is_main:
        print("Loading frozen OpenVLA model...")
    processor, model = load_model(device=str(device))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    # ── Detect token boundaries ──
    from PIL import Image as PILImage
    dummy_image = PILImage.new("RGB", (256, 256), color=(128, 128, 128))
    boundaries = detect_token_boundaries(
        processor, model, dummy_image, "pick up the object", str(device)
    )
    vision_end = boundaries["vision_end"]
    if is_main:
        print(f"  Vision tokens: 0..{vision_end - 1}")

    # ── Action tokenizer ──
    tokenizer = ActionTokenizer(model)

    # ── Adapter model ──
    hidden_dim = model.config.text_config.hidden_size  # 4096
    if args.adapter_version == 2:
        adapter = AttentionAdapterV2(hidden_dim=hidden_dim)
    else:
        adapter = AttentionAdapter(hidden_dim=hidden_dim)

    # Freeze blend_alpha if requested (v2-prop config)
    if args.freeze_blend and args.adapter_version == 2:
        raw = adapter
        raw._blend_logit.requires_grad_(False)
        # Force blend_alpha to 0 by setting logit to -20 (sigmoid(-20) ~ 2e-9)
        with torch.no_grad():
            raw._blend_logit.fill_(-20.0)
        if is_main:
            print(f"  blend_alpha FROZEN at {raw.blend_alpha.item():.6f}")

    if is_main:
        print(f"Adapter v{args.adapter_version} parameters: {adapter.param_count():,}")

    # ── Optimizer + Scheduler ──
    optimizer = AdamW(
        adapter.parameters(),
        lr=args.lr,
        weight_decay=config.ADAPTER_WEIGHT_DECAY,
    )

    warmup = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=config.ADAPTER_WARMUP_STEPS,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=config.ADAPTER_MAX_STEPS - config.ADAPTER_WARMUP_STEPS,
        eta_min=config.ADAPTER_MIN_LR,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[config.ADAPTER_WARMUP_STEPS],
    )

    # ── Create data loaders ──
    if is_main:
        print("Creating data loaders...")
    train_loader, val_loader, _ = create_dataloaders(
        num_episodes=args.num_episodes,
        batch_size=per_gpu_bs,
        source="tfrecord",
        accelerator=accelerator,
        use_object_masks=(args.adapter_version == 2),
    )

    # ── Prepare with accelerate (wraps adapter in DDP, distributes data) ──
    adapter, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
        adapter, optimizer, scheduler, train_loader, val_loader,
    )

    # ── V3 Context for VAR ──
    ctx = V3Context(
        active=True,
        use_var=True,
        var_p=config.VAR_P,
        var_rho=config.VAR_RHO,
        var_sink_indices=list(config.VAR_SINK_INDICES),
        vision_end=vision_end,
        enhancement_layers=set(config.ADAPTER_TARGET_LAYERS),
    )
    set_v3_context(ctx)
    install_v3_patch(ctx)
    set_var_differentiable(enabled=True, temperature=10.0)

    # ── Training state ──
    global_step = 0
    best_val_loss = float("inf")
    patience_counter = 0

    # ── Resume ──
    if args.resume:
        global_step, best_val_loss, patience_counter = load_checkpoint(
            args.resume, adapter, optimizer, scheduler, accelerator, device,
        )

    # ── Create output dirs ──
    if args.output_dir:
        ckpt_dir = Path(args.output_dir) / "checkpoints"
        log_dir = Path(args.output_dir) / "logs"
    else:
        ckpt_dir = config.ADAPTER_CHECKPOINT_DIR
        log_dir = config.ADAPTER_LOG_DIR
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

    accelerator.wait_for_everyone()

    # ═════════════════════════════════════════════════════════════════════
    # Training loop
    # ═════════════════════════════════════════════════════════════════════
    adapter.train()
    log_entries = []
    t_start = time.time()

    while global_step < config.ADAPTER_MAX_STEPS:
        for batch in train_loader:
            if global_step >= config.ADAPTER_MAX_STEPS:
                break

            # Per-sample gradient accumulation: backward per sample to keep
            # only 1 computation graph in memory at a time (prevents OOM).
            batch_p_stats = []
            local_bs = len(batch["images"])
            batch_loss_value = 0.0  # scalar for logging (no grad)

            for i in range(local_bs):
                image = batch["images"][i]
                instruction = batch["instructions"][i]
                gt_action = batch["actions"][i]
                if isinstance(gt_action, torch.Tensor):
                    gt_action = gt_action.cpu().numpy()

                # Object mask (v2)
                obj_mask = None
                if "object_masks" in batch:
                    obj_mask = torch.from_numpy(batch["object_masks"][i]).to(device)

                target_tokens = tokenizer.action_to_token_ids(gt_action)

                loss_i, p_matrix = forward_with_adapter(
                    model, adapter,
                    processor, ctx, image, instruction, target_tokens, device,
                    object_mask=obj_mask,
                )

                # L1 sparsity penalty
                l1_penalty = config.ADAPTER_L1_LAMBDA * p_matrix.abs().mean()
                sample_loss = (loss_i + l1_penalty) / local_bs

                # DDP: skip gradient all-reduce for all but last sample
                sync_ctx = accelerator.no_sync(adapter) if i < local_bs - 1 else nullcontext()
                with sync_ctx:
                    accelerator.backward(sample_loss)

                batch_loss_value += sample_loss.item()
                batch_p_stats.append(
                    accelerator.unwrap_model(adapter).sparsity_stats(p_matrix)
                )

            # Gradient clipping + optimizer step
            grad_norm = accelerator.clip_grad_norm_(adapter.parameters(), config.ADAPTER_GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            # ── Logging (main process only) ──
            if is_main and global_step % 50 == 0:
                mean_p = np.mean([s["mean_p"] for s in batch_p_stats])
                active = np.mean([s["active_ratio"] for s in batch_p_stats])
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t_start

                log_entry = {
                    "step": global_step,
                    "loss": batch_loss_value,
                    "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm),
                    "lr": lr,
                    "mean_p": float(mean_p),
                    "active_ratio": float(active),
                    "elapsed_s": elapsed,
                }

                raw = accelerator.unwrap_model(adapter)
                if hasattr(raw, 'blend_alpha'):
                    log_entry["blend_alpha"] = raw.blend_alpha.item()

                log_entries.append(log_entry)

                print(
                    f"Step {global_step:6d} | "
                    f"Loss {batch_loss_value:.4f} | "
                    f"GradNorm {log_entry['grad_norm']:.4f} | "
                    f"LR {lr:.2e} | "
                    f"MeanP {mean_p:.4f} | "
                    f"Active {active:.2%}"
                    + (f" | Blend {log_entry['blend_alpha']:.4f}" if 'blend_alpha' in log_entry else "")
                )

            # ── Evaluation ──
            if global_step % config.ADAPTER_EVAL_EVERY == 0:
                val_loss = evaluate(
                    model, adapter, processor, tokenizer, ctx,
                    val_loader, device, accelerator,
                )

                if is_main:
                    print(f"  >>> Val Loss: {val_loss:.4f} (best: {best_val_loss:.4f})")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        save_checkpoint(
                            adapter, optimizer, scheduler, global_step,
                            best_val_loss, patience_counter, accelerator, "best.pt",
                            checkpoint_dir=ckpt_dir,
                        )
                    else:
                        patience_counter += 1

                # Broadcast early stopping decision
                should_stop = torch.tensor(
                    [1 if patience_counter >= config.ADAPTER_PATIENCE else 0],
                    device=device,
                )
                should_stop = accelerator.reduce(should_stop, reduction="max")
                if should_stop.item() >= 1:
                    if is_main:
                        print(f"Early stopping at step {global_step}")
                    break

            # ── Save checkpoint ──
            if global_step % config.ADAPTER_SAVE_EVERY == 0:
                save_checkpoint(
                    adapter, optimizer, scheduler, global_step,
                    best_val_loss, patience_counter, accelerator,
                    f"step_{global_step}.pt",
                    checkpoint_dir=ckpt_dir,
                )

        # Check outer loop termination
        if global_step >= config.ADAPTER_MAX_STEPS:
            break
        # Early stopping check
        stop_flag = torch.tensor(
            [1 if patience_counter >= config.ADAPTER_PATIENCE else 0],
            device=device,
        )
        stop_flag = accelerator.reduce(stop_flag, reduction="max")
        if stop_flag.item() >= 1:
            break

    # ═════════════════════════════════════════════════════════════════════
    # Cleanup
    # ═════════════════════════════════════════════════════════════════════
    set_var_differentiable(enabled=False)
    uninstall_v3_patch()

    save_checkpoint(
        adapter, optimizer, scheduler, global_step,
        best_val_loss, patience_counter, accelerator, "final.pt",
        checkpoint_dir=ckpt_dir,
    )

    if is_main:
        log_path = log_dir / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(log_entries, f, indent=2)
        print(f"\nTraining complete. Logs: {log_path}")


if __name__ == "__main__":
    train()
