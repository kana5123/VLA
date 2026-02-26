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
from extract_attention import load_model_from_registry, get_layers
from model_registry import get_model, list_experiment_models


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
    target_token_ids: list[int] | None,
    device: torch.device,
    model_cfg,
    adapter_cfg: dict,
    object_mask: torch.Tensor | None = None,
    gt_action: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a single sample through the model with adapter-controlled VAR.

    Uses teacher forcing: feeds ground-truth tokens instead of predictions.
    Supports both discrete (CE loss) and continuous (MSE loss) action models.

    Returns:
        total_loss: scalar loss (requires_grad via adapter)
        p_matrix: (1, num_target_layers, num_heads) adapter output
    """
    prompt = model_cfg.prompt_template.format(instruction=instruction)
    inputs = processor(prompt, image, return_tensors="pt").to(device)
    if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

    # SpatialVLA requires camera intrinsic matrix for backproject_patch
    if "intrinsic" not in inputs and model_cfg.architecture == "gemma2":
        inputs["intrinsic"] = torch.tensor(
            [[[224.0, 0.0, 112.0],
              [0.0, 224.0, 112.0],
              [0.0, 0.0, 1.0]]], device=device, dtype=torch.float32,
        )

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs.get("pixel_values")

    # ── Step 1: Forward through model to capture source layer hidden (no grad) ──
    captured_hidden = {}

    def capture_hook(module, args, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        captured_hidden["h_last"] = h[:, -1, :].detach()           # (1, hidden_dim)
        captured_hidden["h_vision"] = h[:, :ctx.vision_end, :].detach()  # (1, V, hidden_dim)

    layers = get_layers(model, model_cfg)

    hook = layers[adapter_cfg["source_layer"]].register_forward_hook(capture_hook)

    capture_inputs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        use_cache=False,
    )
    # Pass architecture-specific extra inputs for capture forward
    for extra_key in ("intrinsic", "image_sizes"):
        if extra_key in inputs:
            capture_inputs[extra_key] = inputs[extra_key]

    # Deactivate V3 during capture (no attention modification needed)
    _was_active = ctx.active
    ctx.active = False
    with torch.no_grad():
        _ = model(**capture_inputs)
    ctx.active = _was_active
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
            # Resize mask to match model's vision token count
            V_model = ctx.vision_end
            if mask_tensor.shape[-1] != V_model:
                mask_tensor = F.interpolate(
                    mask_tensor.unsqueeze(1).float(),
                    size=V_model, mode="nearest",
                ).squeeze(1)

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

    # Map (num_target_layers, num_heads) → full (num_layers, num_heads)
    full_p = torch.zeros(
        model_cfg.num_layers, model_cfg.num_heads, device=device, dtype=p_matrix.dtype,
    )
    _target_idx = torch.tensor(
        adapter_cfg["target_layers"], device=device,
    ).unsqueeze(1).expand(-1, model_cfg.num_heads)
    full_p = full_p.scatter(0, _target_idx, p_matrix[0])
    ctx.per_head_var_strength = full_p

    # ── Step 3: Teacher-forced autoregressive (grad through target layers) ──
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    is_continuous = (adapter_cfg.get("action_type", "discrete") == "continuous")

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
    }
    # Pass architecture-specific extra inputs (e.g., SpatialVLA intrinsic)
    for extra_key in ("intrinsic", "image_sizes"):
        if extra_key in inputs:
            model_inputs[extra_key] = inputs[extra_key]

    for token_idx in range(model_cfg.action_tokens):
        ctx.current_token_idx = token_idx

        outputs = model(**model_inputs, use_cache=False)
        logits = outputs.logits[:, -1, :]  # (1, vocab_size)

        if is_continuous and gt_action is not None:
            # Continuous action: MSE loss on raw output (e.g., TraceVLA)
            # The model outputs logits which are treated as continuous values
            pred_action_dim = logits[:, :model_cfg.action_tokens].float()
            gt_val = torch.tensor(
                [gt_action[token_idx]], device=device, dtype=torch.float32
            )
            loss_i = F.mse_loss(pred_action_dim[:, token_idx:token_idx+1], gt_val.unsqueeze(0))
        else:
            # Discrete action: CE loss on token prediction
            target = torch.tensor(
                [target_token_ids[token_idx]], device=device, dtype=torch.long
            )
            loss_i = F.cross_entropy(logits.float(), target)

        total_loss = total_loss + loss_i

        # Teacher forcing: grow sequence for discrete, keep fixed for continuous
        if not is_continuous:
            if target_token_ids is not None:
                gt_token = torch.tensor(
                    [[target_token_ids[token_idx]]], device=device, dtype=torch.long
                )
            else:
                gt_token = logits.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, gt_token], dim=-1)
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(1, 1, device=device, dtype=attention_mask.dtype),
                ], dim=-1)
        # Continuous models: input_ids stays fixed (no token appending)

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }
        for extra_key in ("intrinsic", "image_sizes"):
            if extra_key in inputs:
                model_inputs[extra_key] = inputs[extra_key]

    total_loss = total_loss / model_cfg.action_tokens
    return total_loss, p_matrix


# ═══════════════════════════════════════════════════════════════════════════
# Micro-batch forward: process M samples simultaneously
# ═══════════════════════════════════════════════════════════════════════════

def forward_micro_batch(
    model,
    adapter,
    processor,
    ctx: V3Context,
    images: list,
    instructions: list[str],
    target_token_ids_list: list[list[int] | None],
    device: torch.device,
    model_cfg,
    adapter_cfg: dict,
    object_masks: list[torch.Tensor | None] | None = None,
    gt_actions: list[np.ndarray | None] | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Process M samples together through frozen model + adapter.

    Steps:
      1. Tokenize each sample, left-pad to max length, stack
      2. Batched no_grad forward → capture M hidden states
      3. M adapter forwards → M p_matrices, M redistribution_weights
      4. Set batched tensors in ctx (B, L, H) and (B, V)
      5. 7 batched teacher-forced forward passes
      6. Per-sample loss from batched logits

    Returns:
        total_loss: scalar (sum of per-sample losses / M / action_tokens)
        p_matrices: list of (1, num_target_layers, num_heads) tensors
    """
    M = len(images)
    is_continuous = (adapter_cfg.get("action_type", "discrete") == "continuous")

    # ── Step 1: Tokenize and left-pad ──
    all_inputs = []
    for img, instr in zip(images, instructions):
        prompt = model_cfg.prompt_template.format(instruction=instr)
        inp = processor(prompt, img, return_tensors="pt")
        all_inputs.append(inp)

    # Find max sequence length for left-padding
    lengths = [inp["input_ids"].shape[1] for inp in all_inputs]
    max_len = max(lengths)

    # Get pad token id (use 0 as fallback)
    pad_id = getattr(processor, "pad_token_id", None)
    if pad_id is None and hasattr(processor, "tokenizer"):
        pad_id = getattr(processor.tokenizer, "pad_token_id", 0)
    if pad_id is None:
        pad_id = 0

    # Left-pad input_ids and attention_mask
    padded_ids = []
    padded_masks = []
    for inp, L in zip(all_inputs, lengths):
        pad_len = max_len - L
        ids = inp["input_ids"][0]  # (L,)
        mask = inp.get("attention_mask")
        if mask is not None:
            mask = mask[0]  # (L,)
        else:
            mask = torch.ones(L, dtype=torch.long)

        if pad_len > 0:
            ids = torch.cat([torch.full((pad_len,), pad_id, dtype=ids.dtype), ids])
            mask = torch.cat([torch.zeros(pad_len, dtype=mask.dtype), mask])
        padded_ids.append(ids)
        padded_masks.append(mask)

    input_ids = torch.stack(padded_ids).to(device)         # (M, max_len)
    attention_mask = torch.stack(padded_masks).to(device)   # (M, max_len)

    # Per-sample padding offsets for correct vision token extraction
    pad_offsets = [max_len - L for L in lengths]  # pad_len per sample

    # Stack pixel_values — all images same size
    pixel_values = torch.cat([inp["pixel_values"] for inp in all_inputs], dim=0).to(device)
    if pixel_values.dtype != model.dtype:
        pixel_values = pixel_values.to(model.dtype)

    # Architecture-specific extras
    extra_kv = {}
    if model_cfg.architecture == "gemma2":
        extra_kv["intrinsic"] = torch.tensor(
            [[[224.0, 0.0, 112.0],
              [0.0, 224.0, 112.0],
              [0.0, 0.0, 1.0]]], device=device, dtype=torch.float32,
        ).expand(M, -1, -1)
    if "image_sizes" in all_inputs[0]:
        extra_kv["image_sizes"] = torch.cat(
            [inp["image_sizes"] for inp in all_inputs], dim=0
        ).to(device)

    # ── Step 2: Batched capture forward (no_grad) ──
    captured_hidden = {}
    layers = get_layers(model, model_cfg)

    def capture_hook(module, args, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        # h_last: last token is always correctly positioned (right-aligned content)
        captured_hidden["h_last"] = h[:, -1, :].detach()            # (M, hidden_dim)
        # h_vision: account for left-padding offset per sample
        V = ctx.vision_end
        h_vis_list = []
        for b_idx in range(h.size(0)):
            off = pad_offsets[b_idx]
            h_vis_list.append(h[b_idx, off:off + V, :])
        captured_hidden["h_vision"] = torch.stack(h_vis_list).detach()  # (M, V, hidden_dim)

    hook = layers[adapter_cfg["source_layer"]].register_forward_hook(capture_hook)
    capture_inputs = dict(
        input_ids=input_ids, attention_mask=attention_mask,
        pixel_values=pixel_values, use_cache=False, **extra_kv,
    )
    # Deactivate V3 during capture (no attention modification needed)
    _was_active = ctx.active
    ctx.active = False
    with torch.no_grad():
        _ = model(**capture_inputs)
    ctx.active = _was_active
    hook.remove()

    # ── Step 3: Per-sample adapter forward → batched p_matrix ──
    raw_adapter = adapter.module if hasattr(adapter, 'module') else adapter
    is_v2 = isinstance(raw_adapter, AttentionAdapterV2)

    all_p = []
    all_redist = []
    h_last_all = captured_hidden["h_last"].float()    # (M, hidden_dim)
    h_vision_all = captured_hidden["h_vision"].float()  # (M, V, hidden_dim)

    for b in range(M):
        h_last_b = h_last_all[b:b+1]     # (1, hidden_dim)
        h_vision_b = h_vision_all[b:b+1]  # (1, V, hidden_dim)

        if is_v2:
            mask_tensor = None
            if object_masks is not None and object_masks[b] is not None:
                mask_tensor = object_masks[b].float().to(device)
                if mask_tensor.dim() == 1:
                    mask_tensor = mask_tensor.unsqueeze(0)
                V_model = ctx.vision_end
                if mask_tensor.shape[-1] != V_model:
                    mask_tensor = F.interpolate(
                        mask_tensor.unsqueeze(1).float(),
                        size=V_model, mode="nearest",
                    ).squeeze(1)
            p_mat, redist_raw = adapter(h_last_b, h_vision_b, mask_tensor)
        else:
            p_mat = adapter(h_last_b)
            redist_raw = None

        all_p.append(p_mat)  # (1, num_target_layers, num_heads)

        if is_v2 and redist_raw is not None:
            blend = raw_adapter.blend_alpha
            V = ctx.vision_end
            prop_weights = torch.ones(1, V, device=device, dtype=torch.float32)
            sink_set = set(ctx.var_sink_indices)
            for si in sink_set:
                if si < V:
                    prop_weights[0, si] = 0.0
            prop_weights = prop_weights / prop_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            final_redist = blend * redist_raw + (1 - blend) * prop_weights
            all_redist.append(final_redist.squeeze(0))  # (V,)
        else:
            all_redist.append(None)

    # Stack p_matrices → (M, num_target_layers, num_heads)
    p_batch = torch.cat(all_p, dim=0)  # (M, TL, H)

    # Build full (M, num_layers, num_heads) and set on ctx
    # NOTE: Use advanced indexing instead of in-place scatter_ to preserve
    # gradient flow from p_batch (adapter output) through to ctx → apply_var → loss.
    # In-place scatter_ on a leaf tensor silently drops gradients.
    target_idx = torch.tensor(adapter_cfg["target_layers"], device=device)
    full_p = torch.zeros(M, model_cfg.num_layers, model_cfg.num_heads,
                         device=device, dtype=p_batch.dtype)
    full_p = full_p.clone()  # clone makes non-leaf (CloneBackward); in-place write on non-leaf is autograd-safe
    full_p[:, target_idx, :] = p_batch  # CopySlices backward routes grad to p_batch
    ctx.per_head_var_strength = full_p  # (M, L, H) — batched, grad-connected

    # Set batched redistribution_weights
    if all_redist[0] is not None:
        ctx.redistribution_weights = torch.stack(all_redist)  # (M, V)
    else:
        ctx.redistribution_weights = None

    # ── Step 4: Batched teacher-forced autoregressive ──
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    model_inputs = dict(
        input_ids=input_ids, attention_mask=attention_mask,
        pixel_values=pixel_values, **extra_kv,
    )

    for token_idx in range(model_cfg.action_tokens):
        ctx.current_token_idx = token_idx

        outputs = model(**model_inputs, use_cache=False)
        logits = outputs.logits[:, -1, :]  # (M, vocab_size)

        if is_continuous and gt_actions is not None:
            pred_dims = logits[:, :model_cfg.action_tokens].float()
            gt_vals = torch.tensor(
                [gt_actions[b][token_idx] for b in range(M)],
                device=device, dtype=torch.float32,
            )
            loss_i = F.mse_loss(pred_dims[:, token_idx], gt_vals)
        else:
            targets = torch.tensor(
                [target_token_ids_list[b][token_idx] for b in range(M)],
                device=device, dtype=torch.long,
            )
            loss_i = F.cross_entropy(logits.float(), targets)

        total_loss = total_loss + loss_i

        # Teacher forcing: grow sequence for discrete models
        if not is_continuous:
            if target_token_ids_list[0] is not None:
                gt_tokens = torch.tensor(
                    [[target_token_ids_list[b][token_idx]] for b in range(M)],
                    device=device, dtype=torch.long,
                )
            else:
                gt_tokens = logits.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, gt_tokens], dim=-1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(M, 1, device=device, dtype=attention_mask.dtype),
            ], dim=-1)

        model_inputs = dict(
            input_ids=input_ids, attention_mask=attention_mask,
            pixel_values=pixel_values, **extra_kv,
        )

    total_loss = total_loss / model_cfg.action_tokens
    return total_loss, all_p


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
    model_cfg,
    adapter_cfg: dict,
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

            is_continuous = (adapter_cfg.get("action_type", "discrete") == "continuous")
            if not is_continuous:
                target_tokens = tokenizer.action_to_token_ids(gt_action)
            else:
                target_tokens = None

            # Object mask (v2)
            obj_mask_np = None
            if "object_masks" in batch:
                obj_mask_np = batch["object_masks"][i]

            # Get hidden state from source layer
            prompt = model_cfg.prompt_template.format(instruction=instruction)
            inputs = processor(prompt, image, return_tensors="pt").to(device)
            if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
                inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

            # SpatialVLA requires camera intrinsic matrix
            if "intrinsic" not in inputs and model_cfg.architecture == "gemma2":
                inputs["intrinsic"] = torch.tensor(
                    [[[224.0, 0.0, 112.0],
                      [0.0, 224.0, 112.0],
                      [0.0, 0.0, 1.0]]], device=device, dtype=torch.float32,
                )

            captured = {}

            def hook_fn(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                captured["h_last"] = h[:, -1, :]
                captured["h_vision"] = h[:, :ctx.vision_end, :]

            layers = get_layers(model, model_cfg)
            hook_layer = layers[adapter_cfg["source_layer"]]

            hook = hook_layer.register_forward_hook(hook_fn)
            # Deactivate V3 during capture
            _was_active = ctx.active
            ctx.active = False
            model(**{k: v for k, v in inputs.items()}, use_cache=False)
            ctx.active = _was_active
            hook.remove()

            # Adapter prediction
            raw_adapter = accelerator.unwrap_model(adapter)
            is_v2 = isinstance(raw_adapter, AttentionAdapterV2)

            if is_v2:
                mask_tensor = None
                if obj_mask_np is not None:
                    mask_tensor = torch.tensor(obj_mask_np, dtype=torch.float32).unsqueeze(0).to(device)
                    V_model = ctx.vision_end
                    if mask_tensor.shape[-1] != V_model:
                        mask_tensor = F.interpolate(
                            mask_tensor.unsqueeze(1).float(),
                            size=V_model, mode="nearest",
                        ).squeeze(1)
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
                model_cfg.num_layers, model_cfg.num_heads, device=device, dtype=p_matrix.dtype,
            )
            _target_idx = torch.tensor(
                adapter_cfg["target_layers"], device=device,
            ).unsqueeze(1).expand(-1, model_cfg.num_heads)
            full_p = full_p.scatter(0, _target_idx, p_matrix[0])
            ctx.per_head_var_strength = full_p

            # Autoregressive eval (predicted tokens, not teacher forcing)
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask")
            pixel_values = inputs.get("pixel_values")

            step_loss = 0.0
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
            }
            for extra_key in ("intrinsic", "image_sizes"):
                if extra_key in inputs:
                    model_inputs[extra_key] = inputs[extra_key]

            for token_idx in range(model_cfg.action_tokens):
                ctx.current_token_idx = token_idx
                outputs = model(**model_inputs, use_cache=False)
                logits = outputs.logits[:, -1, :]

                if is_continuous:
                    pred = logits[:, :model_cfg.action_tokens].float()
                    gt_val = torch.tensor(
                        [gt_action[token_idx]], device=device, dtype=torch.float32
                    )
                    step_loss += F.mse_loss(pred[:, token_idx:token_idx+1], gt_val.unsqueeze(0)).item()
                else:
                    target = torch.tensor(
                        [target_tokens[token_idx]], device=device, dtype=torch.long
                    )
                    step_loss += F.cross_entropy(logits.float(), target).item()

                # Grow sequence for discrete models only
                if not is_continuous:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                    input_ids = torch.cat([input_ids, next_token], dim=-1)
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
                    for extra_key in ("intrinsic", "image_sizes"):
                        if extra_key in inputs:
                            model_inputs[extra_key] = inputs[extra_key]

            total_loss += step_loss / model_cfg.action_tokens
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
    model_name: str = "openvla-7b",
    adapter_cfg: dict = None,
    model=None,
    use_lora: bool = False,
):
    """Save adapter checkpoint (main process only).

    adapter_cfg must be provided — it contains model-specific parameters.
    If use_lora=True and model is provided, also saves LoRA weights.
    """
    if not accelerator.is_main_process:
        return
    if adapter_cfg is None:
        raise ValueError("adapter_cfg is required for save_checkpoint")

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
            "model_name": model_name,
            "architecture": adapter_cfg["architecture"],
            "hidden_dim": adapter_cfg["hidden_dim"],
            "num_heads": adapter_cfg["num_heads"],
            "num_target_layers": adapter_cfg["num_target_layers"],
            "target_layers": adapter_cfg["target_layers"],
            "source_layer": adapter_cfg["source_layer"],
            "vision_tokens": adapter_cfg["vision_tokens"],
            "action_type": adapter_cfg["action_type"],
            "seed": adapter_cfg.get("seed", 42),
            "lr": config.ADAPTER_LR,
            "l1_lambda": config.ADAPTER_L1_LAMBDA,
            "adapter_version": 2 if is_v2 else 1,
            "use_lora": use_lora,
            **({"query_dim": config.ADAPTER_V2_QUERY_DIM,
                "temperature": config.ADAPTER_V2_TEMPERATURE,
                "blend_init": config.ADAPTER_V2_BLEND_INIT,
                "mask_dim": config.ADAPTER_V2_MASK_DIM,
               } if is_v2 else {}),
        },
    }, path)
    print(f"  Checkpoint saved: {path}")

    # Save LoRA weights alongside adapter checkpoint
    if use_lora and model is not None:
        lora_suffix = filename.replace(".pt", "").replace(".", "_")
        lora_dir = save_dir / f"lora_{lora_suffix}"
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(lora_dir)
        print(f"  LoRA checkpoint saved: {lora_dir}")


def load_checkpoint(
    path: str,
    adapter,
    optimizer,
    scheduler,
    accelerator: Accelerator,
    device: torch.device,
    model=None,
    use_lora: bool = False,
) -> tuple[int, float, int]:
    """Load checkpoint and return (global_step, best_val_loss, patience_counter)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw_adapter = accelerator.unwrap_model(adapter)
    raw_adapter.load_state_dict(ckpt["adapter_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    step = ckpt["global_step"]
    best = ckpt["best_val_loss"]
    patience = ckpt["patience_counter"]

    # Reload LoRA weights if joint training
    if use_lora and model is not None:
        from pathlib import Path
        lora_suffix = Path(path).stem.replace(".", "_")
        lora_dir = Path(path).parent / f"lora_{lora_suffix}"
        if lora_dir.exists():
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.load_adapter(str(lora_dir), adapter_name="default")
            if accelerator.is_main_process:
                print(f"  LoRA weights reloaded from {lora_dir}")
        elif accelerator.is_main_process:
            print(f"  WARNING: LoRA dir not found at {lora_dir}, starting LoRA from scratch")

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
    parser.add_argument("--model", type=str, default="openvla-7b",
                        choices=list_experiment_models(),
                        help="VLA model from model_registry (experiment-ready only)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--use_lora", action="store_true",
                        help="Joint training: apply LoRA to model weights alongside adapter")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override config.ADAPTER_MAX_STEPS (for smoke tests)")
    parser.add_argument("--eval_every", type=int, default=None,
                        help="Override config.ADAPTER_EVAL_EVERY (for smoke tests)")
    parser.add_argument("--micro_batch_size", type=int, default=8,
                        help="Samples processed simultaneously in one forward pass (default: 8)")
    args = parser.parse_args()

    # Override config values if CLI args provided
    if args.max_steps is not None:
        config.ADAPTER_MAX_STEPS = args.max_steps
    if args.eval_every is not None:
        config.ADAPTER_EVAL_EVERY = args.eval_every

    # ── Initialize accelerator ──
    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device
    set_seed(args.seed)
    if accelerator.is_main_process:
        print(f"  Seed: {args.seed}")

    is_main = accelerator.is_main_process

    per_gpu_bs = args.batch_size // max(accelerator.num_processes, 1)

    # ── Load frozen model (each GPU gets a copy) ──
    if is_main:
        print(f"Loading frozen {args.model} model...")
    processor, model, model_cfg = load_model_from_registry(args.model, device=str(device))

    # Build adapter config from the authoritative model_cfg returned by loader
    adapter_cfg = model_cfg.get_adapter_config()
    adapter_cfg["architecture"] = model_cfg.architecture  # for checkpoint saving
    adapter_cfg["seed"] = args.seed
    assert len(adapter_cfg["target_layers"]) == adapter_cfg["num_target_layers"], (
        f"target_layers length {len(adapter_cfg['target_layers'])} "
        f"!= num_target_layers {adapter_cfg['num_target_layers']}"
    )

    if is_main:
        print(f"\n{'=' * 60}")
        print(f"Differentiable Attention Adapter Training")
        print(f"  Model: {args.model} ({model_cfg.architecture})")
        print(f"  Devices: {accelerator.num_processes} GPU(s)")
        print(f"  Episodes: {args.num_episodes}")
        print(f"  Batch size: {args.batch_size} (effective = {per_gpu_bs}/GPU × {accelerator.num_processes} GPUs)")
        print(f"  Micro-batch: {args.micro_batch_size} samples/forward")
        print(f"  Per-GPU batch: {per_gpu_bs}")
        print(f"  LR: {args.lr} → {config.ADAPTER_MIN_LR}")
        print(f"  Max steps: {config.ADAPTER_MAX_STEPS}")
        print(f"  Target layers: {adapter_cfg['target_layers']}")
        print(f"  Source layer: {adapter_cfg['source_layer']}")
        print(f"  Action type: {adapter_cfg['action_type']}")
        print(f"{'=' * 60}\n")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    # ── Vision/text boundaries from model config ──
    vision_end = model_cfg.num_vision_tokens
    # For text_end, probe with a dummy input
    from PIL import Image as PILImage
    dummy_image = PILImage.new("RGB", (256, 256), color=(128, 128, 128))
    prompt = model_cfg.prompt_template.format(instruction="pick up the object")
    dummy_inputs = processor(prompt, dummy_image, return_tensors="pt").to(device)
    num_text_tokens = dummy_inputs["input_ids"].shape[-1]
    text_end = vision_end + num_text_tokens  # absolute index in [vision | text | action] layout
    if is_main:
        print(f"  Vision tokens: 0..{vision_end - 1}")
        print(f"  Text tokens: {vision_end}..{text_end - 1} ({num_text_tokens} tokens)")
        print(f"  Text end index: {text_end}")

    # ── Action tokenizer (must be created BEFORE LoRA wrapping) ──
    if model_cfg.action_type == "discrete":
        tokenizer = ActionTokenizer(model, model_cfg=model_cfg)
    else:
        tokenizer = None  # Continuous models use MSE loss, no tokenizer needed

    # ── Optional: Apply LoRA for joint training ──
    if args.use_lora:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            target_modules=config.LORA_TARGET_MODULES,
            layers_to_transform=adapter_cfg["target_layers"],
            lora_dropout=config.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.train()  # LoRA params need train mode
        if is_main:
            model.print_trainable_parameters()
            print(f"  LoRA + Adapter joint training enabled")

    # ── Adapter model ──
    if args.adapter_version == 2:
        adapter = AttentionAdapterV2(
            hidden_dim=adapter_cfg["hidden_dim"],
            num_target_layers=adapter_cfg["num_target_layers"],
            num_heads=adapter_cfg["num_heads"],
            vision_tokens=adapter_cfg["vision_tokens"],
        )
    else:
        adapter = AttentionAdapter(
            hidden_dim=adapter_cfg["hidden_dim"],
            num_target_layers=adapter_cfg["num_target_layers"],
            num_heads=adapter_cfg["num_heads"],
        )

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
    if args.use_lora:
        # Joint training: optimize both adapter and LoRA parameters
        lora_params = [p for p in model.parameters() if p.requires_grad]
        all_params = [
            {"params": list(adapter.parameters()), "lr": args.lr},
            {"params": lora_params, "lr": config.LORA_LR},
        ]
        optimizer = AdamW(all_params, weight_decay=config.ADAPTER_WEIGHT_DECAY)
        if is_main:
            n_lora = sum(p.numel() for p in lora_params)
            print(f"  Optimizer: adapter ({adapter.param_count():,}) + LoRA ({n_lora:,}) params")
    else:
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
    if args.use_lora:
        # LoRA params live on model — must also wrap model for DDP gradient sync
        adapter, model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
            adapter, model, optimizer, scheduler, train_loader, val_loader,
        )
    else:
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
        dynamic_sink_detection=config.DYNAMIC_SINK_DETECTION,
        sink_alpha=config.SINK_ALPHA,
        vision_end=vision_end,
        enhancement_layers=set(adapter_cfg["target_layers"]),
        text_end=text_end,
        text_sink_enabled=config.VAR_TEXT_SINK_ENABLED,
        text_sink_p=config.VAR_TEXT_SINK_P,
        text_sink_threshold=config.VAR_TEXT_SINK_THRESHOLD,
    )
    set_v3_context(ctx)
    install_v3_patch(ctx, architecture=model_cfg.architecture, model=model)
    set_var_differentiable(enabled=True, temperature=10.0)
    if is_main:
        if ctx.dynamic_sink_detection:
            print(f"  Dynamic sink detection: ON (α={ctx.sink_alpha})")
        else:
            print(f"  Dynamic sink detection: OFF (hardcoded sinks={ctx.var_sink_indices})")

    # ── Training state ──
    global_step = 0
    best_val_loss = float("inf")
    patience_counter = 0

    # ── Resume ──
    if args.resume:
        global_step, best_val_loss, patience_counter = load_checkpoint(
            args.resume, adapter, optimizer, scheduler, accelerator, device,
            model=model if args.use_lora else None,
            use_lora=args.use_lora,
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

            # Micro-batch gradient accumulation: process micro_batch_size
            # samples per forward pass for better GPU utilization.
            batch_p_stats = []
            local_bs = len(batch["images"])
            batch_loss_value = 0.0  # scalar for logging (no grad)
            mbs = min(args.micro_batch_size, local_bs)
            n_micro = (local_bs + mbs - 1) // mbs  # ceil division

            for mi in range(n_micro):
                start = mi * mbs
                end = min(start + mbs, local_bs)
                micro_size = end - start

                # Collect micro-batch samples
                mb_images = batch["images"][start:end]
                mb_instructions = batch["instructions"][start:end]
                mb_actions = []
                mb_masks = []
                mb_targets = []
                for i in range(start, end):
                    gt_action = batch["actions"][i]
                    if isinstance(gt_action, torch.Tensor):
                        gt_action = gt_action.cpu().numpy()
                    mb_actions.append(gt_action)

                    obj_mask = None
                    if "object_masks" in batch:
                        obj_mask = torch.tensor(batch["object_masks"][i], dtype=torch.float32).to(device)
                    mb_masks.append(obj_mask)

                    mb_targets.append(
                        tokenizer.action_to_token_ids(gt_action) if tokenizer is not None else None
                    )

                # Forward micro-batch
                mb_loss, mb_p_list = forward_micro_batch(
                    model, adapter,
                    processor, ctx, mb_images, mb_instructions,
                    mb_targets, device,
                    model_cfg=model_cfg, adapter_cfg=adapter_cfg,
                    object_masks=mb_masks,
                    gt_actions=mb_actions,
                )

                # L1 sparsity penalty with warmup schedule
                # During LR warmup: L1=0 (let adapter freely explore p space)
                # After warmup: linearly ramp L1 over another warmup_steps
                p_cat = torch.cat(mb_p_list, dim=0)  # (micro_size, TL, H)
                l1_warmup_end = config.ADAPTER_WARMUP_STEPS * 2  # ramp finishes at 2x warmup
                if global_step < config.ADAPTER_WARMUP_STEPS:
                    l1_scale = 0.0
                elif global_step < l1_warmup_end:
                    l1_scale = (global_step - config.ADAPTER_WARMUP_STEPS) / config.ADAPTER_WARMUP_STEPS
                else:
                    l1_scale = 1.0
                l1_penalty = (config.ADAPTER_L1_LAMBDA * l1_scale) * p_cat.abs().mean()
                micro_loss = (mb_loss + l1_penalty) * micro_size / local_bs

                # DDP: skip gradient all-reduce for all but last micro-batch
                is_last_micro = (mi == n_micro - 1)
                if not is_last_micro:
                    adapter_sync = accelerator.no_sync(adapter)
                    # Also no_sync model when LoRA is active (model is DDP-wrapped)
                    model_sync = accelerator.no_sync(model) if args.use_lora else nullcontext()
                else:
                    adapter_sync = nullcontext()
                    model_sync = nullcontext()
                with adapter_sync, model_sync:
                    accelerator.backward(micro_loss)

                batch_loss_value += micro_loss.item()
                for p in mb_p_list:
                    batch_p_stats.append(
                        accelerator.unwrap_model(adapter).sparsity_stats(p)
                    )

            # Gradient clipping + optimizer step
            if args.use_lora:
                all_trainable = list(adapter.parameters()) + [
                    p for p in model.parameters() if p.requires_grad
                ]
                grad_norm = accelerator.clip_grad_norm_(all_trainable, config.ADAPTER_GRAD_CLIP)
            else:
                grad_norm = accelerator.clip_grad_norm_(adapter.parameters(), config.ADAPTER_GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            # ── Logging (main process only) ──
            if is_main and global_step % 50 == 0:
                mean_p = np.mean([s["mean_p"] for s in batch_p_stats])
                active = np.mean([s["active_ratio"] for s in batch_p_stats])
                max_p = np.max([s["max_p"] for s in batch_p_stats])
                min_p = np.min([s["min_p"] for s in batch_p_stats])
                all_means = [s["mean_p"] for s in batch_p_stats]
                std_p = float(np.std(all_means)) if len(all_means) > 1 else 0.0
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t_start

                # Compute current L1 scale for logging
                _l1_warmup_end = config.ADAPTER_WARMUP_STEPS * 2
                if global_step < config.ADAPTER_WARMUP_STEPS:
                    _l1_scale_log = 0.0
                elif global_step < _l1_warmup_end:
                    _l1_scale_log = (global_step - config.ADAPTER_WARMUP_STEPS) / config.ADAPTER_WARMUP_STEPS
                else:
                    _l1_scale_log = 1.0

                log_entry = {
                    "step": global_step,
                    "loss": batch_loss_value,
                    "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm),
                    "lr": lr,
                    "mean_p": float(mean_p),
                    "max_p": float(max_p),
                    "min_p": float(min_p),
                    "std_p": std_p,
                    "active_ratio": float(active),
                    "l1_scale": _l1_scale_log,
                    "elapsed_s": elapsed,
                }

                raw = accelerator.unwrap_model(adapter)
                if hasattr(raw, 'blend_alpha'):
                    log_entry["blend_alpha"] = raw.blend_alpha.item()
                # Redistribution entropy (V2 only)
                redist_entries = [s.get("redist_entropy") for s in batch_p_stats if s.get("redist_entropy") is not None]
                if redist_entries:
                    log_entry["redist_entropy"] = float(np.mean(redist_entries))

                log_entries.append(log_entry)

                print(
                    f"Step {global_step:6d} | "
                    f"Loss {batch_loss_value:.4f} | "
                    f"GradNorm {log_entry['grad_norm']:.4f} | "
                    f"LR {lr:.2e} | "
                    f"P [{min_p:.3f}/{mean_p:.3f}/{max_p:.3f}] | "
                    f"StdP {std_p:.4f} | "
                    f"Active {active:.2%} | "
                    f"L1s {_l1_scale_log:.2f}"
                    + (f" | Blend {log_entry['blend_alpha']:.4f}" if 'blend_alpha' in log_entry else "")
                    + (f" | RdEnt {log_entry['redist_entropy']:.2f}" if 'redist_entropy' in log_entry else "")
                )

            # ── Evaluation ──
            if global_step % config.ADAPTER_EVAL_EVERY == 0:
                val_loss = evaluate(
                    model, adapter, processor, tokenizer, ctx,
                    val_loader, device, accelerator,
                    model_cfg=model_cfg, adapter_cfg=adapter_cfg,
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
                            model_name=args.model, adapter_cfg=adapter_cfg,
                            model=model, use_lora=args.use_lora,
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
                    model_name=args.model, adapter_cfg=adapter_cfg,
                    model=model, use_lora=args.use_lora,
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
        model_name=args.model, adapter_cfg=adapter_cfg,
        model=model, use_lora=args.use_lora,
    )

    if is_main:
        log_path = log_dir / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(log_entries, f, indent=2)
        print(f"\nTraining complete. Logs: {log_path}")


if __name__ == "__main__":
    train()
