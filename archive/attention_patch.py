"""Monkey-patch engine for attention enhancement at inference time.

The patched `eager_attention_forward` is a drop-in replacement for the
function in `transformers.models.llama.modeling_llama`.

Three enhancement methods:
  1. logit_bias:     add bias to pre-softmax logits at object patch positions
  2. weight_rescale: scale post-softmax weights on object patches, re-normalize
  3. head_steering:  amplify object-patch weights only for vision-attending heads
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from torch import nn

import config


# ======================================================================
# Enhancement context
# ======================================================================

@dataclass
class EnhancementContext:
    """Mutable context read by the patched attention function."""

    active: bool = False
    method: str = "weight_rescale"
    object_patch_indices: list[int] = field(default_factory=list)

    logit_bias_alpha: float = config.LOGIT_BIAS_ALPHA
    weight_rescale_lambda: float = config.WEIGHT_RESCALE_LAMBDA
    head_steer_top_k: int = config.HEAD_STEER_TOP_K_HEADS
    head_steer_amplify: float = config.HEAD_STEER_AMPLIFY

    # V2 additions
    bg_suppress_gamma: float = config.BG_SUPPRESS_GAMMA
    gripper_exempt: bool = False
    current_token_idx: int = 0  # tracks which action token is being generated

    enhancement_layers: Optional[set[int]] = None

    def set_object_patches(self, indices: list[int]) -> None:
        self.object_patch_indices = indices

    def is_layer_active(self, layer_idx: int) -> bool:
        if not self.active or not self.object_patch_indices:
            return False
        if self.enhancement_layers is not None and layer_idx not in self.enhancement_layers:
            return False
        # Gripper-exempt: disable enhancement for the 7th action token (gripper)
        if self.gripper_exempt and self.current_token_idx >= 6:
            return False
        return True


_ctx: EnhancementContext = EnhancementContext()


def get_context() -> EnhancementContext:
    return _ctx


def set_context(ctx: EnhancementContext) -> None:
    global _ctx
    _ctx = ctx


# ======================================================================
# Enhancement method implementations
# ======================================================================

def apply_logit_bias(
    attn_logits: torch.Tensor,
    patch_indices: list[int],
    alpha: float,
) -> torch.Tensor:
    """Method 1: Pre-softmax additive bias on object patch key positions.

    Only modifies the last query row (the token being generated).
    """
    valid = [i for i in patch_indices if i < attn_logits.shape[-1]]
    if not valid:
        return attn_logits
    idx = torch.tensor(valid, dtype=torch.long, device=attn_logits.device)
    attn_logits[:, :, -1, idx] = attn_logits[:, :, -1, idx] + alpha
    return attn_logits


def apply_weight_rescale(
    attn_weights: torch.Tensor,
    patch_indices: list[int],
    lambda_scale: float,
) -> torch.Tensor:
    """Method 2: Post-softmax scale on object patches then re-normalize.

    Inspired by Atlas (ACL 2025).
    """
    valid = [i for i in patch_indices if i < attn_weights.shape[-1]]
    if not valid:
        return attn_weights
    idx = torch.tensor(valid, dtype=torch.long, device=attn_weights.device)
    last = attn_weights[:, :, -1, :].clone()
    last[:, :, idx] = last[:, :, idx] * lambda_scale
    last = last / last.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    attn_weights = attn_weights.clone()
    attn_weights[:, :, -1, :] = last
    return attn_weights


def apply_background_suppression(
    attn_weights: torch.Tensor,
    object_patch_indices: list[int],
    vision_end: int,
    gamma: float,
) -> torch.Tensor:
    """Method 4: Suppress non-object vision patches instead of boosting objects.

    Gentler than boosting — preserves relative distribution among task-relevant tokens.
    """
    all_vision = set(range(vision_end))
    keep = set(i for i in object_patch_indices if i < vision_end)
    suppress = sorted(all_vision - keep)
    if not suppress:
        return attn_weights
    idx = torch.tensor(suppress, dtype=torch.long, device=attn_weights.device)
    last = attn_weights[:, :, -1, :].clone()
    last[:, :, idx] = last[:, :, idx] * gamma
    last = last / last.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    attn_weights = attn_weights.clone()
    attn_weights[:, :, -1, :] = last
    return attn_weights


def apply_head_steering(
    attn_weights: torch.Tensor,
    patch_indices: list[int],
    vision_end: int,
    top_k_heads: int,
    amplify: float,
) -> torch.Tensor:
    """Method 3: Amplify object patches only in vision-attending heads."""
    valid = [i for i in patch_indices if i < attn_weights.shape[-1]]
    if not valid:
        return attn_weights

    last = attn_weights[:, :, -1, :]  # (B, H, kv_len)
    vision_attn = last[:, :, :vision_end].sum(dim=-1).mean(dim=0)  # (H,)

    k = min(top_k_heads, vision_attn.shape[0])
    _, top_head_idx = vision_attn.topk(k)
    top_mask = torch.zeros(vision_attn.shape[0], dtype=torch.bool, device=attn_weights.device)
    top_mask[top_head_idx] = True

    idx = torch.tensor(valid, dtype=torch.long, device=attn_weights.device)
    modified = attn_weights[:, :, -1, :].clone()
    selected = modified[:, top_mask, :]
    selected[:, :, idx] = selected[:, :, idx] * amplify
    selected = selected / selected.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    modified[:, top_mask, :] = selected
    attn_weights = attn_weights.clone()
    attn_weights[:, :, -1, :] = modified
    return attn_weights


# ======================================================================
# Monkey-patch engine
# ======================================================================

_original_fn: Optional[Callable] = None


def _make_patched_forward(ctx: EnhancementContext) -> Callable:
    from transformers.models.llama.modeling_llama import repeat_kv

    def patched_eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)

        attn_logits = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_logits = attn_logits + causal_mask

        # Hook A: pre-softmax logit bias
        layer_idx = getattr(module, "layer_idx", -1)
        if ctx.is_layer_active(layer_idx) and ctx.method == "logit_bias":
            attn_logits = apply_logit_bias(
                attn_logits, ctx.object_patch_indices, ctx.logit_bias_alpha
            )

        attn_weights = nn.functional.softmax(
            attn_logits, dim=-1, dtype=torch.float32
        ).to(query.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=dropout, training=module.training
        )

        # Hook B: post-softmax methods
        if ctx.is_layer_active(layer_idx):
            if ctx.method == "weight_rescale":
                attn_weights = apply_weight_rescale(
                    attn_weights, ctx.object_patch_indices, ctx.weight_rescale_lambda
                )
            elif ctx.method == "head_steering":
                vision_end = getattr(module, "_atlasvla_vision_end", 256)
                attn_weights = apply_head_steering(
                    attn_weights, ctx.object_patch_indices,
                    vision_end, ctx.head_steer_top_k, ctx.head_steer_amplify,
                )
            elif ctx.method == "bg_suppress":
                vision_end = getattr(module, "_atlasvla_vision_end", 256)
                attn_weights = apply_background_suppression(
                    attn_weights, ctx.object_patch_indices,
                    vision_end, ctx.bg_suppress_gamma,
                )

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    return patched_eager_attention_forward


def install_patch(ctx: Optional[EnhancementContext] = None) -> None:
    """Replace eager_attention_forward in the transformers llama module."""
    global _original_fn
    import transformers.models.llama.modeling_llama as llama_module

    if _original_fn is not None:
        return

    _original_fn = llama_module.eager_attention_forward
    active_ctx = ctx if ctx is not None else _ctx
    llama_module.eager_attention_forward = _make_patched_forward(active_ctx)
    print("[attention_patch] Patched eager_attention_forward installed.")


def uninstall_patch() -> None:
    """Restore the original eager_attention_forward."""
    global _original_fn
    import transformers.models.llama.modeling_llama as llama_module

    if _original_fn is None:
        return
    llama_module.eager_attention_forward = _original_fn
    _original_fn = None
    print("[attention_patch] Original eager_attention_forward restored.")


def inject_vision_end(model, vision_end: int) -> None:
    """Attach vision token boundary to each attention module for head_steering."""
    if hasattr(model, "language_model"):
        layers = model.language_model.model.layers
    else:
        layers = model.model.layers
    for layer in layers:
        layer.self_attn._atlasvla_vision_end = vision_end
