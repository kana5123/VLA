"""V3 attention enhancement methods — research-based, sink-aware.

Implements four families of inference-time attention manipulation:

  1. VAR  (Visual Attention Redistribution, ICLR 2025)
     Redistributes attention from sink tokens to non-sink visual tokens.
     Only modifies image-centric heads (visual non-sink ratio >= rho).
     Extensions: decaying schedule, layer-selective application.

  2. ACT  (Attention Calibration Technique, arXiv 2406.15765)
     Identifies sink tokens dynamically (score > alpha/N), scales them
     down by beta, redistributes freed budget proportionally.

  3. SPIN (Head Suppression, EMNLP 2025)
     Identifies top-K vision-attending heads and suppresses others,
     forcing the model to rely on visual information more heavily.

  4. VTR  (Vision-Text Rebalance)
     Shifts attention budget from text tokens to vision tokens,
     counteracting the model's tendency to over-attend to language.

All methods operate post-softmax on the last query row (autoregressive
generation) and preserve the sum-to-1 constraint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from torch import nn

import config


# ======================================================================
# Dynamic sink detection
# ======================================================================

def detect_sinks(
    attn_weights: torch.Tensor,
    alpha: float = 5.0,
) -> list[int]:
    """Detect attention sink tokens using ACT-style α/N threshold.

    A token is a sink if its attention weight exceeds α/N where N is the
    sequence length. Detection is per-head; the union across all heads
    is returned.

    Args:
        attn_weights: (B, H, K) or (B, H, Q, K) attention weights (post-softmax).
        alpha: threshold multiplier (default 5.0 = 5x uniform average).
    Returns:
        Sorted list of sink token indices (union across batch and heads).
    """
    attn = attn_weights.detach().float()

    if attn.dim() == 4:
        attn = attn[:, :, -1, :]  # (B, H, K)

    N = attn.shape[-1]
    if N == 0:
        return []

    threshold = alpha / N
    sink_mask = attn > threshold
    any_sink = sink_mask.any(dim=0).any(dim=0)  # (K,)
    sink_indices = any_sink.nonzero(as_tuple=True)[0].tolist()
    return sorted(sink_indices)


# ======================================================================
# V3 enhancement context
# ======================================================================

@dataclass
class V3Context:
    """Mutable context for V3 attention methods."""

    active: bool = False

    # --- VAR ---
    use_var: bool = False
    var_p: float = config.VAR_P
    var_rho: float = config.VAR_RHO
    var_sink_indices: list[int] = field(
        default_factory=lambda: list(config.VAR_SINK_INDICES)
    )

    # --- VAR + object-aware redistribution ---
    use_object_redirect: bool = False
    object_patch_indices: list[int] = field(default_factory=list)
    object_redirect_weight: float = config.VAR_OBJECT_REDIRECT_WEIGHT

    # --- VAR decay schedule (p decreases across action tokens) ---
    var_decay: bool = False

    # --- Temporal motion-aware attention ---
    use_temporal: bool = False
    temporal_patch_indices: list[int] = field(default_factory=list)
    temporal_boost_weight: float = config.TEMPORAL_BOOST_WEIGHT

    # --- Vision-Text Rebalance ---
    use_vt_rebalance: bool = False
    vt_shift_fraction: float = config.VT_SHIFT_FRACTION

    # --- BG suppress (applied after VAR) ---
    use_bg_suppress: bool = False
    bg_gamma: float = config.BG_SUPPRESS_GAMMA

    # --- ACT ---
    use_act: bool = False
    act_alpha: float = config.ACT_SINK_ALPHA
    act_beta: float = config.ACT_SCALE_BETA

    # --- SPIN ---
    use_spin: bool = False
    spin_top_k: int = config.SPIN_TOP_K_HEADS
    spin_suppress_alpha: float = config.SPIN_SUPPRESS_ALPHA

    # --- Targeted intervention (V5) ---
    # Per-dimension VAR multiplier: length 7, gates VAR on/off per token
    dim_var_factors: Optional[list[float]] = None
    # Per-head VAR strength: (num_layers, num_heads) tensor on device
    per_head_var_strength: Optional[object] = None  # torch.Tensor

    # Per-patch redistribution weights from adapter v2 cross-attention
    redistribution_weights: Optional[object] = None  # torch.Tensor (V,) or None

    # --- Text sink redistribution ---
    text_sink_enabled: bool = False
    text_sink_p: float = 0.3
    text_sink_threshold: float = 0.15
    text_end: int = 0

    # --- Dynamic sink detection ---
    dynamic_sink_detection: bool = config.DYNAMIC_SINK_DETECTION
    sink_alpha: float = config.SINK_ALPHA

    # --- Common ---
    gripper_exempt: bool = False
    current_token_idx: int = 0
    vision_end: int = 256
    enhancement_layers: Optional[set[int]] = None

    def is_active(self, layer_idx: int) -> bool:
        if not self.active:
            return False
        if self.gripper_exempt and self.current_token_idx >= 6:
            return False
        if self.enhancement_layers is not None and layer_idx not in self.enhancement_layers:
            return False
        return True

    def effective_var_p(self) -> float:
        """Compute effective VAR redistribution fraction (scalar).

        With var_decay=True, p linearly decays from var_p to ~0 across
        the 7 action tokens:
          token 0 (x):       p * 1.0
          token 6 (gripper): p * 0.05 (near-zero)
        This prevents over-fixation on later tokens where contextual
        awareness matters more than spatial precision.
        """
        base_p = self.var_p

        # Per-dimension gating (V5 targeted)
        if self.dim_var_factors is not None:
            base_p = base_p * self.dim_var_factors[self.current_token_idx]

        if not self.var_decay:
            return base_p
        n_tokens = config.NUM_ACTION_TOKENS  # 7
        decay = max(0.05, 1.0 - self.current_token_idx / (n_tokens - 1))
        return base_p * decay

    def get_per_head_p(self, layer_idx: int) -> Optional[object]:
        """Get per-head VAR strength for a specific layer.

        Returns (H,) or (B, H) tensor if per_head_var_strength is set, else None.
        Supports both single-sample (L, H) and micro-batch (B, L, H) layouts.
        """
        if self.per_head_var_strength is None:
            return None
        phs = self.per_head_var_strength
        if phs.dim() == 2:
            head_p = phs[layer_idx]  # (H,)
        else:
            head_p = phs[:, layer_idx, :]  # (B, H)
        # Apply per-dimension gating
        if self.dim_var_factors is not None:
            head_p = head_p * self.dim_var_factors[self.current_token_idx]
        return head_p

    def set_object_patches(self, indices: list[int]) -> None:
        self.object_patch_indices = indices


_v3_ctx = V3Context()


def get_v3_context() -> V3Context:
    return _v3_ctx


def set_v3_context(ctx: V3Context) -> None:
    global _v3_ctx
    _v3_ctx = ctx


# ======================================================================
# Method 1: VAR — Visual Attention Redistribution
# ======================================================================


def set_var_differentiable(enabled: bool = True, temperature: float = 10.0):
    """Enable/disable differentiable mode for apply_var.

    When enabled, the hard binary rho-threshold head_mask is replaced with
    a soft sigmoid mask so gradients can flow through.
    """
    apply_var._differentiable = enabled
    apply_var._soft_temperature = temperature


def apply_var(
    attn_weights: torch.Tensor,
    sink_indices: list[int],
    vision_end: int,
    p: float,
    rho: float,
    object_indices: Optional[list[int]] = None,
    object_weight: float = 1.0,
    extra_boost_map: Optional[dict[int, float]] = None,
    per_head_p: Optional[torch.Tensor] = None,
    redistribution_weights: Optional[torch.Tensor] = None,
    text_sink_enabled: bool = False,
    text_sink_p: float = 0.3,
    text_sink_threshold: float = 0.15,
    text_end: int = 0,
) -> torch.Tensor:
    """Redistribute attention from sink tokens to non-sink visual tokens.

    Based on: Kang et al. "See What You Are Told" (ICLR 2025)

    Args:
        attn_weights: (B, H, Q, K) attention weights after softmax
        sink_indices: indices of sink tokens within vision range
        vision_end: index marking end of vision tokens
        p: fraction of sink attention to redistribute (scalar fallback)
        rho: threshold for selecting image-centric heads
        object_indices: if provided, preferentially redistribute to these
        object_weight: extra weight multiplier for object indices
        extra_boost_map: {patch_idx: weight} for additional boosted patches
                        (e.g. gripper patches, temporal motion patches)
        per_head_p: (H,) tensor of per-head redistribution strengths.
                    If provided, overrides scalar p for per-head control.
        redistribution_weights: (V,) tensor of per-patch weights from adapter
                    v2 cross-attention. If provided, overrides proportional
                    and object-weighted redistribution.
    """
    orig_dtype = attn_weights.dtype
    last = attn_weights[:, :, -1, :].clone().float()  # (B, H, K), compute in fp32
    B, H, K = last.shape

    sink_set = set(i for i in sink_indices if i < vision_end)
    non_sink_visual = [i for i in range(vision_end) if i not in sink_set]

    if not non_sink_visual or not sink_set:
        return attn_weights

    sink_t = torch.tensor(sorted(sink_set), device=last.device, dtype=torch.long)
    nonsink_t = torch.tensor(non_sink_visual, device=last.device, dtype=torch.long)

    # Step 1: Image-centric head selection (sink-aware)
    # Compute vision ratio EXCLUDING sink tokens
    non_sink_vision_mask = torch.ones(vision_end, device=last.device)
    for si in sink_indices:
        if si < vision_end:
            non_sink_vision_mask[si] = 0.0
    # Useful vision ratio = attention to non-sink vision tokens
    useful_vision = (last[:, :, :vision_end] * non_sink_vision_mask).sum(dim=-1)  # (B, H)

    # Soft sigmoid mask for differentiability (gradient flows through)
    # When differentiable=False (default), uses original hard threshold
    differentiable = getattr(apply_var, "_differentiable", False)
    if differentiable:
        _soft_temp = getattr(apply_var, "_soft_temperature", 10.0)
        head_mask = torch.sigmoid((useful_vision.mean(dim=0) - rho) * _soft_temp)  # (H,)
    else:
        head_mask = (useful_vision.mean(dim=0) >= rho).float()  # (H,)

    if head_mask.sum() == 0:
        return attn_weights

    # Step 2: Compute per-head effective p
    if per_head_p is not None:
        # Per-head targeted intervention:
        # per_head_p: (H,) for single-sample, (B, H) for micro-batch
        effective_p = per_head_p.float().to(last.device) * head_mask  # (H,) or (B, H)
        if effective_p.dim() == 1:
            ep = effective_p.unsqueeze(0).unsqueeze(-1)  # (1, H, 1)
        else:
            ep = effective_p.unsqueeze(-1)  # (B, H, 1)
    else:
        # Original scalar p × binary mask
        ep = p * head_mask.unsqueeze(0).unsqueeze(-1)  # (1, H, 1)

    # Reduce sink attention for selected heads
    sink_vals = last[:, :, sink_t]  # (B, H, S)
    freed = sink_vals.sum(dim=-1, keepdim=True) * ep  # (B, H, 1)
    new_sink = sink_vals * (1 - ep)

    # Compute redistribution weights
    nonsink_vals = last[:, :, nonsink_t]  # (B, H, NS)

    if redistribution_weights is not None:
        # V2: Use learned per-patch weights from adapter cross-attention
        # redistribution_weights: (V,) or (B, V) for micro-batch
        if redistribution_weights.dim() == 1:
            redist_for_nonsink = redistribution_weights[nonsink_t]  # (NS,)
            redist_for_nonsink = redist_for_nonsink.float().to(last.device)
            rw = redist_for_nonsink.unsqueeze(0).unsqueeze(0)  # (1, 1, NS)
        else:
            redist_for_nonsink = redistribution_weights[:, nonsink_t]  # (B, NS)
            redist_for_nonsink = redist_for_nonsink.float().to(last.device)
            rw = redist_for_nonsink.unsqueeze(1)  # (B, 1, NS)
        bonus = freed * rw  # (B, H, NS)
    else:
        # Original: proportional or object-weighted redistribution
        has_obj = object_indices and object_weight > 1.0
        has_extra = extra_boost_map and len(extra_boost_map) > 0
        non_sink_set = set(non_sink_visual)

        if has_obj or has_extra:
            weight_vec = torch.ones(len(non_sink_visual), device=last.device)
            if has_obj:
                obj_set = set(i for i in object_indices if i in non_sink_set)
                for idx, token_id in enumerate(non_sink_visual):
                    if token_id in obj_set:
                        weight_vec[idx] = object_weight
            if has_extra:
                for idx, token_id in enumerate(non_sink_visual):
                    if token_id in extra_boost_map:
                        weight_vec[idx] *= extra_boost_map[token_id]
            weighted_nonsink = nonsink_vals * weight_vec.unsqueeze(0).unsqueeze(0)
            weighted_sum = weighted_nonsink.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            bonus = freed * (weighted_nonsink / weighted_sum)
        else:
            nonsink_sum = nonsink_vals.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            bonus = freed * (nonsink_vals / nonsink_sum)

    new_nonsink = nonsink_vals + bonus

    # Write back
    modified = last.clone()
    modified[:, :, sink_t] = new_sink
    modified[:, :, nonsink_t] = new_nonsink

    attn_weights = attn_weights.clone()
    attn_weights[:, :, -1, :] = modified.to(orig_dtype)

    # Step 4: Text sink redistribution (optional)
    # Redistribute from the highest-attention text token (typically "\n")
    # to non-sink vision tokens
    if text_sink_enabled and text_end > vision_end:
        last = attn_weights[:, :, -1, :].clone().float()  # refresh after write-back
        text_region = last[:, :, vision_end:text_end]  # (B, H, T_text)
        text_max_val, text_max_local = text_region.max(dim=-1)  # (B, H)
        text_max_global = text_max_local + vision_end  # global index

        # Only redistribute from text tokens that hog > threshold of total attention
        text_hog_mask = (text_max_val > text_sink_threshold) & (head_mask.unsqueeze(0) > 0.5)  # (B, H)

        # Amount to move from each text sink
        text_to_move = text_max_val * text_sink_p * text_hog_mask.float()  # (B, H)

        # Reduce the text sink token
        # Use scatter_add approach: subtract from text sink, add to non-sink vision
        for b in range(attn_weights.size(0)):
            for h in range(attn_weights.size(1)):
                if text_hog_mask[b, h]:
                    move_amt = text_to_move[b, h]
                    text_idx = text_max_global[b, h].item()

                    # Subtract from text sink
                    attn_weights[b, h, -1, text_idx] = attn_weights[b, h, -1, text_idx] - move_amt.to(orig_dtype)

                    # Add to non-sink vision tokens using same redistribution as vision sink
                    # Use redistribution_weights if available, else proportional
                    if redistribution_weights is not None:
                        redist = redistribution_weights
                        if redist.dim() == 1:
                            redist = redist.unsqueeze(0)  # (1, V)
                        # Zero out sink positions and renormalize
                        redist_clean = redist.clone()
                        for si in sink_indices:
                            if si < vision_end:
                                redist_clean[:, si] = 0.0
                        redist_sum = redist_clean.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                        redist_norm = redist_clean / redist_sum
                        # Use correct batch index (b) for batched redist, or 0 for single-sample
                        ridx = b if redist_norm.size(0) > 1 else 0
                        attn_weights[b, h, -1, :vision_end] = (
                            attn_weights[b, h, -1, :vision_end] + move_amt.to(orig_dtype) * redist_norm[ridx].to(orig_dtype)
                        )
                    else:
                        # Proportional: distribute equally to non-sink vision
                        n_targets = vision_end - len(sink_indices)
                        if n_targets > 0:
                            per_token = move_amt / n_targets
                            for vi in range(vision_end):
                                if vi not in sink_set:
                                    attn_weights[b, h, -1, vi] = attn_weights[b, h, -1, vi] + per_token.to(orig_dtype)

    return attn_weights


# ======================================================================
# Method 2: ACT — Attention Calibration Technique
# ======================================================================

def apply_act(
    attn_weights: torch.Tensor,
    vision_end: int,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """Dynamic sink identification and calibration.

    Based on: Yu et al. "Unveiling and Harnessing Hidden Attention Sinks"

    Args:
        attn_weights: (B, H, Q, K) attention weights after softmax
        vision_end: end of vision token range
        alpha: sink threshold multiplier (sink if score > alpha/N)
        beta: scale factor for identified sinks
    """
    orig_dtype = attn_weights.dtype
    last = attn_weights[:, :, -1, :].clone().float()  # (B, H, K), compute in fp32
    B, H, K = last.shape

    if vision_end <= 0:
        return attn_weights

    vis_attn = last[:, :, :vision_end]  # (B, H, V)
    V = vision_end
    threshold = alpha / V  # per-head threshold

    # Identify sinks per head: tokens with attention > threshold
    # Average over batch for stability
    vis_mean = vis_attn.mean(dim=0)  # (H, V)
    sink_mask = vis_mean > threshold  # (H, V)

    if not sink_mask.any():
        return attn_weights

    # For each head, scale down sinks and redistribute
    for h in range(H):
        h_sinks = sink_mask[h]  # (V,)
        if not h_sinks.any():
            continue

        sink_idx = h_sinks.nonzero(as_tuple=True)[0]
        nonsink_idx = (~h_sinks).nonzero(as_tuple=True)[0]

        if nonsink_idx.numel() == 0:
            continue

        # Scale down sinks
        old_sink = last[:, h, sink_idx].clone()
        new_sink_vals = old_sink * beta
        freed = (old_sink - new_sink_vals).sum(dim=-1, keepdim=True)

        # Redistribute proportionally to non-sinks (within vision range)
        nonsink_vals = last[:, h, nonsink_idx]
        nonsink_sum = nonsink_vals.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        bonus = freed * (nonsink_vals / nonsink_sum)

        last[:, h, sink_idx] = new_sink_vals
        last[:, h, nonsink_idx] = nonsink_vals + bonus

    attn_weights = attn_weights.clone()
    attn_weights[:, :, -1, :] = last.to(orig_dtype)
    return attn_weights


# ======================================================================
# Method 3: SPIN — Head Suppression
# ======================================================================

def apply_spin(
    attn_output: torch.Tensor,
    attn_weights: torch.Tensor,
    vision_end: int,
    top_k: int,
    suppress_alpha: float,
) -> torch.Tensor:
    """Suppress image-inattentive attention heads.

    Based on: "Mitigating Hallucinations via Image-Guided Head Suppression"

    Unlike VAR/ACT which modify weights, SPIN operates on the attention
    output tensor, suppressing heads that don't attend to visual tokens.

    Args:
        attn_output: (B, H, Q, D_head) per-head attention output
        attn_weights: (B, H, Q, K) attention weights (for head selection)
        vision_end: end of vision token range
        top_k: number of vision-attending heads to keep at full strength
        suppress_alpha: multiplier for suppressed heads (e.g., 0.05)
    """
    if vision_end <= 0:
        return attn_output

    B, H, Q, D = attn_output.shape

    # Compute total visual attention per head (last query only)
    vis_attn = attn_weights[:, :, -1, :vision_end].sum(dim=-1).mean(dim=0)  # (H,)

    # Select top-K vision-attending heads
    k = min(top_k, H)
    _, top_idx = vis_attn.topk(k)
    keep_mask = torch.zeros(H, device=attn_output.device)
    keep_mask[top_idx] = 1.0

    # Build per-head scaling: kept heads = 1.0, suppressed = suppress_alpha
    scale = keep_mask + (1 - keep_mask) * suppress_alpha  # (H,)
    scale = scale.to(attn_output.dtype).view(1, H, 1, 1)  # broadcast

    return attn_output * scale


# ======================================================================
# BG suppress (reused from V2, applied after VAR)
# ======================================================================

def apply_bg_suppress_v3(
    attn_weights: torch.Tensor,
    object_indices: list[int],
    vision_end: int,
    gamma: float,
) -> torch.Tensor:
    """Suppress non-object vision patches after sink redistribution."""
    all_vision = set(range(vision_end))
    keep = set(i for i in object_indices if i < vision_end)
    suppress = sorted(all_vision - keep)
    if not suppress:
        return attn_weights
    orig_dtype = attn_weights.dtype
    idx = torch.tensor(suppress, dtype=torch.long, device=attn_weights.device)
    last = attn_weights[:, :, -1, :].clone().float()
    last[:, :, idx] = last[:, :, idx] * gamma
    last = last / last.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    attn_weights = attn_weights.clone()
    attn_weights[:, :, -1, :] = last.to(orig_dtype)
    return attn_weights


# ======================================================================
# Method 4: VTR — Vision-Text Rebalance
# ======================================================================

def apply_vt_rebalance(
    attn_weights: torch.Tensor,
    vision_end: int,
    shift_fraction: float,
) -> torch.Tensor:
    """Shift attention budget from text tokens to vision tokens.

    The model tends to over-attend to language instruction tokens.
    This method moves a fraction of the text-token attention to
    vision tokens, re-weighted proportionally.

    Args:
        attn_weights: (B, H, Q, K) attention weights after softmax
        vision_end: index marking end of vision tokens
        shift_fraction: fraction of text attention to shift to vision
    """
    orig_dtype = attn_weights.dtype
    last = attn_weights[:, :, -1, :].clone().float()  # (B, H, K)
    B, H, K = last.shape

    if vision_end <= 0 or vision_end >= K:
        return attn_weights

    vis_attn = last[:, :, :vision_end]         # (B, H, V)
    text_attn = last[:, :, vision_end:]         # (B, H, T)

    text_sum = text_attn.sum(dim=-1, keepdim=True)    # (B, H, 1)
    vis_sum = vis_attn.sum(dim=-1, keepdim=True)      # (B, H, 1)

    # Only rebalance heads where text dominates (text_sum > vis_sum)
    text_dominant = (text_sum > vis_sum).float()       # (B, H, 1)

    # Compute shift amount
    shift = shift_fraction * text_sum * text_dominant  # (B, H, 1)

    # Reduce text proportionally
    text_scale = (text_sum - shift) / text_sum.clamp(min=1e-9)
    new_text = text_attn * text_scale

    # Add to vision proportionally
    vis_bonus = shift * (vis_attn / vis_sum.clamp(min=1e-9))
    new_vis = vis_attn + vis_bonus

    modified = last.clone()
    modified[:, :, :vision_end] = new_vis
    modified[:, :, vision_end:] = new_text

    attn_weights = attn_weights.clone()
    attn_weights[:, :, -1, :] = modified.to(orig_dtype)
    return attn_weights


# ======================================================================
# Monkey-patch engine (V3)
# ======================================================================

_original_fn_v3: Optional[Callable] = None
_patched_arch: Optional[str] = None  # "llama", "gemma2", or "phi3_v"
_patched_class: Optional[type] = None  # The attention class that was patched


def _apply_v3_enhancements(ctx: V3Context, attn_weights: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """Apply V3 attention enhancements (VAR/ACT/VTR/BG) to attention weights.

    Shared logic extracted to avoid duplication across architecture-specific
    patched forwards. Operates on post-softmax (B, H, Q, K) weights.
    """
    if not ctx.is_active(layer_idx):
        return attn_weights

    if ctx.use_vt_rebalance:
        attn_weights = apply_vt_rebalance(
            attn_weights, ctx.vision_end, ctx.vt_shift_fraction,
        )

    if ctx.use_var:
        # Determine sink indices: dynamic detection or hardcoded
        if ctx.dynamic_sink_detection:
            last_attn = attn_weights[:, :, -1, :]
            sink_indices = detect_sinks(last_attn, alpha=ctx.sink_alpha)
            if not sink_indices:
                sink_indices = ctx.var_sink_indices
        else:
            sink_indices = ctx.var_sink_indices

        obj_idx = ctx.object_patch_indices if ctx.use_object_redirect else None
        obj_w = ctx.object_redirect_weight if ctx.use_object_redirect else 1.0
        extra_map = None
        if ctx.use_temporal and ctx.temporal_patch_indices:
            extra_map = {idx: ctx.temporal_boost_weight for idx in ctx.temporal_patch_indices}

        attn_weights = apply_var(
            attn_weights, sink_indices, ctx.vision_end,
            ctx.effective_var_p(), ctx.var_rho,
            object_indices=obj_idx, object_weight=obj_w,
            extra_boost_map=extra_map,
            per_head_p=ctx.get_per_head_p(layer_idx),
            redistribution_weights=ctx.redistribution_weights,
            text_sink_enabled=ctx.text_sink_enabled,
            text_sink_p=ctx.text_sink_p,
            text_sink_threshold=ctx.text_sink_threshold,
            text_end=ctx.text_end,
        )

    if ctx.use_act:
        attn_weights = apply_act(attn_weights, ctx.vision_end, ctx.act_alpha, ctx.act_beta)

    if ctx.use_bg_suppress and ctx.object_patch_indices:
        attn_weights = apply_bg_suppress_v3(
            attn_weights, ctx.object_patch_indices, ctx.vision_end, ctx.bg_gamma,
        )

    return attn_weights


def _apply_v3_spin(ctx: V3Context, attn_output: torch.Tensor, attn_weights: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """Apply SPIN head suppression (operates on output, not weights)."""
    if ctx.is_active(layer_idx) and ctx.use_spin:
        attn_output = apply_spin(
            attn_output, attn_weights, ctx.vision_end,
            ctx.spin_top_k, ctx.spin_suppress_alpha,
        )
    return attn_output


def _make_v3_patched_forward(ctx: V3Context) -> Callable:
    """Create a patched LlamaAttention.forward that injects VAR/ACT/SPIN/VTR.

    Compatible with transformers 4.57.x where LlamaAttention uses:
      - hidden_shape = (*input_shape, -1, self.head_dim) for reshaping
      - self.num_key_value_groups (not self.num_heads / self.num_key_value_heads)
      - self.scaling (not 1/sqrt(head_dim))
      - position_embeddings as required tuple(cos, sin)
      - Returns (attn_output, attn_weights) — 2 values, not 3
    """
    from transformers.models.llama.modeling_llama import (
        apply_rotary_pos_emb,
        repeat_kv,
    )

    def patched_llama_attention_forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Inline eager attention with V3 injection
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=0.0 if not self.training else self.attention_dropout, training=self.training,
        )

        # ── V3 Enhancement injection point ──
        layer_idx = getattr(self, "layer_idx", -1)
        attn_weights = _apply_v3_enhancements(ctx, attn_weights, layer_idx)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = _apply_v3_spin(ctx, attn_output, attn_weights, layer_idx)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights

    return patched_llama_attention_forward


def _make_v3_patched_forward_gemma2(ctx: V3Context) -> Callable:
    """Create a patched Gemma2Attention.forward with V3 injection.

    Compatible with transformers 4.57.x Gemma2Attention API.
    Handles Gemma2-specific features: logit softcapping, sliding window.
    """
    from transformers.models.gemma2.modeling_gemma2 import (
        apply_rotary_pos_emb,
        repeat_kv,
    )

    def patched_gemma2_attention_forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs,
            )

        # Inline eager attention with V3 injection (instead of delegating)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        scaling = self.scaling if hasattr(self, "scaling") else self.head_dim ** -0.5
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling

        # Gemma2 logit softcapping
        softcap = getattr(self, "attn_logit_softcapping", None)
        if softcap is not None:
            attn_weights = attn_weights / softcap
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * softcap

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        dropout_p = self.attention_dropout if self.training else 0.0
        attn_weights = nn.functional.dropout(attn_weights, p=dropout_p, training=self.training)

        # ── V3 Enhancement injection point ──
        layer_idx = getattr(self, "layer_idx", -1)
        attn_weights = _apply_v3_enhancements(ctx, attn_weights, layer_idx)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = _apply_v3_spin(ctx, attn_output, attn_weights, layer_idx)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights

    return patched_gemma2_attention_forward


def _make_v3_patched_forward_phi3v(ctx: V3Context) -> Callable:
    """Create a patched Phi3Attention.forward with V3 injection.

    Handles Phi3V-specific features: fused qkv_proj (single projection
    for Q, K, V concatenated), custom RoPE variants.
    """
    import math

    def patched_phi3v_attention_forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        # Phi3V uses fused QKV projection
        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos:query_pos + self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim:]

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError("layer_idx must be set for cache use")
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)

        # Import apply_rotary_pos_emb from the model's own module
        _apply_rope = getattr(self, "_apply_rotary_pos_emb", None)
        if _apply_rope is None:
            # Fallback: use the module-level function from the model's module
            model_module = type(self).__module__
            import importlib
            mod = importlib.import_module(model_module)
            _apply_rope = mod.apply_rotary_pos_emb
        query_states, key_states = _apply_rope(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # GQA expansion
        _repeat_kv = None
        try:
            model_module = type(self).__module__
            import importlib
            mod = importlib.import_module(model_module)
            _repeat_kv = mod.repeat_kv
        except (ImportError, AttributeError):
            from transformers.models.llama.modeling_llama import repeat_kv as _repeat_kv
        key_states = _repeat_kv(key_states, self.num_key_value_groups)
        value_states = _repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout if hasattr(self, 'attention_dropout') else 0.0, training=self.training)

        # ── V3 Enhancement injection point ──
        layer_idx = getattr(self, "layer_idx", -1)
        attn_weights = _apply_v3_enhancements(ctx, attn_weights, layer_idx)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = _apply_v3_spin(ctx, attn_output, attn_weights, layer_idx)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, "
                f"but is {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    return patched_phi3v_attention_forward


def _get_attention_class(architecture: str, model=None):
    """Get the attention class for a given architecture.

    Returns (attention_class, architecture_name) tuple.
    """
    if architecture == "llama":
        from transformers.models.llama.modeling_llama import LlamaAttention
        return LlamaAttention, "llama"
    elif architecture == "gemma2":
        from transformers.models.gemma2.modeling_gemma2 import Gemma2Attention
        return Gemma2Attention, "gemma2"
    elif architecture == "phi3_v":
        # Phi3V uses trust_remote_code — get attention class from loaded model
        if model is not None:
            # Unwrap PeftModel/DDP if needed
            base = model
            if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
                base = base.base_model.model
            elif hasattr(base, "module"):
                base = base.module
            # Walk the model to find the attention class
            layers_path = "model.layers"
            obj = base
            for attr in layers_path.split("."):
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None and len(obj) > 0:
                attn_module = getattr(obj[0], "self_attn", None)
                if attn_module is not None:
                    return type(attn_module), "phi3_v"
        # Fallback: try importing from transformers
        try:
            from transformers.models.phi3.modeling_phi3 import Phi3Attention
            return Phi3Attention, "phi3_v"
        except ImportError:
            raise ValueError(
                "Cannot find Phi3V attention class. Pass the loaded model "
                "to install_v3_patch() so it can discover the class dynamically."
            )
    else:
        raise ValueError(
            f"Unsupported architecture for V3 patching: {architecture}. "
            f"Supported: llama, gemma2, phi3_v"
        )


def _get_patched_forward(architecture: str, ctx: V3Context) -> Callable:
    """Get the appropriate patched forward function for an architecture."""
    if architecture == "llama":
        return _make_v3_patched_forward(ctx)
    elif architecture == "gemma2":
        return _make_v3_patched_forward_gemma2(ctx)
    elif architecture == "phi3_v":
        return _make_v3_patched_forward_phi3v(ctx)
    else:
        raise ValueError(f"No patched forward for architecture: {architecture}")


def install_v3_patch(
    ctx: Optional[V3Context] = None,
    architecture: str = "llama",
    model=None,
) -> None:
    """Patch the attention forward for the specified architecture.

    Args:
        ctx: V3Context with enhancement parameters. Uses global if None.
        architecture: "llama", "gemma2", or "phi3_v"
        model: The loaded model instance (required for phi3_v to discover
               the attention class from trust_remote_code modules).
    """
    global _original_fn_v3, _patched_arch, _patched_class

    if _original_fn_v3 is not None:
        return

    attn_class, arch_name = _get_attention_class(architecture, model=model)
    active_ctx = ctx if ctx is not None else _v3_ctx
    patched_fn = _get_patched_forward(arch_name, active_ctx)

    _original_fn_v3 = attn_class.forward
    _patched_arch = arch_name
    _patched_class = attn_class
    attn_class.forward = patched_fn
    print(f"[attention_v3] V3 patched {attn_class.__name__}.forward installed ({arch_name}).")


def uninstall_v3_patch() -> None:
    """Restore the original attention forward."""
    global _original_fn_v3, _patched_arch, _patched_class

    if _original_fn_v3 is None:
        return

    if _patched_class is not None:
        _patched_class.forward = _original_fn_v3
        print(f"[attention_v3] Original {_patched_class.__name__}.forward restored.")
    else:
        # Legacy fallback
        from transformers.models.llama.modeling_llama import LlamaAttention
        LlamaAttention.forward = _original_fn_v3
        print("[attention_v3] Original LlamaAttention.forward restored.")

    _original_fn_v3 = None
    _patched_arch = None
    _patched_class = None
