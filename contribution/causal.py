# contribution/causal.py
"""
Causal verification via length-preserving masking (Design Section 4.3).

Method: Value-zero (V=0) — zero out value projections for target tokens.
Attention knockout is DEPRECATED (was a no-op bug — see AttentionKnockoutHook docstring).
"""
import torch
import torch.nn as nn
from typing import Optional


class AttentionKnockoutHook:
    """DEPRECATED (Phase 2.5): This hook is a complete no-op.

    Bug: register_forward_hook fires AFTER self_attn.forward() completes.
    By that point, attn_output = attn_weights @ value has already been computed.
    Modifying output[1] (weights) does NOT change output[0] (attn_output).
    Result: KL=0.0 on all models (confirmed ECoT, OpenVLA, SpatialVLA).

    Use ValueZeroHook instead — it hooks v_proj directly (in computation path).

    For attention-level intervention, modify attention_mask BEFORE forward pass:
        attention_mask[:, :, query_range, target_positions] = -inf
    """

    def __init__(self, target_positions: list[int], query_range: tuple[int, int] | None = None):
        self.target_positions = target_positions
        self.query_range = query_range
        self._handles = []

    def apply_mask(self, scores: torch.Tensor) -> torch.Tensor:
        """Apply knockout mask to attention scores (pre-softmax).

        Args:
            scores: (batch, H, seq, seq) — raw attention scores
        Returns:
            masked scores with -inf at target positions
        """
        masked = scores.clone()
        q_start = self.query_range[0] if self.query_range else 0
        q_end = self.query_range[1] if self.query_range else scores.shape[2]

        for t in self.target_positions:
            if t < scores.shape[3]:
                masked[:, :, q_start:q_end, t] = float("-inf")

        return masked

    def register(self, model, model_cfg, get_layers_fn):
        """Register hooks on all attention layers."""
        layers = get_layers_fn(model, model_cfg)
        for layer in layers:
            handle = layer.self_attn.register_forward_hook(self._make_hook())
            self._handles.append(handle)

    def _make_hook(self):
        knockout = self

        def hook_fn(module, args, output):
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                attn_weights = output[1].clone()
                q_start = knockout.query_range[0] if knockout.query_range else 0
                q_end = knockout.query_range[1] if knockout.query_range else attn_weights.shape[2]
                for t in knockout.target_positions:
                    if t < attn_weights.shape[3]:
                        attn_weights[:, :, q_start:q_end, t] = 0.0
                row_sums = attn_weights[:, :, q_start:q_end, :].sum(dim=-1, keepdim=True).clamp(min=1e-10)
                attn_weights[:, :, q_start:q_end, :] = attn_weights[:, :, q_start:q_end, :] / row_sums
                return (output[0], attn_weights) + output[2:]
            return output

        return hook_fn

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


class ValueZeroHook:
    """Zero out value vectors for specific tokens (length-preserving).

    Hooks v_proj (LLaMA/Gemma) or qkv_proj (Phi3V) to zero out value
    projections at target positions before attention is computed.
    """

    def __init__(self, target_positions: list[int], target_layers: list[int] | None = None):
        self.target_positions = target_positions
        self.target_layers = target_layers  # None = all layers
        self._handles = []
        self._sanity_changed = False

    def register(self, model, model_cfg, get_layers_fn):
        """Register hooks to zero out V projections for target tokens.
        If target_layers is set, only hooks those specific layer indices.
        """
        layers = get_layers_fn(model, model_cfg)
        num_heads = model_cfg.num_heads
        num_kv_heads = getattr(model_cfg, 'num_kv_heads', None) or num_heads
        head_dim = model_cfg.hidden_dim // num_heads

        for layer_idx, layer in enumerate(layers):
            if self.target_layers is not None and layer_idx not in self.target_layers:
                continue
            attn = layer.self_attn
            if hasattr(attn, "v_proj"):
                handle = attn.v_proj.register_forward_hook(self._make_v_proj_hook())
                self._handles.append(handle)
            elif hasattr(attn, "qkv_proj"):
                q_dim = num_heads * head_dim
                kv_dim = num_kv_heads * head_dim
                v_start = q_dim + kv_dim
                v_end = q_dim + 2 * kv_dim
                handle = attn.qkv_proj.register_forward_hook(
                    self._make_fused_qkv_hook(v_start, v_end)
                )
                self._handles.append(handle)

    def _make_v_proj_hook(self):
        vzero = self
        def hook_fn(module, args, output):
            modified = output.clone()
            for t in vzero.target_positions:
                if t < modified.shape[1]:
                    modified[:, t, :] = 0.0
                    vzero._sanity_changed = True
            return modified
        return hook_fn

    def _make_fused_qkv_hook(self, v_start: int, v_end: int):
        vzero = self
        def hook_fn(module, args, output):
            modified = output.clone()
            for t in vzero.target_positions:
                if t < modified.shape[1]:
                    modified[:, t, v_start:v_end] = 0.0
                    vzero._sanity_changed = True
            return modified
        return hook_fn

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


class ValueMeanHook:
    """Replace value vectors at target positions with dataset mean (mean ablation).

    Unlike V=0, this avoids OOD activation shift by replacing with a
    plausible value vector, making the causal claim more robust.
    Complements V=0 ablation per Hase & Bansal (2021) recommendations.
    """

    def __init__(self, target_positions: list[int],
                 mean_values: torch.Tensor | None = None,
                 target_layers: list[int] | None = None):
        """
        Args:
            target_positions: Token indices to ablate.
            mean_values: Pre-computed mean V output per layer.
                Shape: (hidden_dim,) — same mean for all target positions.
                If None, uses running mean computed during first N forward passes.
            target_layers: Layer indices to hook (None = all layers).
        """
        self.target_positions = target_positions
        self.target_layers = target_layers
        self.mean_values = mean_values  # Will be set per-layer if needed
        self._handles = []
        self._sanity_changed = False
        self._layer_means = {}  # layer_idx → mean tensor

    def set_layer_means(self, layer_means: dict):
        """Set pre-computed per-layer mean V projections."""
        self._layer_means = layer_means

    def register(self, model, model_cfg, get_layers_fn):
        layers = get_layers_fn(model, model_cfg)
        num_heads = model_cfg.num_heads
        num_kv_heads = getattr(model_cfg, 'num_kv_heads', None) or num_heads
        head_dim = model_cfg.hidden_dim // num_heads

        for layer_idx, layer in enumerate(layers):
            if self.target_layers is not None and layer_idx not in self.target_layers:
                continue
            attn = layer.self_attn
            if hasattr(attn, "v_proj"):
                handle = attn.v_proj.register_forward_hook(
                    self._make_v_proj_hook(layer_idx)
                )
                self._handles.append(handle)
            elif hasattr(attn, "qkv_proj"):
                q_dim = num_heads * head_dim
                kv_dim = num_kv_heads * head_dim
                v_start = q_dim + kv_dim
                v_end = q_dim + 2 * kv_dim
                handle = attn.qkv_proj.register_forward_hook(
                    self._make_fused_qkv_hook(layer_idx, v_start, v_end)
                )
                self._handles.append(handle)

    def _make_v_proj_hook(self, layer_idx):
        hook_self = self
        def hook_fn(module, args, output):
            modified = output.clone()
            mean_v = hook_self._layer_means.get(layer_idx)
            for t in hook_self.target_positions:
                if t < modified.shape[1]:
                    if mean_v is not None:
                        modified[:, t, :] = mean_v.to(modified.device, modified.dtype)
                    else:
                        # Fallback: use mean of all positions in this forward pass
                        modified[:, t, :] = modified[:, :, :].mean(dim=1)
                    hook_self._sanity_changed = True
            return modified
        return hook_fn

    def _make_fused_qkv_hook(self, layer_idx, v_start, v_end):
        hook_self = self
        def hook_fn(module, args, output):
            modified = output.clone()
            mean_v = hook_self._layer_means.get(layer_idx)
            for t in hook_self.target_positions:
                if t < modified.shape[1]:
                    if mean_v is not None:
                        modified[:, t, v_start:v_end] = mean_v.to(
                            modified.device, modified.dtype
                        )
                    else:
                        modified[:, t, v_start:v_end] = modified[
                            :, :, v_start:v_end
                        ].mean(dim=1)
                    hook_self._sanity_changed = True
            return modified
        return hook_fn

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def compute_output_kl(
    logits_orig: torch.Tensor,
    logits_masked: torch.Tensor,
) -> float:
    """KL(p_orig || p_masked) for action token logits (Design Section 4.3).

    Args:
        logits_orig: original model output logits
        logits_masked: logits after masking intervention
    Returns:
        KL divergence (scalar)
    """
    p = torch.softmax(logits_orig.float(), dim=-1).clamp(min=1e-10)
    q = torch.softmax(logits_masked.float(), dim=-1).clamp(min=1e-10)
    kl = (p * (p.log() - q.log())).sum(dim=-1).mean()
    return float(kl.item())


def compute_top1_change_rate(logits_orig: torch.Tensor, logits_masked: torch.Tensor) -> float:
    """Fraction where top-1 predicted token changes after masking."""
    return float((logits_orig.argmax(-1) != logits_masked.argmax(-1)).float().mean().item())


def run_vzero_sanity_check(model, model_cfg, get_layers_fn, inputs, target_positions: list[int]) -> dict:
    """Verify that V=0 hook actually changes model output.
    Returns: {"hook_fired": bool, "logits_changed": bool, "kl_divergence": float}
    """
    with torch.no_grad():
        out_orig = model(**inputs)
    logits_orig = out_orig.logits[0, -1, :]

    vzero = ValueZeroHook(target_positions)
    vzero.register(model, model_cfg, get_layers_fn)
    with torch.no_grad():
        out_masked = model(**inputs)
    logits_masked = out_masked.logits[0, -1, :]
    vzero.remove()

    kl = compute_output_kl(logits_orig, logits_masked)
    return {
        "hook_fired": vzero._sanity_changed if hasattr(vzero, '_sanity_changed') else True,
        "logits_changed": not torch.allclose(logits_orig, logits_masked, atol=1e-5),
        "kl_divergence": kl,
    }


def get_deep_layer_ranges(num_layers: int) -> dict[str, list[int]]:
    """Return deep layer ranges for block-level V=0 experiments.

    For a model with N layers, deep = last 10 layers.
    block1 = first half of deep, block2 = second half.

    Args:
        num_layers: Total number of transformer layers.

    Returns:
        {"all": [...], "block1": [...], "block2": [...]}
    """
    deep_start = max(0, num_layers - 10)
    deep_end = num_layers
    mid = deep_start + (deep_end - deep_start) // 2
    return {
        "all": list(range(deep_start, deep_end)),
        "block1": list(range(deep_start, mid)),
        "block2": list(range(mid, deep_end)),
    }
