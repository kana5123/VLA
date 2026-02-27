"""Text masking hooks for Gate 3 leakage control.

Two modes:
1. TextValueZeroHook: Zeros V projections for text tokens.
   Q/K routing remains active -- verb routing info can still flow.
2. TextKVMaskHook: Creates a 4D causal mask with text columns set to -inf.
   Fully blocks text tokens from being attended to -- kills routing AND content.

CRITICAL: Both hooks accept text_ranges (list of (start,end) tuples) to handle
disjoint text regions (e.g. Phi3V: [text_prefix][vision][text_suffix]).

Usage:
    boundaries = detect_token_boundaries(...)
    text_ranges = boundaries["text_ranges"]  # list of (start, end) tuples

    # Text V=0 (reuses ValueZeroHook with text range)
    hook = TextValueZeroHook(text_ranges)
    hook.register(model, model_cfg, get_layers_fn)

    # Text KV-mask (creates 4D mask -- MUST use 4D to bypass LLaMA mask regen)
    hook = TextKVMaskHook(text_ranges)
    inputs["attention_mask"] = hook.apply(inputs["attention_mask"])
"""
import torch
import numpy as np
from contribution.causal import ValueZeroHook


def _ranges_to_positions(ranges: list[tuple[int, int]]) -> list[int]:
    """Convert list of (start, end) ranges to flat list of positions."""
    positions = []
    for s, e in ranges:
        positions.extend(range(s, e))
    return positions


def _make_4d_causal_mask(seq_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Create a standard 4D causal attention mask.

    Returns: (1, 1, seq_len, seq_len) with 0 for allowed positions, -inf for future.
    """
    # Lower triangular = causal (query at i can attend to j <= i)
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)  # upper triangle = -inf
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)


class TextValueZeroHook(ValueZeroHook):
    """Zero out V projections for all text tokens.

    Inherits from ValueZeroHook -- converts text ranges to target positions.
    Accepts either a single (start, end) tuple or a list of ranges.
    """

    def __init__(self, text_ranges, target_layers: list[int] | None = None):
        """
        Args:
            text_ranges: Either (text_start, text_end) or [(s1,e1), (s2,e2), ...]
                         Disjoint text regions from boundaries["text_ranges"].
            target_layers: Optional layer filter (None = all layers).
        """
        # Normalize: single tuple -> list of tuples
        if isinstance(text_ranges, tuple) and len(text_ranges) == 2 and isinstance(text_ranges[0], int):
            text_ranges = [text_ranges]
        self.text_ranges = text_ranges
        target_positions = _ranges_to_positions(text_ranges)
        super().__init__(target_positions, target_layers=target_layers)


class TextKVMaskHook:
    """Block text tokens entirely via 4D attention mask.

    CRITICAL: LLaMA/Phi3V ignores 2D mask modifications because forward()
    calls create_causal_mask() which regenerates 4D masks from scratch.
    Only pre-built 4D masks bypass this regeneration.

    This hook ALWAYS creates a 4D causal mask with text columns set to -inf,
    regardless of input mask dimensionality.

    Accepts disjoint text ranges for Phi3V-style [prefix][vision][suffix] layouts.
    """

    def __init__(self, text_ranges):
        """
        Args:
            text_ranges: Either (text_start, text_end) or [(s1,e1), (s2,e2), ...]
                         Disjoint text regions from boundaries["text_ranges"].
        """
        # Normalize: single tuple -> list of tuples
        if isinstance(text_ranges, tuple) and len(text_ranges) == 2 and isinstance(text_ranges[0], int):
            text_ranges = [text_ranges]
        self.text_ranges = text_ranges
        self.text_positions = _ranges_to_positions(text_ranges)

    def apply(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Create 4D causal mask with text columns blocked.

        ALWAYS returns a 4D mask (B, 1, seq, seq) regardless of input dim.
        This is the ONLY way to reliably block tokens in LLaMA/Phi3V.

        Args:
            attention_mask: Shape (B, 1, seq, seq), (B, H, seq, seq), or (B, seq).

        Returns:
            4D attention_mask with text columns set to -inf.
        """
        if attention_mask.dim() == 4:
            # Already 4D -- just block text columns
            mask = attention_mask.clone().float()
        elif attention_mask.dim() == 2:
            # 2D padding mask (B, seq) -- convert to 4D causal mask
            # CRITICAL: 2D masks are often int64; must use float for -inf
            B, seq_len = attention_mask.shape
            float_dtype = torch.float32  # always float for mask math
            mask = _make_4d_causal_mask(seq_len, float_dtype, attention_mask.device)
            mask = mask.expand(B, -1, -1, -1).clone()  # (B, 1, seq, seq)

            # Apply padding from original 2D mask (0 = pad -> -inf in 4D)
            pad_positions = (attention_mask == 0)  # (B, seq)
            if pad_positions.any():
                # Block columns for padded positions
                pad_cols = pad_positions.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, seq)
                mask = mask.masked_fill(pad_cols, float("-inf"))
        else:
            raise ValueError(f"Unexpected attention_mask dim: {attention_mask.dim()}")

        # Block text columns: no query can attend to text tokens
        for pos in self.text_positions:
            if pos < mask.shape[-1]:
                mask[:, :, :, pos] = float("-inf")

        return mask

    # Keep old name as alias for backward compat
    def apply_to_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.apply(attention_mask)

    def get_masked_positions(self) -> list[int]:
        """Return the absolute positions being masked."""
        return list(self.text_positions)

    def get_n_masked(self) -> int:
        """Return total number of text tokens being masked."""
        return len(self.text_positions)

    def get_masked_token_strs(self, input_ids, tokenizer) -> list[str]:
        """Return the actual token strings being masked (for report verification).

        Args:
            input_ids: Full sequence token IDs (list or 1D tensor).
            tokenizer: For decoding.

        Returns:
            List of token strings in the masked ranges.
        """
        if hasattr(input_ids, 'tolist'):
            input_ids = input_ids.tolist()
        strs = []
        for pos in self.text_positions:
            if pos < len(input_ids):
                try:
                    s = tokenizer.decode([input_ids[pos]], skip_special_tokens=False).strip()
                except Exception:
                    s = f"<id:{input_ids[pos]}>"
                strs.append(s)
        return strs


class TextKVZeroHook:
    """Block text tokens by zeroing both K and V projections at text positions.

    For Prismatic models (OpenVLA, ECoT) where the external attention_mask
    is concatenated internally with vision masks (can't pass 4D).

    This hooks into k_proj and v_proj forward hooks to zero their output
    at text token positions, which effectively:
    - Prevents Q from matching K at those positions (attention scores → ~0)
    - Prevents V content from flowing even if attention somehow remains

    This is stronger than TextValueZeroHook (V-only) because it kills
    both routing (K) and content (V).
    """

    def __init__(self, text_positions: list[int], model, model_cfg, get_layers_fn):
        self.text_positions = text_positions
        self._handles = []
        self._any_fired = False

        layers = get_layers_fn(model, model_cfg)
        for layer in layers:
            attn = layer.self_attn
            # Hook both K and V projections
            if hasattr(attn, "k_proj"):
                self._handles.append(attn.k_proj.register_forward_hook(self._make_zero_hook()))
            if hasattr(attn, "v_proj"):
                self._handles.append(attn.v_proj.register_forward_hook(self._make_zero_hook()))
            if hasattr(attn, "qkv_proj"):
                # Fused QKV (Phi3V): zero both K and V portions
                num_heads = model_cfg.num_heads
                num_kv_heads = getattr(model_cfg, 'num_kv_heads', num_heads)
                head_dim = model_cfg.hidden_dim // num_heads
                q_dim = num_heads * head_dim
                kv_dim = num_kv_heads * head_dim
                k_start = q_dim
                v_end = q_dim + 2 * kv_dim
                self._handles.append(
                    attn.qkv_proj.register_forward_hook(self._make_fused_kv_hook(k_start, v_end))
                )

    @classmethod
    def from_ranges(cls, text_ranges, model, model_cfg, get_layers_fn):
        if isinstance(text_ranges, tuple) and len(text_ranges) == 2 and isinstance(text_ranges[0], int):
            text_ranges = [text_ranges]
        positions = _ranges_to_positions(text_ranges)
        return cls(positions, model, model_cfg, get_layers_fn)

    def _make_zero_hook(self):
        hook_self = self
        def hook_fn(module, args, output):
            modified = output.clone()
            for t in hook_self.text_positions:
                if t < modified.shape[1]:
                    modified[:, t, :] = 0.0
                    hook_self._any_fired = True
            return modified
        return hook_fn

    def _make_fused_kv_hook(self, k_start: int, v_end: int):
        hook_self = self
        def hook_fn(module, args, output):
            modified = output.clone()
            for t in hook_self.text_positions:
                if t < modified.shape[1]:
                    modified[:, t, k_start:v_end] = 0.0
                    hook_self._any_fired = True
            return modified
        return hook_fn

    def fired(self) -> bool:
        return self._any_fired

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def get_n_masked(self) -> int:
        return len(self.text_positions)


def create_text_kv_mask(text_ranges, model, model_cfg, get_layers_fn):
    """Factory function: choose the right text KV-masking strategy per architecture.

    For Prismatic (OpenVLA, ECoT): Uses TextKVZeroHook (k_proj/v_proj forward hooks).
        Prismatic internally concatenates 2D text mask with 2D vision patch mask,
        so we CANNOT pass a 4D external mask. Instead, we hook into k_proj and v_proj
        to zero their outputs at text positions.
    For others (SpatialVLA, TraceVLA): Uses TextKVMaskHook (4D mask).

    Args:
        text_ranges: List of (start, end) tuples or single tuple.
        model: The model.
        model_cfg: Model config.
        get_layers_fn: Function to get layers.

    Returns:
        (hook_or_none, apply_fn)
        - hook_or_none: TextKVZeroHook instance if hook-based, else None
        - apply_fn: function(inputs) -> modified inputs (for mask-based approach)
    """
    # Prismatic-based models: use hook approach (can't modify attention_mask)
    is_prismatic = hasattr(model, 'language_model') and hasattr(model, 'projector')

    if is_prismatic:
        hook = TextKVZeroHook.from_ranges(text_ranges, model, model_cfg, get_layers_fn)
        return hook, None
    else:
        # Standard models: use 4D mask approach
        kv_hook = TextKVMaskHook(text_ranges)
        def apply_mask(inputs):
            inputs["attention_mask"] = kv_hook.apply(inputs["attention_mask"])
            return inputs
        return None, apply_mask


def sample_normalized_vision_positions(
    vision_range: tuple[int, int],
    n_tokens: int,
    seed: int = 42,
) -> list[int]:
    """Sample n_tokens random vision positions for normalized comparison.

    Gate 3 Condition D: masks the SAME number of vision tokens as text tokens
    to make modality comparison fair (256 vision vs ~17 text).

    Args:
        vision_range: (vision_start, vision_end) absolute positions.
        n_tokens: Number of tokens to mask (= number of text tokens).
        seed: RNG seed for reproducibility.

    Returns:
        Sorted list of absolute vision positions to mask.
    """
    vs, ve = vision_range
    n_vision = ve - vs
    rng = np.random.default_rng(seed)
    n = min(n_tokens, n_vision)
    selected = rng.choice(n_vision, size=n, replace=False)
    return sorted(int(vs + idx) for idx in selected)
