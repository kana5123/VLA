"""Text masking hooks for Gate 3 leakage control.

Two modes:
1. TextValueZeroHook: Zeros V projections for text tokens.
   Q/K routing remains active -- verb routing info can still flow.
2. TextKVMaskHook: Sets attention_mask[:, :, :, text_range] = -inf.
   Fully blocks text tokens from being attended to -- kills routing AND content.

Usage:
    # Text V=0 (reuses ValueZeroHook with text range)
    boundaries = detect_token_boundaries(...)
    text_range = (boundaries["text_start"], boundaries["text_end"])
    hook = TextValueZeroHook(text_range)
    hook.register(model, model_cfg, get_layers_fn)

    # Text KV-mask (modify attention_mask before forward)
    hook = TextKVMaskHook(text_range)
    inputs["attention_mask"] = hook.apply_to_attention_mask(inputs["attention_mask"])
"""
import torch
import numpy as np
from contribution.causal import ValueZeroHook


class TextValueZeroHook(ValueZeroHook):
    """Zero out V projections for all text tokens.

    Inherits from ValueZeroHook -- just converts a text range
    to a list of target positions.
    """

    def __init__(self, text_range: tuple[int, int], target_layers: list[int] | None = None):
        """
        Args:
            text_range: (text_start, text_end) absolute positions.
                        From detect_token_boundaries().
            target_layers: Optional layer filter (None = all layers).
        """
        self.text_range = text_range
        target_positions = list(range(text_range[0], text_range[1]))
        super().__init__(target_positions, target_layers=target_layers)


class TextKVMaskHook:
    """Block text tokens entirely via attention mask modification.

    Sets attention_mask[:, :, :, text_range] = -inf BEFORE the forward pass.
    This prevents any query from attending to text tokens -- kills both
    value content AND Q/K routing information.

    Unlike TextValueZeroHook, this does NOT use forward hooks.
    Instead, modify the attention_mask tensor directly before model(**inputs).
    """

    def __init__(self, text_range: tuple[int, int]):
        """
        Args:
            text_range: (text_start, text_end) absolute positions.
        """
        self.text_range = text_range

    def apply_to_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Modify attention_mask to block text tokens.

        Args:
            attention_mask: Shape (B, 1, seq, seq) or (B, seq).
                If 2D, creates a 4D causal mask first.

        Returns:
            Modified attention_mask with text columns set to -inf.
        """
        mask = attention_mask.clone()
        ts, te = self.text_range

        if mask.dim() == 4:
            # (B, 1, seq, seq) or (B, H, seq, seq)
            mask[:, :, :, ts:te] = float("-inf")
        elif mask.dim() == 2:
            # (B, seq) -- set text positions to 0 (will be expanded to -inf by model)
            mask[:, ts:te] = 0
        else:
            raise ValueError(f"Unexpected attention_mask dim: {mask.dim()}")

        return mask

    def get_masked_token_strs(self, input_ids: list[int], tokenizer) -> list[str]:
        """Return the actual token strings being masked (for report verification).

        Args:
            input_ids: Full sequence token IDs.
            tokenizer: For decoding.

        Returns:
            List of token strings in the masked range.
        """
        ts, te = self.text_range
        strs = []
        for pos in range(ts, min(te, len(input_ids))):
            try:
                s = tokenizer.decode([input_ids[pos]], skip_special_tokens=False).strip()
            except Exception:
                s = f"<id:{input_ids[pos]}>"
            strs.append(s)
        return strs


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
