"""Tests for text masking hooks."""
import sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from contribution.text_mask import TextValueZeroHook, TextKVMaskHook


def test_text_value_zero_hook_init():
    hook = TextValueZeroHook(text_range=(256, 280))
    assert hook.text_range == (256, 280)
    assert len(hook._handles) == 0


def test_text_kv_mask_hook_init():
    hook = TextKVMaskHook(text_range=(256, 280))
    assert hook.text_range == (256, 280)


def test_text_kv_mask_modifies_attention_mask():
    """KV-mask should set text columns to -inf in attention_mask."""
    hook = TextKVMaskHook(text_range=(5, 10))
    # Simulate a 4D attention mask: (B, 1, seq, seq)
    seq = 15
    mask = torch.zeros(1, 1, seq, seq)
    modified = hook.apply_to_attention_mask(mask)
    # Text range columns should be -inf
    assert torch.isinf(modified[0, 0, 0, 5]).item()
    assert torch.isinf(modified[0, 0, 0, 9]).item()
    # Non-text columns should be 0
    assert modified[0, 0, 0, 0].item() == 0.0
    assert modified[0, 0, 0, 11].item() == 0.0


def test_normalized_vision_mask_count():
    """Vision V=0 (normalized) should mask same number of tokens as text."""
    from contribution.text_mask import sample_normalized_vision_positions
    text_range = (256, 273)  # 17 text tokens
    vision_range = (0, 256)  # 256 vision tokens
    n_text = text_range[1] - text_range[0]  # 17
    positions = sample_normalized_vision_positions(vision_range, n_text, seed=42)
    assert len(positions) == n_text
    # All positions should be within vision range
    for p in positions:
        assert vision_range[0] <= p < vision_range[1]


if __name__ == "__main__":
    test_text_value_zero_hook_init()
    test_text_kv_mask_hook_init()
    test_text_kv_mask_modifies_attention_mask()
    test_normalized_vision_mask_count()
    print("All tests passed!")
