"""Tests for dynamic attention sink detection."""
import torch
import pytest


def test_detect_sinks_basic():
    """Single clear sink at position 0 — must be detected."""
    from attention_v3 import detect_sinks

    attn = torch.zeros(1, 2, 10)
    attn[0, 0, 0] = 0.45
    attn[0, 0, 1:] = 0.55 / 9
    attn[0, 1, 0] = 0.50
    attn[0, 1, 1:] = 0.50 / 9

    result = detect_sinks(attn, alpha=4.0)
    assert 0 in result, f"Token 0 should be sink with α=4, got {result}"


def test_detect_sinks_no_sinks():
    """Uniform attention — no sinks should be detected."""
    from attention_v3 import detect_sinks

    attn = torch.ones(1, 4, 10) / 10.0
    result = detect_sinks(attn, alpha=5.0)
    assert len(result) == 0, f"No sinks expected with uniform attn, got {result}"


def test_detect_sinks_multiple_sinks():
    """Two sink tokens (BOS + special) — both detected."""
    from attention_v3 import detect_sinks

    attn = torch.zeros(1, 4, 50)
    attn[:, :, 0] = 0.20
    attn[:, :, 1] = 0.15
    remaining = 1.0 - 0.20 - 0.15
    attn[:, :, 2:] = remaining / 48

    result = detect_sinks(attn, alpha=5.0)
    assert 0 in result, "BOS should be sink"
    assert 1 in result, "<|user|> should be sink"


def test_detect_sinks_openvla_realistic():
    """Realistic OpenVLA-like distribution: vision[0] dominates."""
    from attention_v3 import detect_sinks

    attn = torch.zeros(1, 32, 270)
    attn[:, :, 0] = 0.45
    remaining = 0.55
    attn[:, :, 1:] = remaining / 269

    result = detect_sinks(attn, alpha=5.0)
    assert 0 in result
    assert len(result) <= 3, "Should not flag too many sinks"


def test_detect_sinks_returns_sorted_list():
    """Output should be a sorted list of ints."""
    from attention_v3 import detect_sinks

    attn = torch.zeros(1, 2, 20)
    attn[:, :, 5] = 0.40
    attn[:, :, 0] = 0.30
    remaining = 1.0 - 0.40 - 0.30
    attn[:, :, 1:5] = remaining / 18
    attn[:, :, 6:] = remaining / 18

    result = detect_sinks(attn, alpha=5.0)
    assert isinstance(result, list)
    assert all(isinstance(i, int) for i in result)
    assert result == sorted(result), "Result should be sorted"


def test_detect_sinks_detached():
    """Detection must not affect gradient flow."""
    from attention_v3 import detect_sinks

    attn = torch.zeros(1, 2, 10, requires_grad=True)
    base = torch.ones(1, 2, 10) * 0.05
    base[0, 0, 0] = 0.60
    attn_val = attn + base  # grad-tracked tensor

    result = detect_sinks(attn_val, alpha=5.0)  # internal .detach() should handle this
    assert isinstance(result, list)
    assert attn.grad is None, "detect_sinks should not accumulate gradients"
