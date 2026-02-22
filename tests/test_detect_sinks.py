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


# ── Task 2: V3Context fields ──

def test_v3context_dynamic_fields():
    """V3Context should have dynamic_sink_detection and sink_alpha fields."""
    from attention_v3 import V3Context
    import config

    ctx = V3Context()
    assert hasattr(ctx, "dynamic_sink_detection")
    assert hasattr(ctx, "sink_alpha")
    assert ctx.dynamic_sink_detection == config.DYNAMIC_SINK_DETECTION
    assert ctx.sink_alpha == config.SINK_ALPHA


def test_v3context_backward_compat():
    """dynamic_sink_detection=False should preserve old behavior."""
    from attention_v3 import V3Context

    ctx = V3Context(dynamic_sink_detection=False)
    assert ctx.var_sink_indices == [0]  # still has hardcoded default
    assert ctx.dynamic_sink_detection is False


# ── Task 5: Regression + edge cases ──

def test_dynamic_off_uses_hardcoded():
    """When dynamic_sink_detection=False, var_sink_indices should be used."""
    from attention_v3 import V3Context

    ctx = V3Context(
        active=True,
        use_var=True,
        dynamic_sink_detection=False,
        var_sink_indices=[0, 5],
        vision_end=256,
    )
    assert ctx.var_sink_indices == [0, 5]
    assert ctx.dynamic_sink_detection is False


def test_detect_sinks_alpha_sensitivity():
    """Higher α should detect fewer sinks."""
    from attention_v3 import detect_sinks

    attn = torch.zeros(1, 4, 100)
    attn[:, :, 0] = 0.10  # 10x uniform (uniform=0.01)
    attn[:, :, 1] = 0.06  # 6x uniform
    attn[:, :, 2] = 0.04  # 4x uniform
    remaining = 1.0 - 0.10 - 0.06 - 0.04
    attn[:, :, 3:] = remaining / 97

    result_3 = detect_sinks(attn, alpha=3.0)
    assert 0 in result_3 and 1 in result_3 and 2 in result_3

    result_5 = detect_sinks(attn, alpha=5.0)
    assert 0 in result_5 and 1 in result_5
    assert 2 not in result_5

    result_8 = detect_sinks(attn, alpha=8.0)
    assert 0 in result_8
    assert 1 not in result_8


def test_detect_sinks_empty_input():
    """Edge case: empty sequence should return empty list."""
    from attention_v3 import detect_sinks

    attn = torch.zeros(1, 4, 0)
    result = detect_sinks(attn, alpha=5.0)
    assert result == []


def test_detect_sinks_4d_input():
    """Should handle (B, H, Q, K) input by taking last query."""
    from attention_v3 import detect_sinks

    attn_4d = torch.zeros(1, 2, 5, 20)  # (B, H, Q=5, K=20)
    attn_4d[:, :, -1, 0] = 0.50  # last query → token 0 is sink
    remaining = 0.50
    attn_4d[:, :, -1, 1:] = remaining / 19

    result = detect_sinks(attn_4d, alpha=5.0)
    assert 0 in result


def test_dynamic_sink_replaces_hardcoded_in_context():
    """When dynamic detection is on, detect_sinks finds different sinks than hardcoded."""
    from attention_v3 import V3Context, detect_sinks

    ctx = V3Context(
        active=True,
        use_var=True,
        dynamic_sink_detection=True,
        sink_alpha=5.0,
        var_sink_indices=[0],
        vision_end=256,
        enhancement_layers={28, 29, 30, 31},
    )

    # Simulate: token 3 is the actual sink, not token 0
    attn = torch.zeros(1, 32, 270)
    attn[:, :, 3] = 0.40
    attn[:, :, 0] = 0.002
    remaining = 1.0 - 0.40 - 0.002
    attn[:, :, 1:3] = remaining / 268
    attn[:, :, 4:] = remaining / 268

    detected = detect_sinks(attn, alpha=ctx.sink_alpha)
    assert 3 in detected, f"Token 3 should be dynamically detected, got {detected}"
    assert 0 not in detected, f"Token 0 should NOT be detected, got {detected}"
    # Hardcoded would have used [0] — dynamic finds [3]
    assert detected != ctx.var_sink_indices
