"""Tests for apply_var with redistribution_weights parameter."""
import torch


def _make_attn_weights(B=1, H=4, K=32, vision_end=16):
    """Create synthetic attention weights with a sink at index 0."""
    w = torch.rand(B, H, 1, K)
    w[:, :, :, 0] = 2.0  # heavy sink
    w = w / w.sum(dim=-1, keepdim=True)
    return w


def test_apply_var_with_redistribution_weights():
    """When redistribution_weights provided, freed attention goes to weighted patches."""
    from attention_v3 import apply_var

    B, H, K, V = 1, 4, 32, 16
    attn = _make_attn_weights(B, H, K, V)
    original_bg = attn[:, :, -1, 5].clone()

    redist = torch.zeros(V)
    redist[10] = 0.5
    redist[11] = 0.3
    redist[12] = 0.2

    result = apply_var(
        attn, sink_indices=[0], vision_end=V, p=0.5, rho=0.0,
        redistribution_weights=redist,
    )

    # Background patch 5 should NOT increase (redist[5] == 0)
    assert result[0, 0, -1, 5].item() <= original_bg[0, 0].item() + 1e-6

    # Object patches should increase
    for idx in [10, 11, 12]:
        assert result[0, 0, -1, idx].item() > attn[0, 0, -1, idx].item()


def test_apply_var_redist_weights_override_proportional():
    """redistribution_weights should override proportional redistribution."""
    from attention_v3 import apply_var

    B, H, K, V = 1, 2, 20, 10
    attn = _make_attn_weights(B, H, K, V)

    redist = torch.zeros(V)
    redist[5] = 1.0  # ALL weight on patch 5

    result = apply_var(
        attn, sink_indices=[0], vision_end=V, p=0.8, rho=0.0,
        redistribution_weights=redist,
    )

    # Patch 5 should get ALL freed attention
    assert result[0, 0, -1, 5].item() > attn[0, 0, -1, 5].item()
    # Other non-sink patches should NOT change
    for idx in range(1, V):
        if idx == 5:
            continue
        diff = abs(result[0, 0, -1, idx].item() - attn[0, 0, -1, idx].item())
        assert diff < 1e-5, f"Patch {idx} changed by {diff}"


def test_apply_var_without_redist_unchanged():
    """Without redistribution_weights, behavior should be identical to v1."""
    from attention_v3 import apply_var

    attn = _make_attn_weights(1, 4, 32, 16)
    result = apply_var(attn, sink_indices=[0], vision_end=16, p=0.5, rho=0.0)
    # Should still work (proportional redistribution)
    assert result.shape == attn.shape
    row_sum = result[0, 0, -1, :].sum().item()
    assert abs(row_sum - 1.0) < 0.01


def test_v3_context_has_redistribution_weights():
    """V3Context should have redistribution_weights field."""
    from attention_v3 import V3Context
    ctx = V3Context()
    assert hasattr(ctx, "redistribution_weights")
    assert ctx.redistribution_weights is None
