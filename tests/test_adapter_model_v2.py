"""Comprehensive tests for AttentionAdapterV2.

Tests cover both branches (per-head p and redistribution weights),
blend_alpha initialization, shape correctness, masking, and gradient flow.
"""

from __future__ import annotations

import math

import pytest
import torch

import config
from adapter_model import AttentionAdapterV2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def adapter():
    """Create a fresh AttentionAdapterV2 instance."""
    return AttentionAdapterV2()


@pytest.fixture
def h_last():
    """Single hidden state vector (no batch dim)."""
    torch.manual_seed(42)
    return torch.randn(4096)


@pytest.fixture
def h_last_batched():
    """Batched hidden state vectors."""
    torch.manual_seed(42)
    return torch.randn(2, 4096)


@pytest.fixture
def h_vision():
    """Vision hidden states, V=256 tokens, no batch dim."""
    torch.manual_seed(42)
    return torch.randn(256, 4096)


@pytest.fixture
def h_vision_batched():
    """Batched vision hidden states, V=256."""
    torch.manual_seed(42)
    return torch.randn(2, 256, 4096)


@pytest.fixture
def object_mask():
    """Binary object mask with ~30% object patches, no batch dim."""
    torch.manual_seed(42)
    mask = torch.zeros(256)
    mask[:77] = 1.0  # first 77 patches are "object"
    return mask


@pytest.fixture
def object_mask_batched():
    """Batched binary object mask."""
    torch.manual_seed(42)
    mask = torch.zeros(2, 256)
    mask[0, :77] = 1.0
    mask[1, :40] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Test 1: Branch 1 output shape
# ---------------------------------------------------------------------------

class TestBranch1OutputShape:
    """test_branch1_output_shape — (h_last, object_mask) -> p shape == (4, 32)"""

    def test_unbatched(self, adapter, h_last, object_mask):
        p, _ = adapter(h_last, h_vision=None, object_mask=object_mask)
        assert p.shape == (config.ADAPTER_NUM_TARGET_LAYERS, config.NUM_HEADS), (
            f"Expected shape ({config.ADAPTER_NUM_TARGET_LAYERS}, {config.NUM_HEADS}), "
            f"got {p.shape}"
        )

    def test_batched(self, adapter, h_last_batched, object_mask_batched):
        p, _ = adapter(h_last_batched, h_vision=None, object_mask=object_mask_batched)
        assert p.shape == (2, config.ADAPTER_NUM_TARGET_LAYERS, config.NUM_HEADS), (
            f"Expected shape (2, {config.ADAPTER_NUM_TARGET_LAYERS}, {config.NUM_HEADS}), "
            f"got {p.shape}"
        )


# ---------------------------------------------------------------------------
# Test 2: Branch 1 init near VAR-optimal p
# ---------------------------------------------------------------------------

class TestBranch1InitNearVarOptimal:
    """test_branch1_init — mean p ≈ 0.4 at initialization (conservative VAR start)."""

    def test_mean_p_near_target(self, adapter, h_last, object_mask):
        with torch.no_grad():
            p, _ = adapter(h_last, h_vision=None, object_mask=object_mask)
        mean_p = p.mean().item()
        assert 0.3 < mean_p < 0.5, (
            f"At initialization, mean p should be ≈ 0.4 (sigmoid(-0.405)≈0.40), "
            f"got {mean_p:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 3: Branch 2 output shape
# ---------------------------------------------------------------------------

class TestBranch2OutputShape:
    """test_branch2_output_shape — cross-attention -> redist shape == (V,)"""

    def test_unbatched(self, adapter, h_last, h_vision, object_mask):
        _, redist = adapter(h_last, h_vision=h_vision, object_mask=object_mask)
        V = h_vision.shape[0]
        assert redist is not None
        assert redist.shape == (V,), f"Expected shape ({V},), got {redist.shape}"

    def test_batched(self, adapter, h_last_batched, h_vision_batched, object_mask_batched):
        _, redist = adapter(h_last_batched, h_vision=h_vision_batched, object_mask=object_mask_batched)
        B, V = h_vision_batched.shape[:2]
        assert redist is not None
        assert redist.shape == (B, V), f"Expected shape ({B}, {V}), got {redist.shape}"


# ---------------------------------------------------------------------------
# Test 4: Branch 2 — object-only redistribution
# ---------------------------------------------------------------------------

class TestBranch2ObjectOnly:
    """test_branch2_object_only — background patches have 0 weight, object patches sum to ~1."""

    def test_background_zero(self, adapter, h_last, h_vision, object_mask):
        with torch.no_grad():
            _, redist = adapter(h_last, h_vision=h_vision, object_mask=object_mask)
        assert redist is not None

        # Background patches (mask == 0) should have zero redistribution weight
        bg_mask = object_mask == 0
        bg_weights = redist[bg_mask]
        assert torch.allclose(bg_weights, torch.zeros_like(bg_weights), atol=1e-6), (
            f"Background patches should have zero weight, got max={bg_weights.max().item():.6f}"
        )

    def test_object_patches_sum_to_one(self, adapter, h_last, h_vision, object_mask):
        with torch.no_grad():
            _, redist = adapter(h_last, h_vision=h_vision, object_mask=object_mask)
        assert redist is not None

        # Object patches should sum to approximately 1 (softmax output)
        obj_mask = object_mask == 1
        obj_sum = redist[obj_mask].sum().item()
        assert abs(obj_sum - 1.0) < 1e-4, (
            f"Object patch weights should sum to ~1.0, got {obj_sum:.6f}"
        )


# ---------------------------------------------------------------------------
# Test 5: Branch 2 returns None without h_vision
# ---------------------------------------------------------------------------

class TestBranch2NoneWithoutHVision:
    """test_branch2_none_without_h_vision — returns None if h_vision not given."""

    def test_none_when_no_vision(self, adapter, h_last, object_mask):
        _, redist = adapter(h_last, h_vision=None, object_mask=object_mask)
        assert redist is None, f"Expected None redistribution, got {type(redist)}"


# ---------------------------------------------------------------------------
# Test 6: blend_alpha initialization
# ---------------------------------------------------------------------------

class TestBlendAlphaInit:
    """test_blend_alpha_init — starts at sigmoid(ADAPTER_V2_BLEND_INIT)."""

    def test_blend_alpha_value(self, adapter):
        expected = torch.sigmoid(torch.tensor(config.ADAPTER_V2_BLEND_INIT)).item()
        actual = adapter.blend_alpha.item()
        assert abs(actual - expected) < 1e-4, (
            f"blend_alpha should start at sigmoid({config.ADAPTER_V2_BLEND_INIT})={expected:.4f}, "
            f"got {actual:.4f}"
        )

    def test_blend_alpha_near_expected(self, adapter):
        expected = torch.sigmoid(torch.tensor(config.ADAPTER_V2_BLEND_INIT)).item()
        assert abs(adapter.blend_alpha.item() - expected) < 0.005, (
            f"blend_alpha should be ~{expected:.3f}, got {adapter.blend_alpha.item():.4f}"
        )


# ---------------------------------------------------------------------------
# Test 7: Dual encoder (V=512)
# ---------------------------------------------------------------------------

class TestDualEncoder512:
    """test_dual_encoder_512 — V=512 works (Prismatic dual-encoder produces 512 vision tokens)."""

    def test_v512_shapes(self, h_last):
        V = 512
        adapter_512 = AttentionAdapterV2(vision_tokens=V)
        h_vision = torch.randn(V, 4096)
        object_mask = torch.zeros(V)
        object_mask[:200] = 1.0

        p, redist = adapter_512(h_last, h_vision=h_vision, object_mask=object_mask)

        assert p.shape == (config.ADAPTER_NUM_TARGET_LAYERS, config.NUM_HEADS)
        assert redist is not None
        assert redist.shape == (V,)

    def test_v512_batched(self, h_last_batched):
        B, V = 2, 512
        adapter_512 = AttentionAdapterV2(vision_tokens=V)
        h_vision = torch.randn(B, V, 4096)
        object_mask = torch.zeros(B, V)
        object_mask[0, :200] = 1.0
        object_mask[1, :100] = 1.0

        p, redist = adapter_512(h_last_batched, h_vision=h_vision, object_mask=object_mask)

        assert p.shape == (B, config.ADAPTER_NUM_TARGET_LAYERS, config.NUM_HEADS)
        assert redist is not None
        assert redist.shape == (B, V)


# ---------------------------------------------------------------------------
# Test 8: Gradient flows through both branches
# ---------------------------------------------------------------------------

class TestGradientFlowsBothBranches:
    """test_gradient_flows_both_branches — gradients flow through both p_head and key_proj."""

    def test_gradient_flow(self, adapter):
        h_last = torch.randn(4096, requires_grad=False)
        h_vision = torch.randn(256, 4096, requires_grad=False)
        object_mask = torch.zeros(256)
        object_mask[:77] = 1.0

        p, redist = adapter(h_last, h_vision=h_vision, object_mask=object_mask)

        # Combined loss from both branches.
        # Use entropy-like loss for redist (sum of softmax == 1, so .sum() has
        # zero grad w.r.t. scores; instead use a weighted sum that depends on
        # the *distribution* shape).
        target_weights = torch.randn(redist.shape).softmax(dim=-1)
        redist_loss = -(target_weights * (redist + 1e-10).log()).sum()
        loss = p.mean() + redist_loss
        loss.backward()

        # Branch 1: p_head MLP should have gradients
        p_head_last = adapter.p_head[-1]  # last Linear in p_head
        assert p_head_last.weight.grad is not None, "p_head should receive gradients"
        assert p_head_last.weight.grad.abs().sum().item() > 0, (
            "p_head gradients should be non-zero"
        )

        # Branch 2: key_proj should have gradients
        assert adapter.key_proj.weight.grad is not None, "key_proj should receive gradients"
        assert adapter.key_proj.weight.grad.abs().sum().item() > 0, (
            "key_proj gradients should be non-zero"
        )

        # Also check query_proj
        assert adapter.query_proj.weight.grad is not None, "query_proj should receive gradients"
        assert adapter.query_proj.weight.grad.abs().sum().item() > 0, (
            "query_proj gradients should be non-zero"
        )


# ---------------------------------------------------------------------------
# Additional utility tests
# ---------------------------------------------------------------------------

class TestUtilityMethods:
    """Tests for param_count and sparsity_stats."""

    def test_param_count_positive(self, adapter):
        count = adapter.param_count()
        assert count > 0, "param_count should be positive"
        assert isinstance(count, int)

    def test_sparsity_stats_keys(self, adapter, h_last, h_vision, object_mask):
        with torch.no_grad():
            p, redist = adapter(h_last, h_vision=h_vision, object_mask=object_mask)
        stats = adapter.sparsity_stats(p, redist)
        assert "mean_p" in stats
        assert "active_ratio" in stats
        assert "max_p" in stats
        assert "min_p" in stats
        assert "blend_alpha" in stats

    def test_sparsity_stats_blend_alpha(self, adapter, h_last, object_mask):
        with torch.no_grad():
            p, _ = adapter(h_last, h_vision=None, object_mask=object_mask)
        stats = adapter.sparsity_stats(p, None)
        assert abs(stats["blend_alpha"] - adapter.blend_alpha.item()) < 1e-6
