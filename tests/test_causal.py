# tests/test_causal.py
import torch
import pytest
from contribution.causal import AttentionKnockoutHook, ValueZeroHook, compute_output_kl, get_deep_layer_ranges


class TestAttentionKnockout:
    def test_knockout_zeros_attention_to_target(self):
        H, seq = 4, 32
        target_positions = [0, 1]
        query_range = (28, 32)
        hook = AttentionKnockoutHook(target_positions, query_range)
        scores = torch.randn(1, H, seq, seq)
        masked = hook.apply_mask(scores)
        for q in range(28, 32):
            for t in target_positions:
                assert masked[0, :, q, t].max() < -1e8
        assert torch.equal(scores[0, :, 0, :], masked[0, :, 0, :])


class TestValueZero:
    def test_value_zeroed(self):
        H, seq, D = 4, 32, 64
        target_positions = [0, 1]
        hook = ValueZeroHook(target_positions)
        values = torch.randn(1, H, seq, D // H)
        zeroed = hook.apply(values)
        for t in target_positions:
            assert zeroed[0, :, t, :].abs().max() < 1e-10
        assert torch.equal(values[0, :, 5, :], zeroed[0, :, 5, :])


class TestGetDeepLayerRanges:
    def test_32L(self):
        ranges = get_deep_layer_ranges(32)
        assert ranges["all"] == list(range(22, 32))
        assert ranges["block1"] == list(range(22, 27))
        assert ranges["block2"] == list(range(27, 32))

    def test_26L(self):
        ranges = get_deep_layer_ranges(26)
        assert ranges["all"] == list(range(16, 26))
        assert ranges["block1"] == list(range(16, 21))
        assert ranges["block2"] == list(range(21, 26))


class TestValueZeroTargetLayers:
    def test_target_layers_none(self):
        """target_layers=None should behave same as before (all layers)."""
        hook = ValueZeroHook([0, 1], target_layers=None)
        assert hook.target_layers is None

    def test_target_layers_list(self):
        hook = ValueZeroHook([0, 1], target_layers=[22, 23, 24])
        assert hook.target_layers == [22, 23, 24]


class TestCausalExperiment:
    def test_kl_positive_when_bottleneck_masked(self):
        logits_orig = torch.zeros(256)
        logits_orig[42] = 10.0
        logits_masked = torch.zeros(256)
        kl = compute_output_kl(logits_orig, logits_masked)
        assert kl > 1.0

    def test_kl_small_when_sink_masked(self):
        logits_orig = torch.zeros(256)
        logits_orig[42] = 10.0
        logits_masked = torch.zeros(256)
        logits_masked[42] = 9.8
        kl = compute_output_kl(logits_orig, logits_masked)
        assert kl < 0.1
