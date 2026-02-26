# tests/test_causal.py
import torch
import pytest
from contribution.causal import AttentionKnockoutHook, ValueZeroHook, compute_output_kl


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
