# tests/test_classify.py
import numpy as np
import torch
import pytest
from contribution.classify import (
    classify_token,
    classify_layer,
    TokenClassification,
    compute_mismatch,
    compute_phi_all_tokens,
)


class TestTokenClassification:
    def test_bottleneck(self):
        result = classify_token(a_share=0.6, c_share=0.85, phi=30.0)
        assert result == TokenClassification.BOTTLENECK

    def test_sink(self):
        result = classify_token(a_share=0.5, c_share=0.02, phi=25.0)
        assert result == TokenClassification.SINK

    def test_normal(self):
        result = classify_token(a_share=0.01, c_share=0.01, phi=5.0)
        assert result == TokenClassification.NORMAL

    def test_mismatch_score(self):
        a = np.array([0.8, 0.1, 0.05, 0.05])
        c = np.array([0.1, 0.3, 0.3, 0.3])
        delta = compute_mismatch(a, c)
        assert delta > 0.25

        a2 = np.array([0.8, 0.1, 0.05, 0.05])
        c2 = np.array([0.85, 0.08, 0.04, 0.03])
        delta2 = compute_mismatch(a2, c2)
        assert delta2 < 0.1


class TestLayerClassification:
    def test_classifies_all_tokens(self):
        a_tilde = np.array([0.6, 0.2, 0.1, 0.1])
        c_tilde = np.array([0.85, 0.08, 0.04, 0.03])
        boundaries = {"vision_start": 0, "vision_end": 2}
        result = classify_layer(a_tilde, c_tilde, boundaries, phi_values=None)
        assert result["dominant_type"] in ("sink", "bottleneck", "normal")
        assert "mismatch" in result
        assert "entropy" in result


class TestPhiComputation:
    def test_phi_spike_detection(self):
        seq, D = 32, 768
        torch.manual_seed(42)
        h = torch.randn(seq, D)
        h[0, :] = 0.0       # zero out row so spike dominates RMS
        h[0, 5] = 500.0     # phi = sqrt(D) ≈ 27.7 for single-spike row
        phi = compute_phi_all_tokens(h)
        assert phi.shape == (seq,)
        assert phi[0] > 20.0
        assert phi[1:].mean() < 5.0
