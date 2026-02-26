import torch
import numpy as np
import pytest

from contribution.compute import (
    compute_perhead_contribution,
    aggregate_contributions,
    find_topk_candidates,
    ContributionResult,
)


class TestPerHeadContribution:
    def test_shapes(self):
        H, seq, D = 4, 32, 64
        attn = torch.rand(H, seq, seq)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        hidden = torch.randn(seq, D)
        w_v = torch.randn(D, D)
        w_o = torch.randn(D, D)
        query_positions = [28, 29, 30, 31]
        result = compute_perhead_contribution(attn, hidden, w_v, w_o, query_positions)
        assert result.shape == (H, len(query_positions), seq)
        assert (result >= 0).all()

    def test_zero_attention_zero_contribution(self):
        H, seq, D = 2, 16, 32
        attn = torch.zeros(H, seq, seq)
        attn[:, :, 0] = 1.0
        hidden = torch.randn(seq, D)
        w_v = torch.randn(D, D)
        w_o = torch.randn(D, D)
        result = compute_perhead_contribution(attn, hidden, w_v, w_o, [8])
        assert (result[:, :, 1:] == 0).all()
        assert (result[:, :, 0] > 0).all()

    def test_contribution_proportional_to_attention(self):
        H, seq, D = 1, 8, 16
        hidden = torch.randn(seq, D)
        w_v = torch.randn(D, D)
        w_o = torch.randn(D, D)
        attn1 = torch.zeros(H, seq, seq)
        attn1[0, 4, 2] = 0.3
        attn1[0, 4, 0] = 0.7
        attn2 = torch.zeros(H, seq, seq)
        attn2[0, 4, 2] = 0.6
        attn2[0, 4, 0] = 0.4
        r1 = compute_perhead_contribution(attn1, hidden, w_v, w_o, [4])
        r2 = compute_perhead_contribution(attn2, hidden, w_v, w_o, [4])
        ratio_attn = 0.6 / 0.3
        ratio_contrib = r2[0, 0, 2] / r1[0, 0, 2]
        assert abs(ratio_contrib - ratio_attn) < 1e-4


class TestAggregateContributions:
    def test_output_shape(self):
        H, n_query, seq = 4, 3, 32
        attn = torch.rand(H, seq, seq)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        contrib = torch.rand(H, n_query, seq)
        query_positions = list(range(29, 32))
        a_j, c_j = aggregate_contributions(attn, contrib, query_positions)
        assert a_j.shape == (seq,)
        assert c_j.shape == (seq,)

    def test_normalized_to_distribution(self):
        H, n_query, seq = 4, 3, 32
        attn = torch.rand(H, seq, seq)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        contrib = torch.rand(H, n_query, seq) + 0.01
        a_j, c_j = aggregate_contributions(attn, contrib, list(range(29, 32)))
        assert abs(a_j.sum().item() - 1.0) < 1e-5
        assert abs(c_j.sum().item() - 1.0) < 1e-5


class TestTopKCandidates:
    def test_topk_returns_correct_count(self):
        c_j = torch.zeros(256)
        c_j[0] = 0.8
        c_j[100] = 0.15
        c_j[200] = 0.05
        candidates = find_topk_candidates(c_j, k=3)
        assert len(candidates) == 3
        assert candidates[0]["position"] == 0
        assert candidates[0]["share"] == pytest.approx(0.8, abs=1e-5)

    def test_topk_includes_type_classification(self):
        c_j = torch.zeros(300)
        c_j[0] = 0.5
        c_j[10] = 0.3
        c_j[260] = 0.2
        boundaries = {"vision_start": 5, "vision_end": 256}
        candidates = find_topk_candidates(c_j, k=3, boundaries=boundaries)
        types = {c["type"] for c in candidates}
        assert "pre_vision" in types
        assert "vision" in types
        assert "text" in types


class TestExtractSampleContributions:
    def test_returns_all_layers(self):
        from contribution.compute import extract_sample_contributions
        n_layers, H, seq, D = 4, 2, 32, 64
        attention_weights = {i: torch.rand(H, seq, seq) for i in range(n_layers)}
        for i in attention_weights:
            attention_weights[i] = attention_weights[i] / attention_weights[i].sum(dim=-1, keepdim=True)
        hidden_states = {i: torch.randn(seq, D) for i in range(n_layers)}

        def mock_get_wov(model, cfg, layer_idx):
            return torch.randn(D, D), torch.randn(D, D)

        boundaries = {"vision_start": 0, "vision_end": 20, "text_start": 20, "text_end": 32}
        query_positions = [28, 29, 30, 31]

        results = extract_sample_contributions(
            attention_weights=attention_weights,
            hidden_states=hidden_states,
            get_wov_fn=mock_get_wov,
            model=None,
            model_cfg=None,
            boundaries=boundaries,
            query_positions=query_positions,
            top_k=5,
        )

        assert len(results) == n_layers
        for r in results:
            assert isinstance(r, ContributionResult)
            assert r.a_tilde.shape == (seq,)
            assert abs(r.a_tilde.sum() - 1.0) < 1e-4


class TestFrequencyAnalysis:
    def test_frequency_counts(self):
        from contribution.compute import compute_candidate_frequency
        all_topk = [
            [{"position": 0, "share": 0.8}, {"position": 5, "share": 0.1}],
            [{"position": 0, "share": 0.7}, {"position": 10, "share": 0.15}],
            [{"position": 0, "share": 0.9}, {"position": 5, "share": 0.05}],
        ]
        freq = compute_candidate_frequency(all_topk, seq_len=32)
        assert freq[0] == 3 / 3
        assert freq[5] == 2 / 3
        assert freq[10] == 1 / 3
