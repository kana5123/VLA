# tests/test_signature.py
import numpy as np
import pytest
from contribution.signature import (
    compute_skill_signature,
    compute_within_between_distance,
    run_linear_probe,
    label_skill_from_instruction,
    generate_counterfactual_instructions,
)


class TestSkillLabeling:
    def test_pick_verb(self):
        label = label_skill_from_instruction("pick up the red can from the table")
        assert label == "pick"

    def test_place_verb(self):
        label = label_skill_from_instruction("place the pot on the stove")
        assert label == "place"

    def test_move_verb(self):
        label = label_skill_from_instruction("move the block near the cup")
        assert label == "move"

    def test_open_verb(self):
        label = label_skill_from_instruction("open the top drawer")
        assert label == "open"

    def test_close_verb(self):
        label = label_skill_from_instruction("close the oven door")
        assert label == "close"


class TestSkillSignature:
    def test_signature_shape(self):
        c_tildes = [np.random.dirichlet(np.ones(256)) for _ in range(5)]
        sig = compute_skill_signature(c_tildes)
        assert sig.shape == (256,)
        assert abs(sig.sum() - 1.0) < 1e-5


class TestWithinBetweenDistance:
    def test_separable_skills(self):
        np.random.seed(42)
        sig_a = [np.zeros(64) for _ in range(20)]
        for s in sig_a:
            s[:10] = np.random.dirichlet(np.ones(10))
            s /= s.sum()

        sig_b = [np.zeros(64) for _ in range(20)]
        for s in sig_b:
            s[50:60] = np.random.dirichlet(np.ones(10))
            s /= s.sum()

        labels = ["A"] * 20 + ["B"] * 20
        signatures = sig_a + sig_b

        d_within, d_between = compute_within_between_distance(signatures, labels)
        assert d_within < d_between


class TestLinearProbe:
    def test_separable_data_high_accuracy(self):
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(50, 64) + np.array([1]*32 + [0]*32),
            np.random.randn(50, 64) + np.array([0]*32 + [1]*32),
        ])
        y = np.array([0]*50 + [1]*50)
        acc = run_linear_probe(X, y)
        assert acc > 0.8


class TestCounterfactual:
    def test_generates_swap_pairs(self):
        pairs = generate_counterfactual_instructions("pick up the red can")
        assert len(pairs) >= 1
        assert pairs[0][0] == "push"
        assert "push" in pairs[0][1]

    def test_no_swap_for_unknown_verb(self):
        pairs = generate_counterfactual_instructions("prepare the salad")
        assert len(pairs) == 0
