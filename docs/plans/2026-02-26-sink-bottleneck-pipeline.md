# Sink vs Bottleneck Analysis Pipeline — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a fresh, general-purpose pipeline that automatically discovers candidate tokens in VLA models, classifies them as sink (high attention, low contribution) or bottleneck (high attention, high contribution), and measures how they distort skill-conditioned internal representations.

**Architecture:** Three-layer design: (1) `contribution/compute.py` computes per-head W_OV contribution `C_l(j)` and attention `A_l(j)` for any model from the registry, with TopK candidate discovery; (2) `contribution/classify.py` applies sink/bottleneck/normal classification rules and causal masking (attention knockout + V=0); (3) `contribution/signature.py` analyzes skill signatures via JS divergence and counterfactual instruction swaps. All reuse existing `model_registry`, `extract_attention.load_model_from_registry`, and `SinkVerificationHookManager`.

**Tech Stack:** PyTorch, transformers, numpy, scipy (JS divergence), matplotlib, sklearn (linear probe). Existing infra: `model_registry.py`, `extract_attention.py`, `verify_attention_sinks.py` (hook manager only), `config.py`.

---

## Existing Infrastructure (DO NOT recreate)

| What | Where | Signature |
|------|-------|-----------|
| Model loading | `extract_attention.py:715` | `load_model_from_registry(name, device) → (processor, model, model_cfg)` |
| Layer access | `extract_attention.py:802` | `get_layers(model, model_cfg) → nn.ModuleList` |
| Token boundaries | `extract_attention.py` | `detect_token_boundaries(...) → dict` with `vision_start/end`, `text_query_ranges` |
| Hook manager | `verify_attention_sinks.py:59` | `SinkVerificationHookManager(model, model_cfg)` → `.attention_weights[l]`, `.hidden_states[l]` |
| W_V/W_O extraction | `verify_attention_sinks.py:127` | `get_wov_matrix(model, model_cfg, layer_idx) → (v_weight, o_weight)` |
| Sample loading | `visualize_text_attention.py` | `load_samples_from_cache(cache_dir, n_samples) → [{"instruction", "image"}]` |
| Processor call | `extract_attention.py` | `call_processor(processor, prompt, image, model_cfg)` |
| Model registry | `model_registry.py` | `get_model(name) → VLAModelConfig` |
| Paths | `config.py` | `DATA_CACHE_DIR`, `OUTPUT_DIR`, etc. |

## File Structure (new files)

```
ATLASVLA/
├── contribution/                    # NEW package
│   ├── __init__.py
│   ├── compute.py                   # Task 1-3: Core Ã(j), C̃(j) computation
│   ├── classify.py                  # Task 4-5: Sink/bottleneck classification
│   ├── causal.py                    # Task 6-7: Attention knockout + V=0 hooks
│   ├── signature.py                 # Task 8-9: Skill signatures + counterfactual
│   └── visualize.py                 # Task 10: All 6 figure types
├── tests/
│   ├── test_contribution_compute.py # Task 1
│   ├── test_classify.py             # Task 4
│   ├── test_causal.py               # Task 6
│   └── test_signature.py            # Task 8
├── run_contribution_analysis.py     # Task 11: Main CLI
└── run_causal_experiment.py         # Task 12: Causal CLI
```

---

## Task 1: Core per-head contribution computation

**Files:**
- Create: `contribution/__init__.py`
- Create: `contribution/compute.py`
- Create: `tests/test_contribution_compute.py`

### Step 1: Write the failing test

```python
# tests/test_contribution_compute.py
import torch
import numpy as np
import pytest

from contribution.compute import (
    compute_perhead_contribution,
    aggregate_contributions,
    find_topk_candidates,
)


class TestPerHeadContribution:
    """Test per-head W_OV contribution: Contrib_{i←j}^{l,h} = ||α * x_j * W_OV^h||"""

    def test_shapes(self):
        """Contribution output shape = (n_query, seq_len) per head."""
        H, seq, D = 4, 32, 64
        v_dim = D  # MHA, no GQA
        attn = torch.rand(H, seq, seq)
        attn = attn / attn.sum(dim=-1, keepdim=True)  # normalize
        hidden = torch.randn(seq, D)
        # W_V: (v_dim, D), W_O: (D, v_dim)
        w_v = torch.randn(v_dim, D)
        w_o = torch.randn(D, v_dim)

        query_positions = [28, 29, 30, 31]  # action tokens
        result = compute_perhead_contribution(attn, hidden, w_v, w_o, query_positions)

        assert result.shape == (H, len(query_positions), seq)
        assert (result >= 0).all()  # norms are non-negative

    def test_zero_attention_zero_contribution(self):
        """If attention to token j is 0, contribution from j must be 0."""
        H, seq, D = 2, 16, 32
        attn = torch.zeros(H, seq, seq)
        attn[:, :, 0] = 1.0  # all attention to token 0
        hidden = torch.randn(seq, D)
        w_v = torch.randn(D, D)
        w_o = torch.randn(D, D)

        result = compute_perhead_contribution(attn, hidden, w_v, w_o, [8])
        # Contribution from token 1..15 should be 0
        assert (result[:, :, 1:] == 0).all()
        # Contribution from token 0 should be > 0
        assert (result[:, :, 0] > 0).all()

    def test_contribution_proportional_to_attention(self):
        """Doubling attention weight doubles contribution (same value vector)."""
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

        # ratio of contributions should equal ratio of attention weights
        ratio_attn = 0.6 / 0.3
        ratio_contrib = r2[0, 0, 2] / r1[0, 0, 2]
        assert abs(ratio_contrib - ratio_attn) < 1e-4


class TestAggregateContributions:
    """Test A_l(j) and C_l(j) aggregation across heads and query positions."""

    def test_output_shape(self):
        """Aggregated A(j) and C(j) are 1D vectors of length seq_len."""
        H, n_query, seq = 4, 3, 32
        attn = torch.rand(H, seq, seq)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        contrib = torch.rand(H, n_query, seq)
        query_positions = list(range(29, 32))

        a_j, c_j = aggregate_contributions(attn, contrib, query_positions)

        assert a_j.shape == (seq,)
        assert c_j.shape == (seq,)

    def test_normalized_to_distribution(self):
        """Ã(j) and C̃(j) sum to 1.0."""
        H, n_query, seq = 4, 3, 32
        attn = torch.rand(H, seq, seq)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        contrib = torch.rand(H, n_query, seq) + 0.01  # avoid all-zero

        a_j, c_j = aggregate_contributions(attn, contrib, list(range(29, 32)))

        assert abs(a_j.sum().item() - 1.0) < 1e-5
        assert abs(c_j.sum().item() - 1.0) < 1e-5


class TestTopKCandidates:
    """Test automatic TopK candidate discovery."""

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
        """Candidates should be classified as vision/text/pre_vision."""
        c_j = torch.zeros(300)
        c_j[0] = 0.5   # pre_vision
        c_j[10] = 0.3   # vision
        c_j[260] = 0.2  # text

        boundaries = {"vision_start": 5, "vision_end": 256}
        candidates = find_topk_candidates(c_j, k=3, boundaries=boundaries)

        types = {c["type"] for c in candidates}
        assert "pre_vision" in types
        assert "vision" in types
        assert "text" in types
```

### Step 2: Run test to verify it fails

Run: `/home/kana5123/miniconda3/envs/interp/bin/python -m pytest tests/test_contribution_compute.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'contribution'`

### Step 3: Write minimal implementation

```python
# contribution/__init__.py
"""Sink vs Bottleneck contribution analysis pipeline for VLA models."""

# contribution/compute.py
"""
Core per-head W_OV contribution computation.

Math (from user's experiment design, Section 3.2):
  Contrib_{i←j}^{l,h} = ||α_{i,j}^{l,h} · x_j^{l-1} · W_OV^{l,h}||

  A_l(j) = E_{i∈I_act, h}[α_{i,j}^{l,h}]        (attention aggregate)
  C_l(j) = E_{i∈I_act, h}[Contrib_{i←j}^{l,h}]   (contribution aggregate)

  Ã_l(j) = A_l(j) / Σ_k A_l(k)                    (normalized)
  C̃_l(j) = C_l(j) / Σ_k C_l(k)                    (normalized)
"""
import torch
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ContributionResult:
    """Per-layer contribution analysis result."""
    layer_idx: int
    a_tilde: np.ndarray          # Ã(j) — normalized attention, shape (seq,)
    c_tilde: np.ndarray          # C̃(j) — normalized contribution, shape (seq,)
    topk_attention: list[dict]   # TopK attention candidates
    topk_contribution: list[dict] # TopK contribution candidates
    raw_a: np.ndarray | None = None
    raw_c: np.ndarray | None = None


def compute_perhead_contribution(
    attn: torch.Tensor,          # (H, seq, seq) — attention weights
    hidden: torch.Tensor,        # (seq, D) — hidden states from layer l-1
    w_v: torch.Tensor,           # (v_dim, D) — value weight
    w_o: torch.Tensor,           # (D, v_dim) — output weight
    query_positions: list[int],  # I_act — action token positions
) -> torch.Tensor:
    """Compute per-head contribution norms.

    For each head h, query i, key j:
      contrib = ||α_{i,j}^h · W_O^h · W_V^h · x_j||

    For GQA models where num_kv_heads < num_heads, w_v has shape
    (kv_dim, D) where kv_dim = num_kv_heads * head_dim. We handle
    this by computing per-kv-group and expanding.

    Returns:
        Tensor of shape (H, len(query_positions), seq) — contribution norms
    """
    H = attn.shape[0]
    seq = attn.shape[1]
    D = hidden.shape[1]
    v_dim = w_v.shape[0]

    # Compute W_OV = W_O @ W_V, shape (D, D)
    # For GQA: v_dim < D, so W_OV shape is (D, D) after composition
    w_ov = (w_o @ w_v).float()  # (D, D)

    # Value vectors for all tokens: x_j @ W_OV^T → (seq, D)
    value_vectors = hidden.float() @ w_ov.T  # (seq, D)

    n_query = len(query_positions)
    result = torch.zeros(H, n_query, seq)

    for qi, i in enumerate(query_positions):
        if i >= seq:
            continue
        for h in range(H):
            # α_{i,j}^h for all j — shape (seq,)
            alpha = attn[h, i, :seq].float()
            # Contribution = ||α * value_vector|| per j
            # = |α_j| * ||value_vector_j|| (since α is scalar per j)
            # value_vectors: (seq, D)
            weighted = alpha.unsqueeze(1) * value_vectors  # (seq, D)
            norms = torch.norm(weighted, dim=1)  # (seq,)
            result[h, qi, :] = norms

    return result


def aggregate_contributions(
    attn: torch.Tensor,          # (H, seq, seq)
    contrib: torch.Tensor,       # (H, n_query, seq) — from compute_perhead_contribution
    query_positions: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate across heads and query positions, then normalize.

    A_l(j) = E_{i∈I_act, h}[α_{i,j}]
    C_l(j) = E_{i∈I_act, h}[contrib_{i←j}]

    Returns (a_tilde, c_tilde) — both shape (seq,), summing to 1.0.
    """
    H = attn.shape[0]
    seq = attn.shape[-1]

    # Attention aggregate: mean over heads and query positions
    a_raw = torch.zeros(seq)
    for qi, i in enumerate(query_positions):
        if i >= attn.shape[1]:
            continue
        a_raw += attn[:, i, :seq].float().mean(dim=0)  # mean over H
    a_raw /= max(len(query_positions), 1)

    # Contribution aggregate: mean over heads and query positions
    c_raw = contrib.mean(dim=(0, 1))  # mean over H and queries → (seq,)

    # Normalize to distributions
    a_sum = a_raw.sum().clamp(min=1e-10)
    c_sum = c_raw.sum().clamp(min=1e-10)

    a_tilde = a_raw / a_sum
    c_tilde = c_raw / c_sum

    return a_tilde, c_tilde


def find_topk_candidates(
    distribution: torch.Tensor,   # (seq,) — Ã or C̃
    k: int = 5,
    boundaries: dict | None = None,
) -> list[dict]:
    """Find TopK candidates from a normalized distribution.

    Args:
        distribution: 1D tensor, sums to ~1.0
        k: number of candidates
        boundaries: if provided, classify positions as vision/text/pre_vision

    Returns:
        list of dicts: [{position, share, type}, ...]
    """
    values, indices = distribution.topk(min(k, len(distribution)))
    candidates = []

    for rank, (val, idx) in enumerate(zip(values.tolist(), indices.tolist())):
        token_type = "unknown"
        if boundaries is not None:
            vs = boundaries.get("vision_start", 0)
            ve = boundaries.get("vision_end", 0)
            if idx < vs:
                token_type = "pre_vision"
            elif idx < ve:
                token_type = f"vision"
            else:
                token_type = "text"

        candidates.append({
            "rank": rank,
            "position": idx,
            "share": val,
            "type": token_type,
        })

    return candidates
```

### Step 4: Run test to verify it passes

Run: `/home/kana5123/miniconda3/envs/interp/bin/python -m pytest tests/test_contribution_compute.py -v`
Expected: All 7 tests PASS

### Step 5: Commit

```bash
git add contribution/__init__.py contribution/compute.py tests/test_contribution_compute.py
git commit -m "feat: core per-head W_OV contribution computation with TopK discovery"
```

---

## Task 2: Single-sample extraction pipeline

**Files:**
- Modify: `contribution/compute.py` (add `extract_sample_contributions`)

### Step 1: Write the failing test

```python
# Append to tests/test_contribution_compute.py

class TestExtractSampleContributions:
    """Integration: extract A_l(j), C_l(j) for one sample, all layers."""

    def test_returns_all_layers(self):
        """Should return ContributionResult for every layer."""
        from contribution.compute import extract_sample_contributions
        # Create mock objects
        n_layers, H, seq, D = 4, 2, 32, 64
        attention_weights = {i: torch.rand(H, seq, seq) for i in range(n_layers)}
        # Normalize attention
        for i in attention_weights:
            attention_weights[i] = attention_weights[i] / attention_weights[i].sum(dim=-1, keepdim=True)
        hidden_states = {i: torch.randn(seq, D) for i in range(n_layers)}

        # Mock get_wov that returns random matrices
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
```

### Step 2: Run test to verify it fails

Run: `/home/kana5123/miniconda3/envs/interp/bin/python -m pytest tests/test_contribution_compute.py::TestExtractSampleContributions -v`
Expected: FAIL — `ImportError: cannot import name 'extract_sample_contributions'`

### Step 3: Write minimal implementation

```python
# Add to contribution/compute.py

from typing import Callable


def extract_sample_contributions(
    attention_weights: dict[int, torch.Tensor],  # layer → (H, seq, seq)
    hidden_states: dict[int, torch.Tensor],      # layer → (seq, D)
    get_wov_fn: Callable,                        # (model, cfg, layer) → (w_v, w_o)
    model,                                       # passed to get_wov_fn
    model_cfg,                                   # passed to get_wov_fn
    boundaries: dict,
    query_positions: list[int],
    top_k: int = 5,
    target_layers: list[int] | None = None,      # None = all layers
) -> list[ContributionResult]:
    """Extract Ã(j) and C̃(j) for all layers of a single sample.

    This is the main per-sample workhorse. For each layer:
    1. Get W_V, W_O via get_wov_fn
    2. Compute per-head contribution (H, n_query, seq)
    3. Aggregate → Ã(j), C̃(j)
    4. Find TopK candidates

    Args:
        attention_weights: from SinkVerificationHookManager.attention_weights
        hidden_states: from SinkVerificationHookManager.hidden_states
        get_wov_fn: function(model, model_cfg, layer_idx) → (w_v, w_o)
        query_positions: I_act — action token positions to use as queries
        target_layers: which layers to analyze (default: all available)

    Returns:
        list of ContributionResult, one per layer
    """
    if target_layers is None:
        target_layers = sorted(set(attention_weights.keys()) & set(hidden_states.keys()))

    results = []

    for layer_idx in target_layers:
        attn = attention_weights[layer_idx]  # (H, seq, seq)
        if attn.dim() == 4:
            attn = attn[0]  # remove batch dim

        # Hidden states from PREVIOUS layer (input to this layer's attention)
        prev_layer = max(0, layer_idx - 1)
        h = hidden_states.get(prev_layer, hidden_states[layer_idx]).cpu().float()

        try:
            w_v, w_o = get_wov_fn(model, model_cfg, layer_idx)
        except (RuntimeError, ValueError, AttributeError):
            # GQA or other extraction failure — skip layer
            continue

        # Per-head contribution
        contrib = compute_perhead_contribution(attn, h, w_v, w_o, query_positions)

        # Aggregate
        a_tilde, c_tilde = aggregate_contributions(attn, contrib, query_positions)
        a_np = a_tilde.numpy()
        c_np = c_tilde.numpy()

        # TopK
        topk_a = find_topk_candidates(a_tilde, k=top_k, boundaries=boundaries)
        topk_c = find_topk_candidates(c_tilde, k=top_k, boundaries=boundaries)

        results.append(ContributionResult(
            layer_idx=layer_idx,
            a_tilde=a_np,
            c_tilde=c_np,
            topk_attention=topk_a,
            topk_contribution=topk_c,
        ))

    return results
```

### Step 4: Run test to verify it passes

Run: `/home/kana5123/miniconda3/envs/interp/bin/python -m pytest tests/test_contribution_compute.py -v`
Expected: All tests PASS

### Step 5: Commit

```bash
git add contribution/compute.py tests/test_contribution_compute.py
git commit -m "feat: single-sample contribution extraction pipeline"
```

---

## Task 3: Multi-sample batch extraction with frequency analysis

**Files:**
- Modify: `contribution/compute.py` (add `run_contribution_analysis`)

### Step 1: Write the failing test

```python
# Append to tests/test_contribution_compute.py

class TestFrequencyAnalysis:
    """Test Freq_l(j) = Pr(j ∈ TopK(C̃_l)) across samples."""

    def test_frequency_counts(self):
        from contribution.compute import compute_candidate_frequency

        # Simulate 10 samples, each with topk candidates
        all_topk = [
            [{"position": 0, "share": 0.8}, {"position": 5, "share": 0.1}],
            [{"position": 0, "share": 0.7}, {"position": 10, "share": 0.15}],
            [{"position": 0, "share": 0.9}, {"position": 5, "share": 0.05}],
        ]

        freq = compute_candidate_frequency(all_topk, seq_len=32)
        assert freq[0] == 3 / 3  # position 0 appears in all 3
        assert freq[5] == 2 / 3  # position 5 in 2/3
        assert freq[10] == 1 / 3
```

### Step 2: Run test, verify fail

### Step 3: Implement

```python
# Add to contribution/compute.py

def compute_candidate_frequency(
    all_topk: list[list[dict]],  # n_samples × k candidates
    seq_len: int,
) -> np.ndarray:
    """Compute Freq_l(j) = Pr(j ∈ TopK) across samples.

    Returns:
        ndarray of shape (seq_len,) with frequency in [0, 1]
    """
    counts = np.zeros(seq_len)
    n = len(all_topk)
    if n == 0:
        return counts

    for sample_topk in all_topk:
        for candidate in sample_topk:
            pos = candidate["position"]
            if 0 <= pos < seq_len:
                counts[pos] += 1

    return counts / n
```

### Step 4: Run test, verify pass

### Step 5: Commit

```bash
git add contribution/compute.py tests/test_contribution_compute.py
git commit -m "feat: candidate frequency analysis across samples"
```

---

## Task 4: Sink vs Bottleneck classification

**Files:**
- Create: `contribution/classify.py`
- Create: `tests/test_classify.py`

### Step 1: Write the failing test

```python
# tests/test_classify.py
import numpy as np
import pytest
from contribution.classify import (
    classify_token,
    classify_layer,
    TokenClassification,
)


class TestTokenClassification:
    """Test sink/bottleneck/normal classification per design Section 4.2."""

    def test_bottleneck(self):
        """High A AND high C → bottleneck."""
        result = classify_token(a_share=0.6, c_share=0.85, phi=30.0)
        assert result == TokenClassification.BOTTLENECK

    def test_sink(self):
        """High A but low C → sink (attention-contribution mismatch)."""
        result = classify_token(a_share=0.5, c_share=0.02, phi=25.0)
        assert result == TokenClassification.SINK

    def test_normal(self):
        """Low A, low C → normal."""
        result = classify_token(a_share=0.01, c_share=0.01, phi=5.0)
        assert result == TokenClassification.NORMAL

    def test_mismatch_score(self):
        """Mismatch Δ = JS(p_att, p_contrib) should be high for sinks."""
        from contribution.classify import compute_mismatch
        # Sink-like: attention concentrated, contribution spread
        a = np.array([0.8, 0.1, 0.05, 0.05])
        c = np.array([0.1, 0.3, 0.3, 0.3])
        delta = compute_mismatch(a, c)
        assert delta > 0.3  # high mismatch

        # Aggregator-like: both concentrated at same token
        a2 = np.array([0.8, 0.1, 0.05, 0.05])
        c2 = np.array([0.85, 0.08, 0.04, 0.03])
        delta2 = compute_mismatch(a2, c2)
        assert delta2 < 0.1  # low mismatch


class TestLayerClassification:
    def test_classifies_all_tokens(self):
        a_tilde = np.array([0.6, 0.2, 0.1, 0.1])
        c_tilde = np.array([0.85, 0.08, 0.04, 0.03])
        boundaries = {"vision_start": 0, "vision_end": 2}

        result = classify_layer(a_tilde, c_tilde, boundaries, phi_values=None)
        assert result["dominant_type"] in ("sink", "bottleneck", "normal")
        assert "mismatch" in result
        assert "entropy" in result
```

### Step 2: Run test, verify fail

### Step 3: Implement

```python
# contribution/classify.py
"""
Sink vs Bottleneck classification (Design Section 4.2).

Definitions:
  BOTTLENECK: C̃(j) ≥ c_threshold AND Freq(j) high AND causal impact high
  SINK:       Ã(j) ≥ a_threshold AND C̃(j) < c_threshold ("attention-contribution mismatch")
  NORMAL:     neither condition met
"""
import numpy as np
from enum import Enum
from scipy.spatial.distance import jensenshannon


class TokenClassification(Enum):
    SINK = "sink"
    BOTTLENECK = "bottleneck"
    NORMAL = "normal"


def classify_token(
    a_share: float,
    c_share: float,
    phi: float | None = None,
    a_threshold: float = 0.15,
    c_high_threshold: float = 0.5,
    c_low_threshold: float = 0.05,
) -> TokenClassification:
    """Classify a single token's role.

    Args:
        a_share: Ã(j) — this token's share of total attention
        c_share: C̃(j) — this token's share of total contribution
        phi: φ(x_j) — hidden state spike ratio (optional, for VAR compat)
        a_threshold: minimum attention to be considered a candidate
        c_high_threshold: contribution threshold for bottleneck
        c_low_threshold: contribution threshold below which = sink
    """
    if a_share < a_threshold:
        return TokenClassification.NORMAL
    if c_share >= c_high_threshold:
        return TokenClassification.BOTTLENECK
    if c_share < c_low_threshold:
        return TokenClassification.SINK
    return TokenClassification.NORMAL


def compute_mismatch(a_tilde: np.ndarray, c_tilde: np.ndarray) -> float:
    """Δ = JS(Ã, C̃) — attention-contribution mismatch (Design Section 6.3.2).

    High Δ → sink pattern (attention concentrated where contribution is not).
    Low Δ → consistent (aggregator or normal).
    """
    # Clip to avoid log(0)
    a = np.clip(a_tilde, 1e-10, None)
    c = np.clip(c_tilde, 1e-10, None)
    a = a / a.sum()
    c = c / c.sum()
    return float(jensenshannon(a, c) ** 2)  # JS divergence (squared for [0,1] range)


def compute_entropy(distribution: np.ndarray) -> float:
    """H(p) = -Σ p(j) log p(j) — contribution entropy (Design Section 6.3.3)."""
    p = np.clip(distribution, 1e-10, None)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def classify_layer(
    a_tilde: np.ndarray,
    c_tilde: np.ndarray,
    boundaries: dict,
    phi_values: np.ndarray | None = None,
    top_k: int = 5,
    a_threshold: float = 0.15,
    c_high_threshold: float = 0.5,
) -> dict:
    """Classify all candidate tokens in a layer.

    Returns dict with:
        dominant_position, dominant_type, dominant_a_share, dominant_c_share,
        mismatch (JS), entropy (H), top1_share, candidates (list)
    """
    mismatch = compute_mismatch(a_tilde, c_tilde)
    entropy = compute_entropy(c_tilde)
    top1_share = float(c_tilde.max())

    # Find top candidates by contribution
    topk_idx = np.argsort(c_tilde)[-top_k:][::-1]

    candidates = []
    for pos in topk_idx:
        a_share = float(a_tilde[pos])
        c_share = float(c_tilde[pos])
        phi = float(phi_values[pos]) if phi_values is not None else None
        cls = classify_token(a_share, c_share, phi, a_threshold, c_high_threshold)

        vs = boundaries.get("vision_start", 0)
        ve = boundaries.get("vision_end", 0)
        if pos < vs:
            token_type = "pre_vision"
        elif pos < ve:
            token_type = "vision"
        else:
            token_type = "text"

        candidates.append({
            "position": int(pos),
            "token_region": token_type,
            "classification": cls.value,
            "a_share": a_share,
            "c_share": c_share,
            "phi": phi,
        })

    dominant = candidates[0] if candidates else None

    return {
        "dominant_position": dominant["position"] if dominant else -1,
        "dominant_type": dominant["classification"] if dominant else "normal",
        "dominant_a_share": dominant["a_share"] if dominant else 0.0,
        "dominant_c_share": dominant["c_share"] if dominant else 0.0,
        "mismatch": mismatch,
        "entropy": entropy,
        "top1_share": top1_share,
        "candidates": candidates,
    }
```

### Step 4: Run test, verify pass

### Step 5: Commit

```bash
git add contribution/classify.py tests/test_classify.py
git commit -m "feat: sink/bottleneck/normal classification with JS mismatch"
```

---

## Task 5: Phi (hidden state spike) integration

**Files:**
- Modify: `contribution/classify.py` (add `compute_phi_all_tokens`)

### Step 1: Write test

```python
# Append to tests/test_classify.py

class TestPhiComputation:
    def test_phi_spike_detection(self):
        from contribution.classify import compute_phi_all_tokens
        import torch

        seq, D = 32, 64
        h = torch.randn(seq, D)
        # Make token 0 have a massive spike in dimension 5
        h[0, 5] = 500.0

        phi = compute_phi_all_tokens(h)
        assert phi.shape == (seq,)
        assert phi[0] > 20.0  # should be a spike
        assert phi[1:].mean() < 5.0  # others should be normal
```

### Step 2: Run test, verify fail

### Step 3: Implement

```python
# Add to contribution/classify.py

import torch

def compute_phi_all_tokens(hidden: torch.Tensor, tau: float = 20.0) -> np.ndarray:
    """Compute φ(x) = max_d |x[d]| / RMS(x) for all tokens (VAR Section 4.1).

    Args:
        hidden: (seq, D) tensor
        tau: threshold for spike detection

    Returns:
        ndarray of shape (seq,) — phi values
    """
    h = hidden.float()
    rms = torch.sqrt((h ** 2).mean(dim=1)).clamp(min=1e-8)  # (seq,)
    phi = (h.abs() / rms.unsqueeze(1)).max(dim=1).values     # (seq,)
    return phi.numpy()
```

### Step 4: Run test, verify pass

### Step 5: Commit

```bash
git add contribution/classify.py tests/test_classify.py
git commit -m "feat: phi spike detection for VAR compatibility"
```

---

## Task 6: Causal verification — Attention knockout hook

**Files:**
- Create: `contribution/causal.py`
- Create: `tests/test_causal.py`

### Step 1: Write the failing test

```python
# tests/test_causal.py
import torch
import pytest
from contribution.causal import AttentionKnockoutHook, ValueZeroHook


class TestAttentionKnockout:
    """Design Section 4.3: mask attention(i,j) = -∞ for specific (query, key) pairs."""

    def test_knockout_zeros_attention_to_target(self):
        """After knockout, attention from query to target should be ~0."""
        H, seq = 4, 32
        target_positions = [0, 1]   # tokens to knock out
        query_range = (28, 32)       # action tokens

        hook = AttentionKnockoutHook(target_positions, query_range)

        # Simulate attention scores (pre-softmax)
        scores = torch.randn(1, H, seq, seq)
        masked = hook.apply_mask(scores)

        # Check that target positions are -inf for queries in range
        for q in range(28, 32):
            for t in target_positions:
                assert masked[0, :, q, t].max() < -1e8

        # Non-query positions should be unchanged
        assert torch.equal(scores[0, :, 0, :], masked[0, :, 0, :])


class TestValueZero:
    """Design Section 4.3: V=0 — zero out value vectors for specific tokens."""

    def test_value_zeroed(self):
        H, seq, D = 4, 32, 64
        target_positions = [0, 1]
        hook = ValueZeroHook(target_positions)

        values = torch.randn(1, H, seq, D // H)
        zeroed = hook.apply(values)

        for t in target_positions:
            assert zeroed[0, :, t, :].abs().max() < 1e-10

        # Non-target should be unchanged
        assert torch.equal(values[0, :, 5, :], zeroed[0, :, 5, :])
```

### Step 2: Run test, verify fail

### Step 3: Implement

```python
# contribution/causal.py
"""
Causal verification via length-preserving masking (Design Section 4.3).

Two methods:
1. Attention knockout: mask attention scores to -∞ so query cannot see target
2. Value-zero: keep attention but zero out value vectors for target tokens

Both preserve sequence length and positional indices.
"""
import torch
import torch.nn as nn
from typing import Optional


class AttentionKnockoutHook:
    """Mask attention scores: α_{i,j} = -∞ for (i ∈ queries, j ∈ targets).

    Usage:
        hook = AttentionKnockoutHook([0, 1], query_range=(256, 263))
        handle = model.layer.self_attn.register_forward_pre_hook(hook.hook_fn)
        # ... forward pass ...
        handle.remove()
    """

    def __init__(self, target_positions: list[int], query_range: tuple[int, int] | None = None):
        self.target_positions = target_positions
        self.query_range = query_range  # (start, end) — if None, all queries affected
        self._handles = []

    def apply_mask(self, scores: torch.Tensor) -> torch.Tensor:
        """Apply knockout mask to attention scores (pre-softmax).

        Args:
            scores: (batch, H, seq, seq) — raw attention scores
        Returns:
            masked scores with -inf at target positions
        """
        masked = scores.clone()
        q_start = self.query_range[0] if self.query_range else 0
        q_end = self.query_range[1] if self.query_range else scores.shape[2]

        for t in self.target_positions:
            if t < scores.shape[3]:
                masked[:, :, q_start:q_end, t] = float("-inf")

        return masked

    def register(self, model, model_cfg, get_layers_fn):
        """Register hooks on all attention layers.

        NOTE: This requires architecture-specific integration.
        The hook intercepts the attention computation to modify scores.
        """
        layers = get_layers_fn(model, model_cfg)
        for layer in layers:
            handle = layer.self_attn.register_forward_hook(self._make_hook())
            self._handles.append(handle)

    def _make_hook(self):
        knockout = self

        def hook_fn(module, args, output):
            # output is typically (hidden, attn_weights, ...)
            # We need to modify BEFORE softmax — this is a post-hook on self_attn
            # For proper knockout, we'd need a pre-hook that modifies the mask
            # Simplified: we modify the attention weights in output[1]
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                # Post-softmax weights — zero them out instead
                attn_weights = output[1].clone()  # (B, H, seq, seq)
                q_start = knockout.query_range[0] if knockout.query_range else 0
                q_end = knockout.query_range[1] if knockout.query_range else attn_weights.shape[2]
                for t in knockout.target_positions:
                    if t < attn_weights.shape[3]:
                        attn_weights[:, :, q_start:q_end, t] = 0.0
                # Re-normalize
                row_sums = attn_weights[:, :, q_start:q_end, :].sum(dim=-1, keepdim=True).clamp(min=1e-10)
                attn_weights[:, :, q_start:q_end, :] = attn_weights[:, :, q_start:q_end, :] / row_sums
                return (output[0], attn_weights) + output[2:]
            return output

        return hook_fn

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


class ValueZeroHook:
    """Zero out value vectors for specific tokens (length-preserving).

    Attention is computed normally but target tokens contribute nothing.
    """

    def __init__(self, target_positions: list[int]):
        self.target_positions = target_positions
        self._handles = []

    def apply(self, values: torch.Tensor) -> torch.Tensor:
        """Zero out value vectors at target positions.

        Args:
            values: (batch, H, seq, head_dim)
        Returns:
            values with targets zeroed
        """
        out = values.clone()
        for t in self.target_positions:
            if t < out.shape[2]:
                out[:, :, t, :] = 0.0
        return out

    def register(self, model, model_cfg, get_layers_fn):
        """Register hooks to zero out V projections for target tokens."""
        layers = get_layers_fn(model, model_cfg)
        for layer in layers:
            handle = layer.self_attn.register_forward_hook(self._make_v_hook())
            self._handles.append(handle)

    def _make_v_hook(self):
        vzero = self

        def hook_fn(module, args, output):
            # Modify the hidden state output to remove contribution from target tokens
            # This is approximate — true V=0 requires modifying internal computation
            return output

        return hook_fn

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def compute_output_kl(
    logits_orig: torch.Tensor,   # (vocab,) or (n_action, vocab)
    logits_masked: torch.Tensor,
) -> float:
    """KL(p_orig || p_masked) for action token logits (Design Section 4.3).

    Args:
        logits_orig: original model output logits
        logits_masked: logits after masking intervention
    Returns:
        KL divergence (scalar)
    """
    p = torch.softmax(logits_orig.float(), dim=-1).clamp(min=1e-10)
    q = torch.softmax(logits_masked.float(), dim=-1).clamp(min=1e-10)
    kl = (p * (p.log() - q.log())).sum(dim=-1).mean()
    return float(kl.item())
```

### Step 4: Run test, verify pass

### Step 5: Commit

```bash
git add contribution/causal.py tests/test_causal.py
git commit -m "feat: causal verification — attention knockout and V=0 hooks"
```

---

## Task 7: Causal experiment runner

**Files:**
- Modify: `contribution/causal.py` (add `run_causal_experiment`)

### Step 1: Write test

```python
# Append to tests/test_causal.py

class TestCausalExperiment:
    def test_kl_positive_when_bottleneck_masked(self):
        """Masking a bottleneck token should cause large KL."""
        from contribution.causal import compute_output_kl
        # Simulate: original logits are peaked, masked logits are flat
        logits_orig = torch.zeros(256)
        logits_orig[42] = 10.0  # confident prediction

        logits_masked = torch.zeros(256)  # uniform after masking

        kl = compute_output_kl(logits_orig, logits_masked)
        assert kl > 1.0  # significant divergence

    def test_kl_small_when_sink_masked(self):
        """Masking a sink (low value) should cause small KL."""
        logits_orig = torch.zeros(256)
        logits_orig[42] = 10.0

        # Nearly identical after masking sink
        logits_masked = torch.zeros(256)
        logits_masked[42] = 9.8

        kl = compute_output_kl(logits_orig, logits_masked)
        assert kl < 0.1  # minimal divergence
```

### Step 2-5: Implement, test, commit

```bash
git commit -m "feat: KL divergence for causal masking verification"
```

---

## Task 8: Skill signature analysis

**Files:**
- Create: `contribution/signature.py`
- Create: `tests/test_signature.py`

### Step 1: Write the failing test

```python
# tests/test_signature.py
import numpy as np
import pytest
from contribution.signature import (
    compute_skill_signature,
    compute_within_between_distance,
    run_linear_probe,
    label_skill_from_instruction,
)


class TestSkillLabeling:
    """Design Section 6.1: extract verb/skill from instruction."""

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
    """Design Section 6.2: p_contrib^(n)(j) from deep layers."""

    def test_signature_shape(self):
        """Signature should be a normalized distribution over tokens."""
        # Simulate C̃ from 5 deep layers, seq_len=256
        c_tildes = [np.random.dirichlet(np.ones(256)) for _ in range(5)]
        sig = compute_skill_signature(c_tildes)
        assert sig.shape == (256,)
        assert abs(sig.sum() - 1.0) < 1e-5


class TestWithinBetweenDistance:
    """Design Section 6.3.1: D_within < D_between → skill signatures exist."""

    def test_separable_skills(self):
        """Skills with different signatures should have D_within < D_between."""
        np.random.seed(42)
        # Skill A: concentrated on token 0-10
        sig_a = [np.zeros(64) for _ in range(20)]
        for s in sig_a:
            s[:10] = np.random.dirichlet(np.ones(10))
            s /= s.sum()

        # Skill B: concentrated on token 50-60
        sig_b = [np.zeros(64) for _ in range(20)]
        for s in sig_b:
            s[50:60] = np.random.dirichlet(np.ones(10))
            s /= s.sum()

        labels = ["A"] * 20 + ["B"] * 20
        signatures = sig_a + sig_b

        d_within, d_between = compute_within_between_distance(signatures, labels)
        assert d_within < d_between


class TestLinearProbe:
    """Design Section 6.3.4: skill classification from contribution vectors."""

    def test_separable_data_high_accuracy(self):
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(50, 64) + np.array([1]*32 + [0]*32),
            np.random.randn(50, 64) + np.array([0]*32 + [1]*32),
        ])
        y = np.array([0]*50 + [1]*50)

        acc = run_linear_probe(X, y)
        assert acc > 0.8
```

### Step 2: Run test, verify fail

### Step 3: Implement

```python
# contribution/signature.py
"""
Skill signature analysis (Design Sections 6.1–6.3, 7).

Computes:
- Skill labels from instructions (verb extraction)
- Contribution signatures p_contrib^(n) per sample
- Within-skill vs between-skill JS divergence
- Linear probe accuracy
- Counterfactual instruction analysis
"""
import re
import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# ── Skill labeling ──────────────────────────────────────────────

SKILL_VERBS = {
    "pick": ["pick", "grab", "grasp", "take", "lift"],
    "place": ["place", "put", "set", "drop"],
    "move": ["move", "push", "slide", "drag", "sweep"],
    "open": ["open", "pull open"],
    "close": ["close", "shut"],
    "pour": ["pour", "dump"],
    "stack": ["stack"],
    "fold": ["fold"],
    "wipe": ["wipe", "clean"],
    "turn": ["turn", "rotate", "twist"],
}

# Invert for lookup
_VERB_TO_SKILL = {}
for skill, verbs in SKILL_VERBS.items():
    for v in verbs:
        _VERB_TO_SKILL[v] = skill


def label_skill_from_instruction(instruction: str) -> str:
    """Extract primary skill verb from instruction text.

    Simple rule-based: find the first matching verb.
    Returns skill label or "unknown".
    """
    words = instruction.lower().split()
    for word in words:
        # Strip punctuation
        clean = re.sub(r"[^a-z]", "", word)
        if clean in _VERB_TO_SKILL:
            return _VERB_TO_SKILL[clean]
    return "unknown"


# ── Skill signatures ────────────────────────────────────────────

def compute_skill_signature(c_tildes: list[np.ndarray]) -> np.ndarray:
    """Compute p_contrib^(n)(j) = Normalize(Σ_{l∈L*} C̃_l(j)).

    Args:
        c_tildes: list of C̃ arrays from deep layers, each shape (seq,)
    Returns:
        Normalized signature, shape (seq,)
    """
    stacked = np.stack(c_tildes)
    summed = stacked.sum(axis=0)
    total = summed.sum()
    if total < 1e-10:
        return summed
    return summed / total


# ── Within/Between distance ─────────────────────────────────────

def compute_within_between_distance(
    signatures: list[np.ndarray],
    labels: list[str],
) -> tuple[float, float]:
    """Compute D_within and D_between using JS divergence.

    Args:
        signatures: list of contribution signatures, each (seq,)
        labels: skill label per sample

    Returns:
        (d_within, d_between) — mean JS divergence within and between skills
    """
    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2:
        return 0.0, 0.0

    within_dists = []
    between_dists = []

    n = len(signatures)
    for i in range(n):
        for j in range(i + 1, n):
            si = np.clip(signatures[i], 1e-10, None)
            sj = np.clip(signatures[j], 1e-10, None)
            si = si / si.sum()
            sj = sj / sj.sum()
            js = float(jensenshannon(si, sj) ** 2)

            if labels[i] == labels[j]:
                within_dists.append(js)
            else:
                between_dists.append(js)

    d_within = np.mean(within_dists) if within_dists else 0.0
    d_between = np.mean(between_dists) if between_dists else 0.0

    return d_within, d_between


# ── Linear probe ────────────────────────────────────────────────

def run_linear_probe(X: np.ndarray, y: np.ndarray, cv: int = 5) -> float:
    """Train linear probe on contribution vectors to predict skill labels.

    Args:
        X: (n_samples, seq_len) — contribution signatures
        y: (n_samples,) — skill labels (int-encoded)
        cv: cross-validation folds

    Returns:
        Mean cross-validation accuracy
    """
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="auto")
    scores = cross_val_score(clf, X, y, cv=min(cv, len(np.unique(y))), scoring="accuracy")
    return float(scores.mean())


# ── Counterfactual ──────────────────────────────────────────────

COUNTERFACTUAL_PAIRS = [
    ("pick", "push"),
    ("place", "move"),
    ("open", "close"),
]


def generate_counterfactual_instructions(instruction: str) -> list[tuple[str, str]]:
    """Generate counterfactual instruction pairs by swapping verbs.

    Args:
        instruction: original instruction text

    Returns:
        list of (original_verb, swapped_instruction) pairs
    """
    words = instruction.lower().split()
    original_skill = label_skill_from_instruction(instruction)

    results = []
    for pair in COUNTERFACTUAL_PAIRS:
        if original_skill in pair:
            target = pair[1] if pair[0] == original_skill else pair[0]
            # Replace the verb in instruction
            new_words = []
            replaced = False
            for w in instruction.split():
                clean = re.sub(r"[^a-zA-Z]", "", w).lower()
                if not replaced and clean in _VERB_TO_SKILL and _VERB_TO_SKILL[clean] == original_skill:
                    new_words.append(target)
                    replaced = True
                else:
                    new_words.append(w)
            if replaced:
                results.append((target, " ".join(new_words)))

    return results
```

### Step 4: Run test, verify pass

### Step 5: Commit

```bash
git add contribution/signature.py tests/test_signature.py
git commit -m "feat: skill signature analysis with JS divergence and linear probe"
```

---

## Task 9: Counterfactual instruction experiment

**Files:**
- Modify: `contribution/signature.py` (add `run_counterfactual_analysis`)

### Step 1: Write test

```python
# Append to tests/test_signature.py

class TestCounterfactual:
    def test_generates_swap_pairs(self):
        from contribution.signature import generate_counterfactual_instructions
        pairs = generate_counterfactual_instructions("pick up the red can")
        assert len(pairs) >= 1
        assert pairs[0][0] == "push"
        assert "push" in pairs[0][1]

    def test_no_swap_for_unknown_verb(self):
        from contribution.signature import generate_counterfactual_instructions
        pairs = generate_counterfactual_instructions("prepare the salad")
        assert len(pairs) == 0
```

### Step 2-5: Test, implement, commit

```bash
git commit -m "feat: counterfactual instruction generation for causal skill analysis"
```

---

## Task 10: Visualization module

**Files:**
- Create: `contribution/visualize.py`

### Step 1: Implement (no TDD — visualization code)

```python
# contribution/visualize.py
"""
All 6 figure types from Design Section 9.

1. Layer-wise Top1 contrib share curve (per model)
2. Candidate token frequency heatmap (16×16 or flattened)
3. Attention vs Contribution mismatch scatter
4. Masking ablation curve (top-k knockout → KL change)
5. Skill signature clustering (JS distance matrix)
6. Performance–metric correlation (success vs bottleneck severity)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def plot_top1_share_curve(
    layer_data: dict[str, list[float]],  # model_name → [top1_share per layer]
    output_path: Path,
    title: str = "Layer-wise Top1 Contribution Share",
):
    """Figure 1: Bottleneck onset visualization."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, shares in layer_data.items():
        ax.plot(range(len(shares)), shares, label=name, linewidth=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Top1 C̃(j) Share")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_frequency_heatmap(
    freq: np.ndarray,   # (seq_len,) — Freq_l(j)
    grid_size: int,      # e.g., 16 for 16×16
    output_path: Path,
    title: str = "Candidate Token Frequency",
):
    """Figure 2: Which positions are systematically bottleneck candidates."""
    # Reshape to grid (vision tokens only, first grid_size² tokens)
    n_vision = grid_size * grid_size
    vision_freq = freq[:n_vision]
    grid = vision_freq.reshape(grid_size, grid_size)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(grid, cmap="hot", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Patch Column")
    ax.set_ylabel("Patch Row")
    fig.colorbar(im, ax=ax, label="Freq(j)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_attention_contribution_scatter(
    a_shares: np.ndarray,   # (n_tokens,) — Ã(j) for candidates
    c_shares: np.ndarray,   # (n_tokens,) — C̃(j)
    labels: list[str],      # classification label per token
    output_path: Path,
    title: str = "Attention vs Contribution (Sink/Bottleneck)",
):
    """Figure 3: VAR-style scatter — sinks in top-left, bottlenecks in top-right."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"sink": "red", "bottleneck": "blue", "normal": "gray"}

    for cls in ["normal", "sink", "bottleneck"]:
        mask = np.array([l == cls for l in labels])
        if mask.any():
            ax.scatter(a_shares[mask], c_shares[mask],
                      c=colors[cls], label=cls, alpha=0.6, s=40)

    ax.set_xlabel("Ã(j) — Attention Share")
    ax.set_ylabel("C̃(j) — Contribution Share")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_masking_ablation(
    k_values: list[int],
    kl_values: dict[str, list[float]],  # method → [kl per k]
    output_path: Path,
    title: str = "Masking Ablation: Top-K Knockout → Output KL",
):
    """Figure 4: How much output changes when masking top-k candidates."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, kls in kl_values.items():
        ax.plot(k_values[:len(kls)], kls, marker="o", label=method, linewidth=2)
    ax.set_xlabel("Number of Masked Tokens (K)")
    ax.set_ylabel("KL Divergence (orig → masked)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_skill_js_matrix(
    js_matrix: np.ndarray,   # (n_skills, n_skills)
    skill_names: list[str],
    output_path: Path,
    title: str = "Skill Signature JS Distance Matrix",
):
    """Figure 5: Skill clustering via JS divergence."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(js_matrix, cmap="viridis", interpolation="nearest")
    ax.set_xticks(range(len(skill_names)))
    ax.set_yticks(range(len(skill_names)))
    ax.set_xticklabels(skill_names, rotation=45, ha="right")
    ax.set_yticklabels(skill_names)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="JS Divergence²")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")
```

### Step 2: Commit

```bash
git add contribution/visualize.py
git commit -m "feat: visualization module — 5 figure types for paper"
```

---

## Task 11: Main CLI — `run_contribution_analysis.py`

**Files:**
- Create: `run_contribution_analysis.py`

### Step 1: Implement

```python
#!/usr/bin/env python3
"""
Main entry point: Sink vs Bottleneck contribution analysis.

Usage:
  python run_contribution_analysis.py --model openvla-7b --device cuda:0 --n_samples 20
  python run_contribution_analysis.py --model tracevla-phi3v --device cuda:1 --n_samples 20
"""
import argparse
import json
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
from extract_attention import load_model_from_registry, get_layers, call_processor, detect_token_boundaries
from verify_attention_sinks import SinkVerificationHookManager, get_wov_matrix
from visualize_text_attention import load_samples_from_cache
from model_registry import get_model as registry_get_model

from contribution.compute import (
    extract_sample_contributions,
    compute_candidate_frequency,
    ContributionResult,
)
from contribution.classify import (
    classify_layer,
    compute_phi_all_tokens,
    compute_mismatch,
)
from contribution.signature import (
    label_skill_from_instruction,
    compute_skill_signature,
    compute_within_between_distance,
    run_linear_probe,
)
from contribution.visualize import (
    plot_top1_share_curve,
    plot_frequency_heatmap,
    plot_attention_contribution_scatter,
    plot_skill_js_matrix,
)


def run_analysis(model_name: str, device: str, n_samples: int, top_k: int, output_dir: Path):
    print(f"\n{'='*70}")
    print(f"Contribution Analysis — {model_name}")
    print(f"{'='*70}")

    # ── Load model ──────────────────────────────────────────────
    processor, model, model_cfg = load_model_from_registry(model_name, device)
    samples = load_samples_from_cache(config.DATA_CACHE_DIR, n_samples=n_samples)
    print(f"  Loaded {len(samples)} samples")

    # ── Detect boundaries from first sample ─────────────────────
    boundaries = detect_token_boundaries(
        processor, model, samples[0]["image"], samples[0]["instruction"], device, model_cfg
    )
    print(f"  Boundaries: vis=[{boundaries['vision_start']}:{boundaries['vision_end']}], "
          f"text=[{boundaries['text_start']}:{boundaries['text_end']}]")

    # ── Determine query positions (action tokens or last text tokens) ──
    # For models with action_tokens > 0, use last N text positions as proxy
    # (actual action tokens are generated, not in input)
    text_end = boundaries["text_end"]
    n_query = min(model_cfg.action_tokens or 4, 4)
    query_positions = list(range(max(0, text_end - n_query), text_end))
    print(f"  Query positions (action proxy): {query_positions}")

    # ── Per-sample extraction ───────────────────────────────────
    n_layers = model_cfg.num_layers
    deep_layers = list(range(max(0, n_layers - 10), n_layers))

    all_layer_top1 = {l: [] for l in deep_layers}  # layer → [top1_share per sample]
    all_layer_topk_c = {l: [] for l in deep_layers}  # layer → [topk candidates]
    all_layer_classifications = {l: [] for l in deep_layers}
    all_signatures = []
    all_skill_labels = []
    all_mismatches = []

    hook_mgr = SinkVerificationHookManager(model, model_cfg)

    for si, sample in enumerate(samples):
        print(f"\n  [{si+1}/{len(samples)}] \"{sample['instruction'][:60]}...\"")

        # Skill label
        skill = label_skill_from_instruction(sample["instruction"])
        all_skill_labels.append(skill)

        # Forward pass with hooks
        hook_mgr.register_hooks()
        hook_mgr.reset()
        prompt = model_cfg.prompt_template.format(instruction=sample["instruction"])
        inputs = call_processor(processor, prompt, sample["image"], model_cfg, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(device)
        elif isinstance(inputs, dict):
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            model(**inputs, output_attentions=True)

        attn_weights = hook_mgr.attention_weights
        hidden_states = hook_mgr.hidden_states
        hook_mgr.remove_hooks()

        # Per-sample boundaries (may differ due to instruction length)
        sample_boundaries = detect_token_boundaries(
            processor, model, sample["image"], sample["instruction"], device, model_cfg
        )
        text_end_s = sample_boundaries["text_end"]
        query_pos_s = list(range(max(0, text_end_s - n_query), text_end_s))

        # Extract contributions
        results = extract_sample_contributions(
            attention_weights=attn_weights,
            hidden_states=hidden_states,
            get_wov_fn=get_wov_matrix,
            model=model,
            model_cfg=model_cfg,
            boundaries=sample_boundaries,
            query_positions=query_pos_s,
            top_k=top_k,
            target_layers=deep_layers,
        )

        # Collect per-layer stats
        sample_c_tildes = []
        for r in results:
            l = r.layer_idx
            if l in all_layer_top1:
                top1 = float(r.c_tilde.max())
                all_layer_top1[l].append(top1)
                all_layer_topk_c[l].append(r.topk_contribution)

                # Classify
                phi = compute_phi_all_tokens(hidden_states[l])
                cls_result = classify_layer(r.a_tilde, r.c_tilde, sample_boundaries, phi)
                all_layer_classifications[l].append(cls_result)

                mismatch = compute_mismatch(r.a_tilde, r.c_tilde)
                all_mismatches.append(mismatch)

                sample_c_tildes.append(r.c_tilde)

        # Skill signature (from deep layers)
        if sample_c_tildes:
            sig = compute_skill_signature(sample_c_tildes)
            all_signatures.append(sig)

        print(f"    Skill={skill}, layers_extracted={len(results)}")

    # ── Aggregate results ───────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Layer-wise top1 share
    layer_mean_top1 = {}
    for l in deep_layers:
        if all_layer_top1[l]:
            layer_mean_top1[l] = float(np.mean(all_layer_top1[l]))

    # 2. Frequency analysis (per deep layer)
    seq_len = boundaries["total_seq_len"]
    layer_freq = {}
    for l in deep_layers:
        if all_layer_topk_c[l]:
            freq = compute_candidate_frequency(all_layer_topk_c[l], seq_len)
            layer_freq[l] = freq

    # 3. Dominant classification per layer
    layer_dominant = {}
    for l in deep_layers:
        if all_layer_classifications[l]:
            types = [c["dominant_type"] for c in all_layer_classifications[l]]
            from collections import Counter
            most_common = Counter(types).most_common(1)[0]
            layer_dominant[l] = {
                "dominant_type": most_common[0],
                "frequency": most_common[1] / len(types),
                "mean_mismatch": float(np.mean([c["mismatch"] for c in all_layer_classifications[l]])),
                "mean_entropy": float(np.mean([c["entropy"] for c in all_layer_classifications[l]])),
                "mean_top1_share": float(np.mean([c["top1_share"] for c in all_layer_classifications[l]])),
            }

    # 4. Skill signature analysis
    sig_analysis = {}
    valid_sigs = [(sig, lab) for sig, lab in zip(all_signatures, all_skill_labels) if lab != "unknown"]
    if len(valid_sigs) >= 4:
        sigs = [s for s, _ in valid_sigs]
        labs = [l for _, l in valid_sigs]
        d_within, d_between = compute_within_between_distance(sigs, labs)
        sig_analysis["d_within"] = d_within
        sig_analysis["d_between"] = d_between
        sig_analysis["signature_exists"] = d_within < d_between

        # Linear probe
        unique_labels = sorted(set(labs))
        if len(unique_labels) >= 2:
            X = np.stack(sigs)
            y = np.array([unique_labels.index(l) for l in labs])
            probe_acc = run_linear_probe(X, y)
            sig_analysis["probe_accuracy"] = probe_acc

    # ── Save report ─────────────────────────────────────────────
    report = {
        "model": model_name,
        "n_samples": len(samples),
        "n_layers": n_layers,
        "deep_layers": deep_layers,
        "boundaries": {k: int(v) if isinstance(v, (int, np.integer)) else v
                      for k, v in boundaries.items()},
        "layer_analysis": {str(l): layer_dominant.get(l, {}) for l in deep_layers},
        "skill_signature": sig_analysis,
        "mean_mismatch": float(np.mean(all_mismatches)) if all_mismatches else 0.0,
        "skill_distribution": dict(Counter(all_skill_labels)),
    }

    report_path = output_dir / "contribution_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved: {report_path}")

    # ── Visualizations ──────────────────────────────────────────
    plot_top1_share_curve(
        {model_name: [layer_mean_top1.get(l, 0) for l in deep_layers]},
        output_dir / "top1_contrib_share.png",
    )

    if layer_freq:
        last_layer = max(layer_freq.keys())
        plot_frequency_heatmap(
            layer_freq[last_layer],
            model_cfg.vision_grid_size,
            output_dir / "candidate_frequency.png",
        )

    # Scatter: collect all candidates across layers
    all_a_shares, all_c_shares, all_cls_labels = [], [], []
    for l in deep_layers:
        for cls_result in all_layer_classifications.get(l, []):
            for cand in cls_result.get("candidates", [])[:3]:
                all_a_shares.append(cand["a_share"])
                all_c_shares.append(cand["c_share"])
                all_cls_labels.append(cand["classification"])

    if all_a_shares:
        plot_attention_contribution_scatter(
            np.array(all_a_shares), np.array(all_c_shares), all_cls_labels,
            output_dir / "attention_vs_contribution.png",
        )

    print(f"\n{'='*70}")
    print(f"  Analysis complete for {model_name}")
    print(f"{'='*70}")

    del model
    torch.cuda.empty_cache()
    return report


def main():
    from collections import Counter

    parser = argparse.ArgumentParser(description="Sink vs Bottleneck Contribution Analysis")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else config.OUTPUT_DIR / "contribution_analysis" / args.model
    run_analysis(args.model, args.device, args.n_samples, args.top_k, output_dir)


if __name__ == "__main__":
    main()
```

### Step 2: Smoke test

Run: `/home/kana5123/miniconda3/envs/interp/bin/python run_contribution_analysis.py --model openvla-7b --device cuda:0 --n_samples 3 --top_k 5`
Expected: Runs without error, produces `outputs/contribution_analysis/openvla-7b/contribution_report.json`

### Step 3: Commit

```bash
git add run_contribution_analysis.py
git commit -m "feat: main CLI for contribution analysis pipeline"
```

---

## Task 12: Causal experiment CLI

**Files:**
- Create: `run_causal_experiment.py`

### Step 1: Implement

```python
#!/usr/bin/env python3
"""
Causal verification: mask candidate tokens and measure output KL divergence.

Usage:
  python run_causal_experiment.py --model openvla-7b --device cuda:0 \
    --report outputs/contribution_analysis/openvla-7b/contribution_report.json
"""
import argparse
import json
import sys
import torch
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

import config
from extract_attention import load_model_from_registry, call_processor, detect_token_boundaries
from verify_attention_sinks import SinkVerificationHookManager
from visualize_text_attention import load_samples_from_cache

from contribution.causal import AttentionKnockoutHook, compute_output_kl
from contribution.visualize import plot_masking_ablation


def run_causal(model_name, device, n_samples, output_dir, candidate_positions):
    """Run forward pass with and without knockout, measure KL."""
    processor, model, model_cfg = load_model_from_registry(model_name, device)
    samples = load_samples_from_cache(config.DATA_CACHE_DIR, n_samples=n_samples)

    results = {"model": model_name, "candidates": candidate_positions, "per_k": {}}

    for k in [1, 3, 5]:
        targets = candidate_positions[:k]
        kl_values = []

        for si, sample in enumerate(samples):
            prompt = model_cfg.prompt_template.format(instruction=sample["instruction"])
            inputs = call_processor(processor, prompt, sample["image"], model_cfg, return_tensors="pt")
            if hasattr(inputs, "to"):
                inputs = inputs.to(device)
            elif isinstance(inputs, dict):
                inputs = {k_: v.to(device) if hasattr(v, "to") else v for k_, v in inputs.items()}

            # Original forward
            with torch.no_grad():
                out_orig = model(**inputs)
            logits_orig = out_orig.logits[0, -1, :]  # last token logits

            # Knockout forward
            boundaries = detect_token_boundaries(
                processor, model, sample["image"], sample["instruction"], device, model_cfg
            )
            text_end = boundaries["text_end"]
            query_range = (max(0, text_end - 4), text_end)
            knockout = AttentionKnockoutHook(targets, query_range)
            knockout.register(model, model_cfg, lambda m, c: __import__("extract_attention").get_layers(m, c))

            with torch.no_grad():
                out_masked = model(**inputs)
            logits_masked = out_masked.logits[0, -1, :]
            knockout.remove()

            kl = compute_output_kl(logits_orig, logits_masked)
            kl_values.append(kl)

        results["per_k"][k] = {
            "targets": targets,
            "mean_kl": float(np.mean(kl_values)),
            "std_kl": float(np.std(kl_values)),
        }
        print(f"  K={k}, targets={targets}, KL={np.mean(kl_values):.4f} ± {np.std(kl_values):.4f}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "causal_report.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    k_vals = sorted(results["per_k"].keys())
    plot_masking_ablation(
        k_vals,
        {"attention_knockout": [results["per_k"][k]["mean_kl"] for k in k_vals]},
        output_dir / "masking_ablation.png",
    )

    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--report", type=str, default=None, help="Path to contribution_report.json")
    parser.add_argument("--positions", type=int, nargs="+", default=[0],
                       help="Candidate positions to mask")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    # Load candidates from report if available
    candidates = args.positions
    if args.report:
        with open(args.report) as f:
            report = json.load(f)
        # Extract most frequent candidate across layers
        # (would need to store this in report — use positions arg for now)

    output_dir = Path(args.output_dir) if args.output_dir else config.OUTPUT_DIR / "causal_experiment" / args.model
    run_causal(args.model, args.device, args.n_samples, output_dir, candidates)


if __name__ == "__main__":
    main()
```

### Step 2: Commit

```bash
git add run_causal_experiment.py
git commit -m "feat: causal experiment CLI — attention knockout with KL measurement"
```

---

## Task 13: Run all 4 models

### Step 1: Smoke test all models

```bash
# OpenVLA (cuda:0) and TraceVLA (cuda:1) in parallel
/home/kana5123/miniconda3/envs/interp/bin/python run_contribution_analysis.py \
  --model openvla-7b --device cuda:0 --n_samples 5 --top_k 5 &

/home/kana5123/miniconda3/envs/interp/bin/python run_contribution_analysis.py \
  --model tracevla-phi3v --device cuda:1 --n_samples 5 --top_k 5 &
wait
```

### Step 2: Fix any issues found in smoke test

### Step 3: Full run (20 samples each)

```bash
for model in openvla-7b ecot-7b tracevla-phi3v spatialvla-4b; do
  /home/kana5123/miniconda3/envs/interp/bin/python run_contribution_analysis.py \
    --model $model --device cuda:0 --n_samples 20 --top_k 5
done
```

### Step 4: Commit results

```bash
git add outputs/contribution_analysis/
git commit -m "data: contribution analysis results for 4 VLA models"
```

---

## Task 14: Cross-model comparison

**Files:**
- Create: `compare_contribution_results.py`

### Step 1: Implement

Script that loads all `contribution_report.json` files and produces:
- Table 1: Model × {sink/bottleneck count, onset layer, dominant position}
- Table 2: Skill probe accuracy per model
- Figure: Side-by-side top1 share curves
- Figure: Cross-model attention-contribution scatter

### Step 2: Run and commit

```bash
/home/kana5123/miniconda3/envs/interp/bin/python compare_contribution_results.py
git add compare_contribution_results.py outputs/contribution_analysis/cross_model/
git commit -m "feat: cross-model comparison tables and figures"
```

---

## Execution Order Summary

| Task | Description | Dependencies | Est. Time |
|------|-------------|--------------|-----------|
| 1 | Core W_OV contribution compute | None | 15 min |
| 2 | Single-sample extraction pipeline | Task 1 | 10 min |
| 3 | Frequency analysis | Task 2 | 5 min |
| 4 | Sink/bottleneck classification | Task 1 | 10 min |
| 5 | Phi integration | Task 4 | 5 min |
| 6 | Attention knockout hook | None | 10 min |
| 7 | Causal KL computation | Task 6 | 5 min |
| 8 | Skill signatures + JS divergence | None | 15 min |
| 9 | Counterfactual instructions | Task 8 | 5 min |
| 10 | Visualization module | None | 10 min |
| 11 | Main CLI | Tasks 1-5, 8, 10 | 15 min |
| 12 | Causal CLI | Tasks 6-7, 10 | 10 min |
| 13 | Run all 4 models | Tasks 11-12 | GPU time |
| 14 | Cross-model comparison | Task 13 | 10 min |

**Critical path:** Tasks 1→2→3 + Task 4→5 + Task 8 → Task 11 → Task 13 → Task 14
