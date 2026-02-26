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
from typing import Callable


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

    Returns:
        Tensor of shape (H, len(query_positions), seq) — contribution norms
    """
    H = attn.shape[0]
    seq = attn.shape[1]
    D = hidden.shape[1]

    # Compute W_OV = W_O @ W_V, shape (D, D)
    w_ov = (w_o @ w_v).float()  # (D, D)

    # Value vectors for all tokens: x_j @ W_OV^T → (seq, D)
    value_vectors = hidden.float() @ w_ov.T  # (seq, D)

    n_query = len(query_positions)
    result = torch.zeros(H, n_query, seq)

    for qi, i in enumerate(query_positions):
        if i >= seq:
            continue
        for h in range(H):
            alpha = attn[h, i, :seq].float()
            weighted = alpha.unsqueeze(1) * value_vectors  # (seq, D)
            norms = torch.norm(weighted, dim=1)  # (seq,)
            result[h, qi, :] = norms

    return result


def aggregate_contributions(
    attn: torch.Tensor,          # (H, seq, seq)
    contrib: torch.Tensor,       # (H, n_query, seq)
    query_positions: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate across heads and query positions, then normalize.

    Returns (a_tilde, c_tilde) — both shape (seq,), summing to 1.0.
    """
    H = attn.shape[0]
    seq = attn.shape[-1]

    a_raw = torch.zeros(seq)
    for qi, i in enumerate(query_positions):
        if i >= attn.shape[1]:
            continue
        a_raw += attn[:, i, :seq].float().mean(dim=0)
    a_raw /= max(len(query_positions), 1)

    c_raw = contrib.mean(dim=(0, 1))

    a_sum = a_raw.sum().clamp(min=1e-10)
    c_sum = c_raw.sum().clamp(min=1e-10)

    a_tilde = a_raw / a_sum
    c_tilde = c_raw / c_sum

    return a_tilde, c_tilde


def find_topk_candidates(
    distribution: torch.Tensor,
    k: int = 5,
    boundaries: dict | None = None,
) -> list[dict]:
    """Find TopK candidates from a normalized distribution."""
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
                token_type = "vision"
            else:
                token_type = "text"

        candidates.append({
            "rank": rank,
            "position": idx,
            "share": val,
            "type": token_type,
        })

    return candidates


def extract_sample_contributions(
    attention_weights: dict[int, torch.Tensor],
    hidden_states: dict[int, torch.Tensor],
    get_wov_fn: Callable,
    model,
    model_cfg,
    boundaries: dict,
    query_positions: list[int],
    top_k: int = 5,
    target_layers: list[int] | None = None,
) -> list[ContributionResult]:
    """Extract Ã(j) and C̃(j) for all layers of a single sample."""
    if target_layers is None:
        target_layers = sorted(set(attention_weights.keys()) & set(hidden_states.keys()))

    results = []

    for layer_idx in target_layers:
        attn = attention_weights[layer_idx]
        if attn.dim() == 4:
            attn = attn[0]

        prev_layer = max(0, layer_idx - 1)
        h = hidden_states.get(prev_layer, hidden_states[layer_idx]).cpu().float()

        try:
            w_v, w_o = get_wov_fn(model, model_cfg, layer_idx)
        except (RuntimeError, ValueError, AttributeError):
            continue

        contrib = compute_perhead_contribution(attn, h, w_v, w_o, query_positions)
        a_tilde, c_tilde = aggregate_contributions(attn, contrib, query_positions)
        a_np = a_tilde.numpy()
        c_np = c_tilde.numpy()

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


def compute_candidate_frequency(
    all_topk: list[list[dict]],
    seq_len: int,
) -> np.ndarray:
    """Compute Freq_l(j) = Pr(j ∈ TopK) across samples."""
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
