# contribution/classify.py
"""
Sink vs Bottleneck classification (Design Section 4.2).

Definitions:
  BOTTLENECK: C̃(j) ≥ c_threshold AND Freq(j) high AND causal impact high
  SINK:       Ã(j) ≥ a_threshold AND C̃(j) < c_threshold ("attention-contribution mismatch")
  NORMAL:     neither condition met
"""
import numpy as np
import torch
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


# Task 5: Phi (hidden state spike) integration
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
