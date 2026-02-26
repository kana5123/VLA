"""Verify attention sinks using the VAR paper's 3-part definition (ICLR 2025).

Three conditions must hold for a token to be a true attention sink:

  (A) Attention Concentration — The token consistently receives high attention
      from multiple query tokens (cross-token consistency).

  (B) Hidden State Spike — The token's hidden state has anomalous dimension
      spikes: φ(x) = max_{d∈D_sink} |x[d] / RMS(x)| ≥ τ (τ=20).
      D_sink = dimensions where sinks consistently spike across samples.

  (C) No Contribution — Despite high attention weight, the actual information
      contribution is small: ||α_{i,j} · x_j · W_OV|| << ||α_{i,k} · x_k · W_OV||
      for non-sink token k with similar attention weight.

Usage:
    python verify_attention_sinks.py --model openvla-7b --device cuda:0 --n_samples 5

Output:
    outputs/sink_verification/<model>/
      ├── sink_report.json         (full numerical results)
      ├── condition_A_heatmap.png  (cross-token attention consistency)
      ├── condition_B_phi.png      (hidden state spike magnitudes)
      └── condition_C_contribution.png  (attention vs contribution comparison)
"""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F

import config
from extract_attention import (
    load_model_from_registry,
    detect_token_boundaries,
    get_layers,
    call_processor,
)
from model_registry import get_model as registry_get_model
from visualize_text_attention import (
    load_samples_from_cache,
    extract_keywords_from_instruction,
    find_keyword_token_positions,
)


# ═══════════════════════════════════════════════════════════════════
# Extended Hook Manager (captures hidden states + attention weights)
# ═══════════════════════════════════════════════════════════════════

class SinkVerificationHookManager:
    """Extended hook manager that captures both attention weights AND hidden states.

    Unlike AttentionHookManager which only captures attention weights (output[1]),
    this also captures:
      - Hidden states from each layer output (output[0]) — needed for condition (B)
      - We'll compute W_OV contributions post-hoc — needed for condition (C)
    """

    def __init__(self, model, model_cfg):
        self.model = model
        self.model_cfg = model_cfg
        self.hooks = []
        self.attention_weights = {}   # layer_idx → (H, seq, seq)
        self.hidden_states = {}       # layer_idx → (seq, D)

    def register_hooks(self):
        """Register hooks on all transformer layers."""
        # Enable output_attentions. Some architectures (Gemma2) block this
        # unless _attn_implementation=="eager". Force eager first, then set.
        try:
            cfg = self.model.language_model.config if hasattr(self.model, "language_model") else self.model.config
            if hasattr(cfg, "_attn_implementation") and cfg._attn_implementation != "eager":
                cfg._attn_implementation = "eager"
            cfg.output_attentions = True
        except (ValueError, AttributeError) as e:
            print(f"  WARNING: Cannot set output_attentions ({e}). "
                  f"Using hook-only extraction (hidden states only, no attn weights).")

        layers = get_layers(self.model, self.model_cfg)

        for i, layer in enumerate(layers):
            # Hook on self_attn for attention weights
            attn_hook = layer.self_attn.register_forward_hook(
                self._make_attn_hook(i)
            )
            self.hooks.append(attn_hook)

            # Hook on the full layer for hidden states (layer output)
            layer_hook = layer.register_forward_hook(
                self._make_hidden_hook(i)
            )
            self.hooks.append(layer_hook)

    def _make_attn_hook(self, layer_idx):
        def hook_fn(module, args, output):
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                # output[1] = attention weights: (batch, heads, seq, seq)
                self.attention_weights[layer_idx] = output[1][0].detach().cpu()
        return hook_fn

    def _make_hidden_hook(self, layer_idx):
        def hook_fn(module, args, output):
            # Layer output: (hidden_states, ...) — shape (batch, seq, D)
            h = output[0] if isinstance(output, tuple) else output
            self.hidden_states[layer_idx] = h[0].detach().cpu().float()
        return hook_fn

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def reset(self):
        self.attention_weights = {}
        self.hidden_states = {}


def get_wov_matrix(model, model_cfg, layer_idx):
    """Extract W_V and W_O for a given layer (multi-architecture).

    Supports:
      - LLaMA/Gemma2: separate v_proj, o_proj
      - Phi3V: fused qkv_proj → slice out V portion
      - InternLM2: wqkv (fused) → slice out V portion

    Returns:
        v_weight: (v_dim, hidden_dim) — W_V
        o_weight: (hidden_dim, v_dim) — W_O
    """
    layers = get_layers(model, model_cfg)
    layer = layers[layer_idx]
    attn = layer.self_attn

    o_weight = attn.o_proj.weight.detach().cpu().float()

    if hasattr(attn, "v_proj"):
        # Standard: LLaMA, Gemma2, Gemma
        v_weight = attn.v_proj.weight.detach().cpu().float()
    elif hasattr(attn, "qkv_proj"):
        # Phi3V: fused QKV → (q_dim + k_dim + v_dim, hidden_dim)
        qkv_weight = attn.qkv_proj.weight.detach().cpu().float()
        num_heads = getattr(attn, "num_heads", model_cfg.num_heads)
        head_dim = getattr(attn, "head_dim", model_cfg.hidden_dim // model_cfg.num_heads)
        num_kv_heads = getattr(attn, "num_key_value_heads", num_heads)
        q_dim = num_heads * head_dim
        k_dim = num_kv_heads * head_dim
        v_dim = num_kv_heads * head_dim
        v_weight = qkv_weight[q_dim + k_dim : q_dim + k_dim + v_dim, :]
    elif hasattr(attn, "wqkv"):
        # InternLM2: fused wqkv
        wqkv = attn.wqkv.weight.detach().cpu().float()
        num_heads = getattr(attn, "num_heads", model_cfg.num_heads)
        head_dim = getattr(attn, "head_dim", model_cfg.hidden_dim // model_cfg.num_heads)
        num_kv_heads = getattr(attn, "num_key_value_heads", num_heads)
        q_dim = num_heads * head_dim
        k_dim = num_kv_heads * head_dim
        v_dim = num_kv_heads * head_dim
        v_weight = wqkv[q_dim + k_dim : q_dim + k_dim + v_dim, :]
    else:
        raise AttributeError(
            f"Cannot find V projection in {type(attn).__name__}. "
            f"Available: {[n for n, _ in attn.named_parameters()]}"
        )

    return v_weight, o_weight


# ═══════════════════════════════════════════════════════════════════
# Condition (A): Cross-Token Attention Consistency
# ═══════════════════════════════════════════════════════════════════

def check_condition_A(attention_weights, boundaries, model_cfg, n_query_tokens=8):
    """Check which tokens in the FULL SEQUENCE consistently receive high
    attention from TEXT-ONLY query positions.

    Generalized version:
      - Queries ONLY from text tokens (uses text_query_ranges to skip vision)
      - Checks attention to ALL tokens (full sequence, not just vision range)
      - Identifies sink candidates anywhere: BOS, vision, separators, etc.

    Returns:
        dict per layer with full-sequence consistency and classified sinks
    """
    vs = boundaries["vision_start"]
    ve = boundaries["vision_end"]

    # Use text_query_ranges to get only real text positions
    text_ranges = boundaries.get(
        "text_query_ranges",
        [(boundaries["text_start"], boundaries["text_end"])],
    )
    all_text_pos = []
    for s, e in text_ranges:
        all_text_pos.extend(range(s, e))

    if not all_text_pos:
        return {}

    if len(all_text_pos) <= n_query_tokens:
        query_positions = all_text_pos
    else:
        step = len(all_text_pos) // n_query_tokens
        query_positions = all_text_pos[::step][:n_query_tokens]

    results = {}

    for layer_idx in sorted(attention_weights.keys()):
        attn = attention_weights[layer_idx]  # (H, seq, seq)
        if attn.dim() == 4:
            attn = attn[0]

        seq_len = attn.shape[-1]
        n_vision = ve - vs

        # For each text query, get attention to ALL tokens
        top5_counts = np.zeros(seq_len)
        n_valid = 0

        for qpos in query_positions:
            if qpos >= attn.shape[1]:
                continue
            q_to_all = attn[:, qpos, :seq_len].float().mean(dim=0).numpy()
            top5_idx = np.argsort(q_to_all)[::-1][:5]
            for idx in top5_idx:
                top5_counts[idx] += 1
            n_valid += 1

        if n_valid == 0:
            continue

        consistency = top5_counts / n_valid

        # Classify consistent sinks (>80%) across full sequence
        sink_candidates = []
        for pos in np.where(consistency > 0.8)[0]:
            pos = int(pos)
            if vs <= pos < ve:
                token_type = f"vision[{pos - vs}]"
            elif pos < vs:
                token_type = f"pre_vision[{pos}]"
            else:
                token_type = f"text[{pos - ve}]" if pos >= ve else f"pos[{pos}]"
            sink_candidates.append({
                "position": pos,
                "type": token_type,
                "consistency": float(consistency[pos]),
            })

        # Vision-only subset (backward compat with plots)
        vision_consistency = consistency[vs:ve].tolist() if n_vision > 0 else []
        vision_sinks = [int(i) for i in np.where(np.array(vision_consistency) > 0.8)[0]] if vision_consistency else []

        results[f"layer_{layer_idx:02d}"] = {
            "consistency_scores": vision_consistency,       # backward compat
            "consistent_sinks": vision_sinks,               # backward compat (vision-relative)
            "full_consistency": consistency.tolist(),        # NEW: full sequence
            "sink_candidates": sink_candidates,             # NEW: classified sinks
            "query_positions_used": [int(q) for q in query_positions],
            "n_valid_queries": n_valid,
            "top1_vision_token": int(np.argmax(vision_consistency)) if vision_consistency else -1,
            "top1_consistency": float(max(vision_consistency)) if vision_consistency else 0.0,
        }

    return results


def check_full_sequence_sinks(attention_weights, boundaries, model_cfg, n_query_tokens=8):
    """Check which tokens in the FULL sequence (not just vision) receive concentrated attention.

    This addresses a blind spot: sinks can occur at BOS, separator, or other
    special tokens outside the vision range. VAR paper's sinks may be text-side.

    For each layer, from the last text token's perspective, find which tokens
    in the entire sequence receive the most attention (head-averaged).

    Returns:
        dict with per-layer full-sequence attention analysis
    """
    te = boundaries["text_end"]
    vs = boundaries["vision_start"]
    ve = boundaries["vision_end"]

    results = {}

    for layer_idx in sorted(attention_weights.keys()):
        attn = attention_weights[layer_idx]  # (H, seq, seq)
        if attn.dim() == 4:
            attn = attn[0]

        seq_len = attn.shape[-1]
        query_pos = te - 1
        if query_pos >= attn.shape[1]:
            continue

        # Last text token's attention to ALL tokens (head-averaged)
        attn_to_all = attn[:, query_pos, :seq_len].float().mean(dim=0).numpy()  # (seq_len,)

        # Also check multiple query positions for consistency
        n_text = te - (boundaries.get("text_start", 0) or 0)
        text_start = boundaries.get("text_start", 0) or 0
        if n_text <= n_query_tokens:
            query_positions = list(range(text_start, te))
        else:
            step = max(1, n_text // n_query_tokens)
            query_positions = list(range(text_start, te, step))[:n_query_tokens]

        # Count how often each token is in top-10 across queries
        top10_counts = np.zeros(seq_len)
        for qpos in query_positions:
            if qpos >= attn.shape[1]:
                continue
            q_to_all = attn[:, qpos, :seq_len].float().mean(dim=0).numpy()
            top10_idx = np.argsort(q_to_all)[::-1][:10]
            for idx in top10_idx:
                top10_counts[idx] += 1
        consistency = top10_counts / max(len(query_positions), 1)

        # Find top-5 most attended tokens overall
        top5_global = np.argsort(attn_to_all)[::-1][:5]
        top5_info = []
        for pos in top5_global:
            pos = int(pos)
            if vs <= pos < ve:
                token_type = f"vision[{pos - vs}]"
            elif pos < vs:
                token_type = f"pre_vision[{pos}]"
            else:
                token_type = f"text[{pos - ve}]" if pos >= ve else f"pos[{pos}]"
            top5_info.append({
                "position": pos,
                "type": token_type,
                "attention_pct": float(attn_to_all[pos] * 100),
                "consistency": float(consistency[pos]),
            })

        # Classify sink candidates (>5% attention from last token)
        sink_candidates = []
        for pos in range(seq_len):
            if attn_to_all[pos] > 0.05:  # >5% of total attention
                pos_int = int(pos)
                if vs <= pos_int < ve:
                    token_type = f"vision[{pos_int - vs}]"
                elif pos_int < vs:
                    token_type = f"pre_vision[{pos_int}]"
                else:
                    token_type = f"text[{pos_int - ve}]" if pos_int >= ve else f"pos[{pos_int}]"
                sink_candidates.append({
                    "position": pos_int,
                    "type": token_type,
                    "attention_pct": float(attn_to_all[pos] * 100),
                    "consistency": float(consistency[pos]),
                })

        results[f"layer_{layer_idx:02d}"] = {
            "top5_tokens": top5_info,
            "sink_candidates": sink_candidates,
            "vision_attention_total_pct": float(attn_to_all[vs:ve].sum() * 100),
            "text_attention_total_pct": float(
                (attn_to_all[:vs].sum() + attn_to_all[ve:].sum()) * 100
            ),
        }

    return results


# ═══════════════════════════════════════════════════════════════════
# Condition (B): Hidden State Dimension Spike
# ═══════════════════════════════════════════════════════════════════

def check_condition_B(hidden_states, boundaries, tau=20.0, top_k_dims=10):
    """Check ALL tokens in the full sequence for hidden state dimension spikes.

    Generalized version: computes φ(x) for every token, not just vision.
    φ(x) = max_d |x[d]| / RMS(x). Tokens with φ(x) ≥ τ have spikes.

    Returns:
        dict per layer with full-sequence phi and classified spike positions
    """
    vs = boundaries["vision_start"]
    ve = boundaries["vision_end"]

    results = {}

    for layer_idx in sorted(hidden_states.keys()):
        h = hidden_states[layer_idx]  # (seq_len, D)
        seq_len = h.shape[0]
        D = h.shape[1]

        # Compute RMS per token (ALL tokens in sequence)
        rms = torch.sqrt((h ** 2).mean(dim=1)).clamp(min=1e-8)  # (seq_len,)
        normalized = h.abs() / rms.unsqueeze(1)  # (seq_len, D)
        phi_values, phi_dim_indices = normalized.max(dim=1)  # (seq_len,)
        phi_all = phi_values.numpy()

        # Find spike tokens across full sequence
        spike_mask = phi_values >= tau
        all_spike_positions = torch.where(spike_mask)[0].tolist()

        # Classify spike positions
        spike_dims_info = {}
        for pos in all_spike_positions:
            if vs <= pos < ve:
                token_type = f"vision[{pos - vs}]"
            elif pos < vs:
                token_type = f"pre_vision[{pos}]"
            else:
                token_type = f"text[{pos - ve}]" if pos >= ve else f"pos[{pos}]"
            tok_normalized = normalized[pos]
            top_vals, top_dims = tok_normalized.topk(min(top_k_dims, D))
            spike_dims_info[str(pos)] = {
                "position": pos,
                "type": token_type,
                "phi": float(phi_values[pos]),
                "max_dim": int(phi_dim_indices[pos]),
                "top_dims": top_dims.tolist(),
                "top_vals": [round(v, 2) for v in top_vals.tolist()],
                "rms": float(rms[pos]),
            }

        # Vision-only subset (backward compat with plots)
        vision_phi = phi_all[vs:ve].tolist() if ve > vs else []
        vision_spike_tokens = [pos - vs for pos in all_spike_positions if vs <= pos < ve]

        results[f"layer_{layer_idx:02d}"] = {
            "phi_values": vision_phi,                   # backward compat (vision-only)
            "spike_tokens": vision_spike_tokens,        # backward compat (vision-relative)
            "full_phi": phi_all.tolist(),                # NEW: full sequence
            "all_spike_positions": all_spike_positions,  # NEW: absolute positions
            "spike_dims_info": spike_dims_info,
            "phi_mean": float(phi_all.mean()),
            "phi_std": float(phi_all.std()),
            "phi_max": float(phi_all.max()),
            "phi_max_position": int(phi_all.argmax()),  # absolute position
            "phi_max_token": int(np.argmax(vision_phi)) if vision_phi else -1,  # compat
            "n_spike_tokens": len(all_spike_positions),
            "tau": tau,
        }

    return results


# ═══════════════════════════════════════════════════════════════════
# Condition (C): Contribution Analysis
# ═══════════════════════════════════════════════════════════════════

def check_condition_C(model, model_cfg, attention_weights, hidden_states,
                      boundaries, sink_position=None, target_layers=None):
    """Check if a sink token has low actual contribution despite high attention.

    Generalized version:
      - Works for ANY sink position (vision, BOS, text separator, etc.)
      - Falls back to hidden-state norm when W_OV extraction fails (GQA)
      - Compares sink's value norm to mean of other tokens

    Args:
        sink_position: absolute position of the sink candidate. None = vision[0].
        target_layers: layers to analyze (default: last 8)

    Returns:
        dict per layer with sink contribution analysis
    """
    vs = boundaries["vision_start"]
    ve = boundaries["vision_end"]

    # Find last text position for query
    text_ranges = boundaries.get(
        "text_query_ranges",
        [(boundaries["text_start"], boundaries["text_end"])],
    )
    last_text_pos = max(e - 1 for _, e in text_ranges)

    if sink_position is None:
        sink_position = vs  # default to first vision token

    if target_layers is None:
        n_layers = model_cfg.num_layers
        target_layers = list(range(max(0, n_layers - 8), n_layers))

    results = {}

    for layer_idx in target_layers:
        if layer_idx not in attention_weights or layer_idx not in hidden_states:
            continue

        attn = attention_weights[layer_idx]  # (H, seq, seq)
        if attn.dim() == 4:
            attn = attn[0]

        if last_text_pos >= attn.shape[1]:
            continue

        # Attention from last text token to ALL tokens
        alpha_all = attn[:, last_text_pos, :].float().mean(dim=0).numpy()

        # Hidden states
        prev_layer = max(0, layer_idx - 1)
        if prev_layer not in hidden_states:
            prev_layer = layer_idx
        h = hidden_states[prev_layer]  # (seq_len, D)

        sink_attention = float(alpha_all[sink_position]) if sink_position < len(alpha_all) else 0.0

        # Try W_OV computation; fall back to hidden-state norm on failure
        wov_available = False
        try:
            v_weight, o_weight = get_wov_matrix(model, model_cfg, layer_idx)

            # Value projection for sink token
            sink_h = h[sink_position:sink_position + 1]  # (1, D)
            sink_v = sink_h @ v_weight.T  # (1, v_dim)
            sink_v_norm = float(torch.norm(sink_v).item())

            # Sample other tokens for comparison (skip sink)
            other_positions = [p for p in range(h.shape[0]) if p != sink_position]
            if len(other_positions) > 50:
                step = len(other_positions) // 50
                other_positions = other_positions[::step][:50]
            other_h = h[other_positions]
            other_v = other_h @ v_weight.T
            other_v_norms = torch.norm(other_v, dim=1).numpy()
            others_mean = float(other_v_norms.mean())
            wov_available = True

        except (RuntimeError, ValueError, AttributeError):
            # GQA dimension mismatch — fall back to hidden state norm
            sink_v_norm = float(torch.norm(h[sink_position]).item())
            all_norms = torch.norm(h, dim=1).numpy()
            mask = np.ones(len(all_norms), dtype=bool)
            mask[sink_position] = False
            others_mean = float(all_norms[mask].mean())

        value_norm_ratio = sink_v_norm / (others_mean + 1e-8)

        # Classify sink position
        if vs <= sink_position < ve:
            sink_type = f"vision[{sink_position - vs}]"
        elif sink_position < vs:
            sink_type = f"pre_vision[{sink_position}]"
        else:
            sink_type = f"text[{sink_position - ve}]" if sink_position >= ve else f"pos[{sink_position}]"

        results[f"layer_{layer_idx:02d}"] = {
            "sink_position": sink_position,
            "sink_type": sink_type,
            "sink_attention": sink_attention,
            "sink_value_norm": sink_v_norm,
            "others_value_norm_mean": others_mean,
            "value_norm_ratio": value_norm_ratio,
            "wov_available": wov_available,
            "is_low_contribution": value_norm_ratio < 0.5,   # true sink
            "is_high_contribution": value_norm_ratio > 2.0,  # context aggregator
            # Backward compat fields
            "token0_value_norm": sink_v_norm,
            "others_value_norm_mean": others_mean,
        }

    return results


# ═══════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════

def plot_condition_A(cond_a_results, output_path, model_name, n_vision):
    """Plot cross-token attention consistency heatmap."""
    layers = sorted(cond_a_results.keys())
    n_layers = len(layers)

    # Select a subset of layers for readability
    if n_layers > 16:
        step = max(1, n_layers // 16)
        selected_layers = layers[::step]
    else:
        selected_layers = layers

    # Build matrix: (layers, vision_tokens)
    mat = np.zeros((len(selected_layers), n_vision))
    for i, lk in enumerate(selected_layers):
        scores = cond_a_results[lk]["consistency_scores"]
        mat[i, :len(scores)] = scores[:n_vision]

    fig, ax = plt.subplots(figsize=(14, max(6, len(selected_layers) * 0.4)))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1.0)

    ax.set_xlabel("Vision Token Index", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    ax.set_title(
        f"Condition (A): Cross-Token Attention Consistency\n"
        f"{model_name} — Fraction of text queries ranking each vision token in top-5",
        fontsize=12,
    )
    ax.set_yticks(range(len(selected_layers)))
    ax.set_yticklabels([lk.replace("layer_", "L") for lk in selected_layers], fontsize=7)

    # Mark vision token 0 with a vertical line
    ax.axvline(x=0, color='cyan', linewidth=1.5, linestyle='--', alpha=0.7, label='Token 0')

    plt.colorbar(im, ax=ax, label="Consistency (0=never, 1=always in top-5)")
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_condition_B(cond_b_results, output_path, model_name, n_vision, tau=20.0):
    """Plot hidden state spike analysis (phi values)."""
    layers = sorted(cond_b_results.keys())
    n_layers = len(layers)

    if n_layers > 16:
        step = max(1, n_layers // 16)
        selected_layers = layers[::step]
    else:
        selected_layers = layers

    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, len(selected_layers) * 0.4)))

    # Left: phi values heatmap (layers × vision tokens)
    mat_phi = np.zeros((len(selected_layers), n_vision))
    for i, lk in enumerate(selected_layers):
        phi = cond_b_results[lk]["phi_values"]
        mat_phi[i, :len(phi)] = phi[:n_vision]

    im1 = axes[0].imshow(mat_phi, aspect="auto", cmap="hot", vmin=0, vmax=max(tau * 2, mat_phi.max()))
    axes[0].set_xlabel("Vision Token Index", fontsize=11)
    axes[0].set_ylabel("Layer", fontsize=11)
    axes[0].set_title(f"φ(x) = max_d |x[d]/RMS(x)| per vision token\n(τ = {tau})", fontsize=11)
    axes[0].set_yticks(range(len(selected_layers)))
    axes[0].set_yticklabels([lk.replace("layer_", "L") for lk in selected_layers], fontsize=7)
    axes[0].axvline(x=0, color='cyan', linewidth=1.5, linestyle='--', alpha=0.7)
    plt.colorbar(im1, ax=axes[0], label="φ(x)")

    # Right: bar chart of phi for token 0 vs mean of others, per layer
    phi_token0 = []
    phi_others_mean = []
    for lk in selected_layers:
        phi = cond_b_results[lk]["phi_values"]
        phi_token0.append(phi[0] if len(phi) > 0 else 0)
        phi_others_mean.append(np.mean(phi[1:]) if len(phi) > 1 else 0)

    x = np.arange(len(selected_layers))
    width = 0.35
    axes[1].barh(x - width/2, phi_token0, width, label="Token 0 (sink candidate)", color='red', alpha=0.8)
    axes[1].barh(x + width/2, phi_others_mean, width, label="Others (mean)", color='steelblue', alpha=0.8)
    axes[1].axvline(x=tau, color='green', linewidth=2, linestyle='--', label=f"τ = {tau}")
    axes[1].set_yticks(x)
    axes[1].set_yticklabels([lk.replace("layer_", "L") for lk in selected_layers], fontsize=7)
    axes[1].set_xlabel("φ(x)", fontsize=11)
    axes[1].set_title("Token 0 vs Others: Hidden State Spike", fontsize=11)
    axes[1].legend(fontsize=9)

    plt.suptitle(
        f"Condition (B): Hidden State Dimension Spike — {model_name}",
        fontsize=13, fontweight='bold', y=1.02,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_condition_C(cond_c_results, output_path, model_name, n_vision):
    """Plot attention vs contribution comparison."""
    layers = sorted(cond_c_results.keys())
    n_layers = len(layers)

    fig, axes = plt.subplots(2, min(4, n_layers), figsize=(5 * min(4, n_layers), 10),
                             squeeze=False)

    for col, lk in enumerate(layers[:4]):
        data = cond_c_results[lk]
        alpha = np.array(data["alpha_mean"])
        output_norms = np.array(data["output_norms"])

        # Normalize for comparison
        alpha_norm = alpha / (alpha.max() + 1e-8)
        contrib_norm = output_norms / (output_norms.max() + 1e-8)

        # Top row: scatter plot (attention vs contribution)
        ax_scatter = axes[0, col]
        colors = ['red' if i == 0 else 'steelblue' for i in range(n_vision)]
        sizes = [80 if i == 0 else 20 for i in range(n_vision)]
        ax_scatter.scatter(alpha_norm, contrib_norm, c=colors, s=sizes, alpha=0.6)
        ax_scatter.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)  # diagonal
        ax_scatter.set_xlabel("Attention (normalized)", fontsize=9)
        ax_scatter.set_ylabel("Contribution (normalized)", fontsize=9)
        ax_scatter.set_title(lk.replace("layer_", "Layer "), fontsize=10)

        # Annotate token 0
        if n_vision > 0:
            ax_scatter.annotate(
                "Token 0",
                (alpha_norm[0], contrib_norm[0]),
                textcoords="offset points", xytext=(10, 10),
                fontsize=8, color='red', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red', lw=1),
            )

        # Bottom row: bar comparison for top-10 attention tokens
        ax_bar = axes[1, col]
        top_tokens = data["top_attention_tokens"]
        labels = [f"V{t['vision_token']}" for t in top_tokens]
        attn_vals = [t["attention_score"] for t in top_tokens]
        contrib_vals = [t["output_norm"] for t in top_tokens]

        # Normalize for bar chart comparison
        max_contrib = max(contrib_vals) if contrib_vals else 1.0
        contrib_vals_norm = [v / (max_contrib + 1e-8) for v in contrib_vals]
        max_attn = max(attn_vals) if attn_vals else 1.0
        attn_vals_norm = [v / (max_attn + 1e-8) for v in attn_vals]

        x = np.arange(len(labels))
        width = 0.35
        bar_colors_attn = ['red' if t['vision_token'] == 0 else 'lightcoral' for t in top_tokens]
        bar_colors_cont = ['darkblue' if t['vision_token'] == 0 else 'steelblue' for t in top_tokens]

        ax_bar.bar(x - width/2, attn_vals_norm, width, color=bar_colors_attn, alpha=0.7, label="Attention")
        ax_bar.bar(x + width/2, contrib_vals_norm, width, color=bar_colors_cont, alpha=0.7, label="Contribution")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(labels, fontsize=7, rotation=45)
        ax_bar.set_ylabel("Normalized value", fontsize=9)
        ax_bar.set_title(f"{lk.replace('layer_', 'L')}: Attention vs Contribution", fontsize=9)
        if col == 0:
            ax_bar.legend(fontsize=7)

    plt.suptitle(
        f"Condition (C): Attention Weight vs Information Contribution — {model_name}\n"
        f"Sinks have HIGH attention but LOW contribution (below diagonal)",
        fontsize=12, fontweight='bold', y=1.04,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_summary(cond_a, cond_b, cond_c, output_path, model_name, n_vision):
    """Create a single summary figure with all 3 conditions for key layers."""
    # Pick 4 representative layers
    all_layers_a = sorted(cond_a.keys())
    all_layers_b = sorted(cond_b.keys())
    all_layers_c = sorted(cond_c.keys())

    # Use condition C layers (last 8) as anchor
    summary_layers = all_layers_c[:4] if len(all_layers_c) >= 4 else all_layers_c

    fig = plt.figure(figsize=(20, 5 * len(summary_layers)))
    gs = gridspec.GridSpec(len(summary_layers), 3, figure=fig, wspace=0.3, hspace=0.4)

    for row, lk in enumerate(summary_layers):
        # (A) Consistency bar for this layer
        ax_a = fig.add_subplot(gs[row, 0])
        if lk in cond_a:
            scores = np.array(cond_a[lk]["consistency_scores"][:n_vision])
            colors_a = ['red' if i == 0 else 'steelblue' for i in range(len(scores))]
            ax_a.bar(range(len(scores)), scores, color=colors_a, alpha=0.6, width=1.0)
            ax_a.axhline(y=0.8, color='green', linestyle='--', linewidth=1, label='80% threshold')
            ax_a.set_ylabel("Consistency", fontsize=8)
            consistent = cond_a[lk]["consistent_sinks"]
            ax_a.set_title(f"{lk.replace('layer_', 'L')}: (A) Consistency\nSinks: {consistent}", fontsize=9)
        ax_a.set_xlim(-0.5, min(n_vision, 20) - 0.5)  # Zoom to first 20 tokens

        # (B) Phi bar for this layer
        ax_b = fig.add_subplot(gs[row, 1])
        if lk in cond_b:
            phi = np.array(cond_b[lk]["phi_values"][:n_vision])
            colors_b = ['red' if i == 0 else 'steelblue' for i in range(len(phi))]
            ax_b.bar(range(len(phi)), phi, color=colors_b, alpha=0.6, width=1.0)
            ax_b.axhline(y=20.0, color='green', linestyle='--', linewidth=1, label=f'τ=20')
            ax_b.set_ylabel("φ(x)", fontsize=8)
            n_spikes = cond_b[lk]["n_spike_tokens"]
            ax_b.set_title(f"{lk.replace('layer_', 'L')}: (B) φ Spike\n{n_spikes} tokens ≥ τ", fontsize=9)
        ax_b.set_xlim(-0.5, min(n_vision, 20) - 0.5)

        # (C) Attention vs Contribution scatter
        ax_c = fig.add_subplot(gs[row, 2])
        if lk in cond_c:
            alpha = np.array(cond_c[lk]["alpha_mean"])
            output_norms = np.array(cond_c[lk]["output_norms"])
            alpha_n = alpha / (alpha.max() + 1e-8)
            contrib_n = output_norms / (output_norms.max() + 1e-8)
            colors_c = ['red' if i == 0 else 'steelblue' for i in range(len(alpha_n))]
            sizes_c = [60 if i == 0 else 15 for i in range(len(alpha_n))]
            ax_c.scatter(alpha_n, contrib_n, c=colors_c, s=sizes_c, alpha=0.5)
            ax_c.plot([0, 1], [0, 1], 'k--', alpha=0.3)
            ax_c.set_xlabel("Attention", fontsize=8)
            ax_c.set_ylabel("Contribution", fontsize=8)
            # Annotate token 0
            if len(alpha_n) > 0:
                ax_c.annotate("T0", (alpha_n[0], contrib_n[0]),
                              fontsize=8, color='red', fontweight='bold')
            ax_c.set_title(f"{lk.replace('layer_', 'L')}: (C) Attn vs Contribution", fontsize=9)

    plt.suptitle(
        f"Attention Sink Verification Summary — {model_name}\n"
        f"Red = Token 0 (candidate sink), Blue = Other vision tokens",
        fontsize=14, fontweight='bold', y=1.02,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def run_verification(model_name, device, n_samples=5, tau=20.0, output_dir=None):
    """Run the full 3-part sink verification pipeline."""

    # ── Setup ────────────────────────────────────────────────────────
    if output_dir is None:
        output_dir = config.OUTPUT_DIR / "sink_verification" / model_name
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Attention Sink Verification — {model_name}")
    print(f"VAR Paper 3-Part Definition (ICLR 2025)")
    print(f"{'='*60}")

    # ── Load model ───────────────────────────────────────────────────
    print("\n[1/5] Loading model...")
    processor, model, model_cfg = load_model_from_registry(model_name, device=device)

    # ── Load samples ─────────────────────────────────────────────────
    print(f"\n[2/5] Loading {n_samples} samples...")
    samples = load_samples_from_cache(config.DATA_CACHE_DIR, n_samples=n_samples)
    print(f"  Loaded {len(samples)} samples")

    # ── Run verification on each sample ──────────────────────────────
    all_cond_a = []
    all_cond_b = []
    all_cond_c = []
    sample_sink_positions = []  # per-sample detected sink position

    n_vision = model_cfg.num_vision_tokens
    last_boundaries = None

    for idx, sample in enumerate(samples):
        print(f"\n[3/5] Sample {idx}/{len(samples)}: \"{sample['instruction'][:50]}...\"")

        # Prepare inputs
        image = sample["image"]
        instruction = sample["instruction"]
        prompt = model_cfg.prompt_template.format(instruction=instruction)
        inputs = call_processor(processor, prompt, image, model_cfg,
                                return_tensors="pt").to(device)
        if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

        # Detect boundaries (includes text_query_ranges)
        boundaries = detect_token_boundaries(processor, model, image, instruction, device, model_cfg=model_cfg)
        last_boundaries = boundaries

        # Register hooks
        hook_mgr = SinkVerificationHookManager(model, model_cfg)
        hook_mgr.register_hooks()

        # Forward pass
        with torch.no_grad():
            fwd_kwargs = {k: v for k, v in inputs.items()}
            fwd_kwargs["use_cache"] = False
            if model_cfg.architecture == "gemma2" and "intrinsic" not in fwd_kwargs:
                fwd_kwargs["intrinsic"] = torch.tensor(
                    [[[218.26, 0.0, 111.83],
                      [0.0, 218.26, 111.79],
                      [0.0, 0.0, 1.0]]], device=device, dtype=torch.float32,
                )
            model(**fwd_kwargs, output_attentions=True)

        # ── Condition A (full-sequence) ──────────────────────────────
        print("    Checking Condition (A): Cross-token consistency (text→all)...")
        cond_a = check_condition_A(hook_mgr.attention_weights, boundaries, model_cfg)
        all_cond_a.append(cond_a)

        # ── Condition B (full-sequence) ──────────────────────────────
        print("    Checking Condition (B): Hidden state spikes (full seq)...")
        cond_b = check_condition_B(hook_mgr.hidden_states, boundaries, tau=tau)
        all_cond_b.append(cond_b)

        # ── Detect per-sample dominant sink from A ───────────────────
        # Vote across layers: which position appears most often as a sink?
        sink_votes = {}
        for lk in sorted(cond_a.keys()):
            for sc in cond_a[lk].get("sink_candidates", []):
                pos = sc["position"]
                sink_votes[pos] = sink_votes.get(pos, 0) + 1
        if sink_votes:
            detected_sink = max(sink_votes, key=sink_votes.get)
        else:
            # No consistent sink → use position with highest avg full_consistency
            layer_keys = sorted(cond_a.keys())
            if layer_keys:
                last_lk = layer_keys[-1]
                fc = cond_a[last_lk].get("full_consistency", [])
                detected_sink = int(np.argmax(fc)) if fc else boundaries["vision_start"]
            else:
                detected_sink = boundaries["vision_start"]
        sample_sink_positions.append(detected_sink)
        print(f"    Detected sink candidate: position {detected_sink}")

        # ── Condition C (on detected sink) ───────────────────────────
        print(f"    Checking Condition (C): Contribution at position {detected_sink}...")
        cond_c = check_condition_C(
            model, model_cfg,
            hook_mgr.attention_weights, hook_mgr.hidden_states,
            boundaries, sink_position=detected_sink,
        )
        all_cond_c.append(cond_c)

        # Clean up
        hook_mgr.remove_hooks()
        hook_mgr.reset()
        torch.cuda.empty_cache()

    # ── Aggregate results across samples ─────────────────────────────
    from collections import Counter
    print(f"\n[4/5] Aggregating results across {len(samples)} samples...")

    vs = last_boundaries["vision_start"]
    ve = last_boundaries["vision_end"]

    # ── Find dominant sink position across samples ────────────────
    sink_counter = Counter(sample_sink_positions)
    dominant_sink = sink_counter.most_common(1)[0][0]
    dominant_sink_freq = sink_counter.most_common(1)[0][1]
    print(f"  Dominant sink position: {dominant_sink} "
          f"(detected in {dominant_sink_freq}/{len(samples)} samples)")

    # Classify dominant sink
    if vs <= dominant_sink < ve:
        dominant_sink_type = f"vision[{dominant_sink - vs}]"
    elif dominant_sink < vs:
        dominant_sink_type = f"pre_vision[{dominant_sink}]"
    else:
        dominant_sink_type = f"text[{dominant_sink - ve}]" if dominant_sink >= ve else f"pos[{dominant_sink}]"
    print(f"  Sink type: {dominant_sink_type}")

    # ── Aggregate condition A (vision-only subset for plots) ──────
    agg_cond_a = {}
    for lk in all_cond_a[0].keys():
        # Vision-only scores (backward compat with plots)
        n_scores = len(all_cond_a[0][lk]["consistency_scores"]) if all_cond_a[0][lk]["consistency_scores"] else 0
        if n_scores > 0:
            avg_scores = np.zeros(n_scores)
            for sample_a in all_cond_a:
                if lk in sample_a:
                    scores = np.array(sample_a[lk]["consistency_scores"])
                    avg_scores[:min(len(scores), n_scores)] += scores[:n_scores]
            avg_scores /= len(all_cond_a)
        else:
            avg_scores = np.array([])

        # Full-sequence sink candidates (aggregated across samples)
        candidate_votes = {}
        for sample_a in all_cond_a:
            if lk in sample_a:
                for sc in sample_a[lk].get("sink_candidates", []):
                    pos = sc["position"]
                    if pos not in candidate_votes:
                        candidate_votes[pos] = {"type": sc["type"], "consistencies": []}
                    candidate_votes[pos]["consistencies"].append(sc["consistency"])

        agg_sink_candidates = []
        for pos, info in sorted(candidate_votes.items()):
            avg_cons = float(np.mean(info["consistencies"]))
            freq = len(info["consistencies"])
            agg_sink_candidates.append({
                "position": pos,
                "type": info["type"],
                "avg_consistency": avg_cons,
                "frequency": freq,
            })
        agg_sink_candidates.sort(key=lambda x: x["avg_consistency"], reverse=True)

        agg_cond_a[lk] = {
            "consistency_scores": avg_scores.tolist(),
            "consistent_sinks": np.where(avg_scores > 0.8)[0].tolist() if len(avg_scores) > 0 else [],
            "sink_candidates": agg_sink_candidates,
            "top1_vision_token": int(np.argmax(avg_scores)) if len(avg_scores) > 0 else -1,
            "top1_consistency": float(avg_scores.max()) if len(avg_scores) > 0 else 0.0,
        }

    # ── Aggregate condition B (vision-only subset + full-seq spikes) ──
    agg_cond_b = {}
    for lk in all_cond_b[0].keys():
        # Vision-only phi (backward compat)
        n_tokens = len(all_cond_b[0][lk]["phi_values"]) if all_cond_b[0][lk]["phi_values"] else 0
        if n_tokens > 0:
            avg_phi = np.zeros(n_tokens)
            for sample_b in all_cond_b:
                if lk in sample_b:
                    phi = np.array(sample_b[lk]["phi_values"])
                    avg_phi[:min(len(phi), n_tokens)] += phi[:n_tokens]
            avg_phi /= len(all_cond_b)
        else:
            avg_phi = np.array([])

        # Check dominant sink's phi across samples
        sink_phi_values = []
        for sample_b in all_cond_b:
            if lk in sample_b:
                full_phi = sample_b[lk].get("full_phi", [])
                if dominant_sink < len(full_phi):
                    sink_phi_values.append(full_phi[dominant_sink])
        avg_sink_phi = float(np.mean(sink_phi_values)) if sink_phi_values else 0.0

        spike_tokens = np.where(avg_phi >= tau)[0].tolist() if len(avg_phi) > 0 else []

        agg_cond_b[lk] = {
            "phi_values": avg_phi.tolist(),
            "spike_tokens": spike_tokens,
            "n_spike_tokens": len(spike_tokens),
            "phi_mean": float(avg_phi.mean()) if len(avg_phi) > 0 else 0.0,
            "phi_max": float(avg_phi.max()) if len(avg_phi) > 0 else 0.0,
            "phi_max_token": int(avg_phi.argmax()) if len(avg_phi) > 0 else -1,
            "dominant_sink_phi": avg_sink_phi,
            "tau": tau,
        }

    # ── Aggregate condition C (now per-sink, not per-vision-token) ──
    agg_cond_c = {}
    if all_cond_c:
        for lk in all_cond_c[0].keys():
            avg_ratio = []
            avg_sink_vn = []
            avg_others_vn = []
            n_low = 0
            n_high = 0
            for sample_c in all_cond_c:
                if lk in sample_c:
                    d = sample_c[lk]
                    avg_ratio.append(d["value_norm_ratio"])
                    avg_sink_vn.append(d["sink_value_norm"])
                    avg_others_vn.append(d["others_value_norm_mean"])
                    if d.get("is_low_contribution"):
                        n_low += 1
                    if d.get("is_high_contribution"):
                        n_high += 1

            agg_cond_c[lk] = {
                "sink_position": dominant_sink,
                "sink_type": dominant_sink_type,
                "avg_value_norm_ratio": float(np.mean(avg_ratio)) if avg_ratio else 0.0,
                "avg_sink_value_norm": float(np.mean(avg_sink_vn)) if avg_sink_vn else 0.0,
                "avg_others_value_norm": float(np.mean(avg_others_vn)) if avg_others_vn else 0.0,
                "n_low_contribution": n_low,
                "n_high_contribution": n_high,
                "n_samples": len(avg_ratio),
                # backward compat
                "token0_value_norm": float(np.mean(avg_sink_vn)) if avg_sink_vn else 0.0,
                "others_value_norm_mean": float(np.mean(avg_others_vn)) if avg_others_vn else 0.0,
                "value_norm_ratio": float(np.mean(avg_ratio)) if avg_ratio else 0.0,
            }

    # ── Determine final verdict ──────────────────────────────────────
    print("\n[5/5] Generating verdict and report...")

    # Count layers where dominant sink passes each condition
    # Condition A: sink is in sink_candidates for >50% of layers
    sink_a_layers = 0
    for lk in agg_cond_a:
        for sc in agg_cond_a[lk].get("sink_candidates", []):
            if sc["position"] == dominant_sink and sc["frequency"] >= len(samples) * 0.5:
                sink_a_layers += 1
                break

    # Condition B: dominant sink has phi >= tau
    sink_b_layers = 0
    for lk in agg_cond_b:
        if agg_cond_b[lk].get("dominant_sink_phi", 0) >= tau:
            sink_b_layers += 1

    # Condition C: low vs high contribution
    sink_c_low = sum(1 for lk in agg_cond_c if agg_cond_c[lk].get("avg_value_norm_ratio", 1.0) < 0.5)
    sink_c_high = sum(1 for lk in agg_cond_c if agg_cond_c[lk].get("avg_value_norm_ratio", 1.0) > 2.0)

    total_layers_a = len(agg_cond_a)
    total_layers_b = len(agg_cond_b)
    total_layers_c = len(agg_cond_c)

    verdict = {
        "dominant_sink_position": dominant_sink,
        "dominant_sink_type": dominant_sink_type,
        "dominant_sink_frequency": f"{dominant_sink_freq}/{len(samples)}",
        "condition_A": {
            "pass": sink_a_layers > total_layers_a * 0.5,
            "layers_passed": sink_a_layers,
            "layers_total": total_layers_a,
            "description": f"Attention consistency: {dominant_sink_type} in top-5 for >80% of text queries",
        },
        "condition_B": {
            "pass": sink_b_layers > total_layers_b * 0.3,
            "layers_passed": sink_b_layers,
            "layers_total": total_layers_b,
            "description": f"Hidden state spike: φ({dominant_sink_type}) ≥ τ={tau}",
        },
        "condition_C": {
            "pass": sink_c_low > total_layers_c * 0.3 if total_layers_c > 0 else False,
            "layers_low_value": sink_c_low,
            "layers_high_value": sink_c_high,
            "layers_total": total_layers_c,
            "description": f"Contribution: {dominant_sink_type} value norm vs others",
            "alternative": f"HIGH value in {sink_c_high}/{total_layers_c} layers → context aggregator" if sink_c_high > 0 else "",
        },
        "is_true_sink": (sink_a_layers > total_layers_a * 0.5) and
                        (sink_b_layers > total_layers_b * 0.3) and
                        (sink_c_low > total_layers_c * 0.3 if total_layers_c > 0 else False),
        "is_context_aggregator": (sink_a_layers > total_layers_a * 0.5) and
                                  (sink_b_layers > total_layers_b * 0.3) and
                                  (sink_c_high > total_layers_c * 0.3 if total_layers_c > 0 else False),
    }

    # ── Print verdict ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SINK VERIFICATION VERDICT — {model_name}")
    print(f"{'='*60}")
    print(f"  Dominant sink: {dominant_sink_type} (position {dominant_sink})")
    print(f"  Detected in {dominant_sink_freq}/{len(samples)} samples")
    for cond_key in ["condition_A", "condition_B", "condition_C"]:
        v = verdict[cond_key]
        status = "PASS" if v["pass"] else "FAIL"
        print(f"  [{status}] {v['description']}")
        n_key = "layers_passed" if "layers_passed" in v else "layers_low_value"
        print(f"         {v[n_key]}/{v['layers_total']} layers")
        if v.get("alternative"):
            print(f"         Note: {v['alternative']}")

    if verdict["is_true_sink"]:
        print(f"\n  *** TRUE ATTENTION SINK (VAR definition) at {dominant_sink_type} ***")
        print(f"  → High attention + hidden spike + LOW contribution = information discarded")
    elif verdict["is_context_aggregator"]:
        print(f"\n  *** CONTEXT AGGREGATOR at {dominant_sink_type} ***")
        print(f"  → High attention + hidden spike + HIGH contribution = information bottleneck")
    else:
        print(f"\n  No clear sink/aggregator pattern at {dominant_sink_type}")
    print(f"{'='*60}\n")

    # ── Save report ───────────────────────────────────────────────────
    report = {
        "model": model_name,
        "n_samples": len(samples),
        "n_vision_tokens": n_vision,
        "tau": tau,
        "sample_sink_positions": sample_sink_positions,
        "verdict": verdict,
        "condition_A": agg_cond_a,
        "condition_B": agg_cond_b,
        "condition_C": agg_cond_c,
    }

    report_path = output_dir / "sink_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Report saved: {report_path}")

    # Generate visualizations (wrapped in try/except for robustness)
    try:
        plot_condition_A(agg_cond_a, output_dir / "condition_A_heatmap.png", model_name, n_vision)
    except Exception as e:
        print(f"  WARNING: plot_condition_A failed: {e}")
    try:
        plot_condition_B(agg_cond_b, output_dir / "condition_B_phi.png", model_name, n_vision, tau=tau)
    except Exception as e:
        print(f"  WARNING: plot_condition_B failed: {e}")
    # Condition C plots skipped — data structure changed to per-sink format

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Verify attention sinks using VAR paper's 3-part definition"
    )
    parser.add_argument("--model", default="openvla-7b", help="Model name from registry")
    parser.add_argument("--device", default="cuda:0", help="Device (e.g., cuda:0)")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of samples to verify")
    parser.add_argument("--tau", type=float, default=20.0, help="Spike threshold τ for condition B")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    args = parser.parse_args()

    run_verification(
        model_name=args.model,
        device=args.device,
        n_samples=args.n_samples,
        tau=args.tau,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
