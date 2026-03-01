#!/usr/bin/env python3
"""Debug trace: follow 1 sample through Gate2 (V=0) and Gap1 (V=mean) pipelines.

Prints ALL intermediate values at each step for verification.

Usage:
    MUJOCO_GL=egl python debug_trace_pipeline.py --device cuda:6
"""
import os
os.environ["MUJOCO_GL"] = "egl"

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

import config
from extract_attention import (
    load_model_from_registry,
    get_layers,
    call_processor,
    detect_token_boundaries,
)
from data_sampler import load_balanced_samples
from verify_attention_sinks import get_wov_matrix, SinkVerificationHookManager
from contribution.compute import (
    compute_perhead_contribution,
    aggregate_contributions,
    find_topk_candidates,
)
from contribution.causal import (
    ValueZeroHook,
    ValueMeanHook,
    compute_output_kl,
    compute_top1_change_rate,
)

P = lambda *a, **kw: print(*a, **kw, flush=True)

BANNER = lambda title: P(f"\n{'='*72}\n  STEP: {title}\n{'='*72}")


def prepare_inputs(processor, model, model_cfg, sample, device):
    """Prepare model inputs from a sample (shared helper)."""
    prompt = model_cfg.prompt_template.format(instruction=sample["instruction"])
    inputs = call_processor(
        processor, prompt, sample["image"], model_cfg, return_tensors="pt"
    ).to(device)
    if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

    fwd_kwargs = {k: v for k, v in inputs.items()}
    fwd_kwargs["use_cache"] = False
    # SpatialVLA intrinsic (not needed for ecot, but keep for generality)
    if model_cfg.architecture == "gemma2" and "intrinsic" not in fwd_kwargs:
        fwd_kwargs["intrinsic"] = torch.tensor(
            [[[218.26, 0.0, 111.83], [0.0, 218.26, 111.79], [0.0, 0.0, 1.0]]],
            device=device, dtype=torch.float32,
        )
    return inputs, fwd_kwargs


def main():
    parser = argparse.ArgumentParser(description="Debug trace pipeline")
    parser.add_argument("--device", type=str, default="cuda:6")
    parser.add_argument("--model", type=str, default="ecot-7b")
    parser.add_argument("--deep_layer", type=int, default=None,
                        help="Specific deep layer to inspect (default: last-2)")
    args = parser.parse_args()

    device = args.device
    model_name = args.model

    # ─────────────────────────────────────────────────────────────────
    # STEP 1: Load model
    # ─────────────────────────────────────────────────────────────────
    BANNER("1. Load model")
    processor, model, model_cfg = load_model_from_registry(model_name, device)
    model.eval()

    num_layers = model_cfg.num_layers
    num_heads = model_cfg.num_heads
    hidden_dim = model_cfg.hidden_dim
    head_dim = hidden_dim // num_heads
    num_kv_heads = getattr(model_cfg, "num_kv_heads", None) or num_heads
    deep_layers = list(range(max(0, num_layers - 10), num_layers))
    inspect_layer = args.deep_layer if args.deep_layer is not None else num_layers - 2

    P(f"  Model: {model_name}")
    P(f"  num_layers={num_layers}, num_heads={num_heads}, hidden_dim={hidden_dim}")
    P(f"  head_dim={head_dim}, num_kv_heads={num_kv_heads}")
    P(f"  deep_layers={deep_layers}")
    P(f"  Inspect layer for detailed analysis: {inspect_layer}")

    # ─────────────────────────────────────────────────────────────────
    # STEP 2: Load 1 Bridge V2 sample
    # ─────────────────────────────────────────────────────────────────
    BANNER("2. Load 1 Bridge V2 sample")
    samples = load_balanced_samples(config.DATA_CACHE_DIR, n_per_skill=1, seed=42)
    sample = samples[0]
    P(f"  Instruction: {sample['instruction']}")
    P(f"  Skill: {sample.get('skill', 'unknown')}")
    P(f"  Episode ID: {sample.get('episode_id', '?')}")
    P(f"  Image size: {sample['image'].size}")

    # ─────────────────────────────────────────────────────────────────
    # STEP 3: detect_token_boundaries
    # ─────────────────────────────────────────────────────────────────
    BANNER("3. detect_token_boundaries")
    with torch.no_grad():
        bounds = detect_token_boundaries(
            processor, model, sample["image"], sample["instruction"],
            device, model_cfg,
        )
    P(f"  vision_start  = {bounds['vision_start']}")
    P(f"  vision_end    = {bounds['vision_end']}")
    P(f"  text_start    = {bounds['text_start']}")
    P(f"  text_end      = {bounds['text_end']}")
    P(f"  total_seq_len = {bounds['total_seq_len']}")
    P(f"  num_vision_tokens = {bounds.get('num_vision_tokens', '?')}")
    P(f"  num_text_tokens   = {bounds.get('num_text_tokens', '?')}")
    P(f"  text_ranges       = {bounds.get('text_ranges', '?')}")

    vs = bounds["vision_start"]
    ve = bounds["vision_end"]
    seq_len = bounds["total_seq_len"]

    # ─────────────────────────────────────────────────────────────────
    # STEP 4: SinkVerificationHookManager — capture attention + hidden
    # ─────────────────────────────────────────────────────────────────
    BANNER("4. SinkVerificationHookManager forward pass")
    hook_mgr = SinkVerificationHookManager(model, model_cfg)
    hook_mgr.register_hooks()

    inputs, fwd_kwargs = prepare_inputs(processor, model, model_cfg, sample, device)
    with torch.no_grad():
        out_clean = model(**fwd_kwargs)
    hook_mgr.remove_hooks()

    P(f"\n  Captured attention_weights for {len(hook_mgr.attention_weights)} layers")
    for li in sorted(hook_mgr.attention_weights.keys()):
        P(f"    Layer {li:2d}: attn shape = {tuple(hook_mgr.attention_weights[li].shape)}")
    P(f"\n  Captured hidden_states for {len(hook_mgr.hidden_states)} layers")
    for li in sorted(hook_mgr.hidden_states.keys()):
        P(f"    Layer {li:2d}: hidden shape = {tuple(hook_mgr.hidden_states[li].shape)}")

    # Top-5 attention positions from the last layer (query = last text token)
    last_layer = max(hook_mgr.attention_weights.keys())
    attn_last = hook_mgr.attention_weights[last_layer]  # (H, seq, seq)
    last_text_tok = seq_len - 1  # last token in sequence
    P(f"\n  Last layer = {last_layer}, query position = {last_text_tok} (last text token)")

    # Mean over heads for that query
    attn_last_query = attn_last[:, last_text_tok, :].mean(dim=0)  # (seq,)
    top5_vals, top5_idx = attn_last_query.topk(5)
    P(f"  Top-5 attention positions (mean over heads):")
    for rank, (val, idx) in enumerate(zip(top5_vals.tolist(), top5_idx.tolist())):
        region = "vision" if vs <= idx < ve else "text"
        rel = idx - vs if vs <= idx < ve else idx
        P(f"    rank {rank}: pos={idx} ({region}, rel={rel}), attn={val:.6f}")

    # ─────────────────────────────────────────────────────────────────
    # STEP 5: get_wov_matrix for inspect_layer
    # ─────────────────────────────────────────────────────────────────
    BANNER(f"5. get_wov_matrix (layer {inspect_layer})")
    w_v, w_o = get_wov_matrix(model, model_cfg, inspect_layer)
    P(f"  v_weight shape: {tuple(w_v.shape)}  (expected: ({num_kv_heads * head_dim}, {hidden_dim}))")
    P(f"  o_weight shape: {tuple(w_o.shape)}  (expected: ({hidden_dim}, {num_heads * head_dim}))")

    # BUG-3 check: W_O @ W_V dimension compatibility
    o_input_dim = w_o.shape[1]    # num_heads_q * head_dim
    v_output_dim = w_v.shape[0]   # num_heads_kv * head_dim
    dim_match = (o_input_dim == v_output_dim)
    P(f"  w_o input dim  = {o_input_dim}")
    P(f"  w_v output dim = {v_output_dim}")
    P(f"  W_O @ W_V dimension match (no GQA expansion needed): {dim_match}")
    if not dim_match:
        P(f"  GQA detected: group ratio = {o_input_dim // v_output_dim}")

    # ─────────────────────────────────────────────────────────────────
    # STEP 6: compute_perhead_contribution
    # ─────────────────────────────────────────────────────────────────
    BANNER(f"6. compute_perhead_contribution (layer {inspect_layer})")

    attn_l = hook_mgr.attention_weights[inspect_layer]
    if attn_l.dim() == 4:
        attn_l = attn_l[0]
    P(f"  attn shape: {tuple(attn_l.shape)}")

    # Use hidden state from previous layer (l-1) as input to layer l
    prev_layer = max(0, inspect_layer - 1)
    hidden_l = hook_mgr.hidden_states.get(prev_layer, hook_mgr.hidden_states[inspect_layer]).cpu().float()
    P(f"  hidden shape (layer {prev_layer} output): {tuple(hidden_l.shape)}")

    # Query positions = last few text tokens (action prediction)
    query_positions = [seq_len - 1]
    P(f"  query_positions: {query_positions}")

    contrib = compute_perhead_contribution(attn_l, hidden_l, w_v, w_o, query_positions)
    P(f"  Contribution tensor shape: {tuple(contrib.shape)}  (expected: (H, n_query, seq))")

    # Per-head top-1 position
    P(f"\n  Per-head top-1 position (is it always the same?):")
    head_top1_positions = []
    for h in range(min(num_heads, contrib.shape[0])):
        top1_pos = contrib[h, 0, :].argmax().item()
        top1_val = contrib[h, 0, :].max().item()
        head_top1_positions.append(top1_pos)
        if h < 8 or h == num_heads - 1:  # Print first 8 + last
            region = "vision" if vs <= top1_pos < ve else "text"
            P(f"    head {h:2d}: top1_pos={top1_pos} ({region}), norm={top1_val:.6f}")
        elif h == 8:
            P(f"    ... (heads 8-{num_heads-2} omitted, showing last) ...")

    unique_positions = set(head_top1_positions)
    P(f"\n  Unique top-1 positions across all heads: {sorted(unique_positions)}")
    P(f"  All heads agree: {len(unique_positions) == 1}")

    # Check: contribution ranking vs attention ranking for each head
    P(f"\n  Contribution ranking vs Attention ranking (per head):")
    rank_agreements = 0
    for h in range(min(num_heads, contrib.shape[0])):
        c_top1 = contrib[h, 0, :].argmax().item()
        a_top1 = attn_l[h, query_positions[0], :].argmax().item()
        agree = (c_top1 == a_top1)
        rank_agreements += int(agree)
        if h < 4 or h == num_heads - 1:
            P(f"    head {h:2d}: attn_top1={a_top1}, contrib_top1={c_top1}, agree={agree}")
        elif h == 4:
            P(f"    ... (remaining heads omitted) ...")
    P(f"  Agreement rate: {rank_agreements}/{num_heads} = {rank_agreements/num_heads:.1%}")

    # ─────────────────────────────────────────────────────────────────
    # STEP 7: aggregate_contributions
    # ─────────────────────────────────────────────────────────────────
    BANNER(f"7. aggregate_contributions (layer {inspect_layer})")
    a_tilde, c_tilde = aggregate_contributions(attn_l, contrib, query_positions)
    P(f"  a_tilde shape: {tuple(a_tilde.shape)}, sum={a_tilde.sum().item():.6f}")
    P(f"  c_tilde shape: {tuple(c_tilde.shape)}, sum={c_tilde.sum().item():.6f}")

    # Top-5 for a_tilde
    a_top5_vals, a_top5_idx = a_tilde.topk(5)
    P(f"\n  a_tilde top-5:")
    for rank, (val, idx) in enumerate(zip(a_top5_vals.tolist(), a_top5_idx.tolist())):
        region = "vision" if vs <= idx < ve else "text"
        rel = idx - vs if vs <= idx < ve else idx
        P(f"    rank {rank}: pos={idx} ({region}, rel={rel}), share={val:.6f}")

    # Top-5 for c_tilde
    c_top5_vals, c_top5_idx = c_tilde.topk(5)
    P(f"\n  c_tilde top-5:")
    for rank, (val, idx) in enumerate(zip(c_top5_vals.tolist(), c_top5_idx.tolist())):
        region = "vision" if vs <= idx < ve else "text"
        rel = idx - vs if vs <= idx < ve else idx
        P(f"    rank {rank}: pos={idx} ({region}, rel={rel}), share={val:.6f}")

    # Mismatch
    a_peak = a_top5_idx[0].item()
    c_peak = c_top5_idx[0].item()
    mismatch = abs(a_tilde[a_peak].item() - c_tilde[a_peak].item())
    P(f"\n  Mismatch (|a_tilde[a_peak] - c_tilde[a_peak]|): {mismatch:.6f}")
    P(f"  a_tilde peak position: {a_peak}")
    P(f"  c_tilde peak position: {c_peak}")
    P(f"  Top-1 position SAME between a_tilde and c_tilde: {a_peak == c_peak}")

    # ─────────────────────────────────────────────────────────────────
    # STEP 8: ValueZeroHook (Gate2: V=0 ablation at position 0)
    # ─────────────────────────────────────────────────────────────────
    BANNER("8. ValueZeroHook (V=0 ablation at position 0 = A_mode target)")

    # Anchor position = vision_start (position 0 relative to vision)
    anchor_abs = vs  # vision[0] absolute position
    P(f"  Target position (anchor_abs): {anchor_abs}")
    P(f"  Deep layers for hook: {deep_layers}")

    # Original logits (from the clean forward we already did)
    logits_orig = out_clean.logits[0, -1, :].detach().cpu().float()
    top5_orig_vals, top5_orig_idx = logits_orig.topk(5)
    P(f"\n  Original logits:")
    P(f"    argmax = {logits_orig.argmax().item()}")
    P(f"    top-5 indices: {top5_orig_idx.tolist()}")
    P(f"    top-5 values:  {[f'{v:.4f}' for v in top5_orig_vals.tolist()]}")

    # V=0 forward
    vzero = ValueZeroHook(target_positions=[anchor_abs], target_layers=deep_layers)
    vzero.register(model, model_cfg, get_layers)
    with torch.no_grad():
        out_vzero = model(**fwd_kwargs)
    vzero.remove()

    logits_vzero = out_vzero.logits[0, -1, :].detach().cpu().float()
    top5_vz_vals, top5_vz_idx = logits_vzero.topk(5)
    P(f"\n  V=0 ablated logits:")
    P(f"    argmax = {logits_vzero.argmax().item()}")
    P(f"    top-5 indices: {top5_vz_idx.tolist()}")
    P(f"    top-5 values:  {[f'{v:.4f}' for v in top5_vz_vals.tolist()]}")
    P(f"    hook_fired = {vzero._sanity_changed}")

    kl_vzero = compute_output_kl(logits_orig, logits_vzero)
    top1_changed_vz = (logits_orig.argmax().item() != logits_vzero.argmax().item())
    P(f"\n  KL divergence (V=0): {kl_vzero:.6f}")
    P(f"  top1_changed: {top1_changed_vz}")

    # ─────────────────────────────────────────────────────────────────
    # STEP 9: ValueMeanHook (Gap1: V=mean ablation)
    # ─────────────────────────────────────────────────────────────────
    BANNER("9. ValueMeanHook (V=mean ablation)")

    # Collect mean V from remaining positions in this single sample
    # (In production, we'd use N calibration samples; here we use 1 for debug)
    P("  Collecting per-layer mean V projections from remaining vision tokens...")

    layer_means = {}
    collector_handles = []
    layers_list = get_layers(model, model_cfg)

    for layer_idx in deep_layers:
        layer = layers_list[layer_idx]
        attn_mod = layer.self_attn

        if hasattr(attn_mod, "v_proj"):
            def make_collector_hook(lidx):
                def hook_fn(module, args, output):
                    # Mean over vision tokens EXCLUDING anchor (position 0)
                    start = max(vs + 1, 1)
                    end = min(ve, output.shape[1])
                    if start < end:
                        mean_v = output[0, start:end, :].detach().float().mean(dim=0)
                    else:
                        mean_v = output[0, vs:end, :].detach().float().mean(dim=0)
                    layer_means[lidx] = mean_v.cpu()
                return hook_fn
            h = attn_mod.v_proj.register_forward_hook(make_collector_hook(layer_idx))
            collector_handles.append(h)

    with torch.no_grad():
        model(**fwd_kwargs)
    for h in collector_handles:
        h.remove()

    P(f"  Collected means for {len(layer_means)} layers")
    for lidx in sorted(layer_means.keys()):
        mv = layer_means[lidx]
        P(f"    Layer {lidx}: mean_v shape={tuple(mv.shape)}, "
          f"norm={mv.norm().item():.4f}, mean={mv.mean().item():.6f}")

    # V=mean forward
    vmean = ValueMeanHook(target_positions=[anchor_abs], target_layers=deep_layers)
    vmean.set_layer_means(layer_means)
    vmean.register(model, model_cfg, get_layers)
    with torch.no_grad():
        out_vmean = model(**fwd_kwargs)
    vmean.remove()

    logits_vmean = out_vmean.logits[0, -1, :].detach().cpu().float()
    top5_vm_vals, top5_vm_idx = logits_vmean.topk(5)
    P(f"\n  V=mean ablated logits:")
    P(f"    argmax = {logits_vmean.argmax().item()}")
    P(f"    top-5 indices: {top5_vm_idx.tolist()}")
    P(f"    top-5 values:  {[f'{v:.4f}' for v in top5_vm_vals.tolist()]}")
    P(f"    hook_fired = {vmean._sanity_changed}")

    kl_vmean = compute_output_kl(logits_orig, logits_vmean)
    top1_changed_vm = (logits_orig.argmax().item() != logits_vmean.argmax().item())
    P(f"\n  KL divergence (V=mean): {kl_vmean:.6f}")
    P(f"  top1_changed: {top1_changed_vm}")

    # Comparison summary
    P(f"\n  === V=0 vs V=mean comparison ===")
    P(f"  V=0:    KL={kl_vzero:.6f}, flip={top1_changed_vz}")
    P(f"  V=mean: KL={kl_vmean:.6f}, flip={top1_changed_vm}")
    P(f"  V=mean > V=0: {kl_vmean > kl_vzero}")
    if kl_vzero > 0:
        P(f"  Ratio (V=mean / V=0): {kl_vmean / kl_vzero:.2f}x")

    # ─────────────────────────────────────────────────────────────────
    # STEP 10: BUG-3 verification — per-head W_OV decomposition
    # ─────────────────────────────────────────────────────────────────
    BANNER(f"10. BUG-3 Verification: per-head W_OV (layer {inspect_layer})")

    P(f"  Goal: Check if the current all-heads W_OV computation differs from")
    P(f"         summing individual per-head W_OV contributions.\n")

    # Get full W_V and W_O again
    w_v_full, w_o_full = get_wov_matrix(model, model_cfg, inspect_layer)
    P(f"  w_v_full shape: {tuple(w_v_full.shape)}")
    P(f"  w_o_full shape: {tuple(w_o_full.shape)}")

    hidden_for_bug3 = hidden_l.float()  # (seq, D)
    P(f"  hidden shape: {tuple(hidden_for_bug3.shape)}")

    # Current all-heads computation (from compute_perhead_contribution logic):
    o_in = w_o_full.shape[1]    # num_heads_q * head_dim
    v_out = w_v_full.shape[0]   # num_heads_kv * head_dim

    if o_in == v_out:
        w_ov_full = (w_o_full @ w_v_full).float()
        value_vectors_allheads = hidden_for_bug3 @ w_ov_full.T
        P(f"  All-heads W_OV = W_O @ W_V, shape: {tuple(w_ov_full.shape)}")
        P(f"  All-heads value_vectors shape: {tuple(value_vectors_allheads.shape)}")
    else:
        num_groups = o_in // v_out
        P(f"  GQA mode: groups={num_groups}")
        v_out_raw = hidden_for_bug3 @ w_v_full.T.float()
        v_out_expanded = v_out_raw.reshape(seq_len, -1, head_dim).repeat(1, num_groups, 1).reshape(seq_len, -1)
        value_vectors_allheads = v_out_expanded @ w_o_full.T.float()
        P(f"  All-heads value_vectors shape: {tuple(value_vectors_allheads.shape)}")

    # Per-head W_OV for head 0 and head 16
    test_heads = [0, min(16, num_heads - 1)]
    P(f"\n  Testing individual heads: {test_heads}")

    for h_idx in test_heads:
        P(f"\n  --- Head {h_idx} ---")
        # Extract per-head slices
        # W_V_h: rows [h*head_dim : (h+1)*head_dim] of w_v  (for MHA; GQA uses KV head mapping)
        # W_O_h: cols [h*head_dim : (h+1)*head_dim] of w_o

        if o_in == v_out:
            # MHA: straightforward slicing
            v_h_start = h_idx * head_dim
            v_h_end = (h_idx + 1) * head_dim
            o_h_start = h_idx * head_dim
            o_h_end = (h_idx + 1) * head_dim

            w_v_h = w_v_full[v_h_start:v_h_end, :].float()   # (head_dim, D)
            w_o_h = w_o_full[:, o_h_start:o_h_end].float()    # (D, head_dim)

            P(f"    W_V_h shape: {tuple(w_v_h.shape)}")
            P(f"    W_O_h shape: {tuple(w_o_h.shape)}")

            # Per-head value vector: hidden @ W_V_h^T @ W_O_h^T
            # = hidden @ (head_dim, D)^T = hidden @ (D, head_dim) → (seq, head_dim)
            # then @ (D, head_dim)^T = @ (head_dim, D) → (seq, D)
            v_h = hidden_for_bug3 @ w_v_h.T          # (seq, head_dim)
            value_vectors_h = v_h @ w_o_h.T           # (seq, D)
        else:
            # GQA: map Q head to KV head
            kv_heads = v_out // head_dim
            kv_h_idx = h_idx // (num_heads // kv_heads)

            v_h_start = kv_h_idx * head_dim
            v_h_end = (kv_h_idx + 1) * head_dim
            o_h_start = h_idx * head_dim
            o_h_end = (h_idx + 1) * head_dim

            w_v_h = w_v_full[v_h_start:v_h_end, :].float()
            w_o_h = w_o_full[:, o_h_start:o_h_end].float()

            P(f"    W_V_h shape: {tuple(w_v_h.shape)}  (KV head {kv_h_idx})")
            P(f"    W_O_h shape: {tuple(w_o_h.shape)}")

            v_h = hidden_for_bug3 @ w_v_h.T
            value_vectors_h = v_h @ w_o_h.T

        P(f"    Per-head value_vectors_h shape: {tuple(value_vectors_h.shape)}")
        P(f"    Per-head value_vectors_h norm (pos 0): {value_vectors_h[0].norm().item():.6f}")
        P(f"    Per-head value_vectors_h norm (pos -1): {value_vectors_h[-1].norm().item():.6f}")

    # Now compare: sum of per-head value_vectors vs all-heads value_vectors
    P(f"\n  --- Summing ALL per-head value vectors ---")
    sum_perhead = torch.zeros_like(value_vectors_allheads)

    for h_idx in range(num_heads):
        if o_in == v_out:
            v_h_s = h_idx * head_dim
            v_h_e = (h_idx + 1) * head_dim
            o_h_s = h_idx * head_dim
            o_h_e = (h_idx + 1) * head_dim
            w_v_h = w_v_full[v_h_s:v_h_e, :].float()
            w_o_h = w_o_full[:, o_h_s:o_h_e].float()
        else:
            kv_heads = v_out // head_dim
            kv_h_idx = h_idx // (num_heads // kv_heads)
            v_h_s = kv_h_idx * head_dim
            v_h_e = (kv_h_idx + 1) * head_dim
            o_h_s = h_idx * head_dim
            o_h_e = (h_idx + 1) * head_dim
            w_v_h = w_v_full[v_h_s:v_h_e, :].float()
            w_o_h = w_o_full[:, o_h_s:o_h_e].float()

        v_h = hidden_for_bug3 @ w_v_h.T
        val_h = v_h @ w_o_h.T
        sum_perhead += val_h

    # Compare norms
    norm_allheads = value_vectors_allheads.norm().item()
    norm_sumperhead = sum_perhead.norm().item()
    diff = (value_vectors_allheads - sum_perhead).norm().item()
    reldiff = diff / max(norm_allheads, 1e-10)

    P(f"  All-heads value_vectors norm:     {norm_allheads:.6f}")
    P(f"  Sum-of-per-head norm:             {norm_sumperhead:.6f}")
    P(f"  Absolute difference norm:         {diff:.6f}")
    P(f"  Relative difference:              {reldiff:.8f}")
    P(f"  Are they different (rel > 1e-5)?  {reldiff > 1e-5}")

    # Also compare at specific positions
    for pos in [0, vs, ve - 1, seq_len - 1]:
        if pos < seq_len:
            d = (value_vectors_allheads[pos] - sum_perhead[pos]).norm().item()
            a = value_vectors_allheads[pos].norm().item()
            P(f"    pos={pos}: allheads_norm={a:.6f}, diff_norm={d:.8f}, "
              f"rel={d/max(a,1e-10):.8f}")

    P(f"\n  BUG-3 CONCLUSION: ", end="")
    if reldiff > 1e-3:
        P("SIGNIFICANT MISMATCH -- per-head decomposition != all-heads computation")
        P("  The current compute_perhead_contribution uses all-heads W_OV jointly,")
        P("  which may not correctly represent per-head contributions.")
    elif reldiff > 1e-5:
        P("MINOR MISMATCH (likely numerical precision)")
    else:
        P("MATCH -- per-head sum == all-heads (no BUG-3 issue for this architecture)")

    # ─────────────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────────────────────────────────────────
    P(f"\n{'='*72}")
    P(f"  FINAL SUMMARY")
    P(f"{'='*72}")
    P(f"  Model: {model_name}")
    P(f"  Sample: '{sample['instruction']}' (skill={sample.get('skill','?')})")
    P(f"  Seq len: {seq_len} (vision [{vs}:{ve}], {ve-vs} tokens)")
    P(f"  Inspect layer: {inspect_layer}")
    P(f"  a_tilde peak: pos={a_peak} (share={a_tilde[a_peak].item():.4f})")
    P(f"  c_tilde peak: pos={c_peak} (share={c_tilde[c_peak].item():.4f})")
    P(f"  a==c peak: {a_peak == c_peak}")
    P(f"  V=0  KL: {kl_vzero:.6f}, flip={top1_changed_vz}")
    P(f"  V=mean KL: {kl_vmean:.6f}, flip={top1_changed_vm}")
    P(f"  BUG-3 relative diff: {reldiff:.8f}")
    P(f"{'='*72}\n")

    # Cleanup
    del model
    torch.cuda.empty_cache()
    P("Done. GPU memory freed.")


if __name__ == "__main__":
    main()
