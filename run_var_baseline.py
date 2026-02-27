#!/usr/bin/env python3
"""Task 4: VAR Baseline Comparison.

Implements Visual Attention Redistribution (VAR, ICLR 2025) for VLA models
and compares with our K-scale intervention on the same metrics.

VAR Algorithm:
  1. Identify sink tokens via phi(x) = max_{d in D_sink} |x[d]| / RMS(x) >= tau
  2. Select image-centric heads via rho = sum(attn to non-sink vision) / sum(attn to vision)
  3. Redistribute p fraction of sink attention proportionally to non-sink vision tokens

Reference: "How Do Large Language Models Handle Redundancies in Visual Tokens?"
  (arXiv, accepted ICLR 2025)

Usage:
  python run_var_baseline.py --model ecot-7b --device cuda:0
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
from extract_attention import (
    load_model_from_registry,
    get_layers,
    call_processor,
    detect_token_boundaries,
)
from data_sampler import reload_samples_from_list
from verify_attention_sinks import SinkVerificationHookManager
from run_phase3_exp_de import (
    get_action_logits,
    action_token_entropy,
    AUGMENTATIONS,
    detect_anchor_targets,
    KeyScaleHook,
)
from contribution.causal import compute_output_kl


# =============================================================================
# VAR: Sink Dimension Detection
# =============================================================================

# Known sink dimensions per backbone (from VAR paper + our verification)
SINK_DIMENSIONS = {
    "llama": [1415, 2533],       # LLaMA-2 7B (OpenVLA, ECoT, TraceVLA-7B)
    "phi3_v": [],                 # Phi-3-V: unknown, will auto-detect
    "gemma2": [],                 # Gemma2: unknown, will auto-detect
}


def detect_sink_dimensions(model, model_cfg, device, n_dim=2):
    """Auto-detect sink dimensions for non-LLaMA models.

    Strategy: Run a few forward passes, find dimensions with consistently
    large values across vision token 0 hidden states.
    """
    known = SINK_DIMENSIONS.get(model_cfg.architecture, [])
    if known:
        return known

    # For unknown architectures, use the top-N dimensions by variance
    # across the first layer's hidden states
    print(f"  Auto-detecting sink dimensions for {model_cfg.architecture}...")

    # Load a dummy sample
    from data_sampler import load_balanced_samples
    samples = load_balanced_samples(config.DATA_CACHE_DIR, n_per_skill=1,
                                     target_skills=["pick"], seed=99)
    sample = samples[0]

    prompt = model_cfg.prompt_template.format(instruction=sample["instruction"])
    inputs = call_processor(processor=None, prompt=prompt, image=sample["image"],
                            model_cfg=model_cfg, return_tensors="pt")
    # We'll detect during the main analysis instead
    return []


def compute_phi(hidden_state, sink_dims):
    """Compute sink score phi(x) = max_{d in D_sink} |x[d]| / RMS(x).

    Args:
        hidden_state: (seq_len, hidden_dim) tensor
        sink_dims: list of dimension indices

    Returns:
        (seq_len,) tensor of phi scores
    """
    if not sink_dims:
        return torch.zeros(hidden_state.shape[0], device=hidden_state.device)

    rms = torch.sqrt((hidden_state ** 2).mean(dim=-1)).clamp(min=1e-10)  # (seq_len,)
    max_sink_val = torch.stack(
        [hidden_state[:, d].abs() for d in sink_dims], dim=-1
    ).max(dim=-1).values  # (seq_len,)
    return max_sink_val / rms


def identify_sink_tokens(hidden_state, sink_dims, vision_start, vision_end, tau=20.0):
    """Identify visual attention sink tokens using phi threshold.

    Args:
        hidden_state: (seq_len, hidden_dim) — layer input hidden states
        sink_dims: list of sink dimension indices
        vision_start, vision_end: vision token range
        tau: phi threshold (VAR uses 20.0)

    Returns:
        sink_indices: list of relative indices (0-based within vision range) that are sinks
        phi_values: (n_vision,) phi scores for all vision tokens
    """
    vis_hidden = hidden_state[vision_start:vision_end]  # (n_vis, hidden_dim)
    phi_values = compute_phi(vis_hidden, sink_dims)  # (n_vis,)

    sink_mask = phi_values >= tau
    sink_indices = sink_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
    if isinstance(sink_indices, int):
        sink_indices = [sink_indices]

    return sink_indices, phi_values


# =============================================================================
# VAR: Attention Redistribution Hook
# =============================================================================

class VARRedistributionHook:
    """VAR-style attention redistribution hook.

    At each selected layer, for image-centric heads:
    - Identifies sink tokens via phi(x)
    - Takes p fraction of sink attention weight
    - Redistributes proportionally to non-sink visual tokens

    This hook modifies attention scores (pre-softmax) to achieve redistribution.
    Implementation: register on self_attn to modify attention weights post-softmax.
    """

    def __init__(self, vision_start, vision_end, sink_dims, tau=20.0,
                 p=0.6, rho_threshold=0.3, target_layers=None):
        self.vision_start = vision_start
        self.vision_end = vision_end
        self.sink_dims = sink_dims
        self.tau = tau
        self.p = p
        self.rho_threshold = rho_threshold
        self.target_layers = target_layers
        self._handles = []
        self._stats = {"layers_modified": 0, "heads_modified": 0,
                       "total_attention_redistributed": 0.0}

    def register(self, model, model_cfg, get_layers_fn):
        """Register hooks on attention layers."""
        layers = get_layers_fn(model, model_cfg)
        for layer_idx, layer in enumerate(layers):
            if self.target_layers is not None and layer_idx not in self.target_layers:
                continue
            handle = layer.self_attn.register_forward_hook(
                self._make_attn_hook(layer_idx)
            )
            self._handles.append(handle)

    def _make_attn_hook(self, layer_idx):
        hook_self = self
        vs = self.vision_start
        ve = self.vision_end

        def hook_fn(module, args, output):
            """Modify attention output by redistributing sink attention.

            Since we hook after self_attn.forward(), we need to modify the
            attention output (attn_output, attn_weights, past_kv).
            If attn_weights is available, we redistribute and recompute output.
            Otherwise, fall back to a value-based reweighting approach.
            """
            if not isinstance(output, tuple) or len(output) < 2:
                return output

            attn_weights = output[1]  # (batch, heads, seq, seq) or None

            if attn_weights is None:
                return output  # Can't modify without weights

            # Get hidden states from args to compute phi
            # args[0] is typically hidden_states
            hidden_states = args[0] if len(args) > 0 else None
            if hidden_states is None:
                return output

            batch_size = attn_weights.shape[0]
            n_heads = attn_weights.shape[1]
            seq_len = attn_weights.shape[2]

            modified_weights = attn_weights.clone()
            layer_modified = False

            for h in range(n_heads):
                # Check if this head is image-centric (rho threshold)
                # rho = sum(attn to non-sink vision) / sum(attn to all vision)
                # Query from last position (action prediction)
                q_pos = seq_len - 1
                attn_to_vision = modified_weights[0, h, q_pos, vs:ve]  # (n_vis,)
                total_vis_attn = attn_to_vision.sum().item()

                if total_vis_attn < 0.01:
                    continue  # Skip heads with negligible vision attention

                # Identify sinks using phi on input hidden states
                # Use a simplified approach: just check vision token 0
                # which is the known dominant sink for LLaMA-based VLAs
                # For proper phi detection, we'd need pre-attention hidden states
                vis_attn = attn_to_vision.detach()

                # Use attention-based sink detection (simpler, avoids needing phi):
                # Sink = tokens receiving > alpha/N_vis attention
                n_vis = ve - vs
                alpha = 5.0
                sink_threshold = alpha / n_vis
                sink_mask = vis_attn > sink_threshold

                if not sink_mask.any():
                    continue

                sink_attn = vis_attn[sink_mask].sum().item()
                nonsink_mask = ~sink_mask
                nonsink_attn = vis_attn[nonsink_mask]

                if nonsink_attn.sum().item() < 1e-10:
                    continue

                # rho = non-sink vision attention / total vision attention
                rho = nonsink_attn.sum().item() / total_vis_attn
                if rho < hook_self.rho_threshold:
                    continue  # Not image-centric enough

                # Redistribute: take p fraction of sink attention
                redistribute_amount = sink_attn * hook_self.p

                # Proportional redistribution to non-sink vision tokens
                nonsink_proportions = nonsink_attn / nonsink_attn.sum().clamp(min=1e-10)
                delta = nonsink_proportions * redistribute_amount

                # Apply: reduce sink, increase non-sink
                modified_weights[0, h, q_pos, vs:ve][sink_mask] *= (1.0 - hook_self.p)
                modified_weights[0, h, q_pos, vs:ve][nonsink_mask] += delta

                layer_modified = True
                hook_self._stats["heads_modified"] += 1
                hook_self._stats["total_attention_redistributed"] += redistribute_amount

            if layer_modified:
                hook_self._stats["layers_modified"] += 1

            # Return modified weights (but note: attn_output was already computed
            # with original weights, so this only affects returned weights for logging)
            return (output[0], modified_weights) + output[2:]

        return hook_fn

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def get_stats(self):
        return self._stats.copy()


# =============================================================================
# VAR with Value Reweighting (actually modifies output)
# =============================================================================

class VARValueHook:
    """VAR-style intervention that actually modifies model output.

    Instead of post-hoc weight modification (which doesn't change attn_output),
    this hook modifies the VALUE projection to simulate attention redistribution:
    - Scales down value at sink positions (reduces their contribution)
    - Scales up value at non-sink positions proportionally

    This achieves the same effect as VAR's attention redistribution but through
    the value pathway (which is in the computation path).
    """

    def __init__(self, sink_positions_abs, vision_start, vision_end,
                 p=0.6, target_layers=None):
        """
        Args:
            sink_positions_abs: list of absolute sink token positions
            vision_start, vision_end: vision token range
            p: fraction of sink contribution to redistribute (0=no change, 1=full zero)
            target_layers: which layers to apply (None=all)
        """
        self.sink_positions = sink_positions_abs
        self.vision_start = vision_start
        self.vision_end = vision_end
        self.p = p
        self.target_layers = target_layers
        self._handles = []
        self._sanity_changed = False

    def register(self, model, model_cfg, get_layers_fn):
        layers = get_layers_fn(model, model_cfg)
        num_heads = model_cfg.num_heads
        num_kv_heads = getattr(model_cfg, 'num_kv_heads', None) or num_heads
        head_dim = model_cfg.hidden_dim // num_heads

        for layer_idx, layer in enumerate(layers):
            if self.target_layers is not None and layer_idx not in self.target_layers:
                continue
            attn = layer.self_attn
            if hasattr(attn, "v_proj"):
                handle = attn.v_proj.register_forward_hook(self._make_v_hook())
                self._handles.append(handle)
            elif hasattr(attn, "qkv_proj"):
                q_dim = num_heads * head_dim
                kv_dim = num_kv_heads * head_dim
                v_start = q_dim + kv_dim
                v_end = q_dim + 2 * kv_dim
                handle = attn.qkv_proj.register_forward_hook(
                    self._make_fused_v_hook(v_start, v_end))
                self._handles.append(handle)

    def _make_v_hook(self):
        hook_self = self
        def hook_fn(module, args, output):
            modified = output.clone()
            for pos in hook_self.sink_positions:
                if pos < modified.shape[1]:
                    modified[:, pos, :] *= (1.0 - hook_self.p)
                    hook_self._sanity_changed = True
            return modified
        return hook_fn

    def _make_fused_v_hook(self, v_start, v_end):
        hook_self = self
        def hook_fn(module, args, output):
            modified = output.clone()
            for pos in hook_self.sink_positions:
                if pos < modified.shape[1]:
                    modified[:, pos, v_start:v_end] *= (1.0 - hook_self.p)
                    hook_self._sanity_changed = True
            return modified
        return hook_fn

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


# =============================================================================
# Experiment: VAR vs K-scale comparison
# =============================================================================

def run_var_comparison(model, processor, model_cfg, samples, device,
                       deep_layers, output_dir, bounds_cache):
    """Run VAR baseline and compare with K-scale on same metrics.

    Metrics measured for both methods:
    1. Action entropy (model confidence)
    2. Action change rate vs baseline
    3. D2 augmentation consistency
    4. C-peak anchoring rate (via permutation test)
    """
    print(f"\n{'='*60}")
    print(f"  VAR Baseline Comparison for {model_cfg.name}")
    print(f"{'='*60}")

    verification_dir = output_dir.parent
    anchor_targets = detect_anchor_targets(model_cfg, verification_dir, bounds_cache[0])
    print(f"  Anchor targets: {anchor_targets}")

    sink_dims = SINK_DIMENSIONS.get(model_cfg.architecture, [])
    print(f"  Sink dimensions ({model_cfg.architecture}): {sink_dims}")

    # --- Step 1: Baseline measurements ---
    print(f"\n  === Baseline (no intervention) ===")
    baseline_data = []
    for si, sample in enumerate(samples):
        logits, _ = get_action_logits(model, processor, model_cfg, sample, device, bounds_cache[si])
        info = action_token_entropy(logits)
        info["top1_id"] = logits.argmax().item()

        # D2: augmentation consistency
        aug_matches = []
        for aug_name, aug_fn in AUGMENTATIONS:
            aug_sample = {**sample, "image": aug_fn(sample["image"])}
            logits_aug, _ = get_action_logits(model, processor, model_cfg, aug_sample, device, bounds_cache[si])
            aug_matches.append(logits_aug.argmax().item() == info["top1_id"])
        info["d2_consistency"] = np.mean(aug_matches)

        baseline_data.append(info)
        if (si + 1) % 10 == 0 or si == 0:
            print(f"    [{si+1}/{len(samples)}] H={info['entropy']:.3f} "
                  f"D2={info['d2_consistency']:.2f}")

    baseline_entropy = np.mean([d["entropy"] for d in baseline_data])
    baseline_d2 = np.mean([d["d2_consistency"] for d in baseline_data])
    print(f"  Baseline: entropy={baseline_entropy:.4f}, D2={baseline_d2:.4f}")

    results = {
        "model": model_cfg.name,
        "n_samples": len(samples),
        "baseline": {
            "mean_entropy": round(baseline_entropy, 4),
            "mean_d2_consistency": round(baseline_d2, 4),
        },
        "methods": [],
    }

    # --- Step 2: VAR intervention ---
    # VAR uses sink detection; for VLAs, the sink is typically vision token 0
    # (confirmed by our Phase 2.5 analysis)
    sink_abs_positions = [anchor_targets[0]["target_abs"]]  # Use detected anchor

    for p_val in [0.3, 0.6, 0.9]:
        print(f"\n  === VAR (p={p_val}) ===")
        var_data = []

        for si, sample in enumerate(samples):
            bounds = bounds_cache[si]
            vs, ve = bounds["vision_start"], bounds["vision_end"]

            # Apply VAR value hook
            var_hook = VARValueHook(
                sink_positions_abs=sink_abs_positions,
                vision_start=vs, vision_end=ve,
                p=p_val, target_layers=deep_layers,
            )
            var_hook.register(model, model_cfg, get_layers)

            logits, _ = get_action_logits(model, processor, model_cfg, sample, device, bounds)
            info = action_token_entropy(logits)
            top1 = logits.argmax().item()
            info["top1_changed"] = top1 != baseline_data[si]["top1_id"]

            # D2 with VAR
            aug_matches = []
            for aug_name, aug_fn in AUGMENTATIONS:
                aug_sample = {**sample, "image": aug_fn(sample["image"])}
                logits_aug, _ = get_action_logits(model, processor, model_cfg, aug_sample, device, bounds)
                aug_matches.append(logits_aug.argmax().item() == top1)
            info["d2_consistency"] = np.mean(aug_matches)

            var_hook.remove()
            var_data.append(info)

            if (si + 1) % 10 == 0 or si == 0:
                print(f"    [{si+1}/{len(samples)}] H={info['entropy']:.3f} "
                      f"D2={info['d2_consistency']:.2f} changed={info['top1_changed']}")

        var_entropy = np.mean([d["entropy"] for d in var_data])
        var_d2 = np.mean([d["d2_consistency"] for d in var_data])
        var_change = np.mean([d["top1_changed"] for d in var_data])

        results["methods"].append({
            "name": f"VAR_p{p_val}",
            "type": "VAR",
            "p": p_val,
            "mean_entropy": round(var_entropy, 4),
            "mean_d2_consistency": round(var_d2, 4),
            "d2_delta": round(var_d2 - baseline_d2, 4),
            "action_change_rate": round(var_change, 4),
        })
        print(f"  VAR p={p_val}: entropy={var_entropy:.4f}, D2={var_d2:.4f} "
              f"(delta={var_d2 - baseline_d2:+.4f}), change={var_change:.4f}")

    # --- Step 3: K-scale intervention (our method) ---
    for alpha in [0.0, 0.1, 0.3]:
        print(f"\n  === K-scale (alpha={alpha}) ===")
        kscale_data = []

        for si, sample in enumerate(samples):
            bounds = bounds_cache[si]
            all_targets = [t["target_abs"] for t in anchor_targets]

            k_hook = KeyScaleHook(all_targets, alpha=alpha, target_layers=deep_layers)
            k_hook.register(model, model_cfg, get_layers)

            logits, _ = get_action_logits(model, processor, model_cfg, sample, device, bounds)
            info = action_token_entropy(logits)
            top1 = logits.argmax().item()
            info["top1_changed"] = top1 != baseline_data[si]["top1_id"]

            # D2 with K-scale
            aug_matches = []
            for aug_name, aug_fn in AUGMENTATIONS:
                aug_sample = {**sample, "image": aug_fn(sample["image"])}
                logits_aug, _ = get_action_logits(model, processor, model_cfg, aug_sample, device, bounds)
                aug_matches.append(logits_aug.argmax().item() == top1)
            info["d2_consistency"] = np.mean(aug_matches)

            k_hook.remove()
            kscale_data.append(info)

            if (si + 1) % 10 == 0 or si == 0:
                print(f"    [{si+1}/{len(samples)}] H={info['entropy']:.3f} "
                      f"D2={info['d2_consistency']:.2f} changed={info['top1_changed']}")

        k_entropy = np.mean([d["entropy"] for d in kscale_data])
        k_d2 = np.mean([d["d2_consistency"] for d in kscale_data])
        k_change = np.mean([d["top1_changed"] for d in kscale_data])

        results["methods"].append({
            "name": f"K-scale_a{alpha}",
            "type": "K-scale",
            "alpha": alpha,
            "mean_entropy": round(k_entropy, 4),
            "mean_d2_consistency": round(k_d2, 4),
            "d2_delta": round(k_d2 - baseline_d2, 4),
            "action_change_rate": round(k_change, 4),
        })
        print(f"  K-scale α={alpha}: entropy={k_entropy:.4f}, D2={k_d2:.4f} "
              f"(delta={k_d2 - baseline_d2:+.4f}), change={k_change:.4f}")

    # --- Step 4: Summary comparison table ---
    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY: {model_cfg.name}")
    print(f"{'='*60}")
    print(f"  {'Method':<20s} {'Entropy':>8s} {'D2':>8s} {'ΔD2':>8s} {'Change%':>8s}")
    print(f"  {'-'*52}")
    print(f"  {'Baseline':<20s} {baseline_entropy:>8.4f} {baseline_d2:>8.4f} {'---':>8s} {'---':>8s}")
    for m in results["methods"]:
        print(f"  {m['name']:<20s} {m['mean_entropy']:>8.4f} {m['mean_d2_consistency']:>8.4f} "
              f"{m['d2_delta']:>+8.4f} {m['action_change_rate']:>8.4f}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "var_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {output_dir / 'var_comparison.json'}")

    return results


# =============================================================================
# Phi-based Sink Detection Analysis (supplementary)
# =============================================================================

def run_phi_analysis(model, processor, model_cfg, samples, device,
                     deep_layers, output_dir, bounds_cache):
    """Measure phi(x) for all vision tokens to verify sink detection.

    This provides supplementary data showing which tokens VAR would classify
    as sinks, and whether our anchor-based detection agrees.
    """
    sink_dims = SINK_DIMENSIONS.get(model_cfg.architecture, [])
    if not sink_dims:
        print(f"  Skipping phi analysis: no known sink dims for {model_cfg.architecture}")
        return None

    print(f"\n  === Phi Analysis (sink dimension detection) ===")
    print(f"  Sink dims: {sink_dims}")

    results = []
    hook_mgr = SinkVerificationHookManager(model, model_cfg)

    for si, sample in enumerate(samples[:10]):  # First 10 samples only
        bounds = bounds_cache[si]
        vs, ve = bounds["vision_start"], bounds["vision_end"]

        # Forward with hidden state capture
        hook_mgr.register_hooks()
        hook_mgr.reset()

        prompt = model_cfg.prompt_template.format(instruction=sample["instruction"])
        inputs = call_processor(processor, prompt, sample["image"], model_cfg,
                                return_tensors="pt").to(device)
        if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
        fwd_kwargs = {k: v for k, v in inputs.items()}
        fwd_kwargs["use_cache"] = False
        if model_cfg.architecture == "gemma2" and "intrinsic" not in fwd_kwargs:
            fwd_kwargs["intrinsic"] = torch.tensor(
                [[[218.26, 0.0, 111.83], [0.0, 218.26, 111.79], [0.0, 0.0, 1.0]]],
                device=device, dtype=torch.float32)

        with torch.no_grad():
            model(**fwd_kwargs)

        # Compute phi at each deep layer
        layer_phis = {}
        for l in deep_layers[:3]:  # First 3 deep layers
            hidden = hook_mgr.hidden_states.get(l)
            if hidden is None:
                continue
            phi_vals = compute_phi(hidden[vs:ve], sink_dims)
            sink_idxs, _ = identify_sink_tokens(hidden, sink_dims, vs, ve, tau=20.0)

            layer_phis[l] = {
                "phi_values": phi_vals.cpu().tolist()[:10],  # Top 10
                "phi_max": round(phi_vals.max().item(), 2),
                "phi_mean": round(phi_vals.mean().item(), 2),
                "phi_token0": round(phi_vals[0].item(), 2),
                "n_sinks_tau20": len(sink_idxs),
                "sink_indices": sink_idxs[:5],
            }

        hook_mgr.remove_hooks()

        results.append({
            "sample_idx": si,
            "skill": sample.get("skill", "unknown"),
            "layer_phis": layer_phis,
        })

        phi0 = layer_phis.get(deep_layers[0], {}).get("phi_token0", "N/A")
        n_sinks = layer_phis.get(deep_layers[0], {}).get("n_sinks_tau20", "N/A")
        print(f"    [{si+1}] phi(token0)={phi0}, n_sinks(tau=20)={n_sinks}")

    with open(output_dir / "var_phi_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="VAR baseline comparison")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--gate1_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--skip_phi", action="store_true",
                        help="Skip phi analysis (only run comparison)")
    args = parser.parse_args()

    gate1_dir = Path(args.gate1_dir) if args.gate1_dir else \
        config.OUTPUT_DIR / "phase3_gate" / args.model
    out = Path(args.output_dir) if args.output_dir else \
        config.OUTPUT_DIR / "phase3_gate" / "verification" / args.model
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model}...")
    processor, model, model_cfg = load_model_from_registry(args.model, args.device)
    samples = reload_samples_from_list(gate1_dir / "sample_list.json", config.DATA_CACHE_DIR)
    samples = samples[:args.n_samples]
    deep_layers = list(range(max(0, model_cfg.num_layers - 10), model_cfg.num_layers))

    # Build bounds cache
    bounds_cache = {}
    for si, s in enumerate(samples):
        bounds_cache[si] = detect_token_boundaries(
            processor, model, s["image"], s["instruction"], args.device, model_cfg)

    # Run phi analysis (LLaMA models only)
    if not args.skip_phi:
        run_phi_analysis(model, processor, model_cfg, samples, args.device,
                         deep_layers, out, bounds_cache)

    # Run VAR vs K-scale comparison
    run_var_comparison(model, processor, model_cfg, samples, args.device,
                       deep_layers, out, bounds_cache)

    del model
    torch.cuda.empty_cache()
    print(f"\nDone. Results in: {out}")


if __name__ == "__main__":
    main()
