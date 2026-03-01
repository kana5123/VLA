#!/usr/bin/env python3
"""Per-sample routing → D2 causal analysis for TraceVLA.

Within-model experiment: TraceVLA has natural routing variation across samples
(82.2% neither, 17.8% content-following). We correlate per-sample routing quality
with D2 (augmentation consistency) to establish causal evidence that
better routing → better robustness WITHOUT model capability confound.

Usage:
  python run_routing_causality.py --device cuda:5 --n_per_skill 35 --n_perm 3
"""
import argparse
import json
import sys
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

import config
from extract_attention import (
    load_model_from_registry, get_layers, call_processor, detect_token_boundaries,
)
from verify_attention_sinks import SinkVerificationHookManager, get_wov_matrix
from data_sampler import load_balanced_samples
from contribution.compute import extract_sample_contributions
from run_phase3_exp_de import (
    get_action_logits, AUGMENTATIONS, compute_output_kl,
)


def measure_d2_per_sample(model, processor, model_cfg, sample, device, bounds):
    """Measure D2 (augmentation consistency) for a single sample."""
    logits_orig, _ = get_action_logits(model, processor, model_cfg, sample, device, bounds)
    top1_orig = logits_orig.argmax().item()

    matches = []
    kls = []
    for aug_name, aug_fn in AUGMENTATIONS:
        aug_sample = {**sample, "image": aug_fn(sample["image"])}
        logits_aug, _ = get_action_logits(model, processor, model_cfg, aug_sample, device, bounds)
        top1_aug = logits_aug.argmax().item()
        matches.append(int(top1_aug == top1_orig))
        kls.append(compute_output_kl(logits_orig, logits_aug))

    return {
        "consistency": float(np.mean(matches)),
        "mean_kl": float(np.mean(kls)),
    }


def _make_embed_perm_hook(vs, ve, perm_tensor):
    """Create a post-hook for embed_tokens that permutes vision positions.

    Matches the reference implementation in run_phase3_exp_de.py (lines 767-783).
    """
    def hook_fn(module, args, output):
        if isinstance(output, tuple):
            t = output[0].clone()
            t[0, vs:ve] = t[0, vs:ve][perm_tensor]
            return (t,) + output[1:]
        m = output.clone()
        m[0, vs:ve] = m[0, vs:ve][perm_tensor]
        return m
    return hook_fn


def _find_embed_module(model, model_cfg):
    """Find the embedding module to hook for permutation."""
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        return model.model.embed_tokens
    elif hasattr(model, 'embed_tokens'):
        return model.embed_tokens
    else:
        return get_layers(model, model_cfg)[0]


def _clone_inputs(inputs):
    """Deep clone inputs dict to avoid Phi3V in-place input_ids modification."""
    cloned = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            cloned[k] = v.clone()
        else:
            cloned[k] = v
    return cloned


def measure_anchoring_per_sample(model, processor, model_cfg, sample, device,
                                  bounds, deep_layers, n_perm=3, seed=42):
    """Measure position anchoring score for a single sample via permutation test.

    For each permutation, shuffle vision token embedding positions via a hook on
    embed_tokens (matching run_phase3_exp_de.py reference implementation).
    Then check if attention peak follows content or stays at position.

    IMPORTANT: Phi3V modifies input_ids in-place during forward pass,
    so we clone inputs before each forward.

    Returns:
        dict with per-sample anchoring metrics
    """
    hook_mgr = SinkVerificationHookManager(model, model_cfg)

    # Prepare inputs (will clone before each forward)
    prompt = model_cfg.prompt_template.format(instruction=sample["instruction"])
    inputs_base = call_processor(processor, prompt, sample["image"],
                                 model_cfg, return_tensors="pt").to(device)
    if "pixel_values" in inputs_base and inputs_base["pixel_values"].dtype != model.dtype:
        inputs_base["pixel_values"] = inputs_base["pixel_values"].to(model.dtype)

    vs = bounds["vision_start"]
    ve = bounds["vision_end"]
    n_vis = ve - vs

    # Original forward: get attention peaks (clone inputs for Phi3V safety)
    hook_mgr.register_hooks()
    hook_mgr.reset()
    inputs_orig = _clone_inputs(inputs_base)
    with torch.no_grad():
        model(**inputs_orig, output_attentions=True)

    orig_attn = hook_mgr.attention_weights
    hook_mgr.remove_hooks()

    # Get original peaks per layer
    orig_peaks = {}
    for l in deep_layers:
        attn = orig_attn.get(l)
        if attn is None:
            continue
        if attn.dim() == 3:  # (H, seq, seq)
            qpos = attn.shape[-2] - 1
            a_vis = attn[:, qpos, vs:ve].mean(dim=0)  # (n_vis,)
        else:
            qpos = attn.shape[-2] - 1
            a_vis = attn[0, :, qpos, vs:ve].mean(dim=0)
        orig_peaks[l] = int(a_vis.argmax().item())

    if not orig_peaks:
        return {"anchoring_score": 0.0, "content_following_rate": 0.0, "n_tests": 0}

    # Find embedding module for permutation hook
    embed_mod = _find_embed_module(model, model_cfg)

    # Permutation test
    rng = np.random.default_rng(seed)
    a_followed_content_counts = []
    a_stayed_same_pos_counts = []

    for perm_i in range(n_perm):
        perm = rng.permutation(n_vis)
        perm_tensor = torch.tensor(perm, device=device, dtype=torch.long)

        # Register permutation hook on embed_tokens output (reference approach)
        perm_handle = embed_mod.register_forward_hook(
            _make_embed_perm_hook(vs, ve, perm_tensor)
        )

        # Register attention capture hooks
        hook_mgr.register_hooks()
        hook_mgr.reset()

        # Clone inputs (critical for Phi3V which modifies input_ids in-place)
        inputs_perm = _clone_inputs(inputs_base)

        with torch.no_grad():
            model(**inputs_perm, output_attentions=True)

        perm_attn = hook_mgr.attention_weights
        hook_mgr.remove_hooks()
        perm_handle.remove()

        # Check peaks after permutation
        for l in deep_layers:
            if l not in orig_peaks:
                continue
            attn = perm_attn.get(l)
            if attn is None:
                continue
            if attn.dim() == 3:
                qpos = attn.shape[-2] - 1
                a_vis = attn[:, qpos, vs:ve].mean(dim=0)
            else:
                qpos = attn.shape[-2] - 1
                a_vis = attn[0, :, qpos, vs:ve].mean(dim=0)
            perm_peak = int(a_vis.argmax().item())

            orig_peak = orig_peaks[l]
            # "stayed same position": peak didn't move (position anchoring)
            stayed_same = (perm_peak == orig_peak)
            # "followed content": peak moved to where original content went
            followed = (perm_peak == int(perm[orig_peak]))

            a_stayed_same_pos_counts.append(int(stayed_same))
            a_followed_content_counts.append(int(followed))

    n_total = len(a_followed_content_counts)
    if n_total == 0:
        return {"anchoring_score": 0.0, "content_following_rate": 0.0, "n_tests": 0}

    return {
        "anchoring_score": float(np.mean(a_stayed_same_pos_counts)),
        "content_following_rate": float(np.mean(a_followed_content_counts)),
        "n_tests": n_total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="tracevla-phi3v")
    parser.add_argument("--device", type=str, default="cuda:5")
    parser.add_argument("--n_per_skill", type=int, default=35,
                        help="Samples per skill (6 skills × 35 = 210 total)")
    parser.add_argument("--n_perm", type=int, default=3,
                        help="Permutations per sample for anchoring test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path("outputs/routing_causality") / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Loading {args.model} on {args.device} ===", flush=True)
    processor, model, model_cfg = load_model_from_registry(args.model, args.device)
    model.eval()

    # Load balanced samples
    target_skills = ["place", "move", "pick", "fold", "open", "close"]
    samples = load_balanced_samples(
        config.DATA_CACHE_DIR, n_per_skill=args.n_per_skill,
        target_skills=target_skills, seed=args.seed,
    )
    print(f"Loaded {len(samples)} balanced samples", flush=True)

    n_layers = model_cfg.num_layers
    deep_layers = list(range(max(0, n_layers - 10), n_layers))

    # Per-sample measurements
    results = []
    for si, sample in enumerate(samples):
        bounds = detect_token_boundaries(
            processor, model, sample["image"], sample["instruction"],
            args.device, model_cfg,
        )

        # D2
        d2 = measure_d2_per_sample(model, processor, model_cfg, sample,
                                    args.device, bounds)

        # Anchoring
        anchoring = measure_anchoring_per_sample(
            model, processor, model_cfg, sample, args.device,
            bounds, deep_layers, n_perm=args.n_perm, seed=args.seed + si,
        )

        result = {
            "sample_idx": si,
            "skill": sample.get("skill", "unknown"),
            "d2_consistency": d2["consistency"],
            "d2_mean_kl": d2["mean_kl"],
            "anchoring_score": anchoring["anchoring_score"],
            "content_following_rate": anchoring["content_following_rate"],
            "n_anchoring_tests": anchoring["n_tests"],
        }
        results.append(result)

        if (si + 1) % 10 == 0:
            print(f"  [{si+1}/{len(samples)}] D2={d2['consistency']:.2f} "
                  f"anchor={anchoring['anchoring_score']:.2f} "
                  f"content={anchoring['content_following_rate']:.2f}", flush=True)

    # Compute correlations
    d2_vals = np.array([r["d2_consistency"] for r in results])
    anchor_vals = np.array([r["anchoring_score"] for r in results])
    content_vals = np.array([r["content_following_rate"] for r in results])

    print(f"\n{'='*70}")
    print(f"RESULTS (N={len(results)})")
    print(f"{'='*70}")
    print(f"D2 mean: {d2_vals.mean():.4f} ± {d2_vals.std():.4f}")
    print(f"Anchoring mean: {anchor_vals.mean():.4f} ± {anchor_vals.std():.4f}")
    print(f"Content-following mean: {content_vals.mean():.4f} ± {content_vals.std():.4f}")

    # Spearman correlations
    correlations = {}

    # D2 vs Content-following (expected: positive — more content → more consistent)
    rho_d2_content, p_d2_content = stats.spearmanr(d2_vals, content_vals)
    print(f"\nD2 vs Content-following: rho={rho_d2_content:.4f}, p={p_d2_content:.4f}")
    correlations["d2_vs_content_following"] = {
        "rho": float(rho_d2_content), "p": float(p_d2_content),
    }

    # D2 vs Anchoring (expected: negative — more position-anchored → less consistent)
    rho_d2_anchor, p_d2_anchor = stats.spearmanr(d2_vals, anchor_vals)
    print(f"D2 vs Anchoring: rho={rho_d2_anchor:.4f}, p={p_d2_anchor:.4f}")
    correlations["d2_vs_anchoring"] = {
        "rho": float(rho_d2_anchor), "p": float(p_d2_anchor),
    }

    # D2_KL vs Content-following
    kl_vals = np.array([r["d2_mean_kl"] for r in results])
    rho_kl_content, p_kl_content = stats.spearmanr(kl_vals, content_vals)
    print(f"D2_KL vs Content-following: rho={rho_kl_content:.4f}, p={p_kl_content:.4f}")
    correlations["kl_vs_content_following"] = {
        "rho": float(rho_kl_content), "p": float(p_kl_content),
    }

    # Bootstrap 95% CI for primary correlation
    n_boot = 5000
    boot_rhos = []
    rng = np.random.default_rng(args.seed + 9999)
    for _ in range(n_boot):
        idx = rng.choice(len(results), size=len(results), replace=True)
        r, _ = stats.spearmanr(d2_vals[idx], content_vals[idx])
        boot_rhos.append(r)
    ci_lo = np.percentile(boot_rhos, 2.5)
    ci_hi = np.percentile(boot_rhos, 97.5)
    print(f"\nBootstrap 95% CI for D2-Content rho: [{ci_lo:.4f}, {ci_hi:.4f}]")
    correlations["d2_vs_content_bootstrap_ci"] = {
        "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
    }

    # Interpretation
    if p_d2_content < 0.05 and rho_d2_content > 0:
        interpretation = "SIGNIFICANT POSITIVE: Content-following routing → higher D2 (robustness)"
    elif p_d2_content < 0.05 and rho_d2_content < 0:
        interpretation = "SIGNIFICANT NEGATIVE: Content-following routing → lower D2 (unexpected)"
    else:
        interpretation = "NOT SIGNIFICANT: No clear relationship found"
    print(f"\nInterpretation: {interpretation}")

    # Save results
    output = {
        "model": args.model,
        "n_samples": len(results),
        "correlations": correlations,
        "interpretation": interpretation,
        "per_sample": results,
        "summary": {
            "d2_mean": float(d2_vals.mean()),
            "d2_std": float(d2_vals.std()),
            "anchoring_mean": float(anchor_vals.mean()),
            "content_following_mean": float(content_vals.mean()),
        },
    }

    with open(output_dir / "routing_causality.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {output_dir / 'routing_causality.json'}", flush=True)


if __name__ == "__main__":
    main()
