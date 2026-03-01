#!/usr/bin/env python3
"""Post-hoc analysis of routing causality results.

Handles zero-variance content_following_rate (all zeros) gracefully,
focuses on D2 vs Anchoring correlation as primary analysis.

Usage:
  python analyze_routing_causality.py
"""
import json
import numpy as np
from pathlib import Path
from scipy import stats


def analyze(results_path):
    with open(results_path) as f:
        data = json.load(f)

    results = data["per_sample"]
    n = len(results)
    print(f"Model: {data['model']}, N={n}")

    d2_vals = np.array([r["d2_consistency"] for r in results])
    anchor_vals = np.array([r["anchoring_score"] for r in results])
    content_vals = np.array([r["content_following_rate"] for r in results])
    kl_vals = np.array([r["d2_mean_kl"] for r in results])

    print(f"\n{'='*70}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*70}")
    print(f"D2 consistency:  {d2_vals.mean():.4f} ± {d2_vals.std():.4f}  (range [{d2_vals.min():.2f}, {d2_vals.max():.2f}])")
    print(f"Anchoring score: {anchor_vals.mean():.4f} ± {anchor_vals.std():.4f}  (range [{anchor_vals.min():.2f}, {anchor_vals.max():.2f}])")
    print(f"Content follow:  {content_vals.mean():.4f} ± {content_vals.std():.4f}  (range [{content_vals.min():.2f}, {content_vals.max():.2f}])")
    print(f"D2 KL:           {kl_vals.mean():.4f} ± {kl_vals.std():.4f}")

    print(f"\n{'='*70}")
    print(f"CORRELATION ANALYSIS")
    print(f"{'='*70}")

    correlations = {}

    # Primary: D2 vs Anchoring
    if anchor_vals.std() > 0 and d2_vals.std() > 0:
        rho, p = stats.spearmanr(d2_vals, anchor_vals)
        print(f"\n★ PRIMARY: D2 vs Anchoring")
        print(f"  Spearman ρ = {rho:.4f}, p = {p:.6f}")
        print(f"  {'SIGNIFICANT' if p < 0.05 else 'NOT SIGNIFICANT'} (α=0.05)")

        if p < 0.05:
            direction = "POSITIVE (more anchored → more consistent)" if rho > 0 else "NEGATIVE (more anchored → less consistent)"
            print(f"  Direction: {direction}")

        correlations["d2_vs_anchoring"] = {"rho": float(rho), "p": float(p)}

        # Pearson too
        r_pearson, p_pearson = stats.pearsonr(d2_vals, anchor_vals)
        print(f"  Pearson r = {r_pearson:.4f}, p = {p_pearson:.6f}")

        # Bootstrap 95% CI
        rng = np.random.default_rng(42)
        boot_rhos = []
        for _ in range(10000):
            idx = rng.choice(n, size=n, replace=True)
            r_b, _ = stats.spearmanr(d2_vals[idx], anchor_vals[idx])
            if not np.isnan(r_b):
                boot_rhos.append(r_b)
        ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])
        print(f"  Bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
        correlations["d2_vs_anchoring_ci"] = {"lo": float(ci_lo), "hi": float(ci_hi)}
    else:
        print("\n★ PRIMARY: D2 vs Anchoring — SKIPPED (zero variance)")

    # Secondary: D2_KL vs Anchoring
    if anchor_vals.std() > 0 and kl_vals.std() > 0:
        rho, p = stats.spearmanr(kl_vals, anchor_vals)
        print(f"\n  D2_KL vs Anchoring: ρ = {rho:.4f}, p = {p:.6f}")
        correlations["kl_vs_anchoring"] = {"rho": float(rho), "p": float(p)}

    # Content following (likely zero-variance)
    if content_vals.std() > 0:
        rho, p = stats.spearmanr(d2_vals, content_vals)
        print(f"\n  D2 vs Content-following: ρ = {rho:.4f}, p = {p:.6f}")
        correlations["d2_vs_content"] = {"rho": float(rho), "p": float(p)}
    else:
        print(f"\n  D2 vs Content-following: SKIPPED (content_following has zero variance = {content_vals.mean():.4f})")
        print(f"  → All samples show 0% content-following: attention is PURELY positional")

    # Skill breakdown
    print(f"\n{'='*70}")
    print(f"PER-SKILL BREAKDOWN")
    print(f"{'='*70}")
    from collections import defaultdict
    skill_data = defaultdict(list)
    for r in results:
        skill_data[r["skill"]].append(r)

    for skill in sorted(skill_data.keys()):
        sr = skill_data[skill]
        d2s = [r["d2_consistency"] for r in sr]
        ancs = [r["anchoring_score"] for r in sr]
        print(f"  {skill:10s}: N={len(sr):3d}  D2={np.mean(d2s):.3f}±{np.std(d2s):.3f}  Anchor={np.mean(ancs):.3f}±{np.std(ancs):.3f}")

    # Interpretation
    print(f"\n{'='*70}")
    print(f"INTERPRETATION")
    print(f"{'='*70}")

    if "d2_vs_anchoring" in correlations:
        rho = correlations["d2_vs_anchoring"]["rho"]
        p = correlations["d2_vs_anchoring"]["p"]
        if p < 0.05:
            if rho > 0:
                print("Position anchoring POSITIVELY correlates with augmentation robustness.")
                print("→ Counterintuitive: fixed-position attention provides STABLE routing.")
                print("→ Paper implication: 'bottleneck' may be a feature, not a bug.")
            else:
                print("Position anchoring NEGATIVELY correlates with augmentation robustness.")
                print("→ Supports hypothesis: position-anchored routing → fragile to augmentation.")
                print("→ Paper implication: routing quality causally affects robustness.")
        else:
            print("No significant correlation between anchoring and augmentation robustness.")
            print("→ Position anchoring and D2 are independent within TraceVLA.")
            print("→ Paper implication: routing variation doesn't explain robustness variation.")
    print(f"{'='*70}")

    # Save corrected analysis
    output_path = results_path.parent / "routing_causality_analysis.json"
    output = {
        "model": data["model"],
        "n_samples": n,
        "correlations": correlations,
        "summary": {
            "d2_mean": float(d2_vals.mean()),
            "d2_std": float(d2_vals.std()),
            "anchoring_mean": float(anchor_vals.mean()),
            "anchoring_std": float(anchor_vals.std()),
            "content_following_mean": float(content_vals.mean()),
            "content_following_is_zero_variance": bool(content_vals.std() == 0),
        },
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    results_path = Path("outputs/routing_causality/tracevla-phi3v/routing_causality.json")
    if results_path.exists():
        analyze(results_path)
    else:
        print(f"Results not ready yet: {results_path}")
        print("Run this after run_routing_causality.py completes.")
