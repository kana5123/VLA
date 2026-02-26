"""Cross-model bottleneck diagnosis comparison.

Reads diagnosis_report.json from each model's output directory and generates:
  1. Summary comparison table (printed + saved as JSON)
  2. Multi-panel comparison figure

Usage:
    python compare_bottleneck_diagnosis.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import config

OUTPUT_DIR = config.OUTPUT_DIR / "bottleneck_diagnosis"


def load_all_reports():
    """Load all diagnosis reports from output directory."""
    reports = {}
    for report_path in sorted(OUTPUT_DIR.glob("*/diagnosis_report.json")):
        model_name = report_path.parent.name
        with open(report_path, encoding="utf-8") as f:
            reports[model_name] = json.load(f)
    return reports


def build_comparison_table(reports):
    """Build structured comparison from all reports."""
    rows = []
    for model_name, report in reports.items():
        test1 = report.get("test1_cls_vs_patch", {})
        test2 = report.get("test2_shift", {})
        test3 = report.get("test3_ablation", {})

        # Test 2: average contribution across shifts
        shift_data = test2.get("shifts", [])
        orig_contrib = 0
        avg_shifted_contrib = 0
        if shift_data:
            orig_contrib = shift_data[0].get("avg_contribution_pct", 0)
            shifted = [s.get("avg_contribution_pct", 0) for s in shift_data[1:]]
            avg_shifted_contrib = np.mean(shifted) if shifted else 0

        # Test 3: ablation metrics
        summary = test3.get("summary", {})
        kl_t0 = summary.get("avg_kl_token0", 0)
        kl_rand = summary.get("avg_kl_random", 0)
        kl_ratio = summary.get("avg_kl_ratio", 0)
        n_changed_t0 = summary.get("n_action_changed_t0", 0)
        n_changed_rand = summary.get("n_action_changed_rand", 0)

        rows.append({
            "model": model_name,
            "has_cls": test1.get("has_cls", "unknown"),
            "vision_encoder": test1.get("vision_encoder", ""),
            "num_vision_tokens": test1.get("num_vision_tokens", 0),
            "token0_identity": "CLS" if test1.get("has_cls") is True else "Patch" if test1.get("has_cls") is False else "Unknown",
            "orig_contribution_pct": orig_contrib,
            "shifted_contribution_pct": avg_shifted_contrib,
            "shift_verdict": test2.get("verdict", ""),
            "kl_token0": kl_t0,
            "kl_random": kl_rand,
            "kl_ratio": kl_ratio,
            "action_changed_t0": n_changed_t0,
            "action_changed_rand": n_changed_rand,
            "ablation_verdict": test3.get("verdict", ""),
            "overall_verdict": report.get("overall_verdict", ""),
        })

    return rows


def print_comparison_table(rows):
    """Print a formatted comparison table."""
    print("\n" + "=" * 100)
    print("CROSS-MODEL BOTTLENECK DIAGNOSIS COMPARISON")
    print("=" * 100)

    # Sort by KL ratio (highest first)
    rows_sorted = sorted(rows, key=lambda r: r["kl_ratio"], reverse=True)

    print(f"\n{'Model':<20} {'Backbone':<12} {'Token0':<8} {'Contrib%':<10} "
          f"{'KL ratio':<12} {'Changed':<10} {'Verdict'}")
    print("-" * 100)

    backbone_map = {
        "openvla-7b": "LLaMA-2",
        "ecot-7b": "LLaMA-2",
        "llava-1.5-7b": "LLaMA-2",
        "tracevla-phi3v": "Phi3V",
        "spatialvla-4b": "Gemma2",
        "paligemma-3b": "Gemma",
        "internvl2-8b": "InternLM2",
    }

    for r in rows_sorted:
        backbone = backbone_map.get(r["model"], "?")
        changed = f"{r['action_changed_t0']}/3 vs {r['action_changed_rand']}/3"

        # Verdict emoji
        if r["kl_ratio"] > 100:
            verdict = "CRITICAL BOTTLENECK"
        elif r["kl_ratio"] > 5:
            verdict = "SIGNIFICANT"
        elif r["kl_ratio"] > 2:
            verdict = "MODERATE"
        else:
            verdict = "NO BOTTLENECK"

        print(f"{r['model']:<20} {backbone:<12} {r['token0_identity']:<8} "
              f"{r['orig_contribution_pct']:<10.1f} {r['kl_ratio']:<12.1f} "
              f"{changed:<10} {verdict}")

    print()


def plot_comparison(rows, output_path):
    """Generate multi-panel comparison figure."""
    rows_sorted = sorted(rows, key=lambda r: r["kl_ratio"], reverse=True)
    models = [r["model"] for r in rows_sorted]
    n = len(models)

    backbone_map = {
        "openvla-7b": "LLaMA-2+Prismatic",
        "ecot-7b": "LLaMA-2+Prismatic",
        "llava-1.5-7b": "LLaMA-2+CLIP",
        "tracevla-phi3v": "Phi3V+CLIP",
        "spatialvla-4b": "Gemma2+SigLIP",
        "paligemma-3b": "Gemma+SigLIP",
        "internvl2-8b": "InternLM2+InternViT",
    }

    model_type_map = {
        "openvla-7b": "VLA",
        "ecot-7b": "VLA",
        "llava-1.5-7b": "VLM",
        "tracevla-phi3v": "VLA",
        "spatialvla-4b": "VLA",
        "paligemma-3b": "VLM",
        "internvl2-8b": "VLM",
    }

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # Color coding by backbone
    colors = []
    for m in models:
        if "prismatic" in backbone_map.get(m, "").lower():
            colors.append("#d32f2f")  # red for Prismatic
        elif "llama" in backbone_map.get(m, "").lower():
            colors.append("#1976d2")  # blue for LLaMA
        elif "gemma" in backbone_map.get(m, "").lower():
            colors.append("#388e3c")  # green for Gemma
        elif "phi3" in backbone_map.get(m, "").lower():
            colors.append("#f57c00")  # orange for Phi3
        else:
            colors.append("#7b1fa2")  # purple for others

    labels = [f"{m}\n({backbone_map.get(m, '?')})" for m in models]

    # Panel 1: KL Ratio (log scale)
    ax1 = axes[0]
    kl_ratios = [max(r["kl_ratio"], 0.1) for r in rows_sorted]  # min 0.1 for log
    bars1 = ax1.barh(range(n), kl_ratios, color=colors, alpha=0.85, edgecolor='white')
    ax1.set_xscale('log')
    ax1.set_yticks(range(n))
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel("KL Ratio (token0 / random)", fontsize=10)
    ax1.set_title("Ablation KL Ratio\n(higher = more dependent on token 0)", fontsize=11)
    ax1.axvline(x=10, color='orange', linestyle='--', alpha=0.5, label='threshold=10x')
    ax1.axvline(x=100, color='red', linestyle='--', alpha=0.5, label='threshold=100x')
    ax1.legend(fontsize=7)
    ax1.invert_yaxis()

    # Annotate bars
    for i, (bar, val) in enumerate(zip(bars1, kl_ratios)):
        ax1.text(bar.get_width() * 1.1, bar.get_y() + bar.get_height()/2,
                 f'{val:.1f}x', va='center', fontsize=8, fontweight='bold')

    # Panel 2: Token 0 Contribution % (original vs shifted)
    ax2 = axes[1]
    orig_pcts = [r["orig_contribution_pct"] for r in rows_sorted]
    shifted_pcts = [r["shifted_contribution_pct"] for r in rows_sorted]

    y_pos = np.arange(n)
    bar_height = 0.35
    bars_orig = ax2.barh(y_pos - bar_height/2, orig_pcts, bar_height,
                          color=colors, alpha=0.85, label='Original', edgecolor='white')
    bars_shift = ax2.barh(y_pos + bar_height/2, shifted_pcts, bar_height,
                           color=colors, alpha=0.4, label='Shifted (avg)', edgecolor='white')

    ax2.set_yticks(range(n))
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel("Token 0 Contribution (%)", fontsize=10)
    ax2.set_title("Token 0 Contribution\n(solid=original, faded=shifted)", fontsize=11)
    ax2.axvline(x=50, color='orange', linestyle='--', alpha=0.5)
    ax2.axvline(x=90, color='red', linestyle='--', alpha=0.5)
    ax2.legend(fontsize=8)
    ax2.invert_yaxis()

    # Panel 3: Action Changed Summary
    ax3 = axes[2]
    changed_t0 = [r["action_changed_t0"] for r in rows_sorted]
    changed_rand = [r["action_changed_rand"] for r in rows_sorted]

    bars_t0 = ax3.barh(y_pos - bar_height/2, changed_t0, bar_height,
                        color='#d32f2f', alpha=0.8, label='Token 0 ablated')
    bars_rand = ax3.barh(y_pos + bar_height/2, changed_rand, bar_height,
                          color='#1976d2', alpha=0.8, label='Random ablated')

    ax3.set_yticks(range(n))
    ax3.set_yticklabels(labels, fontsize=8)
    ax3.set_xlabel("Samples with Action Changed (out of 3)", fontsize=10)
    ax3.set_title("Action Change on Ablation\n(red=token0, blue=random)", fontsize=11)
    ax3.set_xlim(0, 3.5)
    ax3.legend(fontsize=8)
    ax3.invert_yaxis()

    # VLA/VLM badge
    for i, m in enumerate(models):
        mtype = model_type_map.get(m, "?")
        badge_color = '#e91e63' if mtype == "VLA" else '#3f51b5'
        ax3.text(3.3, i, mtype, ha='center', va='center', fontsize=7,
                 fontweight='bold', color='white',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor=badge_color, alpha=0.8))

    plt.suptitle("Cross-Architecture Token 0 Bottleneck Diagnosis\n"
                 "6 Models: 4 VLAs + 2 VLMs", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nComparison figure saved: {output_path}")


def main():
    reports = load_all_reports()
    if not reports:
        print("No diagnosis reports found!")
        return

    print(f"Found {len(reports)} diagnosis reports: {', '.join(reports.keys())}")

    rows = build_comparison_table(reports)
    print_comparison_table(rows)

    # Save comparison JSON
    comparison_path = OUTPUT_DIR / "cross_model_comparison.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"Comparison data saved: {comparison_path}")

    # Generate comparison figure
    plot_comparison(rows, OUTPUT_DIR / "cross_model_comparison.png")


if __name__ == "__main__":
    main()
