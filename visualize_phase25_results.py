#!/usr/bin/env python3
"""
Phase 2.5 Cross-Model Comparison Visualizations.

Generates 8 publication-quality figures from contribution + causal reports:
  1. Dual-Track A-peak vs C-peak comparison (4 models x 10 layers)
  2. Top1 C̃ share curve overlay (4 models)
  3. A-C mismatch (JS divergence) per model per layer
  4. V=0 Causal KL divergence comparison (bar chart)
  5. Token identity heatmap (which tokens are A/C/R peaks)
  6. Phi (φ) hidden state spike comparison
  7. Attention vs Contribution scatter (all models combined)
  8. Model taxonomy radar chart (bottleneck severity, mismatch, entropy, KL)
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict

# ── Configuration ──────────────────────────────────────────────
OUTPUT_DIR = Path("outputs/phase2.5_analysis")
CONTRIB_DIR = Path("outputs/contribution_analysis")
CAUSAL_DIR = Path("outputs/causal_experiment")

MODELS = {
    "ecot-7b": {"dir": "ecot-7b-phase2.5-test", "causal": "ecot-7b-phase2.5", "color": "#e74c3c", "short": "ECoT"},
    "openvla-7b": {"dir": "openvla-7b-phase2.5-test", "causal": "openvla-7b-phase2.5", "color": "#3498db", "short": "OpenVLA"},
    "spatialvla-4b": {"dir": "spatialvla-4b-phase2.5-test", "causal": "spatialvla-4b-phase2.5", "color": "#2ecc71", "short": "SpatialVLA"},
    "tracevla-phi3v": {"dir": "tracevla-phi3v-phase2.5-test", "causal": "tracevla-phi3v-phase2.5", "color": "#9b59b6", "short": "TraceVLA"},
}

# ── Load Data ──────────────────────────────────────────────────
def load_all_reports():
    contrib_reports = {}
    causal_reports = {}
    for model_name, cfg in MODELS.items():
        cpath = CONTRIB_DIR / cfg["dir"] / "contribution_report.json"
        if cpath.exists():
            with open(cpath) as f:
                contrib_reports[model_name] = json.load(f)

        capath = CAUSAL_DIR / cfg["causal"] / "causal_report.json"
        if capath.exists():
            with open(capath) as f:
                causal_reports[model_name] = json.load(f)

    return contrib_reports, causal_reports


# ── Figure 1: Dual-Track A-peak vs C-peak ─────────────────────
def fig1_dual_track_peaks(contrib_reports, out_dir):
    """Shows where A-peak and C-peak fall for each model across layers."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (model_name, report) in enumerate(contrib_reports.items()):
        ax = axes[idx]
        cfg = MODELS[model_name]
        layers = report["deep_layers"]
        analysis = report["layer_analysis"]
        boundaries = report["boundaries"]
        ve = boundaries["vision_end"]

        a_peaks = []
        c_peaks = []
        r_peaks = []
        a_types = []
        c_types = []

        for l in layers:
            la = analysis.get(str(l), {})
            if not la:
                continue
            a_peak = la.get("a_peak", {})
            c_peak = la.get("c_peak", {})
            r_peak = la.get("r_peak", {})

            a_peaks.append(a_peak.get("abs_t", 0))
            c_peaks.append(c_peak.get("abs_t", 0))
            r_peaks.append(r_peak.get("abs_t", 0))
            a_types.append(a_peak.get("token_type", ""))
            c_types.append(c_peak.get("token_type", ""))

        x = range(len(layers))
        ax.scatter(x, a_peaks, marker="^", s=100, c="red", label="A-peak (attn)", zorder=5, edgecolors="darkred")
        ax.scatter(x, c_peaks, marker="s", s=100, c="blue", label="C-peak (contrib)", zorder=5, edgecolors="darkblue")
        ax.scatter(x, r_peaks, marker="D", s=60, c="orange", alpha=0.6, label="R-peak (sink)", zorder=4)

        # Vision/text boundary
        ax.axhline(y=ve, color="gray", linestyle="--", alpha=0.5, label=f"Vision end ({ve})")
        ax.axhspan(0, ve, alpha=0.05, color="green")
        ax.axhspan(ve, max(max(a_peaks), max(c_peaks), max(r_peaks)) + 10, alpha=0.05, color="yellow")

        ax.set_xticks(list(x))
        ax.set_xticklabels([str(l) for l in layers], fontsize=8)
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Token Position (abs_t)")
        ax.set_title(f"{cfg['short']} — {analysis.get(str(layers[0]), {}).get('dominant_type', '?')}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.2)

    fig.suptitle("Phase 2.5: Dual-Track Peak Positions (A-peak vs C-peak vs R-peak)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "fig1_dual_track_peaks.png", dpi=200)
    plt.close(fig)
    print(f"  Saved: fig1_dual_track_peaks.png")


# ── Figure 2: Top1 C̃ Share Overlay ────────────────────────────
def fig2_top1_share_overlay(contrib_reports, out_dir):
    """Overlay Top1 contribution share curves for all 4 models."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for model_name, report in contrib_reports.items():
        cfg = MODELS[model_name]
        layers = report["deep_layers"]
        analysis = report["layer_analysis"]

        shares = []
        for l in layers:
            la = analysis.get(str(l), {})
            shares.append(la.get("mean_top1_share", 0))

        ax.plot(range(len(layers)), shares,
                marker="o", linewidth=2.5, markersize=6,
                color=cfg["color"], label=f"{cfg['short']} ({analysis.get(str(layers[0]), {}).get('dominant_type', '?')})")

    ax.axhline(y=0.5, color="red", linestyle=":", alpha=0.7, label="Bottleneck threshold (0.5)")
    ax.set_xlabel("Layer (deep layers only)", fontsize=12)
    ax.set_ylabel("Top1 C̃(j) Share", fontsize=12)
    ax.set_title("Layer-wise Top1 Contribution Share — All Models", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_top1_share_overlay.png", dpi=200)
    plt.close(fig)
    print(f"  Saved: fig2_top1_share_overlay.png")


# ── Figure 3: A-C Mismatch (JS Divergence) ────────────────────
def fig3_mismatch_curves(contrib_reports, out_dir):
    """JS divergence(Ã, C̃) per layer per model — high = sink pattern."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for model_name, report in contrib_reports.items():
        cfg = MODELS[model_name]
        layers = report["deep_layers"]
        analysis = report["layer_analysis"]

        mismatches = []
        for l in layers:
            la = analysis.get(str(l), {})
            mismatches.append(la.get("mean_mismatch", 0))

        ax.plot(range(len(layers)), mismatches,
                marker="s", linewidth=2, markersize=5,
                color=cfg["color"], label=cfg["short"])

    ax.set_xlabel("Layer (deep layers only)", fontsize=12)
    ax.set_ylabel("JS Divergence (Ã ∥ C̃)", fontsize=12)
    ax.set_title("Attention–Contribution Mismatch — Higher = Stronger Sink Pattern", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_ac_mismatch.png", dpi=200)
    plt.close(fig)
    print(f"  Saved: fig3_ac_mismatch.png")


# ── Figure 4: V=0 Causal KL Divergence ────────────────────────
def fig4_causal_kl_comparison(causal_reports, out_dir):
    """Bar chart comparing V=0 KL divergence across models and K values."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: KL at K=1
    models_sorted = list(causal_reports.keys())
    kl_k1 = []
    kl_k1_std = []
    top1_change = []
    colors = []

    for m in models_sorted:
        r = causal_reports[m]
        pk = r["per_k"]["1"]
        kl_k1.append(pk["vzero_mean_kl"])
        kl_k1_std.append(pk["vzero_std_kl"])
        top1_change.append(pk["vzero_mean_top1_change"])
        colors.append(MODELS[m]["color"])

    short_names = [MODELS[m]["short"] for m in models_sorted]
    x = range(len(models_sorted))

    bars = ax1.bar(x, kl_k1, yerr=kl_k1_std, capsize=5,
                   color=colors, alpha=0.8, edgecolor="black")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(short_names, fontsize=11)
    ax1.set_ylabel("KL Divergence", fontsize=12)
    ax1.set_title("V=0 Causal Impact (K=1)", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.2, axis="y")

    # Add KL values on top of bars
    for bar, kl, std in zip(bars, kl_k1, kl_k1_std):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.2,
                f'{kl:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Right: Top-1 prediction change rate
    bars2 = ax2.bar(x, top1_change, color=colors, alpha=0.8, edgecolor="black")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(short_names, fontsize=11)
    ax2.set_ylabel("Top-1 Change Rate", fontsize=12)
    ax2.set_title("Prediction Flip Rate After V=0 (K=1)", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, 1.15)
    ax2.grid(True, alpha=0.2, axis="y")

    for bar, rate in zip(bars2, top1_change):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{rate:.0%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    fig.suptitle("Phase 2.5: Causal Verification via Value-Zero Masking", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "fig4_causal_kl.png", dpi=200)
    plt.close(fig)
    print(f"  Saved: fig4_causal_kl.png")


# ── Figure 5: Token Identity Matrix ───────────────────────────
def fig5_token_identity_heatmap(contrib_reports, out_dir):
    """Which tokens are A-peak, C-peak, R-peak across models & layers."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)

    peak_types = [("a_peak", "A-peak (Attention Max)", "Reds"),
                  ("c_peak", "C-peak (Contribution Max)", "Blues"),
                  ("r_peak", "R-peak (Sink Candidate)", "Oranges")]

    all_models = list(contrib_reports.keys())
    short_names = [MODELS[m]["short"] for m in all_models]

    for ax, (peak_key, title, cmap) in zip(axes, peak_types):
        # Build matrix: models x layers → token_str
        token_strs = []
        token_types = []

        for model_name in all_models:
            report = contrib_reports[model_name]
            layers = report["deep_layers"]
            analysis = report["layer_analysis"]

            row_strs = []
            row_types = []
            for l in layers:
                la = analysis.get(str(l), {})
                peak = la.get(peak_key, {})
                ts = peak.get("token_str", "?")
                tt = peak.get("token_type", "?")
                share = peak.get("a_share" if peak_key == "a_peak" else "c_share", 0)
                row_strs.append(f"{ts}\n({share:.2f})")
                row_types.append(tt)
            token_strs.append(row_strs)
            token_types.append(row_types)

        # Color matrix by token_type
        type_map = {"vision": 0.8, "text": 0.3, "pre_vision": 0.1}
        color_matrix = np.array([[type_map.get(t, 0.5) for t in row] for row in token_types])

        im = ax.imshow(color_matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

        # Annotate cells
        for i in range(len(all_models)):
            for j in range(len(layers)):
                ax.text(j, i, token_strs[i][j],
                       ha="center", va="center", fontsize=7, fontweight="bold",
                       color="black" if color_matrix[i, j] > 0.5 else "white")

        # Use first model's layer indices for x labels (they differ across models)
        first_report = list(contrib_reports.values())[0]
        ax.set_xticks(range(len(first_report["deep_layers"])))
        ax.set_xticklabels([str(l) for l in first_report["deep_layers"]], fontsize=9)
        ax.set_yticks(range(len(all_models)))
        ax.set_yticklabels(short_names, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")

    axes[-1].set_xlabel("Layer Index", fontsize=12)

    # Legend
    vision_patch = mpatches.Patch(facecolor=plt.cm.Reds(0.8), label="Vision region")
    text_patch = mpatches.Patch(facecolor=plt.cm.Reds(0.3), label="Text region")
    fig.legend(handles=[vision_patch, text_patch], loc="lower center", ncol=2, fontsize=10)

    fig.suptitle("Phase 2.5: Peak Token Identity Across Models & Layers", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(out_dir / "fig5_token_identity.png", dpi=200)
    plt.close(fig)
    print(f"  Saved: fig5_token_identity.png")


# ── Figure 6: Phi (Hidden State Spike) Comparison ─────────────
def fig6_phi_comparison(contrib_reports, out_dir):
    """φ values at A-peak and C-peak positions — higher = more outlier dimensions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for model_name, report in contrib_reports.items():
        cfg = MODELS[model_name]
        layers = report["deep_layers"]
        analysis = report["layer_analysis"]

        a_phis = []
        c_phis = []
        for l in layers:
            la = analysis.get(str(l), {})
            a_phi = la.get("a_peak", {}).get("phi", 0)
            c_phi = la.get("c_peak", {}).get("phi", 0)
            a_phis.append(a_phi)
            c_phis.append(c_phi)

        ax1.plot(range(len(layers)), a_phis,
                marker="^", linewidth=2, color=cfg["color"], label=cfg["short"])
        ax2.plot(range(len(layers)), c_phis,
                marker="s", linewidth=2, color=cfg["color"], label=cfg["short"])

    ax1.set_title("φ at A-peak (Attention Max)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("φ = max|x[d]| / RMS(x)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_title("φ at C-peak (Contribution Max)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("φ = max|x[d]| / RMS(x)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Hidden State Spike (φ) at Peak Positions — VAR Criterion", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "fig6_phi_comparison.png", dpi=200)
    plt.close(fig)
    print(f"  Saved: fig6_phi_comparison.png")


# ── Figure 7: Entropy Comparison ──────────────────────────────
def fig7_entropy_curves(contrib_reports, out_dir):
    """H(C̃) entropy per layer — low entropy = concentrated, high = distributed."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for model_name, report in contrib_reports.items():
        cfg = MODELS[model_name]
        layers = report["deep_layers"]
        analysis = report["layer_analysis"]

        entropies = []
        for l in layers:
            la = analysis.get(str(l), {})
            entropies.append(la.get("mean_entropy", 0))

        ax.plot(range(len(layers)), entropies,
                marker="D", linewidth=2, markersize=5,
                color=cfg["color"], label=cfg["short"])

    ax.set_xlabel("Layer (deep layers only)", fontsize=12)
    ax.set_ylabel("H(C̃) — Contribution Entropy", fontsize=12)
    ax.set_title("Contribution Distribution Entropy — Low = Concentrated (Bottleneck)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig7_entropy_curves.png", dpi=200)
    plt.close(fig)
    print(f"  Saved: fig7_entropy_curves.png")


# ── Figure 8: Model Taxonomy Summary ──────────────────────────
def fig8_model_taxonomy_table(contrib_reports, causal_reports, out_dir):
    """Summary table figure comparing all models on key metrics."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis("off")

    # Build data
    headers = [
        "Model", "Backbone", "Layers",
        "Dominant\nType", "A-C\nMatch",
        "Avg Top1\nC̃ Share", "Avg\nMismatch",
        "Avg\nEntropy",
        "A-peak\nToken", "C-peak\nToken",
        "V=0 KL\n(K=1)", "Top1\nChange",
        "φ(A)", "φ(C)"
    ]

    backbones = {"ecot-7b": "LLaMA-7B", "openvla-7b": "LLaMA-7B",
                 "spatialvla-4b": "Gemma2-2B", "tracevla-phi3v": "Phi3V-4B"}

    rows = []
    for model_name in MODELS:
        r = contrib_reports.get(model_name, {})
        cr = causal_reports.get(model_name, {})

        analysis = r.get("layer_analysis", {})
        layers = r.get("deep_layers", [])

        # Average metrics across layers
        top1_shares = [analysis.get(str(l), {}).get("mean_top1_share", 0) for l in layers]
        mismatches = [analysis.get(str(l), {}).get("mean_mismatch", 0) for l in layers]
        entropies = [analysis.get(str(l), {}).get("mean_entropy", 0) for l in layers]

        # Most common dominant type
        types = [analysis.get(str(l), {}).get("dominant_type", "?") for l in layers]
        dom_type = max(set(types), key=types.count) if types else "?"

        # A-C match rate (average)
        match_rates = [analysis.get(str(l), {}).get("a_c_match_rate", 0) for l in layers]

        # Representative peak tokens (from last layer)
        last_l = str(layers[-1]) if layers else "0"
        a_token = analysis.get(last_l, {}).get("a_peak", {}).get("token_str", "?")
        c_token = analysis.get(last_l, {}).get("c_peak", {}).get("token_str", "?")

        # Phi at last layer
        a_phi = analysis.get(last_l, {}).get("a_peak", {}).get("phi", 0)
        c_phi = analysis.get(last_l, {}).get("c_peak", {}).get("phi", 0)

        # Causal
        kl_k1 = cr.get("per_k", {}).get("1", {}).get("vzero_mean_kl", 0)
        top1_ch = cr.get("per_k", {}).get("1", {}).get("vzero_mean_top1_change", 0)

        rows.append([
            MODELS[model_name]["short"],
            backbones.get(model_name, "?"),
            str(r.get("n_layers", "?")),
            dom_type.upper(),
            f"{np.mean(match_rates):.0%}",
            f"{np.mean(top1_shares):.3f}",
            f"{np.mean(mismatches):.4f}",
            f"{np.mean(entropies):.2f}",
            a_token,
            c_token,
            f"{kl_k1:.2f}",
            f"{top1_ch:.0%}",
            f"{a_phi:.1f}",
            f"{c_phi:.1f}",
        ])

    # Color cells
    cell_colors = []
    type_colors = {
        "BOTTLENECK": "#ffcccc",
        "COEXIST": "#fff3cc",
        "NORMAL": "#ccffcc",
    }

    for row in rows:
        row_colors = ["white"] * len(row)
        dom = row[3]
        row_colors[3] = type_colors.get(dom, "white")
        # Color KL by severity
        kl_val = float(row[10])
        if kl_val > 10:
            row_colors[10] = "#ff9999"
        elif kl_val > 3:
            row_colors[10] = "#ffcc99"
        else:
            row_colors[10] = "#99ff99"
        cell_colors.append(row_colors)

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellColours=cell_colors,
        colColours=["#d4e6f1"] * len(headers),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.0)

    # Bold header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold", fontsize=8)
        cell.set_edgecolor("gray")

    ax.set_title("Phase 2.5: VLA Attention Routing Taxonomy — Cross-Model Summary",
                 fontsize=15, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(out_dir / "fig8_model_taxonomy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig8_model_taxonomy.png")


# ── Figure 9: Causal KL Scaling (K=1,3,5) ─────────────────────
def fig9_causal_scaling(causal_reports, out_dir):
    """How KL scales with number of masked tokens."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, report in causal_reports.items():
        cfg = MODELS[model_name]
        k_vals = sorted([int(k) for k in report["per_k"].keys()])
        kls = [report["per_k"][str(k)]["vzero_mean_kl"] for k in k_vals]
        ax.plot(k_vals, kls, marker="o", linewidth=2.5, markersize=8,
                color=cfg["color"], label=cfg["short"])

    ax.set_xlabel("Number of Masked Tokens (K)", fontsize=12)
    ax.set_ylabel("KL Divergence", fontsize=12)
    ax.set_title("V=0 Ablation Scaling: KL vs # Masked Tokens", fontsize=14, fontweight="bold")
    ax.set_xticks([1, 3, 5])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig9_causal_scaling.png", dpi=200)
    plt.close(fig)
    print(f"  Saved: fig9_causal_scaling.png")


# ── Main ───────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nPhase 2.5 Cross-Model Visualization")
    print(f"{'='*50}")

    contrib_reports, causal_reports = load_all_reports()
    print(f"Loaded: {len(contrib_reports)} contribution reports, {len(causal_reports)} causal reports")

    fig1_dual_track_peaks(contrib_reports, OUTPUT_DIR)
    fig2_top1_share_overlay(contrib_reports, OUTPUT_DIR)
    fig3_mismatch_curves(contrib_reports, OUTPUT_DIR)
    fig4_causal_kl_comparison(causal_reports, OUTPUT_DIR)
    fig5_token_identity_heatmap(contrib_reports, OUTPUT_DIR)
    fig6_phi_comparison(contrib_reports, OUTPUT_DIR)
    fig7_entropy_curves(contrib_reports, OUTPUT_DIR)
    fig8_model_taxonomy_table(contrib_reports, causal_reports, OUTPUT_DIR)
    fig9_causal_scaling(causal_reports, OUTPUT_DIR)

    print(f"\nAll figures saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
