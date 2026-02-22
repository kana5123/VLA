"""Cross-model comparison of VLA attention sink patterns.

Loads per-head JSON results from all models in the cross_model_analysis
directory and generates comparison visualizations and summary tables.

Usage:
    python cross_model_compare.py
    python cross_model_compare.py --base-dir outputs/cross_model_analysis
    python cross_model_compare.py --output-dir outputs/cross_model_comparison
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import config

# Use non-interactive backend for server environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════

def load_all_perhead(base_dir: Path) -> dict[str, dict]:
    """Load perhead JSONs from all model subdirectories.

    Scans base_dir/<model_name>/<dataset>/*_perhead.json and returns
    a dict mapping model_name -> perhead_analysis dict.

    If multiple perhead files exist for a model, uses the first one found.
    """
    results = {}
    if not base_dir.exists():
        print(f"Base directory not found: {base_dir}")
        return results

    for model_dir in sorted(base_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        # Search all dataset subdirectories for perhead JSON files
        perhead_files = sorted(model_dir.glob("**/*_perhead.json"))
        if not perhead_files:
            print(f"  No perhead JSON found for {model_name}, skipping")
            continue

        # Use the first perhead file found
        perhead_path = perhead_files[0]
        with open(perhead_path) as f:
            data = json.load(f)

        if "perhead_analysis" in data:
            results[model_name] = data["perhead_analysis"]
            print(f"  Loaded {model_name} from {perhead_path.name}")
        else:
            print(f"  No perhead_analysis key in {perhead_path}, skipping")

    return results


# ═══════════════════════════════════════════════════════════════════════
# Summary Computation
# ═══════════════════════════════════════════════════════════════════════

def compute_sink_summary(model_name: str, perhead: dict) -> dict:
    """Compute mean sink ratios across all layers/heads for one model.

    Args:
        model_name: Display name of the model.
        perhead: The perhead_analysis dict from extraction JSON.
            Structure: action_key -> layer_key -> head_key -> {vision_token0, ...}

    Returns:
        dict with:
            - model: model name
            - mean_vision0: mean vision[0] ratio across all layers/heads/actions
            - mean_text_total: mean text_total ratio
            - mean_vision_other: mean useful vision (vision_other) ratio
            - mean_action_tokens: mean action_tokens ratio
            - per_layer_vision0: list of per-layer mean vision[0] ratios
            - num_layers: number of layers found
            - num_heads: number of heads found
    """
    all_v0 = []
    all_text = []
    all_v_other = []
    all_action = []
    all_early_sink = []

    # Collect per-layer aggregates (averaged over actions and heads)
    layer_v0_accum = {}  # layer_key -> list of vision_token0 values
    layer_early_accum = {}  # layer_key -> list of early_sink values

    for action_key, layers_data in perhead.items():
        for layer_key, heads_data in layers_data.items():
            if layer_key not in layer_v0_accum:
                layer_v0_accum[layer_key] = []
                layer_early_accum[layer_key] = []

            for head_key, stats in heads_data.items():
                v0 = stats.get("vision_token0", 0.0)
                text = stats.get("text_total", 0.0)
                v_other = stats.get("vision_other", 0.0)
                action = stats.get("action_tokens", 0.0)
                early = stats.get("early_sink", v0)  # fallback to v0 for older data

                all_v0.append(v0)
                all_text.append(text)
                all_v_other.append(v_other)
                all_action.append(action)
                all_early_sink.append(early)
                layer_v0_accum[layer_key].append(v0)
                layer_early_accum[layer_key].append(early)

    # Per-layer mean vision[0], sorted by layer index
    sorted_layers = sorted(layer_v0_accum.keys())
    per_layer_vision0 = [
        float(np.mean(layer_v0_accum[lk])) for lk in sorted_layers
    ]
    per_layer_early_sink = [
        float(np.mean(layer_early_accum[lk])) for lk in sorted_layers
    ]

    # Detect num_heads from first action/layer
    num_heads = 0
    for action_key in perhead:
        for layer_key in perhead[action_key]:
            num_heads = len(perhead[action_key][layer_key])
            break
        break

    return {
        "model": model_name,
        "mean_vision0": float(np.mean(all_v0)) if all_v0 else 0.0,
        "mean_text_total": float(np.mean(all_text)) if all_text else 0.0,
        "mean_vision_other": float(np.mean(all_v_other)) if all_v_other else 0.0,
        "mean_action_tokens": float(np.mean(all_action)) if all_action else 0.0,
        "mean_early_sink": float(np.mean(all_early_sink)) if all_early_sink else 0.0,
        "per_layer_vision0": per_layer_vision0,
        "per_layer_early_sink": per_layer_early_sink,
        "num_layers": len(sorted_layers),
        "num_heads": num_heads,
        "layer_keys": sorted_layers,
    }


# ═══════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════

def plot_cross_model_comparison(summaries: dict[str, dict], output_dir: Path):
    """Generate all cross-model comparison figures.

    Produces:
        1. cross_model_sink_comparison.png - grouped bar chart of vision[0] mean per model
        2. cross_model_heatmap.png - heatmap (models x layers) of vision[0] fraction
        3. cross_model_dual_sink.png - stacked bar showing vision[0] + text + useful + action
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = list(summaries.keys())
    n_models = len(model_names)

    if n_models == 0:
        print("No models to compare.")
        return

    # ── Figure 1: Grouped bar chart of vision[0] mean per model ──
    fig1, ax1 = plt.subplots(figsize=(max(8, n_models * 1.5), 6))
    v0_means = [summaries[m]["mean_vision0"] for m in model_names]
    bars = ax1.bar(range(n_models), v0_means, color="firebrick", alpha=0.85, edgecolor="black")

    ax1.set_xticks(range(n_models))
    ax1.set_xticklabels(model_names, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("Mean Vision[0] Attention Ratio", fontsize=11)
    ax1.set_title("Cross-Model Attention Sink Comparison\n(Vision Token 0 = Sink)", fontsize=13)
    ax1.set_ylim(0, max(v0_means) * 1.25 if v0_means else 1.0)

    # Annotate bars with values
    for bar, val in zip(bars, v0_means):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax1.axhline(y=np.mean(v0_means), color="gray", linestyle="--", alpha=0.5,
                label=f"Mean: {np.mean(v0_means):.3f}")
    ax1.legend(fontsize=9)
    plt.tight_layout()
    path1 = output_dir / "cross_model_sink_comparison.png"
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved: {path1}")

    # ── Figure 2: Heatmap (models x layers) of vision[0] fraction ──
    max_layers = max(s["num_layers"] for s in summaries.values())
    heatmap_data = np.full((n_models, max_layers), np.nan)

    for i, m in enumerate(model_names):
        per_layer = summaries[m]["per_layer_vision0"]
        for j, val in enumerate(per_layer):
            heatmap_data[i, j] = val

    fig2, ax2 = plt.subplots(figsize=(max(10, max_layers * 0.4), max(4, n_models * 0.8)))
    im = ax2.imshow(heatmap_data, aspect="auto", cmap="YlOrRd", vmin=0,
                    vmax=np.nanmax(heatmap_data) if not np.all(np.isnan(heatmap_data)) else 1.0)

    ax2.set_yticks(range(n_models))
    ax2.set_yticklabels(model_names, fontsize=9)
    ax2.set_xlabel("Layer Index", fontsize=11)
    ax2.set_ylabel("Model", fontsize=11)
    ax2.set_title("Vision[0] Attention Sink by Layer\n(Heatmap: Models x Layers)", fontsize=13)

    # Layer ticks
    tick_step = max(1, max_layers // 16)
    ax2.set_xticks(range(0, max_layers, tick_step))
    ax2.set_xticklabels([str(i) for i in range(0, max_layers, tick_step)], fontsize=8)

    plt.colorbar(im, ax=ax2, label="Mean Vision[0] Ratio", shrink=0.8)
    plt.tight_layout()
    path2 = output_dir / "cross_model_heatmap.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {path2}")

    # ── Figure 3: Stacked bar showing vision[0] + text + useful + action per model ──
    fig3, ax3 = plt.subplots(figsize=(max(8, n_models * 1.5), 6))

    v0_vals = [summaries[m]["mean_vision0"] for m in model_names]
    text_vals = [summaries[m]["mean_text_total"] for m in model_names]
    useful_vals = [summaries[m]["mean_vision_other"] for m in model_names]
    action_vals = [summaries[m]["mean_action_tokens"] for m in model_names]

    x = np.arange(n_models)
    width = 0.6

    # Stacked bars
    b1 = ax3.bar(x, v0_vals, width, label="Vision[0] (sink)", color="firebrick", alpha=0.85)
    b2 = ax3.bar(x, text_vals, width, bottom=v0_vals, label="Text tokens", color="steelblue", alpha=0.85)
    bottom2 = [v + t for v, t in zip(v0_vals, text_vals)]
    b3 = ax3.bar(x, useful_vals, width, bottom=bottom2, label="Vision[1:] (useful)", color="forestgreen", alpha=0.85)
    bottom3 = [b + u for b, u in zip(bottom2, useful_vals)]
    b4 = ax3.bar(x, action_vals, width, bottom=bottom3, label="Action tokens", color="mediumpurple", alpha=0.85)

    ax3.set_xticks(x)
    ax3.set_xticklabels(model_names, rotation=45, ha="right", fontsize=9)
    ax3.set_ylabel("Mean Attention Fraction", fontsize=11)
    ax3.set_title("Cross-Model Attention Distribution Breakdown\n"
                   "(Stacked: Sink + Text + Useful Vision + Action)", fontsize=13)
    ax3.set_ylim(0, 1.05)
    ax3.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    path3 = output_dir / "cross_model_dual_sink.png"
    fig3.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved: {path3}")


# ═══════════════════════════════════════════════════════════════════════
# LaTeX Table Generation
# ═══════════════════════════════════════════════════════════════════════

def generate_latex_table(summaries: dict[str, dict], output_dir: Path):
    """Generate LaTeX table of sink ratios per model.

    Produces cross_model_table.tex with columns:
        Model | Layers | Heads | Vision[0] | Text | Vision[1:] | Action
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Cross-model attention distribution analysis. "
                 r"Vision[0] denotes the attention sink token; Vision[1:] denotes "
                 r"useful (non-sink) vision tokens. All values are mean attention "
                 r"fractions averaged across all layers, heads, and action tokens.}")
    lines.append(r"  \label{tab:cross_model_sink}")
    lines.append(r"  \begin{tabular}{lcccccc}")
    lines.append(r"    \toprule")
    lines.append(r"    Model & Layers & Heads & Vision[0] $\downarrow$ & Text & "
                 r"Vision[1:] $\uparrow$ & Action \\")
    lines.append(r"    \midrule")

    for model_name, summary in summaries.items():
        v0 = summary["mean_vision0"]
        text = summary["mean_text_total"]
        useful = summary["mean_vision_other"]
        action = summary["mean_action_tokens"]
        n_layers = summary["num_layers"]
        n_heads = summary["num_heads"]

        # Format model name for LaTeX (escape underscores)
        latex_name = model_name.replace("_", r"\_")

        lines.append(
            f"    {latex_name} & {n_layers} & {n_heads} & "
            f"{v0:.3f} & {text:.3f} & {useful:.3f} & {action:.3f} \\\\"
        )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    tex_content = "\n".join(lines)

    tex_path = output_dir / "cross_model_table.tex"
    with open(tex_path, "w") as f:
        f.write(tex_content)
    print(f"  Saved: {tex_path}")

    return tex_content


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Cross-model VLA attention sink comparison and visualization"
    )
    parser.add_argument(
        "--base-dir", type=str,
        default=str(config.OUTPUT_DIR / "cross_model_analysis"),
        help="Base directory containing per-model extraction results",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=None,
        help="Output directory for comparison figures (default: base_dir/comparison)",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "comparison"

    print(f"Loading per-head data from: {base_dir}")
    all_perhead = load_all_perhead(base_dir)

    if not all_perhead:
        print("No per-head data found. Run cross_model_extract.py first.")
        return

    print(f"\nFound {len(all_perhead)} models: {list(all_perhead.keys())}")

    # Compute summaries
    print("\nComputing sink summaries...")
    summaries = {}
    for model_name, perhead in all_perhead.items():
        summary = compute_sink_summary(model_name, perhead)
        summaries[model_name] = summary
        print(f"  {model_name}: vision[0]={summary['mean_vision0']:.3f}, "
              f"text={summary['mean_text_total']:.3f}, "
              f"useful={summary['mean_vision_other']:.3f}, "
              f"action={summary['mean_action_tokens']:.3f}, "
              f"early_sink={summary['mean_early_sink']:.3f}")

    # Save summary JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "cross_model_summary.json"
    # Convert numpy types for JSON serialization
    serializable = {}
    for k, v in summaries.items():
        s = {}
        for sk, sv in v.items():
            if isinstance(sv, (np.floating, np.integer)):
                s[sk] = float(sv)
            elif isinstance(sv, list):
                s[sk] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in sv]
            else:
                s[sk] = sv
        serializable[k] = s

    with open(summary_path, "w") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\nSaved summary: {summary_path}")

    # Generate plots
    print("\nGenerating comparison plots...")
    plot_cross_model_comparison(summaries, output_dir)

    # Generate LaTeX table
    print("\nGenerating LaTeX table...")
    tex = generate_latex_table(summaries, output_dir)
    print("\nLaTeX table preview:")
    print(tex)

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
