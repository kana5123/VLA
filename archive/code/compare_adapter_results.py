"""Compare adapter experiment results across all configurations.

Loads eval_results.json from each config directory and produces:
1. Summary table (JSON + stdout)
2. Per-dimension MSE bar chart
3. Improvement heatmap
4. LaTeX table for paper

Usage:
    python compare_adapter_results.py
    python compare_adapter_results.py --experiment_dir outputs/experiment_results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import config

EXPERIMENT_DIR = config.OUTPUT_DIR / "experiment_results"
DIM_NAMES = config.ACTION_DIM_NAMES  # ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def load_results(experiment_dir: Path) -> dict[str, dict]:
    """Load eval_results.json from each config directory."""
    results = {}
    for config_dir in sorted(experiment_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        eval_path = config_dir / "eval" / "eval_results.json"
        if eval_path.exists():
            data = json.loads(eval_path.read_text())
            results[config_dir.name] = data
            print(f"  Loaded: {config_dir.name} ({eval_path})")
        else:
            print(f"  Skipped: {config_dir.name} (no eval_results.json)")
    return results


def compute_comparison(results: dict[str, dict]) -> dict:
    """Compute comparison statistics relative to baseline."""
    comparison = {}

    base_summary = None
    if "base" in results and "baseline" in results["base"]:
        base_summary = results["base"]["baseline"].get("summary")
    elif "base" in results and "summary" in results["base"]:
        base_summary = results["base"]["summary"]

    if base_summary is None:
        # Use first config's baseline as reference
        for name, data in results.items():
            if "baseline" in data and "summary" in data["baseline"]:
                base_summary = data["baseline"]["summary"]
                break

    if base_summary is None:
        print("WARNING: No baseline found for comparison")
        return comparison

    for name, data in results.items():
        # Get the adapter summary (or baseline for base config)
        if name == "base":
            summary = base_summary
        elif "adapter" in data and "summary" in data["adapter"]:
            summary = data["adapter"]["summary"]
        elif "summary" in data:
            summary = data["summary"]
        else:
            continue

        comparison[name] = {
            "overall_mse": summary["overall_mse"],
            "spatial_mse": summary["spatial_mse"],
            "rotational_mse": summary.get("rotational_mse", 0),
            "per_dim_mse": summary["per_dim_mse"],
            "n_steps": summary.get("n_steps", 0),
        }

        if name != "base":
            base_overall = base_summary["overall_mse"]
            comparison[name]["overall_change_pct"] = (
                (summary["overall_mse"] - base_overall) / base_overall * 100
            )
            comparison[name]["spatial_change_pct"] = (
                (summary["spatial_mse"] - base_summary["spatial_mse"])
                / base_summary["spatial_mse"] * 100
            )
            comparison[name]["per_dim_change_pct"] = {}
            for dim in DIM_NAMES:
                base_dim = base_summary["per_dim_mse"][dim]
                if base_dim > 0:
                    comparison[name]["per_dim_change_pct"][dim] = (
                        (summary["per_dim_mse"][dim] - base_dim) / base_dim * 100
                    )

    return comparison


def print_summary(comparison: dict):
    """Print formatted summary table to stdout."""
    print(f"\n{'=' * 70}")
    print("ADAPTER EXPERIMENT RESULTS COMPARISON")
    print(f"{'=' * 70}")

    # Header
    configs = list(comparison.keys())
    header = f"{'Metric':<20s}"
    for name in configs:
        header += f" {name:>12s}"
    print(header)
    print("-" * 70)

    # Overall MSE
    row = f"{'Overall MSE':<20s}"
    for name in configs:
        row += f" {comparison[name]['overall_mse']:>12.6f}"
    print(row)

    # Spatial MSE
    row = f"{'Spatial MSE':<20s}"
    for name in configs:
        row += f" {comparison[name]['spatial_mse']:>12.6f}"
    print(row)

    # Per-dimension
    print(f"\n{'Per-Dimension MSE':}")
    for dim in DIM_NAMES:
        row = f"  {dim:<18s}"
        for name in configs:
            val = comparison[name]["per_dim_mse"].get(dim, 0)
            row += f" {val:>12.6f}"
        print(row)

    # Change %
    print(f"\n{'Change vs Baseline (%)'}")
    row = f"{'Overall':<20s}"
    for name in configs:
        pct = comparison[name].get("overall_change_pct", 0)
        row += f" {pct:>+11.2f}%"
    print(row)

    for dim in DIM_NAMES:
        row = f"  {dim:<18s}"
        for name in configs:
            pct = comparison[name].get("per_dim_change_pct", {}).get(dim, 0)
            row += f" {pct:>+11.2f}%"
        print(row)

    print(f"{'=' * 70}")


def plot_per_dim_bar(comparison: dict, output_dir: Path):
    """Bar chart of per-dimension MSE for each config."""
    configs = list(comparison.keys())
    n_configs = len(configs)
    n_dims = len(DIM_NAMES)
    x = np.arange(n_dims)
    width = 0.8 / n_configs

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, n_configs))

    for i, name in enumerate(configs):
        values = [comparison[name]["per_dim_mse"].get(d, 0) for d in DIM_NAMES]
        offset = (i - n_configs / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=name, color=colors[i])

    ax.set_xlabel("Action Dimension")
    ax.set_ylabel("MSE")
    ax.set_title("Per-Dimension MSE by Adapter Configuration")
    ax.set_xticks(x)
    ax.set_xticklabels(DIM_NAMES)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "per_dim_mse_bar.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_improvement_heatmap(comparison: dict, output_dir: Path):
    """Heatmap of per-dimension improvement % vs baseline."""
    configs = [c for c in comparison if c != "base"]
    if not configs:
        return

    n_configs = len(configs)
    data = np.zeros((n_configs, len(DIM_NAMES)))

    for i, name in enumerate(configs):
        for j, dim in enumerate(DIM_NAMES):
            data[i, j] = comparison[name].get("per_dim_change_pct", {}).get(dim, 0)

    fig, ax = plt.subplots(figsize=(10, 4))
    vmax = max(abs(data.min()), abs(data.max()), 1)
    im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(DIM_NAMES)))
    ax.set_xticklabels(DIM_NAMES)
    ax.set_yticks(range(n_configs))
    ax.set_yticklabels(configs)
    ax.set_title("MSE Change (%) vs Baseline (green = improvement)")

    for i in range(n_configs):
        for j in range(len(DIM_NAMES)):
            ax.text(j, i, f"{data[i, j]:+.1f}%", ha="center", va="center", fontsize=9)

    plt.colorbar(im, ax=ax, label="Change %")
    plt.tight_layout()
    path = output_dir / "improvement_heatmap.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def generate_latex_table(comparison: dict, output_dir: Path):
    """Generate LaTeX table for paper inclusion."""
    configs = list(comparison.keys())

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Adapter experiment results: per-dimension MSE and change vs.\ baseline.}",
        r"\label{tab:adapter-results}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "c" * len(configs) + "}",
        r"\toprule",
    ]

    # Header
    header = "Metric"
    for name in configs:
        header += f" & {name}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Overall MSE
    row = "Overall MSE"
    for name in configs:
        row += f" & {comparison[name]['overall_mse']:.6f}"
    row += r" \\"
    lines.append(row)

    # Spatial MSE
    row = "Spatial MSE"
    for name in configs:
        row += f" & {comparison[name]['spatial_mse']:.6f}"
    row += r" \\"
    lines.append(row)

    lines.append(r"\midrule")

    # Per-dim
    for dim in DIM_NAMES:
        row = dim
        for name in configs:
            val = comparison[name]["per_dim_mse"].get(dim, 0)
            row += f" & {val:.6f}"
        row += r" \\"
        lines.append(row)

    lines.append(r"\midrule")

    # Change %
    row = r"$\Delta$ Overall (\%)"
    for name in configs:
        pct = comparison[name].get("overall_change_pct", 0)
        row += f" & {pct:+.2f}"
    row += r" \\"
    lines.append(row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ])

    path = output_dir / "results_table.tex"
    path.write_text("\n".join(lines))
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Compare adapter experiment results")
    parser.add_argument("--experiment_dir", type=str, default=str(EXPERIMENT_DIR))
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    print(f"Loading results from: {experiment_dir}")

    results = load_results(experiment_dir)
    if not results:
        print("No results found!")
        return

    comparison = compute_comparison(results)
    if not comparison:
        print("Could not compute comparison (no baseline)")
        return

    print_summary(comparison)

    # Plots
    plot_dir = experiment_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_per_dim_bar(comparison, plot_dir)
    plot_improvement_heatmap(comparison, plot_dir)
    generate_latex_table(comparison, plot_dir)

    # Save comparison JSON
    comp_path = experiment_dir / "comparison_summary.json"
    # Convert numpy to native types for JSON
    def to_native(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2, default=to_native)
    print(f"\nComparison saved: {comp_path}")


if __name__ == "__main__":
    main()
