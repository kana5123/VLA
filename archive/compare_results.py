"""Aggregate enhancement experiment results and generate comparison report.

Usage:
    python compare_results.py

Produces:
    outputs/enhancement_results/comparison_summary.json
    outputs/enhancement_results/mse_comparison.png
    outputs/enhancement_results/mse_timestep.png
    outputs/enhancement_results/mse_per_dim.png
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

METHODS = ["baseline", "logit_bias", "weight_rescale", "head_steering"]
METHOD_COLORS = {
    "baseline": "#555555",
    "logit_bias": "#2196F3",
    "weight_rescale": "#4CAF50",
    "head_steering": "#FF9800",
}


def load_method_results(results_dir: Path, method: str) -> list[dict]:
    method_dir = results_dir / method
    if not method_dir.exists():
        return []
    return [
        json.loads(f.read_text())
        for f in sorted(method_dir.glob("ep*_step*.json"))
    ]


def aggregate(results: list[dict]) -> dict:
    if not results:
        return {}

    per_episode = defaultdict(list)
    per_step = []
    dim_acc = []

    for r in results:
        ep_id = r["episode_id"]
        per_episode[ep_id].append(r["mse_mean"])
        per_step.append((ep_id, r["step_id"], r["mse_mean"]))
        dim_acc.append(r["mse_per_dim"])

    episode_avgs = {ep: float(np.mean(v)) for ep, v in per_episode.items()}
    return {
        "episode_avg_mse": float(np.mean(list(episode_avgs.values()))),
        "per_episode": episode_avgs,
        "per_step_mse": per_step,
        "per_dim_avg": np.mean(dim_acc, axis=0).tolist(),
    }


def build_summary(results_dir: Path) -> dict:
    summary = {}
    for method in METHODS:
        raw = load_method_results(results_dir, method)
        if raw:
            summary[method] = aggregate(raw)
        else:
            print(f"  No results for '{method}'")
    return summary


def print_table(summary: dict) -> None:
    print("\n" + "=" * 70)
    print(f"{'Method':<20} {'Episode-Avg MSE':>16} {'vs Baseline':>16}")
    print("-" * 70)
    bl = summary.get("baseline", {}).get("episode_avg_mse")
    for m in METHODS:
        if m not in summary:
            continue
        mse = summary[m]["episode_avg_mse"]
        if bl is not None and m != "baseline":
            d = mse - bl
            pct = 100 * d / bl if bl != 0 else 0
            change = f"{d:+.6f} ({pct:+.1f}%)"
        else:
            change = "-"
        print(f"  {m:<18} {mse:>16.6f} {change:>16}")
    print("=" * 70)

    present = [m for m in METHODS if m in summary]
    print(f"\n{'Dim':<10}", end="")
    for m in present:
        print(f"  {m[:14]:>14}", end="")
    print()
    print("-" * (10 + 16 * len(present)))
    for i, name in enumerate(config.ACTION_DIM_NAMES):
        print(f"  {name:<8}", end="")
        for m in present:
            val = summary[m].get("per_dim_avg", [0]*7)[i]
            print(f"  {val:>14.6f}", end="")
        print()


def plot_episode_bar(summary: dict, out_path: Path) -> None:
    present = [m for m in METHODS if m in summary]
    vals = [summary[m]["episode_avg_mse"] for m in present]
    colors = [METHOD_COLORS[m] for m in present]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(present, vals, color=colors, edgecolor="black", linewidth=0.7)
    ax.bar_label(bars, fmt="%.5f", padding=3, fontsize=9)
    ax.set_ylabel("Episode-Average MSE")
    ax.set_title("Attention Enhancement: Episode-Average Action MSE")
    ax.set_ylim(0, max(vals) * 1.15 if vals else 1)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_per_timestep(summary: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    for m in METHODS:
        if m not in summary:
            continue
        steps = summary[m]["per_step_mse"]
        if not steps:
            continue
        y = [s[2] for s in steps]
        ax.plot(range(len(y)), y, label=m, color=METHOD_COLORS[m], alpha=0.8, linewidth=1.2)

    ax.set_xlabel("Timestep (all episodes concatenated)")
    ax.set_ylabel("MSE (7-dim mean)")
    ax.set_title("Per-Timestep Action MSE: Baseline vs Enhancement Methods")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_per_dim(summary: dict, out_path: Path) -> None:
    present = [m for m in METHODS if m in summary]
    dims = config.ACTION_DIM_NAMES
    n_dims = len(dims)
    n_m = len(present)
    x = np.arange(n_dims)
    width = 0.8 / max(n_m, 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, m in enumerate(present):
        vals = summary[m].get("per_dim_avg", [0]*n_dims)
        offset = (i - n_m / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=m,
               color=METHOD_COLORS[m], edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(dims)
    ax.set_ylabel("Mean MSE")
    ax.set_title("Per Action-Dimension MSE by Enhancement Method")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def to_native(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(i) for i in obj]
    return obj


def run_comparison(results_dir: Path = None) -> dict:
    if results_dir is None:
        results_dir = config.ENHANCEMENT_RESULTS_DIR

    print(f"Loading results from {results_dir}/...")
    summary = build_summary(results_dir)

    if not summary:
        print("No results found. Run run_enhancement.py first.")
        return {}

    print_table(summary)

    summary_path = results_dir / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(to_native(summary), f, indent=2)
    print(f"\nSummary JSON: {summary_path}")

    plot_episode_bar(summary, results_dir / "mse_comparison.png")
    plot_per_timestep(summary, results_dir / "mse_timestep.png")
    plot_per_dim(summary, results_dir / "mse_per_dim.png")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Compare enhancement results")
    parser.add_argument("--results_dir", type=str, default=None)
    args = parser.parse_args()
    results_dir = Path(args.results_dir) if args.results_dir else None
    run_comparison(results_dir)


if __name__ == "__main__":
    main()
