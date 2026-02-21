"""Aggregate V2 enhancement experiment results and produce ranking.

Usage:
    python compare_v2_results.py
    python compare_v2_results.py --results_dir outputs/v2_enhancement_results
    python compare_v2_results.py --top 5

Produces:
    comparison_summary.json   — full aggregate data
    ranking_overall.png       — bar chart ranked by overall MSE
    ranking_spatial.png       — bar chart ranked by spatial-only MSE
    mse_heatmap.png           — episode × condition heatmap
    mse_per_dim.png           — per-dimension grouped bar chart
    mse_timestep_top5.png     — timestep trace for top-5 conditions
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import config

# Import condition list for ordering
try:
    from run_v2_experiment import ALL_CONDITION_NAMES, CONDITIONS, describe_condition
except ImportError:
    ALL_CONDITION_NAMES = None
    CONDITIONS = None
    describe_condition = lambda x: x


def load_condition_results(results_dir: Path, condition: str) -> list[dict]:
    cond_dir = results_dir / condition
    if not cond_dir.exists():
        return []
    return [
        json.loads(f.read_text())
        for f in sorted(cond_dir.glob("ep*_step*.json"))
    ]


def aggregate(results: list[dict]) -> dict:
    if not results:
        return {}

    per_episode = defaultdict(list)
    per_episode_spatial = defaultdict(list)
    per_episode_gripper = defaultdict(list)
    per_step = []
    dim_acc = []

    for r in results:
        ep = r["episode_id"]
        per_episode[ep].append(r["mse_mean"])
        per_episode_spatial[ep].append(r.get("mse_spatial", float(np.mean(r["mse_per_dim"][:6]))))
        per_episode_gripper[ep].append(r.get("mse_gripper", r["mse_per_dim"][6] if len(r["mse_per_dim"]) > 6 else 0))
        per_step.append((ep, r["step_id"], r["mse_mean"]))
        dim_acc.append(r["mse_per_dim"])

    episode_avgs = {ep: float(np.mean(v)) for ep, v in per_episode.items()}
    episode_spatial = {ep: float(np.mean(v)) for ep, v in per_episode_spatial.items()}
    episode_gripper = {ep: float(np.mean(v)) for ep, v in per_episode_gripper.items()}

    return {
        "mse_overall": float(np.mean(list(episode_avgs.values()))),
        "mse_spatial": float(np.mean(list(episode_spatial.values()))),
        "mse_gripper": float(np.mean(list(episode_gripper.values()))),
        "per_episode_mse": episode_avgs,
        "per_episode_spatial": episode_spatial,
        "per_episode_gripper": episode_gripper,
        "per_step_mse": per_step,
        "per_dim_avg": np.mean(dim_acc, axis=0).tolist(),
        "num_steps": len(results),
    }


def discover_conditions(results_dir: Path) -> list[str]:
    """Find all condition dirs that have result files."""
    found = []
    for d in sorted(results_dir.iterdir()):
        if d.is_dir() and list(d.glob("ep*_step*.json")):
            found.append(d.name)
    return found


def build_summary(results_dir: Path) -> dict:
    conditions = discover_conditions(results_dir)
    summary = {}
    for cond in conditions:
        raw = load_condition_results(results_dir, cond)
        if raw:
            summary[cond] = aggregate(raw)
        else:
            print(f"  [skip] No results for '{cond}'")
    return summary


def print_ranking(summary: dict) -> None:
    if not summary:
        return

    bl = summary.get("baseline", {}).get("mse_overall")

    # Rank by overall MSE
    ranked = sorted(summary.items(), key=lambda x: x[1]["mse_overall"])

    print(f"\n{'='*90}")
    print(f"  V2 ENHANCEMENT RESULTS — RANKED BY OVERALL MSE")
    print(f"{'='*90}")
    print(f"  {'#':>3} {'Condition':<30} {'Overall':>10} {'Spatial':>10} {'Gripper':>10} {'vs BL':>10}")
    print(f"  {'-'*83}")

    for i, (name, data) in enumerate(ranked):
        mse = data["mse_overall"]
        sp = data["mse_spatial"]
        gr = data["mse_gripper"]
        if bl is not None and name != "baseline":
            delta = 100 * (mse - bl) / bl if bl != 0 else 0
            vs = f"{delta:+.1f}%"
        else:
            vs = "-"
        marker = " *" if name == "baseline" else ""
        print(f"  {i+1:>3} {name:<30} {mse:>10.6f} {sp:>10.6f} {gr:>10.6f} {vs:>10}{marker}")

    # Also rank by spatial-only
    ranked_sp = sorted(summary.items(), key=lambda x: x[1]["mse_spatial"])
    print(f"\n  {'─'*83}")
    print(f"  RANKED BY SPATIAL MSE (excl. gripper)")
    print(f"  {'─'*83}")
    for i, (name, data) in enumerate(ranked_sp[:10]):
        sp = data["mse_spatial"]
        if bl is not None and name != "baseline":
            bl_sp = summary["baseline"]["mse_spatial"]
            delta = 100 * (sp - bl_sp) / bl_sp if bl_sp != 0 else 0
            vs = f"{delta:+.1f}%"
        else:
            vs = "-"
        print(f"  {i+1:>3} {name:<30} {sp:>10.6f} {vs:>10}")

    # Per-dimension table
    print(f"\n  {'─'*83}")
    print(f"  PER-DIMENSION MSE (top 5 by overall)")
    print(f"  {'─'*83}")
    top5 = [name for name, _ in ranked[:6]]  # Include baseline + top 5
    if "baseline" not in top5 and "baseline" in summary:
        top5.insert(0, "baseline")

    header = f"  {'Dim':<10}"
    for name in top5:
        header += f" {name[:12]:>12}"
    print(header)
    print(f"  {'-'*(10 + 13*len(top5))}")
    for i, dim_name in enumerate(config.ACTION_DIM_NAMES):
        row = f"  {dim_name:<10}"
        for name in top5:
            val = summary[name].get("per_dim_avg", [0]*7)[i]
            row += f" {val:>12.6f}"
        print(row)

    # Improvement count
    print(f"\n  {'─'*83}")
    print(f"  PER-EPISODE IMPROVEMENT COUNT (overall MSE < baseline)")
    print(f"  {'─'*83}")
    if bl is not None:
        bl_ep = summary["baseline"]["per_episode_mse"]
        for name, data in ranked:
            if name == "baseline":
                continue
            improved = 0
            total = 0
            for ep, ep_mse in data["per_episode_mse"].items():
                total += 1
                if ep in bl_ep and ep_mse < bl_ep[ep]:
                    improved += 1
            print(f"  {name:<30} {improved}/{total} episodes improved")

    print(f"{'='*90}")


def to_native(obj):
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(i) for i in obj]
    return obj


# ======================================================================
# Plots
# ======================================================================

def get_colors(n: int):
    """Generate distinct colors for n conditions."""
    cmap = cm.get_cmap("tab20", max(n, 20))
    return [cmap(i) for i in range(n)]


def plot_ranking_bar(summary: dict, metric: str, title: str, out_path: Path) -> None:
    ranked = sorted(summary.items(), key=lambda x: x[1][metric])
    names = [n for n, _ in ranked]
    vals = [d[metric] for _, d in ranked]
    colors = get_colors(len(names))

    # Highlight baseline
    bar_colors = []
    for i, n in enumerate(names):
        if n == "baseline":
            bar_colors.append("#333333")
        else:
            bar_colors.append(colors[i])

    fig, ax = plt.subplots(figsize=(max(12, len(names) * 0.7), 6))
    bars = ax.barh(range(len(names)), vals, color=bar_colors, edgecolor="black", linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.5f}", va="center", fontsize=8)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel(f"{metric.replace('_', ' ').title()}")
    ax.set_title(title)
    ax.invert_yaxis()

    # Baseline reference line
    bl_val = summary.get("baseline", {}).get(metric)
    if bl_val is not None:
        ax.axvline(bl_val, color="red", linestyle="--", alpha=0.7, label=f"baseline={bl_val:.5f}")
        ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_heatmap(summary: dict, out_path: Path) -> None:
    """Episode × condition heatmap of MSE."""
    conditions = sorted(summary.keys())
    all_eps = sorted(set(
        ep for data in summary.values()
        for ep in data.get("per_episode_mse", {}).keys()
    ))

    if not conditions or not all_eps:
        return

    matrix = np.full((len(conditions), len(all_eps)), np.nan)
    for i, cond in enumerate(conditions):
        for j, ep in enumerate(all_eps):
            val = summary[cond].get("per_episode_mse", {}).get(ep)
            if val is not None:
                matrix[i, j] = val

    fig, ax = plt.subplots(figsize=(max(10, len(all_eps) * 0.8), max(8, len(conditions) * 0.4)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(all_eps)))
    ax.set_xticklabels([f"ep{e}" for e in all_eps], fontsize=8, rotation=45)
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels(conditions, fontsize=8)
    ax.set_title("MSE Heatmap: Condition × Episode")
    fig.colorbar(im, ax=ax, label="MSE")

    # Annotate cells
    for i in range(len(conditions)):
        for j in range(len(all_eps)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i,j]:.4f}", ha="center", va="center",
                        fontsize=6, color="white" if matrix[i,j] > np.nanmedian(matrix) else "black")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_per_dim(summary: dict, out_path: Path) -> None:
    """Per-dimension grouped bar chart for top conditions + baseline."""
    ranked = sorted(summary.items(), key=lambda x: x[1]["mse_overall"])
    # Top 5 + baseline
    selected = []
    for name, _ in ranked[:6]:
        selected.append(name)
    if "baseline" in summary and "baseline" not in selected:
        selected.insert(0, "baseline")

    dims = config.ACTION_DIM_NAMES
    n_dims = len(dims)
    n_m = len(selected)
    x = np.arange(n_dims)
    width = 0.8 / max(n_m, 1)
    colors = get_colors(n_m)

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, name in enumerate(selected):
        vals = summary[name].get("per_dim_avg", [0] * n_dims)
        offset = (i - n_m / 2 + 0.5) * width
        color = "#333333" if name == "baseline" else colors[i]
        ax.bar(x + offset, vals, width, label=name, color=color, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(dims)
    ax.set_ylabel("Mean MSE")
    ax.set_title("Per Action-Dimension MSE: Top Conditions vs Baseline")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_timestep_top(summary: dict, out_path: Path, top_n: int = 5) -> None:
    """Per-timestep MSE trace for top-N conditions."""
    ranked = sorted(summary.items(), key=lambda x: x[1]["mse_overall"])
    selected = [name for name, _ in ranked[:top_n + 1]]
    if "baseline" in summary and "baseline" not in selected:
        selected.insert(0, "baseline")

    colors = get_colors(len(selected))

    fig, ax = plt.subplots(figsize=(16, 5))
    for i, name in enumerate(selected):
        steps = summary[name].get("per_step_mse", [])
        if not steps:
            continue
        y = [s[2] for s in steps]
        color = "#333333" if name == "baseline" else colors[i]
        ax.plot(range(len(y)), y, label=name, color=color, alpha=0.8, linewidth=1.0)

    ax.set_xlabel("Timestep (all episodes concatenated)")
    ax.set_ylabel("MSE (7-dim mean)")
    ax.set_title(f"Per-Timestep MSE: Baseline vs Top-{top_n} Conditions")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ======================================================================
# Main
# ======================================================================

def run_v2_comparison(results_dir: Path = None, top_n: int = 5) -> dict:
    if results_dir is None:
        results_dir = config.V2_RESULTS_DIR

    print(f"Loading V2 results from {results_dir}/...")
    summary = build_summary(results_dir)

    if not summary:
        print("No results found. Run run_v2_experiment.py first.")
        return {}

    print_ranking(summary)

    # Save JSON
    summary_path = results_dir / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(to_native(summary), f, indent=2)
    print(f"\nSummary JSON: {summary_path}")

    # Generate plots
    plot_ranking_bar(summary, "mse_overall", "V2: Overall MSE Ranking", results_dir / "ranking_overall.png")
    plot_ranking_bar(summary, "mse_spatial", "V2: Spatial MSE Ranking", results_dir / "ranking_spatial.png")
    plot_heatmap(summary, results_dir / "mse_heatmap.png")
    plot_per_dim(summary, results_dir / "mse_per_dim.png")
    plot_timestep_top(summary, results_dir / "mse_timestep_top5.png", top_n=top_n)

    # Find best condition
    if len(summary) > 1:
        bl = summary.get("baseline", {}).get("mse_overall")
        non_bl = [(n, d) for n, d in summary.items() if n != "baseline"]
        best_overall = min(non_bl, key=lambda x: x[1]["mse_overall"])
        best_spatial = min(non_bl, key=lambda x: x[1]["mse_spatial"])

        print(f"\n  BEST OVERALL:  {best_overall[0]}  (MSE={best_overall[1]['mse_overall']:.6f})")
        print(f"  BEST SPATIAL:  {best_spatial[0]}  (MSE={best_spatial[1]['mse_spatial']:.6f})")
        if bl is not None:
            delta_o = 100 * (best_overall[1]["mse_overall"] - bl) / bl if bl else 0
            delta_s = 100 * (best_spatial[1]["mse_spatial"] - summary["baseline"]["mse_spatial"]) / summary["baseline"]["mse_spatial"] if summary["baseline"]["mse_spatial"] else 0
            print(f"  vs baseline:   overall {delta_o:+.1f}%,  spatial {delta_s:+.1f}%")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Compare V2 enhancement results")
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--top", type=int, default=5, help="Number of top conditions to show in timestep plot")
    args = parser.parse_args()
    results_dir = Path(args.results_dir) if args.results_dir else None
    run_v2_comparison(results_dir, top_n=args.top)


if __name__ == "__main__":
    main()
