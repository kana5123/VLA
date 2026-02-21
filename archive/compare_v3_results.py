"""Aggregate V3 enhancement experiment results and produce ranking.

Usage:
    python compare_v3_results.py
    python compare_v3_results.py --top 5
    python compare_v3_results.py --filtered   # SAM2 patches>0 episodes only

Produces comparison_summary.json + ranking/heatmap/per-dim/timestep plots.
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

try:
    from run_v3_experiment import ALL_CONDITION_NAMES, describe_condition
except ImportError:
    ALL_CONDITION_NAMES = None
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
        per_episode_spatial[ep].append(
            r.get("mse_spatial", float(np.mean(r["mse_per_dim"][:6])))
        )
        per_episode_gripper[ep].append(
            r.get("mse_gripper", r["mse_per_dim"][6] if len(r["mse_per_dim"]) > 6 else 0)
        )
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
    return [
        d.name for d in sorted(results_dir.iterdir())
        if d.is_dir() and list(d.glob("ep*_step*.json"))
    ]


def build_summary(results_dir: Path) -> dict:
    conditions = discover_conditions(results_dir)
    summary = {}
    for cond in conditions:
        raw = load_condition_results(results_dir, cond)
        if raw:
            summary[cond] = aggregate(raw)
    return summary


# ── Filtered comparison (SAM2 patches > 0 only) ──────────────────────

def find_sam2_grounded_episodes(results_dir: Path) -> set:
    """Find episodes where SAM2 grounding produced patches > 0.

    Uses the FIRST step (step_id=0) of each episode as representative.
    An episode counts as "grounded" if any SAM2 condition found
    patches > 0 for that episode's first step.
    """
    sam2_conditions = [
        d.name for d in sorted(results_dir.iterdir())
        if d.is_dir() and d.name.startswith("sam_")
    ]
    grounded_eps = set()
    for cond in sam2_conditions:
        results = load_condition_results(results_dir, cond)
        for r in results:
            if r["step_id"] != 0:
                continue
            g = r.get("grounding", {})
            if g.get("num_patches", 0) > 0:
                grounded_eps.add(r["episode_id"])
    return grounded_eps


def build_filtered_summary(results_dir: Path, keep_episodes: set) -> dict:
    """Build summary using only specified episodes."""
    conditions = discover_conditions(results_dir)
    summary = {}
    for cond in conditions:
        raw = load_condition_results(results_dir, cond)
        filtered = [r for r in raw if r["episode_id"] in keep_episodes]
        if filtered:
            summary[cond] = aggregate(filtered)
    return summary


def run_filtered_comparison(results_dir: Path) -> dict:
    """Compare conditions using only episodes where SAM2 grounding succeeded."""
    grounded_eps = find_sam2_grounded_episodes(results_dir)
    if not grounded_eps:
        print("No SAM2-grounded episodes found.")
        return {}

    print(f"\n{'='*95}")
    print(f"  FILTERED ANALYSIS: SAM2-grounded episodes only")
    print(f"  Episodes with patches > 0: {sorted(grounded_eps)} ({len(grounded_eps)} total)")
    print(f"{'='*95}")

    summary = build_filtered_summary(results_dir, grounded_eps)
    if not summary:
        return {}

    # Print per-episode grounding details
    sam2_conds = [c for c in summary if c.startswith("sam_")]
    if sam2_conds:
        print(f"\n  Grounding coverage (first SAM2 condition):")
        raw = load_condition_results(results_dir, sam2_conds[0])
        ep_info = {}
        for r in raw:
            ep = r["episode_id"]
            if ep in grounded_eps and ep not in ep_info:
                g = r.get("grounding", {})
                ep_info[ep] = g
        for ep in sorted(ep_info):
            g = ep_info[ep]
            print(f"    ep{ep:03d}: patches={g.get('num_patches', 0)}, "
                  f"coverage={g.get('patch_coverage', 0):.1%}, "
                  f"nouns={g.get('nouns', [])}")

    # Print ranking for filtered subset
    print_ranking(summary)

    # Save filtered summary
    filtered_path = results_dir / "filtered_comparison_summary.json"
    with open(filtered_path, "w") as f:
        json.dump(to_native({
            "filter": "sam2_patches_gt_0",
            "episodes": sorted(grounded_eps),
            "num_episodes": len(grounded_eps),
            "conditions": summary,
        }), f, indent=2)
    print(f"\nFiltered summary: {filtered_path}")

    # Plots for filtered
    plot_ranking_bar(
        summary, "mse_overall",
        f"V3 Filtered ({len(grounded_eps)} eps, SAM2 patches>0): Overall MSE",
        results_dir / "filtered_ranking_overall.png",
    )
    plot_ranking_bar(
        summary, "mse_spatial",
        f"V3 Filtered ({len(grounded_eps)} eps): Spatial MSE",
        results_dir / "filtered_ranking_spatial.png",
    )
    plot_heatmap(summary, results_dir / "filtered_mse_heatmap.png")

    return summary


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


def print_ranking(summary: dict) -> None:
    if not summary:
        return
    bl = summary.get("baseline", {}).get("mse_overall")
    ranked = sorted(summary.items(), key=lambda x: x[1]["mse_overall"])

    print(f"\n{'='*95}")
    print(f"  V3 RESULTS (Research-based Methods) — RANKED BY OVERALL MSE")
    print(f"{'='*95}")
    print(f"  {'#':>3} {'Condition':<25} {'Method':>30} {'Overall':>10} {'Spatial':>10} {'Gripper':>10} {'vs BL':>8}")
    print(f"  {'-'*98}")

    for i, (name, data) in enumerate(ranked):
        mse = data["mse_overall"]
        sp = data["mse_spatial"]
        gr = data["mse_gripper"]
        desc = describe_condition(name) if describe_condition else name
        if bl is not None and name != "baseline":
            delta = 100 * (mse - bl) / bl if bl != 0 else 0
            vs = f"{delta:+.1f}%"
        else:
            vs = "-"
        marker = " *" if name == "baseline" else ""
        print(
            f"  {i+1:>3} {name:<25} {desc:>30} "
            f"{mse:>10.6f} {sp:>10.6f} {gr:>10.6f} {vs:>8}{marker}"
        )

    # Spatial ranking
    ranked_sp = sorted(summary.items(), key=lambda x: x[1]["mse_spatial"])
    bl_sp = summary.get("baseline", {}).get("mse_spatial")
    print(f"\n  {'─'*98}")
    print(f"  RANKED BY SPATIAL MSE (excl. gripper)")
    print(f"  {'─'*98}")
    for i, (name, data) in enumerate(ranked_sp):
        sp = data["mse_spatial"]
        if bl_sp is not None and name != "baseline":
            delta = 100 * (sp - bl_sp) / bl_sp if bl_sp != 0 else 0
            vs = f"{delta:+.1f}%"
        else:
            vs = "-"
        print(f"  {i+1:>3} {name:<25} {sp:>10.6f} {vs:>8}")

    # Per-dim
    print(f"\n  {'─'*98}")
    print(f"  PER-DIMENSION MSE (top 5 + baseline)")
    print(f"  {'─'*98}")
    top5 = [n for n, _ in ranked[:6]]
    if "baseline" not in top5 and "baseline" in summary:
        top5.insert(0, "baseline")
    header = f"  {'Dim':<10}"
    for name in top5:
        header += f" {name[:14]:>14}"
    print(header)
    print(f"  {'-'*(10 + 15*len(top5))}")
    for i, dim_name in enumerate(config.ACTION_DIM_NAMES):
        row = f"  {dim_name:<10}"
        for name in top5:
            val = summary[name].get("per_dim_avg", [0]*7)[i]
            row += f" {val:>14.6f}"
        print(row)

    # Per-episode improvement
    print(f"\n  {'─'*98}")
    print(f"  PER-EPISODE IMPROVEMENT COUNT")
    print(f"  {'─'*98}")
    if bl is not None:
        bl_ep = summary["baseline"]["per_episode_mse"]
        bl_sp_ep = summary["baseline"]["per_episode_spatial"]
        for name, data in ranked:
            if name == "baseline":
                continue
            imp_overall = sum(
                1 for ep, m in data["per_episode_mse"].items()
                if ep in bl_ep and m < bl_ep[ep]
            )
            imp_spatial = sum(
                1 for ep, m in data["per_episode_spatial"].items()
                if ep in bl_sp_ep and m < bl_sp_ep[ep]
            )
            total = len(data["per_episode_mse"])
            print(
                f"  {name:<25} overall: {imp_overall}/{total}  "
                f"spatial: {imp_spatial}/{total}"
            )

    print(f"{'='*95}")


# ── Plots ─────────────────────────────────────────────────────────────

def get_colors(n):
    cmap = plt.colormaps.get_cmap("tab20") if hasattr(plt, "colormaps") else cm.get_cmap("tab20", max(n, 20))
    return [cmap(i / max(n, 1)) for i in range(n)]


def plot_ranking_bar(summary, metric, title, out_path):
    ranked = sorted(summary.items(), key=lambda x: x[1][metric])
    names = [n for n, _ in ranked]
    vals = [d[metric] for _, d in ranked]
    colors = get_colors(len(names))
    bar_colors = ["#333333" if n == "baseline" else colors[i] for i, n in enumerate(names)]

    fig, ax = plt.subplots(figsize=(max(12, len(names) * 0.7), 6))
    bars = ax.barh(range(len(names)), vals, color=bar_colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_width() + max(vals) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.5f}", va="center", fontsize=8,
        )
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.invert_yaxis()
    bl_val = summary.get("baseline", {}).get(metric)
    if bl_val is not None:
        ax.axvline(bl_val, color="red", linestyle="--", alpha=0.7, label=f"baseline={bl_val:.5f}")
        ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_heatmap(summary, out_path):
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
    ax.set_title("V3: MSE Heatmap (Condition x Episode)")
    fig.colorbar(im, ax=ax, label="MSE")
    for i in range(len(conditions)):
        for j in range(len(all_eps)):
            if not np.isnan(matrix[i, j]):
                ax.text(
                    j, i, f"{matrix[i,j]:.4f}", ha="center", va="center",
                    fontsize=6, color="white" if matrix[i, j] > np.nanmedian(matrix) else "black",
                )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_per_dim(summary, out_path):
    ranked = sorted(summary.items(), key=lambda x: x[1]["mse_overall"])
    selected = [n for n, _ in ranked[:6]]
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
    ax.set_title("V3: Per Action-Dimension MSE")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_timestep_top(summary, out_path, top_n=5):
    ranked = sorted(summary.items(), key=lambda x: x[1]["mse_overall"])
    selected = [n for n, _ in ranked[:top_n + 1]]
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
    ax.set_title(f"V3: Per-Timestep MSE — Baseline vs Top-{top_n}")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────

def run_v3_comparison(results_dir=None, top_n=5, filtered=False):
    if results_dir is None:
        results_dir = config.V3_RESULTS_DIR

    print(f"Loading V3 results from {results_dir}/...")
    summary = build_summary(results_dir)

    if not summary:
        print("No results found. Run run_v3_experiment.py first.")
        return {}

    print_ranking(summary)

    summary_path = results_dir / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(to_native(summary), f, indent=2)
    print(f"\nSummary JSON: {summary_path}")

    plot_ranking_bar(summary, "mse_overall", "V3: Overall MSE Ranking", results_dir / "ranking_overall.png")
    plot_ranking_bar(summary, "mse_spatial", "V3: Spatial MSE Ranking", results_dir / "ranking_spatial.png")
    plot_heatmap(summary, results_dir / "mse_heatmap.png")
    plot_per_dim(summary, results_dir / "mse_per_dim.png")
    plot_timestep_top(summary, results_dir / "mse_timestep_top5.png", top_n=top_n)

    if len(summary) > 1:
        bl = summary.get("baseline", {}).get("mse_overall")
        bl_sp = summary.get("baseline", {}).get("mse_spatial")
        non_bl = [(n, d) for n, d in summary.items() if n != "baseline"]
        best_overall = min(non_bl, key=lambda x: x[1]["mse_overall"])
        best_spatial = min(non_bl, key=lambda x: x[1]["mse_spatial"])

        print(f"\n  BEST OVERALL:  {best_overall[0]}  (MSE={best_overall[1]['mse_overall']:.6f})")
        print(f"  BEST SPATIAL:  {best_spatial[0]}  (MSE={best_spatial[1]['mse_spatial']:.6f})")
        if bl is not None:
            delta_o = 100 * (best_overall[1]["mse_overall"] - bl) / bl if bl else 0
            delta_s = 100 * (best_spatial[1]["mse_spatial"] - bl_sp) / bl_sp if bl_sp else 0
            print(f"  vs baseline:   overall {delta_o:+.1f}%,  spatial {delta_s:+.1f}%")

    # Run filtered comparison if requested
    if filtered:
        run_filtered_comparison(results_dir)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Compare V3 enhancement results")
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--filtered", action="store_true",
                        help="Also show SAM2 filtered comparison (patches>0 only)")
    args = parser.parse_args()
    results_dir = Path(args.results_dir) if args.results_dir else None
    run_v3_comparison(results_dir, top_n=args.top, filtered=args.filtered)


if __name__ == "__main__":
    main()
