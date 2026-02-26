# contribution/visualize.py
"""
All 5 figure types from Design Section 9.

1. Layer-wise Top1 contrib share curve (per model)
2. Candidate token frequency heatmap (16×16 or flattened)
3. Attention vs Contribution mismatch scatter
4. Masking ablation curve (top-k knockout → KL change)
5. Skill signature clustering (JS distance matrix)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def plot_top1_share_curve(
    layer_data: dict[str, list[float]],
    output_path: Path,
    title: str = "Layer-wise Top1 Contribution Share",
):
    """Figure 1: Bottleneck onset visualization."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, shares in layer_data.items():
        ax.plot(range(len(shares)), shares, label=name, linewidth=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Top1 C̃(j) Share")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_frequency_heatmap(
    freq: np.ndarray,
    grid_size: int,
    output_path: Path,
    title: str = "Candidate Token Frequency",
):
    """Figure 2: Which positions are systematically bottleneck candidates."""
    n_vision = grid_size * grid_size
    vision_freq = freq[:n_vision]
    grid = vision_freq.reshape(grid_size, grid_size)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(grid, cmap="hot", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Patch Column")
    ax.set_ylabel("Patch Row")
    fig.colorbar(im, ax=ax, label="Freq(j)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_attention_contribution_scatter(
    a_shares: np.ndarray,
    c_shares: np.ndarray,
    labels: list[str],
    output_path: Path,
    title: str = "Attention vs Contribution (Sink/Bottleneck)",
):
    """Figure 3: VAR-style scatter — sinks in top-left, bottlenecks in top-right."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"sink": "red", "bottleneck": "blue", "normal": "gray"}

    for cls in ["normal", "sink", "bottleneck"]:
        mask = np.array([l == cls for l in labels])
        if mask.any():
            ax.scatter(a_shares[mask], c_shares[mask],
                      c=colors[cls], label=cls, alpha=0.6, s=40)

    ax.set_xlabel("Ã(j) — Attention Share")
    ax.set_ylabel("C̃(j) — Contribution Share")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_masking_ablation(
    k_values: list[int],
    kl_values: dict[str, list[float]],
    output_path: Path,
    title: str = "Masking Ablation: Top-K Knockout → Output KL",
):
    """Figure 4: How much output changes when masking top-k candidates."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, kls in kl_values.items():
        ax.plot(k_values[:len(kls)], kls, marker="o", label=method, linewidth=2)
    ax.set_xlabel("Number of Masked Tokens (K)")
    ax.set_ylabel("KL Divergence (orig → masked)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_skill_js_matrix(
    js_matrix: np.ndarray,
    skill_names: list[str],
    output_path: Path,
    title: str = "Skill Signature JS Distance Matrix",
):
    """Figure 5: Skill clustering via JS divergence."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(js_matrix, cmap="viridis", interpolation="nearest")
    ax.set_xticks(range(len(skill_names)))
    ax.set_yticks(range(len(skill_names)))
    ax.set_xticklabels(skill_names, rotation=45, ha="right")
    ax.set_yticklabels(skill_names)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="JS Divergence²")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")
