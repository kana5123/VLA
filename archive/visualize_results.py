"""Visualize attention analysis results.

Usage:
    python visualize_results.py [--episodes 0,1,2] [--steps 0,1,2]

Reads JSON files from outputs/attention_results/ and produces:
  1. Grid images (32 layers x 5 top-k) per action token per step
  2. Individual patch crops (vision) / text files (text tokens)
  3. Summary statistics (summary.json)
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from tqdm import tqdm

import config


# ═══════════════════════════════════════════════════════════════════════════
# Patch cropping
# ═══════════════════════════════════════════════════════════════════════════

def crop_vision_patch(image, patch_row, patch_col, grid_size):
    """Crop a single patch from the image based on grid coordinates.

    Args:
        image: PIL.Image
        patch_row, patch_col: grid coordinates (0-indexed)
        grid_size: size of the grid (e.g., 16 for 16x16)

    Returns:
        PIL.Image of the cropped patch
    """
    w, h = image.size
    patch_w = w / grid_size
    patch_h = h / grid_size

    left = int(patch_col * patch_w)
    top = int(patch_row * patch_h)
    right = int((patch_col + 1) * patch_w)
    bottom = int((patch_row + 1) * patch_h)

    return image.crop((left, top, right, bottom))


def draw_patch_highlight(image, patch_row, patch_col, grid_size, color="red", alpha=0.4):
    """Draw a semi-transparent highlight on a patch location.

    Returns:
        PIL.Image with highlight overlay
    """
    img_array = np.array(image).copy().astype(np.float32)
    h, w = img_array.shape[:2]
    patch_h = h / grid_size
    patch_w = w / grid_size

    top = int(patch_row * patch_h)
    bottom = int((patch_row + 1) * patch_h)
    left = int(patch_col * patch_w)
    right = int((patch_col + 1) * patch_w)

    # Create colored overlay
    color_map = {
        "red": [255, 0, 0],
        "green": [0, 255, 0],
        "blue": [0, 0, 255],
        "yellow": [255, 255, 0],
        "cyan": [0, 255, 255],
    }
    overlay_color = np.array(color_map.get(color, [255, 0, 0]), dtype=np.float32)

    img_array[top:bottom, left:right] = (
        img_array[top:bottom, left:right] * (1 - alpha) + overlay_color * alpha
    )

    # Draw border
    border_width = max(1, int(min(patch_h, patch_w) * 0.1))
    img_array[top:top+border_width, left:right] = overlay_color
    img_array[bottom-border_width:bottom, left:right] = overlay_color
    img_array[top:bottom, left:left+border_width] = overlay_color
    img_array[top:bottom, right-border_width:right] = overlay_color

    return Image.fromarray(img_array.clip(0, 255).astype(np.uint8))


# ═══════════════════════════════════════════════════════════════════════════
# Grid visualization
# ═══════════════════════════════════════════════════════════════════════════

RANK_COLORS = ["red", "green", "blue", "yellow", "cyan"]


def create_action_grid(result_json, image, action_key):
    """Create a 32-layer x 5-rank grid visualization for one action token.

    Each cell shows:
      - Vision token: image with highlighted patch + crop
      - Text token: text display
      - Score overlay

    Args:
        result_json: parsed JSON result for this step
        image: PIL.Image of the original observation
        action_key: e.g., "action_0_x"

    Returns:
        matplotlib figure
    """
    analysis = result_json["attention_analysis"][action_key]
    num_layers = len(analysis)
    k = config.TOP_K

    fig, axes = plt.subplots(
        num_layers, k,
        figsize=(k * 2.5, num_layers * 2),
        squeeze=False,
    )
    fig.suptitle(
        f"{action_key} | ep{result_json['episode_id']:03d} step{result_json['step_id']:03d}\n"
        f"instruction: \"{result_json['instruction'][:60]}\"",
        fontsize=10, y=1.02,
    )

    for layer_idx in range(num_layers):
        layer_key = f"layer_{layer_idx:02d}"
        if layer_key not in analysis:
            continue

        top5 = analysis[layer_key]["top5"]

        for rank_idx in range(min(k, len(top5))):
            ax = axes[layer_idx][rank_idx]
            token = top5[rank_idx]

            if token["type"] == "vision":
                grid_size = token.get("grid_size", config.VISION_GRID_SIZE)
                highlighted = draw_patch_highlight(
                    image, token["patch_row"], token["patch_col"],
                    grid_size, color=RANK_COLORS[rank_idx % len(RANK_COLORS)]
                )
                ax.imshow(highlighted)
                encoder_label = f" ({token['encoder']})" if "encoder" in token else ""
                ax.set_title(
                    f"V[{token['patch_row']},{token['patch_col']}]{encoder_label}\n"
                    f"score={token['score']:.4f}",
                    fontsize=5, pad=1,
                )
            elif token["type"] == "text":
                # Show text token as centered text on white background
                ax.set_facecolor("lightyellow")
                ax.text(
                    0.5, 0.5, f"\"{token['text']}\"",
                    ha="center", va="center",
                    fontsize=8, fontweight="bold",
                    transform=ax.transAxes,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                )
                ax.set_title(
                    f"T[{token['text_idx']}] \"{token['text'][:8]}\"\n"
                    f"score={token['score']:.4f}",
                    fontsize=5, pad=1,
                )
            else:
                # Special token
                ax.set_facecolor("lightgray")
                ax.text(
                    0.5, 0.5, f"{token['label']}\npos={token['position']}",
                    ha="center", va="center",
                    fontsize=7,
                    transform=ax.transAxes,
                )
                ax.set_title(
                    f"S[{token['position']}]\nscore={token['score']:.4f}",
                    fontsize=5, pad=1,
                )

            ax.set_xticks([])
            ax.set_yticks([])

            # Row label (layer) on leftmost column
            if rank_idx == 0:
                ax.set_ylabel(f"L{layer_idx}", fontsize=6, rotation=0, labelpad=15)

    # Column headers
    for rank_idx in range(k):
        axes[0][rank_idx].set_xlabel(f"Rank {rank_idx}", fontsize=6)
        axes[0][rank_idx].xaxis.set_label_position("top")

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Individual patches
# ═══════════════════════════════════════════════════════════════════════════

def save_individual_patches(result_json, image, output_dir):
    """Save individual patch crops and text token files.

    Vision tokens → PNG files
    Text tokens → TXT files with token info
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for action_key, layers in result_json["attention_analysis"].items():
        for layer_key, layer_data in layers.items():
            for token in layer_data["top5"]:
                rank = token["rank"]
                prefix = f"{action_key}_{layer_key}_rank{rank}"

                if token["type"] == "vision":
                    grid_size = token.get("grid_size", config.VISION_GRID_SIZE)
                    patch = crop_vision_patch(
                        image, token["patch_row"], token["patch_col"], grid_size
                    )
                    encoder_suffix = f"_{token['encoder']}" if "encoder" in token else ""
                    filename = f"{prefix}_vis_r{token['patch_row']:02d}c{token['patch_col']:02d}{encoder_suffix}.png"
                    patch.save(output_dir / filename)

                elif token["type"] == "text":
                    filename = f"{prefix}_text_tok{token['text_idx']:02d}.txt"
                    with open(output_dir / filename, "w") as f:
                        f.write(f"Token index: {token['text_idx']}\n")
                        f.write(f"Token text: {token['text']}\n")
                        f.write(f"Global position: {token['token_idx']}\n")
                        f.write(f"Attention score: {token['score']:.6f}\n")


# ═══════════════════════════════════════════════════════════════════════════
# Summary statistics
# ═══════════════════════════════════════════════════════════════════════════

def compute_summary(all_results):
    """Compute summary statistics across all results.

    Returns:
        dict with overall statistics
    """
    # Track: per layer, how often each type appears in top-5
    layer_type_counts = defaultdict(lambda: defaultdict(int))
    # Per action dimension
    action_type_counts = defaultdict(lambda: defaultdict(int))
    # Most common vision positions
    vision_position_counts = defaultdict(int)
    # Most common text tokens
    text_token_counts = defaultdict(int)
    # Total counts
    total_entries = 0

    for result in all_results:
        for action_key, layers in result["attention_analysis"].items():
            action_dim = action_key.split("_")[1] + "_" + action_key.split("_")[2]
            for layer_key, layer_data in layers.items():
                for token in layer_data["top5"]:
                    total_entries += 1
                    layer_type_counts[layer_key][token["type"]] += 1
                    action_type_counts[action_dim][token["type"]] += 1

                    if token["type"] == "vision":
                        pos_key = f"r{token['patch_row']:02d}_c{token['patch_col']:02d}"
                        vision_position_counts[pos_key] += 1
                    elif token["type"] == "text":
                        text_token_counts[token.get("text", "unknown")] += 1

    # Convert to serializable format
    summary = {
        "total_top_k_entries": total_entries,
        "layer_type_distribution": {},
        "action_dim_type_distribution": {},
        "top_20_vision_positions": [],
        "top_20_text_tokens": [],
    }

    # Layer-wise type distribution
    for layer_key in sorted(layer_type_counts.keys()):
        counts = layer_type_counts[layer_key]
        total = sum(counts.values())
        summary["layer_type_distribution"][layer_key] = {
            t: {"count": c, "ratio": round(c / total, 4)} for t, c in counts.items()
        }

    # Action-dim type distribution
    for action_key in sorted(action_type_counts.keys()):
        counts = action_type_counts[action_key]
        total = sum(counts.values())
        summary["action_dim_type_distribution"][action_key] = {
            t: {"count": c, "ratio": round(c / total, 4)} for t, c in counts.items()
        }

    # Top vision positions
    sorted_vpos = sorted(vision_position_counts.items(), key=lambda x: -x[1])[:20]
    summary["top_20_vision_positions"] = [
        {"position": pos, "count": cnt} for pos, cnt in sorted_vpos
    ]

    # Top text tokens
    sorted_text = sorted(text_token_counts.items(), key=lambda x: -x[1])[:20]
    summary["top_20_text_tokens"] = [
        {"token": tok, "count": cnt} for tok, cnt in sorted_text
    ]

    return summary


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def run_visualization(episode_ids=None, step_ids=None):
    """Generate visualizations from attention result JSONs."""

    # ── Find all result files ─────────────────────────────────────────────
    result_files = sorted(config.ATTENTION_RESULTS_DIR.glob("ep*_step*.json"))

    if not result_files:
        print(f"ERROR: No result files found in {config.ATTENTION_RESULTS_DIR}/")
        print("Run extract_attention.py first.")
        return

    # ── Filter by episode/step ────────────────────────────────────────────
    if episode_ids is not None:
        result_files = [
            f for f in result_files
            if int(f.stem.split("_")[0].replace("ep", "")) in episode_ids
        ]
    if step_ids is not None:
        result_files = [
            f for f in result_files
            if int(f.stem.split("_")[1].replace("step", "")) in step_ids
        ]

    print(f"Processing {len(result_files)} result files...")

    all_results = []

    for result_file in tqdm(result_files, desc="Visualizing"):
        with open(result_file) as f:
            result = json.load(f)
        all_results.append(result)

        ep_id = result["episode_id"]
        step_id = result["step_id"]
        step_name = f"ep{ep_id:03d}_step{step_id:03d}"

        # ── Load original image ───────────────────────────────────────────
        image_path = config.DATA_DIR / f"episode_{ep_id:03d}" / f"step_{step_id:03d}.png"
        if not image_path.exists():
            print(f"  WARNING: Image not found: {image_path}")
            continue
        image = Image.open(image_path).convert("RGB")

        # ── Create grid visualizations ────────────────────────────────────
        vis_dir = config.VISUALIZATIONS_DIR / step_name
        vis_dir.mkdir(parents=True, exist_ok=True)

        for action_key in result["attention_analysis"]:
            fig = create_action_grid(result, image, action_key)
            fig_path = vis_dir / f"{action_key}_top5_grid.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        # ── Save individual patches ───────────────────────────────────────
        patches_dir = config.PATCHES_DIR / step_name
        save_individual_patches(result, image, patches_dir)

    # ── Compute and save summary ──────────────────────────────────────────
    if all_results:
        summary = compute_summary(all_results)
        summary_path = config.OUTPUT_DIR / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSummary saved to {summary_path}")

    print(f"Visualizations: {config.VISUALIZATIONS_DIR}/")
    print(f"Patches: {config.PATCHES_DIR}/")
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Visualize attention analysis results")
    parser.add_argument(
        "--episodes",
        type=str,
        default=None,
        help="Comma-separated episode IDs (default: all)",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated step IDs (default: all)",
    )
    args = parser.parse_args()

    episode_ids = [int(x) for x in args.episodes.split(",")] if args.episodes else None
    step_ids = [int(x) for x in args.steps.split(",")] if args.steps else None

    run_visualization(episode_ids=episode_ids, step_ids=step_ids)


if __name__ == "__main__":
    main()
