"""Visualize text-token → visual-token attention maps across all layers.

Inspired by VAR (ICLR 2025) Figure 1: shows where specific text tokens
(task verbs like "pick", object nouns like "cup") attend in visual space.

Usage:
    python visualize_text_attention.py --model openvla-7b --device cuda:5

For each sample, produces a figure with:
  - Rows: layers (0..N)
  - Columns: selected text tokens (verbs + nouns)
  - Each cell: original image with attention heatmap overlay (top-10 patches highlighted)
"""

import argparse
import pickle
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from PIL import Image

import config
from extract_attention import (
    load_model_from_registry,
    detect_token_boundaries,
    get_layers,
    AttentionHookManager,
)
from model_registry import get_model as registry_get_model


# ═══════════════════════════════════════════════════════════════════
# Token position finding
# ═══════════════════════════════════════════════════════════════════

def find_keyword_token_positions(tokenizer, prompt_template, instruction, keywords, text_start):
    """Find token positions for given keywords in the full sequence.

    Tokenizes the prompt text separately, then maps keyword positions
    back to the full sequence (vision + text) positions.

    Args:
        tokenizer: HuggingFace tokenizer
        prompt_template: model's prompt template string
        instruction: raw instruction string
        keywords: list of keyword strings to find (e.g., ["pick", "cup"])
        text_start: index where text tokens begin (after vision tokens)

    Returns:
        dict mapping keyword → list of (position_in_sequence, token_string) tuples
    """
    # Tokenize the full prompt (text only, without image)
    prompt = prompt_template.format(instruction=instruction)
    text_ids = tokenizer.encode(prompt, add_special_tokens=True)

    # Decode each text token for debugging
    text_tokens = []
    for tid in text_ids:
        decoded = tokenizer.decode([tid]).strip().lower()
        text_tokens.append(decoded)

    result = {}
    for kw in keywords:
        kw_lower = kw.lower()

        # Strategy 1: Encode the keyword and find exact token ID match
        kw_ids = tokenizer.encode(kw, add_special_tokens=False)

        if len(kw_ids) == 1:
            # Single-token keyword: find in text_ids
            for i, tid in enumerate(text_ids):
                if tid == kw_ids[0]:
                    seq_pos = text_start + i
                    result[kw] = [(seq_pos, tokenizer.decode([tid]).strip())]
                    break
        else:
            # Multi-token keyword (e.g., "oven" → ["o", "ven"])
            # Find the subsequence in text_ids
            for i in range(len(text_ids) - len(kw_ids) + 1):
                if text_ids[i:i + len(kw_ids)] == kw_ids:
                    # Use the first subword token position
                    seq_pos = text_start + i
                    tok_str = tokenizer.decode(kw_ids).strip()
                    result[kw] = [(seq_pos, tok_str)]
                    break

        # Strategy 2: Fallback — fuzzy match on decoded tokens
        if kw not in result:
            for i, tok_str in enumerate(text_tokens):
                if kw_lower == tok_str or (len(kw_lower) > 2 and kw_lower in tok_str):
                    seq_pos = text_start + i
                    result[kw] = [(seq_pos, tok_str)]
                    break

    return result


def extract_keywords_from_instruction(instruction):
    """Extract task verbs and object nouns from a robot instruction.

    Simple heuristic: verbs are action words, nouns are objects.
    """
    # Common robot task verbs
    verbs = {"pick", "place", "put", "move", "push", "pull", "grasp",
             "lift", "open", "close", "turn", "slide", "wipe", "sweep",
             "stack", "pour", "rotate", "flip", "press", "reach", "grab"}
    # Stopwords to exclude
    stopwords = {"the", "a", "an", "to", "in", "on", "up", "down",
                 "into", "onto", "from", "with", "and", "of", "at",
                 "should", "what", "action", "robot", "take", "that",
                 "this", "it", "is", "be", "do", "can", "will"}

    words = re.findall(r'\b[a-zA-Z]+\b', instruction.lower())
    found_verbs = []
    found_nouns = []

    for w in words:
        if w in verbs:
            if w not in found_verbs:
                found_verbs.append(w)
        elif w not in stopwords and len(w) > 2:
            if w not in found_nouns:
                found_nouns.append(w)

    return found_verbs, found_nouns


# ═══════════════════════════════════════════════════════════════════
# Attention extraction
# ═══════════════════════════════════════════════════════════════════

def extract_all_layer_attentions(model, processor, model_cfg, image, instruction, device):
    """Run a single forward pass and capture attention from all layers.

    Returns:
        attentions: dict[layer_idx] → tensor (num_heads, seq_len, seq_len)
        boundaries: dict with vision_start, vision_end, text_start, etc.
        input_ids: 1D tensor of input token IDs
    """
    prompt = model_cfg.prompt_template.format(instruction=instruction)
    inputs = processor(prompt, image, return_tensors="pt").to(device)
    if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

    input_ids = inputs["input_ids"][0]

    # Detect token boundaries
    boundaries = detect_token_boundaries(processor, model, image, instruction, device)

    # Register hooks on all attention layers
    hook_mgr = AttentionHookManager(model)
    hook_mgr.register_hooks()

    # Forward pass
    with torch.no_grad():
        # Architecture-specific extras
        extra_kv = {}
        if model_cfg.architecture == "gemma2":
            extra_kv["intrinsic"] = torch.tensor(
                [[[218.26, 0.0, 111.83],
                  [0.0, 218.26, 111.79],
                  [0.0, 0.0, 1.0]]], device=device, dtype=torch.float32,
            )

        model(**{k: v for k, v in inputs.items()}, use_cache=False, **extra_kv)

    # Collect attentions from step 0 (the prompt pass)
    attentions = hook_mgr.get_attentions_for_step(0)
    hook_mgr.remove_hooks()

    return attentions, boundaries, input_ids


# ═══════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════

def create_attention_heatmap(
    image: Image.Image,
    attention_scores: np.ndarray,
    grid_h: int,
    grid_w: int,
    top_k: int = 10,
    sink_indices: list = None,
):
    """Create VAR Figure 1-style attention heatmap overlay on image.

    Two visual categories:
      - Attention sinks (irrelevant patches): RED box outline only, no fill
      - Relevant top-K patches: PURPLE fill with rank-based saturation

    Args:
        image: PIL Image (original)
        attention_scores: (num_vision_tokens,) attention weights
        grid_h, grid_w: vision token grid dimensions
        top_k: number of top patches to highlight
        sink_indices: list of known attention sink patch indices (e.g., [0])

    Returns:
        numpy array of the composited image
    """
    if sink_indices is None:
        sink_indices = [0]  # Vision token 0 is the confirmed sink for OpenVLA

    # Resize image to patch-aligned size
    img_w = grid_w * 32
    img_h = grid_h * 32
    img_array = np.array(image.resize((img_w, img_h))).astype(np.float32)
    patch_h = img_h // grid_h
    patch_w = img_w // grid_w

    # Normalize scores to [0, 1]
    scores = attention_scores.copy()
    if scores.max() > 0:
        scores = scores / scores.max()

    # Handle size mismatch
    n_patches = grid_h * grid_w
    if len(scores) > n_patches:
        scores = scores[:n_patches]
    elif len(scores) < n_patches:
        padded = np.zeros(n_patches)
        padded[:len(scores)] = scores
        scores = padded

    # Detect attention sinks dynamically: patches with score > alpha/N
    sink_alpha = 5.0
    uniform_level = 1.0 / n_patches
    dynamic_sinks = set(np.where(scores > sink_alpha * uniform_level)[0].tolist())
    # Merge with known sinks
    all_sinks = dynamic_sinks | set(sink_indices)

    # Find top-K patches (including sinks — we'll separate them visually)
    top_indices = np.argsort(scores)[::-1][:top_k]

    # Separate into relevant patches and sink patches
    relevant_indices = [idx for idx in top_indices if idx not in all_sinks]
    sink_top_indices = [idx for idx in top_indices if idx in all_sinks]

    # ── Purple fill for relevant patches (rank-based saturation) ──
    overlay = img_array.copy()

    for rank, idx in enumerate(relevant_indices):
        row = idx // grid_w
        col = idx % grid_w
        y1 = row * patch_h
        y2 = (row + 1) * patch_h
        x1 = col * patch_w
        x2 = (col + 1) * patch_w

        # Saturation decreases with rank: rank 0 = deepest purple, last = lightest
        n_relevant = max(len(relevant_indices) - 1, 1)
        t = rank / n_relevant  # 0.0 (top) to 1.0 (bottom)

        # Deep purple (80, 0, 160) → Light purple (180, 140, 220)
        deep = np.array([80.0, 0.0, 160.0])
        light = np.array([180.0, 140.0, 220.0])
        purple = deep * (1 - t) + light * t

        # Alpha also varies: 0.85 (rank 0) → 0.45 (last rank)
        alpha = 0.85 - 0.40 * t

        overlay[y1:y2, x1:x2] = (
            (1 - alpha) * overlay[y1:y2, x1:x2] + alpha * purple
        )

    result = overlay.astype(np.uint8)

    # ── Red box outlines for attention sinks (no fill) ──
    border = 2
    for idx in sink_top_indices:
        row = idx // grid_w
        col = idx % grid_w
        y1 = row * patch_h
        y2 = (row + 1) * patch_h - 1
        x1 = col * patch_w
        x2 = (col + 1) * patch_w - 1

        # Red border only (no purple fill inside)
        result[y1:y1+border, x1:x2, :] = [255, 0, 0]
        result[y2-border:y2, x1:x2, :] = [255, 0, 0]
        result[y1:y2, x1:x1+border, :] = [255, 0, 0]
        result[y1:y2, x2-border:x2, :] = [255, 0, 0]

    return result


def visualize_text_token_attention(
    model, processor, model_cfg, tokenizer_or_processor,
    image, instruction, device,
    output_path,
    top_k=10,
    layer_step=1,
):
    """Main visualization function.

    Creates a figure showing attention from task-verb and object-noun tokens
    to visual tokens, across all layers.
    """
    print(f"\n{'='*60}")
    print(f"Instruction: {instruction}")
    print(f"{'='*60}")

    # 1. Extract attention from all layers
    attentions, boundaries, input_ids = extract_all_layer_attentions(
        model, processor, model_cfg, image, instruction, device
    )

    if attentions is None:
        print("ERROR: No attention weights captured!")
        return

    vision_start = boundaries["vision_start"]
    vision_end = boundaries["vision_end"]
    text_start = boundaries["text_start"]
    num_vision = boundaries["num_vision_tokens"]
    num_layers = model_cfg.num_layers
    grid_size = model_cfg.vision_grid_size

    print(f"Vision tokens: {num_vision} ({grid_size}x{grid_size})")
    print(f"Text starts at: {text_start}")

    # 2. Find keyword positions
    found_verbs, found_nouns = extract_keywords_from_instruction(instruction)
    all_keywords = found_verbs + found_nouns
    print(f"Keywords - Verbs: {found_verbs}, Nouns: {found_nouns}")

    # Get tokenizer for position finding
    if hasattr(tokenizer_or_processor, "tokenizer"):
        tok = tokenizer_or_processor.tokenizer
    elif hasattr(tokenizer_or_processor, "decode"):
        tok = tokenizer_or_processor
    else:
        tok = tokenizer_or_processor

    kw_positions = find_keyword_token_positions(
        tok, model_cfg.prompt_template, instruction, all_keywords, text_start
    )
    print(f"Found token positions: {kw_positions}")

    if not kw_positions:
        print("WARNING: No keyword tokens found! Trying all content words...")
        words = re.findall(r'\b[a-zA-Z]+\b', instruction.lower())
        stopwords = {"the", "a", "an", "to", "in", "on", "up", "down", "into",
                     "onto", "from", "with", "and", "of", "at", "should", "what",
                     "action", "robot", "take", "that"}
        content_words = [w for w in words if w not in stopwords and len(w) > 2][:3]
        kw_positions = find_keyword_token_positions(
            tok, model_cfg.prompt_template, instruction, content_words, text_start
        )
        if kw_positions:
            found_verbs = []
            found_nouns = list(kw_positions.keys())
            all_keywords = found_nouns

    if not kw_positions:
        print("ERROR: Could not find any keyword token positions!")
        return

    # Select layers to visualize (all, but can step)
    layer_indices = list(range(0, num_layers, layer_step))

    # 3. Build the figure
    n_tokens = len(kw_positions)
    n_layers = len(layer_indices)

    fig, axes = plt.subplots(
        n_layers, n_tokens + 1,
        figsize=(4 * (n_tokens + 1), 3 * n_layers),
        squeeze=False,
    )

    # Column 0: layer labels with original image
    for row_idx, layer_idx in enumerate(layer_indices):
        ax = axes[row_idx, 0]
        ax.imshow(np.array(image.resize((grid_size * 32, grid_size * 32))))
        ax.set_ylabel(f"Layer {layer_idx}", fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        if row_idx == 0:
            ax.set_title("Original", fontsize=11, fontweight='bold')

    # Columns 1+: attention heatmaps per keyword
    keyword_list = list(kw_positions.keys())
    for col_idx, kw in enumerate(keyword_list):
        positions = kw_positions[kw]
        # Use first matching position
        seq_pos, tok_str = positions[0]

        is_verb = kw in found_verbs
        label_prefix = "[V]" if is_verb else "[N]"

        for row_idx, layer_idx in enumerate(layer_indices):
            ax = axes[row_idx, col_idx + 1]

            if layer_idx not in attentions:
                ax.text(0.5, 0.5, "N/A", ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # attn shape: (1, num_heads, seq_len, seq_len)
            attn = attentions[layer_idx]
            if attn.dim() == 4:
                attn = attn[0]  # (num_heads, seq_len, seq_len)

            # Check seq_pos is valid
            if seq_pos >= attn.shape[1]:
                ax.text(0.5, 0.5, f"pos {seq_pos} > seq_len {attn.shape[1]}",
                        ha='center', va='center', fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # Extract attention from this text token to vision tokens
            # attn shape: (H, Q, K) — we want attn[:, seq_pos, vision_start:vision_end]
            text_to_vision = attn[:, seq_pos, vision_start:vision_end]  # (H, V)
            # Head-average
            avg_attn = text_to_vision.float().mean(dim=0).numpy()  # (V,)

            # Create heatmap with sink separation
            heatmap = create_attention_heatmap(
                image, avg_attn, grid_size, grid_size,
                top_k=top_k, sink_indices=[0],
            )
            ax.imshow(heatmap)
            ax.set_xticks([])
            ax.set_yticks([])

            # Add attention stats as text
            top3_val = np.sort(avg_attn)[::-1][:3]
            ax.text(0.02, 0.98, f"max:{top3_val[0]:.3f}",
                    transform=ax.transAxes, fontsize=6,
                    va='top', ha='left', color='white',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))

            if row_idx == 0:
                ax.set_title(f'{label_prefix} "{kw}" (pos={seq_pos})',
                             fontsize=11, fontweight='bold',
                             color='red' if is_verb else 'blue')

    # Add legend explaining color coding
    legend_text = (
        r"$\bf{Purple\ fill}$: relevant attention (darker = higher rank)   "
        r"$\bf{Red\ box}$: attention sink (irrelevant)"
    )
    plt.suptitle(
        f'{model_cfg.name}: Text-Token → Visual Attention\n"{instruction}"',
        fontsize=14, fontweight='bold', y=1.02,
    )
    fig.text(0.5, 1.005, legend_text, fontsize=10, ha='center',
             color='#333333', style='italic')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print(f"Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════
# Data loading helpers
# ═══════════════════════════════════════════════════════════════════

def load_samples_from_cache(cache_dir, n_samples=3):
    """Load a few diverse samples from the cached bridge dataset (memmap).

    The cache is a numpy memmap of shape (N, 256, 256, 3) uint8 + metadata pickle.
    """
    import json
    cache_dir = Path(cache_dir)

    with open(cache_dir / "cache_info.json") as f:
        info = json.load(f)
    total_steps = info["total_steps"]
    img_h = info["image_height"]
    img_w = info["image_width"]

    with open(cache_dir / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    # Open memmap (lazy read, no RAM usage)
    images_mmap = np.memmap(
        str(cache_dir / "images.dat"), dtype=np.uint8, mode="r",
        shape=(total_steps, img_h, img_w, 3),
    )

    # Select diverse samples by unique verbs
    seen_verbs = set()
    selected = []

    for entry in metadata:
        instr = entry.get("instruction", "")
        verbs, nouns = extract_keywords_from_instruction(instr)
        verb_key = tuple(sorted(verbs))
        if verb_key not in seen_verbs and verbs and nouns:
            seen_verbs.add(verb_key)
            selected.append(entry)
            if len(selected) >= n_samples:
                break

    if len(selected) < n_samples:
        selected = metadata[:n_samples]

    # Load images from memmap
    samples = []
    for entry in selected:
        global_idx = entry["global_idx"]
        img_array = np.array(images_mmap[global_idx])  # copy from memmap
        image = Image.fromarray(img_array)
        samples.append({
            "instruction": entry.get("instruction", ""),
            "image": image,
        })
        print(f"  Sample: [{entry.get('instruction', '')}] (idx={global_idx})")

    del images_mmap  # close memmap
    return samples


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Visualize text-token attention maps")
    parser.add_argument("--model", default="openvla-7b", help="Model name from registry")
    parser.add_argument("--device", default="cuda", help="Device (e.g., cuda:5)")
    parser.add_argument("--n_samples", type=int, default=3, help="Number of samples")
    parser.add_argument("--top_k", type=int, default=10, help="Top-K patches to highlight")
    parser.add_argument("--layer_step", type=int, default=1, help="Layer step (1=all, 2=every other)")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    args = parser.parse_args()

    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = config.OUTPUT_DIR / "text_attention_viz" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    processor, model, model_cfg = load_model_from_registry(args.model, device=args.device)

    # Load samples from memmap cache
    print("Loading samples from cache...")
    samples = load_samples_from_cache(
        config.DATA_CACHE_DIR, n_samples=args.n_samples
    )

    print(f"Loaded {len(samples)} samples")

    # Run visualization for each sample
    for i, sample in enumerate(samples):
        if sample["image"] is None:
            print(f"Skipping sample {i}: no image")
            continue

        output_path = out_dir / f"sample_{i:02d}_attention.png"
        visualize_text_token_attention(
            model, processor, model_cfg, processor,
            sample["image"], sample["instruction"],
            args.device, output_path,
            top_k=args.top_k,
            layer_step=args.layer_step,
        )

    # Also create a summary: layer-averaged attention for each keyword
    print(f"\nAll visualizations saved to: {out_dir}")


if __name__ == "__main__":
    main()
