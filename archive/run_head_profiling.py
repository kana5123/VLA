"""Head profiling via gradient-based attribution for dimension-aware intervention.

For each calibration step (ep0-4, ~200 steps), for each of 7 action tokens:
  1. Run forward pass WITH gradients (no torch.no_grad)
  2. Hook into o_proj input to capture per-head attention outputs
  3. Backprop from predicted action token's logit
  4. Compute per-head importance as |gradient × activation|

Output:
  - importance_mean.npy:       (7, 32, 32) — [action_dim, layer, head]
  - importance_std.npy:        (7, 32, 32) — standard deviation
  - importance_normalized.npy: (7, 32, 32) — per-layer normalized
  - head_classification.json:  per-head type + dimension scores
  - profiling_summary.json:    statistics + top heads per dimension

Research basis:
  - Attribution Patching (Neel Nanda et al., 2023)
  - "Are Sixteen Heads Really Better than One?" (Michel et al., NeurIPS 2019)
  - Causal Head Gating (arXiv 2505.13737)

Usage:
    python run_head_profiling.py --episodes 0,1,2,3,4 --device cuda
"""

import argparse
import json
import sys
import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import config
from extract_attention import load_model, detect_token_boundaries


# ======================================================================
# Head Importance Profiler
# ======================================================================

class HeadImportanceProfiler:
    """Capture per-head importance via gradient × activation at o_proj input.

    The o_proj linear layer in LlamaAttention combines all heads:
        o_proj(concat(head_0, ..., head_31))

    By hooking into o_proj's input, we capture each head's contribution
    and measure its effect on the predicted action token via:
        importance[l, h] = |activation[l, h] · ∇activation[l, h]|

    This is the first-order Taylor approximation of each head's
    contribution (Michel et al., NeurIPS 2019).
    """

    def __init__(self, model):
        if hasattr(model, "language_model"):
            self.layers = model.language_model.model.layers
            lm_config = model.language_model.config
        else:
            self.layers = model.model.layers
            lm_config = model.config

        self.num_layers = len(self.layers)
        self.num_heads = lm_config.num_attention_heads
        self.hidden_size = lm_config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads

        self.hooks = []
        self.activations = {}  # layer_idx -> tensor (with retain_grad)

    def register_hooks(self):
        """Register forward pre-hooks on all o_proj modules."""
        for l_idx in range(self.num_layers):
            o_proj = self.layers[l_idx].self_attn.o_proj
            hook = o_proj.register_forward_pre_hook(self._make_hook(l_idx))
            self.hooks.append(hook)

    def _make_hook(self, layer_idx):
        def hook_fn(module, args):
            # args[0] shape: (B, seq_len, num_heads * head_dim)
            x = args[0]
            x.retain_grad()
            self.activations[layer_idx] = x
        return hook_fn

    def compute_importance(self):
        """After backward(), compute |grad × activation| per head.

        Returns:
            np.ndarray of shape (num_layers, num_heads) — importance scores
        """
        importance = np.zeros((self.num_layers, self.num_heads))

        for l_idx in range(self.num_layers):
            x = self.activations.get(l_idx)
            if x is None or x.grad is None:
                continue

            # Take last sequence position (the generated action token)
            # x shape: (B, seq, H*D) → (B, H, D) for last position
            x_last = x[:, -1, :].float().reshape(-1, self.num_heads, self.head_dim)
            g_last = x.grad[:, -1, :].float().reshape(-1, self.num_heads, self.head_dim)

            # Attribution: |activation × gradient| summed over head_dim
            imp = (x_last * g_last).abs().sum(dim=-1).mean(dim=0)  # (H,)
            importance[l_idx] = imp.detach().cpu().numpy()

        return importance

    def clear(self):
        """Clear stored activations (frees memory)."""
        self.activations = {}

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.activations = {}


# ======================================================================
# Main profiling
# ======================================================================

def run_profiling(episode_ids=None, device="cuda", save_raw=False):
    """Profile all 1024 attention heads on calibration episodes.

    For each valid step and each of 7 action tokens:
    - Forward pass (with gradients)
    - Backward from predicted token logit
    - Collect per-head importance via gradient × activation

    Args:
        episode_ids: list of calibration episode IDs (default: [0,1,2,3,4])
        device: CUDA device
        save_raw: if True, save per-step raw importance (large: ~115MB)
    """
    if episode_ids is None:
        episode_ids = [0, 1, 2, 3, 4]

    if not config.METADATA_PATH.exists():
        print(f"ERROR: Metadata not found at {config.METADATA_PATH}")
        sys.exit(1)

    with open(config.METADATA_PATH) as f:
        metadata = json.load(f)

    episodes = [ep for ep in metadata["episodes"] if ep["episode_id"] in episode_ids]
    if not episodes:
        print("No matching episodes found.")
        return

    # Collect valid steps (skip is_first/is_last — dummy zero actions)
    valid_steps = []
    for ep in episodes:
        for step in ep["steps"]:
            if not step.get("is_first", False) and not step.get("is_last", False):
                valid_steps.append((ep["episode_id"], step))

    total_steps = len(valid_steps)
    total_passes = total_steps * config.NUM_ACTION_TOKENS

    print(f"\n{'='*70}")
    print(f"  Head Profiling — Gradient-Based Attribution")
    print(f"  Calibration episodes: {episode_ids}")
    print(f"  Valid steps: {total_steps}")
    print(f"  Total forward+backward passes: {total_passes}")
    print(f"{'='*70}\n")

    # Load model (NO attention patch — profile baseline behavior)
    print("Loading model...")
    processor, model = load_model(device)

    # Architecture info from actual model config
    profiler = HeadImportanceProfiler(model)
    num_layers = profiler.num_layers
    num_heads = profiler.num_heads
    head_dim = profiler.head_dim

    print(f"  Architecture: {num_layers}L × {num_heads}H × {head_dim}D")
    print(f"  Total heads: {num_layers * num_heads}")

    # Detect token boundaries
    first_step = valid_steps[0][1]
    sample_image = Image.open(
        config.PROJECT_ROOT / first_step["image_path"]
    ).convert("RGB")
    boundaries = detect_token_boundaries(
        processor, model, sample_image, first_step["instruction"], device
    )
    vision_end = boundaries["vision_end"]
    print(f"  Vision tokens: {boundaries['num_vision_tokens']}, vision_end={vision_end}")

    # Register hooks
    profiler.register_hooks()
    print(f"  Hooks registered on {num_layers} o_proj modules")

    # Storage: (N_steps, 7, num_layers, num_heads)
    all_importance = np.zeros(
        (total_steps, config.NUM_ACTION_TOKENS, num_layers, num_heads),
        dtype=np.float32,
    )

    config.HEAD_PROFILING_DIR.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(total=total_passes, desc="Profiling heads")
    t_start = time.time()

    for step_idx, (ep_id, step) in enumerate(valid_steps):
        step_id = step["step_id"]
        image = Image.open(
            config.PROJECT_ROOT / step["image_path"]
        ).convert("RGB")
        instruction = step["instruction"]

        # Prepare inputs
        prompt = config.PROMPT_TEMPLATE.format(instruction=instruction)
        inputs = processor(prompt, image, return_tensors="pt").to(device)
        if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        pixel_values = inputs.get("pixel_values")

        # Autoregressive generation with per-token gradient profiling
        for token_idx in range(config.NUM_ACTION_TOKENS):
            # Clear state from previous token
            model.zero_grad()
            profiler.clear()

            # Forward pass (WITH gradients — builds computation graph)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                use_cache=False,
            )

            # Predicted token logit as scalar objective
            logits = outputs.logits[:, -1, :]  # (1, vocab_size)
            predicted_id = logits.argmax(dim=-1)  # (1,)
            target_logit = logits[0, predicted_id[0]]

            # Backward pass — computes gradients through all layers
            target_logit.backward()

            # Collect per-head importance from hooked activations
            importance = profiler.compute_importance()  # (num_layers, num_heads)
            all_importance[step_idx, token_idx] = importance

            # Prepare inputs for next token (detach to break graph)
            next_token = predicted_id.unsqueeze(0).detach()
            input_ids = torch.cat([input_ids.detach(), next_token], dim=-1)
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask.detach(),
                    torch.ones(1, 1, device=device, dtype=attention_mask.dtype),
                ], dim=-1)

            pbar.update(1)
            pbar.set_postfix(
                ep=ep_id, step=step_id,
                tok=config.ACTION_DIM_NAMES[token_idx],
            )

    pbar.close()
    profiler.remove_hooks()
    elapsed = time.time() - t_start

    # ==================================================================
    # Aggregate results
    # ==================================================================

    # Mean over steps: (7, num_layers, num_heads)
    mean_importance = all_importance.mean(axis=0)
    std_importance = all_importance.std(axis=0)

    # Per-layer normalized: within each (action_dim, layer), heads sum to 1
    norm_importance = mean_importance.copy()
    for d in range(config.NUM_ACTION_TOKENS):
        for l in range(num_layers):
            layer_sum = norm_importance[d, l].sum()
            if layer_sum > 0:
                norm_importance[d, l] /= layer_sum

    # ==================================================================
    # Head classification
    # ==================================================================

    head_classification = {}
    for l in range(num_layers):
        for h in range(num_heads):
            dim_scores = mean_importance[:, l, h]  # (7,)
            total = float(dim_scores.sum())
            top_dim = int(np.argmax(dim_scores))
            top_score = float(dim_scores[top_dim])
            dominance = top_score / total if total > 0 else 0.0

            # Specialist: one dimension has >40% of total importance
            head_type = "specialist" if dominance > 0.4 else "generalist"

            head_classification[f"L{l}_H{h}"] = {
                "layer": l,
                "head": h,
                "top_dim": config.ACTION_DIM_NAMES[top_dim],
                "top_dim_idx": top_dim,
                "top_score": round(top_score, 6),
                "dominance_ratio": round(dominance, 4),
                "type": head_type,
                "scores": {
                    config.ACTION_DIM_NAMES[d]: round(float(dim_scores[d]), 6)
                    for d in range(config.NUM_ACTION_TOKENS)
                },
            }

    # ==================================================================
    # Dimension similarity (cosine of importance vectors)
    # ==================================================================

    dim_vectors = mean_importance.reshape(config.NUM_ACTION_TOKENS, -1)  # (7, L*H)
    dim_norms = np.linalg.norm(dim_vectors, axis=1, keepdims=True) + 1e-9
    dim_similarity = (dim_vectors @ dim_vectors.T) / (dim_norms @ dim_norms.T)

    # ==================================================================
    # Save results
    # ==================================================================

    out_dir = config.HEAD_PROFILING_DIR
    np.save(out_dir / "importance_mean.npy", mean_importance)
    np.save(out_dir / "importance_std.npy", std_importance)
    np.save(out_dir / "importance_normalized.npy", norm_importance)

    if save_raw:
        np.save(out_dir / "importance_raw.npy", all_importance)

    with open(out_dir / "head_classification.json", "w") as f:
        json.dump(head_classification, f, indent=2)

    # Build summary
    specialist_count = sum(
        1 for v in head_classification.values() if v["type"] == "specialist"
    )
    generalist_count = sum(
        1 for v in head_classification.values() if v["type"] == "generalist"
    )

    dim_specialist_counts = {d: 0 for d in config.ACTION_DIM_NAMES}
    for v in head_classification.values():
        if v["type"] == "specialist":
            dim_specialist_counts[v["top_dim"]] += 1

    summary = {
        "calibration_episodes": episode_ids,
        "num_steps": total_steps,
        "elapsed_seconds": round(elapsed, 1),
        "architecture": {
            "num_layers": num_layers,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "total_heads": num_layers * num_heads,
        },
        "token_boundaries": boundaries,
        "head_stats": {
            "total_heads": num_layers * num_heads,
            "specialists": specialist_count,
            "generalists": generalist_count,
            "specialist_ratio": round(specialist_count / (num_layers * num_heads), 4),
        },
        "dim_specialist_counts": dim_specialist_counts,
        "dim_similarity": {
            f"{config.ACTION_DIM_NAMES[i]}-{config.ACTION_DIM_NAMES[j]}":
            round(float(dim_similarity[i, j]), 4)
            for i in range(config.NUM_ACTION_TOKENS)
            for j in range(i + 1, config.NUM_ACTION_TOKENS)
        },
        "action_dim_names": config.ACTION_DIM_NAMES,
        "importance_shape": list(mean_importance.shape),
    }

    # Top-10 heads per dimension
    for d in range(config.NUM_ACTION_TOKENS):
        dim_name = config.ACTION_DIM_NAMES[d]
        flat_importance = mean_importance[d].ravel()
        top_idx = np.argsort(flat_importance)[::-1][:10]
        top10 = []
        for idx in top_idx:
            l = int(idx // num_heads)
            h = int(idx % num_heads)
            top10.append({
                "layer": l,
                "head": h,
                "importance": round(float(mean_importance[d, l, h]), 6),
                "importance_std": round(float(std_importance[d, l, h]), 6),
            })
        summary[f"top10_{dim_name}"] = top10

    with open(out_dir / "profiling_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ==================================================================
    # Print summary
    # ==================================================================

    print(f"\n{'='*70}")
    print(f"  Head Profiling Complete — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Steps profiled: {total_steps}")
    print(f"  Results: {out_dir}/")
    print(f"{'='*70}")

    print(f"\n  Head Classification (threshold: 40% dominance):")
    print(f"    Specialists: {specialist_count} / {num_layers * num_heads}")
    print(f"    Generalists: {generalist_count} / {num_layers * num_heads}")

    print(f"\n  Specialists per dimension:")
    for dim_name, count in dim_specialist_counts.items():
        print(f"    {dim_name:>8}: {count} heads")

    print(f"\n  Top-3 most important heads per dimension:")
    for d in range(config.NUM_ACTION_TOKENS):
        dim_name = config.ACTION_DIM_NAMES[d]
        top3 = summary[f"top10_{dim_name}"][:3]
        heads_str = ", ".join(
            f"L{h['layer']}H{h['head']}({h['importance']:.4f})"
            for h in top3
        )
        print(f"    {dim_name:>8}: {heads_str}")

    print(f"\n  Dimension similarity (cosine):")
    pairs = [
        ("x", "y"), ("x", "z"), ("y", "z"),
        ("roll", "pitch"), ("roll", "yaw"), ("pitch", "yaw"),
        ("x", "gripper"), ("z", "gripper"),
    ]
    for a, b in pairs:
        key = f"{a}-{b}"
        if key not in summary["dim_similarity"]:
            key = f"{b}-{a}"
        if key in summary["dim_similarity"]:
            print(f"    {a:>8} ↔ {b:<8}: {summary['dim_similarity'][key]:.4f}")

    print(f"\n  Files saved:")
    print(f"    importance_mean.npy:       ({config.NUM_ACTION_TOKENS}, {num_layers}, {num_heads})")
    print(f"    importance_std.npy:        ({config.NUM_ACTION_TOKENS}, {num_layers}, {num_heads})")
    print(f"    importance_normalized.npy: ({config.NUM_ACTION_TOKENS}, {num_layers}, {num_heads})")
    if save_raw:
        print(f"    importance_raw.npy:        ({total_steps}, {config.NUM_ACTION_TOKENS}, {num_layers}, {num_heads})")
    print(f"    head_classification.json:  1024 heads classified")
    print(f"    profiling_summary.json:    stats + top-10 per dim")


def main():
    parser = argparse.ArgumentParser(
        description="Head profiling via gradient-based attribution"
    )
    parser.add_argument(
        "--episodes", type=str, default="0,1,2,3,4",
        help="Comma-separated calibration episode IDs (default: 0,1,2,3,4)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (default: cuda)",
    )
    parser.add_argument(
        "--save_raw", action="store_true",
        help="Save per-step raw importance (~115MB)",
    )
    args = parser.parse_args()

    episode_ids = [int(x) for x in args.episodes.split(",")]
    run_profiling(
        episode_ids=episode_ids,
        device=args.device,
        save_raw=args.save_raw,
    )


if __name__ == "__main__":
    main()
