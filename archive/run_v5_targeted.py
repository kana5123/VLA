"""V5 targeted experiment — dimension-aware per-head VAR intervention.

Uses head profiling results (importance_mean.npy) + V4 per-dimension MSE to
compute optimal per-head VAR strength. Only applies VAR to heads whose
primary dimensions benefit from redistribution.

Conditions:
  1. baseline         — no modification
  2. v4_replay        — same as V4: sam_var_obj_laysel (uniform VAR)
  3. dim_gate         — per-dimension VAR gate: only apply for z/pitch/yaw
  4. targeted         — per-dimension gate + per-head strength modulation

Usage:
    # All conditions on one GPU:
    python run_v5_targeted.py --device cuda

    # Parallel across GPUs (one condition each):
    CUDA_VISIBLE_DEVICES=0 python run_v5_targeted.py --conditions baseline &
    CUDA_VISIBLE_DEVICES=1 python run_v5_targeted.py --conditions v4_replay &
    CUDA_VISIBLE_DEVICES=2 python run_v5_targeted.py --conditions dim_gate &
    CUDA_VISIBLE_DEVICES=3 python run_v5_targeted.py --conditions targeted &
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
from extract_attention import load_model, detect_token_boundaries, detokenize_actions
from object_grounder import ObjectGrounder, precompute_grounding_for_episode
from attention_v3 import (
    V3Context, install_v3_patch, uninstall_v3_patch,
    get_v3_context, set_v3_context,
)


# ======================================================================
# V4 per-dimension MSE change (%) — used to determine which dims benefit
# ======================================================================

# From V4 corrected results: (enhanced MSE - baseline MSE) / baseline MSE
# Negative = improvement, Positive = worsening
V4_DIM_MSE_CHANGE = {
    "x":       +0.0189,   # +1.89% worsened
    "y":       +0.0196,   # +1.96% worsened
    "z":       -0.0210,   # -2.10% improved
    "roll":    +0.0975,   # +9.75% worsened
    "pitch":   -0.0048,   # -0.48% improved
    "yaw":     -0.0293,   # -2.93% improved
    "gripper":  0.0,      # neutral
}


# ======================================================================
# Per-dimension VAR gate factors
# ======================================================================

def compute_dim_var_factors(threshold=0.0):
    """Compute per-dimension VAR on/off multipliers.

    Dimensions where VAR improved MSE (negative change) get factor=1.0.
    Dimensions where VAR worsened MSE (positive change) get factor=0.0.

    Args:
        threshold: MSE change threshold. Dimensions with change > threshold
                   are turned off.
    Returns:
        list of 7 floats (one per action dimension)
    """
    factors = []
    for dim_name in config.ACTION_DIM_NAMES:
        change = V4_DIM_MSE_CHANGE[dim_name]
        if change <= threshold:
            factors.append(1.0)  # VAR helped this dimension
        else:
            factors.append(0.0)  # VAR hurt this dimension
    return factors


# ======================================================================
# Per-head VAR strength from profiling
# ======================================================================

def compute_per_head_var_strength(importance_mean, base_p=0.6):
    """Compute per-layer, per-head VAR strength from profiling data.

    For each head (l, h):
    1. Normalize importance across dimensions → dim_weight (sums to 1)
    2. Compute weighted benefit = sum(dim_weight * dim_benefit_sign)
    3. If weighted benefit > 0: head benefits from VAR → strength ∝ benefit
    4. If weighted benefit <= 0: head hurts from VAR → strength = 0

    Args:
        importance_mean: (7, L, H) importance tensor from profiling
        base_p: maximum redistribution fraction

    Returns:
        (L, H) numpy array of per-head VAR strengths in [0, base_p]
    """
    n_dims, n_layers, n_heads = importance_mean.shape

    # Convert MSE changes to benefit signs: improved → positive
    dim_benefit = np.array([
        -V4_DIM_MSE_CHANGE[d] for d in config.ACTION_DIM_NAMES
    ])  # positive = good, negative = bad

    per_head_p = np.zeros((n_layers, n_heads), dtype=np.float32)

    for l in range(n_layers):
        for h in range(n_heads):
            dim_imp = importance_mean[:, l, h]  # (7,)
            total = dim_imp.sum()
            if total < 1e-9:
                continue

            # Normalized importance weights
            weights = dim_imp / total  # (7,), sums to 1

            # Weighted benefit score
            benefit = (weights * dim_benefit).sum()

            if benefit > 0:
                # Head overall benefits from VAR
                # Scale: max benefit ≈ max(dim_benefit) ≈ 0.0975
                max_benefit = np.abs(dim_benefit).max()
                strength = base_p * min(1.0, benefit / max_benefit)
                per_head_p[l, h] = strength
            # else: strength = 0, head should not have VAR

    return per_head_p


# ======================================================================
# Experimental conditions
# ======================================================================

ALL_CONDITION_NAMES = ["baseline", "v4_replay", "dim_gate", "targeted"]


def describe_condition(name):
    descs = {
        "baseline": "no enhancement",
        "v4_replay": "VAR(p=0.6,rho=0.5) + obj + SAM2 + L8-24 (same as V4)",
        "dim_gate": "VAR only for z/pitch/yaw tokens + obj + SAM2 + L8-24",
        "targeted": "per-head VAR strength + dim gate + obj + SAM2 + L8-24",
    }
    return descs.get(name, name)


# ======================================================================
# Helpers (same as V4)
# ======================================================================

def compute_mse(pred, gt):
    pred_arr = np.array(pred, dtype=float)
    gt_arr = np.array(gt, dtype=float)
    per_dim = ((pred_arr - gt_arr) ** 2).tolist()
    return {
        "mse_per_dim": per_dim,
        "mse_mean": float(np.mean(per_dim)),
        "mse_spatial": float(np.mean(per_dim[:6])),
        "mse_gripper": float(per_dim[6]) if len(per_dim) > 6 else 0.0,
        "mse_per_dim_names": config.ACTION_DIM_NAMES,
    }


def run_inference_step(processor, model, image, instruction, boundaries, device, ctx):
    """Autoregressive inference with per-token tracking."""
    prompt = config.PROMPT_TEMPLATE.format(instruction=instruction)
    inputs = processor(prompt, image, return_tensors="pt").to(device)
    if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs.get("pixel_values")

    generated_tokens = []
    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
    }

    with torch.no_grad():
        for token_idx in range(config.NUM_ACTION_TOKENS):
            ctx.current_token_idx = token_idx
            outputs = model(**model_inputs, use_cache=False)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_tokens.append(next_token[0].item())

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(1, 1, device=device, dtype=attention_mask.dtype),
                ], dim=-1)
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
            }

    return generated_tokens


# ======================================================================
# Condition configuration
# ======================================================================

def configure_condition(ctx, name, grounding, vision_end,
                        dim_factors, per_head_p_tensor):
    """Configure V3Context for a specific condition."""
    # Full reset
    ctx.active = False
    ctx.current_token_idx = 0
    ctx.use_var = False
    ctx.var_decay = False
    ctx.use_vt_rebalance = False
    ctx.use_temporal = False
    ctx.temporal_patch_indices = []
    ctx.use_act = False
    ctx.use_spin = False
    ctx.use_bg_suppress = False
    ctx.use_object_redirect = False
    ctx.gripper_exempt = False
    ctx.vision_end = vision_end
    ctx.object_patch_indices = []
    ctx.enhancement_layers = None
    ctx.dim_var_factors = None
    ctx.per_head_var_strength = None

    if name == "baseline":
        return

    patch_indices = grounding.patch_indices if grounding else []

    # Common: VAR + object + SAM2 + layer selective
    ctx.use_var = True
    ctx.var_p = 0.6
    ctx.var_rho = 0.5
    ctx.var_sink_indices = list(config.VAR_SINK_INDICES)

    ctx.use_object_redirect = True
    ctx.object_patch_indices = patch_indices
    ctx.object_redirect_weight = config.VAR_OBJECT_REDIRECT_WEIGHT

    lo, hi = config.LAYER_SELECTIVE_RANGE
    ctx.enhancement_layers = set(range(lo, hi))

    # Condition-specific
    if name == "v4_replay":
        pass  # uniform VAR, same as V4

    elif name == "dim_gate":
        ctx.dim_var_factors = dim_factors

    elif name == "targeted":
        ctx.dim_var_factors = dim_factors
        ctx.per_head_var_strength = per_head_p_tensor

    ctx.active = True


# ======================================================================
# Main experiment
# ======================================================================

def run_v5_experiment(episode_ids=None, device="cuda", conditions=None):
    """Run V5 targeted experiment."""
    if conditions is None:
        conditions = ALL_CONDITION_NAMES

    if not config.METADATA_PATH.exists():
        print(f"ERROR: Metadata not found at {config.METADATA_PATH}")
        sys.exit(1)

    # Load profiling results
    profiling_dir = config.HEAD_PROFILING_DIR
    if not (profiling_dir / "importance_mean.npy").exists():
        print(f"ERROR: Profiling results not found at {profiling_dir}")
        print("Run run_head_profiling.py first.")
        sys.exit(1)

    importance_mean = np.load(profiling_dir / "importance_mean.npy")
    print(f"Loaded profiling: importance shape = {importance_mean.shape}")

    # Compute intervention parameters
    dim_factors = compute_dim_var_factors()
    per_head_p = compute_per_head_var_strength(importance_mean, base_p=0.6)

    print(f"\nDimension VAR factors: {dict(zip(config.ACTION_DIM_NAMES, dim_factors))}")
    active_heads = (per_head_p > 0).sum()
    print(f"Per-head VAR: {active_heads}/{per_head_p.size} heads active "
          f"(mean p={per_head_p[per_head_p > 0].mean():.3f}, "
          f"max p={per_head_p.max():.3f})")

    # Load metadata
    with open(config.METADATA_PATH) as f:
        metadata = json.load(f)

    episodes = metadata["episodes"]
    if episode_ids is not None:
        episodes = [ep for ep in episodes if ep["episode_id"] in episode_ids]
    if not episodes:
        print("No episodes to process.")
        return

    # Valid steps (skip is_first/is_last)
    valid_steps = []
    for ep in episodes:
        for s in ep["steps"]:
            if not s.get("is_first", False) and not s.get("is_last", False):
                valid_steps.append((ep["episode_id"], s))

    total_steps = len(valid_steps)
    total_raw = sum(len(ep["steps"]) for ep in episodes)
    skipped = total_raw - total_steps

    print(f"\n{'='*70}")
    print(f"  V5 Targeted Experiment")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Valid steps: {total_steps} (filtered {skipped})")
    print(f"  Conditions: {len(conditions)}")
    print(f"  Total inference runs: {total_steps * len(conditions)}")
    print(f"{'='*70}")
    for c in conditions:
        print(f"  {c:<20} {describe_condition(c)}")
    print(f"{'='*70}\n")

    # Load model
    print("Loading model...")
    processor, model = load_model(device)

    # Check unnormalization
    norm_stats = getattr(model, "norm_stats", None) or getattr(model.config, "norm_stats", None)
    if norm_stats and config.BRIDGE_UNNORM_KEY in norm_stats:
        print(f"  Unnormalization: OK")
    else:
        print(f"  WARNING: No norm_stats for '{config.BRIDGE_UNNORM_KEY}'")

    # Token boundaries
    first_step = valid_steps[0][1]
    sample_image = Image.open(
        config.PROJECT_ROOT / first_step["image_path"]
    ).convert("RGB")
    boundaries = detect_token_boundaries(
        processor, model, sample_image, first_step["instruction"], device
    )
    vision_end = boundaries["vision_end"]
    print(f"  vision_end={vision_end}")

    # Install patch
    ctx = V3Context()
    ctx.vision_end = vision_end
    ctx.active = False
    set_v3_context(ctx)
    install_v3_patch(ctx)

    # Attach vision_end to self_attn modules
    layers = (model.language_model.model.layers
              if hasattr(model, "language_model") else model.model.layers)
    for layer in layers:
        layer.self_attn._atlasvla_vision_end = vision_end

    # Convert per_head_p to tensor on device
    per_head_p_tensor = torch.from_numpy(per_head_p).float().to(device)

    # Init grounder
    need_grounding = any(c != "baseline" for c in conditions)
    grounder = ObjectGrounder(device=device) if need_grounding else None

    # Output dirs
    out_dir = config.V5_RESULTS_DIR
    for c in conditions:
        (out_dir / c).mkdir(parents=True, exist_ok=True)

    # Main loop
    pbar = tqdm(total=total_steps * len(conditions), desc="V5 experiment")
    all_indices = {c: [] for c in conditions}
    t_start = time.time()

    for ep in episodes:
        ep_id = ep["episode_id"]

        # Pre-compute grounding
        grounding_cache = {}
        if need_grounding and grounder is not None:
            print(f"\n  Grounding episode {ep_id}...")
            grounding_cache = precompute_grounding_for_episode(
                grounder, ep, boundaries["num_vision_tokens"],
                use_sam2=True,
            )

        for step in ep["steps"]:
            if step.get("is_first", False) or step.get("is_last", False):
                pbar.update(len(conditions))
                continue

            step_id = step["step_id"]
            image = Image.open(
                config.PROJECT_ROOT / step["image_path"]
            ).convert("RGB")
            instruction = step["instruction"]
            gt_action = step["action"]

            for cond_name in conditions:
                grounding = grounding_cache.get(step_id)
                configure_condition(
                    ctx, cond_name, grounding, vision_end,
                    dim_factors, per_head_p_tensor,
                )

                token_ids = run_inference_step(
                    processor, model, image, instruction,
                    boundaries, device, ctx,
                )
                ctx.active = False

                action_info = detokenize_actions(model, token_ids)
                pred = action_info["unnormalized_action"]
                if pred is None:
                    pred = action_info["normalized_action"]

                mse = compute_mse(pred, gt_action)

                result = {
                    "episode_id": ep_id,
                    "step_id": step_id,
                    "instruction": instruction,
                    "condition": cond_name,
                    "condition_desc": describe_condition(cond_name),
                    "predicted_action": pred,
                    "ground_truth_action": gt_action,
                    "action_token_ids": token_ids,
                    **mse,
                }

                if grounding and cond_name != "baseline":
                    result["grounding"] = {
                        "nouns": grounding.nouns,
                        "num_patches": len(grounding.patch_indices),
                        "patch_coverage": grounding.patch_coverage,
                    }

                out_path = out_dir / cond_name / f"ep{ep_id:03d}_step{step_id:03d}.json"
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)

                all_indices[cond_name].append({
                    "episode_id": ep_id,
                    "step_id": step_id,
                    "mse_mean": result["mse_mean"],
                    "mse_spatial": result["mse_spatial"],
                    "mse_gripper": result["mse_gripper"],
                })

                pbar.update(1)
                pbar.set_postfix(ep=ep_id, step=step_id, cond=cond_name[:12])

    pbar.close()
    elapsed = time.time() - t_start

    # Save index files
    for cond_name in conditions:
        idx_path = out_dir / cond_name / "index.json"
        with open(idx_path, "w") as f:
            json.dump({
                "condition": cond_name,
                "description": describe_condition(cond_name),
                "dim_var_factors": dim_factors if cond_name in ("dim_gate", "targeted") else None,
                "per_head_active": int(active_heads) if cond_name == "targeted" else None,
                "token_boundaries": boundaries,
                "results": all_indices[cond_name],
            }, f, indent=2)

    uninstall_v3_patch()

    # ==================================================================
    # Summary
    # ==================================================================

    print(f"\n{'='*70}")
    print(f"  V5 experiment complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*70}")

    bl_mse = None
    bl_spatial = None
    if "baseline" in all_indices and all_indices["baseline"]:
        bl_mse = np.mean([r["mse_mean"] for r in all_indices["baseline"]])
        bl_spatial = np.mean([r["mse_spatial"] for r in all_indices["baseline"]])

    print(f"\n  {'Condition':<20} {'MSE':>10} {'Spatial':>10} {'Gripper':>10} {'vs BL':>10} {'Sp.vsBL':>10}")
    print("  " + "-" * 72)
    for cond_name in conditions:
        if not all_indices[cond_name]:
            continue
        vals = all_indices[cond_name]
        avg_mse = np.mean([r["mse_mean"] for r in vals])
        avg_sp = np.mean([r["mse_spatial"] for r in vals])
        avg_gr = np.mean([r["mse_gripper"] for r in vals])

        vs_bl = "-"
        vs_sp = "-"
        if bl_mse is not None and cond_name != "baseline":
            vs_bl = f"{100*(avg_mse-bl_mse)/bl_mse:+.2f}%" if bl_mse != 0 else "N/A"
            vs_sp = f"{100*(avg_sp-bl_spatial)/bl_spatial:+.2f}%" if bl_spatial != 0 else "N/A"
        print(f"  {cond_name:<20} {avg_mse:>10.6f} {avg_sp:>10.6f} {avg_gr:>10.6f} {vs_bl:>10} {vs_sp:>10}")

    # Per-dimension breakdown
    print(f"\n  Per-dimension MSE:")
    print(f"  {'Condition':<20} {'x':>8} {'y':>8} {'z':>8} {'roll':>8} {'pitch':>8} {'yaw':>8} {'grip':>8}")
    print("  " + "-" * 76)

    cond_per_dim = {}
    for cond_name in conditions:
        cond_dir = out_dir / cond_name
        files = sorted(cond_dir.glob("ep*_step*.json"))
        if not files:
            continue
        all_per_dim = []
        for f_path in files:
            d = json.loads(f_path.read_text())
            all_per_dim.append(d["mse_per_dim"])
        avg_per_dim = np.mean(all_per_dim, axis=0)
        cond_per_dim[cond_name] = avg_per_dim
        dims = " ".join(f"{v:>8.6f}" for v in avg_per_dim)
        print(f"  {cond_name:<20} {dims}")

    # Per-dimension change vs baseline
    if "baseline" in cond_per_dim:
        bl_pd = cond_per_dim["baseline"]
        print(f"\n  Per-dimension change vs baseline (%):")
        print(f"  {'Condition':<20} {'x':>8} {'y':>8} {'z':>8} {'roll':>8} {'pitch':>8} {'yaw':>8} {'grip':>8}")
        print("  " + "-" * 76)
        for cond_name in conditions:
            if cond_name == "baseline" or cond_name not in cond_per_dim:
                continue
            pd = cond_per_dim[cond_name]
            pcts = []
            for i in range(len(pd)):
                if bl_pd[i] != 0:
                    pcts.append(f"{100*(pd[i]-bl_pd[i])/bl_pd[i]:>+7.2f}%")
                else:
                    pcts.append(f"{'N/A':>8}")
            print(f"  {cond_name:<20} {' '.join(pcts)}")


def main():
    parser = argparse.ArgumentParser(description="V5 targeted experiment")
    parser.add_argument("--episodes", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conditions", type=str, default="all")
    args = parser.parse_args()

    episode_ids = (
        [int(x) for x in args.episodes.split(",")]
        if args.episodes else None
    )
    conditions = (
        ALL_CONDITION_NAMES if args.conditions == "all"
        else [c.strip() for c in args.conditions.split(",")]
    )

    invalid = [c for c in conditions if c not in ALL_CONDITION_NAMES]
    if invalid:
        print(f"ERROR: Unknown conditions: {invalid}")
        print(f"Available: {ALL_CONDITION_NAMES}")
        sys.exit(1)

    run_v5_experiment(
        episode_ids=episode_ids,
        device=args.device,
        conditions=conditions,
    )


if __name__ == "__main__":
    main()
