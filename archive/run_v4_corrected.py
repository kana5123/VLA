"""V4 corrected experiment — clean baseline vs SAM2+VAR+LayerSelective+Object.

Bug fixes from V3:
  1. Filter is_first / is_last steps (dummy zero actions in RLDS)
  2. Fixed off-by-one in detokenize bin centers (255 → 256)
  3. Added unnormalization warning

Usage:
    python run_v4_corrected.py --conditions baseline,sam_var_obj_laysel --device cuda
    CUDA_VISIBLE_DEVICES=0 python run_v4_corrected.py --conditions baseline &
    CUDA_VISIBLE_DEVICES=1 python run_v4_corrected.py --conditions sam_var_obj_laysel &
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
# Experimental conditions (minimal — only what we need)
# ======================================================================

CONDITIONS = {
    "baseline": dict(),

    "sam_var_obj_laysel": dict(
        use_var=True, var_p=0.6, var_rho=0.5,
        use_object_redirect=True,
        _use_sam2=True, _layer_selective=True,
    ),
}

ALL_CONDITION_NAMES = list(CONDITIONS.keys())


# ======================================================================
# Helpers
# ======================================================================

def compute_mse(pred: list[float], gt: list[float]) -> dict:
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


def describe_condition(name: str) -> str:
    if name == "baseline":
        return "no enhancement"
    cond = CONDITIONS[name]
    parts = []
    if cond.get("use_var"):
        parts.append(f"VAR(p={cond.get('var_p', 0.6)},rho={cond.get('var_rho', 0.5)})")
    if cond.get("use_object_redirect"):
        parts.append("obj_redirect")
    if cond.get("_layer_selective"):
        lo, hi = config.LAYER_SELECTIVE_RANGE
        parts.append(f"L{lo}-{hi-1}")
    if cond.get("_use_sam2"):
        parts.append("SAM2")
    return " + ".join(parts) if parts else name


def configure_condition(ctx, condition_name, grounding_result, vision_end):
    """Configure ctx for a given condition before inference."""
    cond = CONDITIONS[condition_name]

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

    if condition_name == "baseline":
        return

    patch_indices = grounding_result.patch_indices if grounding_result else []

    # VAR
    if cond.get("use_var"):
        ctx.use_var = True
        ctx.var_p = cond.get("var_p", config.VAR_P)
        ctx.var_rho = cond.get("var_rho", config.VAR_RHO)
        ctx.var_sink_indices = list(config.VAR_SINK_INDICES)

    # Object redirect
    if cond.get("use_object_redirect"):
        ctx.use_object_redirect = True
        ctx.object_patch_indices = patch_indices
        ctx.object_redirect_weight = cond.get(
            "object_redirect_weight", config.VAR_OBJECT_REDIRECT_WEIGHT
        )

    # Layer-selective
    if cond.get("_layer_selective"):
        lo, hi = config.LAYER_SELECTIVE_RANGE
        ctx.enhancement_layers = set(range(lo, hi))

    ctx.active = True


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
# Main experiment
# ======================================================================

def run_v4_experiment(episode_ids=None, device="cuda", conditions=None):
    """Run corrected V4 experiment with filtered steps."""
    if conditions is None:
        conditions = ALL_CONDITION_NAMES

    if not config.METADATA_PATH.exists():
        print(f"ERROR: Metadata not found at {config.METADATA_PATH}")
        sys.exit(1)

    with open(config.METADATA_PATH) as f:
        metadata = json.load(f)

    episodes = metadata["episodes"]
    if episode_ids is not None:
        episodes = [ep for ep in episodes if ep["episode_id"] in episode_ids]
    if not episodes:
        print("No episodes to process.")
        return

    # Count valid steps (filter is_first / is_last)
    total_raw = sum(len(ep["steps"]) for ep in episodes)
    valid_steps = []
    for ep in episodes:
        for s in ep["steps"]:
            if not s.get("is_first", False) and not s.get("is_last", False):
                valid_steps.append((ep["episode_id"], s))
    total_steps = len(valid_steps)
    skipped = total_raw - total_steps

    print(f"\n{'='*70}")
    print(f"  V4 Corrected Experiment")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Valid steps: {total_steps} (filtered {skipped} is_first/is_last)")
    print(f"  Conditions: {len(conditions)}")
    print(f"  Total inference runs: {total_steps * len(conditions)}")
    print(f"{'='*70}")
    for c in conditions:
        print(f"  {c:<25} {describe_condition(c)}")
    print(f"{'='*70}\n")

    # Load model
    print("Loading model...")
    processor, model = load_model(device)

    # Check unnormalization
    norm_stats = getattr(model, "norm_stats", None) or getattr(model.config, "norm_stats", None)
    if norm_stats and config.BRIDGE_UNNORM_KEY in norm_stats:
        print(f"  Unnormalization: OK (key='{config.BRIDGE_UNNORM_KEY}')")
    else:
        print(f"  WARNING: No norm_stats for '{config.BRIDGE_UNNORM_KEY}' — MSE may be in wrong scale!")

    # Detect token boundaries
    first_valid = valid_steps[0][1]
    sample_image = Image.open(
        config.PROJECT_ROOT / first_valid["image_path"]
    ).convert("RGB")
    boundaries = detect_token_boundaries(
        processor, model, sample_image, first_valid["instruction"], device
    )
    vision_end = boundaries["vision_end"]
    num_vision_tokens = boundaries["num_vision_tokens"]
    print(f"  vision_end={vision_end}, num_vision_tokens={num_vision_tokens}")

    # Install V3 monkey-patch
    ctx = V3Context()
    ctx.vision_end = vision_end
    ctx.active = False
    set_v3_context(ctx)
    install_v3_patch(ctx)

    # Attach vision_end to self_attn modules
    if hasattr(model, "language_model"):
        layers = model.language_model.model.layers
    else:
        layers = model.model.layers
    for layer in layers:
        layer.self_attn._atlasvla_vision_end = vision_end

    # Init grounder if needed
    need_sam2 = any(
        CONDITIONS[c].get("_use_sam2") for c in conditions if c != "baseline"
    )
    need_grounding = any(
        CONDITIONS[c].get("use_object_redirect") for c in conditions if c != "baseline"
    )
    grounder = ObjectGrounder(device=device) if need_grounding else None

    # Prepare output dirs
    for cond_name in conditions:
        (config.V4_RESULTS_DIR / cond_name).mkdir(parents=True, exist_ok=True)

    # Main loop — iterate per episode
    pbar = tqdm(total=total_steps * len(conditions), desc="V4 experiment")
    all_indices = {c: [] for c in conditions}
    t_start = time.time()

    for ep in episodes:
        ep_id = ep["episode_id"]

        # Pre-compute grounding for episode
        grounding_cache = {}
        if need_grounding and grounder is not None:
            print(f"\n  Grounding episode {ep_id} (SAM2)...")
            grounding_cache = precompute_grounding_for_episode(
                grounder, ep, num_vision_tokens,
                use_sam2=need_sam2,
            )

        for step in ep["steps"]:
            # Filter is_first / is_last
            if step.get("is_first", False) or step.get("is_last", False):
                pbar.update(len(conditions))
                continue

            step_id = step["step_id"]
            image = Image.open(
                config.PROJECT_ROOT / step["image_path"]
            ).convert("RGB")
            instruction = step["instruction"]
            gt_action = step["action"]

            # Validate GT is not near-zero (extra safety)
            spatial_gt = gt_action[:6]
            gt_is_zero = all(abs(v) < 1e-8 for v in spatial_gt)

            for cond_name in conditions:
                grounding = grounding_cache.get(step_id)
                configure_condition(ctx, cond_name, grounding, vision_end)

                token_ids = run_inference_step(
                    processor, model, image, instruction,
                    boundaries, device, ctx,
                )
                ctx.active = False

                action_info = detokenize_actions(model, token_ids)
                pred = action_info["unnormalized_action"]
                if pred is None:
                    print(f"  WARNING: unnormalized_action is None at ep{ep_id} step{step_id}")
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
                    "gt_is_zero_spatial": gt_is_zero,
                    **mse,
                }

                if grounding and cond_name != "baseline":
                    result["grounding"] = {
                        "nouns": grounding.nouns,
                        "num_boxes": len(grounding.boxes_xyxy),
                        "num_patches": len(grounding.patch_indices),
                        "patch_coverage": grounding.patch_coverage,
                    }

                out_path = (
                    config.V4_RESULTS_DIR / cond_name
                    / f"ep{ep_id:03d}_step{step_id:03d}.json"
                )
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)

                all_indices[cond_name].append({
                    "episode_id": ep_id,
                    "step_id": step_id,
                    "mse_mean": result["mse_mean"],
                    "mse_spatial": result["mse_spatial"],
                    "mse_gripper": result["mse_gripper"],
                    "gt_is_zero_spatial": gt_is_zero,
                })

                pbar.update(1)
                pbar.set_postfix(ep=ep_id, step=step_id, cond=cond_name[:15])

    pbar.close()
    elapsed = time.time() - t_start

    # Save index files
    for cond_name in conditions:
        idx_path = config.V4_RESULTS_DIR / cond_name / "index.json"
        with open(idx_path, "w") as f:
            json.dump({
                "condition": cond_name,
                "description": describe_condition(cond_name),
                "config": CONDITIONS[cond_name],
                "token_boundaries": boundaries,
                "results": all_indices[cond_name],
            }, f, indent=2)

    uninstall_v3_patch()

    # Summary
    print(f"\n{'='*70}")
    print(f"  V4 experiment complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Results: {config.V4_RESULTS_DIR}/")
    print(f"{'='*70}")

    bl_mse = None
    if "baseline" in all_indices and all_indices["baseline"]:
        bl_mse = np.mean([r["mse_mean"] for r in all_indices["baseline"]])

    header = "Condition"
    print(f"\n  {header:<25} {'MSE':>10} {'Spatial':>10} {'Gripper':>10} {'vs BL':>10}")
    print("  " + "-" * 67)
    for cond_name in conditions:
        if cond_name not in all_indices or not all_indices[cond_name]:
            continue
        vals = all_indices[cond_name]
        avg_mse = np.mean([r["mse_mean"] for r in vals])
        avg_sp = np.mean([r["mse_spatial"] for r in vals])
        avg_gr = np.mean([r["mse_gripper"] for r in vals])
        if bl_mse is not None and cond_name != "baseline":
            delta_pct = 100 * (avg_mse - bl_mse) / bl_mse if bl_mse != 0 else 0
            vs_bl = f"{delta_pct:+.2f}%"
        else:
            vs_bl = "-"
        print(f"  {cond_name:<25} {avg_mse:>10.6f} {avg_sp:>10.6f} {avg_gr:>10.6f} {vs_bl:>10}")

    # Also show per-dimension breakdown
    print(f"\n  Per-dimension MSE breakdown:")
    print(f"  {'Condition':<25} {'x':>8} {'y':>8} {'z':>8} {'roll':>8} {'pitch':>8} {'yaw':>8} {'grip':>8}")
    print("  " + "-" * 81)
    for cond_name in conditions:
        if cond_name not in all_indices or not all_indices[cond_name]:
            continue
        # Load per-dim from files
        cond_dir = config.V4_RESULTS_DIR / cond_name
        files = sorted(cond_dir.glob("ep*_step*.json"))
        all_per_dim = []
        for f_path in files:
            d = json.loads(f_path.read_text())
            all_per_dim.append(d["mse_per_dim"])
        avg_per_dim = np.mean(all_per_dim, axis=0)
        dims = " ".join(f"{v:>8.6f}" for v in avg_per_dim)
        print(f"  {cond_name:<25} {dims}")


def main():
    parser = argparse.ArgumentParser(description="V4 corrected experiment")
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

    invalid = [c for c in conditions if c not in CONDITIONS]
    if invalid:
        print(f"ERROR: Unknown conditions: {invalid}")
        sys.exit(1)

    run_v4_experiment(
        episode_ids=episode_ids,
        device=args.device,
        conditions=conditions,
    )


if __name__ == "__main__":
    main()
