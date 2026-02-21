"""V2 attention enhancement experiment — systematic combination search.

Tests 5 improvements individually, in pairs/triples, and fully combined:
  1. micro_params:  conservative hyperparameters (λ=1.1, middle layers)
  2. gripper_exempt: disable enhancement for gripper token (7th action token)
  3. bg_suppress:   suppress non-object vision patches (γ=0.85)
  4. mid_layers:    restrict enhancement to middle layers [12,14,16,18,20]
  5. resid_steer:   residual stream steering (α=1.5, layer 16)

Usage:
    python run_v2_experiment.py [--episodes 0,1,2] [--device cuda]
    python run_v2_experiment.py --conditions baseline,ind1_micro,full_bg
    python run_v2_experiment.py --reuse_baseline
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
from attention_patch import (
    EnhancementContext, install_patch, uninstall_patch,
    inject_vision_end, get_context,
)
from residual_steer import ResidualSteerer


# ======================================================================
# Experimental conditions
# ======================================================================
# Each condition is a dict specifying which improvements are active.
#   attn_method: None | "weight_rescale" | "bg_suppress" | "logit_bias"
#   wr_lambda:   weight_rescale lambda (default V2_WEIGHT_RESCALE_LAMBDA)
#   bg_gamma:    bg_suppress gamma (default BG_SUPPRESS_GAMMA)
#   layers:      None (all) or list of layer indices
#   gripper_exempt: bool
#   use_residual: bool
#   resid_alpha:  residual steer alpha

CONDITIONS = {
    # ── Baseline ──
    "baseline": dict(attn_method=None),

    # ── Individual improvements (5) ──
    "ind1_micro": dict(
        attn_method="weight_rescale",
        wr_lambda=config.V2_WEIGHT_RESCALE_LAMBDA,  # 1.1
    ),
    "ind2_gripex": dict(
        attn_method="weight_rescale",
        wr_lambda=config.WEIGHT_RESCALE_LAMBDA,  # 2.0 (original)
        gripper_exempt=True,
    ),
    "ind3_bgsup": dict(
        attn_method="bg_suppress",
        bg_gamma=config.BG_SUPPRESS_GAMMA,  # 0.85
    ),
    "ind4_midlay": dict(
        attn_method="weight_rescale",
        wr_lambda=config.WEIGHT_RESCALE_LAMBDA,  # 2.0
        layers=config.V2_ENHANCEMENT_LAYERS,  # [12,14,16,18,20]
    ),
    "ind5_resid": dict(
        attn_method=None,
        use_residual=True,
        resid_alpha=config.RESIDUAL_STEER_ALPHA,  # 1.5
    ),

    # ── Pairs (7) ──
    "pair_micro_gripex": dict(
        attn_method="weight_rescale",
        wr_lambda=config.V2_WEIGHT_RESCALE_LAMBDA,
        gripper_exempt=True,
    ),
    "pair_micro_midlay": dict(
        attn_method="weight_rescale",
        wr_lambda=config.V2_WEIGHT_RESCALE_LAMBDA,
        layers=config.V2_ENHANCEMENT_LAYERS,
    ),
    "pair_gripex_midlay": dict(
        attn_method="weight_rescale",
        wr_lambda=config.WEIGHT_RESCALE_LAMBDA,
        gripper_exempt=True,
        layers=config.V2_ENHANCEMENT_LAYERS,
    ),
    "pair_bgsup_gripex": dict(
        attn_method="bg_suppress",
        bg_gamma=config.BG_SUPPRESS_GAMMA,
        gripper_exempt=True,
    ),
    "pair_bgsup_midlay": dict(
        attn_method="bg_suppress",
        bg_gamma=config.BG_SUPPRESS_GAMMA,
        layers=config.V2_ENHANCEMENT_LAYERS,
    ),
    "pair_micro_resid": dict(
        attn_method="weight_rescale",
        wr_lambda=config.V2_WEIGHT_RESCALE_LAMBDA,
        use_residual=True,
    ),
    "pair_bgsup_resid": dict(
        attn_method="bg_suppress",
        bg_gamma=config.BG_SUPPRESS_GAMMA,
        use_residual=True,
    ),

    # ── Triples (4) ──
    "tri_micro_gripex_midlay": dict(
        attn_method="weight_rescale",
        wr_lambda=config.V2_WEIGHT_RESCALE_LAMBDA,
        gripper_exempt=True,
        layers=config.V2_ENHANCEMENT_LAYERS,
    ),
    "tri_bgsup_gripex_midlay": dict(
        attn_method="bg_suppress",
        bg_gamma=config.BG_SUPPRESS_GAMMA,
        gripper_exempt=True,
        layers=config.V2_ENHANCEMENT_LAYERS,
    ),
    "tri_micro_gripex_resid": dict(
        attn_method="weight_rescale",
        wr_lambda=config.V2_WEIGHT_RESCALE_LAMBDA,
        gripper_exempt=True,
        use_residual=True,
    ),
    "tri_bgsup_gripex_resid": dict(
        attn_method="bg_suppress",
        bg_gamma=config.BG_SUPPRESS_GAMMA,
        gripper_exempt=True,
        use_residual=True,
    ),

    # ── Full combos (2) ──
    "full_wr": dict(
        attn_method="weight_rescale",
        wr_lambda=config.V2_WEIGHT_RESCALE_LAMBDA,
        gripper_exempt=True,
        layers=config.V2_ENHANCEMENT_LAYERS,
        use_residual=True,
    ),
    "full_bg": dict(
        attn_method="bg_suppress",
        bg_gamma=config.BG_SUPPRESS_GAMMA,
        gripper_exempt=True,
        layers=config.V2_ENHANCEMENT_LAYERS,
        use_residual=True,
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
    """Human-readable one-liner for a condition."""
    cond = CONDITIONS[name]
    if name == "baseline":
        return "no enhancement"
    parts = []
    m = cond.get("attn_method")
    if m == "weight_rescale":
        parts.append(f"wr(λ={cond.get('wr_lambda', '?')})")
    elif m == "bg_suppress":
        parts.append(f"bg_sup(γ={cond.get('bg_gamma', '?')})")
    elif m == "logit_bias":
        parts.append(f"lb(α={cond.get('lb_alpha', '?')})")
    if cond.get("gripper_exempt"):
        parts.append("gripex")
    if cond.get("layers"):
        parts.append(f"L{cond['layers']}")
    if cond.get("use_residual"):
        parts.append(f"resid(α={cond.get('resid_alpha', config.RESIDUAL_STEER_ALPHA)})")
    return " + ".join(parts) if parts else name


def configure_condition(
    ctx: EnhancementContext,
    steerer: ResidualSteerer,
    condition_name: str,
    grounding_result,
    vision_end: int,
) -> None:
    """Set up ctx and steerer for a given condition before inference."""
    cond = CONDITIONS[condition_name]

    # Reset everything
    ctx.active = False
    ctx.current_token_idx = 0
    steerer.active = False
    steerer.current_token_idx = 0

    if condition_name == "baseline":
        return

    patch_indices = (
        grounding_result.patch_indices if grounding_result else []
    )

    # Attention-level enhancement
    attn_method = cond.get("attn_method")
    if attn_method:
        ctx.active = True
        ctx.method = attn_method
        ctx.weight_rescale_lambda = cond.get("wr_lambda", config.V2_WEIGHT_RESCALE_LAMBDA)
        ctx.logit_bias_alpha = cond.get("lb_alpha", config.V2_LOGIT_BIAS_ALPHA)
        ctx.bg_suppress_gamma = cond.get("bg_gamma", config.BG_SUPPRESS_GAMMA)
        ctx.head_steer_top_k = cond.get("hs_top_k", config.V2_HEAD_STEER_TOP_K_HEADS)
        ctx.head_steer_amplify = cond.get("hs_amplify", config.V2_HEAD_STEER_AMPLIFY)
        ctx.gripper_exempt = cond.get("gripper_exempt", False)
        ctx.current_token_idx = 0
        ctx.set_object_patches(patch_indices)

        layers = cond.get("layers")
        ctx.enhancement_layers = set(layers) if layers else None

    # Residual stream steering
    if cond.get("use_residual"):
        alpha = cond.get("resid_alpha", config.RESIDUAL_STEER_ALPHA)
        gripper_ex = cond.get("gripper_exempt", False)
        steerer.configure(patch_indices, vision_end, alpha=alpha, gripper_exempt=gripper_ex)
        steerer.active = True


def run_inference_step_v2(
    processor, model, image, instruction, boundaries, device,
    ctx, steerer,
):
    """Autoregressive inference with per-token index tracking."""
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
            # Update token index for gripper-exempt logic
            ctx.current_token_idx = token_idx
            steerer.current_token_idx = token_idx

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

def run_v2_experiment(
    episode_ids=None,
    device="cuda",
    conditions=None,
    reuse_baseline=False,
):
    """Run the full V2 enhancement experiment."""
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

    # Print experiment plan
    print(f"\n{'='*70}")
    print(f"  V2 Enhancement Experiment")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Conditions: {len(conditions)}")
    total_steps = sum(len(ep['steps']) for ep in episodes)
    print(f"  Total steps per condition: {total_steps}")
    print(f"  Total inference runs: {total_steps * len(conditions)}")
    print(f"{'='*70}")
    for c in conditions:
        print(f"  {c:<30} {describe_condition(c)}")
    print(f"{'='*70}\n")

    # Copy baseline from V1 if requested
    if reuse_baseline and "baseline" in conditions:
        v1_bl_dir = config.ENHANCEMENT_RESULTS_DIR / "baseline"
        v2_bl_dir = config.V2_RESULTS_DIR / "baseline"
        if v1_bl_dir.exists() and list(v1_bl_dir.glob("ep*_step*.json")):
            import shutil
            v2_bl_dir.mkdir(parents=True, exist_ok=True)
            for f in v1_bl_dir.glob("ep*_step*.json"):
                shutil.copy2(f, v2_bl_dir / f.name)
            # Also copy index
            idx = v1_bl_dir / "index.json"
            if idx.exists():
                shutil.copy2(idx, v2_bl_dir / "index.json")
            print(f"[v2] Reused V1 baseline ({len(list(v2_bl_dir.glob('ep*_step*.json')))} files)")
            conditions = [c for c in conditions if c != "baseline"]
            if not conditions:
                print("Only baseline requested and reused. Done.")
                return

    # Load model
    print("Loading model...")
    processor, model = load_model(device)

    # Detect token boundaries
    first_step = episodes[0]["steps"][0]
    sample_image = Image.open(config.PROJECT_ROOT / first_step["image_path"]).convert("RGB")
    boundaries = detect_token_boundaries(
        processor, model, sample_image, first_step["instruction"], device
    )
    vision_end = boundaries["vision_end"]
    print(f"  vision_end={vision_end}, num_vision_tokens={boundaries['num_vision_tokens']}")

    # Install monkey-patch (always installed, activation controlled per-condition)
    ctx = get_context()
    ctx.active = False
    install_patch(ctx)
    inject_vision_end(model, vision_end)

    # Install residual steerer (always installed, activation controlled per-condition)
    steerer = ResidualSteerer()
    steerer.install(model, layer_idx=config.RESIDUAL_STEER_LAYER)

    # Init grounder
    need_grounding = any(
        CONDITIONS[c].get("attn_method") or CONDITIONS[c].get("use_residual")
        for c in conditions if c != "baseline"
    )
    grounder = ObjectGrounder(device=device) if need_grounding else None

    # Prepare output dirs
    for cond_name in conditions:
        (config.V2_RESULTS_DIR / cond_name).mkdir(parents=True, exist_ok=True)

    # Main loop
    pbar = tqdm(total=total_steps * len(conditions), desc="V2 experiment")
    all_indices = {c: [] for c in conditions}
    t_start = time.time()

    for ep in episodes:
        ep_id = ep["episode_id"]

        grounding_cache = {}
        if need_grounding and grounder is not None:
            print(f"\n  Grounding episode {ep_id}...")
            grounding_cache = precompute_grounding_for_episode(
                grounder, ep, boundaries["num_vision_tokens"]
            )

        for step in ep["steps"]:
            step_id = step["step_id"]
            image = Image.open(
                config.PROJECT_ROOT / step["image_path"]
            ).convert("RGB")
            instruction = step["instruction"]
            gt_action = step["action"]
            grounding = grounding_cache.get(step_id)

            for cond_name in conditions:
                # Configure for this condition
                configure_condition(ctx, steerer, cond_name, grounding, vision_end)

                # Run inference
                token_ids = run_inference_step_v2(
                    processor, model, image, instruction, boundaries, device,
                    ctx, steerer,
                )

                # Deactivate
                ctx.active = False
                steerer.active = False

                # Decode and compute MSE
                action_info = detokenize_actions(model, token_ids)
                pred = action_info["unnormalized_action"] or action_info["normalized_action"]
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
                        "num_boxes": len(grounding.boxes_xyxy),
                        "num_patches": len(grounding.patch_indices),
                        "patch_coverage": (
                            len(grounding.patch_indices) / vision_end
                            if vision_end > 0 else 0
                        ),
                    }

                out_path = (
                    config.V2_RESULTS_DIR / cond_name
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
                })

                pbar.update(1)
                pbar.set_postfix(ep=ep_id, step=step_id, cond=cond_name[:12])

    pbar.close()
    elapsed = time.time() - t_start

    # Save index files
    for cond_name in conditions:
        idx_path = config.V2_RESULTS_DIR / cond_name / "index.json"
        with open(idx_path, "w") as f:
            json.dump({
                "condition": cond_name,
                "description": describe_condition(cond_name),
                "config": {
                    k: (list(v) if isinstance(v, (set, list)) else v)
                    for k, v in CONDITIONS[cond_name].items()
                },
                "token_boundaries": boundaries,
                "results": all_indices[cond_name],
            }, f, indent=2)

    # Cleanup
    steerer.remove()
    uninstall_patch()

    print(f"\n{'='*70}")
    print(f"  V2 experiment complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Results: {config.V2_RESULTS_DIR}/")
    print(f"{'='*70}")

    # Quick summary table
    bl_mse = None
    if "baseline" in all_indices and all_indices["baseline"]:
        bl_mse = np.mean([r["mse_mean"] for r in all_indices["baseline"]])
    elif reuse_baseline:
        # Load from saved baseline
        bl_dir = config.V2_RESULTS_DIR / "baseline"
        bl_files = sorted(bl_dir.glob("ep*_step*.json"))
        if bl_files:
            bl_vals = [json.loads(f.read_text())["mse_mean"] for f in bl_files]
            bl_mse = np.mean(bl_vals)

    print(f"\n{'Condition':<30} {'MSE':>10} {'Spatial':>10} {'Gripper':>10} {'vs BL':>10}")
    print("-" * 72)
    for cond_name in ALL_CONDITION_NAMES:
        if cond_name not in all_indices or not all_indices[cond_name]:
            # Check reused baseline
            if cond_name == "baseline" and reuse_baseline:
                bl_dir = config.V2_RESULTS_DIR / "baseline"
                bl_files = sorted(bl_dir.glob("ep*_step*.json"))
                if bl_files:
                    data = [json.loads(f.read_text()) for f in bl_files]
                    avg_mse = np.mean([d["mse_mean"] for d in data])
                    avg_sp = np.mean([d.get("mse_spatial", 0) for d in data])
                    avg_gr = np.mean([d.get("mse_gripper", 0) for d in data])
                    print(f"  {cond_name:<28} {avg_mse:>10.6f} {avg_sp:>10.6f} {avg_gr:>10.6f} {'(reused)':>10}")
            continue
        vals = all_indices[cond_name]
        avg_mse = np.mean([r["mse_mean"] for r in vals])
        avg_sp = np.mean([r["mse_spatial"] for r in vals])
        avg_gr = np.mean([r["mse_gripper"] for r in vals])
        if bl_mse is not None and cond_name != "baseline":
            delta_pct = 100 * (avg_mse - bl_mse) / bl_mse if bl_mse != 0 else 0
            vs_bl = f"{delta_pct:+.1f}%"
        else:
            vs_bl = "-"
        print(f"  {cond_name:<28} {avg_mse:>10.6f} {avg_sp:>10.6f} {avg_gr:>10.6f} {vs_bl:>10}")


def main():
    parser = argparse.ArgumentParser(description="V2 attention enhancement experiment")
    parser.add_argument("--episodes", type=str, default=None,
                        help="Comma-separated episode IDs (default: all)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conditions", type=str, default="all",
                        help="Comma-separated condition names or 'all'")
    parser.add_argument("--reuse_baseline", action="store_true",
                        help="Copy baseline results from V1 instead of re-running")
    parser.add_argument("--list", action="store_true",
                        help="Just list all conditions and exit")
    args = parser.parse_args()

    if args.list:
        print(f"{'Condition':<30} Description")
        print("-" * 70)
        for name in ALL_CONDITION_NAMES:
            print(f"  {name:<28} {describe_condition(name)}")
        return

    episode_ids = (
        [int(x) for x in args.episodes.split(",")]
        if args.episodes else None
    )
    conditions = (
        ALL_CONDITION_NAMES if args.conditions == "all"
        else [c.strip() for c in args.conditions.split(",")]
    )

    # Validate condition names
    invalid = [c for c in conditions if c not in CONDITIONS]
    if invalid:
        print(f"ERROR: Unknown conditions: {invalid}")
        print(f"Valid: {ALL_CONDITION_NAMES}")
        sys.exit(1)

    run_v2_experiment(
        episode_ids=episode_ids,
        device=args.device,
        conditions=conditions,
        reuse_baseline=args.reuse_baseline,
    )


if __name__ == "__main__":
    main()
