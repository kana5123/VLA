"""V3 attention enhancement experiment — research-based, sink-aware methods.

Implements three method families from recent publications:
  1. VAR  (Visual Attention Redistribution, ICLR 2025)
  2. ACT  (Attention Calibration Technique, arXiv 2406.15765)
  3. SPIN (Head Suppression, EMNLP 2025)

Usage:
    python run_v3_experiment.py [--episodes 0,1,2] [--device cuda]
    python run_v3_experiment.py --conditions baseline,var_default,act_default
    python run_v3_experiment.py --reuse_baseline
    python run_v3_experiment.py --list
"""

import argparse
import json
import sys
import time
from typing import Optional

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
from residual_steer import ResidualSteerer


# ======================================================================
# Experimental conditions
# ======================================================================

CONDITIONS = {
    # ── Baseline ──
    "baseline": dict(),

    # ── VAR family ──
    "var_default": dict(
        use_var=True, var_p=0.6, var_rho=0.5,
    ),
    "var_conservative": dict(
        use_var=True, var_p=0.3, var_rho=0.5,
    ),
    "var_strict": dict(
        use_var=True, var_p=0.6, var_rho=0.8,
    ),
    "var_gripex": dict(
        use_var=True, var_p=0.6, var_rho=0.5,
        gripper_exempt=True,
    ),
    "var_object": dict(
        use_var=True, var_p=0.6, var_rho=0.5,
        use_object_redirect=True,
    ),
    "var_obj_gripex": dict(
        use_var=True, var_p=0.6, var_rho=0.5,
        use_object_redirect=True, gripper_exempt=True,
    ),
    "var_bgsup_gripex": dict(
        use_var=True, var_p=0.6, var_rho=0.5,
        use_bg_suppress=True, gripper_exempt=True,
    ),

    # ── ACT family ──
    "act_default": dict(
        use_act=True, act_alpha=5.0, act_beta=0.4,
    ),
    "act_gripex": dict(
        use_act=True, act_alpha=5.0, act_beta=0.4,
        gripper_exempt=True,
    ),

    # ── SPIN family ──
    "spin_k8": dict(
        use_spin=True, spin_top_k=8, spin_suppress_alpha=0.05,
    ),
    "spin_k16": dict(
        use_spin=True, spin_top_k=16, spin_suppress_alpha=0.05,
    ),

    # ── SAM2 Grounded-SAM2 family (precise segmentation) ──
    "sam_var_object": dict(
        use_var=True, var_p=0.6, var_rho=0.5,
        use_object_redirect=True,
        _use_sam2=True,
    ),
    "sam_var_obj_gripex": dict(
        use_var=True, var_p=0.6, var_rho=0.5,
        use_object_redirect=True, gripper_exempt=True,
        _use_sam2=True,
    ),
    "sam_var_bgsup_gripex": dict(
        use_var=True, var_p=0.6, var_rho=0.5,
        use_bg_suppress=True, gripper_exempt=True,
        _use_sam2=True,
    ),

    # ── Phase 1 Quick Wins: standalone methods ──
    "var_decay": dict(
        use_var=True, var_p=0.6, var_rho=0.5,
        var_decay=True,
    ),
    "var_laysel": dict(
        use_var=True, var_p=0.6, var_rho=0.5,
        _layer_selective=True,
    ),
    "var_vt_rebal": dict(
        use_var=True, var_p=0.6, var_rho=0.5,
        use_vt_rebalance=True,
    ),

    # ── Phase 1 + SAM2 object combinations ──
    "sam_var_obj_decay": dict(
        use_var=True, var_p=0.6, var_rho=0.5,
        use_object_redirect=True, var_decay=True,
        _use_sam2=True,
    ),
    "sam_var_obj_laysel": dict(
        use_var=True, var_p=0.6, var_rho=0.5,
        use_object_redirect=True,
        _use_sam2=True, _layer_selective=True,
    ),
    "sam_var_obj_vt": dict(
        use_var=True, var_p=0.6, var_rho=0.5,
        use_object_redirect=True, use_vt_rebalance=True,
        _use_sam2=True,
    ),

    # ── Phase 1 combined: best standalone combo ──
    "sam_var_obj_decay_laysel": dict(
        use_var=True, var_p=0.6, var_rho=0.5,
        use_object_redirect=True, var_decay=True,
        _use_sam2=True, _layer_selective=True,
    ),

    # ── E: Temporal motion-aware attention ──
    "sam_var_obj_temporal": dict(
        use_var=True, var_p=0.6, var_rho=0.5,
        use_object_redirect=True, use_temporal=True,
        _use_sam2=True,
    ),
}

ALL_CONDITION_NAMES = list(CONDITIONS.keys())


# ======================================================================
# Helpers
# ======================================================================

def compute_temporal_patches(
    current_image: Image.Image,
    prev_image: Optional[Image.Image],
    grid_size: int = config.VISION_GRID_SIZE,
    num_vision_tokens: int = 256,
    diff_threshold: float = config.TEMPORAL_DIFF_THRESHOLD,
    min_motion: float = config.TEMPORAL_PATCH_MIN_MOTION,
) -> list[int]:
    """Detect motion between consecutive frames and return patch indices.

    Computes per-pixel grayscale difference, thresholds to get a motion mask,
    then maps to 16x16 grid patches where motion exceeds min_motion fraction.
    """
    if prev_image is None:
        return []

    # Convert to grayscale numpy
    cur = np.array(current_image.convert("L"), dtype=float)
    prev = np.array(prev_image.convert("L"), dtype=float)

    # Resize if different sizes
    if cur.shape != prev.shape:
        prev_pil = prev_image.convert("L").resize(current_image.size)
        prev = np.array(prev_pil, dtype=float)

    # Absolute difference
    diff = np.abs(cur - prev)
    motion_mask = diff > diff_threshold  # bool (H, W)

    H, W = motion_mask.shape
    cell_h = H / grid_size
    cell_w = W / grid_size
    dual = (num_vision_tokens == grid_size * grid_size * 2)
    per_enc = grid_size * grid_size

    patch_set = set()
    for r in range(grid_size):
        for c in range(grid_size):
            r_start = int(r * cell_h)
            r_end = int((r + 1) * cell_h)
            c_start = int(c * cell_w)
            c_end = int((c + 1) * cell_w)

            cell = motion_mask[r_start:r_end, c_start:c_end]
            if cell.size == 0:
                continue

            motion_frac = cell.sum() / cell.size
            if motion_frac >= min_motion:
                idx = r * grid_size + c
                patch_set.add(idx)
                if dual:
                    patch_set.add(idx + per_enc)

    return sorted(patch_set)


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
        p = cond.get("var_p", config.VAR_P)
        rho = cond.get("var_rho", config.VAR_RHO)
        parts.append(f"VAR(p={p},rho={rho})")
    if cond.get("use_object_redirect"):
        parts.append("obj_redirect")
    if cond.get("use_bg_suppress"):
        parts.append(f"bg_sup(g={cond.get('bg_gamma', config.BG_SUPPRESS_GAMMA)})")
    if cond.get("use_act"):
        a = cond.get("act_alpha", config.ACT_SINK_ALPHA)
        b = cond.get("act_beta", config.ACT_SCALE_BETA)
        parts.append(f"ACT(a={a},b={b})")
    if cond.get("use_spin"):
        k = cond.get("spin_top_k", config.SPIN_TOP_K_HEADS)
        sa = cond.get("spin_suppress_alpha", config.SPIN_SUPPRESS_ALPHA)
        parts.append(f"SPIN(k={k},a={sa})")
    if cond.get("var_decay"):
        parts.append("decay")
    if cond.get("use_vt_rebalance"):
        parts.append(f"VTR(f={cond.get('vt_shift_fraction', config.VT_SHIFT_FRACTION)})")
    if cond.get("use_temporal"):
        parts.append("temporal")
    if cond.get("_layer_selective"):
        lo, hi = config.LAYER_SELECTIVE_RANGE
        parts.append(f"L{lo}-{hi-1}")
    if cond.get("gripper_exempt"):
        parts.append("gripex")
    if cond.get("_use_sam2"):
        parts.append("SAM2")
    return " + ".join(parts) if parts else name


def configure_condition(
    ctx: V3Context,
    condition_name: str,
    grounding_result,
    vision_end: int,
) -> None:
    """Configure ctx for a given condition before inference."""
    cond = CONDITIONS[condition_name]

    # Reset
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

    # Object redirect (needs grounding)
    if cond.get("use_object_redirect"):
        ctx.use_object_redirect = True
        ctx.object_patch_indices = patch_indices
        ctx.object_redirect_weight = cond.get(
            "object_redirect_weight", config.VAR_OBJECT_REDIRECT_WEIGHT
        )

    # VAR decay schedule
    if cond.get("var_decay"):
        ctx.var_decay = True

    # Vision-Text Rebalance
    if cond.get("use_vt_rebalance"):
        ctx.use_vt_rebalance = True
        ctx.vt_shift_fraction = cond.get(
            "vt_shift_fraction", config.VT_SHIFT_FRACTION
        )

    # Temporal motion-aware
    if cond.get("use_temporal"):
        ctx.use_temporal = True
        ctx.temporal_boost_weight = cond.get(
            "temporal_boost_weight", config.TEMPORAL_BOOST_WEIGHT
        )

    # Layer-selective
    if cond.get("_layer_selective"):
        lo, hi = config.LAYER_SELECTIVE_RANGE
        ctx.enhancement_layers = set(range(lo, hi))

    # BG suppress (needs grounding)
    if cond.get("use_bg_suppress"):
        ctx.use_bg_suppress = True
        ctx.bg_gamma = cond.get("bg_gamma", config.BG_SUPPRESS_GAMMA)
        ctx.object_patch_indices = patch_indices

    # ACT
    if cond.get("use_act"):
        ctx.use_act = True
        ctx.act_alpha = cond.get("act_alpha", config.ACT_SINK_ALPHA)
        ctx.act_beta = cond.get("act_beta", config.ACT_SCALE_BETA)

    # SPIN
    if cond.get("use_spin"):
        ctx.use_spin = True
        ctx.spin_top_k = cond.get("spin_top_k", config.SPIN_TOP_K_HEADS)
        ctx.spin_suppress_alpha = cond.get(
            "spin_suppress_alpha", config.SPIN_SUPPRESS_ALPHA
        )

    # Gripper exempt
    ctx.gripper_exempt = cond.get("gripper_exempt", False)

    # Activate
    ctx.active = True


def run_inference_step(
    processor, model, image, instruction, boundaries, device, ctx,
):
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

def run_v3_experiment(
    episode_ids=None,
    device="cuda",
    conditions=None,
    reuse_baseline=False,
):
    """Run the full V3 enhancement experiment."""
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

    total_steps_raw = sum(len(ep["steps"]) for ep in episodes)
    # Count valid steps (excluding is_first / is_last dummy actions)
    total_steps = sum(
        1 for ep in episodes for s in ep["steps"]
        if not s.get("is_first", False) and not s.get("is_last", False)
    )
    skipped = total_steps_raw - total_steps

    # Print experiment plan
    print(f"\n{'='*70}")
    print(f"  V3 Enhancement Experiment (Research-based Methods)")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Conditions: {len(conditions)}")
    print(f"  Total steps per condition: {total_steps} (skipped {skipped} dummy steps)")
    print(f"  Total inference runs: {total_steps * len(conditions)}")
    print(f"{'='*70}")
    for c in conditions:
        print(f"  {c:<25} {describe_condition(c)}")
    print(f"{'='*70}\n")

    # Copy baseline from V2 if requested
    if reuse_baseline and "baseline" in conditions:
        v2_bl_dir = config.V2_RESULTS_DIR / "baseline"
        v3_bl_dir = config.V3_RESULTS_DIR / "baseline"
        if v2_bl_dir.exists() and list(v2_bl_dir.glob("ep*_step*.json")):
            import shutil
            v3_bl_dir.mkdir(parents=True, exist_ok=True)
            for f in v2_bl_dir.glob("ep*_step*.json"):
                shutil.copy2(f, v3_bl_dir / f.name)
            idx = v2_bl_dir / "index.json"
            if idx.exists():
                shutil.copy2(idx, v3_bl_dir / "index.json")
            count = len(list(v3_bl_dir.glob("ep*_step*.json")))
            print(f"[v3] Reused V2 baseline ({count} files)")
            conditions = [c for c in conditions if c != "baseline"]
            if not conditions:
                print("Only baseline requested and reused. Done.")
                return

    # Load model
    print("Loading model...")
    processor, model = load_model(device)

    # Detect token boundaries
    first_step = episodes[0]["steps"][0]
    sample_image = Image.open(
        config.PROJECT_ROOT / first_step["image_path"]
    ).convert("RGB")
    boundaries = detect_token_boundaries(
        processor, model, sample_image, first_step["instruction"], device
    )
    vision_end = boundaries["vision_end"]
    print(f"  vision_end={vision_end}")

    # Install V3 monkey-patch
    ctx = V3Context()
    ctx.vision_end = vision_end
    ctx.active = False
    set_v3_context(ctx)
    install_v3_patch(ctx)

    # Attach vision_end to self_attn modules (for any methods that need it)
    if hasattr(model, "language_model"):
        layers = model.language_model.model.layers
    else:
        layers = model.model.layers
    for layer in layers:
        layer.self_attn._atlasvla_vision_end = vision_end

    # Init grounder (needed for object_redirect and bg_suppress conditions)
    need_grounding_bbox = any(
        (CONDITIONS[c].get("use_object_redirect") or CONDITIONS[c].get("use_bg_suppress"))
        and not CONDITIONS[c].get("_use_sam2")
        for c in conditions if c != "baseline"
    )
    need_grounding_sam2 = any(
        CONDITIONS[c].get("_use_sam2")
        for c in conditions if c != "baseline"
    )
    need_grounding = need_grounding_bbox or need_grounding_sam2
    grounder = ObjectGrounder(device=device) if need_grounding else None

    # Check if temporal motion detection is needed
    need_temporal = any(
        CONDITIONS[c].get("use_temporal")
        for c in conditions if c != "baseline"
    )

    # Prepare output dirs
    for cond_name in conditions:
        (config.V3_RESULTS_DIR / cond_name).mkdir(parents=True, exist_ok=True)

    # Main loop
    pbar = tqdm(total=total_steps * len(conditions), desc="V3 experiment")
    all_indices = {c: [] for c in conditions}
    t_start = time.time()

    for ep in episodes:
        ep_id = ep["episode_id"]

        grounding_cache_bbox = {}
        grounding_cache_sam2 = {}
        if need_grounding and grounder is not None:
            if need_grounding_bbox:
                print(f"\n  Grounding episode {ep_id} (bbox)...")
                grounding_cache_bbox = precompute_grounding_for_episode(
                    grounder, ep, boundaries["num_vision_tokens"],
                    use_sam2=False,
                )
            if need_grounding_sam2:
                print(f"\n  Grounding episode {ep_id} (SAM2)...")
                grounding_cache_sam2 = precompute_grounding_for_episode(
                    grounder, ep, boundaries["num_vision_tokens"],
                    use_sam2=True,
                )

        prev_image = None  # for temporal motion detection
        for step in ep["steps"]:
            # Skip steps with dummy zero actions (is_first / is_last)
            if step.get("is_first", False) or step.get("is_last", False):
                pbar.update(len(conditions))
                continue

            step_id = step["step_id"]
            image = Image.open(
                config.PROJECT_ROOT / step["image_path"]
            ).convert("RGB")
            instruction = step["instruction"]
            gt_action = step["action"]

            # Compute temporal motion patches (once per step, shared across conditions)
            temporal_patches = []
            if need_temporal:
                temporal_patches = compute_temporal_patches(
                    image, prev_image,
                    grid_size=config.VISION_GRID_SIZE,
                    num_vision_tokens=boundaries["num_vision_tokens"],
                )
                prev_image = image

            for cond_name in conditions:
                # Pick SAM2 or bbox grounding based on condition
                if CONDITIONS.get(cond_name, {}).get("_use_sam2"):
                    grounding = grounding_cache_sam2.get(step_id)
                else:
                    grounding = grounding_cache_bbox.get(step_id)

                configure_condition(ctx, cond_name, grounding, vision_end)

                # Set temporal patches if this condition uses them
                if ctx.use_temporal:
                    ctx.temporal_patch_indices = temporal_patches

                token_ids = run_inference_step(
                    processor, model, image, instruction,
                    boundaries, device, ctx,
                )

                ctx.active = False

                action_info = detokenize_actions(model, token_ids)
                pred = (
                    action_info["unnormalized_action"]
                    or action_info["normalized_action"]
                )
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
                    is_sam2 = CONDITIONS.get(cond_name, {}).get("_use_sam2", False)
                    result["grounding"] = {
                        "nouns": grounding.nouns,
                        "num_boxes": len(grounding.boxes_xyxy),
                        "num_patches": len(grounding.patch_indices),
                        "patch_coverage": grounding.patch_coverage,
                        "segmentation": "sam2" if is_sam2 else "bbox",
                    }

                if ctx.use_temporal:
                    result["temporal"] = {
                        "num_motion_patches": len(temporal_patches),
                        "motion_coverage": len(temporal_patches) / boundaries["num_vision_tokens"],
                    }

                out_path = (
                    config.V3_RESULTS_DIR / cond_name
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
                pbar.set_postfix(ep=ep_id, step=step_id, cond=cond_name[:15])

    pbar.close()
    elapsed = time.time() - t_start

    # Save index files
    for cond_name in conditions:
        idx_path = config.V3_RESULTS_DIR / cond_name / "index.json"
        with open(idx_path, "w") as f:
            json.dump({
                "condition": cond_name,
                "description": describe_condition(cond_name),
                "config": CONDITIONS[cond_name],
                "token_boundaries": boundaries,
                "results": all_indices[cond_name],
            }, f, indent=2)

    # Cleanup
    uninstall_v3_patch()

    print(f"\n{'='*70}")
    print(f"  V3 experiment complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Results: {config.V3_RESULTS_DIR}/")
    print(f"{'='*70}")

    # Quick summary table
    bl_mse = None
    if "baseline" in all_indices and all_indices["baseline"]:
        bl_mse = np.mean([r["mse_mean"] for r in all_indices["baseline"]])
    elif reuse_baseline:
        bl_dir = config.V3_RESULTS_DIR / "baseline"
        bl_files = sorted(bl_dir.glob("ep*_step*.json"))
        if bl_files:
            bl_vals = [json.loads(f.read_text())["mse_mean"] for f in bl_files]
            bl_mse = np.mean(bl_vals)

    print(f"\n{'Condition':<25} {'MSE':>10} {'Spatial':>10} {'Gripper':>10} {'vs BL':>10}")
    print("-" * 67)
    for cond_name in ALL_CONDITION_NAMES:
        if cond_name not in all_indices or not all_indices[cond_name]:
            if cond_name == "baseline" and reuse_baseline:
                bl_dir = config.V3_RESULTS_DIR / "baseline"
                bl_files = sorted(bl_dir.glob("ep*_step*.json"))
                if bl_files:
                    data = [json.loads(f.read_text()) for f in bl_files]
                    avg_mse = np.mean([d["mse_mean"] for d in data])
                    avg_sp = np.mean([d.get("mse_spatial", 0) for d in data])
                    avg_gr = np.mean([d.get("mse_gripper", 0) for d in data])
                    print(
                        f"  {cond_name:<23} {avg_mse:>10.6f} "
                        f"{avg_sp:>10.6f} {avg_gr:>10.6f} {'(reused)':>10}"
                    )
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
        print(
            f"  {cond_name:<23} {avg_mse:>10.6f} "
            f"{avg_sp:>10.6f} {avg_gr:>10.6f} {vs_bl:>10}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="V3 attention enhancement experiment (research-based methods)",
    )
    parser.add_argument("--episodes", type=str, default=None,
                        help="Comma-separated episode IDs (default: all)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conditions", type=str, default="all",
                        help="Comma-separated condition names or 'all'")
    parser.add_argument("--reuse_baseline", action="store_true",
                        help="Copy baseline results from V2")
    parser.add_argument("--list", action="store_true",
                        help="List all conditions and exit")
    args = parser.parse_args()

    if args.list:
        print(f"{'Condition':<25} Description")
        print("-" * 70)
        for name in ALL_CONDITION_NAMES:
            print(f"  {name:<23} {describe_condition(name)}")
        return

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
        print(f"Valid: {ALL_CONDITION_NAMES}")
        sys.exit(1)

    run_v3_experiment(
        episode_ids=episode_ids,
        device=args.device,
        conditions=conditions,
        reuse_baseline=args.reuse_baseline,
    )


if __name__ == "__main__":
    main()
