"""Run attention enhancement experiment.

Usage:
    python run_enhancement.py [--episodes 0,1,2] [--device cuda] [--method all]
    python run_enhancement.py --method logit_bias --episodes 0
    python run_enhancement.py --skip_baseline

For each (episode, step) and each method:
  1. Optionally apply attention enhancement via monkey-patch
  2. Run OpenVLA autoregressive inference (7 action tokens)
  3. De-tokenize to continuous action values
  4. Compute MSE against ground truth
  5. Save per-step JSON to outputs/enhancement_results/<method>/
"""

import argparse
import json
import sys

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


METHODS = ["baseline", "logit_bias", "weight_rescale", "head_steering"]


def compute_mse(pred: list[float], gt: list[float]) -> dict:
    pred_arr = np.array(pred, dtype=float)
    gt_arr = np.array(gt, dtype=float)
    per_dim = ((pred_arr - gt_arr) ** 2).tolist()
    return {
        "mse_per_dim": per_dim,
        "mse_mean": float(np.mean(per_dim)),
        "mse_per_dim_names": config.ACTION_DIM_NAMES,
    }


def run_inference_step(processor, model, image, instruction, boundaries, device):
    """Run single-step inference, return list of 7 action token IDs."""
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
        for _ in range(config.NUM_ACTION_TOKENS):
            outputs = model(**model_inputs, use_cache=False)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tok = next_token[0].item()
            generated_tokens.append(tok)

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


def run_step_for_method(
    processor, model, image, instruction, gt_action,
    boundaries, device, method, grounding_result=None,
):
    """Run inference for one (episode, step) under one method. Returns result dict."""
    ctx = get_context()

    if method == "baseline":
        ctx.active = False
    else:
        ctx.method = method
        ctx.active = True
        if grounding_result is not None:
            ctx.set_object_patches(grounding_result.patch_indices)
        else:
            ctx.set_object_patches([])

    token_ids = run_inference_step(
        processor, model, image, instruction, boundaries, device
    )

    # Deactivate after inference
    ctx.active = False

    action_info = detokenize_actions(model, token_ids)
    pred = action_info["unnormalized_action"] or action_info["normalized_action"]
    mse = compute_mse(pred, gt_action)

    result = {
        "predicted_action": pred,
        "ground_truth_action": gt_action,
        "action_token_ids": token_ids,
        **mse,
    }

    if grounding_result is not None and method != "baseline":
        result["grounding"] = {
            "nouns": grounding_result.nouns,
            "num_boxes": len(grounding_result.boxes_xyxy),
            "num_patches": len(grounding_result.patch_indices),
            "patch_indices": grounding_result.patch_indices[:20],  # truncate for readability
        }

    return result


def run_experiment(
    episode_ids=None, device="cuda", methods=None, skip_baseline=False,
):
    """Run the full enhancement experiment."""
    if methods is None:
        methods = METHODS

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

    # Load model
    processor, model = load_model(device)

    # Detect token boundaries
    first_step = episodes[0]["steps"][0]
    sample_image = Image.open(config.PROJECT_ROOT / first_step["image_path"]).convert("RGB")
    boundaries = detect_token_boundaries(
        processor, model, sample_image, first_step["instruction"], device
    )

    # Install monkey-patch
    ctx = get_context()
    ctx.active = False
    if config.ENHANCEMENT_LAYERS is not None:
        ctx.enhancement_layers = set(config.ENHANCEMENT_LAYERS)
    install_patch(ctx)
    inject_vision_end(model, boundaries["vision_end"])

    # Init grounder if needed
    need_grounding = any(m != "baseline" for m in methods)
    grounder = ObjectGrounder(device=device) if need_grounding else None

    # Prepare output dirs
    for method in methods:
        (config.ENHANCEMENT_RESULTS_DIR / method).mkdir(parents=True, exist_ok=True)

    # Main loop
    total_steps = sum(len(ep["steps"]) for ep in episodes)
    pbar = tqdm(total=total_steps * len(methods), desc="Enhancement experiment")
    all_indices = {m: [] for m in methods}

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

            for method in methods:
                if method == "baseline" and skip_baseline:
                    pbar.update(1)
                    continue

                result = run_step_for_method(
                    processor, model, image, instruction, gt_action,
                    boundaries, device, method, grounding,
                )
                result.update({
                    "episode_id": ep_id,
                    "step_id": step_id,
                    "instruction": instruction,
                    "method": method,
                })

                out_path = (
                    config.ENHANCEMENT_RESULTS_DIR / method
                    / f"ep{ep_id:03d}_step{step_id:03d}.json"
                )
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)

                all_indices[method].append({
                    "episode_id": ep_id,
                    "step_id": step_id,
                    "mse_mean": result["mse_mean"],
                    "output_path": str(out_path.relative_to(config.PROJECT_ROOT)),
                })

                pbar.update(1)
                pbar.set_postfix(ep=ep_id, step=step_id, method=method[:8])

    pbar.close()

    # Save index files
    for method in methods:
        if method == "baseline" and skip_baseline:
            continue
        idx_path = config.ENHANCEMENT_RESULTS_DIR / method / "index.json"
        with open(idx_path, "w") as f:
            json.dump({
                "method": method,
                "token_boundaries": boundaries,
                "results": all_indices[method],
            }, f, indent=2)

    uninstall_patch()
    print(f"\nExperiment complete. Results: {config.ENHANCEMENT_RESULTS_DIR}/")


def main():
    parser = argparse.ArgumentParser(description="Run attention enhancement experiment")
    parser.add_argument("--episodes", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--method", type=str, default="all",
        help="Comma-separated methods or 'all' (baseline,logit_bias,weight_rescale,head_steering)",
    )
    parser.add_argument("--skip_baseline", action="store_true")
    args = parser.parse_args()

    episode_ids = (
        [int(x) for x in args.episodes.split(",")]
        if args.episodes else None
    )
    methods = (
        METHODS if args.method == "all"
        else [m.strip() for m in args.method.split(",")]
    )
    run_experiment(
        episode_ids=episode_ids, device=args.device,
        methods=methods, skip_baseline=args.skip_baseline,
    )


if __name__ == "__main__":
    main()
