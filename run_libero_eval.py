"""LIBERO downstream evaluation for adaptive routing interventions.

Measures closed-loop task success rate with and without attention hooks
(VARValueHook, KeyScaleHook) to validate that D2 improvements translate
to actual downstream performance gains.

Usage:
    MUJOCO_GL=egl python run_libero_eval.py --model openvla-7b --device cuda:0 \
        --suite libero_spatial --num_episodes 10 --intervention var --var_p 0.9

Conditions tested:
    - baseline: No intervention
    - var: VARValueHook (redistributes sink token value)
    - kscale: KeyScaleHook (scales down anchor key projection)
    - lora: Entropy-regularized LoRA adapter (training-time fix)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Force unbuffered output for nohup/background execution
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None

# Must set before LIBERO imports
os.environ.setdefault("MUJOCO_GL", "egl")

import config
from extract_attention import (
    load_model_from_registry,
    get_layers,
    detect_token_boundaries,
    detokenize_actions,
)
from model_registry import get_model, list_experiment_models
from run_phase3_exp_de import KeyScaleHook, detect_anchor_targets
from run_var_baseline import VARValueHook, SINK_DIMENSIONS


# ── LIBERO helpers ───────────────────────────────────────────────────

def load_libero_suite(suite_name: str):
    from libero.libero import benchmark
    SuiteClass = benchmark.get_benchmark(suite_name)
    suite = SuiteClass()
    tasks = []
    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        bddl_path = suite.get_task_bddl_file_path(task_id)
        init_states = suite.get_task_init_states(task_id)
        tasks.append((task.name, task, bddl_path, init_states))
    return tasks, suite


def create_env(bddl_path: str, init_states: np.ndarray,
               episode_idx: int = 0, image_size: int = 256,
               render_gpu_device_id: int = -1):
    from libero.libero.envs import OffScreenRenderEnv
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="agentview",
        use_camera_obs=True,
        camera_heights=image_size,
        camera_widths=image_size,
        render_gpu_device_id=render_gpu_device_id,
    )
    env.seed(0)
    # Set initial state from demonstration
    state_idx = episode_idx % len(init_states)
    env.set_init_state(init_states[state_idx])
    env.reset()
    return env


def obs_to_image(obs: dict) -> Image.Image:
    img = obs["agentview_image"]
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)


# ── Hook management ──────────────────────────────────────────────────

def install_hooks(model, model_cfg, intervention: str, params: dict,
                  anchor_targets: list, deep_layers: list, bounds: dict):
    """Install VARValueHook or KeyScaleHook based on intervention type.

    Returns list of installed hooks (for cleanup).
    """
    hooks = []
    arch = model_cfg.architecture
    vs = bounds["vision_start"]
    ve = bounds["vision_end"]

    if intervention == "var":
        sink_dims = SINK_DIMENSIONS.get(arch, [])
        if not sink_dims:
            print(f"  WARNING: No sink dims for {arch}, VAR may be ineffective")
        # Anchor position (absolute index in sequence)
        anchor_abs = anchor_targets[0]["target_abs"] if anchor_targets else vs
        hook = VARValueHook(
            sink_positions_abs=[anchor_abs],
            vision_start=vs,
            vision_end=ve,
            p=params.get("p", 0.6),
            target_layers=deep_layers,
        )
        hook.register(model, model_cfg, get_layers)
        hooks.append(hook)

    elif intervention == "kscale":
        anchor_abs = anchor_targets[0]["target_abs"] if anchor_targets else vs
        hook = KeyScaleHook(
            target_positions=[anchor_abs],
            alpha=params.get("alpha", 0.0),
            target_layers=deep_layers,
        )
        hook.register(model, model_cfg, get_layers)
        hooks.append(hook)

    elif intervention == "hybrid":
        # Both VAR + K-scale
        anchor_abs = anchor_targets[0]["target_abs"] if anchor_targets else vs
        sink_dims = SINK_DIMENSIONS.get(arch, [])
        var_hook = VARValueHook(
            sink_positions_abs=[anchor_abs],
            vision_start=vs,
            vision_end=ve,
            p=params.get("p", 0.6),
            target_layers=deep_layers,
        )
        var_hook.register(model, model_cfg, get_layers)
        hooks.append(var_hook)

        ks_hook = KeyScaleHook(
            target_positions=[anchor_abs],
            alpha=params.get("alpha", 0.3),
            target_layers=deep_layers,
        )
        ks_hook.register(model, model_cfg, get_layers)
        hooks.append(ks_hook)

    return hooks


def remove_hooks(hooks: list):
    for h in hooks:
        h.remove()


# ── Predict action ───────────────────────────────────────────────────

@torch.no_grad()
def predict_action(model, processor, model_cfg, image, instruction, device,
                   use_libero_actions=False):
    """Predict 7D action from image + instruction via greedy autoregressive decoding.

    Args:
        use_libero_actions: If True, use normalized_action directly ([-1,1] range)
            instead of Bridge V2 unnormalization. Required for LIBERO fine-tuned models
            since they were trained on [-1,1] actions directly.
    """
    prompt = model_cfg.prompt_template.format(instruction=instruction)
    inputs = processor(prompt, image, return_tensors="pt").to(device)
    if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs.get("pixel_values")

    generated = []
    for _ in range(model_cfg.action_tokens):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            use_cache=False,
        )
        next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_tok.item())
        input_ids = torch.cat([input_ids, next_tok], dim=-1)
        if attention_mask is not None:
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(1, 1, device=device, dtype=attention_mask.dtype),
            ], dim=-1)

    result = detokenize_actions(model, generated)

    # Free GPU memory to prevent CUDA/EGL rendering contention in MuJoCo
    del outputs, inputs, input_ids, attention_mask, pixel_values, next_tok
    torch.cuda.empty_cache()

    if use_libero_actions:
        # LIBERO fine-tuned models: actions are already in [-1,1], use directly
        action = result.get("normalized_action")
        return np.array(action) if action is not None else np.zeros(7)
    else:
        # Bridge V2 pre-trained models: unnormalize using dataset statistics
        action = result.get("unnormalized_action")
        return np.array(action) if action is not None else np.zeros(7)


# ── Rollout ──────────────────────────────────────────────────────────

def rollout(model, processor, model_cfg, env, instruction, device, max_steps=300,
            use_libero_actions=False):
    obs = env.reset()
    total_reward = 0.0

    for step in range(max_steps):
        image = obs_to_image(obs)
        action = predict_action(model, processor, model_cfg, image, instruction, device,
                                use_libero_actions=use_libero_actions)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        # LIBERO: check_success() returns dict of success conditions
        success_dict = env.check_success()
        task_success = all(success_dict.values()) if isinstance(success_dict, dict) else bool(success_dict)
        if task_success or done:
            break

    return {"success": bool(task_success), "reward": float(total_reward), "steps": step + 1}


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LIBERO eval with adaptive hooks")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--suite", default="libero_spatial",
                        choices=config.LIBERO_SUITES + ["all"])
    parser.add_argument("--num_episodes", type=int, default=10,
                        help="Episodes per task (default: 10)")
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--max_tasks", type=int, default=None,
                        help="Limit number of tasks per suite (for quick testing)")
    parser.add_argument("--intervention", default="baseline",
                        choices=["baseline", "var", "kscale", "hybrid"])
    parser.add_argument("--var_p", type=float, default=0.6)
    parser.add_argument("--kscale_alpha", type=float, default=0.0)
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to LoRA adapter dir")
    parser.add_argument("--libero_actions", action="store_true",
                        help="Use [-1,1] normalized actions directly (for LIBERO FT models)")
    parser.add_argument("--render_gpu", type=int, default=-1,
                        help="GPU device for MuJoCo offscreen rendering (-1=same as model)")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Load model ──
    print(f"\nLoading {args.model}...")
    processor, model, model_cfg = load_model_from_registry(args.model, args.device)
    model.eval()

    # ── Load LoRA if specified ──
    if args.lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_path)
        model.eval()
        print(f"  LoRA loaded from {args.lora_path}")

    # Auto-detect LIBERO actions if lora_path contains libero_ft
    use_libero_actions = args.libero_actions
    if not use_libero_actions and args.lora_path and "libero_ft" in args.lora_path:
        use_libero_actions = True
        print("  Auto-detected LIBERO fine-tuned model → using [-1,1] actions directly")

    # ── Setup anchor detection ──
    verification_dir = config.OUTPUT_DIR / "phase3_gate" / "verification"
    deep_layers = list(range(max(0, model_cfg.num_layers - 10), model_cfg.num_layers))

    # Detect boundaries from dummy image
    dummy_img = Image.new("RGB", (256, 256), (128, 128, 128))
    bounds = detect_token_boundaries(
        processor, model, dummy_img, "pick up the object",
        args.device, model_cfg,
    )
    anchor_targets = detect_anchor_targets(model_cfg, verification_dir, bounds)
    print(f"  Anchor targets: {anchor_targets}")
    print(f"  Deep layers: {deep_layers}")

    # ── Install hooks ──
    hook_params = {"p": args.var_p, "alpha": args.kscale_alpha}
    hooks = install_hooks(
        model, model_cfg, args.intervention, hook_params,
        anchor_targets, deep_layers, bounds,
    )
    if hooks:
        print(f"  Installed {len(hooks)} hooks for intervention={args.intervention}")

    # ── Output dir ──
    out_dir = Path(args.output_dir) if args.output_dir else \
        config.OUTPUT_DIR / "libero_results" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Evaluate ──
    suites = config.LIBERO_SUITES if args.suite == "all" else [args.suite]

    all_results = {
        "model": args.model,
        "intervention": args.intervention,
        "params": hook_params,
        "lora_path": args.lora_path,
        "seed": args.seed,
        "suites": {},
    }

    t0 = time.time()
    for suite_name in suites:
        print(f"\n{'='*60}", flush=True)
        print(f"  LIBERO Suite: {suite_name}", flush=True)
        print(f"  Intervention: {args.intervention} (params: {hook_params})", flush=True)
        print(f"{'='*60}", flush=True)

        print(f"  Loading LIBERO suite '{suite_name}'...", flush=True)
        tasks, suite = load_libero_suite(suite_name)
        print(f"  Loaded {len(tasks)} tasks from {suite_name}", flush=True)
        if args.max_tasks:
            tasks = tasks[:args.max_tasks]

        suite_results = {"tasks": {}, "num_tasks": len(tasks)}
        all_successes = []

        for ti, (task_name, task, bddl_path, init_states) in enumerate(tasks):
            print(f"\n  Task {ti+1}/{len(tasks)}: {task_name}", flush=True)
            instruction = task.language if hasattr(task, 'language') else task_name

            task_successes = []
            for ep in range(args.num_episodes):
                print(f"    Creating env for ep {ep+1}...", end="", flush=True)
                env = create_env(bddl_path, init_states, episode_idx=ep,
                                render_gpu_device_id=args.render_gpu)
                print(f" rolling out...", end="", flush=True)
                result = rollout(
                    model, processor, model_cfg, env,
                    instruction, args.device, args.max_steps,
                    use_libero_actions=use_libero_actions,
                )
                task_successes.append(result["success"])
                env.close()
                sr = np.mean(task_successes) * 100
                print(f" done (steps={result['steps']}, success={result['success']}, SR={sr:.1f}%)", flush=True)

            sr = float(np.mean(task_successes))
            suite_results["tasks"][task_name] = {
                "success_rate": sr,
                "successes": int(sum(task_successes)),
                "episodes": args.num_episodes,
            }
            all_successes.extend(task_successes)
            print(f"  => {task_name}: {sr*100:.1f}%")

        suite_results["aggregate_sr"] = float(np.mean(all_successes)) if all_successes else 0.0
        all_results["suites"][suite_name] = suite_results
        print(f"\n  Suite {suite_name}: {suite_results['aggregate_sr']*100:.1f}%")

    elapsed = time.time() - t0
    all_results["elapsed_s"] = elapsed

    # Overall
    all_srs = []
    for sr in all_results["suites"].values():
        for tr in sr["tasks"].values():
            all_srs.append(tr["success_rate"])
    all_results["overall_sr"] = float(np.mean(all_srs)) if all_srs else 0.0

    # Save
    tag = args.intervention
    if args.lora_path:
        tag += "_lora"
    out_path = out_dir / f"libero_{tag}_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Cleanup
    remove_hooks(hooks)
    del model
    torch.cuda.empty_cache()

    print(f"\n{'#'*60}")
    print(f"  LIBERO Complete: {args.model} [{args.intervention}]")
    print(f"  Overall SR: {all_results['overall_sr']*100:.1f}%")
    print(f"  Time: {elapsed/60:.1f}min")
    print(f"  Saved: {out_path}")
    for sn, sr in all_results["suites"].items():
        print(f"    {sn}: {sr['aggregate_sr']*100:.1f}%")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
