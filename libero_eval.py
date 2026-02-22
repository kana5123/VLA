"""LIBERO benchmark evaluation for VLA models with/without adapter.

Runs closed-loop rollouts in LIBERO simulation environments and measures
task success rate — the gold standard metric for VLA evaluation.

Supports multiple conditions:
    - baseline: Raw VLA model (no attention modification)
    - adapter: VLA with trained attention adapter (VAR)
    - fixed-var: VLA with static VAR parameters
    - lora: VLA with LoRA fine-tuned weights

Usage:
    # Evaluate baseline on a single suite
    MUJOCO_GL=egl python libero_eval.py --model openvla-7b --suite libero_spatial --baseline_only

    # Evaluate adapter vs baseline
    MUJOCO_GL=egl python libero_eval.py --model openvla-7b --checkpoint best.pt --suite libero_spatial

    # Full benchmark (all 4 suites)
    MUJOCO_GL=egl python libero_eval.py --model openvla-7b --checkpoint best.pt --suite all

Prerequisites:
    bash setup_libero.sh  (installs mujoco, robosuite, libero)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import config
from extract_attention import load_model_from_registry, get_layers
from model_registry import get_model, list_experiment_models

# Lazy imports for LIBERO (may not be installed)
_LIBERO_AVAILABLE = None


def _check_libero():
    global _LIBERO_AVAILABLE
    if _LIBERO_AVAILABLE is None:
        try:
            import libero  # noqa: F401
            import robosuite  # noqa: F401
            _LIBERO_AVAILABLE = True
        except ImportError:
            _LIBERO_AVAILABLE = False
    if not _LIBERO_AVAILABLE:
        raise ImportError(
            "LIBERO is not installed. Run: bash setup_libero.sh\n"
            "LIBERO requires: mujoco, robosuite, libero"
        )


def load_libero_suite(suite_name: str):
    """Load a LIBERO task suite.

    Returns list of (task_name, env_factory) tuples.
    """
    _check_libero()
    from libero.libero import benchmark

    suite = benchmark.get_benchmark(suite_name)
    tasks = []
    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        task_name = task.name
        env_args = suite.get_task_init_kwargs(task_id)
        tasks.append((task_name, task, env_args))
    return tasks, suite


def create_env(task, env_args, image_size: int = 256):
    """Create a LIBERO environment for a task."""
    _check_libero()
    import robosuite as suite
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        **env_args,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="agentview",
        use_camera_obs=True,
        camera_heights=image_size,
        camera_widths=image_size,
        control_freq=20,
    )
    env.reset()
    return env


def get_observation_image(obs: dict, camera: str = "agentview_image") -> Image.Image:
    """Extract RGB image from environment observation."""
    img_array = obs[camera]
    if img_array.dtype != np.uint8:
        img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array)


class LiberoEvaluator:
    """Evaluate VLA model on LIBERO benchmark tasks."""

    def __init__(
        self,
        model_name: str,
        checkpoint_path: str | None = None,
        method: str = "adapter",
        device: str = "cuda",
    ):
        self.device = device
        self.method = method

        # Load model
        print(f"Loading {model_name}...")
        self.processor, self.model, self.model_cfg = load_model_from_registry(
            model_name, device=device,
        )
        self.model.eval()

        # Load adapter/LoRA if needed
        self.adapter = None
        self.adapter_cfg = self.model_cfg.get_adapter_config()

        if method == "adapter" and checkpoint_path:
            self._load_adapter(checkpoint_path)
        elif method == "lora" and checkpoint_path:
            self._load_lora(checkpoint_path)
        elif method == "fixed-var":
            print("Using fixed VAR (no adapter)")
        elif method == "baseline":
            print("Baseline evaluation (no attention modification)")

    def _load_adapter(self, checkpoint_path: str):
        """Load trained attention adapter."""
        from adapter_model import AttentionAdapter, AttentionAdapterV2
        from attention_v3 import (
            V3Context, install_v3_patch, set_v3_context, set_var_differentiable,
        )

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        ckpt_cfg = ckpt.get("config", {})
        version = ckpt_cfg.get("adapter_version", 2)

        if version == 2:
            self.adapter = AttentionAdapterV2(
                hidden_dim=ckpt_cfg.get("hidden_dim", self.adapter_cfg["hidden_dim"]),
                num_target_layers=ckpt_cfg.get("num_target_layers", self.adapter_cfg["num_target_layers"]),
                num_heads=ckpt_cfg.get("num_heads", self.adapter_cfg["num_heads"]),
                vision_tokens=ckpt_cfg.get("vision_tokens", self.adapter_cfg["vision_tokens"]),
            ).to(self.device)
        else:
            self.adapter = AttentionAdapter(
                hidden_dim=ckpt_cfg.get("hidden_dim", self.adapter_cfg["hidden_dim"]),
                num_target_layers=ckpt_cfg.get("num_target_layers", self.adapter_cfg["num_target_layers"]),
                num_heads=ckpt_cfg.get("num_heads", self.adapter_cfg["num_heads"]),
            ).to(self.device)

        self.adapter.load_state_dict(ckpt["adapter_state_dict"])
        self.adapter.eval()
        print(f"Adapter v{version} loaded from {checkpoint_path}")

    def _load_lora(self, checkpoint_path: str):
        """Load PEFT LoRA checkpoint."""
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, checkpoint_path)
        self.model.eval()
        print(f"LoRA model loaded from {checkpoint_path}")

    def predict_action(self, image: Image.Image, instruction: str) -> np.ndarray:
        """Predict a single action from image + instruction.

        Returns unnormalized 7D action array.
        """
        prompt = self.model_cfg.prompt_template.format(instruction=instruction)
        inputs = self.processor(prompt, image, return_tensors="pt").to(self.device)
        if "pixel_values" in inputs and inputs["pixel_values"].dtype != self.model.dtype:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        pixel_values = inputs.get("pixel_values")

        generated_tokens = []
        with torch.no_grad():
            for _ in range(self.model_cfg.action_tokens):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    use_cache=False,
                )
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_tokens.append(next_token.item())

                input_ids = torch.cat([input_ids, next_token], dim=-1)
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones(1, 1, device=self.device, dtype=attention_mask.dtype),
                    ], dim=-1)

        # Detokenize to continuous action
        from extract_attention import detokenize_actions
        result = detokenize_actions(self.model, generated_tokens)
        action = result.get("unnormalized_action")
        if action is None:
            action = np.zeros(7)
        return np.array(action)

    def rollout(
        self,
        env,
        instruction: str,
        max_steps: int = 300,
    ) -> dict:
        """Run a single episode rollout.

        Returns dict with success, total_reward, num_steps.
        """
        obs = env.reset()
        total_reward = 0.0

        for step in range(max_steps):
            image = get_observation_image(obs)
            action = self.predict_action(image, instruction)

            # LIBERO expects (7,) action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
            obs, reward, done, info = env.step(action)
            total_reward += reward

            if done:
                break

        success = env.check_success() if hasattr(env, 'check_success') else (total_reward > 0)
        return {
            "success": bool(success),
            "total_reward": float(total_reward),
            "num_steps": step + 1,
        }

    def evaluate_suite(
        self,
        suite_name: str,
        num_episodes: int = 20,
        max_steps: int = 300,
        image_size: int = 256,
    ) -> dict:
        """Evaluate on a full LIBERO suite.

        Returns per-task and aggregate success rates.
        """
        tasks, suite = load_libero_suite(suite_name)

        results = {
            "suite": suite_name,
            "num_tasks": len(tasks),
            "num_episodes_per_task": num_episodes,
            "tasks": {},
        }

        all_successes = []

        for task_idx, (task_name, task, env_args) in enumerate(tasks):
            print(f"\n  Task {task_idx + 1}/{len(tasks)}: {task_name}")
            env = create_env(task, env_args, image_size=image_size)

            # Get instruction from task
            instruction = task.language if hasattr(task, 'language') else task_name

            task_successes = []
            for ep in range(num_episodes):
                result = self.rollout(env, instruction, max_steps=max_steps)
                task_successes.append(result["success"])
                if (ep + 1) % 5 == 0:
                    sr = np.mean(task_successes) * 100
                    print(f"    Episode {ep + 1}/{num_episodes}: SR = {sr:.1f}%")

            success_rate = np.mean(task_successes)
            results["tasks"][task_name] = {
                "success_rate": float(success_rate),
                "num_successes": int(sum(task_successes)),
                "num_episodes": num_episodes,
            }
            all_successes.extend(task_successes)

            env.close()
            print(f"  {task_name}: {success_rate * 100:.1f}%")

        results["aggregate_success_rate"] = float(np.mean(all_successes))
        print(f"\n  Suite {suite_name} aggregate: {results['aggregate_success_rate'] * 100:.1f}%")
        return results


def main():
    parser = argparse.ArgumentParser(description="LIBERO benchmark evaluation")
    parser.add_argument("--model", type=str, default="openvla-7b",
                        choices=list_experiment_models())
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Adapter or LoRA checkpoint path")
    parser.add_argument("--method", type=str, default="adapter",
                        choices=["baseline", "adapter", "fixed-var", "lora"])
    parser.add_argument("--suite", type=str, default="libero_spatial",
                        choices=config.LIBERO_SUITES + ["all"],
                        help="LIBERO suite to evaluate")
    parser.add_argument("--num_episodes", type=int, default=config.LIBERO_EPISODES_PER_TASK)
    parser.add_argument("--max_steps", type=int, default=config.LIBERO_MAX_STEPS)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--baseline_only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _check_libero()

    from accelerate.utils import set_seed
    set_seed(args.seed)

    method = "baseline" if args.baseline_only else args.method
    evaluator = LiberoEvaluator(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        method=method,
        device=args.device,
    )

    suites = config.LIBERO_SUITES if args.suite == "all" else [args.suite]

    all_results = {
        "model": args.model,
        "method": method,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "suites": {},
    }

    t0 = time.time()
    for suite_name in suites:
        print(f"\n{'=' * 60}")
        print(f"  LIBERO Suite: {suite_name}")
        print(f"{'=' * 60}")

        suite_results = evaluator.evaluate_suite(
            suite_name,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
        )
        all_results["suites"][suite_name] = suite_results

    total_time = time.time() - t0
    all_results["total_time_s"] = total_time

    # Compute overall success rate across all suites
    all_task_rates = []
    for suite_res in all_results["suites"].values():
        for task_res in suite_res["tasks"].values():
            all_task_rates.append(task_res["success_rate"])
    all_results["overall_success_rate"] = float(np.mean(all_task_rates)) if all_task_rates else 0.0

    # Save results
    out_dir = Path(args.output_dir) if args.output_dir else config.LIBERO_RESULTS_DIR / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"libero_{method}_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'#' * 60}")
    print(f"  LIBERO Evaluation Complete")
    print(f"  Model: {args.model}, Method: {method}")
    print(f"  Overall Success Rate: {all_results['overall_success_rate'] * 100:.1f}%")
    print(f"  Time: {total_time / 60:.1f} min")
    print(f"  Results: {out_path}")
    for suite_name, suite_res in all_results["suites"].items():
        print(f"    {suite_name}: {suite_res['aggregate_success_rate'] * 100:.1f}%")
    print(f"{'#' * 60}")


if __name__ == "__main__":
    main()
