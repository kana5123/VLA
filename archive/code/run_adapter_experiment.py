"""Run the full adapter experiment: train + eval for base vs v2-full.

Configurations:
    base    — No adapter, raw VLA baseline (eval only)
    v2-full — AttentionAdapterV2, object-centric dynamic adapter

Supports multiple VLA models via --model flag (openvla-7b, ecot-7b, etc.).

Usage:
    python run_adapter_experiment.py --model openvla-7b
    python run_adapter_experiment.py --model spatialvla-4b --gpus 5,6
    python run_adapter_experiment.py --skip_training   # eval only (checkpoints must exist)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import config
from model_registry import list_experiment_models

# ── Experiment Configurations ────────────────────────────────────────────

EXPERIMENT_DIR = config.OUTPUT_DIR / "experiment_results"

CONFIGS = {
    # ── Baselines (eval-only, no training) ──
    "base": {
        "skip_training": True,
        "adapter_version": None,
        "description": "Raw VLA baseline (no adapter)",
    },
    "fixed-var": {
        "skip_training": True,
        "adapter_version": None,
        "eval_method": "fixed-var",
        "description": "Static VAR with hand-tuned parameters (ICLR 2025 method)",
    },
    "act": {
        "skip_training": True,
        "adapter_version": None,
        "eval_method": "act",
        "description": "ACT sink scaling baseline (arXiv 2406.15765)",
    },
    "random": {
        "adapter_version": 2,
        "freeze_blend": False,
        "skip_training": True,
        "random_init": True,
        "description": "Randomly initialized adapter (no training, proves learning matters)",
    },
    "lora": {
        "adapter_version": None,
        "use_lora": True,
        "description": "LoRA fine-tuning baseline (q_proj + v_proj, rank 16)",
    },
    # ── Ablations (optional, for internal analysis only) ──
    "v1": {
        "adapter_version": 1,
        "description": "[Ablation] AttentionAdapter V1, MLP-only per-head p-matrix (no object mask)",
    },
    "v2-prop": {
        "adapter_version": 2,
        "freeze_blend": True,
        "description": "[Ablation] V2 with blend frozen (proportional redistribution only)",
    },
    # ── Main methods (these are what we train and compare) ──
    "v2-full": {
        "adapter_version": 2,
        "freeze_blend": False,
        "description": "AttentionAdapterV2, object-centric dynamic adapter (OUR METHOD)",
    },
    "lora+adapter": {
        "adapter_version": 2,
        "freeze_blend": False,
        "use_lora": True,
        "joint_training": True,
        "description": "LoRA + AttentionAdapterV2 joint training (OUR METHOD + LoRA)",
    },
}


def _create_random_checkpoint(ckpt_path: Path, model_name: str):
    """Create a random-initialized adapter checkpoint for the 'random' baseline."""
    import torch
    from model_registry import get_model
    from adapter_model import AttentionAdapterV2

    model_cfg = get_model(model_name)
    adapter_cfg = model_cfg.get_adapter_config()

    adapter = AttentionAdapterV2(
        hidden_dim=adapter_cfg["hidden_dim"],
        num_target_layers=adapter_cfg["num_target_layers"],
        num_heads=adapter_cfg["num_heads"],
        vision_tokens=adapter_cfg["vision_tokens"],
    )

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "adapter_state_dict": adapter.state_dict(),
        "global_step": 0,
        "best_val_loss": float("inf"),
        "patience_counter": 0,
        "config": {
            "model_name": model_name,
            "architecture": model_cfg.architecture,
            "hidden_dim": adapter_cfg["hidden_dim"],
            "num_heads": adapter_cfg["num_heads"],
            "num_target_layers": adapter_cfg["num_target_layers"],
            "target_layers": adapter_cfg["target_layers"],
            "source_layer": adapter_cfg["source_layer"],
            "vision_tokens": adapter_cfg["vision_tokens"],
            "action_type": adapter_cfg["action_type"],
            "adapter_version": 2,
            "random_init": True,
        },
    }, ckpt_path)
    print(f"  Random checkpoint saved: {ckpt_path}")


def run_command(cmd: list[str], description: str, log_path: Path | None = None) -> int:
    """Run a command, streaming output to both stdout and optional log file."""
    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as log_f:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            for line in proc.stdout:
                sys.stdout.write(line)
                log_f.write(line)
            proc.wait()
            return proc.returncode
    else:
        return subprocess.call(cmd)


def train_config(
    name: str, cfg: dict, gpus: str, num_episodes: int | None,
    model_name: str = "openvla-7b", seed: int = 42,
) -> bool:
    """Train a single adapter configuration. Returns True on success."""
    if cfg.get("skip_training"):
        print(f"[{name}] Skipping training (eval-only config)")
        return True

    output_dir = EXPERIMENT_DIR / model_name / name
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_list = gpus.split(",")
    n_gpus = len(gpu_list)

    # Choose training script
    if cfg.get("joint_training"):
        train_script = "adapter_train.py"  # adapter_train handles LoRA+adapter jointly
    elif cfg.get("use_lora"):
        train_script = "lora_train.py"
    else:
        train_script = "adapter_train.py"

    if n_gpus > 1:
        cmd = [
            sys.executable, "-m", "accelerate.commands.launch",
            "--num_processes", str(n_gpus),
            train_script,
        ]
    else:
        cmd = [sys.executable, train_script]

    cmd.extend(["--model", model_name, "--output_dir", str(output_dir)])

    if cfg.get("joint_training"):
        # Joint LoRA + Adapter: pass both flags to adapter_train.py
        cmd.extend(["--adapter_version", str(cfg["adapter_version"])])
        cmd.append("--use_lora")
        if cfg.get("freeze_blend"):
            cmd.append("--freeze_blend")
    elif not cfg.get("use_lora"):
        cmd.extend(["--adapter_version", str(cfg["adapter_version"])])
        if cfg.get("freeze_blend"):
            cmd.append("--freeze_blend")

    cmd.extend(["--seed", str(seed)])

    if num_episodes is not None:
        cmd.extend(["--num_episodes", str(num_episodes)])

    env_prefix = f"CUDA_VISIBLE_DEVICES={gpus}"
    full_cmd = ["env", f"CUDA_VISIBLE_DEVICES={gpus}"] + cmd

    log_path = output_dir / "train.log"
    t0 = time.time()
    rc = run_command(full_cmd, f"Training [{name}]: {cfg['description']}", log_path)
    elapsed = time.time() - t0

    if rc != 0:
        print(f"[{name}] Training FAILED (exit code {rc}) after {elapsed:.0f}s")
        return False

    print(f"[{name}] Training completed in {elapsed:.0f}s")
    return True


def eval_config(
    name: str, cfg: dict, device: str, num_eval_episodes: int,
    eval_gpu: str = "1", model_name: str = "openvla-7b",
) -> bool:
    """Evaluate a single configuration. Returns True on success."""
    output_dir = EXPERIMENT_DIR / model_name / name
    eval_dir = output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    eval_method = cfg.get("eval_method")

    if eval_method in ("fixed-var", "act"):
        # Static attention method (no checkpoint needed)
        print(f"[{name}] Running {eval_method} evaluation...")
        cmd = [
            sys.executable, "adapter_eval.py",
            "--model", model_name,
            "--device", device,
            "--num_episodes", str(num_eval_episodes),
            "--output_dir", str(eval_dir),
            "--method", eval_method,
        ]
    elif cfg.get("random_init"):
        # Random adapter: create random checkpoint on-the-fly, then eval
        print(f"[{name}] Creating random adapter checkpoint and evaluating...")
        random_ckpt = output_dir / "checkpoints" / "random.pt"
        _create_random_checkpoint(random_ckpt, model_name)
        cmd = [
            sys.executable, "adapter_eval.py",
            "--model", model_name,
            "--checkpoint", str(random_ckpt),
            "--device", device,
            "--num_episodes", str(num_eval_episodes),
            "--output_dir", str(eval_dir),
        ]
    elif cfg.get("skip_training") and not eval_method:
        # Base config: raw baseline (no adapter, no attention method)
        print(f"[{name}] Running baseline-only evaluation...")
        cmd = [
            sys.executable, "adapter_eval.py",
            "--model", model_name,
            "--checkpoint", "NONE",  # special marker
            "--device", device,
            "--num_episodes", str(num_eval_episodes),
            "--output_dir", str(eval_dir),
            "--baseline_only",
        ]
    elif cfg.get("joint_training"):
        # Joint LoRA + Adapter: need both LoRA dir and adapter .pt checkpoint
        ckpt_dir = output_dir / "checkpoints"
        adapter_ckpt = ckpt_dir / "best.pt"
        if not adapter_ckpt.exists():
            adapter_ckpt = ckpt_dir / "final.pt"
        lora_ckpt = ckpt_dir / "lora_best"
        if not lora_ckpt.exists():
            lora_ckpt = ckpt_dir / "lora_final"
        if not adapter_ckpt.exists() or not lora_ckpt.exists():
            print(f"[{name}] ERROR: Missing LoRA+Adapter checkpoints in {ckpt_dir}")
            return False

        cmd = [
            sys.executable, "adapter_eval.py",
            "--model", model_name,
            "--checkpoint", str(adapter_ckpt),
            "--lora_checkpoint", str(lora_ckpt),
            "--device", device,
            "--num_episodes", str(num_eval_episodes),
            "--output_dir", str(eval_dir),
        ]
    elif cfg.get("use_lora"):
        # LoRA: PEFT saves as directories, use lora_eval for these
        ckpt_dir = output_dir / "checkpoints"
        best_ckpt = ckpt_dir / "best"
        if not best_ckpt.exists():
            best_ckpt = ckpt_dir / "final"
        if not best_ckpt.exists():
            print(f"[{name}] ERROR: No LoRA checkpoint found in {ckpt_dir}")
            return False

        cmd = [
            sys.executable, "adapter_eval.py",
            "--model", model_name,
            "--checkpoint", str(best_ckpt),
            "--device", device,
            "--num_episodes", str(num_eval_episodes),
            "--output_dir", str(eval_dir),
            "--method", "lora",
        ]
    else:
        # Find best checkpoint (adapter .pt format)
        ckpt_dir = output_dir / "checkpoints"
        best_ckpt = ckpt_dir / "best.pt"
        if not best_ckpt.exists():
            best_ckpt = ckpt_dir / "final.pt"
        if not best_ckpt.exists():
            print(f"[{name}] ERROR: No checkpoint found in {ckpt_dir}")
            return False

        cmd = [
            sys.executable, "adapter_eval.py",
            "--model", model_name,
            "--checkpoint", str(best_ckpt),
            "--device", device,
            "--num_episodes", str(num_eval_episodes),
            "--output_dir", str(eval_dir),
        ]

    # Set CUDA_VISIBLE_DEVICES for eval to use the correct GPU
    full_cmd = ["env", f"CUDA_VISIBLE_DEVICES={eval_gpu}"] + cmd

    log_path = output_dir / "eval.log"
    t0 = time.time()
    rc = run_command(full_cmd, f"Evaluating [{name}]: {cfg['description']}", log_path)
    elapsed = time.time() - t0

    if rc != 0:
        print(f"[{name}] Evaluation FAILED (exit code {rc}) after {elapsed:.0f}s")
        return False

    print(f"[{name}] Evaluation completed in {elapsed:.0f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run full adapter experiment")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()),
                        choices=list(CONFIGS.keys()),
                        help="Which configs to run (default: all)")
    parser.add_argument("--gpus", type=str, default="1,2,3,4",
                        help="GPU IDs for training (comma-separated)")
    parser.add_argument("--eval_gpu", type=str, default=None,
                        help="GPU ID for evaluation (default: first GPU in --gpus)")
    parser.add_argument("--eval_device", type=str, default="cuda:0",
                        help="Device for evaluation (within CUDA_VISIBLE_DEVICES scope)")
    parser.add_argument("--num_episodes", type=int, default=None,
                        help="Training episodes (None=all)")
    parser.add_argument("--num_eval_episodes", type=int, default=200,
                        help="Evaluation episodes")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training, eval only (checkpoints must exist)")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip evaluation, training only")
    parser.add_argument("--model", type=str, default="openvla-7b",
                        choices=list_experiment_models(),
                        help="VLA model name from model_registry")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Multiple seeds for multi-seed runs (e.g. --seeds 42 123 456)")
    args = parser.parse_args()

    # Resolve seed list: --seeds takes precedence over --seed
    seed_list = args.seeds if args.seeds else [args.seed]

    print(f"\n{'#' * 60}")
    print(f"  ADAPTER EXPERIMENT PIPELINE")
    print(f"  Model: {args.model}")
    print(f"  Configs: {args.configs}")
    print(f"  Seeds: {seed_list}")
    print(f"  GPUs (train): {args.gpus}")
    print(f"  Eval device: {args.eval_device}")
    print(f"{'#' * 60}\n")

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    multi_seed = len(seed_list) > 1
    results_summary = {}
    t_total = time.time()

    for name in args.configs:
        cfg = CONFIGS[name]

        for seed in seed_list:
            # Key includes seed suffix for multi-seed runs
            run_key = f"{name}/seed_{seed}" if multi_seed else name
            # Output subdirectory includes seed for multi-seed
            run_subdir = f"{name}/seed_{seed}" if multi_seed else name

            print(f"\n{'*' * 60}")
            print(f"  CONFIG: {run_key} — {cfg['description']} (seed={seed})")
            print(f"{'*' * 60}")

            # Training phase
            if not args.skip_training:
                ok = train_config(
                    run_subdir, cfg, args.gpus, args.num_episodes,
                    model_name=args.model, seed=seed,
                )
                if not ok and not cfg.get("skip_training"):
                    results_summary[run_key] = {"status": "train_failed", "seed": seed}
                    continue

            # Evaluation phase
            eval_gpu = args.eval_gpu or args.gpus.split(",")[0]
            if not args.skip_eval:
                ok = eval_config(
                    run_subdir, cfg, args.eval_device, args.num_eval_episodes,
                    eval_gpu=eval_gpu, model_name=args.model,
                )
                if ok:
                    eval_path = EXPERIMENT_DIR / args.model / run_subdir / "eval" / "eval_results.json"
                    if eval_path.exists():
                        results_summary[run_key] = json.loads(eval_path.read_text())
                        results_summary[run_key]["status"] = "complete"
                        results_summary[run_key]["seed"] = seed
                    else:
                        results_summary[run_key] = {"status": "eval_no_output", "seed": seed}
                else:
                    results_summary[run_key] = {"status": "eval_failed", "seed": seed}
            else:
                results_summary[run_key] = {"status": "train_only", "seed": seed}

    total_time = time.time() - t_total

    # Save combined summary
    summary_dir = EXPERIMENT_DIR / args.model
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "model": args.model,
            "configs_run": args.configs,
            "seeds": seed_list,
            "total_time_s": total_time,
            "results": results_summary,
        }, f, indent=2)

    print(f"\n{'#' * 60}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"  Total time: {total_time / 60:.1f} min")
    print(f"  Summary: {summary_path}")
    for run_key, res in results_summary.items():
        status = res.get("status", "unknown")
        mse = ""
        if "comparison" in res:
            pct = res["comparison"].get("overall_change_pct", 0)
            mse = f" | MSE change: {pct:+.2f}%"
        print(f"    {run_key:25s}: {status}{mse}")
    print(f"{'#' * 60}\n")


if __name__ == "__main__":
    main()
