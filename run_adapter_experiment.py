"""Run the full adapter experiment: train + eval for 4 configurations.

Configurations:
    base    — No adapter, raw OpenVLA baseline (eval only)
    v1      — AttentionAdapter (MLP-only, no object masks)
    v2-prop — AttentionAdapterV2, blend_alpha frozen at 0 (proportional redistribution)
    v2-full — AttentionAdapterV2, blend_alpha learnable (SAM masks + learned redistribution)

Usage:
    python run_adapter_experiment.py
    python run_adapter_experiment.py --configs v1 v2-full
    python run_adapter_experiment.py --skip_training   # eval only (checkpoints must exist)
    python run_adapter_experiment.py --gpus 0,1,2,3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import config

# ── Experiment Configurations ────────────────────────────────────────────

EXPERIMENT_DIR = config.OUTPUT_DIR / "experiment_results"

CONFIGS = {
    "base": {
        "skip_training": True,
        "adapter_version": None,
        "description": "Raw OpenVLA baseline (no adapter)",
    },
    "v1": {
        "adapter_version": 1,
        "freeze_blend": False,
        "description": "AttentionAdapter v1 (MLP only)",
    },
    "v2-prop": {
        "adapter_version": 2,
        "freeze_blend": True,
        "description": "AttentionAdapterV2, proportional redistribution (blend frozen)",
    },
    "v2-full": {
        "adapter_version": 2,
        "freeze_blend": False,
        "description": "AttentionAdapterV2, learned redistribution (blend learnable)",
    },
}


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


def train_config(name: str, cfg: dict, gpus: str, num_episodes: int | None) -> bool:
    """Train a single adapter configuration. Returns True on success."""
    if cfg.get("skip_training"):
        print(f"[{name}] Skipping training (eval-only config)")
        return True

    output_dir = EXPERIMENT_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = []
    gpu_list = gpus.split(",")
    n_gpus = len(gpu_list)

    if n_gpus > 1:
        cmd = [
            "accelerate", "launch",
            "--num_processes", str(n_gpus),
            "adapter_train.py",
        ]
    else:
        cmd = ["python", "adapter_train.py"]

    cmd.extend([
        "--adapter_version", str(cfg["adapter_version"]),
        "--output_dir", str(output_dir),
    ])

    if cfg.get("freeze_blend"):
        cmd.append("--freeze_blend")

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
) -> bool:
    """Evaluate a single configuration. Returns True on success."""
    output_dir = EXPERIMENT_DIR / name
    eval_dir = output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    if cfg.get("skip_training"):
        # Base config: run eval without adapter
        # We use adapter_eval.py in a special mode — the base eval doesn't need
        # an adapter checkpoint. We'll handle this by running inference for
        # baseline only (no adapter condition).
        print(f"[{name}] Running baseline-only evaluation...")
        cmd = [
            "python", "adapter_eval.py",
            "--checkpoint", "NONE",  # special marker
            "--device", device,
            "--num_episodes", str(num_eval_episodes),
            "--output_dir", str(eval_dir),
            "--baseline_only",
        ]
    else:
        # Find best checkpoint
        ckpt_dir = output_dir / "checkpoints"
        best_ckpt = ckpt_dir / "best.pt"
        if not best_ckpt.exists():
            # Fallback to final
            best_ckpt = ckpt_dir / "final.pt"
        if not best_ckpt.exists():
            print(f"[{name}] ERROR: No checkpoint found in {ckpt_dir}")
            return False

        cmd = [
            "python", "adapter_eval.py",
            "--checkpoint", str(best_ckpt),
            "--device", device,
            "--num_episodes", str(num_eval_episodes),
            "--output_dir", str(eval_dir),
        ]

    log_path = output_dir / "eval.log"
    t0 = time.time()
    rc = run_command(cmd, f"Evaluating [{name}]: {cfg['description']}", log_path)
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
    parser.add_argument("--gpus", type=str, default="0,1,2,3",
                        help="GPU IDs for training (comma-separated)")
    parser.add_argument("--eval_device", type=str, default="cuda:0",
                        help="Device for evaluation")
    parser.add_argument("--num_episodes", type=int, default=None,
                        help="Training episodes (None=all)")
    parser.add_argument("--num_eval_episodes", type=int, default=200,
                        help="Evaluation episodes")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training, eval only (checkpoints must exist)")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip evaluation, training only")
    args = parser.parse_args()

    print(f"\n{'#' * 60}")
    print(f"  ADAPTER EXPERIMENT PIPELINE")
    print(f"  Configs: {args.configs}")
    print(f"  GPUs (train): {args.gpus}")
    print(f"  Eval device: {args.eval_device}")
    print(f"{'#' * 60}\n")

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    results_summary = {}
    t_total = time.time()

    for name in args.configs:
        cfg = CONFIGS[name]
        print(f"\n{'*' * 60}")
        print(f"  CONFIG: {name} — {cfg['description']}")
        print(f"{'*' * 60}")

        # Training phase
        if not args.skip_training:
            ok = train_config(name, cfg, args.gpus, args.num_episodes)
            if not ok and not cfg.get("skip_training"):
                results_summary[name] = {"status": "train_failed"}
                continue

        # Evaluation phase
        if not args.skip_eval:
            ok = eval_config(name, cfg, args.eval_device, args.num_eval_episodes)
            if ok:
                eval_path = EXPERIMENT_DIR / name / "eval" / "eval_results.json"
                if eval_path.exists():
                    results_summary[name] = json.loads(eval_path.read_text())
                    results_summary[name]["status"] = "complete"
                else:
                    results_summary[name] = {"status": "eval_no_output"}
            else:
                results_summary[name] = {"status": "eval_failed"}
        else:
            results_summary[name] = {"status": "train_only"}

    total_time = time.time() - t_total

    # Save combined summary
    summary_path = EXPERIMENT_DIR / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "configs_run": args.configs,
            "total_time_s": total_time,
            "results": results_summary,
        }, f, indent=2)

    print(f"\n{'#' * 60}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"  Total time: {total_time / 60:.1f} min")
    print(f"  Summary: {summary_path}")
    for name, res in results_summary.items():
        status = res.get("status", "unknown")
        mse = ""
        if "comparison" in res:
            pct = res["comparison"].get("overall_change_pct", 0)
            mse = f" | MSE change: {pct:+.2f}%"
        print(f"    {name:10s}: {status}{mse}")
    print(f"{'#' * 60}\n")


if __name__ == "__main__":
    main()
