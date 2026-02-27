"""SimplerEnv Downstream Evaluation — Task 3.

Orchestrates SimplerEnv simulation evaluations for VLA models (OpenVLA, SpatialVLA)
by launching the existing bash-script infrastructure via subprocess, then parsing
success/failure video files to compute task success rates.

SimplerEnv runs in the `simpler` conda env, not our `interp` env, so we shell out
to the SimplerEnv evaluation scripts rather than importing their policy classes.

Usage:
    python run_simplerenv_eval.py --model openvla --task pick_coke_can --device 0
    python run_simplerenv_eval.py --model openvla --task all --device 0
    python run_simplerenv_eval.py --model spatialvla --task all --device 1
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

SIMPLERENV_DIR = Path("/home/kana5123/capston/external/SimplerEnv-OpenVLA")
SIMPLERENV_PYTHON = "/home/kana5123/miniconda3/envs/simpler/bin/python"

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase3_gate" / "simplerenv"

# ── Model checkpoints ───────────────────────────────────────────────────────

MODEL_CKPTS = {
    "openvla": "openvla/openvla-7b",
    "spatialvla": "IPEC-COMMUNITY/spatialvla-4b-224-pt",
}

# Action ensemble temperature (negative = enabled, magnitude is temperature).
# SimplerEnv convention: pass as a string argument to the shell scripts.
ACTION_ENSEMBLE_TEMP = "-0.8"

# ── Evaluation task definitions ──────────────────────────────────────────────

EVAL_TASKS = {
    "pick_coke_can": {
        "script": "pick_coke_can_visual_matching.sh",
        "skill": "pick",
        "description": "Pick up coke can (visual matching, 4 URDF x 3 orientations)",
    },
    "move_near": {
        "script": "move_near_visual_matching.sh",
        "skill": "move",
        "description": "Move object near target (visual matching)",
    },
    "open_drawer": {
        "script": "drawer_visual_matching.sh",
        "skill": "open/close",
        "description": "Open/close drawer (visual matching, ray-tracing)",
    },
    "bridge_tasks": {
        "script": "bridge.sh",
        "skill": "bridge",
        "description": "WidowX bridge tasks (spoon, carrot, stack, eggplant)",
    },
}


# ── Result parsing ───────────────────────────────────────────────────────────


def parse_results(logging_dir: str) -> dict:
    """Walk *logging_dir* recursively and count success/failure .mp4 video files.

    SimplerEnv names episode recordings like:
        success_obj_<x>_<y>.mp4   or   success_obj_episode_<id>.mp4
        failure_obj_<x>_<y>.mp4   or   failure_obj_episode_<id>.mp4

    The file-stem prefix (before the first underscore-delimited word "obj")
    is always exactly "success" or "failure".
    """
    successes = 0
    failures = 0

    logging_path = Path(logging_dir)
    if not logging_path.exists():
        print(f"  [WARN] Logging directory does not exist: {logging_dir}")
        return {
            "successes": 0,
            "failures": 0,
            "total": 0,
            "success_rate": None,
        }

    for root, _dirs, files in os.walk(logging_dir):
        for f in files:
            if not f.endswith(".mp4"):
                continue
            stem = Path(f).stem
            if stem.startswith("success"):
                successes += 1
            elif stem.startswith("failure"):
                failures += 1

    total = successes + failures
    return {
        "successes": successes,
        "failures": failures,
        "total": total,
        "success_rate": successes / total if total > 0 else None,
    }


# ── Subprocess launcher ─────────────────────────────────────────────────────


def run_simplerenv_task(
    model_name: str,
    task_name: str,
    device_id: str,
    logging_dir: str,
) -> dict:
    """Launch a SimplerEnv evaluation task via its bash script.

    The shell scripts follow a uniform interface:
        bash scripts/<script.sh> <ckpt_path> <policy_model> <ensemble_temp> <logging_dir> <gpu_id>

    Returns a dict with parsed success/failure counts and the process returncode.
    """
    if task_name not in EVAL_TASKS:
        raise ValueError(
            f"Unknown task '{task_name}'. Choose from: {list(EVAL_TASKS.keys())}"
        )
    if model_name not in MODEL_CKPTS:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {list(MODEL_CKPTS.keys())}"
        )

    task_info = EVAL_TASKS[task_name]
    ckpt_path = MODEL_CKPTS[model_name]
    script_path = f"scripts/{task_info['script']}"

    # Ensure the script exists
    full_script_path = SIMPLERENV_DIR / script_path
    if not full_script_path.exists():
        raise FileNotFoundError(
            f"SimplerEnv script not found: {full_script_path}"
        )

    # Build the command
    cmd = [
        "bash",
        script_path,
        ckpt_path,        # $1 — checkpoint path / HF model id
        model_name,        # $2 — policy model name (openvla, spatialvla, ...)
        ACTION_ENSEMBLE_TEMP,  # $3 — action ensemble temperature
        logging_dir,       # $4 — where to write evaluation videos
        device_id,         # $5 — GPU id
    ]

    # Environment: inherit current env, override CUDA_VISIBLE_DEVICES
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device_id)
    # Ensure the simpler conda env python is first on PATH so bare `python`
    # in the scripts resolves to the correct interpreter.
    simpler_bin_dir = str(Path(SIMPLERENV_PYTHON).parent)
    env["PATH"] = simpler_bin_dir + os.pathsep + env.get("PATH", "")

    print(f"\n{'='*70}")
    print(f"  Task:    {task_name} — {task_info['description']}")
    print(f"  Model:   {model_name} (ckpt: {ckpt_path})")
    print(f"  Device:  GPU {device_id}")
    print(f"  LogDir:  {logging_dir}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    start_time = time.time()

    result = subprocess.run(
        cmd,
        cwd=str(SIMPLERENV_DIR),
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    elapsed = time.time() - start_time
    print(f"\n  [{task_name}] Process exited with code {result.returncode} "
          f"({elapsed:.1f}s elapsed)")

    # Parse the results from the logging directory
    task_results = parse_results(logging_dir)
    task_results["returncode"] = result.returncode
    task_results["elapsed_seconds"] = round(elapsed, 1)

    return task_results


# ── Summary printing ─────────────────────────────────────────────────────────


def print_summary_table(model_name: str, all_results: dict):
    """Print a formatted summary table of task results."""
    print(f"\n{'='*70}")
    print(f"  SimplerEnv Evaluation Summary — {model_name}")
    print(f"{'='*70}")
    print(f"  {'Task':<20} {'Success':>8} {'Fail':>8} {'Total':>8} {'Rate':>10}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    overall_succ = 0
    overall_total = 0

    for task_name, res in all_results.items():
        rate_str = (
            f"{res['success_rate']:.3f}" if res["success_rate"] is not None else "N/A"
        )
        print(
            f"  {task_name:<20} {res['successes']:>8} {res['failures']:>8} "
            f"{res['total']:>8} {rate_str:>10}"
        )
        overall_succ += res["successes"]
        overall_total += res["total"]

    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    overall_fail = overall_total - overall_succ
    overall_rate = overall_succ / overall_total if overall_total > 0 else None
    rate_str = f"{overall_rate:.3f}" if overall_rate is not None else "N/A"
    print(
        f"  {'OVERALL':<20} {overall_succ:>8} {overall_fail:>8} "
        f"{overall_total:>8} {rate_str:>10}"
    )
    print(f"{'='*70}\n")


# ── Main entry point ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Run SimplerEnv downstream evaluation for VLA models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_simplerenv_eval.py --model openvla --task pick_coke_can --device 0\n"
            "  python run_simplerenv_eval.py --model openvla --task all --device 0\n"
            "  python run_simplerenv_eval.py --model spatialvla --task all --device 1\n"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_CKPTS.keys()),
        help="Model to evaluate (openvla or spatialvla).",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help=(
            "Task name to run, or 'all' to run every task. "
            f"Choices: {list(EVAL_TASKS.keys()) + ['all']}"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="GPU device id (default: 0).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Base output directory. Defaults to "
            "outputs/phase3_gate/simplerenv/<model_name>/"
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing them.",
    )
    args = parser.parse_args()

    # Determine tasks to run
    if args.task == "all":
        tasks_to_run = list(EVAL_TASKS.keys())
    else:
        if args.task not in EVAL_TASKS:
            parser.error(
                f"Unknown task '{args.task}'. "
                f"Choose from: {list(EVAL_TASKS.keys()) + ['all']}"
            )
        tasks_to_run = [args.task]

    # Set up output directory
    if args.output_dir is not None:
        output_base = Path(args.output_dir)
    else:
        output_base = DEFAULT_OUTPUT_DIR / args.model
    output_base.mkdir(parents=True, exist_ok=True)

    # Validate SimplerEnv directory
    if not SIMPLERENV_DIR.exists():
        print(f"ERROR: SimplerEnv directory not found at {SIMPLERENV_DIR}")
        sys.exit(1)
    if not Path(SIMPLERENV_PYTHON).exists():
        print(
            f"WARNING: SimplerEnv Python interpreter not found at {SIMPLERENV_PYTHON}. "
            "Evaluation may fail if 'simpler' conda env is not set up."
        )

    # Print run configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"SimplerEnv Evaluation — {timestamp}")
    print(f"  Model:      {args.model} ({MODEL_CKPTS[args.model]})")
    print(f"  Tasks:      {tasks_to_run}")
    print(f"  Device:     GPU {args.device}")
    print(f"  Output dir: {output_base}")
    if args.dry_run:
        print("  ** DRY RUN — commands will not be executed **")

    # Run each task
    all_results = {}
    for task_name in tasks_to_run:
        # Each task gets its own logging subdirectory
        task_logging_dir = str(output_base / task_name)
        Path(task_logging_dir).mkdir(parents=True, exist_ok=True)

        if args.dry_run:
            task_info = EVAL_TASKS[task_name]
            ckpt_path = MODEL_CKPTS[args.model]
            cmd = (
                f"bash scripts/{task_info['script']} "
                f"{ckpt_path} {args.model} {ACTION_ENSEMBLE_TEMP} "
                f"{task_logging_dir} {args.device}"
            )
            print(f"\n  [DRY RUN] Would execute in {SIMPLERENV_DIR}:")
            print(f"    {cmd}")

            # In dry-run mode, still attempt to parse any existing results
            all_results[task_name] = parse_results(task_logging_dir)
            continue

        try:
            task_results = run_simplerenv_task(
                model_name=args.model,
                task_name=task_name,
                device_id=args.device,
                logging_dir=task_logging_dir,
            )
            all_results[task_name] = task_results

            sr = task_results["success_rate"]
            sr_str = f"{sr:.3f}" if sr is not None else "N/A"
            print(
                f"  [{task_name}] Result: {task_results['successes']}/{task_results['total']} "
                f"(success rate: {sr_str})"
            )

        except Exception as e:
            print(f"  [ERROR] Task '{task_name}' failed: {e}")
            all_results[task_name] = {
                "successes": 0,
                "failures": 0,
                "total": 0,
                "success_rate": None,
                "error": str(e),
            }

    # Print summary table
    print_summary_table(args.model, all_results)

    # Save results JSON
    output_json = {
        "model": args.model,
        "ckpt": MODEL_CKPTS[args.model],
        "device": args.device,
        "timestamp": timestamp,
        "tasks": all_results,
    }

    results_path = output_base / "results.json"
    with open(results_path, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"Results saved to: {results_path}")

    # Exit with non-zero status if any task had a non-zero returncode
    any_failure = any(
        res.get("returncode", 0) != 0
        for res in all_results.values()
        if isinstance(res.get("returncode"), int)
    )
    if any_failure:
        print("\nWARNING: One or more tasks exited with a non-zero return code.")
        sys.exit(1)


if __name__ == "__main__":
    main()
