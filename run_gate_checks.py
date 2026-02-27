#!/usr/bin/env python3
"""Gate Check Orchestrator: Gate ① → ② + ③ → go/no-go decision.

Execution flow:
  Gate ① (150 balanced, GPU 0-3, ~75min)
    ├── sample_list.json → Gate ②, ③
    ├── mode_tokens.json → Gate ②
    └── probe baselines → Gate ③

  Gate ② (layer-local V=0, GPU 0-3, ~45min)   ← independent
  Gate ③ (text masking, GPU 4-7, ~2hr)         ← independent

  Both complete → pass criteria check → go/no-go

Usage:
  python run_gate_checks.py --gate 1 --models ecot-7b openvla-7b spatialvla-4b tracevla-phi3v
  python run_gate_checks.py --gate 2 --models ecot-7b --gate1_dir outputs/phase3_gate/ecot-7b
  python run_gate_checks.py --gate 3 --models ecot-7b --gate1_dir outputs/phase3_gate/ecot-7b
  python run_gate_checks.py --check_pass --gate1_dir outputs/phase3_gate
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import config


MODELS = ["ecot-7b", "openvla-7b", "spatialvla-4b", "tracevla-phi3v"]
GPU_MAP = {"ecot-7b": "cuda:0", "openvla-7b": "cuda:1", "spatialvla-4b": "cuda:2", "tracevla-phi3v": "cuda:3"}
GPU_MAP_GATE3 = {"ecot-7b": "cuda:4", "openvla-7b": "cuda:5", "spatialvla-4b": "cuda:6", "tracevla-phi3v": "cuda:7"}
TARGET_SKILLS = ["place", "move", "pick", "fold", "open", "close"]


def run_gate1(models, output_base, n_per_skill=25, seed=42):
    """Gate ①: Run contribution analysis with balanced sampling."""
    procs = []
    for model in models:
        device = GPU_MAP.get(model, "cuda:0")
        out_dir = Path(output_base) / model
        sample_list = out_dir / "sample_list.json"
        cmd = [
            sys.executable, "run_contribution_analysis.py",
            "--model", model, "--device", device,
            "--balanced", "--n_per_skill", str(n_per_skill), "--seed", str(seed),
            "--sample_list", str(sample_list),
            "--target_skills", *TARGET_SKILLS,
            "--output_dir", str(out_dir),
        ]
        print(f"  Launching Gate ① for {model} on {device}")
        p = subprocess.Popen(cmd)
        procs.append((model, p))

    # Wait for all
    for model, p in procs:
        rc = p.wait()
        if rc != 0:
            print(f"  ERROR: Gate ① failed for {model} (exit code {rc})")
        else:
            print(f"  Gate ① complete for {model}")


def run_gate2(models, gate1_base, output_base):
    """Gate ②: Layer-local V=0 for each model × peak × block."""
    procs = []
    for model in models:
        device = GPU_MAP.get(model, "cuda:0")
        gate1_dir = Path(gate1_base) / model
        mode_tokens_path = gate1_dir / "mode_tokens.json"
        sample_list_path = gate1_dir / "sample_list.json"

        if not mode_tokens_path.exists():
            print(f"  SKIP Gate ② for {model}: no mode_tokens.json")
            continue

        for peak_type in ["A_mode", "C_mode", "R_mode"]:
            for layer_mode in ["all", "block1", "block2"]:
                out_dir = Path(output_base) / model / f"{peak_type}_{layer_mode}"
                cmd = [
                    sys.executable, "run_causal_experiment.py",
                    "--model", model, "--device", device,
                    "--candidates_json", str(mode_tokens_path),
                    "--peak_type", peak_type,
                    "--layer_mode", layer_mode,
                    "--sample_list", str(sample_list_path),
                    "--n_samples", "20",
                    "--output_dir", str(out_dir),
                ]
                print(f"  Launching Gate ② for {model}/{peak_type}/{layer_mode}")
                p = subprocess.Popen(cmd)
                procs.append((f"{model}/{peak_type}/{layer_mode}", p))

    for name, p in procs:
        rc = p.wait()
        status = "OK" if rc == 0 else f"FAIL (rc={rc})"
        print(f"  Gate ② {name}: {status}")


def run_gate3(models, gate1_base, output_base):
    """Gate 3: Text masking + mini counterfactual on GPUs 4-7."""
    procs = []
    for model in models:
        device = GPU_MAP_GATE3.get(model, "cuda:4")
        gate1_dir = Path(gate1_base) / model
        sample_list = gate1_dir / "sample_list.json"
        if not sample_list.exists():
            print(f"  SKIP Gate 3 for {model}: no sample_list.json")
            continue
        out_dir = Path(output_base) / model
        cmd = [
            sys.executable, "run_gate3_text_mask.py",
            "--model", model, "--device", device,
            "--gate1_dir", str(gate1_dir),
            "--output_dir", str(out_dir),
            "--n_samples", "20",
        ]
        print(f"  Launching Gate 3 for {model} on {device}")
        p = subprocess.Popen(cmd)
        procs.append((model, p))

    for model, p in procs:
        rc = p.wait()
        status = "OK" if rc == 0 else f"FAIL (rc={rc})"
        print(f"  Gate 3 {model}: {status}")


def check_gate1_pass(gate1_base):
    """Check Gate ① pass criteria from design doc."""
    results = {}
    for model in MODELS:
        report_path = Path(gate1_base) / model / "contribution_report.json"
        if not report_path.exists():
            continue
        with open(report_path) as f:
            report = json.load(f)

        mode_path = Path(gate1_base) / model / "mode_tokens.json"
        mode_tokens = {}
        if mode_path.exists():
            with open(mode_path) as f:
                mode_tokens = json.load(f)

        # Extract medians from layer_analysis
        top1_shares = []
        mismatches = []
        entropies = []
        for l_info in report.get("layer_analysis", {}).values():
            if "mean_top1_share" in l_info:
                top1_shares.append(l_info["mean_top1_share"])
            if "mean_mismatch" in l_info:
                mismatches.append(l_info["mean_mismatch"])
            if "mean_entropy" in l_info:
                entropies.append(l_info["mean_entropy"])

        import numpy as np
        median_top1 = float(np.median(top1_shares)) if top1_shares else 0
        median_mismatch = float(np.median(mismatches)) if mismatches else 0
        median_entropy = float(np.median(entropies)) if entropies else 0

        a_mode = mode_tokens.get("A_mode", {}).get("abs_t", -1)
        c_mode = mode_tokens.get("C_mode", {}).get("abs_t", -1)

        # Check model-specific criteria
        passed = True
        reasons = []

        if model == "ecot-7b":
            if median_top1 <= 0.8:
                passed = False
                reasons.append(f"Top1 C_tilde median {median_top1:.3f} <= 0.8")
            if a_mode != c_mode:
                passed = False
                reasons.append(f"A_mode({a_mode}) != C_mode({c_mode})")
        elif model == "openvla-7b":
            if median_mismatch <= 0.15:
                passed = False
                reasons.append(f"mismatch median {median_mismatch:.3f} <= 0.15")
            # Critical fix: Verify coexist identity -- A_mode must be vision, C_mode must be text
            a_type = mode_tokens.get("A_mode", {}).get("token_type", "unknown")
            c_type = mode_tokens.get("C_mode", {}).get("token_type", "unknown")
            if a_type != "vision":
                passed = False
                reasons.append(f"A_mode.token_type={a_type}, expected 'vision'")
            if c_type != "text":
                passed = False
                reasons.append(f"C_mode.token_type={c_type}, expected 'text'")
        elif model == "spatialvla-4b":
            if median_mismatch >= 0.05:
                passed = False
                reasons.append(f"mismatch median {median_mismatch:.3f} >= 0.05")
            if median_entropy <= 2.0:
                passed = False
                reasons.append(f"entropy median {median_entropy:.3f} <= 2.0")
        elif model == "tracevla-phi3v":
            if median_top1 >= 0.2:
                passed = False
                reasons.append(f"Top1 C_tilde median {median_top1:.3f} >= 0.2")

        results[model] = {
            "passed": passed,
            "median_top1": median_top1,
            "median_mismatch": median_mismatch,
            "median_entropy": median_entropy,
            "a_mode": a_mode,
            "c_mode": c_mode,
            "reasons": reasons,
        }

    # Print results
    print("\n" + "=" * 60)
    print("Gate 1 Pass Criteria Check")
    print("=" * 60)
    all_pass = True
    for model, r in results.items():
        status = "PASS" if r["passed"] else "FAIL"
        if not r["passed"]:
            all_pass = False
        print(f"  {model}: {status}")
        print(f"    Top1 C_tilde median={r['median_top1']:.3f}, mismatch={r['median_mismatch']:.3f}, "
              f"entropy={r['median_entropy']:.3f}")
        print(f"    A_mode={r['a_mode']}, C_mode={r['c_mode']}")
        for reason in r["reasons"]:
            print(f"    -> {reason}")

    print(f"\nOverall: {'ALL PASS -> proceed to Gate 2+3' if all_pass else 'SOME FAILED -> investigate'}")
    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Gate Check Orchestrator")
    parser.add_argument("--gate", type=int, choices=[1, 2, 3],
                        help="Which gate to run (1, 2, or 3)")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--output_base", default="outputs/phase3_gate")
    parser.add_argument("--gate1_dir", default="outputs/phase3_gate",
                        help="Gate 1 output dir (for Gate 2 and 3)")
    parser.add_argument("--n_per_skill", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check_pass", action="store_true",
                        help="Check Gate 1 pass criteria")
    args = parser.parse_args()

    if args.check_pass:
        check_gate1_pass(args.gate1_dir)
        return

    if args.gate == 1:
        run_gate1(args.models, args.output_base, args.n_per_skill, args.seed)
    elif args.gate == 2:
        run_gate2(args.models, args.gate1_dir, Path(args.output_base) / "gate2")
    elif args.gate == 3:
        run_gate3(args.models, args.gate1_dir, Path(args.output_base) / "gate3")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
