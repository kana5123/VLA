#!/usr/bin/env python3
"""Task 2: Sample Expansion — N=20 → N=200 balanced.

Creates an expanded sample list (8 skills × 25 per skill = 200 samples)
and re-runs Exp D+E+F with the larger sample set.

This script:
1. Creates a new balanced sample_list.json with 200 samples
2. Runs run_phase3_exp_de.py with the new sample list
3. Saves results to outputs/phase3_gate_expanded/

Usage (parallel on 4 GPUs):
  python run_expanded_samples.py --model ecot-7b --device cuda:0
  python run_expanded_samples.py --model openvla-7b --device cuda:1
  python run_expanded_samples.py --model spatialvla-4b --device cuda:2
  python run_expanded_samples.py --model tracevla-phi3v --device cuda:3
"""
import argparse
import json
import sys
import time
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config
from data_sampler import (
    load_balanced_samples,
    save_sample_list,
    build_skill_episode_index,
)

EXPANDED_N_PER_SKILL = 25
EXPANDED_SEED = 2024  # Different seed from original (42) for independence
EXPANDED_TARGET_SKILLS = ["place", "move", "pick", "fold", "open", "close", "turn"]
# Total: 7 skills × 25 = 175 samples (vs original 6 × 25 = 150, used only 20)
# sweep/stack/pour/wipe have too few episodes (<100) for balanced sampling


def create_expanded_sample_list(output_dir: Path) -> Path:
    """Create expanded sample list with 200 balanced samples."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_list_path = output_dir / "sample_list.json"

    if sample_list_path.exists():
        with open(sample_list_path) as f:
            existing = json.load(f)
        n_existing = len(existing.get("samples", []))
        print(f"  Found existing sample_list with {n_existing} samples")
        if n_existing >= 150:  # Close enough to target
            print(f"  Reusing existing sample list")
            return sample_list_path

    # Check available skills first
    print(f"  Building skill episode index from {config.DATA_CACHE_DIR}...")
    skill_index = build_skill_episode_index(config.DATA_CACHE_DIR)
    print(f"  Available skills:")
    for skill in sorted(skill_index.keys()):
        print(f"    {skill:12s}: {len(skill_index[skill]):5d} episodes")

    # Filter to skills with enough episodes
    available_skills = []
    for skill in EXPANDED_TARGET_SKILLS:
        n_avail = len(skill_index.get(skill, []))
        if n_avail >= EXPANDED_N_PER_SKILL:
            available_skills.append(skill)
        else:
            print(f"  WARNING: skill '{skill}' has only {n_avail} episodes "
                  f"(need {EXPANDED_N_PER_SKILL}), skipping")

    print(f"\n  Sampling {EXPANDED_N_PER_SKILL} per skill × {len(available_skills)} skills "
          f"= {EXPANDED_N_PER_SKILL * len(available_skills)} total")

    samples = load_balanced_samples(
        config.DATA_CACHE_DIR,
        n_per_skill=EXPANDED_N_PER_SKILL,
        target_skills=available_skills,
        seed=EXPANDED_SEED,
    )

    save_sample_list(samples, sample_list_path,
                     seed=EXPANDED_SEED,
                     n_per_skill=EXPANDED_N_PER_SKILL,
                     target_skills=available_skills)

    # Print skill distribution
    from collections import Counter
    skill_dist = Counter(s["skill"] for s in samples)
    print(f"  Skill distribution:")
    for skill, count in sorted(skill_dist.items()):
        print(f"    {skill:12s}: {count:3d}")

    return sample_list_path


def run_experiments(model: str, device: str, sample_list_path: Path,
                    gate1_dir: Path, output_dir: Path, experiments: str = "d,e,f"):
    """Run Exp D+E+F using run_phase3_exp_de.py with expanded samples."""
    python = sys.executable
    n_samples = 200  # Use all expanded samples

    cmd = [
        python, "run_phase3_exp_de.py",
        "--model", model,
        "--device", device,
        "--gate1_dir", str(gate1_dir),
        "--output_dir", str(output_dir),
        "--n_samples", str(n_samples),
        "--experiments", experiments,
    ]

    print(f"\n{'='*60}")
    print(f"  Running Exp {experiments.upper()} for {model}")
    print(f"  Device: {device}")
    print(f"  N samples: {n_samples}")
    print(f"  Output: {output_dir}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    elapsed = time.time() - start

    print(f"\n  {model} completed in {elapsed/60:.1f} minutes "
          f"(exit code: {result.returncode})")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Expanded sample experiments")
    parser.add_argument("--model", required=True,
                        choices=["ecot-7b", "openvla-7b", "spatialvla-4b", "tracevla-phi3v"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--experiments", default="d,e,f",
                        help="Which experiments to run (d,e,f)")
    parser.add_argument("--skip_create", action="store_true",
                        help="Skip sample list creation (reuse existing)")
    args = parser.parse_args()

    # Directories
    expanded_base = config.OUTPUT_DIR / "phase3_gate_expanded"
    gate1_dir = expanded_base / args.model
    output_dir = expanded_base / "verification" / args.model

    # Step 1: Create expanded sample list (shared across models)
    if not args.skip_create:
        sample_list_path = create_expanded_sample_list(gate1_dir)
    else:
        sample_list_path = gate1_dir / "sample_list.json"
        assert sample_list_path.exists(), f"No sample list at {sample_list_path}"

    # Copy Exp C results from original run (needed for anchor detection)
    orig_verification = config.OUTPUT_DIR / "phase3_gate" / "verification" / args.model
    exp_c_src = orig_verification / "exp_c_position_anchoring.json"
    exp_c_dst = output_dir / "exp_c_position_anchoring.json"
    if exp_c_src.exists() and not exp_c_dst.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(exp_c_src, exp_c_dst)
        print(f"  Copied Exp C results from original run")

    # Step 2: Run experiments
    rc = run_experiments(args.model, args.device, sample_list_path,
                         gate1_dir, output_dir, args.experiments)
    sys.exit(rc)


if __name__ == "__main__":
    main()
