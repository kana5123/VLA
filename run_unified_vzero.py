#!/usr/bin/env python3
"""Unified V=0 ablation on EXACTLY the same samples as Gap1 test set.

Resolves discrepancy between Gate2 (32%) and Gap1 (62%) flip rates by
running V=0 on the identical sample subset: samples[30:80].

Usage:
    python run_unified_vzero.py --model ecot-7b --device cuda:0
    python run_unified_vzero.py --model openvla-7b --device cuda:2
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.path.insert(0, str(Path(__file__).parent))

import config
from extract_attention import (
    load_model_from_registry, get_layers, call_processor, detect_token_boundaries,
)
from data_sampler import reload_samples_from_list
from contribution.causal import (
    ValueZeroHook, compute_output_kl, compute_top1_change_rate,
)
from run_phase3_exp_de import get_action_logits


@torch.no_grad()
def run_vzero_on_samples(model, processor, model_cfg, samples, device,
                          target_pos, deep_layers):
    """Run V=0 ablation, return per-sample KL and flip."""
    results = []
    for si, sample in enumerate(samples):
        bounds = detect_token_boundaries(
            processor, model, sample["image"], sample["instruction"],
            device, model_cfg,
        )
        # Original logits
        logits_orig, _ = get_action_logits(
            model, processor, model_cfg, sample, device, bounds
        )
        # V=0
        vzero = ValueZeroHook([target_pos], target_layers=deep_layers)
        vzero.register(model, model_cfg, get_layers)
        logits_vz, _ = get_action_logits(
            model, processor, model_cfg, sample, device, bounds
        )
        vzero.remove()

        kl = compute_output_kl(logits_orig, logits_vz)
        flip = int(logits_orig.argmax().item() != logits_vz.argmax().item())

        results.append({
            "sample_idx": si,
            "skill": sample.get("skill", "unknown"),
            "kl": round(kl, 6),
            "flip": bool(flip),
        })

        if (si + 1) % 10 == 0 or si == 0:
            print(f"  [{si+1}/{len(samples)}] KL={kl:.4f} flip={flip}", flush=True)

    kls = [r["kl"] for r in results]
    flips = [r["flip"] for r in results]
    summary = {
        "n_samples": len(results),
        "target_pos": target_pos,
        "mean_kl": round(float(np.mean(kls)), 4),
        "std_kl": round(float(np.std(kls)), 4),
        "median_kl": round(float(np.median(kls)), 4),
        "flip_rate": round(float(np.mean(flips)), 4),
        "n_flipped": sum(flips),
    }
    return results, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_cal", type=int, default=30, help="Number of calibration samples to skip")
    parser.add_argument("--n_test", type=int, default=50, help="Number of test samples")
    args = parser.parse_args()

    model_name = args.model
    out_dir = config.OUTPUT_DIR / "unified_vzero" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Unified V=0: {model_name}")
    print(f"  Sample range: [{args.n_cal}:{args.n_cal + args.n_test}]")
    print(f"  (Same as Gap1 test set)")
    print(f"{'='*60}\n", flush=True)

    # Load model
    processor, model, model_cfg = load_model_from_registry(model_name, args.device)
    model.eval()

    deep_layers = list(range(max(0, model_cfg.num_layers - 10), model_cfg.num_layers))

    # Load exact same samples as Gap1
    sample_list_path = config.OUTPUT_DIR / "phase3_gate" / model_name / "sample_list.json"
    all_samples = reload_samples_from_list(str(sample_list_path), config.DATA_CACHE_DIR)
    print(f"  Total samples in list: {len(all_samples)}")

    # Gap1 test set: skip n_cal, take n_test
    test_samples = all_samples[args.n_cal:args.n_cal + args.n_test]
    print(f"  Test samples: {len(test_samples)} (indices {args.n_cal}-{args.n_cal + len(test_samples) - 1})")

    # Also run on Gate2's sample range for comparison
    gate2_samples = all_samples[:args.n_test]
    print(f"  Gate2-range samples: {len(gate2_samples)} (indices 0-{len(gate2_samples) - 1})")

    # Target position: always pos 0 (A_mode anchor for bottleneck models)
    target_pos = 0

    # Run on Gap1 test set
    print(f"\n--- V=0 on Gap1 test set (samples[{args.n_cal}:{args.n_cal+args.n_test}]) ---", flush=True)
    t0 = time.time()
    gap1_results, gap1_summary = run_vzero_on_samples(
        model, processor, model_cfg, test_samples, args.device, target_pos, deep_layers
    )
    print(f"  Gap1-range: KL={gap1_summary['mean_kl']:.4f}, flip={gap1_summary['flip_rate']:.2%} "
          f"({gap1_summary['n_flipped']}/{gap1_summary['n_samples']}) [{time.time()-t0:.1f}s]")

    # Run on Gate2 sample range
    print(f"\n--- V=0 on Gate2 range (samples[0:{args.n_test}]) ---", flush=True)
    t0 = time.time()
    gate2_results, gate2_summary = run_vzero_on_samples(
        model, processor, model_cfg, gate2_samples, args.device, target_pos, deep_layers
    )
    print(f"  Gate2-range: KL={gate2_summary['mean_kl']:.4f}, flip={gate2_summary['flip_rate']:.2%} "
          f"({gate2_summary['n_flipped']}/{gate2_summary['n_samples']}) [{time.time()-t0:.1f}s]")

    # Run on ALL 150 samples
    print(f"\n--- V=0 on ALL samples (samples[0:{len(all_samples)}]) ---", flush=True)
    t0 = time.time()
    all_results, all_summary = run_vzero_on_samples(
        model, processor, model_cfg, all_samples, args.device, target_pos, deep_layers
    )
    print(f"  All: KL={all_summary['mean_kl']:.4f}, flip={all_summary['flip_rate']:.2%} "
          f"({all_summary['n_flipped']}/{all_summary['n_samples']}) [{time.time()-t0:.1f}s]")

    # Save
    output = {
        "model": model_name,
        "target_pos": target_pos,
        "deep_layers": deep_layers,
        "gap1_range": {
            "sample_range": f"[{args.n_cal}:{args.n_cal + len(test_samples)}]",
            "summary": gap1_summary,
            "per_sample": gap1_results,
        },
        "gate2_range": {
            "sample_range": f"[0:{len(gate2_samples)}]",
            "summary": gate2_summary,
            "per_sample": gate2_results,
        },
        "all_samples": {
            "sample_range": f"[0:{len(all_samples)}]",
            "summary": all_summary,
            "per_sample": all_results,
        },
    }

    out_path = out_dir / "unified_vzero_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print comparison table
    print(f"\n{'='*60}")
    print(f"  COMPARISON: {model_name}")
    print(f"  {'Source':<25} {'N':>5} {'Mean KL':>10} {'Flip Rate':>12}")
    print(f"  {'-'*52}")
    print(f"  {'Gap1 range [30:80]':<25} {gap1_summary['n_samples']:>5} "
          f"{gap1_summary['mean_kl']:>10.4f} {gap1_summary['flip_rate']:>12.2%}")
    print(f"  {'Gate2 range [0:50]':<25} {gate2_summary['n_samples']:>5} "
          f"{gate2_summary['mean_kl']:>10.4f} {gate2_summary['flip_rate']:>12.2%}")
    print(f"  {'All [0:150]':<25} {all_summary['n_samples']:>5} "
          f"{all_summary['mean_kl']:>10.4f} {all_summary['flip_rate']:>12.2%}")
    print(f"  Saved: {out_path}")
    print(f"{'='*60}\n", flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
