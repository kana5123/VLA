#!/usr/bin/env python3
"""
Causal verification: mask candidate tokens and measure output KL divergence.

Usage:
  python run_causal_experiment.py --model openvla-7b --device cuda:0 \
    --report outputs/contribution_analysis/openvla-7b/contribution_report.json
"""
import argparse
import json
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
from extract_attention import load_model_from_registry, get_layers, call_processor, detect_token_boundaries
from verify_attention_sinks import SinkVerificationHookManager
from visualize_text_attention import load_samples_from_cache

from contribution.causal import AttentionKnockoutHook, compute_output_kl
from contribution.visualize import plot_masking_ablation


def run_causal(model_name, device, n_samples, output_dir, candidate_positions):
    """Run forward pass with and without knockout, measure KL."""
    print(f"\n{'='*70}")
    print(f"Causal Experiment — {model_name}")
    print(f"  Candidates: {candidate_positions}")
    print(f"{'='*70}")

    processor, model, model_cfg = load_model_from_registry(model_name, device)
    samples = load_samples_from_cache(config.DATA_CACHE_DIR, n_samples=n_samples)

    results = {"model": model_name, "candidates": candidate_positions, "per_k": {}}

    for k in [1, 3, 5]:
        targets = candidate_positions[:k]
        kl_values = []

        for si, sample in enumerate(samples):
            prompt = model_cfg.prompt_template.format(instruction=sample["instruction"])
            inputs = call_processor(processor, prompt, sample["image"], model_cfg, return_tensors="pt").to(device)
            if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
                inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

            # Original forward
            with torch.no_grad():
                out_orig = model(**inputs)
            logits_orig = out_orig.logits[0, -1, :]  # last token logits

            # Knockout forward
            boundaries = detect_token_boundaries(
                processor, model, sample["image"], sample["instruction"], device, model_cfg
            )
            text_end = boundaries["text_end"]
            query_range = (max(0, text_end - 4), text_end)
            knockout = AttentionKnockoutHook(targets, query_range)
            knockout.register(model, model_cfg, get_layers)

            with torch.no_grad():
                out_masked = model(**inputs, output_attentions=True)
            logits_masked = out_masked.logits[0, -1, :]
            knockout.remove()

            kl = compute_output_kl(logits_orig, logits_masked)
            kl_values.append(kl)

        results["per_k"][k] = {
            "targets": targets,
            "mean_kl": float(np.mean(kl_values)),
            "std_kl": float(np.std(kl_values)),
        }
        print(f"  K={k}, targets={targets}, KL={np.mean(kl_values):.4f} +/- {np.std(kl_values):.4f}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "causal_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Report saved: {report_path}")

    # Plot
    k_vals = sorted(results["per_k"].keys())
    plot_masking_ablation(
        k_vals,
        {"attention_knockout": [results["per_k"][k]["mean_kl"] for k in k_vals]},
        output_dir / "masking_ablation.png",
    )

    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description="Causal Verification Experiment")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--report", type=str, default=None,
                       help="Path to contribution_report.json (to auto-detect candidates)")
    parser.add_argument("--candidates", type=int, nargs="+", default=None,
                       help="Manual candidate positions (e.g., --candidates 0 1 5)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else config.OUTPUT_DIR / "causal_experiment" / args.model

    # Get candidate positions
    if args.candidates:
        candidate_positions = args.candidates
    elif args.report:
        with open(args.report) as f:
            report = json.load(f)
        # Find most common dominant position across layers
        positions = []
        for layer_info in report.get("layer_analysis", {}).values():
            if "dominant_position" in layer_info:
                positions.append(layer_info["dominant_position"])
        if positions:
            from collections import Counter
            candidate_positions = [p for p, _ in Counter(positions).most_common(5)]
        else:
            candidate_positions = [0]
    else:
        candidate_positions = [0]
        print("  WARNING: No candidates specified, defaulting to position 0")

    run_causal(args.model, args.device, args.n_samples, output_dir, candidate_positions)


if __name__ == "__main__":
    main()
