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
from extract_attention import load_model_from_registry, get_layers, call_processor
from visualize_text_attention import load_samples_from_cache

from contribution.causal import (
    ValueZeroHook,
    compute_output_kl,
    compute_top1_change_rate,
    run_vzero_sanity_check,
)
from contribution.visualize import plot_masking_ablation


def run_causal(model_name, device, n_samples, output_dir, candidate_positions,
               target_layers=None, sample_list_path=None):
    """Run V=0 causal ablation, measure KL divergence.

    AttentionKnockoutHook was removed — it was a no-op bug (Phase 2.5).
    """
    print(f"\n{'='*70}")
    print(f"Causal Experiment — {model_name} (V=0 only, knockout removed)")
    print(f"  Candidates (abs_t): {candidate_positions}")
    print(f"{'='*70}")

    processor, model, model_cfg = load_model_from_registry(model_name, device)
    if sample_list_path:
        from data_sampler import reload_samples_from_list
        samples = reload_samples_from_list(sample_list_path, config.DATA_CACHE_DIR)
        print(f"  Loaded {len(samples)} samples from {sample_list_path}")
    else:
        samples = load_samples_from_cache(config.DATA_CACHE_DIR, n_samples=n_samples)

    # B4: Sanity check V=0 on first sample
    print("\n  B4 Sanity Check: Verifying V=0 hook works...")
    sample0 = samples[0]
    prompt = model_cfg.prompt_template.format(instruction=sample0["instruction"])
    inputs0 = call_processor(processor, prompt, sample0["image"], model_cfg, return_tensors="pt").to(device)
    if "pixel_values" in inputs0 and inputs0["pixel_values"].dtype != model.dtype:
        inputs0["pixel_values"] = inputs0["pixel_values"].to(model.dtype)

    sanity = run_vzero_sanity_check(model, model_cfg, get_layers, inputs0, candidate_positions[:3])
    print(f"    hook_fired={sanity['hook_fired']}, "
          f"logits_changed={sanity['logits_changed']}, "
          f"kl={sanity['kl_divergence']:.6f}")
    if not sanity["hook_fired"]:
        print("    WARNING: V=0 hook did NOT fire! Check hook registration.")
    if not sanity["logits_changed"]:
        print("    WARNING: V=0 did NOT change logits! Hook may be on wrong module.")

    results = {
        "model": model_name,
        "method": "v_zero",
        "candidates_abs_t": candidate_positions,
        "target_layers": target_layers,
        "sanity_check": sanity,
        "per_k": {},
    }

    for k in [1, 3, 5]:
        targets = candidate_positions[:k]
        vzero_kls = []
        vzero_changes = []

        for si, sample in enumerate(samples):
            prompt = model_cfg.prompt_template.format(instruction=sample["instruction"])
            inputs = call_processor(processor, prompt, sample["image"], model_cfg, return_tensors="pt").to(device)
            if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
                inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

            # Original forward
            with torch.no_grad():
                out_orig = model(**inputs)
            logits_orig = out_orig.logits[0, -1, :]

            # V=0 forward
            vzero = ValueZeroHook(targets, target_layers=target_layers)
            vzero.register(model, model_cfg, get_layers)
            with torch.no_grad():
                out_vz = model(**inputs)
            logits_vz = out_vz.logits[0, -1, :]
            vzero.remove()

            kl_vz = compute_output_kl(logits_orig, logits_vz)
            change_vz = compute_top1_change_rate(logits_orig.unsqueeze(0), logits_vz.unsqueeze(0))
            vzero_kls.append(kl_vz)
            vzero_changes.append(change_vz)

        results["per_k"][k] = {
            "targets": targets,
            "vzero_mean_kl": float(np.mean(vzero_kls)),
            "vzero_std_kl": float(np.std(vzero_kls)),
            "vzero_mean_top1_change": float(np.mean(vzero_changes)),
        }

        print(f"  K={k}, V0_KL={np.mean(vzero_kls):.4f}+-{np.std(vzero_kls):.4f}, "
              f"top1_change={np.mean(vzero_changes):.3f}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "causal_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Report saved: {report_path}")

    # Plot
    k_vals = sorted(results["per_k"].keys())
    plot_data = {"value_zero": [results["per_k"][k]["vzero_mean_kl"] for k in k_vals]}
    plot_masking_ablation(k_vals, plot_data, output_dir / "masking_ablation.png")

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
    parser.add_argument("--layer_mode", choices=["all", "block1", "block2"],
                        default="all",
                        help="Layer range for V=0: all=deep22-31, block1=22-26, block2=27-31")
    parser.add_argument("--candidates_json", type=str, default=None,
                        help="Path to mode_tokens.json (from Gate ①)")
    parser.add_argument("--sample_list", type=str, default=None,
                        help="Path to sample_list.json (reuse Gate ① samples)")
    parser.add_argument("--peak_type", choices=["A_mode", "C_mode", "R_mode"],
                        default=None,
                        help="Which mode token to target (requires --candidates_json)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else config.OUTPUT_DIR / "causal_experiment" / args.model

    # Get candidate positions
    if args.candidates:
        candidate_positions = args.candidates
    elif args.candidates_json and args.peak_type:
        # Gate ②: use mode tokens from Gate ①
        with open(args.candidates_json) as f:
            mode_tokens = json.load(f)
        peak = mode_tokens.get(args.peak_type, {})
        abs_t = peak.get("abs_t", 0)
        freq = peak.get("freq", 0)
        candidate_positions = [abs_t]
        print(f"  Gate ② mode: {args.peak_type} → abs_t={abs_t}, freq={freq:.2f}")
        if freq < 0.7:
            print(f"  WARNING: freq={freq:.2f} < 0.7 (unstable). Consider testing Top-3.")
    elif args.report:
        with open(args.report) as f:
            report = json.load(f)
        positions = []
        for layer_info in report.get("layer_analysis", {}).values():
            if "a_peak" in layer_info:
                positions.append(layer_info["a_peak"].get("abs_t", 0))
            if "c_peak" in layer_info:
                pos = layer_info["c_peak"].get("abs_t", 0)
                if pos not in positions:
                    positions.append(pos)
            elif "dominant_position" in layer_info:
                positions.append(layer_info["dominant_position"])
        if positions:
            from collections import Counter
            candidate_positions = [p for p, _ in Counter(positions).most_common(5)]
        else:
            candidate_positions = [0]
    else:
        candidate_positions = [0]
        print("  WARNING: No candidates specified, defaulting to position 0")

    # Resolve layer_mode → target_layers
    from contribution.causal import get_deep_layer_ranges
    from model_registry import get_model as _get_model
    model_cfg_temp = _get_model(args.model)
    layer_ranges = get_deep_layer_ranges(model_cfg_temp.num_layers)
    target_layers = layer_ranges.get(args.layer_mode, layer_ranges["all"])
    print(f"  Layer mode: {args.layer_mode} → layers {target_layers}")

    run_causal(args.model, args.device, args.n_samples, output_dir, candidate_positions,
               target_layers=target_layers, sample_list_path=args.sample_list)


if __name__ == "__main__":
    main()
