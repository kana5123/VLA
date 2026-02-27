#!/usr/bin/env python3
"""Gap 1 Fix: Mean Ablation + Activation Patching for Bottleneck Tokens.

Complements existing V=0 ablation with mean ablation to address
reviewer concern about OOD activation shifts (Hase & Bansal, 2021).

Steps:
  1. Collect per-layer mean V projections from N calibration samples
  2. Run V=0 ablation (existing) and V=mean ablation at anchor position
  3. Compare KL divergence and top-1 change rate for both methods
  4. If both produce large effects → causal claim is bulletproof

Usage:
    python run_gap1_mean_ablation.py --model openvla-7b --device cuda:0
    python run_gap1_mean_ablation.py --model ecot-7b --device cuda:0
    python run_gap1_mean_ablation.py --all --device cuda:0
"""

from __future__ import annotations

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
    load_model_from_registry,
    get_layers,
    call_processor,
    detect_token_boundaries,
)
from data_sampler import reload_samples_from_list, get_action_for_sample
from contribution.causal import (
    ValueZeroHook,
    ValueMeanHook,
    compute_output_kl,
    compute_top1_change_rate,
)
from run_phase3_exp_de import detect_anchor_position, get_action_logits


# ── Collect Layer Means ──────────────────────────────────────────────

class VMeanCollector:
    """Collects per-layer mean V projections across ALL vision positions.

    The mean is over all vision tokens (not just the target), so replacing
    vision[0] with this mean creates a meaningful ablation that tests
    causal importance without OOD activation shifts.
    """

    def __init__(self, model, model_cfg, target_layers, vision_start, vision_end):
        self.model = model
        self.model_cfg = model_cfg
        self.target_layers = target_layers
        self.vision_start = vision_start
        self.vision_end = vision_end
        self._accum = {}  # layer_idx → running sum tensor
        self._count = 0
        self._handles = []

    def register(self):
        layers = get_layers(self.model, self.model_cfg)
        for layer_idx, layer in enumerate(layers):
            if layer_idx not in self.target_layers:
                continue
            attn = layer.self_attn
            if hasattr(attn, "v_proj"):
                handle = attn.v_proj.register_forward_hook(
                    self._make_hook(layer_idx)
                )
                self._handles.append(handle)

    def _make_hook(self, layer_idx):
        collector = self
        def hook_fn(module, args, output):
            vs = collector.vision_start
            ve = min(collector.vision_end, output.shape[1])
            if vs < ve:
                # Mean over ALL vision token V projections (excluding target=0)
                # Use positions 1:ve to avoid including the sink itself
                start = max(vs + 1, 1)  # Skip vision[0] in mean
                if start < ve:
                    v_mean = output[0, start:ve, :].detach().float().mean(dim=0)
                else:
                    v_mean = output[0, vs:ve, :].detach().float().mean(dim=0)
                if layer_idx not in collector._accum:
                    collector._accum[layer_idx] = torch.zeros_like(v_mean)
                collector._accum[layer_idx] += v_mean
        return hook_fn

    def collect_sample(self, model, processor, model_cfg, sample, device, bounds):
        """Run one forward pass to accumulate V projection."""
        prompt = model_cfg.prompt_template.format(instruction=sample["instruction"])
        inputs = call_processor(
            processor, prompt, sample["image"], model_cfg, return_tensors="pt"
        ).to(device)
        if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

        fwd_kwargs = {k: v for k, v in inputs.items()}
        fwd_kwargs["use_cache"] = False
        if model_cfg.architecture == "gemma2" and "intrinsic" not in fwd_kwargs:
            fwd_kwargs["intrinsic"] = torch.tensor(
                [[[218.26, 0.0, 111.83], [0.0, 218.26, 111.79], [0.0, 0.0, 1.0]]],
                device=device, dtype=torch.float32,
            )

        with torch.no_grad():
            model(**fwd_kwargs)
        self._count += 1

    def get_means(self):
        """Return {layer_idx: mean_tensor}."""
        return {l: v / self._count for l, v in self._accum.items()}

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


# ── Ablation Experiment ──────────────────────────────────────────────

@torch.no_grad()
def run_ablation_comparison(model, processor, model_cfg, samples, device,
                             bounds_cache, deep_layers, anchor_abs,
                             layer_means, output_dir):
    """Run V=0 and V=mean ablation side by side."""
    results = []

    for si, sample in enumerate(samples):
        bounds = bounds_cache[si]

        # Original logits
        logits_orig, _ = get_action_logits(
            model, processor, model_cfg, sample, device, bounds
        )

        # --- V=0 ablation ---
        vzero = ValueZeroHook(
            target_positions=[anchor_abs], target_layers=deep_layers
        )
        vzero.register(model, model_cfg, get_layers)
        logits_vzero, _ = get_action_logits(
            model, processor, model_cfg, sample, device, bounds
        )
        vzero.remove()

        kl_vzero = compute_output_kl(logits_orig, logits_vzero)
        top1_changed_vzero = (
            logits_orig.argmax().item() != logits_vzero.argmax().item()
        )

        # --- V=mean ablation ---
        vmean = ValueMeanHook(
            target_positions=[anchor_abs], target_layers=deep_layers
        )
        vmean.set_layer_means(layer_means)
        vmean.register(model, model_cfg, get_layers)
        logits_vmean, _ = get_action_logits(
            model, processor, model_cfg, sample, device, bounds
        )
        vmean.remove()

        kl_vmean = compute_output_kl(logits_orig, logits_vmean)
        top1_changed_vmean = (
            logits_orig.argmax().item() != logits_vmean.argmax().item()
        )

        # Cosine similarity
        cos_vzero = float(torch.nn.functional.cosine_similarity(
            logits_orig.unsqueeze(0), logits_vzero.unsqueeze(0)
        ).item())
        cos_vmean = float(torch.nn.functional.cosine_similarity(
            logits_orig.unsqueeze(0), logits_vmean.unsqueeze(0)
        ).item())

        entry = {
            "sample_idx": si,
            "skill": sample.get("skill", "unknown"),
            "anchor_abs": anchor_abs,
            "vzero": {
                "kl_divergence": round(kl_vzero, 4),
                "top1_changed": top1_changed_vzero,
                "cosine_similarity": round(cos_vzero, 6),
            },
            "vmean": {
                "kl_divergence": round(kl_vmean, 4),
                "top1_changed": top1_changed_vmean,
                "cosine_similarity": round(cos_vmean, 6),
            },
        }
        results.append(entry)

        if (si + 1) % 10 == 0 or si == 0:
            print(f"  [{si+1}/{len(samples)}] "
                  f"V=0: KL={kl_vzero:.3f} flip={top1_changed_vzero} | "
                  f"V=mean: KL={kl_vmean:.3f} flip={top1_changed_vmean}",
                  flush=True)

    # Aggregate
    vzero_kls = [r["vzero"]["kl_divergence"] for r in results]
    vmean_kls = [r["vmean"]["kl_divergence"] for r in results]
    vzero_flips = [r["vzero"]["top1_changed"] for r in results]
    vmean_flips = [r["vmean"]["top1_changed"] for r in results]

    summary = {
        "n_samples": len(results),
        "anchor_abs": anchor_abs,
        "vzero": {
            "mean_kl": round(float(np.mean(vzero_kls)), 4),
            "std_kl": round(float(np.std(vzero_kls)), 4),
            "median_kl": round(float(np.median(vzero_kls)), 4),
            "flip_rate": round(float(np.mean(vzero_flips)), 4),
        },
        "vmean": {
            "mean_kl": round(float(np.mean(vmean_kls)), 4),
            "std_kl": round(float(np.std(vmean_kls)), 4),
            "median_kl": round(float(np.median(vmean_kls)), 4),
            "flip_rate": round(float(np.mean(vmean_flips)), 4),
        },
        "both_methods_agree": round(
            float(np.mean([
                (vz["top1_changed"] == vm["top1_changed"])
                for vz, vm in zip(
                    [r["vzero"] for r in results],
                    [r["vmean"] for r in results]
                )
            ])), 4
        ),
    }

    return results, summary


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gap 1: Mean Ablation Comparison")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--all", action="store_true",
                        help="Run for all 4 models")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_samples", type=int, default=50,
                        help="Number of test samples")
    parser.add_argument("--n_calibration", type=int, default=30,
                        help="Number of calibration samples for mean computation")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    models = (
        ["openvla-7b", "ecot-7b", "tracevla-phi3v", "spatialvla-4b"]
        if args.all else [args.model]
    )

    for model_name in models:
        if model_name is None:
            continue

        out_dir = Path(args.output_dir) if args.output_dir else \
            config.OUTPUT_DIR / "gap1_mean_ablation" / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Gap 1 Mean Ablation: {model_name}")
        print(f"  Device: {args.device}")
        print(f"  N calibration: {args.n_calibration}")
        print(f"  N test: {args.n_samples}")
        print(f"{'='*60}\n", flush=True)

        # Load model
        print("Loading model...", flush=True)
        processor, model, model_cfg = load_model_from_registry(model_name, args.device)
        model.eval()

        deep_layers = list(range(
            max(0, model_cfg.num_layers - 10), model_cfg.num_layers
        ))

        # Load samples
        gate1_dir = config.OUTPUT_DIR / "phase3_gate" / model_name
        sample_list_path = gate1_dir / "sample_list.json"
        print(f"Loading samples from {sample_list_path}...", flush=True)
        all_samples = reload_samples_from_list(sample_list_path, config.DATA_CACHE_DIR)

        n_cal = min(args.n_calibration, len(all_samples))
        n_test = min(args.n_samples, len(all_samples) - n_cal)
        cal_samples = all_samples[:n_cal]
        test_samples = all_samples[n_cal:n_cal + n_test]
        print(f"  Calibration: {len(cal_samples)}, Test: {len(test_samples)}")

        # Detect boundaries
        bounds_cache = {}
        for si, s in enumerate(cal_samples + test_samples):
            bounds_cache[si] = detect_token_boundaries(
                processor, model, s["image"], s["instruction"],
                args.device, model_cfg,
            )
        print(f"  Boundaries cached for {len(bounds_cache)} samples")

        # Detect anchor position
        verification_dir = config.OUTPUT_DIR / "phase3_gate" / "verification"
        bounds_0 = bounds_cache[0]
        anchor_rel, anchor_abs, is_anchored = detect_anchor_position(
            model_name, verification_dir, bounds_0
        )
        print(f"  Anchor: rel={anchor_rel}, abs={anchor_abs}, anchored={is_anchored}")

        # Step 1: Collect mean V projections (over all vision tokens excl. target)
        print("\n  Collecting mean V projections (all vision tokens excl. anchor)...",
              flush=True)
        t0 = time.time()
        vs = bounds_0["vision_start"]
        ve = bounds_0["vision_end"]
        collector = VMeanCollector(model, model_cfg, deep_layers, vs, ve)
        collector.register()

        for ci, sample in enumerate(cal_samples):
            collector.collect_sample(
                model, processor, model_cfg, sample, args.device,
                bounds_cache[ci],
            )
            if (ci + 1) % 10 == 0:
                print(f"    Calibration {ci+1}/{n_cal}", flush=True)

        layer_means = collector.get_means()
        collector.remove()
        print(f"  Mean V collected for {len(layer_means)} layers ({time.time()-t0:.1f}s)")

        # Step 2: Run ablation comparison
        print("\n  Running ablation comparison...", flush=True)
        t0 = time.time()
        # Re-index bounds for test samples
        test_bounds = {
            i: bounds_cache[n_cal + i]
            for i in range(len(test_samples))
        }
        results, summary = run_ablation_comparison(
            model, processor, model_cfg, test_samples, args.device,
            test_bounds, deep_layers, anchor_abs, layer_means, out_dir,
        )
        elapsed = time.time() - t0

        # Save
        full_result = {
            "model": model_name,
            "n_calibration": n_cal,
            "n_test": len(test_samples),
            "anchor_abs": anchor_abs,
            "anchor_rel": anchor_rel,
            "is_anchored": is_anchored,
            "deep_layers": deep_layers,
            "summary": summary,
            "per_sample": results,
            "elapsed_s": round(elapsed, 1),
        }

        out_path = out_dir / "ablation_comparison.json"
        with open(out_path, "w") as f:
            json.dump(full_result, f, indent=2)

        # Print summary
        print(f"\n{'='*60}")
        print(f"  RESULTS: {model_name}")
        print(f"  {'Method':<15} {'Mean KL':>10} {'Median KL':>12} {'Flip Rate':>12}")
        print(f"  {'-'*49}")
        print(f"  {'V=0':<15} "
              f"{summary['vzero']['mean_kl']:>10.4f} "
              f"{summary['vzero']['median_kl']:>12.4f} "
              f"{summary['vzero']['flip_rate']:>12.1%}")
        print(f"  {'V=mean':<15} "
              f"{summary['vmean']['mean_kl']:>10.4f} "
              f"{summary['vmean']['median_kl']:>12.4f} "
              f"{summary['vmean']['flip_rate']:>12.1%}")
        print(f"  Agreement rate: {summary['both_methods_agree']:.1%}")
        print(f"  Saved: {out_path}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"{'='*60}\n", flush=True)

        # Cleanup
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
