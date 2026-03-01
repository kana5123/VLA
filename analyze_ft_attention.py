#!/usr/bin/env python3
"""Pre-trained vs Fine-tuned Attention Routing Comparison.

Core experiment: Does LIBERO fine-tuning resolve position-anchoring bottleneck?
Measures 3 key metrics before/after fine-tuning:
  1. Vision[0] attention share (action queries → vision token 0)
  2. W_OV contribution top-1 concentration
  3. Taxonomy classification (Bottleneck / Normal / Sink / Coexist)

Usage:
    python analyze_ft_attention.py --model openvla-7b --device cuda:0
    python analyze_ft_attention.py --model ecot-7b --device cuda:0 \
        --lora_path outputs/libero_ft/ecot-7b/libero_spatial/lora_adapter
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import h5py
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
from verify_attention_sinks import SinkVerificationHookManager, get_wov_matrix
from contribution.compute import compute_perhead_contribution


# ── LIBERO Sample Loader ─────────────────────────────────────────────

LIBERO_DATA_DIR = Path("/ceph_data/kana5123/libero_datasets")


def load_libero_samples(suite: str = "libero_spatial", n: int = 20,
                        image_size: int = 224, seed: int = 42):
    """Load N random LIBERO samples for attention analysis."""
    from PIL import Image as PILImage

    suite_dir = LIBERO_DATA_DIR / suite
    hdf5_files = sorted(suite_dir.glob("*.hdf5"))

    all_entries = []
    for hdf5_path in hdf5_files:
        task_name = hdf5_path.stem.replace("_demo", "")
        instruction = task_name.replace("_", " ")
        with h5py.File(hdf5_path, "r") as f:
            demo_keys = [k for k in f["data"].keys() if k.startswith("demo_")]
            for dk in demo_keys[:10]:  # Max 10 demos per task
                n_steps = f[f"data/{dk}/actions"].shape[0]
                # Pick middle timestep (most informative)
                mid = n_steps // 2
                all_entries.append((str(hdf5_path), dk, mid, instruction))

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(all_entries), size=min(n, len(all_entries)), replace=False)

    samples = []
    for idx in indices:
        hdf5_path, dk, si, instruction = all_entries[idx]
        with h5py.File(hdf5_path, "r") as f:
            img_arr = f[f"data/{dk}/obs/agentview_rgb"][si]
        img = PILImage.fromarray(img_arr).resize((image_size, image_size), PILImage.LANCZOS)
        samples.append({"image": img, "instruction": instruction})

    print(f"  Loaded {len(samples)} LIBERO samples from {suite}")
    return samples


# ── Metric 1: Vision[0] Attention Share ──────────────────────────────

@torch.no_grad()
def measure_vision0_attention(model, processor, model_cfg, samples, device,
                               deep_layers, bounds):
    """Measure attention share of vision[0] across deep layers.

    Returns per-layer mean attention share of vision[0] from action query.
    """
    vs = bounds["vision_start"]
    ve = bounds["vision_end"]

    hook_mgr = SinkVerificationHookManager(model, model_cfg)
    hook_mgr.register_hooks()

    per_layer_shares = {l: [] for l in deep_layers}

    for si, sample in enumerate(samples):
        prompt = model_cfg.prompt_template.format(instruction=sample["instruction"])
        inputs = call_processor(
            processor, prompt, sample["image"], model_cfg, return_tensors="pt"
        ).to(device)
        if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

        fwd_kwargs = {k: v for k, v in inputs.items()}
        fwd_kwargs["use_cache"] = False
        fwd_kwargs["output_attentions"] = True

        # SpatialVLA intrinsic
        if model_cfg.architecture == "gemma2" and "intrinsic" not in fwd_kwargs:
            fwd_kwargs["intrinsic"] = torch.tensor(
                [[[218.26, 0.0, 111.83], [0.0, 218.26, 111.79], [0.0, 0.0, 1.0]]],
                device=device, dtype=torch.float32,
            )

        hook_mgr.attention_weights.clear()
        hook_mgr.hidden_states.clear()
        model(**fwd_kwargs)

        seq_len = inputs["input_ids"].shape[1]
        qpos = seq_len - 1  # Last input position (action query proxy)

        for l in deep_layers:
            if l not in hook_mgr.attention_weights:
                continue
            attn = hook_mgr.attention_weights[l]  # (H, seq, seq)
            # Per-head vision[0] share
            attn_q = attn[:, qpos, vs:ve]  # (H, n_vision)
            total_vision = attn_q.sum(dim=-1)  # (H,)
            v0_share = attn_q[:, 0] / (total_vision + 1e-10)  # (H,)
            per_layer_shares[l].append(v0_share.mean().item())

    hook_mgr.remove_hooks()

    result = {}
    for l in deep_layers:
        vals = per_layer_shares[l]
        result[l] = {
            "mean": float(np.mean(vals)) if vals else 0.0,
            "std": float(np.std(vals)) if vals else 0.0,
            "n": len(vals),
        }

    overall_mean = float(np.mean([r["mean"] for r in result.values()]))
    return {"per_layer": result, "overall_mean": overall_mean}


# ── Metric 2: W_OV Contribution Top-1 Concentration ─────────────────

@torch.no_grad()
def measure_contribution_concentration(model, processor, model_cfg, samples, device,
                                        deep_layers, bounds):
    """Measure top-1 contribution concentration in vision range.

    Returns per-layer mean top-1 share and entropy of contribution distribution.
    """
    vs = bounds["vision_start"]
    ve = bounds["vision_end"]

    hook_mgr = SinkVerificationHookManager(model, model_cfg)
    hook_mgr.register_hooks()

    per_layer_top1 = {l: [] for l in deep_layers}
    per_layer_entropy = {l: [] for l in deep_layers}

    for si, sample in enumerate(samples):
        prompt = model_cfg.prompt_template.format(instruction=sample["instruction"])
        inputs = call_processor(
            processor, prompt, sample["image"], model_cfg, return_tensors="pt"
        ).to(device)
        if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

        fwd_kwargs = {k: v for k, v in inputs.items()}
        fwd_kwargs["use_cache"] = False
        fwd_kwargs["output_attentions"] = True

        if model_cfg.architecture == "gemma2" and "intrinsic" not in fwd_kwargs:
            fwd_kwargs["intrinsic"] = torch.tensor(
                [[[218.26, 0.0, 111.83], [0.0, 218.26, 111.79], [0.0, 0.0, 1.0]]],
                device=device, dtype=torch.float32,
            )

        hook_mgr.attention_weights.clear()
        hook_mgr.hidden_states.clear()
        model(**fwd_kwargs)

        seq_len = inputs["input_ids"].shape[1]
        qpos = seq_len - 1

        for l in deep_layers:
            if l not in hook_mgr.attention_weights or l not in hook_mgr.hidden_states:
                continue

            attn = hook_mgr.attention_weights[l]  # (H, seq, seq)
            hidden = hook_mgr.hidden_states.get(l - 1)  # Previous layer hidden
            if hidden is None:
                # Use current layer input (layer 0 edge case)
                hidden = hook_mgr.hidden_states.get(l)
                if hidden is None:
                    continue

            try:
                v_weight, o_weight = get_wov_matrix(model, model_cfg, l)
                contrib = compute_perhead_contribution(
                    attn, hidden, v_weight, o_weight, [qpos]
                )  # (H, 1, seq)

                # Vision range contribution
                c_vision = contrib[:, 0, vs:ve]  # (H, n_vision)
                c_sum = c_vision.sum(dim=-1, keepdim=True) + 1e-10
                c_norm = c_vision / c_sum  # (H, n_vision)

                # Top-1 share per head
                top1 = c_norm.max(dim=-1).values  # (H,)
                per_layer_top1[l].append(top1.mean().item())

                # Entropy per head
                log_c = torch.log(c_norm + 1e-10)
                ent = -(c_norm * log_c).sum(dim=-1)  # (H,)
                per_layer_entropy[l].append(ent.mean().item())

            except Exception as e:
                if si == 0:
                    print(f"  WARNING: Contribution failed at layer {l}: {e}")
                continue

    hook_mgr.remove_hooks()

    result = {}
    for l in deep_layers:
        t1 = per_layer_top1[l]
        ent = per_layer_entropy[l]
        result[l] = {
            "top1_share_mean": float(np.mean(t1)) if t1 else 0.0,
            "top1_share_std": float(np.std(t1)) if t1 else 0.0,
            "entropy_mean": float(np.mean(ent)) if ent else 0.0,
            "entropy_std": float(np.std(ent)) if ent else 0.0,
            "n": len(t1),
        }

    overall_top1 = float(np.mean([r["top1_share_mean"] for r in result.values()]))
    overall_entropy = float(np.mean([r["entropy_mean"] for r in result.values()]))
    return {
        "per_layer": result,
        "overall_top1_share": overall_top1,
        "overall_entropy": overall_entropy,
    }


# ── Metric 3: Taxonomy Classification ───────────────────────────────

def classify_routing_type(v0_attn_results, contrib_results):
    """Classify routing type based on attention + contribution patterns.

    Rules (from ADAPTIVE_D2_RESULTS.md):
      IF top1_share > 0.8 AND v0_attn > 0.5 → Bottleneck
      IF v0_attn > 0.5 AND top1_share < 0.5 → Sink (attention without contribution)
      IF v0_attn < 0.3 → Normal
      ELSE → Coexist
    """
    v0_mean = v0_attn_results["overall_mean"]
    top1_mean = contrib_results["overall_top1_share"]
    entropy = contrib_results["overall_entropy"]

    if top1_mean > 0.6 and v0_mean > 0.4:
        taxonomy = "bottleneck"
    elif v0_mean > 0.4 and top1_mean < 0.4:
        taxonomy = "sink"
    elif v0_mean < 0.25:
        taxonomy = "normal"
    else:
        taxonomy = "coexist"

    return {
        "taxonomy": taxonomy,
        "vision0_attn_share": round(v0_mean, 4),
        "top1_contribution_share": round(top1_mean, 4),
        "contribution_entropy": round(entropy, 4),
    }


# ── Load Pre-trained Results ─────────────────────────────────────────

def load_pretrained_results(model_name: str):
    """Load pre-trained analysis results from existing JSON files."""
    contrib_path = config.OUTPUT_DIR / "phase3_gate" / model_name / "contribution_report.json"
    if not contrib_path.exists():
        return None

    with open(contrib_path) as f:
        data = json.load(f)

    deep_layers = data.get("deep_layers", list(range(22, 32)))
    layer_analysis = data.get("layer_analysis", {})

    # Extract metrics from existing data
    v0_shares = []
    top1_shares = []
    for l_str, la in layer_analysis.items():
        l = int(l_str)
        if l in deep_layers:
            a_peak = la.get("a_peak", {})
            c_peak = la.get("c_peak", {})
            # Vision[0] attention share
            if a_peak.get("token_type") == "vision" and a_peak.get("vision_j", -1) == 0:
                v0_shares.append(a_peak.get("a_share", 0))
            # Top-1 contribution share
            top1_shares.append(c_peak.get("c_share", 0))

    return {
        "vision0_attn_share": float(np.mean(v0_shares)) if v0_shares else 0.0,
        "top1_contribution_share": float(np.mean(top1_shares)) if top1_shares else 0.0,
        "contribution_entropy": 0.0,  # Not stored in old format, compute fresh
        "source": "contribution_report.json",
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pre-trained vs Fine-tuned Attention Comparison"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lora_path", default=None,
                        help="Path to LoRA adapter. If None, auto-detect from outputs/libero_ft/")
    parser.add_argument("--ft_model", default=None,
                        help="Registry name for full FT model (e.g. openvla-7b-ft-libero). "
                             "Use instead of --lora_path for official merged FT models.")
    parser.add_argument("--n_samples", type=int, default=20,
                        help="Number of LIBERO samples for analysis")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--pretrained_only", action="store_true",
                        help="Only run pre-trained analysis (for re-measurement)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else \
        config.OUTPUT_DIR / "ft_attention_analysis" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Attention Routing Analysis: {args.model}")
    print(f"  Device: {args.device}")
    print(f"  N samples: {args.n_samples}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}\n", flush=True)

    # ── Load LIBERO samples ──
    print("Loading LIBERO analysis samples...", flush=True)
    samples = load_libero_samples(args.suite, n=args.n_samples)

    # ── Analysis function ──
    def run_analysis(model, processor, model_cfg, label: str):
        print(f"\n--- {label} Analysis ---", flush=True)
        deep_layers = list(range(
            max(0, model_cfg.num_layers - 10), model_cfg.num_layers
        ))
        bounds = detect_token_boundaries(
            processor, model, samples[0]["image"],
            samples[0]["instruction"], args.device, model_cfg,
        )
        print(f"  Deep layers: {deep_layers}")
        print(f"  Vision range: [{bounds['vision_start']}:{bounds['vision_end']}]")

        print("  Measuring vision[0] attention share...", flush=True)
        t0 = time.time()
        v0_results = measure_vision0_attention(
            model, processor, model_cfg, samples, args.device,
            deep_layers, bounds,
        )
        print(f"  Vision[0] attention share: {v0_results['overall_mean']:.4f} "
              f"({time.time()-t0:.1f}s)")

        print("  Measuring contribution concentration...", flush=True)
        t0 = time.time()
        contrib_results = measure_contribution_concentration(
            model, processor, model_cfg, samples, args.device,
            deep_layers, bounds,
        )
        print(f"  Top-1 contribution share: {contrib_results['overall_top1_share']:.4f}")
        print(f"  Contribution entropy:     {contrib_results['overall_entropy']:.4f} "
              f"({time.time()-t0:.1f}s)")

        taxonomy = classify_routing_type(v0_results, contrib_results)
        print(f"  Taxonomy: {taxonomy['taxonomy']}")

        return {
            "vision0_attention": v0_results,
            "contribution": contrib_results,
            "taxonomy": taxonomy,
        }

    # ── Pre-trained analysis ──
    print("Loading pre-trained model...", flush=True)
    processor, model, model_cfg = load_model_from_registry(args.model, args.device)
    model.eval()

    pretrained_results = run_analysis(model, processor, model_cfg, "Pre-trained")

    # Save pre-trained results
    pre_path = out_dir / "pretrained_analysis.json"
    with open(pre_path, "w") as f:
        json.dump(pretrained_results, f, indent=2, default=str)
    print(f"\n  Saved pre-trained results: {pre_path}")

    if args.pretrained_only:
        print("\n  --pretrained_only: skipping fine-tuned analysis.")
        return

    # ── Fine-tuned analysis ──
    if args.ft_model:
        # Full FT model from registry (e.g. official openvla-7b-finetuned-libero-spatial)
        print(f"\nLoading full FT model: {args.ft_model}...", flush=True)
        del model
        torch.cuda.empty_cache()
        processor_ft, model_ft, model_cfg_ft = load_model_from_registry(args.ft_model, args.device)
        model_ft.eval()
        # Use FT model's config for analysis
        model_cfg = model_cfg_ft
        processor = processor_ft
        lora_path = f"hf:{args.ft_model}"
        print(f"  Full FT model loaded.")
    else:
        lora_path = args.lora_path
        if lora_path is None:
            # Auto-detect
            auto_path = config.OUTPUT_DIR / "libero_ft" / args.model / args.suite / "lora_adapter"
            if auto_path.exists():
                lora_path = str(auto_path)
            else:
                print(f"\n  ERROR: No LoRA adapter found at {auto_path}")
                print(f"  Run fine-tuning first or specify --lora_path or --ft_model")
                return

        print(f"\nLoading fine-tuned model (LoRA: {lora_path})...", flush=True)
        from peft import PeftModel
        model_ft = PeftModel.from_pretrained(model, lora_path)
        model_ft.eval()
        print(f"  LoRA loaded.")

    finetuned_results = run_analysis(model_ft, processor, model_cfg, "Fine-tuned")

    # Save fine-tuned results
    ft_path = out_dir / "finetuned_analysis.json"
    with open(ft_path, "w") as f:
        json.dump(finetuned_results, f, indent=2, default=str)

    # ── Comparison ──
    pre_tax = pretrained_results["taxonomy"]
    ft_tax = finetuned_results["taxonomy"]

    comparison = {
        "model": args.model,
        "lora_path": lora_path,
        "suite": args.suite,
        "n_samples": args.n_samples,
        "pretrained": pre_tax,
        "finetuned": ft_tax,
        "delta": {
            "vision0_attn_share": round(
                ft_tax["vision0_attn_share"] - pre_tax["vision0_attn_share"], 4
            ),
            "top1_contribution_share": round(
                ft_tax["top1_contribution_share"] - pre_tax["top1_contribution_share"], 4
            ),
            "contribution_entropy": round(
                ft_tax["contribution_entropy"] - pre_tax["contribution_entropy"], 4
            ),
            "taxonomy_changed": pre_tax["taxonomy"] != ft_tax["taxonomy"],
        },
    }

    comp_path = out_dir / "comparison.json"
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)

    # ── Print Summary ──
    print(f"\n{'='*60}")
    print(f"  COMPARISON: {args.model}")
    print(f"{'='*60}")
    print(f"  {'Metric':<30} {'Pre-trained':>12} {'Fine-tuned':>12} {'Delta':>10}")
    print(f"  {'-'*64}")
    print(f"  {'Vision[0] Attn Share':<30} "
          f"{pre_tax['vision0_attn_share']:>12.4f} "
          f"{ft_tax['vision0_attn_share']:>12.4f} "
          f"{comparison['delta']['vision0_attn_share']:>+10.4f}")
    print(f"  {'Top-1 Contrib Share':<30} "
          f"{pre_tax['top1_contribution_share']:>12.4f} "
          f"{ft_tax['top1_contribution_share']:>12.4f} "
          f"{comparison['delta']['top1_contribution_share']:>+10.4f}")
    print(f"  {'Contrib Entropy':<30} "
          f"{pre_tax['contribution_entropy']:>12.4f} "
          f"{ft_tax['contribution_entropy']:>12.4f} "
          f"{comparison['delta']['contribution_entropy']:>+10.4f}")
    print(f"  {'Taxonomy':<30} "
          f"{pre_tax['taxonomy']:>12} "
          f"{ft_tax['taxonomy']:>12} "
          f"{'CHANGED!' if comparison['delta']['taxonomy_changed'] else 'same':>10}")
    print(f"{'='*60}")
    print(f"  Results saved to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
