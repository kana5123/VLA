#!/usr/bin/env python3
"""Recompute skill probe (N=300, balanced) + D2/D3 with corrected get_action_logits.

Usage:
  python run_recompute_all.py --model ecot-7b --device cuda:1
  python run_recompute_all.py --model openvla-7b --device cuda:3
  python run_recompute_all.py --model spatialvla-4b --device cuda:7
  python run_recompute_all.py --model tracevla-phi3v --device cuda:4
"""
import argparse
import json
import sys
import torch
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

import config
from extract_attention import (
    load_model_from_registry, get_layers, call_processor, detect_token_boundaries,
)
from verify_attention_sinks import SinkVerificationHookManager, get_wov_matrix
from data_sampler import load_balanced_samples, save_sample_list

from contribution.compute import extract_sample_contributions, compute_candidate_frequency
from contribution.classify import classify_layer, compute_phi_all_tokens, compute_mismatch
from contribution.classify import classify_layer_dual_track, compute_phi_universal
from contribution.signature import (
    label_skill_from_instruction, compute_skill_signature,
    compute_within_between_distance, run_linear_probe,
)

from run_phase3_exp_de import (
    get_action_logits, AUGMENTATIONS, compute_output_kl,
)
from contribution.causal import ValueZeroHook


def run_skill_probe(model, model_cfg, processor, device, samples,
                    output_dir, top_k=5):
    """Run contribution analysis + skill probe with balanced samples."""
    print(f"\n{'='*70}")
    print(f"SKILL PROBE (N={len(samples)}, balanced)")
    print(f"{'='*70}")

    boundaries = detect_token_boundaries(
        processor, model, samples[0]["image"], samples[0]["instruction"],
        device, model_cfg,
    )
    print(f"  Boundaries: vis=[{boundaries['vision_start']}:{boundaries['vision_end']}], "
          f"text=[{boundaries.get('text_start', boundaries['vision_end'])}:{boundaries['text_end']}]")

    n_layers = model_cfg.num_layers
    deep_layers = list(range(max(0, n_layers - 10), n_layers))
    text_end = boundaries["text_end"]
    n_query = min(model_cfg.action_tokens or 4, 4)

    all_signatures = []
    all_skill_labels = []
    all_mismatches = []
    all_layer_dual_track = {l: [] for l in deep_layers}
    all_layer_top1 = {l: [] for l in deep_layers}

    tokenizer = getattr(processor, 'tokenizer', None)
    if tokenizer is None and hasattr(processor, 'decode'):
        tokenizer = processor

    hook_mgr = SinkVerificationHookManager(model, model_cfg)

    for si, sample in enumerate(samples):
        if si % 50 == 0:
            print(f"\n  [{si+1}/{len(samples)}] Processing...", flush=True)

        skill = sample.get("skill") or label_skill_from_instruction(sample["instruction"])
        all_skill_labels.append(skill)

        hook_mgr.register_hooks()
        hook_mgr.reset()
        prompt = model_cfg.prompt_template.format(instruction=sample["instruction"])
        inputs = call_processor(processor, prompt, sample["image"],
                                model_cfg, return_tensors="pt").to(device)
        if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

        with torch.no_grad():
            model(**inputs, output_attentions=True)

        attn_weights = hook_mgr.attention_weights
        hidden_states = hook_mgr.hidden_states
        hook_mgr.remove_hooks()

        sample_boundaries = detect_token_boundaries(
            processor, model, sample["image"], sample["instruction"], device, model_cfg,
        )
        text_end_s = sample_boundaries["text_end"]
        query_pos_s = list(range(max(0, text_end_s - n_query), text_end_s))

        results = extract_sample_contributions(
            attention_weights=attn_weights,
            hidden_states=hidden_states,
            get_wov_fn=get_wov_matrix,
            model=model,
            model_cfg=model_cfg,
            boundaries=sample_boundaries,
            query_positions=query_pos_s,
            top_k=top_k,
            target_layers=deep_layers,
        )

        sample_c_tildes = []
        sample_input_ids = inputs.get("input_ids", None)
        if sample_input_ids is not None:
            sample_input_ids = sample_input_ids[0].tolist()

        for r in results:
            l = r.layer_idx
            if l in all_layer_top1:
                all_layer_top1[l].append(float(r.c_tilde.max()))
                mismatch = compute_mismatch(r.a_tilde, r.c_tilde)
                all_mismatches.append(mismatch)
                sample_c_tildes.append(r.c_tilde)

                dual = classify_layer_dual_track(
                    r.a_tilde, r.c_tilde, sample_boundaries,
                    hidden_states_layer=hidden_states.get(l),
                    input_ids=sample_input_ids, tokenizer=tokenizer,
                )
                all_layer_dual_track[l].append(dual)

        if sample_c_tildes:
            sig = compute_skill_signature(sample_c_tildes)
            all_signatures.append(sig)

    # Aggregate
    skill_dist = Counter(all_skill_labels)
    print(f"\n  Skill distribution: {dict(skill_dist)}")

    # Probe
    valid_sigs = [(sig, lab) for sig, lab in zip(all_signatures, all_skill_labels)
                  if lab != "unknown"]
    sig_analysis = {"n_valid": len(valid_sigs), "skill_distribution": dict(skill_dist)}

    if len(valid_sigs) >= 4:
        sigs = [s for s, _ in valid_sigs]
        labs = [l for _, l in valid_sigs]
        d_within, d_between = compute_within_between_distance(sigs, labs)
        sig_analysis["d_within"] = d_within
        sig_analysis["d_between"] = d_between
        sig_analysis["signature_exists"] = d_within < d_between

        unique_labels = sorted(set(labs))
        if len(unique_labels) >= 2:
            max_len = max(len(s) for s in sigs)
            padded_sigs = [np.pad(s, (0, max_len - len(s)), constant_values=0.0)
                           if len(s) < max_len else s for s in sigs]
            X = np.stack(padded_sigs)
            y = np.array([unique_labels.index(l) for l in labs])
            probe_acc = run_linear_probe(X, y)
            sig_analysis["probe_accuracy"] = probe_acc
            sig_analysis["feature_dim"] = X.shape[1]
            sig_analysis["n_classes"] = len(unique_labels)
            sig_analysis["classes"] = unique_labels
            sig_analysis["samples_per_class"] = {
                lbl: int((y == i).sum()) for i, lbl in enumerate(unique_labels)
            }
            print(f"\n  Probe accuracy: {probe_acc:.4f} ({len(valid_sigs)} samples, "
                  f"{len(unique_labels)} classes, feature_dim={X.shape[1]})")

    # Layer top1 shares
    layer_mean_top1 = {}
    for l in deep_layers:
        if all_layer_top1[l]:
            layer_mean_top1[l] = float(np.mean(all_layer_top1[l]))

    report = {
        "n_samples": len(samples),
        "deep_layers": deep_layers,
        "skill_signature": sig_analysis,
        "mean_mismatch": float(np.mean(all_mismatches)) if all_mismatches else 0.0,
        "layer_mean_top1": {str(l): v for l, v in layer_mean_top1.items()},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "skill_probe_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Saved: {output_dir / 'skill_probe_report.json'}")

    return report


def run_d2_recompute(model, model_cfg, processor, device, samples,
                     output_dir, bounds_cache):
    """Recompute D2 (augmentation consistency) with fixed get_action_logits."""
    print(f"\n{'='*70}")
    print(f"D2 RECOMPUTE (N={len(samples)})")
    print(f"{'='*70}")

    results = []
    for si, sample in enumerate(samples):
        bounds = bounds_cache[si]
        logits_orig, _ = get_action_logits(model, processor, model_cfg,
                                           sample, device, bounds)
        top1_orig = logits_orig.argmax().item()

        aug_results = []
        for aug_name, aug_fn in AUGMENTATIONS:
            aug_sample = {**sample, "image": aug_fn(sample["image"])}
            logits_aug, _ = get_action_logits(model, processor, model_cfg,
                                              aug_sample, device, bounds)
            top1_aug = logits_aug.argmax().item()
            aug_results.append({
                "aug_name": aug_name,
                "top1": top1_aug,
                "matches": top1_aug == top1_orig,
                "kl": compute_output_kl(logits_orig, logits_aug),
            })

        consistency = float(np.mean([a["matches"] for a in aug_results]))
        mean_kl = float(np.mean([a["kl"] for a in aug_results]))
        results.append({
            "sample_idx": si,
            "skill": sample.get("skill", "unknown"),
            "orig_top1": top1_orig,
            "consistency_rate": consistency,
            "mean_augmentation_kl": mean_kl,
        })
        if (si + 1) % 10 == 0:
            print(f"  D2 [{si+1}/{len(samples)}] consistency={consistency:.2f} kl={mean_kl:.3f}",
                  flush=True)

    mean_cons = float(np.mean([r["consistency_rate"] for r in results]))
    mean_kl = float(np.mean([r["mean_augmentation_kl"] for r in results]))
    summary = {
        "d2_mean_consistency": mean_cons,
        "d2_mean_aug_kl": mean_kl,
        "n_samples": len(results),
        "per_skill": {},
    }
    for skill in sorted(set(r["skill"] for r in results)):
        skill_results = [r for r in results if r["skill"] == skill]
        summary["per_skill"][skill] = {
            "consistency": float(np.mean([r["consistency_rate"] for r in skill_results])),
            "kl": float(np.mean([r["mean_augmentation_kl"] for r in skill_results])),
            "n": len(skill_results),
        }

    print(f"\n  D2 Summary: mean_consistency={mean_cons:.4f}, mean_kl={mean_kl:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "d2_recompute.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "d2_recompute_detail.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {output_dir / 'd2_recompute.json'}")

    return summary


def run_d3_recompute(model, model_cfg, processor, device, samples,
                     output_dir, bounds_cache):
    """Recompute D3 (V[0] ablation KL + flip rate) with fixed code."""
    print(f"\n{'='*70}")
    print(f"D3 RECOMPUTE (N={len(samples)})")
    print(f"{'='*70}")

    results = []
    for si, sample in enumerate(samples):
        bounds = bounds_cache[si]
        vs = bounds["vision_start"]

        # Original logits
        logits_orig, _ = get_action_logits(model, processor, model_cfg,
                                           sample, device, bounds)

        # V[0] ablation
        n_layers = model_cfg.num_layers
        deep_layers = list(range(max(0, n_layers - 10), n_layers))

        hook = ValueZeroHook(model, model_cfg, target_positions=[vs],
                             target_layers=deep_layers)
        hook.register()
        logits_vz, _ = get_action_logits(model, processor, model_cfg,
                                         sample, device, bounds)
        hook.remove()

        kl = compute_output_kl(logits_orig, logits_vz)
        flip = int(logits_orig.argmax().item() != logits_vz.argmax().item())

        results.append({
            "sample_idx": si,
            "skill": sample.get("skill", "unknown"),
            "kl": kl,
            "flip": flip,
        })
        if (si + 1) % 10 == 0:
            print(f"  D3 [{si+1}/{len(samples)}] kl={kl:.3f} flip={flip}", flush=True)

    mean_kl = float(np.mean([r["kl"] for r in results]))
    flip_rate = float(np.mean([r["flip"] for r in results]))
    summary = {
        "d3_mean_kl": mean_kl,
        "d3_top1_change_rate": flip_rate,
        "n_samples": len(results),
    }
    print(f"\n  D3 Summary: mean_kl={mean_kl:.4f}, flip_rate={flip_rate:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "d3_recompute.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {output_dir / 'd3_recompute.json'}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_per_skill", type=int, default=50,
                        help="Samples per skill for balanced probe (default: 50)")
    parser.add_argument("--n_d2d3", type=int, default=50,
                        help="Samples for D2/D3 recompute (default: 50)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_probe", action="store_true")
    parser.add_argument("--skip_d2d3", action="store_true")
    args = parser.parse_args()

    output_dir = Path("outputs/recompute_v2") / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Loading {args.model} on {args.device} ===", flush=True)
    processor, model, model_cfg = load_model_from_registry(args.model, args.device)
    model.eval()

    # ── Skill Probe ──
    if not args.skip_probe:
        target_skills = ["place", "move", "pick", "fold", "open", "close"]
        samples = load_balanced_samples(
            config.DATA_CACHE_DIR, n_per_skill=args.n_per_skill,
            target_skills=target_skills, seed=args.seed,
        )
        print(f"Loaded {len(samples)} balanced samples")

        sample_list_path = output_dir / "sample_list.json"
        save_sample_list(samples, sample_list_path, args.seed,
                         args.n_per_skill, target_skills)

        run_skill_probe(model, model_cfg, processor, args.device,
                        samples, output_dir)

    # ── D2/D3 Recompute ──
    if not args.skip_d2d3:
        # Use first n_d2d3 from balanced samples (or load separately)
        d2d3_samples = load_balanced_samples(
            config.DATA_CACHE_DIR, n_per_skill=max(args.n_d2d3 // 6, 10),
            target_skills=["place", "move", "pick", "fold", "open", "close"],
            seed=args.seed + 1000,  # Different seed to avoid overlap
        )
        d2d3_samples = d2d3_samples[:args.n_d2d3]
        print(f"\nD2/D3: Using {len(d2d3_samples)} samples")

        # Cache boundaries
        bounds_cache = []
        for s in d2d3_samples:
            b = detect_token_boundaries(
                processor, model, s["image"], s["instruction"],
                args.device, model_cfg,
            )
            bounds_cache.append(b)

        run_d2_recompute(model, model_cfg, processor, args.device,
                         d2d3_samples, output_dir, bounds_cache)
        run_d3_recompute(model, model_cfg, processor, args.device,
                         d2d3_samples, output_dir, bounds_cache)

    print(f"\n{'='*70}")
    print(f"ALL DONE — {args.model}")
    print(f"Results: {output_dir}")
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
