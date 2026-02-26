#!/usr/bin/env python3
"""
Main entry point: Sink vs Bottleneck contribution analysis.

Usage:
  python run_contribution_analysis.py --model openvla-7b --device cuda:0 --n_samples 20
  python run_contribution_analysis.py --model tracevla-phi3v --device cuda:1 --n_samples 20
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
from extract_attention import load_model_from_registry, get_layers, call_processor, detect_token_boundaries
from verify_attention_sinks import SinkVerificationHookManager, get_wov_matrix
from visualize_text_attention import load_samples_from_cache
from model_registry import get_model as registry_get_model

from contribution.compute import (
    extract_sample_contributions,
    compute_candidate_frequency,
    ContributionResult,
)
from contribution.classify import (
    classify_layer,
    compute_phi_all_tokens,
    compute_mismatch,
    classify_layer_dual_track,   # Phase 2.5
    compute_phi_universal,       # Phase 2.5
)
from contribution.signature import (
    label_skill_from_instruction,
    compute_skill_signature,
    compute_within_between_distance,
    run_linear_probe,
)
from contribution.visualize import (
    plot_top1_share_curve,
    plot_frequency_heatmap,
    plot_attention_contribution_scatter,
    plot_skill_js_matrix,
)


def run_analysis(model_name: str, device: str, n_samples: int, top_k: int, output_dir: Path,
                 balanced=False, n_per_skill=25, seed=42, sample_list_path=None,
                 target_skills=None):
    print(f"\n{'='*70}")
    print(f"Contribution Analysis — {model_name} (Phase 2.5: Dual-Track)")
    print(f"{'='*70}")

    # ── Load model ──────────────────────────────────────────────
    processor, model, model_cfg = load_model_from_registry(model_name, device)
    # Tokenizer for token identity decode (Phase 2.5)
    tokenizer = getattr(processor, 'tokenizer', None)
    if tokenizer is None and hasattr(processor, 'decode'):
        tokenizer = processor  # processor itself has decode
    # ── Load samples ─────────────────────────────────────────────
    if sample_list_path and Path(sample_list_path).exists():
        from data_sampler import reload_samples_from_list
        samples = reload_samples_from_list(sample_list_path, config.DATA_CACHE_DIR)
        print(f"  Loaded {len(samples)} samples from {sample_list_path}")
    elif balanced:
        from data_sampler import load_balanced_samples, save_sample_list
        samples = load_balanced_samples(
            config.DATA_CACHE_DIR, n_per_skill=n_per_skill,
            target_skills=target_skills, seed=seed,
        )
        print(f"  Loaded {len(samples)} balanced samples ({n_per_skill}/skill)")
        if sample_list_path:
            save_sample_list(samples, sample_list_path, seed, n_per_skill, target_skills)
    else:
        samples = load_samples_from_cache(config.DATA_CACHE_DIR, n_samples=n_samples)
        print(f"  Loaded {len(samples)} samples (legacy)")

    # ── Detect boundaries from first sample ─────────────────────
    boundaries = detect_token_boundaries(
        processor, model, samples[0]["image"], samples[0]["instruction"], device, model_cfg
    )
    print(f"  Boundaries: vis=[{boundaries['vision_start']}:{boundaries['vision_end']}], "
          f"text=[{boundaries.get('text_start', boundaries['vision_end'])}:{boundaries['text_end']}]")

    # ── Determine query positions (last N text positions as proxy for action tokens) ──
    text_end = boundaries["text_end"]
    n_query = min(model_cfg.action_tokens or 4, 4)
    query_positions = list(range(max(0, text_end - n_query), text_end))
    print(f"  Query positions (action proxy): {query_positions}")

    # ── Per-sample extraction ───────────────────────────────────
    n_layers = model_cfg.num_layers
    deep_layers = list(range(max(0, n_layers - 10), n_layers))

    all_layer_top1 = {l: [] for l in deep_layers}
    all_layer_topk_c = {l: [] for l in deep_layers}
    all_layer_classifications = {l: [] for l in deep_layers}
    all_signatures = []
    all_skill_labels = []
    all_mismatches = []
    # Phase 2.5: Dual-track per-layer per-sample
    all_layer_dual_track = {l: [] for l in deep_layers}

    hook_mgr = SinkVerificationHookManager(model, model_cfg)

    for si, sample in enumerate(samples):
        print(f"\n  [{si+1}/{len(samples)}] \"{sample['instruction'][:60]}...\"")

        # Skill label
        skill = label_skill_from_instruction(sample["instruction"])
        all_skill_labels.append(skill)

        # Forward pass with hooks
        hook_mgr.register_hooks()
        hook_mgr.reset()
        prompt = model_cfg.prompt_template.format(instruction=sample["instruction"])
        inputs = call_processor(processor, prompt, sample["image"], model_cfg, return_tensors="pt").to(device)
        if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

        with torch.no_grad():
            model(**inputs, output_attentions=True)

        attn_weights = hook_mgr.attention_weights
        hidden_states = hook_mgr.hidden_states
        hook_mgr.remove_hooks()

        # Per-sample boundaries (may differ due to instruction length)
        sample_boundaries = detect_token_boundaries(
            processor, model, sample["image"], sample["instruction"], device, model_cfg
        )
        text_end_s = sample_boundaries["text_end"]
        query_pos_s = list(range(max(0, text_end_s - n_query), text_end_s))

        # Extract contributions
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

        # Collect per-layer stats
        sample_c_tildes = []
        # Phase 2.5: Extract input_ids once per sample (not per layer)
        sample_input_ids = inputs.get("input_ids", None)
        if sample_input_ids is not None:
            sample_input_ids = sample_input_ids[0].tolist()

        for r in results:
            l = r.layer_idx
            if l in all_layer_top1:
                top1 = float(r.c_tilde.max())
                all_layer_top1[l].append(top1)
                all_layer_topk_c[l].append(r.topk_contribution)

                # Classify
                phi = compute_phi_all_tokens(hidden_states[l])
                cls_result = classify_layer(r.a_tilde, r.c_tilde, sample_boundaries, phi)
                all_layer_classifications[l].append(cls_result)

                # Phase 2.5: Dual-track classification
                dual = classify_layer_dual_track(
                    r.a_tilde, r.c_tilde,
                    sample_boundaries,
                    hidden_states_layer=hidden_states.get(l),
                    input_ids=sample_input_ids,
                    tokenizer=tokenizer,
                )
                all_layer_dual_track[l].append(dual)

                mismatch = compute_mismatch(r.a_tilde, r.c_tilde)
                all_mismatches.append(mismatch)

                sample_c_tildes.append(r.c_tilde)

        # Skill signature (from deep layers)
        if sample_c_tildes:
            sig = compute_skill_signature(sample_c_tildes)
            all_signatures.append(sig)

        print(f"    Skill={skill}, layers_extracted={len(results)}")

    # ── Aggregate results ───────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Layer-wise top1 share
    layer_mean_top1 = {}
    for l in deep_layers:
        if all_layer_top1[l]:
            layer_mean_top1[l] = float(np.mean(all_layer_top1[l]))

    # 2. Frequency analysis (per deep layer)
    seq_len = boundaries["total_seq_len"]
    layer_freq = {}
    for l in deep_layers:
        if all_layer_topk_c[l]:
            freq = compute_candidate_frequency(all_layer_topk_c[l], seq_len)
            layer_freq[l] = freq

    # 3. Dominant classification per layer
    layer_dominant = {}
    for l in deep_layers:
        if all_layer_classifications[l]:
            types = [c["dominant_type"] for c in all_layer_classifications[l]]
            most_common = Counter(types).most_common(1)[0]
            layer_dominant[l] = {
                "dominant_type": most_common[0],
                "frequency": most_common[1] / len(types),
                "mean_mismatch": float(np.mean([c["mismatch"] for c in all_layer_classifications[l]])),
                "mean_entropy": float(np.mean([c["entropy"] for c in all_layer_classifications[l]])),
                "mean_top1_share": float(np.mean([c["top1_share"] for c in all_layer_classifications[l]])),
            }

    # Phase 2.5: Aggregate dual-track per layer
    layer_dual_track_agg = {}
    for l in deep_layers:
        duals = all_layer_dual_track.get(l, [])
        if not duals:
            continue

        # Dominant type by vote
        types = [d["dominant_type"] for d in duals]
        dominant_vote = Counter(types).most_common(1)[0]

        # A-peak: most common position
        a_positions = [d["a_peak"]["abs_t"] for d in duals]
        a_dom_pos = Counter(a_positions).most_common(1)[0][0]

        # C-peak: most common position
        c_positions = [d["c_peak"]["abs_t"] for d in duals]
        c_dom_pos = Counter(c_positions).most_common(1)[0][0]

        # R-peak: most common position
        r_positions = [d["r_peak"]["abs_t"] for d in duals]
        r_dom_pos = Counter(r_positions).most_common(1)[0][0]

        # Find representative sample for each peak
        def find_representative(duals, peak_key, dom_pos):
            for d in duals:
                if d[peak_key]["abs_t"] == dom_pos:
                    return d[peak_key]
            return duals[0][peak_key]

        a_peak_rep = find_representative(duals, "a_peak", a_dom_pos)
        c_peak_rep = find_representative(duals, "c_peak", c_dom_pos)
        r_peak_rep = find_representative(duals, "r_peak", r_dom_pos)

        # Average metrics
        a_c_match_rate = sum(1 for d in duals if d["a_c_match"]) / len(duals)

        layer_dual_track_agg[l] = {
            "dominant_type": dominant_vote[0],
            "frequency": dominant_vote[1] / len(duals),
            "a_peak": a_peak_rep,
            "c_peak": c_peak_rep,
            "r_peak": r_peak_rep,
            "a_c_match": a_c_match_rate > 0.5,
            "a_c_match_rate": a_c_match_rate,
            "mean_entropy": float(np.mean([d["mean_entropy"] for d in duals])),
            "mean_top1_share": float(np.mean([d["mean_top1_share"] for d in duals])),
            "mean_mismatch": float(np.mean([d["mismatch"] for d in duals])),
        }

    # ── Mode token extraction (B4) ─────────────────────────────
    # Find the most frequent peak position across deep layers
    from collections import Counter as _Counter

    def _extract_mode_token(all_dual_track, peak_key, deep_layers):
        """Extract the mode (most frequent) peak position across deep layers."""
        positions = []
        for l in deep_layers:
            for dual in all_dual_track.get(l, []):
                positions.append(dual[peak_key]["abs_t"])
        if not positions:
            return {"abs_t": -1, "freq": 0.0}
        counter = _Counter(positions)
        mode_pos, mode_count = counter.most_common(1)[0]
        # Find a representative entry to get token_str
        token_str = None
        for l in deep_layers:
            for dual in all_dual_track.get(l, []):
                if dual[peak_key]["abs_t"] == mode_pos:
                    token_str = dual[peak_key].get("token_str")
                    break
            if token_str is not None:
                break
        # freq = fraction of deep layers where this position is the peak
        # Average across all samples: for each sample, count how many layers have this peak
        per_sample_freqs = []
        n_deep = len(deep_layers)
        n_samples_seen = len(all_dual_track.get(deep_layers[0], [])) if deep_layers else 0
        for si in range(n_samples_seen):
            count = 0
            for l in deep_layers:
                duals = all_dual_track.get(l, [])
                if si < len(duals) and duals[si][peak_key]["abs_t"] == mode_pos:
                    count += 1
            per_sample_freqs.append(count / n_deep if n_deep > 0 else 0)
        freq = float(np.mean(per_sample_freqs)) if per_sample_freqs else 0.0
        return {"abs_t": mode_pos, "token_str": token_str, "freq": freq}

    mode_tokens = {
        "A_mode": _extract_mode_token(all_layer_dual_track, "a_peak", deep_layers),
        "C_mode": _extract_mode_token(all_layer_dual_track, "c_peak", deep_layers),
        "R_mode": _extract_mode_token(all_layer_dual_track, "r_peak", deep_layers),
    }

    # CRITICAL FIX (핵심1): Add token_type (vision/text/pre_vision) to each mode token.
    # Gate① pass check needs this to verify "OpenVLA: A_mode ∈ vision, C_mode ∈ text".
    for peak_key in ["A_mode", "C_mode", "R_mode"]:
        mode_pos = mode_tokens[peak_key]["abs_t"]
        vs = boundaries.get("vision_start", 0)
        ve = boundaries.get("vision_end", 0)
        if mode_pos < vs:
            mode_tokens[peak_key]["token_type"] = "pre_vision"
        elif mode_pos < ve:
            mode_tokens[peak_key]["token_type"] = "vision"
        else:
            mode_tokens[peak_key]["token_type"] = "text"

    # 4. Skill signature analysis
    sig_analysis = {}
    valid_sigs = [(sig, lab) for sig, lab in zip(all_signatures, all_skill_labels) if lab != "unknown"]
    if len(valid_sigs) >= 4:
        sigs = [s for s, _ in valid_sigs]
        labs = [l for _, l in valid_sigs]
        d_within, d_between = compute_within_between_distance(sigs, labs)
        sig_analysis["d_within"] = d_within
        sig_analysis["d_between"] = d_between
        sig_analysis["signature_exists"] = d_within < d_between

        # Linear probe (pad signatures to common length for np.stack)
        unique_labels = sorted(set(labs))
        if len(unique_labels) >= 2:
            max_len = max(len(s) for s in sigs)
            padded_sigs = [np.pad(s, (0, max_len - len(s)), constant_values=0.0) if len(s) < max_len else s for s in sigs]
            X = np.stack(padded_sigs)
            y = np.array([unique_labels.index(l) for l in labs])
            probe_acc = run_linear_probe(X, y)
            sig_analysis["probe_accuracy"] = probe_acc

    # ── Save report ─────────────────────────────────────────────
    report = {
        "model": model_name,
        "phase": 2.5,
        "n_samples": len(samples),
        "n_layers": n_layers,
        "deep_layers": deep_layers,
        "boundaries": {k: int(v) if isinstance(v, (int, np.integer)) else v
                      for k, v in boundaries.items()},
        "layer_analysis": {str(l): layer_dual_track_agg.get(l, {}) for l in deep_layers},
        "skill_signature": sig_analysis,
        "mean_mismatch": float(np.mean(all_mismatches)) if all_mismatches else 0.0,
        "skill_distribution": dict(Counter(all_skill_labels)),
        "mode_tokens": mode_tokens,
    }

    report_path = output_dir / "contribution_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved: {report_path}")

    mode_path = output_dir / "mode_tokens.json"
    with open(mode_path, "w") as f:
        json.dump(mode_tokens, f, indent=2)
    print(f"  Mode tokens saved: {mode_path}")

    # ── Visualizations ──────────────────────────────────────────
    plot_top1_share_curve(
        {model_name: [layer_mean_top1.get(l, 0) for l in deep_layers]},
        output_dir / "top1_contrib_share.png",
    )

    if layer_freq:
        last_layer = max(layer_freq.keys())
        plot_frequency_heatmap(
            layer_freq[last_layer],
            model_cfg.vision_grid_size,
            output_dir / "candidate_frequency.png",
        )

    # Scatter: collect all candidates across layers
    all_a_shares, all_c_shares, all_cls_labels = [], [], []
    for l in deep_layers:
        for cls_result in all_layer_classifications.get(l, []):
            for cand in cls_result.get("candidates", [])[:3]:
                all_a_shares.append(cand["a_share"])
                all_c_shares.append(cand["c_share"])
                all_cls_labels.append(cand["classification"])

    if all_a_shares:
        plot_attention_contribution_scatter(
            np.array(all_a_shares), np.array(all_c_shares), all_cls_labels,
            output_dir / "attention_vs_contribution.png",
        )

    print(f"\n{'='*70}")
    print(f"  Analysis complete for {model_name}")
    print(f"{'='*70}")

    del model
    torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description="Sink vs Bottleneck Contribution Analysis")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--balanced", action="store_true",
                        help="Use skill-balanced sampling from BridgeData V2 cache")
    parser.add_argument("--n_per_skill", type=int, default=25,
                        help="Samples per skill when --balanced (default: 25)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for balanced sampling")
    parser.add_argument("--sample_list", type=str, default=None,
                        help="Path to save/load sample_list.json (reuse across gates)")
    parser.add_argument("--target_skills", nargs="+",
                        default=["place", "move", "pick", "fold", "open", "close"],
                        help="Target skills for balanced sampling")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else config.OUTPUT_DIR / "contribution_analysis" / args.model
    run_analysis(args.model, args.device, args.n_samples, args.top_k, output_dir,
                 balanced=args.balanced, n_per_skill=args.n_per_skill,
                 seed=args.seed, sample_list_path=args.sample_list,
                 target_skills=args.target_skills)


if __name__ == "__main__":
    main()
