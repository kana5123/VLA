#!/usr/bin/env python3
"""Gate 3: Text Masking Control + Mini Counterfactual.

Part A: Compare hidden probe accuracy under 4 conditions:
  A. Original (baseline from Gate 1)
  B. Text V=0 (value zeroed, Q/K routing alive)
  C. Text KV-mask (fully blocked via 4D mask)
  D. Vision V=0 (normalized -- same token count as text)

Part B: Mini counterfactual under text masking
  20 pairs x 3 verb swaps: pick<->place, open<->close, move<->fold
  Measure delta-hidden under orig / textKV conditions.

CRITICAL FIXES (v2):
  - TextKVMaskHook now creates proper 4D causal mask (was no-op on 2D)
  - Uses boundaries["text_ranges"] for disjoint text regions (Phi3V fix)
  - Logs masked token strings for sanity verification
  - Adds mask sanity check on first sample

Usage:
  python run_gate3_text_mask.py --model ecot-7b --device cuda:4 \
    --gate1_dir outputs/phase3_gate/ecot-7b
"""
import argparse
import json
import sys
import re
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

import config
from extract_attention import load_model_from_registry, get_layers, call_processor, detect_token_boundaries
from data_sampler import reload_samples_from_list
from verify_attention_sinks import SinkVerificationHookManager
from contribution.text_mask import (
    TextValueZeroHook, TextKVMaskHook, TextKVZeroHook,
    create_text_kv_mask, sample_normalized_vision_positions,
)
from contribution.causal import ValueZeroHook

# Gate 3 Part B: verb swap pairs
VERB_PAIRS = [("pick", "place"), ("open", "close"), ("move", "fold")]


def extract_hidden_at_query(model, inputs, model_cfg, query_pos, deep_layers):
    """Run forward and extract hidden states at query position from deep layers.
    Returns dict: {layer: (D,) numpy array}
    """
    hook_mgr = SinkVerificationHookManager(model, model_cfg)
    hook_mgr.register_hooks()
    hook_mgr.reset()
    with torch.no_grad():
        model(**inputs, output_attentions=True)
    hidden_states = hook_mgr.hidden_states
    hook_mgr.remove_hooks()

    result = {}
    for l in deep_layers:
        h = hidden_states.get(l)
        if h is not None and query_pos < h.shape[0]:
            result[l] = h[query_pos].cpu().float().numpy()
    return result


def _get_text_ranges(bounds):
    """Get text_ranges from boundaries, with fallback for old format."""
    if "text_ranges" in bounds:
        return bounds["text_ranges"]
    # Fallback: single contiguous range
    ts = bounds.get("text_start", bounds["vision_end"])
    te = bounds["text_end"]
    return [(ts, te)]


def _count_text_tokens(text_ranges):
    """Count total text tokens from disjoint ranges."""
    return sum(e - s for s, e in text_ranges)


def run_mask_sanity_check(model, processor, model_cfg, sample, device, text_ranges, deep_layers):
    """Verify that text KV masking actually changes model hidden states.

    Uses create_text_kv_mask() factory to pick the right strategy per architecture.
    Returns dict with sanity check results.
    """
    prompt = model_cfg.prompt_template.format(instruction=sample["instruction"])
    inputs = call_processor(processor, prompt, sample["image"], model_cfg, return_tensors="pt").to(device)
    if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

    bounds = detect_token_boundaries(processor, model, sample["image"], sample["instruction"], device, model_cfg)
    qpos = bounds["text_end"] - 1

    # Original forward
    h_orig = extract_hidden_at_query(model, inputs, model_cfg, qpos, deep_layers)

    # Masked forward (factory picks hook vs 4D mask)
    inputs_masked = {k: v.clone() for k, v in inputs.items()}
    attn_hook, apply_mask_fn = create_text_kv_mask(text_ranges, model, model_cfg, get_layers)
    if apply_mask_fn is not None:
        inputs_masked = apply_mask_fn(inputs_masked)
        strategy = "4d_mask"
    else:
        strategy = "kv_zero_hook"

    h_masked = extract_hidden_at_query(model, inputs_masked, model_cfg, qpos, deep_layers)

    if attn_hook is not None:
        hook_fired = attn_hook.fired()
        n_masked = attn_hook.get_n_masked()
        attn_hook.remove()
    else:
        hook_fired = None
        n_masked = sum(e - s for s, e in text_ranges) if isinstance(text_ranges[0], (list, tuple)) else text_ranges[1] - text_ranges[0]

    # Compare
    any_changed = False
    max_diff = 0.0
    for l in deep_layers:
        if l in h_orig and l in h_masked:
            diff = np.linalg.norm(h_orig[l] - h_masked[l])
            max_diff = max(max_diff, diff)
            if diff > 1e-5:
                any_changed = True

    return {
        "strategy": strategy,
        "hook_fired": hook_fired,
        "n_text_tokens_masked": n_masked,
        "hidden_changed": any_changed,
        "max_hidden_diff": float(max_diff),
    }


def run_gate3(model_name, device, gate1_dir, output_dir, n_samples=20):
    """Run Gate 3 for one model."""
    print(f"\n{'='*60}")
    print(f"Gate 3 v2: Text Masking -- {model_name}")
    print(f"{'='*60}")

    gate1_dir = Path(gate1_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    processor, model, model_cfg = load_model_from_registry(model_name, device)
    tokenizer = getattr(processor, 'tokenizer', processor)

    # Load samples from Gate 1
    sample_list_path = gate1_dir / "sample_list.json"
    samples = reload_samples_from_list(sample_list_path, config.DATA_CACHE_DIR)
    samples = samples[:n_samples]
    print(f"  Using {len(samples)} samples from Gate 1")

    # Detect boundaries on first sample for logging
    _s0_bounds = detect_token_boundaries(
        processor, model, samples[0]["image"], samples[0]["instruction"], device, model_cfg
    )
    text_ranges_s0 = _get_text_ranges(_s0_bounds)
    vis_r_s0 = (_s0_bounds["vision_start"], _s0_bounds["vision_end"])
    n_text_s0 = _count_text_tokens(text_ranges_s0)
    print(f"  text_ranges (sample 0): {text_ranges_s0} ({n_text_s0} tokens)")
    print(f"  vision_range (sample 0): {vis_r_s0} ({vis_r_s0[1]-vis_r_s0[0]} tokens)")

    # Log masked token strings for sanity
    kv_hook_s0 = TextKVMaskHook(text_ranges_s0)
    prompt_s0 = model_cfg.prompt_template.format(instruction=samples[0]["instruction"])
    inputs_s0 = call_processor(processor, prompt_s0, samples[0]["image"], model_cfg, return_tensors="pt").to(device)
    if "pixel_values" in inputs_s0 and inputs_s0["pixel_values"].dtype != model.dtype:
        inputs_s0["pixel_values"] = inputs_s0["pixel_values"].to(model.dtype)
    masked_strs = kv_hook_s0.get_masked_token_strs(inputs_s0["input_ids"][0], tokenizer)
    print(f"  Masked token strings (first 10): {masked_strs[:10]}")
    if len(masked_strs) > 10:
        print(f"  ... ({len(masked_strs)} total)")

    # Deep layers
    deep_layers = list(range(max(0, model_cfg.num_layers - 10), model_cfg.num_layers))

    # == Sanity check: verify mask actually changes output ==
    print("\n  Running mask sanity check on sample 0...")
    sanity = run_mask_sanity_check(model, processor, model_cfg, samples[0], device, text_ranges_s0, deep_layers)
    print(f"  Sanity: strategy={sanity['strategy']}, hook_fired={sanity.get('hook_fired')}")
    print(f"  Sanity: n_text_masked={sanity['n_text_tokens_masked']}")
    print(f"  Sanity: hidden_changed={sanity['hidden_changed']}, max_diff={sanity['max_hidden_diff']:.6f}")

    if not sanity["hidden_changed"]:
        print("  WARNING: TextKVMaskHook did NOT change hidden states! Check mask application.")

    # Save sanity check
    with open(output_dir / "mask_sanity_check.json", "w") as f:
        json.dump(sanity, f, indent=2)

    # == Part A: Probe accuracy under 4 conditions ==
    print("\n  Part A: Collecting hidden states under 4 conditions...")
    conditions = {}

    for cond_name in ["original", "text_v0", "text_kv", "vision_v0_norm"]:
        layer_hiddens = defaultdict(list)

        for si, sample in enumerate(samples):
            prompt = model_cfg.prompt_template.format(instruction=sample["instruction"])
            inputs = call_processor(processor, prompt, sample["image"], model_cfg, return_tensors="pt").to(device)
            if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
                inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

            sample_bounds = detect_token_boundaries(
                processor, model, sample["image"], sample["instruction"], device, model_cfg
            )
            query_pos = sample_bounds["text_end"] - 1
            text_ranges = _get_text_ranges(sample_bounds)
            vis_r = (sample_bounds["vision_start"], sample_bounds["vision_end"])
            n_t = _count_text_tokens(text_ranges)

            if cond_name == "text_v0":
                hook = TextValueZeroHook(text_ranges)
                hook.register(model, model_cfg, get_layers)
                h = extract_hidden_at_query(model, inputs, model_cfg, query_pos, deep_layers)
                hook.remove()
            elif cond_name == "text_kv":
                attn_hook, apply_mask_fn = create_text_kv_mask(text_ranges, model, model_cfg, get_layers)
                if apply_mask_fn is not None:
                    inputs = apply_mask_fn(inputs)
                h = extract_hidden_at_query(model, inputs, model_cfg, query_pos, deep_layers)
                if attn_hook is not None:
                    attn_hook.remove()
            elif cond_name == "vision_v0_norm":
                vis_positions = sample_normalized_vision_positions(vis_r, n_t, seed=42 + si)
                hook = ValueZeroHook(vis_positions)
                hook.register(model, model_cfg, get_layers)
                h = extract_hidden_at_query(model, inputs, model_cfg, query_pos, deep_layers)
                hook.remove()
            else:
                h = extract_hidden_at_query(model, inputs, model_cfg, query_pos, deep_layers)

            for l, vec in h.items():
                layer_hiddens[l].append(vec)

        conditions[cond_name] = {l: np.stack(vecs) for l, vecs in layer_hiddens.items()}
        print(f"    {cond_name}: collected {len(samples)} samples x {len(deep_layers)} layers")

    # Save Part A hidden states for offline probe evaluation
    for cond_name, layer_data in conditions.items():
        for l, mat in layer_data.items():
            np.save(output_dir / f"hidden_{cond_name}_layer{l}.npy", mat)

    # Save skill labels for probe
    labels = [s["skill"] for s in samples]
    with open(output_dir / "skill_labels.json", "w") as f:
        json.dump(labels, f)

    print(f"\n  Part A hidden states saved to {output_dir}/")

    # == Part B: Mini counterfactual ==
    print("\n  Part B: Mini counterfactual (verb swap)...")
    cf_results = []

    for si, sample in enumerate(samples):
        skill = sample["skill"]
        # Find matching verb pair
        swap_verb = None
        for v1, v2 in VERB_PAIRS:
            if skill == v1:
                swap_verb = v2
                break
            elif skill == v2:
                swap_verb = v1
                break

        if swap_verb is None:
            continue

        # Generate swapped instruction
        orig_instr = sample["instruction"]
        swapped_instr = re.sub(r'\b' + skill + r'\b', swap_verb, orig_instr.lower(), count=1)
        if swapped_instr == orig_instr.lower():
            continue

        # Forward with original and swapped instruction
        prompt_orig = model_cfg.prompt_template.format(instruction=orig_instr)
        prompt_swap = model_cfg.prompt_template.format(instruction=swapped_instr)

        inputs_orig = call_processor(processor, prompt_orig, sample["image"], model_cfg, return_tensors="pt").to(device)
        inputs_swap = call_processor(processor, prompt_swap, sample["image"], model_cfg, return_tensors="pt").to(device)

        for inp in [inputs_orig, inputs_swap]:
            if "pixel_values" in inp and inp["pixel_values"].dtype != model.dtype:
                inp["pixel_values"] = inp["pixel_values"].to(model.dtype)

        bounds = detect_token_boundaries(
            processor, model, sample["image"], orig_instr, device, model_cfg
        )
        qpos = bounds["text_end"] - 1
        text_ranges = _get_text_ranges(bounds)

        # Condition: original
        h_orig = extract_hidden_at_query(model, inputs_orig, model_cfg, qpos, deep_layers)
        h_swap = extract_hidden_at_query(model, inputs_swap, model_cfg, qpos, deep_layers)

        # Condition: text KV-mask (factory picks hook vs 4D mask)
        inputs_orig_kv = {k: v.clone() for k, v in inputs_orig.items()}
        inputs_swap_kv = {k: v.clone() for k, v in inputs_swap.items()}

        # For hook-based (Prismatic): register once, run both, remove
        attn_hook_o, apply_fn_o = create_text_kv_mask(text_ranges, model, model_cfg, get_layers)
        if apply_fn_o is not None:
            inputs_orig_kv = apply_fn_o(inputs_orig_kv)
        h_orig_kv = extract_hidden_at_query(model, inputs_orig_kv, model_cfg, qpos, deep_layers)
        if attn_hook_o is not None:
            attn_hook_o.remove()

        attn_hook_s, apply_fn_s = create_text_kv_mask(text_ranges, model, model_cfg, get_layers)
        if apply_fn_s is not None:
            inputs_swap_kv = apply_fn_s(inputs_swap_kv)
        h_swap_kv = extract_hidden_at_query(model, inputs_swap_kv, model_cfg, qpos, deep_layers)
        if attn_hook_s is not None:
            attn_hook_s.remove()

        # Compute delta-hidden for each layer
        for l in deep_layers:
            if l not in h_orig or l not in h_swap:
                continue
            norm_orig = np.linalg.norm(h_orig[l])
            delta_orig = np.linalg.norm(h_orig[l] - h_swap[l]) / max(norm_orig, 1e-10)
            delta_kv = 0.0
            if l in h_orig_kv and l in h_swap_kv:
                delta_kv = np.linalg.norm(h_orig_kv[l] - h_swap_kv[l]) / max(np.linalg.norm(h_orig_kv[l]), 1e-10)

            cf_results.append({
                "sample_idx": si,
                "layer": l,
                "skill": skill,
                "swap_verb": swap_verb,
                "delta_orig": float(delta_orig),
                "delta_textKV": float(delta_kv),
            })

    # Save Part B results
    cf_path = output_dir / "counterfactual_results.json"
    with open(cf_path, "w") as f:
        json.dump(cf_results, f, indent=2)
    print(f"  Part B: {len(cf_results)} measurements saved to {cf_path}")

    # Summary
    if cf_results:
        delta_orig_mean = np.mean([r["delta_orig"] for r in cf_results])
        delta_kv_mean = np.mean([r["delta_textKV"] for r in cf_results])
        print(f"\n  Summary: delta_hidden_orig={delta_orig_mean:.4f}, delta_hidden_textKV={delta_kv_mean:.4f}")
        if delta_kv_mean > 1e-10:
            print(f"  Ratio: delta_orig/delta_KV = {delta_orig_mean/delta_kv_mean:.2f}x")
        else:
            print(f"  Ratio: inf (delta_textKV ~ 0)")

    # Save metadata
    meta = {
        "model": model_name,
        "n_samples": len(samples),
        "text_ranges_sample0": text_ranges_s0,
        "n_text_tokens_sample0": n_text_s0,
        "vision_range_sample0": list(vis_r_s0),
        "masked_tokens_sample0": masked_strs[:20],
        "sanity_check": sanity,
        "version": 2,
    }
    with open(output_dir / "gate3_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Gate 3 v2: Text Masking + Mini Counterfactual")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--gate1_dir", required=True,
                        help="Gate 1 output dir for this model")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--n_samples", type=int, default=20)
    args = parser.parse_args()

    out = Path(args.output_dir) if args.output_dir else config.OUTPUT_DIR / "phase3_gate" / "gate3" / args.model
    run_gate3(args.model, args.device, args.gate1_dir, out, args.n_samples)


if __name__ == "__main__":
    main()
