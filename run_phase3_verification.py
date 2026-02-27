#!/usr/bin/env python3
"""Phase 3 Verification Experiments.

Exp A: move↔fold tokenization boundary check
  - Verify text_ranges are correct for BOTH original and swapped instructions
  - Check if token count differs (could shift mask boundaries → false anomaly)

Exp B: text_v0 vs text_kv counterfactual comparison
  - V-only zeroing (routing via Q·K still alive) vs K+V zeroing (routing AND content blocked)
  - If text_v0 shows anomaly but text_kv doesn't → routing channel matters
  - If both show anomaly → signal comes from vision tokens themselves

Exp C: Position anchoring test
  - Permute vision patch embedding positions (shuffle 0..255)
  - Re-run forward, check if A_peak/C_peak move → content-anchored
  - Or stay at same index → position-anchored

Usage:
  python run_phase3_verification.py --model ecot-7b --device cuda:0 \
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
from verify_attention_sinks import SinkVerificationHookManager, get_wov_matrix
from contribution.text_mask import (
    TextValueZeroHook, TextKVZeroHook,
    create_text_kv_mask, _ranges_to_positions,
)
from contribution.causal import ValueZeroHook
from contribution.compute import compute_perhead_contribution

VERB_PAIRS = [("pick", "place"), ("open", "close"), ("move", "fold")]


def extract_hidden_at_query(model, inputs, model_cfg, query_pos, deep_layers):
    """Run forward and extract hidden states at query position from deep layers."""
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
    if "text_ranges" in bounds:
        return bounds["text_ranges"]
    ts = bounds.get("text_start", bounds["vision_end"])
    te = bounds["text_end"]
    return [(ts, te)]


# ═══════════════════════════════════════════════════════════════
# Experiment A: Tokenization Boundary Check
# ═══════════════════════════════════════════════════════════════

def run_exp_a(model, processor, model_cfg, samples, device, output_dir):
    """Check that verb swap doesn't shift tokenization boundaries incorrectly."""
    print("\n" + "="*60)
    print("Exp A: Tokenization Boundary Check for Verb Swaps")
    print("="*60)

    tokenizer = getattr(processor, 'tokenizer', processor)
    results = []

    for si, sample in enumerate(samples):
        skill = sample["skill"]
        for v1, v2 in VERB_PAIRS:
            if skill == v1:
                swap_verb = v2
            elif skill == v2:
                swap_verb = v1
            else:
                continue

            orig_instr = sample["instruction"]
            swapped_instr = re.sub(r'\b' + skill + r'\b', swap_verb, orig_instr.lower(), count=1)
            if swapped_instr == orig_instr.lower():
                continue

            # Get boundaries for BOTH instructions
            bounds_orig = detect_token_boundaries(
                processor, model, sample["image"], orig_instr, device, model_cfg
            )
            bounds_swap = detect_token_boundaries(
                processor, model, sample["image"], swapped_instr, device, model_cfg
            )

            tr_orig = _get_text_ranges(bounds_orig)
            tr_swap = _get_text_ranges(bounds_swap)
            n_text_orig = sum(e - s for s, e in tr_orig)
            n_text_swap = sum(e - s for s, e in tr_swap)

            # Tokenize just the verb to check token count
            orig_verb_tokens = tokenizer.encode(skill, add_special_tokens=False)
            swap_verb_tokens = tokenizer.encode(swap_verb, add_special_tokens=False)

            entry = {
                "sample_idx": si,
                "skill": skill,
                "swap_verb": swap_verb,
                "orig_instruction": orig_instr,
                "swapped_instruction": swapped_instr,
                "orig_text_ranges": tr_orig,
                "swap_text_ranges": tr_swap,
                "n_text_orig": n_text_orig,
                "n_text_swap": n_text_swap,
                "text_count_match": n_text_orig == n_text_swap,
                "orig_total_seq": bounds_orig["total_seq_len"],
                "swap_total_seq": bounds_swap["total_seq_len"],
                "seq_len_match": bounds_orig["total_seq_len"] == bounds_swap["total_seq_len"],
                "orig_verb_n_tokens": len(orig_verb_tokens),
                "swap_verb_n_tokens": len(swap_verb_tokens),
                "verb_tokenlen_match": len(orig_verb_tokens) == len(swap_verb_tokens),
                "vision_range_match": (
                    bounds_orig["vision_start"] == bounds_swap["vision_start"] and
                    bounds_orig["vision_end"] == bounds_swap["vision_end"]
                ),
            }
            results.append(entry)
            status = "OK" if entry["seq_len_match"] and entry["vision_range_match"] else "MISMATCH"
            print(f"  [{status}] s{si} {skill}→{swap_verb}: "
                  f"seq {entry['orig_total_seq']}→{entry['swap_total_seq']}, "
                  f"text {n_text_orig}→{n_text_swap}, "
                  f"verb_tokens {len(orig_verb_tokens)}→{len(swap_verb_tokens)}")

    # Summary
    n_mismatch = sum(1 for r in results if not r["seq_len_match"])
    n_verb_mismatch = sum(1 for r in results if not r["verb_tokenlen_match"])
    print(f"\n  Summary: {len(results)} pairs, {n_mismatch} seq_len mismatches, "
          f"{n_verb_mismatch} verb token-length mismatches")

    if n_mismatch > 0:
        print("  WARNING: Sequence length mismatch found! Mask ranges may be shifted.")
        for r in results:
            if not r["seq_len_match"]:
                print(f"    s{r['sample_idx']} {r['skill']}→{r['swap_verb']}: "
                      f"seq {r['orig_total_seq']}→{r['swap_total_seq']}")

    with open(output_dir / "exp_a_tokenization_check.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ═══════════════════════════════════════════════════════════════
# Experiment B: text_v0 vs text_kv Counterfactual Comparison
# ═══════════════════════════════════════════════════════════════

def run_exp_b(model, processor, model_cfg, samples, device, deep_layers, output_dir):
    """Compare text_v0 (V-only zeroing) vs text_kv (K+V zeroing) under verb swap.

    text_v0: Q·K routing still active → verb routing info can flow through attention
    text_kv: both K and V zeroed → routing AND content fully blocked

    If move↔fold anomaly appears in BOTH → signal comes from vision side (true anomaly)
    If only in text_v0 (not text_kv) → routing channel carries the verb info
    """
    print("\n" + "="*60)
    print("Exp B: text_v0 vs text_kv Counterfactual Comparison")
    print("="*60)

    results = []

    for si, sample in enumerate(samples):
        skill = sample["skill"]
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

        orig_instr = sample["instruction"]
        swapped_instr = re.sub(r'\b' + skill + r'\b', swap_verb, orig_instr.lower(), count=1)
        if swapped_instr == orig_instr.lower():
            continue

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

        # Condition 1: original (no masking)
        h_orig = extract_hidden_at_query(model, inputs_orig, model_cfg, qpos, deep_layers)
        h_swap = extract_hidden_at_query(model, inputs_swap, model_cfg, qpos, deep_layers)

        # Condition 2: text_v0 (V-only zeroing — routing via Q·K alive)
        hook_v0_o = TextValueZeroHook(text_ranges)
        hook_v0_o.register(model, model_cfg, get_layers)
        h_orig_v0 = extract_hidden_at_query(model, inputs_orig, model_cfg, qpos, deep_layers)
        hook_v0_o.remove()

        hook_v0_s = TextValueZeroHook(text_ranges)
        hook_v0_s.register(model, model_cfg, get_layers)
        h_swap_v0 = extract_hidden_at_query(model, inputs_swap, model_cfg, qpos, deep_layers)
        hook_v0_s.remove()

        # Condition 3: text_kv (K+V zeroing — routing AND content blocked)
        inputs_orig_kv = {k: v.clone() for k, v in inputs_orig.items()}
        inputs_swap_kv = {k: v.clone() for k, v in inputs_swap.items()}

        hook_kv_o, apply_fn_o = create_text_kv_mask(text_ranges, model, model_cfg, get_layers)
        if apply_fn_o is not None:
            inputs_orig_kv = apply_fn_o(inputs_orig_kv)
        h_orig_kv = extract_hidden_at_query(model, inputs_orig_kv, model_cfg, qpos, deep_layers)
        if hook_kv_o is not None:
            hook_kv_o.remove()

        hook_kv_s, apply_fn_s = create_text_kv_mask(text_ranges, model, model_cfg, get_layers)
        if apply_fn_s is not None:
            inputs_swap_kv = apply_fn_s(inputs_swap_kv)
        h_swap_kv = extract_hidden_at_query(model, inputs_swap_kv, model_cfg, qpos, deep_layers)
        if hook_kv_s is not None:
            hook_kv_s.remove()

        for l in deep_layers:
            if l not in h_orig or l not in h_swap:
                continue
            norm_orig = max(np.linalg.norm(h_orig[l]), 1e-10)
            delta_none = np.linalg.norm(h_orig[l] - h_swap[l]) / norm_orig
            delta_v0 = 0.0
            delta_kv = 0.0
            if l in h_orig_v0 and l in h_swap_v0:
                delta_v0 = np.linalg.norm(h_orig_v0[l] - h_swap_v0[l]) / max(np.linalg.norm(h_orig_v0[l]), 1e-10)
            if l in h_orig_kv and l in h_swap_kv:
                delta_kv = np.linalg.norm(h_orig_kv[l] - h_swap_kv[l]) / max(np.linalg.norm(h_orig_kv[l]), 1e-10)

            results.append({
                "sample_idx": si,
                "layer": l,
                "skill": skill,
                "swap_verb": swap_verb,
                "delta_orig": float(delta_none),
                "delta_text_v0": float(delta_v0),
                "delta_text_kv": float(delta_kv),
            })

        if si % 5 == 0:
            print(f"  Processed sample {si}/{len(samples)}")

    # Summary by verb pair
    print("\n  Per-verb-pair summary (mean across layers and samples):")
    verb_groups = defaultdict(lambda: {"orig": [], "v0": [], "kv": []})
    for r in results:
        key = f"{r['skill']}↔{r['swap_verb']}"
        verb_groups[key]["orig"].append(r["delta_orig"])
        verb_groups[key]["v0"].append(r["delta_text_v0"])
        verb_groups[key]["kv"].append(r["delta_text_kv"])

    summary = {}
    for vp, vals in sorted(verb_groups.items()):
        m_orig = np.mean(vals["orig"])
        m_v0 = np.mean(vals["v0"])
        m_kv = np.mean(vals["kv"])
        print(f"    {vp}: delta_orig={m_orig:.4f}, delta_v0={m_v0:.4f}, delta_kv={m_kv:.4f}")
        if m_kv > 1e-6:
            print(f"      v0/kv ratio = {m_v0/m_kv:.2f}x, orig/kv ratio = {m_orig/m_kv:.2f}x")
        else:
            print(f"      delta_kv ≈ 0 → text is sole channel for this verb pair")
        summary[vp] = {"delta_orig": m_orig, "delta_v0": m_v0, "delta_kv": m_kv}

    with open(output_dir / "exp_b_v0_vs_kv.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(output_dir / "exp_b_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return results


# ═══════════════════════════════════════════════════════════════
# Experiment C: Position Anchoring Test
# ═══════════════════════════════════════════════════════════════

def run_exp_c(model, processor, model_cfg, samples, device, deep_layers, output_dir, n_permutations=5):
    """Test if A/C peaks are position-anchored or content-anchored.

    Method:
      1. Original forward → record A_peak, C_peak positions
      2. Permute vision token embeddings (post-vision-encoder, pre-LLM)
         by hooking into the projector/connector output
      3. Re-run → record new A_peak, C_peak
      4. If peaks stay at same index → position-anchored
         If peaks move to follow permuted content → content-anchored
    """
    print("\n" + "="*60)
    print("Exp C: Position Anchoring Test (Vision Token Permutation)")
    print("="*60)

    results = []
    n_samples_to_use = min(10, len(samples))  # Use fewer samples for speed

    for si in range(n_samples_to_use):
        sample = samples[si]
        prompt = model_cfg.prompt_template.format(instruction=sample["instruction"])
        inputs = call_processor(processor, prompt, sample["image"], model_cfg, return_tensors="pt").to(device)
        if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

        bounds = detect_token_boundaries(
            processor, model, sample["image"], sample["instruction"], device, model_cfg
        )
        vs = bounds["vision_start"]
        ve = bounds["vision_end"]
        n_vis = ve - vs
        qpos = bounds["text_end"] - 1

        # === Original forward: get A_peak, C_peak ===
        hook_mgr = SinkVerificationHookManager(model, model_cfg)
        hook_mgr.register_hooks()
        hook_mgr.reset()
        with torch.no_grad():
            model(**inputs, output_attentions=True)

        orig_peaks = {}
        for l in deep_layers:
            attn = hook_mgr.attention_weights.get(l)
            hidden = hook_mgr.hidden_states.get(l)
            if attn is None or hidden is None:
                continue
            # Get previous layer's hidden for W_OV contribution
            prev_hidden = hook_mgr.hidden_states.get(l - 1, hidden)

            # A_peak: mean attention from query to vision tokens
            if attn.dim() == 3:  # (H, seq, seq)
                attn_to_vis = attn[:, qpos, vs:ve].mean(dim=0)  # (n_vis,)
            else:
                attn_to_vis = attn[0, :, qpos, vs:ve].mean(dim=0)
            a_peak_rel = int(attn_to_vis.argmax().item())

            # C_peak: approximate via attention * hidden norm (fast proxy)
            h_vis = prev_hidden[vs:ve]  # (n_vis, D)
            h_norms = torch.norm(h_vis, dim=-1)  # (n_vis,)
            c_proxy = attn_to_vis.cpu() * h_norms  # attention * hidden magnitude
            c_peak_rel = int(c_proxy.argmax().item())

            orig_peaks[l] = {"a_peak_rel": a_peak_rel, "c_peak_rel": c_peak_rel}

        hook_mgr.remove_hooks()

        # === Permuted forwards ===
        for perm_idx in range(n_permutations):
            rng = np.random.default_rng(seed=42 + si * 100 + perm_idx)
            perm = rng.permutation(n_vis)
            perm_tensor = torch.tensor(perm, device=device, dtype=torch.long)

            # Hook to permute vision embeddings after they enter the LLM
            handles = []
            is_prismatic = hasattr(model, 'language_model') and hasattr(model, 'projector')

            if is_prismatic:
                # Prismatic: hook on projector output to permute projected patches
                target_module = model.projector
            else:
                # Standard HF: hook on first LLM layer input to permute vision positions
                layers = get_layers(model, model_cfg)
                target_module = layers[0]

            def make_permute_hook(vs, ve, perm_t, is_proj=False):
                def hook_fn(module, args, output):
                    if is_proj:
                        # Projector output: (batch, n_patches, D)
                        modified = output.clone()
                        modified[0] = modified[0, perm_t]
                        return modified
                    else:
                        # First layer input: permute vision range in hidden_states
                        h = args[0] if isinstance(args, tuple) else args
                        modified = h.clone()
                        modified[0, vs:ve] = modified[0, vs:ve][perm_t]
                        return (modified,) + args[1:] if isinstance(args, tuple) else modified
                return hook_fn

            if is_prismatic:
                handle = target_module.register_forward_hook(
                    make_permute_hook(vs, ve, perm_tensor, is_proj=True)
                )
            else:
                handle = target_module.register_forward_pre_hook(
                    lambda module, args: (
                        make_permute_hook(vs, ve, perm_tensor, is_proj=False)(module, args, None)
                        if False else None  # pre_hook needs different approach
                    )
                )
                # Actually use a forward hook on embeddings
                handle.remove()
                # For non-Prismatic: hook on the model's embed layer output
                if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                    embed_module = model.model.embed_tokens
                elif hasattr(model, 'embed_tokens'):
                    embed_module = model.embed_tokens
                else:
                    embed_module = target_module

                def embed_permute_hook(module, args, output):
                    if isinstance(output, tuple):
                        tensor = output[0].clone()
                        tensor[0, vs:ve] = tensor[0, vs:ve][perm_tensor]
                        return (tensor,) + output[1:]
                    else:
                        modified = output.clone()
                        modified[0, vs:ve] = modified[0, vs:ve][perm_tensor]
                        return modified

                handle = embed_module.register_forward_hook(embed_permute_hook)

            handles.append(handle)

            # Run permuted forward
            hook_mgr2 = SinkVerificationHookManager(model, model_cfg)
            hook_mgr2.register_hooks()
            hook_mgr2.reset()
            with torch.no_grad():
                model(**inputs, output_attentions=True)

            for l in deep_layers:
                if l not in orig_peaks:
                    continue
                attn = hook_mgr2.attention_weights.get(l)
                hidden = hook_mgr2.hidden_states.get(l)
                if attn is None or hidden is None:
                    continue
                prev_hidden = hook_mgr2.hidden_states.get(l - 1, hidden)

                if attn.dim() == 3:
                    attn_to_vis = attn[:, qpos, vs:ve].mean(dim=0)
                else:
                    attn_to_vis = attn[0, :, qpos, vs:ve].mean(dim=0)
                perm_a_peak_rel = int(attn_to_vis.argmax().item())

                h_vis = prev_hidden[vs:ve]
                h_norms = torch.norm(h_vis, dim=-1)
                c_proxy = attn_to_vis.cpu() * h_norms
                perm_c_peak_rel = int(c_proxy.argmax().item())

                orig_a = orig_peaks[l]["a_peak_rel"]
                orig_c = orig_peaks[l]["c_peak_rel"]

                # Check if peak followed the content or stayed at same position
                # If position-anchored: perm_peak == orig_peak (same index)
                # If content-anchored: perm_peak == perm[orig_peak] (followed the content)
                content_target_a = int(np.where(perm == orig_a)[0][0]) if orig_a < len(perm) else -1
                content_target_c = int(np.where(perm == orig_c)[0][0]) if orig_c < len(perm) else -1

                results.append({
                    "sample_idx": si,
                    "perm_idx": perm_idx,
                    "layer": l,
                    "orig_a_peak": orig_a,
                    "perm_a_peak": perm_a_peak_rel,
                    "a_stayed_same_pos": perm_a_peak_rel == orig_a,
                    "a_followed_content": perm_a_peak_rel == content_target_a,
                    "orig_c_peak": orig_c,
                    "perm_c_peak": perm_c_peak_rel,
                    "c_stayed_same_pos": perm_c_peak_rel == orig_c,
                    "c_followed_content": perm_c_peak_rel == content_target_c,
                })

            hook_mgr2.remove_hooks()
            for h in handles:
                h.remove()

        print(f"  Sample {si}/{n_samples_to_use}: "
              f"orig A_peak={orig_peaks.get(deep_layers[-1], {}).get('a_peak_rel', '?')}, "
              f"C_peak={orig_peaks.get(deep_layers[-1], {}).get('c_peak_rel', '?')}")

    # Summary
    if results:
        n_a_same = sum(1 for r in results if r["a_stayed_same_pos"])
        n_a_follow = sum(1 for r in results if r["a_followed_content"])
        n_c_same = sum(1 for r in results if r["c_stayed_same_pos"])
        n_c_follow = sum(1 for r in results if r["c_followed_content"])
        total = len(results)

        print(f"\n  A_peak: stayed_same_pos={n_a_same}/{total} ({n_a_same/total:.1%}), "
              f"followed_content={n_a_follow}/{total} ({n_a_follow/total:.1%}), "
              f"other={total - n_a_same - n_a_follow}/{total}")
        print(f"  C_peak: stayed_same_pos={n_c_same}/{total} ({n_c_same/total:.1%}), "
              f"followed_content={n_c_follow}/{total} ({n_c_follow/total:.1%}), "
              f"other={total - n_c_same - n_c_follow}/{total}")

        if n_a_same / total > 0.7:
            print("  → A_peak is POSITION-ANCHORED (stays at same index regardless of content)")
        elif n_a_follow / total > 0.7:
            print("  → A_peak is CONTENT-ANCHORED (follows permuted content)")
        else:
            print("  → A_peak shows MIXED anchoring behavior")

        if n_c_same / total > 0.7:
            print("  → C_peak is POSITION-ANCHORED")
        elif n_c_follow / total > 0.7:
            print("  → C_peak is CONTENT-ANCHORED")
        else:
            print("  → C_peak shows MIXED anchoring behavior")

    summary = {
        "n_results": len(results),
        "a_stayed_same_pos_rate": n_a_same / total if results else 0,
        "a_followed_content_rate": n_a_follow / total if results else 0,
        "c_stayed_same_pos_rate": n_c_same / total if results else 0,
        "c_followed_content_rate": n_c_follow / total if results else 0,
    }

    with open(output_dir / "exp_c_position_anchoring.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(output_dir / "exp_c_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 3 Verification Experiments")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gate1_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--experiments", default="a,b,c", help="Comma-separated: a,b,c")
    args = parser.parse_args()

    gate1_dir = Path(args.gate1_dir)
    out = Path(args.output_dir) if args.output_dir else config.OUTPUT_DIR / "phase3_gate" / "verification" / args.model
    out.mkdir(parents=True, exist_ok=True)

    # Load model
    processor, model, model_cfg = load_model_from_registry(args.model, args.device)

    # Load samples
    sample_list_path = gate1_dir / "sample_list.json"
    samples = reload_samples_from_list(sample_list_path, config.DATA_CACHE_DIR)
    samples = samples[:args.n_samples]
    print(f"Loaded {len(samples)} samples")

    deep_layers = list(range(max(0, model_cfg.num_layers - 10), model_cfg.num_layers))
    exps = [e.strip() for e in args.experiments.split(",")]

    if "a" in exps:
        run_exp_a(model, processor, model_cfg, samples, args.device, out)

    if "b" in exps:
        run_exp_b(model, processor, model_cfg, samples, args.device, deep_layers, out)

    if "c" in exps:
        run_exp_c(model, processor, model_cfg, samples, args.device, deep_layers, out)

    print(f"\nAll results saved to {out}/")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
