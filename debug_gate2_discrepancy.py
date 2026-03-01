#!/usr/bin/env python3
"""Diagnose EXACT cause of Gate2 vs Unified V=0 discrepancy.

Tests on same 5 samples:
  A) Gate2 code path (run_causal_experiment.py style)
  B) Unified code path (get_action_logits style)
  C) Gate2 + use_cache=False
  D) Gate2 + fresh inputs per forward pass

This isolates whether the difference comes from:
  1. use_cache setting
  2. Input tensor reuse vs fresh creation
  3. Something else
"""
import sys, json, torch, numpy as np
from pathlib import Path
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.path.insert(0, str(Path(__file__).parent))

import config
from extract_attention import load_model_from_registry, get_layers, call_processor, detect_token_boundaries
from data_sampler import reload_samples_from_list
from contribution.causal import ValueZeroHook, compute_output_kl, compute_top1_change_rate
from run_phase3_exp_de import get_action_logits

MODEL = "ecot-7b"
DEVICE = "cuda:0"
TARGET_POS = 0
N_TEST = 10


def method_A_gate2_original(model, processor, model_cfg, sample, device, deep_layers):
    """Exact Gate2 code path from run_causal_experiment.py lines 81-98"""
    prompt = model_cfg.prompt_template.format(instruction=sample["instruction"])
    inputs = call_processor(processor, prompt, sample["image"], model_cfg, return_tensors="pt").to(device)
    if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

    # Original forward (NO use_cache setting, same inputs object)
    with torch.no_grad():
        out_orig = model(**inputs)
    logits_orig = out_orig.logits[0, -1, :]

    # V=0 forward (same inputs object reused)
    vzero = ValueZeroHook([TARGET_POS], target_layers=deep_layers)
    vzero.register(model, model_cfg, get_layers)
    with torch.no_grad():
        out_vz = model(**inputs)
    logits_vz = out_vz.logits[0, -1, :]
    vzero.remove()

    return logits_orig, logits_vz


def method_B_unified(model, processor, model_cfg, sample, device, deep_layers, bounds):
    """Unified code path using get_action_logits (use_cache=False, fresh inputs)"""
    logits_orig, _ = get_action_logits(model, processor, model_cfg, sample, device, bounds)

    vzero = ValueZeroHook([TARGET_POS], target_layers=deep_layers)
    vzero.register(model, model_cfg, get_layers)
    logits_vz, _ = get_action_logits(model, processor, model_cfg, sample, device, bounds)
    vzero.remove()

    return logits_orig, logits_vz


def method_C_gate2_nocache(model, processor, model_cfg, sample, device, deep_layers):
    """Gate2 path BUT with use_cache=False (test hypothesis 1)"""
    prompt = model_cfg.prompt_template.format(instruction=sample["instruction"])
    inputs = call_processor(processor, prompt, sample["image"], model_cfg, return_tensors="pt").to(device)
    if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

    fwd_kwargs = {k: v for k, v in inputs.items()}
    fwd_kwargs["use_cache"] = False

    with torch.no_grad():
        out_orig = model(**fwd_kwargs)
    logits_orig = out_orig.logits[0, -1, :]

    # V=0 with SAME fwd_kwargs (reused)
    vzero = ValueZeroHook([TARGET_POS], target_layers=deep_layers)
    vzero.register(model, model_cfg, get_layers)
    with torch.no_grad():
        out_vz = model(**fwd_kwargs)
    logits_vz = out_vz.logits[0, -1, :]
    vzero.remove()

    return logits_orig, logits_vz


def method_D_gate2_fresh_inputs(model, processor, model_cfg, sample, device, deep_layers):
    """Gate2 path BUT with fresh inputs per forward pass (test hypothesis 2)"""
    prompt = model_cfg.prompt_template.format(instruction=sample["instruction"])

    # Original: fresh inputs
    inputs1 = call_processor(processor, prompt, sample["image"], model_cfg, return_tensors="pt").to(device)
    if "pixel_values" in inputs1 and inputs1["pixel_values"].dtype != model.dtype:
        inputs1["pixel_values"] = inputs1["pixel_values"].to(model.dtype)
    with torch.no_grad():
        out_orig = model(**inputs1)
    logits_orig = out_orig.logits[0, -1, :]

    # V=0: fresh inputs
    inputs2 = call_processor(processor, prompt, sample["image"], model_cfg, return_tensors="pt").to(device)
    if "pixel_values" in inputs2 and inputs2["pixel_values"].dtype != model.dtype:
        inputs2["pixel_values"] = inputs2["pixel_values"].to(model.dtype)
    vzero = ValueZeroHook([TARGET_POS], target_layers=deep_layers)
    vzero.register(model, model_cfg, get_layers)
    with torch.no_grad():
        out_vz = model(**inputs2)
    logits_vz = out_vz.logits[0, -1, :]
    vzero.remove()

    return logits_orig, logits_vz


def main():
    print(f"Loading {MODEL}...", flush=True)
    processor, model, model_cfg = load_model_from_registry(MODEL, DEVICE)
    model.eval()

    deep_layers = list(range(max(0, model_cfg.num_layers - 10), model_cfg.num_layers))

    # Load same samples
    sample_list_path = config.OUTPUT_DIR / "phase3_gate" / MODEL / "sample_list.json"
    all_samples = reload_samples_from_list(str(sample_list_path), config.DATA_CACHE_DIR)
    test_samples = all_samples[:N_TEST]

    print(f"\nTesting {N_TEST} samples with 4 methods...")
    print(f"Target: pos={TARGET_POS}, layers={deep_layers}\n")

    results = []
    for si, sample in enumerate(test_samples):
        bounds = detect_token_boundaries(
            processor, model, sample["image"], sample["instruction"], DEVICE, model_cfg
        )

        # Run all 4 methods
        logits_A_orig, logits_A_vz = method_A_gate2_original(model, processor, model_cfg, sample, DEVICE, deep_layers)
        logits_B_orig, logits_B_vz = method_B_unified(model, processor, model_cfg, sample, DEVICE, deep_layers, bounds)
        logits_C_orig, logits_C_vz = method_C_gate2_nocache(model, processor, model_cfg, sample, DEVICE, deep_layers)
        logits_D_orig, logits_D_vz = method_D_gate2_fresh_inputs(model, processor, model_cfg, sample, DEVICE, deep_layers)

        # Compute metrics
        entry = {"sample": si}
        for name, lo, lv in [
            ("A_gate2_orig", logits_A_orig, logits_A_vz),
            ("B_unified", logits_B_orig, logits_B_vz),
            ("C_gate2_nocache", logits_C_orig, logits_C_vz),
            ("D_gate2_fresh", logits_D_orig, logits_D_vz),
        ]:
            kl = compute_output_kl(lo, lv)
            flip = int(lo.argmax().item() != lv.argmax().item())
            orig_top1 = lo.argmax().item()
            vz_top1 = lv.argmax().item()
            entry[name] = {
                "kl": round(kl, 6),
                "flip": flip,
                "orig_top1": orig_top1,
                "vz_top1": vz_top1,
            }

        # Check orig logits agreement
        entry["orig_logits_match"] = {
            "A_vs_B": bool(torch.allclose(logits_A_orig, logits_B_orig, atol=1e-4)),
            "A_vs_C": bool(torch.allclose(logits_A_orig, logits_C_orig, atol=1e-4)),
            "A_vs_D": bool(torch.allclose(logits_A_orig, logits_D_orig, atol=1e-4)),
            "A_vs_B_maxdiff": float((logits_A_orig - logits_B_orig).abs().max().item()),
            "A_vs_C_maxdiff": float((logits_A_orig - logits_C_orig).abs().max().item()),
        }
        entry["vz_logits_match"] = {
            "A_vs_B": bool(torch.allclose(logits_A_vz, logits_B_vz, atol=1e-4)),
            "A_vs_C": bool(torch.allclose(logits_A_vz, logits_C_vz, atol=1e-4)),
            "A_vs_D": bool(torch.allclose(logits_A_vz, logits_D_vz, atol=1e-4)),
            "A_vs_B_maxdiff": float((logits_A_vz - logits_B_vz).abs().max().item()),
            "A_vs_C_maxdiff": float((logits_A_vz - logits_C_vz).abs().max().item()),
        }

        results.append(entry)

        a = entry["A_gate2_orig"]
        b = entry["B_unified"]
        print(f"  [{si}] A(gate2): KL={a['kl']:.4f} flip={a['flip']} top1={a['orig_top1']}→{a['vz_top1']}  |  "
              f"B(unified): KL={b['kl']:.4f} flip={b['flip']} top1={b['orig_top1']}→{b['vz_top1']}  |  "
              f"orig_match={entry['orig_logits_match']['A_vs_B']} "
              f"(maxdiff={entry['orig_logits_match']['A_vs_B_maxdiff']:.6f})  |  "
              f"vz_match={entry['vz_logits_match']['A_vs_B']} "
              f"(maxdiff={entry['vz_logits_match']['A_vs_B_maxdiff']:.6f})",
              flush=True)

    # Summary
    print(f"\n{'='*80}")
    print(f"{'Method':<25} {'Mean KL':>10} {'Flip Rate':>12} {'Flips':>8}")
    print(f"{'-'*80}")
    for method in ["A_gate2_orig", "B_unified", "C_gate2_nocache", "D_gate2_fresh"]:
        kls = [r[method]["kl"] for r in results]
        flips = [r[method]["flip"] for r in results]
        print(f"  {method:<23} {np.mean(kls):>10.4f} {np.mean(flips):>12.2%} "
              f"{sum(flips):>5}/{len(flips)}")

    print(f"\nOrig logits A==B: {sum(r['orig_logits_match']['A_vs_B'] for r in results)}/{len(results)}")
    print(f"Orig logits A==C: {sum(r['orig_logits_match']['A_vs_C'] for r in results)}/{len(results)}")
    print(f"Orig logits A==D: {sum(r['orig_logits_match']['A_vs_D'] for r in results)}/{len(results)}")
    print(f"VZ logits A==B: {sum(r['vz_logits_match']['A_vs_B'] for r in results)}/{len(results)}")
    print(f"VZ logits A==C: {sum(r['vz_logits_match']['A_vs_C'] for r in results)}/{len(results)}")
    print(f"VZ logits A==D: {sum(r['vz_logits_match']['A_vs_D'] for r in results)}/{len(results)}")

    # Save
    out_path = Path("outputs/debug_gate2_discrepancy/results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
