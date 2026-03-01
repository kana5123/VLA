#!/usr/bin/env python3
"""
TraceVLA Position Sweep: Ablate various positions to test "global fragility" hypothesis.
Tests whether KL≈13 is position-independent (global fragility) or position-specific.
"""
import sys, json, torch, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import config
from extract_attention import load_model_from_registry, get_layers, call_processor
from visualize_text_attention import load_samples_from_cache
from contribution.causal import ValueZeroHook, compute_output_kl, compute_top1_change_rate, get_deep_layer_ranges

MODEL = "tracevla-phi3v"
DEVICE = "cuda:0"
N_SAMPLES = 25

# Positions to test
# vision: 0-312, text: 313+
TEST_POSITIONS = {
    "vision_start_0": 0,       # Already tested (A_mode)
    "vision_early_3": 3,       # Already tested (R_mode)
    "vision_early_10": 10,
    "vision_mid_75": 75,
    "vision_mid_150": 150,     # Key: mid-vision
    "vision_late_250": 250,
    "vision_late_300": 300,
    "vision_end_312": 312,
    "text_start_313": 313,     # First text token
    "text_mid_320": 320,       # Mid text
}

def main():
    print(f"Loading {MODEL}...")
    processor, model, model_cfg = load_model_from_registry(MODEL, DEVICE)
    
    # Load sample list from Gate 1 if available
    sample_list_path = Path(f"outputs/phase3_gate/{MODEL}/sample_list.json")
    if sample_list_path.exists():
        from data_sampler import reload_samples_from_list
        samples = reload_samples_from_list(str(sample_list_path), config.DATA_CACHE_DIR)
        print(f"Loaded {len(samples)} samples from Gate 1 sample list")
        samples = samples[:N_SAMPLES]
    else:
        samples = load_samples_from_cache(config.DATA_CACHE_DIR, n_samples=N_SAMPLES)
    
    layer_ranges = get_deep_layer_ranges(model_cfg.num_layers)
    target_layers = layer_ranges["all"]
    print(f"Target layers: {target_layers}")
    
    results = {}
    
    for name, pos in TEST_POSITIONS.items():
        print(f"\n--- Ablating position {pos} ({name}) ---")
        kls = []
        flips = []
        
        for si, sample in enumerate(samples):
            prompt = model_cfg.prompt_template.format(instruction=sample["instruction"])
            inputs = call_processor(processor, prompt, sample["image"], model_cfg, return_tensors="pt").to(DEVICE)
            if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
                inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
            
            seq_len = inputs["input_ids"].shape[1]
            if pos >= seq_len:
                print(f"  Skip sample {si}: pos {pos} >= seq_len {seq_len}")
                continue
            
            with torch.no_grad():
                out_orig = model(**inputs)
            logits_orig = out_orig.logits[0, -1, :]
            
            vzero = ValueZeroHook([pos], target_layers=target_layers)
            vzero.register(model, model_cfg, get_layers)
            with torch.no_grad():
                out_vz = model(**inputs)
            logits_vz = out_vz.logits[0, -1, :]
            vzero.remove()
            
            kl = compute_output_kl(logits_orig, logits_vz)
            flip = compute_top1_change_rate(logits_orig.unsqueeze(0), logits_vz.unsqueeze(0))
            kls.append(kl)
            flips.append(flip)
        
        if kls:
            results[name] = {
                "position": pos,
                "mean_kl": float(np.mean(kls)),
                "std_kl": float(np.std(kls)),
                "mean_flip": float(np.mean(flips)),
                "n_samples": len(kls),
            }
            print(f"  KL={np.mean(kls):.4f}±{np.std(kls):.4f}, flip={np.mean(flips):.3f} (n={len(kls)})")
    
    # Summary table
    print(f"\n{'='*70}")
    print(f"{'Position':<25} {'Pos':>5} {'KL':>10} {'±std':>10} {'Flip':>8}")
    print(f"{'-'*70}")
    for name in TEST_POSITIONS:
        if name in results:
            r = results[name]
            print(f"{name:<25} {r['position']:>5} {r['mean_kl']:>10.4f} {r['std_kl']:>10.4f} {r['mean_flip']:>8.3f}")
    
    # Save
    out_path = Path("outputs/tracevla_position_sweep/sweep_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
