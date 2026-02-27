#!/usr/bin/env python3
"""Adaptive Routing: Diagnose VLA routing failure mode and auto-select intervention.

Implements the core "diagnose -> prescribe" pipeline for the VLA routing paper:
  1. AdaptiveRouter: measures phi, D3 KL, flip rate, A-peak vs C-peak to classify
     the model's routing failure mode (bottleneck / coexist / sink / normal).
  2. evaluate_d2_with_intervention(): measures D2 augmentation consistency with
     any intervention hook installed (var, kscale, hybrid, or baseline).
  3. run_full_comparison(): tests all intervention configs and auto-selection,
     produces comparison table.
  4. CLI main(): python adaptive_routing.py --model ecot-7b --device cuda:0 --mode both

Classification rules (from our experiments):
  if D3_KL > 1.0 and not has_coexist:  -> "bottleneck" -> No intervention
  if D3_KL > 1.0 and has_coexist:      -> "coexist"    -> VARValueHook(p=0.9)
  if D3_KL < 0.1:                       -> "sink"       -> KeyScaleHook(alpha=0.0)
  else:                                  -> "normal"     -> VARValueHook(p=0.6)

Expected results from existing experiments:
  ECoT       = bottleneck (D3 KL=1.66, neither intervention helps)
  OpenVLA    = coexist    (D3 KL=1.66, VAR +7%)
  TraceVLA   = sink       (D3 KL=0.01, K-scale +3%)
  SpatialVLA = normal     (VAR +4%, K-scale +2%)

Usage:
  python adaptive_routing.py --model ecot-7b --device cuda:0 --mode both
  python adaptive_routing.py --model openvla-7b --device cuda:0 --mode diagnose
  python adaptive_routing.py --model tracevla-7b --device cuda:0 --mode compare
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
    load_model_from_registry,
    get_layers,
    call_processor,
    detect_token_boundaries,
)
from data_sampler import reload_samples_from_list
from verify_attention_sinks import SinkVerificationHookManager
from run_phase3_exp_de import (
    ValueScaleHook,
    KeyScaleHook,
    get_action_logits,
    detect_anchor_targets,
    AUGMENTATIONS,
    action_token_entropy,
)
from run_var_baseline import VARValueHook, compute_phi, SINK_DIMENSIONS
from contribution.causal import ValueZeroHook, compute_output_kl


# =============================================================================
# Classification thresholds
# =============================================================================

D3_KL_HIGH_THRESHOLD = 1.0   # Above this: bottleneck or coexist
D3_KL_LOW_THRESHOLD = 0.1    # Below this: sink


# =============================================================================
# AdaptiveRouter: Diagnose + Prescribe
# =============================================================================

class AdaptiveRouter:
    """Diagnoses a VLA model's routing failure mode and applies optimal intervention.

    Workflow:
        1. diagnose(n_samples) -> measures D3 KL, phi, flip rate, A-peak vs C-peak
        2. apply_optimal_hook() -> installs VARValueHook or KeyScaleHook based on diagnosis
        3. remove_hook() -> cleanup

    Attributes:
        diagnosis: dict with classification result and supporting metrics
        active_hooks: list of currently installed hooks
    """

    def __init__(self, model, processor, model_cfg, device, samples,
                 bounds_cache, deep_layers, output_dir, verification_dir):
        """
        Args:
            model: loaded VLA model
            processor: model processor
            model_cfg: VLAModelConfig from registry
            device: torch device string
            samples: list of sample dicts (image, instruction, ...)
            bounds_cache: dict mapping sample_idx -> token boundaries
            deep_layers: list of layer indices (last 10 layers)
            output_dir: Path for saving results
            verification_dir: Path to Exp C anchoring data
        """
        self.model = model
        self.processor = processor
        self.model_cfg = model_cfg
        self.device = device
        self.samples = samples
        self.bounds_cache = bounds_cache
        self.deep_layers = deep_layers
        self.output_dir = output_dir
        self.verification_dir = verification_dir
        self.diagnosis = None
        self.active_hooks = []

    def diagnose(self, n_samples=3):
        """Run diagnosis on n_samples to classify the model's routing failure mode.

        Measures:
            - phi: mean sink score across vision tokens (VAR paper)
            - D3 KL: KL divergence when V-zeroing the anchor token at deep layers
            - flip_rate: fraction of samples where top-1 action changes after V-zero
            - A-peak vs C-peak: whether attention peak and contribution peak differ
              (coexist = A-peak != C-peak)

        Returns:
            dict with keys: classification, d3_kl_mean, flip_rate, has_coexist,
            phi_mean, anchor_targets, recommended_hook, recommended_params
        """
        n = min(n_samples, len(self.samples))
        samples = self.samples[:n]

        print(f"\n{'='*60}")
        print(f"  Adaptive Router: Diagnosing {self.model_cfg.name}")
        print(f"  Using {n} samples, deep layers: {self.deep_layers}")
        print(f"{'='*60}")

        # --- Step 1: Detect anchor targets (A-peak vs C-peak) ---
        anchor_targets = detect_anchor_targets(
            self.model_cfg, self.verification_dir, self.bounds_cache[0]
        )
        has_coexist = len(anchor_targets) > 1
        print(f"  Anchor targets: {anchor_targets}")
        print(f"  Has coexist (A-peak != C-peak): {has_coexist}")

        # --- Step 2: D3 ablation — V-zero anchor at deep layers ---
        d3_kls = []
        flip_count = 0
        for si in range(n):
            sample = samples[si]
            bounds = self.bounds_cache[si]
            vs = bounds["vision_start"]

            # Determine ablation target (primary anchor's absolute position)
            anchor_rel = anchor_targets[0]["anchor_rel"]
            target_abs = vs + anchor_rel

            # Original forward
            logits_orig, _ = get_action_logits(
                self.model, self.processor, self.model_cfg,
                sample, self.device, bounds,
            )

            # V-zero forward at anchor, deep layers only
            vzero = ValueZeroHook(
                target_positions=[target_abs],
                target_layers=self.deep_layers,
            )
            vzero.register(self.model, self.model_cfg, get_layers)
            logits_masked, _ = get_action_logits(
                self.model, self.processor, self.model_cfg,
                sample, self.device, bounds,
            )
            vzero.remove()

            kl = compute_output_kl(logits_orig, logits_masked)
            top1_changed = logits_orig.argmax().item() != logits_masked.argmax().item()
            d3_kls.append(kl)
            flip_count += int(top1_changed)

            print(f"    D3 [{si+1}/{n}] KL={kl:.3f} flip={top1_changed}")

        d3_kl_mean = float(np.mean(d3_kls))
        flip_rate = flip_count / n

        # --- Step 3: Phi (hidden state sink score) ---
        sink_dims = SINK_DIMENSIONS.get(self.model_cfg.architecture, [])
        phi_values_all = []

        if sink_dims:
            hook_mgr = SinkVerificationHookManager(self.model, self.model_cfg)
            for si in range(min(n, 3)):
                sample = samples[si]
                bounds = self.bounds_cache[si]
                vs = bounds["vision_start"]
                ve = bounds["vision_end"]

                hook_mgr.register_hooks()
                hook_mgr.reset()

                prompt = self.model_cfg.prompt_template.format(
                    instruction=sample["instruction"]
                )
                inputs = call_processor(
                    self.processor, prompt, sample["image"],
                    self.model_cfg, return_tensors="pt",
                ).to(self.device)
                if "pixel_values" in inputs and inputs["pixel_values"].dtype != self.model.dtype:
                    inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)

                fwd_kwargs = {k: v for k, v in inputs.items()}
                fwd_kwargs["use_cache"] = False
                if self.model_cfg.architecture == "gemma2" and "intrinsic" not in fwd_kwargs:
                    fwd_kwargs["intrinsic"] = torch.tensor(
                        [[[218.26, 0.0, 111.83],
                          [0.0, 218.26, 111.79],
                          [0.0, 0.0, 1.0]]],
                        device=self.device, dtype=torch.float32,
                    )

                with torch.no_grad():
                    self.model(**fwd_kwargs)

                # Compute phi at one deep layer
                l = self.deep_layers[0]
                hidden = hook_mgr.hidden_states.get(l)
                if hidden is not None and ve > vs:
                    phi_vals = compute_phi(hidden[vs:ve], sink_dims)
                    phi_values_all.append(float(phi_vals.mean().item()))

                hook_mgr.remove_hooks()
        else:
            print(f"  No known sink dims for {self.model_cfg.architecture}, skipping phi")

        phi_mean = float(np.mean(phi_values_all)) if phi_values_all else 0.0

        # --- Step 4: Classification ---
        if d3_kl_mean > D3_KL_HIGH_THRESHOLD and not has_coexist:
            classification = "bottleneck"
            recommended_hook = None
            recommended_params = None
        elif d3_kl_mean > D3_KL_HIGH_THRESHOLD and has_coexist:
            classification = "coexist"
            recommended_hook = "var"
            recommended_params = {"p": 0.9}
        elif d3_kl_mean < D3_KL_LOW_THRESHOLD:
            classification = "sink"
            recommended_hook = "kscale"
            recommended_params = {"alpha": 0.0}
        else:
            classification = "normal"
            recommended_hook = "var"
            recommended_params = {"p": 0.6}

        self.diagnosis = {
            "model": self.model_cfg.name,
            "classification": classification,
            "d3_kl_mean": round(d3_kl_mean, 4),
            "d3_kls": [round(k, 4) for k in d3_kls],
            "flip_rate": round(flip_rate, 4),
            "has_coexist": has_coexist,
            "phi_mean": round(phi_mean, 4),
            "anchor_targets": anchor_targets,
            "n_samples_used": n,
            "recommended_hook": recommended_hook,
            "recommended_params": recommended_params,
        }

        # Print diagnosis summary
        print(f"\n  {'='*50}")
        print(f"  DIAGNOSIS: {classification.upper()}")
        print(f"  {'='*50}")
        print(f"  D3 KL (mean) = {d3_kl_mean:.4f}")
        print(f"  Flip rate    = {flip_rate:.4f}")
        print(f"  Has coexist  = {has_coexist}")
        print(f"  Phi (mean)   = {phi_mean:.4f}")
        if recommended_hook:
            print(f"  Recommended  = {recommended_hook} {recommended_params}")
        else:
            print(f"  Recommended  = No intervention (bottleneck)")
        print(f"  {'='*50}\n")

        # Save diagnosis
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "diagnosis.json", "w") as f:
            json.dump(self.diagnosis, f, indent=2)
        print(f"  Diagnosis saved to {self.output_dir / 'diagnosis.json'}")

        return self.diagnosis

    def apply_optimal_hook(self):
        """Install the recommended intervention hook based on the diagnosis.

        Must call diagnose() first.

        Returns:
            list of installed hooks (may be empty for bottleneck)
        """
        if self.diagnosis is None:
            raise RuntimeError("Must call diagnose() before apply_optimal_hook()")

        # Remove any previously installed hooks
        self.remove_hook()

        classification = self.diagnosis["classification"]
        anchor_targets = self.diagnosis["anchor_targets"]
        params = self.diagnosis["recommended_params"]

        if classification == "bottleneck":
            print("  No hook installed (bottleneck: intervention not helpful)")
            return []

        # Compute sink and target positions from first sample's bounds
        bounds = self.bounds_cache[0]
        vs = bounds["vision_start"]
        ve = bounds["vision_end"]

        if classification in ("coexist", "normal"):
            # VARValueHook: targets the primary anchor (vision sink)
            sink_abs = [anchor_targets[0]["target_abs"]]
            p = params["p"]
            hook = VARValueHook(
                sink_positions_abs=sink_abs,
                vision_start=vs,
                vision_end=ve,
                p=p,
                target_layers=self.deep_layers,
            )
            hook.register(self.model, self.model_cfg, get_layers)
            self.active_hooks.append(hook)
            print(f"  Installed VARValueHook(p={p}) at sink={sink_abs}")

        elif classification == "sink":
            # KeyScaleHook: targets all anchor positions
            all_targets = [t["target_abs"] for t in anchor_targets]
            alpha = params["alpha"]
            hook = KeyScaleHook(
                target_positions=all_targets,
                alpha=alpha,
                target_layers=self.deep_layers,
            )
            hook.register(self.model, self.model_cfg, get_layers)
            self.active_hooks.append(hook)
            print(f"  Installed KeyScaleHook(alpha={alpha}) at targets={all_targets}")

        return self.active_hooks

    def remove_hook(self):
        """Remove all active intervention hooks."""
        for hook in self.active_hooks:
            hook.remove()
        if self.active_hooks:
            print(f"  Removed {len(self.active_hooks)} hook(s)")
        self.active_hooks = []


# =============================================================================
# D2 Measurement with Intervention
# =============================================================================

def evaluate_d2_with_intervention(
    model, processor, model_cfg, samples, device, bounds_cache,
    deep_layers, anchor_targets, intervention_config=None,
):
    """Measure D2 (augmentation consistency) with an optional intervention hook.

    Replicates D2 measurement from run_phase3_exp_de.py:483-521:
      For each sample:
        1. Get original logits (with hook if applicable), record top1_orig
        2. For each of 5 AUGMENTATIONS:
           - Apply augmentation to sample image
           - Get augmented logits (with same hook)
           - Check if top1 matches original
        3. D2 = mean(matches) across all augmentations and samples

    Args:
        model, processor, model_cfg: model stack
        samples: list of sample dicts
        device: torch device
        bounds_cache: dict si -> bounds
        deep_layers: list of deep layer indices
        anchor_targets: list of anchor target dicts from detect_anchor_targets
        intervention_config: None (baseline) or dict with:
            {"method": "var", "p": float}
            {"method": "kscale", "alpha": float}
            {"method": "hybrid", "p": float, "alpha": float}

    Returns:
        dict with d2_mean, d2_per_sample, intervention_name, entropy_mean
    """
    method = intervention_config["method"] if intervention_config else None
    config_name = _make_config_name(intervention_config)

    print(f"\n  --- D2 measurement: {config_name} ---")

    bounds0 = bounds_cache[0]
    vs = bounds0["vision_start"]
    ve = bounds0["vision_end"]
    sink_abs = [anchor_targets[0]["target_abs"]]
    all_target_abs = [t["target_abs"] for t in anchor_targets]

    d2_per_sample = []
    entropies = []

    for si, sample in enumerate(samples):
        bounds = bounds_cache[si]
        local_vs = bounds["vision_start"]
        local_ve = bounds["vision_end"]
        local_sink_abs = [local_vs + anchor_targets[0]["anchor_rel"]]
        local_all_targets = [local_vs + t["anchor_rel"] for t in anchor_targets]

        # --- Install hooks for this sample ---
        hooks = _install_hooks(
            model, model_cfg, intervention_config,
            local_sink_abs, local_vs, local_ve,
            local_all_targets, deep_layers,
        )

        # Original logits
        logits_orig, _ = get_action_logits(
            model, processor, model_cfg, sample, device, bounds,
        )
        top1_orig = logits_orig.argmax().item()
        info = action_token_entropy(logits_orig)
        entropies.append(info["entropy"])

        # Augmentation consistency
        aug_matches = []
        for aug_name, aug_fn in AUGMENTATIONS:
            aug_sample = {**sample, "image": aug_fn(sample["image"])}
            logits_aug, _ = get_action_logits(
                model, processor, model_cfg, aug_sample, device, bounds,
            )
            aug_matches.append(logits_aug.argmax().item() == top1_orig)

        d2 = float(np.mean(aug_matches))
        d2_per_sample.append(d2)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        if (si + 1) % 5 == 0 or si == 0:
            print(f"    [{si+1}/{len(samples)}] D2={d2:.2f} H={info['entropy']:.3f}")

    d2_mean = float(np.mean(d2_per_sample))
    entropy_mean = float(np.mean(entropies))

    print(f"  {config_name}: D2={d2_mean:.4f}, H={entropy_mean:.4f}")

    return {
        "intervention_name": config_name,
        "intervention_config": intervention_config,
        "d2_mean": round(d2_mean, 4),
        "d2_per_sample": [round(d, 4) for d in d2_per_sample],
        "entropy_mean": round(entropy_mean, 4),
        "n_samples": len(samples),
    }


def _install_hooks(model, model_cfg, intervention_config,
                   sink_abs, vs, ve, all_target_abs, deep_layers):
    """Install hooks based on intervention config. Returns list of hooks."""
    hooks = []
    if intervention_config is None:
        return hooks

    method = intervention_config["method"]

    if method in ("var", "hybrid"):
        p = intervention_config.get("p", 0.6)
        var_hook = VARValueHook(
            sink_positions_abs=sink_abs,
            vision_start=vs,
            vision_end=ve,
            p=p,
            target_layers=deep_layers,
        )
        var_hook.register(model, model_cfg, get_layers)
        hooks.append(var_hook)

    if method in ("kscale", "hybrid"):
        alpha = intervention_config.get("alpha", 0.0)
        k_hook = KeyScaleHook(
            target_positions=all_target_abs,
            alpha=alpha,
            target_layers=deep_layers,
        )
        k_hook.register(model, model_cfg, get_layers)
        hooks.append(k_hook)

    return hooks


def _make_config_name(intervention_config):
    """Create a human-readable name for an intervention config."""
    if intervention_config is None:
        return "baseline"
    method = intervention_config["method"]
    if method == "var":
        return f"VAR_p{intervention_config['p']}"
    elif method == "kscale":
        return f"Kscale_a{intervention_config['alpha']}"
    elif method == "hybrid":
        return f"hybrid_p{intervention_config['p']}_a{intervention_config['alpha']}"
    return str(intervention_config)


# =============================================================================
# Full Comparison: test all configs + auto-selection
# =============================================================================

# Comparison configs to test
COMPARISON_CONFIGS = [
    {"name": "baseline", "config": None},
    {"name": "VAR_p0.3", "config": {"method": "var", "p": 0.3}},
    {"name": "VAR_p0.6", "config": {"method": "var", "p": 0.6}},
    {"name": "VAR_p0.9", "config": {"method": "var", "p": 0.9}},
    {"name": "Kscale_a0.0", "config": {"method": "kscale", "alpha": 0.0}},
    {"name": "Kscale_a0.1", "config": {"method": "kscale", "alpha": 0.1}},
    {"name": "Kscale_a0.3", "config": {"method": "kscale", "alpha": 0.3}},
    {"name": "hybrid_p0.6_a0.3", "config": {"method": "hybrid", "p": 0.6, "alpha": 0.3}},
    {"name": "hybrid_p0.9_a0.0", "config": {"method": "hybrid", "p": 0.9, "alpha": 0.0}},
    {"name": "hybrid_p0.3_a0.1", "config": {"method": "hybrid", "p": 0.3, "alpha": 0.1}},
]


def run_full_comparison(
    model, processor, model_cfg, samples, device,
    bounds_cache, deep_layers, anchor_targets,
    output_dir, diagnosis=None,
):
    """Test all intervention configs and auto-selection, produce comparison table.

    Args:
        model, processor, model_cfg: model stack
        samples: list of sample dicts
        device: torch device
        bounds_cache: dict si -> bounds
        deep_layers: list of deep layer indices
        anchor_targets: list of anchor targets
        output_dir: Path to save results
        diagnosis: optional diagnosis dict (for auto-selection comparison)

    Returns:
        dict with all results and comparison table
    """
    print(f"\n{'='*60}")
    print(f"  Full D2 Comparison: {model_cfg.name}")
    print(f"  Testing {len(COMPARISON_CONFIGS)} configs on {len(samples)} samples")
    print(f"{'='*60}")

    results = []
    baseline_d2 = None

    for entry in COMPARISON_CONFIGS:
        cfg_name = entry["name"]
        cfg = entry["config"]

        result = evaluate_d2_with_intervention(
            model, processor, model_cfg, samples, device,
            bounds_cache, deep_layers, anchor_targets,
            intervention_config=cfg,
        )
        results.append(result)

        if cfg is None:
            baseline_d2 = result["d2_mean"]

    # Compute deltas
    for r in results:
        r["d2_delta"] = round(r["d2_mean"] - baseline_d2, 4) if baseline_d2 is not None else 0.0

    # Auto-selection result (if diagnosis available)
    auto_result = None
    if diagnosis and diagnosis.get("recommended_hook"):
        auto_config = _build_auto_config(diagnosis)
        auto_result = evaluate_d2_with_intervention(
            model, processor, model_cfg, samples, device,
            bounds_cache, deep_layers, anchor_targets,
            intervention_config=auto_config,
        )
        auto_result["d2_delta"] = round(
            auto_result["d2_mean"] - baseline_d2, 4
        ) if baseline_d2 is not None else 0.0
        auto_result["is_auto_selected"] = True
        auto_result["classification"] = diagnosis["classification"]
        results.append(auto_result)

    # --- Print comparison table ---
    print(f"\n{'='*60}")
    print(f"  COMPARISON TABLE: {model_cfg.name}")
    print(f"{'='*60}")
    print(f"  {'Config':<28s} {'D2':>8s} {'dD2':>8s} {'H':>8s}")
    print(f"  {'-'*52}")
    for r in results:
        name = r["intervention_name"]
        if r.get("is_auto_selected"):
            name = f"*AUTO* {name}"
        print(f"  {name:<28s} {r['d2_mean']:>8.4f} {r['d2_delta']:>+8.4f} "
              f"{r['entropy_mean']:>8.4f}")

    # Find best config
    best = max(results, key=lambda r: r["d2_mean"])
    print(f"\n  Best config: {best['intervention_name']} (D2={best['d2_mean']:.4f})")
    if auto_result:
        print(f"  Auto-selected: {auto_result['intervention_name']} "
              f"(D2={auto_result['d2_mean']:.4f}, "
              f"delta={auto_result['d2_delta']:+.4f})")

    # Save results
    comparison = {
        "model": model_cfg.name,
        "n_samples": len(samples),
        "baseline_d2": baseline_d2,
        "results": results,
        "best_config": best["intervention_name"],
        "best_d2": best["d2_mean"],
        "diagnosis_classification": diagnosis["classification"] if diagnosis else None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  Comparison saved to {output_dir / 'comparison.json'}")

    return comparison


def _build_auto_config(diagnosis):
    """Build intervention config from diagnosis recommendation."""
    hook = diagnosis["recommended_hook"]
    params = diagnosis["recommended_params"]

    if hook == "var":
        return {"method": "var", "p": params["p"]}
    elif hook == "kscale":
        return {"method": "kscale", "alpha": params["alpha"]}
    elif hook == "hybrid":
        return {"method": "hybrid", "p": params["p"], "alpha": params["alpha"]}
    return None


# =============================================================================
# CLI Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Routing: Diagnose VLA routing failure and auto-select intervention"
    )
    parser.add_argument("--model", required=True,
                        help="Model name from registry (e.g., ecot-7b, openvla-7b)")
    parser.add_argument("--device", default="cuda:0",
                        help="Torch device (default: cuda:0)")
    parser.add_argument("--n_samples", type=int, default=20,
                        help="Number of samples for D2 comparison (default: 20)")
    parser.add_argument("--n_diag", type=int, default=3,
                        help="Number of samples for diagnosis (default: 3)")
    parser.add_argument("--mode", default="both",
                        choices=["diagnose", "compare", "both"],
                        help="Run mode: diagnose only, compare only, or both")
    parser.add_argument("--gate1_dir", default=None,
                        help="Path to gate1 directory with sample_list.json")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory override")
    args = parser.parse_args()

    # --- Paths ---
    model_name = args.model
    gate1_dir = Path(args.gate1_dir) if args.gate1_dir else \
        config.OUTPUT_DIR / "phase3_gate" / model_name
    out = Path(args.output_dir) if args.output_dir else \
        config.OUTPUT_DIR / "phase3_gate" / "adaptive" / model_name
    verification_dir = config.OUTPUT_DIR / "phase3_gate" / "verification"
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Adaptive Routing Pipeline")
    print(f"  Model: {model_name}")
    print(f"  Mode:  {args.mode}")
    print(f"  Output: {out}")
    print(f"{'='*60}")

    # --- Load model ---
    print(f"\nLoading {model_name}...")
    processor, model, model_cfg = load_model_from_registry(model_name, args.device)

    # --- Load samples ---
    sample_list_path = gate1_dir / "sample_list.json"
    print(f"Loading samples from {sample_list_path}...")
    samples = reload_samples_from_list(sample_list_path, config.DATA_CACHE_DIR)
    samples = samples[:args.n_samples]
    print(f"  Loaded {len(samples)} samples")

    # --- Compute deep layers and bounds ---
    deep_layers = list(range(
        max(0, model_cfg.num_layers - 10), model_cfg.num_layers
    ))
    print(f"  Deep layers: {deep_layers}")

    bounds_cache = {}
    for si, s in enumerate(samples):
        bounds_cache[si] = detect_token_boundaries(
            processor, model, s["image"], s["instruction"],
            args.device, model_cfg,
        )

    # --- Detect anchor targets ---
    anchor_targets = detect_anchor_targets(
        model_cfg, verification_dir, bounds_cache[0]
    )
    print(f"  Anchor targets: {anchor_targets}")

    # --- Run diagnosis ---
    diagnosis = None
    if args.mode in ("diagnose", "both"):
        router = AdaptiveRouter(
            model, processor, model_cfg, args.device, samples,
            bounds_cache, deep_layers, out, verification_dir,
        )
        diagnosis = router.diagnose(n_samples=args.n_diag)

        # Also try applying and immediately removing the hook (sanity check)
        hooks = router.apply_optimal_hook()
        router.remove_hook()

    # --- Run full comparison ---
    if args.mode in ("compare", "both"):
        # If we didn't diagnose, try loading existing diagnosis
        if diagnosis is None:
            diag_path = out / "diagnosis.json"
            if diag_path.exists():
                with open(diag_path) as f:
                    diagnosis = json.load(f)
                print(f"  Loaded existing diagnosis: {diagnosis['classification']}")
            else:
                print("  WARNING: No diagnosis available, running without auto-selection")

        run_full_comparison(
            model, processor, model_cfg, samples, args.device,
            bounds_cache, deep_layers, anchor_targets,
            out, diagnosis=diagnosis,
        )

    # --- Cleanup ---
    del model
    torch.cuda.empty_cache()
    print(f"\nDone. Results in: {out}")


if __name__ == "__main__":
    main()
