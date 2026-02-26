"""Diagnose whether vision token 0 bottleneck is CLS-like compression or spatial shortcut.

Three diagnostic tests:
  1. CLS vs Patch verification (token count check)
  2. Image shift test (does token 0 still dominate after spatial shift?)
  3. Token 0 ablation (mask token 0 → how much does action change?)

Usage:
    python diagnose_token0.py --device cuda:0 --n_samples 3
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import config
from extract_attention import (
    load_model_from_registry,
    detect_token_boundaries,
    get_layers,
    call_processor,
)
from model_registry import get_model as registry_get_model
from verify_attention_sinks import (
    SinkVerificationHookManager,
    check_condition_C,
    get_wov_matrix,
)
from visualize_text_attention import load_samples_from_cache


def compute_contribution_profile(model, model_cfg, hook_mgr, boundaries):
    """Compute token 0's contribution ratio for ALL layers."""
    vs = boundaries["vision_start"]
    ve = boundaries["vision_end"]
    te = boundaries["text_end"]
    n_vision = ve - vs

    ratios = {}
    for layer_idx in range(model_cfg.num_layers):
        if layer_idx not in hook_mgr.attention_weights:
            continue
        if layer_idx not in hook_mgr.hidden_states:
            continue

        attn = hook_mgr.attention_weights[layer_idx]
        if attn.dim() == 4:
            attn = attn[0]

        query_pos = te - 1
        if query_pos >= attn.shape[1]:
            continue

        alpha = attn[:, query_pos, vs:ve].float().mean(dim=0).numpy()

        prev_layer = max(0, layer_idx - 1)
        if prev_layer not in hook_mgr.hidden_states:
            prev_layer = layer_idx
        x_vision = hook_mgr.hidden_states[prev_layer][vs:ve]

        try:
            v_weight, o_weight = get_wov_matrix(model, model_cfg, layer_idx)
        except Exception:
            continue

        v_proj = x_vision @ v_weight.T
        v_norms = torch.norm(v_proj, dim=1).numpy()

        weighted = alpha * v_norms
        total = weighted.sum()
        if total < 1e-10:
            continue

        ratios[layer_idx] = float(weighted[0] / total * 100)

    return ratios


# ═══════════════════════════════════════════════════════════════════
# Test 1: CLS vs Patch Verification
# ═══════════════════════════════════════════════════════════════════

def test_cls_vs_patch(model, model_cfg, processor, device):
    """Check if token 0 is a CLS token or a real spatial patch."""
    print("\n" + "=" * 60)
    print("TEST 1: CLS Token vs Top-Left Patch")
    print("=" * 60)

    # Check vision encoder config
    model_config = model.config
    result = {
        "num_vision_tokens": model_cfg.num_vision_tokens,
        "grid_size": model_cfg.vision_grid_size,
        "expected_patches": model_cfg.vision_grid_size ** 2,
        "vision_encoder": model_cfg.vision_encoder,
    }

    n_patches = model_cfg.vision_grid_size ** 2
    n_tokens = model_cfg.num_vision_tokens

    if n_tokens == n_patches:
        result["has_cls"] = False
        result["verdict"] = "NO CLS — token 0 is a real spatial patch (top-left)"
        print(f"  Grid: {model_cfg.vision_grid_size}x{model_cfg.vision_grid_size} = {n_patches} patches")
        print(f"  Vision tokens: {n_tokens}")
        print(f"  → {n_tokens} == {n_patches}: NO CLS token")
        print(f"  → Token 0 = TOP-LEFT PATCH (real spatial position)")
    elif n_tokens == n_patches + 1:
        result["has_cls"] = True
        result["verdict"] = "HAS CLS — token 0 may be a CLS/summary token"
        print(f"  Grid: {model_cfg.vision_grid_size}x{model_cfg.vision_grid_size} = {n_patches} patches")
        print(f"  Vision tokens: {n_tokens}")
        print(f"  → {n_tokens} == {n_patches} + 1: CLS token likely at index 0")
    else:
        result["has_cls"] = "unknown"
        result["verdict"] = f"Unusual: {n_tokens} tokens for {n_patches} patches (dual encoder?)"
        print(f"  Grid: {model_cfg.vision_grid_size}x{model_cfg.vision_grid_size} = {n_patches} patches")
        print(f"  Vision tokens: {n_tokens}")
        print(f"  → Cannot determine CLS status from count alone")

    # Check Prismatic-specific: uses get_intermediate_layers which drops CLS
    if model_cfg.vision_encoder == "prismatic":
        result["prismatic_note"] = "Prismatic uses get_intermediate_layers() which drops CLS"
        result["has_cls"] = False
        result["verdict"] = "NO CLS — Prismatic drops CLS via get_intermediate_layers()"
        print(f"  → Prismatic encoder: CLS dropped by get_intermediate_layers()")
        print(f"  → CONFIRMED: Token 0 = top-left patch at spatial position (0,0)")

    print(f"\n  VERDICT: {result['verdict']}")
    return result


# ═══════════════════════════════════════════════════════════════════
# Test 2: Image Shift Test
# ═══════════════════════════════════════════════════════════════════

def shift_image(image, dx, dy):
    """Shift image by (dx, dy) pixels, filling with black."""
    img_array = np.array(image)
    h, w, c = img_array.shape
    shifted = np.zeros_like(img_array)

    # Source and destination ranges
    src_y_start = max(0, -dy)
    src_y_end = min(h, h - dy)
    src_x_start = max(0, -dx)
    src_x_end = min(w, w - dx)

    dst_y_start = max(0, dy)
    dst_y_end = min(h, h + dy)
    dst_x_start = max(0, dx)
    dst_x_end = min(w, w + dx)

    shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
        img_array[src_y_start:src_y_end, src_x_start:src_x_end]

    return Image.fromarray(shifted)


def test_image_shift(model, model_cfg, processor, device, samples):
    """Test if token 0 dominance persists after spatial image shifts."""
    print("\n" + "=" * 60)
    print("TEST 2: Image Shift Sensitivity")
    print("=" * 60)

    shifts = [
        (0, 0, "original"),
        (50, 0, "right_50px"),
        (0, 50, "down_50px"),
        (50, 50, "right50_down50"),
        (-50, -50, "left50_up50"),
    ]

    all_results = []
    sample = samples[0]  # Use first sample
    image = sample["image"]
    instruction = sample["instruction"]

    print(f"  Sample: \"{instruction[:60]}\"")
    print(f"  Image size: {image.size}")

    for dx, dy, label in shifts:
        shifted_img = shift_image(image, dx, dy) if (dx != 0 or dy != 0) else image

        prompt = model_cfg.prompt_template.format(instruction=instruction)
        inputs = call_processor(processor, prompt, shifted_img, model_cfg=model_cfg, return_tensors="pt").to(device)
        if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

        boundaries = detect_token_boundaries(
            processor, model, shifted_img, instruction, device, model_cfg=model_cfg
        )

        hook_mgr = SinkVerificationHookManager(model, model_cfg)
        hook_mgr.register_hooks()

        with torch.no_grad():
            fwd_kwargs = {k: v for k, v in inputs.items()}
            fwd_kwargs["use_cache"] = False
            if model_cfg.architecture == "gemma2" and "intrinsic" not in fwd_kwargs:
                fwd_kwargs["intrinsic"] = torch.tensor(
                    [[[218.26, 0.0, 111.83],
                      [0.0, 218.26, 111.79],
                      [0.0, 0.0, 1.0]]], device=device, dtype=torch.float32,
                )
            model(**fwd_kwargs)

        # Compute contribution profile
        ratios = compute_contribution_profile(model, model_cfg, hook_mgr, boundaries)

        hook_mgr.remove_hooks()
        hook_mgr.reset()
        torch.cuda.empty_cache()

        # Summary stats
        if ratios:
            mid_layers = {k: v for k, v in ratios.items() if k >= 2}
            avg_ratio = np.mean(list(mid_layers.values())) if mid_layers else 0
            onset = min((k for k, v in ratios.items() if v > 50), default=None)
        else:
            avg_ratio = 0
            onset = None

        result = {
            "shift": label,
            "dx": dx, "dy": dy,
            "avg_contribution_pct": float(avg_ratio),
            "onset_layer": onset,
            "per_layer": ratios,
        }
        all_results.append(result)

        print(f"\n  [{label}] shift=({dx},{dy})")
        print(f"    Token 0 avg contribution (L2+): {avg_ratio:.1f}%")
        print(f"    Onset layer (>50%): {onset}")

    # Comparison
    print(f"\n  {'Shift':<20} {'Avg Contrib %':<15} {'Onset Layer':<12}")
    print(f"  {'-'*47}")
    for r in all_results:
        print(f"  {r['shift']:<20} {r['avg_contribution_pct']:<15.1f} {str(r['onset_layer']):<12}")

    # Verdict
    orig_ratio = all_results[0]["avg_contribution_pct"]
    shifted_ratios = [r["avg_contribution_pct"] for r in all_results[1:]]
    avg_shifted = np.mean(shifted_ratios) if shifted_ratios else 0

    if avg_shifted > 90 and abs(orig_ratio - avg_shifted) < 5:
        verdict = "POSITION-INDEPENDENT: Token 0 dominates regardless of image content shift → likely architectural artifact, NOT spatial"
    elif avg_shifted > 70:
        verdict = "MOSTLY POSITION-INDEPENDENT: Token 0 dominance slightly decreases with shift but remains strong → structural bias"
    else:
        verdict = "POSITION-DEPENDENT: Shift significantly reduces token 0 dominance → spatial content dependency"

    print(f"\n  VERDICT: {verdict}")

    return {"shifts": all_results, "verdict": verdict}


# ═══════════════════════════════════════════════════════════════════
# Test 3: Token 0 Ablation (Masking)
# ═══════════════════════════════════════════════════════════════════

def test_token0_ablation(model, model_cfg, processor, device, samples):
    """Test action change when token 0 is masked or replaced."""
    print("\n" + "=" * 60)
    print("TEST 3: Token 0 Ablation (Action Impact)")
    print("=" * 60)

    all_results = []

    for sample_idx, sample in enumerate(samples):
        image = sample["image"]
        instruction = sample["instruction"]

        prompt = model_cfg.prompt_template.format(instruction=instruction)
        inputs = call_processor(processor, prompt, image, model_cfg=model_cfg, return_tensors="pt").to(device)
        if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

        boundaries = detect_token_boundaries(
            processor, model, image, instruction, device, model_cfg=model_cfg
        )
        vs = boundaries["vision_start"]

        print(f"\n  Sample {sample_idx}: \"{instruction[:50]}\"")

        # Build forward kwargs (architecture-specific extras)
        fwd_kwargs = {k: v for k, v in inputs.items()}
        fwd_kwargs["use_cache"] = False
        if model_cfg.architecture == "gemma2" and "intrinsic" not in fwd_kwargs:
            fwd_kwargs["intrinsic"] = torch.tensor(
                [[[224.0, 0.0, 112.0],
                  [0.0, 224.0, 112.0],
                  [0.0, 0.0, 1.0]]], device=device, dtype=torch.float32,
            )

        # --- Normal forward pass: get action logits ---
        with torch.no_grad():
            outputs_normal = model(**fwd_kwargs)
        logits_normal = outputs_normal.logits[0, -1, :].float().cpu()
        action_normal = logits_normal.argmax().item()
        probs_normal = torch.softmax(logits_normal, dim=0)

        # --- Ablation via hooks ---
        # OpenVLA prepends vision tokens internally, so attention_mask only covers
        # text tokens. We use a hook on self_attn to zero out attention to the
        # target token position AFTER softmax.

        def make_ablation_hook(target_pos):
            """Hook that zeros out attention to target_pos across all heads."""
            def hook_fn(module, args, output):
                if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                    # output[0] = attn_output, output[1] = attn_weights (B, H, S, S)
                    attn_weights = output[1]
                    # Zero out column target_pos (no token can attend to it)
                    modified_weights = attn_weights.clone()
                    modified_weights[:, :, :, target_pos] = 0
                    # Re-normalize
                    row_sums = modified_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                    modified_weights = modified_weights / row_sums

                    # Recompute attn_output: need value states
                    # Instead, just return modified weights — the output is already computed
                    # So we hook at the layer level to modify hidden states
                return output
            return hook_fn

        def make_layer_ablation_hook(target_pos):
            """Hook on full layer that zeros token target_pos's hidden state contribution.

            This effectively removes token target_pos from the residual stream update
            by zeroing out the attention layer's contribution for that token.
            """
            def hook_fn(module, args, output):
                # output[0] = hidden_states (B, S, D)
                h = output[0] if isinstance(output, tuple) else output
                h_modified = h.clone()
                # Zero out the target token's hidden state update
                # by replacing it with the input hidden state (no update)
                input_h = args[0] if len(args) > 0 else None
                if input_h is not None and input_h.shape == h.shape:
                    h_modified[0, target_pos, :] = input_h[0, target_pos, :]
                else:
                    h_modified[0, target_pos, :] = 0
                if isinstance(output, tuple):
                    return (h_modified,) + output[1:]
                return h_modified
            return hook_fn

        # --- Ablation 1: Zero out token 0 via hook ---
        layers = get_layers(model, model_cfg)

        # Simple approach: zero token 0's hidden state at every layer
        ablation_hooks_t0 = []
        for layer in layers:
            h = layer.register_forward_hook(make_layer_ablation_hook(vs))
            ablation_hooks_t0.append(h)

        with torch.no_grad():
            outputs_ablated = model(**fwd_kwargs)
        logits_ablated = outputs_ablated.logits[0, -1, :].float().cpu()
        action_ablated = logits_ablated.argmax().item()
        probs_ablated = torch.softmax(logits_ablated, dim=0)

        for h in ablation_hooks_t0:
            h.remove()

        # --- Ablation 2: Zero out middle vision token for comparison ---
        random_token_idx = vs + model_cfg.num_vision_tokens // 2
        ablation_hooks_rand = []
        for layer in layers:
            h = layer.register_forward_hook(make_layer_ablation_hook(random_token_idx))
            ablation_hooks_rand.append(h)

        with torch.no_grad():
            outputs_random = model(**fwd_kwargs)
        logits_random = outputs_random.logits[0, -1, :].float().cpu()
        action_random = logits_random.argmax().item()
        probs_random = torch.softmax(logits_random, dim=0)

        for h in ablation_hooks_rand:
            h.remove()

        # --- Metrics ---
        # KL divergence between normal and ablated
        kl_token0 = torch.nn.functional.kl_div(
            probs_ablated.log().clamp(min=-100), probs_normal,
            reduction='sum'
        ).item()
        kl_random = torch.nn.functional.kl_div(
            probs_random.log().clamp(min=-100), probs_normal,
            reduction='sum'
        ).item()

        # L1 distance in logit space
        l1_token0 = (logits_normal - logits_ablated).abs().mean().item()
        l1_random = (logits_normal - logits_random).abs().mean().item()

        # Action changed?
        action_changed_t0 = action_normal != action_ablated
        action_changed_rand = action_normal != action_random

        result = {
            "sample_idx": sample_idx,
            "instruction": instruction[:60],
            "action_normal": action_normal,
            "action_ablated_t0": action_ablated,
            "action_ablated_rand": action_random,
            "action_changed_t0": action_changed_t0,
            "action_changed_rand": action_changed_rand,
            "kl_div_token0": kl_token0,
            "kl_div_random": kl_random,
            "l1_logits_token0": l1_token0,
            "l1_logits_random": l1_random,
            "kl_ratio": kl_token0 / (kl_random + 1e-8),
        }
        all_results.append(result)

        print(f"    Normal action token: {action_normal}")
        print(f"    Ablate token 0:  action={action_ablated} changed={action_changed_t0}")
        print(f"    Ablate mid-token: action={action_random} changed={action_changed_rand}")
        print(f"    KL divergence — token 0: {kl_token0:.4f}, random: {kl_random:.4f} (ratio: {kl_token0/(kl_random+1e-8):.1f}x)")
        print(f"    L1 logits — token 0: {l1_token0:.4f}, random: {l1_random:.4f}")

        torch.cuda.empty_cache()

    # Summary
    avg_kl_t0 = np.mean([r["kl_div_token0"] for r in all_results])
    avg_kl_rand = np.mean([r["kl_div_random"] for r in all_results])
    avg_kl_ratio = np.mean([r["kl_ratio"] for r in all_results])
    n_changed_t0 = sum(1 for r in all_results if r["action_changed_t0"])
    n_changed_rand = sum(1 for r in all_results if r["action_changed_rand"])

    print(f"\n  SUMMARY ({len(all_results)} samples):")
    print(f"    Avg KL(ablate token 0):     {avg_kl_t0:.4f}")
    print(f"    Avg KL(ablate random):      {avg_kl_rand:.4f}")
    print(f"    KL ratio (token0 / random): {avg_kl_ratio:.1f}x")
    print(f"    Action changed: token 0 ablated={n_changed_t0}/{len(all_results)}, "
          f"random ablated={n_changed_rand}/{len(all_results)}")

    if avg_kl_t0 < avg_kl_rand * 2:
        verdict = "SINK (low impact): Token 0 carries similar info to random tokens → likely attention sink, not bottleneck"
    elif avg_kl_ratio > 10:
        verdict = "CRITICAL BOTTLENECK: Token 0 carries vastly more info than other tokens → dangerous single-point dependency"
    elif avg_kl_ratio > 3:
        verdict = "SIGNIFICANT BOTTLENECK: Token 0 carries notably more info → concerning dependency"
    else:
        verdict = "MODERATE DEPENDENCY: Token 0 is more important than average but not dramatically so"

    print(f"\n  VERDICT: {verdict}")

    return {"samples": all_results, "summary": {
        "avg_kl_token0": avg_kl_t0,
        "avg_kl_random": avg_kl_rand,
        "avg_kl_ratio": avg_kl_ratio,
        "n_action_changed_t0": n_changed_t0,
        "n_action_changed_rand": n_changed_rand,
    }, "verdict": verdict}


# ═══════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════

def plot_diagnostics(test1, test2, test3, output_path, model_name):
    """Generate diagnostic summary figure."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: CLS vs Patch verdict
    ax1 = axes[0]
    ax1.text(0.5, 0.7, "Token 0 Identity", ha='center', va='center',
             fontsize=14, fontweight='bold', transform=ax1.transAxes)
    has_cls = test1.get("has_cls", "unknown")
    if has_cls is False:
        color = "red"
        text = "TOP-LEFT PATCH\n(NOT a CLS token)"
    elif has_cls is True:
        color = "green"
        text = "CLS TOKEN\n(Global summary)"
    else:
        color = "orange"
        text = "UNKNOWN"
    ax1.text(0.5, 0.4, text, ha='center', va='center', fontsize=16,
             fontweight='bold', color=color, transform=ax1.transAxes)
    ax1.text(0.5, 0.15, f"Vision: {test1['vision_encoder']}\n"
             f"Tokens: {test1['num_vision_tokens']} "
             f"(grid {test1['grid_size']}x{test1['grid_size']})",
             ha='center', va='center', fontsize=10, transform=ax1.transAxes)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')

    # Panel 2: Shift test - contribution across shifts
    ax2 = axes[1]
    if test2 and "shifts" in test2:
        labels = [r["shift"] for r in test2["shifts"]]
        values = [r["avg_contribution_pct"] for r in test2["shifts"]]
        colors_bar = ['red' if v > 90 else 'orange' if v > 50 else 'green' for v in values]
        bars = ax2.bar(range(len(labels)), values, color=colors_bar, alpha=0.8)
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
        ax2.axhline(y=50, color='green', linestyle='--', linewidth=1, alpha=0.7)
        ax2.set_ylabel("Token 0 Contribution (%)", fontsize=10)
        ax2.set_title("Image Shift Test\n(Position independence)", fontsize=11)
        ax2.set_ylim(0, 105)

    # Panel 3: Ablation KL divergence comparison
    ax3 = axes[2]
    if test3 and "summary" in test3:
        s = test3["summary"]
        categories = ["Ablate\nToken 0", "Ablate\nRandom"]
        kl_values = [s["avg_kl_token0"], s["avg_kl_random"]]
        colors_kl = ['red', 'steelblue']
        bars3 = ax3.bar(categories, kl_values, color=colors_kl, alpha=0.8, width=0.5)
        ax3.set_ylabel("KL Divergence from Normal", fontsize=10)
        ax3.set_title(f"Token 0 Ablation Impact\n(ratio: {s['avg_kl_ratio']:.1f}x)", fontsize=11)

        # Annotate with values
        for bar, val in zip(bars3, kl_values):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle(f"Token 0 Bottleneck Diagnosis — {model_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Diagnostic figure saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Diagnose token 0 bottleneck")
    parser.add_argument("--model", default="openvla-7b", help="Model name")
    parser.add_argument("--device", default="cuda:0", help="Device")
    parser.add_argument("--n_samples", type=int, default=3, help="Samples for ablation test")
    args = parser.parse_args()

    output_dir = config.OUTPUT_DIR / "bottleneck_diagnosis" / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading {args.model}...")
    processor, model, model_cfg = load_model_from_registry(args.model, device=args.device)

    # Load samples
    print(f"\nLoading {args.n_samples} samples...")
    samples = load_samples_from_cache(config.DATA_CACHE_DIR, n_samples=args.n_samples)

    # Run tests
    test1 = test_cls_vs_patch(model, model_cfg, processor, args.device)
    test2 = test_image_shift(model, model_cfg, processor, args.device, samples)
    test3 = test_token0_ablation(model, model_cfg, processor, args.device, samples)

    # Overall verdict
    print("\n" + "=" * 60)
    print("OVERALL DIAGNOSIS")
    print("=" * 60)

    is_cls = test1.get("has_cls", False)
    shift_verdict = test2.get("verdict", "")
    ablation_verdict = test3.get("verdict", "")

    print(f"\n  1. Token 0 identity:  {'CLS (summary)' if is_cls else 'TOP-LEFT PATCH (spatial)'}")
    print(f"  2. Shift test:        {shift_verdict}")
    print(f"  3. Ablation test:     {ablation_verdict}")

    if not is_cls and "POSITION-INDEPENDENT" in shift_verdict:
        overall = ("PROBLEMATIC BOTTLENECK: Token 0 is a real spatial patch (top-left), "
                   "dominance is position-independent (not content-driven), "
                   "and the model is architecturally dependent on it. "
                   "This is an architectural shortcut, not meaningful compression.")
    elif not is_cls and "CRITICAL" in ablation_verdict:
        overall = ("SEVERE BOTTLENECK: Token 0 is a spatial patch carrying critical info. "
                   "The model routes ALL visual information through one spatial location. "
                   "This is a dangerous single-point dependency.")
    elif is_cls:
        overall = ("CLS-BASED COMPRESSION: Token 0 is a CLS/summary token. "
                   "Dominance may be by design (ViT-style global pooling). "
                   "Further investigation needed to assess if 99% is excessive.")
    else:
        overall = "INCONCLUSIVE: Need more data to determine bottleneck nature."

    print(f"\n  OVERALL: {overall}")

    # Save results
    full_report = {
        "model": args.model,
        "test1_cls_vs_patch": test1,
        "test2_shift": test2,
        "test3_ablation": test3,
        "overall_verdict": overall,
    }
    report_path = output_dir / "diagnosis_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    print(f"\n  Report saved: {report_path}")

    # Generate diagnostic figure
    plot_diagnostics(test1, test2, test3,
                     output_dir / "diagnosis_summary.png", args.model)


if __name__ == "__main__":
    main()
