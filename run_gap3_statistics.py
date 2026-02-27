#!/usr/bin/env python3
"""Gap3: Statistical Strengthening — Bootstrap CI + Significance Tests.

Addresses Gap 3 from SCI readiness assessment:
  - Bootstrap 95% CI on all key metrics (vision[0] share, top-1 contribution, entropy)
  - Wilcoxon signed-rank for cross-model comparisons
  - Cohen's d effect sizes
  - Per-skill variance analysis
  - Subsampling robustness check (50% splits)

Two modes:
  1. --collect: Run per-sample metric collection (requires GPU + model)
  2. --analyze: Compute statistics from collected per-sample data (CPU only)

Usage:
  # Collect per-sample data (one model per GPU)
  CUDA_VISIBLE_DEVICES=0 python run_gap3_statistics.py --collect --model ecot-7b --device cuda:0 --n_per_skill 50
  CUDA_VISIBLE_DEVICES=2 python run_gap3_statistics.py --collect --model openvla-7b --device cuda:0 --n_per_skill 50

  # Analyze all collected data
  python run_gap3_statistics.py --analyze --models ecot-7b openvla-7b spatialvla-4b tracevla-phi3v
"""
import argparse
import json
import sys
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

import config
from extract_attention import load_model_from_registry, get_layers, call_processor, detect_token_boundaries
from verify_attention_sinks import SinkVerificationHookManager, get_wov_matrix
from contribution.compute import compute_perhead_contribution


# ═══════════════════════════════════════════════════════════════
# Phase 1: Per-Sample Metric Collection (GPU required)
# ═══════════════════════════════════════════════════════════════

def collect_per_sample_metrics(
    model_name: str,
    device: str,
    n_per_skill: int = 50,
    seed: int = 42,
    output_dir: Path = None,
):
    """Collect per-sample raw metrics for bootstrap analysis.

    For each sample, measures:
      - vision[0] attention share (per deep layer, averaged over heads)
      - top-1 contribution share (per deep layer)
      - contribution entropy (per deep layer)
      - skill label
    """
    from data_sampler import load_balanced_samples
    from model_registry import get_model as registry_get_model

    output_dir = output_dir or Path("outputs/gap3_statistics") / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading model: {model_name}")
    processor, model, model_cfg = load_model_from_registry(model_name, device)
    model.eval()

    layers = get_layers(model, model_cfg)
    n_layers = len(layers)
    deep_layers = list(range(n_layers - 10, n_layers))
    print(f"  Deep layers: {deep_layers}")

    # Load balanced samples (expanded count)
    target_skills = ["place", "move", "pick", "fold", "open", "close"]
    print(f"  Loading {n_per_skill} samples per skill ({len(target_skills)} skills = {n_per_skill * len(target_skills)} total)")

    samples = load_balanced_samples(
        cache_dir=config.DATA_CACHE_DIR,
        n_per_skill=n_per_skill,
        target_skills=target_skills,
        seed=seed,
    )
    print(f"  Loaded {len(samples)} samples")

    # Detect token boundaries from first sample
    sample0 = samples[0]
    bounds = detect_token_boundaries(processor, model, sample0["image"], sample0["instruction"], device, model_cfg=model_cfg)

    vs = bounds["vision_start"]
    ve = bounds["vision_end"]
    n_vision = ve - vs
    print(f"  Vision range: [{vs}:{ve}] ({n_vision} tokens)")

    # Build prompt template
    prompt_template = model_cfg.prompt_template if model_cfg else "In: What action should the robot take to {instruction}?\nOut:"

    # Collect per-sample metrics
    all_sample_metrics = []
    hook_mgr = SinkVerificationHookManager(model, model_cfg)

    for si, sample in enumerate(samples):
        if si % 20 == 0:
            print(f"  Processing sample {si+1}/{len(samples)}...")

        prompt = prompt_template.format(instruction=sample["instruction"])
        inputs = call_processor(processor, prompt, sample["image"], model_cfg, return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
        seq_len = inputs["input_ids"].shape[1]
        query_pos = seq_len - 1  # last token

        # Register hooks and run forward
        hook_mgr.register_hooks()
        hook_mgr.reset()
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, use_cache=False)

        # Use model outputs for attention (move to CPU) and hooks for hidden states (already CPU)
        attentions = outputs.attentions if hasattr(outputs, "attentions") and outputs.attentions else None
        hidden_states = hook_mgr.hidden_states  # dict: layer_idx -> (seq, D) on CPU
        hook_mgr.remove_hooks()

        sample_metrics = {
            "sample_idx": si,
            "skill": sample.get("skill", "unknown"),
            "instruction": sample.get("instruction", ""),
            "per_layer": {},
        }

        for l in deep_layers:
            # Get attention weights from model output (move to CPU)
            if attentions is not None and l < len(attentions):
                attn = attentions[l][0].detach().cpu()  # (H, seq, seq)
            else:
                sample_metrics["per_layer"][str(l)] = {"error": "no attention"}
                continue

            # Vision[0] attention share (averaged over heads)
            # attn[h, query_pos, vision[0]] / sum(attn[h, query_pos, vs:ve])
            h_count = attn.shape[0]
            v0_shares = []
            for h in range(h_count):
                vision_attn = attn[h, query_pos, vs:ve].float()
                total_vision = vision_attn.sum().item()
                if total_vision > 1e-10:
                    v0_share = attn[h, query_pos, vs].float().item() / total_vision
                else:
                    v0_share = 0.0
                v0_shares.append(v0_share)
            v0_attn_mean = float(np.mean(v0_shares))

            # W_OV contribution
            h_prev = hidden_states.get(l)
            if h_prev is not None:
                try:
                    w_v, w_o = get_wov_matrix(model, model_cfg, l)
                    contrib = compute_perhead_contribution(
                        attn, h_prev, w_v, w_o, [query_pos]
                    )  # (H, 1, seq)
                    # Average over heads, get vision-range contribution
                    c_mean = contrib[:, 0, :].mean(dim=0).numpy()  # (seq,)
                    c_vision = c_mean[vs:ve]
                    c_total = c_vision.sum()
                    if c_total > 1e-10:
                        c_norm = c_vision / c_total
                        top1_share = float(c_norm.max())
                        # Entropy
                        c_clip = np.clip(c_norm, 1e-10, None)
                        entropy = float(-np.sum(c_clip * np.log(c_clip)))
                    else:
                        top1_share = 0.0
                        entropy = 0.0
                except Exception as e:
                    sample_metrics["per_layer"][str(l)] = {"error": str(e)}
                    continue
            else:
                top1_share = 0.0
                entropy = 0.0

            sample_metrics["per_layer"][str(l)] = {
                "v0_attn_share": v0_attn_mean,
                "top1_contrib_share": top1_share,
                "contrib_entropy": entropy,
            }

        all_sample_metrics.append(sample_metrics)

    # Save per-sample data
    out_path = output_dir / "per_sample_metrics.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": model_name,
            "n_samples": len(samples),
            "n_per_skill": n_per_skill,
            "deep_layers": deep_layers,
            "seed": seed,
            "samples": all_sample_metrics,
        }, f, indent=2)
    print(f"\n  Saved {len(samples)} per-sample metrics to {out_path}")

    del model
    torch.cuda.empty_cache()
    return all_sample_metrics


# ═══════════════════════════════════════════════════════════════
# Phase 2: Statistical Analysis (CPU only)
# ═══════════════════════════════════════════════════════════════

def bootstrap_ci(data, n_bootstrap=10000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    data = np.array(data)
    n = len(data)
    if n == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0}

    boot_means = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_means.append(np.mean(data[idx]))
    boot_means = np.array(boot_means)

    alpha = 1 - ci
    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return {
        "mean": float(np.mean(data)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std": float(np.std(data)),
        "n": n,
    }


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return float((m1 - m2) / pooled_std)


def wilcoxon_test(group1, group2):
    """Wilcoxon rank-sum test (non-parametric)."""
    from scipy import stats
    if len(group1) < 3 or len(group2) < 3:
        return {"statistic": 0.0, "p_value": 1.0}
    stat, p = stats.mannwhitneyu(group1, group2, alternative="two-sided")
    return {"statistic": float(stat), "p_value": float(p)}


def subsampling_robustness(data, metric_key, n_splits=100, frac=0.5, seed=42):
    """Check if metric holds on random 50% splits."""
    rng = np.random.RandomState(seed)
    n = len(data)
    half = int(n * frac)
    split_means = []
    for _ in range(n_splits):
        idx = rng.choice(n, size=half, replace=False)
        vals = [data[i] for i in idx]
        split_means.append(np.mean(vals))
    return {
        "full_mean": float(np.mean(data)),
        "split_mean_mean": float(np.mean(split_means)),
        "split_mean_std": float(np.std(split_means)),
        "split_mean_range": [float(np.min(split_means)), float(np.max(split_means))],
        "stable": float(np.std(split_means)) < 0.05 * abs(np.mean(data)) if abs(np.mean(data)) > 1e-10 else True,
    }


def analyze_collected_data(model_names: list[str], data_dir: Path):
    """Run full statistical analysis on collected per-sample data."""
    from scipy import stats

    all_model_data = {}
    for m in model_names:
        path = data_dir / m / "per_sample_metrics.json"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping {m}")
            continue
        with open(path) as f:
            all_model_data[m] = json.load(f)
        print(f"  Loaded {all_model_data[m]['n_samples']} samples for {m}")

    if not all_model_data:
        print("No data found!")
        return

    results = {"models": {}, "cross_model": {}, "subsampling": {}}

    # ── Per-Model Analysis ──────────────────────────────────────
    for m, data in all_model_data.items():
        samples = data["samples"]
        deep_layers = data["deep_layers"]

        # Aggregate per-sample: average across deep layers
        v0_attn_per_sample = []
        top1_per_sample = []
        entropy_per_sample = []
        skill_metrics = defaultdict(lambda: {"v0": [], "top1": [], "entropy": []})

        for s in samples:
            v0_vals = []
            top1_vals = []
            ent_vals = []
            for l in deep_layers:
                lm = s["per_layer"].get(str(l), {})
                if "error" in lm:
                    continue
                v0_vals.append(lm["v0_attn_share"])
                top1_vals.append(lm["top1_contrib_share"])
                ent_vals.append(lm["contrib_entropy"])

            if v0_vals:
                v0_mean = np.mean(v0_vals)
                top1_mean = np.mean(top1_vals)
                ent_mean = np.mean(ent_vals)
                v0_attn_per_sample.append(v0_mean)
                top1_per_sample.append(top1_mean)
                entropy_per_sample.append(ent_mean)

                skill = s.get("skill", "unknown")
                skill_metrics[skill]["v0"].append(v0_mean)
                skill_metrics[skill]["top1"].append(top1_mean)
                skill_metrics[skill]["entropy"].append(ent_mean)

        # Bootstrap CIs
        model_results = {
            "n_samples": len(v0_attn_per_sample),
            "v0_attn_share": bootstrap_ci(v0_attn_per_sample),
            "top1_contrib_share": bootstrap_ci(top1_per_sample),
            "contrib_entropy": bootstrap_ci(entropy_per_sample),
        }

        # Per-skill variance
        skill_analysis = {}
        for skill, metrics in sorted(skill_metrics.items()):
            skill_analysis[skill] = {
                "n": len(metrics["v0"]),
                "v0_attn": bootstrap_ci(metrics["v0"]),
                "top1_contrib": bootstrap_ci(metrics["top1"]),
                "entropy": bootstrap_ci(metrics["entropy"]),
            }
        model_results["per_skill"] = skill_analysis

        # Subsampling robustness
        model_results["robustness"] = {
            "v0_attn": subsampling_robustness(v0_attn_per_sample, "v0"),
            "top1_contrib": subsampling_robustness(top1_per_sample, "top1"),
            "entropy": subsampling_robustness(entropy_per_sample, "entropy"),
        }

        results["models"][m] = model_results
        print(f"\n  {m}:")
        print(f"    Vision[0] attn: {model_results['v0_attn_share']['mean']:.4f} "
              f"[{model_results['v0_attn_share']['ci_lower']:.4f}, "
              f"{model_results['v0_attn_share']['ci_upper']:.4f}]")
        print(f"    Top-1 contrib:  {model_results['top1_contrib_share']['mean']:.4f} "
              f"[{model_results['top1_contrib_share']['ci_lower']:.4f}, "
              f"{model_results['top1_contrib_share']['ci_upper']:.4f}]")
        print(f"    Entropy:        {model_results['contrib_entropy']['mean']:.4f} "
              f"[{model_results['contrib_entropy']['ci_lower']:.4f}, "
              f"{model_results['contrib_entropy']['ci_upper']:.4f}]")

    # ── Cross-Model Comparisons ────────────────────────────────
    model_list = list(all_model_data.keys())
    for i in range(len(model_list)):
        for j in range(i + 1, len(model_list)):
            m1, m2 = model_list[i], model_list[j]
            d1, d2 = all_model_data[m1], all_model_data[m2]

            # Collect per-sample means
            def get_per_sample_means(data):
                out = {"v0": [], "top1": [], "entropy": []}
                for s in data["samples"]:
                    v0_vals, t1_vals, ent_vals = [], [], []
                    for l in data["deep_layers"]:
                        lm = s["per_layer"].get(str(l), {})
                        if "error" not in lm:
                            v0_vals.append(lm["v0_attn_share"])
                            t1_vals.append(lm["top1_contrib_share"])
                            ent_vals.append(lm["contrib_entropy"])
                    if v0_vals:
                        out["v0"].append(np.mean(v0_vals))
                        out["top1"].append(np.mean(t1_vals))
                        out["entropy"].append(np.mean(ent_vals))
                return out

            ps1 = get_per_sample_means(d1)
            ps2 = get_per_sample_means(d2)

            comparison = {}
            for metric in ["v0", "top1", "entropy"]:
                comparison[metric] = {
                    "wilcoxon": wilcoxon_test(ps1[metric], ps2[metric]),
                    "cohens_d": cohens_d(ps1[metric], ps2[metric]),
                    f"{m1}_mean": float(np.mean(ps1[metric])),
                    f"{m2}_mean": float(np.mean(ps2[metric])),
                }
            results["cross_model"][f"{m1}_vs_{m2}"] = comparison

            print(f"\n  {m1} vs {m2}:")
            for metric in ["v0", "top1", "entropy"]:
                c = comparison[metric]
                sig = "***" if c["wilcoxon"]["p_value"] < 0.001 else "**" if c["wilcoxon"]["p_value"] < 0.01 else "*" if c["wilcoxon"]["p_value"] < 0.05 else "ns"
                print(f"    {metric}: d={c['cohens_d']:.3f}, p={c['wilcoxon']['p_value']:.4e} {sig}")

    # ── Save Results ────────────────────────────────────────────
    out_path = data_dir / "gap3_statistical_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Full analysis saved: {out_path}")

    # ── Summary Table ───────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  Gap3 Statistical Summary")
    print(f"{'='*80}")
    print(f"  {'Model':<20} {'Vision[0] Share':>25} {'Top-1 Contrib':>25} {'Entropy':>25}")
    print(f"  {'─'*95}")
    for m in model_list:
        r = results["models"][m]
        v0 = r["v0_attn_share"]
        t1 = r["top1_contrib_share"]
        en = r["contrib_entropy"]
        print(f"  {m:<20} {v0['mean']:.3f} [{v0['ci_lower']:.3f},{v0['ci_upper']:.3f}]"
              f"   {t1['mean']:.3f} [{t1['ci_lower']:.3f},{t1['ci_upper']:.3f}]"
              f"   {en['mean']:.3f} [{en['ci_lower']:.3f},{en['ci_upper']:.3f}]")

    return results


def main():
    parser = argparse.ArgumentParser(description="Gap3: Statistical Strengthening")
    parser.add_argument("--collect", action="store_true", help="Phase 1: collect per-sample metrics (GPU)")
    parser.add_argument("--analyze", action="store_true", help="Phase 2: compute statistics (CPU)")
    parser.add_argument("--model", type=str, help="Model name for --collect")
    parser.add_argument("--models", nargs="+", help="Models for --analyze")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_per_skill", type=int, default=50,
                        help="Samples per skill (default 50, total = 300)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/gap3_statistics")

    args = parser.parse_args()
    data_dir = Path(args.output_dir)

    if args.collect:
        if not args.model:
            parser.error("--collect requires --model")
        collect_per_sample_metrics(
            model_name=args.model,
            device=args.device,
            n_per_skill=args.n_per_skill,
            seed=args.seed,
            output_dir=data_dir / args.model,
        )

    elif args.analyze:
        models = args.models or ["ecot-7b", "openvla-7b", "spatialvla-4b", "tracevla-phi3v"]
        analyze_collected_data(models, data_dir)

    else:
        parser.error("Specify --collect or --analyze")


if __name__ == "__main__":
    main()
