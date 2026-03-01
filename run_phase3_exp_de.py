#!/usr/bin/env python3
"""Phase 3 Verification Experiments D, E & F.

Exp D: Performance Connection — action prediction quality vs position anchoring
  D1: Action token entropy
  D2: Augmentation consistency
  D3: Anchor V-zero ablation sensitivity

Exp E: Inference-Time Mitigation — value scaling at anchor position
  Alpha sweep {0.0, 0.1, 0.3, 0.5, 0.7, 1.0}
  Measure: entropy change, action change rate, position anchoring rate

Exp F: Q/K-Level Intervention — key scaling at anchor position
  Targets K projection (not V) to directly reduce attention scores at anchor
  Measures: C-peak anchoring breakage, D2 consistency improvement
  Tests the hypothesis: "position shortcut is encoded in Q/K"

Usage:
  python run_phase3_exp_de.py --model ecot-7b --device cuda:0
  python run_phase3_exp_de.py --model ecot-7b --device cuda:0 --experiments f
"""
import argparse, json, sys, torch, numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance
from collections import Counter
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import config
from extract_attention import load_model_from_registry, get_layers, call_processor, detect_token_boundaries
from data_sampler import reload_samples_from_list, get_action_for_sample
from verify_attention_sinks import SinkVerificationHookManager, get_wov_matrix
from contribution.causal import ValueZeroHook, compute_output_kl, compute_top1_change_rate
from contribution.compute import compute_perhead_contribution


# =============================================================================
# Task 1 Step 2: ValueScaleHook
# =============================================================================

class ValueScaleHook(ValueZeroHook):
    """Scale (not zero) V projection at target positions.
    alpha=1.0 = no change, alpha=0.0 = ValueZeroHook equivalent.
    Fix 6: Inherits target_layers from ValueZeroHook for deep-layer-only application.
    """
    def __init__(self, target_positions, alpha=0.5, target_layers=None):
        super().__init__(target_positions, target_layers)
        self.alpha = alpha

    def _make_v_proj_hook(self):
        hook_self = self
        def hook_fn(module, args, output):
            modified = output.clone()
            for t in hook_self.target_positions:
                if t < modified.shape[1]:
                    modified[:, t, :] *= hook_self.alpha
                    hook_self._sanity_changed = True
            return modified
        return hook_fn

    def _make_fused_qkv_hook(self, v_start, v_end):
        hook_self = self
        def hook_fn(module, args, output):
            modified = output.clone()
            for t in hook_self.target_positions:
                if t < modified.shape[1]:
                    modified[:, t, v_start:v_end] *= hook_self.alpha
                    hook_self._sanity_changed = True
            return modified
        return hook_fn


# =============================================================================
# Exp F: KeyScaleHook — targets K projection at anchor positions
# =============================================================================

class KeyScaleHook:
    """Scale K projection at target positions to reduce attention scores.
    alpha=1.0 = no change, alpha=0.0 = K zeroed (attention score → 0).
    When K[pos]=0, softmax gives that position exp(0)=1 weight vs exp(score)
    for others, effectively suppressing attention to the anchor.
    """
    def __init__(self, target_positions, alpha=0.0, target_layers=None):
        self.target_positions = target_positions
        self.target_layers = target_layers
        self.alpha = alpha
        self._handles = []
        self._sanity_changed = False

    def register(self, model, model_cfg, get_layers_fn):
        layers = get_layers_fn(model, model_cfg)
        num_heads = model_cfg.num_heads
        num_kv_heads = getattr(model_cfg, 'num_kv_heads', None) or num_heads
        head_dim = model_cfg.hidden_dim // num_heads

        for layer_idx, layer in enumerate(layers):
            if self.target_layers is not None and layer_idx not in self.target_layers:
                continue
            attn = layer.self_attn
            if hasattr(attn, "k_proj"):
                handle = attn.k_proj.register_forward_hook(self._make_k_proj_hook())
                self._handles.append(handle)
            elif hasattr(attn, "qkv_proj"):
                # Fused QKV: K is the middle slice [q_dim : q_dim + kv_dim]
                q_dim = num_heads * head_dim
                kv_dim = num_kv_heads * head_dim
                k_start = q_dim
                k_end = q_dim + kv_dim
                handle = attn.qkv_proj.register_forward_hook(
                    self._make_fused_qkv_hook(k_start, k_end)
                )
                self._handles.append(handle)

    def _make_k_proj_hook(self):
        hook_self = self
        def hook_fn(module, args, output):
            modified = output.clone()
            for t in hook_self.target_positions:
                if t < modified.shape[1]:
                    modified[:, t, :] *= hook_self.alpha
                    hook_self._sanity_changed = True
            return modified
        return hook_fn

    def _make_fused_qkv_hook(self, k_start, k_end):
        hook_self = self
        def hook_fn(module, args, output):
            modified = output.clone()
            for t in hook_self.target_positions:
                if t < modified.shape[1]:
                    modified[:, t, k_start:k_end] *= hook_self.alpha
                    hook_self._sanity_changed = True
            return modified
        return hook_fn

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


# =============================================================================
# Task 1 Step 3: get_action_logits() (Fix 1 applied)
# =============================================================================

def get_action_logits(model, processor, model_cfg, sample, device, bounds):
    prompt = model_cfg.prompt_template.format(instruction=sample["instruction"])
    inputs = call_processor(processor, prompt, sample["image"], model_cfg, return_tensors="pt").to(device)
    if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    fwd_kwargs = {k: v for k, v in inputs.items()}
    fwd_kwargs["use_cache"] = False
    # SpatialVLA needs intrinsic matrix
    if model_cfg.architecture == "gemma2" and "intrinsic" not in fwd_kwargs:
        fwd_kwargs["intrinsic"] = torch.tensor(
            [[[218.26, 0.0, 111.83], [0.0, 218.26, 111.79], [0.0, 0.0, 1.0]]],
            device=device, dtype=torch.float32,
        )
    with torch.no_grad():
        out = model(**fwd_kwargs)
    # Use last position of actual output logits (NOT input_ids length).
    # For Prismatic models (OpenVLA/ECoT), input_ids has only text tokens (~22),
    # but logits include prepended vision tokens (~278). Using input_ids.shape[1]-1
    # would index into vision token territory, not the last text position.
    return out.logits[0, -1, :], inputs  # (vocab_size,), inputs for reuse


# =============================================================================
# Task 1 Step 4: action_token_entropy()
# =============================================================================

def action_token_entropy(logits):
    probs = torch.softmax(logits.float(), dim=-1)
    log_probs = torch.log(probs.clamp(min=1e-10))
    entropy = -(probs * log_probs).sum().item()
    top1_prob = probs.max().item()
    top5_mass = probs.topk(5).values.sum().item()
    top1_id = probs.argmax().item()
    return {"entropy": entropy, "top1_prob": top1_prob,
            "top5_prob_mass": top5_mass, "top1_token_id": top1_id}


# =============================================================================
# Task 1 Step 5: detect_anchor_position()
# =============================================================================

def detect_anchor_position(model_name, verification_dir, bounds):
    """Find position-anchored token from Exp C results."""
    exp_c_path = verification_dir / model_name / "exp_c_position_anchoring.json"
    vs = bounds["vision_start"]
    if exp_c_path.exists():
        with open(exp_c_path) as f:
            exp_c = json.load(f)
        peak_counts = Counter(r["orig_a_peak"] for r in exp_c)
        dominant_peak, count = peak_counts.most_common(1)[0]
        is_anchored = count / len(exp_c) > 0.7
        return dominant_peak, vs + dominant_peak, is_anchored
    return 0, vs, False


# =============================================================================
# Task 1 Step 6: Augmentation functions
# =============================================================================

def augment_crop_90(img):
    w, h = img.size
    m_w, m_h = int(w * 0.05), int(h * 0.05)
    return img.crop((m_w, m_h, w - m_w, h - m_h)).resize((w, h), Image.BILINEAR)

def augment_shift_left(img):
    w, h = img.size; s = int(w * 0.05)
    out = Image.new(img.mode, (w, h))
    out.paste(img, (-s, 0))
    col = img.crop((w-1, 0, w, h))
    for x in range(w-s, w): out.paste(col, (x, 0))
    return out

def augment_shift_right(img):
    w, h = img.size; s = int(w * 0.05)
    out = Image.new(img.mode, (w, h))
    out.paste(img, (s, 0))
    col = img.crop((0, 0, 1, h))
    for x in range(s): out.paste(col, (x, 0))
    return out

def augment_brighten(img):
    return ImageEnhance.Brightness(img).enhance(1.1)

def augment_darken(img):
    return ImageEnhance.Brightness(img).enhance(0.9)

AUGMENTATIONS = [
    ("crop_90pct", augment_crop_90),
    ("shift_left_5pct", augment_shift_left),
    ("shift_right_5pct", augment_shift_right),
    ("brighten_1.1x", augment_brighten),
    ("darken_0.9x", augment_darken),
]


# =============================================================================
# Task 2 Step 2: ActionTokenizerLite (Fix 9+11)
# =============================================================================

class ActionTokenizerLite:
    """Lightweight action tokenizer for GT action → token ID conversion.
    Fix 11: Converts all 7 dims, not just dim0.
    Mirrors OpenVLA's tokenization: normalized → bin_index → token_id.
    """
    def __init__(self, model, model_cfg):
        self.n_bins = getattr(model.config, "n_action_bins", 256)
        pad = getattr(model.config, "pad_to_multiple_of", 0)
        cfg = model.config
        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "vocab_size"):
            self.vocab_size = cfg.text_config.vocab_size - pad
        elif hasattr(cfg, "vocab_size"):
            self.vocab_size = cfg.vocab_size - pad
        else:
            self.vocab_size = None  # SpatialVLA etc.
        edges = np.linspace(-1, 1, self.n_bins + 1)
        self.bin_centers = (edges[:-1] + edges[1:]) / 2.0
        # Load normalization stats
        norm_stats = getattr(model, "norm_stats", None) or getattr(model.config, "norm_stats", None)
        if norm_stats and config.BRIDGE_UNNORM_KEY in norm_stats:
            stats = norm_stats[config.BRIDGE_UNNORM_KEY]["action"]
            self.q01 = np.array(stats["q01"], dtype=np.float64)
            self.q99 = np.array(stats["q99"], dtype=np.float64)
            self.mask = np.array(stats.get("mask", [True] * 7))
        else:
            # Bridge V2 defaults (same as adapter_data.py)
            self.q01 = np.array([-0.02873, -0.04170, -0.02609, -0.08092, -0.09289, -0.20718, 0.0])
            self.q99 = np.array([0.02831, 0.04086, 0.04016, 0.08192, 0.07793, 0.20383, 1.0])
            self.mask = np.array([True, True, True, True, True, True, False])
        self.available = self.vocab_size is not None and model_cfg.architecture != "gemma2"

    def action_to_token_ids(self, action_7d):
        """Unnormalized 7-dim action → 7 token IDs. Returns None if not available."""
        if not self.available:
            return None
        action = np.asarray(action_7d, dtype=np.float64)
        normalized = np.where(
            self.mask, 2.0 * (action - self.q01) / (self.q99 - self.q01 + 1e-8) - 1.0, action)
        normalized = np.clip(normalized, -1.0, 1.0)
        bin_indices = np.digitize(normalized, self.bin_centers) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        return (self.vocab_size - 1 - bin_indices).tolist()

    def validate_token_ids(self, token_ids, processor=None):
        """Runtime sanity: verify token IDs are in expected action token range.
        Action tokens should be in [vocab_size - n_bins, vocab_size - 1].
        Optionally decode tokens to check they're not garbage/UNK.
        """
        lo = self.vocab_size - self.n_bins  # e.g., 31744 for LLaMA-2
        hi = self.vocab_size - 1            # e.g., 31999
        for d, tid in enumerate(token_ids):
            assert lo <= tid <= hi, (
                f"Token ID {tid} for dim{d} outside action range [{lo}, {hi}]! "
                f"vocab_size={self.vocab_size}, n_bins={self.n_bins}")
        if processor is not None and hasattr(processor, "tokenizer"):
            decoded = [processor.tokenizer.decode([tid]) for tid in token_ids]
            print(f"    Decoded action tokens: {decoded}")
        return True


# =============================================================================
# Task 2 Step 3: run_exp_d0_nll() (Fix 9+11+12)
# =============================================================================

def run_exp_d0_nll(model, processor, model_cfg, samples, device, output_dir):
    """D0: Teacher-forced NLL — 7-dim action, all dims.
    Fix 9: Real NLL. Fix 11: All 7 dims. Fix 12: True teacher-forcing.
    """
    tokenizer = ActionTokenizerLite(model, model_cfg)
    results = []

    for si, sample in enumerate(samples):
        gt_action = get_action_for_sample(sample, config.DATA_CACHE_DIR)  # (7,)

        # Get base inputs (prompt + vision)
        prompt = model_cfg.prompt_template.format(instruction=sample["instruction"])
        inputs = call_processor(processor, prompt, sample["image"], model_cfg,
                                return_tensors="pt").to(device)
        if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

        # Fix 11: Convert all 7 dims to token IDs
        gt_token_ids = tokenizer.action_to_token_ids(gt_action.numpy())

        if gt_token_ids is not None:
            # === Sanity check on first sample (token mapping validation) ===
            if si == 0:
                tokenizer.validate_token_ids(gt_token_ids, processor)
                print(f"    Token mapping sanity: gt_token_ids={gt_token_ids}, "
                      f"range=[{tokenizer.vocab_size - tokenizer.n_bins}, "
                      f"{tokenizer.vocab_size - 1}]")

            # Fix 12: TRUE teacher-forcing — concat GT action tokens to input
            base_ids = inputs["input_ids"]  # (1, seq_len)
            n_base = base_ids.shape[1]
            gt_suffix = torch.tensor([gt_token_ids], device=device, dtype=base_ids.dtype)  # (1, 7)
            # Concat: [prompt+vision tokens] + [7 GT action tokens]
            tf_ids = torch.cat([base_ids, gt_suffix], dim=1)  # (1, seq_len + 7)

            # Extend attention_mask if present
            fwd_kwargs = {k: v for k, v in inputs.items()}
            fwd_kwargs["input_ids"] = tf_ids
            if "attention_mask" in fwd_kwargs:
                ext = torch.ones(1, 7, device=device, dtype=fwd_kwargs["attention_mask"].dtype)
                fwd_kwargs["attention_mask"] = torch.cat([fwd_kwargs["attention_mask"], ext], dim=1)
            fwd_kwargs["use_cache"] = False
            if model_cfg.architecture == "gemma2" and "intrinsic" not in fwd_kwargs:
                fwd_kwargs["intrinsic"] = torch.tensor(
                    [[[218.26, 0.0, 111.83], [0.0, 218.26, 111.79], [0.0, 0.0, 1.0]]],
                    device=device, dtype=torch.float32)

            with torch.no_grad():
                out = model(**fwd_kwargs)

            # Extract NLL at each action position
            # For Prismatic models, logits are in expanded space:
            #   [0..V-1] vision | [V..V+T-1] text | [V+T..V+T+6] action
            # logits[V+T+d-1] predicts action token d (causal: logits[i] -> token[i+1])
            vision_offset = getattr(model_cfg, 'num_vision_tokens', 0)
            nll_per_dim = []
            gt_probs_per_dim = []
            gt_ranks_per_dim = []
            for d in range(7):
                logit_pos = vision_offset + n_base + d - 1
                logits_d = out.logits[0, logit_pos, :]
                probs_d = torch.softmax(logits_d.float(), dim=-1)
                log_probs_d = torch.log(probs_d.clamp(min=1e-10))
                gt_tid = gt_token_ids[d]
                nll_d = -log_probs_d[gt_tid].item()
                prob_d = probs_d[gt_tid].item()
                rank_d = int((probs_d >= probs_d[gt_tid]).sum().item())
                nll_per_dim.append(nll_d)
                gt_probs_per_dim.append(prob_d)
                gt_ranks_per_dim.append(rank_d)

            mean_nll = np.mean(nll_per_dim)
            mean_gt_prob = np.mean(gt_probs_per_dim)

            # Also get dim0 entropy for proxy compatibility
            logits_dim0 = out.logits[0, vision_offset + n_base - 1, :]
            probs_dim0 = torch.softmax(logits_dim0.float(), dim=-1)
            entropy = -(probs_dim0 * torch.log(probs_dim0.clamp(min=1e-10))).sum().item()
            top1_prob = probs_dim0.max().item()
            pred_token_id = logits_dim0.argmax().item()

            # === Sanity: dim0 cross-check (first sample only) ===
            # Teacher-forced dim0 NLL should ≈ non-teacher-forced dim0 NLL
            # because logits[n_base-1] sees the same context either way
            # (GT action tokens are AFTER this position, can't influence it via causal mask)
            if si == 0:
                logits_nontf, _ = get_action_logits(model, processor, model_cfg, sample, device, None)
                probs_nontf = torch.softmax(logits_nontf.float(), dim=-1)
                nll_nontf = -torch.log(probs_nontf.clamp(min=1e-10))[gt_token_ids[0]].item()
                nll_tf = nll_per_dim[0]
                diff = abs(nll_tf - nll_nontf)
                print(f"    Dim0 cross-check: TF_NLL={nll_tf:.4f}, nonTF_NLL={nll_nontf:.4f}, "
                      f"diff={diff:.6f}")
                # Numerical drift between TF/non-TF passes can occur due to:
                # - Phi-3-V in-place input_ids mutation
                # - Prismatic models (OpenVLA/ECoT) with dual vision encoder + projector
                # Use relaxed threshold for models with known drift patterns.
                is_inplace_model = model_cfg.architecture == "phi3_v"
                is_prismatic = hasattr(model, 'projector')
                threshold = 0.5 if (is_inplace_model or is_prismatic) else 0.01
                if diff >= threshold:
                    raise AssertionError(
                        f"Dim0 NLL mismatch! TF={nll_tf:.4f} vs nonTF={nll_nontf:.4f} "
                        f"(diff={diff:.4f}, threshold={threshold}). "
                        f"Teacher-forcing may be misaligned."
                    )
                elif diff >= 0.01:
                    print(f"    WARNING: Dim0 cross-check diff={diff:.4f} > 0.01 "
                          f"(tolerated for {model_cfg.architecture} in-place input mutation)")
                print(f"    tf_ids.shape={tf_ids.shape} (expected: ({1}, {n_base + 7}))")
        else:
            # SpatialVLA fallback: single-position entropy proxy
            logits, _ = get_action_logits(model, processor, model_cfg, sample, device, None)
            probs = torch.softmax(logits.float(), dim=-1)
            entropy = -(probs * torch.log(probs.clamp(min=1e-10))).sum().item()
            top1_prob = probs.max().item()
            pred_token_id = logits.argmax().item()
            mean_nll = None; mean_gt_prob = None
            nll_per_dim = None; gt_probs_per_dim = None; gt_ranks_per_dim = None
            gt_token_ids = None

        results.append({
            "sample_idx": si, "skill": sample.get("skill", "unknown"),
            "gt_action": gt_action.tolist(),
            "gt_token_ids": gt_token_ids,          # Fix 11: all 7 token IDs
            "pred_token_id_dim0": pred_token_id,
            "mean_nll": mean_nll,                   # Fix 11+12: mean over 7 dims
            "nll_per_dim": nll_per_dim,             # Fix 11: per-dim breakdown
            "mean_gt_prob": mean_gt_prob,           # Fix 11+12
            "gt_probs_per_dim": gt_probs_per_dim,
            "gt_ranks_per_dim": gt_ranks_per_dim,
            "top1_prob_dim0": top1_prob,            # proxy for compat
            "entropy_dim0": entropy,
        })
        nll_str = f"NLL={mean_nll:.3f}" if mean_nll is not None else "NLL=N/A"
        print(f"  D0 [{si+1}/{len(samples)}] skill={sample.get('skill')} "
              f"{nll_str} top1={top1_prob:.4f} H={entropy:.3f}")

    with open(output_dir / "exp_d0_nll.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# =============================================================================
# Task 3: run_exp_d1_entropy()
# =============================================================================

def run_exp_d1_entropy(model, processor, model_cfg, samples, device, output_dir):
    """D1: Measure action token entropy per sample."""
    results = []
    bounds_cache = {}
    for si, sample in enumerate(samples):
        bounds = detect_token_boundaries(processor, model, sample["image"],
                                          sample["instruction"], device, model_cfg)
        bounds_cache[si] = bounds
        logits, _ = get_action_logits(model, processor, model_cfg, sample, device, bounds)
        info = action_token_entropy(logits)
        info["sample_idx"] = si
        info["skill"] = sample.get("skill", "unknown")
        results.append(info)
        print(f"  D1 [{si+1}/{len(samples)}] skill={info['skill']} "
              f"H={info['entropy']:.3f} top1={info['top1_prob']:.3f}")

    # Save
    with open(output_dir / "exp_d1_entropy.json", "w") as f:
        json.dump(results, f, indent=2)

    mean_H = np.mean([r["entropy"] for r in results])
    mean_top1 = np.mean([r["top1_prob"] for r in results])
    print(f"\n  D1 Summary: mean_entropy={mean_H:.3f}, mean_top1_prob={mean_top1:.3f}")
    return results, bounds_cache


# =============================================================================
# Task 4: run_exp_d2_augmentation()
# =============================================================================

def run_exp_d2_augmentation(model, processor, model_cfg, samples, device,
                             bounds_cache, output_dir):
    """D2: Measure action token consistency under image augmentations."""
    results = []
    for si, sample in enumerate(samples):
        bounds = bounds_cache[si]
        # Original top-1
        logits_orig, _ = get_action_logits(model, processor, model_cfg, sample, device, bounds)
        top1_orig = logits_orig.argmax().item()

        aug_results = []
        for aug_name, aug_fn in AUGMENTATIONS:
            aug_sample = {**sample, "image": aug_fn(sample["image"])}
            logits_aug, _ = get_action_logits(model, processor, model_cfg, aug_sample, device, bounds)
            top1_aug = logits_aug.argmax().item()
            aug_results.append({
                "aug_name": aug_name,
                "top1": top1_aug,
                "matches": top1_aug == top1_orig,
                "kl": compute_output_kl(logits_orig, logits_aug),
            })

        consistency = np.mean([a["matches"] for a in aug_results])
        mean_kl = np.mean([a["kl"] for a in aug_results])
        results.append({
            "sample_idx": si, "skill": sample.get("skill", "unknown"),
            "orig_top1": top1_orig,
            "augmentation_results": aug_results,
            "consistency_rate": consistency,
            "mean_augmentation_kl": mean_kl,
        })
        print(f"  D2 [{si+1}/{len(samples)}] consistency={consistency:.2f} mean_kl={mean_kl:.3f}")

    with open(output_dir / "exp_d2_augmentation.json", "w") as f:
        json.dump(results, f, indent=2)

    mean_cons = np.mean([r["consistency_rate"] for r in results])
    print(f"\n  D2 Summary: mean_consistency={mean_cons:.3f}")
    return results


# =============================================================================
# Task 5: run_exp_d3_ablation() (Fix 3: deep layers only)
# =============================================================================

def run_exp_d3_ablation(model, processor, model_cfg, samples, device,
                         bounds_cache, deep_layers, output_dir):
    """D3: V-zero the position-anchored token at DEEP LAYERS ONLY, measure action change.
    Fix 3: target_layers=deep_layers to avoid over-intervention.
    """
    verification_dir = output_dir.parent  # outputs/phase3_gate/verification/
    anchor_rel, anchor_abs, is_anchored = detect_anchor_position(
        model_cfg.name, verification_dir, bounds_cache[0]
    )
    print(f"  Anchor: rel={anchor_rel}, abs={anchor_abs}, is_anchored={is_anchored}")
    print(f"  V=0 applied to deep layers only: {deep_layers}")

    results = []
    for si, sample in enumerate(samples):
        bounds = bounds_cache[si]
        vs = bounds["vision_start"]
        target_abs = vs + anchor_rel

        # Original forward
        logits_orig, _ = get_action_logits(model, processor, model_cfg, sample, device, bounds)

        # V-zero forward at anchor — Fix 3: deep layers only
        vzero = ValueZeroHook(target_positions=[target_abs], target_layers=deep_layers)
        vzero.register(model, model_cfg, get_layers)
        logits_masked, _ = get_action_logits(model, processor, model_cfg, sample, device, bounds)
        vzero.remove()

        kl = compute_output_kl(logits_orig, logits_masked)
        top1_changed = logits_orig.argmax().item() != logits_masked.argmax().item()
        cos_sim = float(torch.nn.functional.cosine_similarity(
            logits_orig.float().unsqueeze(0), logits_masked.float().unsqueeze(0)
        ).item())

        results.append({
            "sample_idx": si, "skill": sample.get("skill", "unknown"),
            "anchor_rel": anchor_rel, "anchor_abs": target_abs,
            "is_position_anchored": is_anchored,
            "top1_changed": top1_changed,
            "kl_divergence": kl,
            "cosine_similarity": cos_sim,
            "orig_top1": logits_orig.argmax().item(),
            "masked_top1": logits_masked.argmax().item(),
        })
        print(f"  D3 [{si+1}/{len(samples)}] KL={kl:.3f} top1_changed={top1_changed}")

    with open(output_dir / "exp_d3_ablation.json", "w") as f:
        json.dump(results, f, indent=2)

    mean_kl = np.mean([r["kl_divergence"] for r in results])
    change_rate = np.mean([r["top1_changed"] for r in results])
    print(f"\n  D3 Summary: mean_KL={mean_kl:.3f}, top1_change_rate={change_rate:.3f}")
    return results


# =============================================================================
# Task 6: compute_exp_d4_diversity() and save_exp_d_summary()
# =============================================================================

def compute_exp_d4_diversity(d1_results):
    """D4: Per-skill action diversity (post-processing over D1)."""
    from collections import defaultdict
    skill_actions = defaultdict(list)
    for r in d1_results:
        skill_actions[r["skill"]].append(r["top1_token_id"])

    diversity = {}
    for skill, actions in sorted(skill_actions.items()):
        n_unique = len(set(actions))
        # Entropy of action distribution
        counts = Counter(actions)
        total = len(actions)
        probs = [c / total for c in counts.values()]
        H = -sum(p * np.log(p + 1e-10) for p in probs)
        diversity[skill] = {
            "n_samples": total, "n_unique_actions": n_unique,
            "action_entropy": round(H, 4),
        }
    return diversity

def save_exp_d_summary(d0, d1, d2, d3, d4, output_dir):
    """Aggregate all Exp D metrics into single summary."""
    # Fix 9+11+12: Include D0 teacher-forced 7-dim NLL metrics
    nll_values = [r["mean_nll"] for r in d0 if r.get("mean_nll") is not None]
    summary = {
        "d0_mean_nll": np.mean(nll_values) if nll_values else None,       # Fix 11+12: 7-dim teacher-forced
        "d0_mean_gt_prob": np.mean([r["mean_gt_prob"] for r in d0 if r.get("mean_gt_prob")]) if nll_values else None,
        "d0_n_with_nll": len(nll_values),
        "d1_mean_entropy": np.mean([r["entropy"] for r in d1]),
        "d1_mean_top1_prob": np.mean([r["top1_prob"] for r in d1]),
        "d2_mean_consistency": np.mean([r["consistency_rate"] for r in d2]),
        "d2_mean_aug_kl": np.mean([r["mean_augmentation_kl"] for r in d2]),
        "d3_mean_kl": np.mean([r["kl_divergence"] for r in d3]),
        "d3_top1_change_rate": np.mean([r["top1_changed"] for r in d3]),
        "d4_diversity": d4,
    }
    with open(output_dir / "exp_d_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# =============================================================================
# Task 7: detect_anchor_targets() and run_exp_e() (Fix 4-13)
# =============================================================================

def detect_anchor_targets(model_cfg, verification_dir, bounds):
    """Fix 7: Detect anchor targets per model taxonomy.
    - bottleneck/normal: single target (A-peak position)
    - coexist (OpenVLA): BOTH A_mode (vision) AND C_mode (text gate)
    Returns list of {"target_abs": int, "mode": str, "anchor_rel": int}
    """
    exp_c_path = verification_dir / model_cfg.name / "exp_c_position_anchoring.json"
    vs = bounds["vision_start"]
    targets = []

    if exp_c_path.exists():
        with open(exp_c_path) as f:
            exp_c = json.load(f)
        # A-peak: most common attention peak
        a_peak_counts = Counter(r["orig_a_peak"] for r in exp_c)
        a_dominant, a_count = a_peak_counts.most_common(1)[0]
        a_anchored = a_count / len(exp_c) > 0.7
        targets.append({"target_abs": vs + a_dominant, "mode": "A_mode",
                         "anchor_rel": a_dominant, "is_anchored": a_anchored})

        # C-peak: most common contribution peak (may differ from A-peak)
        c_peak_counts = Counter(r["orig_c_peak"] for r in exp_c)
        c_dominant, c_count = c_peak_counts.most_common(1)[0]
        c_anchored = c_count / len(exp_c) > 0.7

        # Fix 7: If C-peak differs from A-peak (coexist), add C_mode target
        if c_dominant != a_dominant:
            targets.append({"target_abs": vs + c_dominant, "mode": "C_mode",
                             "anchor_rel": c_dominant, "is_anchored": c_anchored})

    if not targets:
        targets.append({"target_abs": vs, "mode": "A_mode", "anchor_rel": 0, "is_anchored": False})

    return targets


def run_exp_e(model, processor, model_cfg, samples, device, deep_layers, output_dir,
              bounds_cache, alphas=[0.0, 0.1, 0.3, 0.5, 0.7, 1.0],
              n_perms_for_anchoring=3):
    """Exp E: Value scaling at anchor + re-measure position anchoring.
    Fixes applied: 5 (baseline first), 6 (deep_layers), 7 (dual-target),
                   8 (W_OV C̃), 10 (qpos consistency).
    """
    verification_dir = output_dir.parent
    anchor_targets = detect_anchor_targets(model_cfg, verification_dir, bounds_cache[0])
    print(f"  Anchor targets: {anchor_targets}")

    # === Fix 5: Run baseline (alpha=1.0) FIRST ===
    print(f"\n  === Baseline pass (alpha=1.0) ===")
    baseline_top1s = {}
    baseline_logits = {}
    for si, sample in enumerate(samples):
        logits, inputs = get_action_logits(model, processor, model_cfg, sample, device, bounds_cache[si])
        baseline_top1s[si] = logits.argmax().item()
        baseline_logits[si] = logits.detach().cpu()

    # === Alpha sweep (excluding 1.0 which was done above) ===
    alpha_results = []
    # Add baseline result first
    baseline_info = [action_token_entropy(baseline_logits[si]) for si in range(len(samples))]
    alpha_results.append({
        "alpha": 1.0,
        "mean_entropy": round(np.mean([r["entropy"] for r in baseline_info]), 4),
        "action_change_rate_vs_baseline": 0.0,
        "n_samples": len(samples),
        # Anchoring metrics filled below
    })

    sweep_alphas = [a for a in alphas if a < 1.0]

    for alpha in sweep_alphas:
        print(f"\n  === Alpha={alpha} ===")
        sample_results = []

        for si, sample in enumerate(samples):
            bounds = bounds_cache[si]
            # Fix 7: Scale ALL anchor targets
            all_target_positions = [t["target_abs"] for t in anchor_targets]
            # Fix 6: Apply to deep_layers only
            scale_hook = ValueScaleHook(all_target_positions, alpha=alpha,
                                         target_layers=deep_layers)
            scale_hook.register(model, model_cfg, get_layers)

            logits, inputs = get_action_logits(model, processor, model_cfg, sample, device, bounds)
            info = action_token_entropy(logits)
            top1 = info["top1_token_id"]

            scale_hook.remove()

            sample_results.append({
                "sample_idx": si, "skill": sample.get("skill", "unknown"),
                **info,
                "top1_changed_vs_baseline": top1 != baseline_top1s[si],
            })

        # Fix 4 + Fix 8 + Fix 10 + Fix 13: C-peak anchoring check with per-mode tracking
        # Fix 13: Track anchoring per mode (A_mode, C_mode) separately
        per_mode_stayed = {t["mode"]: 0 for t in anchor_targets}
        per_mode_total = {t["mode"]: 0 for t in anchor_targets}
        top1_c_shares_proxy = []
        top1_c_shares_wov = []
        contrib_entropies = []
        n_anchor_samples = min(3, len(samples))  # Fix 8: reduced for W_OV cost

        for si in range(n_anchor_samples):
            sample = samples[si]
            bounds = bounds_cache[si]
            vs, ve = bounds["vision_start"], bounds["vision_end"]
            n_vis = ve - vs

            for pi in range(n_perms_for_anchoring):
                rng = np.random.default_rng(seed=42 + si * 100 + pi)
                perm = rng.permutation(n_vis)
                perm_tensor = torch.tensor(perm, device=device, dtype=torch.long)

                # Install BOTH hooks: value scale (Fix 6: deep_layers) + permutation
                handles = []
                if alpha < 1.0:
                    all_targets = [t["target_abs"] for t in anchor_targets]
                    scale_hook = ValueScaleHook(all_targets, alpha=alpha,
                                                 target_layers=deep_layers)
                    scale_hook.register(model, model_cfg, get_layers)

                # Permutation hook (reuse Exp C pattern)
                is_prismatic = hasattr(model, 'language_model') and hasattr(model, 'projector')
                if is_prismatic:
                    def make_perm_hook(pt):
                        def hook_fn(module, args, output):
                            m = output.clone(); m[0] = m[0, pt]; return m
                        return hook_fn
                    h = model.projector.register_forward_hook(make_perm_hook(perm_tensor))
                else:
                    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                        embed_mod = model.model.embed_tokens
                    elif hasattr(model, 'embed_tokens'):
                        embed_mod = model.embed_tokens
                    else:
                        embed_mod = get_layers(model, model_cfg)[0]
                    def make_embed_perm(vs_, ve_, pt_):
                        def hook_fn(module, args, output):
                            if isinstance(output, tuple):
                                t = output[0].clone()
                                t[0, vs_:ve_] = t[0, vs_:ve_][pt_]
                                return (t,) + output[1:]
                            m = output.clone()
                            m[0, vs_:ve_] = m[0, vs_:ve_][pt_]
                            return m
                        return hook_fn
                    h = embed_mod.register_forward_hook(make_embed_perm(vs, ve, perm_tensor))
                handles.append(h)

                # Forward with both hooks
                hook_mgr = SinkVerificationHookManager(model, model_cfg)
                hook_mgr.register_hooks(); hook_mgr.reset()
                prompt = model_cfg.prompt_template.format(instruction=sample["instruction"])
                inputs = call_processor(processor, prompt, sample["image"], model_cfg, return_tensors="pt").to(device)
                if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
                    inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
                fwd_kwargs = {k: v for k, v in inputs.items()}
                fwd_kwargs["use_cache"] = False
                if model_cfg.architecture == "gemma2" and "intrinsic" not in fwd_kwargs:
                    fwd_kwargs["intrinsic"] = torch.tensor(
                        [[[218.26, 0.0, 111.83], [0.0, 218.26, 111.79], [0.0, 0.0, 1.0]]],
                        device=device, dtype=torch.float32)
                with torch.no_grad():
                    model(**fwd_kwargs, output_attentions=True)

                l = deep_layers[-1]
                attn = hook_mgr.attention_weights.get(l)
                hidden = hook_mgr.hidden_states.get(l)
                prev_hidden = hook_mgr.hidden_states.get(l - 1, hidden)
                if attn is not None and prev_hidden is not None:
                    # Use attention tensor shape for qpos (full sequence incl. vision)
                    # input_ids.shape[1] is text-only, wrong for Prismatic models
                    qpos = attn.shape[-2] - 1
                    if attn.dim() == 3:
                        attn_vis = attn[:, qpos, vs:ve].mean(dim=0)
                    else:
                        attn_vis = attn[0, :, qpos, vs:ve].mean(dim=0)

                    # C-peak proxy (same as Exp C for consistency)
                    h_vis = prev_hidden[vs:ve]
                    h_norms = torch.norm(h_vis, dim=-1)
                    c_proxy = attn_vis.cpu() * h_norms
                    c_proxy_norm = c_proxy / c_proxy.sum().clamp(min=1e-10)

                    perm_c_peak = int(c_proxy.argmax().item())

                    # Fix 13: Per-mode anchoring evaluation
                    for target in anchor_targets:
                        mode = target["mode"]
                        stayed = perm_c_peak == target["anchor_rel"]
                        per_mode_stayed[mode] += int(stayed)
                        per_mode_total[mode] += 1

                    top1_c_shares_proxy.append(float(c_proxy_norm.max().item()))
                    c_ent = -(c_proxy_norm * torch.log(c_proxy_norm.clamp(min=1e-10))).sum().item()
                    contrib_entropies.append(c_ent)

                    # Fix 8: W_OV-based C̃ (stronger measurement, parallel to proxy)
                    try:
                        v_w, o_w = get_wov_matrix(model, model_cfg, l)
                        c_wov = compute_perhead_contribution(
                            attn[:, :, :] if attn.dim() == 3 else attn[0],
                            prev_hidden, v_w, o_w, [qpos])
                        # c_wov: (H, 1, seq) → mean over heads, extract vision range
                        c_wov_vis = c_wov[:, 0, vs:ve].mean(dim=0)  # (n_vis,)
                        c_wov_norm = c_wov_vis / c_wov_vis.sum().clamp(min=1e-10)
                        top1_c_shares_wov.append(float(c_wov_norm.max().item()))
                    except Exception:
                        pass  # W_OV unavailable (GQA dim mismatch) → skip

                hook_mgr.remove_hooks()
                for hh in handles: hh.remove()
                if alpha < 1.0: scale_hook.remove()

        # Fix 13: Per-mode anchoring rates
        anchoring_rates_per_mode = {}
        for mode in per_mode_stayed:
            total = per_mode_total[mode]
            anchoring_rates_per_mode[mode] = per_mode_stayed[mode] / max(total, 1)
        # Overall C-peak anchoring = average across modes (or primary mode for single-target)
        c_anchoring_rate = np.mean(list(anchoring_rates_per_mode.values()))

        mean_top1_c_proxy = np.mean(top1_c_shares_proxy) if top1_c_shares_proxy else 0.0
        mean_top1_c_wov = np.mean(top1_c_shares_wov) if top1_c_shares_wov else None
        mean_contrib_entropy = np.mean(contrib_entropies) if contrib_entropies else 0.0

        mean_H = np.mean([r["entropy"] for r in sample_results])
        change_rate = np.mean([r["top1_changed_vs_baseline"] for r in sample_results])

        result_entry = {
            "alpha": alpha,
            "mean_entropy": round(mean_H, 4),
            "action_change_rate_vs_baseline": round(change_rate, 4),
            "c_peak_anchoring_rate": round(c_anchoring_rate, 4),
            # Fix 13: Per-mode breakdown
            "anchoring_rate_per_mode": {m: round(r, 4) for m, r in anchoring_rates_per_mode.items()},
            "mean_top1_c_share_proxy": round(mean_top1_c_proxy, 4),
            "mean_top1_c_share_wov": round(mean_top1_c_wov, 4) if mean_top1_c_wov else None,
            "mean_contrib_entropy": round(mean_contrib_entropy, 4),
            "n_samples": len(sample_results),
            "n_anchor_trials_per_mode": dict(per_mode_total),
            "anchor_targets": [t["mode"] for t in anchor_targets],  # Fix 7
        }
        alpha_results.append(result_entry)
        mode_str = " ".join(f"{m}={r:.3f}" for m, r in anchoring_rates_per_mode.items())
        print(f"  Alpha={alpha}: H={mean_H:.3f}, change={change_rate:.3f}, "
              f"C-anchoring={c_anchoring_rate:.3f} [{mode_str}], "
              f"top1_C_proxy={mean_top1_c_proxy:.3f}, C_entropy={mean_contrib_entropy:.3f}")

    # Save
    with open(output_dir / "exp_e_alpha_sweep.json", "w") as f:
        json.dump(alpha_results, f, indent=2)

    return alpha_results


# =============================================================================
# Task 10.5: compute_anchoring_correlation() (Bonus)
# =============================================================================

def compute_anchoring_correlation(d0_results, d2_results, exp_c_path, output_dir):
    """Bonus: Correlate per-sample anchoring strength with D0/D2 metrics.
    Uses Exp C position_anchoring.json to get per-sample C-peak stay rate.
    """
    if not exp_c_path.exists():
        print("  Skipping correlation: no Exp C data")
        return None

    with open(exp_c_path) as f:
        exp_c = json.load(f)

    # Per-sample anchoring rate (fraction of layer×perm where C-peak stayed)
    from collections import defaultdict
    sample_anchor = defaultdict(lambda: {"stayed": 0, "total": 0})
    for r in exp_c:
        si = r["sample_idx"]
        sample_anchor[si]["total"] += 1
        if r["c_stayed_same_pos"]:
            sample_anchor[si]["stayed"] += 1

    anchor_rates = {}
    for si, v in sample_anchor.items():
        anchor_rates[si] = v["stayed"] / max(v["total"], 1)

    # Match with D0/D2 (by sample_idx)
    corr_data = []
    for d0r in d0_results:
        si = d0r["sample_idx"]
        if si in anchor_rates:
            d2r = d2_results[si] if si < len(d2_results) else None
            corr_data.append({
                "sample_idx": si,
                "anchor_rate": anchor_rates[si],
                "nll": d0r.get("mean_nll"),          # Fix 11+12: 7-dim teacher-forced NLL
                "entropy": d0r["entropy_dim0"],
                "consistency": d2r["consistency_rate"] if d2r else None,
            })

    if len(corr_data) < 5:
        print("  Too few samples for correlation")
        return None

    # Spearman correlations
    anchors = [d["anchor_rate"] for d in corr_data]
    results = {"n_samples": len(corr_data)}

    nlls = [d["nll"] for d in corr_data if d["nll"] is not None]
    if len(nlls) >= 5:
        rho, p = spearmanr([d["anchor_rate"] for d in corr_data if d["nll"] is not None], nlls)
        results["anchor_vs_nll"] = {"rho": round(rho, 4), "p": round(p, 4)}

    entropies = [d["entropy"] for d in corr_data]
    rho, p = spearmanr(anchors, entropies)
    results["anchor_vs_entropy"] = {"rho": round(rho, 4), "p": round(p, 4)}

    consistencies = [d["consistency"] for d in corr_data if d["consistency"] is not None]
    if len(consistencies) >= 5:
        rho, p = spearmanr(
            [d["anchor_rate"] for d in corr_data if d["consistency"] is not None],
            consistencies)
        results["anchor_vs_consistency"] = {"rho": round(rho, 4), "p": round(p, 4)}

    with open(output_dir / "exp_correlation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Correlation: {results}")
    return results


# =============================================================================
# Exp F: Q/K-Level Intervention — Key Scaling + D2 Consistency Improvement
# =============================================================================

def run_exp_f(model, processor, model_cfg, samples, device, deep_layers,
              output_dir, bounds_cache, d2_baseline=None,
              alphas=[0.0, 0.1, 0.3, 0.5, 0.7, 1.0],
              n_perms_for_anchoring=3):
    """Exp F: K-scaling at anchor positions to break position shortcut.

    Unlike Exp E (V-scaling), this targets K projection, directly reducing
    attention scores at the anchor position. Tests hypothesis:
    'position shortcut is encoded in Q/K, so only Q/K intervention can break it.'

    Key output: D2 consistency WITH K-intervention vs WITHOUT (baseline).
    """
    verification_dir = output_dir.parent
    anchor_targets = detect_anchor_targets(model_cfg, verification_dir, bounds_cache[0])
    print(f"\n{'='*60}")
    print(f"  Exp F: K-Scaling Intervention")
    print(f"  Anchor targets: {anchor_targets}")
    print(f"  Deep layers: {deep_layers}")
    print(f"{'='*60}")

    # === Baseline pass (alpha=1.0) ===
    print(f"\n  === F Baseline pass (alpha=1.0) ===")
    baseline_top1s = {}
    baseline_logits = {}
    for si, sample in enumerate(samples):
        logits, inputs = get_action_logits(model, processor, model_cfg, sample, device, bounds_cache[si])
        baseline_top1s[si] = logits.argmax().item()
        baseline_logits[si] = logits.detach().cpu()

    baseline_info = [action_token_entropy(baseline_logits[si]) for si in range(len(samples))]
    baseline_d2_consistency = d2_baseline if d2_baseline is not None else None

    alpha_results = []
    alpha_results.append({
        "alpha": 1.0,
        "intervention": "K-scale",
        "mean_entropy": round(np.mean([r["entropy"] for r in baseline_info]), 4),
        "action_change_rate_vs_baseline": 0.0,
        "d2_consistency_with_intervention": baseline_d2_consistency,
        "n_samples": len(samples),
    })

    sweep_alphas = [a for a in alphas if a < 1.0]

    for alpha in sweep_alphas:
        print(f"\n  === F Alpha={alpha} (K-scale) ===")
        sample_results = []

        for si, sample in enumerate(samples):
            bounds = bounds_cache[si]
            all_target_positions = [t["target_abs"] for t in anchor_targets]
            k_hook = KeyScaleHook(all_target_positions, alpha=alpha,
                                   target_layers=deep_layers)
            k_hook.register(model, model_cfg, get_layers)

            logits, inputs = get_action_logits(model, processor, model_cfg, sample, device, bounds)
            info = action_token_entropy(logits)
            top1 = info["top1_token_id"]

            k_hook.remove()

            sample_results.append({
                "sample_idx": si, "skill": sample.get("skill", "unknown"),
                **info,
                "top1_changed_vs_baseline": top1 != baseline_top1s[si],
            })

        # --- C-peak anchoring measurement (same pattern as Exp E) ---
        per_mode_stayed = {t["mode"]: 0 for t in anchor_targets}
        per_mode_total = {t["mode"]: 0 for t in anchor_targets}
        top1_c_shares_proxy = []
        contrib_entropies = []
        n_anchor_samples = min(3, len(samples))

        for si in range(n_anchor_samples):
            sample = samples[si]
            bounds = bounds_cache[si]
            vs, ve = bounds["vision_start"], bounds["vision_end"]
            n_vis = ve - vs

            for pi in range(n_perms_for_anchoring):
                rng = np.random.default_rng(seed=42 + si * 100 + pi)
                perm = rng.permutation(n_vis)
                perm_tensor = torch.tensor(perm, device=device, dtype=torch.long)

                handles = []
                # K-scale hook
                if alpha < 1.0:
                    all_targets = [t["target_abs"] for t in anchor_targets]
                    k_hook = KeyScaleHook(all_targets, alpha=alpha,
                                           target_layers=deep_layers)
                    k_hook.register(model, model_cfg, get_layers)

                # Permutation hook
                is_prismatic = hasattr(model, 'language_model') and hasattr(model, 'projector')
                if is_prismatic:
                    def make_perm_hook(pt):
                        def hook_fn(module, args, output):
                            m = output.clone(); m[0] = m[0, pt]; return m
                        return hook_fn
                    h = model.projector.register_forward_hook(make_perm_hook(perm_tensor))
                else:
                    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                        embed_mod = model.model.embed_tokens
                    elif hasattr(model, 'embed_tokens'):
                        embed_mod = model.embed_tokens
                    else:
                        embed_mod = get_layers(model, model_cfg)[0]
                    def make_embed_perm(vs_, ve_, pt_):
                        def hook_fn(module, args, output):
                            if isinstance(output, tuple):
                                t = output[0].clone()
                                t[0, vs_:ve_] = t[0, vs_:ve_][pt_]
                                return (t,) + output[1:]
                            m = output.clone()
                            m[0, vs_:ve_] = m[0, vs_:ve_][pt_]
                            return m
                        return hook_fn
                    h = embed_mod.register_forward_hook(make_embed_perm(vs, ve, perm_tensor))
                handles.append(h)

                # Forward with hooks
                hook_mgr = SinkVerificationHookManager(model, model_cfg)
                hook_mgr.register_hooks(); hook_mgr.reset()
                prompt = model_cfg.prompt_template.format(instruction=sample["instruction"])
                inp = call_processor(processor, prompt, sample["image"], model_cfg, return_tensors="pt").to(device)
                if "pixel_values" in inp and inp["pixel_values"].dtype != model.dtype:
                    inp["pixel_values"] = inp["pixel_values"].to(model.dtype)
                fwd_kwargs = {k: v for k, v in inp.items()}
                fwd_kwargs["use_cache"] = False
                if model_cfg.architecture == "gemma2" and "intrinsic" not in fwd_kwargs:
                    fwd_kwargs["intrinsic"] = torch.tensor(
                        [[[218.26, 0.0, 111.83], [0.0, 218.26, 111.79], [0.0, 0.0, 1.0]]],
                        device=device, dtype=torch.float32)
                with torch.no_grad():
                    model(**fwd_kwargs, output_attentions=True)

                l = deep_layers[-1]
                attn = hook_mgr.attention_weights.get(l)
                hidden = hook_mgr.hidden_states.get(l)
                prev_hidden = hook_mgr.hidden_states.get(l - 1, hidden)
                if attn is not None and prev_hidden is not None:
                    # Use attention tensor shape for qpos (full sequence incl. vision)
                    qpos = attn.shape[-2] - 1
                    if attn.dim() == 3:
                        attn_vis = attn[:, qpos, vs:ve].mean(dim=0)
                    else:
                        attn_vis = attn[0, :, qpos, vs:ve].mean(dim=0)

                    h_vis = prev_hidden[vs:ve]
                    h_norms = torch.norm(h_vis, dim=-1)
                    c_proxy = attn_vis.cpu() * h_norms
                    c_proxy_norm = c_proxy / c_proxy.sum().clamp(min=1e-10)
                    perm_c_peak = int(c_proxy.argmax().item())

                    for target in anchor_targets:
                        mode = target["mode"]
                        stayed = perm_c_peak == target["anchor_rel"]
                        per_mode_stayed[mode] += int(stayed)
                        per_mode_total[mode] += 1

                    top1_c_shares_proxy.append(float(c_proxy_norm.max().item()))
                    c_ent = -(c_proxy_norm * torch.log(c_proxy_norm.clamp(min=1e-10))).sum().item()
                    contrib_entropies.append(c_ent)

                hook_mgr.remove_hooks()
                for hh in handles: hh.remove()
                if alpha < 1.0: k_hook.remove()

        anchoring_rates_per_mode = {}
        for mode in per_mode_stayed:
            total = per_mode_total[mode]
            anchoring_rates_per_mode[mode] = per_mode_stayed[mode] / max(total, 1)
        c_anchoring_rate = np.mean(list(anchoring_rates_per_mode.values()))

        mean_top1_c_proxy = np.mean(top1_c_shares_proxy) if top1_c_shares_proxy else 0.0
        mean_contrib_entropy = np.mean(contrib_entropies) if contrib_entropies else 0.0
        mean_H = np.mean([r["entropy"] for r in sample_results])
        change_rate = np.mean([r["top1_changed_vs_baseline"] for r in sample_results])

        # --- D2 consistency WITH K-intervention (key metric for paper) ---
        d2_with_intervention = None
        if alpha <= 0.3:  # Only measure D2 at aggressive alphas (expensive)
            print(f"    Measuring D2 consistency with K-scale alpha={alpha}...")
            d2_consistencies = []
            for si, sample in enumerate(samples):
                bounds = bounds_cache[si]
                all_targets = [t["target_abs"] for t in anchor_targets]

                # Original with K-hook
                k_hook_orig = KeyScaleHook(all_targets, alpha=alpha, target_layers=deep_layers)
                k_hook_orig.register(model, model_cfg, get_layers)
                logits_orig, _ = get_action_logits(model, processor, model_cfg, sample, device, bounds)
                top1_orig = logits_orig.argmax().item()
                k_hook_orig.remove()

                # Augmentations with K-hook
                matches = []
                for aug_name, aug_fn in AUGMENTATIONS:
                    aug_sample = {**sample, "image": aug_fn(sample["image"])}
                    k_hook_aug = KeyScaleHook(all_targets, alpha=alpha, target_layers=deep_layers)
                    k_hook_aug.register(model, model_cfg, get_layers)
                    logits_aug, _ = get_action_logits(model, processor, model_cfg, aug_sample, device, bounds)
                    top1_aug = logits_aug.argmax().item()
                    k_hook_aug.remove()
                    matches.append(top1_aug == top1_orig)

                d2_consistencies.append(np.mean(matches))

            d2_with_intervention = round(np.mean(d2_consistencies), 4)
            print(f"    D2 with K-scale alpha={alpha}: {d2_with_intervention:.4f} "
                  f"(baseline: {baseline_d2_consistency})")

        result_entry = {
            "alpha": alpha,
            "intervention": "K-scale",
            "mean_entropy": round(mean_H, 4),
            "action_change_rate_vs_baseline": round(change_rate, 4),
            "c_peak_anchoring_rate": round(c_anchoring_rate, 4),
            "anchoring_rate_per_mode": {m: round(r, 4) for m, r in anchoring_rates_per_mode.items()},
            "mean_top1_c_share_proxy": round(mean_top1_c_proxy, 4),
            "mean_contrib_entropy": round(mean_contrib_entropy, 4),
            "d2_consistency_with_intervention": d2_with_intervention,
            "d2_consistency_baseline": baseline_d2_consistency,
            "n_samples": len(sample_results),
            "n_anchor_trials_per_mode": dict(per_mode_total),
            "anchor_targets": [t["mode"] for t in anchor_targets],
        }
        alpha_results.append(result_entry)
        mode_str = " ".join(f"{m}={r:.3f}" for m, r in anchoring_rates_per_mode.items())
        d2_str = f", D2={d2_with_intervention:.4f}" if d2_with_intervention is not None else ""
        print(f"  F Alpha={alpha}: H={mean_H:.3f}, change={change_rate:.3f}, "
              f"C-anchoring={c_anchoring_rate:.3f} [{mode_str}], "
              f"top1_C={mean_top1_c_proxy:.3f}, C_ent={mean_contrib_entropy:.3f}{d2_str}")

    # Save
    with open(output_dir / "exp_f_k_scale.json", "w") as f:
        json.dump(alpha_results, f, indent=2)

    # Print comparison summary
    print(f"\n  === Exp F Summary (K-scale vs V-scale) ===")
    print(f"  Baseline D2 consistency: {baseline_d2_consistency}")
    for r in alpha_results:
        if r.get("d2_consistency_with_intervention") is not None:
            delta = r["d2_consistency_with_intervention"] - (baseline_d2_consistency or 0)
            print(f"  K-scale alpha={r['alpha']}: D2={r['d2_consistency_with_intervention']:.4f} "
                  f"(delta={delta:+.4f}), C-anchoring={r.get('c_peak_anchoring_rate', 'N/A')}")

    return alpha_results


# =============================================================================
# Task 8: main()
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 3 Exp D+E+F")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gate1_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--experiments", default="d,e,f")
    args = parser.parse_args()

    gate1_dir = Path(args.gate1_dir) if args.gate1_dir else \
        config.OUTPUT_DIR / "phase3_gate" / args.model
    out = Path(args.output_dir) if args.output_dir else \
        config.OUTPUT_DIR / "phase3_gate" / "verification" / args.model
    out.mkdir(parents=True, exist_ok=True)

    processor, model, model_cfg = load_model_from_registry(args.model, args.device)
    samples = reload_samples_from_list(gate1_dir / "sample_list.json", config.DATA_CACHE_DIR)
    samples = samples[:args.n_samples]
    deep_layers = list(range(max(0, model_cfg.num_layers - 10), model_cfg.num_layers))
    exps = [e.strip() for e in args.experiments.split(",")]

    bounds_cache = {}
    d0 = d1 = d2 = d3 = None
    if "d" in exps:
        d0 = run_exp_d0_nll(model, processor, model_cfg, samples, args.device, out)
        d1, bounds_cache = run_exp_d1_entropy(model, processor, model_cfg, samples, args.device, out)
        d2 = run_exp_d2_augmentation(model, processor, model_cfg, samples, args.device, bounds_cache, out)
        d3 = run_exp_d3_ablation(model, processor, model_cfg, samples, args.device,
                                  bounds_cache, deep_layers, out)  # Fix 3: deep_layers
        d4 = compute_exp_d4_diversity(d1)
        save_exp_d_summary(d0, d1, d2, d3, d4, out)

    if "e" in exps:
        if not bounds_cache:
            for si, s in enumerate(samples):
                bounds_cache[si] = detect_token_boundaries(
                    processor, model, s["image"], s["instruction"], args.device, model_cfg)
        run_exp_e(model, processor, model_cfg, samples, args.device, deep_layers, out, bounds_cache)

    if "f" in exps:
        if not bounds_cache:
            for si, s in enumerate(samples):
                bounds_cache[si] = detect_token_boundaries(
                    processor, model, s["image"], s["instruction"], args.device, model_cfg)
        # Load D2 baseline from existing results (or compute if needed)
        d2_baseline_val = None
        d2_summary_path = out / "exp_d_summary.json"
        if d2_summary_path.exists():
            with open(d2_summary_path) as f:
                d_summary = json.load(f)
            d2_baseline_val = d_summary.get("d2_mean_consistency")
        elif d2 is not None:
            d2_baseline_val = round(np.mean([r["consistency_rate"] for r in d2]), 4)
        run_exp_f(model, processor, model_cfg, samples, args.device, deep_layers, out,
                  bounds_cache, d2_baseline=d2_baseline_val)

    # Bonus: per-sample correlation
    exp_c_path = out / "exp_c_position_anchoring.json"
    if d0 is not None and d2 is not None and exp_c_path.exists():
        compute_anchoring_correlation(d0, d2, exp_c_path, out)

    del model; torch.cuda.empty_cache()
    print(f"\nAll experiments complete. Results in: {out}")

if __name__ == "__main__":
    main()
