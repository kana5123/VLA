#!/usr/bin/env python3
"""LoRA Fine-Tune with Attention Entropy Regularization.

Training-time approach to fix Bottleneck models (e.g., ECoT) by distributing
attention more evenly across vision tokens, preventing concentration on a
single anchor/sink token.

Unlike inference-time hooks (VAR/K-scale) which both hurt ECoT because the
bottleneck carries critical routing information, this approach regularizes
attention entropy during training so the model learns not to rely on a
single token.

Key idea:
  total_loss = CE_loss + lambda_ent * entropy_penalty
  entropy_penalty = relu(h_target - H(action -> vision_tokens))

Usage:
  python train_entropy_reg.py --model ecot-7b --device cuda:0
  python train_entropy_reg.py --model openvla-7b --device cuda:0 --lambda_ent 0.1
  python train_entropy_reg.py --model ecot-7b --device cuda:0 --max_steps 200 --eval_every 50
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

import config
from extract_attention import (
    load_model_from_registry,
    get_layers,
    call_processor,
    detect_token_boundaries,
)
from data_sampler import reload_samples_from_list, get_action_for_sample
from run_phase3_exp_de import ActionTokenizerLite


# =============================================================================
# AttentionCaptureHook: Register hooks on self_attn to capture attention weights
# =============================================================================

class AttentionCaptureHook:
    """Register forward hooks on self_attn modules of target layers to
    capture attention weights WITH gradient flow.

    Unlike the inference-time AttentionHookManager in extract_attention.py,
    this class does NOT detach attention weights, allowing gradients to flow
    back through the entropy regularization loss.

    After a forward pass with output_attentions=True, each target layer's
    self_attn module will have a ``_last_attn_weights`` attribute containing
    the attention weights tensor of shape (B, H, S, S).
    """

    def __init__(self, model, model_cfg, target_layers):
        """
        Args:
            model: The loaded (possibly LoRA-wrapped) VLA model.
            model_cfg: VLAModelConfig from model_registry.
            target_layers: List of layer indices to hook (e.g., deep layers).
        """
        self.handles = []
        self.target_layers = target_layers

        layers = get_layers(model, model_cfg)
        for layer_idx in target_layers:
            if layer_idx >= len(layers):
                continue
            layer = layers[layer_idx]
            attn_module = layer.self_attn
            handle = attn_module.register_forward_hook(self._make_hook())
            self.handles.append(handle)

    @staticmethod
    def _make_hook():
        """Create a hook function that captures attention weights without detaching."""
        def hook_fn(module, args, output):
            # LlamaAttention / GemmaAttention returns:
            #   (attn_output, attn_weights, past_key_value)
            # attn_weights shape: (B, H, S, S)
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                module._last_attn_weights = output[1]  # Keep grad flow
        return hook_fn

    def remove(self):
        """Remove all registered hooks."""
        for h in self.handles:
            h.remove()
        self.handles = []


# =============================================================================
# compute_attention_entropy_loss: Entropy penalty on action->vision attention
# =============================================================================

def compute_attention_entropy_loss(model, model_cfg, deep_layers,
                                   vision_start, vision_end,
                                   h_target, action_pos):
    """Compute attention entropy regularization loss.

    For each deep layer, extract the attention weights from the action token
    position to vision tokens, compute per-head entropy, and penalize heads
    whose entropy falls below the target threshold.

    Args:
        model: The model (with _last_attn_weights stored by hooks).
        model_cfg: VLAModelConfig.
        deep_layers: List of layer indices where hooks are registered.
        vision_start: Start index of vision tokens in the sequence.
        vision_end: End index of vision tokens in the sequence.
        h_target: Target entropy value (in nats).
        action_pos: Sequence position(s) of the action token query.
            Can be an int or a list of ints.

    Returns:
        Scalar tensor with the entropy penalty (differentiable).
    """
    layers = get_layers(model, model_cfg)
    penalties = []

    # Normalize action_pos to a list
    if isinstance(action_pos, int):
        action_positions = [action_pos]
    else:
        action_positions = list(action_pos)

    for layer_idx in deep_layers:
        if layer_idx >= len(layers):
            continue
        attn_module = layers[layer_idx].self_attn
        attn_weights = getattr(attn_module, "_last_attn_weights", None)
        if attn_weights is None:
            continue

        # attn_weights shape: (B, H, S, S)
        # Extract attention from action positions to vision tokens
        for apos in action_positions:
            # attn[:, :, apos, vision_start:vision_end] -> (B, H, n_vis)
            a = attn_weights[:, :, apos, vision_start:vision_end]

            # Renormalize over vision tokens (they may not sum to 1 due to
            # other tokens in the full softmax)
            a = a / (a.sum(dim=-1, keepdim=True) + 1e-10)

            # Per-head entropy: H = -sum(a * log(a))
            log_a = torch.log(a + 1e-10)
            entropy = -(a * log_a).sum(dim=-1)  # (B, H)

            # Penalty: relu(h_target - H) -- penalize when entropy is too low
            penalty = F.relu(h_target - entropy).mean()  # scalar
            penalties.append(penalty)

    if not penalties:
        return torch.tensor(0.0, device=next(model.parameters()).device,
                            requires_grad=True)

    return torch.stack(penalties).mean()


# =============================================================================
# forward_with_entropy_reg: Single-sample forward with CE + entropy reg
# =============================================================================

def forward_with_entropy_reg(model, model_cfg, processor, sample, device,
                             bounds, gt_token_ids, deep_layers,
                             h_target, lambda_ent):
    """Single-sample forward pass with cross-entropy + entropy regularization.

    Steps:
      1. Build teacher-forced input: concat GT action tokens to input_ids
      2. Forward with output_attentions=True
      3. CE loss: average NLL across 7 action dims
      4. Entropy loss: compute_attention_entropy_loss()
      5. Return total_loss = ce + lambda_ent * ent

    Args:
        model: LoRA-wrapped model.
        model_cfg: VLAModelConfig.
        processor: Model processor.
        sample: Sample dict with 'image' and 'instruction'.
        device: Torch device.
        bounds: Token boundary dict from detect_token_boundaries().
        gt_token_ids: List of 7 ground-truth action token IDs.
        deep_layers: Layer indices for entropy regularization.
        h_target: Target entropy value.
        lambda_ent: Entropy regularization coefficient.

    Returns:
        Tuple of (total_loss, ce_loss_value, ent_loss_value, dict of details).
    """
    # Build prompt and process inputs
    prompt = model_cfg.prompt_template.format(instruction=sample["instruction"])
    inputs = call_processor(processor, prompt, sample["image"],
                            model_cfg, return_tensors="pt").to(device)
    if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

    # Build teacher-forced input: [prompt+vision] + [7 GT action tokens]
    base_ids = inputs["input_ids"]  # (1, seq_len)
    n_base = base_ids.shape[1]
    gt_suffix = torch.tensor([gt_token_ids], device=device, dtype=base_ids.dtype)  # (1, 7)
    tf_ids = torch.cat([base_ids, gt_suffix], dim=1)  # (1, seq_len + 7)

    # Extend attention_mask if present
    fwd_kwargs = {k: v for k, v in inputs.items()}
    fwd_kwargs["input_ids"] = tf_ids
    if "attention_mask" in fwd_kwargs:
        ext = torch.ones(1, 7, device=device, dtype=fwd_kwargs["attention_mask"].dtype)
        fwd_kwargs["attention_mask"] = torch.cat([fwd_kwargs["attention_mask"], ext], dim=1)
    fwd_kwargs["use_cache"] = False
    fwd_kwargs["output_attentions"] = True

    # SpatialVLA needs intrinsic matrix
    if model_cfg.architecture == "gemma2" and "intrinsic" not in fwd_kwargs:
        fwd_kwargs["intrinsic"] = torch.tensor(
            [[[218.26, 0.0, 111.83],
              [0.0, 218.26, 111.79],
              [0.0, 0.0, 1.0]]],
            device=device, dtype=torch.float32,
        )

    # Forward pass (with gradients)
    out = model(**fwd_kwargs)

    # CE loss: average NLL across 7 action dims
    # For Prismatic models, logits are in expanded space:
    #   [0..V-1] vision | [V..V+T-1] text | [V+T..V+T+6] action
    # logits[V+T+d-1] predicts action token d (causal: logits[i] -> token[i+1])
    vision_offset = bounds.get("num_vision_tokens", 0)
    ce_losses = []
    for d in range(7):
        logit_pos = vision_offset + n_base + d - 1
        logits_d = out.logits[0, logit_pos, :]  # (vocab_size,)
        target_d = torch.tensor([gt_token_ids[d]], device=device, dtype=torch.long)
        ce_d = F.cross_entropy(logits_d.unsqueeze(0), target_d)
        ce_losses.append(ce_d)
    ce_loss = torch.stack(ce_losses).mean()

    # Entropy regularization loss
    vision_start = bounds["vision_start"]
    vision_end = bounds["vision_end"]
    # Action positions in the teacher-forced expanded sequence
    action_positions = [vision_offset + n_base + d - 1 for d in range(7)]
    ent_loss = compute_attention_entropy_loss(
        model, model_cfg, deep_layers,
        vision_start, vision_end,
        h_target, action_positions,
    )

    # Total loss
    total_loss = ce_loss + lambda_ent * ent_loss

    details = {
        "ce_loss": ce_loss.item(),
        "ent_loss": ent_loss.item(),
        "total_loss": total_loss.item(),
        "lambda_ent": lambda_ent,
    }

    return total_loss, ce_loss.item(), ent_loss.item(), details


# =============================================================================
# train(): CLI entry point
# =============================================================================

def train():
    """Main training function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="LoRA Fine-Tune with Attention Entropy Regularization"
    )
    parser.add_argument("--model", required=True,
                        help="Model name from registry (e.g., ecot-7b, openvla-7b)")
    parser.add_argument("--device", default="cuda:0",
                        help="Torch device (default: cuda:0)")
    parser.add_argument("--lambda_ent", type=float, default=0.05,
                        help="Entropy regularization coefficient (default: 0.05)")
    parser.add_argument("--h_target_frac", type=float, default=0.3,
                        help="Target entropy as fraction of log(n_vision_tokens) (default: 0.3)")
    parser.add_argument("--max_steps", type=int, default=100,
                        help="Maximum training steps (default: 100)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--eval_every", type=int, default=25,
                        help="Evaluate every N steps (default: 25)")
    parser.add_argument("--n_train", type=int, default=50,
                        help="Number of training samples (default: 50)")
    parser.add_argument("--n_eval", type=int, default=20,
                        help="Number of evaluation samples (default: 20)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory override")
    parser.add_argument("--gate1_dir", default=None,
                        help="Path to gate1 directory with sample_list.json")
    args = parser.parse_args()

    model_name = args.model
    device = args.device

    # --- Paths ---
    gate1_dir = Path(args.gate1_dir) if args.gate1_dir else \
        config.OUTPUT_DIR / "phase3_gate" / model_name
    output_dir = Path(args.output_dir) if args.output_dir else \
        config.OUTPUT_DIR / "phase3_gate" / "entropy_reg" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Entropy Regularization Training")
    print(f"  Model:      {model_name}")
    print(f"  Device:     {device}")
    print(f"  lambda_ent: {args.lambda_ent}")
    print(f"  h_target_frac: {args.h_target_frac}")
    print(f"  max_steps:  {args.max_steps}")
    print(f"  lr:         {args.lr}")
    print(f"  eval_every: {args.eval_every}")
    print(f"  n_train:    {args.n_train}")
    print(f"  n_eval:     {args.n_eval}")
    print(f"  output_dir: {output_dir}")
    print(f"{'='*60}\n")

    # ── Step 1: Load model ──────────────────────────────────────────────
    print("Loading model...")
    processor, model, model_cfg = load_model_from_registry(model_name, device)

    # Determine deep layers (last 10 or all if <10)
    deep_layers = list(range(
        max(0, model_cfg.num_layers - 10), model_cfg.num_layers
    ))
    print(f"  Deep layers for entropy reg: {deep_layers}")

    # Determine target layers for LoRA (same as deep layers)
    target_layers = deep_layers

    # ── Step 2: Apply LoRA via PEFT ─────────────────────────────────────
    print("Applying LoRA...")
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGET_MODULES,
        layers_to_transform=target_layers,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Save model_cfg reference before wrapping (get_peft_model changes model object)
    saved_model_cfg = model_cfg

    model = get_peft_model(model, lora_config)
    model.train()

    # Restore model_cfg reference
    model_cfg = saved_model_cfg

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  LoRA trainable params: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    # Enable output_attentions in the underlying model config
    if hasattr(model, "config"):
        model.config.output_attentions = True
    base_model = model.base_model.model if hasattr(model, "base_model") else model
    if hasattr(base_model, "language_model"):
        base_model.language_model.config.output_attentions = True
    elif hasattr(base_model, "config"):
        base_model.config.output_attentions = True

    # ── Step 3: Register AttentionCaptureHook ───────────────────────────
    print("Registering attention capture hooks on deep layers...")
    attn_hook = AttentionCaptureHook(model, model_cfg, deep_layers)
    print(f"  Registered {len(attn_hook.handles)} hooks")

    # ── Step 4: Compute H_target ────────────────────────────────────────
    num_vision_tokens = model_cfg.num_vision_tokens
    h_target = args.h_target_frac * math.log(num_vision_tokens)
    print(f"  num_vision_tokens: {num_vision_tokens}")
    print(f"  H_target = {args.h_target_frac} * log({num_vision_tokens}) = {h_target:.4f}")

    # ── Step 5: Load samples and tokenize GT actions ────────────────────
    sample_list_path = gate1_dir / "sample_list.json"
    print(f"Loading samples from {sample_list_path}...")
    all_samples = reload_samples_from_list(sample_list_path, config.DATA_CACHE_DIR)

    # Split into train and eval
    n_total = len(all_samples)
    n_train = min(args.n_train, n_total)
    n_eval = min(args.n_eval, n_total - n_train)
    train_samples = all_samples[:n_train]
    eval_samples = all_samples[n_train:n_train + n_eval] if n_eval > 0 else []
    print(f"  Train: {len(train_samples)}, Eval: {len(eval_samples)}")

    # Initialize ActionTokenizerLite
    # Need to access base model for config attributes
    base_for_tokenizer = model.base_model.model if hasattr(model, "base_model") else model
    tokenizer = ActionTokenizerLite(base_for_tokenizer, model_cfg)

    if not tokenizer.available:
        print(f"  WARNING: ActionTokenizerLite not available for {model_cfg.architecture}.")
        print(f"  Skipping NLL-based training (SpatialVLA uses continuous actions).")
        print(f"  Only entropy regularization will be applied (no CE loss).")

    # Pre-tokenize GT actions for training samples
    train_gt_token_ids = []
    skipped = 0
    for si, sample in enumerate(train_samples):
        gt_action = get_action_for_sample(sample, config.DATA_CACHE_DIR)
        if tokenizer.available:
            tids = tokenizer.action_to_token_ids(gt_action.numpy())
            if tids is None:
                skipped += 1
                train_gt_token_ids.append(None)
            else:
                train_gt_token_ids.append(tids)
        else:
            train_gt_token_ids.append(None)
    print(f"  Tokenized GT actions: {n_train - skipped}/{n_train} successful")

    # Detect token boundaries using first training sample
    print("Detecting token boundaries...")
    bounds = detect_token_boundaries(
        processor, model, train_samples[0]["image"],
        train_samples[0]["instruction"], device, model_cfg,
    )
    print(f"  Vision tokens: [{bounds['vision_start']}:{bounds['vision_end']}]")

    # ── Step 6: Training loop ───────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    history = {
        "model": model_name,
        "lambda_ent": args.lambda_ent,
        "h_target_frac": args.h_target_frac,
        "h_target": h_target,
        "lr": args.lr,
        "lora_r": config.LORA_R,
        "lora_alpha": config.LORA_ALPHA,
        "steps": [],
        "eval_results": [],
    }

    rng = np.random.default_rng(seed=42)

    print(f"\n{'='*60}")
    print(f"  Starting training loop ({args.max_steps} steps)")
    print(f"{'='*60}\n")

    t_start = time.time()

    for step in range(1, args.max_steps + 1):
        # Pick a random training sample
        while True:
            si = int(rng.integers(0, n_train))
            if train_gt_token_ids[si] is not None:
                break
            # If all are None (SpatialVLA), allow None and skip CE
            if not tokenizer.available:
                break

        sample = train_samples[si]
        gt_tids = train_gt_token_ids[si]

        optimizer.zero_grad()

        if gt_tids is not None:
            # Full forward with CE + entropy regularization
            total_loss, ce_val, ent_val, details = forward_with_entropy_reg(
                model, model_cfg, processor, sample, device,
                bounds, gt_tids, deep_layers, h_target, args.lambda_ent,
            )
        else:
            # Entropy-only mode (SpatialVLA or failed tokenization)
            # Forward without CE loss, only entropy regularization
            prompt = model_cfg.prompt_template.format(instruction=sample["instruction"])
            inputs = call_processor(processor, prompt, sample["image"],
                                    model_cfg, return_tensors="pt").to(device)
            if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
                inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

            fwd_kwargs = {k: v for k, v in inputs.items()}
            fwd_kwargs["use_cache"] = False
            fwd_kwargs["output_attentions"] = True
            if model_cfg.architecture == "gemma2" and "intrinsic" not in fwd_kwargs:
                fwd_kwargs["intrinsic"] = torch.tensor(
                    [[[218.26, 0.0, 111.83],
                      [0.0, 218.26, 111.79],
                      [0.0, 0.0, 1.0]]],
                    device=device, dtype=torch.float32,
                )

            out = model(**fwd_kwargs)

            # Use last position as action position proxy
            n_seq = inputs["input_ids"].shape[1]
            action_pos = n_seq - 1
            ent_loss = compute_attention_entropy_loss(
                model, model_cfg, deep_layers,
                bounds["vision_start"], bounds["vision_end"],
                h_target, action_pos,
            )
            total_loss = args.lambda_ent * ent_loss
            ce_val = 0.0
            ent_val = ent_loss.item()

        # Backward + clip + step
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Log
        elapsed = time.time() - t_start
        step_info = {
            "step": step,
            "total_loss": total_loss.item(),
            "ce_loss": ce_val,
            "ent_loss": ent_val,
            "sample_idx": si,
            "elapsed_sec": round(elapsed, 1),
        }
        history["steps"].append(step_info)

        if step % 5 == 0 or step == 1:
            print(f"  Step {step:4d}/{args.max_steps} | "
                  f"loss={total_loss.item():.4f} "
                  f"(CE={ce_val:.4f} + {args.lambda_ent}*Ent={ent_val:.4f}) | "
                  f"{elapsed:.1f}s")

        # ── Step 7: Periodic evaluation ─────────────────────────────────
        if step % args.eval_every == 0 or step == args.max_steps:
            print(f"\n  --- Eval at step {step} ---")
            model.eval()

            try:
                from adaptive_routing import evaluate_d2_with_intervention
                from run_phase3_exp_de import detect_anchor_targets

                # Compute bounds for eval samples
                eval_bounds_cache = {}
                eval_subset = eval_samples[:min(10, len(eval_samples))]
                for ei, es in enumerate(eval_subset):
                    eval_bounds_cache[ei] = detect_token_boundaries(
                        processor, model, es["image"],
                        es["instruction"], device, model_cfg,
                    )

                # Detect anchor targets
                verification_dir = config.OUTPUT_DIR / "phase3_gate" / "verification"
                anchor_targets = detect_anchor_targets(
                    model_cfg, verification_dir, eval_bounds_cache.get(0, bounds),
                )

                # Run D2 evaluation with no intervention (baseline = trained model)
                d2_result = evaluate_d2_with_intervention(
                    model, processor, model_cfg, eval_subset, device,
                    eval_bounds_cache, deep_layers, anchor_targets,
                    intervention_config=None,
                )
                eval_entry = {
                    "step": step,
                    "d2_mean": d2_result["d2_mean"],
                    "entropy_mean": d2_result["entropy_mean"],
                    "n_eval_samples": len(eval_subset),
                }
                history["eval_results"].append(eval_entry)
                print(f"  D2={d2_result['d2_mean']:.4f}, "
                      f"H={d2_result['entropy_mean']:.4f}")
            except Exception as e:
                print(f"  Eval failed: {e}")
                history["eval_results"].append({
                    "step": step,
                    "error": str(e),
                })

            model.train()
            print()

    # ── Step 8: Save LoRA weights and training history ──────────────────
    t_total = time.time() - t_start

    # Save LoRA adapter weights
    lora_save_path = output_dir / "lora_adapter"
    print(f"Saving LoRA weights to {lora_save_path}...")
    model.save_pretrained(str(lora_save_path))
    print(f"  LoRA adapter saved.")

    # Save training history
    history["total_time_sec"] = round(t_total, 1)
    history["final_step"] = args.max_steps
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Training history saved to {history_path}")

    # Cleanup
    attn_hook.remove()

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Total time: {t_total:.1f}s ({t_total / 60:.1f}min)")
    print(f"  Output dir: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
