"""Evaluate trained Attention Adapter vs baseline OpenVLA.

Computes per-dimension MSE, overall MSE, and adapter behavior statistics
on the held-out test set. Supports both v1 and v2 adapters with auto-detection.

Usage:
    python adapter_eval.py --checkpoint outputs/adapter_results/checkpoints/best.pt
    python adapter_eval.py --checkpoint best.pt --num_episodes 100
    python adapter_eval.py --baseline_only --num_episodes 200 --output_dir outputs/base/eval
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import config
from adapter_data import ActionTokenizer, create_dataloaders
from adapter_model import AttentionAdapter, AttentionAdapterV2
from attention_v3 import (
    V3Context,
    install_v3_patch,
    set_v3_context,
    set_var_differentiable,
    uninstall_v3_patch,
)
from extract_attention import load_model_from_registry, get_layers, detokenize_actions
from model_registry import get_model


class AdapterEvaluator:
    """Compare adapter-modified vs baseline action predictions."""

    def __init__(
        self,
        checkpoint_path: str | None,
        model_name: str = "openvla-7b",
        device: str = "cuda",
        baseline_only: bool = False,
    ):
        self.device = device

        # ── Load model ──
        self.model_cfg = get_model(model_name)
        self.adapter_cfg = self.model_cfg.get_adapter_config()
        self.adapter_cfg["architecture"] = self.model_cfg.architecture
        print(f"Loading {model_name}...")
        self.processor, self.model, self.model_cfg = load_model_from_registry(model_name, device=device)
        self.model.eval()

        # ── Detect boundaries ──
        self.vision_end = self.model_cfg.num_vision_tokens
        # Probe text_end with dummy input
        from PIL import Image
        dummy = Image.new("RGB", (256, 256), (128, 128, 128))
        prompt = self.model_cfg.prompt_template.format(instruction="pick up the object")
        dummy_inputs = self.processor(prompt, dummy, return_tensors="pt").to(device)
        self.text_end = dummy_inputs["input_ids"].shape[-1]

        # ── Load adapter (skip in baseline-only mode) ──
        self.baseline_only = baseline_only
        self.adapter = None
        self.adapter_version = 0

        if not baseline_only and checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location=device)
            self.adapter_version = ckpt.get("config", {}).get("adapter_version", 1)

            if self.adapter_version == 2:
                self.adapter = AttentionAdapterV2(
                    hidden_dim=self.adapter_cfg["hidden_dim"],
                    num_target_layers=self.adapter_cfg["num_target_layers"],
                    num_heads=self.adapter_cfg["num_heads"],
                    vision_tokens=self.adapter_cfg["vision_tokens"],
                ).to(device)
            else:
                self.adapter = AttentionAdapter(
                    hidden_dim=self.adapter_cfg["hidden_dim"],
                    num_target_layers=self.adapter_cfg["num_target_layers"],
                    num_heads=self.adapter_cfg["num_heads"],
                ).to(device)

            self.adapter.load_state_dict(ckpt["adapter_state_dict"])
            self.adapter.eval()
            print(
                f"Adapter v{self.adapter_version} loaded from {checkpoint_path} "
                f"(step {ckpt.get('global_step', '?')})"
            )

        # ── Action tokenizer ──
        if self.model_cfg.action_type == "discrete":
            self.tokenizer = ActionTokenizer(self.model)
        else:
            raise NotImplementedError(
                f"Continuous action evaluation ({model_name}) is not yet implemented."
            )

    def _run_inference(
        self, image, instruction, adapter_enabled: bool, object_mask=None,
    ) -> dict:
        """Run autoregressive inference, optionally with adapter.

        Returns:
            dict with token_ids, normalized_action, unnormalized_action
        """
        prompt = self.model_cfg.prompt_template.format(instruction=instruction)
        inputs = self.processor(prompt, image, return_tensors="pt").to(self.device)
        if "pixel_values" in inputs and inputs["pixel_values"].dtype != self.model.dtype:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)

        ctx = get_v3_ctx_for_eval(
            self.model, self.adapter, self.device, self.vision_end,
            adapter_enabled=adapter_enabled,
            inputs=inputs,
            adapter_version=self.adapter_version,
            object_mask=object_mask,
            text_end=self.text_end,
            model_cfg=self.model_cfg,
            adapter_cfg=self.adapter_cfg,
        ) if adapter_enabled and self.adapter is not None else None

        if adapter_enabled and ctx is not None:
            set_v3_context(ctx)
            install_v3_patch(ctx, architecture=self.model_cfg.architecture, model=self.model)

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        pixel_values = inputs.get("pixel_values")
        generated_tokens = []

        with torch.no_grad():
            for token_idx in range(self.model_cfg.action_tokens):
                if ctx is not None:
                    ctx.current_token_idx = token_idx

                model_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "pixel_values": pixel_values,
                }
                outputs = self.model(**model_inputs, use_cache=False)
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_tokens.append(next_token.item())

                input_ids = torch.cat([input_ids, next_token], dim=-1)
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones(1, 1, device=self.device, dtype=attention_mask.dtype),
                    ], dim=-1)

        if adapter_enabled:
            uninstall_v3_patch()

        result = detokenize_actions(self.model, generated_tokens)
        return result

    def evaluate(
        self,
        num_episodes: int = 100,
        data_source: str = "tfrecord",
    ) -> dict:
        """Run full evaluation: baseline vs adapter on test split."""
        conditions = ["baseline"] if self.baseline_only else ["baseline", "adapter"]
        print(f"\nEvaluating on test set ({num_episodes} episodes, conditions: {conditions})...")

        _, _, test_loader = create_dataloaders(
            num_episodes=num_episodes,
            batch_size=1,
            source=data_source,
            use_object_masks=(self.adapter_version == 2),
        )

        results = {cond: {"per_step": []} for cond in conditions}

        for batch in tqdm_wrapper(test_loader, desc="Eval"):
            image = batch["images"][0]
            instruction = batch["instructions"][0]
            gt_action = batch["actions"][0]
            ep_id = batch["episode_ids"][0]
            step_id = batch["step_ids"][0]

            # Load object mask for this step if available
            obj_mask = None
            if "object_masks" in batch:
                obj_mask = batch["object_masks"][0]

            for condition in conditions:
                adapter_on = condition == "adapter"
                pred = self._run_inference(
                    image, instruction, adapter_enabled=adapter_on,
                    object_mask=obj_mask if adapter_on else None,
                )

                if pred["unnormalized_action"] is not None:
                    pred_arr = np.array(pred["unnormalized_action"])
                    mse_per_dim = (pred_arr - gt_action) ** 2

                    results[condition]["per_step"].append({
                        "episode_id": ep_id,
                        "step_id": step_id,
                        "mse_per_dim": mse_per_dim.tolist(),
                        "mse_overall": float(mse_per_dim.mean()),
                        "pred_action": pred_arr.tolist(),
                        "gt_action": gt_action.tolist(),
                    })

        # Aggregate
        for condition in conditions:
            steps = results[condition]["per_step"]
            if not steps:
                continue

            all_mse = np.array([s["mse_per_dim"] for s in steps])  # (N, 7)

            results[condition]["summary"] = {
                "n_steps": len(steps),
                "overall_mse": float(all_mse.mean()),
                "spatial_mse": float(all_mse[:, :3].mean()),
                "rotational_mse": float(all_mse[:, 3:6].mean()),
                "per_dim_mse": {
                    name: float(all_mse[:, i].mean())
                    for i, name in enumerate(config.ACTION_DIM_NAMES)
                },
            }

        # Compare (only if both conditions present)
        if "adapter" in results and "summary" in results.get("baseline", {}) and "summary" in results.get("adapter", {}):
            bl = results["baseline"]["summary"]
            ad = results["adapter"]["summary"]
            comparison = {
                "overall_change_pct": (ad["overall_mse"] - bl["overall_mse"]) / bl["overall_mse"] * 100,
                "spatial_change_pct": (ad["spatial_mse"] - bl["spatial_mse"]) / bl["spatial_mse"] * 100,
                "per_dim_change_pct": {
                    name: (ad["per_dim_mse"][name] - bl["per_dim_mse"][name]) / bl["per_dim_mse"][name] * 100
                    for name in config.ACTION_DIM_NAMES
                },
            }
            results["comparison"] = comparison

            print(f"\n{'=' * 50}")
            print("EVALUATION RESULTS")
            print(f"{'=' * 50}")
            print(f"  Baseline overall MSE: {bl['overall_mse']:.6f}")
            print(f"  Adapter  overall MSE: {ad['overall_mse']:.6f}")
            print(f"  Change: {comparison['overall_change_pct']:+.2f}%")
            print(f"\nPer-dimension changes:")
            for name in config.ACTION_DIM_NAMES:
                pct = comparison["per_dim_change_pct"][name]
                print(f"  {name:8s}: {pct:+.2f}%")

        # Save results
        eval_dir = config.ADAPTER_RESULTS_DIR / "eval" / self.model_cfg.name
        eval_dir.mkdir(parents=True, exist_ok=True)
        summary_results = {}
        for k, v in results.items():
            if isinstance(v, dict) and "per_step" in v:
                summary_results[k] = {sk: sv for sk, sv in v.items() if sk != "per_step"}
            else:
                summary_results[k] = v
        out_path = eval_dir / "eval_results.json"
        with open(out_path, "w") as f:
            json.dump(summary_results, f, indent=2)
        print(f"\nResults saved: {out_path}")

        return results


def get_v3_ctx_for_eval(
    model, adapter, device, vision_end, adapter_enabled, inputs,
    adapter_version=1, object_mask=None, text_end=0,
    model_cfg=None, adapter_cfg=None,
):
    """Create V3Context with adapter-predicted p values for evaluation.

    For v2 adapters, also computes redistribution_weights via cross-attention
    blended with proportional fallback using blend_alpha.
    """
    if not adapter_enabled:
        return None

    # Get hidden state from layer 27
    captured = {}

    def hook_fn(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["h_last"] = h[:, -1, :]
        captured["h_vision"] = h[:, :vision_end, :]

    layers = get_layers(model, model_cfg)
    hook_layer = layers[adapter_cfg["source_layer"]]

    hook = hook_layer.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**{k: v for k, v in inputs.items()}, use_cache=False)
    hook.remove()

    # Adapter prediction
    redistribution_weights = None

    with torch.no_grad():
        if adapter_version == 2:
            h_last = captured["h_last"].float()
            h_vision = captured["h_vision"].float()

            mask_tensor = None
            if object_mask is not None:
                mask_tensor = torch.from_numpy(object_mask).float().unsqueeze(0).to(device)

            p_matrix, redist_raw = adapter(h_last, h_vision, mask_tensor)

            # Blend learned redistribution with proportional (same as adapter_train.py)
            if redist_raw is not None:
                blend = adapter.blend_alpha
                V = vision_end
                prop_weights = torch.ones(1, V, device=device, dtype=torch.float32)
                for si in config.VAR_SINK_INDICES:
                    if si < V:
                        prop_weights[0, si] = 0.0
                prop_weights = prop_weights / prop_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                final_redist = blend * redist_raw + (1 - blend) * prop_weights
                redistribution_weights = final_redist.squeeze(0)  # (V,)
        else:
            p_matrix = adapter(captured["h_last"].float())  # (1, 4, 32)

    # Build full p tensor
    full_p = torch.zeros(
        model_cfg.num_layers, model_cfg.num_heads, device=device, dtype=p_matrix.dtype,
    )
    _target_idx = torch.tensor(
        adapter_cfg["target_layers"], device=device,
    ).unsqueeze(1).expand(-1, model_cfg.num_heads)
    full_p = full_p.scatter(0, _target_idx, p_matrix[0])

    ctx = V3Context(
        active=True,
        use_var=True,
        var_p=config.VAR_P,
        var_rho=config.VAR_RHO,
        var_sink_indices=list(config.VAR_SINK_INDICES),
        vision_end=vision_end,
        enhancement_layers=set(adapter_cfg["target_layers"]),
        per_head_var_strength=full_p,
        redistribution_weights=redistribution_weights,
        text_end=text_end,
        text_sink_enabled=config.VAR_TEXT_SINK_ENABLED,
        text_sink_p=config.VAR_TEXT_SINK_P,
        text_sink_threshold=config.VAR_TEXT_SINK_THRESHOLD,
    )
    return ctx


def tqdm_wrapper(iterable, **kwargs):
    """tqdm wrapper that works with DataLoader."""
    try:
        from tqdm import tqdm
        return tqdm(iterable, **kwargs)
    except ImportError:
        return iterable


def main():
    parser = argparse.ArgumentParser(description="Evaluate Attention Adapter")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model", type=str, default="openvla-7b",
                        choices=["openvla-7b", "ecot-7b", "spatialvla-4b", "tracevla-phi3v"],
                        help="VLA model from model_registry")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory for eval results")
    parser.add_argument("--baseline_only", action="store_true",
                        help="Evaluate baseline only (no adapter)")
    args = parser.parse_args()

    if args.baseline_only:
        evaluator = AdapterEvaluator(
            checkpoint_path=None, model_name=args.model, device=args.device, baseline_only=True,
        )
    else:
        if not args.checkpoint:
            parser.error("--checkpoint required unless --baseline_only")
        evaluator = AdapterEvaluator(args.checkpoint, model_name=args.model, device=args.device)

    results = evaluator.evaluate(num_episodes=args.num_episodes)

    if args.output_dir:
        eval_dir = Path(args.output_dir)
        eval_dir.mkdir(parents=True, exist_ok=True)
        summary_results = {}
        for k, v in results.items():
            if isinstance(v, dict) and "per_step" in v:
                summary_results[k] = {sk: sv for sk, sv in v.items() if sk != "per_step"}
            else:
                summary_results[k] = v
        out_path = eval_dir / "eval_results.json"
        with open(out_path, "w") as f:
            json.dump(summary_results, f, indent=2)
        print(f"Results saved: {out_path}")


if __name__ == "__main__":
    main()
