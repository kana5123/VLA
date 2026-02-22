"""Cross-model attention extraction for VLA models.

Extracts per-head attention weights from any model in the registry.
Produces both head-averaged top-5 JSON and per-head breakdown JSON,
plus heatmap visualizations.

Usage:
    python cross_model_extract.py --model openvla-7b --dataset bridge_v2
    python cross_model_extract.py --model cogact-base --dataset bridge_v2 --device cuda:0
    python cross_model_extract.py --all --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import gc
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

import config
from model_registry import get_model, list_models, VLAModelConfig
from dataset_registry import load_sample, DatasetSample
from extract_attention import (
    compute_perhead_stats,
    analyze_top_k,
    visualize_perhead_sink,
)


CROSS_MODEL_DIR = config.OUTPUT_DIR / "cross_model_analysis"


def load_vla_model(model_cfg: VLAModelConfig, device: str = "cuda"):
    """Load any VLA model from registry with eager attention."""
    from transformers import AutoModelForVision2Seq, AutoProcessor

    print(f"Loading {model_cfg.name} ({model_cfg.hf_id})...")
    processor = AutoProcessor.from_pretrained(
        model_cfg.hf_id, trust_remote_code=model_cfg.trust_remote_code,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        model_cfg.hf_id,
        torch_dtype=getattr(torch, model_cfg.torch_dtype),
        trust_remote_code=model_cfg.trust_remote_code,
        attn_implementation=model_cfg.attn_impl,
    ).to(device).eval()
    print(f"  Loaded: {model_cfg.num_layers}L x {model_cfg.num_heads}H, "
          f"hidden={model_cfg.hidden_dim}")
    return processor, model


def get_layers(model, model_cfg: VLAModelConfig):
    """Navigate to the transformer layers using the model config path."""
    obj = model
    for attr in model_cfg.layers_path.split("."):
        obj = getattr(obj, attr)
    return obj


def detect_boundaries(processor, model, model_cfg, sample: DatasetSample, device):
    """Detect vision/text token boundaries for a specific model."""
    prompt = model_cfg.prompt_template.format(instruction=sample.instruction)
    inputs = processor(prompt, sample.image, return_tensors="pt").to(device)

    if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

    input_ids = inputs["input_ids"][0]
    num_text = len(input_ids)

    # Capture sequence length via hook on first layer
    captured = {}
    layers = get_layers(model, model_cfg)

    def hook_fn(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["seq_len"] = h.shape[1]

    hook = layers[0].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**inputs, use_cache=False)
    hook.remove()

    full_seq = captured["seq_len"]
    num_vision = full_seq - num_text

    return {
        "vision_start": 0,
        "vision_end": num_vision,
        "text_start": num_vision,
        "text_end": full_seq,
        "total_seq_len": full_seq,
        "num_vision_tokens": num_vision,
        "num_text_tokens": num_text,
    }


def extract_attention_generic(
    processor, model, model_cfg: VLAModelConfig,
    sample: DatasetSample, boundaries: dict, device: str,
) -> dict:
    """Extract attention from any VLA model for a single sample.

    Returns dict with attention_analysis (head-averaged) and
    perhead_analysis (per-head breakdown).
    """
    prompt = model_cfg.prompt_template.format(instruction=sample.instruction)
    inputs = processor(prompt, sample.image, return_tensors="pt").to(device)
    if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    # Build expanded input_ids for text decoding
    original_ids = inputs["input_ids"][0]
    num_vision = boundaries["num_vision_tokens"]
    dummy_vision = torch.zeros(num_vision, dtype=torch.long)
    expanded_ids = torch.cat([dummy_vision, original_ids.cpu()])

    # Enable output_attentions
    layers = get_layers(model, model_cfg)
    attn_store = {}

    def make_hook(layer_idx):
        def hook_fn(module, args, output):
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                attn_store[layer_idx] = output[1].detach().cpu()
        return hook_fn

    # Set output_attentions on the model config
    if hasattr(model, "language_model"):
        model.language_model.config.output_attentions = True
    else:
        model.config.output_attentions = True

    hooks = []
    for i, layer in enumerate(layers):
        attn_mod = getattr(layer, model_cfg.attn_module)
        hooks.append(attn_mod.register_forward_hook(make_hook(i)))

    # Generate action tokens autoregressively
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs.get("pixel_values")

    num_action_tokens = model_cfg.action_tokens
    attention_analysis = {}
    perhead_analysis = {}
    generated_tokens = []

    dim_names = config.ACTION_DIM_NAMES[:num_action_tokens]

    with torch.no_grad():
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }

        for act_idx in range(num_action_tokens):
            attn_store.clear()

            outputs = model(**model_inputs, use_cache=False)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_tokens.append(next_token.item() if next_token.dim() == 1
                                    else next_token[0].item())

            dim_name = dim_names[act_idx] if act_idx < len(dim_names) else f"dim{act_idx}"
            action_key = f"action_{act_idx}_{dim_name}"
            attention_analysis[action_key] = {}
            perhead_analysis[action_key] = {}

            for layer_idx in sorted(attn_store.keys()):
                attn = attn_store[layer_idx]  # (1, H, S, S)
                attn_last = attn[0, :, -1:, :]  # (H, 1, S)
                layer_key = f"layer_{layer_idx:02d}"

                # Head-averaged top-5
                top_k = analyze_top_k(attn_last, boundaries, tokenizer, expanded_ids)
                attention_analysis[action_key][layer_key] = {"top5": top_k}

                # Per-head breakdown
                perhead_analysis[action_key][layer_key] = compute_perhead_stats(
                    attn_last, boundaries
                )

            # Extend sequence for next token
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)
                                   if next_token.dim() == 1 else next_token], dim=-1)
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(1, 1, device=device, dtype=attention_mask.dtype)
                ], dim=-1)
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
            }

    # Cleanup hooks
    for h in hooks:
        h.remove()

    return {
        "attention_analysis": attention_analysis,
        "perhead_analysis": perhead_analysis,
        "generated_tokens": generated_tokens,
    }


def run_single_model(
    model_name: str, dataset_name: str = "bridge_v2",
    device: str = "cuda", episode_id: int = 0, step_id: int = 0,
):
    """Run extraction for a single model + dataset combo."""
    model_cfg = get_model(model_name)

    # Load sample via universal dispatcher
    sample = load_sample(dataset_name, episode_id, step_id)

    # Load model
    processor, model = load_vla_model(model_cfg, device)

    # Detect boundaries
    boundaries = detect_boundaries(processor, model, model_cfg, sample, device)
    print(f"  Boundaries: {boundaries['num_vision_tokens']} vision, "
          f"{boundaries['num_text_tokens']} text")

    # Extract
    result = extract_attention_generic(
        processor, model, model_cfg, sample, boundaries, device
    )

    # Save outputs
    out_dir = CROSS_MODEL_DIR / model_name / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Head-averaged JSON
    avg_path = out_dir / f"ep{episode_id:03d}_step{step_id:03d}.json"
    avg_output = {
        "model": model_name,
        "model_hf_id": model_cfg.hf_id,
        "architecture": model_cfg.architecture,
        "dataset": dataset_name,
        "episode_id": episode_id,
        "step_id": step_id,
        "instruction": sample.instruction,
        "token_boundaries": boundaries,
        "attention_analysis": result["attention_analysis"],
    }
    with open(avg_path, "w") as f:
        json.dump(avg_output, f, indent=2, ensure_ascii=False)

    # Per-head JSON
    perhead_path = out_dir / f"ep{episode_id:03d}_step{step_id:03d}_perhead.json"
    perhead_output = {
        "model": model_name,
        "dataset": dataset_name,
        "episode_id": episode_id,
        "step_id": step_id,
        "instruction": sample.instruction,
        "token_boundaries": boundaries,
        "perhead_analysis": result["perhead_analysis"],
    }
    with open(perhead_path, "w") as f:
        json.dump(perhead_output, f, indent=2, ensure_ascii=False)

    # Per-head heatmap
    visualize_perhead_sink(perhead_path, output_dir=out_dir)

    print(f"  Saved: {avg_path}")
    print(f"  Saved: {perhead_path}")

    # Free GPU memory
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    return avg_output, perhead_output


def main():
    parser = argparse.ArgumentParser(description="Cross-model VLA attention extraction")
    parser.add_argument("--model", type=str, default=None, help="Model name from registry")
    parser.add_argument("--dataset", type=str, default="bridge_v2", help="Dataset name")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--all", action="store_true",
                        help="Run all models with bridge_v2")
    parser.add_argument("--list", action="store_true", help="List available models")
    args = parser.parse_args()

    if args.list:
        from model_registry import MODELS
        for name, cfg in MODELS.items():
            print(f"  {name:20s} {cfg.hf_id:45s} ({cfg.architecture}, {cfg.num_layers}L)")
        return

    if args.all:
        models = list_models()
        for m in models:
            print(f"\n{'=' * 60}")
            print(f"  MODEL: {m}")
            print(f"{'=' * 60}")
            try:
                run_single_model(m, args.dataset, args.device, args.episode, args.step)
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()
    elif args.model:
        run_single_model(args.model, args.dataset, args.device, args.episode, args.step)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
