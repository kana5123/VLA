# Comprehensive VLA Attention Sink Analysis & Adapter Experiment Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Validate the attention sink phenomenon across multiple VLA models, redesign the rho filter to handle dual sinks (vision[0] + text "\n"), and run the 4-config adapter experiment with corrected architecture.

**Architecture:** Three parallel work streams: (A) Cross-model attention analysis pipeline with per-head extraction for 7+ VLA models across 5+ datasets, (B) Rho filter & VAR redesign based on empirical dual-sink observations, (C) Adapter v2 experiment execution (4 configs: base/v1/v2-prop/v2-full). Stream A provides evidence for the paper's universality claim. Stream B fixes the architectural flaw before final training. Stream C generates quantitative results.

**Tech Stack:** PyTorch, HuggingFace Transformers, Accelerate (DDP), matplotlib/seaborn, numpy, SAM2, GroundingDINO, peft (LoRA). Models: OpenVLA, CogACT, TraceVLA, SpatialVLA, SmolVLA, ECoT, RoboFlamingo.

---

## Current State (2026-02-22)

| Component | Status | Notes |
|-----------|--------|-------|
| extract_attention.py | **Complete** | Head-averaged + per-head extraction, per-head heatmap visualization |
| adapter_model.py (V1+V2) | **Complete** | MLP + two-branch object-aware |
| adapter_train.py | **Complete** | V1/V2 DDP, freeze_blend, output_dir |
| adapter_eval.py | **Complete** | V2 auto-detect, object masks, baseline_only |
| adapter_data.py | **Complete** | Cache + tfrecord, object mask loading |
| attention_v3.py | **Complete** | VAR/ACT/SPIN/VTR, redistribution_weights |
| run_adapter_experiment.py | **Complete** | 4-config orchestrator |
| compare_adapter_results.py | **Complete** | Plots, heatmap, LaTeX table |
| adapter_lora.py | **Complete** | LoRA infra (peft wrapper) |
| sam_preprocess.py | **83% done** | ~234K steps remaining on GPUs 1-7 |
| Per-head analysis | **Code ready** | `compute_perhead_stats()` + `visualize_perhead_sink()` in extract_attention.py |
| Cross-model pipeline | **Not started** | New: model_registry, dataset_registry, cross_model_analysis |
| Rho filter redesign | **Not started** | Dual-sink fix needed in attention_v3.py |

### Critical Finding (Previous Session)

Empirical analysis of OpenVLA attention data revealed:
- **vision[0] absorbs 25-53%** of attention in layers 28-31
- **Text "\n" token absorbs 20-35%** of attention
- Together: **45-75% of all attention goes to two sink tokens**
- Task-relevant tokens ("spoon", "cube") get only 2-7%
- Current rho filter (vision ratio > 0.5) is **flawed**: counts sink attention as "vision attention", making sink-heavy heads appear image-centric
- **Two sinks, not one**: VAR only handles vision[0], text "\n" sink is untouched

---

## Stream A: Cross-Model VLA Attention Sink Analysis

### Task 1: Create model_registry.py — VLA model definitions

**Files:**
- Create: `model_registry.py`

**Step 1: Write the model registry**

```python
"""Registry of VLA models for cross-model attention analysis.

Each entry defines how to load the model, what architecture it uses,
and how to extract attention weights from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VLAModelConfig:
    """Configuration for a single VLA model."""
    name: str                          # Display name
    hf_id: str                         # HuggingFace model ID
    architecture: str                  # "llama", "phi3", "qwen2", "mistral", etc.
    vision_encoder: str                # "prismatic", "siglip", "dinov2", etc.
    num_layers: int                    # LLM backbone layers
    num_heads: int                     # Attention heads
    hidden_dim: int                    # Hidden dimension
    vision_grid_size: int              # e.g., 16 for 16x16 = 256 tokens
    num_vision_tokens: int             # Total vision tokens
    action_tokens: int = 7             # Number of action tokens generated
    prompt_template: str = ""          # Model-specific prompt format
    trust_remote_code: bool = True
    attn_impl: str = "eager"           # Must be eager for weight extraction
    torch_dtype: str = "bfloat16"
    notes: str = ""
    native_datasets: list[str] = field(default_factory=list)
    # Architecture-specific: where to find attention layers
    layers_path: str = "language_model.model.layers"  # dot-separated path
    attn_module: str = "self_attn"     # attribute name for attention in each layer


# ═══════════════════════════════════════════════════════════════════════
# Model Registry
# ═══════════════════════════════════════════════════════════════════════

MODELS: dict[str, VLAModelConfig] = {}


def register(cfg: VLAModelConfig):
    MODELS[cfg.name] = cfg


# ── OpenVLA (LLaMA-2 7B + Prismatic) ──
register(VLAModelConfig(
    name="openvla-7b",
    hf_id="openvla/openvla-7b",
    architecture="llama",
    vision_encoder="prismatic",
    num_layers=32,
    num_heads=32,
    hidden_dim=4096,
    vision_grid_size=16,
    num_vision_tokens=256,
    action_tokens=7,
    prompt_template="In: What action should the robot take to {instruction}?\nOut:",
    native_datasets=["bridge_v2", "oxe"],
    notes="Dual encoder (DINOv2+SigLIP) fused to 256 tokens",
))

# ── CogACT (CogVLM2-Llama3-8B backbone) ──
register(VLAModelConfig(
    name="cogact-base",
    hf_id="CogACT/CogACT-Base",
    architecture="llama",
    vision_encoder="eva2-clip-e",
    num_layers=32,
    num_heads=32,
    hidden_dim=4096,
    vision_grid_size=14,
    num_vision_tokens=196,  # 14x14
    action_tokens=7,
    prompt_template="What action should the robot take to {instruction}?",
    native_datasets=["oxe", "bridge_v2"],
    notes="CogVLM2 backbone, EVA2-CLIP-E vision encoder",
    layers_path="model.layers",
))

# ── TraceVLA (Phi-3-V backbone) ──
register(VLAModelConfig(
    name="tracevla-phi3v",
    hf_id="zxliu/TraceVLA-Phi3V",
    architecture="phi3",
    vision_encoder="clip-vit",
    num_layers=32,
    num_heads=32,
    hidden_dim=3072,
    vision_grid_size=14,
    num_vision_tokens=196,
    action_tokens=7,
    prompt_template="What action should the robot take to {instruction}?",
    native_datasets=["bridge_v2", "oxe"],
    notes="Phi-3 Vision backbone, visual trace overlays",
    layers_path="model.layers",
))

# ── SpatialVLA (Qwen2-VL backbone) ──
register(VLAModelConfig(
    name="spatialvla-4b",
    hf_id="IPEC-COMMUNITY/spatialvla-4b-224-pt",
    architecture="qwen2",
    vision_encoder="qwen2-vl-vit",
    num_layers=28,
    num_heads=20,
    hidden_dim=2560,
    vision_grid_size=14,
    num_vision_tokens=196,
    action_tokens=7,
    prompt_template="What action should the robot take to {instruction}?",
    native_datasets=["bridge_v2", "oxe"],
    notes="Qwen2-VL 4B, adaptive resolution vision",
    layers_path="model.layers",
))

# ── SmolVLA (SmolLM2 backbone) ──
register(VLAModelConfig(
    name="smolvla-base",
    hf_id="HuggingFaceTB/SmolVLA-base",
    architecture="smollm2",
    vision_encoder="siglip",
    num_layers=30,
    num_heads=16,
    hidden_dim=1536,
    vision_grid_size=14,
    num_vision_tokens=196,
    action_tokens=7,
    prompt_template="What action should the robot take to {instruction}?",
    native_datasets=["lerobot"],
    notes="SmolLM2 360M backbone, lightweight",
    layers_path="model.layers",
))

# ── ECoT (OpenVLA + Chain-of-Thought) ──
register(VLAModelConfig(
    name="ecot-7b",
    hf_id="Embodied-CoT/ecot-openvla-7b-bridge",
    architecture="llama",
    vision_encoder="prismatic",
    num_layers=32,
    num_heads=32,
    hidden_dim=4096,
    vision_grid_size=16,
    num_vision_tokens=256,
    action_tokens=7,
    prompt_template="In: What action should the robot take to {instruction}?\nOut:",
    native_datasets=["bridge_v2"],
    notes="Same backbone as OpenVLA, fine-tuned with chain-of-thought",
))

# ── RoboFlamingo (MPT-1B or 3B backbone) ──
register(VLAModelConfig(
    name="roboflamingo",
    hf_id="roboflamingo/RoboFlamingo",
    architecture="mpt",
    vision_encoder="clip-vit",
    num_layers=24,
    num_heads=16,
    hidden_dim=2048,
    vision_grid_size=14,
    num_vision_tokens=196,
    action_tokens=7,
    prompt_template="{instruction}",
    native_datasets=["calvin"],
    notes="OpenFlamingo-based, cross-attention (different attn pattern)",
    layers_path="transformer.blocks",
    attn_module="attn",
))


def get_model(name: str) -> VLAModelConfig:
    if name not in MODELS:
        available = ", ".join(MODELS.keys())
        raise ValueError(f"Unknown model: {name}. Available: {available}")
    return MODELS[name]


def list_models() -> list[str]:
    return list(MODELS.keys())
```

**Step 2: Verify**

Run: `python -c "from model_registry import list_models; print(list_models())"`
Expected: List of 7 model names

**Step 3: Commit**

```bash
git add model_registry.py
git commit -m "feat: add VLA model registry with 7 models for cross-model analysis"
```

---

### Task 2: Create dataset_registry.py — Dataset download & loading

**Files:**
- Create: `dataset_registry.py`

**Step 1: Write the dataset registry**

```python
"""Registry of robot manipulation datasets for cross-model attention analysis.

Each entry defines how to download, load, and sample from the dataset.
We only need 1 demo (episode) per dataset for attention analysis.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

import config

DATASET_CACHE = Path("/ceph_data/kana5123/cross_model_datasets")


@dataclass
class DatasetConfig:
    """Configuration for a single robot dataset."""
    name: str
    display_name: str
    download_cmd: str               # Shell command to download
    download_size: str              # Approximate download size
    format: str                     # "tfrecord", "hdf5", "lerobot", "custom"
    default_instruction: str        # Fallback instruction if not in data
    image_key: str = "image"        # Key for image in data dict
    action_key: str = "action"      # Key for action in data dict
    instruction_key: str = ""       # Key for instruction (empty = use default)
    notes: str = ""


@dataclass
class DatasetSample:
    """A single demo sample from a dataset."""
    dataset_name: str
    episode_id: int
    step_id: int
    image: Image.Image              # PIL RGB image
    instruction: str
    action: Optional[list[float]]   # Ground truth action (if available)


DATASETS: dict[str, DatasetConfig] = {}


def register_dataset(cfg: DatasetConfig):
    DATASETS[cfg.name] = cfg


# ── Bridge V2 (already downloaded) ──
register_dataset(DatasetConfig(
    name="bridge_v2",
    display_name="Bridge V2",
    download_cmd="# Already available at /ceph_data/kana5123/bridge_v2_data",
    download_size="~25GB (already cached)",
    format="tfrecord",
    default_instruction="pick up the object",
    notes="Primary dataset, already cached and preprocessed",
))

# ── CALVIN (debug split, small) ──
register_dataset(DatasetConfig(
    name="calvin_debug",
    display_name="CALVIN Debug",
    download_cmd=(
        "wget -P {cache_dir}/calvin "
        "http://calvin.cs.uni-freiburg.de/dataset/task_D_D_debug.zip && "
        "cd {cache_dir}/calvin && unzip task_D_D_debug.zip"
    ),
    download_size="~1.3GB",
    format="hdf5",
    default_instruction="slide the door to the right",
    notes="CALVIN benchmark debug split (small, fast download)",
))

# ── DROID (droid_100 subset) ──
register_dataset(DatasetConfig(
    name="droid_100",
    display_name="DROID-100",
    download_cmd=(
        "pip install gdown && "
        "gdown --folder https://drive.google.com/drive/folders/droid_100_id "
        "-O {cache_dir}/droid_100"
    ),
    download_size="~2GB",
    format="hdf5",
    default_instruction="pick up the object and place it in the bin",
    notes="100-episode subset of DROID",
))

# ── LeRobot (HuggingFace datasets) ──
register_dataset(DatasetConfig(
    name="lerobot_pusht",
    display_name="LeRobot PushT",
    download_cmd=(
        "pip install lerobot && "
        "python -c \"from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; "
        "ds = LeRobotDataset('lerobot/pusht', root='{cache_dir}/lerobot')\""
    ),
    download_size="~500MB",
    format="lerobot",
    default_instruction="push the T-shaped block to the target",
    notes="LeRobot PushT task, standard benchmark",
))

# ── RH20T (mini subset) ──
register_dataset(DatasetConfig(
    name="rh20t_mini",
    display_name="RH20T Mini",
    download_cmd=(
        "# Download from https://rh20t.github.io/ — select 1 scene\n"
        "# Manual download required, ~26GB minimum"
    ),
    download_size="~26GB",
    format="custom",
    default_instruction="grasp the object",
    notes="Real-world bimanual dataset, large",
))


def get_dataset(name: str) -> DatasetConfig:
    if name not in DATASETS:
        available = ", ".join(DATASETS.keys())
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")
    return DATASETS[name]


def list_datasets() -> list[str]:
    return list(DATASETS.keys())


def load_bridge_sample(episode_id: int = 0, step_id: int = 0) -> DatasetSample:
    """Load a sample from the already-cached Bridge V2 dataset."""
    meta_path = config.METADATA_PATH
    with open(meta_path) as f:
        metadata = json.load(f)

    ep = metadata["episodes"][episode_id]
    step = ep["steps"][step_id]
    image_path = config.PROJECT_ROOT / step["image_path"]
    image = Image.open(image_path).convert("RGB")

    return DatasetSample(
        dataset_name="bridge_v2",
        episode_id=ep["episode_id"],
        step_id=step["step_id"],
        image=image,
        instruction=step["instruction"],
        action=step.get("action"),
    )


def download_dataset(name: str) -> Path:
    """Download a dataset to the cache directory."""
    cfg = get_dataset(name)
    cache_dir = DATASET_CACHE / name
    cache_dir.mkdir(parents=True, exist_ok=True)

    cmd = cfg.download_cmd.format(cache_dir=str(cache_dir))
    print(f"Downloading {cfg.display_name}...")
    print(f"  Command: {cmd}")
    print(f"  Size: {cfg.download_size}")

    os.system(cmd)
    return cache_dir
```

**Step 2: Verify**

Run: `python -c "from dataset_registry import list_datasets, load_bridge_sample; print(list_datasets()); s = load_bridge_sample(); print(f'{s.dataset_name} ep{s.episode_id} step{s.step_id}: {s.instruction[:40]}')" `
Expected: Dataset list + bridge sample loaded

**Step 3: Commit**

```bash
git add dataset_registry.py
git commit -m "feat: add dataset registry with 5 datasets for cross-model analysis"
```

---

### Task 3: Create cross_model_extract.py — Generic attention extraction for any VLA

**Files:**
- Create: `cross_model_extract.py`

**Context:** The existing `extract_attention.py` is hardcoded for OpenVLA (Prismatic vision encoder, LLaMA backbone). This new script abstracts the extraction to work with any model from the registry. It reuses `compute_perhead_stats()` and `analyze_top_k()` from the existing code, and adds model-specific loading logic.

**Step 1: Write the generic extractor**

Create `cross_model_extract.py` (~350 lines). Key functions:

```python
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
from dataset_registry import load_bridge_sample, DatasetSample
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

    # Load sample
    if dataset_name == "bridge_v2":
        sample = load_bridge_sample(episode_id, step_id)
    else:
        raise NotImplementedError(f"Dataset loader for {dataset_name} not yet implemented")

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
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('cross_model_extract.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add cross_model_extract.py
git commit -m "feat: add cross-model attention extraction for VLA model comparison"
```

---

### Task 4: Create cross_model_compare.py — Cross-model comparison & visualization

**Files:**
- Create: `cross_model_compare.py`

**Context:** After running `cross_model_extract.py --all`, this script loads per-head JSON from all models and produces comparison visualizations: (1) Per-model sink ratio heatmap, (2) Cross-model bar chart of vision[0] / text_sink / useful vision ratios, (3) Summary table for the paper.

**Step 1: Write the comparison script**

Create `cross_model_compare.py` (~300 lines). Key outputs:
- `cross_model_sink_comparison.png` — Bar chart: mean vision[0] ratio per model across all layers
- `cross_model_heatmap.png` — Heatmap (models × layers) of vision[0] attention fraction
- `cross_model_dual_sink.png` — Stacked bar: vision[0] + text_max + useful per model
- `cross_model_summary.json` — JSON with all numeric results
- `cross_model_table.tex` — LaTeX table for paper

Key function signatures:
```python
def load_all_perhead(base_dir: Path) -> dict[str, dict]:
    """Load perhead JSONs from all model subdirectories."""

def compute_sink_summary(model_name: str, perhead: dict) -> dict:
    """Compute mean sink ratios across all layers/heads for one model."""

def plot_cross_model_comparison(summaries: dict, output_dir: Path):
    """Generate all cross-model comparison figures."""

def generate_latex_table(summaries: dict, output_dir: Path):
    """Generate LaTeX table of sink ratios per model."""
```

**Step 2: Verify**

Run: `python -c "import ast; ast.parse(open('cross_model_compare.py').read()); print('OK')"`

**Step 3: Commit**

```bash
git add cross_model_compare.py
git commit -m "feat: add cross-model attention sink comparison with plots and LaTeX table"
```

---

### Task 5: Download additional datasets and run cross-model extraction

**Files:**
- No new files (uses cross_model_extract.py)

**Step 1: Download CALVIN debug**

```bash
mkdir -p /ceph_data/kana5123/cross_model_datasets/calvin
cd /ceph_data/kana5123/cross_model_datasets/calvin
wget http://calvin.cs.uni-freiburg.de/dataset/task_D_D_debug.zip
unzip task_D_D_debug.zip
```

**Step 2: Run extraction for all models on bridge_v2**

```bash
# Run all models sequentially on GPU 0 (other GPUs doing SAM)
cd /home/kana5123/ATLASVLA
CUDA_VISIBLE_DEVICES=0 python cross_model_extract.py --all --device cuda:0

# If some models fail, run individually:
CUDA_VISIBLE_DEVICES=0 python cross_model_extract.py --model openvla-7b --device cuda:0
CUDA_VISIBLE_DEVICES=0 python cross_model_extract.py --model cogact-base --device cuda:0
# ... etc for each model
```

**Step 3: Generate comparison**

```bash
python cross_model_compare.py
```

**Step 4: Commit results**

```bash
git add outputs/cross_model_analysis/
git commit -m "data: add cross-model attention sink analysis results"
```

---

## Stream B: Rho Filter & VAR Redesign

### Task 6: Redesign rho filter in attention_v3.py — Sink-aware head selection

**Files:**
- Modify: `attention_v3.py:220-240` (rho filter section in apply_var)

**Context:** The current rho filter at line 234 computes:
```python
ratio = attn[:, :, :, :ve].sum(dim=-1)  # vision ratio per head
head_mask = (ratio.mean(dim=0) >= rho).float()
```
This counts vision[0] (the sink) as "vision attention", so sink-heavy heads pass the filter. The fix: compute vision ratio EXCLUDING token 0.

**Step 1: Replace rho filter logic**

In `attention_v3.py`, replace lines 220-240 (the head selection block inside `apply_var`):

From (current):
```python
    # Step 1: Image-centric head selection
    # ratio = fraction of attention on vision tokens, per head
    ratio = attn[:, :, :, :ve].sum(dim=-1)  # (B, H, S)
    head_mask = (ratio.mean(dim=0) >= rho).float()  # (H,)
```

To (new — sink-aware):
```python
    # Step 1: Image-centric head selection (sink-aware)
    # Compute vision ratio EXCLUDING sink tokens
    non_sink_vision_mask = torch.ones(ve, device=last.device)
    for si in sink_indices:
        if si < ve:
            non_sink_vision_mask[si] = 0.0
    # Useful vision ratio = attention to non-sink vision tokens
    useful_vision = (attn[:, :, :, :ve] * non_sink_vision_mask).sum(dim=-1)  # (B, H, S)
    head_mask = (useful_vision.mean(dim=0) >= rho).float()  # (H,)
```

**Step 2: Verify syntax**

Run: `python -c "from attention_v3 import apply_var; print('OK')"`

**Step 3: Commit**

```bash
git add attention_v3.py
git commit -m "fix: rho filter now excludes sink tokens from vision ratio calculation"
```

---

### Task 7: Add text sink handling to VAR — Redistribute from "\n" token too

**Files:**
- Modify: `attention_v3.py:178-300` (apply_var function)
- Modify: `config.py` (add TEXT_SINK_HANDLING flag)

**Context:** Currently VAR only handles vision[0] as a sink. The text "\n" token also absorbs 20-35% of attention. We add an optional text sink redistribution that identifies the max-attention text token per head and redistributes a fraction of it to non-sink vision tokens.

**Step 1: Add config constants**

In `config.py`, after line 88 (VAR_OBJECT_REDIRECT_WEIGHT), add:

```python
VAR_TEXT_SINK_ENABLED = True        # Also redistribute from text sink ("\n")
VAR_TEXT_SINK_P = 0.3               # Fraction of text sink attention to redistribute
VAR_TEXT_SINK_THRESHOLD = 0.15      # Only redistribute if text max > 15% of total
```

**Step 2: Extend apply_var to handle text sinks**

In `attention_v3.py`, after the existing sink redistribution block (after the vision sink loop ~line 289), add a new text sink block:

```python
    # Step 4: Text sink redistribution (optional)
    if text_sink_enabled and ve < seq_len:
        text_region = attn[:, :, :, ve:te]  # (B, H, S, T_text)
        text_max_val, text_max_idx = text_region.max(dim=-1)  # (B, H, S)
        # Only redistribute from text tokens that hog > threshold
        text_sink_mask = (text_max_val > text_sink_threshold)  # (B, H, S)
        text_to_move = text_max_val * text_sink_p * text_sink_mask.float()  # (B, H, S)
        # Reduce the text sink
        # ... (scatter subtract from text_max_idx, add to non-sink vision)
```

This requires updating the `apply_var` signature to accept `text_sink_enabled`, `text_sink_p`, `text_sink_threshold`, and `te` (text_end).

**Step 3: Update V3Context and patched forward**

Add fields to V3Context dataclass:
```python
text_sink_enabled: bool = False
text_sink_p: float = 0.3
text_sink_threshold: float = 0.15
text_end: int = 0
```

Update `_make_v3_patched_forward()` to pass these to `apply_var()`.

**Step 4: Verify syntax**

Run: `python -c "from attention_v3 import apply_var, V3Context; print('OK')"`

**Step 5: Run tests**

Run: `python -m pytest tests/test_apply_var_v2.py -v`
Expected: Existing tests pass (text sink disabled by default)

**Step 6: Commit**

```bash
git add attention_v3.py config.py
git commit -m "feat: add text sink redistribution to VAR (dual-sink handling)"
```

---

### Task 8: Update adapter training to pass text_end and text_sink params

**Files:**
- Modify: `adapter_train.py:100-145` (forward_with_adapter, V3Context creation)
- Modify: `adapter_eval.py:254-270` (get_v3_ctx_for_eval)

**Step 1: Update V3Context creation in training**

In `adapter_train.py`, where `V3Context` is created (inside `forward_with_adapter`), add the text sink fields:

```python
    ctx = V3Context(
        active=True,
        use_var=True,
        var_p=config.VAR_P,
        var_rho=config.VAR_RHO,
        var_sink_indices=list(config.VAR_SINK_INDICES),
        vision_end=vision_end,
        text_end=text_end,                          # NEW
        text_sink_enabled=config.VAR_TEXT_SINK_ENABLED,  # NEW
        text_sink_p=config.VAR_TEXT_SINK_P,              # NEW
        text_sink_threshold=config.VAR_TEXT_SINK_THRESHOLD,  # NEW
        enhancement_layers=set(config.ADAPTER_TARGET_LAYERS),
        per_head_var_strength=full_p,
        redistribution_weights=redistribution_weights,
    )
```

**Step 2: Same for evaluation**

Update `get_v3_ctx_for_eval()` in `adapter_eval.py` with the same fields.

**Step 3: Verify syntax**

Run: `python -c "from adapter_train import train; print('OK')"`

**Step 4: Commit**

```bash
git add adapter_train.py adapter_eval.py
git commit -m "feat: pass text_end and text sink params to V3Context in training/eval"
```

---

## Stream C: Adapter Experiment Execution

### Task 9: Verify SAM preprocessing completion

**Files:** None (monitoring only)

**Step 1: Check SAM progress**

```bash
python -c "
import numpy as np
masks = np.memmap('/ceph_data/kana5123/bridge_data_cache/object_masks.dat',
                  dtype='uint8', mode='r', shape=(1382356, 256))
done = 0
batch = 10000
for start in range(0, 1382356, batch):
    end = min(start + batch, 1382356)
    chunk = np.array(masks[start:end])
    done += (~(chunk == 255).all(axis=1)).sum()
print(f'SAM progress: {done:,} / 1,382,356 ({100*done/1382356:.1f}%)')
"
```

**Step 2: Wait for 100% or assess if partial is sufficient**

If SAM stalls, the v2-prop config (frozen blend, no SAM masks needed for proportional redistribution) can still run. Only v2-full requires SAM masks.

---

### Task 10: Run the 4-config adapter experiment

**Files:** None (uses existing run_adapter_experiment.py)

**Prerequisites:** SAM preprocessing complete (or at least 95%+ for v2-full).

**Step 1: Run full experiment**

```bash
cd /home/kana5123/ATLASVLA
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_adapter_experiment.py \
    --gpus 0,1,2,3 \
    --num_eval_episodes 200

# If you want to run configs individually:
# Base (eval only, fast):
python run_adapter_experiment.py --configs base --eval_device cuda:0

# V1 (train + eval):
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_adapter_experiment.py --configs v1 --gpus 0,1,2,3

# V2-prop (train + eval, no SAM needed):
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_adapter_experiment.py --configs v2-prop --gpus 0,1,2,3

# V2-full (train + eval, needs SAM):
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_adapter_experiment.py --configs v2-full --gpus 0,1,2,3
```

**Step 2: Compare results**

```bash
python compare_adapter_results.py
```

**Step 3: Review outputs**

```bash
cat outputs/experiment_results/comparison_summary.json
ls outputs/experiment_results/plots/
```

**Step 4: Commit results**

```bash
git add outputs/experiment_results/
git commit -m "data: add 4-config adapter experiment results"
```

---

### Task 11: Re-run experiment with text sink fix (optional, after Task 7)

**Files:** None (uses same pipeline)

**Context:** After implementing the text sink fix (Task 7), re-run the experiment to compare with/without text sink handling. This provides ablation data for the paper.

**Step 1: Add new configs to run_adapter_experiment.py**

```python
# In CONFIGS dict, add:
"v2-full-textsink": {
    "adapter_version": 2,
    "freeze_blend": False,
    "extra_flags": ["--text_sink_enabled"],
    "description": "V2 + text sink redistribution",
},
```

**Step 2: Run with text sink**

```bash
python run_adapter_experiment.py --configs v2-full-textsink --gpus 0,1,2,3
```

**Step 3: Compare all 5 configs**

```bash
python compare_adapter_results.py
```

---

## Execution Order & Dependencies

```
[Now]  SAM preprocessing finishing (83% → 100%)
  │
  ├─ Stream A (can start immediately, uses GPU 0)
  │    Task 1: model_registry.py ─────────────────────┐
  │    Task 2: dataset_registry.py ───────────────────┤ parallel
  │    Task 3: cross_model_extract.py ────────────────┘ depends on 1+2
  │    Task 4: cross_model_compare.py ──────── depends on 3
  │    Task 5: Download + run extraction ───── depends on 3+4
  │
  ├─ Stream B (can start immediately, code only)
  │    Task 6: Rho filter fix ────────────────┐
  │    Task 7: Text sink handling ────────────┤ sequential
  │    Task 8: Training/eval integration ─────┘
  │
  └─ Stream C (requires SAM complete)
       Task 9: Verify SAM completion ─────────┐
       Task 10: Run 4-config experiment ──────┤ sequential
       Task 11: Re-run with text sink ────────┘ depends on B.Task 8
```

**Parallelization:**
- Tasks 1-2 (registries) are independent — run in parallel
- Task 3 depends on 1+2
- Tasks 6-8 (Stream B) are independent of Stream A
- Stream B Tasks 6-8 can run parallel with Stream A Tasks 1-5
- Stream C waits for SAM + optionally Stream B completion

**Estimated timeline:**
- Stream A Tasks 1-4: Code implementation (~2 hours)
- Stream A Task 5: Model downloads + extraction (~4-6 hours, GPU-bound)
- Stream B Tasks 6-8: Code implementation (~1 hour)
- Stream C Task 10: Training (~8-12 hours on 4×H100) + Eval (~1 hour)

---

## Success Criteria

1. **Cross-model analysis produces**: Per-head heatmaps for 5+ VLA models showing vision[0] sink pattern, JSON summaries, comparison plots for paper
2. **Rho filter is fixed**: Sink-heavy heads correctly identified as non-useful; text sink redistributed
3. **4-config experiment completes**: baseline MSE, v1 MSE, v2-prop MSE, v2-full MSE with per-dimension breakdown
4. **Paper figures ready**: Cross-model sink comparison bar chart, adapter improvement heatmap, per-dimension MSE comparison
