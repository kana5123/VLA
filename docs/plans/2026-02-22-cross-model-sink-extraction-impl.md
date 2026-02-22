# Cross-Model Sink Extraction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract attention sink patterns from 4 unextracted VLA models (CogACT, SpatialVLA, SmolVLA, RoboFlamingo), update cross-model comparison to 7 models, verify dynamic `detect_sinks()` works across all architectures, then prepare adapter training with dynamic detection.

**Architecture:** The extraction pipeline (`cross_model_extract.py`) already works for 3 models (OpenVLA, ECoT, TraceVLA). We run it on 4 new models across GPUs 5-7 in parallel, fixing any model-specific loading issues that arise. After extraction, `cross_model_compare.py` regenerates all comparison visualizations. Finally, `detect_sinks(alpha=5.0)` is validated against each model's extracted attention data.

**Tech Stack:** PyTorch 2.5.1, HuggingFace Transformers 4.57.6, matplotlib, numpy. Models loaded via `AutoModelForVision2Seq` (or `AutoModelForCausalLM` for special architectures). Data: Bridge V2 (cached at `/ceph_data/kana5123/bridge_data_cache/`).

---

## Context for Implementer

### Files you'll be working with
- `cross_model_extract.py` — Main extraction script (396 lines). Loads any VLA model from registry, extracts per-head attention weights, saves JSON + heatmaps.
- `model_registry.py` — VLA model configs (187 lines). Each model has `layers_path`, `attn_module`, `architecture` fields.
- `cross_model_compare.py` — Comparison visualization script (400 lines). Reads perhead JSONs, generates bar charts, heatmaps, LaTeX table.
- `attention_v3.py` — Contains `detect_sinks()` function (lines 41-70) using ACT-style α/N threshold.
- `dataset_registry.py` — Dataset loading. `load_sample("bridge_v2", 0, 0)` returns a `DatasetSample` with image + instruction.
- `run_cross_model.sh` — Existing GPU parallel execution script (needs updating for 4 new models).

### Environment
- Python venv: `/home/kana5123/ATLASVLA/venv/bin/python`
- Conda env (for GPU scripts): `/home/kana5123/miniconda3/envs/interp/bin/python`
- HF cache: `HF_HOME=/ceph_data/kana5123/hf_cache`
- Output dir: `outputs/cross_model_analysis/<model_name>/bridge_v2/`
- GPUs available: 5, 6, 7 (GPUs 1-4 running adapter training)

### Already extracted (3 models)
| Model | Architecture | Sink Pattern |
|-------|-------------|-------------|
| openvla-7b | LLaMA-2 + Prismatic | vision[0] = 45.4% |
| ecot-7b | LLaMA-2 + Prismatic | vision[0] = 74.5% |
| tracevla-phi3v | Phi-3-V | vision[0] = 0.2% (no sink) |

### Models to extract (4 models)
| Model | HF ID | Architecture | layers_path | VRAM Est. |
|-------|-------|-------------|------------|-----------|
| cogact-base | CogACT/CogACT-Base | llama (CogVLM2) | model.layers | ~17GB |
| spatialvla-4b | IPEC-COMMUNITY/spatialvla-4b-224-pt | gemma2 | language_model.model.layers | ~10GB |
| smolvla-base | lerobot/smolvla_base (VLM: SmolVLM2-500M) | llama | model.text_model.model.layers | ~2GB |
| roboflamingo | roboflamingo/RoboFlamingo | mpt (cross-attn) | transformer.blocks | ~5GB |

---

## Task 1: Pre-flight — Validate Bridge V2 sample + model list

**Files:**
- Read: `dataset_registry.py`
- Read: `model_registry.py`

**Step 1: Verify Bridge V2 sample loads correctly**

```bash
cd /home/kana5123/ATLASVLA
/home/kana5123/miniconda3/envs/interp/bin/python -c "
from dataset_registry import load_sample
s = load_sample('bridge_v2', 0, 0)
print(f'Dataset: {s.dataset_name}')
print(f'Episode: {s.episode_id}, Step: {s.step_id}')
print(f'Instruction: {s.instruction}')
print(f'Image size: {s.image.size}')
print('OK: Bridge V2 sample loads correctly')
"
```

Expected: Prints dataset info, image size, and "OK" message.

**Step 2: List registered models and confirm 4 targets**

```bash
cd /home/kana5123/ATLASVLA
/home/kana5123/miniconda3/envs/interp/bin/python -c "
from model_registry import MODELS
for name, cfg in MODELS.items():
    extracted = name in ('openvla-7b', 'ecot-7b', 'tracevla-phi3v')
    status = 'DONE' if extracted else 'TODO'
    print(f'  [{status}] {name:20s} {cfg.hf_id:45s} ({cfg.architecture}, {cfg.num_layers}L)')
"
```

Expected: 7 models listed, 3 marked DONE, 4 marked TODO.

---

## Task 2: CogACT-Base extraction (GPU 5)

**Files:**
- Modify (if needed): `cross_model_extract.py`
- Modify (if needed): `model_registry.py`

**Step 1: Test CogACT model loading**

```bash
cd /home/kana5123/ATLASVLA
CUDA_VISIBLE_DEVICES=5 /home/kana5123/miniconda3/envs/interp/bin/python -c "
import os
os.environ['HF_HOME'] = '/ceph_data/kana5123/hf_cache'
os.environ['HF_HUB_CACHE'] = '/ceph_data/kana5123/hf_cache/hub'
from cross_model_extract import load_vla_model
from model_registry import get_model
cfg = get_model('cogact-base')
print(f'Loading {cfg.name}...')
processor, model = load_vla_model(cfg, 'cuda:0')
print(f'Model type: {type(model).__name__}')
print(f'Model config: {model.config.model_type}')
# Verify layers_path
from cross_model_extract import get_layers
layers = get_layers(model, cfg)
print(f'Num layers found: {len(layers)}')
print(f'Layer 0 type: {type(layers[0]).__name__}')
attn_mod = getattr(layers[0], cfg.attn_module, None)
print(f'Attn module: {type(attn_mod).__name__ if attn_mod else \"NOT FOUND\"}')
print('OK: CogACT loads correctly')
del model
import torch; torch.cuda.empty_cache()
"
```

Expected: Model loads, layers found, attention module accessible. If this fails, fix `model_registry.py` (update `layers_path`, `attn_module`, or add special loading in `cross_model_extract.py`).

**Step 2: Run full CogACT extraction**

```bash
cd /home/kana5123/ATLASVLA
export HF_HOME=/ceph_data/kana5123/hf_cache
export HF_HUB_CACHE=/ceph_data/kana5123/hf_cache/hub
CUDA_VISIBLE_DEVICES=5 /home/kana5123/miniconda3/envs/interp/bin/python cross_model_extract.py \
    --model cogact-base --dataset bridge_v2 --device cuda:0 \
    2>&1 | tee outputs/cross_model_analysis/cogact-base_bridge_v2.log
```

Expected: Creates `outputs/cross_model_analysis/cogact-base/bridge_v2/ep000_step000.json`, `ep000_step000_perhead.json`, and 14 heatmap PNGs.

**Step 3: Validate extraction output**

```bash
/home/kana5123/miniconda3/envs/interp/bin/python -c "
import json
with open('outputs/cross_model_analysis/cogact-base/bridge_v2/ep000_step000_perhead.json') as f:
    data = json.load(f)
print(f'Model: {data[\"model\"]}')
print(f'Actions: {list(data[\"perhead_analysis\"].keys())}')
first_action = list(data['perhead_analysis'].keys())[0]
layers = list(data['perhead_analysis'][first_action].keys())
print(f'Layers: {len(layers)} (first={layers[0]}, last={layers[-1]})')
first_layer = data['perhead_analysis'][first_action][layers[0]]
heads = list(first_layer.keys())
print(f'Heads: {len(heads)}')
# Check sink pattern
v0_values = []
for lk, heads_data in data['perhead_analysis'][first_action].items():
    for hk, stats in heads_data.items():
        v0_values.append(stats.get('vision_token0', 0))
import numpy as np
print(f'Mean vision[0]: {np.mean(v0_values):.4f}')
print(f'Max vision[0]:  {np.max(v0_values):.4f}')
print('OK: CogACT extraction validated')
"
```

Expected: Shows model info, layer/head counts, and vision[0] sink ratio.

**Step 4: Commit (if code changes were needed)**

```bash
cd /home/kana5123/ATLASVLA
git add cross_model_extract.py model_registry.py
git commit -m "fix: update CogACT model loading for cross-model extraction"
```

Only commit if modifications to loading code were required. If extraction ran without changes, skip this step.

---

## Task 3: SpatialVLA-4B extraction (GPU 6)

**Files:**
- Modify (if needed): `cross_model_extract.py`
- Modify (if needed): `model_registry.py`

**Step 1: Test SpatialVLA model loading**

```bash
cd /home/kana5123/ATLASVLA
CUDA_VISIBLE_DEVICES=6 /home/kana5123/miniconda3/envs/interp/bin/python -c "
import os
os.environ['HF_HOME'] = '/ceph_data/kana5123/hf_cache'
os.environ['HF_HUB_CACHE'] = '/ceph_data/kana5123/hf_cache/hub'
from cross_model_extract import load_vla_model
from model_registry import get_model
cfg = get_model('spatialvla-4b')
print(f'Loading {cfg.name} (architecture={cfg.architecture})...')
processor, model = load_vla_model(cfg, 'cuda:0')
print(f'Model type: {type(model).__name__}')
from cross_model_extract import get_layers
layers = get_layers(model, cfg)
print(f'Num layers: {len(layers)}')
attn_mod = getattr(layers[0], cfg.attn_module, None)
print(f'Attn module: {type(attn_mod).__name__ if attn_mod else \"NOT FOUND\"}')
print('OK: SpatialVLA loads correctly')
del model
import torch; torch.cuda.empty_cache()
"
```

Expected: Model loads with Gemma-2 architecture. Previous transformers 4.46.3 incompatibility should be resolved with 4.57.6. If it fails, check error message for missing classes or config issues.

**Step 2: Run full SpatialVLA extraction**

```bash
cd /home/kana5123/ATLASVLA
export HF_HOME=/ceph_data/kana5123/hf_cache
export HF_HUB_CACHE=/ceph_data/kana5123/hf_cache/hub
CUDA_VISIBLE_DEVICES=6 /home/kana5123/miniconda3/envs/interp/bin/python cross_model_extract.py \
    --model spatialvla-4b --dataset bridge_v2 --device cuda:0 \
    2>&1 | tee outputs/cross_model_analysis/spatialvla-4b_bridge_v2.log
```

Expected: Creates `outputs/cross_model_analysis/spatialvla-4b/bridge_v2/` with JSON + heatmaps.

**Step 3: Validate output**

```bash
/home/kana5123/miniconda3/envs/interp/bin/python -c "
import json, numpy as np
with open('outputs/cross_model_analysis/spatialvla-4b/bridge_v2/ep000_step000_perhead.json') as f:
    data = json.load(f)
first_action = list(data['perhead_analysis'].keys())[0]
v0_vals = []
for lk, heads in data['perhead_analysis'][first_action].items():
    for hk, stats in heads.items():
        v0_vals.append(stats.get('vision_token0', 0))
print(f'SpatialVLA: {len(data[\"perhead_analysis\"][first_action])} layers, mean vision[0]={np.mean(v0_vals):.4f}')
print('OK')
"
```

**Step 4: Commit if changes needed**

```bash
git add cross_model_extract.py model_registry.py
git commit -m "fix: update SpatialVLA loading for cross-model extraction"
```

---

## Task 4: SmolVLA-base extraction (GPU 6, after SpatialVLA)

**Files:**
- Modify (if needed): `cross_model_extract.py` (SmolVLA uses underlying VLM, special loading at line 53-63)

**Step 1: Test SmolVLA model loading**

The extraction code already has a special case for SmolVLA (lines 53-63): it loads the underlying VLM `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` instead of the LeRobot policy wrapper.

```bash
cd /home/kana5123/ATLASVLA
CUDA_VISIBLE_DEVICES=6 /home/kana5123/miniconda3/envs/interp/bin/python -c "
import os
os.environ['HF_HOME'] = '/ceph_data/kana5123/hf_cache'
os.environ['HF_HUB_CACHE'] = '/ceph_data/kana5123/hf_cache/hub'
from cross_model_extract import load_vla_model
from model_registry import get_model
cfg = get_model('smolvla-base')
print(f'Loading {cfg.name} (underlying VLM: SmolVLM2-500M-Video-Instruct)...')
processor, model = load_vla_model(cfg, 'cuda:0')
print(f'Model type: {type(model).__name__}')
from cross_model_extract import get_layers
try:
    layers = get_layers(model, cfg)
    print(f'Num layers: {len(layers)}')
    attn_mod = getattr(layers[0], cfg.attn_module, None)
    print(f'Attn module: {type(attn_mod).__name__ if attn_mod else \"NOT FOUND\"}')
except AttributeError as e:
    print(f'layers_path={cfg.layers_path} FAILED: {e}')
    # Try alternatives
    for path in ['model.text_model.model.layers', 'model.layers', 'language_model.model.layers']:
        try:
            obj = model
            for attr in path.split('.'):
                obj = getattr(obj, attr)
            print(f'  FOUND: {path} -> {len(obj)} layers')
        except AttributeError:
            print(f'  MISS:  {path}')
del model
import torch; torch.cuda.empty_cache()
"
```

Expected: Model loads, layers_path resolves. If `model.text_model.model.layers` fails, update `model_registry.py` with the correct path.

**Step 2: Run SmolVLA extraction**

```bash
cd /home/kana5123/ATLASVLA
export HF_HOME=/ceph_data/kana5123/hf_cache
export HF_HUB_CACHE=/ceph_data/kana5123/hf_cache/hub
CUDA_VISIBLE_DEVICES=6 /home/kana5123/miniconda3/envs/interp/bin/python cross_model_extract.py \
    --model smolvla-base --dataset bridge_v2 --device cuda:0 \
    2>&1 | tee outputs/cross_model_analysis/smolvla-base_bridge_v2.log
```

Expected: Creates `outputs/cross_model_analysis/smolvla-base/bridge_v2/` with JSON + heatmaps.

**Step 3: Validate output + commit if needed**

Same validation pattern as Tasks 2-3. Commit any model-specific fixes.

---

## Task 5: RoboFlamingo extraction (GPU 7)

**Files:**
- Modify (if needed): `cross_model_extract.py`
- Modify (if needed): `model_registry.py`

**Important:** RoboFlamingo uses **cross-attention** (OpenFlamingo architecture), not self-attention with concatenated vision+text tokens. The attention pattern is fundamentally different. The `layers_path="transformer.blocks"` and `attn_module="attn"` may not work with the standard extraction pipeline.

**Step 1: Test RoboFlamingo model loading**

```bash
cd /home/kana5123/ATLASVLA
CUDA_VISIBLE_DEVICES=7 /home/kana5123/miniconda3/envs/interp/bin/python -c "
import os
os.environ['HF_HOME'] = '/ceph_data/kana5123/hf_cache'
os.environ['HF_HUB_CACHE'] = '/ceph_data/kana5123/hf_cache/hub'
from model_registry import get_model
cfg = get_model('roboflamingo')
print(f'Loading {cfg.name} ({cfg.hf_id})...')
print(f'Architecture: {cfg.architecture}')
print(f'layers_path: {cfg.layers_path}')
print(f'attn_module: {cfg.attn_module}')
# Try loading
try:
    from cross_model_extract import load_vla_model
    processor, model = load_vla_model(cfg, 'cuda:0')
    print(f'Model type: {type(model).__name__}')
    print(f'Model modules: {[n for n, _ in model.named_children()]}')
except Exception as e:
    print(f'Loading FAILED: {e}')
    print('RoboFlamingo may need custom loading - check HuggingFace model card')
"
```

Expected: If loading fails (likely for a cross-attention architecture like OpenFlamingo), document the failure mode and skip this model for now. Cross-attention models have fundamentally different attention patterns — the sink concept may not apply.

**Step 2: Run extraction (if loading succeeded)**

```bash
cd /home/kana5123/ATLASVLA
export HF_HOME=/ceph_data/kana5123/hf_cache
export HF_HUB_CACHE=/ceph_data/kana5123/hf_cache/hub
CUDA_VISIBLE_DEVICES=7 /home/kana5123/miniconda3/envs/interp/bin/python cross_model_extract.py \
    --model roboflamingo --dataset bridge_v2 --device cuda:0 \
    2>&1 | tee outputs/cross_model_analysis/roboflamingo_bridge_v2.log
```

**Step 3: If RoboFlamingo fails, document and skip**

If loading or extraction fails due to cross-attention architecture incompatibility:

1. Log the error in `outputs/cross_model_analysis/roboflamingo_bridge_v2.log`
2. Add a comment to `model_registry.py` noting the incompatibility:
```python
# ── RoboFlamingo (MPT-1B or 3B backbone) ──
# NOTE: Cross-attention architecture. Standard extraction pipeline incompatible.
# Requires custom handling for gated cross-attention layers (OpenFlamingo-style).
# Skipped in initial cross-model analysis.
```
3. Proceed with 6 models instead of 7

**Step 4: Commit**

```bash
git add model_registry.py cross_model_extract.py
git commit -m "feat: extract CogACT, SpatialVLA, SmolVLA sink patterns (+ RoboFlamingo note)"
```

---

## Task 6: Update cross-model comparison (all extracted models)

**Files:**
- Run: `cross_model_compare.py`
- Output: `outputs/cross_model_analysis/comparison/`

**Step 1: Run comparison script**

```bash
cd /home/kana5123/ATLASVLA
/home/kana5123/miniconda3/envs/interp/bin/python cross_model_compare.py
```

Expected output:
```
Loading per-head data from: outputs/cross_model_analysis
  Loaded openvla-7b from ep000_step000_perhead.json
  Loaded ecot-7b from ep000_step000_perhead.json
  Loaded tracevla-phi3v from ep000_step000_perhead.json
  Loaded cogact-base from ep000_step000_perhead.json
  Loaded spatialvla-4b from ep000_step000_perhead.json
  Loaded smolvla-base from ep000_step000_perhead.json

Found 6 models: [...]
```

**Step 2: Verify comparison outputs**

```bash
ls -la outputs/cross_model_analysis/comparison/
# Should contain:
# cross_model_sink_comparison.png
# cross_model_heatmap.png
# cross_model_dual_sink.png
# cross_model_summary.json
# cross_model_table.tex
```

**Step 3: Print summary table**

```bash
/home/kana5123/miniconda3/envs/interp/bin/python -c "
import json
with open('outputs/cross_model_analysis/comparison/cross_model_summary.json') as f:
    data = json.load(f)
print(f'{'Model':<20} {'Vision[0]':>10} {'Text':>10} {'Useful':>10} {'Early Sink':>10}')
print('-' * 62)
for model, summary in data.items():
    print(f'{model:<20} {summary[\"mean_vision0\"]:>10.4f} {summary[\"mean_text_total\"]:>10.4f} '
          f'{summary[\"mean_vision_other\"]:>10.4f} {summary[\"mean_early_sink\"]:>10.4f}')
"
```

**Step 4: Commit results**

```bash
cd /home/kana5123/ATLASVLA
git add outputs/cross_model_analysis/comparison/
git commit -m "data: update cross-model comparison with 6-7 VLA models"
```

---

## Task 7: Validate dynamic detect_sinks() across all models

**Files:**
- Read: `attention_v3.py:41-70` (detect_sinks function)
- Output: `outputs/cross_model_analysis/comparison/detected_sinks.json`

**Step 1: Run detect_sinks on each model's extracted attention data**

```bash
cd /home/kana5123/ATLASVLA
/home/kana5123/miniconda3/envs/interp/bin/python -c "
import json, torch
import numpy as np
from pathlib import Path
from attention_v3 import detect_sinks

results = {}
base_dir = Path('outputs/cross_model_analysis')

for model_dir in sorted(base_dir.iterdir()):
    if not model_dir.is_dir() or model_dir.name == 'comparison':
        continue
    model_name = model_dir.name
    perhead_files = sorted(model_dir.glob('**/*_perhead.json'))
    if not perhead_files:
        continue

    with open(perhead_files[0]) as f:
        data = json.load(f)

    boundaries = data.get('token_boundaries', {})
    perhead = data['perhead_analysis']
    first_action = list(perhead.keys())[0]

    # Reconstruct attention-like tensor from perhead stats
    # Use vision_token0 ratio as proxy for full attention distribution
    layer_keys = sorted(perhead[first_action].keys())
    head_keys = sorted(perhead[first_action][layer_keys[0]].keys())
    n_layers = len(layer_keys)
    n_heads = len(head_keys)

    # Report per-layer sink detection
    print(f'\n{model_name} ({n_layers}L x {n_heads}H):')
    total_v0 = []
    total_early = []
    for lk in layer_keys:
        v0_vals = [perhead[first_action][lk][hk].get('vision_token0', 0) for hk in head_keys]
        early_vals = [perhead[first_action][lk][hk].get('early_sink', 0) for hk in head_keys]
        total_v0.extend(v0_vals)
        total_early.extend(early_vals)

    mean_v0 = np.mean(total_v0)
    mean_early = np.mean(total_early) if total_early else 0

    # Estimate whether detect_sinks would find v0 as sink
    # threshold = alpha / N where N = total seq len
    N = boundaries.get('total_seq_len', 270)
    threshold = 5.0 / N
    v0_is_sink = mean_v0 > threshold
    early_is_sink = mean_early > threshold

    results[model_name] = {
        'n_layers': n_layers,
        'n_heads': n_heads,
        'mean_vision0': float(mean_v0),
        'mean_early_sink': float(mean_early),
        'threshold_alpha5': float(threshold),
        'v0_detected_as_sink': v0_is_sink,
        'early_detected_as_sink': early_is_sink,
        'seq_len': N,
    }

    sink_label = 'SINK' if v0_is_sink else 'not sink'
    early_label = 'SINK' if early_is_sink else 'not sink'
    print(f'  vision[0]: {mean_v0:.4f} (threshold={threshold:.4f}) -> {sink_label}')
    print(f'  early_sink: {mean_early:.4f} (threshold={threshold:.4f}) -> {early_label}')

# Save results
out_path = Path('outputs/cross_model_analysis/comparison/detected_sinks.json')
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved: {out_path}')
"
```

Expected: Each model shows whether vision[0] exceeds the α/N threshold. Should match the known patterns:
- OpenVLA, ECoT: vision[0] IS sink
- TraceVLA: vision[0] is NOT sink (but early tokens might be)
- CogACT, SpatialVLA, SmolVLA: to be discovered

**Step 2: Commit**

```bash
git add outputs/cross_model_analysis/comparison/detected_sinks.json
git commit -m "data: add per-model dynamic sink detection results (detect_sinks α=5.0)"
```

---

## Task 8: Prepare adapter training with dynamic detection

**Files:**
- Read: `config.py` (verify DYNAMIC_SINK_DETECTION=True, SINK_ALPHA=5.0)
- Read: `adapter_train.py:527-550` (V3Context creation with dynamic detection)
- Read: `run_adapter_experiment.py` (experiment orchestrator)

**Step 1: Verify dynamic detection is configured**

```bash
cd /home/kana5123/ATLASVLA
/home/kana5123/miniconda3/envs/interp/bin/python -c "
import config
print(f'DYNAMIC_SINK_DETECTION: {config.DYNAMIC_SINK_DETECTION}')
print(f'SINK_ALPHA: {config.SINK_ALPHA}')
print(f'VAR_SINK_INDICES: {config.VAR_SINK_INDICES}')
print()
if config.DYNAMIC_SINK_DETECTION:
    print('Dynamic detection is ON')
    print('Adapter training will use detect_sinks(alpha={}) per forward pass'.format(config.SINK_ALPHA))
else:
    print('Dynamic detection is OFF')
    print('Will use hardcoded sinks: {}'.format(config.VAR_SINK_INDICES))
"
```

Expected: `DYNAMIC_SINK_DETECTION: True`, `SINK_ALPHA: 5.0`

**Step 2: Check current adapter training status**

```bash
# Check if GPUs 1-4 are still busy with adapter v1
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | head -5
# Check training process
ps aux | grep adapter_train | grep -v grep | head -5
```

If GPUs 1-4 are free, proceed to Step 3. If still training, wait and re-check later.

**Step 3: Run adapter experiment with dynamic detection**

When GPUs 1-4 are free:

```bash
cd /home/kana5123/ATLASVLA
export HF_HOME=/ceph_data/kana5123/hf_cache
CUDA_VISIBLE_DEVICES=1,2,3,4 /home/kana5123/miniconda3/envs/interp/bin/python \
    run_adapter_experiment.py --gpus 0,1,2,3 --num_eval_episodes 200
```

This runs 4 configs (base, v1, v2-prop, v2-full) with `DYNAMIC_SINK_DETECTION=True`.

**Step 4: Compare results**

```bash
/home/kana5123/miniconda3/envs/interp/bin/python compare_adapter_results.py
cat outputs/experiment_results/comparison_summary.json
```

**Step 5: Commit results**

```bash
git add outputs/experiment_results/
git commit -m "data: adapter experiment results with dynamic sink detection"
```

---

## Execution Order & Parallelization

```
Phase 1 (immediate, parallel on GPUs 5-7):
  Task 1: Pre-flight validation (CPU, 1 min)
  ├── Task 2: CogACT on GPU 5      (~15 min)
  ├── Task 3: SpatialVLA on GPU 6   (~10 min)
  │   └── Task 4: SmolVLA on GPU 6  (~5 min, after SpatialVLA)
  └── Task 5: RoboFlamingo on GPU 7 (~10 min)

Phase 2 (after Phase 1 completes, CPU):
  Task 6: Cross-model comparison    (~2 min)
  Task 7: detect_sinks validation   (~1 min)

Phase 3 (after GPUs 1-4 free):
  Task 8: Adapter training + eval   (~8-12 hours)
```

**Tasks 2, 3+4, 5 can run in parallel** on GPUs 5, 6, 7 respectively. Launch all three with background processes.

---

## Success Criteria

1. **4 new models extracted**: Each has `ep000_step000_perhead.json` + heatmap PNGs
2. **Cross-model comparison updated**: Bar chart, heatmap, LaTeX table include all extracted models
3. **detect_sinks validated**: `detected_sinks.json` shows correct sink/non-sink classification for each model
4. **Adapter experiment**: MSE results for 4 configs with dynamic detection (Task 8, may be deferred)
