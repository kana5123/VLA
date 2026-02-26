# Cross-Architecture Bottleneck Analysis — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Verify whether the vision token 0 information bottleneck (99%+ contribution from Layer 2) discovered in OpenVLA is universal across VLA/VLM architectures, or specific to certain model families.

**Architecture:** Extend the existing `verify_attention_sinks.py` pipeline to support cross-model analysis. Fix the hardcoded `detect_token_boundaries()` to use per-model prompt templates. Add VLM models (LLaVA-1.5, InternVL2) to the registry. For diffusion VLAs (π0, Dita), implement custom attention extraction since they use fundamentally different architectures.

**Tech Stack:** Python 3.10, PyTorch 2.5.1, transformers 4.57.6, conda env `interp`, 8× H100 80GB GPUs

---

## Pre-Requisites

- Existing bridge data cache at `/ceph_data/kana5123/bridge_data_cache/` (memmap images + metadata)
- OpenVLA-7B sink verification already complete at `outputs/sink_verification/openvla-7b/`
- `model_registry.py` has TraceVLA and SpatialVLA entries (experiment_ready=True)

---

### Task 0: Fix `detect_token_boundaries()` for Cross-Model Use

The function at `extract_attention.py:38-107` hardcodes `config.PROMPT_TEMPLATE` instead of using the per-model `model_cfg.prompt_template`. This breaks for TraceVLA (Phi3V format) and SpatialVLA.

**Files:**
- Modify: `extract_attention.py:48` (the PROMPT_TEMPLATE reference)

**Step 1: Fix the hardcoded prompt template**

In `extract_attention.py:38`, the function signature is:
```python
def detect_token_boundaries(processor, model, sample_image, sample_instruction, device):
```

Add `model_cfg=None` parameter and use its prompt_template:

```python
def detect_token_boundaries(processor, model, sample_image, sample_instruction, device, model_cfg=None):
```

Change line 48 from:
```python
prompt = config.PROMPT_TEMPLATE.format(instruction=sample_instruction)
```
to:
```python
if model_cfg is not None:
    prompt = model_cfg.prompt_template.format(instruction=sample_instruction)
else:
    prompt = config.PROMPT_TEMPLATE.format(instruction=sample_instruction)
```

**Step 2: Update callers in `verify_attention_sinks.py`**

At line 720-721, update the call:
```python
# Before:
boundaries = detect_token_boundaries(processor, model, image, instruction, device)
# After:
boundaries = detect_token_boundaries(processor, model, image, instruction, device, model_cfg=model_cfg)
```

**Step 3: Update callers in `visualize_text_attention.py`**

At line 155, update:
```python
# Before:
boundaries = detect_token_boundaries(processor, model, image, instruction, device)
# After:
boundaries = detect_token_boundaries(processor, model, image, instruction, device, model_cfg=model_cfg)
```

**Step 4: Verify no other callers use the old signature**

Run: `grep -rn "detect_token_boundaries" /home/kana5123/ATLASVLA/*.py`

Ensure all callers still work (the default `model_cfg=None` preserves backward compat).

**Step 5: Commit**

```bash
git add extract_attention.py verify_attention_sinks.py visualize_text_attention.py
git commit -m "fix: use per-model prompt_template in detect_token_boundaries"
```

---

### Task 1: Fix `verify_attention_sinks.py` for Architecture-Specific W_V/W_O Extraction

The `get_wov_matrix()` function at line 121-144 assumes `v_proj` and `o_proj` attributes, which is LLaMA-specific. Phi3V uses `qkv_proj` (fused QKV) and Gemma2 may use different naming.

**Files:**
- Modify: `verify_attention_sinks.py:121-144`

**Step 1: Inspect Phi3V attention module structure**

Run a quick check (no model loading needed):
```bash
conda run -n interp python -c "
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('furonghuang-lab/tracevla_phi3v', trust_remote_code=True)
print(cfg.model_type)
"
```

Then look at the Phi3V modeling code:
```bash
conda run -n interp python -c "
from transformers import AutoModelForCausalLM
import inspect
# Check module source for attention class
m = AutoModelForCausalLM.from_pretrained('furonghuang-lab/tracevla_phi3v', trust_remote_code=True, device_map='cpu', torch_dtype='auto')
layer = m.model.layers[0]
attn = layer.self_attn
print(dir(attn))
print([n for n, _ in attn.named_modules()])
print([n for n, _ in attn.named_parameters()])
del m
"
```

**Step 2: Update `get_wov_matrix()` for multi-architecture support**

Replace the function body with architecture-aware extraction:

```python
def get_wov_matrix(model, model_cfg, layer_idx):
    """Extract W_V and W_O for a given layer (architecture-aware)."""
    layers = get_layers(model, model_cfg)
    layer = layers[layer_idx]
    attn = layer.self_attn

    if model_cfg.architecture == "phi3_v":
        # Phi3V uses fused qkv_proj: (3 * hidden_dim, hidden_dim)
        qkv_weight = attn.qkv_proj.weight.detach().cpu().float()
        hidden_dim = model_cfg.hidden_dim
        # Split: [Q, K, V] each of size (hidden_dim, hidden_dim)
        v_weight = qkv_weight[2 * hidden_dim:3 * hidden_dim, :]  # (hidden_dim, hidden_dim)
        o_weight = attn.o_proj.weight.detach().cpu().float()
    elif model_cfg.architecture == "gemma2":
        # Gemma2 uses separate q/k/v projections but may have different dim names
        v_weight = attn.v_proj.weight.detach().cpu().float()
        o_weight = attn.o_proj.weight.detach().cpu().float()
    else:
        # Default LLaMA-style
        v_weight = attn.v_proj.weight.detach().cpu().float()
        o_weight = attn.o_proj.weight.detach().cpu().float()

    return v_weight, o_weight
```

NOTE: The actual Phi3V QKV split may differ — the Step 1 inspection will reveal the exact layout. Adjust the slicing accordingly. Key attributes to check: `qkv_proj`, `q_proj`, `k_proj`, `v_proj`, `o_proj`.

**Step 3: Verify the fix with a minimal test**

```bash
cd /home/kana5123/ATLASVLA
conda run -n interp python -c "
from verify_attention_sinks import get_wov_matrix
from extract_attention import load_model_from_registry
proc, model, cfg = load_model_from_registry('openvla-7b', device='cuda:0')
v, o = get_wov_matrix(model, cfg, 31)
print(f'OpenVLA L31: V={v.shape}, O={o.shape}')
del model; import torch; torch.cuda.empty_cache()
"
```

**Step 4: Commit**

```bash
git add verify_attention_sinks.py
git commit -m "feat: multi-architecture W_V/W_O extraction in get_wov_matrix"
```

---

### Task 2: Add `--all-layers` Flag and All-Layer Contribution Profile

Currently `check_condition_C` only analyzes the last 8 layers. For the bottleneck analysis, we need ALL layers to find the "onset layer" where contribution exceeds 50%.

**Files:**
- Modify: `verify_attention_sinks.py`

**Step 1: Add `--all-layers` CLI flag**

In `main()` at line 943:
```python
parser.add_argument("--all-layers", action="store_true",
                    help="Run condition C on ALL layers (not just last 8)")
```

**Step 2: Pass `all_layers` to `run_verification()`**

Update signature and forward the flag:
```python
def run_verification(model_name, device, n_samples=5, tau=20.0, output_dir=None, all_layers=False):
```

When calling `check_condition_C`, if `all_layers=True`:
```python
target_layers_c = list(range(model_cfg.num_layers)) if all_layers else None
cond_c = check_condition_C(
    model, model_cfg,
    hook_mgr.attention_weights, hook_mgr.hidden_states,
    boundaries,
    target_layers=target_layers_c,
)
```

**Step 3: Compute bottleneck onset layer and severity metrics**

After aggregation, add to the report:
```python
# Bottleneck metrics
contribution_ratios = {}
for lk in sorted(agg_cond_c.keys()):
    data = agg_cond_c[lk]
    v_norms = np.array(data["value_norms"])
    alpha = np.array(data["alpha_mean"])
    weighted = alpha * v_norms
    total = weighted.sum()
    token0_ratio = (weighted[0] / (total + 1e-8)) * 100
    contribution_ratios[lk] = token0_ratio

# Onset: first layer where token 0 contribution > 50%
onset_layer = None
for lk in sorted(contribution_ratios.keys()):
    if contribution_ratios[lk] > 50.0:
        onset_layer = int(lk.split("_")[1])
        break

# Severity: mean contribution from layer 2 onwards
severity_layers = {lk: v for lk, v in contribution_ratios.items()
                   if int(lk.split("_")[1]) >= 2}
severity = np.mean(list(severity_layers.values())) if severity_layers else 0.0

report["bottleneck_metrics"] = {
    "contribution_ratios": contribution_ratios,
    "onset_layer": onset_layer,
    "severity_percent": float(severity),
}
```

**Step 4: Add all-layer contribution profile visualization**

```python
def plot_contribution_profile(contribution_ratios, output_path, model_name):
    """Bar chart of token 0 contribution ratio across all layers."""
    layers = sorted(contribution_ratios.keys())
    ratios = [contribution_ratios[lk] for lk in layers]
    layer_nums = [int(lk.split("_")[1]) for lk in layers]

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ['red' if r > 50 else 'steelblue' for r in ratios]
    ax.bar(layer_nums, ratios, color=colors, alpha=0.8)
    ax.axhline(y=50, color='green', linestyle='--', linewidth=1.5, label='50% threshold')
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Token 0 Contribution (%)", fontsize=11)
    ax.set_title(f"All-Layer Token 0 Contribution Profile — {model_name}", fontsize=12)
    ax.legend()
    ax.set_ylim(0, 105)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
```

Call it after generating other plots:
```python
if "bottleneck_metrics" in report:
    plot_contribution_profile(
        report["bottleneck_metrics"]["contribution_ratios"],
        output_dir / "contribution_profile.png",
        model_name,
    )
```

**Step 5: Commit**

```bash
git add verify_attention_sinks.py
git commit -m "feat: add --all-layers flag and contribution profile visualization"
```

---

### Task 3: Run Phase 1 — TraceVLA-Phi3V Sink Verification

**Files:** No code changes — execution only.

**Step 1: Run TraceVLA verification**

```bash
cd /home/kana5123/ATLASVLA
conda run -n interp python verify_attention_sinks.py \
    --model tracevla-phi3v \
    --device cuda:1 \
    --n_samples 5 \
    --all-layers
```

Expected: ~10-15 min on H100, outputs to `outputs/sink_verification/tracevla-phi3v/`

**Step 2: Check output exists**

```bash
ls -la outputs/sink_verification/tracevla-phi3v/
cat outputs/sink_verification/tracevla-phi3v/sink_report.json | python -m json.tool | head -40
```

Verify:
- `sink_report.json` contains verdict section
- All 4 PNG visualizations generated
- `contribution_profile.png` generated (from Task 2)
- `bottleneck_metrics.onset_layer` and `severity_percent` are populated

**Step 3: Note findings**

Record: Does TraceVLA show the same pattern as OpenVLA?
- Condition A: PASS/FAIL?
- Condition B: PASS/FAIL?
- Condition C: Token 0 value norm ratio?
- Onset layer?
- Severity?

NOTE: TraceVLA uses Phi3V with `[text] [vision] [text]` layout (BOS before vision). The "sink" token may be the BOS token rather than vision token 0. If `detect_token_boundaries` reports vision_start > 0, the analysis needs to check BOTH vision[0] AND position 0 (BOS).

**Step 4: Commit results**

```bash
git add outputs/sink_verification/tracevla-phi3v/
git commit -m "data: TraceVLA-Phi3V sink verification results"
```

---

### Task 4: Run Phase 1 — SpatialVLA-4B Sink Verification

**Step 1: Run SpatialVLA verification**

```bash
cd /home/kana5123/ATLASVLA
conda run -n interp python verify_attention_sinks.py \
    --model spatialvla-4b \
    --device cuda:2 \
    --n_samples 5 \
    --all-layers
```

Expected: ~5-8 min (smaller 4B model)

**Step 2: Check output and note findings**

Same checks as Task 3. SpatialVLA uses Gemma-2 backbone with SigLIP vision encoder.

Expected differences:
- 26 layers (vs 32 in LLaMA models)
- 8 heads (vs 32)
- Different tokenizer (Gemma sentencepiece)
- May require `intrinsic` parameter in forward pass (already handled in verify_attention_sinks.py line 730-734)

**Step 3: Commit results**

```bash
git add outputs/sink_verification/spatialvla-4b/
git commit -m "data: SpatialVLA-4B sink verification results"
```

---

### Task 5: Phase 1 Comparison — Autoregressive VLA Summary

Produce a side-by-side comparison of OpenVLA, TraceVLA, and SpatialVLA.

**Files:**
- Create: `compare_sink_results.py`

**Step 1: Write comparison script**

```python
"""Compare sink verification results across models."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config


def load_report(model_name):
    path = config.OUTPUT_DIR / "sink_verification" / model_name / "sink_report.json"
    with open(path) as f:
        return json.load(f)


def compare_models(model_names):
    reports = {name: load_report(name) for name in model_names}
    output_dir = config.OUTPUT_DIR / "sink_verification" / "cross_model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Table
    print(f"\n{'Model':<20} {'Cond A':<10} {'Cond B':<10} {'Cond C':<10} "
          f"{'Sink?':<8} {'Aggregator?':<12} {'Onset':<8} {'Severity':<10}")
    print("-" * 88)

    comparison = {}
    for name, report in reports.items():
        v = report["verdict"]
        bm = report.get("bottleneck_metrics", {})
        ca = "PASS" if v["condition_A"]["pass"] else "FAIL"
        cb = "PASS" if v["condition_B"]["pass"] else "FAIL"
        cc = "PASS" if v["condition_C"]["pass"] else "FAIL"
        sink = "YES" if v["is_true_sink"] else "NO"
        agg = "YES" if v.get("is_context_aggregator") else "NO"
        onset = str(bm.get("onset_layer", "N/A"))
        severity = f"{bm.get('severity_percent', 0):.1f}%"

        print(f"{name:<20} {ca:<10} {cb:<10} {cc:<10} {sink:<8} {agg:<12} {onset:<8} {severity:<10}")

        comparison[name] = {
            "condition_A": ca, "condition_B": cb, "condition_C": cc,
            "is_true_sink": v["is_true_sink"],
            "is_context_aggregator": v.get("is_context_aggregator", False),
            "onset_layer": bm.get("onset_layer"),
            "severity_percent": bm.get("severity_percent", 0),
        }

    # Save comparison JSON
    with open(output_dir / "comparison_table.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # Contribution profile overlay plot
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf"]
    for i, (name, report) in enumerate(reports.items()):
        bm = report.get("bottleneck_metrics", {})
        ratios = bm.get("contribution_ratios", {})
        if ratios:
            layers_sorted = sorted(ratios.keys())
            x = [int(lk.split("_")[1]) for lk in layers_sorted]
            y = [ratios[lk] for lk in layers_sorted]
            ax.plot(x, y, marker="o", markersize=3, label=name,
                    color=colors[i % len(colors)], linewidth=2)

    ax.axhline(y=50, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="50% threshold")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Token 0 Contribution (%)", fontsize=12)
    ax.set_title("Cross-Model Contribution Profile Comparison", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    fig.savefig(output_dir / "contribution_profiles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {output_dir / 'contribution_profiles.png'}")
    print(f"Saved: {output_dir / 'comparison_table.json'}")


if __name__ == "__main__":
    models = sys.argv[1:] if len(sys.argv) > 1 else ["openvla-7b", "tracevla-phi3v", "spatialvla-4b"]
    compare_models(models)
```

**Step 2: Run comparison**

```bash
cd /home/kana5123/ATLASVLA
conda run -n interp python compare_sink_results.py openvla-7b tracevla-phi3v spatialvla-4b
```

**Step 3: Commit**

```bash
git add compare_sink_results.py outputs/sink_verification/cross_model_comparison/
git commit -m "feat: cross-model sink comparison script + Phase 1 results"
```

---

### Task 6: Add LLaVA-1.5-7B to Model Registry (Phase 2)

LLaVA-1.5-7B shares the same LLaMA-2 backbone as OpenVLA. If the bottleneck appears in OpenVLA but NOT in LLaVA, it proves that VLA fine-tuning (not the backbone) causes the bottleneck.

**Files:**
- Modify: `model_registry.py`

**Step 1: Research LLaVA-1.5-7B architecture**

```bash
conda run -n interp python -c "
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('llava-hf/llava-1.5-7b-hf')
print(f'model_type: {cfg.model_type}')
print(f'text_config: {cfg.text_config.model_type if hasattr(cfg, \"text_config\") else \"N/A\"}')
print(f'hidden_size: {cfg.text_config.hidden_size}')
print(f'num_layers: {cfg.text_config.num_hidden_layers}')
print(f'num_heads: {cfg.text_config.num_attention_heads}')
"
```

**Step 2: Add LLaVA-1.5-7B to registry**

Add to `model_registry.py` after the existing VLA entries:

```python
# ── LLaVA-1.5-7B (VLM — VQA Only, same backbone as OpenVLA) ──
register(VLAModelConfig(
    name="llava-1.5-7b",
    hf_id="llava-hf/llava-1.5-7b-hf",
    architecture="llama",
    vision_encoder="clip-vit-l",
    num_layers=32,
    num_heads=32,
    hidden_dim=4096,
    vision_grid_size=24,        # CLIP ViT-L/14 @ 336px → 24x24 = 576 tokens
    num_vision_tokens=576,
    action_tokens=0,            # VLM — no action tokens
    action_type="none",
    prompt_template="USER: <image>\n{instruction}\nASSISTANT:",
    native_datasets=[],
    notes="VLM (VQA only), same LLaMA-2-7B backbone as OpenVLA, CLIP ViT-L/14 vision",
    layers_path="language_model.model.layers",
    auto_model_class="AutoModelForVision2Seq",
))
```

NOTE: Verify the exact prompt template and vision token count by checking the HuggingFace model card. The `vision_grid_size=24` (576 tokens) is for the 336px variant; the 224px variant uses 16x16=256.

**Step 3: Test loading**

```bash
cd /home/kana5123/ATLASVLA
conda run -n interp python -c "
from model_registry import get_model
cfg = get_model('llava-1.5-7b')
print(f'OK: {cfg.name}, {cfg.hf_id}, {cfg.num_layers}L, {cfg.num_vision_tokens}V')
"
```

**Step 4: Commit**

```bash
git add model_registry.py
git commit -m "feat: add LLaVA-1.5-7B to model registry"
```

---

### Task 7: Add InternVL2-8B to Model Registry (Phase 2)

**Files:**
- Modify: `model_registry.py`

**Step 1: Research InternVL2-8B architecture**

```bash
conda run -n interp python -c "
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('OpenGVLab/InternVL2-8B', trust_remote_code=True)
print(type(cfg).__name__)
print(dir(cfg))
# Check LLM backbone
if hasattr(cfg, 'llm_config'):
    llm = cfg.llm_config
    print(f'LLM type: {llm.model_type}')
    print(f'hidden: {llm.hidden_size}')
    print(f'layers: {llm.num_hidden_layers}')
    print(f'heads: {llm.num_attention_heads}')
"
```

**Step 2: Add InternVL2-8B to registry**

```python
# ── InternVL2-8B (VLM — VQA Only, InternLM2 backbone) ──
register(VLAModelConfig(
    name="internvl2-8b",
    hf_id="OpenGVLab/InternVL2-8B",
    architecture="internlm2",
    vision_encoder="internvit-6b",
    num_layers=32,              # InternLM2 has 32 layers
    num_heads=32,
    hidden_dim=4096,
    vision_grid_size=16,
    num_vision_tokens=256,      # Verify via inspection
    action_tokens=0,
    action_type="none",
    prompt_template="<|im_start|>user\n<image>\n{instruction}<|im_end|>\n<|im_start|>assistant\n",
    native_datasets=[],
    notes="VLM (VQA only), InternLM2 backbone, InternViT-6B vision encoder",
    layers_path="language_model.model.layers",
    auto_model_class="AutoModel",
    trust_remote_code=True,
))
```

NOTE: InternVL2 uses `trust_remote_code=True` and may need special loading. Verify layers_path by inspecting the model structure.

**Step 3: Test loading (config only, don't load weights)**

```bash
conda run -n interp python -c "
from model_registry import get_model
cfg = get_model('internvl2-8b')
print(f'OK: {cfg.name}, {cfg.hf_id}')
"
```

**Step 4: Commit**

```bash
git add model_registry.py
git commit -m "feat: add InternVL2-8B to model registry"
```

---

### Task 8: Adapt `verify_attention_sinks.py` for VLM Models

VLMs don't generate action tokens — they generate text. The verification script needs to handle this gracefully.

**Files:**
- Modify: `verify_attention_sinks.py`

**Step 1: Handle `action_type="none"` in data loading**

The current script calls `load_samples_from_cache()` which loads bridge dataset images+instructions. For VLMs doing VQA, we can reuse the same images but the instruction interpretation changes.

VLMs don't predict actions — they answer questions. The "query token" for condition C should be the last text token before generation, not an action token.

No code change needed here — `check_condition_C` already uses `te - 1` (last text token) as the query position. This works for both VLA and VLM.

**Step 2: Handle VLM-specific model loading**

LLaVA and InternVL2 may need different loading patterns. Add to `load_model_from_registry()` in `extract_attention.py` if needed:

For LLaVA-1.5-7B (using `llava-hf` HuggingFace version):
- AutoModelForVision2Seq should work
- Processor: `LlavaProcessor` (auto-detected)

For InternVL2-8B:
- May need custom loading via `trust_remote_code=True`
- Check if `get_layers()` navigates correctly

**Step 3: Handle `get_wov_matrix()` for InternLM2**

InternLM2 likely uses standard `v_proj` / `o_proj` (like LLaMA). Verify:
```bash
conda run -n interp python -c "
from transformers import AutoModel
m = AutoModel.from_pretrained('OpenGVLab/InternVL2-8B', trust_remote_code=True, device_map='cpu', torch_dtype='auto')
layer = None
# Navigate to layers
for name, mod in m.named_modules():
    if 'layers.0.self_attn' in name and hasattr(mod, 'v_proj'):
        print(f'Found v_proj at: {name}')
        print(f'  v_proj shape: {mod.v_proj.weight.shape}')
        print(f'  o_proj shape: {mod.o_proj.weight.shape}')
        break
del m
"
```

If InternLM2 uses `wqkv` (fused), add a branch in `get_wov_matrix()` similar to Phi3V.

**Step 4: Add VLM-specific instruction for the bridge images**

VLMs don't understand "What action should the robot take to X?" — rewrite instructions:

```python
# In verify_attention_sinks.py, run_verification():
if model_cfg.action_type == "none":
    # VLM: rephrase as VQA question
    instruction = f"Describe what you see and what the robot should do to {sample['instruction']}"
```

Or simpler: just use the original instruction. The attention pattern analysis doesn't depend on output quality — we're analyzing the prompt processing, not the response.

**Step 5: Commit**

```bash
git add verify_attention_sinks.py extract_attention.py
git commit -m "feat: adapt sink verification for VLM models (action_type=none)"
```

---

### Task 9: Run Phase 2 — LLaVA-1.5-7B Verification

**Step 1: Test model loading + boundary detection**

```bash
cd /home/kana5123/ATLASVLA
conda run -n interp python -c "
from extract_attention import load_model_from_registry, detect_token_boundaries
from model_registry import get_model
from PIL import Image
import numpy as np

# Load model
proc, model, cfg = load_model_from_registry('llava-1.5-7b', device='cuda:3')

# Test boundary detection with a dummy image
img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
boundaries = detect_token_boundaries(proc, model, img, 'pick up the cup', 'cuda:3', model_cfg=cfg)
print(f'Boundaries: {boundaries}')

del model
import torch; torch.cuda.empty_cache()
"
```

**Step 2: Run full verification**

```bash
conda run -n interp python verify_attention_sinks.py \
    --model llava-1.5-7b \
    --device cuda:3 \
    --n_samples 5 \
    --all-layers
```

**Step 3: Check results and record findings**

Key question: Does LLaVA-1.5 (same LLaMA-2 backbone as OpenVLA) show the same token 0 bottleneck?

If YES → bottleneck is in the backbone, not VLA-specific
If NO → VLA fine-tuning introduces the bottleneck

**Step 4: Commit**

```bash
git add outputs/sink_verification/llava-1.5-7b/
git commit -m "data: LLaVA-1.5-7B sink verification results (Phase 2)"
```

---

### Task 10: Run Phase 2 — InternVL2-8B Verification

**Step 1: Test model loading + boundary detection**

Same pattern as Task 9 but with `internvl2-8b`.

**Step 2: Run full verification**

```bash
conda run -n interp python verify_attention_sinks.py \
    --model internvl2-8b \
    --device cuda:4 \
    --n_samples 5 \
    --all-layers
```

**Step 3: Record findings**

InternVL2 uses InternLM2 (not LLaMA-2). This tests whether the bottleneck is backbone-specific.

**Step 4: Commit**

```bash
git add outputs/sink_verification/internvl2-8b/
git commit -m "data: InternVL2-8B sink verification results (Phase 2)"
```

---

### Task 11: Phase 2 Comparison — VLA vs VLM

**Step 1: Run comparison across all 5 models**

```bash
cd /home/kana5123/ATLASVLA
conda run -n interp python compare_sink_results.py \
    openvla-7b tracevla-phi3v spatialvla-4b llava-1.5-7b internvl2-8b
```

**Step 2: Analyze the comparison table**

Key questions:
1. VLA vs VLM (same backbone): OpenVLA vs LLaVA-1.5 → isolates VLA fine-tuning effect
2. Different backbones: LLaMA-2 vs InternLM2 → backbone-specific?
3. Different vision encoders: Prismatic vs CLIP vs InternViT → vision encoder effect?

**Step 3: Commit**

```bash
git add outputs/sink_verification/cross_model_comparison/
git commit -m "data: Phase 2 cross-model comparison (VLA vs VLM)"
```

---

### Task 12: Phase 3 Research — π0 Architecture Analysis

π0 uses PaliGemma 3B VLM + a separate "Flow Expert" for action generation via flow matching. The attention pattern analysis is different because:
1. The VLM part (PaliGemma) processes vision+text
2. The Flow Expert processes noised actions conditioned on VLM features
3. Cross-attention between Flow Expert and VLM features determines vision-action connection

**Files:**
- Create: `pi0_attention_analysis.py` (standalone script)

**Step 1: Clone open-pi-zero repository**

```bash
cd /home/kana5123
git clone https://github.com/allenzren/open-pi-zero.git 2>/dev/null || echo "Already cloned"
ls open-pi-zero/
```

**Step 2: Investigate PaliGemma attention structure**

```bash
conda run -n interp python -c "
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('google/paligemma-3b-pt-224', trust_remote_code=True)
print(f'model_type: {cfg.model_type}')
print(f'text: hidden={cfg.text_config.hidden_size}, layers={cfg.text_config.num_hidden_layers}, heads={cfg.text_config.num_attention_heads}')
print(f'vision: {cfg.vision_config.model_type}')
"
```

**Step 3: Decide analysis approach**

For π0, there are two separate attention mechanisms:
- **PaliGemma self-attention**: Same analysis as VLM (vision+text tokens)
- **Flow Expert cross-attention**: Action tokens attend to VLM features → this is where vision information flows to actions

The self-attention analysis can reuse our existing pipeline. The cross-attention requires a new analysis.

Write `pi0_attention_analysis.py` that:
1. Loads PaliGemma-3B (without the Flow Expert)
2. Runs the standard 3-part sink verification on PaliGemma's self-attention
3. Documents findings about whether PaliGemma shows the bottleneck

**Step 4: Register PaliGemma-3B**

```python
# ── PaliGemma 3B (VLM component of π0) ──
register(VLAModelConfig(
    name="paligemma-3b",
    hf_id="google/paligemma-3b-pt-224",
    architecture="gemma",
    vision_encoder="siglip",
    num_layers=18,              # Verify
    num_heads=8,
    hidden_dim=2048,
    vision_grid_size=16,
    num_vision_tokens=256,
    action_tokens=0,
    action_type="none",
    prompt_template="{instruction}",
    native_datasets=[],
    notes="VLM component of π0 (PaliGemma-3B), SigLIP vision encoder",
    layers_path="language_model.model.layers",
    auto_model_class="AutoModelForVision2Seq",
))
```

**Step 5: Commit**

```bash
git add model_registry.py pi0_attention_analysis.py
git commit -m "feat: π0 (PaliGemma) attention analysis setup"
```

---

### Task 13: Phase 3 Research — Dita Architecture Analysis

Dita uses a DiT (Diffusion Transformer) for action generation. Unlike autoregressive VLAs, Dita processes vision tokens through cross-attention in a denoising loop.

**Step 1: Clone Dita repository**

```bash
cd /home/kana5123
git clone https://github.com/Dita-Robotics/Dita.git 2>/dev/null || echo "Already cloned"
ls Dita/
```

**Step 2: Investigate Dita architecture**

Read the model architecture files to understand:
- How vision features are processed (vision encoder → DiT)
- Where cross-attention happens
- Whether there's a "token 0 bottleneck" possible in the DiT framework

```bash
find /home/kana5123/Dita -name "*.py" | head -20
```

**Step 3: Analyze Dita's attention mechanism**

Dita DiT blocks use:
- Self-attention among noised action tokens
- Cross-attention: action tokens → vision features
- The vision features come from a separate vision encoder

In cross-attention, there's no causal mask → all vision tokens are equally accessible.
The "bottleneck" pattern from autoregressive models (where earlier tokens accumulate residual information) may not apply.

Write analysis notes documenting the architectural difference.

**Step 4: Commit findings**

```bash
git add docs/plans/
git commit -m "docs: Phase 3 diffusion VLA architecture analysis notes"
```

---

### Task 14: Run Phase 3 — PaliGemma-3B Verification

**Step 1: Run sink verification on PaliGemma-3B**

```bash
cd /home/kana5123/ATLASVLA
conda run -n interp python verify_attention_sinks.py \
    --model paligemma-3b \
    --device cuda:5 \
    --n_samples 5 \
    --all-layers
```

**Step 2: Record findings**

Does PaliGemma (Gemma backbone, SigLIP vision) show the same bottleneck?

**Step 3: Commit**

```bash
git add outputs/sink_verification/paligemma-3b/
git commit -m "data: PaliGemma-3B sink verification results (Phase 3)"
```

---

### Task 15: Final Cross-Architecture Comparison

**Step 1: Run full comparison across all models**

```bash
cd /home/kana5123/ATLASVLA
conda run -n interp python compare_sink_results.py \
    openvla-7b tracevla-phi3v spatialvla-4b \
    llava-1.5-7b internvl2-8b paligemma-3b
```

**Step 2: Generate final comparison table**

The comparison should produce:
```
outputs/sink_verification/cross_model_comparison/
├── comparison_table.json       (all models × all metrics)
├── contribution_profiles.png   (overlaid line plots)
└── bottleneck_comparison.png   (bar chart of severity)
```

**Step 3: Interpret results and document**

Create a summary document with:
1. Which models show the bottleneck (onset layer, severity)
2. VLA vs VLM comparison (same backbone)
3. Autoregressive vs Diffusion comparison
4. Hypothesis validation:
   - H1: VLA fine-tuning causes bottleneck? → Compare OpenVLA vs LLaVA-1.5
   - H2: Vision encoder causes bottleneck? → Compare Prismatic vs SigLIP vs CLIP models
   - H3: Autoregressive structure causes bottleneck? → Compare with PaliGemma/Dita

**Step 4: Commit all final results**

```bash
git add outputs/sink_verification/ compare_sink_results.py
git commit -m "feat: complete cross-architecture bottleneck analysis (Part A)"
```

---

## Execution Notes

### GPU Assignment
- cuda:0 — OpenVLA-7B (already done)
- cuda:1 — TraceVLA-Phi3V (Task 3)
- cuda:2 — SpatialVLA-4B (Task 4)
- cuda:3 — LLaVA-1.5-7B (Task 9)
- cuda:4 — InternVL2-8B (Task 10)
- cuda:5 — PaliGemma-3B (Task 14)

Tasks 3 & 4 can run in parallel. Tasks 9 & 10 can run in parallel.

### Expected Issues
1. **Phi3V fused QKV**: `get_wov_matrix()` needs architecture-specific W_V extraction (Task 1)
2. **InternVL2 custom code**: May need `trust_remote_code=True` patches in loader
3. **LLaVA vision tokens**: 576 tokens (24×24) vs OpenVLA's 256 — larger memory footprint
4. **PaliGemma prompt format**: Different from LLaMA-style, may need special handling
5. **Dita**: Cannot run standard self-attention analysis on DiT blocks (fundamentally different architecture)

### Estimated Timeline
- Task 0-2: Code changes (~30 min)
- Task 3-4: Phase 1 runs (~30 min, parallel)
- Task 5: Phase 1 comparison (~10 min)
- Task 6-8: Phase 2 setup (~30 min)
- Task 9-10: Phase 2 runs (~30 min, parallel)
- Task 11: Phase 2 comparison (~10 min)
- Task 12-13: Phase 3 research (~30 min)
- Task 14: Phase 3 run (~15 min)
- Task 15: Final comparison (~15 min)
- **Total: ~3-4 hours**
