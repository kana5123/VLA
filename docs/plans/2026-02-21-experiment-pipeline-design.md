# Full Experiment Pipeline Design

## Goal

End-to-end experiment pipeline: train 4 adapter configurations, evaluate offline with MSE metrics, compare results, and prepare LoRA infrastructure for future extension.

## Current State

| Component | File | Status |
|-----------|------|--------|
| V2 Model | adapter_model.py | Complete |
| Training | adapter_train.py | Complete (v1/v2, DDP, object masks) |
| Data Pipeline | adapter_data.py | Complete (object_masks loading) |
| Attention Engine | attention_v3.py | Complete (redistribution_weights) |
| SAM Preprocessing | sam_preprocess.py | Running (~4.5h ETA) |
| Evaluation | adapter_eval.py | **V1 only — needs v2 upgrade** |
| Experiment Runner | — | **Missing** |
| Results Comparison | compare_v3_results.py | V3 heuristic only, not adapter |
| LoRA | — | **Missing** |

## Approach: Incremental Modification (A)

Keep existing working code, fill gaps with minimal new files.

---

## Component 1: adapter_eval.py V2 Upgrade

### Problem

```python
# Line 56 — v1 only:
self.adapter = AttentionAdapter(hidden_dim=hidden_dim)

# Line 232 — h_vision not captured:
captured["h"] = h[:, -1, :]

# Line 246 — v2 signature mismatch:
p_matrix = adapter(captured["h"].float())
```

- `get_v3_ctx_for_eval()` does not capture `h_vision`
- No `object_mask` support
- No `redistribution_weights` computation
- No v1/v2 auto-detection

### Solution

1. **Auto-detect adapter version** from checkpoint:
   ```python
   ckpt = torch.load(path, map_location=device)
   version = ckpt.get("config", {}).get("adapter_version", 1)
   if version == 2:
       adapter = AttentionAdapterV2(hidden_dim=hidden_dim)
   else:
       adapter = AttentionAdapter(hidden_dim=hidden_dim)
   ```

2. **Upgrade `get_v3_ctx_for_eval()`**:
   - Capture both `h_last = h[:, -1, :]` and `h_vision = h[:, :vision_end, :]`
   - Accept `object_mask` parameter
   - For v2: call `adapter(h_last, h_vision, mask)` → get `(p_matrix, redist_raw)`
   - Compute blended redistribution weights with `blend_alpha`
   - Set `ctx.redistribution_weights`

3. **Upgrade `_run_inference()`**:
   - Load object masks from SAM memmap for current step
   - Pass `object_mask` to `get_v3_ctx_for_eval()`

4. **CLI options**:
   - `--use_object_masks` flag (default: auto-detect from checkpoint version)

### Files Modified

- `adapter_eval.py` (~80 lines changed)

---

## Component 2: run_adapter_experiment.py (New)

### Purpose

Automate training and evaluation of 4 experiment configurations.

### Configurations

| Config | Adapter | Training | Eval Details |
|--------|---------|----------|--------------|
| `base` | None | Skip (no training) | Raw OpenVLA MSE baseline |
| `v1` | AttentionAdapter | `adapter_train.py` (v1 mode) | Eval with v1 adapter |
| `v2-prop` | AttentionAdapterV2 | `adapter_train.py --freeze_blend` | Eval without learned redistribution |
| `v2-full` | AttentionAdapterV2 | `adapter_train.py` (default) | Eval with SAM masks + redistribution |

### Design

```python
CONFIGS = {
    "base": {"skip_training": True, "adapter_version": None},
    "v1": {"adapter_version": 1},
    "v2-prop": {"adapter_version": 2, "freeze_blend": True},
    "v2-full": {"adapter_version": 2, "freeze_blend": False},
}
```

For each config:
1. Create output directory: `outputs/experiment_results/<config>/`
2. Train: `accelerate launch adapter_train.py --config <config> --output_dir <dir>`
3. Eval: `python adapter_eval.py --checkpoint <dir>/checkpoints/best.pt --output_dir <dir>/eval/`

### Required Change in adapter_train.py

Add `--freeze_blend` flag (~15 lines):
```python
if args.freeze_blend:
    raw_adapter._blend_logit.requires_grad_(False)
```

### Files

- Create: `run_adapter_experiment.py` (~200 lines)
- Modify: `adapter_train.py` (~15 lines for `--freeze_blend` + `--adapter_version` + `--output_dir`)

---

## Component 3: compare_adapter_results.py (New)

### Purpose

Load evaluation results from all 4 configs and produce comparison outputs.

### Outputs

1. **Summary table** (JSON + stdout):
   - Overall MSE, Spatial MSE (x,y,z), Rotational MSE (roll,pitch,yaw), Gripper accuracy
   - Per-config improvement % vs baseline

2. **Plots**:
   - Bar chart: per-dimension MSE across configs
   - Heatmap: per-dimension improvement %
   - Episode-level scatter: adapter MSE vs baseline MSE

3. **LaTeX table** for paper inclusion

### Design

Follows `compare_v3_results.py` patterns:
- Load `eval_results.json` from each config directory
- Aggregate and compute statistics
- Generate matplotlib figures

### Files

- Create: `compare_adapter_results.py` (~250 lines)

---

## Component 4: LoRA Infrastructure (adapter_lora.py, New)

### Purpose

Prepare LoRA fine-tuning capability. Infrastructure only — actual training deferred until v2 results are analyzed.

### Design

```python
from peft import LoraConfig, get_peft_model

def create_lora_model(model, lora_config):
    """Wrap OpenVLA with LoRA adapters on LLaMA layers."""
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)
```

Two training modes:
1. **LoRA only**: Fine-tune OpenVLA with LoRA, no attention adapter
2. **LoRA + Adapter**: Two-stage — first train adapter (frozen LoRA), then train LoRA (frozen adapter)

### Config additions (config.py)

```python
# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LORA_DROPOUT = 0.05
LORA_LR = 1e-4
LORA_MAX_STEPS = 20000
```

### Files

- Create: `adapter_lora.py` (~200 lines)
- Modify: `config.py` (~15 lines for LoRA constants)

---

## Execution Order

```
[Running] SAM Preprocessing (GPU 1-7, ~4.5h ETA)
      │
      ▼
[Phase 1] adapter_eval.py v2 upgrade
      │   Can start NOW (no SAM dependency for code changes)
      │
[Phase 2] adapter_train.py --freeze_blend flag
      │   + run_adapter_experiment.py
      │
[Phase 3] Training (after SAM completes)
      │   ├─ base: eval only
      │   ├─ v1: train adapter v1
      │   ├─ v2-prop: train v2, blend frozen at 0
      │   └─ v2-full: train v2, blend learnable
      │
[Phase 4] Evaluation (all 4 configs)
      │
[Phase 5] compare_adapter_results.py + analysis
      │
[Phase 6] LoRA infrastructure (adapter_lora.py)
      │   Training deferred to post-analysis
      │
      ▼
   Paper results
```

**Key dependency**: Phase 3-4 training/eval requires SAM preprocessing complete (v2-full needs object_masks.dat).

---

## File Change Summary

| Action | File | Type | Est. Lines |
|--------|------|------|-----------|
| Modify | adapter_eval.py | V2 upgrade | ~80 |
| Modify | adapter_train.py | --freeze_blend, --adapter_version, --output_dir | ~15 |
| Modify | config.py | LoRA constants | ~15 |
| Create | run_adapter_experiment.py | Experiment runner | ~200 |
| Create | compare_adapter_results.py | Results comparison | ~250 |
| Create | adapter_lora.py | LoRA infrastructure | ~200 |

**Total**: 6 files (3 modify + 3 create), ~760 lines

---

## Decision Log

- **Approach A (incremental modification)** chosen over full refactoring — preserves working code, minimizes risk
- **4 configs** (base, v1, v2-prop, v2-full) to isolate each component's contribution
- **LoRA infra only** now — train after v2 analysis to avoid premature optimization
- **Offline eval only** — no real-time robot inference needed
