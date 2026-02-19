# Object-Aware Differentiable Attention Adapter v2

**Date**: 2026-02-19
**Status**: Approved

## Problem

The current adapter (v1) learns per-head VAR redistribution strengths but distributes freed attention proportionally to ALL non-sink vision tokens. This dilutes the benefit — background patches get attention alongside task-relevant object patches.

For robot manipulation, spatial precision is critical. Attention should be directed specifically to task-relevant object patches (e.g., the cup to pick up, the gripper).

## Solution Overview

Extend the adapter with two capabilities:
1. **Per-head p** (existing): which heads to modify and how much sink attention to free
2. **Redistribution weights** (new): where to direct freed attention, using cross-attention between the task representation and vision patch representations

Use SAM2 + GroundingDINO to pre-compute object patch masks, ensuring freed attention goes ONLY to object patches (background receives nothing).

## Architecture

### Inputs (3 sources)

| Input | Shape | Source |
|-------|-------|--------|
| h_27_last | (4096,) | Layer 27 last token hidden state |
| h_27_vision | (V, 4096) | Layer 27 all vision token hidden states (V = vision_end, 256 or 512) |
| object_mask | (V,) | Pre-computed SAM binary mask (1 = object, 0 = background) |

### Branch 1: Per-head p (how much to free from sink)

```
h_27_last (4096) -> Linear(4096, 256) -> SiLU --------+
                                                       | concat (320)
object_mask (V) -> Linear(V, 64) -> SiLU -------------+
                                                       v
                                          Linear(320, 128) -> SiLU -> Dropout(0.1)
                                          Linear(128, 128) -> Sigmoid
                                          reshape -> (4 layers, 32 heads)
```

Zero-init: last layer weight=0, bias=-4 -> sigmoid(-4) ~ 0.018 at start.

### Branch 2: Redistribution weights (where to direct freed attention)

```
h_27_last (4096) -> Linear(4096, 128) -> query  (128,)
                                                   |  dot product
h_27_vision (V, 4096) -> Linear(4096, 128) -> keys (V, 128)
                                                   v
                                         scores = query . keys^T / (sqrt(128) * tau)
                                         scores[~object_mask] = -inf
                                         softmax(scores) -> redistribution_weights (V,)
```

Temperature tau=2.0 for smoother initial gradients.

### Blending for training stability

```python
# blend_alpha: learnable scalar, initialized to sigmoid(-4) ~ 0.018
# Starts with proportional redistribution, gradually transitions to learned weights
final_weights = blend_alpha * redistribution_weights + (1 - blend_alpha) * proportional_weights
```

### Parameter Count

| Component | Parameters |
|-----------|-----------|
| Branch 1 (per_head_p) | ~1.12M |
| Branch 2 (query + key proj) | ~1.05M |
| blend_alpha | 1 |
| **Total** | **~2.17M** (0.031% of frozen 7B) |

## apply_var Modification

```python
# Current (proportional to all non-sink):
bonus = freed * (nonsink_vals / nonsink_sum)

# New (object-targeted with learned weights):
bonus = freed * redistribution_weights[nonsink_t]
# Object patches: positive weights (sum to 1)
# Background patches: 0.0
```

## Data Pipeline

### Phase 1: Image cache (existing, currently building)

```
BridgeData V2 tfrecords -> memmap cache
  images.dat:      (total_steps, 256, 256, 3) uint8
  metadata.pkl:    [{instruction, action, episode_id, ...}]
  cache_info.json: {total_steps, total_episodes, ...}
```

### Phase 2: SAM preprocessing (new, ~8h on 4 GPU)

For each step:
1. Extract noun phrases from instruction (spaCy en_core_web_sm)
2. GroundingDINO(image, noun_phrase) -> bounding boxes
3. SAM2(image, boxes) -> pixel mask (256x256)
4. Map to vision grid: 16x16 grid, patch is "object" if mask overlap > threshold
5. Store: object_masks.dat (total_steps, V) uint8

Failure handling:
- Step-level: SAM failure -> step excluded from training (mask marker = 255)
- Episode-level: >50% SAM failure rate -> entire episode excluded

### Phase 3: Metadata update

- Compute valid_step_indices (exclude SAM-failed steps)
- Compute valid_episodes (exclude >50% failure rate episodes)
- Split valid_episodes into train/val/test (AFTER filtering)

### Dual encoder handling (V=512)

If OpenVLA uses Prismatic dual encoder (DINOv2 + SigLIP):
- vision_end = 512 (detected at runtime)
- SAM produces 256-dim mask (one 16x16 grid)
- Duplicate for both encoders: object_mask_512 = cat(mask_256, mask_256)
- Both encoders share the same spatial layout

## Training Loop

### Hook modification

```python
# Capture both last token and all vision tokens from layer 27
captured["h_last"] = output[0][:, -1, :]            # (1, 4096)
captured["h_vision"] = output[0][:, :vision_end, :]  # (1, V, 4096)
```

### forward_with_adapter

```python
def forward_with_adapter(model, adapter, processor, ctx,
                         image, instruction, target_tokens,
                         device, object_mask):
    # Step 1: Capture h_27 (no_grad)
    # Step 2: Adapter (gradient tracked)
    per_head_p, redistribution_weights = adapter(
        h_last.float(), h_vision.float(), object_mask.float()
    )
    # Step 3: Set context (scatter per_head_p, set redistribution_weights)
    # Step 4: Teacher-forced forward x7 (gradient through layers 28-31)
```

### Gradient path

```
CE Loss -> logits -> LM head (frozen) -> h_31
  -> Layer 31 attention -> apply_var
     -> per_head_p (Branch 1: MLP) -> adapter params [GRADIENT]
     -> redistribution_weights (Branch 2: cross-attn) -> adapter params [GRADIENT]
```

Both branches receive gradients through the same apply_var function.

### Per-sample gradient accumulation

- Process 1 sample at a time, backward immediately
- DDP no_sync for all but last sample per batch
- Peak memory: ~18 GB/GPU (model 14GB + 1 sample activations ~4GB)

## Inference

1. Capture image
2. GroundingDINO + SAM2 (real-time, ~80ms on H100)
3. Adapter predicts per_head_p + redistribution_weights
4. apply_var in layers 28-31
5. Generate 7 action tokens

Additional inference cost: ~80ms for GroundingDINO + SAM2. Compatible with 5-10 Hz robot control.

## Known Issues and Mitigations

| Issue | Severity | Mitigation |
|-------|----------|-----------|
| Softmax vanishing gradient | Medium | Temperature tau=2.0, Xavier init |
| Cross-attention initial instability | Medium | blend_alpha (sigmoid, init=-4) |
| Vision tokens 256 vs 512 | High | Runtime detection, mask duplication |
| spaCy noun phrase extraction failure | Low | Full instruction as fallback query |
| Episode filtering affects split ratios | Medium | Filter first, then split |
| Memory increase from h_27_vision | Low | Per-sample grad accum sufficient |

## Related Work

- [VAR (ICLR 2025)](https://arxiv.org/abs/2503.03321): Proportional redistribution to all non-sink visual tokens
- [Localization Heads (CVPR 2025)](https://arxiv.org/abs/2503.06287): Few attention heads sufficient for visual grounding
- [Attention Sink (ICLR 2025 Spotlight)](https://github.com/sail-sg/Attention-Sink): Sink phenomenon analysis

Our contribution: **Object-targeted redistribution with learned per-head control** — combining VAR's redistribution mechanism with SAM-based object grounding and a cross-attention adapter for dynamic weight prediction.
