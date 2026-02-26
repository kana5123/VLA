# Text-Token → Visual-Token Attention Map Visualization

**Date**: 2026-02-25
**Status**: Active
**Purpose**: Investigate where task-verb and object-noun tokens attend in visual space across all layers

## Motivation

Current adapter approach assumes last token → object attention matters.
New hypothesis: **text tokens referencing tasks/objects should attend to corresponding visual regions**.
Must verify this before redesigning the attention redistribution strategy.

## Design

### What We Visualize

For a given instruction like "pick up the yellow cup":
- Extract **verb tokens** ("pick") and **noun tokens** ("cup") positions
- For each such token, at each layer, extract attention weights to visual tokens
- Head-average the attention → (num_vision_tokens,) vector
- Map to 16×16 grid → overlay on original image as heatmap
- Top-10 patches: purple/blue color intensity proportional to attention
- Remaining high-attention patches: red boxes (irrelevant attention sinks)

### Output Format

Per sample: one large figure with layout:
```
Row 0: [Original Image] [Token legend]
Row 1-N: Layer 0..31 attention heatmaps for each selected token (columns)
```

### Technical Approach

1. **Model loading**: `load_model_from_registry()` from extract_attention.py
2. **Attention capture**: Forward hooks on all layers, store (B, H, Q, K) weights
3. **Token position finding**: Tokenize instruction, identify verb/noun indices via keyword matching
4. **Vision attention extraction**: `attn[layer][:, :, token_pos, vision_start:vision_end].mean(dim=1)` → (V,)
5. **Grid mapping**: Reshape (V,) → (grid_h, grid_w) → resize to image dimensions
6. **Visualization**: matplotlib imshow with alpha-blended heatmap overlay

### Models

Start with OpenVLA-7B, extend to all 4 experiment-ready models.

### Data

2-3 Bridge V2 samples with diverse instructions.
