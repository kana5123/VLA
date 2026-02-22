# Cross-Model Attention Sink Verification — Design Document

## Goal

Verify that the attention sink phenomenon (vision[0] + text "\n" absorbing 45-75% of attention)
is **universal across diverse VLA architectures**, not specific to OpenVLA.

## Scope

- **5 Tier-1 VLA models** with VLM backbones (standard HuggingFace loading)
- **5 robot manipulation datasets** (1 episode each per model)
- **25 extraction runs** total
- Per-head JSON + heatmap visualization per run
- Integrated cross-model comparison in `outputs/visualizations/`

## Models (Tier 1)

| Model | HF ID | LLM Backbone | Vision Encoder | Params | GPU RAM |
|-------|--------|-------------|----------------|--------|---------|
| OpenVLA-7B | `openvla/openvla-7b` | LLaMA-2 7B (32L/32H) | DINOv2+SigLIP | 7B | ~15GB |
| ECoT-7B | `Embodied-CoT/ecot-openvla-7b-bridge` | LLaMA-2 7B (32L/32H) | DINOv2+SigLIP | 7B | ~15GB |
| TraceVLA-Phi3 | `furonghuang-lab/tracevla_phi3v` | Phi-3 mini (32L/32H) | CLIP ViT-L/14 | 4.2B | ~9GB |
| SpatialVLA-4B | `IPEC-COMMUNITY/spatialvla-4b-224-pt` | Gemma-2 2B (26L/8H) | SigLIP-So400m | 4B | ~9GB |
| SmolVLA-450M | `lerobot/smolvla_base` | SmolLM2 (12L/16H) | SigLIP | 450M | ~2GB |

## Datasets

| Dataset | Format | Size | Loader Status |
|---------|--------|------|---------------|
| Bridge V2 | TFRecord/cache | Cached | Fix needed (metadata.json → pkl cache) |
| CALVIN Debug | HDF5 | ~1.3GB | Needs implementation |
| LeRobot PushT | LeRobot native | ~500MB | Needs implementation |
| DROID-100 | HDF5 | ~2GB | Needs implementation |
| RH20T Mini | Custom | ~26GB | Needs implementation |

## Extraction Matrix

```
                Bridge V2   CALVIN    LeRobot   DROID-100   RH20T
OpenVLA-7B      ✓           ✓         ✓         ✓           ✓
ECoT-7B         ✓           ✓         ✓         ✓           ✓
TraceVLA-4B     ✓           ✓         ✓         ✓           ✓
SpatialVLA-4B   ✓           ✓         ✓         ✓           ✓
SmolVLA-450M    ✓           ✓         ✓         ✓           ✓
```

## GPU Allocation

```
GPU 1-4: Adapter experiment (v1 training in progress)
GPU 5:   OpenVLA-7B (15GB) → 5 datasets sequential
GPU 6:   ECoT-7B (15GB) → 5 datasets sequential
GPU 7:   TraceVLA (9GB) → SpatialVLA (9GB) → SmolVLA (2GB) sequential
```

## Output Structure

```
outputs/
├── cross_model_analysis/          # Per-model raw results
│   ├── <model-name>/
│   │   ├── <dataset-name>/
│   │   │   ├── ep000_step000.json          # head-averaged top-5
│   │   │   ├── ep000_step000_perhead.json  # per-head breakdown
│   │   │   └── perhead_heatmap.png         # per-head heatmap
│   │   └── ...
│   └── ...
│
└── visualizations/                # Integrated comparison
    ├── cross_model_sink_comparison.png   # vision[0] bar chart per model
    ├── cross_model_heatmap.png           # model × layer heatmap
    ├── cross_model_dual_sink.png         # stacked bar (sink+text+useful+action)
    ├── cross_model_summary.json          # summary JSON
    ├── cross_model_table.tex             # LaTeX table
    └── per_dataset_comparison.png        # per-dataset sink comparison
```

## Blockers and Fixes

1. **`load_bridge_sample()`**: Reads `metadata.json` which doesn't exist.
   Fix: Use `metadata.pkl` + `images.dat` memmap from `/ceph_data/kana5123/bridge_data_cache/`.

2. **Dataset loaders**: CALVIN, LeRobot, DROID, RH20T loaders not implemented.
   Fix: Add `load_calvin_sample()`, `load_lerobot_sample()`, `load_droid_sample()`, `load_rh20t_sample()`.

3. **SpatialVLA registry**: Listed as `qwen2` architecture but actually uses PaliGemma2 (Gemma-2).
   Fix: Update model_registry.py entry.

4. **SmolVLA loading**: Uses LeRobot framework, not standard `AutoModelForVision2Seq`.
   Fix: Add custom loader or check if HuggingFace Transformers loading works.

5. **Dataset downloads**: Only Bridge V2 is cached. Need ~30GB downloads to `/ceph_data/`.

## Files to Modify

| File | Changes |
|------|---------|
| `dataset_registry.py` | Fix `load_bridge_sample()`, add 4 dataset loaders |
| `model_registry.py` | Fix SpatialVLA, verify SmolVLA config |
| `cross_model_extract.py` | Multi-dataset support, `--dataset` expansion |
| `cross_model_compare.py` | Per-dataset comparison visualization |

## Success Criteria

1. All 5 models show vision[0] ratio > 0.15 → sink is universal
2. Per-head heatmaps generated for all 25 model×dataset pairs
3. Cross-model comparison figures demonstrate consistent pattern
4. Paper-ready visualizations in `outputs/visualizations/`

## Approach

GPU-parallel extraction (Approach B):
- 3 GPUs (5, 6, 7) running simultaneously
- Independent of adapter experiment on GPUs 1-4
- Estimated completion: ~1.5 hours
