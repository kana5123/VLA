# SAM2 Grounded Segmentation for ATLASVLA

## Date: 2026-02-18

## Problem

GroundingDINO bounding boxes produce unstable patch coverage (0-100%) for VAR object-aware attention redistribution:
- ep0: 43% (acceptable but imprecise)
- ep1: 100% (5 nouns including "side", "top" -> all patches selected)
- ep4/ep6: 0% (empty instruction or grounding failure)
- ep9: 87.5% (overly broad)

Result: `var_object` condition degraded MSE by +5.6% vs baseline.

Root causes:
1. SpaCy extracts non-object nouns (side, top, bottom, it, piece)
2. GroundingDINO bboxes are too coarse for 16x16 grid mapping

## Solution

Adopt Dream project's proven GroundingDINO+SAM architecture but replace SAM with SAM2:

1. **Noun filtering**: Remove abstract/positional nouns from SpaCy output
2. **Area filtering + NMS**: From Dream's ObjectSegmentor (max 50% area, IoU 0.5)
3. **SAM2 masks**: Pixel-level segmentation via `facebook/sam2.1-hiera-tiny` (156MB)
4. **Mask-to-patch conversion**: Grid overlap calculation with configurable threshold

## Architecture

```
Instruction -> SpaCy (filtered) -> physical object nouns only
                                    |
Image + Nouns -> GroundingDINO -> bboxes
                                    |
NMS (IoU 0.5) + Area Filter (max 50%) <- from Dream
                                    |
Image + filtered bboxes -> SAM2 (hiera-tiny) -> pixel masks
                                    |
Merge same-label masks <- from Dream
                                    |
Masks -> 16x16 grid overlap -> patch_indices (~10-30/256)
                                    |
attention_v3.py: VAR object_redirect with precise indices
```

## Changes

### object_grounder.py (refactor)
- Add `ABSTRACT_NOUNS` blacklist for noun filtering
- Add `_filter_detections()` with NMS + area filtering (from Dream)
- Add `_merge_same_label()` (from Dream)
- Replace `boxes_to_patch_indices()` with `masks_to_patch_indices()`
- Add SAM2 model loading and inference via transformers API

### config.py
- `SAM2_MODEL_ID = "facebook/sam2.1-hiera-tiny"`
- `GROUNDING_MAX_AREA_FRACTION = 0.5`
- `GROUNDING_NMS_IOU_THRESHOLD = 0.5`
- `SAM_PATCH_OVERLAP_THRESHOLD = 0.1`

### run_v3_experiment.py
- Add conditions: `sam_var_object`, `sam_var_obj_gripex`, `sam_var_bgsup_gripex`

## Expected Results

| Metric | var_object (bbox) | sam_var_object (expected) |
|--------|-------------------|--------------------------|
| Coverage | 0-100% (unstable) | 2-10% (stable) |
| vs baseline | +5.6% (degraded) | -0.3% to -1% (improved) |
