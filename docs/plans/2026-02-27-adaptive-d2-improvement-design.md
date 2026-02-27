# Adaptive D2 Improvement: Dynamic Routing-Aware Intervention for VLAs

## Problem
D2 (augmentation consistency) varies dramatically across VLA models:
- ECoT-7b: 0.63 (Bottleneck) — inference-time intervention hurts
- OpenVLA-7b: 0.38 (Coexist) — VAR helps +7%, K-scale hurts
- TraceVLA: 0.55 (Sink) — K-scale helps +3%, VAR neutral
- SpatialVLA: 0.79 (Normal) — both help slightly

No single intervention works for all models. The optimal method depends on the routing failure mode.

## Solution: 3-Pronged Approach

### Approach 1: Adaptive Inference-Time Router
- Diagnose routing failure mode at model load time (3 samples, ~30s)
- Cache diagnosis and select optimal hook (VAR/K-scale/none)
- Zero additional inference cost after diagnosis

### Approach 2: Hybrid VAR+K-scale Hook
- Apply both VAR (V scaling) and K-scale (K scaling) simultaneously
- Model-type-specific weight tuning via grid search
- Test for synergy effects between the two methods

### Approach 3: Attention Entropy Regularization (Training-Time)
- Add attention entropy loss during fine-tuning
- Prevents bottleneck/sink formation at the source
- Only approach that can fix Bottleneck models (ECoT)

## Expected Outcome
A complete "diagnose → intervene → prevent" pipeline that improves D2 for all VLA routing types.

## Design approved: 2026-02-27
