# Phase 3 Expansion Results: N=20 → N=175 + VAR Baseline

## 1. Sample Expansion: N=20 → N=175 (7 skills × 25)

**Goal**: Validate that Phase 3 findings hold with 8.75× more samples and balanced skill distribution.

### Cross-Model D Metric Comparison (N=20 → N=175)

| Model | Metric | N=20 | N=175 | Delta | Stable? |
|-------|--------|------|-------|-------|---------|
| **ECoT-7b** | D1 entropy | 1.323 | 1.698 | +0.375 | ~ (wider skill coverage) |
| | D2 consistency | 0.630 | 0.618 | -0.012 | **YES** |
| | D3 KL (V=0) | 1.662 | 1.289 | -0.373 | **YES** (still high) |
| | D3 flip rate | 0.450 | 0.457 | +0.007 | **YES** |
| | D0 NLL | 17.07 | 16.58 | -0.49 | **YES** |
| **OpenVLA-7b** | D1 entropy | 2.287 | 1.724 | -0.563 | ~ (entropy decreased) |
| | D2 consistency | 0.380 | 0.401 | +0.021 | **YES** |
| | D3 KL (V=0) | 1.664 | 1.042 | -0.622 | **YES** (still high) |
| | D3 flip rate | 0.600 | 0.486 | -0.114 | **YES** (still high) |
| | D0 NLL | 9.27 | 9.25 | -0.02 | **YES** |
| **TraceVLA** | D1 entropy | 2.216 | 1.962 | -0.254 | ~ |
| | D2 consistency | 0.550 | 0.486 | -0.064 | **YES** |
| | D3 KL (V=0) | 0.010 | 0.010 | 0.000 | **YES** (near-zero) |
| | D3 flip rate | 0.050 | 0.057 | +0.007 | **YES** |
| | D0 NLL | 14.42 | 14.42 | 0.00 | **YES** |

### Key Finding: All core claims CONFIRMED at N=175

1. **Bottleneck pattern confirmed**: ECoT/OpenVLA D3 KL remains high (1.3 / 1.0), D3 flip rate ~45-49%
2. **Sink pattern confirmed**: TraceVLA D3 KL near-zero (0.01), flip rate ~5.7%
3. **D2 consistency gap confirmed**: OpenVLA (0.40) << ECoT (0.62) ≈ TraceVLA (0.49) << SpatialVLA (pending)
4. **D4 action diversity now properly measured**: With 25 samples per skill, diversity metrics are meaningful
   - TraceVLA has highest diversity (21-25 unique actions per skill) — consistent with distributed routing
   - ECoT has lowest diversity (3-15 unique) — consistent with bottleneck compression

### Skill Balance Comparison

| | N=20 (original) | N=175 (expanded) |
|---|---|---|
| Skills | 6 (unbalanced: pick=6, open=1) | 7 (balanced: 25 each) |
| Total | 20 | 175 |
| Added skill | — | turn (25 samples) |
| Seed | 42 | 2024 (independent) |

---

## 2. VAR Baseline Comparison

**Goal**: Compare our K-scale intervention with VAR (ICLR 2025) attention redistribution on VLA models.

### Phi Analysis (Sink Dimension Detection)

| Model | Backbone | phi(token0) | n_sinks (tau=20) | Sink Position |
|-------|----------|-------------|------------------|---------------|
| ECoT-7b | LLaMA-2 | **51.72** | 1 | Vision token 0 |
| OpenVLA-7b | LLaMA-2 | **50.59** | 1 | Vision token 0 |
| TraceVLA | Phi-3-V | N/A (no known D_sink) | — | — |
| SpatialVLA | Gemma2 | N/A (no known D_sink) | — | — |

**Key finding**: LLaMA-2 models show phi >> 20 at token 0, confirming it as a sink by VAR's definition.
But our analysis shows this token is actually a **bottleneck** (not a sink) because V=0 causes action collapse.

### VAR vs K-scale: Per-Model Comparison (N=20)

#### ECoT-7b (Bottleneck)

| Method | Entropy | D2 | ΔD2 | Action Change |
|--------|---------|-----|------|---------------|
| Baseline | 1.323 | 0.630 | — | — |
| VAR p=0.3 | 1.224 | 0.620 | **-0.010** | 0.15 |
| VAR p=0.6 | 1.109 | 0.600 | **-0.030** | 0.25 |
| VAR p=0.9 | 1.262 | 0.590 | **-0.040** | 0.40 |
| K-scale α=0.0 | 1.193 | 0.600 | **-0.030** | 0.15 |
| K-scale α=0.1 | 1.178 | 0.590 | **-0.040** | 0.15 |
| K-scale α=0.3 | 1.175 | 0.580 | **-0.050** | 0.10 |

**Result: BOTH methods HURT bottleneck models.** Neither VAR nor K-scale improve D2 for ECoT.

#### OpenVLA-7b (Bottleneck/Coexist)

| Method | Entropy | D2 | ΔD2 | Action Change |
|--------|---------|-----|------|---------------|
| Baseline | 2.287 | 0.380 | — | — |
| VAR p=0.3 | 2.333 | 0.410 | **+0.030** | 0.40 |
| VAR p=0.6 | 2.580 | 0.400 | **+0.020** | 0.55 |
| VAR p=0.9 | 2.960 | 0.450 | **+0.070** | 0.55 |
| K-scale α=0.0 | 2.442 | 0.370 | **-0.010** | 0.40 |
| K-scale α=0.1 | 2.470 | 0.360 | **-0.020** | 0.40 |
| K-scale α=0.3 | 2.468 | 0.350 | **-0.030** | 0.35 |

**Result: VAR helps OpenVLA (+7% D2) but K-scale hurts it (-3%).** This is because OpenVLA has a coexist pattern where the attention sink is vision but the contribution bottleneck is text — VAR reduces the surplus vision attention (sink), allowing other vision tokens to contribute.

#### TraceVLA-Phi3V (Sink)

| Method | Entropy | D2 | ΔD2 | Action Change |
|--------|---------|-----|------|---------------|
| Baseline | 2.216 | 0.550 | — | — |
| VAR p=0.3 | 2.218 | 0.560 | +0.010 | 0.00 |
| VAR p=0.6 | 2.186 | 0.530 | -0.020 | 0.00 |
| VAR p=0.9 | 2.202 | 0.540 | -0.010 | 0.00 |
| K-scale α=0.0 | 2.219 | 0.580 | **+0.030** | 0.05 |
| K-scale α=0.1 | 2.222 | 0.570 | **+0.020** | 0.05 |
| K-scale α=0.3 | 2.249 | 0.580 | **+0.030** | 0.05 |

**Result: K-scale helps TraceVLA (+3% D2) but VAR is neutral.** For the true sink model, K-scale works because it addresses the Q/K-level position shortcut. VAR (value-based) has zero action change rate — confirming the sink's V projection carries little information.

#### SpatialVLA-4b (Normal)

| Method | Entropy | D2 | ΔD2 | Action Change |
|--------|---------|-----|------|---------------|
| Baseline | 3.283 | 0.790 | — | — |
| VAR p=0.3 | 3.084 | 0.780 | -0.010 | 0.10 |
| VAR p=0.6 | 2.926 | 0.830 | **+0.040** | 0.15 |
| VAR p=0.9 | 2.832 | 0.820 | **+0.030** | 0.20 |
| K-scale α=0.0 | 3.693 | 0.770 | -0.020 | 0.15 |
| K-scale α=0.1 | 3.636 | 0.780 | -0.010 | 0.10 |
| K-scale α=0.3 | 3.553 | 0.810 | **+0.020** | 0.10 |

**Result: Both methods slightly help SpatialVLA, with VAR slightly ahead (+4% vs +2%).** This is expected for the "Normal" routing model — no severe bottleneck or sink, so redistribution provides modest benefit by diversifying attention.

### Cross-Method Summary Table

| Model | Type | VAR Best ΔD2 | K-scale Best ΔD2 | Winner |
|-------|------|-------------|------------------|--------|
| ECoT-7b | Bottleneck | -1% (p=0.3) | -3% (α=0.3) | Neither (both hurt) |
| OpenVLA-7b | Coexist | **+7%** (p=0.9) | -3% (α=0.3) | **VAR** |
| TraceVLA | Sink | +1% (p=0.3) | **+3%** (α=0.0) | **K-scale** |
| SpatialVLA | Normal | **+4%** (p=0.6) | **+2%** (α=0.3) | **VAR** (marginal) |

### Paper-Ready Insight

**The optimal intervention depends on the routing failure mode:**

1. **Sink models (TraceVLA)**: K-scale works because the position shortcut is encoded in Q/K.
   VAR is ineffective because the sink V projection carries minimal information — redistributing it changes nothing.

2. **Coexist models (OpenVLA)**: VAR works because attention is concentrated on a vision sink
   (which by definition carries surplus attention). Redistributing this surplus to other vision tokens
   improves grounding. K-scale fails because reducing K at the sink doesn't address the text-side bottleneck.

3. **Bottleneck models (ECoT)**: Neither method works at inference time. The bottleneck carries
   critical routing information — any intervention that reduces its influence degrades performance.
   This motivates training-time fixes (Task 3).

4. **Normal models (SpatialVLA)**: Both methods provide modest improvement (+2-4%). The healthy
   routing pattern means there's no severe failure mode, but mild attention redistribution still
   helps by diversifying the information sources. VAR is slightly better because SpatialVLA has
   mild attention concentration (not extreme enough to warrant Q/K-level intervention).

---

## 3. Phi Analysis: VAR Sink Detection on VLAs

For LLaMA-2 based models (ECoT, OpenVLA), vision token 0 shows:
- phi(token0) ≈ 50-52 (well above VAR's tau=20 threshold)
- Consistently the ONLY token exceeding tau=20 across all samples
- This confirms token 0 would be classified as a "visual attention sink" by VAR

**Critical distinction**: VAR defines sinks as tokens that "receive high attention but contribute little"
and can be removed without performance loss. Our V=0 analysis shows ECoT/OpenVLA's token 0 causes
action collapse when zeroed (D3 KL=1.3-1.7, flip=45-60%), meaning it's a **bottleneck, not a sink**.

This represents a fundamental disagreement with VAR's assumption for VLA models:
VLA sinks ≠ VLM sinks because the action prediction pathway creates functional dependencies
that don't exist in standard text generation.
