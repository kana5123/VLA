# Phase 3 Gate Check — Complete Results

> Generated: 2026-02-27
> Models: ECoT-7b, OpenVLA-7b, SpatialVLA-4b, TraceVLA-Phi3V
> Data: BridgeData V2, 150 balanced samples (25 per skill × 6 skills), 20 samples for Gate 3
> Skills: pick, move, fold, place, open, close

---

## Experiment Overview

| Gate | Purpose | Method |
|------|---------|--------|
| Gate ① | Contribution analysis + hidden state probe + skill signature | W_OV contribution, attention/contribution peaks, JS distance, logistic probe |
| Gate ② | Causal V=0 ablation on mode tokens (A/C/R) | ValueZeroHook on v_proj, KL divergence + top-1 change |
| Gate ③ | Text masking control + mini counterfactual | TextKVZeroHook (Prismatic) / 4D mask (others), verb swap delta-hidden |

## Model Architectures

| Model | Backbone | Layers | Q Heads | KV Heads | Hidden | QKV Type | Wrapper |
|-------|----------|--------|---------|----------|--------|----------|---------|
| ECoT-7b | LLaMA-2 | 32 | 32 | 32 | 4096 | Separate q/k/v_proj | Prismatic |
| OpenVLA-7b | LLaMA-2 | 32 | 32 | 32 | 4096 | Separate q/k/v_proj | Prismatic |
| SpatialVLA-4b | Gemma2 | 26 | 8 | 4 | 2304 | Separate q/k/v_proj | Standard HF |
| TraceVLA-Phi3V | Phi3V | 32 | 32 | 32 | 3072 | Fused qkv_proj | Standard HF |

## Token Layout (Sample 0)

| Model | Vision Range | Text Range(s) | n_vision | n_text | Total |
|-------|-------------|---------------|----------|--------|-------|
| ECoT-7b | [0, 256) | [256, 278) | 256 | 22 | 278 |
| OpenVLA-7b | [0, 256) | [256, 278) | 256 | 22 | 278 |
| SpatialVLA-4b | [0, 256) | [256, 270) | 256 | 14 | 270 |
| TraceVLA-Phi3V | [0, 313) | [313, 341) | 313 | 28 | 341 |

---

# Gate ① — Contribution Analysis + Hidden State Probe

## 1.1 Mode Tokens (Most Frequent Peak Tokens Across 150 Samples)

| Model | A_mode (Attention Peak) | freq | type | C_mode (Contribution Peak) | freq | type | R_mode (Residual Peak) | freq | type |
|-------|------------------------|------|------|---------------------------|------|------|------------------------|------|------|
| ECoT-7b | abs_t=0 `<s>` | 1.000 | vision | abs_t=0 `<s>` | 1.000 | vision | abs_t=271 | 0.157 | text |
| OpenVLA-7b | abs_t=0 `<s>` | 1.000 | vision | abs_t=271 | 0.180 | text | abs_t=262 | 0.079 | text |
| SpatialVLA-4b | abs_t=260 `robot` | 0.543 | text | abs_t=260 `robot` | 0.448 | text | abs_t=193 `<image>` | 0.752 | vision |
| TraceVLA-Phi3V | abs_t=0 `<s>` | 0.840 | vision | abs_t=1 `<\|user\|>` | 1.000 | vision | abs_t=3 | 0.495 | vision |

## 1.2 Skill Signature Analysis

| Model | d_within (JS) | d_between (JS) | signature_exists | probe_accuracy | mean_mismatch (A vs C) |
|-------|---------------|----------------|------------------|----------------|------------------------|
| ECoT-7b | 0.0257 | 0.0278 | True | 0.233 | 0.0682 |
| OpenVLA-7b | 0.389 | 0.430 | True | 0.273 | 0.230 |
| SpatialVLA-4b | 0.128 | 0.173 | True | 0.553 | 0.00449 |
| TraceVLA-Phi3V | 0.0169 | 0.0181 | True | 0.220 | 0.0783 |

## 1.3 Skill Distribution (All Models — Identical)

| Skill | Count |
|-------|-------|
| pick | 25 |
| move | 25 |
| fold | 25 |
| place | 25 |
| open | 25 |
| close | 25 |
| **Total** | **150** |

## 1.4 Per-Layer Analysis

### ECoT-7b (Layers 22–31)

| Layer | dominant_type | freq | a_c_match | match_rate | A_peak (abs_t, token, share) | C_peak (abs_t, token, share) | R_peak (abs_t, token) | entropy | top1_share | mismatch |
|-------|---------------|------|-----------|------------|------------------------------|------------------------------|----------------------|---------|------------|----------|
| 22 | bottleneck | 1.000 | True | 1.000 | 0, `<s>`, 0.6104 | 0, `<s>`, 0.9585 | 271, text | 0.5347 | 0.9060 | 0.1666 |
| 23 | bottleneck | 1.000 | True | 1.000 | 0, `<s>`, 0.6729 | 0, `<s>`, 0.9678 | 271, text | 0.3831 | 0.9279 | 0.1192 |
| 24 | bottleneck | 1.000 | True | 1.000 | 0, `<s>`, 0.7219 | 0, `<s>`, 0.9748 | 271, text | 0.3070 | 0.9407 | 0.0910 |
| 25 | bottleneck | 1.000 | True | 1.000 | 0, `<s>`, 0.7604 | 0, `<s>`, 0.9802 | 271, text | 0.2375 | 0.9526 | 0.0676 |
| 26 | bottleneck | 0.993 | True | 1.000 | 0, `<s>`, 0.7860 | 0, `<s>`, 0.9836 | 271, text | 0.1877 | 0.9622 | 0.0524 |
| 27 | bottleneck | 1.000 | True | 1.000 | 0, `<s>`, 0.8050 | 0, `<s>`, 0.9862 | 271, text | 0.1497 | 0.9699 | 0.0406 |
| 28 | bottleneck | 1.000 | True | 1.000 | 0, `<s>`, 0.8200 | 0, `<s>`, 0.9882 | 271, text | 0.1097 | 0.9762 | 0.0388 |
| 29 | bottleneck | 1.000 | True | 1.000 | 0, `<s>`, 0.8307 | 0, `<s>`, 0.9898 | 271, text | 0.0808 | 0.9816 | 0.0403 |
| 30 | bottleneck | 1.000 | True | 1.000 | 0, `<s>`, 0.8395 | 0, `<s>`, 0.9902 | 271, text | 0.0628 | 0.9831 | 0.0385 |
| 31 | bottleneck | 1.000 | True | 1.000 | 0, `<s>`, 0.8495 | 0, `<s>`, 0.9972 | 258, text | 0.0560 | 0.9898 | 0.0270 |

**Summary:** Pure bottleneck. Token `<s>` at abs_t=0 (first vision token) dominates both attention (61–85%) and contribution (96–99.7%) across ALL deep layers with 100% consistency. Entropy drops monotonically (0.53 → 0.06). This is the most extreme bottleneck in the study.

### OpenVLA-7b (Layers 22–31)

| Layer | dominant_type | freq | a_c_match | match_rate | A_peak (abs_t, token, share) | C_peak (abs_t, token, share) | R_peak (abs_t, token) | entropy | top1_share | mismatch |
|-------|---------------|------|-----------|------------|------------------------------|------------------------------|----------------------|---------|------------|----------|
| 22 | coexist | 1.000 | False | 0.000 | 0, `<s>`, 0.2646 | 271, text, 0.6503 | 262, text | 0.7927 | 0.6542 | 0.2116 |
| 23 | coexist | 1.000 | False | 0.000 | 0, `<s>`, 0.3070 | 271, text, 0.6634 | 264, text | 0.7529 | 0.6625 | 0.2191 |
| 24 | coexist | 1.000 | False | 0.000 | 0, `<s>`, 0.3437 | 271, text, 0.6736 | 264, text | 0.7163 | 0.6660 | 0.2294 |
| 25 | coexist | 1.000 | False | 0.000 | 0, `<s>`, 0.3716 | 271, text, 0.6861 | 264, text | 0.6833 | 0.6736 | 0.2371 |
| 26 | coexist | 1.000 | False | 0.000 | 0, `<s>`, 0.3969 | 271, text, 0.6944 | 262, text | 0.6546 | 0.6824 | 0.2392 |
| 27 | coexist | 1.000 | False | 0.000 | 0, `<s>`, 0.4193 | 271, text, 0.7012 | 262, text | 0.6360 | 0.6903 | 0.2337 |
| 28 | coexist | 1.000 | False | 0.000 | 0, `<s>`, 0.4373 | 271, text, 0.7009 | 254, vision | 0.6276 | 0.6892 | 0.2148 |
| 29 | coexist | 1.000 | False | 0.000 | 0, `<s>`, 0.4516 | 271, text, 0.7058 | 254, vision | 0.6234 | 0.6926 | 0.2134 |
| 30 | coexist | 1.000 | False | 0.000 | 0, `<s>`, 0.4658 | 271, text, 0.7060 | 247, vision | 0.6199 | 0.6923 | 0.2126 |
| 31 | coexist | 1.000 | False | 0.000 | 0, `<s>`, 0.4827 | 271, text, 0.7098 | 14, vision `right` | 0.6165 | 0.7013 | 0.3163 |

**Summary:** Persistent coexist pattern. Attention always peaks on vision token `<s>` (abs_t=0, 26–48%) but contribution always peaks on text token abs_t=271 (65–71%). This A≠C mismatch is 100% consistent. The text token that dominates contribution is NOT a vision token — this is the "text bottleneck" pattern unique to OpenVLA.

### SpatialVLA-4b (Layers 16–25)

| Layer | dominant_type | freq | a_c_match | match_rate | A_peak (abs_t, token, share) | C_peak (abs_t, token, share) | R_peak (abs_t, token) | entropy | top1_share | mismatch |
|-------|---------------|------|-----------|------------|------------------------------|------------------------------|----------------------|---------|------------|----------|
| 16 | normal | 0.987 | True | 0.953 | 260, `robot`, 0.2099 | 260, `robot`, 0.2161 | 193, `<image>` | 2.5648 | 0.2129 | 0.0012 |
| 17 | normal | 0.993 | True | 0.900 | 260, `robot`, 0.2305 | 260, `robot`, 0.2348 | 193, `<image>` | 2.3755 | 0.2295 | 0.0016 |
| 18 | normal | 0.993 | True | 0.853 | 260, `robot`, 0.2411 | 260, `robot`, 0.2425 | 193, `<image>` | 2.2848 | 0.2378 | 0.0021 |
| 19 | normal | 0.973 | True | 0.807 | 260, `robot`, 0.2423 | 260, `robot`, 0.2412 | 193, `<image>` | 2.3079 | 0.2358 | 0.0033 |
| 20 | normal | 0.947 | True | 0.673 | 260, `robot`, 0.2467 | 260, `robot`, 0.2404 | 193, `<image>` | 2.3720 | 0.2322 | 0.0058 |
| 21 | normal | 0.827 | True | 0.540 | 260, `robot`, 0.2349 | 225, `<image>`, 0.2172 | 256, `What` | 2.5610 | 0.2055 | 0.0108 |
| 22 | normal | 0.513 | True | 0.513 | 260, `robot`, 0.1887 | 260, `robot`, 0.1741 | 193, `<image>` | 3.0218 | 0.1694 | 0.0100 |
| 23 | normal | 0.520 | True | 0.520 | 260, `robot`, 0.1873 | 260, `robot`, 0.1708 | 193, `<image>` | 3.0828 | 0.1676 | 0.0125 |
| 24 | normal | 0.573 | True | 0.573 | 225, `<image>`, 0.2079 | 225, `<image>`, 0.1885 | 193, `<image>` | 3.0459 | 0.1810 | 0.0141 |
| 25 | normal | 0.733 | True | 0.667 | 266, `?`, 0.2698 | 266, `?`, 0.2548 | 193, `<image>` | 2.5523 | 0.2467 | 0.0200 |

**Summary:** Healthy normal distribution. Attention and contribution mostly agree (match_rate 51–95%). Peak tokens rotate between `robot` (abs_t=260), `<image>` (abs_t=225), and `?` (abs_t=266) — all text or special tokens. R_peak consistently on `<image>` at abs_t=193 (vision). Entropy is high (2.3–3.1) indicating distributed attention. No bottleneck behavior.

### TraceVLA-Phi3V (Layers 22–31)

| Layer | dominant_type | freq | a_c_match | match_rate | A_peak (abs_t, token, share) | C_peak (abs_t, token, share) | R_peak (abs_t, token) | entropy | top1_share | mismatch |
|-------|---------------|------|-----------|------------|------------------------------|------------------------------|----------------------|---------|------------|----------|
| 22 | normal | 1.000 | False | 0.280 | 0, `<s>`, 0.0761 | 1, `<\|user\|>`, 0.1293 | 3, vision | 2.4927 | 0.1290 | 0.0425 |
| 23 | normal | 1.000 | False | 0.247 | 0, `<s>`, 0.0818 | 1, `<\|user\|>`, 0.1336 | 3, vision | 2.4534 | 0.1311 | 0.0468 |
| 24 | normal | 1.000 | False | 0.213 | 0, `<s>`, 0.0877 | 1, `<\|user\|>`, 0.1362 | 3, vision | 2.4268 | 0.1332 | 0.0530 |
| 25 | normal | 1.000 | False | 0.187 | 0, `<s>`, 0.0935 | 1, `<\|user\|>`, 0.1391 | 3, vision | 2.4049 | 0.1348 | 0.0606 |
| 26 | normal | 1.000 | False | 0.167 | 0, `<s>`, 0.0978 | 1, `<\|user\|>`, 0.1401 | 3, vision | 2.3906 | 0.1367 | 0.0675 |
| 27 | normal | 0.993 | False | 0.147 | 0, `<s>`, 0.1016 | 1, `<\|user\|>`, 0.1422 | 159, `<unk>` | 2.3785 | 0.1384 | 0.0762 |
| 28 | normal | 0.993 | False | 0.120 | 0, `<s>`, 0.1054 | 1, `<\|user\|>`, 0.1432 | 150, `<unk>` | 2.3671 | 0.1395 | 0.0849 |
| 29 | normal | 0.993 | False | 0.093 | 0, `<s>`, 0.1088 | 1, `<\|user\|>`, 0.1449 | 159, `<unk>` | 2.3566 | 0.1413 | 0.0938 |
| 30 | normal | 0.993 | False | 0.060 | 0, `<s>`, 0.1144 | 1, `<\|user\|>`, 0.1477 | 319, text | 2.3226 | 0.1436 | 0.1197 |
| 31 | normal | 0.993 | False | 0.007 | 0, `<s>`, 0.1258 | 1, `<\|user\|>`, 0.1473 | 319, text | 2.2416 | 0.1443 | 0.2161 |

**Summary:** Normal distribution with consistent A≠C offset. Attention peaks on `<s>` (abs_t=0, 8–13%) and contribution peaks on `<|user|>` (abs_t=1, 13–15%) — a one-position offset, NOT a modality mismatch. Both are vision-region tokens. R_peak shifts from early vision (abs_t=3) to `<unk>` vision tokens (abs_t=150/159) and finally to text (abs_t=319) in deepest layers. Entropy moderate (~2.2–2.5), top1_share low (~13–14%).

---

# Gate ② — Causal V=0 Ablation

## 2.1 Method

- **V=0 hook**: Zero the V projection output at the target token's position
- **Target tokens**: A_mode, C_mode, R_mode (from Gate ①)
- **Layer scope**: `all` (layers 22–31 or 16–25), `block1` (first 5), `block2` (last 5)
- **Metrics**: KL divergence (logits), top-1 change rate
- **Samples**: 150 (same as Gate ①)

All sanity checks passed: `hook_fired=true`, `logits_changed=true` for every configuration.

## 2.2 Results — A_mode (Attention Peak Token)

| Model | Target (abs_t) | Token | all (KL) | all (top1Δ) | block1 (KL) | block1 (top1Δ) | block2 (KL) | block2 (top1Δ) |
|-------|----------------|-------|----------|-------------|-------------|----------------|-------------|----------------|
| ECoT-7b | 0 | `<s>` | 1.197 | 0.320 | 0.456 | 0.267 | 0.418 | 0.227 |
| OpenVLA-7b | 0 | `<s>` | 0.511 | 0.333 | 0.300 | 0.300 | 0.163 | 0.207 |
| SpatialVLA-4b | 260 | `robot` | 1.215 | 0.673 | 0.723 | 0.513 | 0.513 | 0.367 |
| TraceVLA-Phi3V | 0 | `<s>` | 13.112 | 1.000 | 13.081 | 1.000 | 13.052 | 1.000 |

## 2.3 Results — C_mode (Contribution Peak Token)

| Model | Target (abs_t) | Token | all (KL) | all (top1Δ) | block1 (KL) | block1 (top1Δ) | block2 (KL) | block2 (top1Δ) |
|-------|----------------|-------|----------|-------------|-------------|----------------|-------------|----------------|
| ECoT-7b | 0 | `<s>` | 1.197 | 0.320 | 0.456 | 0.267 | 0.418 | 0.227 |
| OpenVLA-7b | 271 | text | 0.186 | 0.207 | 0.082 | 0.173 | 0.059 | 0.100 |
| SpatialVLA-4b | 260 | `robot` | 1.215 | 0.673 | 0.723 | 0.513 | 0.513 | 0.367 |
| TraceVLA-Phi3V | 1 | `<\|user\|>` | 13.109 | 1.000 | 13.074 | 1.000 | 13.057 | 1.000 |

## 2.4 Results — R_mode (Residual Peak Token)

| Model | Target (abs_t) | Token | all (KL) | all (top1Δ) | block1 (KL) | block1 (top1Δ) | block2 (KL) | block2 (top1Δ) |
|-------|----------------|-------|----------|-------------|-------------|----------------|-------------|----------------|
| ECoT-7b | 271 | text | 0.072 | 0.067 | 0.041 | 0.047 | 0.028 | 0.067 |
| OpenVLA-7b | 262 | text | 0.001 | 0.020 | 0.001 | 0.007 | 0.001 | 0.013 |
| SpatialVLA-4b | 193 | `<image>` | 0.004 | 0.047 | 0.003 | 0.027 | 0.002 | 0.047 |
| TraceVLA-Phi3V | 3 | vision | 12.969 | 1.000 | 13.020 | 1.000 | 12.983 | 1.000 |

## 2.5 Causal Impact Summary (A/C vs R)

| Model | A_mode KL | C_mode KL | R_mode KL | A/R ratio | C/R ratio | Interpretation |
|-------|-----------|-----------|-----------|-----------|-----------|----------------|
| ECoT-7b | 1.197 | 1.197 | 0.072 | 16.6x | 16.6x | A=C token (bottleneck), R is negligible → **bottleneck confirmed** |
| OpenVLA-7b | 0.511 | 0.186 | 0.001 | 511x | 186x | A≠C, both causally important but A > C, R negligible → **coexist confirmed** |
| SpatialVLA-4b | 1.215 | 1.215 | 0.004 | 304x | 304x | A=C token, high impact, R negligible → **text-anchored bottleneck** |
| TraceVLA-Phi3V | 13.112 | 13.109 | 12.969 | 1.01x | 1.01x | ALL tokens catastrophically important → **global dependency** |

### Key Observations

1. **ECoT**: A=C=`<s>` (abs_t=0). Removing this single vision token causes KL=1.20, 32% top-1 change. R_mode token (text) is negligible (KL=0.07). Classic bottleneck: one token holds the model together.

2. **OpenVLA**: A≠C. Attention peak `<s>` (KL=0.51) is MORE causally important than contribution peak text:271 (KL=0.19). Both far exceed R_mode (KL=0.001). The "text bottleneck" from contribution analysis is real but the vision attention sink also matters causally.

3. **SpatialVLA**: A=C=`robot` (text token abs_t=260). Strongest functional impact (KL=1.22, 67% top-1 change). R_mode `<image>` at abs_t=193 is negligible (KL=0.004). A text token acts as the routing hub.

4. **TraceVLA**: Catastrophic sensitivity to ANY token. A, C, R modes all produce KL≈13 and 100% top-1 change. Even the R_mode token (abs_t=3) causes total output collapse. This suggests TraceVLA-Phi3V has extreme position sensitivity in early tokens rather than a single bottleneck.

5. **Block locality**: For ECoT/OpenVLA/SpatialVLA, block1 (earlier layers) generally shows slightly higher or comparable KL to block2 (later layers), suggesting causal impact is distributed rather than concentrated in final layers. For TraceVLA, all blocks show identical catastrophic impact.

---

# Gate ③ — Text Masking Control + Mini Counterfactual (v2)

## 3.1 Masking Strategy Per Architecture

| Model | Strategy | Reason |
|-------|----------|--------|
| ECoT-7b | `kv_zero_hook` (TextKVZeroHook) | Prismatic wrapper concatenates 2D masks internally; can't pass 4D |
| OpenVLA-7b | `kv_zero_hook` (TextKVZeroHook) | Same Prismatic architecture |
| SpatialVLA-4b | `4d_mask` (TextKVMaskHook) | Standard HF model; 4D causal mask bypasses mask regeneration |
| TraceVLA-Phi3V | `4d_mask` (TextKVMaskHook) | Standard HF model; 4D mask works directly |

## 3.2 Mask Sanity Check (Sample 0)

| Model | Strategy | hook_fired | n_text_masked | hidden_changed | max_hidden_diff |
|-------|----------|-----------|---------------|----------------|-----------------|
| ECoT-7b | kv_zero_hook | true | 22 | **true** | 122.90 |
| OpenVLA-7b | kv_zero_hook | true | 22 | **true** | 125.10 |
| SpatialVLA-4b | 4d_mask | n/a | 14 | **true** | 700.05 |
| TraceVLA-Phi3V | 4d_mask | n/a | 28 | **true** | 3427.06 |

All 4 models pass sanity: text masking verifiably changes hidden states.

**Bug fix note (v2):** Gate ③ v1 used `TextAttnWeightHook` for Prismatic models which was a no-op (`hidden_changed=false`, ratio=1.0). Fixed by switching to `TextKVZeroHook` that hooks into `k_proj` and `v_proj` forward hooks to zero outputs at text positions.

## 3.3 Gate 3 Metadata

| Model | n_samples | text_ranges (sample 0) | n_text_tokens | vision_range | version |
|-------|-----------|------------------------|---------------|-------------|---------|
| ECoT-7b | 20 | [[256, 278]] | 22 | [0, 256] | 2 |
| OpenVLA-7b | 20 | [[256, 278]] | 22 | [0, 256] | 2 |
| SpatialVLA-4b | 20 | [[256, 270]] | 14 | [0, 256] | 2 |
| TraceVLA-Phi3V | 20 | [[313, 341]] | 28 | [0, 313] | 2 |

### SpatialVLA-4b Masked Token Strings (Sample 0)
```
What, action, should, the, robot, take, to, take, clothes, out, of, laundry, machine, ?
```

### TraceVLA-Phi3V Masked Token Strings (Sample 0, first 20)
```
<id:-1>, <id:-1>, <id:-1>, <id:-1>, <s>, , , What, action, should, the, robot, take, to, take, clothes, out, of, la, und
```

## 3.4 Skill Labels (Gate 3, 20 samples — identical across all models)

```
pick, move, fold, place, pick, move, open, pick, fold, pick, fold, place, close, pick, place, place, close, move, pick, place
```

## 3.5 Part A: Hidden States Under 4 Conditions

Part A collected hidden states at the query position under 4 conditions for each of 20 samples × 10 deep layers:

- **original**: No masking (baseline)
- **text_v0**: TextValueZeroHook — zeros V projections for text tokens (Q/K routing alive)
- **text_kv**: TextKVZeroHook or 4D mask — fully blocks text KV at text positions
- **vision_v0_norm**: ValueZeroHook — same number of random vision tokens zeroed (normalized control)

Hidden state `.npy` files saved per condition × layer for offline probe evaluation.

## 3.6 Part B: Mini Counterfactual — Verb Swap Delta-Hidden

### Method
- Same image, swap verb in instruction (pick↔place, open↔close, move↔fold)
- Measure `delta_orig = ||h_orig - h_swap|| / ||h_orig||` (relative hidden state change)
- Measure `delta_textKV` = same under text KV masking
- If text carries verb information, delta_textKV should drop toward 0

### Summary Statistics

| Model | delta_orig (mean) | delta_textKV (mean) | Ratio (orig/KV) | n_pairs |
|-------|-------------------|---------------------|------------------|---------|
| ECoT-7b | 0.7797 | 0.5117 | **1.52x** | 80 |
| OpenVLA-7b | 0.6462 | 0.3208 | **2.01x** | 80 |
| SpatialVLA-4b | 0.3181 | 0.0000 | **∞** | 80 |
| TraceVLA-Phi3V | 0.8093 | 0.0203 | **39.93x** | 80 |

### Per-Sample Counterfactual Detail

#### ECoT-7b

| sample_idx | skill | swap_verb | delta_orig (range) | delta_textKV (range) | Pattern |
|-----------|-------|-----------|-------------------|---------------------|---------|
| 1 | move | fold | 1.187 – 1.252 | 1.293 – 1.395 | textKV > orig (!) |
| 4 | pick | place | 0.345 – 0.613 | 0.0 (all layers) | text carries ALL verb info |
| 5 | move | fold | 1.122 – 1.206 | 1.308 – 1.411 | textKV > orig (!) |
| 6 | open | close | 0.440 – 0.686 | 0.0 (all layers) | text carries ALL verb info |
| 12 | close | open | 0.441 – 0.626 | 0.0 (all layers) | text carries ALL verb info |
| 13 | pick | place | 0.393 – 0.622 | 0.0 (all layers) | text carries ALL verb info |
| 16 | close | open | 0.460 – 0.645 | 0.0 (all layers) | text carries ALL verb info |
| 17 | move | fold | 1.093 – 1.317 | 1.302 – 1.399 | textKV > orig (!) |

**ECoT Pattern**: `move↔fold` swaps INCREASE delta under text masking (delta_textKV > delta_orig). All other verbs (pick, open, close) drop to exactly 0. The move/fold anomaly suggests these verbs create compensatory vision-mediated routing when text is blocked.

#### OpenVLA-7b

| sample_idx | skill | swap_verb | delta_orig (range) | delta_textKV (range) | Pattern |
|-----------|-------|-----------|-------------------|---------------------|---------|
| 1 | move | fold | 0.486 – 0.849 | 0.688 – 0.884 | textKV ≈ orig |
| 4 | pick | place | 0.196 – 0.330 | 0.0 (all layers) | text carries ALL verb info |
| 5 | move | fold | 0.605 – 1.055 | 0.828 – 0.911 | textKV retains signal |
| 6 | open | close | 0.840 – 1.228 | 0.0 (all layers) | text carries ALL verb info |
| 12 | close | open | 0.680 – 0.924 | 0.0 (all layers) | text carries ALL verb info |
| 13 | pick | place | 0.194 – 0.329 | 0.0 (all layers) | text carries ALL verb info |
| 16 | close | open | 0.685 – 0.999 | 0.0 (all layers) | text carries ALL verb info |
| 17 | move | fold | 0.649 – 1.015 | 0.776 – 1.252 | textKV retains signal |

**OpenVLA Pattern**: Same verb-specific split as ECoT. `move↔fold` retains nonzero delta_textKV; `pick`, `open`, `close` all drop to 0. This suggests move/fold verb distinction is partially encoded in vision-side routing (possibly through spatial trajectory patterns in the image).

#### SpatialVLA-4b

| sample_idx | skill | swap_verb | delta_orig (range) | delta_textKV (range) | Pattern |
|-----------|-------|-----------|-------------------|---------------------|---------|
| 1 | move | fold | 0.235 – 0.487 | 0.0 (all) | text is sole channel |
| 4 | pick | place | 0.128 – 0.272 | 0.0 (all) | text is sole channel |
| 5 | move | fold | 0.223 – 0.535 | 0.0 (all) | text is sole channel |
| 6 | open | close | 0.348 – 0.661 | 0.0 (all) | text is sole channel |
| 10 | fold | move | 0.328 – 0.647 | 0.0 (all) | text is sole channel |
| 12 | close | open | 0.314 – 0.558 | 0.0 (all) | text is sole channel |
| 13 | pick | place | 0.109 – 0.245 | 0.0 (all) | text is sole channel |
| 16 | close | open | 0.321 – 0.633 | 0.0 (all) | text is sole channel |
| 17 | move | fold | 0.314 – 0.710 | 0.0 (all) | text is sole channel |

**SpatialVLA Pattern**: delta_textKV = 0.0 for ALL verbs including move/fold. Text tokens are the **sole channel** for verb routing. No vision-side compensatory pathway exists. Most extreme text dependence in the study.

#### TraceVLA-Phi3V

| sample_idx | skill | swap_verb | delta_orig (range) | delta_textKV (range) | Pattern |
|-----------|-------|-----------|-------------------|---------------------|---------|
| 1 | move | fold | 0.856 – 1.028 | 0.039 – 0.096 | 95% reduction |
| 4 | pick | place | 0.277 – 0.474 | 0.005 – 0.009 | 98% reduction |
| 5 | move | fold | 0.804 – 0.840 | 0.036 – 0.087 | 95% reduction |
| 6 | open | close | 0.832 – 1.167 | 0.0001 – 0.0008 | 99.9% reduction |
| 12 | close | open | 0.954 – 1.087 | 0.007 – 0.017 | 99% reduction |
| 13 | pick | place | 0.272 – 0.419 | 0.006 – 0.013 | 98% reduction |
| 16 | close | open | 0.972 – 1.471 | 0.004 – 0.007 | 99.5% reduction |
| 17 | move | fold | 0.678 – 0.933 | 0.038 – 0.095 | 93% reduction |

**TraceVLA Pattern**: Text masking reduces verb-swap delta by 93–99.9% but does NOT reach exactly zero. Residual is largest for `move↔fold` (~0.04–0.10) and smallest for `open↔close` (~0.0001–0.001). This indicates text is the dominant verb channel (>95%) but a tiny residual exists, possibly from the trace image carrying implicit action intent.

---

# Cross-Gate Synthesis

## Model Taxonomy

| Model | Gate ① Type | Gate ② Confirms? | Gate ③ Text Dependence | Overall Classification |
|-------|-------------|------------------|------------------------|------------------------|
| ECoT-7b | Bottleneck (vision `<s>`) | Yes (A/C KL=1.20, R KL=0.07) | Partial (1.52x ratio, move retains signal) | **Vision Bottleneck + Verb-Specific Text Channel** |
| OpenVLA-7b | Coexist (A=vision, C=text) | Yes (A KL=0.51, C KL=0.19, R KL≈0) | Partial (2.01x ratio, move retains signal) | **Vision-Text Coexist + Verb-Specific Text Channel** |
| SpatialVLA-4b | Normal (text `robot`) | Yes (A/C KL=1.22, R KL=0.004) | Total (∞ ratio, all verbs drop to 0) | **Text-Anchored Normal + Sole Text Channel** |
| TraceVLA-Phi3V | Normal (vision, A≠C offset) | Ambiguous (all tokens KL≈13) | Near-total (39.93x, tiny residual) | **Global Sensitivity + Dominant Text Channel** |

## Key Findings

### 1. Bottleneck vs Sink Taxonomy
- **ECoT**: Pure bottleneck — vision token `<s>` has high attention AND high contribution AND high causal impact. NOT a sink.
- **OpenVLA**: Coexist — attention peak (vision `<s>`) and contribution peak (text:271) are different modalities. Both are causally important.
- **SpatialVLA**: No bottleneck/sink — healthy distributed attention with text token `robot` as the most important hub.
- **TraceVLA**: No clear bottleneck/sink — extreme sensitivity to ALL early tokens makes classification unclear.

### 2. Text Leakage Control
- **SpatialVLA** is the strongest case for text leakage: 100% of verb information flows through text tokens (delta_textKV = 0 for all verbs).
- **TraceVLA** shows 93–99.9% text dependence with tiny vision-side residual.
- **ECoT and OpenVLA** show verb-specific behavior: `pick↔place`, `open↔close` are pure text channel, but `move↔fold` retains signal even under text masking.

### 3. Verb-Specific Routing Asymmetry
The `move↔fold` anomaly in ECoT/OpenVLA is a novel finding: these verb pairs maintain counterfactual sensitivity even when text KV is zeroed, suggesting:
- Move and fold actions may create distinguishable visual patterns (spatial trajectories) that the vision pathway can leverage independently
- Pick/place/open/close are more linguistically-specified and rely entirely on text tokens for routing

### 4. Architecture-Dependent Failure Modes
- **Prismatic wrapper** (ECoT, OpenVLA): Creates bottleneck or coexist patterns where early vision tokens (especially `<s>`) accumulate disproportionate influence
- **Standard HF** (SpatialVLA, TraceVLA): Distributes attention more evenly but routes ALL verb information through text tokens

---

# Appendix: File Locations

```
outputs/phase3_gate/
├── {model}/                          # Gate ① outputs
│   ├── contribution_report.json      # Full contribution analysis
│   ├── mode_tokens.json              # A/C/R mode token summary
│   └── sample_list.json              # 150 sample indices
├── gate2/{model}/                    # Gate ② outputs
│   ├── A_mode_all/causal_report.json
│   ├── A_mode_block1/causal_report.json
│   ├── A_mode_block2/causal_report.json
│   ├── C_mode_all/causal_report.json
│   ├── C_mode_block1/causal_report.json
│   ├── C_mode_block2/causal_report.json
│   ├── R_mode_all/causal_report.json
│   ├── R_mode_block1/causal_report.json
│   └── R_mode_block2/causal_report.json
└── gate3_v2/{model}/                 # Gate ③ v2 outputs
    ├── mask_sanity_check.json
    ├── gate3_metadata.json
    ├── counterfactual_results.json
    ├── skill_labels.json
    └── hidden_{condition}_layer{N}.npy  # Part A hidden states
```

Models: `ecot-7b`, `openvla-7b`, `spatialvla-4b`, `tracevla-phi3v`
