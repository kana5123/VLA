# Phase 3 Gate Check — Full Results (Unabridged)

**Date:** 2026-02-26
**Models:** ECoT-7B, OpenVLA-7B, SpatialVLA-4B, TraceVLA-Phi3V
**Data:** BridgeData V2, 150 balanced samples (6 skills x 25)
**Hardware:** 8x H100-80GB GPUs
**Conda env:** interp (torch 2.5.1+cu121, transformers 4.57.6)

---

# Gate 1 — Contribution Analysis (150 Balanced Samples)

## ecot-7b

- **n_samples:** 150
- **n_layers:** 32
- **deep_layers:** [22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

### Sequence Boundaries

| Field | Value |
|-------|-------|
| vision_start | 0 |
| vision_end | 256 |
| text_start | 256 |
| text_end | 278 |
| total_seq_len | 278 |
| num_vision_tokens | 256 |
| num_text_tokens | 22 |
| pre_image_tokens | 0 |
| text_query_ranges | [[256, 278]] |

### Mode Tokens

| Mode | abs_t | token_type | token_str | freq | a_share_mean | c_share_mean |
|------|-------|------------|-----------|------|-------------|-------------|
| A_mode | 0 | vision | `<s>` | 1.0000 | N/A | N/A |
| C_mode | 0 | vision | `<s>` | 1.0000 | N/A | N/A |
| R_mode | 271 | text | `None` | 0.1567 | N/A | N/A |

### Skill Distribution

| Skill | Count |
|-------|-------|
| close | 25 |
| fold | 25 |
| move | 25 |
| open | 25 |
| pick | 25 |
| place | 25 |

### Per-Layer Analysis

#### Layer 22

- **dominant_type:** bottleneck
- **frequency:** 1.0
- **a_c_match:** True
- **a_c_match_rate:** 1.0
- **mean_entropy:** 0.12278109493354956
- **mean_top1_share:** 0.9634697012106578
- **mean_mismatch:** 0.07521559086938699

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.771204 |
| c_share | 0.991796 |
| sink_score | -0.251564 |
| vision_j | 0 |
| token_str | <s> |
| phi | 51.716385 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.771204 |
| c_share | 0.991796 |
| sink_score | -0.251564 |
| vision_j | 0 |
| token_str | <s> |
| phi | 51.716385 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 271 |
| token_type | text |
| a_share | 0.019001 |
| c_share | 0.000308 |
| sink_score | 4.123374 |
| phi | 41.069008 |

#### Layer 23

- **dominant_type:** bottleneck
- **frequency:** 1.0
- **a_c_match:** True
- **a_c_match_rate:** 1.0
- **mean_entropy:** 0.22892444009582202
- **mean_top1_share:** 0.9510824477672577
- **mean_mismatch:** 0.038856344893574715

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.835375 |
| c_share | 0.979743 |
| sink_score | -0.159409 |
| vision_j | 0 |
| token_str | <s> |
| phi | 51.717499 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.835375 |
| c_share | 0.979743 |
| sink_score | -0.159409 |
| vision_j | 0 |
| token_str | <s> |
| phi | 51.717499 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 271 |
| token_type | text |
| a_share | 0.011272 |
| c_share | 0.000714 |
| sink_score | 2.759538 |
| phi | 40.589138 |

#### Layer 24

- **dominant_type:** bottleneck
- **frequency:** 1.0
- **a_c_match:** True
- **a_c_match_rate:** 1.0
- **mean_entropy:** 0.3324857192238172
- **mean_top1_share:** 0.9359922254085541
- **mean_mismatch:** 0.05515251969297727

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.757831 |
| c_share | 0.965173 |
| sink_score | -0.241847 |
| vision_j | 0 |
| token_str | <s> |
| phi | 51.718479 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.757831 |
| c_share | 0.965173 |
| sink_score | -0.241847 |
| vision_j | 0 |
| token_str | <s> |
| phi | 51.718479 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 271 |
| token_type | text |
| a_share | 0.013055 |
| c_share | 0.000792 |
| sink_score | 2.802629 |
| phi | 39.867954 |

#### Layer 25

- **dominant_type:** bottleneck
- **frequency:** 1.0
- **a_c_match:** True
- **a_c_match_rate:** 1.0
- **mean_entropy:** 0.35273731023073196
- **mean_top1_share:** 0.9329344471295674
- **mean_mismatch:** 0.04363650149355332

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.803971 |
| c_share | 0.966690 |
| sink_score | -0.184315 |
| vision_j | 0 |
| token_str | <s> |
| phi | 51.718464 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.803971 |
| c_share | 0.966690 |
| sink_score | -0.184315 |
| vision_j | 0 |
| token_str | <s> |
| phi | 51.718464 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 271 |
| token_type | text |
| a_share | 0.012221 |
| c_share | 0.000952 |
| sink_score | 2.552472 |
| phi | 38.826187 |

#### Layer 26

- **dominant_type:** bottleneck
- **frequency:** 0.9933333333333333
- **a_c_match:** True
- **a_c_match_rate:** 1.0
- **mean_entropy:** 0.535088159441948
- **mean_top1_share:** 0.9061445679267247
- **mean_mismatch:** 0.05396421946585178

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.745229 |
| c_share | 0.949793 |
| sink_score | -0.242553 |
| vision_j | 0 |
| token_str | <s> |
| phi | 51.718452 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.745229 |
| c_share | 0.949793 |
| sink_score | -0.242553 |
| vision_j | 0 |
| token_str | <s> |
| phi | 51.718452 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 271 |
| token_type | text |
| a_share | 0.012979 |
| c_share | 0.001199 |
| sink_score | 2.381602 |
| phi | 37.580845 |

#### Layer 27

- **dominant_type:** bottleneck
- **frequency:** 1.0
- **a_c_match:** True
- **a_c_match_rate:** 1.0
- **mean_entropy:** 0.2630232231815656
- **mean_top1_share:** 0.9461174702644348
- **mean_mismatch:** 0.05477899491786957

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.814336 |
| c_share | 0.979758 |
| sink_score | -0.184932 |
| vision_j | 0 |
| token_str | <s> |
| phi | 51.724285 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.814336 |
| c_share | 0.979758 |
| sink_score | -0.184932 |
| vision_j | 0 |
| token_str | <s> |
| phi | 51.724285 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 271 |
| token_type | text |
| a_share | 0.012783 |
| c_share | 0.000683 |
| sink_score | 2.929321 |
| phi | 36.747456 |

#### Layer 28

- **dominant_type:** bottleneck
- **frequency:** 1.0
- **a_c_match:** True
- **a_c_match_rate:** 1.0
- **mean_entropy:** 0.25625208238760633
- **mean_top1_share:** 0.9489048937956492
- **mean_mismatch:** 0.047251448296010495

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.852636 |
| c_share | 0.983043 |
| sink_score | -0.142321 |
| vision_j | 0 |
| token_str | <s> |
| phi | 51.726154 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.852636 |
| c_share | 0.983043 |
| sink_score | -0.142321 |
| vision_j | 0 |
| token_str | <s> |
| phi | 51.726154 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 271 |
| token_type | text |
| a_share | 0.017792 |
| c_share | 0.001085 |
| sink_score | 2.796891 |
| phi | 33.809029 |

#### Layer 29

- **dominant_type:** bottleneck
- **frequency:** 1.0
- **a_c_match:** True
- **a_c_match_rate:** 1.0
- **mean_entropy:** 0.3298047761122386
- **mean_top1_share:** 0.9371200672785441
- **mean_mismatch:** 0.07417462373773256

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.655476 |
| c_share | 0.957848 |
| sink_score | -0.379327 |
| vision_j | 0 |
| token_str | <s> |
| phi | 51.728008 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.655476 |
| c_share | 0.957848 |
| sink_score | -0.379327 |
| vision_j | 0 |
| token_str | <s> |
| phi | 51.728008 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 271 |
| token_type | text |
| a_share | 0.020899 |
| c_share | 0.001433 |
| sink_score | 2.679948 |
| phi | 31.740158 |

#### Layer 30

- **dominant_type:** bottleneck
- **frequency:** 1.0
- **a_c_match:** True
- **a_c_match_rate:** 1.0
- **mean_entropy:** 0.28138127565383914
- **mean_top1_share:** 0.9463368372122447
- **mean_mismatch:** 0.07203245806197325

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.704529 |
| c_share | 0.968941 |
| sink_score | -0.318674 |
| vision_j | 0 |
| token_str | <s> |
| phi | 51.563858 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.704529 |
| c_share | 0.968941 |
| sink_score | -0.318674 |
| vision_j | 0 |
| token_str | <s> |
| phi | 51.563858 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 271 |
| token_type | text |
| a_share | 0.015454 |
| c_share | 0.001086 |
| sink_score | 2.655445 |
| phi | 25.840214 |

#### Layer 31

- **dominant_type:** bottleneck
- **frequency:** 1.0
- **a_c_match:** True
- **a_c_match_rate:** 1.0
- **mean_entropy:** 0.05619201015681028
- **mean_top1_share:** 0.9902228820323944
- **mean_mismatch:** 0.1667621866861979

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.608941 |
| c_share | 0.997109 |
| sink_score | -0.493139 |
| vision_j | 0 |
| token_str | <s> |
| phi | 50.543926 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.608941 |
| c_share | 0.997109 |
| sink_score | -0.493139 |
| vision_j | 0 |
| token_str | <s> |
| phi | 50.543926 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 258 |
| token_type | text |
| a_share | 0.011726 |
| c_share | 0.000079 |
| sink_score | 4.998527 |
| phi | 5.545178 |

---

## openvla-7b

- **n_samples:** 150
- **n_layers:** 32
- **deep_layers:** [22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

### Sequence Boundaries

| Field | Value |
|-------|-------|
| vision_start | 0 |
| vision_end | 256 |
| text_start | 256 |
| text_end | 278 |
| total_seq_len | 278 |
| num_vision_tokens | 256 |
| num_text_tokens | 22 |
| pre_image_tokens | 0 |
| text_query_ranges | [[256, 278]] |

### Mode Tokens

| Mode | abs_t | token_type | token_str | freq | a_share_mean | c_share_mean |
|------|-------|------------|-----------|------|-------------|-------------|
| A_mode | 0 | vision | `<s>` | 1.0000 | N/A | N/A |
| C_mode | 271 | text | `None` | 0.1800 | N/A | N/A |
| R_mode | 262 | text | `None` | 0.0787 | N/A | N/A |

### Skill Distribution

| Skill | Count |
|-------|-------|
| close | 25 |
| fold | 25 |
| move | 25 |
| open | 25 |
| pick | 25 |
| place | 25 |

### Per-Layer Analysis

#### Layer 22

- **dominant_type:** coexist
- **frequency:** 1.0
- **a_c_match:** False
- **a_c_match_rate:** 0.0
- **mean_entropy:** 0.6583793346087138
- **mean_top1_share:** 0.6777702609697978
- **mean_mismatch:** 0.21019748250643414

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.457617 |
| c_share | 0.325675 |
| sink_score | 0.340135 |
| vision_j | 0 |
| token_str | <s> |
| phi | 50.589710 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 271 |
| token_type | text |
| a_share | 0.160124 |
| c_share | 0.685799 |
| sink_score | -1.454634 |
| phi | 50.713024 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 264 |
| token_type | text |
| a_share | 0.000804 |
| c_share | 0.000004 |
| sink_score | 5.259184 |
| phi | 43.882736 |

#### Layer 23

- **dominant_type:** coexist
- **frequency:** 1.0
- **a_c_match:** False
- **a_c_match_rate:** 0.0
- **mean_entropy:** 0.7281499103705088
- **mean_top1_share:** 0.6632779161135356
- **mean_mismatch:** 0.21583571434020996

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.406161 |
| c_share | 0.324664 |
| sink_score | 0.223961 |
| vision_j | 0 |
| token_str | <s> |
| phi | 50.590012 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 271 |
| token_type | text |
| a_share | 0.153164 |
| c_share | 0.663216 |
| sink_score | -1.465591 |
| phi | 50.713306 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 262 |
| token_type | text |
| a_share | 0.000009 |
| c_share | 0.000000 |
| sink_score | 4.626628 |
| phi | 43.423634 |

#### Layer 24

- **dominant_type:** coexist
- **frequency:** 1.0
- **a_c_match:** False
- **a_c_match_rate:** 0.0
- **mean_entropy:** 0.7233652087052663
- **mean_top1_share:** 0.6674936032295227
- **mean_mismatch:** 0.23718253364165623

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.411791 |
| c_share | 0.328107 |
| sink_score | 0.227175 |
| vision_j | 0 |
| token_str | <s> |
| phi | 50.590355 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 271 |
| token_type | text |
| a_share | 0.144234 |
| c_share | 0.670841 |
| sink_score | -1.537096 |
| phi | 50.713589 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 262 |
| token_type | text |
| a_share | 0.000915 |
| c_share | 0.000016 |
| sink_score | 4.074576 |
| phi | 38.719940 |

#### Layer 25

- **dominant_type:** coexist
- **frequency:** 1.0
- **a_c_match:** False
- **a_c_match_rate:** 0.0
- **mean_entropy:** 0.7702118746439616
- **mean_top1_share:** 0.6593360877037049
- **mean_mismatch:** 0.21803384860356648

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.423181 |
| c_share | 0.328474 |
| sink_score | 0.253343 |
| vision_j | 0 |
| token_str | <s> |
| phi | 50.590363 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 271 |
| token_type | text |
| a_share | 0.149295 |
| c_share | 0.656793 |
| sink_score | -1.481447 |
| phi | 50.712811 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 262 |
| token_type | text |
| a_share | 0.000889 |
| c_share | 0.000020 |
| sink_score | 3.805006 |
| phi | 40.404560 |

#### Layer 26

- **dominant_type:** coexist
- **frequency:** 1.0
- **a_c_match:** False
- **a_c_match_rate:** 0.0
- **mean_entropy:** 0.776180590391159
- **mean_top1_share:** 0.6554338244597117
- **mean_mismatch:** 0.2154764833052953

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.382880 |
| c_share | 0.327895 |
| sink_score | 0.155028 |
| vision_j | 0 |
| token_str | <s> |
| phi | 50.590454 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 271 |
| token_type | text |
| a_share | 0.140654 |
| c_share | 0.664317 |
| sink_score | -1.552458 |
| phi | 50.713688 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 254 |
| token_type | vision |
| a_share | 0.000682 |
| c_share | 0.000014 |
| sink_score | 3.897238 |
| vision_j | 254 |
| phi | 49.785378 |

#### Layer 27

- **dominant_type:** coexist
- **frequency:** 1.0
- **a_c_match:** False
- **a_c_match_rate:** 0.0
- **mean_entropy:** 0.7572385557492574
- **mean_top1_share:** 0.6487835991382599
- **mean_mismatch:** 0.22955394277969995

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.372974 |
| c_share | 0.334735 |
| sink_score | 0.108168 |
| vision_j | 0 |
| token_str | <s> |
| phi | 50.591644 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 271 |
| token_type | text |
| a_share | 0.148744 |
| c_share | 0.647864 |
| sink_score | -1.471451 |
| phi | 50.713089 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 254 |
| token_type | vision |
| a_share | 0.000493 |
| c_share | 0.000008 |
| sink_score | 4.143862 |
| vision_j | 254 |
| phi | 47.222576 |

#### Layer 28

- **dominant_type:** coexist
- **frequency:** 1.0
- **a_c_match:** False
- **a_c_match_rate:** 0.0
- **mean_entropy:** 0.7543034172058105
- **mean_top1_share:** 0.6477701286474864
- **mean_mismatch:** 0.20834402551253636

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.477091 |
| c_share | 0.356914 |
| sink_score | 0.290211 |
| vision_j | 0 |
| token_str | <s> |
| phi | 50.591171 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 271 |
| token_type | text |
| a_share | 0.154054 |
| c_share | 0.654609 |
| sink_score | -1.446739 |
| phi | 50.713116 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 254 |
| token_type | vision |
| a_share | 0.000262 |
| c_share | 0.000005 |
| sink_score | 3.953531 |
| vision_j | 254 |
| phi | 48.932487 |

#### Layer 29

- **dominant_type:** coexist
- **frequency:** 1.0
- **a_c_match:** False
- **a_c_match_rate:** 0.0
- **mean_entropy:** 0.7944635768731435
- **mean_top1_share:** 0.6481642591953277
- **mean_mismatch:** 0.22941142161687214

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.402098 |
| c_share | 0.349504 |
| sink_score | 0.140179 |
| vision_j | 0 |
| token_str | <s> |
| phi | 50.591232 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 271 |
| token_type | text |
| a_share | 0.127635 |
| c_share | 0.658206 |
| sink_score | -1.640344 |
| phi | 50.713955 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 247 |
| token_type | vision |
| a_share | 0.000382 |
| c_share | 0.000009 |
| sink_score | 3.756280 |
| vision_j | 247 |
| phi | 46.208534 |

#### Layer 30

- **dominant_type:** coexist
- **frequency:** 1.0
- **a_c_match:** False
- **a_c_match_rate:** 0.0
- **mean_entropy:** 0.7387906579176585
- **mean_top1_share:** 0.6486717402935028
- **mean_mismatch:** 0.2176171287894249

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.410665 |
| c_share | 0.345181 |
| sink_score | 0.173708 |
| vision_j | 0 |
| token_str | <s> |
| phi | 50.581669 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 271 |
| token_type | text |
| a_share | 0.156408 |
| c_share | 0.656726 |
| sink_score | -1.434798 |
| phi | 50.796925 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 254 |
| token_type | vision |
| a_share | 0.000526 |
| c_share | 0.000009 |
| sink_score | 4.086718 |
| vision_j | 254 |
| phi | 34.839111 |

#### Layer 31

- **dominant_type:** coexist
- **frequency:** 1.0
- **a_c_match:** False
- **a_c_match_rate:** 0.0
- **mean_entropy:** 0.6160164284706116
- **mean_top1_share:** 0.7185152876377106
- **mean_mismatch:** 0.3185901037851969

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.262012 |
| c_share | 0.280805 |
| sink_score | -0.069270 |
| vision_j | 0 |
| token_str | <s> |
| phi | 50.352886 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 271 |
| token_type | text |
| a_share | 0.121848 |
| c_share | 0.713439 |
| sink_score | -1.767322 |
| phi | 50.794800 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 14 |
| token_type | vision |
| a_share | 0.009758 |
| c_share | 0.000020 |
| sink_score | 6.186571 |
| vision_j | 14 |
| token_str | right |
| phi | 23.728947 |

---

## spatialvla-4b

- **n_samples:** 150
- **n_layers:** 26
- **deep_layers:** [16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

### Sequence Boundaries

| Field | Value |
|-------|-------|
| vision_start | 0 |
| vision_end | 256 |
| text_start | 256 |
| text_end | 270 |
| total_seq_len | 270 |
| num_vision_tokens | 256 |
| num_text_tokens | 270 |
| pre_image_tokens | 0 |
| text_query_ranges | [[256, 270]] |

### Mode Tokens

| Mode | abs_t | token_type | token_str | freq | a_share_mean | c_share_mean |
|------|-------|------------|-----------|------|-------------|-------------|
| A_mode | 260 | text | `robot` | 0.5427 | N/A | N/A |
| C_mode | 260 | text | `robot` | 0.4480 | N/A | N/A |
| R_mode | 193 | vision | `<image>` | 0.7520 | N/A | N/A |

### Skill Distribution

| Skill | Count |
|-------|-------|
| close | 25 |
| fold | 25 |
| move | 25 |
| open | 25 |
| pick | 25 |
| place | 25 |

### Per-Layer Analysis

#### Layer 16

- **dominant_type:** normal
- **frequency:** 0.94
- **a_c_match:** True
- **a_c_match_rate:** 0.9133333333333333
- **mean_entropy:** 3.155214703877767
- **mean_top1_share:** 0.22713977629939716
- **mean_mismatch:** 0.0019950538699049503

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 260 |
| token_type | text |
| a_share | 0.207981 |
| c_share | 0.203193 |
| sink_score | 0.023290 |
| token_str | robot |
| phi | 36.088188 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 260 |
| token_type | text |
| a_share | 0.207981 |
| c_share | 0.203193 |
| sink_score | 0.023290 |
| token_str | robot |
| phi | 36.088188 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 193 |
| token_type | vision |
| a_share | 0.000112 |
| c_share | 0.000057 |
| sink_score | 0.674375 |
| vision_j | 193 |
| token_str | <image> |
| phi | 35.228264 |

#### Layer 17

- **dominant_type:** normal
- **frequency:** 0.94
- **a_c_match:** True
- **a_c_match_rate:** 0.9133333333333333
- **mean_entropy:** 2.7168381849924725
- **mean_top1_share:** 0.2000841285288334
- **mean_mismatch:** 0.0012261816215080519

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 260 |
| token_type | text |
| a_share | 0.181725 |
| c_share | 0.193949 |
| sink_score | -0.065098 |
| token_str | robot |
| phi | 35.078209 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 260 |
| token_type | text |
| a_share | 0.181725 |
| c_share | 0.193949 |
| sink_score | -0.065098 |
| token_str | robot |
| phi | 35.078209 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 193 |
| token_type | vision |
| a_share | 0.000510 |
| c_share | 0.000293 |
| sink_score | 0.553492 |
| vision_j | 193 |
| token_str | <image> |
| phi | 37.548309 |

#### Layer 18

- **dominant_type:** normal
- **frequency:** 0.9333333333333333
- **a_c_match:** True
- **a_c_match_rate:** 0.9133333333333333
- **mean_entropy:** 2.579474541346232
- **mean_top1_share:** 0.21980851580699284
- **mean_mismatch:** 0.001647299201770996

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 260 |
| token_type | text |
| a_share | 0.134572 |
| c_share | 0.132002 |
| sink_score | 0.019283 |
| token_str | robot |
| phi | 35.469147 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 260 |
| token_type | text |
| a_share | 0.134572 |
| c_share | 0.132002 |
| sink_score | 0.019283 |
| token_str | robot |
| phi | 35.469147 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 193 |
| token_type | vision |
| a_share | 0.005719 |
| c_share | 0.002785 |
| sink_score | 0.719421 |
| vision_j | 193 |
| token_str | <image> |
| phi | 36.899876 |

#### Layer 19

- **dominant_type:** normal
- **frequency:** 0.8
- **a_c_match:** True
- **a_c_match_rate:** 0.7866666666666666
- **mean_entropy:** 2.5146342198053997
- **mean_top1_share:** 0.20580355127652486
- **mean_mismatch:** 0.0027565315121319146

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 260 |
| token_type | text |
| a_share | 0.230949 |
| c_share | 0.209490 |
| sink_score | 0.097524 |
| token_str | robot |
| phi | 35.582706 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 260 |
| token_type | text |
| a_share | 0.230949 |
| c_share | 0.209490 |
| sink_score | 0.097524 |
| token_str | robot |
| phi | 35.582706 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 193 |
| token_type | vision |
| a_share | 0.000676 |
| c_share | 0.000316 |
| sink_score | 0.761482 |
| vision_j | 193 |
| token_str | <image> |
| phi | 40.127163 |

#### Layer 20

- **dominant_type:** normal
- **frequency:** 0.76
- **a_c_match:** True
- **a_c_match_rate:** 0.6266666666666667
- **mean_entropy:** 2.5836858463287355
- **mean_top1_share:** 0.19182103246450424
- **mean_mismatch:** 0.0061136986392860615

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 260 |
| token_type | text |
| a_share | 0.155512 |
| c_share | 0.139539 |
| sink_score | 0.108376 |
| token_str | robot |
| phi | 36.422153 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 260 |
| token_type | text |
| a_share | 0.207775 |
| c_share | 0.187407 |
| sink_score | 0.103173 |
| token_str | robot |
| phi | 38.028049 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 193 |
| token_type | vision |
| a_share | 0.004890 |
| c_share | 0.002235 |
| sink_score | 0.783058 |
| vision_j | 193 |
| token_str | <image> |
| phi | 40.230400 |

#### Layer 21

- **dominant_type:** normal
- **frequency:** 0.5133333333333333
- **a_c_match:** True
- **a_c_match_rate:** 0.5133333333333333
- **mean_entropy:** 2.2819899853070575
- **mean_top1_share:** 0.2648932605981827
- **mean_mismatch:** 0.01964153052618106

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 260 |
| token_type | text |
| a_share | 0.369545 |
| c_share | 0.318854 |
| sink_score | 0.147538 |
| token_str | robot |
| phi | 36.191566 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 225 |
| token_type | vision |
| a_share | 0.120097 |
| c_share | 0.218862 |
| sink_score | -0.600143 |
| vision_j | 225 |
| token_str | <image> |
| phi | 44.520432 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 256 |
| token_type | text |
| a_share | 0.000812 |
| c_share | 0.000619 |
| sink_score | 0.271715 |
| token_str | What |
| phi | 32.672344 |

#### Layer 22

- **dominant_type:** normal
- **frequency:** 0.92
- **a_c_match:** True
- **a_c_match_rate:** 0.9133333333333333
- **mean_entropy:** 2.3025963306427
- **mean_top1_share:** 0.268906703988711
- **mean_mismatch:** 0.0012159422102073828

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 260 |
| token_type | text |
| a_share | 0.340343 |
| c_share | 0.338265 |
| sink_score | 0.006125 |
| token_str | robot |
| phi | 31.609787 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 260 |
| token_type | text |
| a_share | 0.340343 |
| c_share | 0.338265 |
| sink_score | 0.006125 |
| token_str | robot |
| phi | 31.609787 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 193 |
| token_type | vision |
| a_share | 0.001314 |
| c_share | 0.000538 |
| sink_score | 0.892757 |
| vision_j | 193 |
| token_str | <image> |
| phi | 32.723709 |

#### Layer 23

- **dominant_type:** normal
- **frequency:** 0.82
- **a_c_match:** True
- **a_c_match_rate:** 0.7866666666666666
- **mean_entropy:** 2.4472403224309285
- **mean_top1_share:** 0.22628549272815388
- **mean_mismatch:** 0.0039957449515350164

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 260 |
| token_type | text |
| a_share | 0.248904 |
| c_share | 0.261188 |
| sink_score | -0.048175 |
| token_str | robot |
| phi | 22.294710 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 260 |
| token_type | text |
| a_share | 0.248904 |
| c_share | 0.261188 |
| sink_score | -0.048175 |
| token_str | robot |
| phi | 22.294710 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 193 |
| token_type | vision |
| a_share | 0.004875 |
| c_share | 0.001990 |
| sink_score | 0.896093 |
| vision_j | 193 |
| token_str | <image> |
| phi | 40.735424 |

#### Layer 24

- **dominant_type:** normal
- **frequency:** 0.84
- **a_c_match:** True
- **a_c_match_rate:** 0.8333333333333334
- **mean_entropy:** 2.37463982741038
- **mean_top1_share:** 0.26707770546277365
- **mean_mismatch:** 0.004243087309102217

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 225 |
| token_type | vision |
| a_share | 0.321531 |
| c_share | 0.260414 |
| sink_score | 0.210821 |
| vision_j | 225 |
| token_str | <image> |
| phi | 35.865162 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 225 |
| token_type | vision |
| a_share | 0.321531 |
| c_share | 0.260414 |
| sink_score | 0.210821 |
| vision_j | 225 |
| token_str | <image> |
| phi | 35.865162 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 193 |
| token_type | vision |
| a_share | 0.002195 |
| c_share | 0.001112 |
| sink_score | 0.680270 |
| vision_j | 193 |
| token_str | <image> |
| phi | 43.145264 |

#### Layer 25

- **dominant_type:** normal
- **frequency:** 0.9933333333333333
- **a_c_match:** True
- **a_c_match_rate:** 0.9466666666666667
- **mean_entropy:** 3.252409764925639
- **mean_top1_share:** 0.16619116773207981
- **mean_mismatch:** 0.002041918150304506

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 266 |
| token_type | text |
| a_share | 0.146785 |
| c_share | 0.162118 |
| sink_score | -0.099354 |
| token_str | ? |
| phi | 7.526434 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 266 |
| token_type | text |
| a_share | 0.146785 |
| c_share | 0.162118 |
| sink_score | -0.099354 |
| token_str | ? |
| phi | 7.526434 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 193 |
| token_type | vision |
| a_share | 0.023332 |
| c_share | 0.016116 |
| sink_score | 0.370054 |
| vision_j | 193 |
| token_str | <image> |
| phi | 12.461736 |

---

## tracevla-phi3v

- **n_samples:** 150
- **n_layers:** 32
- **deep_layers:** [22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

### Sequence Boundaries

| Field | Value |
|-------|-------|
| vision_start | 0 |
| vision_end | 313 |
| text_start | 313 |
| text_end | 341 |
| total_seq_len | 341 |
| num_vision_tokens | 313 |
| num_text_tokens | 341 |
| pre_image_tokens | 0 |
| text_query_ranges | [[313, 341]] |

### Mode Tokens

| Mode | abs_t | token_type | token_str | freq | a_share_mean | c_share_mean |
|------|-------|------------|-----------|------|-------------|-------------|
| A_mode | 0 | vision | `<s>` | 0.8400 | N/A | N/A |
| C_mode | 1 | vision | `<|user|>` | 1.0000 | N/A | N/A |
| R_mode | 3 | vision | `` | 0.4947 | N/A | N/A |

### Skill Distribution

| Skill | Count |
|-------|-------|
| close | 25 |
| fold | 25 |
| move | 25 |
| open | 25 |
| pick | 25 |
| place | 25 |

### Per-Layer Analysis

#### Layer 22

- **dominant_type:** normal
- **frequency:** 1.0
- **a_c_match:** False
- **a_c_match_rate:** 0.28
- **mean_entropy:** 2.3340478515625
- **mean_top1_share:** 0.1332390887538592
- **mean_mismatch:** 0.053162105586379765

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.091844 |
| c_share | 0.124437 |
| sink_score | -0.303705 |
| vision_j | 0 |
| token_str | <s> |
| phi | 33.487976 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 1 |
| token_type | vision |
| a_share | 0.091594 |
| c_share | 0.133712 |
| sink_score | -0.378324 |
| vision_j | 1 |
| token_str | <|user|> |
| phi | 33.630703 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 3 |
| token_type | vision |
| a_share | 0.000166 |
| c_share | 0.000006 |
| sink_score | 3.308012 |
| vision_j | 3 |
| token_str |  |
| phi | 27.760595 |

#### Layer 23

- **dominant_type:** normal
- **frequency:** 1.0
- **a_c_match:** False
- **a_c_match_rate:** 0.12666666666666668
- **mean_entropy:** 2.3437156963348387
- **mean_top1_share:** 0.1335187190771103
- **mean_mismatch:** 0.06200975666443507

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.076425 |
| c_share | 0.120191 |
| sink_score | -0.452768 |
| vision_j | 0 |
| token_str | <s> |
| phi | 33.487991 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 1 |
| token_type | vision |
| a_share | 0.076265 |
| c_share | 0.129261 |
| sink_score | -0.527615 |
| vision_j | 1 |
| token_str | <|user|> |
| phi | 33.630730 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 3 |
| token_type | vision |
| a_share | 0.000223 |
| c_share | 0.000009 |
| sink_score | 3.173362 |
| vision_j | 3 |
| token_str |  |
| phi | 28.929302 |

#### Layer 24

- **dominant_type:** normal
- **frequency:** 1.0
- **a_c_match:** False
- **a_c_match_rate:** 0.19333333333333333
- **mean_entropy:** 2.2902826690673828
- **mean_top1_share:** 0.13576353778441747
- **mean_mismatch:** 0.057785115564862884

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.100974 |
| c_share | 0.126380 |
| sink_score | -0.224429 |
| vision_j | 0 |
| token_str | <s> |
| phi | 33.488007 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 1 |
| token_type | vision |
| a_share | 0.100534 |
| c_share | 0.135801 |
| sink_score | -0.300696 |
| vision_j | 1 |
| token_str | <|user|> |
| phi | 33.630787 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 3 |
| token_type | vision |
| a_share | 0.000228 |
| c_share | 0.000007 |
| sink_score | 3.542164 |
| vision_j | 3 |
| token_str |  |
| phi | 33.519409 |

#### Layer 25

- **dominant_type:** normal
- **frequency:** 1.0
- **a_c_match:** False
- **a_c_match_rate:** 0.013333333333333334
- **mean_entropy:** 2.2631529633204144
- **mean_top1_share:** 0.13893840253353118
- **mean_mismatch:** 0.06403569259991249

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.093085 |
| c_share | 0.131343 |
| sink_score | -0.344296 |
| vision_j | 0 |
| token_str | <s> |
| phi | 33.488049 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 1 |
| token_type | vision |
| a_share | 0.092364 |
| c_share | 0.140505 |
| sink_score | -0.419510 |
| vision_j | 1 |
| token_str | <|user|> |
| phi | 33.630798 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 3 |
| token_type | vision |
| a_share | 0.000436 |
| c_share | 0.000014 |
| sink_score | 3.411558 |
| vision_j | 3 |
| token_str |  |
| phi | 34.201790 |

#### Layer 26

- **dominant_type:** normal
- **frequency:** 1.0
- **a_c_match:** False
- **a_c_match_rate:** 0.09333333333333334
- **mean_entropy:** 2.296719861030579
- **mean_top1_share:** 0.14008624563614527
- **mean_mismatch:** 0.051064567553500334

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.102736 |
| c_share | 0.132364 |
| sink_score | -0.253385 |
| vision_j | 0 |
| token_str | <s> |
| phi | 33.487953 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 1 |
| token_type | vision |
| a_share | 0.101521 |
| c_share | 0.140994 |
| sink_score | -0.328450 |
| vision_j | 1 |
| token_str | <|user|> |
| phi | 33.630688 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 3 |
| token_type | vision |
| a_share | 0.000424 |
| c_share | 0.000017 |
| sink_score | 3.189269 |
| vision_j | 3 |
| token_str |  |
| phi | 34.577278 |

#### Layer 27

- **dominant_type:** normal
- **frequency:** 1.0
- **a_c_match:** False
- **a_c_match_rate:** 0.0
- **mean_entropy:** 2.2447503741582233
- **mean_top1_share:** 0.14147414028644562
- **mean_mismatch:** 0.07212668083608151

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.089806 |
| c_share | 0.135611 |
| sink_score | -0.412143 |
| vision_j | 0 |
| token_str | <s> |
| phi | 33.451809 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 1 |
| token_type | vision |
| a_share | 0.089182 |
| c_share | 0.145253 |
| sink_score | -0.487790 |
| vision_j | 1 |
| token_str | <|user|> |
| phi | 33.679413 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 159 |
| token_type | vision |
| a_share | 0.000005 |
| c_share | 0.000000 |
| sink_score | 4.081097 |
| vision_j | 159 |
| token_str | <unk> |
| phi | 21.657412 |

#### Layer 28

- **dominant_type:** normal
- **frequency:** 0.9933333333333333
- **a_c_match:** False
- **a_c_match_rate:** 0.02
- **mean_entropy:** 2.2595986620585125
- **mean_top1_share:** 0.14485662033160526
- **mean_mismatch:** 0.041290538515895606

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.125799 |
| c_share | 0.141180 |
| sink_score | -0.115352 |
| vision_j | 0 |
| token_str | <s> |
| phi | 33.459442 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 1 |
| token_type | vision |
| a_share | 0.123381 |
| c_share | 0.149414 |
| sink_score | -0.191447 |
| vision_j | 1 |
| token_str | <|user|> |
| phi | 33.769295 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 159 |
| token_type | vision |
| a_share | 0.000000 |
| c_share | 0.000000 |
| sink_score | 3.115247 |
| vision_j | 159 |
| token_str | <unk> |
| phi | 23.888714 |

#### Layer 29

- **dominant_type:** normal
- **frequency:** 0.9933333333333333
- **a_c_match:** False
- **a_c_match_rate:** 0.15333333333333332
- **mean_entropy:** 2.3283585786819456
- **mean_top1_share:** 0.14438602109750112
- **mean_mismatch:** 0.07573284621040026

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.106587 |
| c_share | 0.137357 |
| sink_score | -0.253619 |
| vision_j | 0 |
| token_str | <s> |
| phi | 33.530975 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 1 |
| token_type | vision |
| a_share | 0.104125 |
| c_share | 0.145045 |
| sink_score | -0.331444 |
| vision_j | 1 |
| token_str | <|user|> |
| phi | 33.821869 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 150 |
| token_type | vision |
| a_share | 0.000000 |
| c_share | 0.000000 |
| sink_score | 2.932549 |
| vision_j | 150 |
| token_str | <unk> |
| phi | 27.259201 |

#### Layer 30

- **dominant_type:** normal
- **frequency:** 1.0
- **a_c_match:** False
- **a_c_match_rate:** 0.006666666666666667
- **mean_entropy:** 2.4967158889770507
- **mean_top1_share:** 0.13586528410514195
- **mean_mismatch:** 0.0876722927391529

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.095341 |
| c_share | 0.132572 |
| sink_score | -0.329662 |
| vision_j | 0 |
| token_str | <s> |
| phi | 33.564835 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 1 |
| token_type | vision |
| a_share | 0.093614 |
| c_share | 0.141074 |
| sink_score | -0.410102 |
| vision_j | 1 |
| token_str | <|user|> |
| phi | 33.857555 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 319 |
| token_type | text |
| a_share | 0.001034 |
| c_share | 0.000078 |
| sink_score | 2.579057 |
| token_str |  |
| phi | 34.569977 |

#### Layer 31

- **dominant_type:** normal
- **frequency:** 1.0
- **a_c_match:** False
- **a_c_match_rate:** 0.006666666666666667
- **mean_entropy:** 2.322074933052063
- **mean_top1_share:** 0.14350408325592676
- **mean_mismatch:** 0.21764838645855586

**A peak (Attention):**

| Field | Value |
|-------|-------|
| abs_t | 0 |
| token_type | vision |
| a_share | 0.082405 |
| c_share | 0.140399 |
| sink_score | -0.532850 |
| vision_j | 0 |
| token_str | <s> |
| phi | 35.521412 |

**C peak (Contribution):**

| Field | Value |
|-------|-------|
| abs_t | 1 |
| token_type | vision |
| a_share | 0.079975 |
| c_share | 0.148252 |
| sink_score | -0.617199 |
| vision_j | 1 |
| token_str | <|user|> |
| phi | 35.695305 |

**R peak (Residual):**

| Field | Value |
|-------|-------|
| abs_t | 319 |
| token_type | text |
| a_share | 0.008687 |
| c_share | 0.000049 |
| sink_score | 5.186414 |
| token_str |  |
| phi | 31.307709 |

---

# Gate 2 — Layer-Local V=0 Block Sweep (Causal Ablation)

## ecot-7b

### A_mode

#### A_mode_all

- **method:** v_zero
- **candidates_abs_t:** [0]
- **target_layers:** [22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

**Sanity Check:**

| Field | Value |
|-------|-------|
| hook_fired | True |
| logits_changed | True |
| kl_divergence | 0.015298 |

**Per-K Ablation Results:**

| K | targets | vzero_mean_kl | vzero_std_kl | vzero_mean_top1_change |
|---|---------|--------------|-------------|----------------------|
| 1 | [0] | 1.197461 | 1.343250 | 0.3200 |
| 3 | [0] | 1.197461 | 1.343250 | 0.3200 |
| 5 | [0] | 1.197461 | 1.343250 | 0.3200 |

#### A_mode_block1

- **method:** v_zero
- **candidates_abs_t:** [0]
- **target_layers:** [22, 23, 24, 25, 26]

**Sanity Check:**

| Field | Value |
|-------|-------|
| hook_fired | True |
| logits_changed | True |
| kl_divergence | 0.015298 |

**Per-K Ablation Results:**

| K | targets | vzero_mean_kl | vzero_std_kl | vzero_mean_top1_change |
|---|---------|--------------|-------------|----------------------|
| 1 | [0] | 0.456111 | 0.643871 | 0.2667 |
| 3 | [0] | 0.456111 | 0.643871 | 0.2667 |
| 5 | [0] | 0.456111 | 0.643871 | 0.2667 |

#### A_mode_block2

- **method:** v_zero
- **candidates_abs_t:** [0]
- **target_layers:** [27, 28, 29, 30, 31]

**Sanity Check:**

| Field | Value |
|-------|-------|
| hook_fired | True |
| logits_changed | True |
| kl_divergence | 0.015298 |

**Per-K Ablation Results:**

| K | targets | vzero_mean_kl | vzero_std_kl | vzero_mean_top1_change |
|---|---------|--------------|-------------|----------------------|
| 1 | [0] | 0.418270 | 0.417134 | 0.2267 |
| 3 | [0] | 0.418270 | 0.417134 | 0.2267 |
| 5 | [0] | 0.418270 | 0.417134 | 0.2267 |

### C_mode

#### C_mode_all

- **method:** v_zero
- **candidates_abs_t:** [0]
- **target_layers:** [22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

**Sanity Check:**

| Field | Value |
|-------|-------|
| hook_fired | True |
| logits_changed | True |
| kl_divergence | 0.015298 |

**Per-K Ablation Results:**

| K | targets | vzero_mean_kl | vzero_std_kl | vzero_mean_top1_change |
|---|---------|--------------|-------------|----------------------|
| 1 | [0] | 1.197461 | 1.343250 | 0.3200 |
| 3 | [0] | 1.197461 | 1.343250 | 0.3200 |
| 5 | [0] | 1.197461 | 1.343250 | 0.3200 |

#### C_mode_block1

- **method:** v_zero
- **candidates_abs_t:** [0]
- **target_layers:** [22, 23, 24, 25, 26]

**Sanity Check:**

| Field | Value |
|-------|-------|
| hook_fired | True |
| logits_changed | True |
| kl_divergence | 0.015298 |

**Per-K Ablation Results:**

| K | targets | vzero_mean_kl | vzero_std_kl | vzero_mean_top1_change |
|---|---------|--------------|-------------|----------------------|
| 1 | [0] | 0.456111 | 0.643871 | 0.2667 |
| 3 | [0] | 0.456111 | 0.643871 | 0.2667 |
| 5 | [0] | 0.456111 | 0.643871 | 0.2667 |

#### C_mode_block2

- **method:** v_zero
- **candidates_abs_t:** [0]
- **target_layers:** [27, 28, 29, 30, 31]

**Sanity Check:**

| Field | Value |
|-------|-------|
| hook_fired | True |
| logits_changed | True |
| kl_divergence | 0.015298 |

**Per-K Ablation Results:**

| K | targets | vzero_mean_kl | vzero_std_kl | vzero_mean_top1_change |
|---|---------|--------------|-------------|----------------------|
| 1 | [0] | 0.418270 | 0.417134 | 0.2267 |
| 3 | [0] | 0.418270 | 0.417134 | 0.2267 |
| 5 | [0] | 0.418270 | 0.417134 | 0.2267 |

### R_mode

#### R_mode_all

- **method:** v_zero
- **candidates_abs_t:** [271]
- **target_layers:** [22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

**Sanity Check:**

| Field | Value |
|-------|-------|
| hook_fired | True |
| logits_changed | True |
| kl_divergence | 0.000230 |

**Per-K Ablation Results:**

| K | targets | vzero_mean_kl | vzero_std_kl | vzero_mean_top1_change |
|---|---------|--------------|-------------|----------------------|
| 1 | [271] | 0.071961 | 0.172982 | 0.0667 |
| 3 | [271] | 0.071961 | 0.172982 | 0.0667 |
| 5 | [271] | 0.071961 | 0.172982 | 0.0667 |

#### R_mode_block1

- **method:** v_zero
- **candidates_abs_t:** [271]
- **target_layers:** [22, 23, 24, 25, 26]

**Sanity Check:**

| Field | Value |
|-------|-------|
| hook_fired | True |
| logits_changed | True |
| kl_divergence | 0.000230 |

**Per-K Ablation Results:**

| K | targets | vzero_mean_kl | vzero_std_kl | vzero_mean_top1_change |
|---|---------|--------------|-------------|----------------------|
| 1 | [271] | 0.040934 | 0.102422 | 0.0467 |
| 3 | [271] | 0.040934 | 0.102422 | 0.0467 |
| 5 | [271] | 0.040934 | 0.102422 | 0.0467 |

#### R_mode_block2

- **method:** v_zero
- **candidates_abs_t:** [271]
- **target_layers:** [27, 28, 29, 30, 31]

**Sanity Check:**

| Field | Value |
|-------|-------|
| hook_fired | True |
| logits_changed | True |
| kl_divergence | 0.000230 |

**Per-K Ablation Results:**

| K | targets | vzero_mean_kl | vzero_std_kl | vzero_mean_top1_change |
|---|---------|--------------|-------------|----------------------|
| 1 | [271] | 0.028259 | 0.073802 | 0.0667 |
| 3 | [271] | 0.028259 | 0.073802 | 0.0667 |
| 5 | [271] | 0.028259 | 0.073802 | 0.0667 |

---

## openvla-7b

### A_mode

#### A_mode_all

- **method:** v_zero
- **candidates_abs_t:** [0]
- **target_layers:** [22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

**Sanity Check:**

| Field | Value |
|-------|-------|
| hook_fired | True |
| logits_changed | True |
| kl_divergence | 4.282147 |

**Per-K Ablation Results:**

| K | targets | vzero_mean_kl | vzero_std_kl | vzero_mean_top1_change |
|---|---------|--------------|-------------|----------------------|
| 1 | [0] | 0.511151 | 0.593234 | 0.3333 |
| 3 | [0] | 0.511151 | 0.593234 | 0.3333 |
| 5 | [0] | 0.511151 | 0.593234 | 0.3333 |

#### A_mode_block1

- **method:** v_zero
- **candidates_abs_t:** [0]
- **target_layers:** [22, 23, 24, 25, 26]

**Sanity Check:**

| Field | Value |
|-------|-------|
| hook_fired | True |
| logits_changed | True |
| kl_divergence | 4.282147 |

**Per-K Ablation Results:**

| K | targets | vzero_mean_kl | vzero_std_kl | vzero_mean_top1_change |
|---|---------|--------------|-------------|----------------------|
| 1 | [0] | 0.299871 | 0.358498 | 0.3000 |
| 3 | [0] | 0.299871 | 0.358498 | 0.3000 |
| 5 | [0] | 0.299871 | 0.358498 | 0.3000 |

#### A_mode_block2

- **method:** v_zero
- **candidates_abs_t:** [0]
- **target_layers:** [27, 28, 29, 30, 31]

**Sanity Check:**

| Field | Value |
|-------|-------|
| hook_fired | True |
| logits_changed | True |
| kl_divergence | 4.282147 |

**Per-K Ablation Results:**

| K | targets | vzero_mean_kl | vzero_std_kl | vzero_mean_top1_change |
|---|---------|--------------|-------------|----------------------|
| 1 | [0] | 0.162520 | 0.174536 | 0.2067 |
| 3 | [0] | 0.162520 | 0.174536 | 0.2067 |
| 5 | [0] | 0.162520 | 0.174536 | 0.2067 |

### C_mode

#### C_mode_all

- **method:** v_zero
- **candidates_abs_t:** [271]
- **target_layers:** [22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

**Sanity Check:**

| Field | Value |
|-------|-------|
| hook_fired | True |
| logits_changed | True |
| kl_divergence | 0.153061 |

**Per-K Ablation Results:**

| K | targets | vzero_mean_kl | vzero_std_kl | vzero_mean_top1_change |
|---|---------|--------------|-------------|----------------------|
| 1 | [271] | 0.186480 | 0.444313 | 0.2067 |
| 3 | [271] | 0.186480 | 0.444313 | 0.2067 |
| 5 | [271] | 0.186480 | 0.444313 | 0.2067 |

#### C_mode_block1

- **method:** v_zero
- **candidates_abs_t:** [271]
- **target_layers:** [22, 23, 24, 25, 26]

**Sanity Check:**

| Field | Value |
|-------|-------|
| hook_fired | True |
| logits_changed | True |
| kl_divergence | 0.153061 |

**Per-K Ablation Results:**

| K | targets | vzero_mean_kl | vzero_std_kl | vzero_mean_top1_change |
|---|---------|--------------|-------------|----------------------|
| 1 | [271] | 0.082259 | 0.154256 | 0.1733 |
| 3 | [271] | 0.082259 | 0.154256 | 0.1733 |
| 5 | [271] | 0.082259 | 0.154256 | 0.1733 |

#### C_mode_block2

- **method:** v_zero
- **candidates_abs_t:** [271]
- **target_layers:** [27, 28, 29, 30, 31]

**Sanity Check:**

| Field | Value |
|-------|-------|
| hook_fired | True |
| logits_changed | True |
| kl_divergence | 0.153061 |

**Per-K Ablation Results:**

| K | targets | vzero_mean_kl | vzero_std_kl | vzero_mean_top1_change |
|---|---------|--------------|-------------|----------------------|
| 1 | [271] | 0.058509 | 0.141097 | 0.1000 |
| 3 | [271] | 0.058509 | 0.141097 | 0.1000 |
| 5 | [271] | 0.058509 | 0.141097 | 0.1000 |

### R_mode

#### R_mode_all

- **method:** v_zero
- **candidates_abs_t:** [262]
- **target_layers:** [22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

**Sanity Check:**

| Field | Value |
|-------|-------|
| hook_fired | True |
| logits_changed | True |
| kl_divergence | 0.016602 |

**Per-K Ablation Results:**

| K | targets | vzero_mean_kl | vzero_std_kl | vzero_mean_top1_change |
|---|---------|--------------|-------------|----------------------|
| 1 | [262] | 0.000961 | 0.001141 | 0.0200 |
| 3 | [262] | 0.000961 | 0.001141 | 0.0200 |
| 5 | [262] | 0.000961 | 0.001141 | 0.0200 |

#### R_mode_block1

- **method:** v_zero
- **candidates_abs_t:** [262]
- **target_layers:** [22, 23, 24, 25, 26]

**Sanity Check:**

| Field | Value |
|-------|-------|
| hook_fired | True |
| logits_changed | True |
| kl_divergence | 0.016602 |

**Per-K Ablation Results:**

| K | targets | vzero_mean_kl | vzero_std_kl | vzero_mean_top1_change |
|---|---------|--------------|-------------|----------------------|
| 1 | [262] | 0.000912 | 0.000963 | 0.0067 |
| 3 | [262] | 0.000912 | 0.000963 | 0.0067 |
| 5 | [262] | 0.000912 | 0.000963 | 0.0067 |

#### R_mode_block2

- **method:** v_zero
- **candidates_abs_t:** [262]
- **target_layers:** [27, 28, 29, 30, 31]

**Sanity Check:**

| Field | Value |
|-------|-------|
| hook_fired | True |
| logits_changed | True |
| kl_divergence | 0.016602 |

**Per-K Ablation Results:**

| K | targets | vzero_mean_kl | vzero_std_kl | vzero_mean_top1_change |
|---|---------|--------------|-------------|----------------------|
| 1 | [262] | 0.000504 | 0.000646 | 0.0133 |
| 3 | [262] | 0.000504 | 0.000646 | 0.0133 |
| 5 | [262] | 0.000504 | 0.000646 | 0.0133 |

---

## spatialvla-4b

### A_mode

#### A_mode_all

- **method:** v_zero
- **candidates_abs_t:** [260]
- **target_layers:** [16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

**Sanity Check:**

| Field | Value |
|-------|-------|
| hook_fired | True |
| logits_changed | True |
| kl_divergence | 1.778777 |

**Per-K Ablation Results:**

| K | targets | vzero_mean_kl | vzero_std_kl | vzero_mean_top1_change |
|---|---------|--------------|-------------|----------------------|
| 1 | [260] | 1.215353 | 0.728086 | 0.6733 |
| 3 | [260] | 1.215353 | 0.728086 | 0.6733 |
| 5 | [260] | 1.215353 | 0.728086 | 0.6733 |

#### A_mode_block1

- **method:** v_zero
- **candidates_abs_t:** [260]
- **target_layers:** [16, 17, 18, 19, 20]

**Sanity Check:**

| Field | Value |
|-------|-------|
| hook_fired | True |
| logits_changed | True |
| kl_divergence | 1.778777 |

**Per-K Ablation Results:**

| K | targets | vzero_mean_kl | vzero_std_kl | vzero_mean_top1_change |
|---|---------|--------------|-------------|----------------------|
| 1 | [260] | 0.723259 | 0.515537 | 0.5133 |
| 3 | [260] | 0.723259 | 0.515537 | 0.5133 |
| 5 | [260] | 0.723259 | 0.515537 | 0.5133 |

#### A_mode_block2

- **method:** v_zero
- **candidates_abs_t:** [260]
- **target_layers:** [21, 22, 23, 24, 25]

**Sanity Check:**

| Field | Value |
|-------|-------|
| hook_fired | True |
| logits_changed | True |
| kl_divergence | 1.778777 |

**Per-K Ablation Results:**

| K | targets | vzero_mean_kl | vzero_std_kl | vzero_mean_top1_change |
|---|---------|--------------|-------------|----------------------|
| 1 | [260] | 0.513002 | 0.377943 | 0.3667 |
| 3 | [260] | 0.513002 | 0.377943 | 0.3667 |
| 5 | [260] | 0.513002 | 0.377943 | 0.3667 |

### C_mode

#### C_mode_all

- **method:** v_zero
- **candidates_abs_t:** [260]
- **target_layers:** [16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

**Sanity Check:**

| Field | Value |
|-------|-------|
| hook_fired | True |
| logits_changed | True |
| kl_divergence | 1.778777 |

**Per-K Ablation Results:**

| K | targets | vzero_mean_kl | vzero_std_kl | vzero_mean_top1_change |
|---|---------|--------------|-------------|----------------------|
| 1 | [260] | 1.215353 | 0.728086 | 0.6733 |
| 3 | [260] | 1.215353 | 0.728086 | 0.6733 |
| 5 | [260] | 1.215353 | 0.728086 | 0.6733 |

#### C_mode_block1

- **method:** v_zero
- **candidates_abs_t:** [260]
- **target_layers:** [16, 17, 18, 19, 20]

**Sanity Check:**

| Field | Value |
|-------|-------|
| hook_fired | True |
| logits_changed | True |
| kl_divergence | 1.778777 |

**Per-K Ablation Results:**

| K | targets | vzero_mean_kl | vzero_std_kl | vzero_mean_top1_change |
|---|---------|--------------|-------------|----------------------|
| 1 | [260] | 0.723259 | 0.515537 | 0.5133 |
| 3 | [260] | 0.723259 | 0.515537 | 0.5133 |
| 5 | [260] | 0.723259 | 0.515537 | 0.5133 |

#### C_mode_block2

- **method:** v_zero
- **candidates_abs_t:** [260]
- **target_layers:** [21, 22, 23, 24, 25]

**Sanity Check:**

| Field | Value |
|-------|-------|
| hook_fired | True |
| logits_changed | True |
| kl_divergence | 1.778777 |

**Per-K Ablation Results:**

| K | targets | vzero_mean_kl | vzero_std_kl | vzero_mean_top1_change |
|---|---------|--------------|-------------|----------------------|
| 1 | [260] | 0.513002 | 0.377943 | 0.3667 |
| 3 | [260] | 0.513002 | 0.377943 | 0.3667 |
| 5 | [260] | 0.513002 | 0.377943 | 0.3667 |

### R_mode

#### R_mode_all

- **method:** v_zero
- **candidates_abs_t:** [193]
- **target_layers:** [16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

**Sanity Check:**

| Field | Value |
|-------|-------|
| hook_fired | True |
| logits_changed | True |
| kl_divergence | 0.018977 |

**Per-K Ablation Results:**

| K | targets | vzero_mean_kl | vzero_std_kl | vzero_mean_top1_change |
|---|---------|--------------|-------------|----------------------|
| 1 | [193] | 0.003841 | 0.002684 | 0.0467 |
| 3 | [193] | 0.003841 | 0.002684 | 0.0467 |
| 5 | [193] | 0.003841 | 0.002684 | 0.0467 |

#### R_mode_block1

- **method:** v_zero
- **candidates_abs_t:** [193]
- **target_layers:** [16, 17, 18, 19, 20]

**Sanity Check:**

| Field | Value |
|-------|-------|
| hook_fired | True |
| logits_changed | True |
| kl_divergence | 0.018977 |

**Per-K Ablation Results:**

| K | targets | vzero_mean_kl | vzero_std_kl | vzero_mean_top1_change |
|---|---------|--------------|-------------|----------------------|
| 1 | [193] | 0.003082 | 0.001868 | 0.0267 |
| 3 | [193] | 0.003082 | 0.001868 | 0.0267 |
| 5 | [193] | 0.003082 | 0.001868 | 0.0267 |

#### R_mode_block2

- **method:** v_zero
- **candidates_abs_t:** [193]
- **target_layers:** [21, 22, 23, 24, 25]

**Sanity Check:**

| Field | Value |
|-------|-------|
| hook_fired | True |
| logits_changed | True |
| kl_divergence | 0.018977 |

**Per-K Ablation Results:**

| K | targets | vzero_mean_kl | vzero_std_kl | vzero_mean_top1_change |
|---|---------|--------------|-------------|----------------------|
| 1 | [193] | 0.001979 | 0.001798 | 0.0467 |
| 3 | [193] | 0.001979 | 0.001798 | 0.0467 |
| 5 | [193] | 0.001979 | 0.001798 | 0.0467 |

---

## tracevla-phi3v

### A_mode

#### A_mode_all

- **method:** v_zero
- **candidates_abs_t:** [0]
- **target_layers:** [22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

**Sanity Check:**

| Field | Value |
|-------|-------|
| hook_fired | True |
| logits_changed | True |
| kl_divergence | 16.022640 |

**Per-K Ablation Results:**

| K | targets | vzero_mean_kl | vzero_std_kl | vzero_mean_top1_change |
|---|---------|--------------|-------------|----------------------|
| 1 | [0] | 13.112485 | 2.759681 | 1.0000 |
| 3 | [0] | 13.112485 | 2.759681 | 1.0000 |
| 5 | [0] | 13.112485 | 2.759681 | 1.0000 |

#### A_mode_block1

- **method:** v_zero
- **candidates_abs_t:** [0]
- **target_layers:** [22, 23, 24, 25, 26]

**Sanity Check:**

| Field | Value |
|-------|-------|
| hook_fired | True |
| logits_changed | True |
| kl_divergence | 16.022640 |

**Per-K Ablation Results:**

| K | targets | vzero_mean_kl | vzero_std_kl | vzero_mean_top1_change |
|---|---------|--------------|-------------|----------------------|
| 1 | [0] | 13.080631 | 2.746928 | 1.0000 |
| 3 | [0] | 13.080631 | 2.746928 | 1.0000 |
| 5 | [0] | 13.080631 | 2.746928 | 1.0000 |

#### A_mode_block2

- **method:** v_zero
- **candidates_abs_t:** [0]
- **target_layers:** [27, 28, 29, 30, 31]

**Sanity Check:**

| Field | Value |
|-------|-------|
| hook_fired | True |
| logits_changed | True |
| kl_divergence | 16.022640 |

**Per-K Ablation Results:**

| K | targets | vzero_mean_kl | vzero_std_kl | vzero_mean_top1_change |
|---|---------|--------------|-------------|----------------------|
| 1 | [0] | 13.051720 | 2.725062 | 1.0000 |
| 3 | [0] | 13.051720 | 2.725062 | 1.0000 |
| 5 | [0] | 13.051720 | 2.725062 | 1.0000 |

### C_mode

#### C_mode_all

- **method:** v_zero
- **candidates_abs_t:** [1]
- **target_layers:** [22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

**Sanity Check:**

| Field | Value |
|-------|-------|
| hook_fired | True |
| logits_changed | True |
| kl_divergence | 14.798369 |

**Per-K Ablation Results:**

| K | targets | vzero_mean_kl | vzero_std_kl | vzero_mean_top1_change |
|---|---------|--------------|-------------|----------------------|
| 1 | [1] | 13.108523 | 2.757982 | 1.0000 |
| 3 | [1] | 13.108523 | 2.757982 | 1.0000 |
| 5 | [1] | 13.108523 | 2.757982 | 1.0000 |

#### C_mode_block1

- **method:** v_zero
- **candidates_abs_t:** [1]
- **target_layers:** [22, 23, 24, 25, 26]

**Sanity Check:**

| Field | Value |
|-------|-------|
| hook_fired | True |
| logits_changed | True |
| kl_divergence | 14.798369 |

**Per-K Ablation Results:**

| K | targets | vzero_mean_kl | vzero_std_kl | vzero_mean_top1_change |
|---|---------|--------------|-------------|----------------------|
| 1 | [1] | 13.074100 | 2.738117 | 1.0000 |
| 3 | [1] | 13.074100 | 2.738117 | 1.0000 |
| 5 | [1] | 13.074100 | 2.738117 | 1.0000 |

#### C_mode_block2

- **method:** v_zero
- **candidates_abs_t:** [1]
- **target_layers:** [27, 28, 29, 30, 31]

**Sanity Check:**

| Field | Value |
|-------|-------|
| hook_fired | True |
| logits_changed | True |
| kl_divergence | 14.798369 |

**Per-K Ablation Results:**

| K | targets | vzero_mean_kl | vzero_std_kl | vzero_mean_top1_change |
|---|---------|--------------|-------------|----------------------|
| 1 | [1] | 13.056668 | 2.724218 | 1.0000 |
| 3 | [1] | 13.056668 | 2.724218 | 1.0000 |
| 5 | [1] | 13.056668 | 2.724218 | 1.0000 |

### R_mode

#### R_mode_all

- **method:** v_zero
- **candidates_abs_t:** [3]
- **target_layers:** [22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

**Sanity Check:**

| Field | Value |
|-------|-------|
| hook_fired | True |
| logits_changed | True |
| kl_divergence | 14.583757 |

**Per-K Ablation Results:**

| K | targets | vzero_mean_kl | vzero_std_kl | vzero_mean_top1_change |
|---|---------|--------------|-------------|----------------------|
| 1 | [3] | 12.968920 | 2.682700 | 1.0000 |
| 3 | [3] | 12.968920 | 2.682700 | 1.0000 |
| 5 | [3] | 12.968920 | 2.682700 | 1.0000 |

#### R_mode_block1

- **method:** v_zero
- **candidates_abs_t:** [3]
- **target_layers:** [22, 23, 24, 25, 26]

**Sanity Check:**

| Field | Value |
|-------|-------|
| hook_fired | True |
| logits_changed | True |
| kl_divergence | 14.583757 |

**Per-K Ablation Results:**

| K | targets | vzero_mean_kl | vzero_std_kl | vzero_mean_top1_change |
|---|---------|--------------|-------------|----------------------|
| 1 | [3] | 13.020450 | 2.702544 | 1.0000 |
| 3 | [3] | 13.020450 | 2.702544 | 1.0000 |
| 5 | [3] | 13.020450 | 2.702544 | 1.0000 |

#### R_mode_block2

- **method:** v_zero
- **candidates_abs_t:** [3]
- **target_layers:** [27, 28, 29, 30, 31]

**Sanity Check:**

| Field | Value |
|-------|-------|
| hook_fired | True |
| logits_changed | True |
| kl_divergence | 14.583757 |

**Per-K Ablation Results:**

| K | targets | vzero_mean_kl | vzero_std_kl | vzero_mean_top1_change |
|---|---------|--------------|-------------|----------------------|
| 1 | [3] | 12.983421 | 2.685167 | 1.0000 |
| 3 | [3] | 12.983421 | 2.685167 | 1.0000 |
| 5 | [3] | 12.983421 | 2.685167 | 1.0000 |

---

# Gate 3 — Text Masking + Counterfactual Verb Swap

## ecot-7b

### Skill Labels

| Skill | Count |
|-------|-------|
| close | 2 |
| fold | 3 |
| move | 3 |
| open | 1 |
| pick | 6 |
| place | 5 |
| **Total** | **20** |

### Counterfactual Results — Raw Records (80 total)

#### Sample 1 (skill=move, swap_verb=fold)

| layer | delta_orig | delta_textKV | ratio |
|-------|-----------|-------------|-------|
| 22 | 1.186758 | 1.186758 | 1.0000 |
| 23 | 1.239056 | 1.239056 | 1.0000 |
| 24 | 1.229022 | 1.229022 | 1.0000 |
| 25 | 1.228514 | 1.228514 | 1.0000 |
| 26 | 1.240551 | 1.240551 | 1.0000 |
| 27 | 1.235439 | 1.235439 | 1.0000 |
| 28 | 1.251168 | 1.251168 | 1.0000 |
| 29 | 1.251210 | 1.251210 | 1.0000 |
| 30 | 1.251719 | 1.251719 | 1.0000 |
| 31 | 1.224376 | 1.224376 | 1.0000 |

#### Sample 4 (skill=pick, swap_verb=place)

| layer | delta_orig | delta_textKV | ratio |
|-------|-----------|-------------|-------|
| 22 | 0.344971 | 0.344971 | 1.0000 |
| 23 | 0.364949 | 0.364949 | 1.0000 |
| 24 | 0.407391 | 0.407391 | 1.0000 |
| 25 | 0.442174 | 0.442174 | 1.0000 |
| 26 | 0.459046 | 0.459046 | 1.0000 |
| 27 | 0.501448 | 0.501448 | 1.0000 |
| 28 | 0.535296 | 0.535296 | 1.0000 |
| 29 | 0.582752 | 0.582752 | 1.0000 |
| 30 | 0.610979 | 0.610979 | 1.0000 |
| 31 | 0.612653 | 0.612653 | 1.0000 |

#### Sample 5 (skill=move, swap_verb=fold)

| layer | delta_orig | delta_textKV | ratio |
|-------|-----------|-------------|-------|
| 22 | 1.122026 | 1.122026 | 1.0000 |
| 23 | 1.159303 | 1.159303 | 1.0000 |
| 24 | 1.151699 | 1.151699 | 1.0000 |
| 25 | 1.165510 | 1.165510 | 1.0000 |
| 26 | 1.158834 | 1.158834 | 1.0000 |
| 27 | 1.161414 | 1.161414 | 1.0000 |
| 28 | 1.177612 | 1.177612 | 1.0000 |
| 29 | 1.176774 | 1.176774 | 1.0000 |
| 30 | 1.206262 | 1.206262 | 1.0000 |
| 31 | 1.199708 | 1.199708 | 1.0000 |

#### Sample 6 (skill=open, swap_verb=close)

| layer | delta_orig | delta_textKV | ratio |
|-------|-----------|-------------|-------|
| 22 | 0.440200 | 0.440200 | 1.0000 |
| 23 | 0.458915 | 0.458915 | 1.0000 |
| 24 | 0.469377 | 0.469377 | 1.0000 |
| 25 | 0.503761 | 0.503761 | 1.0000 |
| 26 | 0.544421 | 0.544421 | 1.0000 |
| 27 | 0.569001 | 0.569001 | 1.0000 |
| 28 | 0.590966 | 0.590966 | 1.0000 |
| 29 | 0.642666 | 0.642666 | 1.0000 |
| 30 | 0.661991 | 0.661991 | 1.0000 |
| 31 | 0.686100 | 0.686100 | 1.0000 |

#### Sample 12 (skill=close, swap_verb=open)

| layer | delta_orig | delta_textKV | ratio |
|-------|-----------|-------------|-------|
| 22 | 0.440890 | 0.440890 | 1.0000 |
| 23 | 0.454948 | 0.454948 | 1.0000 |
| 24 | 0.459020 | 0.459020 | 1.0000 |
| 25 | 0.470209 | 0.470209 | 1.0000 |
| 26 | 0.490535 | 0.490535 | 1.0000 |
| 27 | 0.516926 | 0.516926 | 1.0000 |
| 28 | 0.546789 | 0.546789 | 1.0000 |
| 29 | 0.573032 | 0.573032 | 1.0000 |
| 30 | 0.604528 | 0.604528 | 1.0000 |
| 31 | 0.626007 | 0.626007 | 1.0000 |

#### Sample 13 (skill=pick, swap_verb=place)

| layer | delta_orig | delta_textKV | ratio |
|-------|-----------|-------------|-------|
| 22 | 0.393038 | 0.393038 | 1.0000 |
| 23 | 0.410798 | 0.410798 | 1.0000 |
| 24 | 0.454476 | 0.454476 | 1.0000 |
| 25 | 0.497097 | 0.497097 | 1.0000 |
| 26 | 0.530985 | 0.530985 | 1.0000 |
| 27 | 0.556079 | 0.556079 | 1.0000 |
| 28 | 0.577082 | 0.577082 | 1.0000 |
| 29 | 0.601507 | 0.601507 | 1.0000 |
| 30 | 0.621657 | 0.621657 | 1.0000 |
| 31 | 0.620108 | 0.620108 | 1.0000 |

#### Sample 16 (skill=close, swap_verb=open)

| layer | delta_orig | delta_textKV | ratio |
|-------|-----------|-------------|-------|
| 22 | 0.460234 | 0.460234 | 1.0000 |
| 23 | 0.482285 | 0.482285 | 1.0000 |
| 24 | 0.473425 | 0.473425 | 1.0000 |
| 25 | 0.502414 | 0.502414 | 1.0000 |
| 26 | 0.526066 | 0.526066 | 1.0000 |
| 27 | 0.562317 | 0.562317 | 1.0000 |
| 28 | 0.582977 | 0.582977 | 1.0000 |
| 29 | 0.605668 | 0.605668 | 1.0000 |
| 30 | 0.620881 | 0.620881 | 1.0000 |
| 31 | 0.644827 | 0.644827 | 1.0000 |

#### Sample 17 (skill=move, swap_verb=fold)

| layer | delta_orig | delta_textKV | ratio |
|-------|-----------|-------------|-------|
| 22 | 1.092588 | 1.092588 | 1.0000 |
| 23 | 1.135227 | 1.135227 | 1.0000 |
| 24 | 1.161891 | 1.161891 | 1.0000 |
| 25 | 1.181366 | 1.181366 | 1.0000 |
| 26 | 1.183886 | 1.183886 | 1.0000 |
| 27 | 1.196399 | 1.196399 | 1.0000 |
| 28 | 1.230143 | 1.230143 | 1.0000 |
| 29 | 1.246819 | 1.246819 | 1.0000 |
| 30 | 1.274592 | 1.274592 | 1.0000 |
| 31 | 1.317255 | 1.317255 | 1.0000 |

### Per-Sample Summary

| sample_idx | skill | swap_verb | mean_delta_orig | mean_delta_textKV | mean_ratio |
|-----------|-------|-----------|----------------|------------------|-----------|
| 1 | move | fold | 1.233781 | 1.233781 | 1.0000 |
| 4 | pick | place | 0.486166 | 0.486166 | 1.0000 |
| 5 | move | fold | 1.167914 | 1.167914 | 1.0000 |
| 6 | open | close | 0.556740 | 0.556740 | 1.0000 |
| 12 | close | open | 0.518288 | 0.518288 | 1.0000 |
| 13 | pick | place | 0.526283 | 0.526283 | 1.0000 |
| 16 | close | open | 0.546109 | 0.546109 | 1.0000 |
| 17 | move | fold | 1.202017 | 1.202017 | 1.0000 |

### Overall Model Summary

| Metric | Value |
|--------|-------|
| n_records | 80 |
| n_samples | 8 |
| mean_delta_orig | 0.779662 |
| mean_delta_textKV | 0.779662 |
| overall_ratio (orig/textKV) | 1.0000 |

---

## openvla-7b

### Skill Labels

| Skill | Count |
|-------|-------|
| close | 2 |
| fold | 3 |
| move | 3 |
| open | 1 |
| pick | 6 |
| place | 5 |
| **Total** | **20** |

### Counterfactual Results — Raw Records (80 total)

#### Sample 1 (skill=move, swap_verb=fold)

| layer | delta_orig | delta_textKV | ratio |
|-------|-----------|-------------|-------|
| 22 | 0.654230 | 0.654230 | 1.0000 |
| 23 | 0.665743 | 0.665743 | 1.0000 |
| 24 | 0.648643 | 0.648643 | 1.0000 |
| 25 | 0.626855 | 0.626855 | 1.0000 |
| 26 | 0.539994 | 0.539994 | 1.0000 |
| 27 | 0.530512 | 0.530512 | 1.0000 |
| 28 | 0.519841 | 0.519841 | 1.0000 |
| 29 | 0.501837 | 0.501837 | 1.0000 |
| 30 | 0.485553 | 0.485553 | 1.0000 |
| 31 | 0.848544 | 0.848544 | 1.0000 |

#### Sample 4 (skill=pick, swap_verb=place)

| layer | delta_orig | delta_textKV | ratio |
|-------|-----------|-------------|-------|
| 22 | 0.195799 | 0.195799 | 1.0000 |
| 23 | 0.212072 | 0.212072 | 1.0000 |
| 24 | 0.230490 | 0.230490 | 1.0000 |
| 25 | 0.241642 | 0.241642 | 1.0000 |
| 26 | 0.223615 | 0.223615 | 1.0000 |
| 27 | 0.222689 | 0.222689 | 1.0000 |
| 28 | 0.223686 | 0.223686 | 1.0000 |
| 29 | 0.225104 | 0.225104 | 1.0000 |
| 30 | 0.223745 | 0.223745 | 1.0000 |
| 31 | 0.330214 | 0.330214 | 1.0000 |

#### Sample 5 (skill=move, swap_verb=fold)

| layer | delta_orig | delta_textKV | ratio |
|-------|-----------|-------------|-------|
| 22 | 0.850143 | 0.850143 | 1.0000 |
| 23 | 0.850638 | 0.850638 | 1.0000 |
| 24 | 0.821770 | 0.821770 | 1.0000 |
| 25 | 0.786071 | 0.786071 | 1.0000 |
| 26 | 0.712913 | 0.712913 | 1.0000 |
| 27 | 0.684110 | 0.684110 | 1.0000 |
| 28 | 0.660150 | 0.660150 | 1.0000 |
| 29 | 0.631278 | 0.631278 | 1.0000 |
| 30 | 0.605035 | 0.605035 | 1.0000 |
| 31 | 1.054520 | 1.054520 | 1.0000 |

#### Sample 6 (skill=open, swap_verb=close)

| layer | delta_orig | delta_textKV | ratio |
|-------|-----------|-------------|-------|
| 22 | 1.049454 | 1.049454 | 1.0000 |
| 23 | 1.049835 | 1.049835 | 1.0000 |
| 24 | 1.011367 | 1.011367 | 1.0000 |
| 25 | 0.982629 | 0.982629 | 1.0000 |
| 26 | 0.897487 | 0.897487 | 1.0000 |
| 27 | 0.885736 | 0.885736 | 1.0000 |
| 28 | 0.862671 | 0.862671 | 1.0000 |
| 29 | 0.848507 | 0.848507 | 1.0000 |
| 30 | 0.840417 | 0.840417 | 1.0000 |
| 31 | 1.227919 | 1.227919 | 1.0000 |

#### Sample 12 (skill=close, swap_verb=open)

| layer | delta_orig | delta_textKV | ratio |
|-------|-----------|-------------|-------|
| 22 | 0.827176 | 0.827176 | 1.0000 |
| 23 | 0.830383 | 0.830383 | 1.0000 |
| 24 | 0.825856 | 0.825856 | 1.0000 |
| 25 | 0.807972 | 0.807972 | 1.0000 |
| 26 | 0.756569 | 0.756569 | 1.0000 |
| 27 | 0.754367 | 0.754367 | 1.0000 |
| 28 | 0.726297 | 0.726297 | 1.0000 |
| 29 | 0.709290 | 0.709290 | 1.0000 |
| 30 | 0.680197 | 0.680197 | 1.0000 |
| 31 | 0.923872 | 0.923872 | 1.0000 |

#### Sample 13 (skill=pick, swap_verb=place)

| layer | delta_orig | delta_textKV | ratio |
|-------|-----------|-------------|-------|
| 22 | 0.193818 | 0.193818 | 1.0000 |
| 23 | 0.202987 | 0.202987 | 1.0000 |
| 24 | 0.201842 | 0.201842 | 1.0000 |
| 25 | 0.207538 | 0.207538 | 1.0000 |
| 26 | 0.194585 | 0.194585 | 1.0000 |
| 27 | 0.200675 | 0.200675 | 1.0000 |
| 28 | 0.204189 | 0.204189 | 1.0000 |
| 29 | 0.199154 | 0.199154 | 1.0000 |
| 30 | 0.202630 | 0.202630 | 1.0000 |
| 31 | 0.329054 | 0.329054 | 1.0000 |

#### Sample 16 (skill=close, swap_verb=open)

| layer | delta_orig | delta_textKV | ratio |
|-------|-----------|-------------|-------|
| 22 | 0.836820 | 0.836820 | 1.0000 |
| 23 | 0.852857 | 0.852857 | 1.0000 |
| 24 | 0.835897 | 0.835897 | 1.0000 |
| 25 | 0.826473 | 0.826473 | 1.0000 |
| 26 | 0.765572 | 0.765572 | 1.0000 |
| 27 | 0.760640 | 0.760640 | 1.0000 |
| 28 | 0.738331 | 0.738331 | 1.0000 |
| 29 | 0.706364 | 0.706364 | 1.0000 |
| 30 | 0.685268 | 0.685268 | 1.0000 |
| 31 | 0.999187 | 0.999187 | 1.0000 |

#### Sample 17 (skill=move, swap_verb=fold)

| layer | delta_orig | delta_textKV | ratio |
|-------|-----------|-------------|-------|
| 22 | 0.921833 | 0.921833 | 1.0000 |
| 23 | 0.930539 | 0.930539 | 1.0000 |
| 24 | 0.877884 | 0.877884 | 1.0000 |
| 25 | 0.847131 | 0.847131 | 1.0000 |
| 26 | 0.746910 | 0.746910 | 1.0000 |
| 27 | 0.711367 | 0.711367 | 1.0000 |
| 28 | 0.686799 | 0.686799 | 1.0000 |
| 29 | 0.657510 | 0.657510 | 1.0000 |
| 30 | 0.648868 | 0.648868 | 1.0000 |
| 31 | 1.015361 | 1.015361 | 1.0000 |

### Per-Sample Summary

| sample_idx | skill | swap_verb | mean_delta_orig | mean_delta_textKV | mean_ratio |
|-----------|-------|-----------|----------------|------------------|-----------|
| 1 | move | fold | 0.602175 | 0.602175 | 1.0000 |
| 4 | pick | place | 0.232906 | 0.232906 | 1.0000 |
| 5 | move | fold | 0.765663 | 0.765663 | 1.0000 |
| 6 | open | close | 0.965602 | 0.965602 | 1.0000 |
| 12 | close | open | 0.784198 | 0.784198 | 1.0000 |
| 13 | pick | place | 0.213647 | 0.213647 | 1.0000 |
| 16 | close | open | 0.800741 | 0.800741 | 1.0000 |
| 17 | move | fold | 0.804420 | 0.804420 | 1.0000 |

### Overall Model Summary

| Metric | Value |
|--------|-------|
| n_records | 80 |
| n_samples | 8 |
| mean_delta_orig | 0.646169 |
| mean_delta_textKV | 0.646169 |
| overall_ratio (orig/textKV) | 1.0000 |

---

## spatialvla-4b

### Skill Labels

| Skill | Count |
|-------|-------|
| close | 2 |
| fold | 3 |
| move | 3 |
| open | 1 |
| pick | 6 |
| place | 5 |
| **Total** | **20** |

### Counterfactual Results — Raw Records (90 total)

#### Sample 1 (skill=move, swap_verb=fold)

| layer | delta_orig | delta_textKV | ratio |
|-------|-----------|-------------|-------|
| 16 | 0.235150 | 0.000000 | inf |
| 17 | 0.245784 | 0.000000 | inf |
| 18 | 0.256475 | 0.000000 | inf |
| 19 | 0.249067 | 0.000000 | inf |
| 20 | 0.251095 | 0.000000 | inf |
| 21 | 0.254655 | 0.000000 | inf |
| 22 | 0.284808 | 0.000000 | inf |
| 23 | 0.328719 | 0.000000 | inf |
| 24 | 0.354887 | 0.000000 | inf |
| 25 | 0.487328 | 0.000000 | inf |

#### Sample 4 (skill=pick, swap_verb=place)

| layer | delta_orig | delta_textKV | ratio |
|-------|-----------|-------------|-------|
| 16 | 0.130207 | 0.000000 | inf |
| 17 | 0.137006 | 0.000000 | inf |
| 18 | 0.139289 | 0.000000 | inf |
| 19 | 0.131689 | 0.000000 | inf |
| 20 | 0.130950 | 0.000000 | inf |
| 21 | 0.128084 | 0.000000 | inf |
| 22 | 0.137635 | 0.000000 | inf |
| 23 | 0.154256 | 0.000000 | inf |
| 24 | 0.177148 | 0.000000 | inf |
| 25 | 0.272180 | 0.000000 | inf |

#### Sample 5 (skill=move, swap_verb=fold)

| layer | delta_orig | delta_textKV | ratio |
|-------|-----------|-------------|-------|
| 16 | 0.223407 | 0.000000 | inf |
| 17 | 0.240411 | 0.000000 | inf |
| 18 | 0.248369 | 0.000000 | inf |
| 19 | 0.237283 | 0.000000 | inf |
| 20 | 0.241313 | 0.000000 | inf |
| 21 | 0.257373 | 0.000000 | inf |
| 22 | 0.326867 | 0.000000 | inf |
| 23 | 0.370150 | 0.000000 | inf |
| 24 | 0.439200 | 0.000000 | inf |
| 25 | 0.534608 | 0.000000 | inf |

#### Sample 6 (skill=open, swap_verb=close)

| layer | delta_orig | delta_textKV | ratio |
|-------|-----------|-------------|-------|
| 16 | 0.348376 | 0.000000 | inf |
| 17 | 0.366948 | 0.000000 | inf |
| 18 | 0.360113 | 0.000000 | inf |
| 19 | 0.349279 | 0.000000 | inf |
| 20 | 0.353292 | 0.000000 | inf |
| 21 | 0.356222 | 0.000000 | inf |
| 22 | 0.390280 | 0.000000 | inf |
| 23 | 0.431395 | 0.000000 | inf |
| 24 | 0.486453 | 0.000000 | inf |
| 25 | 0.660573 | 0.000000 | inf |

#### Sample 10 (skill=fold, swap_verb=move)

| layer | delta_orig | delta_textKV | ratio |
|-------|-----------|-------------|-------|
| 16 | 0.328277 | 0.000000 | inf |
| 17 | 0.351215 | 0.000000 | inf |
| 18 | 0.369877 | 0.000000 | inf |
| 19 | 0.354718 | 0.000000 | inf |
| 20 | 0.344595 | 0.000000 | inf |
| 21 | 0.343390 | 0.000000 | inf |
| 22 | 0.384216 | 0.000000 | inf |
| 23 | 0.434989 | 0.000000 | inf |
| 24 | 0.431833 | 0.000000 | inf |
| 25 | 0.646857 | 0.000000 | inf |

#### Sample 12 (skill=close, swap_verb=open)

| layer | delta_orig | delta_textKV | ratio |
|-------|-----------|-------------|-------|
| 16 | 0.345892 | 0.000000 | inf |
| 17 | 0.340827 | 0.000000 | inf |
| 18 | 0.339312 | 0.000000 | inf |
| 19 | 0.335425 | 0.000000 | inf |
| 20 | 0.325626 | 0.000000 | inf |
| 21 | 0.314400 | 0.000000 | inf |
| 22 | 0.334434 | 0.000000 | inf |
| 23 | 0.355308 | 0.000000 | inf |
| 24 | 0.376440 | 0.000000 | inf |
| 25 | 0.557560 | 0.000000 | inf |

#### Sample 13 (skill=pick, swap_verb=place)

| layer | delta_orig | delta_textKV | ratio |
|-------|-----------|-------------|-------|
| 16 | 0.109435 | 0.000000 | inf |
| 17 | 0.117292 | 0.000000 | inf |
| 18 | 0.117154 | 0.000000 | inf |
| 19 | 0.113144 | 0.000000 | inf |
| 20 | 0.113852 | 0.000000 | inf |
| 21 | 0.114772 | 0.000000 | inf |
| 22 | 0.134848 | 0.000000 | inf |
| 23 | 0.163052 | 0.000000 | inf |
| 24 | 0.181325 | 0.000000 | inf |
| 25 | 0.244904 | 0.000000 | inf |

#### Sample 16 (skill=close, swap_verb=open)

| layer | delta_orig | delta_textKV | ratio |
|-------|-----------|-------------|-------|
| 16 | 0.363099 | 0.000000 | inf |
| 17 | 0.350896 | 0.000000 | inf |
| 18 | 0.333677 | 0.000000 | inf |
| 19 | 0.325566 | 0.000000 | inf |
| 20 | 0.320971 | 0.000000 | inf |
| 21 | 0.329119 | 0.000000 | inf |
| 22 | 0.364881 | 0.000000 | inf |
| 23 | 0.420975 | 0.000000 | inf |
| 24 | 0.423535 | 0.000000 | inf |
| 25 | 0.633246 | 0.000000 | inf |

#### Sample 17 (skill=move, swap_verb=fold)

| layer | delta_orig | delta_textKV | ratio |
|-------|-----------|-------------|-------|
| 16 | 0.315399 | 0.000000 | inf |
| 17 | 0.331254 | 0.000000 | inf |
| 18 | 0.326076 | 0.000000 | inf |
| 19 | 0.316298 | 0.000000 | inf |
| 20 | 0.313633 | 0.000000 | inf |
| 21 | 0.319150 | 0.000000 | inf |
| 22 | 0.394363 | 0.000000 | inf |
| 23 | 0.472625 | 0.000000 | inf |
| 24 | 0.531221 | 0.000000 | inf |
| 25 | 0.709521 | 0.000000 | inf |

### Per-Sample Summary

| sample_idx | skill | swap_verb | mean_delta_orig | mean_delta_textKV | mean_ratio |
|-----------|-------|-----------|----------------|------------------|-----------|
| 1 | move | fold | 0.294797 | 0.000000 | inf |
| 4 | pick | place | 0.153844 | 0.000000 | inf |
| 5 | move | fold | 0.311898 | 0.000000 | inf |
| 6 | open | close | 0.410293 | 0.000000 | inf |
| 10 | fold | move | 0.398997 | 0.000000 | inf |
| 12 | close | open | 0.362522 | 0.000000 | inf |
| 13 | pick | place | 0.140978 | 0.000000 | inf |
| 16 | close | open | 0.386596 | 0.000000 | inf |
| 17 | move | fold | 0.402954 | 0.000000 | inf |

### Overall Model Summary

| Metric | Value |
|--------|-------|
| n_records | 90 |
| n_samples | 9 |
| mean_delta_orig | 0.318098 |
| mean_delta_textKV | 0.000000 |
| overall_ratio (orig/textKV) | inf |

---

## tracevla-phi3v

### Skill Labels

| Skill | Count |
|-------|-------|
| close | 2 |
| fold | 3 |
| move | 3 |
| open | 1 |
| pick | 6 |
| place | 5 |
| **Total** | **20** |

### Counterfactual Results — Raw Records (80 total)

#### Sample 1 (skill=move, swap_verb=fold)

| layer | delta_orig | delta_textKV | ratio |
|-------|-----------|-------------|-------|
| 22 | 1.026892 | 0.956275 | 1.0738 |
| 23 | 1.027557 | 0.955867 | 1.0750 |
| 24 | 0.989296 | 0.955142 | 1.0358 |
| 25 | 0.989575 | 0.946378 | 1.0456 |
| 26 | 0.980697 | 0.945606 | 1.0371 |
| 27 | 0.950894 | 0.945719 | 1.0055 |
| 28 | 0.944485 | 0.951865 | 0.9922 |
| 29 | 0.896527 | 0.547754 | 1.6367 |
| 30 | 0.879397 | 0.500962 | 1.7554 |
| 31 | 0.855482 | 1.194089 | 0.7164 |

#### Sample 4 (skill=pick, swap_verb=place)

| layer | delta_orig | delta_textKV | ratio |
|-------|-----------|-------------|-------|
| 22 | 0.276952 | 0.000000 | inf |
| 23 | 0.297873 | 0.000000 | inf |
| 24 | 0.319643 | 0.000000 | inf |
| 25 | 0.337598 | 0.000000 | inf |
| 26 | 0.363825 | 0.000000 | inf |
| 27 | 0.379444 | 0.000000 | inf |
| 28 | 0.378227 | 0.000000 | inf |
| 29 | 0.390258 | 0.000000 | inf |
| 30 | 0.409096 | 0.000000 | inf |
| 31 | 0.473826 | 0.000000 | inf |

#### Sample 5 (skill=move, swap_verb=fold)

| layer | delta_orig | delta_textKV | ratio |
|-------|-----------|-------------|-------|
| 22 | 0.814459 | 0.780437 | 1.0436 |
| 23 | 0.820054 | 0.780166 | 1.0511 |
| 24 | 0.824285 | 0.779658 | 1.0572 |
| 25 | 0.836969 | 0.774858 | 1.0802 |
| 26 | 0.839855 | 0.774393 | 1.0845 |
| 27 | 0.821859 | 0.775406 | 1.0599 |
| 28 | 0.828536 | 0.789825 | 1.0490 |
| 29 | 0.819310 | 0.468805 | 1.7477 |
| 30 | 0.803658 | 0.434539 | 1.8494 |
| 31 | 0.833988 | 0.986033 | 0.8458 |

#### Sample 6 (skill=open, swap_verb=close)

| layer | delta_orig | delta_textKV | ratio |
|-------|-----------|-------------|-------|
| 22 | 1.164824 | 0.000000 | inf |
| 23 | 1.167107 | 0.000000 | inf |
| 24 | 1.084064 | 0.000000 | inf |
| 25 | 1.065208 | 0.000000 | inf |
| 26 | 1.051339 | 0.000000 | inf |
| 27 | 0.989385 | 0.000000 | inf |
| 28 | 0.951883 | 0.000000 | inf |
| 29 | 0.918333 | 0.000000 | inf |
| 30 | 0.899493 | 0.000000 | inf |
| 31 | 0.832196 | 0.000000 | inf |

#### Sample 12 (skill=close, swap_verb=open)

| layer | delta_orig | delta_textKV | ratio |
|-------|-----------|-------------|-------|
| 22 | 1.062278 | 0.000000 | inf |
| 23 | 1.086704 | 0.000000 | inf |
| 24 | 1.059743 | 0.000000 | inf |
| 25 | 1.060615 | 0.000000 | inf |
| 26 | 1.055801 | 0.000000 | inf |
| 27 | 1.020456 | 0.000000 | inf |
| 28 | 0.992421 | 0.000000 | inf |
| 29 | 0.970517 | 0.000000 | inf |
| 30 | 0.954481 | 0.000000 | inf |
| 31 | 1.048346 | 0.000000 | inf |

#### Sample 13 (skill=pick, swap_verb=place)

| layer | delta_orig | delta_textKV | ratio |
|-------|-----------|-------------|-------|
| 22 | 0.272188 | 0.000000 | inf |
| 23 | 0.292673 | 0.000000 | inf |
| 24 | 0.303013 | 0.000000 | inf |
| 25 | 0.319630 | 0.000000 | inf |
| 26 | 0.341519 | 0.000000 | inf |
| 27 | 0.352968 | 0.000000 | inf |
| 28 | 0.364957 | 0.000000 | inf |
| 29 | 0.389408 | 0.000000 | inf |
| 30 | 0.419426 | 0.000000 | inf |
| 31 | 0.376138 | 0.000000 | inf |

#### Sample 16 (skill=close, swap_verb=open)

| layer | delta_orig | delta_textKV | ratio |
|-------|-----------|-------------|-------|
| 22 | 1.050722 | 0.000000 | inf |
| 23 | 1.075039 | 0.000000 | inf |
| 24 | 1.051585 | 0.000000 | inf |
| 25 | 1.053036 | 0.000000 | inf |
| 26 | 1.070789 | 0.000000 | inf |
| 27 | 1.032884 | 0.000000 | inf |
| 28 | 0.997471 | 0.000000 | inf |
| 29 | 0.972558 | 0.000000 | inf |
| 30 | 0.981751 | 0.000000 | inf |
| 31 | 1.471446 | 0.000000 | inf |

#### Sample 17 (skill=move, swap_verb=fold)

| layer | delta_orig | delta_textKV | ratio |
|-------|-----------|-------------|-------|
| 22 | 0.905060 | 0.955948 | 0.9468 |
| 23 | 0.933095 | 0.955558 | 0.9765 |
| 24 | 0.901702 | 0.954866 | 0.9443 |
| 25 | 0.924180 | 0.946479 | 0.9764 |
| 26 | 0.919870 | 0.945775 | 0.9726 |
| 27 | 0.893639 | 0.946080 | 0.9446 |
| 28 | 0.884699 | 0.962748 | 0.9189 |
| 29 | 0.849331 | 0.568752 | 1.4933 |
| 30 | 0.818136 | 0.526614 | 1.5536 |
| 31 | 0.678076 | 1.231227 | 0.5507 |

### Per-Sample Summary

| sample_idx | skill | swap_verb | mean_delta_orig | mean_delta_textKV | mean_ratio |
|-----------|-------|-----------|----------------|------------------|-----------|
| 1 | move | fold | 0.954080 | 0.889966 | 1.0720 |
| 4 | pick | place | 0.362674 | 0.000000 | inf |
| 5 | move | fold | 0.824297 | 0.734412 | 1.1224 |
| 6 | open | close | 1.012383 | 0.000000 | inf |
| 12 | close | open | 1.031136 | 0.000000 | inf |
| 13 | pick | place | 0.343192 | 0.000000 | inf |
| 16 | close | open | 1.075728 | 0.000000 | inf |
| 17 | move | fold | 0.870779 | 0.899405 | 0.9682 |

### Overall Model Summary

| Metric | Value |
|--------|-------|
| n_records | 80 |
| n_samples | 8 |
| mean_delta_orig | 0.809284 |
| mean_delta_textKV | 0.315473 |
| overall_ratio (orig/textKV) | 2.5653 |

---
