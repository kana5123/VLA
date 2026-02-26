# Cross-Architecture Attention Bottleneck Analysis

> Date: 2026-02-25
> Status: Design Phase
> Previous: verify_attention_sinks.py (3-part VAR sink verification)

---

## Motivation

### Key Discovery (OpenVLA-7B)
VAR 논문(ICLR 2025)의 3-part attention sink 정의를 OpenVLA에 적용한 결과:
- **Condition A** (Cross-token consistency): PASS — 32/32 layers
- **Condition B** (Hidden state spike): PASS — φ(token0) ≈ 50.6 >> τ=20
- **Condition C** (Low contribution): **FAIL** — token 0의 value norm이 다른 토큰 대비 14~75x 높음

더 심각한 발견:
- Layer 2부터 token 0이 **출력의 99%+ 독점**
- 256개 vision token 중 사실상 1개만 action prediction에 관여
- 이는 VAR 논문의 VLM (LLaVA 등)과 근본적으로 다른 현상

### Research Question
> VLA 모델에서 vision token 0의 information bottleneck은 보편적 현상인가?
> Autoregressive VLA, Diffusion VLA, VLM 간에 어떤 차이가 있는가?
> 이 bottleneck을 해소하면 VLA 성능이 향상되는가?

---

## Part A: Cross-Architecture Analysis

### 분석 대상 모델 (7개)

#### Autoregressive VLA (3개)
| Model | Backbone | Hidden | Heads | Layers | Vision Tokens | Status |
|-------|----------|--------|-------|--------|---------------|--------|
| OpenVLA-7B | LLaMA-2 | 4096 | 32 | 32 | 256 | **Done** |
| TraceVLA-Phi3V | Phi-3-V | 3072 | 32 | 32 | 313 | Registry ready |
| SpatialVLA-4B | Gemma-2 | 2304 | 8 | 26 | 256 | Registry ready |

#### VLM — VQA Only (2개)
| Model | Backbone | Hidden | Heads | Layers | Vision Tokens | Status |
|-------|----------|--------|-------|--------|---------------|--------|
| LLaVA-1.5-7B | LLaMA-2 | 4096 | 32 | 32 | 576 | Need registry |
| InternVL2-8B | InternLM2 | 4096 | 32 | 32 | 256 | Need registry |

#### Diffusion VLA (2개)
| Model | Architecture | Action Method | Status |
|-------|-------------|---------------|--------|
| π0 (open-pi-zero) | PaliGemma 3B + Flow Expert | Flow matching | Need new impl |
| Dita | LLaMA2-style DiT 334M | Diffusion denoising | Need new impl |

### 분석 항목 (모델당 동일 파이프라인)

1. **3-Part Sink Verification** (Condition A, B, C)
   - A: cross-token attention consistency (8+ text query tokens)
   - B: hidden state φ(x) spike analysis (τ=20)
   - C: per-unit contribution via value norm (||x_j W_V||)

2. **All-Layer Contribution Profile**
   - 모든 레이어에서: token 0 기여율 (α_0 × VN_0) / Σ(α_j × VN_j)
   - Bottleneck onset layer: 기여율이 50%를 넘는 최초 레이어
   - Bottleneck severity: 평균 기여율 (Layer 2+ 에서)

3. **Value Norm Distribution**
   - Per-layer: VN_0 / mean(VN_others) ratio
   - Hidden state dimension spike analysis (D_sink dimensions)
   - Cross-sample variance (same instruction, different images)

4. **Cross-Model Comparison Metrics**
   - Bottleneck Score = mean contribution ratio (L2~last)
   - Onset Layer = first layer where ratio > 50%
   - Sink Dimensions = number of spike dims in D_sink

### 실행 Phase

#### Phase 1: Autoregressive VLA (TraceVLA + SpatialVLA)
- `verify_attention_sinks.py --model tracevla-phi3v` 실행
- `verify_attention_sinks.py --model spatialvla-4b` 실행
- All-layer contribution 스크립트 실행
- 결과: OpenVLA와 동일한 bottleneck 패턴 확인 여부

#### Phase 2: VLM Comparison (LLaVA-1.5 + InternVL2)
- model_registry.py에 LLaVA-1.5-7B, InternVL2-8B 추가
- VLM은 action token 대신 text generation → query token을 마지막 text token으로 사용
- 결과: VLM에서는 bottleneck이 없는지 (VAR 논문 결과 재현)

#### Phase 3: Diffusion VLA (π0 + Dita)
- open-pi-zero repo clone, PaliGemma VLM 부분의 attention 추출
- Dita repo clone, DiT block의 attention 추출
- Diffusion 아키텍처는 cross-attention 기반이므로 분석 방식이 다를 수 있음
- 결과: Diffusion VLA에서도 bottleneck이 존재하는지

---

## Part B: 방법론 제안 (Part A 결과 기반)

Part A 완료 후 결과를 분석하여 방법론 방향을 결정.

### 가설별 방법론 후보

**가설 1: VLA fine-tuning이 bottleneck을 유발한다**
→ Pre-trained VLM (LLaVA)에서는 없고 VLA에서만 있다면:
- Fine-tuning 시 value norm regularization 도입
- Token 0의 W_V norm을 다른 토큰과 유사하게 제한

**가설 2: Vision encoder 구조가 원인이다**
→ Prismatic (dual encoder) vs SigLIP (single) vs CLIP 간 차이가 있다면:
- Vision encoder output에 normalization layer 추가
- Multi-token pooling으로 bottleneck 분산

**가설 3: Autoregressive 구조 자체가 원인이다**
→ Diffusion VLA에서는 bottleneck이 없다면:
- Autoregressive action generation의 근본적 한계 지적
- Hybrid 아키텍처 제안 (autoregressive reasoning + diffusion action)

---

## Output Structure

```
outputs/sink_verification/
├── openvla-7b/           (Done)
│   ├── sink_report.json
│   ├── condition_A_heatmap.png
│   ├── condition_B_phi.png
│   ├── condition_C_contribution.png
│   ├── summary_all_conditions.png
│   └── all_layer_contribution.png
├── tracevla-phi3v/       (Phase 1)
├── spatialvla-4b/        (Phase 1)
├── llava-1.5-7b/         (Phase 2)
├── internvl2-8b/         (Phase 2)
├── pi0/                  (Phase 3)
├── dita/                 (Phase 3)
└── cross_model_comparison/
    ├── comparison_table.json
    ├── bottleneck_comparison.png
    └── contribution_profiles.png
```

---

## References

- [VAR: See What You Are Told (ICLR 2025)](https://arxiv.org/abs/2503.03321)
- [When Attention Sink Emerges (ICLR 2025 Spotlight)](https://github.com/sail-sg/Attention-Sink)
- [π0: A Vision-Language-Action Flow Model](https://arxiv.org/abs/2410.24164)
- [Dita: Scaling Diffusion Transformer (ICCV 2025)](https://arxiv.org/abs/2503.19757)
- [DiffusionVLA (ICML 2025)](https://arxiv.org/abs/2412.03293)
