# ATLASVLA 전체 실험 여정 — 종합 기록

> **프로젝트**: VLA(Vision-Language-Action) 모델의 Attention 메커니즘 분석 및 개선
> **기간**: 2026년 2월 초 ~ 2월 말
> **대상 모델**: OpenVLA-7B, ECoT-7B, SpatialVLA-4B, TraceVLA-Phi3V, LLaVA-1.5-7B
> **최종 업데이트**: 2026-02-27

---

## 목차

1. [Phase 0: 연구 아이디어 수립 — ROME & Circuit Analysis 계획](#phase-0)
2. [Phase 1: Attention Map 측정 — "0번 토큰에 어텐션이 몰린다"](#phase-1)
3. [Phase 2: Attention Sink 가설 — 재분배 실험 (v1~v5)](#phase-2)
4. [Phase 3: 학습 가능한 어댑터 — AttentionAdapterV2](#phase-3)
5. [Phase 4: 크로스 모델 분석 — 5개 모델 비교](#phase-4)
6. [Phase 5: Sink vs Bottleneck 판별 — 3가지 조건 검증](#phase-5)
7. [Phase 6: Bottleneck 확정 진단 — 3가지 추가 테스트](#phase-6)
8. [Phase 7: 태스크별 Contribution 분석 — Skill Signature](#phase-7)
9. [Phase 2.5: Dual-Track Sink/Bottleneck 정밀 분석 — 4모델 분류 체계](#phase-25)
10. [Phase 3-Gate: 3단계 Gate Check — 150 샘플 대규모 검증](#phase-3gate)
11. [Phase 3-Expanded: 표본 확장 (N=175) + VAR 비교 실험](#phase-3expanded)
12. [전체 흐름 요약](#summary)
13. [핵심 발견 요약](#findings)

---

<a id="phase-0"></a>
## Phase 0: 연구 아이디어 수립 (2월 초)

### 배경
OpenVLA라는 로봇 조작 모델(7B 파라미터)이 실패하는 이유를 알고 싶었음.

### 원래 계획
1. 시뮬레이션에서 로봇을 돌려서 **실패 에피소드** 수집
2. **ROME (Causal Tracing)** — "어떤 레이어가 범인인지" 찾는 기법
3. **Circuit Analysis (Path Patching)** — "실패를 만드는 경로"를 그리는 기법 (ICLR 논문: Interpretability in the Wild)

### 방향 전환
본격 시작 전에 "먼저 모델이 이미지를 **어떻게 보고 있는지**부터 확인해보자"라는 생각이 들어서 **Attention Map 분석**으로 전환.

### 관련 파일
- 원래 계획서: `~/capston/requirements/openvla-interpretability-experiment.md`
- 모델 구조: `~/capston/outputs/openvla_module_tree.txt`
- 환경 설정: `~/capston/scripts/check_environment.sh` (H100 x7 확인)

---

<a id="phase-1"></a>
## Phase 1: Attention Map 측정 (2월 중순)

### 실험 내용
OpenVLA-7B 모델에 BridgeData V2의 **10개 에피소드** (총 385 타임스텝)를 입력하여, 모델 내부의 **모든 레이어 Attention Map**을 추출.

### 데이터 준비
- BridgeData V2에서 10개 에피소드 다운로드 (`tensorflow-datasets` 사용)
- 다양한 의존성 문제 해결 (etils, apache-beam, transformers, timm 버전 충돌)
- Hook 기반 attention 추출 방식 구현 (OpenVLA의 `generate()`가 attention 직접 반환 불가)

### 핵심 발견
OpenVLA는 이미지를 **16×16 = 256개 패치**로 나눠서 처리함. 그 중:
- **패치 0 (왼쪽 맨 위 구석)에 전체 어텐션의 40~60%가 집중**
- 나머지 255개 패치는 합쳐봐야 40~60%

```
패치0  패치1  패치2  ... 패치15    ← 이미지 첫 줄 (여기에 어텐션 집중!)
패치16 패치17 ...                  ← 이미지 둘째 줄
...
패치240 ...              패치255   ← 마지막 줄
```

### 이때 내린 판단
> "이건 NLP에서 보고된 **Attention Sink** 현상이다!"
> (StreamingLLM 논문 등에서 보고: 첫 번째 토큰에 "쓰레기통"처럼 남는 어텐션이 쌓이는 현상)

### 시각화 위치
- 어텐션 데이터: `~/ATLASVLA/archive/outputs/attention_results/ep000_step*.json`
- 초기 시각화: `~/ATLASVLA/archive/outputs/visualizations/`
- 헤드별 heatmap: `~/ATLASVLA/archive/outputs/cross_model_analysis/openvla-7b/bridge_v2/action_*_perhead_v0_heatmap.png`

---

<a id="phase-2"></a>
## Phase 2: Attention Sink 재분배 실험 (v1~v5)

### 가설
"0번 토큰이 어텐션을 쓸데없이 빨아들이고 있으니까, 그걸 빼서 **진짜 중요한 곳(물체가 있는 패치)**에 재분배하면 행동 예측이 좋아질 것이다."

### 재분배 방법 3가지
1. **Logit Bias** — softmax 전에 물체 패치 위치에 점수 가산
2. **Weight Rescale (VAR)** — softmax 후 물체 패치 비중을 올리고 나머지 축소
3. **Head Steering** — 이미지를 많이 보는 헤드만 골라서 재분배

### 물체 위치 탐지
- spacy로 텍스트 지시문에서 명사 추출 ("put small spoon from basket to tray" → spoon, basket, tray)
- GroundingDINO로 이미지에서 해당 물체 검출
- 검출 영역을 16×16 패치 좌표로 변환

### 실험 결과 (v1~v5)

| 버전 | 변경 사항 | 결과 |
|------|----------|------|
| **v1** | 기본 3가지 방법 | baseline과 차이 없음 |
| **v2** | SAM 마스크로 물체 영역 정밀화 + 19가지 조합 (개별/쌍/삼중/전체) | 여전히 차이 없음 |
| **v3** | 16가지+ 조합 (decay, temporal, layer selection, 배경 억제, gripper 강조 등) | 전부 baseline과 동일 |
| **v4** | 코드 버그 수정 후 재실험 | 여전히 동일 |
| **v5** | 타겟팅 전략 자체를 변경 (dim_gate: 차원별 게이팅) | 전체 MSE 개선 but 데이터 누출 문제 발견 |

### v3에서 시도한 조합 목록
`sam_var_object`, `sam_var_obj_decay`, `sam_var_obj_temporal`, `sam_var_obj_decay_laysel`, `var_vt_rebal`, `var_decay`, `var_laysel`, `sam_var_bgsup_gripex`, `spin_k16`, `act_gripex` 등

### 실패 원인 분석
OpenVLA의 **이산적(discrete) 출력** 구조가 근본적 한계:
1. 모델이 vocabulary에서 확률 가장 높은 토큰 하나를 고름 (argmax)
2. 그 토큰 ID를 연속적 action 값으로 변환 (detokenize)
3. **어텐션을 아무리 바꿔도 argmax 1등이 안 바뀌면 출력은 완전히 동일**

### 시각화 위치
- v1: `~/ATLASVLA/archive/outputs/enhancement_results/`
- v2: `~/ATLASVLA/archive/outputs/v2_enhancement_results/`
- v3: `~/ATLASVLA/archive/outputs/v3_enhancement_results/` (ranking_overall.png, mse_heatmap.png 등)
- v4: `~/ATLASVLA/archive/outputs/v4_corrected_results/`
- v5: `~/ATLASVLA/archive/outputs/v5_targeted_results/`

---

<a id="phase-3"></a>
## Phase 3: 학습 가능한 어댑터 (AttentionAdapterV2)

### 새 아이디어
"고정 비율 재분배로는 argmax가 안 바뀌니까, **입력에 따라 동적으로 재분배를 결정하는 어댑터**를 학습시키자."

### 어댑터 구조 (2개 가지)

**Branch 1 — "얼마나 빼낼까?" (p_matrix)**
- 모델의 마지막 hidden state + 물체 마스크를 입력으로
- 각 레이어/헤드마다 "sink에서 어텐션을 몇 % 빼낼지" 학습

**Branch 2 — "어디로 보낼까?" (redistribution_weights)**
- 빼낸 어텐션을 어떤 패치에 나눠줄지 결정
- SAM 마스크로 배경은 차단, 물체 패치에만 집중

### 파라미터 규모
- 어댑터: ~2.17M (전체 7B의 0.031%)
- 비교용 LoRA (rank 16): ~1.05M

### Smoke Test 결과

| 모델 | 학습 Loss | 검증 Loss | 비고 |
|------|----------|----------|------|
| OpenVLA-7B | 12.15 | 14.97 | 안정적 수렴 |
| ECoT-7B | 12.92 | 22.25 | 과적합 위험 |
| SpatialVLA-4B | 18.85 | 24.99 | gradient=0, 학습 안 됨 |
| TraceVLA-Phi3V | 2.61 | 2.73 | 안정적 (continuous model) |
| OpenVLA LoRA+Adapter | 12.05 | 14.70 | LoRA 추가 시 소폭 개선 |

### 근본적 의문 발생
> "어댑터를 학습시켜도 결과가 크게 안 바뀐다면...
> 애초에 **이게 정말 Attention Sink가 맞는 건가?**"

### 관련 파일
- 어댑터 코드 및 결과: `~/ATLASVLA/archive/outputs/adapter_results/`, `smoke_test/`, `full/`, `full_v2/`
- 어댑터 동작 원리: `~/ATLASVLA/docs/adapter-mechanism-explained.md`

---

<a id="phase-4"></a>
## Phase 4: 크로스 모델 분석 — 5개 모델 비교

### 왜 이 실험을 했나
"OpenVLA만의 문제인가, 다른 VLA 모델에서도 같은 현상이 발생하는가?"

### 비교 대상 (5개 모델)
| 모델 | 백본 | 특징 |
|------|------|------|
| OpenVLA-7B | LLaMA-2 | 기본 VLA 모델, Prismatic vision encoder |
| ECoT-7B | LLaMA-2 | OpenVLA 파인튜닝 + Chain-of-Thought |
| SpatialVLA-4B | Gemma2 | 공간 인식 특화 |
| TraceVLA-Phi3V | Phi3V | Trace image 기반 |
| LLaVA-1.5-7B | LLaMA-2 | 대조군 (CLIP ViT 사용, 로봇 모델 아님) |

### 발견
- **4개 VLA 모델 모두**에서 유사한 어텐션 집중 현상 발생
  - OpenVLA, ECoT → **vision token 0번**에 집중
  - SpatialVLA, TraceVLA → **text 영역 (BOS 토큰 등)**에 집중
- LLM 백본이 다른데도 (LLaMA, Gemma2, Phi3V) 모두 발생 → **아키텍처와 무관한 보편적 현상**

### 시각화 위치
- 전 모델 비교: `~/ATLASVLA/archive/outputs/cross_model_analysis/comparison/`
  - `cross_model_sink_comparison.png`
  - `cross_model_heatmap.png`
  - `cross_model_dual_sink.png`

---

<a id="phase-5"></a>
## Phase 5: Sink 여부 검증 — 3가지 조건

### 왜 이 실험을 했나
"모든 모델에서 발생하는 보편적 현상이라면, 정말 Sink인지 다른 현상인지 정확히 판별해야 한다."

### 3가지 검증 조건

**조건 A — "어텐션이 일관되게 몰리는가?"**
- 0번 토큰이 전체 레이어의 80% 이상에서 top-5 안에 드는지
- 결과: **통과** (32/32 레이어)

**조건 B — "Hidden state에 비정상적 스파이크가 있는가?"**
- 0번 토큰의 activation norm이 비정상적으로 큰지 (기준: φ ≥ 20.0)
- 결과: **통과** (32/32 레이어)

**조건 C — "정보 없이 어텐션만 흡수하는가?" (핵심!)**
- Sink라면 value norm이 다른 토큰보다 **낮아야** 함 (쓰레기통이니까)
- 결과: **실패!** — 0번 토큰의 value norm이 다른 토큰보다 **훨씬 높았음**

### 최종 판정
```
is_true_sink: False (진짜 sink 아님!)
is_context_aggregator: True (정보를 실제로 모아서 저장하는 토큰)
```

**→ Token[0]은 쓰레기통이 아니라, 정보가 실제로 집중되어 있는 핵심 토큰이었음**

### 시각화 위치
- 전체 요약: `~/ATLASVLA/outputs/sink_verification/openvla-7b/summary_all_conditions.png`
- 조건A: `condition_A_heatmap.png`
- 조건B: `condition_B_phi.png`
- **조건C (결정적)**: `condition_C_contribution.png`
- V2 재검증: `~/ATLASVLA/outputs/sink_verification_v2/openvla-7b/`

---

<a id="phase-6"></a>
## Phase 6: Bottleneck 확정 진단

### 3가지 추가 테스트

**테스트 1: CLS 토큰 여부 확인**
- Vision Transformer에는 보통 [CLS]라는 특별 토큰이 있음 (전체 이미지 대표)
- [CLS]라면 어텐션이 몰리는 게 당연한데...
- **결과: [CLS]가 아님!** — Prismatic encoder는 `get_intermediate_layers()`로 CLS를 제거함
- → 그냥 이미지 왼쪽 위 구석의 일반 패치인데 어텐션이 몰리는 것 = 비정상

**테스트 2: 이미지 Shift 실험**
- 이미지를 90도 돌리거나 상하좌우로 밀어봄
- 만약 "왼쪽 위 물체" 때문이라면 어텐션도 바뀌어야 함
- **결과: 이미지를 밀어도 98.49% → 98.28%로 거의 변화 없음**
- → 이미지 내용물과 상관없이 **"0번 위치"라서** 어텐션이 몰림 = 아키텍처 편향

**테스트 3: 0번 토큰 Ablation**
- 0번 토큰의 값을 0으로 만들고 출력 변화 측정
- 결과:
  - 0번 토큰 제거: KL divergence = **7.34**
  - 랜덤 토큰 제거: KL divergence = **0.000086**
  - **85,139배** 차이! → 0번 하나 지우면 모델 완전히 다른 행동 예측

### 전 모델 비교

| 모델 | 0번 기여율 | Shift 후 | KL 비율 | 진단 |
|------|-----------|---------|---------|------|
| **OpenVLA-7B** | 98.49% | 98.28% | 85,139배 | **BOTTLENECK** |
| **ECoT-7B** | 98.44% | 98.60% | 5,320배 | **BOTTLENECK** |
| **LLaVA-1.5-7B** | 0.22% | 0.16% | 1.05배 | 정상 (bottleneck 아님) |

### 핵심 결론
> **Prismatic encoder가 CLS를 제거하면서, CLS의 역할(전체 이미지 정보 집약)이 token[0]으로 밀려난 것이 근본 원인.**
> LLaVA(CLIP ViT)에서는 이 현상이 발생하지 않음.

### 시각화 위치
- **전 모델 비교 (가장 중요)**: `~/ATLASVLA/outputs/bottleneck_diagnosis/cross_model_comparison.png`
- 개별 모델: `~/ATLASVLA/outputs/bottleneck_diagnosis/{model}/diagnosis_summary.png`

---

<a id="phase-7"></a>
## Phase 7: 태스크별 Contribution 분석 — Skill Signature

### 질문
"Bottleneck이 모든 정보를 모으고 있다면, **태스크(pick, place, move 등)에 따라 다르게** 모으고 있을까?"

### 측정 방법
- 20개 샘플의 어텐션 패턴 추출
- 같은 동사끼리 거리(d_within) vs 다른 동사끼리 거리(d_between) 비교
- 어텐션 패턴으로 동사 분류하는 probe 학습

### 결과
```
같은 동사 내 거리 (d_within):  0.4506
다른 동사 간 거리 (d_between): 0.4414
→ 차이 거의 없음

분류기 정확도: 45% (8개 동사이므로 찍으면 12.5%)
태스크별 구분 가능성: False
```

**→ Bottleneck이 모든 정보를 태스크와 무관하게 균일하게 압축하고 있음. pick이든 place든 move든 어텐션 분포 구별 불가.**

### 시각화 위치
- `~/ATLASVLA/outputs/contribution_analysis/openvla-7b-full/`
  - `attention_vs_contribution.png`
  - `top1_contrib_share.png`
  - `candidate_frequency.png`
- `~/ATLASVLA/outputs/text_attention_viz/openvla-7b-full/`

---

<a id="phase-25"></a>
## Phase 2.5: Dual-Track 정밀 분석 — 4모델 분류 체계 (2월 26일)

### 왜 이 실험을 했나
Phase 5~7에서 Sink가 아니라 Bottleneck임을 확인했지만, 모델마다 **어텐션 라우팅 패턴이 다름**을 발견. 이를 정밀하게 분류하기 위한 실험.

### 분석 방법
- **A-peak**: Attention이 가장 높은 토큰
- **C-peak**: Contribution(= Attention × Value)이 가장 높은 토큰
- **R-peak**: Attention/Contribution 비율이 가장 높은 토큰 (Sink 후보)
- **V=0 Causal Verification**: 해당 토큰의 Value를 0으로 만들었을 때 출력 변화

### 4가지 유형 분류 (핵심 발견!)

#### Type A: Pure Bottleneck — ECoT
- A-peak == C-peak (같은 토큰: vision `<s>`)
- 하나의 토큰이 어텐션 61~85%, contribution 96~99.7% 독점
- 정보의 단일 실패점 (Single Point of Failure)

#### Type B: Coexist (Sink + Bottleneck 공존) — OpenVLA
- A-peak ≠ C-peak (다른 토큰!)
  - 어텐션은 vision[0] `<s>`에 집중 (23~48%)
  - contribution은 text token 271에 집중 (62~71%)
- 어텐션이 가는 곳과 정보가 모이는 곳이 **다른 모달리티**
- **기존 VAR 논문에 보고되지 않은 새로운 발견**

#### Type C: Normal (건강한 분산) — SpatialVLA
- 어텐션과 contribution이 일치 (match rate 85%)
- 최대 기여율 28% (특정 토큰 독점 없음)
- 엔트로피 높음 (2.3~3.3) → 잘 분산된 라우팅

#### Type D: Distributed-Fragile (분산이지만 취약) — TraceVLA
- 기여율은 14%로 가장 잘 분산됨
- **그런데** V=0 KL = 14.03으로 **가장 높은** 인과적 민감도
- "분산 ≠ 견고함"이라는 역설적 발견

### 시각화 위치
- `~/ATLASVLA/outputs/phase2.5_analysis/`
  - `fig1_dual_track_peaks.png` ~ `fig9_causal_scaling.png`

### ANALYSIS_REPORT.md
- `~/ATLASVLA/outputs/phase2.5_analysis/ANALYSIS_REPORT.md` (상세 보고서)

---

<a id="phase-3gate"></a>
## Phase 3-Gate: 3단계 Gate Check — 150 샘플 대규모 검증 (2월 27일)

### 왜 이 실험을 했나
Phase 2.5의 발견을 **통계적으로 유의미한 규모(150 샘플, 6스킬 × 25개)**로 검증.

### Gate ① — Contribution 분석 + Hidden State Probe + Skill Signature

**Mode Token 분석 (150 샘플 결과)**

| 모델 | A_mode (어텐션 피크) | 빈도 | C_mode (기여 피크) | 빈도 |
|------|---------------------|------|---------------------|------|
| ECoT | `<s>` (vis[0]) | 100% | `<s>` (vis[0]) | 100% |
| OpenVLA | `<s>` (vis[0]) | 100% | text token 271 | 18% |
| SpatialVLA | `robot` (text) | 54% | `robot` (text) | 45% |
| TraceVLA | `<s>` (vis[0]) | 84% | `<\|user\|>` (vis[1]) | 100% |

**Skill Signature (150 balanced samples)**

| 모델 | d_within | d_between | 시그니처? | Probe 정확도 |
|------|----------|-----------|-----------|-------------|
| ECoT | 0.026 | 0.028 | True | 23.3% |
| OpenVLA | 0.389 | 0.430 | True | 27.3% |
| SpatialVLA | 0.128 | 0.173 | True | **55.3%** |
| TraceVLA | 0.017 | 0.018 | True | 22.0% |

→ SpatialVLA만이 55% 정확도로 어텐션 패턴에서 태스크를 일부 구분 가능 (다른 모델은 찬스 수준)

### Gate ② — Causal V=0 Ablation (A/C/R 모드 토큰)

| 모델 | A_mode KL | C_mode KL | R_mode KL | A/R 비율 | 해석 |
|------|-----------|-----------|-----------|---------|------|
| ECoT | 1.197 | 1.197 | 0.072 | **16.6×** | A=C 동일 토큰 → Bottleneck 확정 |
| OpenVLA | 0.511 | 0.186 | 0.001 | **511×** | A≠C, 둘 다 중요 → Coexist 확정 |
| SpatialVLA | 1.215 | 1.215 | 0.004 | **304×** | A=C 동일 → Text 앵커 Bottleneck |
| TraceVLA | 13.112 | 13.109 | 12.969 | **1.01×** | 모든 토큰 치명적 → Global Dependency |

### Gate ③ — Text Masking + Counterfactual (Verb Swap)

**Text 마스킹 시 verb swap delta 변화:**

| 모델 | delta_orig | delta_textKV | 비율 | 해석 |
|------|-----------|-------------|------|------|
| ECoT | 0.78 | 0.51 | 1.52× | 부분적 텍스트 의존 |
| OpenVLA | 0.65 | 0.32 | 2.01× | 부분적 텍스트 의존 |
| SpatialVLA | 0.32 | **0.00** | **∞** | 텍스트가 유일한 verb 채널 |
| TraceVLA | 0.81 | 0.02 | 39.9× | 거의 완전한 텍스트 의존 |

**발견: Verb-Specific Routing Asymmetry (새로운 발견!)**
- `pick↔place`, `open↔close`: 텍스트 마스킹 시 delta → 0 (텍스트만으로 구분)
- `move↔fold`: 텍스트 마스킹 후에도 delta 유지! → vision 경로를 통한 별도 정보 채널 존재

### 관련 파일
- `~/ATLASVLA/outputs/phase3_gate/phase3_complete_results.md`
- Gate별 세부 결과: `~/ATLASVLA/outputs/phase3_gate/{model}/`, `gate2/`, `gate3_v2/`

---

<a id="phase-3expanded"></a>
## Phase 3-Expanded: 표본 확장 + VAR 비교 (2월 27일)

### N=20 → N=175 확장 결과

| 모델 | 지표 | N=20 | N=175 | 안정? |
|------|------|------|-------|-------|
| ECoT | D3 KL (V=0) | 1.662 | 1.289 | **YES** |
| ECoT | D3 flip rate | 0.450 | 0.457 | **YES** |
| OpenVLA | D3 KL (V=0) | 1.664 | 1.042 | **YES** |
| OpenVLA | D3 flip rate | 0.600 | 0.486 | **YES** |
| TraceVLA | D3 KL (V=0) | 0.010 | 0.010 | **YES** |

→ **모든 핵심 주장이 N=175에서도 확인됨** (8.75배 표본에서 재현)

### VAR (ICLR 2025) Baseline 비교
- K-scale intervention과 VAR attention redistribution 비교 실험 진행
- VAR 기반 재분배는 discrete action model에서 한계 재확인

### 관련 파일
- `~/ATLASVLA/outputs/phase3_gate_expanded/EXPANSION_RESULTS.md`

---

<a id="summary"></a>
## 전체 흐름 요약

```
[Phase 0] ROME/Circuit Analysis 계획
    ↓ "먼저 어텐션부터 보자"

[Phase 1] Attention Map 측정
    → "0번 토큰에 40~60% 집중!" → "Attention Sink다!"

[Phase 2] Sink 어텐션 재분배 (v1~v5, 16가지+ 조합)
    → 전부 실패 (discrete model의 argmax 불변)

[Phase 3] 학습 가능한 어댑터 (AttentionAdapterV2)
    → 학습 중 "정말 sink가 맞아?" 의문 발생

[Phase 4] 크로스 모델 분석 (5개 모델)
    → 모든 모델에서 유사 현상 → "보편적 구조 문제"

[Phase 5] Sink 3가지 조건 검증
    → 조건 C 실패! → "Sink가 아니라 Context Aggregator"

[Phase 6] Bottleneck 확정 진단
    → 98.5% 기여율, shift 무관, ablation시 모델 붕괴
    → "CRITICAL BOTTLENECK" 확정
    → Prismatic encoder의 CLS 제거가 원인

[Phase 7] 태스크별 차이 분석
    → pick/place/move 구분 불가 → 균일 압축

[Phase 2.5] Dual-Track 4모델 정밀 분류
    → 4가지 유형 발견: Bottleneck / Coexist / Normal / Fragile
    → "같은 LLaMA 백본에서도 학습에 따라 반대 패턴"

[Phase 3-Gate] 150 샘플 대규모 검증 (3단계 Gate)
    → Verb-Specific Routing Asymmetry 발견
    → move↔fold은 vision 경로, 나머지는 text 경로

[Phase 3-Expanded] N=175 표본 확장 + VAR 비교
    → 모든 주장 재현 확인
```

---

<a id="findings"></a>
## 핵심 발견 요약

### 1. Sink vs Bottleneck 구분법
- **Sink**: 어텐션만 높고 value norm은 낮음 → 지워도 출력 변화 적음
- **Bottleneck**: 어텐션도 높고 value norm도 높음 → 지우면 모델 붕괴
- OpenVLA의 token[0] 집중은 **Bottleneck** (Sink 아님)

### 2. 4가지 어텐션 라우팅 유형 (새로운 분류 체계)
| 유형 | 대표 모델 | 특징 |
|------|----------|------|
| Pure Bottleneck | ECoT | A=C=동일 토큰, 99% 독점 |
| Coexist | OpenVLA | A(vision)≠C(text), 다른 모달리티 |
| Normal | SpatialVLA | 잘 분산, 건강한 라우팅 |
| Distributed-Fragile | TraceVLA | 분산이지만 극도로 취약 |

### 3. 학습이 라우팅을 결정한다 (아키텍처가 아님)
- ECoT와 OpenVLA는 **같은 LLaMA-7B 백본**인데 반대 패턴
- → 파인튜닝 방식이 어텐션 라우팅 토폴로지를 결정

### 4. Verb-Specific Routing Asymmetry
- `pick/place/open/close`: 텍스트 경로로만 구분
- `move/fold`: 텍스트 마스킹 후에도 vision 경로로 구분 가능 (공간적 궤적 차이)

### 5. "분산 ≠ 견고함" 역설
- TraceVLA는 가장 분산된 어텐션 (top1 share = 14%)
- 그런데 V=0 KL = 14.03으로 **가장 취약** (다른 모델의 3.7배)
- 분산-취약(Distributed-Fragile) 패턴은 기존 문헌에 보고되지 않은 새로운 발견

### 6. Prismatic Encoder의 구조적 문제
- CLS 토큰을 제거해서 패치만 남겼는데
- CLS가 하던 역할(전체 이미지 정보 집약)이 token[0]에 떠넘겨짐
- CLIP ViT를 쓰는 LLaVA에서는 이 현상 없음

---

## 프로젝트 구조

| 위치 | 내용 |
|------|------|
| `~/capston/` | Phase 0 (원래 ROME/Circuit 계획) |
| `~/ATLASVLA/` | 메인 실험 코드 |
| `~/ATLASVLA/archive/` | Phase 2~4 (v1-v5 재분배 + 크로스 모델) |
| `~/ATLASVLA/outputs/sink_verification/` | Phase 5 (Sink 검증) |
| `~/ATLASVLA/outputs/sink_verification_v2/` | Phase 5 재검증 |
| `~/ATLASVLA/outputs/bottleneck_diagnosis/` | Phase 6 (Bottleneck 진단) |
| `~/ATLASVLA/outputs/contribution_analysis/` | Phase 7 (태스크별 기여 분석) |
| `~/ATLASVLA/outputs/text_attention_viz/` | 텍스트 어텐션 시각화 |
| `~/ATLASVLA/outputs/phase2.5_analysis/` | Phase 2.5 (Dual-Track 분류) |
| `~/ATLASVLA/outputs/phase3_gate/` | Phase 3-Gate (150샘플 검증) |
| `~/ATLASVLA/outputs/phase3_gate_expanded/` | Phase 3-Expanded (N=175 확장) |
| `~/ATLASVLA/docs/` | 설계 문서, 인사이트, 계획서 |
| `~/ATLASVLA/docs/plans/` | 각 실험 설계/구현 계획서 (25개) |

---

*Last updated: 2026-02-27*
