# 2주간 실험 여정 정리: "왜 vision token 0번에 어텐션이 쏠리는가?"

---

## 1단계: 처음 계획 — "모델 내부를 뜯어보자" (capston/)

### 원래 하려고 했던 것
OpenVLA라는 로봇 조작 모델(7B 파라미터)이 있는데, 이 모델이 왜 실패하는지 알고 싶었음.
계획은 이랬음:
- 시뮬레이션에서 로봇을 돌려서 실패 에피소드를 모으고
- ROME(Causal Tracing)이라는 방법으로 "어떤 레이어가 범인인지" 찾고
- Circuit Analysis(Path Patching)로 "실패를 만드는 경로"를 그리자

### 그런데 방향이 바뀜
본격적으로 시작하기 전에, **"먼저 모델이 이미지를 어떻게 보고 있는지부터 확인해보자"** 라는 생각이 들었음.
그래서 attention map(모델이 어디를 주목하는지 보여주는 지도)을 먼저 측정해보기로 함.

### 관련 파일
- 원래 계획서: `~/capston/requirements/openvla-interpretability-experiment.md`
- 모델 구조 정리: `~/capston/outputs/openvla_module_tree.txt`

---

## 2단계: 첫 번째 발견 — "0번 토큰에 어텐션이 몰린다"

### 무엇을 했나
OpenVLA-7B 모델에 Bridge V2 데이터셋의 이미지 10개 에피소드(총 385 타임스텝)를 넣어서, 모델 내부의 attention map을 전부 뽑아봤음.

### 뭘 발견했나
OpenVLA는 이미지를 16x16 = 256개 조각(패치)으로 나눠서 처리함. 그 256개 패치 중에서:

```
패치 0  패치 1  패치 2  ... 패치 15    ← 이미지 첫 줄
패치 16 패치 17 ...                     ← 이미지 둘째 줄
...
패치 240 ...              패치 255      ← 이미지 마지막 줄
```

모델이 "다음 행동(action)을 예측"할 때, 256개 패치를 골고루 보는 게 아니라 **0번 패치(왼쪽 맨 위 구석)에 전체 어텐션의 40~60%를 쏟아붓고** 있었음. 나머지 255개 패치는 합쳐봐야 40~60%밖에 안 됨.

### 이때 내린 판단
> "이거 NLP에서 말하는 **Attention Sink** 현상이다!"

Attention Sink란: LLM에서 첫 번째 토큰에 어텐션이 비정상적으로 몰리는 현상. StreamingLLM 등의 논문에서 보고된 바 있음. 마치 어텐션의 "쓰레기통"처럼, 남는 어텐션을 다 첫 토큰에 버리는 것.

### 시각화 위치
- 원본 어텐션 데이터: `~/ATLASVLA/archive/outputs/attention_results/ep000_step*.json` (385개)
- 초기 시각화: `~/ATLASVLA/archive/outputs/visualizations/`
- 헤드별 heatmap: `~/ATLASVLA/archive/outputs/cross_model_analysis/openvla-7b/bridge_v2/action_*_perhead_v0_heatmap.png`

---

## 3단계: 첫 번째 시도 — "sink에 묶인 어텐션을 물체로 돌려주자" (v1~v5)

### 아이디어
"0번 토큰이 어텐션을 쓸데없이 빨아들이고 있으니까, 그걸 빼서 진짜 중요한 곳(물체가 있는 패치)에 다시 나눠주면 행동 예측이 좋아질 거야."

### 세 가지 재분배 방법을 설계함
1. **Logit Bias**: 어텐션 계산 과정에서 물체 패치 위치에 점수를 더해줌 (softmax 전)
2. **Weight Rescale (VAR)**: 어텐션 가중치를 계산한 뒤, 물체 패치 비중을 올리고 나머지를 줄임 (softmax 후)
3. **Head Steering**: 이미지를 많이 보는 어텐션 헤드만 골라서 거기서만 재분배

### 물체 위치는 어떻게 찾았나
- spacy로 지시문에서 명사 추출 ("put small spoon from basket to tray" → spoon, basket, tray)
- GroundingDINO로 이미지에서 해당 물체 검출
- 물체 위치를 16x16 패치 좌표로 변환

### 엄청나게 많은 조합을 시도함

| 버전 | 뭘 바꿨나 | 결과 파일 | 결과 |
|------|----------|----------|------|
| **v1** | 기본 3가지 방법 | `~/ATLASVLA/archive/outputs/enhancement_results/` | baseline과 차이 없음 |
| **v2** | SAM 마스크로 물체 영역을 더 정밀하게 | `~/ATLASVLA/archive/outputs/v2_enhancement_results/` | 여전히 차이 없음 |
| **v3** | 16가지 이상 조합 (decay, temporal, layer selection, 배경 억제, gripper 강조 등) | `~/ATLASVLA/archive/outputs/v3_enhancement_results/` | 전부 baseline과 거의 동일 |
| **v4** | 코드 버그 발견해서 고친 뒤 재실험 | `~/ATLASVLA/archive/outputs/v4_corrected_results/` | 여전히 동일 |
| **v5** | 타겟팅 전략 자체를 바꿈 | `~/ATLASVLA/archive/outputs/v5_targeted_results/` | 역시 동일 |

v3에서 시도한 조합들의 이름만 봐도 얼마나 많이 시도했는지 알 수 있음:
- `sam_var_object`, `sam_var_obj_decay`, `sam_var_obj_temporal`, `sam_var_obj_decay_laysel`, `var_vt_rebal`, `var_decay`, `var_laysel`, `sam_var_bgsup_gripex`, `spin_k16`, `act_gripex` 등등...

### 왜 전부 실패했나
원인을 분석해보니 이런 구조적 한계가 있었음:

OpenVLA는 행동을 예측할 때 **이산적인(discrete) 토큰**을 출력함. 즉:
1. 모델이 vocabulary에서 확률이 가장 높은 토큰 하나를 고름 (argmax)
2. 그 토큰 ID를 연속적인 action 값으로 변환(detokenize)

문제는: 어텐션을 아무리 재분배해도, 최종 확률 분포에서 **1등 토큰이 바뀌지 않으면** 출력은 완전히 똑같다는 것. 모델의 확신(confidence)이 너무 높아서, 고정된 비율(p=0.6)로 어텐션을 옮겨봤자 1등 토큰의 순위를 뒤집을 수 없었음.

### 시각화 위치
- 전체 방법 비교 순위: `~/ATLASVLA/archive/outputs/v3_enhancement_results/ranking_overall.png`
- MSE 히트맵: `~/ATLASVLA/archive/outputs/v3_enhancement_results/mse_heatmap.png`
- 상위 5개 방법 시계열: `~/ATLASVLA/archive/outputs/v3_enhancement_results/mse_timestep_top5.png`
- 차원별 MSE: `~/ATLASVLA/archive/outputs/v3_enhancement_results/mse_per_dim.png`

### 관련 문서
- 요구사항: `~/ATLASVLA/requirements/attention-enhancement-experiment.md`

---

## 4단계: 두 번째 시도 — "고정 재분배 대신 학습 가능한 어댑터를 만들자"

### 새로운 아이디어
"고정된 비율로 재분배하면 argmax가 안 바뀌니까, **모델이 스스로 배워서 입력에 따라 다르게 재분배하는 어댑터**를 만들자."

### 어댑터(AttentionAdapterV2) 구조

두 개의 가지(branch)로 나뉨:

**Branch 1 — "얼마나 빼낼까?" (p_matrix)**
- 모델의 마지막 hidden state + 물체 마스크를 입력으로 받아서
- 각 레이어/헤드마다 "sink에서 어텐션을 몇 % 빼낼지" 결정
- 예: p=0.7이면 → sink 어텐션의 70%를 빼냄

**Branch 2 — "어디로 보낼까?" (redistribution_weights)**
- 빼낸 어텐션을 어떤 패치에 나눠줄지 결정
- SAM 마스크로 배경은 차단하고, 물체 패치에만 집중적으로 분배

**작동 예시** (레이어 29, 헤드 5):
```
원래: 패치0에 0.45, 물체패치(3,4,5)에 각각 0.008, 0.012, 0.006
   ↓ 어댑터가 p=0.7 결정 → 패치0에서 0.315를 빼냄
결과: 패치0은 0.45→0.135, 물체패치3은 0.008→0.087 (10배↑)
```

### 파라미터 규모
- 어댑터: ~2.17M 파라미터 (전체 7B 모델의 0.031%)
- 비교용 LoRA: rank 16 기준 ~1.05M 파라미터

### 학습(Smoke Test) 결과

| 모델 | 학습 Loss | 검증 Loss | 비고 |
|------|----------|----------|------|
| OpenVLA-7B | 12.15 | 14.97 | 안정적 |
| ECoT-7B | 12.92 | 22.25 | 과적합 위험 |
| SpatialVLA-4B | 18.85 | 24.99 | gradient가 0이라 학습 안 됨 |
| TraceVLA-Phi3V | 2.61 | 2.73 | 안정적 (continuous 모델이라 loss 단위 다름) |
| OpenVLA LoRA+Adapter | 12.05 | 14.70 | LoRA 추가 시 검증 loss 소폭 개선 |

### 이 과정에서 생긴 근본적 의문
어댑터를 학습시키면서 이런 생각이 들었음:
> "어댑터가 열심히 재분배해도 결과가 크게 안 바뀐다면... 애초에 **이게 정말 Attention Sink가 맞는 건가?** Sink라면 어텐션을 빼서 다른 데 주면 좋아져야 하는 거 아닌가?"

### 시각화 위치
- 어댑터 결과: `~/ATLASVLA/archive/outputs/adapter_results/`
- Smoke test: `~/ATLASVLA/archive/outputs/smoke_test/`
- Full 학습: `~/ATLASVLA/archive/outputs/full/`, `~/ATLASVLA/archive/outputs/full_v2/`

### 관련 문서
- 어댑터 동작 원리: `~/ATLASVLA/docs/adapter-mechanism-explained.md`
- LoRA 비교 계획: `~/ATLASVLA/docs/lora_baseline_comparison_plan.md`

---

## 5단계: 다른 모델도 그런가? — 크로스 모델 분석

### 무엇을 했나
"OpenVLA만의 문제인가, 아니면 비슷한 모델에서 다 이런가?" 확인하기 위해 총 5개 모델의 어텐션을 비교함:
- OpenVLA-7B (LLaMA 기반)
- ECoT-7B (OpenVLA 파인튜닝 버전)
- SpatialVLA-4B (Gemma2 기반)
- TraceVLA-Phi3V (Phi3V 기반)
- LLaVA-1.5-7B (비교 대조군, CLIP ViT 사용)

### 발견한 것
- **4개 모델 모두에서 비슷한 현상 발생**, 위치만 다름
  - OpenVLA, ECoT → vision token 0번에 몰림
  - SpatialVLA, TraceVLA → text 영역(BOS 토큰 등)에 몰림
- 사용하는 LLM 백본이 다른데도(LLaMA, Gemma2, Phi3V) 모두 발생 → 아키텍처와 무관한 보편적 현상

### 시각화 위치
- 전 모델 비교: `~/ATLASVLA/archive/outputs/cross_model_analysis/comparison/cross_model_sink_comparison.png`
- 히트맵 비교: `~/ATLASVLA/archive/outputs/cross_model_analysis/comparison/cross_model_heatmap.png`
- 듀얼 싱크: `~/ATLASVLA/archive/outputs/cross_model_analysis/comparison/cross_model_dual_sink.png`

---

## 6단계: "정말 Sink가 맞아?" — 3가지 조건으로 검증 (Sink Verification)

### 왜 이 실험을 했나
"모든 모델에서 일어나는 보편적인 현상이라면, 단순한 잡음이 아니라 구조적인 문제다. 그런데 정말 이게 attention sink가 맞는지, 다른 현상인지 정확히 구분해야 한다."

### 진짜 Sink인지 판별하는 3가지 조건

**조건 A — "어텐션이 정말 일관되게 몰리는가?"**
- 측정: 0번 토큰이 전체 레이어의 80% 이상에서 어텐션 top-5 안에 드는가
- OpenVLA 결과: **통과** (32개 레이어 중 32개 모두 통과)
- → 일관되게 어텐션이 몰리는 건 맞음

**조건 B — "hidden state에 비정상적 스파이크가 있는가?"**
- 측정: 0번 토큰의 activation 크기(norm)가 비정상적으로 큰가 (기준: φ ≥ 20.0)
- OpenVLA 결과: **통과** (32개 레이어 전부 통과)
- → hidden state가 비정상적으로 커져 있음

**조건 C — "정보 없이 어텐션만 흡수하는가?" (핵심!)**
- Sink라면: 0번 토큰은 "쓰레기통"이므로, 실제 담고 있는 정보(value norm)가 다른 토큰보다 **낮아야** 함
- 측정: 0번 토큰의 value vector 크기와 다른 토큰의 value vector 크기를 비교
- OpenVLA 결과: **실패!** — 0번 토큰의 value norm이 다른 토큰보다 **훨씬 높았음** (8개 분석 레이어 전부에서)

### 이게 무슨 뜻이냐면
- **진짜 Sink**: 어텐션은 많이 받지만, 정보는 별로 안 담고 있음. 그냥 남는 어텐션을 버리는 곳.
- **이 경우**: 어텐션도 많이 받고, **정보도 진짜로 담고 있음**. 쓰레기통이 아님.

### 최종 판정
```
is_true_sink: False (진짜 sink 아님)
is_context_aggregator: True (정보를 실제로 모아서 저장하는 토큰)
```

→ Token[0]은 쓰레기통이 아니라, **정보가 실제로 집중되어 있는 핵심 토큰**이었음.

### 시각화 위치
- OpenVLA 전체 요약: `~/ATLASVLA/outputs/sink_verification/openvla-7b/summary_all_conditions.png`
- 조건A 히트맵: `~/ATLASVLA/outputs/sink_verification/openvla-7b/condition_A_heatmap.png`
- 조건B φ 값: `~/ATLASVLA/outputs/sink_verification/openvla-7b/condition_B_phi.png`
- **조건C (결정적)**: `~/ATLASVLA/outputs/sink_verification/openvla-7b/condition_C_contribution.png`
- 다른 모델도 동일: `~/ATLASVLA/outputs/sink_verification/ecot-7b/`, `tracevla-phi3v/` 등

### V2 검증 (더 엄밀하게)
같은 검증을 더 정밀하게 다시 수행한 것이 sink_verification_v2:
- 위치: `~/ATLASVLA/outputs/sink_verification_v2/openvla-7b/`
- 결론 동일: `is_true_sink: False, is_context_aggregator: True`

---

## 7단계: "그러면 이건 뭔가?" — Bottleneck 진단

### 왜 이 실험을 했나
Sink가 아니라면 뭔가? 세 가지 추가 테스트로 정체를 밝힘.

### 테스트 1: "0번 토큰이 특별한 토큰(CLS)인가, 그냥 일반 패치인가?"
- 일반 Vision Transformer에는 [CLS]라는 특별 토큰이 있는데, 전체 이미지를 대표하는 역할을 함
- 만약 0번 토큰이 [CLS]라면 어텐션이 몰리는 게 당연함
- **결과: [CLS]가 아님**. OpenVLA가 쓰는 Prismatic vision encoder는 `get_intermediate_layers()`로 CLS를 제거함. 0번 토큰은 그냥 이미지 왼쪽 맨 위 구석의 일반 패치.
- → CLS 같은 특별한 역할이 없는 그냥 패치인데 어텐션이 몰리는 것 = 비정상

### 테스트 2: "이미지를 밀어보면 어떻게 되나?"
- 이미지를 90도 돌리거나 상하좌우로 밀면, 왼쪽 맨 위에 있던 내용물이 바뀜
- 만약 0번 토큰이 "왼쪽 위에 있는 물체 때문에" 어텐션을 받는 거라면, 이미지를 밀면 어텐션도 바뀌어야 함
- **결과: 이미지를 밀어도 0번 토큰의 기여율이 98.49% → 98.28%로 거의 변화 없음**
- → 이미지 내용물과 상관없이, **"0번 위치"라서** 어텐션이 몰리는 것. 공간적 의미 없는 아키텍처 편향(architectural artifact).

### 테스트 3: "0번 토큰을 지우면 어떻게 되나?"
- 0번 토큰의 값을 0으로 만들어버리고(ablation) 출력이 얼마나 바뀌는지 측정
- 비교 대상: 랜덤하게 고른 다른 토큰을 지웠을 때와 비교
- **결과**:
  - 0번 토큰 제거 시 KL divergence: **7.34**
  - 랜덤 토큰 제거 시 KL divergence: **0.000086**
  - 비율: **85,139배** 차이
  - 7개 행동 차원 중 3개가 변경됨 (랜덤 토큰 제거 시 0개 변경)
- → **0번 토큰 하나를 지우면 모델이 완전히 다른 행동을 예측함**. 모든 정보가 여기에 몰려 있다는 뜻.

### 모든 모델 비교

| 모델 | 0번 토큰 기여율 | 이미지 shift 후 | KL 비율 | 진단 |
|------|---------------|---------------|---------|------|
| **OpenVLA-7B** | 98.49% | 98.28% | 85,139배 | **BOTTLENECK** |
| **ECoT-7B** | 98.44% | 98.60% | 5,320배 | **BOTTLENECK** |
| **LLaVA-1.5-7B** | 0.22% | 0.16% | 1.05배 | 정상 (bottleneck 아님) |

**핵심**: LLaVA-1.5는 이 현상이 없음. LLaVA는 CLIP ViT를 쓰고, OpenVLA/ECoT는 Prismatic encoder를 씀.
→ **Prismatic encoder가 CLS를 제거하면서 그 역할(전체 이미지 정보 집약)이 token[0]으로 밀려난 것**이 원인.

### 최종 진단문
> "PROBLEMATIC BOTTLENECK: Token 0은 실제 공간 패치(왼쪽 위)인데, 이미지 내용과 무관하게 지배적이며, 모델이 이것에 아키텍처적으로 의존하고 있다. 이것은 의미 있는 정보 압축이 아니라 아키텍처적 편법(shortcut)이다."

### 시각화 위치
- OpenVLA 진단 요약: `~/ATLASVLA/outputs/bottleneck_diagnosis/openvla-7b/diagnosis_summary.png`
- **전 모델 비교 (가장 중요)**: `~/ATLASVLA/outputs/bottleneck_diagnosis/cross_model_comparison.png`
- ECoT 진단: `~/ATLASVLA/outputs/bottleneck_diagnosis/ecot-7b/diagnosis_summary.png`
- LLaVA 대조군: `~/ATLASVLA/outputs/bottleneck_diagnosis/llava-1.5-7b/diagnosis_summary.png`
- PaliGemma 진단: `~/ATLASVLA/outputs/bottleneck_diagnosis/paligemma-3b/diagnosis_summary.png`
- TracVLA 진단: `~/ATLASVLA/outputs/bottleneck_diagnosis/tracevla-phi3v/diagnosis_summary.png`

---

## 8단계: "태스크(동사)별로 어텐션 분포가 다른가?" — Contribution Analysis

### 왜 이 실험을 했나
Bottleneck이 확인됐으니 한 가지 더 궁금한 점이 있었음:
"0번 토큰이 모든 정보를 모으고 있다면, 적어도 **다른 태스크(pick, place, move 등)에 따라 다르게 모으고 있을까?**"

만약 다르게 모으고 있다면 → 나름 의미 있는 정보 압축일 수 있음
만약 똑같이 모으고 있다면 → 태스크와 무관하게 그냥 찍어누르는 거라 더 문제임

### 측정 방법: Skill Signature 분석
- 20개 샘플의 어텐션 패턴을 추출
- 같은 동사(예: pick과 pick)끼리의 어텐션 거리(d_within)와
- 다른 동사(예: pick과 place)끼리의 어텐션 거리(d_between)를 비교
- 추가로, 어텐션 패턴만 보고 어떤 동사인지 맞추는 분류기(probe) 학습

### 결과

```
같은 동사 내 거리 (d_within):  0.4506
다른 동사 간 거리 (d_between): 0.4414
차이: 거의 없음

분류기 정확도: 45% (8개 동사니까 찍으면 12.5%)
태스크별 구분 가능성: False
```

### 이게 무슨 뜻이냐면
- **같은 동사끼리의 어텐션 패턴 차이**와 **다른 동사끼리의 차이**가 거의 동일
- pick이든 place든 move든, 어텐션 분포가 거의 구별 불가능
- 분류기 정확도 45%는 찍는 것(12.5%)보다는 높지만, 신뢰성 있게 구분하기엔 턱없이 부족

→ **Bottleneck(token 0)이 모든 정보를 태스크와 무관하게 균일하게 압축하고 있다**. 동사가 뭐든 간에 같은 식으로 정보를 밀어넣고 있으니, 어텐션 패턴만으로는 무슨 태스크인지 알 수 없음.

### 샘플 분포 (참고)
```
move: 7개, pick: 6개, place: 2개, close: 1개,
open: 1개, turn: 1개, wipe: 1개, unknown: 1개
```

### 시각화 위치
- 어텐션 vs 기여도 비교: `~/ATLASVLA/outputs/contribution_analysis/openvla-7b-full/attention_vs_contribution.png`
- Top-1 기여 비율: `~/ATLASVLA/outputs/contribution_analysis/openvla-7b-full/top1_contrib_share.png`
- 후보 빈도 분석: `~/ATLASVLA/outputs/contribution_analysis/openvla-7b-full/candidate_frequency.png`
- 텍스트 어텐션 시각화: `~/ATLASVLA/outputs/text_attention_viz/openvla-7b-full/sample_*_attention.png`
- 다른 모델들: `~/ATLASVLA/outputs/contribution_analysis/ecot-7b-full/`, `spatialvla-4b-full/`, `tracevla-phi3v-full/`

---

## 전체 흐름을 한 줄씩 정리

```
[1단계] ROME/Circuit Analysis 하려다가 → "먼저 어텐션부터 보자"

[2단계] 어텐션 측정 → "0번 토큰에 40~60% 몰린다" 발견
         → "이건 Attention Sink다!" 라고 판단

[3단계] Sink 어텐션을 물체 패치로 재분배 (v1~v5, 16가지+ 조합)
         → 전부 실패. MSE가 baseline과 동일
         → 원인: discrete model의 argmax가 안 바뀜

[4단계] "고정 재분배가 안 되면, 학습 가능한 어댑터를 만들자"
         → MLP 기반 V2 어댑터 설계 및 학습
         → 이 과정에서 "정말 sink가 맞아?" 라는 의문 발생

[5단계] 크로스 모델 분석 → 5개 모델 모두에서 유사 현상 확인
         → "보편적이면 단순한 noise가 아니라 구조적 문제"

[6단계] Sink 여부 검증 → 조건 C 실패!
         → "Sink가 아니라 Context Aggregator다"
         → 쓰레기통이 아니라 정보가 실제로 담겨 있었음

[7단계] Bottleneck 진단 → 98.5% 기여율, 이미지 shift 무관, ablation 시 모델 붕괴
         → "CRITICAL BOTTLENECK" 확정
         → Prismatic encoder의 CLS 제거가 원인
         → LLaVA(CLIP ViT)에서는 발생 안 함

[8단계] 태스크별 차이 분석 → 동사별 어텐션 차이 거의 없음
         → Bottleneck이 모든 정보를 균일하게 압축
         → pick이든 place든 move든 구분 불가
```

---

## 핵심 교훈 정리

1. **"어텐션이 몰린다" ≠ "Attention Sink"**: 현상은 같아 보여도, value norm을 확인해야 진짜 sink인지 bottleneck인지 구분됨

2. **Sink vs Bottleneck 차이**:
   - Sink: 정보 없이 어텐션만 빨아들임 → 지워도 출력 변화 적음
   - Bottleneck: 정보가 실제로 몰려 있음 → 지우면 모델이 망가짐

3. **Prismatic encoder의 구조적 문제**: CLS 토큰을 제거해서 패치만 남겼는데, CLS가 하던 역할(전체 이미지 정보 집약)이 token[0]에 떠넘겨진 것

4. **Fixed attention 재분배의 근본적 한계**: Discrete token model에서는 argmax만 바뀌면 되는데, 어텐션을 살짝 바꾸는 것으로는 argmax 순위가 안 바뀜

---

## 현재 프로젝트 구조

| 위치 | 내용 |
|------|------|
| `~/capston/` | 원래 ROME/Circuit Analysis 계획 (Phase 0) |
| `~/ATLASVLA/` | 메인 실험 코드 |
| `~/ATLASVLA/archive/` | v1~v5 재분배 실험 + 크로스 모델 분석 (Phase 2~5) |
| `~/ATLASVLA/outputs/sink_verification/` | Sink 검증 V1 (Phase 6) |
| `~/ATLASVLA/outputs/sink_verification_v2/` | Sink 검증 V2 (Phase 6 재검증) |
| `~/ATLASVLA/outputs/bottleneck_diagnosis/` | Bottleneck 진단 (Phase 7) |
| `~/ATLASVLA/outputs/contribution_analysis/` | 태스크별 차이 분석 (Phase 8) |
| `~/ATLASVLA/outputs/text_attention_viz/` | 텍스트 어텐션 시각화 |
| `~/ATLASVLA/docs/` | 설계 문서, 인사이트 정리 |
