# Cross-Model Object-Centric Adapter Experiment: 설계 문서

## 1. 실험 목표

4개 오픈소스 VLA 모델에서 **base model(어댑터 없음) vs V2-Full(object-centric dynamic adapter)**의 성능 차이를 측정하여, attention sink 재분배 기반 어댑터가 아키텍처에 무관하게 일반적으로 효과가 있는지 검증한다.

---

## 2. 대상 모델

### 2.1 모델 사양 비교표

| | OpenVLA-7B | ECoT-7B | SpatialVLA-4B | TraceVLA-Phi3V |
|---|---|---|---|---|
| **HF ID** | openvla/openvla-7b | Embodied-CoT/ecot-openvla-7b-bridge | IPEC-COMMUNITY/spatialvla-4b-224-pt | furonghuang-lab/tracevla_phi3v |
| **LLM Backbone** | LLaMA-2-7B | Vicuna-7B-v1.5 | Gemma-2-2B | Phi-3-Vision-128K |
| **hidden_dim** | 4096 | 4096 | 2304 | 3072 |
| **num_heads** | 32 | 32 | 8 | 32 |
| **num_layers** | 32 | 32 | 26 | 32 |
| **vision_tokens** | 256 (16×16) | 256 (16×16) | 256 (16×16) | ~144 |
| **vision_encoder** | DINOv2+SigLIP (dual) | SigLIP | SigLIP+ZoeDepth | CLIP-ViT-L-336 |
| **input_resolution** | 224×224 | 224×224 | 224×224 | 336×336 |
| **action_output** | discrete 7 tokens | discrete 7 tokens | discrete 7 tokens | continuous 7-dim |
| **action_tokenization** | 256-bin discretization | 256-bin discretization | 256-bin discretization | continuous (no binning) |
| **num_action_tokens** | 7 | 7 | 7 | 7 |
| **VRAM (est.)** | ~15GB | ~15GB | ~10GB | ~12GB |
| **layers_path** | model.language_model.model.layers | model.language_model.model.layers | language_model.model.layers | model.model.layers |

### 2.2 어댑터 파라미터 (모델별)

| | OpenVLA-7B | ECoT-7B | SpatialVLA-4B | TraceVLA-Phi3V |
|---|---|---|---|---|
| **adapter hidden_dim** | 4096 | 4096 | 2304 | 3072 |
| **adapter num_heads** | 32 | 32 | 8 | 32 |
| **adapter num_target_layers** | 4 | 4 | 4 | 4 |
| **target_layers** | [28, 29, 30, 31] | [28, 29, 30, 31] | [22, 23, 24, 25] | [28, 29, 30, 31] |
| **source_layer** | 27 | 27 | 21 | 27 |
| **vision_tokens (mask)** | 256 | 256 | 256 | 144 |
| **adapter output_dim** | 4×32=128 | 4×32=128 | 4×8=32 | 4×32=128 |
| **loss function** | CrossEntropy | CrossEntropy | CrossEntropy | MSE |
| **adapter params (est.)** | ~2.17M | ~2.17M | ~0.73M | ~1.50M |
| **% of base model** | 0.031% | 0.031% | 0.018% | 0.039% |

### 2.3 Attention Sink 검증 결과 (이미 완료)

| | Sink Type | Sink Prevalence | Sink Location |
|---|---|---|---|
| **OpenVLA** | vision[0] + text | 32/32 layers (100%) | vision token 0 (v0) |
| **ECoT** | vision[0] + text | 32/32 layers (100%) | vision token 0 (v0) |
| **SpatialVLA** | text only | 26/26 layers (100%) | text tokens (225, 265 등) |
| **TraceVLA** | text only | 31/32 layers (96.9%) | text tokens (0, 1) |

모든 모델에서 sink가 존재하므로, 어댑터가 효과를 발휘할 수 있는 조건이 충족됨.

---

## 3. 데이터셋

### 3.1 데이터셋 선택: Bridge V2

**이유**: 4개 모델 모두 Bridge V2로 사전학습됨 → 공정한 비교 가능

| 속성 | 값 |
|------|-----|
| **이름** | Bridge V2 |
| **위치** | /ceph_data/kana5123/bridge_v2_data |
| **캐시** | /ceph_data/kana5123/bridge_data_cache |
| **총 에피소드** | 53,186 |
| **총 스텝** | 1,382,356 |
| **이미지 해상도** | 256×256 (원본) → 모델별 리사이즈 |
| **Action** | 7-DoF (x, y, z, roll, pitch, yaw, gripper) |
| **SAM Mask** | 완료 (object_masks.dat, 100%) |

### 3.2 데이터 분할

```
총 53,186 에피소드를 에피소드 단위로 분할 (seed=42):
  Train: 80% = 42,549 에피소드
  Val:   10% = 5,319 에피소드
  Test:  10% = 5,318 에피소드
```

**주의**: SAM 실패 에피소드는 `compute_valid_episodes()`로 필터링 후 분할.

### 3.3 데이터 전처리 (모델별)

| | OpenVLA / ECoT | SpatialVLA | TraceVLA |
|---|---|---|---|
| **리사이즈** | 256→224px | 256→224px | 256→336px |
| **SAM mask 해상도** | 16×16 (256 tokens) | 16×16 (256 tokens) | 별도 생성 필요 (~12×12 = 144 tokens) |
| **Action 토큰화** | 256-bin discretization | 256-bin discretization | 연속값 (그대로 사용) |
| **정규화 통계** | model.norm_stats["bridge_orig"] | model.norm_stats | model.action_norm_stats |

---

## 4. 어댑터 동작 메커니즘 (상세)

> 별도 문서: `docs/adapter-mechanism-explained.md` 참조

### 4.1 핵심 요약

어댑터는 **2가지를 학습**합니다:

**Branch 1 (p_matrix)**: 각 헤드에서 sink attention을 얼마나(p%) 빼낼 것인가
- 입력: language hidden state + SAM object mask
- 출력: p ∈ [0,1]^(L×H) — 타겟 레이어 × 헤드 수만큼
- p=0: 해당 헤드 건드리지 않음, p=0.7: sink의 70%를 재분배

**Branch 2 (redistribution_weights)**: 빼낸 attention을 어떤 vision 패치에 나눌 것인가
- 입력: h_last(query) × h_vision(keys), object_mask로 배경 차단
- 출력: softmax weights ∈ [0,1]^V — 물체 패치에 높은 가중치
- blend_alpha가 proportional → learned 전환 제어

### 4.2 어댑터가 결정하는 것들

| 결정 사항 | 방법 | 코드 위치 |
|-----------|------|-----------|
| **어떤 토큰이 sink인가** | α/N threshold (동적) | attention_v3.py:41-70 |
| **어떤 헤드에 적용할 것인가** | useful_vision >= ρ + p > 0 | attention_v3.py:273-302 |
| **얼마나 뺄 것인가** | 어댑터 Branch 1 (p_matrix) | adapter_model.py:305-323 |
| **어디로 보낼 것인가** | 어댑터 Branch 2 (redistribution_weights) | adapter_model.py:328-354 |
| **적용 레이어** | config 설정 (마지막 4레이어) | config.py:134 |

### 4.3 모든 것은 각 헤드 내부에서만 발생

- 헤드 간 교차 없음
- 각 헤드의 attention map 안에서 sink → object patch로 점수 이동
- 모델 weights는 frozen, 어댑터(~2M params)만 학습

---

## 5. 실험 파이프라인

### 5.1 각 모델별 실행 순서

```
For each model in [OpenVLA, ECoT, SpatialVLA, TraceVLA]:
  Step 1: Base model 평가 (어댑터 없음)
    → eval_results/base/{model}/eval_results.json

  Step 2: V2-Full 어댑터 학습
    → 80% train / 10% val 사용
    → early stopping (patience=15)
    → checkpoints/{model}/best.pt

  Step 3: V2-Full 어댑터 평가
    → 10% test 사용
    → eval_results/v2-full/{model}/eval_results.json

  Step 4: 비교
    → base vs v2-full per-dimension MSE
    → improvement % 계산
```

### 5.2 GPU 할당 계획

순차적 실행 (Approach A):

| 순서 | 모델 | GPUs | 예상 VRAM | 예상 소요 |
|------|------|------|----------|----------|
| 1 | OpenVLA-7B | 1,2,3,4 | ~15GB/GPU | 6-12시간 |
| 2 | ECoT-7B | 1,2,3,4 | ~15GB/GPU | 6-12시간 |
| 3 | SpatialVLA-4B | 5,6 | ~10GB/GPU | 4-8시간 |
| 4 | TraceVLA-Phi3V | 5,6,7 | ~12GB/GPU | 6-10시간 |

총 예상: 22-42시간 (순차)

### 5.3 학습 하이퍼파라미터 (동일)

| 파라미터 | 값 | 비고 |
|----------|-----|------|
| LR | 3e-4 → 3e-5 | Cosine annealing |
| Warmup steps | 500 | |
| Max steps | 50,000 | |
| Batch size | 16 | DDP across GPUs |
| Early stopping patience | 15 | eval intervals |
| Eval every | 500 steps | |
| Grad clip | 1.0 | |
| L1 lambda | 0.001 | p sparsity |
| α (sink detection) | 5.0 | Dynamic |
| ρ (head selection) | 0.5 | |

---

## 6. 평가 메트릭

### 6.1 Discrete 모델 (OpenVLA, ECoT, SpatialVLA)

| 메트릭 | 설명 |
|--------|------|
| **Overall MSE** | 7차원 평균 MSE |
| **Spatial MSE** | x, y, z (position) 3차원 MSE |
| **Rotational MSE** | roll, pitch, yaw 3차원 MSE |
| **Gripper MSE** | gripper 1차원 MSE |
| **Per-dim MSE** | 각 7차원 개별 MSE |
| **Improvement %** | (base - adapter) / base × 100 |

### 6.2 Continuous 모델 (TraceVLA)

| 메트릭 | 설명 |
|--------|------|
| **MSE Loss** | 연속값 직접 비교 (CE 대신) |
| **Per-dim MSE** | 동일 |
| **Improvement %** | 동일 |

---

## 7. 코드 수정 필요 사항

### 7.1 어댑터 일반화 (adapter_model.py)

```python
# AttentionAdapterV2의 vision_tokens 파라미터를 모델별로 다르게 설정
class AttentionAdapterV2:
    def __init__(self, hidden_dim, num_heads, num_target_layers, vision_tokens, ...):
        # hidden_dim: OpenVLA=4096, SpatialVLA=2304, TraceVLA=3072
        # num_heads: OpenVLA=32, SpatialVLA=8, TraceVLA=32
        # vision_tokens: OpenVLA=256, TraceVLA=144
```

### 7.2 학습 일반화 (adapter_train.py)

```python
# --model 인자 추가
parser.add_argument("--model", type=str, default="openvla-7b")

# 모델별 설정 로딩
from model_registry import get_model
model_cfg = get_model(args.model)

# 모델별 loss function 분기
if model_cfg.action_type == "discrete":
    loss = F.cross_entropy(logits, target_tokens)
elif model_cfg.action_type == "continuous":
    loss = F.mse_loss(predicted_actions, target_actions)
```

### 7.3 Attention 패칭 일반화 (attention_v3.py)

```python
# 현재: LlamaAttention만 패칭
# 필요: 모델별 attention 클래스 패칭
#   OpenVLA/ECoT: LlamaAttention
#   SpatialVLA: Gemma2Attention (trust_remote_code)
#   TraceVLA: Phi3Attention
```

### 7.4 평가 일반화 (adapter_eval.py)

```python
# 모델별 load_model, detect_token_boundaries 일반화
# TraceVLA: CE loss → MSE loss
```

---

## 8. 예상 결과 및 논문 기여

### 8.1 가설

1. **모든 모델에서 어댑터가 base 대비 MSE를 개선한다** (sink 재분배 → 정보 흐름 개선)
2. **sink가 강한 모델(OpenVLA 99.3%)에서 개선이 크다** vs sink가 약한 모델(SpatialVLA 86.1%)
3. **object-centric 재분배가 proportional 대비 우수하다** (SAM mask 활용)
4. **아키텍처에 무관하게 일반화된다** (LLaMA, Gemma, Phi 모두)

### 8.2 논문 기여점

1. **Cross-model attention sink 현상 검증**: 4개 아키텍처에서 보편적으로 발생
2. **Object-centric dynamic adapter**: 아키텍처 무관 경량 어댑터 (~0.03% params)
3. **Dynamic sink detection**: α/N threshold 기반, 하드코딩 불필요
4. **모델별 sink 위치 차이**: LLaMA=vision[0], Gemma/Phi=text token

---

## 9. 제외된 모델과 그 이유

| 모델 | 제외 이유 |
|------|----------|
| **CogACT-Base** | Diffusion action head (16-step trajectory) — CE/MSE loss 적용 불가, 완전히 다른 패러다임 |
| **SmolVLA** | transformers 4.57.6에서 flex_attention 필요 (PyTorch 2.5+), 현재 환경 호환 불가 |
| **RoboFlamingo** | config.json에 model_type 없음, cross-attention 기반 (self-attention이 아님) |

---

## 10. 데이터 분할 상세

```python
# adapter_data.py:split_episodes()
# 이미 80/10/10 구현되어 있음 (train_ratio=0.8, val_ratio=0.1)

총 에피소드: 53,186 (Bridge V2)
SAM 유효 에피소드: ~45,000 (추정, SAM failure threshold=0.5)
  Train: ~36,000 에피소드
  Val: ~4,500 에피소드
  Test: ~4,500 에피소드
```

---

## 부록: CogACT 제외에 대한 기술적 설명

CogACT는 action 생성에 **Diffusion Transformer (DiT-B)**를 사용합니다:
- 입력: vision+language representation
- 출력: 16-step action trajectory [16, 7]
- 생성 방식: 노이즈에서 시작하여 반복적으로 denoising (8 diffusion steps)
- CE loss 적용 불가: action이 토큰이 아닌 연속 궤적
- MSE loss 적용 어려움: denoising 과정에서의 gradient path가 일반 autoregressive와 다름
- 어댑터 적용: self-attention은 있으나, action head가 autoregressive가 아니므로 현재 파이프라인과 호환 불가
