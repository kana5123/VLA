# ATLASVLA Experiment Insights
> 논문 작성 시 참고용 인사이트 모음 (2026-02-23 실험 기준)

---

## 1. Model-Specific Architectural Quirks

### SpatialVLA (Gemma2)
- `backproject_patch()` 함수가 카메라 intrinsic matrix K(3x3)를 반드시 필요로 함
- Bridge V2 데이터셋에는 카메라 intrinsic이 없음 → 기본값 (224x224 pinhole model) 사용
- depth estimation 기반 3D feature를 생성하는 구조 → 다른 모델보다 공간 인식에 강할 수 있으나, 정확한 intrinsic 없이는 왜곡 가능

### TraceVLA (Phi3V)
- `input_ids`에 -1을 이미지 placeholder로 사용 (특이한 방식)
- forward 시 반드시 `pixel_values` + `image_sizes` 두 가지 모두 전달해야 함
- `image_sizes`가 없으면 `embed_tokens(-1)` 호출 → CUDA assertion 크래시
- **Continuous action model**: 다른 3개 모델(discrete)과 달리 MSE 기반 직접 action 예측
- `input_ids`를 in-place로 수정함 → 여러 번 forward 시 반드시 copy 필요

### OpenVLA (LLaMA)
- 가장 표준적인 아키텍처, 특이사항 없음
- Vision token [0]이 attention sink (전체 attention의 ~40-60% 흡수)
- transformers 버전 호환성 경고 있으나 실제 동작에 영향 없음

### ECoT (LLaMA, OpenVLA 파인튜닝)
- OpenVLA와 동일 아키텍처지만 chain-of-thought 토큰을 action 전에 생성
- Base eval에서 OpenVLA 대비 전반적으로 높은 MSE (특히 yaw, gripper)
- CoT 토큰이 action 정확도에 기여하는 방식 분석 필요

---

## 2. Attention Sink Phenomenon Across Models

### Dynamic Sink Detection (α/N threshold, α=5.0)
- **OpenVLA/ECoT**: Vision token [0]이 일관된 sink → attention score > 5/N 기준 충족
- **SpatialVLA/TraceVLA**: Text token 영역에서 sink 발생 (BOS 또는 특정 text token)
- 모든 4개 모델에서 sink 현상 확인됨, 위치만 다름
- Sink의 존재는 아키텍처-독립적 현상 (LLaMA, Gemma2, Phi3V 모두)

### 논문 시사점
> "Attention sinks in VLA models manifest universally across architectures but localize differently:
> vision-first models (OpenVLA, ECoT) develop sinks at the initial vision token,
> while vision-encoder models (SpatialVLA, TraceVLA) develop sinks in the text prefix region."

---

## 3. Fixed-VAR vs Learned Adapter (핵심 발견)

### Discrete 모델에서 Fixed-VAR의 한계
- **OpenVLA fixed-var = baseline과 동일 MSE (0% 변화)**
- **ECoT fixed-var = baseline과 동일 MSE (0% 변화)**
- 원인: discrete token model에서 action = argmax(logits) → detokenize
- attention redistribution이 softmax를 변경하더라도, **logits의 argmax(top-1 token)가 바뀌지 않으면** MSE는 동일
- 모델 confidence가 높은 경우, 고정 p=0.6 redistribution으로는 argmax 순위 불변

### 논문 시사점
> "Static attention redistribution (fixed p=0.6) fails to produce any measurable
> change in discrete-action models, as the argmax of the prediction logits remains
> invariant under moderate attention weight perturbation. This motivates our
> learned adapter approach, which can produce targeted, input-dependent
> redistribution patterns that may cross decision boundaries."

### TraceVLA (continuous)에서는 다를 수 있음
- Continuous action model은 logits를 직접 regression 값으로 사용
- argmax 불변 문제가 없으므로 fixed-var도 MSE 변화를 보일 가능성 높음
- → 이 대조가 discrete vs continuous action space의 중요한 차이를 보여줌

---

## 4. Object Mask Handling Across Models

### Vision Token Count 불일치
- OpenVLA/ECoT: 256 vision tokens (16x16 patches)
- SpatialVLA: 256 vision tokens (동일)
- TraceVLA: 313 vision tokens (Phi3V의 dynamic resolution)
- SAM mask는 256 elements로 저장됨 → TraceVLA에서 F.interpolate(nearest) 리사이징 필요

### 논문 시사점
> "Object-centric attention masks must be adapted to each model's vision token grid.
> We use nearest-neighbor interpolation to resize masks from the reference resolution
> (256 tokens) to model-specific grids (e.g., 313 for TraceVLA)."

---

## 5. Training Dynamics

### Smoke Test 결과 (100 steps)
| Model | Loss | Val Loss | 비고 |
|-------|------|----------|------|
| openvla-7b | 12.15 | 14.97 | 안정적 수렴 |
| ecot-7b | 12.92 | 22.25 | 높은 val gap → overfitting 주의 |
| spatialvla-4b | 18.85 | 24.99 | 높은 loss, GradNorm=0.0 주의 |
| tracevla-phi3v | 2.61 | 2.73 | MSE loss (continuous) → 낮은 절대값 |
| openvla lora+adapter | 12.05 | 14.70 | LoRA 추가 시 val loss 개선 |
| openvla lora | 11.97 | — | 학습 빠르게 수렴 |

### SpatialVLA GradNorm = 0.0 문제
- 100 steps에서 GradNorm이 거의 0 → 학습이 진행되지 않을 가능성
- 4B 파라미터 모델이라 hidden_dim=2304로 작아서 adapter parameter 수도 적음 (1.2M)
- Full training (50K steps)에서 개선되는지 확인 필요

### ECoT 높은 Val Gap
- Train loss 12.92 vs Val loss 22.25 → 10점 차이
- CoT 토큰이 train/val 분포 차이를 증폭시킬 가능성
- Regularization이나 early stopping 전략 검토 필요

---

## 6. Base Model Evaluation Results

### Per-Dimension MSE (193 test steps)
| Dimension | OpenVLA-7B | ECoT-7B | 차이 |
|-----------|-----------|---------|------|
| x | 0.000111 | 0.000962 | 8.7x |
| y | 0.0000778 | 0.00134 | 17.3x |
| z | 0.000117 | 0.00129 | 11.0x |
| roll | 0.000989 | 0.00608 | 6.1x |
| pitch | 0.00157 | 0.00593 | 3.8x |
| yaw | 0.00235 | 0.03378 | 14.4x |
| gripper | 0.384 | 0.538 | 1.4x |
| **Overall** | **0.0556** | **0.0839** | **1.5x** |

### 분석
- OpenVLA가 모든 차원에서 우위 (특히 y, yaw에서 큰 차이)
- 두 모델 모두 **gripper가 전체 MSE의 ~70-80%를 차지** → gripper 예측이 핵심 병목
- Spatial MSE는 매우 낮음 (0.0001 수준) → 위치 예측은 이미 정확
- Rotational MSE가 spatial보다 ~10-15배 높음 → 회전 예측이 더 어려움
- gripper는 binary(open/close)에 가까운데 MSE가 0.3-0.5 → 자주 틀림

### 논문 시사점
> "Gripper prediction dominates overall MSE across all models (70-80% of total error),
> suggesting that gripper state estimation, not spatial or rotational precision,
> is the primary bottleneck in VLA action prediction. This motivates attention-based
> interventions that could specifically improve gripper-relevant feature extraction."

---

## 7. Cross-Architecture Adapter Design

### Adapter V2 Parameter Counts
| Model | Hidden Dim | Adapter Params | 비율 |
|-------|-----------|---------------|------|
| openvla-7b | 4096 | 2,171,713 | 0.029% |
| ecot-7b | 4096 | 2,171,713 | 0.029% |
| spatialvla-4b | 2304 | 1,207,937 | 0.030% |
| tracevla-phi3v | 3072 | 1,651,073 | 0.024% |

- 모든 모델에서 전체 파라미터의 ~0.03% 이하
- 모델 크기에 비례하는 adapter size → 일관된 overhead

### LoRA + Adapter (Joint Training)
- OpenVLA: adapter (2.17M) + LoRA (1.05M) = 3.22M trainable params
- LoRA target: q_proj, v_proj (rank=16, alpha=32)
- Joint training에서 val loss 개선 관찰 (14.97 → 14.70)

---

## 8. Data Split and Statistics

### Bridge V2 Dataset (80/10/10 split)
- Train: 26,212 episodes → 889,178 steps
- Val: 3,276 episodes → 111,136 steps
- Test: 3,277 episodes → 111,562 steps
- 평균 episode 길이: ~34 steps

### Evaluation Test Set
- 6 episodes, 193 steps (적은 수 → 결과 해석 시 주의)
- Full evaluation에서는 더 많은 에피소드 필요 (100+ recommended)

---

## 9. LIBERO Benchmark Setup

### Available Suites
- `libero_spatial` (10 tasks, max 220 steps)
- `libero_object` (10 tasks, max 280 steps)
- `libero_goal` (10 tasks, max 300 steps)
- `libero_10` (10 tasks, max 520 steps)
- `libero_90` (90 tasks, max 400 steps)
- `libero_100` (100 tasks)

### Evaluation Protocol
- 50 trials per task (deterministic initial states)
- 7D action: dx, dy, dz, droll, dpitch, dyaw, gripper
- Success rate is the gold standard metric
- Image preprocessing: 180-degree rotation + JPEG encode/decode (matching RLDS pipeline)

---

*Last updated: 2026-02-23*
