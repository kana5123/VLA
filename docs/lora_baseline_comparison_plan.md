# LoRA Baseline Comparison Plan

## Overview

Differentiable Attention Adapter vs LoRA/DoRA 비교 평가 설계.
Adapter는 attention pattern만 수정하고, LoRA는 weight matrix를 직접 수정한다는 근본적 차이를 실험적으로 검증.

## LoRA 변형 비교

| 방법 | 핵심 원리 | 파라미터 (last 4 layers, q+v) | PEFT 지원 |
|------|-----------|-------------------------------|-----------|
| **LoRA** | W + BA (low-rank) | rank 4: 262K, rank 8: 524K, rank 16: 1.05M | O |
| **DoRA** | LoRA + weight decomposition (magnitude/direction) | ~1.1x LoRA | `use_dora=True` |
| **AdaLoRA** | rank을 layer별로 동적 할당 (SVD) | 가변 (budget 설정) | `AdaLoraConfig` |
| **QLoRA** | 4-bit quantized base + LoRA | LoRA와 동일 | O (bitsandbytes) |
| **rsLoRA** | rank-scaled (alpha/sqrt(r)) | LoRA와 동일 | `use_rslora=True` |
| **LoRA+** | A/B matrix에 다른 LR 사용 | LoRA와 동일 | `loraplus_lr_ratio` |
| **우리 Adapter** | MLP → per-head VAR strength | **1,098K** | X (커스텀) |

## 파라미터 수 계산

OpenVLA-7B (LLaMA backbone): hidden_dim=4096, q_proj/v_proj 각각 4096x4096

### LoRA on q_proj + v_proj, last 4 layers (28-31)

- 각 projection: A(4096 x r) + B(r x 4096) = 2 x 4096 x r
- 2 projections x 4 layers = 8 modules
- **rank 4**: 8 x 2 x 4096 x 4 = **262,144** (262K)
- **rank 8**: 8 x 2 x 4096 x 8 = **524,288** (524K)
- **rank 16**: 8 x 2 x 4096 x 16 = **1,048,576** (1.05M)
- **rank 32**: 8 x 2 x 4096 x 32 = **2,097,152** (2.1M)

### 우리 Adapter: 1,098,240 (≈ LoRA rank 16)

## 비교 대상 (3+1 methods)

| Method | Trainable Params | 수정 대상 | 추론 오버헤드 |
|--------|-----------------|-----------|--------------|
| **Baseline** (no modification) | 0 | 없음 | 0 |
| **LoRA rank 16** (q+v, L28-31) | 1.05M | weight matrix | 0 (merge 가능) |
| **DoRA** (q+v, L28-31) | ~1.15M | weight magnitude+direction | LoRA + α |
| **Attention Adapter** (ours) | 1.10M | attention redistribution | MLP forward 1회 |

## 추론 Latency 분석

| Scenario | 추가 FLOPs | 추가 Latency | Merge 가능? |
|----------|-----------|-------------|------------|
| LoRA (merged) | **0** | **0%** | Yes |
| LoRA (unmerged, r=16) | ~1.0M/token | <0.5% | Yes |
| MLP adapter (ours) | ~1.08M/token | ~0.5-1% | **No** |

핵심: LoRA는 merge하면 추론 비용 0, Adapter는 매번 MLP forward 필요.
로봇 real-time control (5-10Hz)에서는 이 차이가 중요할 수 있음.

## 평가 지표

| 지표 | 설명 |
|------|------|
| Overall MSE | 7차원 action 전체 평균 |
| Spatial MSE | x, y, z 3차원 |
| Rotational MSE | roll, pitch, yaw 3차원 |
| Per-dim MSE | 각 차원별 |
| Trainable Params | 학습 가능 파라미터 수 |
| Training Time | 동일 step 수 기준 wall-clock |
| Inference Latency | 7토큰 생성 시간 (ms/step) |
| VRAM Usage | 학습/추론 시 GPU 메모리 |

## 구현 계획

### 새로 만들 파일

1. `lora_train.py` — LoRA/DoRA 학습 (PEFT 라이브러리)
2. `lora_eval.py` — LoRA 모델 평가
3. `comparison_eval.py` — 통합 비교 평가

### config.py 추가 설정

```python
LORA_RESULTS_DIR = OUTPUT_DIR / "lora_results"
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LORA_TARGET_LAYERS = [28, 29, 30, 31]
```

### PEFT 핵심 코드

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    layers_to_transform=[28, 29, 30, 31],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
```

### DoRA 전환

```python
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    layers_to_transform=[28, 29, 30, 31],
    use_dora=True,  # 이것만 추가
)
```

## 핵심 인사이트

- DoRA rank 8 ≈ LoRA rank 16 성능 (Liu et al., 2024)
- Merged LoRA = 추론 비용 0 (로봇 deployment에서 결정적 장점)
- `modules_to_save=["lm_head"]`으로 action head는 full fine-tune 가능
- `layers_to_transform`으로 특정 레이어만 타겟팅 가능

## Status

- [x] 분석 완료
- [ ] PEFT 라이브러리 설치
- [ ] lora_train.py 구현
- [ ] lora_eval.py 구현
- [ ] comparison_eval.py 구현
- [ ] 실험 실행
