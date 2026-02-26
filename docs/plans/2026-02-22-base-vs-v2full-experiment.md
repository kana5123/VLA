# Base vs V2-Full Object-Centric Adapter Experiment Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Base model(어댑터 없음) vs V2-Full(object-centric dynamic adapter)의 성능 차이를 OpenVLA에서 측정하고, 이후 다른 VLA 모델로 일반화한다.

**Architecture:** 기존 V1/V2-prop 학습은 건너뛰고, 최종 설계인 V2-Full(SAM masks + cross-attention redistribution + learned blend + dynamic sink detection)만 학습/평가. 데이터 분할은 80% train / 10% val / 10% test (에피소드 단위).

**Tech Stack:** PyTorch 2.4, Accelerate (multi-GPU DDP), OpenVLA-7B (frozen), AttentionAdapterV2 (~2.17M params), SAM2+GroundingDINO masks, Bridge V2 dataset (53K episodes)

---

## 현재 상태 확인

- **GPUs 1-4**: V1 adapter 학습 실행 중 (step 7450+) — **중단 대상** (사용자 요청: 이전 버전 학습 불필요)
- **GPUs 5-7**: 사용 가능
- **GPU 0**: 다른 사용자
- **SAM 전처리**: 100% 완료 (1,382,356 steps, object_masks.dat)
- **데이터 캐시**: Bridge V2 53,186 episodes 완료
- **데이터 분할**: `split_episodes()` 이미 80/10/10 비율 사용 중 ✓
- **Baseline 평가**: `outputs/experiment_results/base/eval/eval_results.json` 존재 (MSE 0.0679)
- **Sink 검증**: 4개 모델 모두 sink 존재 확인 완료 (per-layer/per-head JSON 생성됨)

---

### Task 1: V1 학습 프로세스 중단 및 GPU 확보

**Files:**
- 수정 없음 (프로세스 관리만)

**Step 1: V1 학습 프로세스 확인**

Run: `ps aux | grep "adapter_train.py.*adapter_version 1" | grep -v grep`
Expected: PID 1828703-1828706 (4개 GPU 프로세스)

**Step 2: V1 프로세스 종료**

```bash
# accelerate launch로 실행된 프로세스들이므로 부모 프로세스 종료
kill 1828703 1828704 1828705 1828706
```

Expected: GPUs 1-4 해제

**Step 3: GPU 해제 확인**

Run: `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader`
Expected: GPUs 1-4 메모리 사용량 < 100MB

**Step 4: V1 결과 보존 확인**

Run: `ls outputs/experiment_results/v1/checkpoints/`
Expected: best.pt, step_2000.pt 등 기존 체크포인트 보존됨 (삭제하지 않음)

---

### Task 2: run_adapter_experiment.py 수정 — base vs v2-full만 실행

**Files:**
- Modify: `run_adapter_experiment.py:31-52`

**Step 1: 불필요한 config 제거**

현재 CONFIGS에서 v1, v2-prop을 제거하고 base와 v2-full만 남긴다.

```python
CONFIGS = {
    "base": {
        "skip_training": True,
        "adapter_version": None,
        "description": "Raw OpenVLA baseline (no adapter)",
    },
    "v2-full": {
        "adapter_version": 2,
        "freeze_blend": False,
        "description": "AttentionAdapterV2, object-centric dynamic adapter (learned redistribution + dynamic sink)",
    },
}
```

**Step 2: 확인**

Run: `python -c "from run_adapter_experiment import CONFIGS; print(list(CONFIGS.keys()))"`
Expected: `['base', 'v2-full']`

---

### Task 3: config.py에 동적 sink 탐지 활성화 확인

**Files:**
- Verify: `config.py:94`

**Step 1: 확인**

`DYNAMIC_SINK_DETECTION = True`가 이미 설정되어 있는지 확인.

Run: `grep DYNAMIC_SINK config.py`
Expected: `DYNAMIC_SINK_DETECTION = True`

이미 True로 설정되어 있으므로 수정 불필요.

---

### Task 4: V2-Full 어댑터 학습 실행 (GPUs 1-4)

**Files:**
- 수정 없음 (실행만)

**Step 1: 학습 시작**

```bash
cd /home/kana5123/ATLASVLA
CUDA_VISIBLE_DEVICES=1,2,3,4 accelerate launch \
    --num_processes 4 \
    --mixed_precision bf16 \
    adapter_train.py \
    --adapter_version 2 \
    --output_dir outputs/experiment_results/v2-full \
    2>&1 | tee outputs/experiment_results/v2-full/train.log
```

Expected:
- SAM failure filtering → valid episodes 선정
- 80/10/10 에피소드 분할
- ~50,000 steps 학습 (early stopping 가능)
- `outputs/experiment_results/v2-full/checkpoints/best.pt` 생성
- 예상 소요: 6-12시간

**Step 2: 학습 모니터링**

```bash
# 진행 확인
tail -20 outputs/experiment_results/v2-full/train.log
```

주요 관찰 지표:
- Train loss 감소 추세
- MeanP (adapter 출력 평균) — 0.05-0.30 범위가 적정
- BlendAlpha — 0에서 서서히 증가해야 함
- Val loss — early stopping 판단 기준

---

### Task 5: Baseline 평가 재실행 (최신 코드 반영)

**Files:**
- 수정 없음 (실행만)

**Step 1: Baseline 평가**

```bash
cd /home/kana5123/ATLASVLA
CUDA_VISIBLE_DEVICES=5 python adapter_eval.py \
    --baseline_only \
    --output_dir outputs/experiment_results/base/eval \
    2>&1 | tee outputs/experiment_results/base/eval.log
```

Expected: `eval_results.json` 업데이트 (동적 sink 탐지 반영 baseline)

---

### Task 6: V2-Full 어댑터 평가 (학습 완료 후)

**Files:**
- 수정 없음 (실행만)

**Step 1: V2-Full 평가**

```bash
cd /home/kana5123/ATLASVLA
CUDA_VISIBLE_DEVICES=5 python adapter_eval.py \
    --checkpoint outputs/experiment_results/v2-full/checkpoints/best.pt \
    --output_dir outputs/experiment_results/v2-full/eval \
    2>&1 | tee outputs/experiment_results/v2-full/eval.log
```

Expected:
- `outputs/experiment_results/v2-full/eval/eval_results.json`
- Baseline vs Adapter per-dimension MSE 비교
- Overall MSE, spatial MSE, rotational MSE

---

### Task 7: Base vs V2-Full 비교 결과 생성

**Files:**
- Modify: `compare_adapter_results.py` (v2-full 경로 추가)

**Step 1: 비교 스크립트 실행**

```bash
python compare_adapter_results.py \
    --configs base v2-full \
    --results_dir outputs/experiment_results
```

Expected:
- `outputs/experiment_results/comparison_summary.json`
- `outputs/experiment_results/mse_comparison.png`
- Per-dimension MSE 개선율 (%) 출력
- LaTeX 표 생성

---

### Task 8: 다른 VLA 모델 일반화를 위한 어댑터 코드 수정

**Files:**
- Modify: `adapter_model.py` — hidden_dim 파라미터화
- Modify: `adapter_train.py` — 모델 선택 인자 추가
- Modify: `adapter_eval.py` — 모델 선택 인자 추가
- Modify: `attention_v3.py` — layer path 일반화

**Step 1: adapter_model.py의 AttentionAdapterV2 일반화**

현재 OpenVLA에 하드코딩된 값들을 파라미터화:
- `hidden_dim`: 4096 (OpenVLA) → 각 모델별 설정
- `num_heads`: 32 → 모델별 헤드 수
- `num_target_layers`: 4 → 모델별 타겟 레이어 수
- `vision_tokens`: 256 → 모델별 vision 토큰 수

```python
class AttentionAdapterV2(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 4096,       # 모델별 설정
        num_target_layers: int = 4,    # 모델별 설정
        num_heads: int = 32,           # 모델별 설정
        vision_tokens: int = 256,      # 모델별 설정
        ...
    ):
```

**Step 2: adapter_train.py에 --model 인자 추가**

```python
parser.add_argument("--model", type=str, default="openvla-7b",
                    help="Model from model_registry")
```

모델 로딩 시 `model_registry.get_model(args.model)`에서 아키텍처 정보 가져옴.

**Step 3: attention_v3.py의 패칭 일반화**

현재 `install_v3_patch`는 LLaMA 레이어만 패칭. 다른 아키텍처:
- Gemma-2 (SpatialVLA): `model.language_model.model.layers`
- Phi-3V (TraceVLA): `model.model.layers`

`model_registry`의 `layers_path`를 사용하여 동적으로 레이어 접근:

```python
def install_v3_patch(model, layers_path="model.language_model.model.layers"):
    layers = model
    for attr in layers_path.split("."):
        layers = getattr(layers, attr)
    for layer in layers:
        # 패치 적용
```

---

### Task 9: SpatialVLA 어댑터 학습 (GPU 6)

**Files:**
- 수정 없음 (Task 8 완료 후 실행)

**Step 1: SpatialVLA V2-Full 학습**

```bash
CUDA_VISIBLE_DEVICES=6 python adapter_train.py \
    --model spatialvla-4b \
    --adapter_version 2 \
    --output_dir outputs/experiment_results/spatialvla-4b/v2-full
```

**Step 2: SpatialVLA 평가**

```bash
CUDA_VISIBLE_DEVICES=6 python adapter_eval.py \
    --model spatialvla-4b \
    --checkpoint outputs/experiment_results/spatialvla-4b/v2-full/checkpoints/best.pt \
    --output_dir outputs/experiment_results/spatialvla-4b/v2-full/eval
```

---

### Task 10: TraceVLA 어댑터 학습 (GPU 7)

**Files:**
- 수정 없음 (Task 8 완료 후 실행)

**Step 1: TraceVLA V2-Full 학습**

```bash
CUDA_VISIBLE_DEVICES=7 python adapter_train.py \
    --model tracevla-phi3v \
    --adapter_version 2 \
    --output_dir outputs/experiment_results/tracevla-phi3v/v2-full
```

**Step 2: TraceVLA 평가**

```bash
CUDA_VISIBLE_DEVICES=7 python adapter_eval.py \
    --model tracevla-phi3v \
    --checkpoint outputs/experiment_results/tracevla-phi3v/v2-full/checkpoints/best.pt \
    --output_dir outputs/experiment_results/tracevla-phi3v/v2-full/eval
```

---

### Task 11: 전체 Cross-Model 비교 리포트 생성

**Files:**
- Modify: `compare_adapter_results.py` — 다중 모델 비교 지원

**Step 1: 비교 실행**

모든 모델의 base vs v2-full 결과를 하나의 표로 정리:

| Model | Base MSE | V2-Full MSE | Improvement (%) |
|-------|---------|-------------|----------------|
| OpenVLA-7B | 0.0679 | ? | ? |
| SpatialVLA-4B | ? | ? | ? |
| TraceVLA-Phi3V | ? | ? | ? |

**Step 2: 시각화**

- Per-model, per-dimension MSE bar chart
- Cross-model improvement heatmap
- Sink type vs improvement correlation plot

---

## 실행 순서

```
Phase 1 (즉시):
  [1] V1 학습 중단, GPU 확보
  [2] run_adapter_experiment.py 수정 (base + v2-full only)
  [3] config.py 확인 (DYNAMIC_SINK_DETECTION=True)

Phase 2 (GPUs 1-4, 6-12시간):
  [4] V2-Full 어댑터 학습 (OpenVLA, Bridge V2)
  [5] Baseline 재평가

Phase 3 (학습 완료 후):
  [6] V2-Full 평가
  [7] Base vs V2-Full 비교 생성

Phase 4 (일반화, Phase 3 완료 후):
  [8] 어댑터/패칭 코드 일반화
  [9] SpatialVLA 학습+평가
  [10] TraceVLA 학습+평가
  [11] Cross-model 비교 리포트
```

---

## 검증 방법

1. V2-Full 학습 loss가 단조 감소하는지 확인
2. BlendAlpha가 0에서 서서히 증가하는지 확인
3. Test set MSE가 baseline 대비 개선되는지 확인
4. Per-dimension 분석에서 spatial/rotational 모두 개선되는지 확인
5. 다른 모델에서도 동일한 패턴(sink redistribution → MSE 감소) 관찰되는지 확인
