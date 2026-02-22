# Cross-Model Sink Extraction + Adapter Training Design

## Goal

미추출 VLA 4개 모델(CogACT, SpatialVLA, SmolVLA, RoboFlamingo)의 attention sink 패턴을 추출하여 cross-model universality를 검증하고, OpenVLA에서 dynamic sink detection 기반 adapter 학습 및 평가를 수행한다.

## Architecture

3개 작업을 순차적으로 진행:
1. **Sink 추출**: GPUs 5-7에서 4모델 병렬 추출 (기존 `cross_model_extract.py` 활용)
2. **비교 분석**: 7모델 전체 cross-model 비교 + dynamic `detect_sinks()` 검증
3. **Adapter 학습/평가**: OpenVLA에서 `DYNAMIC_SINK_DETECTION=True`로 adapter 학습 → 200 에피소드 MSE 평가

## Design Decisions

### 1. GPU 배정 (GPUs 5, 6, 7)

| GPU | Model | VRAM Est. | 근거 |
|-----|-------|-----------|------|
| 5 | CogACT-Base | ~17GB | CogVLM2-LLaMA-3-8B, 가장 큰 모델 |
| 6 | SpatialVLA-4B → SmolVLA | ~10GB → ~2GB | SpatialVLA 완료 후 SmolVLA 순차 실행 |
| 7 | RoboFlamingo | ~5GB | cross-attention, 가장 작은 모델 |

GPUs 1-4는 현재 adapter v1 학습 진행 중이므로 사용하지 않음.

### 2. 데이터셋: Bridge V2만 사용

- 모든 모델에 동일 Bridge V2 샘플 (ep0, step0) 사용
- 이미 캐시되어 있어 다운로드 불필요
- 모델 간 공정 비교 가능 (동일 입력에서의 attention 패턴)

### 3. 모델별 알려진 리스크

- **CogACT**: CogVLM2는 visual expert 별도 attention 사용. `layers_path="model.layers"`가 정확한지 실행 시 확인 필요
- **SpatialVLA**: 이전 transformers 4.46.3에서 실패. 현재 4.57.6으로 해결된 것으로 예상
- **SmolVLA**: 실제 VLA 대신 underlying VLM (SmolVLM2-500M) 로딩. 정책 래퍼 없이 VLM 자체의 attention 분석
- **RoboFlamingo**: cross-attention 아키텍처. self-attention sink 패턴과 다를 수 있음. `attn_module="attn"`, `layers_path="transformer.blocks"`

### 4. Adapter 학습 (작업 3)

- GPUs 1-4의 현재 학습 완료 후 진행
- `config.DYNAMIC_SINK_DETECTION = True` (이미 구현 완료)
- 비교: hardcoded `[0]` vs dynamic `detect_sinks(α=5.0)`
- 평가: 200 에피소드 per-dimension MSE

## Expected Output

### 추출 결과 (per model)
```
outputs/cross_model_analysis/{model_name}/bridge_v2/
├── ep000_step000.json          # head-averaged top-5
├── ep000_step000_perhead.json  # per-head breakdown
└── *.png                       # per-head heatmaps
```

### 비교 결과
```
outputs/cross_model_analysis/comparison/
├── cross_model_sink_comparison.png   # 7-model bar chart
├── cross_model_heatmap.png           # model × layer heatmap
├── cross_model_dual_sink.png         # stacked bar
├── cross_model_summary.json          # numeric summary
├── cross_model_table.tex             # LaTeX table
└── detected_sinks.json               # per-model dynamic detection results
```

## Verification

1. 각 모델 extraction 성공 + perhead JSON 생성 확인
2. `detect_sinks(α=5.0)` 결과가 perhead JSON의 실제 sink 위치와 일치
3. Cross-model 비교 시각화 7개 모델 포함
4. Adapter 학습 완료 + evaluation MSE 비교표
