# ATLASVLA 최종 실험 실행 계획

> **이 문서는 모든 이전 계획 파일을 대체하는 최종 계획입니다.**
>
> 이전 계획들(17개 .md 파일)은 각 단계의 설계/구현 기록으로만 보존합니다.

---

## 1. 프로젝트 개요

**목표**: 4개 VLA 모델에서 Object-Centric Attention Adapter (V2-Full)의 cross-model generalizability를 검증한다.

**핵심 질문**: "학습 가능한 attention redistribution이 아키텍처에 무관하게 action prediction을 개선하는가?"

---

## 2. 구현 완료 상태 (새로 만들 것 없음)

모든 핵심 모듈은 **이미 구현/검증 완료**. 아래는 정확한 코드 위치:

### 2.1 Dynamic Sink Detection (α/N = 5.0) ✅

| 항목 | 위치 |
|------|------|
| `detect_sinks()` 함수 | `attention_v3.py:41-70` |
| V3Context 필드 | `attention_v3.py:138-139` (`dynamic_sink_detection`, `sink_alpha`) |
| apply_var() 통합 | `attention_v3.py:631-633` (매 forward pass마다 호출) |
| config 설정 | `config.py:94-95` (`DYNAMIC_SINK_DETECTION=True`, `SINK_ALPHA=5.0`) |

**알고리즘**: `Token t is sink if attn[t] > α/N` (α=5.0, N=seq_len)
- Per-forward-pass 탐지, calibration 불필요
- 탐지 실패 시 `VAR_SINK_INDICES=[0]` fallback
- Non-differentiable (detach) — gradient는 redistribution만 흐름

### 2.2 Cross-Model Sink 검증 결과 ✅

| 모델 | Sink 위치 | vision[0] 비율 | α/N 탐지 |
|------|-----------|---------------|---------|
| OpenVLA-7B | vision[0] | 45.4% | ✅ |
| ECoT-7B | vision[0] | 74.5% | ✅ |
| TraceVLA-Phi3V | text BOS | 0.19% (text=20.3%) | ✅ text sink |
| SpatialVLA-4B | text tokens (225, 265 등) | text-only | ✅ text sink |

**결론**: IMAGE[0]은 보편적 sink가 아님 → 동적 α/N 탐지가 필수 → 이미 구현됨

### 2.3 아키텍처별 Attention Patch (3종) ✅

| 아키텍처 | 함수 | 위치 | API |
|---------|------|------|-----|
| LlamaAttention | `_make_v3_patched_forward()` | `attention_v3.py:679-744` | transformers 4.57.6 NEW |
| Gemma2Attention | `_make_v3_patched_forward_gemma2()` | `attention_v3.py:747-817` | Compatible |
| Phi3VAttention | `_make_v3_patched_forward_phi3v()` | `attention_v3.py:820-914` | trust_remote_code OLD |

### 2.4 Adapter 모델 ✅

| 컴포넌트 | 위치 | 역할 |
|---------|------|------|
| AttentionAdapterV2 | `adapter_model.py:185-412` | 2-branch (p-head + cross-attn) |
| Branch 1 (p_matrix) | `adapter_model.py:232-250` | 헤드별 redistribution 강도 |
| Branch 2 (redistribution) | `adapter_model.py:252-259` | cross-attention으로 redistribution target |
| blend_alpha | `adapter_model.py:261-286` | proportional → learned 전환 (init: sigmoid(-2)≈0.12) |

### 2.5 학습/평가 파이프라인 ✅

| 파일 | 상태 | 핵심 기능 |
|------|------|----------|
| `adapter_train.py` | ✅ 완료 | CE loss (discrete) + MSE loss (continuous) 모두 지원 |
| `adapter_eval.py` | ⚠️ **continuous 미지원** | `NotImplementedError` at line 57 — **유일한 gap** |
| `adapter_data.py` | ✅ 완료 | 80/10/10 split (seed=42), SAM masks 지원 |
| `lora_train.py` | ✅ 완료 | LoRA baseline (q_proj, v_proj, rank 16) |
| `run_adapter_experiment.py` | ✅ 완료 | 9개 config 정의 |
| `compare_adapter_results.py` | ✅ 완료 | Cross-model 비교 + 시각화 |

### 2.6 역할 분담 (어댑터 vs 동적 탐지)

| 역할 | 담당 | 학습? |
|------|------|------|
| **어떤 토큰이 sink인가** | `detect_sinks()` (α/N threshold) | No (non-differentiable) |
| **어떤 헤드에 적용할 것인가** | ρ filter (vision ratio ≥ 0.5) | No (heuristic) |
| **얼마나 뺄 것인가 (p)** | Adapter Branch 1 | **Yes** |
| **어디로 보낼 것인가** | Adapter Branch 2 (cross-attn) | **Yes** |

---

## 3. 모델 선정 (4개, CogACT 제외)

| 모델 | Architecture | Action Type | VRAM | `model_registry.py` |
|------|-------------|-------------|------|---------------------|
| **openvla-7b** | LLaMA-2 + Prismatic | discrete | ~15GB | `experiment_ready=True` (L86) |
| **ecot-7b** | LLaMA-2 + Prismatic | discrete | ~15GB | `experiment_ready=True` (L197) |
| **spatialvla-4b** | Gemma-2 + SigLIP | discrete | ~10GB | `experiment_ready=True` (L157) |
| **tracevla-phi3v** | Phi-3V + CLIP-ViT | **continuous** | ~8GB | `experiment_ready=True` (L131) |

**제외 모델 (확정):**
- ~~CogACT~~: Diffusion action head → CE/MSE loss 적용 불가 (`experiment_ready` 미설정)
- ~~SmolVLA~~: flex_attention 필요, 환경 호환 불가
- ~~RoboFlamingo~~: cross-attention 기반 (self-attention 아님)

---

## 4. 실험 Configs (9개, `run_adapter_experiment.py:31-85`)

### Baselines (eval only)
| Config | Description | Line |
|--------|-------------|------|
| `base` | 원본 VLA, 수정 없음 | L33 |
| `fixed-var` | 고정 VAR (p=0.6, 하드코딩 sink) | L38 |
| `act` | ACT sink scaling만 (재분배 없음) | L44 |
| `random` | 랜덤 초기화 adapter (학습 증명용) | L50 |

### Training
| Config | Description | Line |
|--------|-------------|------|
| `lora` | LoRA fine-tuning only (baseline) | L57 |
| `v1` | [Ablation] V1 adapter (MLP-only, 마스크 없음) | L63 |
| `v2-prop` | [Ablation] V2 blend 고정 (proportional만) | L67 |
| **`v2-full`** | **OUR METHOD** — V2 full (α/N + learned p + SAM) | L73 |
| **`lora+adapter`** | **OUR METHOD + LoRA** joint training | L78 |

---

## 5. 데이터

| 항목 | 값 |
|------|---|
| Dataset | Bridge V2 |
| Cache | `/ceph_data/kana5123/bridge_data_cache/` |
| Episodes | 53,186 (1.38M steps) |
| Split | 80% train / 10% val / 10% test (seed=42) |
| SAM masks | `object_masks.dat` (100% cached) |
| Split function | `adapter_data.py:split_episodes()` (L415-433) |

---

## 6. 실행 계획

### Task 0: Fix Continuous Action Evaluation (유일한 코드 수정)

**문제**: `adapter_eval.py:57` — `NotImplementedError` for continuous models (TraceVLA)
**학습은 이미 가능** (MSE loss, `adapter_train.py:179-186`), 평가만 불가

**수정 사항**:
1. `NotImplementedError` 제거
2. `self.is_continuous` 플래그 추가
3. continuous 모델: tokenizer 생략, logits에서 직접 action 추출
4. discrete 모델: 기존 autoregressive token generation 유지

### Task 1: Smoke Test (4모델 × 주요 config, 100-step)

4개 GPU에 v2-full 병렬 실행 → 크래시/OOM/sink 탐지 로그 확인:

```
GPU 1: openvla-7b     v2-full  (100 steps)
GPU 2: ecot-7b        v2-full  (100 steps)
GPU 3: spatialvla-4b  v2-full  (100 steps)
GPU 4: tracevla-phi3v v2-full  (100 steps)
```

**확인 사항**:
- α/N 동적 탐지가 각 모델에서 **서로 다른** sink를 정확히 탐지하는지 로그 확인
- Training loss가 감소하는지
- OOM 없이 완료되는지

### Task 2: Full Training — Round 1 (7B models)

| GPU | Model | Config |
|-----|-------|--------|
| 1 | openvla-7b | v2-full |
| 2 | openvla-7b | lora+adapter |
| 3 | openvla-7b | lora |
| 4 | ecot-7b | v2-full |
| 5 | ecot-7b | lora+adapter |
| 6 | ecot-7b | lora |

### Task 3: Full Training — Round 2 (smaller models)

| GPU | Model | Config |
|-----|-------|--------|
| 1 | spatialvla-4b | v2-full |
| 2 | spatialvla-4b | lora+adapter |
| 3 | spatialvla-4b | lora |
| 4 | tracevla-phi3v | v2-full |
| 5 | tracevla-phi3v | lora+adapter |
| 6 | tracevla-phi3v | lora |

### Task 4: Evaluation — All Models × All Configs

```bash
for model in openvla-7b ecot-7b spatialvla-4b tracevla-phi3v; do
    python run_adapter_experiment.py \
        --model $model \
        --configs base fixed-var act random v2-full lora+adapter lora \
        --skip_training --num_eval_episodes 200
done
```

Expected: 4 × 7 = 28 `eval_results.json`

### Task 5: Results Compilation

```bash
python compare_adapter_results.py \
    --models openvla-7b ecot-7b spatialvla-4b tracevla-phi3v \
    --output_dir outputs/experiment_results/comparison
```

**핵심 가설 검증:**
1. α/N 동적 탐지 → v2-full이 모든 모델에서 base 대비 개선
2. vision[0] sink 강한 모델(ECoT 75%) → 개선폭 최대
3. lora+adapter ≥ v2-full
4. lora alone < v2-full (attention redistribution의 독립적 효과)

### Task 6 (Optional): LIBERO Simulation Benchmark

MSE 상위 config에 대해 LIBERO 성공률 평가

---

## 7. 실행 순서 요약

```
Task 0 (30min, 코드 수정 1개)
  → Task 1 (1-2h, smoke test)
  → Task 2 + Task 3 (4-7h, 병렬 학습)
  → Task 4 (2-4h, 평가)
  → Task 5 (30min, 비교/시각화)
  → Task 6 (optional, LIBERO)
```

**총 예상: ~10-15h** (학습 병렬 시)

---

## 8. 실패 대응

| 문제 | 대응 |
|------|------|
| Training crash | `--resume` checkpoint에서 재개 |
| Eval crash | `--skip_training`으로 eval만 재실행 |
| OOM | `ADAPTER_BATCH_SIZE` 감소 |
| Sink 미탐지 | `DYNAMIC_SINK_DETECTION=False` fallback 확인 |
| TraceVLA continuous eval 실패 | Task 0 수정 재검증 |

---

## 9. 이전 계획 파일 목록 (기록 보존용)

| 파일 | 내용 | 상태 |
|------|------|------|
| `2026-02-18-sam2-grounded-segmentation-design.md` | SAM2 설계 | ✅ 구현 완료 |
| `2026-02-19-object-aware-adapter-v2-design.md` | V2 어댑터 설계 | ✅ 구현 완료 |
| `2026-02-19-object-aware-adapter-v2-impl.md` | V2 구현 계획 | ✅ 구현 완료 |
| `2026-02-21-experiment-pipeline-design.md` | 실험 파이프라인 설계 | ✅ 구현 완료 |
| `2026-02-21-experiment-pipeline-impl.md` | 파이프라인 구현 | ✅ 구현 완료 |
| `2026-02-22-base-vs-v2full-experiment.md` | 기본 실험 설계 | ✅ configs에 반영 |
| `2026-02-22-comprehensive-experiment-plan.md` | 종합 실험 | ✅ 본 문서로 대체 |
| `2026-02-22-cross-model-adapter-experiment-design.md` | Cross-model 설계 | ✅ 모델 선정 확정 |
| `2026-02-22-cross-model-adapter-impl.md` | Cross-model 구현 | ✅ 구현 완료 |
| `2026-02-22-cross-model-sink-extraction-*.md` (3개) | Sink 추출 | ✅ 완료, 결과 확인 |
| `2026-02-22-cross-model-sink-verification-*.md` (2개) | Sink 검증 | ✅ 완료, α/N 검증됨 |
| `2026-02-22-dynamic-sink-detection-design.md` | 동적 탐지 설계 | ✅ `detect_sinks()` 구현됨 |
| `2026-02-22-dynamic-sink-detection-impl.md` | 동적 탐지 구현 | ✅ 구현 완료 |
