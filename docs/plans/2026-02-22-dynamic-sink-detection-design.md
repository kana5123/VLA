# Dynamic Attention Sink Detection Design

## Problem

현재 VAR(Visual Attention Redistribution)의 sink 토큰이 **하드코딩**(`VAR_SINK_INDICES = [0]`)되어 있어 OpenVLA에서만 정확하게 작동한다.

Cross-model 분석 결과:
- **OpenVLA/ECoT** (LLaMA-2): sink = vision[0] (45-75%)
- **TraceVLA** (Phi-3V): sink = BOS + `<|user|>` (20-34%) — vision[0]은 0.2%에 불과

아키텍처에 따라 sink 위치가 완전히 다르므로, adapter를 다양한 VLA 모델에 적용하려면 **동적 탐지가 필수**.

## Decision: ACT-style α/N Threshold, Per-Forward-Pass

### Algorithm

ACT (ICML 2024) 기반 threshold:

```
Token t is a sink in head h at layer l if:
    attn_h^l[t] > α / N
where:
    α = 5.0 (configurable)
    N = sequence length
```

의미: 평균 attention(1/N)의 α배 이상 받는 토큰 = sink.

### Why This Algorithm

| 후보 | 장점 | 단점 | 선택 이유 |
|------|------|------|----------|
| **ACT α/N** | 수학적 명확, 구현 간단, per-head/per-layer | 고정 α | **채택** — 실측 데이터와 일치 |
| Top-K + Outlier | 분포 적응적 | 파라미터 튜닝 필요, 계산량 | 불필요한 복잡성 |
| SnapKV voting | Cross-head 검증 | 추가 집계 로직, prefill 시점 | VLA 학습 파이프라인과 맞지 않음 |

### Why Per-Forward-Pass (Not Calibration)

- 입력마다 다른 sink 가능 (특히 multi-dataset 학습 시)
- Calibration은 분포 shift에 취약
- Overhead: threshold 비교 O(H × N)로 무시 가능

## Architecture

### Core: `detect_sinks()`

```python
def detect_sinks(attn_weights: Tensor, alpha: float = 5.0) -> Tensor:
    """
    Args:
        attn_weights: (batch, num_heads, seq_len) — last token → all tokens attention
        alpha: threshold multiplier (default 5.0)
    Returns:
        sink_mask: (batch, num_heads, seq_len) boolean — True = sink token
    """
    N = attn_weights.shape[-1]
    threshold = alpha / N
    return (attn_weights > threshold).detach()  # non-differentiable
```

### Integration Point: `_make_v3_patched_forward()`

```
model.forward()
  → LlamaAttention.forward() [patched]
    → original attention computation
    → attn_weights captured
    → if dynamic_sink_detection:
        sink_mask = detect_sinks(attn_weights[:, :, -1, :], alpha)
      else:
        sink_mask = legacy_mask(var_sink_indices)
    → apply_var(last_attn, ctx, sink_mask=sink_mask)
    → return modified attn_output
```

### V3Context Changes

```python
@dataclass
class V3Context:
    # Existing fields (unchanged)
    var_sink_indices: list[int] = field(default_factory=lambda: [0])

    # New fields
    dynamic_sink_detection: bool = True
    sink_alpha: float = 5.0
```

### apply_var() Signature Change

```python
# Before
def apply_var(last, ctx, layer_idx=None):
    sink_mask = build_mask_from_indices(ctx.var_sink_indices)
    ...

# After
def apply_var(last, ctx, layer_idx=None, sink_mask=None):
    if sink_mask is None:
        sink_mask = build_mask_from_indices(ctx.var_sink_indices)
    # sink_mask: (B, H, S) boolean — per-head, per-token
    ...
```

### Gradient Flow

```
detect_sinks() ──detach──→ sink_mask (no gradient)
                              ↓
apply_var(last_attn, sink_mask) → modified_attn (differentiable)
                              ↓
loss.backward() → gradient flows through redistribution, NOT through detection
```

Detection은 non-differentiable (binary threshold). Redistribution은 differentiable (현재와 동일). Adapter는 `p_matrix`를 학습 — sink 위치 탐지는 adapter의 역할이 아님.

## Verification

1. **OpenVLA regression**: `dynamic_sink_detection=True`로 기존 adapter 학습 재실행. Loss/accuracy가 하드코딩 대비 동등 이상.
2. **Cross-model validation**: 추출된 perhead JSON에 대해 detect_sinks 실행 → OpenVLA에서 vision[0], TraceVLA에서 BOS가 탐지되는지 확인.
3. **α sensitivity**: α=3,5,7,10에서 탐지되는 sink 수 비교.
4. **Backward compatibility**: `dynamic_sink_detection=False`로 기존 동작 완벽 유지.

## Files to Modify

| File | Change |
|------|--------|
| `attention_v3.py` | `detect_sinks()` 함수 추가, V3Context 필드 추가, `apply_var()` 시그니처 변경, `_make_v3_patched_forward()` 통합 |
| `config.py` | `DYNAMIC_SINK_DETECTION`, `SINK_ALPHA` 상수 추가 |
| `adapter_train.py` | V3Context 생성 시 dynamic detection 활성화 |
| `tests/test_sink_detector.py` | detect_sinks 단위 테스트 |

## References

- ACT: "Unveiling and Harnessing Hidden Attention Sinks" (ICML 2024) — https://github.com/GATECH-EIC/ACT
- StreamingLLM: "Efficient Streaming Language Models with Attention Sinks" (ICLR 2024)
- "When Attention Sink Emerges in Language Models" (ICLR 2025 Spotlight)
- "See What You Are Told: Visual Attention Sink in Large Multimodal Models" (ICLR 2025)
- VLA-Cache: "Towards Efficient VLA via Adaptive Token Caching" (NeurIPS 2025)
