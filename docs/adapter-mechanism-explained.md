# V2-Full Object-Centric Dynamic Adapter: 동작 메커니즘 해설

## 핵심 파일
- `adapter_model.py:178-362` — AttentionAdapterV2 클래스
- `attention_v3.py:224-407` — apply_var() 재분배 함수
- `attention_v3.py:670-705` — patched forward에서 호출 흐름

---

## 1. 토큰 구조

OpenVLA의 입력 시퀀스:
```
[v0, v1, v2, ..., v255, "In:", "What", ..., "\n", "Out:", a0, a1, ..., a6]
 ├─ 256개 vision tokens ─┤  ├── text tokens ──┤        ├─ 7 action tokens ─┤
```

각 vision token `vi`는 이미지의 16x16 그리드 중 하나의 패치에 대응:
```
v0   v1   v2   ... v15     ← 이미지 첫 번째 줄 (16 패치)
v16  v17  v18  ... v31     ← 이미지 두 번째 줄
...
v240 v241 v242 ... v255    ← 이미지 마지막 줄
```

SAM Object Mask:
```
object_mask = [0, 0, 0, 1, 1, 0, ..., 1, 1, 0, ...]  # (256,)
                           ↑ 물체      ↑ 물체
# SAM2 + GroundingDINO가 instruction에서 물체를 찾아 binary mask 생성
```

---

## 2. 어댑터 아키텍처 (AttentionAdapterV2)

### Branch 1: **어떤 헤드에서 얼마나 sink를 줄일 것인가** (`p_matrix`)
```
입력: h_last (language hidden state, 4096-dim)
      + object_mask (256-dim binary mask)

처리:
  h_emb = SiLU(Linear(h_last))                   # (256,)
  mask_emb = SiLU(Linear(object_mask))            # (64,)
  concat = [h_emb, mask_emb]                      # (320,)
  p_logits = MLP(concat)                          # (128,)
  p_matrix = sigmoid(p_logits)                    # (4 layers × 32 heads) ∈ [0,1]

출력 의미:
  p=0 → 해당 헤드의 sink 건드리지 않음
  p=0.7 → sink attention의 70%를 빼내서 재분배
  p=1 → sink attention 전량 재분배 (실전에서는 드묾)
```

### Branch 2: **빼낸 attention을 어디로 보낼 것인가** (`redistribution_weights`)
```
입력: h_last (query), h_vision (keys, 256×4096), object_mask

처리:
  query = Linear(h_last)                          # (128,)
  keys = Linear(h_vision)                         # (256, 128)
  scores = query · keys^T / √(128 × temperature)  # (256,)
  scores[background] = -∞                         # 배경 마스킹
  weights = softmax(scores)                       # (256,) 합=1

출력 의미:
  물체 패치에 높은 가중치, 배경 패치에 낮은 가중치
  → sink에서 빼낸 attention이 물체 위치에 집중적으로 분배됨
```

### blend_alpha: proportional → learned 전환
```
blend_alpha = sigmoid(learnable_logit)
# 학습 초기: ~0.018 (거의 proportional 분배)
# 학습 후기: 증가 (learned weights 비중 증가)
final_weights = blend_alpha * learned + (1 - blend_alpha) * proportional
```

---

## 3. 전체 동작 흐름 (하나의 레이어, 하나의 헤드 예시)

**Layer 29, Head 5에서 action token a0(x좌표)을 생성할 때:**

### Step 1: Sink 탐지 (`detect_sinks`, attention_v3.py:41-70)
```
threshold = α/N = 5.0/273 = 0.0183

Head 5의 attention map:
  a0 → v0:  0.45   > 0.0183 → ✅ sink
  a0 → v1:  0.002
  a0 → v3:  0.008  (물체 패치)
  a0 → v4:  0.012  (물체 패치)
  a0 → "\n": 0.20  > 0.0183 → ✅ text sink
  ...

sink_indices = [0]  (vision 범위 내 sink)
```

### Step 2: 이미지-중심 헤드 필터링 (`head_mask`, attention_v3.py:273-289)
```
useful_vision = sum(v1 + v2 + ... + v255, sink v0 제외)

if useful_vision >= rho(0.5):
    → 이 헤드는 이미지에 관심 있음 → VAR 적용
else:
    → 관심 부족 → 이 헤드는 건너뜀
```

### Step 3: 어댑터가 p 결정 (`adapter.forward()`)
```
p_matrix[layer29][head5] = 0.7  (어댑터 MLP 출력)
effective_p = 0.7 × head_mask[head5]
```

### Step 4: Sink에서 attention 빼냄 (attention_v3.py:304-307)
```
v0의 원래 attention: 0.45
빼내는 양 (freed): 0.45 × 0.7 = 0.315
v0에 남는 양: 0.45 × 0.3 = 0.135
```

### Step 5: 어디에 나눠주는가 (attention_v3.py:312-320)
```
어댑터 Branch 2의 redistribution_weights (SAM mask로 배경 차단됨):
  v1: 0.01  (배경 → 거의 0)
  v3: 0.25  ← 물체 패치: 높은 가중치
  v4: 0.30  ← 물체 패치: 높은 가중치
  v5: 0.20  ← 물체 패치
  v100: 0.02 (배경)
  합계 = 1.0
```

### Step 6: 실제 재분배 (attention_v3.py:345)
```
v3에 추가: 0.315 × 0.25 = 0.079
v4에 추가: 0.315 × 0.30 = 0.095
v5에 추가: 0.315 × 0.20 = 0.063

최종 결과:
  v0: 0.45 → 0.135  (sink 줄어듦)
  v3: 0.008 → 0.087 (물체 패치 ↑↑ 10배)
  v4: 0.012 → 0.107 (물체 패치 ↑↑ 9배)
  v5: 0.006 → 0.069 (물체 패치 ↑↑ 11배)
  v100: 0.003 → 0.009 (배경은 소폭 증가)
```

---

## 4. 어댑터가 결정하는 것들 (요약)

| 결정 사항 | 누가 결정 | 어떻게 |
|-----------|----------|--------|
| **어떤 토큰이 sink인가** | `detect_sinks()` | α/N threshold (동적, 헤드별 union) |
| **어떤 헤드에 VAR 적용할 것인가** | rho threshold + 어댑터 p값 | useful_vision >= rho인 헤드만, 그중 p>0인 헤드 |
| **얼마나 뺄 것인가** | 어댑터 Branch 1 (p_matrix) | 각 헤드별 p ∈ [0,1], sink attention의 p% 재분배 |
| **어디로 보낼 것인가** | 어댑터 Branch 2 (redistribution_weights) | cross-attention + SAM mask로 물체 패치에 집중 |
| **적용 레이어** | config.ADAPTER_TARGET_LAYERS | [28, 29, 30, 31] (마지막 4개 레이어) |

---

## 5. 모든 것은 각 헤드 내부에서만 발생

```
헤드 간 교차 없음:
  Head 5의 attention map → Head 5 내에서만 재분배
  Head 12의 attention map → Head 12 내에서만 재분배

각 헤드는 독립적으로:
  1. sink가 있는지 확인
  2. 이미지-중심인지 확인
  3. 어댑터의 p값에 따라 sink → 물체 패치로 attention 이동
```

---

## 6. 학습 과정

```
CE Loss = CrossEntropy(predicted_action_tokens, ground_truth_action_tokens)
L1 Loss = λ × ||p_matrix||₁  (p를 sparse하게 유지)

Total Loss = CE + L1

Gradient path:
  Loss → logits → LM head (frozen) → hidden states → modified attention
  → apply_var(p_matrix, redistribution_weights) → Adapter MLP (learnable)

모델 전체는 frozen, 어댑터(~2.17M params)만 학습
```

---

## 7. 파라미터 수

```
Branch 1 (p_head): Linear(4096,256) + Linear(256,64) + MLP(320→128→128) ≈ 1.2M
Branch 2 (cross-attn): Linear(4096,128) + Linear(4096,128) ≈ 1.0M
blend_alpha: 1 param

총: ~2.17M params (7B 모델의 0.031%)
```
