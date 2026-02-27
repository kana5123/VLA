# Phase 3: Causal Verification + Leakage Control + Downstream Connection

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Phase 2 결과(hidden probe 78-95%, signature_exists=True)를 SCI급 논문으로 끌어올리기 위해 남은 3개 핵심 검증을 수행: (A) 다중 후보 인과 실험, (B) 텍스트 라벨 누수 차단, (C) 다운스트림 성능 연결.

**Architecture:** Phase 2 파이프라인(`run_contribution_analysis.py`, `run_causal_experiment.py`)을 확장하고, 새 모듈 2개(`contribution/text_masking.py`, `run_simplerenv_eval.py`)를 추가. 기존 코드 호환성 유지.

**Tech Stack:** PyTorch, scikit-learn, SimplerEnv (ManiSkill2 기반), 기존 hook 인프라, BridgeData V2 cache

**Context:** Phase 2에서 4개 VLA 모델의 skill representation 존재를 확인했으나, SCI 논문 수준에는 (A) bottleneck/sink의 인과적 확인, (B) hidden probe 95%가 텍스트 누수가 아님을 증명, (C) 내부 지표↔정책 성능 상관이 필요함.

---

## Phase 2 핵심 결과 (이 계획의 기반)

| Model | Hidden Probe | Signature | Bottleneck Type | Causal V=0 KL | Top1 Change |
|-------|-------------|-----------|-----------------|---------------|-------------|
| ECoT-7B | 95.2% (L24) | True | pos 0 (vision), 97-99% top1 | 4.36 | 60% |
| OpenVLA-7B | 83.9% (L27) | True | pos 274 (text), 76-81% top1 | 0.30 | 15% |
| SpatialVLA-4B | 83.5% (L16) | True | normal (distributed) | 2.16-2.76 | 70% |
| TraceVLA-Phi3 | 78.3% (L25) | True | pos 1 (vision), 13-15% top1 | 13.60 | 100% |

**Critical files:**

| File | Role |
|------|------|
| `run_contribution_analysis.py` | Phase 2 메인 (balanced, query_mode, head_filter) |
| `run_causal_experiment.py` | V=0 / knockout 인과 실험 |
| `contribution/causal.py` | ValueZeroHook, AttentionKnockoutHook |
| `contribution/probe.py` | Hidden state probe (StratifiedKFold) |
| `contribution/signature.py` | Skill labels, JS distance, counterfactual |
| `contribution/compute.py` | W_OV contribution, aggregate |
| `data_sampler.py` | Balanced skill sampling |
| `model_registry.py` | 4 models with num_kv_heads |
| `config.py` | DATA_CACHE_DIR, paths |

**Environment:** `interp` conda env, 8x A100 80GB, CephFS `/ceph_data/kana5123/`

---

## Task 1: 다중 후보 인과 실험 확장 — Multi-Candidate Causal

**Why:** Phase 2 인과 실험은 레이어별 단일 dominant position만 테스트. SCI급 논문에는 Top-1/3/5 후보를 체계적으로 마스킹하면서 "몇 개를 제거해야 출력이 무너지는지"를 보여야 함. 특히 텍스트 후보(OpenVLA pos 274)도 포함해야 bottleneck이 모달리티를 가리지 않음을 증명.

**Files:**
- Modify: `run_causal_experiment.py` — multi-candidate 지원
- Modify: `contribution/causal.py` — TextTokenZeroHook 추가 (텍스트 후보용)

**Step 1: `run_causal_experiment.py`에 multi-candidate 로직 추가**

현재 코드는 contribution_report.json에서 `dominant_position_abs_t`를 읽어 단일 후보만 테스트. 이를 확장:

```python
def extract_candidates(report: dict, k_values: list[int] = [1, 3, 5]) -> dict:
    """Phase 2 리포트에서 레이어별 top-K 후보를 추출.

    Returns: {k: list[abs_t positions]}
    """
    layer_analysis = report.get("layer_analysis", {})

    # 모든 레이어에서 dominant_position_abs_t 수집 + 빈도 기반 정렬
    pos_freq = Counter()
    for layer_data in layer_analysis.values():
        abs_t = layer_data.get("dominant_position_abs_t")
        if abs_t is not None:
            pos_freq[abs_t] += 1

    # 빈도 높은 순으로 정렬
    sorted_candidates = [pos for pos, _ in pos_freq.most_common()]

    result = {}
    for k in k_values:
        result[k] = sorted_candidates[:min(k, len(sorted_candidates))]

    return result
```

그리고 각 K값에 대해 별도로 V=0 + knockout 실험을 수행:

```python
for k, targets in candidates_per_k.items():
    print(f"\n--- K={k}: masking positions {targets} ---")
    # V=0 실험
    vzero = ValueZeroHook(targets)
    vzero.register(model, model_cfg, get_layers)
    with torch.no_grad():
        out_vzero = model(**inputs)
    # ... KL + top1_change 기록 ...
    vzero.remove()

    # Knockout 실험
    knockout = AttentionKnockoutHook(targets, query_range=(text_start, text_end))
    knockout.register(model, model_cfg, get_layers)
    # ... 동일 측정 ...
    knockout.remove()
```

**Step 2: causal_report.json 구조 확장**

```json
{
  "model": "openvla-7b",
  "method": "both",
  "candidates_abs_t": [274, 225, 267],
  "per_k": {
    "1": {
      "targets": [274],
      "knockout_mean_kl": 0.05,
      "knockout_std_kl": 0.02,
      "vzero_mean_kl": 0.30,
      "vzero_std_kl": 0.44,
      "vzero_mean_top1_change": 0.15
    },
    "3": {
      "targets": [274, 225, 267],
      "knockout_mean_kl": 0.12,
      "vzero_mean_kl": 0.85,
      "vzero_mean_top1_change": 0.35
    },
    "5": { "..." }
  }
}
```

**Step 3: `contribution/causal.py`에 TextTokenZeroHook 추가**

OpenVLA의 pos 274는 텍스트 토큰이므로, 기존 ValueZeroHook (vision용) 외에 텍스트 토큰도 V=0 할 수 있어야 함. 사실 현재 ValueZeroHook은 이미 abs_t를 받으므로 vision/text 구분 없이 작동. 다만 리포트에 `candidate_modality: "text" | "vision"` 필드를 추가해서 명시:

```python
def classify_candidate_modality(abs_t: int, vision_start: int, vision_end: int) -> str:
    """후보 토큰이 vision인지 text인지 분류."""
    if vision_start <= abs_t < vision_end:
        return "vision"
    return "text"
```

**Commit:** `feat: multi-candidate causal experiment with K=1,3,5 + modality classification`

---

## Task 2: 텍스트 라벨 누수 통제 — Text Masking Control

**Why:** Hidden probe 95%(ECoT)가 "진짜 정책 내부 representation"인지 "instruction 텍스트에서 verb 읽은 것"인지 구분해야 함. 텍스트 토큰을 KV-mask해도 probe가 유의미하면 → vision에서 skill 정보가 나온다는 강력한 증거.

**Files:**
- Create: `contribution/text_masking.py`
- Modify: `run_contribution_analysis.py` — `--text_mask` 플래그 추가

**Step 1: `contribution/text_masking.py` 생성**

```python
"""Text token KV-masking for leakage control experiment.

Purpose: Zero out text tokens' KV so model can only use vision+position info.
If hidden probe accuracy remains high → skill representation is NOT text leakage.
"""
import torch
from typing import Optional


class TextKVMaskHook:
    """KV-mask text tokens: force model to use only vision tokens.

    Approach: Zero out K and V projections for all text token positions,
    so attention cannot flow from text tokens to action positions.
    Sequence length and positional encoding are preserved.
    """

    def __init__(self, text_positions: list[int]):
        """
        Args:
            text_positions: Absolute positions of text tokens to mask.
        """
        self.text_positions = text_positions
        self._handles = []

    def register(self, model, model_cfg, get_layers_fn):
        """Register hooks on K and V projections."""
        layers = get_layers_fn(model, model_cfg)
        num_heads = model_cfg.num_heads
        num_kv_heads = getattr(model_cfg, 'num_kv_heads', None) or num_heads
        head_dim = model_cfg.hidden_dim // num_heads

        for layer in layers:
            attn = layer.self_attn
            if hasattr(attn, "k_proj") and hasattr(attn, "v_proj"):
                # Separate projections (LLaMA, Gemma2)
                self._handles.append(
                    attn.k_proj.register_forward_hook(self._make_zero_hook())
                )
                self._handles.append(
                    attn.v_proj.register_forward_hook(self._make_zero_hook())
                )
            elif hasattr(attn, "qkv_proj"):
                # Fused QKV (Phi3V) — zero K and V portions
                q_dim = num_heads * head_dim
                kv_dim = num_kv_heads * head_dim
                k_start = q_dim
                k_end = q_dim + kv_dim
                v_start = q_dim + kv_dim
                v_end = q_dim + 2 * kv_dim
                self._handles.append(
                    attn.qkv_proj.register_forward_hook(
                        self._make_fused_kv_hook(k_start, v_end)  # zero both K and V
                    )
                )

    def _make_zero_hook(self):
        mask_hook = self
        def hook_fn(module, args, output):
            modified = output.clone()
            for t in mask_hook.text_positions:
                if t < modified.shape[1]:
                    modified[:, t, :] = 0.0
            return modified
        return hook_fn

    def _make_fused_kv_hook(self, kv_start: int, kv_end: int):
        mask_hook = self
        def hook_fn(module, args, output):
            modified = output.clone()
            for t in mask_hook.text_positions:
                if t < modified.shape[1]:
                    modified[:, t, kv_start:kv_end] = 0.0
            return modified
        return hook_fn

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []
```

**Step 2: `run_contribution_analysis.py`에 `--text_mask` 플래그 추가**

```python
parser.add_argument("--text_mask", action="store_true",
    help="Control experiment: mask text tokens KV. If probe still high, "
         "skill info comes from vision, not text leakage.")
```

텍스트 마스킹 모드에서는:
1. 텍스트 토큰 위치(text_start ~ text_end) 식별
2. TextKVMaskHook 등록
3. Forward pass → hidden state 추출 → probe 정확도 측정
4. 기존 (마스크 없는) 결과와 비교

```python
if args.text_mask:
    from contribution.text_masking import TextKVMaskHook
    text_positions = list(range(boundaries["text_start"], boundaries["text_end"]))
    text_mask = TextKVMaskHook(text_positions)
    text_mask.register(model, model_cfg, get_layers)
    # Forward pass with text masked → collect hidden states
    # ... (same probe logic, saved to separate key in report)
    text_mask.remove()
```

**Step 3: 리포트에 text_masked_probe 결과 추가**

```json
{
  "hidden_state_probe": {"22": 0.77, "23": 0.76, ...},
  "text_masked_probe": {"22": 0.65, "23": 0.63, ...},
  "leakage_delta": {"22": -0.12, "23": -0.13, ...}
}
```

**해석:**
- text_masked_probe ≫ chance level (≈10%) → vision에서 skill representation 존재
- text_masked_probe ≈ chance level → probe는 텍스트에서 verb 읽은 것 (leakage)
- 예상: ECoT/OpenVLA는 bottleneck이 강해서 probe가 크게 떨어질 수 있고, SpatialVLA는 vision에서 skill info가 남아있을 가능성

**Commit:** `feat: text KV-mask control experiment for probe leakage detection`

---

## Task 3: Counterfactual Δ-Signature 실행 — 코드 연결

**Why:** `--counterfactual` 플래그가 argparse에 있지만 run loop에 미구현. 같은 이미지 + 동사 교체로 "이미지 variance 제거"된 contribution delta를 구하면 skill signature 존재의 가장 강력한 증거.

**Files:**
- Modify: `run_contribution_analysis.py` — counterfactual 모드 실행 로직 추가

**Step 1: run_analysis() 함수의 per-sample loop에 counterfactual 브랜치 추가**

현재 `args.counterfactual`을 받지만 아무 작업 안 함. 추가 로직:

```python
if args.counterfactual:
    # 1. 원본 instruction에서 counterfactual 생성
    from contribution.signature import generate_counterfactual_instructions, compute_counterfactual_delta
    cf_pairs = generate_counterfactual_instructions(sample["instruction"])

    if cf_pairs:
        target_verb, swapped_instr = cf_pairs[0]

        # 2. Swapped instruction으로 forward pass
        inputs_swap = call_processor(processor, sample["image"], swapped_instr, device)
        with torch.no_grad():
            out_swap = model(**inputs_swap, output_attentions=True, output_hidden_states=True)

        # 3. Swapped C̃ 추출 (same layers, same head set)
        c_tildes_swap = []
        for l in deep_layers:
            attn_swap = out_swap.attentions[l]
            hidden_swap = out_swap.hidden_states[l]
            c_tilde_swap_l = extract_sample_contributions(
                attn_swap, hidden_swap, query_positions, wov_matrices[l],
                head_filter=args.head_filter, ...
            )
            c_tildes_swap.append(c_tilde_swap_l)

        # 4. Delta signature
        sig_orig = compute_skill_signature(c_tildes_orig)
        sig_swap = compute_skill_signature(c_tildes_swap)
        delta = compute_counterfactual_delta(sig_orig, sig_swap)

        delta_signatures.append(delta)
        delta_labels.append(sample["skill"])
        delta_cf_labels.append(target_verb)
```

**Step 2: Delta signature로 별도 within/between 분석**

```python
if delta_signatures:
    d_within_delta, d_between_delta = compute_within_between_distance(
        delta_signatures, delta_labels
    )
    report["counterfactual"] = {
        "n_pairs": len(delta_signatures),
        "d_within_delta": d_within_delta,
        "d_between_delta": d_between_delta,
        "delta_signature_exists": str(d_within_delta < d_between_delta),
        "delta_probe_accuracy": run_linear_probe(
            np.stack(delta_signatures), np.array(encoded_delta_labels)
        ),
    }
```

**해석:**
- d_within_delta < d_between_delta → 동사 교체가 contribution을 구조적으로 바꿈 = skill-conditioned routing 존재
- bottleneck 모델에서는 delta가 작을 수 있음 (어떤 verb든 같은 토큰에 몰리므로)

**Commit:** `feat: implement counterfactual Δ-signature execution in run loop`

---

## Task 4: SimplerEnv 평가 통합 — 다운스트림 성능 연결

**Why:** SCI급 논문은 "내부 현상이 정책 성능/일반화와 연결"을 요구. SimplerEnv VM/VA에서 bottleneck severity ↔ success rate를 보여줘야 함.

**Files:**
- Create: `run_simplerenv_eval.py` — SimplerEnv wrapper
- Create: `analysis/downstream_correlation.py` — 내부 지표 ↔ 성능 상관 분석

**Step 1: SimplerEnv 평가 스크립트 생성**

```python
#!/usr/bin/env python3
"""
SimplerEnv evaluation wrapper for VLA models.

Runs Visual Matching (VM) and Variant Aggregation (VA) evaluations.
Logs per-episode internal metrics (bottleneck severity, entropy, probe confidence).

Usage:
  python run_simplerenv_eval.py --model openvla-7b --device cuda:0 \
    --task google_robot_pick_coke_can --eval_type visual_matching --n_episodes 50
"""
import argparse
import json
import numpy as np
from pathlib import Path

# SimplerEnv tasks (subset for MVP)
TASKS = {
    "pick": "google_robot_pick_coke_can",
    "move_near": "google_robot_move_near",
    "drawer_open": "google_robot_open_drawer",
    "drawer_close": "google_robot_close_drawer",
}

EVAL_TYPES = ["visual_matching", "variant_aggregation"]


def setup_simplerenv():
    """SimplerEnv 환경 초기화 (simpler conda env 사용)."""
    try:
        import simpler_env
        return True
    except ImportError:
        print("SimplerEnv not found. Install: pip install simpler-env")
        return False


def run_episode(env, model, processor, model_cfg, device, max_steps=300):
    """Single episode: step through env, collect internal metrics per step.

    Returns:
        success: bool
        metrics: dict with per-step bottleneck severity, entropy, etc.
    """
    obs = env.reset()
    done = False
    step_metrics = []
    total_reward = 0

    for step in range(max_steps):
        if done:
            break

        # Get image from obs
        image = obs["image"] if isinstance(obs, dict) else obs
        instruction = env.get_language_instruction()

        # Model forward with internal metric collection
        # ... (hook-based extraction similar to contribution analysis)

        # Get action from model
        action = model.predict_action(image, instruction)
        obs, reward, done, info = env.step(action)
        total_reward += reward

    success = info.get("success", total_reward > 0)
    return success, step_metrics


def evaluate(args):
    """Full evaluation loop."""
    results = {
        "model": args.model,
        "task": args.task,
        "eval_type": args.eval_type,
        "n_episodes": args.n_episodes,
        "episodes": [],
    }

    # ... setup model, env ...

    successes = 0
    for ep in range(args.n_episodes):
        success, metrics = run_episode(env, model, processor, model_cfg, args.device)
        successes += int(success)
        results["episodes"].append({
            "episode_id": ep,
            "success": success,
            "metrics": metrics,
        })

    results["success_rate"] = successes / args.n_episodes

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "eval_report.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Success rate: {results['success_rate']:.2%} ({successes}/{args.n_episodes})")
    return results
```

**Step 2: downstream_correlation.py — 상관 분석**

```python
"""
Correlate internal metrics (bottleneck severity, probe accuracy, entropy)
with downstream performance (SimplerEnv success rate).
"""
import json
import numpy as np
from scipy import stats
from pathlib import Path


def load_all_results(base_dir: str) -> dict:
    """Load contribution reports + eval reports for all models."""
    results = {}
    base = Path(base_dir)

    for model_dir in base.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        contrib_report = model_dir / "contribution_report.json"
        eval_report = model_dir / "eval_report.json"

        if contrib_report.exists():
            with open(contrib_report) as f:
                results.setdefault(model_name, {})["contribution"] = json.load(f)
        if eval_report.exists():
            with open(eval_report) as f:
                results.setdefault(model_name, {})["eval"] = json.load(f)

    return results


def compute_correlations(results: dict) -> dict:
    """Compute correlation between internal metrics and success rate.

    Internal metrics per model:
    - max_top1_share: highest top1 contrib share across layers
    - mean_entropy: average contribution entropy
    - best_probe_accuracy: highest hidden state probe accuracy
    - d_within: mean within-skill JS distance
    - mean_mismatch: attention-contribution mismatch

    Downstream:
    - success_rate_vm: Visual Matching success
    - success_rate_va: Variant Aggregation success
    """
    metrics = []
    for model_name, data in results.items():
        contrib = data.get("contribution", {})
        eval_data = data.get("eval", {})

        if not contrib or not eval_data:
            continue

        layer_analysis = contrib.get("layer_analysis", {})
        top1_shares = [v.get("mean_top1_share", 0) for v in layer_analysis.values()]

        metrics.append({
            "model": model_name,
            "max_top1_share": max(top1_shares) if top1_shares else 0,
            "mean_mismatch": contrib.get("mean_mismatch", 0),
            "best_probe": max(contrib.get("hidden_state_probe", {}).values(), default=0),
            "d_within": contrib.get("skill_distribution", {}).get("d_within", 0),
            "success_rate": eval_data.get("success_rate", 0),
        })

    if len(metrics) < 3:
        return {"warning": "Need at least 3 models for meaningful correlation"}

    # Spearman rank correlations
    correlations = {}
    for metric_key in ["max_top1_share", "mean_mismatch", "best_probe", "d_within"]:
        x = [m[metric_key] for m in metrics]
        y = [m["success_rate"] for m in metrics]
        rho, p = stats.spearmanr(x, y)
        correlations[metric_key] = {"rho": rho, "p_value": p}

    return {"metrics": metrics, "correlations": correlations}
```

**Note:** SimplerEnv 완전 통합은 환경 설치/호환성 확인이 선행되어야 하므로, 이 Task에서는 scaffold + 분석 코드까지만 작성하고, 실제 실행은 환경 확인 후 진행.

**Commit:** `feat: add SimplerEnv evaluation scaffold + downstream correlation analysis`

---

## Task 5: Position vs Content 분리 실험 설계

**Why:** 연구 목표 Section 5의 핵심 질문 — "bottleneck 토큰이 항상 같은 index(위치)인가, 아니면 이미지 내용에 따라 이동하는가?" Phase 2에서 ECoT는 pos 0 고정, OpenVLA는 pos 274(텍스트) 고정이었지만, 이미지를 변형해도 같은 위치인지 체계적으로 확인 필요.

**Files:**
- Create: `run_position_anchoring_test.py`

**Step 1: 이미지 변형 실험**

동일 instruction + 변형된 이미지(translation, crop+pad, noise) → bottleneck position이 유지되는지:

```python
"""
Position anchoring test: Does the bottleneck position depend on
image content or is it fixed by position?

Augmentations:
1. Translation (+padding): shift image by (dx, dy) pixels
2. Random crop + resize: crop 80% area, resize back
3. Gaussian noise: add noise to image
4. Horizontal flip: mirror image

For each augmented image, run contribution analysis and check if
dominant_position_abs_t matches the original.
"""
import torch
import numpy as np
from PIL import Image, ImageFilter
from collections import Counter

AUGMENTATIONS = {
    "translate_right": lambda img: translate(img, dx=30, dy=0),
    "translate_down": lambda img: translate(img, dx=0, dy=30),
    "crop_resize": lambda img: crop_resize(img, scale=0.8),
    "gaussian_noise": lambda img: add_noise(img, std=25),
    "horizontal_flip": lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
}

def translate(img, dx, dy):
    """Translate image with zero padding."""
    arr = np.array(img)
    result = np.zeros_like(arr)
    h, w = arr.shape[:2]
    src_y = slice(max(0, -dy), min(h, h-dy))
    src_x = slice(max(0, -dx), min(w, w-dx))
    dst_y = slice(max(0, dy), min(h, h+dy))
    dst_x = slice(max(0, dx), min(w, w+dx))
    result[dst_y, dst_x] = arr[src_y, src_x]
    return Image.fromarray(result)

def compute_position_stability(orig_positions: list[int], aug_positions: dict[str, list[int]]) -> dict:
    """Compute position stability metrics.

    Returns:
        exact_match_rate: fraction of augmentations where dominant position is identical
        topk_overlap: average overlap of top-5 positions
    """
    exact_matches = 0
    total = 0
    for aug_name, aug_pos in aug_positions.items():
        if aug_pos and orig_positions:
            if aug_pos[0] == orig_positions[0]:
                exact_matches += 1
            total += 1

    return {
        "exact_match_rate": exact_matches / max(total, 1),
        "n_augmentations": total,
    }
```

**해석:**
- exact_match_rate ≈ 1.0 → position-anchored bottleneck (ECoT/OpenVLA 예상)
- exact_match_rate ≪ 1.0 → content-anchored (정상적인 vision processing)

**Commit:** `feat: add position anchoring test for bottleneck stability`

---

## Task 6: 통합 Phase 3 실행 및 리포트

**Why:** Tasks 1-5의 결과를 모아 논문 표/그림 생성에 필요한 종합 리포트를 만듦.

**Files:**
- Create: `run_phase3_all.sh` — 전체 실험 배치 스크립트
- Create: `analysis/generate_phase3_report.py` — 종합 리포트 생성

**Step 1: 배치 실행 스크립트**

```bash
#!/bin/bash
# Phase 3: 전체 실험 배치 실행
# 사전 요구: Phase 2 결과가 outputs/contribution_analysis/*-phase2/에 존재

BASEDIR=/home/kana5123/ATLASVLA
PYTHON=/home/kana5123/miniconda3/envs/interp/bin/python
MODELS=("openvla-7b" "ecot-7b" "spatialvla-4b" "tracevla-phi3v")
GPUS=(0 1 2 3)

echo "=== Phase 3A: Multi-candidate causal ==="
for i in "${!MODELS[@]}"; do
    nohup $PYTHON $BASEDIR/run_causal_experiment.py \
        --model ${MODELS[$i]} --device cuda:${GPUS[$i]} \
        --method both --balanced --n_per_skill 25 \
        --report $BASEDIR/outputs/contribution_analysis/${MODELS[$i]}-phase2/contribution_report.json \
        --output_dir $BASEDIR/outputs/causal_experiment/${MODELS[$i]}-phase3 \
        > $BASEDIR/outputs/logs/phase3_causal_${MODELS[$i]}.log 2>&1 &
done
wait
echo "Causal done."

echo "=== Phase 3B: Text masking control ==="
for i in "${!MODELS[@]}"; do
    nohup $PYTHON $BASEDIR/run_contribution_analysis.py \
        --model ${MODELS[$i]} --device cuda:${GPUS[$i]} \
        --balanced --n_per_skill 25 --query_mode last_one \
        --text_mask \
        --output_dir $BASEDIR/outputs/contribution_analysis/${MODELS[$i]}-phase3-textmask \
        > $BASEDIR/outputs/logs/phase3_textmask_${MODELS[$i]}.log 2>&1 &
done
wait
echo "Text mask done."

echo "=== Phase 3C: Counterfactual Δ-signature ==="
for i in "${!MODELS[@]}"; do
    nohup $PYTHON $BASEDIR/run_contribution_analysis.py \
        --model ${MODELS[$i]} --device cuda:${GPUS[$i]} \
        --balanced --n_per_skill 25 --query_mode last_one \
        --counterfactual \
        --output_dir $BASEDIR/outputs/contribution_analysis/${MODELS[$i]}-phase3-counterfactual \
        > $BASEDIR/outputs/logs/phase3_cf_${MODELS[$i]}.log 2>&1 &
done
wait
echo "Counterfactual done."

echo "=== Phase 3D: Position anchoring ==="
for i in "${!MODELS[@]}"; do
    nohup $PYTHON $BASEDIR/run_position_anchoring_test.py \
        --model ${MODELS[$i]} --device cuda:${GPUS[$i]} \
        --n_samples 50 \
        --output_dir $BASEDIR/outputs/position_anchoring/${MODELS[$i]}-phase3 \
        > $BASEDIR/outputs/logs/phase3_position_${MODELS[$i]}.log 2>&1 &
done
wait
echo "Position anchoring done."

echo "=== All Phase 3 experiments complete ==="
```

**Step 2: 종합 리포트 생성**

```python
"""
Phase 3 종합 리포트: 논문 Table/Figure 데이터 생성.

Tables:
1. Multi-candidate causal: K={1,3,5}별 KL + top1_change (4 models × 2 methods)
2. Text leakage control: probe accuracy (original vs text-masked) per layer
3. Counterfactual Δ: d_within_delta vs d_between_delta + delta_probe_accuracy
4. Position anchoring: exact_match_rate per augmentation per model

Figures:
1. Causal ablation curve: K vs KL (line plot, 4 models)
2. Probe leakage comparison: bar chart (original vs text-masked per model)
3. Δ-signature clustering: UMAP of delta signatures colored by skill
4. Position stability: heatmap (augmentation × model)
"""
```

**Commit:** `feat: add Phase 3 batch runner + comprehensive report generator`

---

## 실행 순서 정리

| 순서 | Task | 예상 시간 | GPU | 의존성 |
|------|------|-----------|-----|--------|
| 1 | Multi-candidate causal | 30min | 4x A100 | Phase 2 reports |
| 2 | Text masking control | 40min | 4x A100 | Phase 2 code |
| 3 | Counterfactual Δ-signature | 40min | 4x A100 | Phase 2 code |
| 4 | SimplerEnv scaffold | 20min (코드만) | - | simpler env 확인 |
| 5 | Position anchoring | 30min | 4x A100 | Phase 2 code |
| 6 | 종합 리포트 | 10min | - | Tasks 1-5 결과 |

**총 예상: GPU 작업 ~2시간 (병렬), 코드 작업 ~1시간**

---

## SCI급 논문 성립 체크리스트

Phase 3 완료 후 아래 조건을 모두 만족하면 논문 투고 가능:

1. **인과 확인됨**: bottleneck 토큰 K=1 제거 시 KL > 2.0 (ECoT, TraceVLA), sink 토큰 제거 시 KL < 0.5
2. **누수 차단됨**: text-masked probe accuracy > 50% (chance=10%) → vision에서 skill info 존재
3. **시그니처 강화됨**: Δ-signature에서 d_within < d_between 유지
4. **위치 고정 확인됨**: bottleneck 모델은 exact_match_rate > 0.9 (position-anchored)
5. **다운스트림 연결**: bottleneck severity와 OOD 성능 간 유의미한 상관 (p < 0.05)

**최소 4개 이상 달성 시 → 논문 스토리 성립**

---

## Verification Plan

각 Task 완료 후:

1. **Task 1**: `causal_report.json`에 `per_k` 키가 K=1,3,5 모두 있고, targets 리스트가 K만큼 존재
2. **Task 2**: `contribution_report.json`에 `text_masked_probe` 키가 존재하고 값이 [0, 1] 범위
3. **Task 3**: `contribution_report.json`에 `counterfactual` 키가 존재하고 `n_pairs > 0`
4. **Task 4**: `run_simplerenv_eval.py`가 `--help`로 실행 가능 (환경 미설치 시에도)
5. **Task 5**: `position_anchoring_report.json`에 augmentation별 exact_match_rate 존재
6. **Task 6**: `phase3_report.json`에 4개 모델 × 모든 실험 결과 통합
