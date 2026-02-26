# Cross-Model Attention Sink Verification Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 4개 오픈소스 VLA 모델에서 attention sink가 보편적으로 발생하는지 검증하고, 모델별 perhead JSON + 통합 비교 시각화를 생성한다.

**Architecture:** 기존 `cross_model_extract.py` 파이프라인을 사용하되, `load_bridge_sample()`의 데이터 로딩 버그를 먼저 수정. 각 모델에 대해 1개 bridge_v2 샘플로 attention 추출 → perhead JSON 저장 → `cross_model_compare.py`로 통합 시각화.

**Tech Stack:** PyTorch, HuggingFace Transformers, matplotlib, numpy

---

### Task 1: Fix `load_bridge_sample()` — 데이터 캐시에서 직접 로딩

**Files:**
- Modify: `dataset_registry.py:138-156`
- Test: `tests/test_dataset_registry.py` (create)

**Context:**
현재 `load_bridge_sample()`은 `/ceph_data/kana5123/bridge_v2_data/metadata.json`에서 읽지만 이 파일이 존재하지 않음.
실제 데이터는 `/ceph_data/kana5123/bridge_data_cache/`에 `metadata.pkl` (1,382,356 steps) + `images.dat` (256x256x3 memmap)으로 존재.

**Step 1: Write the failing test**

```python
# tests/test_dataset_registry.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataset_registry import load_bridge_sample

def test_load_bridge_sample_returns_valid():
    sample = load_bridge_sample(episode_id=1, step_id=0)
    assert sample.image.size == (256, 256), f"Got {sample.image.size}"
    assert sample.instruction != ""
    assert sample.dataset_name == "bridge_v2"
    assert sample.episode_id == 1

def test_load_bridge_sample_image_is_rgb():
    sample = load_bridge_sample(episode_id=1, step_id=0)
    assert sample.image.mode == "RGB"

if __name__ == "__main__":
    test_load_bridge_sample_returns_valid()
    test_load_bridge_sample_image_is_rgb()
    print("All tests passed!")
```

**Step 2: Run test to verify it fails**

Run: `cd /home/kana5123/ATLASVLA && /home/kana5123/miniconda3/envs/interp/bin/python tests/test_dataset_registry.py`
Expected: FAIL — `FileNotFoundError: /ceph_data/kana5123/bridge_v2_data/metadata.json`

**Step 3: Replace `load_bridge_sample()` to use data cache**

```python
# dataset_registry.py — replace lines 138-156
def load_bridge_sample(episode_id: int = 1, step_id: int = 0) -> DatasetSample:
    """Load a sample from the Bridge V2 data cache (memmap + metadata.pkl)."""
    import pickle
    cache_dir = config.DATA_CACHE_DIR

    with open(cache_dir / "cache_info.json") as f:
        info = json.load(f)
    total_steps = info["total_steps"]
    img_h, img_w = info["image_height"], info["image_width"]

    with open(cache_dir / "metadata.pkl", "rb") as f:
        all_meta = pickle.load(f)

    # Find first step matching episode_id + step_id
    match = None
    for m in all_meta:
        if m["episode_id"] == episode_id and m["step_id"] == step_id:
            match = m
            break
    if match is None:
        raise ValueError(f"No step found for episode={episode_id}, step={step_id}")

    images_mmap = np.memmap(
        str(cache_dir / "images.dat"), dtype=np.uint8, mode="r",
        shape=(total_steps, img_h, img_w, 3),
    )
    image = Image.fromarray(np.array(images_mmap[match["global_idx"]]))

    return DatasetSample(
        dataset_name="bridge_v2",
        episode_id=match["episode_id"],
        step_id=match["step_id"],
        image=image,
        instruction=match["instruction"],
        action=match.get("action"),
    )
```

**Step 4: Run test to verify it passes**

Run: `cd /home/kana5123/ATLASVLA && /home/kana5123/miniconda3/envs/interp/bin/python tests/test_dataset_registry.py`
Expected: PASS — "All tests passed!"

**Step 5: Commit**

```bash
git add dataset_registry.py tests/test_dataset_registry.py
git commit -m "fix: load_bridge_sample reads from data cache instead of missing metadata.json"
```

---

### Task 2: Re-extract OpenVLA with perhead stats

**Files:**
- Run: `cross_model_extract.py --model openvla-7b`
- Output: `outputs/cross_model_analysis/openvla-7b/bridge_v2/ep001_step000.json`
- Output: `outputs/cross_model_analysis/openvla-7b/bridge_v2/ep001_step000_perhead.json`
- Output: `outputs/cross_model_analysis/openvla-7b/bridge_v2/perhead_heatmap_*.png`

**Context:**
기존 `outputs/attention_results/ep000_step000.json`에는 `perhead_analysis`가 없음 (head-averaged top5만 있음).
`cross_model_extract.py`는 per-head 데이터를 포함하므로 이걸로 재추출 필요.
OpenVLA는 이미 다운로드됨 — GPU 5에서 실행.

**Step 1: Verify cross_model_extract.py syntax**

Run: `cd /home/kana5123/ATLASVLA && /home/kana5123/miniconda3/envs/interp/bin/python -c "import ast; ast.parse(open('cross_model_extract.py').read()); print('OK')"`
Expected: `OK`

**Step 2: Run OpenVLA extraction on GPU 5**

```bash
cd /home/kana5123/ATLASVLA
CUDA_VISIBLE_DEVICES=5 /home/kana5123/miniconda3/envs/interp/bin/python \
    cross_model_extract.py --model openvla-7b --dataset bridge_v2 --device cuda:0 \
    --episode 1 --step 0
```

Expected output files:
- `outputs/cross_model_analysis/openvla-7b/bridge_v2/ep001_step000.json` (head-averaged)
- `outputs/cross_model_analysis/openvla-7b/bridge_v2/ep001_step000_perhead.json` (per-head)
- `outputs/cross_model_analysis/openvla-7b/bridge_v2/perhead_heatmap_*.png` (heatmaps)

**Step 3: Verify output**

```bash
python -c "
import json
with open('outputs/cross_model_analysis/openvla-7b/bridge_v2/ep001_step000_perhead.json') as f:
    d = json.load(f)
print('model:', d['model'])
print('has perhead_analysis:', 'perhead_analysis' in d)
# Check structure: action_key -> layer_key -> head_key -> stats
first_action = list(d['perhead_analysis'].keys())[0]
first_layer = list(d['perhead_analysis'][first_action].keys())[0]
heads = d['perhead_analysis'][first_action][first_layer]
print(f'{first_action}/{first_layer}: {len(heads)} heads')
first_head = list(heads.values())[0]
print('Head stats keys:', list(first_head.keys()))
"
```

**Step 4: Commit**

```bash
# Don't commit large output files — just verify they exist
echo "OpenVLA extraction complete"
```

---

### Task 3: Set up HF_HOME for model downloads on /ceph_data

**Files:**
- Modify: `run_experiment.sh` or set env var
- Create: `/ceph_data/kana5123/hf_cache/` dir

**Context:**
`/` only has 35GB free. VLA models (7B = ~14GB each) must download to `/ceph_data`.
Set `HF_HOME=/ceph_data/kana5123/hf_cache` so HuggingFace downloads there.

**Step 1: Create cache dir and symlink**

```bash
mkdir -p /ceph_data/kana5123/hf_cache
export HF_HOME=/ceph_data/kana5123/hf_cache
```

**Step 2: Verify with a quick model info check**

```bash
HF_HOME=/ceph_data/kana5123/hf_cache /home/kana5123/miniconda3/envs/interp/bin/python -c "
from huggingface_hub import model_info
for model_id in ['IPEC-COMMUNITY/spatialvla-4b-224-pt', 'HuggingFaceTB/SmolVLA-base', 'CogACT/CogACT-Base']:
    try:
        info = model_info(model_id)
        size_gb = sum(s.size for s in info.siblings if s.rfilename.endswith(('.bin','.safetensors'))) / 1e9
        print(f'{model_id}: {size_gb:.1f}GB, gated={info.gated}')
    except Exception as e:
        print(f'{model_id}: ERROR — {e}')
"
```

Expected: Model sizes and whether they're gated/public.

**Step 3: Commit**

No file changes needed — just env setup.

---

### Task 4: Extract SpatialVLA-4B attention (Qwen2-VL architecture)

**Files:**
- Run: `cross_model_extract.py --model spatialvla-4b`
- Output: `outputs/cross_model_analysis/spatialvla-4b/bridge_v2/ep001_step000_perhead.json`

**Context:**
Qwen2-VL backbone (28L, 20H) — LLaMA와 완전히 다른 아키텍처.
Vision token 수: 196 (14x14). hidden_dim=2560. ~8GB VRAM.
GPU 6에서 실행.

**Step 1: Download and extract on GPU 6**

```bash
cd /home/kana5123/ATLASVLA
HF_HOME=/ceph_data/kana5123/hf_cache CUDA_VISIBLE_DEVICES=6 \
    /home/kana5123/miniconda3/envs/interp/bin/python \
    cross_model_extract.py --model spatialvla-4b --dataset bridge_v2 --device cuda:0 \
    --episode 1 --step 0
```

**Step 2: If AutoModelForVision2Seq fails, add model-specific loading**

SpatialVLA may need custom loading (Qwen2-VL uses different class).
If it fails, modify `load_vla_model()` in `cross_model_extract.py` to handle Qwen2-VL:

```python
# In cross_model_extract.py, modify load_vla_model()
if model_cfg.architecture == "qwen2":
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    model = Qwen2VLForConditionalGeneration.from_pretrained(...)
```

**Step 3: Verify perhead JSON structure**

Same verification as Task 2 Step 3, but check: 28 layers, 20 heads.

---

### Task 5: Extract SmolVLA-base attention (SmolLM2 architecture)

**Files:**
- Run: `cross_model_extract.py --model smolvla-base`
- Output: `outputs/cross_model_analysis/smolvla-base/bridge_v2/ep001_step000_perhead.json`

**Context:**
SmolLM2 360M backbone (30L, 16H) — 아주 작은 모델.
핵심 질문: **작은 모델에서도 attention sink가 발생하는가?**
~2GB VRAM. GPU 7에서 실행.

**Step 1: Download and extract on GPU 7**

```bash
cd /home/kana5123/ATLASVLA
HF_HOME=/ceph_data/kana5123/hf_cache CUDA_VISIBLE_DEVICES=7 \
    /home/kana5123/miniconda3/envs/interp/bin/python \
    cross_model_extract.py --model smolvla-base --dataset bridge_v2 --device cuda:0 \
    --episode 1 --step 0
```

**Step 2: Handle potential issues**

SmolVLA may use a different prompt format or action token count.
If extraction fails, check model-specific requirements.

**Step 3: Verify perhead JSON structure**

Check: 30 layers, 16 heads.

---

### Task 6: Extract CogACT-Base attention (LLaMA-3 + EVA2-CLIP)

**Files:**
- Run: `cross_model_extract.py --model cogact-base`
- Output: `outputs/cross_model_analysis/cogact-base/bridge_v2/ep001_step000_perhead.json`

**Context:**
CogVLM2-Llama3 backbone (32L, 32H) — 같은 LLaMA 계열이지만 다른 vision encoder (EVA2-CLIP).
Vision tokens: 196 (14x14). ~17GB VRAM.
GPU 5에서 실행 (OpenVLA 추출 완료 후).

**Step 1: Download and extract on GPU 5 (or 6 if free)**

```bash
cd /home/kana5123/ATLASVLA
HF_HOME=/ceph_data/kana5123/hf_cache CUDA_VISIBLE_DEVICES=5 \
    /home/kana5123/miniconda3/envs/interp/bin/python \
    cross_model_extract.py --model cogact-base --dataset bridge_v2 --device cuda:0 \
    --episode 1 --step 0
```

**Step 2: Handle CogVLM2-specific loading if needed**

CogACT may need `CogVLM2ForCausalLM` or custom loading.
Check: `layers_path = "model.layers"` (set in registry — may need adjustment).

**Step 3: Verify perhead JSON structure**

Check: 32 layers, 32 heads, 196 vision tokens.

---

### Task 7: Generate cross-model comparison visualization

**Files:**
- Run: `cross_model_compare.py`
- Output: `outputs/cross_model_analysis/comparison/cross_model_sink_comparison.png`
- Output: `outputs/cross_model_analysis/comparison/cross_model_heatmap.png`
- Output: `outputs/cross_model_analysis/comparison/cross_model_dual_sink.png`
- Output: `outputs/cross_model_analysis/comparison/cross_model_summary.json`
- Output: `outputs/cross_model_analysis/comparison/cross_model_table.tex`

**Context:**
Tasks 2-6에서 생성된 perhead JSON을 통합하여 비교 시각화 생성.
CPU only — GPU 불필요.

**Step 1: Run cross-model comparison**

```bash
cd /home/kana5123/ATLASVLA
/home/kana5123/miniconda3/envs/interp/bin/python cross_model_compare.py \
    --base-dir outputs/cross_model_analysis \
    --output-dir outputs/visualizations
```

**Step 2: Verify all output files**

```bash
ls -la outputs/visualizations/cross_model_*.png
ls -la outputs/visualizations/cross_model_summary.json
ls -la outputs/visualizations/cross_model_table.tex
```

Expected: 3 PNG files, 1 JSON, 1 TEX.

**Step 3: Review summary JSON**

```bash
cat outputs/visualizations/cross_model_summary.json
```

Expected: Per-model sink ratios. Key question:
- vision[0] ratio > 0.15 in ALL models → sink is universal
- If some models show low vision[0] → sink is architecture-dependent

**Step 4: Commit**

```bash
git add dataset_registry.py tests/test_dataset_registry.py cross_model_extract.py
git commit -m "feat: cross-model attention sink verification for 4 VLA models"
```

---

## Execution Order

```
Task 1 (필수 — BLOCKER 수정)
  ↓
Task 2 (GPU 5: OpenVLA)  ─┬─  Task 3 (env setup, no GPU)
  ↓                        │
Task 4 (GPU 6: SpatialVLA) ─┤─ Task 5 (GPU 7: SmolVLA)  ← 병렬 가능
  ↓                        │
Task 6 (GPU 5: CogACT)   ─┘
  ↓
Task 7 (CPU: 통합 비교)
```

## GPU Allocation

| GPU | Task | Model | Est. VRAM | Est. Time |
|-----|------|-------|-----------|-----------|
| 0 | (사용 금지) | vLLM server (다른 사용자) | 73GB | - |
| 1-4 | Adapter 실험 | OpenVLA v1 training | 17GB×4 | 수 시간 |
| 5 | Task 2 → Task 6 | OpenVLA → CogACT | 15GB → 17GB | ~10min each |
| 6 | Task 4 | SpatialVLA | ~10GB | ~10min |
| 7 | Task 5 | SmolVLA | ~2GB | ~5min |
