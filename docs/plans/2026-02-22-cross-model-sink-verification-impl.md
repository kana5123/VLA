# Cross-Model Attention Sink Verification — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Verify attention sink universality across 5 Tier-1 VLA models × 5 datasets, producing per-head JSONs, heatmaps, and integrated cross-model comparison visualizations.

**Architecture:** Fix `load_bridge_sample()` to use the data cache, implement loaders for 4 additional datasets, verify/fix model registry entries, download models+datasets to ceph, then run GPU-parallel extraction on GPUs 5-7. Finally generate comparison figures in `outputs/visualizations/`.

**Tech Stack:** PyTorch, HuggingFace Transformers, h5py, tensorflow-datasets, lerobot, numpy, matplotlib

---

### Task 1: Fix `load_bridge_sample()` to Use Data Cache

The current `load_bridge_sample()` reads from `config.METADATA_PATH` which resolves to `/ceph_data/kana5123/bridge_v2_data/metadata.json`. This path does **not exist** (actual dir is `bridge_data_v2`). Fix it to use the robust data cache (`metadata.pkl` + `images.dat`) at `/ceph_data/kana5123/bridge_data_cache/`.

**Files:**
- Modify: `dataset_registry.py:138-156`
- Test: `tests/test_dataset_registry.py` (create)

**Step 1: Write the failing test**

```python
# tests/test_dataset_registry.py
"""Tests for dataset_registry loaders."""
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image
from pathlib import Path
import pickle
import json

from dataset_registry import DatasetSample, load_bridge_sample


class TestLoadBridgeSample:
    """Test load_bridge_sample from data cache."""

    def test_returns_dataset_sample(self, tmp_path):
        """load_bridge_sample returns a valid DatasetSample."""
        # Create minimal test cache
        total_steps = 10
        img_h, img_w = 256, 256
        images = np.random.randint(0, 255, (total_steps, img_h, img_w, 3), dtype=np.uint8)
        images_path = tmp_path / "images.dat"
        mmap = np.memmap(str(images_path), dtype=np.uint8, mode="w+",
                         shape=(total_steps, img_h, img_w, 3))
        mmap[:] = images
        mmap.flush()

        metadata = []
        for i in range(total_steps):
            metadata.append({
                "episode_id": i // 5,
                "step_id": i % 5,
                "global_idx": i,
                "instruction": f"pick up object {i}",
                "action": [0.1 * j for j in range(7)],
            })
        with open(tmp_path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)
        with open(tmp_path / "cache_info.json", "w") as f:
            json.dump({"total_steps": total_steps, "image_height": img_h,
                        "image_width": img_w, "total_episodes": 2}, f)

        with patch("dataset_registry.DATA_CACHE_DIR", tmp_path):
            sample = load_bridge_sample(episode_id=0, step_id=0)

        assert isinstance(sample, DatasetSample)
        assert sample.dataset_name == "bridge_v2"
        assert sample.episode_id == 0
        assert sample.step_id == 0
        assert isinstance(sample.image, Image.Image)
        assert sample.image.size == (256, 256)
        assert sample.instruction == "pick up object 0"
        assert sample.action is not None
        assert len(sample.action) == 7

    def test_different_episode_step(self, tmp_path):
        """Can load different episode/step combinations."""
        total_steps = 10
        img_h, img_w = 256, 256
        images_path = tmp_path / "images.dat"
        mmap = np.memmap(str(images_path), dtype=np.uint8, mode="w+",
                         shape=(total_steps, img_h, img_w, 3))
        mmap[:] = np.random.randint(0, 255, (total_steps, img_h, img_w, 3), dtype=np.uint8)
        mmap.flush()

        metadata = []
        for i in range(total_steps):
            metadata.append({
                "episode_id": i // 5,
                "step_id": i % 5,
                "global_idx": i,
                "instruction": f"instruction ep{i // 5} step{i % 5}",
                "action": [float(i)] * 7,
            })
        with open(tmp_path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)
        with open(tmp_path / "cache_info.json", "w") as f:
            json.dump({"total_steps": total_steps, "image_height": img_h,
                        "image_width": img_w, "total_episodes": 2}, f)

        with patch("dataset_registry.DATA_CACHE_DIR", tmp_path):
            sample = load_bridge_sample(episode_id=1, step_id=2)

        assert sample.episode_id == 1
        assert sample.step_id == 2
        assert sample.instruction == "instruction ep1 step2"
```

**Step 2: Run test to verify it fails**

Run: `cd /home/kana5123/ATLASVLA && python -m pytest tests/test_dataset_registry.py -v`
Expected: FAIL (import error or `DATA_CACHE_DIR` not patchable)

**Step 3: Implement the fix**

In `dataset_registry.py`, add `DATA_CACHE_DIR` import and rewrite `load_bridge_sample()`:

```python
# At top of dataset_registry.py, after existing imports:
import pickle

# Add module-level constant (after DATASET_CACHE):
DATA_CACHE_DIR = Path("/ceph_data/kana5123/bridge_data_cache")


# Replace load_bridge_sample (lines 138-156) with:
def load_bridge_sample(episode_id: int = 0, step_id: int = 0) -> DatasetSample:
    """Load a sample from the Bridge V2 data cache (memmap + pickle).

    Uses the preprocessed cache at DATA_CACHE_DIR with:
        - metadata.pkl: list of step dicts with episode_id, step_id, global_idx, instruction, action
        - images.dat: numpy memmap (total_steps, 256, 256, 3) uint8
        - cache_info.json: shape info
    """
    cache_dir = DATA_CACHE_DIR
    info_path = cache_dir / "cache_info.json"
    meta_path = cache_dir / "metadata.pkl"
    images_path = cache_dir / "images.dat"

    with open(info_path) as f:
        info = json.load(f)
    with open(meta_path, "rb") as f:
        all_metadata = pickle.load(f)

    # Find the matching step
    for entry in all_metadata:
        if entry["episode_id"] == episode_id and entry["step_id"] == step_id:
            global_idx = entry["global_idx"]
            break
    else:
        raise ValueError(
            f"No step found for episode_id={episode_id}, step_id={step_id}"
        )

    # Load image from memmap
    total_steps = info["total_steps"]
    img_h = info["image_height"]
    img_w = info["image_width"]
    images_mmap = np.memmap(
        str(images_path), dtype=np.uint8, mode="r",
        shape=(total_steps, img_h, img_w, 3),
    )
    image_array = np.array(images_mmap[global_idx])
    image = Image.fromarray(image_array)

    return DatasetSample(
        dataset_name="bridge_v2",
        episode_id=entry["episode_id"],
        step_id=entry["step_id"],
        image=image,
        instruction=entry["instruction"],
        action=entry.get("action"),
    )
```

**Step 4: Run test to verify it passes**

Run: `cd /home/kana5123/ATLASVLA && python -m pytest tests/test_dataset_registry.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
cd /home/kana5123/ATLASVLA
git add dataset_registry.py tests/test_dataset_registry.py
git commit -m "fix: rewrite load_bridge_sample to use data cache instead of missing metadata.json"
```

---

### Task 2: Add Generic `load_sample()` Dispatcher

Add a single entry point that routes to the correct dataset loader. This will be called by `cross_model_extract.py`.

**Files:**
- Modify: `dataset_registry.py` (add `load_sample()` function at bottom)
- Modify: `cross_model_extract.py:225-228` (use `load_sample()` instead of `load_bridge_sample()`)

**Step 1: Add `load_sample()` dispatcher**

Append to `dataset_registry.py`:

```python
def load_sample(dataset_name: str, episode_id: int = 0, step_id: int = 0) -> DatasetSample:
    """Load a single sample from any registered dataset.

    Routes to dataset-specific loader functions.
    """
    loaders = {
        "bridge_v2": load_bridge_sample,
        "calvin_debug": load_calvin_sample,
        "lerobot_pusht": load_lerobot_sample,
        "droid_100": load_droid_sample,
        "rh20t_mini": load_rh20t_sample,
    }
    if dataset_name not in loaders:
        available = ", ".join(loaders.keys())
        raise ValueError(f"No loader for dataset '{dataset_name}'. Available: {available}")
    return loaders[dataset_name](episode_id, step_id)
```

**Step 2: Update `cross_model_extract.py` to use dispatcher**

Replace lines 225-228 in `run_single_model()`:

```python
# Old:
if dataset_name == "bridge_v2":
    sample = load_bridge_sample(episode_id, step_id)
else:
    raise NotImplementedError(f"Dataset loader for {dataset_name} not yet implemented")

# New:
from dataset_registry import load_sample
sample = load_sample(dataset_name, episode_id, step_id)
```

Also update the import at top of `cross_model_extract.py` (line 27):

```python
# Old:
from dataset_registry import load_bridge_sample, DatasetSample

# New:
from dataset_registry import load_sample, DatasetSample
```

**Step 3: Commit**

```bash
cd /home/kana5123/ATLASVLA
git add dataset_registry.py cross_model_extract.py
git commit -m "feat: add generic load_sample() dispatcher for multi-dataset support"
```

---

### Task 3: Implement CALVIN Debug Loader

CALVIN debug split is an HDF5 dataset (~1.3GB) with observation images and language instructions.

**Files:**
- Modify: `dataset_registry.py` (add `load_calvin_sample()`)
- Test: `tests/test_dataset_registry.py` (add CALVIN test)

**Step 1: Write the failing test**

Add to `tests/test_dataset_registry.py`:

```python
class TestLoadCalvinSample:
    """Test load_calvin_sample from CALVIN debug data."""

    def test_returns_dataset_sample_from_mock(self, tmp_path):
        """load_calvin_sample returns valid DatasetSample from mock HDF5."""
        import h5py

        # Create minimal CALVIN-like HDF5
        ep_dir = tmp_path / "training"
        ep_dir.mkdir()
        with h5py.File(ep_dir / "ep_0000.hdf5", "w") as f:
            f.create_dataset("rgb_static", data=np.random.randint(0, 255, (10, 200, 200, 3), dtype=np.uint8))
            f.create_dataset("actions", data=np.random.randn(10, 7).astype(np.float32))
            f.attrs["language"] = "slide the door to the right"

        with patch("dataset_registry.DATASET_CACHE", tmp_path):
            with patch("dataset_registry._calvin_data_dir", return_value=ep_dir):
                sample = load_calvin_sample(episode_id=0, step_id=0)

        assert isinstance(sample, DatasetSample)
        assert sample.dataset_name == "calvin_debug"
        assert isinstance(sample.image, Image.Image)
```

**Step 2: Run test to verify it fails**

Run: `cd /home/kana5123/ATLASVLA && python -m pytest tests/test_dataset_registry.py::TestLoadCalvinSample -v`
Expected: FAIL (function not defined)

**Step 3: Implement `load_calvin_sample()`**

Add to `dataset_registry.py`:

```python
def _calvin_data_dir() -> Path:
    """Return path to CALVIN debug training data."""
    base = DATASET_CACHE / "calvin" / "task_D_D" / "training"
    if not base.exists():
        # Try alternative structure
        alt = DATASET_CACHE / "calvin_debug" / "training"
        if alt.exists():
            return alt
        raise FileNotFoundError(
            f"CALVIN data not found at {base} or {alt}. "
            f"Download with: cd {DATASET_CACHE} && "
            f"wget http://calvin.cs.uni-freiburg.de/dataset/task_D_D_debug.zip && unzip task_D_D_debug.zip"
        )
    return base


def load_calvin_sample(episode_id: int = 0, step_id: int = 0) -> DatasetSample:
    """Load a sample from CALVIN debug dataset (HDF5 format).

    CALVIN stores episodes as individual HDF5 files with:
        - rgb_static: (T, H, W, 3) uint8 images from static camera
        - actions: (T, 7) float32 actions
        - language attribute: task instruction string
    """
    import h5py

    data_dir = _calvin_data_dir()
    episode_files = sorted(data_dir.glob("ep_*.hdf5"))
    if episode_id >= len(episode_files):
        raise ValueError(f"Episode {episode_id} not found. Only {len(episode_files)} episodes available.")

    ep_path = episode_files[episode_id]
    with h5py.File(ep_path, "r") as f:
        images = f["rgb_static"][:]
        actions = f["actions"][:]
        instruction = f.attrs.get("language", "slide the door to the right")
        if isinstance(instruction, bytes):
            instruction = instruction.decode("utf-8")

    if step_id >= len(images):
        raise ValueError(f"Step {step_id} out of range. Episode has {len(images)} steps.")

    image = Image.fromarray(images[step_id])
    action = actions[step_id].tolist() if step_id < len(actions) else None

    return DatasetSample(
        dataset_name="calvin_debug",
        episode_id=episode_id,
        step_id=step_id,
        image=image,
        instruction=instruction,
        action=action,
    )
```

**Step 4: Run test to verify it passes**

Run: `cd /home/kana5123/ATLASVLA && python -m pytest tests/test_dataset_registry.py::TestLoadCalvinSample -v`
Expected: PASS

**Step 5: Commit**

```bash
cd /home/kana5123/ATLASVLA
git add dataset_registry.py tests/test_dataset_registry.py
git commit -m "feat: add CALVIN debug dataset loader"
```

---

### Task 4: Implement LeRobot PushT Loader

LeRobot uses HuggingFace datasets format. Requires `lerobot` package installation.

**Files:**
- Modify: `dataset_registry.py` (add `load_lerobot_sample()`)

**Step 1: Install lerobot**

```bash
cd /home/kana5123/ATLASVLA
pip install lerobot
```

If `lerobot` install fails or is too heavy, use HuggingFace `datasets` directly:

```bash
pip install datasets  # already installed (4.5.0)
```

**Step 2: Implement `load_lerobot_sample()`**

Add to `dataset_registry.py`:

```python
def load_lerobot_sample(episode_id: int = 0, step_id: int = 0) -> DatasetSample:
    """Load a sample from LeRobot PushT dataset.

    Uses HuggingFace datasets to load lerobot/pusht.
    Falls back to a default instruction since PushT has no language annotations.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "lerobot/pusht",
            split="train",
            cache_dir=str(DATASET_CACHE / "lerobot"),
        )
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to load lerobot/pusht: {e}. "
            f"Install: pip install datasets"
        ) from e

    # Filter to requested episode
    ep_data = ds.filter(lambda x: x["episode_index"] == episode_id)
    if len(ep_data) == 0:
        raise ValueError(f"Episode {episode_id} not found in lerobot/pusht.")

    if step_id >= len(ep_data):
        raise ValueError(f"Step {step_id} out of range. Episode has {len(ep_data)} steps.")

    row = ep_data[step_id]

    # PushT stores image as PIL or numpy
    img = row.get("observation.image")
    if img is None:
        img = row.get("image")
    if isinstance(img, np.ndarray):
        image = Image.fromarray(img)
    elif isinstance(img, Image.Image):
        image = img
    else:
        raise ValueError(f"Unexpected image type: {type(img)}")

    # PushT actions are 2D (x, y) — pad to 7 for compatibility
    action_raw = row.get("action")
    if action_raw is not None:
        action_list = list(action_raw) if hasattr(action_raw, "__iter__") else [action_raw]
        # Pad to 7 dimensions
        action = action_list + [0.0] * max(0, 7 - len(action_list))
    else:
        action = None

    cfg = get_dataset("lerobot_pusht")
    return DatasetSample(
        dataset_name="lerobot_pusht",
        episode_id=episode_id,
        step_id=step_id,
        image=image.convert("RGB"),
        instruction=cfg.default_instruction,
        action=action,
    )
```

**Step 3: Commit**

```bash
cd /home/kana5123/ATLASVLA
git add dataset_registry.py
git commit -m "feat: add LeRobot PushT dataset loader"
```

---

### Task 5: Implement DROID-100 and RH20T Loaders

These are HDF5 datasets with similar loading patterns.

**Files:**
- Modify: `dataset_registry.py` (add `load_droid_sample()` and `load_rh20t_sample()`)

**Step 1: Implement `load_droid_sample()`**

Add to `dataset_registry.py`:

```python
def load_droid_sample(episode_id: int = 0, step_id: int = 0) -> DatasetSample:
    """Load a sample from DROID-100 dataset (HDF5 format).

    DROID stores episodes as HDF5 files with:
        - observation/image: (T, H, W, 3) images
        - action: (T, 7) actions (6 DOF + gripper)
    """
    import h5py

    data_dir = DATASET_CACHE / "droid_100"
    if not data_dir.exists():
        raise FileNotFoundError(
            f"DROID-100 not found at {data_dir}. "
            f"Download with: pip install gdown && gdown --folder <google_drive_url> -O {data_dir}"
        )

    episode_files = sorted(data_dir.glob("**/*.hdf5")) + sorted(data_dir.glob("**/*.h5"))
    if not episode_files:
        raise FileNotFoundError(f"No HDF5 files found in {data_dir}")
    if episode_id >= len(episode_files):
        raise ValueError(f"Episode {episode_id} not found. Only {len(episode_files)} episodes.")

    ep_path = episode_files[episode_id]
    with h5py.File(ep_path, "r") as f:
        # Try common DROID key patterns
        img_key = None
        for candidate in ["observation/image", "observations/images/im_256", "obs/image", "image"]:
            if candidate in f:
                img_key = candidate
                break
        if img_key is None:
            # Search recursively
            all_keys = []
            f.visit(lambda k: all_keys.append(k))
            img_candidates = [k for k in all_keys if "image" in k.lower() and isinstance(f[k], h5py.Dataset)]
            if img_candidates:
                img_key = img_candidates[0]
            else:
                raise KeyError(f"No image dataset found in {ep_path}. Keys: {all_keys[:20]}")

        images = f[img_key][:]
        act_key = None
        for candidate in ["action", "actions", "action/abs"]:
            if candidate in f:
                act_key = candidate
                break
        actions = f[act_key][:] if act_key else None

    if step_id >= len(images):
        raise ValueError(f"Step {step_id} out of range. Episode has {len(images)} steps.")

    image = Image.fromarray(images[step_id])
    action = actions[step_id].tolist() if actions is not None and step_id < len(actions) else None
    if action and len(action) < 7:
        action = action + [0.0] * (7 - len(action))

    cfg = get_dataset("droid_100")
    return DatasetSample(
        dataset_name="droid_100",
        episode_id=episode_id,
        step_id=step_id,
        image=image.convert("RGB"),
        instruction=cfg.default_instruction,
        action=action,
    )


def load_rh20t_sample(episode_id: int = 0, step_id: int = 0) -> DatasetSample:
    """Load a sample from RH20T Mini dataset.

    RH20T uses a custom format with per-scene directories.
    Falls back to default instruction.
    """
    data_dir = DATASET_CACHE / "rh20t_mini"
    if not data_dir.exists():
        raise FileNotFoundError(
            f"RH20T Mini not found at {data_dir}. "
            f"Manual download required from https://rh20t.github.io/"
        )

    # RH20T stores scenes with numbered frames
    scene_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not scene_dirs:
        # Try HDF5 files
        import h5py
        h5_files = sorted(data_dir.glob("**/*.hdf5")) + sorted(data_dir.glob("**/*.h5"))
        if not h5_files:
            raise FileNotFoundError(f"No scene dirs or HDF5 files in {data_dir}")

        if episode_id >= len(h5_files):
            raise ValueError(f"Episode {episode_id} not found. Only {len(h5_files)} available.")

        with h5py.File(h5_files[episode_id], "r") as f:
            all_keys = []
            f.visit(lambda k: all_keys.append(k))
            img_candidates = [k for k in all_keys if "image" in k.lower() or "rgb" in k.lower()]
            if not img_candidates:
                raise KeyError(f"No image data in {h5_files[episode_id]}")
            images = f[img_candidates[0]][:]
            if step_id >= len(images):
                raise ValueError(f"Step {step_id} out of range.")
            image = Image.fromarray(images[step_id])

        cfg = get_dataset("rh20t_mini")
        return DatasetSample(
            dataset_name="rh20t_mini",
            episode_id=episode_id,
            step_id=step_id,
            image=image.convert("RGB"),
            instruction=cfg.default_instruction,
            action=None,
        )

    if episode_id >= len(scene_dirs):
        raise ValueError(f"Episode {episode_id} not found. Only {len(scene_dirs)} scenes.")

    scene = scene_dirs[episode_id]
    image_files = sorted(scene.glob("*.jpg")) + sorted(scene.glob("*.png"))
    if step_id >= len(image_files):
        raise ValueError(f"Step {step_id} out of range. Scene has {len(image_files)} frames.")

    image = Image.open(image_files[step_id]).convert("RGB")

    cfg = get_dataset("rh20t_mini")
    return DatasetSample(
        dataset_name="rh20t_mini",
        episode_id=episode_id,
        step_id=step_id,
        image=image,
        instruction=cfg.default_instruction,
        action=None,
    )
```

**Step 2: Commit**

```bash
cd /home/kana5123/ATLASVLA
git add dataset_registry.py
git commit -m "feat: add DROID-100 and RH20T mini dataset loaders"
```

---

### Task 6: Verify and Fix Model Registry Entries

Verify each Tier-1 model's HuggingFace ID, architecture, layers_path, and attn_module are correct. The SpatialVLA entry may be wrong (listed as `qwen2` but may actually be PaliGemma2/Gemma-2).

**Files:**
- Modify: `model_registry.py` (fix SpatialVLA, verify SmolVLA)

**Step 1: Verify SpatialVLA architecture**

Check the HuggingFace model page and config:

```bash
cd /home/kana5123/ATLASVLA
export HF_HOME=/ceph_data/kana5123/hf_cache
python -c "
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('IPEC-COMMUNITY/spatialvla-4b-224-pt', trust_remote_code=True)
print('Model type:', cfg.model_type)
print('Architectures:', getattr(cfg, 'architectures', None))
if hasattr(cfg, 'text_config'):
    tc = cfg.text_config
    print('Text model type:', tc.model_type if hasattr(tc, 'model_type') else 'N/A')
    print('Hidden size:', tc.hidden_size if hasattr(tc, 'hidden_size') else 'N/A')
    print('Num layers:', tc.num_hidden_layers if hasattr(tc, 'num_hidden_layers') else 'N/A')
    print('Num heads:', tc.num_attention_heads if hasattr(tc, 'num_attention_heads') else 'N/A')
print()
print('Full config keys:', list(vars(cfg).keys())[:20])
"
```

**Step 2: Verify SmolVLA architecture**

```bash
python -c "
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('lerobot/smolvla_base', trust_remote_code=True)
print('Model type:', cfg.model_type)
print('Architectures:', getattr(cfg, 'architectures', None))
if hasattr(cfg, 'text_config'):
    tc = cfg.text_config
    print('Text hidden size:', tc.hidden_size)
    print('Text num layers:', tc.num_hidden_layers)
    print('Text num heads:', tc.num_attention_heads)
print('Full config keys:', list(vars(cfg).keys())[:20])
"
```

**Step 3: Verify TraceVLA architecture**

```bash
python -c "
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('furonghuang-lab/tracevla_phi3v', trust_remote_code=True)
print('Model type:', cfg.model_type)
print('Hidden size:', getattr(cfg, 'hidden_size', None))
print('Num layers:', getattr(cfg, 'num_hidden_layers', None))
print('Num heads:', getattr(cfg, 'num_attention_heads', None))
"
```

**Step 4: Update model registry based on verification**

Update `model_registry.py` with any corrections found. Example fix for SpatialVLA if confirmed as PaliGemma2:

```python
# If SpatialVLA is actually PaliGemma2-based:
register(VLAModelConfig(
    name="spatialvla-4b",
    hf_id="IPEC-COMMUNITY/spatialvla-4b-224-pt",
    architecture="paligemma2",  # was "qwen2"
    vision_encoder="siglip-so400m",  # was "qwen2-vl-vit"
    num_layers=26,   # Gemma-2 2B
    num_heads=8,     # Gemma-2 2B (with GQA, 4 KV heads)
    hidden_dim=2304, # Gemma-2 2B
    vision_grid_size=14,
    num_vision_tokens=196,
    action_tokens=7,
    prompt_template="What action should the robot take to {instruction}?",
    native_datasets=["bridge_v2", "oxe"],
    notes="PaliGemma2 3B backbone, SigLIP vision, spatial features",
    layers_path="language_model.model.layers",  # PaliGemma uses language_model prefix
))
```

**Step 5: Commit**

```bash
cd /home/kana5123/ATLASVLA
git add model_registry.py
git commit -m "fix: correct model registry entries based on HF config verification"
```

---

### Task 7: Set Up HF Cache and Download Models

Download all 5 Tier-1 models to ceph-mounted HF cache. Each model needs to be downloaded once and can be shared across extractions.

**Files:** None (infrastructure setup)

**Step 1: Set up HF cache directory**

```bash
export HF_HOME=/ceph_data/kana5123/hf_cache
mkdir -p $HF_HOME
echo 'export HF_HOME=/ceph_data/kana5123/hf_cache' >> ~/.bashrc
```

**Step 2: Download models in parallel (background)**

```bash
export HF_HOME=/ceph_data/kana5123/hf_cache

# OpenVLA (may already be cached in ~/.cache)
python -c "from transformers import AutoModelForVision2Seq; AutoModelForVision2Seq.from_pretrained('openvla/openvla-7b', trust_remote_code=True)" &

# ECoT
python -c "from transformers import AutoModelForVision2Seq; AutoModelForVision2Seq.from_pretrained('Embodied-CoT/ecot-openvla-7b-bridge', trust_remote_code=True)" &

# TraceVLA
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('furonghuang-lab/tracevla_phi3v', trust_remote_code=True)" &

# SpatialVLA
python -c "from transformers import AutoModelForVision2Seq; AutoModelForVision2Seq.from_pretrained('IPEC-COMMUNITY/spatialvla-4b-224-pt', trust_remote_code=True)" &

# SmolVLA — check correct loading class
python -c "from transformers import AutoModelForVision2Seq; AutoModelForVision2Seq.from_pretrained('lerobot/smolvla_base', trust_remote_code=True)" &

wait
echo "All models downloaded"
```

**Note:** Each 7B model is ~14GB. Total download: ~50GB. With 45TB available on ceph, this is fine.

**Step 3: Verify downloads**

```bash
ls -la $HF_HOME/hub/models--*
```

---

### Task 8: Download Datasets

Download the 4 additional datasets (Bridge V2 already cached).

**Files:** None (data setup)

**Step 1: Download CALVIN Debug (~1.3GB)**

```bash
cd /ceph_data/kana5123/cross_model_datasets
mkdir -p calvin
cd calvin
wget http://calvin.cs.uni-freiburg.de/dataset/task_D_D_debug.zip
unzip task_D_D_debug.zip
```

**Step 2: Download LeRobot PushT (~500MB)**

```bash
cd /home/kana5123/ATLASVLA
export HF_HOME=/ceph_data/kana5123/hf_cache
python -c "
from datasets import load_dataset
ds = load_dataset('lerobot/pusht', split='train',
                  cache_dir='/ceph_data/kana5123/cross_model_datasets/lerobot')
print(f'Loaded {len(ds)} samples')
"
```

**Step 3: Download DROID-100 (~2GB)**

```bash
pip install gdown
cd /ceph_data/kana5123/cross_model_datasets
mkdir -p droid_100
# Note: actual Google Drive URL needs to be obtained from DROID project page
# gdown --folder <DROID_100_GDRIVE_URL> -O droid_100/
```

**Step 4: Download RH20T Mini (~26GB)**

```bash
# RH20T requires manual download from https://rh20t.github.io/
# Select 1 scene to minimize size
mkdir -p /ceph_data/kana5123/cross_model_datasets/rh20t_mini
# Manual: download 1 scene (~2-5GB) from RH20T website
```

**Note:** If DROID-100 or RH20T downloads fail or take too long, we can start extraction with Bridge V2 + CALVIN + LeRobot (3 datasets), then add the remaining 2 later.

---

### Task 9: Smoke Test — Single Model + Single Dataset Extraction

Verify the full pipeline works end-to-end before running the 25-cell matrix.

**Files:** None (verification only)

**Step 1: Test OpenVLA + Bridge V2**

```bash
cd /home/kana5123/ATLASVLA
export HF_HOME=/ceph_data/kana5123/hf_cache
CUDA_VISIBLE_DEVICES=5 python cross_model_extract.py \
    --model openvla-7b \
    --dataset bridge_v2 \
    --device cuda:0 \
    --episode 0 --step 0
```

Expected output:
```
Loading openvla-7b (openvla/openvla-7b)...
  Loaded: 32L x 32H, hidden=4096
  Boundaries: 256 vision, ~30 text
  Saved: outputs/cross_model_analysis/openvla-7b/bridge_v2/ep000_step000.json
  Saved: outputs/cross_model_analysis/openvla-7b/bridge_v2/ep000_step000_perhead.json
```

**Step 2: Verify output files**

```bash
# Check JSON structure
python -c "
import json
with open('outputs/cross_model_analysis/openvla-7b/bridge_v2/ep000_step000_perhead.json') as f:
    data = json.load(f)
print('Keys:', list(data.keys()))
print('Perhead keys:', list(data['perhead_analysis'].keys())[:3])
action_key = list(data['perhead_analysis'].keys())[0]
layer_key = list(data['perhead_analysis'][action_key].keys())[0]
print(f'  {action_key} / {layer_key} heads:', list(data['perhead_analysis'][action_key][layer_key].keys())[:3])
head_key = list(data['perhead_analysis'][action_key][layer_key].keys())[0]
print(f'  Stats:', data['perhead_analysis'][action_key][layer_key][head_key])
"
```

**Step 3: Verify heatmap generated**

```bash
ls -la outputs/cross_model_analysis/openvla-7b/bridge_v2/*.png
```

---

### Task 10: Run Full Extraction Matrix — GPU Parallel

Run all 5 models × available datasets in parallel on GPUs 5-7.

**Files:** None (execution)

**Step 1: Create extraction runner script**

Create `run_cross_model.sh`:

```bash
#!/bin/bash
# Cross-model attention sink extraction — GPU parallel
set -e
export HF_HOME=/ceph_data/kana5123/hf_cache
export PYTHONPATH=/home/kana5123/ATLASVLA:$PYTHONPATH
cd /home/kana5123/ATLASVLA

DATASETS="bridge_v2"  # Start with bridge_v2, add others as downloaded
# DATASETS="bridge_v2 calvin_debug lerobot_pusht"  # Expand when ready

run_model() {
    local gpu=$1
    local model=$2
    for ds in $DATASETS; do
        echo "[GPU $gpu] Running $model on $ds..."
        CUDA_VISIBLE_DEVICES=$gpu python cross_model_extract.py \
            --model "$model" --dataset "$ds" --device cuda:0 \
            --episode 0 --step 0 2>&1 | tee "outputs/cross_model_analysis/${model}_${ds}.log"
        echo "[GPU $gpu] Done: $model on $ds"
    done
}

# GPU 5: OpenVLA-7B (~15GB)
run_model 5 openvla-7b &
PID1=$!

# GPU 6: ECoT-7B (~15GB)
run_model 6 ecot-7b &
PID2=$!

# GPU 7: Smaller models sequential
(
    run_model 7 tracevla-phi3v
    run_model 7 spatialvla-4b
    run_model 7 smolvla-base
) &
PID3=$!

echo "Waiting for all extractions to complete..."
echo "  GPU 5 (OpenVLA): PID $PID1"
echo "  GPU 6 (ECoT):    PID $PID2"
echo "  GPU 7 (Small):   PID $PID3"

wait $PID1 $PID2 $PID3
echo "All extractions complete!"
```

**Step 2: Run it**

```bash
chmod +x run_cross_model.sh
./run_cross_model.sh
```

**Step 3: Monitor progress**

```bash
# Check GPU usage
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

# Check output files
find outputs/cross_model_analysis/ -name "*_perhead.json" | wc -l
```

**Step 4: Commit script**

```bash
git add run_cross_model.sh
git commit -m "feat: add GPU-parallel cross-model extraction runner"
```

---

### Task 11: Generate Cross-Model Comparison Visualizations

Once extractions are complete, run comparison pipeline.

**Files:**
- Modify: `cross_model_compare.py` (add per-dataset comparison figure)
- Output: `outputs/visualizations/` (comparison PNGs + summary JSON)

**Step 1: Run existing comparison**

```bash
cd /home/kana5123/ATLASVLA
python cross_model_compare.py \
    --base-dir outputs/cross_model_analysis \
    --output-dir outputs/visualizations
```

Expected output:
```
Loading per-head data from: outputs/cross_model_analysis
  Loaded openvla-7b from ep000_step000_perhead.json
  Loaded ecot-7b from ep000_step000_perhead.json
  ...
Found 5 models: ['ecot-7b', 'openvla-7b', 'smolvla-base', 'spatialvla-4b', 'tracevla-phi3v']

Computing sink summaries...
  openvla-7b: vision[0]=0.XXX, text=0.XXX, useful=0.XXX, action=0.XXX
  ...

Saved summary: outputs/visualizations/cross_model_summary.json
  Saved: outputs/visualizations/cross_model_sink_comparison.png
  Saved: outputs/visualizations/cross_model_heatmap.png
  Saved: outputs/visualizations/cross_model_dual_sink.png
  Saved: outputs/visualizations/cross_model_table.tex
```

**Step 2: Add per-dataset comparison figure**

Add to `cross_model_compare.py`, inside `plot_cross_model_comparison()`:

```python
# ── Figure 4: Per-dataset comparison (if multiple datasets) ──
# Group results by dataset
dataset_groups = {}
for model_dir in sorted(base_dir.iterdir()):
    if not model_dir.is_dir():
        continue
    for ds_dir in sorted(model_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        ds_name = ds_dir.name
        if ds_name not in dataset_groups:
            dataset_groups[ds_name] = {}
        perhead_files = sorted(ds_dir.glob("*_perhead.json"))
        if perhead_files:
            with open(perhead_files[0]) as f:
                data = json.load(f)
            if "perhead_analysis" in data:
                summary = compute_sink_summary(model_dir.name, data["perhead_analysis"])
                dataset_groups[ds_name][model_dir.name] = summary

if len(dataset_groups) > 1:
    fig4, axes = plt.subplots(1, len(dataset_groups), figsize=(6 * len(dataset_groups), 6), sharey=True)
    if len(dataset_groups) == 1:
        axes = [axes]
    for ax, (ds_name, models) in zip(axes, sorted(dataset_groups.items())):
        names = list(models.keys())
        v0_vals = [models[m]["mean_vision0"] for m in names]
        ax.bar(range(len(names)), v0_vals, color="firebrick", alpha=0.85)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_title(ds_name, fontsize=11)
        ax.set_ylabel("Mean Vision[0] Ratio")
    fig4.suptitle("Attention Sink by Dataset", fontsize=13)
    plt.tight_layout()
    path4 = output_dir / "per_dataset_comparison.png"
    fig4.savefig(path4, dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print(f"  Saved: {path4}")
```

**Step 3: Verify all outputs**

```bash
ls -la outputs/visualizations/
# Should see:
# cross_model_sink_comparison.png
# cross_model_heatmap.png
# cross_model_dual_sink.png
# cross_model_summary.json
# cross_model_table.tex
# per_dataset_comparison.png (if multiple datasets)
```

**Step 4: Commit**

```bash
cd /home/kana5123/ATLASVLA
git add cross_model_compare.py
git add outputs/visualizations/cross_model_summary.json
git commit -m "feat: add per-dataset comparison visualization and generate cross-model results"
```

---

### Task 12: Final Verification and Cleanup

**Step 1: Verify sink universality**

```bash
python -c "
import json
with open('outputs/visualizations/cross_model_summary.json') as f:
    summary = json.load(f)
print('MODEL                  VISION[0]   TEXT    USEFUL  ACTION')
print('-' * 65)
for model, s in summary.items():
    print(f'{model:22s} {s[\"mean_vision0\"]:8.3f} {s[\"mean_text_total\"]:7.3f} '
          f'{s[\"mean_vision_other\"]:7.3f} {s[\"mean_action_tokens\"]:7.3f}')

# Check universality
all_v0 = [s['mean_vision0'] for s in summary.values()]
print(f'\nAll vision[0] > 0.15: {all(v > 0.15 for v in all_v0)}')
print(f'Min vision[0]: {min(all_v0):.3f}')
print(f'Max vision[0]: {max(all_v0):.3f}')
print(f'Mean vision[0]: {sum(all_v0)/len(all_v0):.3f}')
```

**Step 2: Run all tests**

```bash
cd /home/kana5123/ATLASVLA
python -m pytest tests/ -v
```

**Step 3: Final commit**

```bash
git add -A
git status
git commit -m "feat: complete cross-model sink verification pipeline (5 models × N datasets)"
```

---

## Execution Summary

| Task | Description | Duration | Dependencies |
|------|-------------|----------|--------------|
| 1 | Fix `load_bridge_sample()` | 5 min | None |
| 2 | Add `load_sample()` dispatcher | 3 min | Task 1 |
| 3 | CALVIN debug loader | 5 min | Task 2 |
| 4 | LeRobot PushT loader | 5 min | Task 2 |
| 5 | DROID + RH20T loaders | 5 min | Task 2 |
| 6 | Verify/fix model registry | 10 min | None |
| 7 | Download models (parallel) | 30-60 min | Task 6 |
| 8 | Download datasets | 15-30 min | None |
| 9 | Smoke test (1 model + 1 dataset) | 5 min | Tasks 1-7 |
| 10 | Full extraction matrix | 60-90 min | Tasks 1-9 |
| 11 | Comparison visualization | 10 min | Task 10 |
| 12 | Verification + cleanup | 5 min | Task 11 |

**Total estimated time:** ~3-4 hours (mostly model/data download + GPU extraction)

**Critical path:** Tasks 1→2→9→10→11→12 (code changes + extraction)
**Parallel track:** Tasks 7+8 (downloads, can run alongside code changes)
