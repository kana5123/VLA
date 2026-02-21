"""Tests for v2 data pipeline with object masks."""
import json
import pickle
from pathlib import Path
import numpy as np
import config


def _make_test_cache(tmp_path, total_steps=20, n_episodes=4, vision_tokens=256):
    """Create a minimal test cache with object masks."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    img_h, img_w = 256, 256

    images = np.memmap(str(cache_dir / "images.dat"), dtype=np.uint8, mode="w+",
                       shape=(total_steps, img_h, img_w, 3))
    images[:] = 128
    images.flush()

    metadata = []
    for i in range(total_steps):
        ep_id = i * n_episodes // total_steps
        metadata.append({
            "global_idx": i, "instruction": "pick up the cup",
            "action": [0.0] * 7, "episode_id": ep_id, "step_id": i % 5,
        })
    with open(cache_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    with open(cache_dir / "cache_info.json", "w") as f:
        json.dump({"total_steps": total_steps, "image_height": img_h,
                    "image_width": img_w, "total_episodes": n_episodes}, f)

    masks = np.memmap(str(cache_dir / "object_masks.dat"), dtype=np.uint8, mode="w+",
                      shape=(total_steps, vision_tokens))
    for i in range(total_steps):
        if i % 4 == 0:
            masks[i] = 255  # SAM failure marker
        else:
            masks[i] = 0
            masks[i, 50:60] = 1
    masks.flush()

    (cache_dir / "DONE").touch()
    return cache_dir


def test_dataset_filters_sam_failures(tmp_path):
    from adapter_data import BridgeTfrecordDataset
    cache_dir = _make_test_cache(tmp_path, total_steps=20, n_episodes=4)
    dataset = BridgeTfrecordDataset(
        cache_dir, episode_indices=list(range(4)), split="train", use_object_masks=True,
    )
    # 20 total, 5 have failure marker (indices 0,4,8,12,16) -> 15 valid
    assert len(dataset) == 15
    item = dataset[0]
    assert "object_mask" in item
    assert item["object_mask"].shape == (256,)
    assert item["object_mask"].max() <= 1


def test_dataset_without_masks_unchanged(tmp_path):
    """Without use_object_masks, all steps included, no object_mask field."""
    from adapter_data import BridgeTfrecordDataset
    cache_dir = _make_test_cache(tmp_path, total_steps=20, n_episodes=4)
    dataset = BridgeTfrecordDataset(
        cache_dir, episode_indices=list(range(4)), split="train", use_object_masks=False,
    )
    assert len(dataset) == 20
    item = dataset[0]
    assert "object_mask" not in item


def test_collate_includes_object_mask():
    from adapter_data import adapter_collate_fn
    from PIL import Image
    batch = [
        {"image": Image.new("RGB", (4, 4)), "instruction": "a", "action": np.zeros(7),
         "episode_id": 0, "step_id": 0, "object_mask": np.zeros(256, dtype=np.uint8)},
        {"image": Image.new("RGB", (4, 4)), "instruction": "b", "action": np.zeros(7),
         "episode_id": 0, "step_id": 1, "object_mask": np.ones(256, dtype=np.uint8)},
    ]
    result = adapter_collate_fn(batch)
    assert "object_masks" in result
    assert result["object_masks"].shape == (2, 256)


def test_collate_without_object_mask():
    from adapter_data import adapter_collate_fn
    from PIL import Image
    batch = [
        {"image": Image.new("RGB", (4, 4)), "instruction": "a", "action": np.zeros(7),
         "episode_id": 0, "step_id": 0},
    ]
    result = adapter_collate_fn(batch)
    assert "object_masks" not in result


def test_compute_valid_episodes(tmp_path):
    from adapter_data import compute_valid_episodes
    cache_dir = _make_test_cache(tmp_path, total_steps=20, n_episodes=4)

    # Rewrite masks: make episode 0 (steps 0-4) ALL failures
    with open(cache_dir / "cache_info.json") as f:
        info = json.load(f)
    with open(cache_dir / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    masks = np.memmap(str(cache_dir / "object_masks.dat"), dtype=np.uint8, mode="r+",
                      shape=(info["total_steps"], 256))
    for m in metadata:
        if m["episode_id"] == 0:
            masks[m["global_idx"]] = 255
    masks.flush()

    valid = compute_valid_episodes(cache_dir, threshold=0.5)
    assert 0 not in valid


def test_compute_valid_episodes_no_masks(tmp_path):
    """Without masks file, all episodes are valid (v1 fallback)."""
    from adapter_data import compute_valid_episodes
    cache_dir = _make_test_cache(tmp_path)
    # Remove masks file
    (cache_dir / "object_masks.dat").unlink()
    valid = compute_valid_episodes(cache_dir)
    assert len(valid) == 4  # all episodes valid
