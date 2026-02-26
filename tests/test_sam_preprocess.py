"""Tests for SAM preprocessing utilities."""
import json
import pickle
from unittest.mock import MagicMock

import numpy as np
import pytest


# ── Task 5: Noun phrase extraction ──

def test_extract_noun_phrases():
    from sam_preprocess import extract_noun_phrases
    phrases = extract_noun_phrases("pick up the blue cup from the table")
    assert isinstance(phrases, list)
    assert len(phrases) > 0
    all_text = " ".join(phrases).lower()
    assert "cup" in all_text

def test_extract_noun_phrases_fallback():
    from sam_preprocess import extract_noun_phrases
    phrases = extract_noun_phrases("go")
    assert len(phrases) >= 1

def test_instruction_to_grounding_query():
    from sam_preprocess import instruction_to_grounding_query
    query = instruction_to_grounding_query("pick up the red block near the bowl")
    assert isinstance(query, str)
    assert len(query) > 0


# ── Task 6: Pixel mask to patch mask ──

def test_pixel_mask_to_patch_mask():
    from sam_preprocess import pixel_mask_to_patch_mask
    pixel_mask = np.zeros((256, 256), dtype=np.uint8)
    pixel_mask[:64, :64] = 1
    patch_mask = pixel_mask_to_patch_mask(pixel_mask, grid_size=16, threshold=0.1)
    assert patch_mask.shape == (256,)
    grid_2d = patch_mask.reshape(16, 16)
    assert grid_2d[:4, :4].sum() == 16
    assert grid_2d[4:, :].sum() == 0

def test_pixel_mask_to_patch_mask_threshold():
    from sam_preprocess import pixel_mask_to_patch_mask
    pixel_mask = np.zeros((256, 256), dtype=np.uint8)
    pixel_mask[0, 0] = 1
    patch_mask = pixel_mask_to_patch_mask(pixel_mask, grid_size=16, threshold=0.1)
    assert patch_mask[0] == 0

def test_pixel_mask_to_patch_mask_512():
    from sam_preprocess import pixel_mask_to_patch_mask
    pixel_mask = np.zeros((256, 256), dtype=np.uint8)
    pixel_mask[:128, :128] = 1
    patch_mask = pixel_mask_to_patch_mask(pixel_mask, grid_size=16, threshold=0.1, vision_tokens=512)
    assert patch_mask.shape == (512,)
    np.testing.assert_array_equal(patch_mask[:256], patch_mask[256:])


# ── Task 7: Per-image pipeline ──

def test_process_single_image_returns_mask():
    from sam_preprocess import process_single_image
    mock_grounding = MagicMock()
    mock_grounding.return_value = {
        "boxes": np.array([[10, 20, 100, 150]]),
        "scores": np.array([0.8]),
    }
    mock_sam = MagicMock()
    pixel_mask = np.zeros((256, 256), dtype=np.uint8)
    pixel_mask[20:150, 10:100] = 1
    mock_sam.return_value = pixel_mask

    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    result = process_single_image(
        image, "pick up the cup",
        grounding_fn=mock_grounding, sam_fn=mock_sam, vision_tokens=256,
    )
    assert result is not None
    assert result.shape == (256,)
    assert result.dtype == np.uint8
    assert result.sum() > 0

def test_process_single_image_failure():
    from sam_preprocess import process_single_image
    mock_grounding = MagicMock()
    mock_grounding.return_value = {"boxes": np.array([]).reshape(0, 4), "scores": np.array([])}
    mock_sam = MagicMock()
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    result = process_single_image(
        image, "pick up the cup",
        grounding_fn=mock_grounding, sam_fn=mock_sam, vision_tokens=256,
    )
    assert result is None


# ── Task 8: Batch processing ──

def test_preprocess_batch_creates_memmap(tmp_path):
    from sam_preprocess import preprocess_all_steps

    total_steps = 10
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    images = np.memmap(
        str(cache_dir / "images.dat"), dtype=np.uint8, mode="w+",
        shape=(total_steps, 256, 256, 3),
    )
    images[:] = np.random.randint(0, 255, (total_steps, 256, 256, 3), dtype=np.uint8)
    images.flush()

    metadata = [
        {"global_idx": i, "instruction": "pick up the cup", "episode_id": i // 3,
         "step_id": i % 3, "action": [0]*7}
        for i in range(total_steps)
    ]
    with open(cache_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    with open(cache_dir / "cache_info.json", "w") as f:
        json.dump({"total_steps": total_steps, "image_height": 256,
                    "image_width": 256, "total_episodes": 4}, f)

    def mock_ground(img, query):
        return {"boxes": np.array([[50, 50, 150, 150]]), "scores": np.array([0.9])}
    def mock_sam(img, boxes):
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[50:150, 50:150] = 1
        return mask

    preprocess_all_steps(
        cache_dir=cache_dir, grounding_fn=mock_ground,
        sam_fn=mock_sam, vision_tokens=256,
    )

    masks_path = cache_dir / "object_masks.dat"
    assert masks_path.exists()
    masks = np.memmap(str(masks_path), dtype=np.uint8, mode="r", shape=(total_steps, 256))
    assert masks.shape == (total_steps, 256)
    for i in range(total_steps):
        assert masks[i].max() <= 1
        assert masks[i].sum() > 0
