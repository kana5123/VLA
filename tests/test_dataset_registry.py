"""Tests for dataset_registry.py — loaders for cross-model sink verification.

Tests use mock data (tmp_path fixtures) so they run without real dataset files.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from PIL import Image

# Ensure the project root is on sys.path so `import dataset_registry` works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dataset_registry
from dataset_registry import (
    DatasetConfig,
    DatasetSample,
    DATASETS,
    get_dataset,
    list_datasets,
    load_bridge_sample,
    load_calvin_sample,
    load_droid_sample,
    load_rh20t_sample,
    load_sample,
    _load_hdf5_sample,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def bridge_cache(tmp_path):
    """Create a mock Bridge V2 cache with memmap + metadata + cache_info."""
    num_steps = 5
    img_h, img_w = 64, 64

    # cache_info.json
    cache_info = {
        "total_steps": num_steps,
        "image_height": img_h,
        "image_width": img_w,
        "total_episodes": 2,
    }
    with open(tmp_path / "cache_info.json", "w") as f:
        json.dump(cache_info, f)

    # metadata.pkl — list of dicts
    metadata = [
        {
            "global_idx": 0,
            "instruction": "pick up the red block",
            "action": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
            "episode_id": 1,
            "step_id": 0,
        },
        {
            "global_idx": 1,
            "instruction": "pick up the red block",
            "action": [0.15, 0.25, 0.35, 0.01, 0.01, 0.01, 1.0],
            "episode_id": 1,
            "step_id": 1,
        },
        {
            "global_idx": 2,
            "instruction": "move arm left",
            "action": [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "episode_id": 2,
            "step_id": 0,
        },
        {
            "global_idx": 3,
            "instruction": "move arm left",
            "action": [-0.4, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            "episode_id": 2,
            "step_id": 1,
        },
        {
            "global_idx": 4,
            "instruction": "move arm left",
            "action": [-0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
            "episode_id": 2,
            "step_id": 2,
        },
    ]
    with open(tmp_path / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    # images.dat — numpy memmap
    images = np.memmap(
        tmp_path / "images.dat",
        dtype=np.uint8,
        mode="w+",
        shape=(num_steps, img_h, img_w, 3),
    )
    # Fill with recognizable patterns (each step has a distinct pixel value)
    for i in range(num_steps):
        images[i] = np.full((img_h, img_w, 3), fill_value=(i * 50) % 256, dtype=np.uint8)
    images.flush()

    return tmp_path


@pytest.fixture
def calvin_hdf5_cache(tmp_path):
    """Create a mock CALVIN cache directory with one HDF5 episode."""
    import h5py

    calvin_dir = tmp_path / "calvin"
    calvin_dir.mkdir()

    num_steps = 10
    img_h, img_w = 64, 64

    ep_path = calvin_dir / "episode_000.hdf5"
    with h5py.File(ep_path, "w") as f:
        # rgb_static images
        imgs = np.random.randint(0, 255, (num_steps, img_h, img_w, 3), dtype=np.uint8)
        f.create_dataset("rgb_static", data=imgs)
        # actions (7-DOF)
        actions = np.random.randn(num_steps, 7).astype(np.float64)
        f.create_dataset("actions", data=actions)
        # language annotation
        f.create_dataset("lang", data=[b"open the drawer"])

    return tmp_path


@pytest.fixture
def droid_hdf5_cache(tmp_path):
    """Create a mock DROID cache directory with one HDF5 episode."""
    import h5py

    droid_dir = tmp_path / "droid_100"
    droid_dir.mkdir()

    num_steps = 8
    img_h, img_w = 64, 64

    ep_path = droid_dir / "episode_000.hdf5"
    with h5py.File(ep_path, "w") as f:
        imgs = np.random.randint(0, 255, (num_steps, img_h, img_w, 3), dtype=np.uint8)
        f.create_dataset("image", data=imgs)
        actions = np.random.randn(num_steps, 7).astype(np.float64)
        f.create_dataset("action", data=actions)
        f.create_dataset("language_instruction", data=[b"pick up the cup"])

    return tmp_path


@pytest.fixture
def rh20t_hdf5_cache(tmp_path):
    """Create a mock RH20T cache directory with one HDF5 episode."""
    import h5py

    rh20t_dir = tmp_path / "rh20t_mini"
    rh20t_dir.mkdir()

    num_steps = 6
    img_h, img_w = 64, 64

    ep_path = rh20t_dir / "episode_000.hdf5"
    with h5py.File(ep_path, "w") as f:
        imgs = np.random.randint(0, 255, (num_steps, img_h, img_w, 3), dtype=np.uint8)
        f.create_dataset("rgb", data=imgs)
        actions = np.random.randn(num_steps, 7).astype(np.float64)
        f.create_dataset("tcp_action", data=actions)

    return tmp_path


# ─── TestDatasetConfig / Registry ─────────────────────────────────────────────

class TestRegistryBasics:
    """Basic tests for dataset registration and lookup."""

    def test_all_five_datasets_registered(self):
        expected = {"bridge_v2", "calvin_debug", "lerobot_pusht", "droid_100", "rh20t_mini"}
        assert expected == set(DATASETS.keys())

    def test_get_dataset_returns_config(self):
        cfg = get_dataset("bridge_v2")
        assert isinstance(cfg, DatasetConfig)
        assert cfg.name == "bridge_v2"

    def test_get_dataset_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_dataset("nonexistent_dataset")

    def test_list_datasets(self):
        names = list_datasets()
        assert "bridge_v2" in names
        assert len(names) == 5


# ─── TestLoadBridgeSample ─────────────────────────────────────────────────────

class TestLoadBridgeSample:
    """Test load_bridge_sample with mock cache (tmp_path, memmap, pickle)."""

    def test_load_first_step(self, bridge_cache):
        with patch.object(dataset_registry, "DATA_CACHE_DIR", bridge_cache):
            sample = load_bridge_sample(episode_id=1, step_id=0)

        assert isinstance(sample, DatasetSample)
        assert sample.dataset_name == "bridge_v2"
        assert sample.episode_id == 1
        assert sample.step_id == 0
        assert sample.instruction == "pick up the red block"
        assert sample.action == [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]

    def test_image_is_pil_rgb(self, bridge_cache):
        with patch.object(dataset_registry, "DATA_CACHE_DIR", bridge_cache):
            sample = load_bridge_sample(episode_id=1, step_id=0)

        assert isinstance(sample.image, Image.Image)
        assert sample.image.mode == "RGB"
        assert sample.image.size == (64, 64)

    def test_image_pixel_values(self, bridge_cache):
        """Verify the image loaded from memmap has the expected pixel pattern."""
        with patch.object(dataset_registry, "DATA_CACHE_DIR", bridge_cache):
            # global_idx=0 -> fill value 0
            sample = load_bridge_sample(episode_id=1, step_id=0)
            arr = np.array(sample.image)
            assert np.all(arr == 0)

            # global_idx=2 -> fill value 100
            sample2 = load_bridge_sample(episode_id=2, step_id=0)
            arr2 = np.array(sample2.image)
            assert np.all(arr2 == 100)

    def test_load_second_episode(self, bridge_cache):
        with patch.object(dataset_registry, "DATA_CACHE_DIR", bridge_cache):
            sample = load_bridge_sample(episode_id=2, step_id=1)

        assert sample.episode_id == 2
        assert sample.step_id == 1
        assert sample.instruction == "move arm left"

    def test_missing_episode_raises(self, bridge_cache):
        with patch.object(dataset_registry, "DATA_CACHE_DIR", bridge_cache):
            with pytest.raises(ValueError, match="No sample found"):
                load_bridge_sample(episode_id=999, step_id=0)

    def test_missing_step_raises(self, bridge_cache):
        with patch.object(dataset_registry, "DATA_CACHE_DIR", bridge_cache):
            with pytest.raises(ValueError, match="No sample found"):
                load_bridge_sample(episode_id=1, step_id=999)

    def test_action_is_list_of_floats(self, bridge_cache):
        with patch.object(dataset_registry, "DATA_CACHE_DIR", bridge_cache):
            sample = load_bridge_sample(episode_id=1, step_id=0)

        assert isinstance(sample.action, list)
        assert all(isinstance(a, float) for a in sample.action)
        assert len(sample.action) == 7


# ─── TestLoadCalvinSample ────────────────────────────────────────────────────

class TestLoadCalvinSample:
    """Test load_calvin_sample with mock HDF5 data."""

    def test_load_step(self, calvin_hdf5_cache):
        with patch.object(dataset_registry, "DATASET_CACHE", calvin_hdf5_cache):
            sample = load_calvin_sample(episode_id=0, step_id=3)

        assert isinstance(sample, DatasetSample)
        assert sample.dataset_name == "calvin_debug"
        assert sample.episode_id == 0
        assert sample.step_id == 3
        assert isinstance(sample.image, Image.Image)
        assert sample.image.mode == "RGB"

    def test_language_annotation_loaded(self, calvin_hdf5_cache):
        with patch.object(dataset_registry, "DATASET_CACHE", calvin_hdf5_cache):
            sample = load_calvin_sample(episode_id=0, step_id=0)

        assert sample.instruction == "open the drawer"

    def test_action_loaded(self, calvin_hdf5_cache):
        with patch.object(dataset_registry, "DATASET_CACHE", calvin_hdf5_cache):
            sample = load_calvin_sample(episode_id=0, step_id=0)

        assert sample.action is not None
        assert len(sample.action) == 7

    def test_step_out_of_range(self, calvin_hdf5_cache):
        with patch.object(dataset_registry, "DATASET_CACHE", calvin_hdf5_cache):
            with pytest.raises(ValueError, match="out of range"):
                load_calvin_sample(episode_id=0, step_id=999)

    def test_no_data_dir_raises(self, tmp_path):
        with patch.object(dataset_registry, "DATASET_CACHE", tmp_path):
            with pytest.raises(FileNotFoundError, match="CALVIN data not found"):
                load_calvin_sample(episode_id=0, step_id=0)


# ─── TestLoadHdf5Generic (DROID, RH20T) ──────────────────────────────────────

class TestLoadDroidSample:
    """Test load_droid_sample with mock HDF5 data."""

    def test_load_step(self, droid_hdf5_cache):
        with patch.object(dataset_registry, "DATASET_CACHE", droid_hdf5_cache):
            sample = load_droid_sample(episode_id=0, step_id=2)

        assert isinstance(sample, DatasetSample)
        assert sample.dataset_name == "droid_100"
        assert sample.episode_id == 0
        assert sample.step_id == 2
        assert isinstance(sample.image, Image.Image)
        assert sample.image.mode == "RGB"

    def test_instruction_from_hdf5(self, droid_hdf5_cache):
        with patch.object(dataset_registry, "DATASET_CACHE", droid_hdf5_cache):
            sample = load_droid_sample(episode_id=0, step_id=0)

        assert sample.instruction == "pick up the cup"

    def test_action_loaded(self, droid_hdf5_cache):
        with patch.object(dataset_registry, "DATASET_CACHE", droid_hdf5_cache):
            sample = load_droid_sample(episode_id=0, step_id=0)

        assert sample.action is not None
        assert len(sample.action) == 7

    def test_no_data_raises(self, tmp_path):
        with patch.object(dataset_registry, "DATASET_CACHE", tmp_path):
            with pytest.raises(FileNotFoundError):
                load_droid_sample(episode_id=0, step_id=0)


class TestLoadRh20tSample:
    """Test load_rh20t_sample with mock HDF5 data."""

    def test_load_step(self, rh20t_hdf5_cache):
        with patch.object(dataset_registry, "DATASET_CACHE", rh20t_hdf5_cache):
            sample = load_rh20t_sample(episode_id=0, step_id=1)

        assert isinstance(sample, DatasetSample)
        assert sample.dataset_name == "rh20t_mini"
        assert sample.episode_id == 0
        assert sample.step_id == 1
        assert isinstance(sample.image, Image.Image)

    def test_default_instruction_fallback(self, rh20t_hdf5_cache):
        """RH20T mock has no instruction key, should fall back to default."""
        with patch.object(dataset_registry, "DATASET_CACHE", rh20t_hdf5_cache):
            sample = load_rh20t_sample(episode_id=0, step_id=0)

        assert sample.instruction == "grasp the object"

    def test_action_loaded(self, rh20t_hdf5_cache):
        with patch.object(dataset_registry, "DATASET_CACHE", rh20t_hdf5_cache):
            sample = load_rh20t_sample(episode_id=0, step_id=0)

        assert sample.action is not None
        assert len(sample.action) == 7


# ─── TestLoadSampleDispatcher ────────────────────────────────────────────────

class TestLoadSampleDispatcher:
    """Test the load_sample() universal dispatcher."""

    def test_dispatch_bridge(self, bridge_cache):
        with patch.object(dataset_registry, "DATA_CACHE_DIR", bridge_cache):
            sample = load_sample("bridge_v2", episode_id=1, step_id=0)

        assert sample.dataset_name == "bridge_v2"
        assert sample.instruction == "pick up the red block"

    def test_dispatch_calvin(self, calvin_hdf5_cache):
        with patch.object(dataset_registry, "DATASET_CACHE", calvin_hdf5_cache):
            sample = load_sample("calvin_debug", episode_id=0, step_id=0)

        assert sample.dataset_name == "calvin_debug"

    def test_dispatch_droid(self, droid_hdf5_cache):
        with patch.object(dataset_registry, "DATASET_CACHE", droid_hdf5_cache):
            sample = load_sample("droid_100", episode_id=0, step_id=0)

        assert sample.dataset_name == "droid_100"

    def test_dispatch_rh20t(self, rh20t_hdf5_cache):
        with patch.object(dataset_registry, "DATASET_CACHE", rh20t_hdf5_cache):
            sample = load_sample("rh20t_mini", episode_id=0, step_id=0)

        assert sample.dataset_name == "rh20t_mini"

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_sample("nonexistent_dataset")


# ─── TestHdf5GenericHelper ────────────────────────────────────────────────────

class TestHdf5GenericHelper:
    """Test the _load_hdf5_sample helper edge cases."""

    def test_grayscale_image_converted_to_rgb(self, tmp_path):
        """If HDF5 image is grayscale (H, W), it should still produce RGB."""
        import h5py

        dataset_dir = tmp_path / "grayscale_test"
        dataset_dir.mkdir()

        ep_path = dataset_dir / "ep.hdf5"
        with h5py.File(ep_path, "w") as f:
            gray_imgs = np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8)
            f.create_dataset("image", data=gray_imgs)
            f.create_dataset("action", data=np.zeros((3, 7)))

        sample = _load_hdf5_sample(
            dataset_dir=dataset_dir,
            dataset_name="droid_100",
            episode_id=0,
            step_id=0,
            image_key_candidates=["image"],
            action_key_candidates=["action"],
            instruction_key_candidates=[],
        )

        assert sample.image.mode == "RGB"

    def test_episode_out_of_range(self, tmp_path):
        import h5py

        dataset_dir = tmp_path / "range_test"
        dataset_dir.mkdir()

        ep_path = dataset_dir / "ep.hdf5"
        with h5py.File(ep_path, "w") as f:
            f.create_dataset("image", data=np.zeros((2, 64, 64, 3), dtype=np.uint8))

        with pytest.raises(ValueError, match="out of range"):
            _load_hdf5_sample(
                dataset_dir=dataset_dir,
                dataset_name="droid_100",
                episode_id=5,
                step_id=0,
                image_key_candidates=["image"],
                action_key_candidates=[],
                instruction_key_candidates=[],
            )

    def test_no_image_key_raises(self, tmp_path):
        import h5py

        dataset_dir = tmp_path / "noimage_test"
        dataset_dir.mkdir()

        ep_path = dataset_dir / "ep.hdf5"
        with h5py.File(ep_path, "w") as f:
            f.create_dataset("something_else", data=np.zeros(10))

        with pytest.raises(KeyError, match="No image key"):
            _load_hdf5_sample(
                dataset_dir=dataset_dir,
                dataset_name="droid_100",
                episode_id=0,
                step_id=0,
                image_key_candidates=["image", "rgb"],
                action_key_candidates=[],
                instruction_key_candidates=[],
            )
