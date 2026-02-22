"""Registry of robot manipulation datasets for cross-model attention analysis.

Each entry defines how to download, load, and sample from the dataset.
We only need 1 demo (episode) per dataset for attention analysis.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

import config

DATASET_CACHE = Path("/ceph_data/kana5123/cross_model_datasets")
DATA_CACHE_DIR = Path("/ceph_data/kana5123/bridge_data_cache")


@dataclass
class DatasetConfig:
    """Configuration for a single robot dataset."""
    name: str
    display_name: str
    download_cmd: str               # Shell command to download
    download_size: str              # Approximate download size
    format: str                     # "tfrecord", "hdf5", "lerobot", "custom"
    default_instruction: str        # Fallback instruction if not in data
    image_key: str = "image"        # Key for image in data dict
    action_key: str = "action"      # Key for action in data dict
    instruction_key: str = ""       # Key for instruction (empty = use default)
    notes: str = ""


@dataclass
class DatasetSample:
    """A single demo sample from a dataset."""
    dataset_name: str
    episode_id: int
    step_id: int
    image: Image.Image              # PIL RGB image
    instruction: str
    action: Optional[list[float]]   # Ground truth action (if available)


DATASETS: dict[str, DatasetConfig] = {}


def register_dataset(cfg: DatasetConfig):
    DATASETS[cfg.name] = cfg


# ── Bridge V2 (already downloaded) ──
register_dataset(DatasetConfig(
    name="bridge_v2",
    display_name="Bridge V2",
    download_cmd="# Already available at /ceph_data/kana5123/bridge_v2_data",
    download_size="~25GB (already cached)",
    format="tfrecord",
    default_instruction="pick up the object",
    notes="Primary dataset, already cached and preprocessed",
))

# ── CALVIN (debug split, small) ──
register_dataset(DatasetConfig(
    name="calvin_debug",
    display_name="CALVIN Debug",
    download_cmd=(
        "wget -P {cache_dir}/calvin "
        "http://calvin.cs.uni-freiburg.de/dataset/task_D_D_debug.zip && "
        "cd {cache_dir}/calvin && unzip task_D_D_debug.zip"
    ),
    download_size="~1.3GB",
    format="hdf5",
    default_instruction="slide the door to the right",
    notes="CALVIN benchmark debug split (small, fast download)",
))

# ── DROID (droid_100 subset) ──
register_dataset(DatasetConfig(
    name="droid_100",
    display_name="DROID-100",
    download_cmd=(
        "pip install gdown && "
        "gdown --folder https://drive.google.com/drive/folders/droid_100_id "
        "-O {cache_dir}/droid_100"
    ),
    download_size="~2GB",
    format="hdf5",
    default_instruction="pick up the object and place it in the bin",
    notes="100-episode subset of DROID",
))

# ── LeRobot (HuggingFace datasets) ──
register_dataset(DatasetConfig(
    name="lerobot_pusht",
    display_name="LeRobot PushT",
    download_cmd=(
        "pip install lerobot && "
        "python -c \"from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; "
        "ds = LeRobotDataset('lerobot/pusht', root='{cache_dir}/lerobot')\""
    ),
    download_size="~500MB",
    format="lerobot",
    default_instruction="push the T-shaped block to the target",
    notes="LeRobot PushT task, standard benchmark",
))

# ── RH20T (mini subset) ──
register_dataset(DatasetConfig(
    name="rh20t_mini",
    display_name="RH20T Mini",
    download_cmd=(
        "# Download from https://rh20t.github.io/ — select 1 scene\n"
        "# Manual download required, ~26GB minimum"
    ),
    download_size="~26GB",
    format="custom",
    default_instruction="grasp the object",
    notes="Real-world bimanual dataset, large",
))


def get_dataset(name: str) -> DatasetConfig:
    if name not in DATASETS:
        available = ", ".join(DATASETS.keys())
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")
    return DATASETS[name]


def list_datasets() -> list[str]:
    return list(DATASETS.keys())


def load_bridge_sample(episode_id: int = 0, step_id: int = 0) -> DatasetSample:
    """Load a sample from the already-cached Bridge V2 dataset.

    Uses the preprocessed cache at DATA_CACHE_DIR which contains:
    - cache_info.json: shape metadata for the memmap
    - metadata.pkl: list of dicts with episode_id, step_id, global_idx, instruction, action
    - images.dat: numpy memmap of shape (total_steps, H, W, 3) uint8
    """
    # 1. Load cache info for memmap shape
    cache_info_path = DATA_CACHE_DIR / "cache_info.json"
    with open(cache_info_path) as f:
        cache_info = json.load(f)

    total_steps = cache_info["total_steps"]
    img_h = cache_info["image_height"]
    img_w = cache_info["image_width"]

    # 2. Load metadata (list of dicts)
    metadata_path = DATA_CACHE_DIR / "metadata.pkl"
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    # 3. Find matching entry by episode_id + step_id
    match = None
    for entry in metadata:
        if entry["episode_id"] == episode_id and entry["step_id"] == step_id:
            match = entry
            break

    if match is None:
        raise ValueError(
            f"No sample found for episode_id={episode_id}, step_id={step_id} "
            f"in Bridge V2 cache"
        )

    global_idx = match["global_idx"]

    # 4. Load image from memmap by global_idx
    images_mmap = np.memmap(
        DATA_CACHE_DIR / "images.dat",
        dtype=np.uint8,
        mode="r",
        shape=(total_steps, img_h, img_w, 3),
    )
    image_array = np.array(images_mmap[global_idx])  # copy out of memmap
    image = Image.fromarray(image_array, mode="RGB")

    # 5. Return DatasetSample
    return DatasetSample(
        dataset_name="bridge_v2",
        episode_id=episode_id,
        step_id=step_id,
        image=image,
        instruction=match["instruction"],
        action=match.get("action"),
    )


def load_calvin_sample(episode_id: int = 0, step_id: int = 0) -> DatasetSample:
    """Load a sample from the CALVIN debug dataset.

    CALVIN stores episodes as HDF5 files in DATASET_CACHE / "calvin".
    Each episode file contains:
    - rgb_static: (T, H, W, 3) uint8 images from the static camera
    - actions: (T, 7) float64 robot actions
    - Optionally: language annotations for the episode

    Common HDF5 structures:
    - episode_{id}.hdf5 with datasets 'rgb_static', 'actions', 'lang'
    - Or npz-based: episode_{id}.npz
    """
    import h5py

    calvin_dir = DATASET_CACHE / "calvin"
    if not calvin_dir.exists():
        raise FileNotFoundError(
            f"CALVIN data not found at {calvin_dir}. "
            f"Run download_dataset('calvin_debug') first."
        )

    cfg = DATASETS["calvin_debug"]

    # Search for episode files (HDF5 or npz)
    episode_files = sorted(calvin_dir.glob("*.hdf5")) + sorted(calvin_dir.glob("*.h5"))
    if not episode_files:
        # CALVIN debug often stores data as episode_XXXXXXX.npz
        episode_files = sorted(calvin_dir.glob("*.npz"))

    if not episode_files:
        # Try subdirectory structure (task_D_D/training/)
        for sub in calvin_dir.rglob("*.hdf5"):
            episode_files.append(sub)
        episode_files = sorted(episode_files)

    if not episode_files:
        raise FileNotFoundError(
            f"No episode files found in {calvin_dir} or subdirectories"
        )

    if episode_id >= len(episode_files):
        raise ValueError(
            f"episode_id={episode_id} out of range, "
            f"only {len(episode_files)} episodes available"
        )

    ep_path = episode_files[episode_id]

    if ep_path.suffix in (".hdf5", ".h5"):
        with h5py.File(ep_path, "r") as f:
            # Search for image key
            image_key = None
            for k in ["rgb_static", "rgb_obs", "image", "obs/rgb_static",
                       "observations/images/rgb_static"]:
                if k in f:
                    image_key = k
                    break
            if image_key is None:
                raise KeyError(
                    f"No known image key found in {ep_path}. "
                    f"Available keys: {list(f.keys())}"
                )

            images = f[image_key]
            if step_id >= len(images):
                raise ValueError(
                    f"step_id={step_id} out of range, "
                    f"episode has {len(images)} steps"
                )
            image_array = np.array(images[step_id])
            image = Image.fromarray(image_array, mode="RGB")

            # Search for action key
            action = None
            for k in ["actions", "action", "rel_actions"]:
                if k in f:
                    action = f[k][step_id].tolist()
                    break

            # Search for language annotation
            instruction = cfg.default_instruction
            for k in ["lang", "language", "instruction", "lang_ann"]:
                if k in f:
                    lang_val = f[k]
                    if hasattr(lang_val, "shape") and len(lang_val.shape) == 0:
                        # scalar string dataset
                        instruction = lang_val[()].decode("utf-8") if isinstance(
                            lang_val[()], bytes) else str(lang_val[()])
                    elif hasattr(lang_val, "__len__") and len(lang_val) > 0:
                        val = lang_val[0]
                        instruction = val.decode("utf-8") if isinstance(
                            val, bytes) else str(val)
                    break
    else:
        # NPZ format
        data = np.load(ep_path, allow_pickle=True)
        # Search for image key
        image_key = None
        for k in ["rgb_static", "rgb_obs", "image", "obs"]:
            if k in data:
                image_key = k
                break
        if image_key is None:
            raise KeyError(
                f"No known image key in {ep_path}. Keys: {list(data.keys())}"
            )
        images = data[image_key]
        if step_id >= len(images):
            raise ValueError(
                f"step_id={step_id} out of range, episode has {len(images)} steps"
            )
        image = Image.fromarray(images[step_id], mode="RGB")

        action = None
        for k in ["actions", "action", "rel_actions"]:
            if k in data:
                action = data[k][step_id].tolist()
                break

        instruction = cfg.default_instruction

    return DatasetSample(
        dataset_name="calvin_debug",
        episode_id=episode_id,
        step_id=step_id,
        image=image,
        instruction=instruction,
        action=action,
    )


def load_lerobot_sample(episode_id: int = 0, step_id: int = 0) -> DatasetSample:
    """Load a sample from the LeRobot PushT dataset via HuggingFace datasets.

    PushT is a 2D pushing task with 2D actions (x, y).
    Actions are padded to 7 dimensions for consistency with other datasets.
    Images are rendered observations of the push environment.
    """
    from datasets import load_dataset

    lerobot_cache = DATASET_CACHE / "lerobot"
    lerobot_cache.mkdir(parents=True, exist_ok=True)

    cfg = DATASETS["lerobot_pusht"]

    # Load via HuggingFace datasets with caching
    ds = load_dataset(
        "lerobot/pusht",
        split="train",
        cache_dir=str(lerobot_cache),
    )

    # Filter to requested episode
    ep_rows = [i for i, row in enumerate(ds) if row.get("episode_index", row.get("episode_id", -1)) == episode_id]
    if not ep_rows:
        # Fallback: treat episode_id as direct index offset
        # PushT episodes are contiguous; try to find by sequential grouping
        ep_rows = list(range(len(ds)))

    if step_id >= len(ep_rows):
        raise ValueError(
            f"step_id={step_id} out of range for episode {episode_id} "
            f"({len(ep_rows)} steps available)"
        )

    row_idx = ep_rows[step_id]
    row = ds[row_idx]

    # Extract image — PushT stores images under various keys
    image = None
    for key in ["observation.image", "image", "observation", "obs"]:
        if key in row:
            val = row[key]
            if isinstance(val, Image.Image):
                image = val.convert("RGB")
            elif isinstance(val, np.ndarray):
                image = Image.fromarray(val, mode="RGB")
            elif isinstance(val, dict) and "bytes" in val:
                import io
                image = Image.open(io.BytesIO(val["bytes"])).convert("RGB")
            break

    if image is None:
        raise KeyError(f"No image found in PushT row. Available keys: {list(row.keys())}")

    # Extract action — PushT has 2D actions, pad to 7
    action = None
    for key in ["action", "actions"]:
        if key in row:
            raw_action = row[key]
            if isinstance(raw_action, (list, np.ndarray)):
                raw_action = list(map(float, raw_action))
            else:
                raw_action = [float(raw_action)]
            # Pad to 7 dimensions
            action = raw_action + [0.0] * (7 - len(raw_action))
            break

    return DatasetSample(
        dataset_name="lerobot_pusht",
        episode_id=episode_id,
        step_id=step_id,
        image=image,
        instruction=cfg.default_instruction,
        action=action,
    )


def download_dataset(name: str) -> Path:
    """Download a dataset to the cache directory."""
    cfg = get_dataset(name)
    cache_dir = DATASET_CACHE / name
    cache_dir.mkdir(parents=True, exist_ok=True)

    cmd = cfg.download_cmd.format(cache_dir=str(cache_dir))
    print(f"Downloading {cfg.display_name}...")
    print(f"  Command: {cmd}")
    print(f"  Size: {cfg.download_size}")

    os.system(cmd)
    return cache_dir


def load_sample(dataset_name: str, episode_id: int = 0, step_id: int = 0) -> DatasetSample:
    """Universal dispatcher: load a sample from any registered dataset.

    Routes to the correct loader based on dataset_name.
    """
    if dataset_name not in DATASETS:
        available = ", ".join(DATASETS.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")

    loaders = {
        "bridge_v2": load_bridge_sample,
        "calvin_debug": load_calvin_sample,
        "lerobot_pusht": load_lerobot_sample,
        "droid_100": load_droid_sample,
        "rh20t_mini": load_rh20t_sample,
    }

    loader = loaders.get(dataset_name)
    if loader is None:
        raise NotImplementedError(
            f"No loader implemented for dataset '{dataset_name}'"
        )

    return loader(episode_id, step_id)
