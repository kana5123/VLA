"""Registry of robot manipulation datasets for cross-model attention analysis.

Each entry defines how to download, load, and sample from the dataset.
We only need 1 demo (episode) per dataset for attention analysis.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

import config

DATASET_CACHE = Path("/ceph_data/kana5123/cross_model_datasets")


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
    """Load a sample from the already-cached Bridge V2 dataset."""
    meta_path = config.METADATA_PATH
    with open(meta_path) as f:
        metadata = json.load(f)

    ep = metadata["episodes"][episode_id]
    step = ep["steps"][step_id]
    image_path = config.PROJECT_ROOT / step["image_path"]
    image = Image.open(image_path).convert("RGB")

    return DatasetSample(
        dataset_name="bridge_v2",
        episode_id=ep["episode_id"],
        step_id=step["step_id"],
        image=image,
        instruction=step["instruction"],
        action=step.get("action"),
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
