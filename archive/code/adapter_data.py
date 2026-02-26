"""Data pipeline for Differentiable Attention Adapter training.

Loads BridgeData V2 from tfrecord shards, caches to disk as numpy memmap
for efficient multi-process access, splits by episode, yields
(image, instruction, gt_action) batches for adapter training.

Cache structure:
    DATA_CACHE_DIR/
        images.dat       — numpy memmap (N, 256, 256, 3) uint8
        metadata.pkl     — list of dicts {instruction, action, episode_id, step_id, global_idx}
        cache_info.json  — {total_steps, image_height, image_width, total_episodes}
        DONE             — sentinel file

Usage:
    from adapter_data import create_dataloaders
    train_loader, val_loader, test_loader = create_dataloaders()
"""

from __future__ import annotations

import json
import pickle
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import config


def resize_mask(mask: np.ndarray, target_tokens: int, source_grid: int = 16) -> np.ndarray:
    """Resize a patch mask from source grid to target number of tokens.

    Used when SAM masks were generated for one vision token count (e.g., 256
    for OpenVLA) but need to be used with a model that has a different count
    (e.g., 313 for TraceVLA). Uses nearest-neighbor interpolation on the 2D
    grid representation.

    Args:
        mask: (V_source,) uint8 binary mask
        target_tokens: desired output length
        source_grid: grid size of the source mask (e.g., 16 for 16x16=256)

    Returns:
        (target_tokens,) uint8 binary mask
    """
    if len(mask) == target_tokens:
        return mask

    # Reshape to 2D grid
    grid_2d = mask[:source_grid * source_grid].reshape(source_grid, source_grid).astype(np.float32)

    # Determine target grid (approximate square root)
    target_grid = int(np.ceil(np.sqrt(target_tokens)))

    # Resize using PIL (nearest-neighbor to keep binary)
    from PIL import Image as _PILImage
    grid_img = _PILImage.fromarray((grid_2d * 255).astype(np.uint8), mode="L")
    resized = grid_img.resize((target_grid, target_grid), _PILImage.NEAREST)
    resized_arr = (np.array(resized) > 127).astype(np.uint8).flatten()

    # Truncate or pad to exact target length
    if len(resized_arr) >= target_tokens:
        return resized_arr[:target_tokens]
    else:
        return np.concatenate([resized_arr, np.zeros(target_tokens - len(resized_arr), dtype=np.uint8)])


# ═══════════════════════════════════════════════════════════════════════════
# Action tokenization (ground truth → target token IDs)
# ═══════════════════════════════════════════════════════════════════════════

class ActionTokenizer:
    """Convert continuous 7-dim actions to/from discrete token IDs.

    Mirrors OpenVLA's tokenization:
      1. unnormalized action → normalized [-1, 1] using q01/q99 stats
      2. normalized → bin index via 256-bin uniform discretization
      3. bin index → token ID via (vocab_size - 1 - bin_index)
    """

    def __init__(self, model, model_cfg=None):
        self.n_bins = getattr(model.config, "n_action_bins", 256)
        pad = getattr(model.config, "pad_to_multiple_of", 0)

        # Resolve vocab_size across different model architectures
        cfg = model.config
        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "vocab_size"):
            self.vocab_size = cfg.text_config.vocab_size - pad
        elif hasattr(cfg, "vocab_size"):
            self.vocab_size = cfg.vocab_size - pad
        else:
            raise RuntimeError(
                f"Cannot determine vocab_size from model config: {type(cfg).__name__}"
            )

        # Bin edges and centers
        edges = np.linspace(-1, 1, self.n_bins + 1)
        self.bin_centers = (edges[:-1] + edges[1:]) / 2.0  # (256,)

        # Load normalization stats
        norm_stats = (
            getattr(model, "norm_stats", None)
            or getattr(model.config, "norm_stats", None)
        )
        if norm_stats and config.BRIDGE_UNNORM_KEY in norm_stats:
            stats = norm_stats[config.BRIDGE_UNNORM_KEY]["action"]
            self.q01 = np.array(stats["q01"], dtype=np.float64)
            self.q99 = np.array(stats["q99"], dtype=np.float64)
            self.mask = np.array(stats.get("mask", [True] * 7))
        else:
            # Fallback: use Bridge V2 dataset statistics from OpenVLA
            # These are dataset-specific, not model-specific
            print(f"  WARNING: No norm_stats for '{config.BRIDGE_UNNORM_KEY}', using Bridge V2 defaults")
            self.q01 = np.array([
                -0.02872725307941437, -0.04170349963009357, -0.026093858778476715,
                -0.08092105075716972, -0.09288699507713317, -0.20718276381492615, 0.0,
            ], dtype=np.float64)
            self.q99 = np.array([
                0.028309678435325586, 0.040855254605412394, 0.040161586627364146,
                0.08192047759890528, 0.07792850524187081, 0.20382574498653397, 1.0,
            ], dtype=np.float64)
            self.mask = np.array([True, True, True, True, True, True, False])

    def action_to_token_ids(self, action: np.ndarray) -> list[int]:
        """Unnormalized 7-dim action → 7 token IDs."""
        action = np.asarray(action, dtype=np.float64)
        # Normalize to [-1, 1]
        normalized = np.where(
            self.mask,
            2.0 * (action - self.q01) / (self.q99 - self.q01 + 1e-8) - 1.0,
            action,
        )
        normalized = np.clip(normalized, -1.0, 1.0)

        # Discretize to bin indices
        bin_indices = np.digitize(normalized, self.bin_centers) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)

        # Convert to token IDs
        token_ids = (self.vocab_size - 1 - bin_indices).tolist()
        return token_ids

    def token_ids_to_bin_centers_tensor(self, device="cpu") -> torch.Tensor:
        """Return bin centers as a tensor for soft-expected-value loss."""
        return torch.tensor(self.bin_centers, dtype=torch.float32, device=device)


# ═══════════════════════════════════════════════════════════════════════════
# Disk cache builder (one-time preprocessing)                                
# ═══════════════════════════════════════════════════════════════════════════

def build_data_cache(
    tfrecord_dir: Path,
    cache_dir: Path,
    num_episodes: Optional[int] = None,
) -> None:
    """Extract tfrecords → numpy memmap + metadata pickle (single pass).

    This is RAM-efficient: images are written directly to the memmap file
    without accumulating in memory.

    Args:
        tfrecord_dir: Path to BridgeData V2 tfrecord directory.
        cache_dir: Where to save the cache files.
        num_episodes: If set, only process this many episodes.
    """
    import tensorflow as tf
    import tensorflow_datasets as tfds

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Building data cache at {cache_dir} ...")

    builder = tfds.builder_from_directory(str(tfrecord_dir))
    split_str = "train" if num_episodes is None else f"train[:{num_episodes}]"
    ds = builder.as_dataset(split=split_str, shuffle_files=False)

    # Pass 1: count steps and collect metadata (no images in RAM)
    print("  Pass 1: Counting steps and collecting metadata ...")
    metadata = []
    ep_step_counter: dict[int, int] = {}
    global_idx = 0
    total_episodes = 0
    img_h, img_w = None, None
    t0 = time.time()

    try:
        for ep_idx, episode in enumerate(ds):
            total_episodes = ep_idx + 1
            for step in episode["steps"]:
                instruction = step["language_instruction"].numpy()
                if isinstance(instruction, bytes):
                    instruction = instruction.decode("utf-8")
                if not instruction.strip():
                    continue

                if img_h is None:
                    shape = step["observation"]["image_0"].numpy().shape
                    img_h, img_w = shape[0], shape[1]

                step_id = ep_step_counter.get(ep_idx, 0)
                ep_step_counter[ep_idx] = step_id + 1

                metadata.append({
                    "global_idx": global_idx,
                    "instruction": instruction,
                    "action": step["action"].numpy().tolist(),
                    "episode_id": ep_idx,
                    "step_id": step_id,
                })
                global_idx += 1

            if (ep_idx + 1) % 5000 == 0:
                elapsed = time.time() - t0
                print(f"    {ep_idx + 1} episodes, {global_idx} steps ({elapsed:.0f}s)")
    except tf.errors.FailedPreconditionError:
        # BridgeData V2 metadata declares 53191 episodes but shards contain 53186.
        # All actual episodes are already collected; ignore the count mismatch.
        print(f"  [WARN] dataset metadata count mismatch — using {total_episodes} episodes collected")

    total_steps = global_idx
    elapsed = time.time() - t0
    print(f"  Pass 1 done: {total_steps} steps from {total_episodes} episodes ({elapsed:.0f}s)")

    if total_steps == 0:
        raise RuntimeError("No valid steps found in dataset!")

    # Save metadata
    meta_path = cache_dir / "metadata.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Metadata saved: {meta_path} ({len(metadata)} entries)")

    # Save cache info
    info_path = cache_dir / "cache_info.json"
    with open(info_path, "w") as f:
        json.dump({
            "total_steps": total_steps,
            "image_height": int(img_h),
            "image_width": int(img_w),
            "total_episodes": total_episodes,
        }, f, indent=2)

    # Pass 2: write images to memmap (sequential, low RAM)
    print(f"  Pass 2: Writing {total_steps} images to memmap ...")
    images_path = cache_dir / "images.dat"
    mmap = np.memmap(
        str(images_path), dtype=np.uint8, mode="w+",
        shape=(total_steps, img_h, img_w, 3),
    )

    ds2 = builder.as_dataset(split=split_str, shuffle_files=False)
    idx = 0
    t1 = time.time()

    try:
        for ep_idx, episode in enumerate(ds2):
            for step in episode["steps"]:
                instruction = step["language_instruction"].numpy()
                if isinstance(instruction, bytes):
                    instruction = instruction.decode("utf-8")
                if not instruction.strip():
                    continue

                if idx >= total_steps:
                    break  # guard against Pass 1/2 count divergence
                mmap[idx] = step["observation"]["image_0"].numpy()
                idx += 1

            if (ep_idx + 1) % 5000 == 0:
                mmap.flush()
                elapsed = time.time() - t1
                print(f"    {ep_idx + 1} episodes, {idx}/{total_steps} images ({elapsed:.0f}s)")
    except tf.errors.FailedPreconditionError:
        print(f"  [WARN] Pass 2 metadata count mismatch — {idx} images written")

    mmap.flush()
    del mmap
    elapsed = time.time() - t1
    print(f"  Pass 2 done: {idx} images written ({elapsed:.0f}s)")

    # Sentinel file
    (cache_dir / "DONE").touch()
    total_bytes = total_steps * img_h * img_w * 3
    print(f"  Cache complete: {total_bytes / 1e9:.1f} GB")


# ═══════════════════════════════════════════════════════════════════════════
# Dataset — reads from disk cache (memmap + metadata)
# ═══════════════════════════════════════════════════════════════════════════

class BridgeStepDataset(Dataset):
    """Dataset of individual (image, instruction, gt_action) steps.

    Each item is one timestep from one episode. Episodes are pre-split
    into train/val/test by episode index.
    """

    def __init__(
        self,
        episode_indices: list[int],
        metadata: dict,
        project_root: Path,
    ):
        self.steps = []
        episodes = metadata["episodes"]
        for ep in episodes:
            if ep["episode_id"] not in episode_indices:
                continue
            for step in ep["steps"]:
                instruction = step.get("instruction", "")
                if not instruction:
                    continue
                self.steps.append({
                    "image_path": project_root / step["image_path"],
                    "instruction": instruction,
                    "action": step["action"],
                    "episode_id": ep["episode_id"],
                    "step_id": step["step_id"],
                })

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, idx):
        item = self.steps[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        return {
            "image": image,
            "instruction": item["instruction"],
            "action": np.array(item["action"], dtype=np.float64),
            "episode_id": item["episode_id"],
            "step_id": item["step_id"],
            "global_step_id": idx,
        }


class BridgeTfrecordDataset(Dataset):
    """Dataset that reads from the disk cache (memmap images + pickle metadata).

    On first use, builds the cache from tfrecords. Subsequent uses load
    from the cache, sharing the memmap across processes (multi-GPU safe).
    """

    def __init__(
        self,
        cache_dir: Path,
        episode_indices: list[int],
        split: str = "train",
        use_object_masks: bool = False,
    ):
        self.split = split
        self.episode_indices = set(episode_indices)
        self.object_masks_mmap = None

        # Load cache info
        with open(cache_dir / "cache_info.json") as f:
            info = json.load(f)
        total_steps = info["total_steps"]
        img_h = info["image_height"]
        img_w = info["image_width"]

        # Load metadata and filter by episode
        with open(cache_dir / "metadata.pkl", "rb") as f:
            all_metadata = pickle.load(f)

        self.steps = [m for m in all_metadata if m["episode_id"] in self.episode_indices]

        # Optionally load object masks and filter out SAM-failed steps
        if use_object_masks:
            masks_path = cache_dir / config.SAM_MASKS_FILENAME
            if masks_path.exists():
                # Infer vision_tokens from file size
                file_bytes = masks_path.stat().st_size
                vision_tokens = file_bytes // total_steps
                self.object_masks_mmap = np.memmap(
                    str(masks_path), dtype=np.uint8, mode="r",
                    shape=(total_steps, vision_tokens),
                )
                # Filter out steps where mask[0] == SAM_FAILURE_MARKER
                self.steps = [
                    m for m in self.steps
                    if self.object_masks_mmap[m["global_idx"]][0] != config.SAM_FAILURE_MARKER
                ]

        # Open memmap (lazy, OS handles page caching — shared across processes)
        self.images_mmap = np.memmap(
            str(cache_dir / "images.dat"), dtype=np.uint8, mode="r",
            shape=(total_steps, img_h, img_w, 3),
        )

        n_eps = len(set(m["episode_id"] for m in self.steps))
        print(f"  [{self.split}] {len(self.steps)} steps from {n_eps} episodes (cached)")

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, idx):
        item = self.steps[idx]
        image_array = np.array(self.images_mmap[item["global_idx"]])  # copy from memmap
        image = Image.fromarray(image_array)
        result = {
            "image": image,
            "instruction": item["instruction"],
            "action": np.array(item["action"], dtype=np.float64),
            "episode_id": item["episode_id"],
            "step_id": item["step_id"],
            "global_step_id": item["global_idx"],
        }
        if self.object_masks_mmap is not None:
            result["object_mask"] = np.array(self.object_masks_mmap[item["global_idx"]])
        return result


# ═══════════════════════════════════════════════════════════════════════════
# Episode splitting
# ═══════════════════════════════════════════════════════════════════════════

def split_episodes(
    total_episodes: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Split episode indices into train/val/test by episode (not step)."""
    indices = list(range(total_episodes))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_train = int(total_episodes * train_ratio)
    n_val = int(total_episodes * val_ratio)

    train_ids = sorted(indices[:n_train])
    val_ids = sorted(indices[n_train : n_train + n_val])
    test_ids = sorted(indices[n_train + n_val :])

    return train_ids, val_ids, test_ids


# ═══════════════════════════════════════════════════════════════════════════
# Collate function (handles variable-size PIL images)
# ═══════════════════════════════════════════════════════════════════════════

def adapter_collate_fn(batch: list[dict]) -> dict:
    """Custom collate: keep images as list, stack actions."""
    result = {
        "images": [item["image"] for item in batch],
        "instructions": [item["instruction"] for item in batch],
        "actions": np.stack([item["action"] for item in batch]),  # (B, 7)
        "episode_ids": [item["episode_id"] for item in batch],
        "step_ids": [item["step_id"] for item in batch],
        "global_step_ids": [item.get("global_step_id", idx) for idx, item in enumerate(batch)],
    }
    if "object_mask" in batch[0]:
        result["object_masks"] = np.stack([item["object_mask"] for item in batch])  # (B, V)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Episode-level SAM failure filtering
# ═══════════════════════════════════════════════════════════════════════════

def compute_valid_episodes(
    cache_dir: Path,
    threshold: float = config.SAM_EPISODE_FAILURE_THRESHOLD,
) -> list[int]:
    """Compute episodes where SAM failure rate is below threshold.

    Args:
        cache_dir: Path to the data cache directory.
        threshold: Maximum allowed fraction of SAM-failed steps per episode.
            Episodes with failure_rate > threshold are excluded.

    Returns:
        Sorted list of episode IDs where failure rate <= threshold.
        If no object_masks.dat exists, returns all episode IDs (v1 fallback).
    """
    # Load metadata
    with open(cache_dir / "metadata.pkl", "rb") as f:
        all_metadata = pickle.load(f)
    with open(cache_dir / "cache_info.json") as f:
        info = json.load(f)

    all_episode_ids = sorted(set(m["episode_id"] for m in all_metadata))

    masks_path = cache_dir / config.SAM_MASKS_FILENAME
    if not masks_path.exists():
        # v1 fallback: no masks file, all episodes are valid
        return all_episode_ids

    total_steps = info["total_steps"]
    file_bytes = masks_path.stat().st_size
    vision_tokens = file_bytes // total_steps

    masks_mmap = np.memmap(
        str(masks_path), dtype=np.uint8, mode="r",
        shape=(total_steps, vision_tokens),
    )

    # Count per-episode total steps and failed steps
    ep_total: dict[int, int] = {}
    ep_failed: dict[int, int] = {}
    for m in all_metadata:
        ep_id = m["episode_id"]
        ep_total[ep_id] = ep_total.get(ep_id, 0) + 1
        if masks_mmap[m["global_idx"]][0] == config.SAM_FAILURE_MARKER:
            ep_failed[ep_id] = ep_failed.get(ep_id, 0) + 1

    valid = []
    for ep_id in all_episode_ids:
        total = ep_total.get(ep_id, 0)
        failed = ep_failed.get(ep_id, 0)
        if total == 0:
            continue
        failure_rate = failed / total
        if failure_rate <= threshold:
            valid.append(ep_id)

    return sorted(valid)


# ═══════════════════════════════════════════════════════════════════════════
# Main factory
# ═══════════════════════════════════════════════════════════════════════════

def create_dataloaders(
    num_episodes: Optional[int] = config.ADAPTER_NUM_TRAIN_EPISODES,
    batch_size: int = config.ADAPTER_BATCH_SIZE,
    num_workers: int = 0,
    source: str = "tfrecord",
    tfrecord_dir: Optional[Path] = None,
    accelerator=None,
    use_object_masks: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders.

    Args:
        num_episodes: Total number of episodes to use.
        batch_size: Steps per batch.
        num_workers: DataLoader workers.
        source: "tfrecord" (read from cache) or "metadata" (pre-extracted PNGs).
        tfrecord_dir: Path to tfrecord directory (for source="tfrecord").
        accelerator: Optional Accelerator instance for multi-GPU cache coordination.
        use_object_masks: If True, load object masks and filter SAM-failed steps/episodes.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    if source == "metadata":
        # Use pre-extracted PNGs + metadata.json
        with open(config.METADATA_PATH) as f:
            metadata = json.load(f)

        if num_episodes is None:
            num_episodes = len(metadata.get("episodes", []))
        train_ids, val_ids, test_ids = split_episodes(num_episodes)
        print(f"Episode split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

        train_ds = BridgeStepDataset(train_ids, metadata, config.PROJECT_ROOT)
        val_ds = BridgeStepDataset(val_ids, metadata, config.PROJECT_ROOT)
        test_ds = BridgeStepDataset(test_ids, metadata, config.PROJECT_ROOT)
    else:
        # Read from disk cache (build if needed)
        tfdir = tfrecord_dir or config.ADAPTER_TFRECORD_DIR
        cache_dir = config.DATA_CACHE_DIR

        if not (cache_dir / "DONE").exists():
            # Only main process builds cache
            is_main = accelerator is None or accelerator.is_main_process
            if is_main:
                print(f"Cache not found. Building from {tfdir} ...")
                build_data_cache(tfdir, cache_dir, num_episodes=num_episodes)
            if accelerator is not None:
                accelerator.wait_for_everyone()

        # Read actual episode count from cache
        with open(cache_dir / "cache_info.json") as f:
            cache_info = json.load(f)
        actual_episodes = cache_info["total_episodes"]

        # Respect num_episodes limit (cap to available episodes)
        if num_episodes is not None:
            actual_episodes = min(actual_episodes, num_episodes)

        if use_object_masks:
            # Filter to valid episodes before splitting
            valid_episodes = compute_valid_episodes(cache_dir)
            if num_episodes is not None:
                valid_episodes = valid_episodes[:min(len(valid_episodes), num_episodes)]
            train_ids, val_ids, test_ids = split_episodes(len(valid_episodes))
            # Map split indices back to actual episode IDs
            train_ids = [valid_episodes[i] for i in train_ids]
            val_ids = [valid_episodes[i] for i in val_ids]
            test_ids = [valid_episodes[i] for i in test_ids]
        else:
            train_ids, val_ids, test_ids = split_episodes(actual_episodes)

        is_main = accelerator is None or accelerator.is_main_process
        if is_main:
            print(f"Episode split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

        train_ds = BridgeTfrecordDataset(cache_dir, train_ids, split="train", use_object_masks=use_object_masks)
        val_ds = BridgeTfrecordDataset(cache_dir, val_ids, split="val", use_object_masks=use_object_masks)
        test_ds = BridgeTfrecordDataset(cache_dir, test_ids, split="test", use_object_masks=use_object_masks)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=adapter_collate_fn, num_workers=num_workers,
        drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=adapter_collate_fn, num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=adapter_collate_fn, num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
