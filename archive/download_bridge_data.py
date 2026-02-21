"""Download BridgeData V2 episodes and save as PNG images + metadata JSON.

Usage:
    python download_bridge_data.py [--num_episodes 10]

This script can run on a CPU-only machine (no GPU required).
It downloads episodes from the BridgeData V2 dataset via tensorflow_datasets,
saves each step's image as a PNG, and collects all metadata (instructions,
ground-truth actions) into a single metadata.json file.
"""

import argparse
import json

import numpy as np
import tensorflow_datasets as tfds
from PIL import Image
from tqdm import tqdm

import config


def download_bridge_data(num_episodes: int = config.NUM_EPISODES) -> None:
    """Download BridgeData V2 and save images + metadata."""

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load dataset from local tfrecord files ────────────────────────────
    print(f"Loading BridgeData V2 ({num_episodes} episodes)...")
    builder_dir = config.TFDS_DATA_DIR / config.BRIDGE_DATASET_NAME / "1.0.0"
    builder = tfds.builder_from_directory(str(builder_dir))
    ds = builder.as_dataset(
        split=f"train[:{num_episodes}]",
        shuffle_files=False,
    )

    metadata = {"episodes": []}

    for ep_idx, episode in enumerate(tqdm(ds, total=num_episodes, desc="Episodes")):
        ep_dir = config.DATA_DIR / f"episode_{ep_idx:03d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        ep_meta = {
            "episode_id": ep_idx,
            "steps": [],
        }

        steps = episode["steps"]
        for step_idx, step in enumerate(steps):
            # ── Extract image ─────────────────────────────────────────────
            # BridgeData V2 uses image_0 as primary camera
            image_array = step["observation"]["image_0"].numpy()  # (H, W, 3) uint8
            image = Image.fromarray(image_array)
            image_path = ep_dir / f"step_{step_idx:03d}.png"
            image.save(image_path)

            # ── Extract action (7-dim) ────────────────────────────────────
            action = step["action"].numpy().tolist()

            # ── Extract instruction ───────────────────────────────────────
            instruction_raw = step["language_instruction"].numpy()
            if isinstance(instruction_raw, bytes):
                instruction = instruction_raw.decode("utf-8")
            else:
                instruction = str(instruction_raw)

            # ── Extract additional fields if available ────────────────────
            step_meta = {
                "step_id": step_idx,
                "image_path": str(image_path.relative_to(config.PROJECT_ROOT)),
                "action": action,
                "instruction": instruction,
            }

            # discount / reward / is_terminal if present
            for key in ["discount", "reward", "is_terminal", "is_first", "is_last"]:
                if key in step:
                    val = step[key].numpy()
                    step_meta[key] = val.item() if hasattr(val, "item") else float(val)

            ep_meta["steps"].append(step_meta)

        metadata["episodes"].append(ep_meta)
        print(f"  Episode {ep_idx:03d}: {len(ep_meta['steps'])} steps saved")

    # ── Save metadata ─────────────────────────────────────────────────────
    with open(config.METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    total_steps = sum(len(ep["steps"]) for ep in metadata["episodes"])
    print(f"\nDone! {num_episodes} episodes, {total_steps} total steps.")
    print(f"Images: {config.DATA_DIR}/episode_XXX/step_XXX.png")
    print(f"Metadata: {config.METADATA_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Download BridgeData V2 episodes")
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=config.NUM_EPISODES,
        help=f"Number of episodes to download (default: {config.NUM_EPISODES})",
    )
    args = parser.parse_args()
    download_bridge_data(num_episodes=args.num_episodes)


if __name__ == "__main__":
    main()
