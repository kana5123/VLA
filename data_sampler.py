"""Balanced skill sampling from BridgeData V2 cache.

Gate 1 requires 150 balanced samples (6 skills x 25 episodes).
This module indexes episodes by skill, samples without replacement,
and caches the sample list for reuse across Gate 2, 3, and Phase 3.

Cache layout (at /ceph_data/kana5123/bridge_data_cache/):
  - cache_info.json: {"total_steps": 1382356, "image_height": 256, "image_width": 256, ...}
  - metadata.pkl: list of dicts with "instruction", "episode_id", "global_idx", ...
  - images.dat: memmap (total_steps, 256, 256, 3) uint8
"""
import json
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

from contribution.signature import label_skill_from_instruction


def build_skill_episode_index(cache_dir: str | Path) -> dict[str, list[dict]]:
    """Pre-index episodes by skill label.

    Groups metadata entries by episode_id (taking the first step per episode),
    labels each episode's skill from its instruction, and returns a dict
    mapping skill -> list of episode dicts.

    Args:
        cache_dir: Path to the BridgeData V2 cache directory.

    Returns:
        {skill: [{"episode_id": int, "global_idx": int, "instruction": str}, ...]}
    """
    cache_dir = Path(cache_dir)
    with open(cache_dir / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    # Group by episode, take first step per episode
    episodes = {}
    for entry in metadata:
        ep_id = entry.get("episode_id", -1)
        if ep_id not in episodes:
            episodes[ep_id] = entry

    skill_index = defaultdict(list)
    for ep_id, entry in episodes.items():
        skill = label_skill_from_instruction(entry["instruction"])
        skill_index[skill].append({
            "episode_id": ep_id,
            "global_idx": entry["global_idx"],
            "instruction": entry["instruction"],
        })

    return dict(skill_index)


def load_balanced_samples(
    cache_dir: str | Path,
    n_per_skill: int = 25,
    target_skills: list[str] | None = None,
    seed: int = 42,
) -> list[dict]:
    """Load skill-balanced samples from cache.

    Samples n_per_skill episodes per target skill (no duplicates),
    loads their images from memmap, and returns as a flat list.

    Args:
        cache_dir: Path to BridgeData V2 cache.
        n_per_skill: Number of episodes per skill.
        target_skills: Skills to sample. None = all non-"unknown" skills.
        seed: RNG seed for reproducibility.

    Returns:
        list of dicts: {image: PIL.Image, instruction: str, skill: str,
                        episode_id: int, global_idx: int}
    """
    cache_dir = Path(cache_dir)
    skill_index = build_skill_episode_index(cache_dir)

    with open(cache_dir / "cache_info.json") as f:
        info = json.load(f)
    images_mmap = np.memmap(
        str(cache_dir / "images.dat"), dtype=np.uint8, mode="r",
        shape=(info["total_steps"], info["image_height"], info["image_width"], 3),
    )

    if target_skills is None:
        target_skills = sorted(s for s in skill_index.keys() if s != "unknown")

    rng = np.random.default_rng(seed)
    samples = []

    for skill in sorted(target_skills):
        episodes = skill_index.get(skill, [])
        n = min(n_per_skill, len(episodes))
        if n == 0:
            print(f"  WARNING: skill '{skill}' has 0 episodes, skipping")
            continue
        selected_indices = rng.choice(len(episodes), size=n, replace=False)
        for idx in selected_indices:
            ep = episodes[idx]
            from PIL import Image
            img_array = images_mmap[ep["global_idx"]]
            samples.append({
                "image": Image.fromarray(img_array),
                "instruction": ep["instruction"],
                "skill": skill,
                "episode_id": ep["episode_id"],
                "global_idx": ep["global_idx"],
            })

    rng.shuffle(samples)
    return samples


def save_sample_list(
    samples: list[dict],
    path: str | Path,
    seed: int,
    n_per_skill: int,
    target_skills: list[str],
) -> None:
    """Save sample list as JSON (without images) for reuse across gates.

    Args:
        samples: Output of load_balanced_samples().
        path: Where to save the JSON file.
        seed: RNG seed used for sampling.
        n_per_skill: Samples per skill.
        target_skills: Target skill list.
    """
    entries = []
    for s in samples:
        entries.append({
            "episode_id": s["episode_id"],
            "global_idx": s["global_idx"],
            "instruction": s["instruction"],
            "skill": s["skill"],
        })

    data = {
        "version": "gate_v1",
        "seed": seed,
        "n_per_skill": n_per_skill,
        "target_skills": target_skills,
        "samples": entries,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved sample_list: {path} ({len(entries)} samples)")


def load_sample_list(path: str | Path) -> dict:
    """Load a saved sample_list.json.

    Returns:
        dict with keys: version, seed, n_per_skill, target_skills, samples
    """
    with open(path) as f:
        return json.load(f)


def reload_samples_from_list(
    sample_list_path: str | Path,
    cache_dir: str | Path,
) -> list[dict]:
    """Reload full sample dicts (with images) from a saved sample_list.json.

    Used by Gate 2, 3 to reuse the exact same samples as Gate 1.

    Args:
        sample_list_path: Path to sample_list.json (from Gate 1).
        cache_dir: Path to BridgeData V2 cache.

    Returns:
        list of dicts matching load_balanced_samples() format.
    """
    sl = load_sample_list(sample_list_path)
    cache_dir = Path(cache_dir)

    with open(cache_dir / "cache_info.json") as f:
        info = json.load(f)
    images_mmap = np.memmap(
        str(cache_dir / "images.dat"), dtype=np.uint8, mode="r",
        shape=(info["total_steps"], info["image_height"], info["image_width"], 3),
    )

    from PIL import Image
    samples = []
    for entry in sl["samples"]:
        img_array = images_mmap[entry["global_idx"]]
        samples.append({
            "image": Image.fromarray(img_array),
            "instruction": entry["instruction"],
            "skill": entry["skill"],
            "episode_id": entry["episode_id"],
            "global_idx": entry["global_idx"],
        })

    return samples


if __name__ == "__main__":
    """Dry run: print skill distribution without loading images."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="/ceph_data/kana5123/bridge_data_cache")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    index = build_skill_episode_index(args.cache_dir)
    print(f"\nSkill distribution ({sum(len(v) for v in index.values())} episodes):")
    for skill in sorted(index.keys()):
        print(f"  {skill:12s}: {len(index[skill]):5d} episodes")
    unknown_pct = len(index.get("unknown", [])) / sum(len(v) for v in index.values()) * 100
    print(f"\n  unknown rate: {unknown_pct:.1f}%")
