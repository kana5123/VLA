"""Tests for balanced skill sampling."""
import json
import sys
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_sampler import build_skill_episode_index, load_balanced_samples, save_sample_list, load_sample_list


def test_build_skill_episode_index():
    """Index should return dict of {skill: [episode_dicts]}."""
    cache_dir = Path("/ceph_data/kana5123/bridge_data_cache")
    if not cache_dir.exists():
        print("SKIP: cache not available")
        return
    index = build_skill_episode_index(cache_dir)
    assert isinstance(index, dict)
    assert len(index) > 0
    # Should have at least some known skills
    known_skills = {"pick", "place", "move", "open", "close", "fold"}
    found = known_skills & set(index.keys())
    assert len(found) >= 4, f"Only found {found} in index"
    # Each skill should have episodes
    for skill, episodes in index.items():
        assert len(episodes) > 0, f"Skill '{skill}' has no episodes"
        assert "episode_id" in episodes[0]
        assert "global_idx" in episodes[0]
        assert "instruction" in episodes[0]
    print(f"Skills found: {sorted(index.keys())}")
    print(f"Counts: {[(s, len(e)) for s, e in sorted(index.items())]}")


def test_load_balanced_samples():
    """Should return exactly n_per_skill × len(target_skills) samples."""
    cache_dir = Path("/ceph_data/kana5123/bridge_data_cache")
    if not cache_dir.exists():
        print("SKIP: cache not available")
        return
    target_skills = ["place", "move", "pick", "fold", "open", "close"]
    samples = load_balanced_samples(cache_dir, n_per_skill=5, target_skills=target_skills, seed=42)
    # Check total count
    assert len(samples) == 30, f"Expected 30, got {len(samples)}"
    # Check per-skill balance
    skill_counts = Counter(s["skill"] for s in samples)
    for skill in target_skills:
        assert skill_counts[skill] == 5, f"Expected 5 for {skill}, got {skill_counts[skill]}"
    # Check required keys
    for s in samples:
        assert "image" in s  # PIL Image
        assert "instruction" in s
        assert "skill" in s
        assert "episode_id" in s
    print(f"Balanced samples: {dict(skill_counts)}")


def test_sample_list_roundtrip(tmp_path):
    """Save and load sample_list.json should preserve all metadata."""
    cache_dir = Path("/ceph_data/kana5123/bridge_data_cache")
    if not cache_dir.exists():
        print("SKIP: cache not available")
        return
    target_skills = ["pick", "place"]
    samples = load_balanced_samples(cache_dir, n_per_skill=3, target_skills=target_skills, seed=42)
    path = tmp_path / "sample_list.json"
    save_sample_list(samples, path, seed=42, n_per_skill=3, target_skills=target_skills)
    loaded = load_sample_list(path)
    assert loaded["seed"] == 42
    assert loaded["n_per_skill"] == 3
    assert len(loaded["samples"]) == 6
    for s in loaded["samples"]:
        assert "episode_id" in s
        assert "global_idx" in s
        assert "instruction" in s
        assert "skill" in s


def test_reproducibility():
    """Same seed should produce same samples."""
    cache_dir = Path("/ceph_data/kana5123/bridge_data_cache")
    if not cache_dir.exists():
        print("SKIP: cache not available")
        return
    s1 = load_balanced_samples(cache_dir, n_per_skill=3, target_skills=["pick", "place"], seed=42)
    s2 = load_balanced_samples(cache_dir, n_per_skill=3, target_skills=["pick", "place"], seed=42)
    ids1 = [s["episode_id"] for s in s1]
    ids2 = [s["episode_id"] for s in s2]
    assert ids1 == ids2, "Same seed should produce same episode IDs"


if __name__ == "__main__":
    test_build_skill_episode_index()
    test_load_balanced_samples()
    test_reproducibility()
    print("All tests passed!")
