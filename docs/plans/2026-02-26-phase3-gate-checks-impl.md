# Phase 3 Gate Checks Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement 3 gate checks (150 balanced samples, layer-local V=0, text masking control) to validate Phase 2.5 taxonomy before Phase 3 proper.

**Architecture:** Extend existing contribution/ pipeline with new modules (data_sampler.py, contribution/text_mask.py) and CLI flags on existing scripts. All changes additive — no breaking changes to Phase 2.5 code.

**Tech Stack:** PyTorch hooks, scikit-learn, numpy memmap, existing SinkVerificationHookManager

---

## Critical References

| File | Role |
|------|------|
| `contribution/causal.py` | ValueZeroHook — needs `target_layers` param |
| `contribution/signature.py` | `label_skill_from_instruction()` — needs stemming/synonyms |
| `run_contribution_analysis.py` | Main CLI — needs `--balanced`, `--n_per_skill`, `--seed`, `--sample_list` |
| `run_causal_experiment.py` | Causal CLI — needs `--layer_mode`, `--candidates_json`, `--sample_list` |
| `extract_attention.py:64` | `detect_token_boundaries()` — returns text_start/text_end/vision_start/vision_end |
| `visualize_text_attention.py:470` | `load_samples_from_cache()` — current (non-balanced) data loader |
| `config.py:29` | `DATA_CACHE_DIR = Path("/ceph_data/kana5123/bridge_data_cache")` |
| `model_registry.py` | 4 experiment-ready models: openvla-7b(32L/32H), ecot-7b(32L/32H), spatialvla-4b(26L/8H), tracevla-phi3v(32L/32H) |
| `verify_attention_sinks.py:59` | `SinkVerificationHookManager` — captures attn_weights + hidden_states |
| `contribution/compute.py` | `extract_sample_contributions()`, `ContributionResult` |
| `contribution/classify.py` | `classify_layer_dual_track()` |

**Data cache:** `/ceph_data/kana5123/bridge_data_cache/` — 1,382,356 steps, 38,642 episodes, memmap images + metadata.pkl

**Model architectures (hook-critical):**
| Model | Backbone | Layers | Q Heads | KV Heads | Hidden | V proj type |
|-------|----------|--------|---------|----------|--------|-------------|
| openvla-7b | LLaMA | 32 | 32 | 32 | 4096 | Separate v_proj (MHA) |
| ecot-7b | LLaMA | 32 | 32 | 32 | 4096 | Separate v_proj (MHA) |
| spatialvla-4b | Gemma2 | 26 | 8 | 4 | 2304 | Separate v_proj (GQA) |
| tracevla-phi3v | Phi3V | 32 | 32 | 32 | 3072 | Fused qkv_proj |

**Deep layer ranges:**
- 32L models (openvla, ecot, tracevla): deep = layers 22-31
- 26L model (spatialvla): deep = layers 16-25

**Design doc:** `docs/plans/2026-02-26-phase3-gate-checks-design.md`

---

## Task 1: Improve Skill Label Extraction (Stemming + Synonyms)

**Why:** Current `label_skill_from_instruction()` matches exact words only. "placed" won't match "place", "put" won't match "place". The design requires unknown rate < 5% (currently ~15%).

**Files:**
- Modify: `contribution/signature.py:20-51`
- Test: `tests/test_signature.py` (create)

**Step 1: Write the test file**

Create `tests/test_signature.py`:

```python
"""Tests for improved skill labeling with stemming + synonyms."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from contribution.signature import label_skill_from_instruction


def test_exact_match():
    assert label_skill_from_instruction("pick up the red cup") == "pick"
    assert label_skill_from_instruction("place the cup on the table") == "place"
    assert label_skill_from_instruction("open the drawer") == "open"
    assert label_skill_from_instruction("close the lid") == "close"
    assert label_skill_from_instruction("fold the towel") == "fold"
    assert label_skill_from_instruction("move the bowl to the right") == "move"


def test_stemming():
    """Past tense and -ing forms should resolve to base skill."""
    assert label_skill_from_instruction("placed the cup on the table") == "place"
    assert label_skill_from_instruction("opened the drawer") == "open"
    assert label_skill_from_instruction("moved the bowl") == "move"
    assert label_skill_from_instruction("folded the cloth") == "fold"
    assert label_skill_from_instruction("picked up the toy") == "pick"
    assert label_skill_from_instruction("closing the jar") == "close"
    assert label_skill_from_instruction("picking up the sponge") == "pick"


def test_synonyms():
    """Synonyms should map to canonical skill."""
    assert label_skill_from_instruction("put the cup down") == "place"
    assert label_skill_from_instruction("slide the plate left") == "move"
    assert label_skill_from_instruction("unfold the cloth") == "fold"


def test_unknown():
    """Instructions with no recognizable verb should return 'unknown'."""
    assert label_skill_from_instruction("do something with the robot") == "unknown"
    assert label_skill_from_instruction("") == "unknown"


if __name__ == "__main__":
    test_exact_match()
    test_stemming()
    test_synonyms()
    test_unknown()
    print("All tests passed!")
```

**Step 2: Run tests — expect failures**

Run: `cd /home/kana5123/ATLASVLA && venv/bin/python -m pytest tests/test_signature.py -v`
Expected: FAIL on `test_stemming` and `test_synonyms` (stemming not implemented)

**Step 3: Implement stemming + synonyms in `contribution/signature.py`**

Replace lines 21-51 with:

```python
SKILL_VERBS = {
    "pick": ["pick", "grab", "grasp", "take", "lift"],
    "place": ["place", "put", "set", "drop"],
    "move": ["move", "push", "slide", "drag", "sweep"],
    "open": ["open", "pull open"],
    "close": ["close", "shut"],
    "pour": ["pour", "dump"],
    "stack": ["stack"],
    "fold": ["fold", "unfold"],
    "wipe": ["wipe", "clean"],
    "turn": ["turn", "rotate", "twist"],
}

_VERB_TO_SKILL = {}
for skill, verbs in SKILL_VERBS.items():
    for v in verbs:
        _VERB_TO_SKILL[v] = skill


def _stem_word(word: str) -> str:
    """Simple suffix-based stemming for skill verbs.
    Handles: -ed, -ing, -s (placed→place, picking→pick, moves→move).
    """
    if word.endswith("ing") and len(word) > 4:
        # closing→close, moving→move, picking→pick
        base = word[:-3]
        if base + "e" in _VERB_TO_SKILL:
            return base + "e"
        if base in _VERB_TO_SKILL:
            return base
        # doubling: putting→put, grabbing→grab
        if len(base) >= 3 and base[-1] == base[-2]:
            shorter = base[:-1]
            if shorter in _VERB_TO_SKILL:
                return shorter
    if word.endswith("ed") and len(word) > 3:
        # placed→place, moved→move
        base = word[:-1]  # placed→place (just remove d)
        if base in _VERB_TO_SKILL:
            return base
        base = word[:-2]  # opened→open
        if base in _VERB_TO_SKILL:
            return base
        # doubled: grabbed→grab
        if len(base) >= 2 and base[-1] == base[-2]:
            shorter = base[:-1]
            if shorter in _VERB_TO_SKILL:
                return shorter
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        base = word[:-1]
        if base in _VERB_TO_SKILL:
            return base
    return word


def label_skill_from_instruction(instruction: str) -> str:
    """Extract primary skill verb from instruction text.

    Applies stemming (placed→place, picking→pick) and synonym matching.
    Returns skill label or "unknown".
    """
    words = instruction.lower().split()
    for word in words:
        clean = re.sub(r"[^a-z]", "", word)
        if not clean:
            continue
        # Direct match
        if clean in _VERB_TO_SKILL:
            return _VERB_TO_SKILL[clean]
        # Stemmed match
        stemmed = _stem_word(clean)
        if stemmed in _VERB_TO_SKILL:
            return _VERB_TO_SKILL[stemmed]
    return "unknown"
```

**Step 4: Run tests — expect pass**

Run: `cd /home/kana5123/ATLASVLA && venv/bin/python -m pytest tests/test_signature.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add tests/test_signature.py contribution/signature.py
git commit -m "feat: add stemming + synonym expansion for skill label extraction"
```

---

## Task 2: Create Balanced Skill Sampler (`data_sampler.py`)

**Why:** Phase 2.5 used 20 random samples with skewed skill distribution (move=7, pick=6, place=2, rest=1). Gate ① needs 150 balanced samples: 6 skills × 25 episodes each. The sampler also saves `sample_list.json` for reuse by Gate ②, ③, and Phase 3.

**Files:**
- Create: `data_sampler.py`
- Test: `tests/test_data_sampler.py` (create)

**Step 1: Write tests**

Create `tests/test_data_sampler.py`:

```python
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
```

**Step 2: Run tests — expect import failure**

Run: `cd /home/kana5123/ATLASVLA && venv/bin/python -m pytest tests/test_data_sampler.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data_sampler'`

**Step 3: Implement `data_sampler.py`**

Create `data_sampler.py` in project root:

```python
"""Balanced skill sampling from BridgeData V2 cache.

Gate ① requires 150 balanced samples (6 skills × 25 episodes).
This module indexes episodes by skill, samples without replacement,
and caches the sample list for reuse across Gate ②, ③, and Phase 3.

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
    mapping skill → list of episode dicts.

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

    Used by Gate ②, ③ to reuse the exact same samples as Gate ①.

    Args:
        sample_list_path: Path to sample_list.json (from Gate ①).
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
```

**Step 4: Run tests — expect pass**

Run: `cd /home/kana5123/ATLASVLA && venv/bin/python -m pytest tests/test_data_sampler.py -v`
Expected: ALL PASS (on machine with cache access)

**Step 5: Dry-run verification**

Run: `cd /home/kana5123/ATLASVLA && venv/bin/python data_sampler.py --dry_run`
Expected: Prints skill distribution with unknown < 5%

**Step 6: Commit**

```bash
git add data_sampler.py tests/test_data_sampler.py
git commit -m "feat: add balanced skill sampler with sample_list.json caching"
```

---

## Task 3: Add `--balanced` and `--sample_list` to `run_contribution_analysis.py`

**Why:** Gate ① needs to run with 150 balanced samples and save the sample list for subsequent gates.

**Files:**
- Modify: `run_contribution_analysis.py:336-350` (argparse + sample loading)

**Step 1: Add CLI flags to argparse**

In `main()` (line 337-345), add after `--output_dir`:

```python
parser.add_argument("--balanced", action="store_true",
                    help="Use skill-balanced sampling from BridgeData V2 cache")
parser.add_argument("--n_per_skill", type=int, default=25,
                    help="Samples per skill when --balanced (default: 25)")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for balanced sampling")
parser.add_argument("--sample_list", type=str, default=None,
                    help="Path to save/load sample_list.json (reuse across gates)")
parser.add_argument("--target_skills", nargs="+",
                    default=["place", "move", "pick", "fold", "open", "close"],
                    help="Target skills for balanced sampling")
```

**Step 2: Replace sample loading in `run_analysis()`**

Change the function signature to accept the new args. In `main()`, pass them through.

Replace the `run_analysis` call in `main()`:
```python
run_analysis(args.model, args.device, args.n_samples, args.top_k, output_dir,
             balanced=args.balanced, n_per_skill=args.n_per_skill,
             seed=args.seed, sample_list_path=args.sample_list,
             target_skills=args.target_skills)
```

Update `run_analysis` function signature (line 51) to accept new params:
```python
def run_analysis(model_name, device, n_samples, top_k, output_dir,
                 balanced=False, n_per_skill=25, seed=42, sample_list_path=None,
                 target_skills=None):
```

Replace sample loading block (line 62):
```python
# ── Load samples ─────────────────────────────────────────────
if sample_list_path and Path(sample_list_path).exists():
    from data_sampler import reload_samples_from_list
    samples = reload_samples_from_list(sample_list_path, config.DATA_CACHE_DIR)
    print(f"  Loaded {len(samples)} samples from {sample_list_path}")
elif balanced:
    from data_sampler import load_balanced_samples, save_sample_list
    samples = load_balanced_samples(
        config.DATA_CACHE_DIR, n_per_skill=n_per_skill,
        target_skills=target_skills, seed=seed,
    )
    print(f"  Loaded {len(samples)} balanced samples ({n_per_skill}/skill)")
    if sample_list_path:
        save_sample_list(samples, sample_list_path, seed, n_per_skill, target_skills)
else:
    samples = load_samples_from_cache(config.DATA_CACHE_DIR, n_samples=n_samples)
    print(f"  Loaded {len(samples)} samples (legacy)")
```

**Step 3: Verify CLI help**

Run: `cd /home/kana5123/ATLASVLA && venv/bin/python run_contribution_analysis.py --help`
Expected: Shows `--balanced`, `--n_per_skill`, `--seed`, `--sample_list`, `--target_skills`

**Step 4: Quick smoke test (1 sample)**

Run: `cd /home/kana5123/ATLASVLA && venv/bin/python run_contribution_analysis.py --model ecot-7b --device cuda:0 --balanced --n_per_skill 1 --seed 42 --sample_list /tmp/test_sample_list.json --output_dir /tmp/test_gate1`
Expected: Runs 6 samples (1 per skill), saves sample_list.json

**Step 5: Commit**

```bash
git add run_contribution_analysis.py
git commit -m "feat: add --balanced, --sample_list flags for Gate ① sampling"
```

---

## Task 4: Add Mode Token Extraction to Report

**Why:** Gate ① must output `mode_tokens.json` — the most frequent peak position per model across deep layers. Gate ② uses these mode tokens to determine which tokens to zero.

**Files:**
- Modify: `run_contribution_analysis.py` (add mode token extraction after aggregation loop)

**Step 1: Add mode token extraction after the per-layer aggregation**

After the `layer_dual_track_agg` computation (around line 255), add:

```python
# ── Mode token extraction (B4) ─────────────────────────────
# Find the most frequent peak position across deep layers
from collections import Counter as _Counter

def _extract_mode_token(all_dual_track, peak_key, deep_layers):
    """Extract the mode (most frequent) peak position across deep layers."""
    positions = []
    for l in deep_layers:
        for dual in all_dual_track.get(l, []):
            positions.append(dual[peak_key]["abs_t"])
    if not positions:
        return {"abs_t": -1, "freq": 0.0}
    counter = _Counter(positions)
    mode_pos, mode_count = counter.most_common(1)[0]
    total = len(deep_layers)  # freq = count per num_deep_layers
    # Find a representative entry to get token_str
    token_str = None
    for l in deep_layers:
        for dual in all_dual_track.get(l, []):
            if dual[peak_key]["abs_t"] == mode_pos:
                token_str = dual[peak_key].get("token_str")
                break
        if token_str is not None:
            break
    # freq = fraction of deep layers where this position is the peak
    # Average across all samples: for each sample, count how many layers have this peak
    per_sample_freqs = []
    n_deep = len(deep_layers)
    n_samples_seen = len(all_dual_track.get(deep_layers[0], [])) if deep_layers else 0
    for si in range(n_samples_seen):
        count = 0
        for l in deep_layers:
            duals = all_dual_track.get(l, [])
            if si < len(duals) and duals[si][peak_key]["abs_t"] == mode_pos:
                count += 1
        per_sample_freqs.append(count / n_deep if n_deep > 0 else 0)
    freq = float(np.mean(per_sample_freqs)) if per_sample_freqs else 0.0
    return {"abs_t": mode_pos, "token_str": token_str, "freq": freq}

mode_tokens = {
    "A_mode": _extract_mode_token(all_layer_dual_track, "a_peak", deep_layers),
    "C_mode": _extract_mode_token(all_layer_dual_track, "c_peak", deep_layers),
    "R_mode": _extract_mode_token(all_layer_dual_track, "r_peak", deep_layers),
}
```

**Step 2: Add mode_tokens to report and save separately**

Add to the report dict (around line 290):
```python
report["mode_tokens"] = mode_tokens
```

After saving the main report, also save mode_tokens.json:
```python
mode_path = output_dir / "mode_tokens.json"
with open(mode_path, "w") as f:
    json.dump(mode_tokens, f, indent=2)
print(f"  Mode tokens saved: {mode_path}")
```

**Step 3: Verify output**

Run: Quick test from Task 3's smoke test output.
Expected: `mode_tokens.json` exists with A_mode, C_mode, R_mode entries.

**Step 4: Commit**

```bash
git add run_contribution_analysis.py
git commit -m "feat: extract mode tokens (B4) with freq stability metric"
```

---

## Task 5: Add `target_layers` to ValueZeroHook

**Why:** Gate ② needs block-level V=0 (e.g., only layers 22-26) instead of all layers. The current ValueZeroHook always registers on all layers.

**Files:**
- Modify: `contribution/causal.py:81-140`
- Test: `tests/test_causal.py` (create)

**Step 1: Write test**

Create `tests/test_causal.py`:

```python
"""Tests for ValueZeroHook with target_layers."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from contribution.causal import ValueZeroHook, get_deep_layer_ranges


def test_get_deep_layer_ranges_32L():
    ranges = get_deep_layer_ranges(32)
    assert ranges["all"] == list(range(22, 32))
    assert ranges["block1"] == list(range(22, 27))
    assert ranges["block2"] == list(range(27, 32))


def test_get_deep_layer_ranges_26L():
    ranges = get_deep_layer_ranges(26)
    assert ranges["all"] == list(range(16, 26))
    assert ranges["block1"] == list(range(16, 21))
    assert ranges["block2"] == list(range(21, 26))


def test_vzero_target_layers_init():
    """target_layers=None should behave same as before (all layers)."""
    hook = ValueZeroHook([0, 1], target_layers=None)
    assert hook.target_layers is None

    hook2 = ValueZeroHook([0, 1], target_layers=[22, 23, 24])
    assert hook2.target_layers == [22, 23, 24]


if __name__ == "__main__":
    test_get_deep_layer_ranges_32L()
    test_get_deep_layer_ranges_26L()
    test_vzero_target_layers_init()
    print("All tests passed!")
```

**Step 2: Run tests — expect failures**

Run: `cd /home/kana5123/ATLASVLA && venv/bin/python -m pytest tests/test_causal.py -v`
Expected: FAIL (no `target_layers` param, no `get_deep_layer_ranges`)

**Step 3: Modify `ValueZeroHook` in `contribution/causal.py`**

Change `__init__` (line 88):
```python
def __init__(self, target_positions: list[int], target_layers: list[int] | None = None):
    self.target_positions = target_positions
    self.target_layers = target_layers  # None = all layers
    self._handles = []
    self._sanity_changed = False
```

Change `register()` (line 93): add layer filtering after `for layer_idx, layer in enumerate(layers):`:
```python
def register(self, model, model_cfg, get_layers_fn):
    """Register hooks to zero out V projections for target tokens.
    If target_layers is set, only hooks those specific layer indices.
    """
    layers = get_layers_fn(model, model_cfg)
    num_heads = model_cfg.num_heads
    num_kv_heads = getattr(model_cfg, 'num_kv_heads', None) or num_heads
    head_dim = model_cfg.hidden_dim // num_heads

    for layer_idx, layer in enumerate(layers):
        if self.target_layers is not None and layer_idx not in self.target_layers:
            continue
        attn = layer.self_attn
        if hasattr(attn, "v_proj"):
            handle = attn.v_proj.register_forward_hook(self._make_v_proj_hook())
            self._handles.append(handle)
        elif hasattr(attn, "qkv_proj"):
            q_dim = num_heads * head_dim
            kv_dim = num_kv_heads * head_dim
            v_start = q_dim + kv_dim
            v_end = q_dim + 2 * kv_dim
            handle = attn.qkv_proj.register_forward_hook(
                self._make_fused_qkv_hook(v_start, v_end)
            )
            self._handles.append(handle)
```

Add `get_deep_layer_ranges()` function at the end of the file:

```python
def get_deep_layer_ranges(num_layers: int) -> dict[str, list[int]]:
    """Return deep layer ranges for block-level V=0 experiments.

    For a model with N layers, deep = last 10 layers.
    block1 = first half of deep, block2 = second half.

    Args:
        num_layers: Total number of transformer layers.

    Returns:
        {"all": [...], "block1": [...], "block2": [...]}
    """
    deep_start = max(0, num_layers - 10)
    deep_end = num_layers
    mid = deep_start + (deep_end - deep_start) // 2
    return {
        "all": list(range(deep_start, deep_end)),
        "block1": list(range(deep_start, mid)),
        "block2": list(range(mid, deep_end)),
    }
```

**Step 4: Run tests — expect pass**

Run: `cd /home/kana5123/ATLASVLA && venv/bin/python -m pytest tests/test_causal.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add contribution/causal.py tests/test_causal.py
git commit -m "feat: add target_layers to ValueZeroHook + get_deep_layer_ranges"
```

---

## Task 6: Add `--layer_mode` and `--candidates_json` to `run_causal_experiment.py`

**Why:** Gate ② tests 9 conditions per model: {A_mode, C_mode, R_mode} × {all, block1, block2}. The script needs to accept mode tokens from Gate ① and apply block-level V=0.

**Files:**
- Modify: `run_causal_experiment.py:125-170` (argparse + experiment loop)

**Step 1: Add new CLI flags**

In `main()` argparse section, add:

```python
parser.add_argument("--layer_mode", choices=["all", "block1", "block2"],
                    default="all",
                    help="Layer range for V=0: all=deep22-31, block1=22-26, block2=27-31")
parser.add_argument("--candidates_json", type=str, default=None,
                    help="Path to mode_tokens.json (from Gate ①)")
parser.add_argument("--sample_list", type=str, default=None,
                    help="Path to sample_list.json (reuse Gate ① samples)")
parser.add_argument("--peak_type", choices=["A_mode", "C_mode", "R_mode"],
                    default=None,
                    help="Which mode token to target (requires --candidates_json)")
```

**Step 2: Update candidate resolution in `main()`**

Replace candidate resolution block (lines 139-164):

```python
# Get candidate positions
if args.candidates:
    candidate_positions = args.candidates
elif args.candidates_json and args.peak_type:
    # Gate ②: use mode tokens from Gate ①
    with open(args.candidates_json) as f:
        mode_tokens = json.load(f)
    peak = mode_tokens.get(args.peak_type, {})
    abs_t = peak.get("abs_t", 0)
    freq = peak.get("freq", 0)
    candidate_positions = [abs_t]
    print(f"  Gate ② mode: {args.peak_type} → abs_t={abs_t}, freq={freq:.2f}")
    if freq < 0.7:
        print(f"  WARNING: freq={freq:.2f} < 0.7 (unstable). Consider testing Top-3.")
elif args.report:
    # ... existing report-based resolution (unchanged)
```

**Step 3: Add layer_mode resolution and pass to run_causal**

After candidate resolution, before calling `run_causal`:

```python
# Resolve layer_mode → target_layers
from contribution.causal import get_deep_layer_ranges
model_cfg_temp = registry_get_model(args.model) if not args.candidates else None
if model_cfg_temp is None:
    from model_registry import get_model as _get_model
    model_cfg_temp = _get_model(args.model)
layer_ranges = get_deep_layer_ranges(model_cfg_temp.num_layers)
target_layers = layer_ranges.get(args.layer_mode, layer_ranges["all"])
print(f"  Layer mode: {args.layer_mode} → layers {target_layers}")
```

**Step 4: Update `run_causal()` to accept `target_layers` and `sample_list_path`**

Change signature:
```python
def run_causal(model_name, device, n_samples, output_dir, candidate_positions,
               target_layers=None, sample_list_path=None):
```

Replace sample loading:
```python
if sample_list_path:
    from data_sampler import reload_samples_from_list
    samples = reload_samples_from_list(sample_list_path, config.DATA_CACHE_DIR)
else:
    samples = load_samples_from_cache(config.DATA_CACHE_DIR, n_samples=n_samples)
```

Pass `target_layers` to `ValueZeroHook`:
```python
vzero = ValueZeroHook(targets, target_layers=target_layers)
```

Add `target_layers` and `layer_mode` to the results dict.

**Step 5: Verify CLI**

Run: `cd /home/kana5123/ATLASVLA && venv/bin/python run_causal_experiment.py --help`
Expected: Shows `--layer_mode`, `--candidates_json`, `--peak_type`, `--sample_list`

**Step 6: Commit**

```bash
git add run_causal_experiment.py
git commit -m "feat: add --layer_mode and --candidates_json for Gate ② block V=0"
```

---

## Task 7: Create Text Masking Module (`contribution/text_mask.py`)

**Why:** Gate ③ needs two text masking modes: Text V=0 (kill value content) and Text KV-mask (kill routing + content). This module provides both hooks.

**Files:**
- Create: `contribution/text_mask.py`
- Test: `tests/test_text_mask.py` (create)

**Step 1: Write test**

Create `tests/test_text_mask.py`:

```python
"""Tests for text masking hooks."""
import sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from contribution.text_mask import TextValueZeroHook, TextKVMaskHook


def test_text_value_zero_hook_init():
    hook = TextValueZeroHook(text_range=(256, 280))
    assert hook.text_range == (256, 280)
    assert len(hook._handles) == 0


def test_text_kv_mask_hook_init():
    hook = TextKVMaskHook(text_range=(256, 280))
    assert hook.text_range == (256, 280)


def test_text_kv_mask_modifies_attention_mask():
    """KV-mask should set text columns to -inf in attention_mask."""
    hook = TextKVMaskHook(text_range=(5, 10))
    # Simulate a 4D attention mask: (B, 1, seq, seq)
    seq = 15
    mask = torch.zeros(1, 1, seq, seq)
    modified = hook.apply_to_attention_mask(mask)
    # Text range columns should be -inf
    assert torch.isinf(modified[0, 0, 0, 5]).item()
    assert torch.isinf(modified[0, 0, 0, 9]).item()
    # Non-text columns should be 0
    assert modified[0, 0, 0, 0].item() == 0.0
    assert modified[0, 0, 0, 11].item() == 0.0


def test_normalized_vision_mask_count():
    """Vision V=0 (normalized) should mask same number of tokens as text."""
    from contribution.text_mask import sample_normalized_vision_positions
    text_range = (256, 273)  # 17 text tokens
    vision_range = (0, 256)  # 256 vision tokens
    n_text = text_range[1] - text_range[0]  # 17
    positions = sample_normalized_vision_positions(vision_range, n_text, seed=42)
    assert len(positions) == n_text
    # All positions should be within vision range
    for p in positions:
        assert vision_range[0] <= p < vision_range[1]


if __name__ == "__main__":
    test_text_value_zero_hook_init()
    test_text_kv_mask_hook_init()
    test_text_kv_mask_modifies_attention_mask()
    test_normalized_vision_mask_count()
    print("All tests passed!")
```

**Step 2: Run tests — expect import failure**

Run: `cd /home/kana5123/ATLASVLA && venv/bin/python -m pytest tests/test_text_mask.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement `contribution/text_mask.py`**

```python
"""Text masking hooks for Gate ③ leakage control.

Two modes:
1. TextValueZeroHook: Zeros V projections for text tokens.
   Q/K routing remains active — verb routing info can still flow.
2. TextKVMaskHook: Sets attention_mask[:, :, :, text_range] = -inf.
   Fully blocks text tokens from being attended to — kills routing AND content.

Usage:
    # Text V=0 (reuses ValueZeroHook with text range)
    boundaries = detect_token_boundaries(...)
    text_range = (boundaries["text_start"], boundaries["text_end"])
    hook = TextValueZeroHook(text_range)
    hook.register(model, model_cfg, get_layers_fn)

    # Text KV-mask (modify attention_mask before forward)
    hook = TextKVMaskHook(text_range)
    inputs["attention_mask"] = hook.apply_to_attention_mask(inputs["attention_mask"])
"""
import torch
import numpy as np
from contribution.causal import ValueZeroHook


class TextValueZeroHook(ValueZeroHook):
    """Zero out V projections for all text tokens.

    Inherits from ValueZeroHook — just converts a text range
    to a list of target positions.
    """

    def __init__(self, text_range: tuple[int, int], target_layers: list[int] | None = None):
        """
        Args:
            text_range: (text_start, text_end) absolute positions.
                        From detect_token_boundaries().
            target_layers: Optional layer filter (None = all layers).
        """
        self.text_range = text_range
        target_positions = list(range(text_range[0], text_range[1]))
        super().__init__(target_positions, target_layers=target_layers)


class TextKVMaskHook:
    """Block text tokens entirely via attention mask modification.

    Sets attention_mask[:, :, :, text_range] = -inf BEFORE the forward pass.
    This prevents any query from attending to text tokens — kills both
    value content AND Q/K routing information.

    Unlike TextValueZeroHook, this does NOT use forward hooks.
    Instead, modify the attention_mask tensor directly before model(**inputs).
    """

    def __init__(self, text_range: tuple[int, int]):
        """
        Args:
            text_range: (text_start, text_end) absolute positions.
        """
        self.text_range = text_range

    def apply_to_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Modify attention_mask to block text tokens.

        Args:
            attention_mask: Shape (B, 1, seq, seq) or (B, seq).
                If 2D, creates a 4D causal mask first.

        Returns:
            Modified attention_mask with text columns set to -inf.
        """
        mask = attention_mask.clone()
        ts, te = self.text_range

        if mask.dim() == 4:
            # (B, 1, seq, seq) or (B, H, seq, seq)
            mask[:, :, :, ts:te] = float("-inf")
        elif mask.dim() == 2:
            # (B, seq) — set text positions to 0 (will be expanded to -inf by model)
            mask[:, ts:te] = 0
        else:
            raise ValueError(f"Unexpected attention_mask dim: {mask.dim()}")

        return mask

    def get_masked_token_strs(self, input_ids: list[int], tokenizer) -> list[str]:
        """Return the actual token strings being masked (for report verification).

        Args:
            input_ids: Full sequence token IDs.
            tokenizer: For decoding.

        Returns:
            List of token strings in the masked range.
        """
        ts, te = self.text_range
        strs = []
        for pos in range(ts, min(te, len(input_ids))):
            try:
                s = tokenizer.decode([input_ids[pos]], skip_special_tokens=False).strip()
            except Exception:
                s = f"<id:{input_ids[pos]}>"
            strs.append(s)
        return strs


def sample_normalized_vision_positions(
    vision_range: tuple[int, int],
    n_tokens: int,
    seed: int = 42,
) -> list[int]:
    """Sample n_tokens random vision positions for normalized comparison.

    Gate ③ Condition D: masks the SAME number of vision tokens as text tokens
    to make modality comparison fair (256 vision vs ~17 text).

    Args:
        vision_range: (vision_start, vision_end) absolute positions.
        n_tokens: Number of tokens to mask (= number of text tokens).
        seed: RNG seed for reproducibility.

    Returns:
        Sorted list of absolute vision positions to mask.
    """
    vs, ve = vision_range
    n_vision = ve - vs
    rng = np.random.default_rng(seed)
    n = min(n_tokens, n_vision)
    selected = rng.choice(n_vision, size=n, replace=False)
    return sorted(int(vs + idx) for idx in selected)
```

**Step 4: Run tests — expect pass**

Run: `cd /home/kana5123/ATLASVLA && venv/bin/python -m pytest tests/test_text_mask.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add contribution/text_mask.py tests/test_text_mask.py
git commit -m "feat: add TextValueZeroHook + TextKVMaskHook for Gate ③"
```

---

## Task 8: Create Gate Check Orchestrator (`run_gate_checks.py`)

**Why:** Single entry point that runs Gate ① → extracts mode tokens → runs Gate ② + ③ in parallel → checks pass criteria → outputs go/no-go decision.

**Files:**
- Create: `run_gate_checks.py`

**Step 1: Implement `run_gate_checks.py`**

```python
#!/usr/bin/env python3
"""Gate Check Orchestrator: Gate ① → ② + ③ → go/no-go decision.

Execution flow:
  Gate ① (150 balanced, GPU 0-3, ~75min)
    ├── sample_list.json → Gate ②, ③
    ├── mode_tokens.json → Gate ②
    └── probe baselines → Gate ③

  Gate ② (layer-local V=0, GPU 0-3, ~45min)   ← independent
  Gate ③ (text masking, GPU 4-7, ~2hr)         ← independent

  Both complete → pass criteria check → go/no-go

Usage:
  python run_gate_checks.py --gate 1 --models ecot-7b openvla-7b spatialvla-4b tracevla-phi3v
  python run_gate_checks.py --gate 2 --models ecot-7b --gate1_dir outputs/phase3_gate/ecot-7b
  python run_gate_checks.py --gate 3 --models ecot-7b --gate1_dir outputs/phase3_gate/ecot-7b
  python run_gate_checks.py --check_pass --gate1_dir outputs/phase3_gate
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import config


MODELS = ["ecot-7b", "openvla-7b", "spatialvla-4b", "tracevla-phi3v"]
GPU_MAP = {"ecot-7b": "cuda:0", "openvla-7b": "cuda:1", "spatialvla-4b": "cuda:2", "tracevla-phi3v": "cuda:3"}
TARGET_SKILLS = ["place", "move", "pick", "fold", "open", "close"]


def run_gate1(models, output_base, n_per_skill=25, seed=42):
    """Gate ①: Run contribution analysis with balanced sampling."""
    procs = []
    for model in models:
        device = GPU_MAP.get(model, "cuda:0")
        out_dir = Path(output_base) / model
        sample_list = out_dir / "sample_list.json"
        cmd = [
            sys.executable, "run_contribution_analysis.py",
            "--model", model, "--device", device,
            "--balanced", "--n_per_skill", str(n_per_skill), "--seed", str(seed),
            "--sample_list", str(sample_list),
            "--target_skills", *TARGET_SKILLS,
            "--output_dir", str(out_dir),
        ]
        print(f"  Launching Gate ① for {model} on {device}")
        p = subprocess.Popen(cmd)
        procs.append((model, p))

    # Wait for all
    for model, p in procs:
        rc = p.wait()
        if rc != 0:
            print(f"  ERROR: Gate ① failed for {model} (exit code {rc})")
        else:
            print(f"  Gate ① complete for {model}")


def run_gate2(models, gate1_base, output_base):
    """Gate ②: Layer-local V=0 for each model × peak × block."""
    procs = []
    for model in models:
        device = GPU_MAP.get(model, "cuda:0")
        gate1_dir = Path(gate1_base) / model
        mode_tokens_path = gate1_dir / "mode_tokens.json"
        sample_list_path = gate1_dir / "sample_list.json"

        if not mode_tokens_path.exists():
            print(f"  SKIP Gate ② for {model}: no mode_tokens.json")
            continue

        for peak_type in ["A_mode", "C_mode", "R_mode"]:
            for layer_mode in ["all", "block1", "block2"]:
                out_dir = Path(output_base) / model / f"{peak_type}_{layer_mode}"
                cmd = [
                    sys.executable, "run_causal_experiment.py",
                    "--model", model, "--device", device,
                    "--candidates_json", str(mode_tokens_path),
                    "--peak_type", peak_type,
                    "--layer_mode", layer_mode,
                    "--sample_list", str(sample_list_path),
                    "--n_samples", "20",
                    "--output_dir", str(out_dir),
                ]
                print(f"  Launching Gate ② for {model}/{peak_type}/{layer_mode}")
                p = subprocess.Popen(cmd)
                procs.append((f"{model}/{peak_type}/{layer_mode}", p))

    for name, p in procs:
        rc = p.wait()
        status = "OK" if rc == 0 else f"FAIL (rc={rc})"
        print(f"  Gate ② {name}: {status}")


def check_gate1_pass(gate1_base):
    """Check Gate ① pass criteria from design doc."""
    results = {}
    for model in MODELS:
        report_path = Path(gate1_base) / model / "contribution_report.json"
        if not report_path.exists():
            continue
        with open(report_path) as f:
            report = json.load(f)

        mode_path = Path(gate1_base) / model / "mode_tokens.json"
        mode_tokens = {}
        if mode_path.exists():
            with open(mode_path) as f:
                mode_tokens = json.load(f)

        # Extract medians from layer_analysis
        top1_shares = []
        mismatches = []
        entropies = []
        for l_info in report.get("layer_analysis", {}).values():
            if "mean_top1_share" in l_info:
                top1_shares.append(l_info["mean_top1_share"])
            if "mean_mismatch" in l_info:
                mismatches.append(l_info["mean_mismatch"])
            if "mean_entropy" in l_info:
                entropies.append(l_info["mean_entropy"])

        import numpy as np
        median_top1 = float(np.median(top1_shares)) if top1_shares else 0
        median_mismatch = float(np.median(mismatches)) if mismatches else 0
        median_entropy = float(np.median(entropies)) if entropies else 0

        a_mode = mode_tokens.get("A_mode", {}).get("abs_t", -1)
        c_mode = mode_tokens.get("C_mode", {}).get("abs_t", -1)

        # Check model-specific criteria
        passed = True
        reasons = []

        if model == "ecot-7b":
            if median_top1 <= 0.8:
                passed = False
                reasons.append(f"Top1 C̃ median {median_top1:.3f} <= 0.8")
            if a_mode != c_mode:
                passed = False
                reasons.append(f"A_mode({a_mode}) != C_mode({c_mode})")
        elif model == "openvla-7b":
            if median_mismatch <= 0.15:
                passed = False
                reasons.append(f"mismatch median {median_mismatch:.3f} <= 0.15")
        elif model == "spatialvla-4b":
            if median_mismatch >= 0.05:
                passed = False
                reasons.append(f"mismatch median {median_mismatch:.3f} >= 0.05")
            if median_entropy <= 2.0:
                passed = False
                reasons.append(f"entropy median {median_entropy:.3f} <= 2.0")
        elif model == "tracevla-phi3v":
            if median_top1 >= 0.2:
                passed = False
                reasons.append(f"Top1 C̃ median {median_top1:.3f} >= 0.2")

        results[model] = {
            "passed": passed,
            "median_top1": median_top1,
            "median_mismatch": median_mismatch,
            "median_entropy": median_entropy,
            "a_mode": a_mode,
            "c_mode": c_mode,
            "reasons": reasons,
        }

    # Print results
    print("\n" + "=" * 60)
    print("Gate ① Pass Criteria Check")
    print("=" * 60)
    all_pass = True
    for model, r in results.items():
        status = "PASS ✓" if r["passed"] else "FAIL ✗"
        if not r["passed"]:
            all_pass = False
        print(f"  {model}: {status}")
        print(f"    Top1 C̃ median={r['median_top1']:.3f}, mismatch={r['median_mismatch']:.3f}, "
              f"entropy={r['median_entropy']:.3f}")
        print(f"    A_mode={r['a_mode']}, C_mode={r['c_mode']}")
        for reason in r["reasons"]:
            print(f"    → {reason}")

    print(f"\nOverall: {'ALL PASS → proceed to Gate ②+③' if all_pass else 'SOME FAILED → investigate'}")
    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Gate Check Orchestrator")
    parser.add_argument("--gate", type=int, choices=[1, 2, 3],
                        help="Which gate to run (1, 2, or 3)")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--output_base", default="outputs/phase3_gate")
    parser.add_argument("--gate1_dir", default="outputs/phase3_gate",
                        help="Gate ① output dir (for Gate ② and ③)")
    parser.add_argument("--n_per_skill", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check_pass", action="store_true",
                        help="Check Gate ① pass criteria")
    args = parser.parse_args()

    if args.check_pass:
        check_gate1_pass(args.gate1_dir)
        return

    if args.gate == 1:
        run_gate1(args.models, args.output_base, args.n_per_skill, args.seed)
    elif args.gate == 2:
        run_gate2(args.models, args.gate1_dir, Path(args.output_base) / "gate2")
    elif args.gate == 3:
        print("Gate ③ not yet implemented — run manually with text_mask.py hooks")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

**Step 2: Verify CLI**

Run: `cd /home/kana5123/ATLASVLA && venv/bin/python run_gate_checks.py --help`
Expected: Shows all flags

**Step 3: Commit**

```bash
git add run_gate_checks.py
git commit -m "feat: add Gate Check orchestrator (Gate ①→②+③ pipeline)"
```

---

## Task 9: Create `tests/__init__.py` and Run Full Test Suite

**Why:** Ensure all test files are importable and passing before running experiments.

**Files:**
- Create: `tests/__init__.py`

**Step 1: Create init file**

```bash
touch tests/__init__.py
```

**Step 2: Run all tests**

Run: `cd /home/kana5123/ATLASVLA && venv/bin/python -m pytest tests/ -v`
Expected: All tests from Task 1, 2, 5, 7 pass

**Step 3: Commit**

```bash
git add tests/__init__.py
git commit -m "chore: add tests/__init__.py, verify full test suite"
```

---

## Task 10: Execute Gate ① (150 balanced samples, 4 models)

**Why:** This is the actual experiment. Everything before this was infrastructure.

**Step 1: Dry-run the sampler**

Run: `cd /home/kana5123/ATLASVLA && venv/bin/python data_sampler.py --dry_run`
Expected: Prints skill distribution, unknown < 5%

**Step 2: Run Gate ① for all 4 models**

Run: `cd /home/kana5123/ATLASVLA && venv/bin/python run_gate_checks.py --gate 1 --models ecot-7b openvla-7b spatialvla-4b tracevla-phi3v --n_per_skill 25 --seed 42`

Expected: ~75 minutes. Each model processes 150 samples. Output:
- `outputs/phase3_gate/{model}/contribution_report.json`
- `outputs/phase3_gate/{model}/sample_list.json`
- `outputs/phase3_gate/{model}/mode_tokens.json`

**Step 3: Check pass criteria**

Run: `cd /home/kana5123/ATLASVLA && venv/bin/python run_gate_checks.py --check_pass --gate1_dir outputs/phase3_gate`
Expected: All 4 models PASS their criteria

**Step 4: Commit results**

```bash
git add outputs/phase3_gate/*/contribution_report.json outputs/phase3_gate/*/mode_tokens.json outputs/phase3_gate/*/sample_list.json
git commit -m "data: Gate ① results — 150 balanced samples × 4 models"
```

---

## Task 11: Execute Gate ② (Layer-Local V=0)

**Depends on:** Task 10 (Gate ① complete, mode_tokens.json exists)

**Step 1: Run Gate ② for all models**

Run: `cd /home/kana5123/ATLASVLA && venv/bin/python run_gate_checks.py --gate 2 --models ecot-7b openvla-7b spatialvla-4b tracevla-phi3v --gate1_dir outputs/phase3_gate`

Expected: 9 conditions per model × 20 samples. ~45 min total.

**Step 2: Check pass criteria manually**

Read each `causal_report.json` from `outputs/phase3_gate/gate2/{model}/{peak}_{block}/` and verify:
1. Ranking preserved: KL(A_mode) vs KL(R_mode) same order as `all`
2. Effect size: block2 KL >= 30% of all KL
3. ΔKL sign: block2 > block1 or block1 > block2 consistent with expectation

**Step 3: Commit results**

```bash
git add outputs/phase3_gate/gate2/
git commit -m "data: Gate ② results — layer-local V=0, 9 conditions × 4 models"
```

---

## Task 12: Execute Gate ③ (Text Masking — Manual)

**Depends on:** Task 10 (Gate ① complete, sample_list.json exists)

Gate ③ requires custom scripting since it combines multiple masking conditions with probe evaluation. This task provides the script template.

**Step 1: Create `run_gate3_text_mask.py`**

This is a focused script for Gate ③ Part A (text V=0 vs KV-mask) + Part B (mini counterfactual). Due to complexity, implement as a separate file.

```python
#!/usr/bin/env python3
"""Gate ③: Text Masking Control + Mini Counterfactual.

Part A: Compare hidden probe accuracy under 4 conditions:
  A. Original (baseline from Gate ①)
  B. Text V=0 (value zeroed, Q/K routing alive)
  C. Text KV-mask (fully blocked)
  D. Vision V=0 (normalized — same token count as text)

Part B: Mini counterfactual under text masking
  20 pairs × 3 verb swaps: pick↔place, open↔close, move↔fold
  Measure Δhidden under orig / textV0 / textKV conditions.

Usage:
  python run_gate3_text_mask.py --model ecot-7b --device cuda:4 \
    --gate1_dir outputs/phase3_gate/ecot-7b
"""
import argparse
import json
import sys
import re
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

import config
from extract_attention import load_model_from_registry, get_layers, call_processor, detect_token_boundaries
from data_sampler import reload_samples_from_list
from verify_attention_sinks import SinkVerificationHookManager
from contribution.text_mask import TextValueZeroHook, TextKVMaskHook, sample_normalized_vision_positions
from contribution.causal import ValueZeroHook

# Gate ③ Part B: verb swap pairs
VERB_PAIRS = [("pick", "place"), ("open", "close"), ("move", "fold")]


def extract_hidden_at_query(model, inputs, model_cfg, query_pos, deep_layers):
    """Run forward and extract hidden states at query position from deep layers.
    Returns dict: {layer: (D,) numpy array}
    """
    hook_mgr = SinkVerificationHookManager(model, model_cfg)
    hook_mgr.register_hooks()
    hook_mgr.reset()
    with torch.no_grad():
        model(**inputs, output_attentions=True)
    hidden_states = hook_mgr.hidden_states
    hook_mgr.remove_hooks()

    result = {}
    for l in deep_layers:
        h = hidden_states.get(l)
        if h is not None and query_pos < h.shape[0]:
            result[l] = h[query_pos].cpu().float().numpy()
    return result


def run_gate3(model_name, device, gate1_dir, output_dir, n_samples=20):
    """Run Gate ③ for one model."""
    print(f"\n{'='*60}")
    print(f"Gate ③: Text Masking — {model_name}")
    print(f"{'='*60}")

    gate1_dir = Path(gate1_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    processor, model, model_cfg = load_model_from_registry(model_name, device)
    tokenizer = getattr(processor, 'tokenizer', processor)

    # Load samples from Gate ①
    sample_list_path = gate1_dir / "sample_list.json"
    samples = reload_samples_from_list(sample_list_path, config.DATA_CACHE_DIR)
    samples = samples[:n_samples]
    print(f"  Using {len(samples)} samples from Gate ①")

    # Detect boundaries from first sample
    boundaries = detect_token_boundaries(
        processor, model, samples[0]["image"], samples[0]["instruction"], device, model_cfg
    )
    text_range = (boundaries.get("text_start", boundaries["vision_end"]), boundaries["text_end"])
    vision_range = (boundaries["vision_start"], boundaries["vision_end"])
    n_text = text_range[1] - text_range[0]
    print(f"  Text range: {text_range}, Vision range: {vision_range}, n_text={n_text}")

    # Deep layers
    deep_layers = list(range(max(0, model_cfg.num_layers - 10), model_cfg.num_layers))

    # ── Part A: Probe accuracy under 4 conditions ──
    # (Simplified: collect hidden states, not full probe — probe runs offline)
    print("\n  Part A: Collecting hidden states under 4 conditions...")
    conditions = {}

    for cond_name in ["original", "text_v0", "text_kv", "vision_v0_norm"]:
        layer_hiddens = defaultdict(list)

        for si, sample in enumerate(samples):
            prompt = model_cfg.prompt_template.format(instruction=sample["instruction"])
            inputs = call_processor(processor, prompt, sample["image"], model_cfg, return_tensors="pt").to(device)
            if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
                inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

            sample_bounds = detect_token_boundaries(
                processor, model, sample["image"], sample["instruction"], device, model_cfg
            )
            query_pos = sample_bounds["text_end"] - 1
            text_r = (sample_bounds.get("text_start", sample_bounds["vision_end"]), sample_bounds["text_end"])
            vis_r = (sample_bounds["vision_start"], sample_bounds["vision_end"])
            n_t = text_r[1] - text_r[0]

            if cond_name == "text_v0":
                hook = TextValueZeroHook(text_r)
                hook.register(model, model_cfg, get_layers)
                h = extract_hidden_at_query(model, inputs, model_cfg, query_pos, deep_layers)
                hook.remove()
            elif cond_name == "text_kv":
                kv_hook = TextKVMaskHook(text_r)
                inputs["attention_mask"] = kv_hook.apply_to_attention_mask(inputs["attention_mask"])
                h = extract_hidden_at_query(model, inputs, model_cfg, query_pos, deep_layers)
            elif cond_name == "vision_v0_norm":
                vis_positions = sample_normalized_vision_positions(vis_r, n_t, seed=42 + si)
                hook = ValueZeroHook(vis_positions)
                hook.register(model, model_cfg, get_layers)
                h = extract_hidden_at_query(model, inputs, model_cfg, query_pos, deep_layers)
                hook.remove()
            else:
                h = extract_hidden_at_query(model, inputs, model_cfg, query_pos, deep_layers)

            for l, vec in h.items():
                layer_hiddens[l].append(vec)

        conditions[cond_name] = {l: np.stack(vecs) for l, vecs in layer_hiddens.items()}
        print(f"    {cond_name}: collected {len(samples)} samples × {len(deep_layers)} layers")

    # Save Part A hidden states for offline probe evaluation
    # (Too large for JSON — save as npz)
    for cond_name, layer_data in conditions.items():
        for l, mat in layer_data.items():
            np.save(output_dir / f"hidden_{cond_name}_layer{l}.npy", mat)

    # Save skill labels for probe
    labels = [s["skill"] for s in samples]
    with open(output_dir / "skill_labels.json", "w") as f:
        json.dump(labels, f)

    print(f"\n  Part A hidden states saved to {output_dir}/")

    # ── Part B: Mini counterfactual ──
    print("\n  Part B: Mini counterfactual (verb swap)...")
    cf_results = []

    for si, sample in enumerate(samples):
        skill = sample["skill"]
        # Find matching verb pair
        swap_verb = None
        for v1, v2 in VERB_PAIRS:
            if skill == v1:
                swap_verb = v2
                break
            elif skill == v2:
                swap_verb = v1
                break

        if swap_verb is None:
            continue

        # Generate swapped instruction
        orig_instr = sample["instruction"]
        swapped_instr = re.sub(r'\b' + skill + r'\b', swap_verb, orig_instr.lower(), count=1)
        if swapped_instr == orig_instr.lower():
            continue

        # Forward with original and swapped instruction
        prompt_orig = model_cfg.prompt_template.format(instruction=orig_instr)
        prompt_swap = model_cfg.prompt_template.format(instruction=swapped_instr)

        inputs_orig = call_processor(processor, prompt_orig, sample["image"], model_cfg, return_tensors="pt").to(device)
        inputs_swap = call_processor(processor, prompt_swap, sample["image"], model_cfg, return_tensors="pt").to(device)

        for inp in [inputs_orig, inputs_swap]:
            if "pixel_values" in inp and inp["pixel_values"].dtype != model.dtype:
                inp["pixel_values"] = inp["pixel_values"].to(model.dtype)

        bounds = detect_token_boundaries(
            processor, model, sample["image"], orig_instr, device, model_cfg
        )
        qpos = bounds["text_end"] - 1
        text_r = (bounds.get("text_start", bounds["vision_end"]), bounds["text_end"])

        # Condition: original
        h_orig = extract_hidden_at_query(model, inputs_orig, model_cfg, qpos, deep_layers)
        h_swap = extract_hidden_at_query(model, inputs_swap, model_cfg, qpos, deep_layers)

        # Condition: text KV-mask
        kv_hook = TextKVMaskHook(text_r)
        inputs_orig_kv = {k: v.clone() for k, v in inputs_orig.items()}
        inputs_swap_kv = {k: v.clone() for k, v in inputs_swap.items()}
        inputs_orig_kv["attention_mask"] = kv_hook.apply_to_attention_mask(inputs_orig_kv["attention_mask"])
        inputs_swap_kv["attention_mask"] = kv_hook.apply_to_attention_mask(inputs_swap_kv["attention_mask"])
        h_orig_kv = extract_hidden_at_query(model, inputs_orig_kv, model_cfg, qpos, deep_layers)
        h_swap_kv = extract_hidden_at_query(model, inputs_swap_kv, model_cfg, qpos, deep_layers)

        # Compute Δhidden for each layer
        for l in deep_layers:
            if l not in h_orig or l not in h_swap:
                continue
            norm_orig = np.linalg.norm(h_orig[l])
            delta_orig = np.linalg.norm(h_orig[l] - h_swap[l]) / max(norm_orig, 1e-10)
            delta_kv = 0.0
            if l in h_orig_kv and l in h_swap_kv:
                delta_kv = np.linalg.norm(h_orig_kv[l] - h_swap_kv[l]) / max(np.linalg.norm(h_orig_kv[l]), 1e-10)

            cf_results.append({
                "sample_idx": si,
                "layer": l,
                "skill": skill,
                "swap_verb": swap_verb,
                "delta_orig": float(delta_orig),
                "delta_textKV": float(delta_kv),
            })

    # Save Part B results
    cf_path = output_dir / "counterfactual_results.json"
    with open(cf_path, "w") as f:
        json.dump(cf_results, f, indent=2)
    print(f"  Part B: {len(cf_results)} measurements saved to {cf_path}")

    # Summary
    if cf_results:
        delta_orig_mean = np.mean([r["delta_orig"] for r in cf_results])
        delta_kv_mean = np.mean([r["delta_textKV"] for r in cf_results])
        print(f"\n  Summary: Δhidden_orig={delta_orig_mean:.4f}, Δhidden_textKV={delta_kv_mean:.4f}")
        print(f"  Ratio: Δorig/ΔKV = {delta_orig_mean/max(delta_kv_mean, 1e-10):.2f}x")

    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Gate ③: Text Masking + Mini Counterfactual")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--gate1_dir", required=True,
                        help="Gate ① output dir for this model")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--n_samples", type=int, default=20)
    args = parser.parse_args()

    out = Path(args.output_dir) if args.output_dir else config.OUTPUT_DIR / "phase3_gate" / "gate3" / args.model
    run_gate3(args.model, args.device, args.gate1_dir, out, args.n_samples)


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add run_gate3_text_mask.py
git commit -m "feat: add Gate ③ text masking + mini counterfactual script"
```

---

## Verification Checklist

After all tasks, verify:

1. **Tests pass:** `venv/bin/python -m pytest tests/ -v` → all green
2. **Balanced sampling:** `venv/bin/python data_sampler.py --dry_run` → unknown < 5%
3. **Gate ① reports:** `outputs/phase3_gate/{model}/contribution_report.json` → 150 samples, mode_tokens present
4. **Gate ② reports:** `outputs/phase3_gate/gate2/{model}/{peak}_{block}/causal_report.json` → KL values differ by block
5. **Gate ③ reports:** `outputs/phase3_gate/gate3/{model}/counterfactual_results.json` → Δorig >> ΔtextKV

**Success criteria:**
- Gate ① taxonomy holds at 150 samples (model-specific thresholds met)
- Gate ② block2 KL >= 30% of all KL (intervention is meaningful, not just noise)
- Gate ③ Δhidden_orig >> Δhidden_textKV (verb info flows through text, not vision artifact)
