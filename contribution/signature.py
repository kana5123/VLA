# contribution/signature.py
"""
Skill signature analysis (Design Sections 6.1–6.3, 7).

Computes:
- Skill labels from instructions (verb extraction)
- Contribution signatures p_contrib^(n) per sample
- Within-skill vs between-skill JS divergence
- Linear probe accuracy
- Counterfactual instruction analysis
"""
import re
import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# ── Skill labeling ──────────────────────────────────────────────

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
    Handles: -ed, -ing, -s (placed->place, picking->pick, moves->move).
    """
    if word.endswith("ing") and len(word) > 4:
        # closing->close, moving->move, picking->pick
        base = word[:-3]
        if base + "e" in _VERB_TO_SKILL:
            return base + "e"
        if base in _VERB_TO_SKILL:
            return base
        # doubling: putting->put, grabbing->grab
        if len(base) >= 3 and base[-1] == base[-2]:
            shorter = base[:-1]
            if shorter in _VERB_TO_SKILL:
                return shorter
    if word.endswith("ed") and len(word) > 3:
        # placed->place, moved->move
        base = word[:-1]  # placed->place (just remove d)
        if base in _VERB_TO_SKILL:
            return base
        base = word[:-2]  # opened->open
        if base in _VERB_TO_SKILL:
            return base
        # doubled: grabbed->grab
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
    Applies stemming (placed->place, picking->pick) and synonym matching.
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


# ── Skill signatures ────────────────────────────────────────────

def compute_skill_signature(c_tildes: list[np.ndarray]) -> np.ndarray:
    """Compute p_contrib^(n)(j) = Normalize(Σ_{l∈L*} C̃_l(j)).

    Args:
        c_tildes: list of C̃ arrays from deep layers, each shape (seq,)
    Returns:
        Normalized signature, shape (seq,)
    """
    stacked = np.stack(c_tildes)
    summed = stacked.sum(axis=0)
    total = summed.sum()
    if total < 1e-10:
        return summed
    return summed / total


# ── Within/Between distance ─────────────────────────────────────

def compute_within_between_distance(
    signatures: list[np.ndarray],
    labels: list[str],
) -> tuple[float, float]:
    """Compute D_within and D_between using JS divergence.

    Args:
        signatures: list of contribution signatures, each (seq,)
        labels: skill label per sample

    Returns:
        (d_within, d_between) — mean JS divergence within and between skills
    """
    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2:
        return 0.0, 0.0

    within_dists = []
    between_dists = []

    # Pad signatures to common length if needed (safety fallback)
    max_len = max(len(s) for s in signatures)
    padded = []
    for s in signatures:
        if len(s) < max_len:
            s = np.pad(s, (0, max_len - len(s)), constant_values=1e-10)
        padded.append(s)

    n = len(padded)
    for i in range(n):
        for j in range(i + 1, n):
            si = np.clip(padded[i], 1e-10, None)
            sj = np.clip(padded[j], 1e-10, None)
            si = si / si.sum()
            sj = sj / sj.sum()
            js = float(jensenshannon(si, sj) ** 2)

            if labels[i] == labels[j]:
                within_dists.append(js)
            else:
                between_dists.append(js)

    d_within = np.mean(within_dists) if within_dists else 0.0
    d_between = np.mean(between_dists) if between_dists else 0.0

    return d_within, d_between


# ── Linear probe ────────────────────────────────────────────────

def run_linear_probe(X: np.ndarray, y: np.ndarray, cv: int = 5) -> float:
    """Train linear probe on contribution vectors to predict skill labels.

    Args:
        X: (n_samples, seq_len) — contribution signatures
        y: (n_samples,) — skill labels (int-encoded)
        cv: cross-validation folds

    Returns:
        Mean cross-validation accuracy
    """
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="auto")
    scores = cross_val_score(clf, X, y, cv=min(cv, len(np.unique(y))), scoring="accuracy")
    return float(scores.mean())


# ── Counterfactual ──────────────────────────────────────────────

COUNTERFACTUAL_PAIRS = [
    ("pick", "push"),
    ("place", "move"),
    ("open", "close"),
]


def generate_counterfactual_instructions(instruction: str) -> list[tuple[str, str]]:
    """Generate counterfactual instruction pairs by swapping verbs.

    Args:
        instruction: original instruction text

    Returns:
        list of (target_verb, swapped_instruction) pairs
    """
    original_skill = label_skill_from_instruction(instruction)

    results = []
    for pair in COUNTERFACTUAL_PAIRS:
        if original_skill in pair:
            target = pair[1] if pair[0] == original_skill else pair[0]
            new_words = []
            replaced = False
            for w in instruction.split():
                clean = re.sub(r"[^a-zA-Z]", "", w).lower()
                if not replaced and clean in _VERB_TO_SKILL and _VERB_TO_SKILL[clean] == original_skill:
                    new_words.append(target)
                    replaced = True
                else:
                    new_words.append(w)
            if replaced:
                results.append((target, " ".join(new_words)))

    return results
