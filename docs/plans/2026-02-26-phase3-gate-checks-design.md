# Phase 2.5→3 Gate Checks Design

**Date:** 2026-02-26
**Status:** Design approved, ready for implementation plan
**Prerequisite:** Phase 2.5 dual-track results (20 samples, 4 models)

---

## Goal

Validate Phase 2.5 taxonomy (bottleneck/coexist/normal/fragile) with larger samples, layer-local causal verification, and text leakage control before proceeding to Phase 3 (counterfactual, position anchoring, SimplerEnv).

## Two-Stage Replication Strategy (B1)

- **Fast replicate (Gate ①):** Top6 skills × 25 = 150 samples
- **Full replicate (Phase 3-①):** Top6 skills × 40 = 240 samples (논문 본문용)

---

## Gate Check ① — Fast Replicate (150 samples)

### Purpose

Verify 20-sample taxonomy holds at 150 samples with balanced skills.

### New File: `data_sampler.py`

Balanced sampling from BridgeData V2 cache (38,642 episodes, 1.38M steps).

**Target skills (6):** place, move, pick, fold, open, close
**Per-skill:** 25 episodes, **total:** 150 samples
**Sampling unit:** episode (not step — no duplicate episodes)

### label_skill_from_instruction Improvements

- Stemming: placed→place, opened→open, moved→move, folded→fold
- Synonyms: put→place, slide→move, unfold→fold
- Target: unknown rate from 15.4% → <5%

### Sample List Caching (B2)

Gate ① saves `sample_list.json` for reuse across all subsequent experiments:

```json
{
    "version": "gate_v1",
    "seed": 42,
    "n_per_skill": 25,
    "target_skills": ["place", "move", "pick", "fold", "open", "close"],
    "samples": [
        {"episode_id": 12345, "global_idx": 67890, "instruction": "...", "skill": "pick"},
        ...
    ]
}
```

All Gate ②, ③, and Phase 3 experiments reuse this exact sample set.

### Mode Token Extraction (B4)

After Gate ① runs, extract per-model mode tokens:

```json
{
    "A_mode": {"abs_t": 0, "token_str": "<s>", "freq": 1.0},
    "C_mode": {"abs_t": 281, "token_str": "?", "freq": 0.9},
    "R_mode": {"abs_t": 3, "token_str": "", "freq": 0.7}
}
```

**Mode stability (new — B4 fix):** Store `freq = count(mode) / num_deep_layers`.
- freq >= 0.7: stable → use single mode token
- freq < 0.7: unstable → Gate ② tests Top-3 most frequent positions

### CLI Changes to `run_contribution_analysis.py`

```
--balanced          Use balanced sampling
--n_per_skill 25    Samples per skill
--seed 42           Reproducibility
--sample_list PATH  Path to save/load sample_list.json
```

### Pass Criteria (C — median-based)

| Model | Metric | Threshold |
|-------|--------|-----------|
| ECoT | Top1 C̃ **median** | > 0.8 |
| ECoT | A_mode == C_mode | True |
| OpenVLA | mismatch **median** | > 0.15 |
| OpenVLA | A_mode ∈ vision, C_mode ∈ text | True |
| SpatialVLA | mismatch **median** | < 0.05 |
| SpatialVLA | entropy **median** | > 2.0 |
| TraceVLA | Top1 C̃ **median** | < 0.2 |

### Execution

```bash
# 4 models parallel on GPU 0-3
venv/bin/python run_contribution_analysis.py --model ecot-7b --device cuda:0 \
    --balanced --n_per_skill 25 --seed 42 \
    --output_dir outputs/phase3_gate/ecot-7b &
# ... same for openvla-7b (cuda:1), spatialvla-4b (cuda:2), tracevla-phi3v (cuda:3)
```

---

## Gate Check ② — Layer-Local V=0

### Purpose

Refute "all-layer V=0 is too strong an intervention" by showing block-level V=0 preserves ranking.

### ValueZeroHook Modification

Add `target_layers` parameter:

```python
class ValueZeroHook:
    def __init__(self, target_positions, target_layers=None):
        self.target_layers = target_layers  # None = all layers

    def register(self, model, model_cfg, get_layers_fn):
        layers = get_layers_fn(model, model_cfg)
        for layer_idx, layer in enumerate(layers):
            if self.target_layers is not None and layer_idx not in self.target_layers:
                continue
            # ... attach hook
```

### Layer Ranges (B3 — deep layers only)

```
all    = full deep range (22-31 for 32L models, 16-25 for 26L)
block1 = first half of deep (22-26 / 16-20)
block2 = second half of deep (27-31 / 21-25)
```

Never touch shallow layers (0-21). Interpretation stays clean.

### Target Tokens (B4 — mode tokens from Gate ①)

```
ECoT:       A_mode=0, C_mode=0, R_mode=281
OpenVLA:    A_mode=0, C_mode=281, R_mode=varies
SpatialVLA: A_mode=260, C_mode=260, R_mode=193
TraceVLA:   A_mode=0, C_mode=1, R_mode=3
```

If mode_freq < 0.7 for any peak, test Top-3 candidates instead.

### Experiment Matrix (per model)

```
             A_mode    C_mode    R_mode
all(22-31)   ✓ KL     ✓ KL      ✓ KL      ← reuse existing if same tokens
block1       ✓ KL     ✓ KL      ✓ KL
block2       ✓ KL     ✓ KL      ✓ KL
```

= 9 conditions × 20 samples (first 20 from sample_list.json)

### CLI Changes to `run_causal_experiment.py`

```
--layer_mode {all, block1, block2}
--candidates_json PATH           # mode_tokens.json from Gate ①
--sample_list PATH               # reuse Gate ① samples
```

### Pass Criteria

1. **Ranking preserved:** If all V=0 gives KL(A_mode) > KL(R_mode), then block2 also gives same ranking
2. **Effect size (new criterion):** block2 KL >= 30% of all KL (local intervention is still meaningful, not just noise)
3. **ΔKL sign:** KL(block2) - KL(block1) sign matches model expectation (deeper layers = more processing = potentially higher impact)
4. **R-peak causal insignificance:** R_mode V=0 KL << A_mode or C_mode V=0 KL → confirms sink definition (high attention but causally unimportant)
5. **TraceVLA fragility:** block V=0 KL still relatively high compared to other models

---

## Gate Check ③ — Text Masking + Mini Counterfactual

### Purpose

Prove hidden probe accuracy is not pure text label leakage. Two parts.

### Part A: Text V=0 vs Text KV-Mask Probe

**Two masking modes (fix from review — Q/K path concern):**

1. **Text V=0:** Zero value projections for text tokens. Q/K paths remain active — instruction routing information can still flow through attention patterns.
2. **Text KV-mask:** Set attention_mask[:, :, :, text_range] = -inf BEFORE forward pass. This fully blocks text tokens from being attended to — both value AND routing are killed.

Why both: Text V=0 kills the "what" (content) but not the "where" (routing). Text KV-mask kills both. Comparing the two reveals whether verb information flows through V (content) or Q/K (routing pattern).

**Implementation:**
- `TextValueZeroHook`: Reuse `ValueZeroHook` with text range from `detect_token_boundaries()` (B5)
- `TextKVMaskHook`: Modify `attention_mask` tensor before forward pass

**Precise text range (B5):** Always use `detect_token_boundaries()` output. Never hardcode. Save `masked_token_strs` in report for verification.

**Experiment conditions:**
```
Condition A: Original               → hidden probe accuracy (from Gate ①)
Condition B: Text V=0               → hidden probe accuracy
Condition C: Text KV-mask           → hidden probe accuracy
Condition D: Vision V=0 (normalized)→ hidden probe accuracy
```

**Normalized vision masking (review fix):**
Vision has ~256 tokens vs text ~17 tokens. Raw comparison is unfair.
- Condition D masks **same number of tokens as text** (randomly sampled vision positions)
- This makes modality importance comparison fair

### Part B: Mini Counterfactual under Text Masking (B6)

Test whether verb swap changes hidden states when text is masked.

**20 pairs from 3 verb swaps:**
- pick ↔ place (~7 pairs)
- open ↔ close (~7 pairs)
- move ↔ fold (~6 pairs)

**Per pair, measure:**
```
Δhidden_orig     = ||h(img, verb1) - h(img, verb2)||₂ / ||h(img, verb1)||₂   (baseline)
Δhidden_textV0   = ||h_V0(img, verb1) - h_V0(img, verb2)||₂ / ...            (V=0)
Δhidden_textKV   = ||h_KV(img, verb1) - h_KV(img, verb2)||₂ / ...            (KV-mask)
```

**Expected interpretation:**
- Δhidden_orig > 0: verb changes hidden states (expected)
- Δhidden_textV0: If still > 0, verb routing info flows through Q/K (not just V)
- Δhidden_textKV ≈ 0: Full text blocking kills verb information (confirms text is the channel)
- Δhidden_textKV > 0: Some verb info from tokenization/position shift artifact — check input_ids length

### Pass Criteria (revised)

**Part A:**
- Text KV-mask probe accuracy should drop significantly (text carries skill info — expected)
- Text KV-mask probe > chance (1/6 = 16.7%) indicates some vision-only skill signal
- Vision-based structural metrics (peak positions, entropy) survive text V=0 (peak token shift < 30%)

**Part B:**
- Δhidden_orig >> Δhidden_textKV confirms verb information flows primarily through text
- If Δhidden_textV0 >> Δhidden_textKV, Q/K routing carries additional verb signal
- Both are valid findings for the paper — the key is characterizing the pathway, not proving one is "better"

---

## Dependency & Execution Order

```
Gate ① (150 balanced, ~75min)
    │
    ├── sample_list.json ───→ Gate ②, ③
    ├── mode_tokens.json ───→ Gate ②
    └── probe baselines   ───→ Gate ③

Gate ① complete
    │
    ├── Gate ② (GPU 0-3, ~45min)  ← independent
    └── Gate ③ (GPU 4-7, ~2hr)    ← independent

Both complete → Phase 3 go/no-go decision
```

---

## Phase 3 (after gate pass)

1. **Full Replicate:** Top6 × 40 = 240 samples (논문 본문 수치)
2. **Counterfactual Δsignature:** 3 verb pairs × 240 samples → JS(C̃_orig, C̃_swap)
3. **Position Anchoring Test:** Permute vision patches → bottleneck follows position or content?
4. **SimplerEnv Performance Link:** bottleneck severity ↔ OOD success rate correlation

---

## Files to Create/Modify

| Action | File | Description |
|--------|------|-------------|
| Create | `data_sampler.py` | Balanced skill sampling + sample_list.json caching |
| Create | `contribution/text_mask.py` | TextValueZeroHook + TextKVMaskHook |
| Modify | `contribution/causal.py` | Add `target_layers` param to ValueZeroHook |
| Modify | `contribution/signature.py` | Stemming + synonym expansion for skill labels |
| Modify | `run_contribution_analysis.py` | --balanced, --n_per_skill, --seed, --sample_list |
| Modify | `run_causal_experiment.py` | --layer_mode, --candidates_json, --sample_list |
| Create | `run_gate_checks.py` | Orchestrator: runs Gate ①→②+③, checks pass criteria |
