# Object-Aware Adapter v2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the v1 Differentiable Attention Adapter to support object-targeted redistribution via cross-attention and SAM2-based object masks.

**Architecture:** Two-branch adapter (Branch 1: per-head p via MLP, Branch 2: redistribution weights via cross-attention) that takes h_27_last, h_27_vision, and a pre-computed SAM object_mask. The `apply_var` function is modified to use learned redistribution weights instead of proportional redistribution. A SAM preprocessing pipeline produces object_masks.dat for the data cache.

**Tech Stack:** PyTorch, HuggingFace transformers, SAM2 (facebook/sam2.1-hiera-tiny), GroundingDINO, spaCy (en_core_web_sm), numpy memmap, HF accelerate (multi-GPU DDP)

**Design Doc:** `docs/plans/2026-02-19-object-aware-adapter-v2-design.md`

---

## Task 1: Add v2 config constants to config.py

**Files:**
- Modify: `config.py:109-145`
- Test: `tests/test_config.py` (create)

**Step 1: Write the failing test**

Create `tests/__init__.py` and `tests/test_config.py`:

```python
# tests/__init__.py
# (empty)
```

```python
# tests/test_config.py
"""Tests for v2 config constants."""

def test_v2_adapter_config_exists():
    import config

    # SAM preprocessing
    assert hasattr(config, "SAM_MASKS_FILENAME")
    assert config.SAM_MASKS_FILENAME == "object_masks.dat"
    assert hasattr(config, "SAM_FAILURE_MARKER")
    assert config.SAM_FAILURE_MARKER == 255
    assert hasattr(config, "SAM_EPISODE_FAILURE_THRESHOLD")
    assert config.SAM_EPISODE_FAILURE_THRESHOLD == 0.5

    # Cross-attention
    assert hasattr(config, "ADAPTER_V2_QUERY_DIM")
    assert config.ADAPTER_V2_QUERY_DIM == 128
    assert hasattr(config, "ADAPTER_V2_TEMPERATURE")
    assert config.ADAPTER_V2_TEMPERATURE == 2.0

    # Blend alpha
    assert hasattr(config, "ADAPTER_V2_BLEND_INIT")
    assert config.ADAPTER_V2_BLEND_INIT == -4.0

    # Object mask MLP
    assert hasattr(config, "ADAPTER_V2_MASK_DIM")
    assert config.ADAPTER_V2_MASK_DIM == 64
```

**Step 2: Run test to verify it fails**

Run: `cd /home/kana5123/ATLASVLA && python -m pytest tests/test_config.py::test_v2_adapter_config_exists -v`
Expected: FAIL with `AttributeError: module 'config' has no attribute 'SAM_MASKS_FILENAME'`

**Step 3: Write minimal implementation**

Append to `config.py` after line 145:

```python
# ── Adapter v2: Object-Aware Differentiable Adapter ─────────────────
# SAM preprocessing
SAM_MASKS_FILENAME = "object_masks.dat"
SAM_FAILURE_MARKER = 255          # uint8 marker for SAM-failed steps
SAM_EPISODE_FAILURE_THRESHOLD = 0.5  # exclude episode if >50% steps fail

# Cross-attention (Branch 2: redistribution weights)
ADAPTER_V2_QUERY_DIM = 128        # projection dim for query/key
ADAPTER_V2_TEMPERATURE = 2.0      # softmax temperature for smoother gradients

# Blend alpha (proportional → learned transition)
ADAPTER_V2_BLEND_INIT = -4.0      # sigmoid(-4) ≈ 0.018

# Object mask embedding (Branch 1 augmentation)
ADAPTER_V2_MASK_DIM = 64          # mask embedding dimension in p-head MLP
```

**Step 4: Run test to verify it passes**

Run: `cd /home/kana5123/ATLASVLA && python -m pytest tests/test_config.py::test_v2_adapter_config_exists -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/__init__.py tests/test_config.py config.py
git commit -m "feat: add v2 adapter config constants (SAM, cross-attention, blend_alpha)"
```

---

## Task 2: Create AttentionAdapterV2 — Branch 1 (per-head p MLP with object_mask)

**Files:**
- Modify: `adapter_model.py` (add new class, keep v1 for backwards compat)
- Test: `tests/test_adapter_model_v2.py` (create)

**Step 1: Write the failing test**

```python
# tests/test_adapter_model_v2.py
"""Tests for AttentionAdapterV2."""
import torch


def test_branch1_output_shape():
    """Branch 1: (h_last, object_mask) -> (num_target_layers, num_heads)."""
    from adapter_model import AttentionAdapterV2

    adapter = AttentionAdapterV2(hidden_dim=4096, vision_tokens=256)
    h_last = torch.randn(4096)
    object_mask = torch.zeros(256)
    object_mask[100:120] = 1.0  # 20 object patches

    p, _ = adapter(h_last, h_vision=None, object_mask=object_mask)
    assert p.shape == (4, 32), f"Expected (4, 32), got {p.shape}"


def test_branch1_init_near_zero():
    """Branch 1 output should start near 0 (sigmoid(-4) ≈ 0.018)."""
    from adapter_model import AttentionAdapterV2

    adapter = AttentionAdapterV2(hidden_dim=4096, vision_tokens=256)
    h_last = torch.randn(4096)
    object_mask = torch.zeros(256)

    with torch.no_grad():
        p, _ = adapter(h_last, h_vision=None, object_mask=object_mask)
    assert p.mean().item() < 0.05, f"Expected near-zero init, got mean={p.mean().item():.4f}"
```

**Step 2: Run test to verify it fails**

Run: `cd /home/kana5123/ATLASVLA && python -m pytest tests/test_adapter_model_v2.py::test_branch1_output_shape -v`
Expected: FAIL with `ImportError: cannot import name 'AttentionAdapterV2'`

**Step 3: Write minimal implementation**

Add to `adapter_model.py` after the existing `AdapterWithHook` class (after line 172):

```python
class AttentionAdapterV2(nn.Module):
    """Object-aware attention adapter with two branches.

    Branch 1: per-head p (how much sink attention to free)
        Input: h_27_last (4096,) + object_mask embedding (64,) → MLP → (4×32,) → sigmoid
    Branch 2: redistribution weights (where to direct freed attention)
        Input: h_27_last as query, h_27_vision as keys → cross-attention → (V,) softmax
    Blending: blend_alpha * learned_weights + (1-blend_alpha) * proportional_weights
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        num_target_layers: int = config.ADAPTER_NUM_TARGET_LAYERS,
        num_heads: int = config.NUM_HEADS,
        intermediate_dim: int = config.ADAPTER_INTERMEDIATE_DIM,
        dropout: float = config.ADAPTER_DROPOUT,
        vision_tokens: int = 256,
        query_dim: int = config.ADAPTER_V2_QUERY_DIM,
        mask_dim: int = config.ADAPTER_V2_MASK_DIM,
        temperature: float = config.ADAPTER_V2_TEMPERATURE,
        blend_init: float = config.ADAPTER_V2_BLEND_INIT,
    ):
        super().__init__()
        self.num_target_layers = num_target_layers
        self.num_heads = num_heads
        self.query_dim = query_dim
        self.temperature = temperature
        output_dim = num_target_layers * num_heads  # 4 × 32 = 128

        # ── Branch 1: per-head p ──
        # h_last → (intermediate_dim,) ; object_mask → (mask_dim,) ; concat → MLP → (output_dim,)
        self.h_proj = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.SiLU(),
        )
        self.mask_proj = nn.Sequential(
            nn.Linear(vision_tokens, mask_dim),
            nn.SiLU(),
        )
        concat_dim = intermediate_dim + mask_dim  # 256 + 64 = 320
        self.p_head = nn.Sequential(
            nn.Linear(concat_dim, output_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )
        self._init_p_head_near_zero()

        # ── Branch 2: cross-attention redistribution weights ──
        self.query_proj = nn.Linear(hidden_dim, query_dim)
        self.key_proj = nn.Linear(hidden_dim, query_dim)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)

        # ── Blend alpha: learnable sigmoid scalar ──
        # sigmoid(blend_init) ≈ 0.018 → starts with proportional redistribution
        self._blend_logit = nn.Parameter(torch.tensor(blend_init))

    def _init_p_head_near_zero(self):
        """Zero-init last layer weights, negative bias → sigmoid ≈ 0 at start."""
        last_linear = self.p_head[-1]
        nn.init.zeros_(last_linear.weight)
        nn.init.constant_(last_linear.bias, -4.0)

    @property
    def blend_alpha(self) -> torch.Tensor:
        return torch.sigmoid(self._blend_logit)

    def forward(
        self,
        h_last: torch.Tensor,
        h_vision: torch.Tensor | None = None,
        object_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass producing per-head p and redistribution weights.

        Args:
            h_last: (hidden_dim,) or (B, hidden_dim) — last token hidden state
            h_vision: (V, hidden_dim) or (B, V, hidden_dim) — all vision token hidden states
            object_mask: (V,) or (B, V) — binary mask (1=object, 0=background)

        Returns:
            p_matrix: (num_target_layers, num_heads) or (B, ...) — per-head redistribution strength
            redistribution_weights: (V,) or (B, V) — per-patch weights (None if h_vision not given)
        """
        squeeze = h_last.dim() == 1
        if squeeze:
            h_last = h_last.unsqueeze(0)

        # ── Branch 1: per-head p ──
        h_emb = self.h_proj(h_last)  # (B, intermediate_dim)

        if object_mask is not None:
            if object_mask.dim() == 1:
                object_mask = object_mask.unsqueeze(0)
            mask_emb = self.mask_proj(object_mask)  # (B, mask_dim)
        else:
            mask_emb = torch.zeros(
                h_last.shape[0], self.mask_proj[0].out_features,
                device=h_last.device, dtype=h_last.dtype,
            )

        concat = torch.cat([h_emb, mask_emb], dim=-1)  # (B, 320)
        p_logits = self.p_head(concat)  # (B, 128)
        p = torch.sigmoid(p_logits)
        p = p.view(-1, self.num_target_layers, self.num_heads)  # (B, 4, 32)

        # ── Branch 2: redistribution weights ──
        redist_weights = None
        if h_vision is not None:
            if h_vision.dim() == 2:
                h_vision = h_vision.unsqueeze(0)

            query = self.query_proj(h_last)  # (B, query_dim)
            keys = self.key_proj(h_vision)   # (B, V, query_dim)

            # Scaled dot-product with temperature
            scores = torch.bmm(
                query.unsqueeze(1), keys.transpose(1, 2)
            ).squeeze(1)  # (B, V)
            scores = scores / (self.query_dim ** 0.5 * self.temperature)

            # Mask background patches → -inf before softmax
            if object_mask is not None:
                bg_mask = object_mask == 0  # (B, V)
                scores = scores.masked_fill(bg_mask, float("-inf"))

            redist_weights = torch.softmax(scores, dim=-1)  # (B, V)

        if squeeze:
            p = p.squeeze(0)
            if redist_weights is not None:
                redist_weights = redist_weights.squeeze(0)

        return p, redist_weights

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def sparsity_stats(self, p_matrix: torch.Tensor) -> dict:
        """Compute sparsity statistics of adapter output."""
        with torch.no_grad():
            flat = p_matrix.flatten().float()
            return {
                "mean_p": flat.mean().item(),
                "active_ratio": (flat > 0.1).float().mean().item(),
                "max_p": flat.max().item(),
                "min_p": flat.min().item(),
                "blend_alpha": self.blend_alpha.item(),
            }
```

**Step 4: Run test to verify it passes**

Run: `cd /home/kana5123/ATLASVLA && python -m pytest tests/test_adapter_model_v2.py -v`
Expected: PASS (both tests)

**Step 5: Commit**

```bash
git add adapter_model.py tests/test_adapter_model_v2.py
git commit -m "feat: add AttentionAdapterV2 with Branch 1 (p-head MLP + object_mask)"
```

---

## Task 3: Test AttentionAdapterV2 — Branch 2 (cross-attention) and blending

**Files:**
- Test: `tests/test_adapter_model_v2.py` (append)

**Step 1: Write the failing test**

Append to `tests/test_adapter_model_v2.py`:

```python
def test_branch2_output_shape():
    """Branch 2: cross-attention → (V,) redistribution weights."""
    from adapter_model import AttentionAdapterV2

    adapter = AttentionAdapterV2(hidden_dim=4096, vision_tokens=256)
    h_last = torch.randn(4096)
    h_vision = torch.randn(256, 4096)
    object_mask = torch.zeros(256)
    object_mask[50:70] = 1.0

    _, redist = adapter(h_last, h_vision, object_mask)
    assert redist is not None
    assert redist.shape == (256,), f"Expected (256,), got {redist.shape}"


def test_branch2_object_only():
    """Background patches must have zero redistribution weight."""
    from adapter_model import AttentionAdapterV2

    adapter = AttentionAdapterV2(hidden_dim=4096, vision_tokens=256)
    h_last = torch.randn(4096)
    h_vision = torch.randn(256, 4096)
    object_mask = torch.zeros(256)
    object_mask[100:110] = 1.0  # only patches 100-109 are object

    with torch.no_grad():
        _, redist = adapter(h_last, h_vision, object_mask)

    # Background patches should be 0
    bg_weight = redist[object_mask == 0].sum().item()
    assert bg_weight < 1e-6, f"Background weight should be ~0, got {bg_weight}"

    # Object patches should sum to ~1
    obj_weight = redist[object_mask == 1].sum().item()
    assert abs(obj_weight - 1.0) < 1e-5, f"Object weight should sum to 1.0, got {obj_weight}"


def test_branch2_none_without_h_vision():
    """Branch 2 returns None if h_vision is not provided."""
    from adapter_model import AttentionAdapterV2

    adapter = AttentionAdapterV2(hidden_dim=4096, vision_tokens=256)
    h_last = torch.randn(4096)
    object_mask = torch.zeros(256)

    p, redist = adapter(h_last, h_vision=None, object_mask=object_mask)
    assert redist is None
    assert p.shape == (4, 32)


def test_blend_alpha_init():
    """blend_alpha should start near 0.018 (sigmoid(-4))."""
    from adapter_model import AttentionAdapterV2

    adapter = AttentionAdapterV2(hidden_dim=4096, vision_tokens=256)
    alpha = adapter.blend_alpha.item()
    assert 0.01 < alpha < 0.03, f"Expected ~0.018, got {alpha}"


def test_dual_encoder_512():
    """V=512 (Prismatic dual encoder) should work."""
    from adapter_model import AttentionAdapterV2

    adapter = AttentionAdapterV2(hidden_dim=4096, vision_tokens=512)
    h_last = torch.randn(4096)
    h_vision = torch.randn(512, 4096)
    object_mask = torch.zeros(512)
    object_mask[:20] = 1.0

    p, redist = adapter(h_last, h_vision, object_mask)
    assert p.shape == (4, 32)
    assert redist.shape == (512,)


def test_gradient_flows_both_branches():
    """Gradients must flow through both branches."""
    from adapter_model import AttentionAdapterV2

    adapter = AttentionAdapterV2(hidden_dim=4096, vision_tokens=256)
    h_last = torch.randn(4096, requires_grad=False)
    h_vision = torch.randn(256, 4096, requires_grad=False)
    object_mask = torch.zeros(256)
    object_mask[50:70] = 1.0

    p, redist = adapter(h_last, h_vision, object_mask)

    # Branch 1 gradient
    loss1 = p.mean()
    loss1.backward(retain_graph=True)
    grad_p_head = adapter.p_head[-1].weight.grad
    assert grad_p_head is not None and grad_p_head.abs().sum() > 0

    adapter.zero_grad()

    # Branch 2 gradient
    loss2 = redist[object_mask == 1].sum()
    loss2.backward()
    grad_key = adapter.key_proj.weight.grad
    assert grad_key is not None and grad_key.abs().sum() > 0
```

**Step 2: Run tests to verify they pass**

Run: `cd /home/kana5123/ATLASVLA && python -m pytest tests/test_adapter_model_v2.py -v`
Expected: ALL PASS (8 tests)

**Step 3: Commit**

```bash
git add tests/test_adapter_model_v2.py
git commit -m "test: add Branch 2 cross-attention and blend_alpha tests"
```

---

## Task 4: Modify apply_var to accept redistribution_weights

**Files:**
- Modify: `attention_v3.py:175-289` (apply_var function)
- Modify: `attention_v3.py:41-95` (V3Context dataclass)
- Test: `tests/test_apply_var_v2.py` (create)

**Step 1: Write the failing test**

```python
# tests/test_apply_var_v2.py
"""Tests for apply_var with redistribution_weights parameter."""
import torch


def _make_attn_weights(B=1, H=4, K=32, vision_end=16):
    """Create synthetic attention weights with a sink at index 0."""
    w = torch.rand(B, H, 1, K)
    # Make index 0 a heavy sink (50% of attention)
    w[:, :, :, 0] = 2.0
    w = w / w.sum(dim=-1, keepdim=True)  # normalize to sum=1
    return w


def test_apply_var_with_redistribution_weights():
    """When redistribution_weights are provided, freed attention goes to weighted patches."""
    from attention_v3 import apply_var

    B, H, K, V = 1, 4, 32, 16
    attn = _make_attn_weights(B, H, K, V)
    original_bg = attn[:, :, -1, 5].clone()  # background patch

    # redistribution_weights: only patches 10-12 get attention
    redist = torch.zeros(V)
    redist[10] = 0.5
    redist[11] = 0.3
    redist[12] = 0.2

    result = apply_var(
        attn, sink_indices=[0], vision_end=V, p=0.5, rho=0.0,
        redistribution_weights=redist,
    )

    # Background patch 5 should NOT increase (redist[5] == 0)
    # It should stay the same or decrease slightly
    assert result[0, 0, -1, 5].item() <= original_bg[0, 0].item() + 1e-6

    # Object patches should increase
    for idx in [10, 11, 12]:
        assert result[0, 0, -1, idx].item() > attn[0, 0, -1, idx].item()


def test_apply_var_redist_weights_override_proportional():
    """redistribution_weights should override proportional redistribution."""
    from attention_v3 import apply_var

    B, H, K, V = 1, 2, 20, 10
    attn = _make_attn_weights(B, H, K, V)

    # All weight on patch 5 only
    redist = torch.zeros(V)
    redist[5] = 1.0

    result = apply_var(
        attn, sink_indices=[0], vision_end=V, p=0.8, rho=0.0,
        redistribution_weights=redist,
    )

    # Patch 5 should get ALL freed attention
    # Other non-sink patches should get nothing extra
    for idx in range(1, V):
        if idx == 5:
            assert result[0, 0, -1, idx].item() > attn[0, 0, -1, idx].item()
        else:
            diff = abs(result[0, 0, -1, idx].item() - attn[0, 0, -1, idx].item())
            assert diff < 1e-5, f"Patch {idx} changed by {diff}"


def test_v3_context_has_redistribution_weights():
    """V3Context should have redistribution_weights field."""
    from attention_v3 import V3Context

    ctx = V3Context()
    assert hasattr(ctx, "redistribution_weights")
    assert ctx.redistribution_weights is None
```

**Step 2: Run test to verify it fails**

Run: `cd /home/kana5123/ATLASVLA && python -m pytest tests/test_apply_var_v2.py -v`
Expected: FAIL — `TypeError: apply_var() got an unexpected keyword argument 'redistribution_weights'`

**Step 3: Write minimal implementation**

**3a. Add `redistribution_weights` field to V3Context** (attention_v3.py line ~90):

After `per_head_var_strength` field (line 90), add:

```python
    # Per-patch redistribution weights from adapter v2 cross-attention
    redistribution_weights: Optional[object] = None  # torch.Tensor (V,) or None
```

**3b. Add `redistribution_weights` parameter to apply_var** (attention_v3.py):

Add parameter to function signature at line 184 (after `per_head_p`):

```python
    redistribution_weights: Optional[torch.Tensor] = None,
```

**3c. Modify redistribution logic** in apply_var (lines 248-278):

Replace the existing redistribution block (lines 248-278) with:

```python
    # Compute redistribution weights
    nonsink_vals = last[:, :, nonsink_t]  # (B, H, NS)

    if redistribution_weights is not None:
        # V2: Use learned per-patch weights from adapter cross-attention
        # redistribution_weights: (V,) — weights for ALL vision tokens
        # Extract only non-sink entries, matching nonsink_t ordering
        redist_for_nonsink = redistribution_weights[nonsink_t]  # (NS,)
        redist_for_nonsink = redist_for_nonsink.float().to(last.device)
        # Expand for (B, H, NS) broadcasting
        rw = redist_for_nonsink.unsqueeze(0).unsqueeze(0)  # (1, 1, NS)
        bonus = freed * rw  # (B, H, NS)
    else:
        # Original logic: proportional or object-weighted redistribution
        has_obj = object_indices and object_weight > 1.0
        has_extra = extra_boost_map and len(extra_boost_map) > 0
        non_sink_set = set(non_sink_visual)

        if has_obj or has_extra:
            weight_vec = torch.ones(len(non_sink_visual), device=last.device)
            if has_obj:
                obj_set = set(i for i in object_indices if i in non_sink_set)
                for idx, token_id in enumerate(non_sink_visual):
                    if token_id in obj_set:
                        weight_vec[idx] = object_weight
            if has_extra:
                for idx, token_id in enumerate(non_sink_visual):
                    if token_id in extra_boost_map:
                        weight_vec[idx] *= extra_boost_map[token_id]
            weighted_nonsink = nonsink_vals * weight_vec.unsqueeze(0).unsqueeze(0)
            weighted_sum = weighted_nonsink.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            bonus = freed * (weighted_nonsink / weighted_sum)
        else:
            nonsink_sum = nonsink_vals.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            bonus = freed * (nonsink_vals / nonsink_sum)
```

**3d. Pass redistribution_weights from context** in patched forward (line ~568-574):

Update the `apply_var` call in `_make_v3_patched_forward`:

```python
                attn_weights = apply_var(
                    attn_weights, ctx.var_sink_indices, ctx.vision_end,
                    ctx.effective_var_p(), ctx.var_rho,
                    object_indices=obj_idx, object_weight=obj_w,
                    extra_boost_map=extra_map,
                    per_head_p=ctx.get_per_head_p(layer_idx),
                    redistribution_weights=ctx.redistribution_weights,
                )
```

**Step 4: Run test to verify it passes**

Run: `cd /home/kana5123/ATLASVLA && python -m pytest tests/test_apply_var_v2.py -v`
Expected: ALL PASS (4 tests)

**Step 5: Run existing tests to check no regression**

Run: `cd /home/kana5123/ATLASVLA && python -m pytest tests/ -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add attention_v3.py tests/test_apply_var_v2.py
git commit -m "feat: add redistribution_weights parameter to apply_var and V3Context"
```

---

## Task 5: Create SAM preprocessing script — noun phrase extraction

**Files:**
- Create: `sam_preprocess.py`
- Test: `tests/test_sam_preprocess.py` (create)

**Step 1: Write the failing test**

```python
# tests/test_sam_preprocess.py
"""Tests for SAM preprocessing utilities."""


def test_extract_noun_phrases():
    """Extract noun phrases from robot instructions."""
    from sam_preprocess import extract_noun_phrases

    phrases = extract_noun_phrases("pick up the blue cup from the table")
    assert isinstance(phrases, list)
    assert len(phrases) > 0
    # Should contain "blue cup" or "cup" and "table"
    all_text = " ".join(phrases).lower()
    assert "cup" in all_text


def test_extract_noun_phrases_fallback():
    """If no nouns found, fall back to full instruction."""
    from sam_preprocess import extract_noun_phrases

    phrases = extract_noun_phrases("go")
    assert len(phrases) >= 1  # fallback to full instruction


def test_instruction_to_grounding_queries():
    """Convert instruction to GroundingDINO query strings."""
    from sam_preprocess import instruction_to_grounding_queries

    queries = instruction_to_grounding_queries("pick up the red block near the bowl")
    assert isinstance(queries, list)
    assert len(queries) > 0
```

**Step 2: Run test to verify it fails**

Run: `cd /home/kana5123/ATLASVLA && python -m pytest tests/test_sam_preprocess.py::test_extract_noun_phrases -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sam_preprocess'`

**Step 3: Write minimal implementation**

Create `sam_preprocess.py`:

```python
"""SAM2 + GroundingDINO preprocessing for object-aware adapter v2.

Produces object_masks.dat — per-step binary masks indicating which
vision grid patches correspond to task-relevant objects.

Pipeline:
    1. Extract noun phrases from instructions (spaCy)
    2. GroundingDINO(image, noun_phrases) → bounding boxes
    3. SAM2(image, boxes) → pixel masks (256×256)
    4. Map pixel masks to vision grid (16×16) → patch mask (V,)
    5. Store as memmap: object_masks.dat (total_steps, V) uint8

Usage:
    python sam_preprocess.py [--num_workers 4] [--vision_tokens 256]
"""

from __future__ import annotations

import spacy

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def extract_noun_phrases(instruction: str) -> list[str]:
    """Extract noun phrases from a robot instruction using spaCy.

    Args:
        instruction: e.g. "pick up the blue cup from the table"

    Returns:
        List of noun phrases, e.g. ["blue cup", "table"]
        Falls back to full instruction if no nouns found.
    """
    nlp = _get_nlp()
    doc = nlp(instruction)

    phrases = []
    for chunk in doc.noun_chunks:
        # Filter out pronouns and very short chunks
        text = chunk.text.strip()
        if len(text) > 1 and chunk.root.pos_ != "PRON":
            phrases.append(text)

    if not phrases:
        # Fallback: use full instruction
        phrases = [instruction.strip()]

    return phrases


def instruction_to_grounding_queries(instruction: str) -> list[str]:
    """Convert instruction to GroundingDINO query strings.

    GroundingDINO expects period-separated queries, e.g. "blue cup. table."
    We extract noun phrases and format them.

    Returns:
        List of individual query strings (each used separately for robustness).
    """
    return extract_noun_phrases(instruction)
```

**Step 4: Run test to verify it passes**

Run: `cd /home/kana5123/ATLASVLA && python -m pytest tests/test_sam_preprocess.py -v`
Expected: ALL PASS (3 tests)

**Step 5: Commit**

```bash
git add sam_preprocess.py tests/test_sam_preprocess.py
git commit -m "feat: add noun phrase extraction for GroundingDINO queries"
```

---

## Task 6: SAM preprocessing — pixel mask to vision grid mapping

**Files:**
- Modify: `sam_preprocess.py` (add function)
- Test: `tests/test_sam_preprocess.py` (append)

**Step 1: Write the failing test**

Append to `tests/test_sam_preprocess.py`:

```python
import numpy as np


def test_pixel_mask_to_patch_mask():
    """Convert 256×256 pixel mask to 16×16 vision grid patch mask."""
    from sam_preprocess import pixel_mask_to_patch_mask

    # Create a pixel mask with a 64×64 object in top-left corner
    pixel_mask = np.zeros((256, 256), dtype=np.uint8)
    pixel_mask[:64, :64] = 1  # covers patches (0,0) to (3,3) in 16×16 grid

    patch_mask = pixel_mask_to_patch_mask(pixel_mask, grid_size=16, threshold=0.1)
    assert patch_mask.shape == (256,), f"Expected (256,), got {patch_mask.shape}"

    # Top-left 4×4 = 16 patches should be 1
    grid_2d = patch_mask.reshape(16, 16)
    assert grid_2d[:4, :4].sum() == 16
    # Rest should be 0
    assert grid_2d[4:, :].sum() == 0
    assert grid_2d[:, 4:].sum() == 0  # except the 4×4 block


def test_pixel_mask_to_patch_mask_threshold():
    """Patches with partial overlap below threshold should be 0."""
    from sam_preprocess import pixel_mask_to_patch_mask

    pixel_mask = np.zeros((256, 256), dtype=np.uint8)
    # Only 1 pixel in patch (0,0) → 1/256 overlap < 0.1 threshold
    pixel_mask[0, 0] = 1

    patch_mask = pixel_mask_to_patch_mask(pixel_mask, grid_size=16, threshold=0.1)
    assert patch_mask[0] == 0, "Patch with <10% overlap should be 0"


def test_pixel_mask_to_patch_mask_512():
    """V=512 (dual encoder): should duplicate 256 mask to 512."""
    from sam_preprocess import pixel_mask_to_patch_mask

    pixel_mask = np.zeros((256, 256), dtype=np.uint8)
    pixel_mask[:128, :128] = 1

    patch_mask = pixel_mask_to_patch_mask(pixel_mask, grid_size=16, threshold=0.1, vision_tokens=512)
    assert patch_mask.shape == (512,)
    # First 256 and second 256 should be identical
    np.testing.assert_array_equal(patch_mask[:256], patch_mask[256:])
```

**Step 2: Run test to verify it fails**

Run: `cd /home/kana5123/ATLASVLA && python -m pytest tests/test_sam_preprocess.py::test_pixel_mask_to_patch_mask -v`
Expected: FAIL with `ImportError: cannot import name 'pixel_mask_to_patch_mask'`

**Step 3: Write minimal implementation**

Add to `sam_preprocess.py`:

```python
import numpy as np

import config


def pixel_mask_to_patch_mask(
    pixel_mask: np.ndarray,
    grid_size: int = config.VISION_GRID_SIZE,
    threshold: float = config.SAM_PATCH_OVERLAP_THRESHOLD,
    vision_tokens: int = 256,
) -> np.ndarray:
    """Convert pixel-level mask (H, W) to vision grid patch mask (V,).

    Divides the image into grid_size × grid_size cells. A cell is marked
    as 'object' if the fraction of masked pixels exceeds threshold.

    Args:
        pixel_mask: (H, W) binary mask from SAM2
        grid_size: number of patches per side (16 for OpenVLA)
        threshold: minimum overlap fraction to mark a patch
        vision_tokens: 256 (single encoder) or 512 (dual encoder)

    Returns:
        patch_mask: (vision_tokens,) uint8 array, 1=object, 0=background
    """
    H, W = pixel_mask.shape
    cell_h = H // grid_size
    cell_w = W // grid_size

    patch_mask_2d = np.zeros((grid_size, grid_size), dtype=np.uint8)

    for r in range(grid_size):
        for c in range(grid_size):
            cell = pixel_mask[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
            overlap = cell.sum() / (cell_h * cell_w)
            if overlap >= threshold:
                patch_mask_2d[r, c] = 1

    patch_mask = patch_mask_2d.flatten()  # (grid_size^2,) = (256,)

    # Dual encoder: duplicate mask for both encoders
    if vision_tokens > len(patch_mask):
        patch_mask = np.concatenate([patch_mask, patch_mask])

    return patch_mask[:vision_tokens]
```

**Step 4: Run test to verify it passes**

Run: `cd /home/kana5123/ATLASVLA && python -m pytest tests/test_sam_preprocess.py -v`
Expected: ALL PASS (6 tests)

**Step 5: Commit**

```bash
git add sam_preprocess.py tests/test_sam_preprocess.py
git commit -m "feat: add pixel_mask_to_patch_mask for vision grid mapping"
```

---

## Task 7: SAM preprocessing — GroundingDINO + SAM2 pipeline per image

**Files:**
- Modify: `sam_preprocess.py` (add core pipeline)
- Test: `tests/test_sam_preprocess.py` (append, with mocking)

**Step 1: Write the failing test**

Append to `tests/test_sam_preprocess.py`:

```python
from unittest.mock import MagicMock, patch


def test_process_single_image_returns_mask():
    """process_single_image should return (V,) uint8 mask or None on failure."""
    from sam_preprocess import process_single_image

    # Mock GroundingDINO and SAM2 — we test the pipeline logic, not the models
    mock_grounding = MagicMock()
    mock_grounding.return_value = {
        "boxes": np.array([[10, 20, 100, 150]]),  # (N, 4) xyxy
        "scores": np.array([0.8]),
    }

    mock_sam = MagicMock()
    mock_sam.return_value = np.zeros((256, 256), dtype=np.uint8)
    mock_sam.return_value[20:150, 10:100] = 1  # fill bbox region

    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    result = process_single_image(
        image, "pick up the cup",
        grounding_fn=mock_grounding,
        sam_fn=mock_sam,
        vision_tokens=256,
    )

    assert result is not None
    assert result.shape == (256,)
    assert result.dtype == np.uint8
    assert result.sum() > 0  # at least some object patches


def test_process_single_image_failure():
    """If GroundingDINO finds nothing, return None."""
    from sam_preprocess import process_single_image

    mock_grounding = MagicMock()
    mock_grounding.return_value = {"boxes": np.array([]).reshape(0, 4), "scores": np.array([])}
    mock_sam = MagicMock()

    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    result = process_single_image(
        image, "pick up the cup",
        grounding_fn=mock_grounding,
        sam_fn=mock_sam,
        vision_tokens=256,
    )
    assert result is None
```

**Step 2: Run test to verify it fails**

Run: `cd /home/kana5123/ATLASVLA && python -m pytest tests/test_sam_preprocess.py::test_process_single_image_returns_mask -v`
Expected: FAIL with `ImportError: cannot import name 'process_single_image'`

**Step 3: Write minimal implementation**

Add to `sam_preprocess.py`:

```python
from typing import Callable, Optional


def process_single_image(
    image: np.ndarray,
    instruction: str,
    grounding_fn: Callable,
    sam_fn: Callable,
    vision_tokens: int = 256,
    max_area_fraction: float = config.GROUNDING_MAX_AREA_FRACTION,
) -> Optional[np.ndarray]:
    """Process a single image through GroundingDINO + SAM2 pipeline.

    Args:
        image: (H, W, 3) uint8 image
        instruction: robot instruction text
        grounding_fn: callable(image, query) -> {"boxes": (N,4), "scores": (N,)}
        sam_fn: callable(image, boxes) -> (H, W) binary mask
        vision_tokens: 256 or 512

    Returns:
        patch_mask: (vision_tokens,) uint8 or None if detection fails
    """
    H, W = image.shape[:2]
    img_area = H * W

    # Extract noun phrases for grounding queries
    queries = instruction_to_grounding_queries(instruction)

    all_boxes = []
    all_scores = []

    for query in queries:
        try:
            result = grounding_fn(image, query)
            boxes = result["boxes"]
            scores = result["scores"]

            if len(boxes) == 0:
                continue

            # Filter large boxes (likely background detections)
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                box_area = (x2 - x1) * (y2 - y1)
                if box_area / img_area <= max_area_fraction:
                    all_boxes.append(box)
                    all_scores.append(scores[i])
        except Exception:
            continue

    if len(all_boxes) == 0:
        return None

    all_boxes = np.array(all_boxes)

    # Run SAM2 with detected boxes
    try:
        pixel_mask = sam_fn(image, all_boxes)
    except Exception:
        return None

    if pixel_mask is None or pixel_mask.sum() == 0:
        return None

    return pixel_mask_to_patch_mask(pixel_mask, vision_tokens=vision_tokens)
```

**Step 4: Run test to verify it passes**

Run: `cd /home/kana5123/ATLASVLA && python -m pytest tests/test_sam_preprocess.py -v`
Expected: ALL PASS (8 tests)

**Step 5: Commit**

```bash
git add sam_preprocess.py tests/test_sam_preprocess.py
git commit -m "feat: add process_single_image pipeline (GroundingDINO + SAM2)"
```

---

## Task 8: SAM preprocessing — batch processing + memmap output

**Files:**
- Modify: `sam_preprocess.py` (add batch processing + CLI)
- Test: `tests/test_sam_preprocess.py` (append)

**Step 1: Write the failing test**

```python
import tempfile
import os


def test_preprocess_batch_creates_memmap(tmp_path):
    """Batch processing should create object_masks.dat memmap."""
    from sam_preprocess import preprocess_all_steps

    # Create a tiny mock cache
    total_steps = 10
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Fake images.dat
    images = np.memmap(
        str(cache_dir / "images.dat"), dtype=np.uint8, mode="w+",
        shape=(total_steps, 256, 256, 3),
    )
    images[:] = np.random.randint(0, 255, (total_steps, 256, 256, 3), dtype=np.uint8)
    images.flush()

    # Fake metadata
    import pickle, json
    metadata = [
        {"global_idx": i, "instruction": "pick up the cup", "episode_id": i // 3, "step_id": i % 3, "action": [0]*7}
        for i in range(total_steps)
    ]
    with open(cache_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    with open(cache_dir / "cache_info.json", "w") as f:
        json.dump({"total_steps": total_steps, "image_height": 256, "image_width": 256, "total_episodes": 4}, f)

    # Mock grounding + SAM that always succeeds
    def mock_ground(img, query):
        return {"boxes": np.array([[50, 50, 150, 150]]), "scores": np.array([0.9])}
    def mock_sam(img, boxes):
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[50:150, 50:150] = 1
        return mask

    preprocess_all_steps(
        cache_dir=cache_dir,
        grounding_fn=mock_ground,
        sam_fn=mock_sam,
        vision_tokens=256,
    )

    # Check output
    masks_path = cache_dir / "object_masks.dat"
    assert masks_path.exists()

    masks = np.memmap(str(masks_path), dtype=np.uint8, mode="r", shape=(total_steps, 256))
    assert masks.shape == (total_steps, 256)
    # All steps should have valid masks (no failures with our mock)
    for i in range(total_steps):
        assert masks[i].max() <= 1  # binary mask, not failure marker (255)
        assert masks[i].sum() > 0   # mock always detects something
```

**Step 2: Run test to verify it fails**

Run: `cd /home/kana5123/ATLASVLA && python -m pytest tests/test_sam_preprocess.py::test_preprocess_batch_creates_memmap -v`
Expected: FAIL with `ImportError: cannot import name 'preprocess_all_steps'`

**Step 3: Write minimal implementation**

Add to `sam_preprocess.py`:

```python
import json
import pickle
import time
from pathlib import Path


def preprocess_all_steps(
    cache_dir: Path,
    grounding_fn: Callable,
    sam_fn: Callable,
    vision_tokens: int = 256,
) -> dict:
    """Run GroundingDINO + SAM2 on all cached images and save object_masks.dat.

    Args:
        cache_dir: Path containing images.dat, metadata.pkl, cache_info.json
        grounding_fn: callable(image, query) -> {"boxes": ..., "scores": ...}
        sam_fn: callable(image, boxes) -> (H, W) mask
        vision_tokens: 256 or 512

    Returns:
        dict with stats: {total, success, failure, failure_rate}
    """
    cache_dir = Path(cache_dir)

    with open(cache_dir / "cache_info.json") as f:
        info = json.load(f)
    total_steps = info["total_steps"]
    img_h, img_w = info["image_height"], info["image_width"]

    with open(cache_dir / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    images = np.memmap(
        str(cache_dir / "images.dat"), dtype=np.uint8, mode="r",
        shape=(total_steps, img_h, img_w, 3),
    )

    # Create output memmap
    masks_path = cache_dir / config.SAM_MASKS_FILENAME
    masks = np.memmap(
        str(masks_path), dtype=np.uint8, mode="w+",
        shape=(total_steps, vision_tokens),
    )
    # Initialize all to failure marker
    masks[:] = config.SAM_FAILURE_MARKER

    success_count = 0
    failure_count = 0
    t0 = time.time()

    for idx in range(total_steps):
        image = np.array(images[idx])
        instruction = metadata[idx]["instruction"]

        result = process_single_image(
            image, instruction,
            grounding_fn=grounding_fn,
            sam_fn=sam_fn,
            vision_tokens=vision_tokens,
        )

        if result is not None:
            masks[idx] = result
            success_count += 1
        else:
            failure_count += 1

        if (idx + 1) % 10000 == 0:
            masks.flush()
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            print(
                f"  [{idx + 1}/{total_steps}] "
                f"success={success_count}, fail={failure_count} "
                f"({rate:.0f} steps/s)"
            )

    masks.flush()
    del masks

    stats = {
        "total": total_steps,
        "success": success_count,
        "failure": failure_count,
        "failure_rate": failure_count / max(total_steps, 1),
    }
    print(f"SAM preprocessing complete: {stats}")
    return stats
```

**Step 4: Run test to verify it passes**

Run: `cd /home/kana5123/ATLASVLA && python -m pytest tests/test_sam_preprocess.py -v`
Expected: ALL PASS (9 tests)

**Step 5: Commit**

```bash
git add sam_preprocess.py tests/test_sam_preprocess.py
git commit -m "feat: add preprocess_all_steps for batch SAM mask generation"
```

---

## Task 9: Update adapter_data.py — load object masks + filter SAM-failed steps

**Files:**
- Modify: `adapter_data.py:285-336` (BridgeTfrecordDataset)
- Modify: `adapter_data.py:343-361` (split_episodes)
- Modify: `adapter_data.py:368-376` (adapter_collate_fn)
- Test: `tests/test_adapter_data_v2.py` (create)

**Step 1: Write the failing test**

```python
# tests/test_adapter_data_v2.py
"""Tests for v2 data pipeline with object masks."""
import json
import pickle
import tempfile
from pathlib import Path

import numpy as np


def _make_test_cache(tmp_path, total_steps=20, n_episodes=4, vision_tokens=256):
    """Create a minimal test cache with object masks."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    img_h, img_w = 256, 256

    # images.dat
    images = np.memmap(
        str(cache_dir / "images.dat"), dtype=np.uint8, mode="w+",
        shape=(total_steps, img_h, img_w, 3),
    )
    images[:] = 128
    images.flush()

    # metadata.pkl
    metadata = []
    for i in range(total_steps):
        ep_id = i * n_episodes // total_steps
        metadata.append({
            "global_idx": i,
            "instruction": "pick up the cup",
            "action": [0.0] * 7,
            "episode_id": ep_id,
            "step_id": i % 5,
        })
    with open(cache_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    # cache_info.json
    with open(cache_dir / "cache_info.json", "w") as f:
        json.dump({
            "total_steps": total_steps,
            "image_height": img_h,
            "image_width": img_w,
            "total_episodes": n_episodes,
        }, f)

    # object_masks.dat — some steps succeed, some fail (255)
    masks = np.memmap(
        str(cache_dir / "object_masks.dat"), dtype=np.uint8, mode="w+",
        shape=(total_steps, vision_tokens),
    )
    for i in range(total_steps):
        if i % 4 == 0:
            masks[i] = 255  # SAM failure marker
        else:
            masks[i] = 0
            masks[i, 50:60] = 1  # object patches
    masks.flush()

    (cache_dir / "DONE").touch()
    return cache_dir


def test_dataset_filters_sam_failures(tmp_path):
    """BridgeTfrecordDataset should exclude steps with SAM failure marker."""
    from adapter_data import BridgeTfrecordDataset

    cache_dir = _make_test_cache(tmp_path, total_steps=20, n_episodes=4)
    dataset = BridgeTfrecordDataset(
        cache_dir, episode_indices=list(range(4)), split="train", use_object_masks=True,
    )

    # 20 total, 5 have failure marker (indices 0, 4, 8, 12, 16) → 15 valid
    assert len(dataset) == 15

    # Check that returned items include object_mask
    item = dataset[0]
    assert "object_mask" in item
    assert item["object_mask"].shape == (256,)
    assert item["object_mask"].max() <= 1  # binary, not failure marker


def test_collate_includes_object_mask(tmp_path):
    """adapter_collate_fn should handle object_mask field."""
    from adapter_data import adapter_collate_fn

    batch = [
        {"image": None, "instruction": "a", "action": np.zeros(7),
         "episode_id": 0, "step_id": 0, "object_mask": np.zeros(256, dtype=np.uint8)},
        {"image": None, "instruction": "b", "action": np.zeros(7),
         "episode_id": 0, "step_id": 1, "object_mask": np.ones(256, dtype=np.uint8)},
    ]
    result = adapter_collate_fn(batch)
    assert "object_masks" in result
    assert result["object_masks"].shape == (2, 256)
```

**Step 2: Run test to verify it fails**

Run: `cd /home/kana5123/ATLASVLA && python -m pytest tests/test_adapter_data_v2.py::test_dataset_filters_sam_failures -v`
Expected: FAIL — `TypeError: BridgeTfrecordDataset.__init__() got an unexpected keyword argument 'use_object_masks'`

**Step 3: Write minimal implementation**

**3a. Modify BridgeTfrecordDataset.__init__** in `adapter_data.py` (lines 292-321):

Add `use_object_masks` parameter and filtering logic:

```python
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
        self.use_object_masks = use_object_masks

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

        # Open memmap (lazy, OS handles page caching — shared across processes)
        self.images_mmap = np.memmap(
            str(cache_dir / "images.dat"), dtype=np.uint8, mode="r",
            shape=(total_steps, img_h, img_w, 3),
        )

        # Object masks (v2)
        self.object_masks_mmap = None
        if use_object_masks:
            masks_path = cache_dir / config.SAM_MASKS_FILENAME
            if masks_path.exists():
                # Detect vision_tokens from file size
                file_bytes = masks_path.stat().st_size
                vision_tokens = file_bytes // total_steps
                self.object_masks_mmap = np.memmap(
                    str(masks_path), dtype=np.uint8, mode="r",
                    shape=(total_steps, vision_tokens),
                )
                # Filter out SAM-failed steps
                self.steps = [
                    m for m in self.steps
                    if self.object_masks_mmap[m["global_idx"]][0] != config.SAM_FAILURE_MARKER
                ]

        n_eps = len(set(m["episode_id"] for m in self.steps))
        print(f"  [{self.split}] {len(self.steps)} steps from {n_eps} episodes (cached)")
```

**3b. Modify __getitem__** to include object_mask:

```python
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
        }
        if self.object_masks_mmap is not None:
            result["object_mask"] = np.array(
                self.object_masks_mmap[item["global_idx"]], dtype=np.uint8,
            )
        return result
```

**3c. Update adapter_collate_fn** to handle object_mask:

```python
def adapter_collate_fn(batch: list[dict]) -> dict:
    """Custom collate: keep images as list, stack actions and masks."""
    result = {
        "images": [item["image"] for item in batch],
        "instructions": [item["instruction"] for item in batch],
        "actions": np.stack([item["action"] for item in batch]),  # (B, 7)
        "episode_ids": [item["episode_id"] for item in batch],
        "step_ids": [item["step_id"] for item in batch],
    }
    if "object_mask" in batch[0]:
        result["object_masks"] = np.stack([item["object_mask"] for item in batch])
    return result
```

**Step 4: Run test to verify it passes**

Run: `cd /home/kana5123/ATLASVLA && python -m pytest tests/test_adapter_data_v2.py -v`
Expected: ALL PASS (2 tests)

**Step 5: Run all tests for regression**

Run: `cd /home/kana5123/ATLASVLA && python -m pytest tests/ -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add adapter_data.py tests/test_adapter_data_v2.py
git commit -m "feat: add object mask loading and SAM-failure filtering to data pipeline"
```

---

## Task 10: Update split_episodes with episode-level SAM failure exclusion

**Files:**
- Modify: `adapter_data.py:343-361` (split_episodes)
- Modify: `adapter_data.py:383-461` (create_dataloaders)
- Test: `tests/test_adapter_data_v2.py` (append)

**Step 1: Write the failing test**

Append to `tests/test_adapter_data_v2.py`:

```python
def test_compute_valid_episodes(tmp_path):
    """Episodes with >50% SAM failure should be excluded."""
    from adapter_data import compute_valid_episodes

    cache_dir = _make_test_cache(tmp_path, total_steps=20, n_episodes=4)

    # Rewrite masks: episode 0 (steps 0-4) has 100% failures
    with open(cache_dir / "cache_info.json") as f:
        info = json.load(f)
    with open(cache_dir / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    masks = np.memmap(
        str(cache_dir / "object_masks.dat"), dtype=np.uint8, mode="r+",
        shape=(info["total_steps"], 256),
    )
    # Make episode 0 all-fail
    for m in metadata:
        if m["episode_id"] == 0:
            masks[m["global_idx"]] = 255
    masks.flush()

    valid = compute_valid_episodes(cache_dir, threshold=0.5)
    assert 0 not in valid, "Episode 0 should be excluded (100% failure)"
    assert len(valid) < 4
```

**Step 2: Run test to verify it fails**

Run: `cd /home/kana5123/ATLASVLA && python -m pytest tests/test_adapter_data_v2.py::test_compute_valid_episodes -v`
Expected: FAIL with `ImportError: cannot import name 'compute_valid_episodes'`

**Step 3: Write minimal implementation**

Add to `adapter_data.py` before `create_dataloaders`:

```python
def compute_valid_episodes(
    cache_dir: Path,
    threshold: float = config.SAM_EPISODE_FAILURE_THRESHOLD,
) -> list[int]:
    """Compute episodes where SAM failure rate is below threshold.

    Args:
        cache_dir: cache directory with object_masks.dat
        threshold: maximum failure rate per episode (0.5 = 50%)

    Returns:
        Sorted list of valid episode IDs
    """
    with open(cache_dir / "cache_info.json") as f:
        info = json.load(f)
    with open(cache_dir / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    masks_path = cache_dir / config.SAM_MASKS_FILENAME
    if not masks_path.exists():
        # No masks → all episodes valid (v1 fallback)
        return sorted(set(m["episode_id"] for m in metadata))

    total_steps = info["total_steps"]
    file_bytes = masks_path.stat().st_size
    vision_tokens = file_bytes // total_steps

    masks = np.memmap(
        str(masks_path), dtype=np.uint8, mode="r",
        shape=(total_steps, vision_tokens),
    )

    # Count per-episode failures
    episode_total: dict[int, int] = {}
    episode_fail: dict[int, int] = {}

    for m in metadata:
        ep = m["episode_id"]
        episode_total[ep] = episode_total.get(ep, 0) + 1
        if masks[m["global_idx"]][0] == config.SAM_FAILURE_MARKER:
            episode_fail[ep] = episode_fail.get(ep, 0) + 1

    valid = []
    for ep, total in episode_total.items():
        fail_rate = episode_fail.get(ep, 0) / total
        if fail_rate <= threshold:
            valid.append(ep)

    return sorted(valid)
```

**Step 4: Run test to verify it passes**

Run: `cd /home/kana5123/ATLASVLA && python -m pytest tests/test_adapter_data_v2.py -v`
Expected: ALL PASS (3 tests)

**Step 5: Commit**

```bash
git add adapter_data.py tests/test_adapter_data_v2.py
git commit -m "feat: add compute_valid_episodes for episode-level SAM failure filtering"
```

---

## Task 11: Update adapter_train.py — capture h_vision + forward_with_adapter v2

**Files:**
- Modify: `adapter_train.py:41-48` (imports)
- Modify: `adapter_train.py:56-163` (forward_with_adapter)
- Modify: `adapter_train.py:389-391` (adapter creation)
- Modify: `adapter_train.py:419-427` (dataloader creation)
- Modify: `adapter_train.py:478-513` (training loop)
- Test: `tests/test_forward_v2.py` (create)

**Step 1: Write the failing test**

```python
# tests/test_forward_v2.py
"""Tests for v2 forward_with_adapter modifications."""
import torch


def test_forward_with_adapter_v2_signature():
    """forward_with_adapter should accept object_mask parameter."""
    from adapter_train import forward_with_adapter
    import inspect

    sig = inspect.signature(forward_with_adapter)
    assert "object_mask" in sig.parameters, \
        f"Missing object_mask param. Params: {list(sig.parameters.keys())}"
```

**Step 2: Run test to verify it fails**

Run: `cd /home/kana5123/ATLASVLA && python -m pytest tests/test_forward_v2.py -v`
Expected: FAIL — `object_mask` not in signature

**Step 3: Write minimal implementation**

**3a. Update imports** in `adapter_train.py` (line 41):

Change:
```python
from adapter_model import AttentionAdapter
```
To:
```python
from adapter_model import AttentionAdapter, AttentionAdapterV2
```

**3b. Modify forward_with_adapter** signature and implementation:

Update the function signature (line 56-65) to add `object_mask`:

```python
def forward_with_adapter(
    model,
    adapter,
    processor,
    ctx: V3Context,
    image,
    instruction: str,
    target_token_ids: list[int],
    device: torch.device,
    object_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
```

Update the hook to capture BOTH last token AND all vision tokens (replace lines 83-107):

```python
    # ── Step 1: Forward through model to capture h_27 (no grad) ──
    captured_hidden = {}

    def capture_hook(module, args, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        captured_hidden["h_last"] = h[:, -1, :].detach()    # (1, hidden_dim)
        captured_hidden["h_vision"] = h[:, :ctx.vision_end, :].detach()  # (1, V, hidden_dim)

    if hasattr(model, "language_model"):
        layers = model.language_model.model.layers
    else:
        layers = model.model.layers

    hook = layers[config.ADAPTER_SOURCE_LAYER].register_forward_hook(capture_hook)

    with torch.no_grad():
        _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            use_cache=False,
        )
    hook.remove()
```

Update adapter call (replace lines 109-123):

```python
    # ── Step 2: Adapter produces p_matrix + redistribution_weights ──
    h_last = captured_hidden["h_last"].float()    # (1, 4096)
    h_vision = captured_hidden["h_vision"].float() # (1, V, 4096)

    if isinstance(adapter, AttentionAdapterV2) or (
        hasattr(adapter, 'module') and isinstance(adapter.module, AttentionAdapterV2)
    ):
        # V2: two-branch adapter
        mask_tensor = None
        if object_mask is not None:
            mask_tensor = object_mask.float().to(device)
            if mask_tensor.dim() == 1:
                mask_tensor = mask_tensor.unsqueeze(0)

        p_matrix, redist_raw = adapter(h_last, h_vision, mask_tensor)
        # p_matrix: (1, 4, 32), redist_raw: (1, V) or None

        # Blend learned redistribution with proportional
        if redist_raw is not None:
            raw_adapter = adapter.module if hasattr(adapter, 'module') else adapter
            blend = raw_adapter.blend_alpha  # scalar in [0, 1]

            # Proportional weights (uniform over non-sink vision tokens)
            V = ctx.vision_end
            prop_weights = torch.ones(1, V, device=device, dtype=torch.float32)
            sink_set = set(ctx.var_sink_indices)
            for si in sink_set:
                if si < V:
                    prop_weights[0, si] = 0.0
            prop_weights = prop_weights / prop_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)

            final_redist = blend * redist_raw + (1 - blend) * prop_weights  # (1, V)
            ctx.redistribution_weights = final_redist.squeeze(0)  # (V,)
        else:
            ctx.redistribution_weights = None
    else:
        # V1: original single-branch adapter
        p_matrix = adapter(h_last)  # (1, 4, 32)
        ctx.redistribution_weights = None

    # Map (4_target_layers, 32_heads) → full (32_layers, 32_heads)
    full_p = torch.zeros(
        config.NUM_LAYERS, config.NUM_HEADS, device=device, dtype=p_matrix.dtype,
    )
    _target_idx = torch.tensor(
        config.ADAPTER_TARGET_LAYERS, device=device,
    ).unsqueeze(1).expand(-1, config.NUM_HEADS)
    full_p = full_p.scatter(0, _target_idx, p_matrix[0])  # differentiable
    ctx.per_head_var_strength = full_p
```

**3c. Update adapter creation** in `train()` (around line 389-391):

Change:
```python
    adapter = AttentionAdapter(hidden_dim=hidden_dim)
```
To:
```python
    adapter = AttentionAdapterV2(
        hidden_dim=hidden_dim,
        vision_tokens=vision_end,
    )
```

**3d. Update dataloader creation** (around line 422-427):

Change:
```python
    train_loader, val_loader, _ = create_dataloaders(
        num_episodes=args.num_episodes,
        batch_size=per_gpu_bs,
        source="tfrecord",
        accelerator=accelerator,
    )
```
To:
```python
    train_loader, val_loader, _ = create_dataloaders(
        num_episodes=args.num_episodes,
        batch_size=per_gpu_bs,
        source="tfrecord",
        accelerator=accelerator,
        use_object_masks=True,
    )
```

This requires also updating `create_dataloaders` to accept and pass through `use_object_masks`:

```python
def create_dataloaders(
    num_episodes: Optional[int] = config.ADAPTER_NUM_TRAIN_EPISODES,
    batch_size: int = config.ADAPTER_BATCH_SIZE,
    num_workers: int = 0,
    source: str = "tfrecord",
    tfrecord_dir: Optional[Path] = None,
    accelerator=None,
    use_object_masks: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
```

And pass it to `BridgeTfrecordDataset`:
```python
        train_ds = BridgeTfrecordDataset(cache_dir, train_ids, split="train", use_object_masks=use_object_masks)
        val_ds = BridgeTfrecordDataset(cache_dir, val_ids, split="val", use_object_masks=use_object_masks)
        test_ds = BridgeTfrecordDataset(cache_dir, test_ids, split="test", use_object_masks=use_object_masks)
```

If `use_object_masks=True`, also filter episodes BEFORE splitting:

```python
        if use_object_masks:
            valid_episodes = compute_valid_episodes(cache_dir)
            train_ids, val_ids, test_ids = split_episodes(len(valid_episodes))
            # Map back to actual episode IDs
            train_ids = [valid_episodes[i] for i in train_ids]
            val_ids = [valid_episodes[i] for i in val_ids]
            test_ids = [valid_episodes[i] for i in test_ids]
        else:
            train_ids, val_ids, test_ids = split_episodes(actual_episodes)
```

**3e. Update training loop** to pass object_mask (around line 484-498):

```python
            for i in range(local_bs):
                image = batch["images"][i]
                instruction = batch["instructions"][i]
                gt_action = batch["actions"][i]
                if isinstance(gt_action, torch.Tensor):
                    gt_action = gt_action.cpu().numpy()

                # Object mask (v2)
                obj_mask = None
                if "object_masks" in batch:
                    obj_mask = torch.from_numpy(batch["object_masks"][i]).to(device)

                target_tokens = tokenizer.action_to_token_ids(gt_action)

                loss_i, p_matrix = forward_with_adapter(
                    model, adapter,
                    processor, ctx, image, instruction, target_tokens, device,
                    object_mask=obj_mask,
                )
```

**Step 4: Run test to verify it passes**

Run: `cd /home/kana5123/ATLASVLA && python -m pytest tests/test_forward_v2.py -v`
Expected: PASS

**Step 5: Run all tests**

Run: `cd /home/kana5123/ATLASVLA && python -m pytest tests/ -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add adapter_train.py adapter_data.py tests/test_forward_v2.py
git commit -m "feat: update training loop for v2 adapter (h_vision capture, object_mask, blending)"
```

---

## Task 12: Update evaluate() function for v2

**Files:**
- Modify: `adapter_train.py:170-273` (evaluate function)

**Step 1: Update evaluate() to handle v2 adapter**

The evaluate function needs the same v2 changes as forward_with_adapter: capture h_vision, pass object_mask, use AttentionAdapterV2, set redistribution_weights.

Key changes:
- Capture hook captures both h_last and h_vision
- Use v2 adapter forward (p_matrix, redist)
- Blend redistribution weights
- Set ctx.redistribution_weights

Replace the eval function body's capture section with the same pattern as forward_with_adapter.

**Step 2: Run all tests**

Run: `cd /home/kana5123/ATLASVLA && python -m pytest tests/ -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add adapter_train.py
git commit -m "feat: update evaluate() for v2 adapter support"
```

---

## Task 13: Update save/load checkpoint for v2

**Files:**
- Modify: `adapter_train.py:280-333` (save_checkpoint, load_checkpoint)

**Step 1: Add v2 config to checkpoint metadata**

In `save_checkpoint`, add v2-specific config to the saved dict:

```python
        "config": {
            "lr": config.ADAPTER_LR,
            "num_target_layers": config.ADAPTER_NUM_TARGET_LAYERS,
            "target_layers": config.ADAPTER_TARGET_LAYERS,
            "source_layer": config.ADAPTER_SOURCE_LAYER,
            "l1_lambda": config.ADAPTER_L1_LAMBDA,
            # v2
            "adapter_version": 2,
            "query_dim": config.ADAPTER_V2_QUERY_DIM,
            "temperature": config.ADAPTER_V2_TEMPERATURE,
            "blend_init": config.ADAPTER_V2_BLEND_INIT,
            "mask_dim": config.ADAPTER_V2_MASK_DIM,
        },
```

**Step 2: Commit**

```bash
git add adapter_train.py
git commit -m "feat: add v2 metadata to checkpoint saving"
```

---

## Task 14: Add SAM preprocessing CLI entry point

**Files:**
- Modify: `sam_preprocess.py` (add `__main__` block with model loading)

**Step 1: Add CLI with argparse**

Add to end of `sam_preprocess.py`:

```python
def load_grounding_and_sam(device: str = "cuda"):
    """Load GroundingDINO and SAM2 models.

    Returns:
        (grounding_fn, sam_fn) callables
    """
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    # GroundingDINO
    gd_processor = AutoProcessor.from_pretrained(config.GROUNDING_MODEL_ID)
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        config.GROUNDING_MODEL_ID
    ).to(device)

    def grounding_fn(image: np.ndarray, query: str) -> dict:
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(image)
        inputs = gd_processor(images=pil_img, text=query, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = gd_model(**inputs)
        results = gd_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=config.GROUNDING_BOX_THRESHOLD,
            text_threshold=config.GROUNDING_TEXT_THRESHOLD,
            target_sizes=[pil_img.size[::-1]],
        )[0]
        return {
            "boxes": results["boxes"].cpu().numpy(),
            "scores": results["scores"].cpu().numpy(),
        }

    # SAM2
    sam_predictor = SAM2ImagePredictor.from_pretrained(config.SAM2_MODEL_ID)
    sam_predictor.model.to(device)

    def sam_fn(image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        masks, _, _ = sam_predictor.predict(
            box=boxes, multimask_output=False,
        )
        # Union all masks
        combined = np.zeros(image.shape[:2], dtype=np.uint8)
        for mask in masks:
            if mask.ndim == 3:
                mask = mask[0]
            combined = np.maximum(combined, mask.astype(np.uint8))
        return combined

    return grounding_fn, sam_fn


if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser(description="SAM preprocessing for adapter v2")
    parser.add_argument("--cache_dir", type=str, default=str(config.DATA_CACHE_DIR))
    parser.add_argument("--vision_tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("Loading GroundingDINO + SAM2...")
    grounding_fn, sam_fn = load_grounding_and_sam(args.device)

    print("Starting SAM preprocessing...")
    stats = preprocess_all_steps(
        cache_dir=Path(args.cache_dir),
        grounding_fn=grounding_fn,
        sam_fn=sam_fn,
        vision_tokens=args.vision_tokens,
    )
    print(f"Done: {stats}")
```

**Step 2: Commit**

```bash
git add sam_preprocess.py
git commit -m "feat: add SAM preprocessing CLI entry point"
```

---

## Task 15: Update logging to include v2 metrics

**Files:**
- Modify: `adapter_train.py` (logging section, ~line 524-548)

**Step 1: Add blend_alpha and redistribution stats to logging**

In the logging block (where `log_entry` is built), add:

```python
                raw = accelerator.unwrap_model(adapter)
                if hasattr(raw, 'blend_alpha'):
                    log_entry["blend_alpha"] = raw.blend_alpha.item()

                print(
                    f"Step {global_step:6d} | "
                    f"Loss {batch_loss_value:.4f} | "
                    f"GradNorm {log_entry['grad_norm']:.4f} | "
                    f"LR {lr:.2e} | "
                    f"MeanP {mean_p:.4f} | "
                    f"Active {active:.2%}"
                    + (f" | Blend {log_entry['blend_alpha']:.4f}" if 'blend_alpha' in log_entry else "")
                )
```

**Step 2: Commit**

```bash
git add adapter_train.py
git commit -m "feat: add blend_alpha to training logs"
```

---

## Task 16: End-to-end integration test (smoke test)

**Files:**
- Test: `tests/test_integration_v2.py` (create)

**Step 1: Write integration test**

```python
# tests/test_integration_v2.py
"""Integration smoke test for v2 adapter pipeline."""
import torch
import numpy as np


def test_full_adapter_v2_pipeline():
    """Test: adapter → apply_var with redistribution_weights → loss."""
    from adapter_model import AttentionAdapterV2
    from attention_v3 import apply_var

    # Simulate
    hidden_dim = 128  # small for test speed
    V = 16
    H = 4
    K = V + 10  # vision + text tokens

    adapter = AttentionAdapterV2(
        hidden_dim=hidden_dim,
        vision_tokens=V,
        query_dim=32,
        mask_dim=16,
        intermediate_dim=64,
    )

    h_last = torch.randn(hidden_dim)
    h_vision = torch.randn(V, hidden_dim)
    object_mask = torch.zeros(V)
    object_mask[5:10] = 1.0

    # Forward through adapter
    p_matrix, redist_weights = adapter(h_last, h_vision, object_mask)

    # Create synthetic attention weights
    attn_weights = torch.rand(1, H, 1, K)
    attn_weights[:, :, :, 0] = 2.0  # sink at 0
    attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)

    # Apply VAR with redistribution weights
    per_head_p = p_matrix[0]  # (4, H) → use first layer
    result = apply_var(
        attn_weights,
        sink_indices=[0],
        vision_end=V,
        p=0.5,
        rho=0.0,
        per_head_p=per_head_p[0],  # first target layer, (H,)
        redistribution_weights=redist_weights,
    )

    assert result.shape == attn_weights.shape
    # sum should still be ~1
    row_sum = result[0, 0, -1, :].sum().item()
    assert abs(row_sum - 1.0) < 0.01, f"Row sum should be ~1.0, got {row_sum}"

    # Gradient test: loss on result should flow back to adapter
    loss = result[0, 0, -1, 5:10].sum()  # object patch attention
    loss.backward()

    # Check gradient exists on adapter parameters
    assert adapter.key_proj.weight.grad is not None
    assert adapter.p_head[-1].weight.grad is not None
```

**Step 2: Run test**

Run: `cd /home/kana5123/ATLASVLA && python -m pytest tests/test_integration_v2.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_integration_v2.py
git commit -m "test: add end-to-end v2 adapter integration smoke test"
```

---

## Task 17: Run full test suite and verify

**Step 1: Run all tests**

Run: `cd /home/kana5123/ATLASVLA && python -m pytest tests/ -v --tb=short`
Expected: ALL PASS

**Step 2: Verify parameter count matches design**

Run:
```python
cd /home/kana5123/ATLASVLA && python -c "
from adapter_model import AttentionAdapterV2
a = AttentionAdapterV2(hidden_dim=4096, vision_tokens=256)
print(f'Total params: {a.param_count():,}')
print(f'Expected: ~2.17M')
"
```
Expected: ~2.17M parameters

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: Object-Aware Adapter v2 implementation complete"
```

---

## Summary of Files

| File | Action | Description |
|------|--------|-------------|
| `config.py` | Modify | Add v2 config constants |
| `adapter_model.py` | Modify | Add `AttentionAdapterV2` class |
| `attention_v3.py` | Modify | Add `redistribution_weights` to `apply_var` and `V3Context` |
| `sam_preprocess.py` | Create | SAM2+GroundingDINO preprocessing pipeline |
| `adapter_data.py` | Modify | Object mask loading, SAM-failure filtering, episode exclusion |
| `adapter_train.py` | Modify | h_vision capture, forward_with_adapter v2, blending logic |
| `tests/__init__.py` | Create | Test package init |
| `tests/test_config.py` | Create | Config constant tests |
| `tests/test_adapter_model_v2.py` | Create | AttentionAdapterV2 unit tests |
| `tests/test_apply_var_v2.py` | Create | apply_var with redistribution_weights tests |
| `tests/test_sam_preprocess.py` | Create | SAM preprocessing tests |
| `tests/test_adapter_data_v2.py` | Create | Data pipeline v2 tests |
| `tests/test_forward_v2.py` | Create | forward_with_adapter v2 signature test |
| `tests/test_integration_v2.py` | Create | End-to-end smoke test |

## Execution Order

```
Task 1  → config.py (v2 constants)
Task 2  → adapter_model.py (Branch 1 + 2 + blend_alpha)
Task 3  → tests for Branch 2 + gradients
Task 4  → attention_v3.py (redistribution_weights in apply_var)
Task 5  → sam_preprocess.py (noun phrase extraction)
Task 6  → sam_preprocess.py (pixel→patch mask mapping)
Task 7  → sam_preprocess.py (per-image pipeline)
Task 8  → sam_preprocess.py (batch processing + memmap)
Task 9  → adapter_data.py (mask loading + filtering)
Task 10 → adapter_data.py (episode-level exclusion)
Task 11 → adapter_train.py (forward_with_adapter v2)
Task 12 → adapter_train.py (evaluate v2)
Task 13 → adapter_train.py (checkpoint v2)
Task 14 → sam_preprocess.py (CLI entry point)
Task 15 → adapter_train.py (logging v2)
Task 16 → integration smoke test
Task 17 → full test suite + param count verification
```

Tasks 1-4 can be done independently. Tasks 5-8 (SAM pipeline) are independent from 1-4. Tasks 9-13 depend on all prior tasks. Tasks 14-17 are final polish.
