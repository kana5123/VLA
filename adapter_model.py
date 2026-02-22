"""Differentiable Attention Adapter — learnable VAR redistribution.

A small MLP that takes hidden states from layer 27 and outputs per-head
VAR redistribution strengths (p values) for the last 4 layers (28-31).

Architecture:
    Input:  h_27 ∈ ℝ^4096  (last token hidden state from layer 27)
    Output: p ∈ [0, 1]^(4 × 32)  (per-head VAR strength for layers 28-31)

Initialization:
    Output layer is zero-initialized (LoRA principle) so the adapter
    starts as identity (p=0.5 after sigmoid of 0, but we use a bias trick
    to start near 0 — see _init_output_near_zero).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class AttentionAdapter(nn.Module):
    """Learnable per-head VAR strength predictor.

    Given the hidden state from a specific layer, outputs a (num_target_layers, num_heads)
    matrix of redistribution strengths in [0, 1].
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        num_target_layers: int = config.ADAPTER_NUM_TARGET_LAYERS,
        num_heads: int = config.NUM_HEADS,
        intermediate_dim: int = 256,
        dropout: float = config.ADAPTER_DROPOUT,
    ):
        super().__init__()
        self.num_target_layers = num_target_layers
        self.num_heads = num_heads
        output_dim = num_target_layers * num_heads  # 4 × 32 = 128

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, intermediate_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim // 2, output_dim),
        )

        # Initialize output layer so sigmoid output starts near 0
        # sigmoid(-4) ≈ 0.018, so the adapter starts with near-zero modification
        self._init_output_near_zero()

    def _init_output_near_zero(self):
        """Small-random weights + negative bias → sigmoid ≈ 0 at start.

        Uses small random weights (std=0.01) instead of zero to allow
        gradient flow to earlier layers. Zero weights block backprop:
        dL/dx = W^T @ dL/dy = 0 when W=0, starving all upstream layers.
        With std=0.01: output ≈ N(0, 0.01)*x + (-4) ≈ -4 ± noise,
        so sigmoid still ≈ 0.018 (near-identity start preserved).
        """
        last_linear = self.net[-1]
        nn.init.normal_(last_linear.weight, std=0.01)
        nn.init.constant_(last_linear.bias, -4.0)  # sigmoid(-4) ≈ 0.018

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Predict per-head VAR strengths.

        Args:
            hidden_state: (B, hidden_dim) or (hidden_dim,)

        Returns:
            p_matrix: (B, num_target_layers, num_heads) in [0, 1]
                      or (num_target_layers, num_heads) if input is 1-D
        """
        squeeze = hidden_state.dim() == 1
        if squeeze:
            hidden_state = hidden_state.unsqueeze(0)

        logits = self.net(hidden_state)                 # (B, L*H)
        p = torch.sigmoid(logits)                       # (B, L*H) in [0,1]
        p = p.view(-1, self.num_target_layers, self.num_heads)  # (B, L, H)

        if squeeze:
            p = p.squeeze(0)  # (L, H)

        return p

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def sparsity_stats(self, p_matrix: torch.Tensor) -> dict:
        """Compute sparsity statistics of adapter output.

        Args:
            p_matrix: (B, L, H) or (L, H) adapter output

        Returns:
            dict with mean_p, active_ratio (p > 0.1), max_p
        """
        with torch.no_grad():
            flat = p_matrix.flatten().float()
            return {
                "mean_p": flat.mean().item(),
                "active_ratio": (flat > 0.1).float().mean().item(),
                "max_p": flat.max().item(),
                "min_p": flat.min().item(),
            }


class AdapterWithHook:
    """Wrapper that hooks into a frozen OpenVLA model to extract hidden states
    from a specified layer and feed them to the adapter.

    Usage:
        hook = AdapterWithHook(adapter, model, source_layer=27)
        hook.install()
        # ... run forward pass ...
        p_matrix = hook.get_p_matrix()  # (num_target_layers, num_heads)
        hook.remove()
    """

    def __init__(
        self,
        adapter: AttentionAdapter,
        model: nn.Module,
        source_layer: int = config.ADAPTER_SOURCE_LAYER,
    ):
        self.adapter = adapter
        self.model = model
        self.source_layer = source_layer
        self._hook = None
        self._hidden_state: torch.Tensor | None = None
        self._p_matrix: torch.Tensor | None = None

    def _hook_fn(self, module, args, output):
        """Capture hidden state from the target layer's output."""
        # LLaMA layer output: (hidden_states, self_attn_weights, present_key_value)
        if isinstance(output, tuple):
            hidden = output[0]  # (B, seq_len, hidden_dim)
        else:
            hidden = output

        # Take the last token's hidden state
        last_token_hidden = hidden[:, -1, :]  # (B, hidden_dim)

        # Detach from the frozen model's graph, but keep on device
        # The adapter creates its own computation graph
        self._hidden_state = last_token_hidden.detach().requires_grad_(False)

        # Run adapter (this creates the gradient-tracked graph)
        self._p_matrix = self.adapter(self._hidden_state.float())

    def install(self):
        """Register forward hook on the source layer."""
        if hasattr(self.model, "language_model"):
            layers = self.model.language_model.model.layers
        else:
            layers = self.model.model.layers
        target = layers[self.source_layer]
        self._hook = target.register_forward_hook(self._hook_fn)

    def remove(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def get_p_matrix(self) -> torch.Tensor | None:
        """Get the most recent adapter output (requires_grad=True)."""
        return self._p_matrix

    def get_hidden_state(self) -> torch.Tensor | None:
        return self._hidden_state


# ── AttentionAdapterV2 ─────────────────────────────────────────────────────


class AttentionAdapterV2(nn.Module):
    """Object-Aware Attention Adapter v2.

    Two-branch architecture:
        Branch 1 (p_head): per-head redistribution strengths p in [0,1]^(L x H).
            Consumes h_last (language hidden state) *and* an aggregated object mask
            embedding for richer context.
        Branch 2 (redistribution weights): learned cross-attention that tells
            *where* freed sink attention should go among vision tokens.

    A learnable ``blend_alpha`` (scalar in [0,1]) controls smooth transition
    from proportional redistribution to the learned weights during training.

    Args:
        hidden_dim: Dimension of language hidden states (default 4096).
        num_target_layers: Number of decoder layers where VAR is applied.
        num_heads: Number of attention heads per layer.
        intermediate_dim: Width of the hidden layers in the p_head MLP.
        mask_dim: Projection dimension for the object mask summary.
        query_dim: Projection dimension for cross-attention query/keys.
        temperature: Softmax temperature in cross-attention.
        blend_init: Initial logit for blend_alpha (sigmoid(blend_init) ~ 0.018).
        dropout: Dropout rate inside the p_head MLP.
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        num_target_layers: int = config.ADAPTER_NUM_TARGET_LAYERS,
        num_heads: int = config.NUM_HEADS,
        intermediate_dim: int = config.ADAPTER_INTERMEDIATE_DIM,
        mask_dim: int = config.ADAPTER_V2_MASK_DIM,
        query_dim: int = config.ADAPTER_V2_QUERY_DIM,
        temperature: float = config.ADAPTER_V2_TEMPERATURE,
        blend_init: float = config.ADAPTER_V2_BLEND_INIT,
        dropout: float = config.ADAPTER_DROPOUT,
        vision_tokens: int = 256,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_target_layers = num_target_layers
        self.num_heads = num_heads
        self.query_dim = query_dim
        self.temperature = temperature
        self.vision_tokens = vision_tokens
        output_dim = num_target_layers * num_heads  # 4 x 32 = 128

        # ── Branch 1: per-head p ──────────────────────────────────────────
        # h_last -> intermediate_dim
        self.h_proj = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.SiLU(),
        )
        # object_mask (V,) -> mask_dim
        self.mask_linear = nn.Linear(vision_tokens, mask_dim)

        # Concat (intermediate_dim + mask_dim) -> output_dim
        concat_dim = intermediate_dim + mask_dim
        self.p_head = nn.Sequential(
            nn.Linear(concat_dim, output_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )
        # Zero-init last layer for near-zero start
        self._init_p_head_near_zero()

        # ── Branch 2: redistribution weights via cross-attention ──────────
        self.query_proj = nn.Linear(hidden_dim, query_dim)
        self.key_proj = nn.Linear(hidden_dim, query_dim)
        # Xavier init for query/key projections
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.zeros_(self.key_proj.bias)

        # ── Blend alpha ───────────────────────────────────────────────────
        self._blend_logit = nn.Parameter(torch.tensor(blend_init))

    # -- Initialization helpers -------------------------------------------

    def _init_p_head_near_zero(self):
        """Small-random weights + negative bias -> sigmoid ~ 0 at start.

        Uses small random weights (std=0.01) instead of zero to allow
        gradient flow to earlier layers (h_proj, mask_linear, p_head.0).
        Zero weights block backprop: dL/dx = W^T @ dL/dy = 0 when W=0.
        """
        last_linear = self.p_head[-1]
        nn.init.normal_(last_linear.weight, std=0.01)
        nn.init.constant_(last_linear.bias, -4.0)  # sigmoid(-4) ~ 0.018

    def _get_mask_linear(self) -> nn.Linear:
        """Return the mask projection layer."""
        return self.mask_linear

    # -- Properties -------------------------------------------------------

    @property
    def blend_alpha(self) -> torch.Tensor:
        """Blend factor in [0, 1], starts at sigmoid(-4) ~ 0.018."""
        return torch.sigmoid(self._blend_logit)

    # -- Forward ----------------------------------------------------------

    def forward(
        self,
        h_last: torch.Tensor,
        h_vision: torch.Tensor | None = None,
        object_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute per-head p and (optionally) redistribution weights.

        Args:
            h_last: Language hidden state, (hidden_dim,) or (B, hidden_dim).
            h_vision: Vision hidden states, (V, hidden_dim) or (B, V, hidden_dim).
                      If None, Branch 2 returns None.
            object_mask: Binary object mask, (V,) or (B, V).
                         If None, treated as all-ones (all object).

        Returns:
            p_matrix: Per-head redistribution strengths in [0,1].
                      Shape (num_target_layers, num_heads) or (B, ...).
            redistribution_weights: Softmax weights over vision tokens,
                      shape (V,) or (B, V), or None if h_vision is None.
        """
        squeeze = h_last.dim() == 1
        if squeeze:
            h_last = h_last.unsqueeze(0)  # (1, hidden_dim)

        B = h_last.shape[0]

        # ── Branch 1: per-head p ──────────────────────────────────────────
        h_emb = self.h_proj(h_last)  # (B, intermediate_dim)

        # Object mask embedding
        if object_mask is not None:
            if object_mask.dim() == 1:
                mask_input = object_mask.unsqueeze(0).expand(B, -1)  # (B, V)
            else:
                mask_input = object_mask  # (B, V)
            mask_emb = F.silu(self.mask_linear(mask_input.float()))  # (B, mask_dim)
        else:
            # No mask provided: use zeros as mask embedding
            mask_dim = self.mask_linear.out_features
            mask_emb = torch.zeros(B, mask_dim, device=h_last.device, dtype=h_last.dtype)

        concat = torch.cat([h_emb, mask_emb], dim=-1)  # (B, intermediate_dim + mask_dim)
        p_logits = self.p_head(concat)  # (B, L*H)
        p_matrix = torch.sigmoid(p_logits)  # (B, L*H) in [0, 1]
        p_matrix = p_matrix.view(B, self.num_target_layers, self.num_heads)  # (B, L, H)

        # ── Branch 2: redistribution weights ──────────────────────────────
        redistribution_weights: torch.Tensor | None = None

        if h_vision is not None:
            if h_vision.dim() == 2:
                h_vision = h_vision.unsqueeze(0).expand(B, -1, -1)  # (B, V, hidden_dim)

            V = h_vision.shape[1]

            query = self.query_proj(h_last)    # (B, query_dim)
            keys = self.key_proj(h_vision)     # (B, V, query_dim)

            # Scaled dot-product: (B, V)
            scores = torch.einsum("bd,bvd->bv", query, keys)
            scores = scores / (self.query_dim ** 0.5 * self.temperature)

            # Mask out background patches
            if object_mask is not None:
                if object_mask.dim() == 1:
                    obj_mask = object_mask.unsqueeze(0).expand(B, -1)  # (B, V)
                else:
                    obj_mask = object_mask
                scores = scores.masked_fill(obj_mask == 0, float("-inf"))
            # If no mask, all patches are treated as object (no masking)

            redistribution_weights = torch.softmax(scores, dim=-1)  # (B, V)

            # Handle NaN from all-masked rows (shouldn't happen in practice)
            redistribution_weights = redistribution_weights.nan_to_num(0.0)

            if squeeze:
                redistribution_weights = redistribution_weights.squeeze(0)  # (V,)

        # ── Unsqueeze if needed ───────────────────────────────────────────
        if squeeze:
            p_matrix = p_matrix.squeeze(0)  # (L, H)

        return p_matrix, redistribution_weights

    # -- Utility methods --------------------------------------------------

    def param_count(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    def sparsity_stats(
        self,
        p_matrix: torch.Tensor,
        redistribution_weights: torch.Tensor | None = None,
    ) -> dict:
        """Compute sparsity statistics of adapter output.

        Args:
            p_matrix: (B, L, H) or (L, H) adapter output.
            redistribution_weights: (B, V) or (V,) or None.

        Returns:
            dict with mean_p, active_ratio (p > 0.1), max_p, min_p, blend_alpha.
        """
        with torch.no_grad():
            flat = p_matrix.flatten().float()
            stats = {
                "mean_p": flat.mean().item(),
                "active_ratio": (flat > 0.1).float().mean().item(),
                "max_p": flat.max().item(),
                "min_p": flat.min().item(),
                "blend_alpha": self.blend_alpha.item(),
            }
            if redistribution_weights is not None:
                rw = redistribution_weights.flatten().float()
                stats["redist_max"] = rw.max().item()
                stats["redist_entropy"] = (
                    -(rw * (rw + 1e-10).log()).sum().item()
                )
            return stats
