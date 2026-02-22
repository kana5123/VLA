"""Comprehensive gradient flow verification for the attention adapter.

Tests that loss.backward() produces non-zero gradients in ALL adapter
parameters, verifying the full chain:

    adapter.forward() → p_matrix → scatter → ctx.per_head_var_strength
    → apply_var (differentiable mode) → modified attn → loss → backward

This is the C6 critical issue: if gradients are zero, training is meaningless.

Run:
    pytest tests/test_gradient_flow.py -v
    # Or standalone:
    python tests/test_gradient_flow.py
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapter_model import AttentionAdapter, AttentionAdapterV2
from attention_v3 import (
    V3Context,
    apply_var,
    set_var_differentiable,
    set_v3_context,
)


# ── Test helpers ──────────────────────────────────────────────────────

def _make_synthetic_attn(B=1, H=8, seq_len=300, vision_end=256):
    """Create realistic synthetic attention weights (post-softmax).

    Mimics a VLA model's attention pattern with sink token 0 absorbing
    disproportionate attention.
    """
    attn = torch.rand(B, H, seq_len, seq_len)
    # Make token 0 a strong sink
    attn[:, :, :, 0] = attn[:, :, :, 0] * 5
    # Normalize to sum-to-1 (softmax-like)
    attn = attn / attn.sum(dim=-1, keepdim=True)
    return attn


def _redistribution_sensitive_loss(modified_attn, vision_end):
    """Compute a loss that's sensitive to WHERE attention is redistributed.

    VAR moves attention from sink (token 0) to non-sink vision tokens.
    A simple sum(:vision_end) is INVARIANT to this redistribution because
    the total stays constant. Instead, we use a weighted sum where non-sink
    tokens have higher weight, simulating how a real model's logits depend
    on which specific tokens receive attention.
    """
    last_row = modified_attn[:, :, -1, :vision_end]  # (B, H, V)
    # Weights: token 0 (sink) gets weight 0, non-sink get weight proportional to position
    weights = torch.arange(vision_end, dtype=last_row.dtype, device=last_row.device).float()
    weights[0] = 0.0  # Sink has zero weight — loss only cares about non-sink
    weights = weights / weights.sum()  # Normalize
    return (last_row * weights).sum()


def _collect_grad_info(module):
    """Collect gradient status for all parameters."""
    info = {}
    for name, param in module.named_parameters():
        if param.grad is not None:
            info[name] = {
                "has_grad": True,
                "grad_norm": param.grad.norm().item(),
                "grad_mean": param.grad.mean().item(),
                "grad_max": param.grad.abs().max().item(),
                "param_norm": param.norm().item(),
            }
        else:
            info[name] = {"has_grad": False}
    return info


# ── V1 Adapter Tests ─────────────────────────────────────────────────

def test_v1_gradient_through_apply_var():
    """V1 adapter: gradient flows through apply_var back to adapter params."""
    set_var_differentiable(enabled=True, temperature=10.0)

    hidden_dim, num_heads, num_layers = 128, 8, 4
    target_layers = [4, 5, 6, 7]  # last 4 of 8 total
    vision_end = 32

    adapter = AttentionAdapter(
        hidden_dim=hidden_dim,
        num_target_layers=num_layers,
        num_heads=num_heads,
        intermediate_dim=64,
    )
    adapter.train()

    # Simulate forward
    h_last = torch.randn(1, hidden_dim)
    p_matrix = adapter(h_last)  # (1, num_layers, num_heads)

    # Scatter into full_p
    full_p = torch.zeros(8, num_heads, dtype=p_matrix.dtype)
    target_idx = torch.tensor(target_layers).unsqueeze(1).expand(-1, num_heads)
    full_p = full_p.scatter(0, target_idx, p_matrix[0])

    # Simulate attention through target layers
    total_loss = torch.tensor(0.0, requires_grad=True)

    for layer_idx in target_layers:
        attn = _make_synthetic_attn(B=1, H=num_heads, seq_len=50, vision_end=vision_end)
        attn.requires_grad_(True)

        per_head_p = full_p[layer_idx]  # (H,)

        modified_attn = apply_var(
            attn_weights=attn,
            sink_indices=[0],
            vision_end=vision_end,
            p=0.6,
            rho=0.5,
            per_head_p=per_head_p,
        )

        # Simulate loss from modified attention
        loss_i = _redistribution_sensitive_loss(modified_attn, vision_end)
        total_loss = total_loss + loss_i

    total_loss.backward()

    grad_info = _collect_grad_info(adapter)

    # Verify ALL parameters have gradients
    for name, info in grad_info.items():
        assert info["has_grad"], f"V1 adapter param '{name}' has NO gradient!"
        assert info["grad_norm"] > 0, f"V1 adapter param '{name}' has zero gradient norm!"

    set_var_differentiable(enabled=False)


def test_v1_gradient_magnitudes_reasonable():
    """V1 adapter: gradient magnitudes are in a reasonable range (not exploding/vanishing)."""
    set_var_differentiable(enabled=True, temperature=10.0)

    hidden_dim, num_heads, num_layers = 128, 8, 4
    target_layers = [4, 5, 6, 7]
    vision_end = 32

    adapter = AttentionAdapter(
        hidden_dim=hidden_dim,
        num_target_layers=num_layers,
        num_heads=num_heads,
        intermediate_dim=64,
    )
    adapter.train()

    h_last = torch.randn(1, hidden_dim)
    p_matrix = adapter(h_last)

    full_p = torch.zeros(8, num_heads, dtype=p_matrix.dtype)
    target_idx = torch.tensor(target_layers).unsqueeze(1).expand(-1, num_heads)
    full_p = full_p.scatter(0, target_idx, p_matrix[0])

    total_loss = torch.tensor(0.0, requires_grad=True)
    for layer_idx in target_layers:
        attn = _make_synthetic_attn(B=1, H=num_heads, seq_len=50, vision_end=vision_end)
        attn.requires_grad_(True)
        per_head_p = full_p[layer_idx]
        modified_attn = apply_var(
            attn_weights=attn, sink_indices=[0], vision_end=vision_end,
            p=0.6, rho=0.5, per_head_p=per_head_p,
        )
        total_loss = total_loss + _redistribution_sensitive_loss(modified_attn, vision_end)

    total_loss.backward()

    for name, param in adapter.named_parameters():
        grad_norm = param.grad.norm().item()
        # Should not be vanishing (< 1e-10) or exploding (> 1e6)
        assert grad_norm > 1e-10, f"V1 '{name}' gradient vanishing: {grad_norm}"
        assert grad_norm < 1e6, f"V1 '{name}' gradient exploding: {grad_norm}"

    set_var_differentiable(enabled=False)


# ── V2 Adapter Tests ─────────────────────────────────────────────────

def test_v2_gradient_through_apply_var():
    """V2 adapter: gradient flows through BOTH branches (p_matrix + redistribution_weights)."""
    set_var_differentiable(enabled=True, temperature=10.0)

    hidden_dim, num_heads, num_layers = 128, 8, 4
    target_layers = [4, 5, 6, 7]
    vision_end = 32

    adapter = AttentionAdapterV2(
        hidden_dim=hidden_dim,
        num_target_layers=num_layers,
        num_heads=num_heads,
        intermediate_dim=64,
        mask_dim=16,
        query_dim=32,
        vision_tokens=vision_end,
    )
    adapter.train()

    h_last = torch.randn(1, hidden_dim)
    h_vision = torch.randn(1, vision_end, hidden_dim)
    object_mask = torch.ones(1, vision_end)
    object_mask[0, vision_end // 2:] = 0.0  # Half are background

    p_matrix, redist_raw = adapter(h_last, h_vision, object_mask)

    # Blend (same as forward_with_adapter)
    blend = adapter.blend_alpha
    prop_weights = torch.ones(1, vision_end)
    prop_weights[0, 0] = 0.0  # sink
    prop_weights = prop_weights / prop_weights.sum(dim=-1, keepdim=True)
    final_redist = blend * redist_raw + (1 - blend) * prop_weights

    # Scatter p_matrix
    full_p = torch.zeros(8, num_heads, dtype=p_matrix.dtype)
    target_idx = torch.tensor(target_layers).unsqueeze(1).expand(-1, num_heads)
    full_p = full_p.scatter(0, target_idx, p_matrix[0])

    total_loss = torch.tensor(0.0, requires_grad=True)
    for layer_idx in target_layers:
        attn = _make_synthetic_attn(B=1, H=num_heads, seq_len=50, vision_end=vision_end)
        attn.requires_grad_(True)

        per_head_p = full_p[layer_idx]
        modified_attn = apply_var(
            attn_weights=attn, sink_indices=[0], vision_end=vision_end,
            p=0.6, rho=0.5,
            per_head_p=per_head_p,
            redistribution_weights=final_redist.squeeze(0),
        )
        total_loss = total_loss + _redistribution_sensitive_loss(modified_attn, vision_end)

    total_loss.backward()

    grad_info = _collect_grad_info(adapter)

    # ALL parameters must have non-zero gradients
    for name, info in grad_info.items():
        assert info["has_grad"], f"V2 adapter param '{name}' has NO gradient!"
        assert info["grad_norm"] > 0, f"V2 adapter param '{name}' has zero gradient norm!"

    set_var_differentiable(enabled=False)


def test_v2_blend_alpha_has_gradient():
    """V2 adapter: blend_alpha (scalar parameter) receives gradient."""
    set_var_differentiable(enabled=True, temperature=10.0)

    hidden_dim, num_heads = 128, 8
    vision_end = 32

    adapter = AttentionAdapterV2(
        hidden_dim=hidden_dim, num_target_layers=4, num_heads=num_heads,
        intermediate_dim=64, mask_dim=16, query_dim=32,
        vision_tokens=vision_end,
    )
    adapter.train()

    h_last = torch.randn(1, hidden_dim)
    h_vision = torch.randn(1, vision_end, hidden_dim)
    object_mask = torch.ones(1, vision_end)

    _, redist_raw = adapter(h_last, h_vision, object_mask)

    # Squeeze redist_raw for 1D operation
    if redist_raw is not None and redist_raw.dim() == 2:
        redist_raw = redist_raw.squeeze(0)

    blend = adapter.blend_alpha
    prop_weights = torch.ones(vision_end) / vision_end
    final_redist = blend * redist_raw + (1 - blend) * prop_weights

    # Use weighted loss that's sensitive to redistribution differences
    # blend's gradient = dL/d_blend * sigmoid'(logit) where
    # dL/d_blend = sum(weights * (redist_raw - prop_weights))
    # Non-uniform weights ensure dL/d_blend != 0
    weights = torch.arange(vision_end, dtype=torch.float32)
    loss = (final_redist * weights).sum()
    loss.backward()

    assert adapter._blend_logit.grad is not None, "blend_logit has no gradient!"
    assert adapter._blend_logit.grad.abs().item() > 0, "blend_logit gradient is zero!"

    set_var_differentiable(enabled=False)


def test_v2_branch2_cross_attention_gradient():
    """V2 adapter Branch 2: query_proj and key_proj receive gradients."""
    set_var_differentiable(enabled=True, temperature=10.0)

    hidden_dim, num_heads = 128, 8
    vision_end = 32

    adapter = AttentionAdapterV2(
        hidden_dim=hidden_dim, num_target_layers=4, num_heads=num_heads,
        intermediate_dim=64, mask_dim=16, query_dim=32,
        vision_tokens=vision_end,
    )
    adapter.train()

    h_last = torch.randn(1, hidden_dim)
    h_vision = torch.randn(1, vision_end, hidden_dim)
    object_mask = torch.ones(1, vision_end)

    p_matrix, redist_raw = adapter(h_last, h_vision, object_mask)

    # Squeeze redist_raw to 1D if needed (apply_var expects (V,))
    if redist_raw is not None and redist_raw.dim() == 2:
        redist_raw = redist_raw.squeeze(0)

    # Use redist_raw in apply_var
    attn = _make_synthetic_attn(B=1, H=num_heads, seq_len=50, vision_end=vision_end)
    attn.requires_grad_(True)

    full_p = torch.zeros(8, num_heads, dtype=p_matrix.dtype)
    target_idx = torch.tensor([4, 5, 6, 7]).unsqueeze(1).expand(-1, num_heads)
    full_p = full_p.scatter(0, target_idx, p_matrix[0])

    modified_attn = apply_var(
        attn_weights=attn, sink_indices=[0], vision_end=vision_end,
        p=0.6, rho=0.5,
        per_head_p=full_p[4],
        redistribution_weights=redist_raw,
    )
    loss = _redistribution_sensitive_loss(modified_attn, vision_end)
    loss.backward()

    # Check cross-attention parameters specifically
    assert adapter.query_proj.weight.grad is not None, "query_proj has no gradient!"
    assert adapter.key_proj.weight.grad is not None, "key_proj has no gradient!"
    assert adapter.query_proj.weight.grad.norm().item() > 0, "query_proj gradient is zero!"
    assert adapter.key_proj.weight.grad.norm().item() > 0, "key_proj gradient is zero!"

    set_var_differentiable(enabled=False)


# ── BF16 Cast Tests ──────────────────────────────────────────────────

def test_bf16_to_float32_preserves_gradient():
    """Verify that bf16 → float32 casts (as in mixed precision) preserve gradients."""
    # Simulate the cast chain in apply_var
    p_bf16 = torch.randn(8, requires_grad=True, dtype=torch.bfloat16)
    p_float = p_bf16.float()  # bf16 → float32

    mask = torch.sigmoid(torch.randn(8))
    effective_p = p_float * mask

    loss = effective_p.sum()
    loss.backward()

    assert p_bf16.grad is not None, "bf16 tensor lost gradient after float() cast!"
    assert p_bf16.grad.norm().item() > 0, "bf16 gradient is zero after float() cast!"


def test_scatter_preserves_gradient():
    """Verify torch.zeros().scatter(src=...) preserves gradient through src."""
    src = torch.randn(4, 8, requires_grad=True)
    target = torch.zeros(8, 8)
    idx = torch.tensor([4, 5, 6, 7]).unsqueeze(1).expand(-1, 8)
    result = target.scatter(0, idx, src)

    loss = result[5].sum()  # Only uses one row from src
    loss.backward()

    assert src.grad is not None, "scatter broke gradient from src!"
    assert src.grad[1].norm().item() > 0, "scatter row 1 (→ target[5]) has zero gradient!"


def test_clone_preserves_gradient():
    """Verify attn_weights.clone() preserves gradient (used in apply_var line 352)."""
    attn = torch.randn(1, 8, 50, 50, requires_grad=True)
    cloned = attn.clone()
    loss = cloned[:, :, -1, :10].sum()
    loss.backward()

    assert attn.grad is not None, "clone() broke gradient!"
    assert attn.grad.norm().item() > 0, "clone() gradient is zero!"


# ── Full Pipeline Simulation ─────────────────────────────────────────

def test_full_pipeline_simulation():
    """End-to-end simulation of the training gradient path.

    Simulates what forward_with_adapter does without needing the actual VLA model:
    1. Adapter produces p_matrix + redistribution_weights
    2. Scatter p_matrix into full_p
    3. Set ctx.per_head_var_strength
    4. Multiple "layers" call apply_var
    5. Loss computed from modified attention
    6. Backward pass
    7. ALL adapter params have non-zero gradients
    """
    set_var_differentiable(enabled=True, temperature=10.0)

    hidden_dim, num_heads, num_layers = 256, 8, 4
    total_layers = 8
    target_layers = [4, 5, 6, 7]
    vision_end = 32

    adapter = AttentionAdapterV2(
        hidden_dim=hidden_dim,
        num_target_layers=num_layers,
        num_heads=num_heads,
        intermediate_dim=64,
        mask_dim=16,
        query_dim=32,
        vision_tokens=vision_end,
    )
    adapter.train()

    # Step 1: Simulate hidden state capture (detached, as in adapter_train.py)
    h_last = torch.randn(1, hidden_dim)  # Detached from frozen model
    h_vision = torch.randn(1, vision_end, hidden_dim)  # Detached

    # Step 2: Object mask
    object_mask = torch.zeros(1, vision_end)
    object_mask[0, 5:15] = 1.0  # Object in patches 5-14

    # Step 3: Adapter forward
    p_matrix, redist_raw = adapter(h_last, h_vision, object_mask)

    # Step 4: Blend (same as forward_with_adapter lines 128-140)
    blend = adapter.blend_alpha
    sink_set = {0}
    prop_weights = torch.ones(1, vision_end)
    for si in sink_set:
        prop_weights[0, si] = 0.0
    prop_weights = prop_weights / prop_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    final_redist = blend * redist_raw + (1 - blend) * prop_weights

    # Step 5: Scatter (same as forward_with_adapter lines 148-155)
    full_p = torch.zeros(total_layers, num_heads, dtype=p_matrix.dtype)
    target_idx = torch.tensor(target_layers).unsqueeze(1).expand(-1, num_heads)
    full_p = full_p.scatter(0, target_idx, p_matrix[0])

    # Step 6: Simulate teacher-forced autoregressive loop (7 action tokens)
    total_loss = torch.tensor(0.0, requires_grad=True)

    for token_idx in range(7):
        for layer_idx in target_layers:
            attn = _make_synthetic_attn(B=1, H=num_heads, seq_len=50, vision_end=vision_end)
            attn.requires_grad_(True)

            per_head_p = full_p[layer_idx]
            modified_attn = apply_var(
                attn_weights=attn,
                sink_indices=[0],
                vision_end=vision_end,
                p=0.6,
                rho=0.5,
                per_head_p=per_head_p,
                redistribution_weights=final_redist.squeeze(0),
            )

            # Simulate logit → CE loss from modified attention
            fake_logit = _redistribution_sensitive_loss(modified_attn, vision_end)
            total_loss = total_loss + fake_logit

    total_loss = total_loss / 7  # Average over tokens

    # Step 7: Backward
    total_loss.backward()

    # Step 8: Verify ALL parameters
    grad_info = _collect_grad_info(adapter)
    print("\n" + "=" * 70)
    print("  GRADIENT FLOW VERIFICATION RESULTS")
    print("=" * 70)

    all_ok = True
    for name, info in grad_info.items():
        if not info["has_grad"]:
            status = "FAIL (no gradient)"
            all_ok = False
        elif info["grad_norm"] == 0:
            status = "FAIL (zero gradient)"
            all_ok = False
        elif info["grad_norm"] < 1e-10:
            status = f"WARN (vanishing: {info['grad_norm']:.2e})"
        elif info["grad_norm"] > 1e6:
            status = f"WARN (exploding: {info['grad_norm']:.2e})"
        else:
            status = f"OK (norm={info['grad_norm']:.4e}, max={info['grad_max']:.4e})"

        print(f"  {name:40s} {status}")

    print("=" * 70)
    print(f"  Overall: {'PASS' if all_ok else 'FAIL'}")
    print("=" * 70 + "\n")

    assert all_ok, "Some adapter parameters have no gradient — training would be meaningless!"

    set_var_differentiable(enabled=False)


def test_gradient_with_mixed_precision_simulation():
    """Simulate mixed precision (bf16 model outputs, float32 adapter).

    This tests the exact cast chain that happens in training:
    - Hidden states: bf16 (from frozen model) → float32 (for adapter)
    - Attention weights: bf16 (from frozen model) → float32 (in apply_var) → bf16 (write back)
    - p_matrix: float32 (adapter output)
    """
    set_var_differentiable(enabled=True, temperature=10.0)

    hidden_dim, num_heads = 128, 8
    vision_end = 32

    adapter = AttentionAdapterV2(
        hidden_dim=hidden_dim, num_target_layers=4, num_heads=num_heads,
        intermediate_dim=64, mask_dim=16, query_dim=32,
        vision_tokens=vision_end,
    )
    adapter.train()

    # Simulate bf16 hidden states from frozen model
    h_last_bf16 = torch.randn(1, hidden_dim, dtype=torch.bfloat16)
    h_vision_bf16 = torch.randn(1, vision_end, hidden_dim, dtype=torch.bfloat16)

    # Cast to float32 for adapter (as in adapter_train.py lines 112-113)
    h_last = h_last_bf16.float()
    h_vision = h_vision_bf16.float()

    object_mask = torch.ones(1, vision_end)
    p_matrix, redist_raw = adapter(h_last, h_vision, object_mask)

    # Squeeze redist_raw to 1D (apply_var expects (V,))
    if redist_raw is not None and redist_raw.dim() == 2:
        redist_raw = redist_raw.squeeze(0)

    # Blend (same as forward_with_adapter — needed so _blend_logit is in the graph)
    blend = adapter.blend_alpha
    prop_weights = torch.ones(vision_end)
    prop_weights[0] = 0.0
    prop_weights = prop_weights / prop_weights.sum()
    final_redist = blend * redist_raw + (1 - blend) * prop_weights

    # Simulate bf16 attention weights (from frozen model's forward pass)
    attn_bf16 = _make_synthetic_attn(B=1, H=num_heads, seq_len=50, vision_end=vision_end)
    attn_bf16 = attn_bf16.to(torch.bfloat16)
    attn_bf16.requires_grad_(True)

    # apply_var casts to float32 internally (line 261)
    full_p = torch.zeros(8, num_heads, dtype=p_matrix.dtype)
    target_idx = torch.tensor([4, 5, 6, 7]).unsqueeze(1).expand(-1, num_heads)
    full_p = full_p.scatter(0, target_idx, p_matrix[0])

    modified_attn = apply_var(
        attn_weights=attn_bf16,
        sink_indices=[0],
        vision_end=vision_end,
        p=0.6,
        rho=0.5,
        per_head_p=full_p[4],
        redistribution_weights=final_redist,
    )

    loss = _redistribution_sensitive_loss(modified_attn, vision_end)
    loss.backward()

    # Check adapter gradients exist despite mixed precision
    for name, param in adapter.named_parameters():
        assert param.grad is not None, f"Mixed precision: '{name}' has no gradient!"
        assert param.grad.norm().item() > 0, f"Mixed precision: '{name}' gradient is zero!"

    set_var_differentiable(enabled=False)


# ── Differential Mode Toggle Test ────────────────────────────────────

def test_differentiable_mode_required():
    """Without differentiable mode, head_mask is binary and may block some gradients."""
    # This test documents the expected behavior difference

    hidden_dim, num_heads = 128, 8
    vision_end = 32

    adapter = AttentionAdapter(
        hidden_dim=hidden_dim, num_target_layers=4, num_heads=num_heads,
        intermediate_dim=64,
    )
    adapter.train()

    h_last = torch.randn(1, hidden_dim)
    p_matrix = adapter(h_last)

    full_p = torch.zeros(8, num_heads, dtype=p_matrix.dtype)
    target_idx = torch.tensor([4, 5, 6, 7]).unsqueeze(1).expand(-1, num_heads)
    full_p = full_p.scatter(0, target_idx, p_matrix[0])

    # Test WITH differentiable mode
    set_var_differentiable(enabled=True, temperature=10.0)
    attn = _make_synthetic_attn(B=1, H=num_heads, seq_len=50, vision_end=vision_end)
    attn.requires_grad_(True)

    modified = apply_var(
        attn_weights=attn, sink_indices=[0], vision_end=vision_end,
        p=0.6, rho=0.5, per_head_p=full_p[4],
    )
    loss = modified[:, :, -1, :vision_end].sum()
    loss.backward()

    grad_norm_diff = sum(p.grad.norm().item() for p in adapter.parameters() if p.grad is not None)

    # Reset
    adapter.zero_grad()

    # Test WITHOUT differentiable mode
    set_var_differentiable(enabled=False)
    attn2 = _make_synthetic_attn(B=1, H=num_heads, seq_len=50, vision_end=vision_end)
    attn2.requires_grad_(True)

    p_matrix2 = adapter(h_last)
    full_p2 = torch.zeros(8, num_heads, dtype=p_matrix2.dtype)
    full_p2 = full_p2.scatter(0, target_idx, p_matrix2[0])

    modified2 = apply_var(
        attn_weights=attn2, sink_indices=[0], vision_end=vision_end,
        p=0.6, rho=0.5, per_head_p=full_p2[4],
    )
    loss2 = modified2[:, :, -1, :vision_end].sum()
    loss2.backward()

    grad_norm_hard = sum(p.grad.norm().item() for p in adapter.parameters() if p.grad is not None)

    # Both should have gradients (the hard threshold still passes gradients through
    # the per_head_p path; it just doesn't pass through the head_mask selection itself)
    assert grad_norm_diff > 0, "Differentiable mode should produce gradients"
    assert grad_norm_hard > 0, "Hard threshold mode should also produce gradients (through ep path)"

    print(f"\n  Gradient norm (differentiable): {grad_norm_diff:.6f}")
    print(f"  Gradient norm (hard threshold): {grad_norm_hard:.6f}")

    set_var_differentiable(enabled=False)


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("scatter preserves gradient", test_scatter_preserves_gradient),
        ("clone preserves gradient", test_clone_preserves_gradient),
        ("bf16→float32 preserves gradient", test_bf16_to_float32_preserves_gradient),
        ("V1 gradient through apply_var", test_v1_gradient_through_apply_var),
        ("V1 gradient magnitudes reasonable", test_v1_gradient_magnitudes_reasonable),
        ("V2 gradient through apply_var", test_v2_gradient_through_apply_var),
        ("V2 blend_alpha has gradient", test_v2_blend_alpha_has_gradient),
        ("V2 cross-attention gradient", test_v2_branch2_cross_attention_gradient),
        ("differentiable mode toggle", test_differentiable_mode_required),
        ("mixed precision simulation", test_gradient_with_mixed_precision_simulation),
        ("FULL PIPELINE simulation", test_full_pipeline_simulation),
    ]

    print("\n" + "=" * 70)
    print("  C6: GRADIENT FLOW VERIFICATION SUITE")
    print("=" * 70)

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1

    print(f"\n  Results: {passed}/{passed + failed} passed")
    if failed > 0:
        print(f"  WARNING: {failed} tests FAILED — gradient flow may be broken!")
        sys.exit(1)
    else:
        print("  All gradient flow tests passed — training is valid.")
    print("=" * 70 + "\n")
