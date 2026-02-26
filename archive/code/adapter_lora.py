"""LoRA fine-tuning infrastructure for OpenVLA.

Wraps the frozen OpenVLA backbone with LoRA adapters on LLaMA's q_proj and
v_proj layers using the peft library.  Two training modes are supported:

    1. LoRA-only: Fine-tune OpenVLA with LoRA, no attention adapter.
    2. LoRA + Adapter: Two-stage training:
       Stage A — Train attention adapter (LoRA frozen)
       Stage B — Train LoRA (attention adapter frozen)

Infrastructure only — actual training invocation deferred until adapter v2
experiment results are analyzed.

Usage (infrastructure test):
    python adapter_lora.py --dry_run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

import config

# Lazy peft import to avoid hard dependency
_peft_available = None


def _check_peft():
    global _peft_available
    if _peft_available is None:
        try:
            import peft  # noqa: F401
            _peft_available = True
        except ImportError:
            _peft_available = False
    return _peft_available


def create_lora_model(
    model: nn.Module,
    r: int = config.LORA_R,
    lora_alpha: int = config.LORA_ALPHA,
    target_modules: list[str] = None,
    lora_dropout: float = config.LORA_DROPOUT,
) -> nn.Module:
    """Wrap OpenVLA's LLaMA backbone with LoRA adapters.

    Args:
        model: The full OpenVLA model (must have .language_model).
        r: LoRA rank.
        lora_alpha: LoRA scaling factor (effective scale = alpha / r).
        target_modules: LLaMA submodule names to apply LoRA to.
        lora_dropout: Dropout rate on LoRA A/B matrices.

    Returns:
        peft-wrapped model with LoRA adapters enabled.
    """
    if not _check_peft():
        raise ImportError(
            "peft is required for LoRA. Install with: pip install peft"
        )
    from peft import LoraConfig, get_peft_model

    if target_modules is None:
        target_modules = list(config.LORA_TARGET_MODULES)

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        task_type="CAUSAL_LM",
        bias="none",
    )

    # Apply to the language model sub-module
    if hasattr(model, "language_model"):
        peft_model = get_peft_model(model.language_model, lora_config)
        model.language_model = peft_model
    else:
        model = get_peft_model(model, lora_config)

    return model


def count_lora_params(model: nn.Module) -> dict:
    """Count trainable (LoRA) vs frozen parameters.

    Returns:
        dict with trainable, frozen, total, and trainable_pct.
    """
    trainable = 0
    frozen = 0
    for p in model.parameters():
        if p.requires_grad:
            trainable += p.numel()
        else:
            frozen += p.numel()
    total = trainable + frozen
    return {
        "trainable": trainable,
        "frozen": frozen,
        "total": total,
        "trainable_pct": trainable / total * 100 if total > 0 else 0,
    }


def save_lora_checkpoint(model: nn.Module, path: Path, extra_state: dict = None):
    """Save only the LoRA adapter weights (much smaller than full model).

    Args:
        model: The peft-wrapped model.
        path: Path to save checkpoint.
        extra_state: Additional state (optimizer, step, etc.) to include.
    """
    if not _check_peft():
        raise ImportError("peft required")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Extract LoRA state dict
    lora_state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            lora_state[name] = param.data.cpu()

    save_dict = {
        "lora_state_dict": lora_state,
        "lora_config": {
            "r": config.LORA_R,
            "alpha": config.LORA_ALPHA,
            "target_modules": config.LORA_TARGET_MODULES,
            "dropout": config.LORA_DROPOUT,
        },
    }
    if extra_state:
        save_dict.update(extra_state)

    torch.save(save_dict, path)
    print(f"LoRA checkpoint saved: {path} ({len(lora_state)} tensors)")


def load_lora_checkpoint(model: nn.Module, path: str | Path, device: str = "cuda"):
    """Load LoRA adapter weights into a peft-wrapped model.

    Args:
        model: The peft-wrapped model (must already have LoRA applied).
        path: Path to the LoRA checkpoint.
        device: Device to map tensors to.

    Returns:
        extra_state dict (optimizer, step, etc.) if present.
    """
    ckpt = torch.load(path, map_location=device)
    lora_state = ckpt["lora_state_dict"]

    model_state = model.state_dict()
    for name, param in lora_state.items():
        if name in model_state:
            model_state[name].copy_(param)
        else:
            print(f"  WARNING: LoRA key not found in model: {name}")

    print(f"LoRA checkpoint loaded: {path}")
    return {k: v for k, v in ckpt.items() if k not in ("lora_state_dict", "lora_config")}


def freeze_for_lora_only(model: nn.Module):
    """Freeze all params except LoRA adapters. For mode 1 (LoRA-only)."""
    for param in model.parameters():
        param.requires_grad_(False)
    # Re-enable LoRA params
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)


def freeze_for_adapter_stage(model: nn.Module, adapter: nn.Module):
    """Freeze model + LoRA, unfreeze adapter. For mode 2, Stage A."""
    for param in model.parameters():
        param.requires_grad_(False)
    for param in adapter.parameters():
        param.requires_grad_(True)


def freeze_for_lora_stage(model: nn.Module, adapter: nn.Module):
    """Freeze model + adapter, unfreeze LoRA. For mode 2, Stage B."""
    for param in model.parameters():
        param.requires_grad_(False)
    for param in adapter.parameters():
        param.requires_grad_(False)
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)


def main():
    """Dry-run test: create LoRA model and print param counts."""
    parser = argparse.ArgumentParser(description="LoRA infrastructure test")
    parser.add_argument("--dry_run", action="store_true", help="Test LoRA creation")
    args = parser.parse_args()

    if not args.dry_run:
        print("Use --dry_run to test LoRA infrastructure.")
        print("Actual LoRA training will be added after adapter v2 analysis.")
        return

    if not _check_peft():
        print("peft not installed. Run: pip install peft")
        print("LoRA infrastructure code is ready, but cannot test without peft.")
        return

    print("Loading model for LoRA test...")
    from extract_attention import load_model
    processor, model = load_model(device="cpu")

    # Before LoRA
    before = count_lora_params(model)
    print(f"\nBefore LoRA: {before['total']:,} params (all frozen)")

    # Apply LoRA
    model = create_lora_model(model)

    # After LoRA
    after = count_lora_params(model)
    print(f"After LoRA:  {after['trainable']:,} trainable / {after['total']:,} total")
    print(f"  Trainable: {after['trainable_pct']:.4f}%")
    print(f"  LoRA rank: {config.LORA_R}, alpha: {config.LORA_ALPHA}")
    print(f"  Target modules: {config.LORA_TARGET_MODULES}")

    # Test save/load
    test_path = Path("/tmp/lora_test.pt")
    save_lora_checkpoint(model, test_path, extra_state={"test": True})
    extra = load_lora_checkpoint(model, test_path)
    print(f"  Save/load test: {'PASS' if extra.get('test') else 'FAIL'}")

    print("\nLoRA infrastructure ready.")


if __name__ == "__main__":
    main()
