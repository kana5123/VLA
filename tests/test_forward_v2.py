"""Tests for v2 forward_with_adapter modifications."""
import inspect


def test_forward_with_adapter_v2_signature():
    """forward_with_adapter should accept object_mask parameter."""
    from adapter_train import forward_with_adapter
    sig = inspect.signature(forward_with_adapter)
    assert "object_mask" in sig.parameters


def test_imports_v2():
    """adapter_train should import AttentionAdapterV2."""
    import adapter_train
    assert hasattr(adapter_train, 'AttentionAdapterV2')
