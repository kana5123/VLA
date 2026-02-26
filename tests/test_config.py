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
    assert config.ADAPTER_V2_BLEND_INIT == -1.0

    # Object mask MLP
    assert hasattr(config, "ADAPTER_V2_MASK_DIM")
    assert config.ADAPTER_V2_MASK_DIM == 64
