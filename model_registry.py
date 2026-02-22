"""Registry of VLA models for cross-model attention analysis.

Each entry defines how to load the model, what architecture it uses,
and how to extract attention weights from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VLAModelConfig:
    """Configuration for a single VLA model."""
    name: str                          # Display name
    hf_id: str                         # HuggingFace model ID
    architecture: str                  # "llama", "phi3", "qwen2", "mistral", etc.
    vision_encoder: str                # "prismatic", "siglip", "dinov2", etc.
    num_layers: int                    # LLM backbone layers
    num_heads: int                     # Attention heads
    hidden_dim: int                    # Hidden dimension
    vision_grid_size: int              # e.g., 16 for 16x16 = 256 tokens
    num_vision_tokens: int             # Total vision tokens
    action_tokens: int = 7             # Number of action tokens generated
    prompt_template: str = ""          # Model-specific prompt format
    trust_remote_code: bool = True
    attn_impl: str = "eager"           # Must be eager for weight extraction
    torch_dtype: str = "bfloat16"
    notes: str = ""
    native_datasets: list[str] = field(default_factory=list)
    # Architecture-specific: where to find attention layers
    layers_path: str = "language_model.model.layers"  # dot-separated path
    attn_module: str = "self_attn"     # attribute name for attention in each layer
    # Adapter experiment fields
    action_type: str = "discrete"      # "discrete" (CE loss) or "continuous" (MSE loss)
    source_layer: int = -5             # Hidden state capture layer (negative = relative to num_layers)
    target_layers: list[int] = field(default_factory=list)  # VAR-applied layers (empty = last 4)
    auto_model_class: str = "AutoModelForVision2Seq"  # HF auto class for loading
    # Experiment support status
    experiment_ready: bool = False     # True = validated for adapter training/eval

    def get_adapter_config(self) -> dict:
        """Return adapter-relevant parameters derived from model config."""
        tl = self.target_layers if self.target_layers else list(range(self.num_layers - 4, self.num_layers))
        sl = self.source_layer if self.source_layer >= 0 else self.num_layers + self.source_layer
        return {
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_target_layers": len(tl),
            "target_layers": tl,
            "source_layer": sl,
            "vision_tokens": self.num_vision_tokens,
            "action_type": self.action_type,
        }


# ═══════════════════════════════════════════════════════════════════════
# Model Registry
# ═══════════════════════════════════════════════════════════════════════

MODELS: dict[str, VLAModelConfig] = {}


def register(cfg: VLAModelConfig):
    MODELS[cfg.name] = cfg


# ── OpenVLA (LLaMA-2 7B + Prismatic) ──
register(VLAModelConfig(
    name="openvla-7b",
    hf_id="openvla/openvla-7b",
    architecture="llama",
    vision_encoder="prismatic",
    num_layers=32,
    num_heads=32,
    hidden_dim=4096,
    vision_grid_size=16,
    num_vision_tokens=256,
    action_tokens=7,
    prompt_template="In: What action should the robot take to {instruction}?\nOut:",
    native_datasets=["bridge_v2", "oxe"],
    notes="Dual encoder (DINOv2+SigLIP) fused to 256 tokens",
    source_layer=27,
    target_layers=[28, 29, 30, 31],
    experiment_ready=True,
))

# ── CogACT (CogVLM2-Llama3-8B backbone) ──
register(VLAModelConfig(
    name="cogact-base",
    hf_id="CogACT/CogACT-Base",
    architecture="llama",
    vision_encoder="eva2-clip-e",
    num_layers=32,
    num_heads=32,
    hidden_dim=4096,
    vision_grid_size=14,
    num_vision_tokens=196,
    action_tokens=7,
    prompt_template="What action should the robot take to {instruction}?",
    native_datasets=["oxe", "bridge_v2"],
    notes="CogVLM2 backbone, EVA2-CLIP-E vision encoder",
    layers_path="model.layers",
))

# ── TraceVLA (Phi-3-V backbone) ──
# BLOCKED: Phi3VConfig not recognized by AutoModelForVision2Seq (transformers 4.57.6)
# Also uses continuous actions which need separate loss implementation
# Verified: model_type=phi3_v, hidden=3072, 32L/32H
# Vision: 313 tokens (detected via negative input_ids from processor)
register(VLAModelConfig(
    name="tracevla-phi3v",
    hf_id="furonghuang-lab/tracevla_phi3v",
    architecture="phi3_v",
    vision_encoder="clip-vit",
    num_layers=32,
    num_heads=32,
    hidden_dim=3072,
    vision_grid_size=18,
    num_vision_tokens=313,
    action_tokens=7,
    action_type="continuous",
    prompt_template="<|user|>\n<|image_1|>\nWhat action should the robot take to {instruction}?\n<|end|>\n<|assistant|>\n",
    native_datasets=["bridge_v2", "oxe"],
    notes="Phi-3 Vision backbone, visual trace overlays, continuous action output",
    layers_path="model.layers",
    source_layer=27,
    target_layers=[28, 29, 30, 31],
    auto_model_class="AutoModelForCausalLM",
))

# ── SpatialVLA (Gemma-2 2B backbone, NOT Qwen2) ──
# BLOCKED: processing_spatialvla.py imports _validate_images_text_input_order
# which was removed in transformers 4.57.6
# Verified: model_type=spatialvla, text=gemma2, 26L/8H/2304D
# Vision: SigLIP image_size=224, patch_size=14 → 16×16 = 256 tokens
register(VLAModelConfig(
    name="spatialvla-4b",
    hf_id="IPEC-COMMUNITY/spatialvla-4b-224-pt",
    architecture="gemma2",
    vision_encoder="siglip-so400m",
    num_layers=26,
    num_heads=8,
    hidden_dim=2304,
    vision_grid_size=16,
    num_vision_tokens=256,
    action_tokens=7,
    prompt_template="What action should the robot take to {instruction}?",
    native_datasets=["bridge_v2", "oxe", "rh20t"],
    notes="SpatialVLA 4B, Gemma-2 text + SigLIP vision, spatial features",
    layers_path="language_model.model.layers",
    source_layer=21,
    target_layers=[22, 23, 24, 25],
    auto_model_class="AutoModel",
))

# ── SmolVLA (SmolVLM2-500M backbone, LeRobot policy) ──
# Verified: type=smolvla, VLM=SmolVLM2-500M (llama, 32L/15H/960D)
# NOTE: Requires LeRobot API or custom loading, not standard AutoModelForVision2Seq
register(VLAModelConfig(
    name="smolvla-base",
    hf_id="lerobot/smolvla_base",
    architecture="llama",
    vision_encoder="siglip",
    num_layers=32,
    num_heads=15,
    hidden_dim=960,
    vision_grid_size=14,
    num_vision_tokens=196,
    action_tokens=7,
    prompt_template="What action should the robot take to {instruction}?",
    native_datasets=["lerobot"],
    notes="SmolVLM2-500M backbone (llama 32L/15H/960D), LeRobot policy format, needs custom loader",
    layers_path="model.text_model.model.layers",
))

# ── ECoT (OpenVLA + Chain-of-Thought) ──
register(VLAModelConfig(
    name="ecot-7b",
    hf_id="Embodied-CoT/ecot-openvla-7b-bridge",
    architecture="llama",
    vision_encoder="prismatic",
    num_layers=32,
    num_heads=32,
    hidden_dim=4096,
    vision_grid_size=16,
    num_vision_tokens=256,
    action_tokens=7,
    prompt_template="In: What action should the robot take to {instruction}?\nOut:",
    native_datasets=["bridge_v2"],
    notes="Same backbone as OpenVLA, fine-tuned with chain-of-thought",
    source_layer=27,
    target_layers=[28, 29, 30, 31],
    experiment_ready=True,
))

# ── RoboFlamingo (MPT-1B or 3B backbone) ──
register(VLAModelConfig(
    name="roboflamingo",
    hf_id="roboflamingo/RoboFlamingo",
    architecture="mpt",
    vision_encoder="clip-vit",
    num_layers=24,
    num_heads=16,
    hidden_dim=2048,
    vision_grid_size=14,
    num_vision_tokens=196,
    action_tokens=7,
    prompt_template="{instruction}",
    native_datasets=["calvin"],
    notes="OpenFlamingo-based, cross-attention (different attn pattern)",
    layers_path="transformer.blocks",
    attn_module="attn",
))


def get_model(name: str) -> VLAModelConfig:
    if name not in MODELS:
        available = ", ".join(MODELS.keys())
        raise ValueError(f"Unknown model: {name}. Available: {available}")
    return MODELS[name]


def list_models() -> list[str]:
    return list(MODELS.keys())


def list_experiment_models() -> list[str]:
    """Return names of models validated for adapter experiments."""
    return [name for name, cfg in MODELS.items() if cfg.experiment_ready]
