"""Extract attention weights from OpenVLA during action generation.

Usage:
    python extract_attention.py [--episodes 0,1,2] [--device cuda]

For each (episode, step) pair, this script:
  1. Loads the image + instruction from metadata.json
  2. Runs OpenVLA inference (greedy, 7 action tokens)
  3. Extracts attention weights from all 32 layers for each generated action token
  4. Identifies the top-5 most attended input tokens (mean over heads)
  5. Classifies each as vision / text / special and saves results to JSON

Requires GPU. Uses `attn_implementation="eager"` (flash attention cannot return weights).
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModel, AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor,
)

import config
from model_registry import get_model as _registry_get_model, VLAModelConfig


# ═══════════════════════════════════════════════════════════════════════════
# Universal processor call (architecture-aware)
# ═══════════════════════════════════════════════════════════════════════════

def call_processor(processor, prompt, image, model_cfg=None, **kwargs):
    """Call processor with the correct calling convention for each architecture.

    Different VLM processors expect different argument styles:
      - OpenVLA, TraceVLA, SpatialVLA: positional (text, images)
      - LLaVA, PaliGemma: keyword (text=..., images=...)
      - InternVL2: keyword (text=..., images=...)

    This function normalizes the call for all architectures.
    """
    arch = model_cfg.architecture if model_cfg else None

    # Architectures that require keyword arguments
    keyword_archs = {"llava", "paligemma", "internlm2"}

    if arch in keyword_archs:
        return processor(text=prompt, images=image, **kwargs)
    else:
        # Default: positional (works for OpenVLA, TraceVLA, SpatialVLA, ECoT, etc.)
        return processor(prompt, image, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# Token boundary detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_token_boundaries(processor, model, sample_image, sample_instruction, device, model_cfg=None):
    """Detect vision/text token boundaries using a forward hook.

    OpenVLA/Prismatic doesn't use <image> placeholders in text.
    Instead, vision features are prepended to text embeddings inside the model.
    We capture attention shape via hooks to determine the actual sequence layout.

    Returns:
        dict with keys: vision_start, vision_end, text_start, text_end, total_seq_len
    """
    if model_cfg is not None:
        prompt = model_cfg.prompt_template.format(instruction=sample_instruction)
    else:
        prompt = config.PROMPT_TEMPLATE.format(instruction=sample_instruction)
    inputs = call_processor(processor, prompt, sample_image, model_cfg=model_cfg, return_tensors="pt").to(device)
    # Cast pixel_values to model dtype (processor returns float32, model is bfloat16)
    if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    input_ids = inputs["input_ids"][0]
    num_text_tokens = len(input_ids)

    print(f"  Text input_ids length: {num_text_tokens}")

    # Use a hook on the first decoder layer to capture sequence length from hidden states
    captured = {}

    def hook_fn(module, args, output):
        # Layer output: (hidden_states, ...) — hidden_states shape is (batch, seq_len, hidden_dim)
        h = output[0] if isinstance(output, tuple) else output
        captured["seq_len"] = h.shape[1]

    layers = get_layers(model, model_cfg) if model_cfg else None
    if layers is not None:
        target_layer = layers[0]
    elif hasattr(model, "language_model"):
        target_layer = model.language_model.model.layers[0]
    else:
        target_layer = model.model.layers[0]

    hook = target_layer.register_forward_hook(hook_fn)

    # Build forward kwargs (some architectures need extra inputs)
    fwd_kwargs = {k: v for k, v in inputs.items()}
    fwd_kwargs["use_cache"] = False
    # SpatialVLA needs intrinsic matrix
    if model_cfg and model_cfg.architecture == "gemma2" and "intrinsic" not in fwd_kwargs:
        fwd_kwargs["intrinsic"] = torch.tensor(
            [[[218.26, 0.0, 111.83],
              [0.0, 218.26, 111.79],
              [0.0, 0.0, 1.0]]], device=device, dtype=torch.float32,
        )

    with torch.no_grad():
        model(**fwd_kwargs)

    hook.remove()

    if "seq_len" not in captured:
        raise RuntimeError("Failed to capture sequence length. Check model architecture.")

    # hidden states shape: (batch, seq_len, hidden_dim)
    full_seq_len = captured["seq_len"]
    num_vision_tokens = full_seq_len - num_text_tokens

    print(f"  Full sequence length (after vision expansion): {full_seq_len}")
    print(f"  Vision tokens (from seq diff): {num_vision_tokens}, Text tokens: {num_text_tokens}")

    # Detect vision token position from input_ids.
    # Some models (Phi3V, LLaVA) encode image placeholders as negative IDs or
    # special token IDs. The expanded sequence inserts vision features at those
    # positions, shifting text tokens around.
    #
    # When the processor already includes image placeholder tokens in input_ids
    # (e.g., SpatialVLA _SimpleProcessor, LLaVA, PaliGemma), the model replaces
    # them in-place → full_seq_len == num_text_tokens → num_vision_tokens=0.
    # In this case, we detect image tokens via special token IDs in input_ids.
    negative_count = int((input_ids < 0).sum().item())
    arch = model_cfg.architecture if model_cfg else None

    if negative_count > 0:
        # Phi3V-style: negative IDs mark image placeholder positions in text
        # Actual layout after expansion: [text_prefix] [vision_tokens] [text_suffix]
        first_neg = int((input_ids < 0).nonzero(as_tuple=True)[0][0].item())
        # Use registry count if seq diff gives 0 (model expands in-place)
        if num_vision_tokens <= 0 and model_cfg:
            num_vision_tokens = negative_count  # negative IDs = vision placeholders
        pre_image_text = first_neg
        vision_start = pre_image_text
        vision_end = vision_start + num_vision_tokens
        text_start = 0  # text wraps around vision
        text_end = full_seq_len
        pre_image_tokens = pre_image_text
    elif num_vision_tokens <= 0 and model_cfg:
        # Processor already embedded image placeholders in input_ids.
        # Detect them by looking for the <image> token ID or repeated special IDs.
        n_vis = model_cfg.num_vision_tokens

        # Try to find image token ID from tokenizer
        image_token_id = None
        if hasattr(processor, "tokenizer"):
            tok = processor.tokenizer
        elif hasattr(processor, "tokenizer"):
            tok = processor.tokenizer
        else:
            tok = processor

        if hasattr(tok, "convert_tokens_to_ids"):
            for marker in ("<image>", "<img>", "<visual>"):
                try:
                    tid = tok.convert_tokens_to_ids(marker)
                    if tid != tok.unk_token_id:
                        image_token_id = tid
                        break
                except Exception:
                    pass

        if image_token_id is not None:
            # Find contiguous block of image token IDs
            img_mask = (input_ids == image_token_id)
            img_positions = img_mask.nonzero(as_tuple=True)[0]
            if len(img_positions) > 0:
                vision_start = int(img_positions[0].item())
                vision_end = int(img_positions[-1].item()) + 1
                num_vision_tokens = vision_end - vision_start
            else:
                # Fallback: use registry value, assume vision-first
                num_vision_tokens = n_vis
                vision_start = 0
                vision_end = n_vis
        else:
            # No marker found: use registry value, assume vision-first
            num_vision_tokens = n_vis
            vision_start = 0
            vision_end = n_vis

        # Text = everything outside vision range
        text_start = 0 if vision_start > 0 else vision_end
        text_end = full_seq_len
        pre_image_tokens = vision_start

        print(f"  (Processor-embedded vision tokens detected: {num_vision_tokens} at [{vision_start}:{vision_end}])")
    elif arch in ("llava", "paligemma"):
        # LLaVA / PaliGemma: vision tokens are typically prepended or placed
        # at a known offset. Use the seq_len difference to infer count.
        # Most HF LLaVA variants prepend vision before text.
        vision_start = 0
        vision_end = num_vision_tokens
        text_start = num_vision_tokens
        text_end = full_seq_len
        pre_image_tokens = 0
    else:
        # Default: vision-first layout (OpenVLA, SpatialVLA, etc.)
        vision_start = 0
        vision_end = num_vision_tokens
        text_start = num_vision_tokens
        text_end = full_seq_len
        pre_image_tokens = 0

    # Compute text-only query ranges (excluding vision tokens).
    # This ensures sink verification queries only from actual text tokens,
    # not accidentally sampling vision tokens as "text queries".
    text_query_ranges = []
    if vision_start > 0:
        text_query_ranges.append((0, vision_start))
    if vision_end < full_seq_len:
        text_query_ranges.append((vision_end, full_seq_len))
    if not text_query_ranges:
        # Fallback: original text range (e.g. vision_start=0, vision_end=seq_len)
        text_query_ranges = [(text_start, text_end)]

    boundaries = {
        "vision_start": vision_start,
        "vision_end": vision_end,
        "text_start": text_start,
        "text_end": text_end,
        "total_seq_len": full_seq_len,
        "num_vision_tokens": num_vision_tokens,
        "num_text_tokens": num_text_tokens,
        "pre_image_tokens": pre_image_tokens,
        "text_query_ranges": text_query_ranges,
    }

    print(f"  Token boundaries: {boundaries}")
    return boundaries


# ═══════════════════════════════════════════════════════════════════════════
# Hook-based attention extraction (fallback)
# ═══════════════════════════════════════════════════════════════════════════

class AttentionHookManager:
    """Register forward hooks to capture attention weights if
    model.generate() doesn't properly return them via output_attentions."""

    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.attention_weights = {}
        self._generation_step = 0

    def register_hooks(self):
        """Register hooks on all self-attention layers."""
        # Enable output_attentions in the language model config
        if hasattr(self.model, "language_model"):
            self.model.language_model.config.output_attentions = True
            layers = self.model.language_model.model.layers
        else:
            self.model.config.output_attentions = True
            layers = self.model.model.layers

        for i, layer in enumerate(layers):
            hook = layer.self_attn.register_forward_hook(
                self._make_hook(i)
            )
            self.hooks.append(hook)
        print("  Attention hooks registered on all layers.")

    def _make_hook(self, layer_idx):
        def hook_fn(module, args, output):
            # LlamaAttention returns (attn_output, attn_weights, past_key_value)
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                step_key = self._generation_step
                if step_key not in self.attention_weights:
                    self.attention_weights[step_key] = {}
                self.attention_weights[step_key][layer_idx] = output[1].detach().cpu()
        return hook_fn

    def new_generation_step(self):
        self._generation_step += 1

    def reset(self):
        self._generation_step = 0
        self.attention_weights = {}

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def get_attentions_for_step(self, step_idx):
        """Return list of (layer_idx, attn_tensor) for a generation step."""
        if step_idx in self.attention_weights:
            return self.attention_weights[step_idx]
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Top-K analysis
# ═══════════════════════════════════════════════════════════════════════════

def analyze_top_k(attn_weights, boundaries, tokenizer, input_ids_expanded, k=config.TOP_K):
    """Given attention weights (num_heads, seq_len), return top-k attended tokens.

    Args:
        attn_weights: tensor of shape (num_heads, 1, seq_len) or (num_heads, seq_len)
        boundaries: dict from detect_token_boundaries
        tokenizer: for decoding text tokens
        input_ids_expanded: the full input_ids after vision expansion (for text decoding)
        k: number of top tokens to return

    Returns:
        list of top-k dicts with rank, token_idx, type, score, and type-specific info
    """
    # Average over heads → (seq_len,)
    if attn_weights.dim() == 3:
        attn_mean = attn_weights[:, 0, :].mean(dim=0)  # (seq_len,)
    elif attn_weights.dim() == 2:
        attn_mean = attn_weights.mean(dim=0)  # (seq_len,)
    else:
        raise ValueError(f"Unexpected attention shape: {attn_weights.shape}")

    attn_mean = attn_mean.float()
    topk = attn_mean.topk(min(k, len(attn_mean)))

    vs = boundaries["vision_start"]
    ve = boundaries["vision_end"]
    ts = boundaries["text_start"]
    te = boundaries["text_end"]
    num_vision = boundaries["num_vision_tokens"]

    # Determine grid size from number of vision tokens
    # Prismatic dual encoder: DINOv2 (256) + SigLIP (256) = 512, or single = 256
    grid_size = int(np.sqrt(num_vision))
    if grid_size * grid_size != num_vision:
        # Dual encoder case: try half
        half = num_vision // 2
        grid_size_half = int(np.sqrt(half))
        if grid_size_half * grid_size_half == half:
            grid_size = grid_size_half
            dual_encoder = True
        else:
            grid_size = config.VISION_GRID_SIZE
            dual_encoder = False
    else:
        dual_encoder = False

    results = []
    for rank, (idx, score) in enumerate(zip(topk.indices, topk.values)):
        idx_val = idx.item()
        score_val = score.item()

        if vs <= idx_val < ve:
            # Vision token
            patch_idx = idx_val - vs
            if dual_encoder:
                half = num_vision // 2
                if patch_idx < half:
                    encoder = "encoder_0"
                    local_idx = patch_idx
                else:
                    encoder = "encoder_1"
                    local_idx = patch_idx - half
                row = local_idx // grid_size
                col = local_idx % grid_size
                token_info = {
                    "rank": rank,
                    "token_idx": idx_val,
                    "type": "vision",
                    "patch_idx": patch_idx,
                    "patch_row": row,
                    "patch_col": col,
                    "grid_size": grid_size,
                    "encoder": encoder,
                    "score": score_val,
                }
            else:
                row = patch_idx // grid_size
                col = patch_idx % grid_size
                token_info = {
                    "rank": rank,
                    "token_idx": idx_val,
                    "type": "vision",
                    "patch_idx": patch_idx,
                    "patch_row": row,
                    "patch_col": col,
                    "grid_size": grid_size,
                    "score": score_val,
                }

        elif ts <= idx_val < te:
            # Text token
            text_idx = idx_val - ts
            # Decode the token
            if input_ids_expanded is not None and idx_val < len(input_ids_expanded):
                token_text = tokenizer.decode([input_ids_expanded[idx_val].item()])
            else:
                token_text = f"<text_pos_{text_idx}>"
            token_info = {
                "rank": rank,
                "token_idx": idx_val,
                "type": "text",
                "text_idx": text_idx,
                "text": token_text,
                "score": score_val,
            }

        else:
            # Special token (BOS, or previously generated action tokens)
            if idx_val < vs:
                label = "bos_or_prefix"
            else:
                label = "action_token"
            token_info = {
                "rank": rank,
                "token_idx": idx_val,
                "type": "special",
                "label": label,
                "position": idx_val,
                "score": score_val,
            }

        results.append(token_info)

    return results


def compute_perhead_stats(attn_weights, boundaries):
    """Compute per-head attention breakdown by token category.

    Args:
        attn_weights: (num_heads, 1, seq_len) or (num_heads, seq_len)
            Attention from the last token (action token) to all other tokens.
        boundaries: dict from detect_token_boundaries
            Must contain: vision_start, vision_end, text_end, total_seq_len

    Returns:
        dict mapping "head_XX" → {vision_token0, vision_other, text_total,
        early_sink, ...}
    """
    if attn_weights.dim() == 3:
        attn = attn_weights[:, 0, :]  # (H, seq_len)
    else:
        attn = attn_weights  # (H, seq_len)

    attn = attn.float()
    num_heads = attn.shape[0]
    seq_len = attn.shape[1]
    vs = boundaries.get("vision_start", 0)
    ve = boundaries["vision_end"]
    te = boundaries.get("text_end", seq_len)

    results = {}
    for h in range(num_heads):
        ha = attn[h]  # (seq_len,)

        # Vision token 0 (first vision token — may not be position 0)
        v0 = ha[vs].item() if vs < seq_len else 0.0

        # Other vision tokens (vision_start+1 to vision_end-1)
        v_other = ha[vs+1:ve].sum().item() if ve > vs + 1 else 0.0
        v_other_max = ha[vs+1:ve].max().item() if ve > vs + 1 else 0.0
        v_other_argmax = int(ha[vs+1:ve].argmax().item()) + vs + 1 if ve > vs + 1 else -1

        # Text tokens — everything that's NOT vision and NOT action
        # For standard VLMs (vs=0): text is at [ve:te]
        # For Phi3V-style (vs>0): text is at [0:vs] + [ve:te]
        if vs > 0:
            # Text prefix before vision + text suffix after vision
            t_prefix = ha[0:vs].sum().item()
            t_suffix = ha[ve:te].sum().item() if te > ve else 0.0
            t_total = t_prefix + t_suffix
            # Find max text token across both ranges
            all_text_scores = []
            if vs > 0:
                all_text_scores.append((ha[0:vs], 0))  # (tensor, offset)
            if te > ve:
                all_text_scores.append((ha[ve:te], ve))
            t_max = 0.0
            t_argmax = -1
            for chunk, offset in all_text_scores:
                if len(chunk) > 0:
                    chunk_max = chunk.max().item()
                    if chunk_max > t_max:
                        t_max = chunk_max
                        t_argmax = int(chunk.argmax().item()) + offset
        else:
            t_total = ha[ve:te].sum().item() if te > ve else 0.0
            if te > ve:
                t_max = ha[ve:te].max().item()
                t_argmax = int(ha[ve:te].argmax().item()) + ve
            else:
                t_max = 0.0
                t_argmax = -1

        # Beyond text (previously generated action tokens)
        beyond = ha[te:].sum().item() if te < seq_len else 0.0
        beyond_max = ha[te:].max().item() if te < seq_len else 0.0

        # Generalized early sink: sum of first N positions (architecture-agnostic)
        # Captures BOS sinks, vision[0] sinks, special token sinks
        n_early = min(max(vs + 1, 1), seq_len)  # at least 1, up to vision_start+1
        early_sink = ha[:n_early].sum().item()

        results[f"head_{h:02d}"] = {
            "vision_token0": round(v0, 6),
            "vision_other": round(v_other, 6),
            "vision_other_max": round(v_other_max, 6),
            "vision_other_argmax": v_other_argmax,
            "text_total": round(t_total, 6),
            "text_max": round(t_max, 6),
            "text_argmax": t_argmax,
            "action_tokens": round(beyond, 6),
            "action_tokens_max": round(beyond_max, 6),
            "sink_total": round(v0 + t_max + beyond_max, 6),
            "early_sink": round(early_sink, 6),
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Action de-tokenization
# ═══════════════════════════════════════════════════════════════════════════

def detokenize_actions(model, token_ids, unnorm_key=config.BRIDGE_UNNORM_KEY):
    """Convert predicted action token IDs to continuous action values.

    Replicates OpenVLAForActionPrediction.predict_action() de-tokenization logic:
        1. token_id → bin index  (vocab_size - token_id - 1)
        2. bin index → normalized value  (bin_centers lookup, range [-1, 1])
        3. normalized → unnormalized  (using dataset action statistics)

    Args:
        model: the loaded OpenVLA model (has .config for vocab info, .norm_stats if available)
        token_ids: list of 7 integer token IDs
        unnorm_key: dataset key for unnormalization stats (default: "bridge_orig")

    Returns:
        dict with normalized_action, unnormalized_action, bin_indices
    """
    n_action_bins = getattr(model.config, "n_action_bins", 256)
    pad_to_multiple = getattr(model.config, "pad_to_multiple_of", 0)
    cfg = model.config
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "vocab_size"):
        vocab_size = cfg.text_config.vocab_size - pad_to_multiple
    elif hasattr(cfg, "vocab_size"):
        vocab_size = cfg.vocab_size - pad_to_multiple
    else:
        raise RuntimeError(f"Cannot determine vocab_size from model config: {type(cfg).__name__}")

    bins = np.linspace(-1, 1, n_action_bins + 1)  # 257 edges → 256 centers
    bin_centers = (bins[:-1] + bins[1:]) / 2.0     # length = n_action_bins = 256

    token_ids_np = np.array(token_ids)
    bin_indices = vocab_size - 1 - token_ids_np     # direct formula
    bin_indices = np.clip(bin_indices, 0, n_action_bins - 1)
    normalized_action = bin_centers[bin_indices].tolist()

    # Unnormalize if norm_stats available
    unnormalized_action = None
    norm_stats = getattr(model, "norm_stats", None) or getattr(model.config, "norm_stats", None)
    if norm_stats and unnorm_key in norm_stats:
        action_stats = norm_stats[unnorm_key]["action"]
        mask = np.array(action_stats.get("mask", [True] * len(normalized_action)))
        q99 = np.array(action_stats["q99"])
        q01 = np.array(action_stats["q01"])
        norm_arr = np.array(normalized_action)
        unnormalized_action = np.where(
            mask,
            0.5 * (norm_arr + 1) * (q99 - q01) + q01,
            norm_arr,
        ).tolist()

    return {
        "bin_indices": bin_indices.tolist(),
        "normalized_action": normalized_action,
        "unnormalized_action": unnormalized_action,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main extraction logic
# ═══════════════════════════════════════════════════════════════════════════

def load_model(device="cuda"):
    """Load OpenVLA model and processor."""
    print(f"Loading model: {config.MODEL_NAME}")
    processor = AutoProcessor.from_pretrained(
        config.MODEL_NAME, trust_remote_code=True
    )
    model = AutoModelForVision2Seq.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=getattr(torch, config.TORCH_DTYPE),
        trust_remote_code=True,
        attn_implementation="eager",  # Required: flash attention can't return weights
    ).to(device).eval()
    print("Model loaded successfully.")
    return processor, model


# ═══════════════════════════════════════════════════════════════════════════
# Universal model loading (registry-based)
# ═══════════════════════════════════════════════════════════════════════════

def _apply_spatialvla_monkey_patch():
    """Apply monkey-patch for SpatialVLA processor compatibility.

    SpatialVLA's custom processor references _validate_images_text_input_order
    which was removed from newer versions of transformers. This patch adds it
    back if missing.
    """
    import transformers.processing_utils as _proc_utils
    if not hasattr(_proc_utils, '_validate_images_text_input_order'):
        from transformers.image_utils import is_valid_image

        def _validate_images_text_input_order(images, text):
            """Validate (images, text) order. Swap if caller passed (text, images)."""
            if images is not None and not is_valid_image(images):
                if isinstance(images, str):
                    images, text = text, images
            return images, text

        _proc_utils._validate_images_text_input_order = _validate_images_text_input_order
        print("  Applied SpatialVLA monkey-patch: _validate_images_text_input_order")


def _apply_tracevla_torch_patch():
    """Bypass torch.load CVE-2025-32434 safety check for TraceVLA.

    TraceVLA uses .bin format weights (no safetensors available).
    Both torch.serialization and transformers.utils enforce version checks.
    This patch makes both no-ops for loading.
    """
    # Patch torch.serialization (torch >= 2.6)
    try:
        import torch.serialization as _ser
        if hasattr(_ser, 'check_torch_load_is_safe'):
            _ser.check_torch_load_is_safe = lambda *a, **kw: None
            print("  Applied TraceVLA torch.serialization safety bypass")
    except Exception:
        pass

    # Patch transformers.utils (transformers >= 4.57)
    try:
        import transformers.utils.import_utils as _tfu
        if hasattr(_tfu, 'check_torch_load_is_safe'):
            _tfu.check_torch_load_is_safe = lambda *a, **kw: None
            print("  Applied TraceVLA transformers safety bypass")
        # Also patch the modeling_utils reference
        import transformers.modeling_utils as _mu
        if hasattr(_mu, 'check_torch_load_is_safe'):
            _mu.check_torch_load_is_safe = lambda *a, **kw: None
    except Exception:
        pass


def _make_spatialvla_simple_processor(model_cfg):
    """Create a simple processor for SpatialVLA that avoids the broken
    custom processor class.

    SpatialVLA's custom processor imports _validate_images_text_input_order
    which may not exist. This processor loads image_processor and tokenizer
    separately and combines them.
    """
    from transformers import AutoImageProcessor, AutoTokenizer
    from transformers.feature_extraction_utils import BatchFeature

    _img_proc = AutoImageProcessor.from_pretrained(
        model_cfg.hf_id, trust_remote_code=True,
    )
    _tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.hf_id, trust_remote_code=True,
    )

    class _SimpleProcessor:
        def __init__(self, image_processor, tokenizer):
            self.image_processor = image_processor
            self.tokenizer = tokenizer

        def __call__(self, text, images, return_tensors="pt", **kwargs):
            image_inputs = self.image_processor(images, return_tensors=return_tensors)
            text_inputs = self.tokenizer(text, return_tensors=return_tensors, **kwargs)
            img_seq_len = getattr(self.image_processor, 'image_seq_length', 256)
            img_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
            img_ids = torch.full((1, img_seq_len), img_token_id, dtype=torch.long)
            text_ids = text_inputs["input_ids"]
            input_ids = torch.cat([img_ids, text_ids], dim=1)
            attention_mask = torch.ones_like(input_ids)
            intrinsic = torch.tensor([[[224.0, 0.0, 112.0],
                                       [0.0, 224.0, 112.0],
                                       [0.0, 0.0, 1.0]]])
            return BatchFeature({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": image_inputs["pixel_values"],
                "intrinsic": intrinsic,
            })

    return _SimpleProcessor(_img_proc, _tokenizer)


def load_model_from_registry(model_name, device="cuda"):
    """Load any VLA model from the registry with eager attention.

    Uses model_registry.get_model() to retrieve config, then loads the
    processor and model using the appropriate HuggingFace auto class
    determined by model_cfg.auto_model_class.

    Supported auto classes:
        - "AutoModelForVision2Seq" : OpenVLA, ECoT (standard VLMs)
        - "AutoModelForCausalLM"   : TraceVLA (Phi3V config not registered for Vision2Seq)
        - "AutoModel"              : SpatialVLA (custom architecture)

    Args:
        model_name: Key in the model registry (e.g., "openvla-7b", "spatialvla-4b")
        device: Target device (default: "cuda")

    Returns:
        Tuple of (processor, model, model_cfg)
    """
    model_cfg = _registry_get_model(model_name)

    print(f"Loading model from registry: {model_cfg.name} ({model_cfg.hf_id})")
    print(f"  Architecture: {model_cfg.architecture}, Auto class: {model_cfg.auto_model_class}")

    # Apply architecture-specific patches before loading
    if model_cfg.architecture == "phi3_v":
        _apply_tracevla_torch_patch()

    if model_cfg.auto_model_class == "AutoModel":
        _apply_spatialvla_monkey_patch()

    # Load processor (SpatialVLA needs special handling)
    if model_cfg.architecture == "gemma2" and model_cfg.auto_model_class == "AutoModel":
        processor = _make_spatialvla_simple_processor(model_cfg)
    else:
        processor = AutoProcessor.from_pretrained(
            model_cfg.hf_id,
            trust_remote_code=model_cfg.trust_remote_code,
        )

    # Select the correct auto class and load the model
    auto_class_map = {
        "AutoModelForVision2Seq": AutoModelForVision2Seq,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoModel": AutoModel,
    }

    auto_cls = auto_class_map.get(model_cfg.auto_model_class)
    if auto_cls is None:
        raise ValueError(
            f"Unknown auto_model_class '{model_cfg.auto_model_class}' for model "
            f"'{model_cfg.name}'. Supported: {list(auto_class_map.keys())}"
        )

    # Some architectures don't support attn_implementation="eager"
    # Gemma2/PaliGemma DO support eager; InternLM2 has custom attention dims
    skip_attn_impl = {"internlm2"}
    load_kwargs = dict(
        torch_dtype=getattr(torch, model_cfg.torch_dtype),
        trust_remote_code=model_cfg.trust_remote_code,
    )
    if model_cfg.architecture not in skip_attn_impl:
        load_kwargs["attn_implementation"] = model_cfg.attn_impl

    model = auto_cls.from_pretrained(
        model_cfg.hf_id, **load_kwargs,
    ).to(device).eval()

    print(f"  Loaded: {model_cfg.num_layers}L x {model_cfg.num_heads}H, "
          f"hidden={model_cfg.hidden_dim}, dtype={model_cfg.torch_dtype}")
    print(f"  Layers path: {model_cfg.layers_path}")
    print("Model loaded successfully.")

    return processor, model, model_cfg


def _unwrap_model(model):
    """Unwrap PeftModel / DDP wrappers to reach the base model."""
    # PeftModel wraps under .base_model.model
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    # DDP / DataParallel
    if hasattr(model, "module"):
        return model.module
    return model


def get_layers(model, model_cfg):
    """Navigate to the transformer layers using the model config's layers_path.

    The layers_path is a dot-separated string (e.g., "language_model.model.layers")
    that specifies how to traverse the model's module hierarchy to reach the
    nn.ModuleList of transformer layers.

    Handles PeftModel / DDP wrappers automatically.

    Args:
        model: The loaded model (may be wrapped by PeftModel or DDP)
        model_cfg: VLAModelConfig with layers_path attribute

    Returns:
        nn.ModuleList of transformer layers
    """
    # Try direct path first
    obj = model
    try:
        for attr in model_cfg.layers_path.split("."):
            obj = getattr(obj, attr)
        return obj
    except AttributeError:
        pass

    # Try after unwrapping PeftModel/DDP
    obj = _unwrap_model(model)
    for attr in model_cfg.layers_path.split("."):
        obj = getattr(obj, attr)
    return obj


def extract_for_step(processor, model, image, instruction, boundaries, device):
    """Run inference for a single step and extract attention analysis.

    Uses hook-based approach since OpenVLA's generate() doesn't return attentions.

    Returns:
        dict with predicted_action and attention_analysis
    """
    return _extract_with_hooks(processor, model, image, instruction, boundaries, device)


def _extract_with_hooks(processor, model, image, instruction, boundaries, device):
    """Fallback: extract attention using forward hooks."""
    tokenizer = processor.tokenizer
    prompt = config.PROMPT_TEMPLATE.format(instruction=instruction)
    inputs = processor(prompt, image, return_tensors="pt").to(device)
    # Cast pixel_values to model dtype (processor returns float32, model is bfloat16)
    if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

    # Build expanded input_ids for text token decoding
    original_input_ids = inputs["input_ids"][0]
    num_vision = boundaries["num_vision_tokens"]
    dummy_vision = torch.zeros(num_vision, dtype=torch.long)
    expanded_input_ids = torch.cat([dummy_vision, original_input_ids.cpu()])

    hook_manager = AttentionHookManager(model)
    hook_manager.register_hooks()

    # We need to run generate step by step to capture per-step attention
    # Use a manual autoregressive loop
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs.get("pixel_values")

    generated_tokens = []
    attention_analysis = {}
    perhead_analysis = {}

    with torch.no_grad():
        # First forward pass: process the full input (image + text)
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }

        for action_idx in range(config.NUM_ACTION_TOKENS):
            hook_manager.reset()
            hook_manager._generation_step = 0

            outputs = model(**model_inputs, use_cache=False)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_tokens.append(next_token.item() if next_token.dim() == 1 else next_token[0].item())

            # Collect attention from hooks
            dim_name = config.ACTION_DIM_NAMES[action_idx]
            action_key = f"action_{action_idx}_{dim_name}"
            attention_analysis[action_key] = {}
            perhead_analysis[action_key] = {}

            step_attns = hook_manager.get_attentions_for_step(0)
            if step_attns:
                for layer_idx in sorted(step_attns.keys()):
                    attn = step_attns[layer_idx]  # (1, heads, seq, seq)
                    # Take the last row (attention from last token)
                    attn_last = attn[0, :, -1:, :]  # (heads, 1, seq_len)

                    layer_key = f"layer_{layer_idx:02d}"
                    # Existing: head-averaged top5
                    top_k_results = analyze_top_k(
                        attn_last, boundaries, tokenizer, expanded_input_ids
                    )
                    attention_analysis[action_key][layer_key] = {"top5": top_k_results}

                    # NEW: per-head breakdown
                    perhead_analysis[action_key][layer_key] = compute_perhead_stats(
                        attn_last, boundaries
                    )

            # Append generated token for next step
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0) if next_token.dim() == 1 else next_token], dim=-1)
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(1, 1, device=device, dtype=attention_mask.dtype)
                ], dim=-1)
            # Must pass pixel_values every time since use_cache=False
            # (the model re-encodes the image each forward pass)
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
            }

    hook_manager.remove_hooks()

    predicted_tokens = [tokenizer.decode([tid]) for tid in generated_tokens]

    return {
        "predicted_tokens": predicted_tokens,
        "action_token_ids": generated_tokens,
        "attention_analysis": attention_analysis,
        "perhead_analysis": perhead_analysis,
    }


def run_extraction(episode_ids=None, device="cuda"):
    """Run attention extraction on specified episodes."""

    # ── Load metadata ─────────────────────────────────────────────────────
    if not config.METADATA_PATH.exists():
        print(f"ERROR: Metadata not found at {config.METADATA_PATH}")
        print("Run download_bridge_data.py first.")
        sys.exit(1)

    with open(config.METADATA_PATH) as f:
        metadata = json.load(f)

    # ── Filter episodes ───────────────────────────────────────────────────
    episodes = metadata["episodes"]
    if episode_ids is not None:
        episodes = [ep for ep in episodes if ep["episode_id"] in episode_ids]

    if not episodes:
        print("No episodes to process.")
        return

    # ── Load model ────────────────────────────────────────────────────────
    processor, model = load_model(device)
    tokenizer = processor.tokenizer

    # ── Detect token boundaries with first available sample ───────────────
    first_ep = episodes[0]
    first_step = first_ep["steps"][0]
    sample_image = Image.open(config.PROJECT_ROOT / first_step["image_path"])
    sample_instruction = first_step["instruction"]

    print("\nDetecting token boundaries...")
    boundaries = detect_token_boundaries(processor, model, sample_image, sample_instruction, device)

    # ── Create output directories ─────────────────────────────────────────
    config.ATTENTION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Process each episode/step ─────────────────────────────────────────
    total_steps = sum(len(ep["steps"]) for ep in episodes)
    print(f"\nProcessing {len(episodes)} episodes, {total_steps} total steps...")

    all_results = []
    pbar = tqdm(total=total_steps, desc="Extracting attention")

    for ep in episodes:
        ep_id = ep["episode_id"]

        for step in ep["steps"]:
            step_id = step["step_id"]
            image_path = config.PROJECT_ROOT / step["image_path"]
            instruction = step["instruction"]
            gt_action = step["action"]

            image = Image.open(image_path).convert("RGB")

            # Extract attention
            result = extract_for_step(
                processor, model, image, instruction, boundaries, device
            )

            # De-tokenize action token IDs to continuous values
            action_info = detokenize_actions(model, result["action_token_ids"])

            # Compute L1 error between predicted and ground truth
            pred_action = action_info["unnormalized_action"] or action_info["normalized_action"]
            l1_error = [abs(p - g) for p, g in zip(pred_action, gt_action)]
            l1_mean = sum(l1_error) / len(l1_error)

            # Build output JSON
            output = {
                "episode_id": ep_id,
                "step_id": step_id,
                "instruction": instruction,
                "ground_truth_action": gt_action,
                "predicted_action": pred_action,
                "action_token_ids": result["action_token_ids"],
                "action_details": action_info,
                "l1_error_per_dim": l1_error,
                "l1_error_mean": l1_mean,
                "predicted_tokens": result["predicted_tokens"],
                "token_boundaries": boundaries,
                "attention_analysis": result["attention_analysis"],
            }

            # Save per-step JSON (head-averaged, existing format)
            output_path = config.ATTENTION_RESULTS_DIR / f"ep{ep_id:03d}_step{step_id:03d}.json"
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

            # Save per-head JSON (NEW: per-head breakdown)
            if "perhead_analysis" in result:
                perhead_output = {
                    "episode_id": ep_id,
                    "step_id": step_id,
                    "instruction": instruction,
                    "token_boundaries": boundaries,
                    "perhead_analysis": result["perhead_analysis"],
                }
                perhead_path = config.ATTENTION_RESULTS_DIR / f"ep{ep_id:03d}_step{step_id:03d}_perhead.json"
                with open(perhead_path, "w") as f:
                    json.dump(perhead_output, f, indent=2, ensure_ascii=False)

                # Generate per-head heatmap visualizations
                try:
                    visualize_perhead_sink(perhead_path)
                except Exception as e:
                    print(f"  WARNING: Per-head visualization failed: {e}")

            all_results.append({
                "episode_id": ep_id,
                "step_id": step_id,
                "output_path": str(output_path.relative_to(config.PROJECT_ROOT)),
            })

            pbar.update(1)
            pbar.set_postfix(ep=ep_id, step=step_id)

    pbar.close()

    # ── Save index file ───────────────────────────────────────────────────
    index_path = config.ATTENTION_RESULTS_DIR / "index.json"
    with open(index_path, "w") as f:
        json.dump({
            "token_boundaries": boundaries,
            "results": all_results,
        }, f, indent=2)

    print(f"\nDone! Results saved to {config.ATTENTION_RESULTS_DIR}/")
    print(f"Index: {index_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Per-head visualization
# ═══════════════════════════════════════════════════════════════════════════

def visualize_perhead_sink(perhead_json_path, output_dir=None):
    """Generate per-head sink heatmaps from a _perhead.json file.

    Produces one heatmap per action token showing (layers × heads) with
    color = vision_token0 ratio. Also shows text_total and action_tokens
    as separate heatmaps side by side.

    Args:
        perhead_json_path: Path to an ep*_perhead.json file.
        output_dir: Where to save PNGs. Defaults to visualizations dir.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    perhead_json_path = Path(perhead_json_path)
    with open(perhead_json_path) as f:
        data = json.load(f)

    ep_id = data["episode_id"]
    step_id = data["step_id"]
    perhead = data["perhead_analysis"]

    if output_dir is None:
        output_dir = config.VISUALIZATIONS_DIR / f"ep{ep_id:03d}_step{step_id:03d}"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for action_key, layers_data in perhead.items():
        layer_keys = sorted(layers_data.keys())
        num_layers = len(layer_keys)

        # Detect number of heads from first layer
        first_layer_data = layers_data[layer_keys[0]]
        head_keys = sorted(first_layer_data.keys())
        num_heads = len(head_keys)

        # Build matrices: (num_layers, num_heads)
        mat_v0 = np.zeros((num_layers, num_heads))
        mat_v_other = np.zeros((num_layers, num_heads))
        mat_text = np.zeros((num_layers, num_heads))
        mat_action = np.zeros((num_layers, num_heads))

        for li, lk in enumerate(layer_keys):
            for hi, hk in enumerate(head_keys):
                hd = layers_data[lk][hk]
                mat_v0[li, hi] = hd["vision_token0"]
                mat_v_other[li, hi] = hd["vision_other"]
                mat_text[li, hi] = hd["text_total"]
                mat_action[li, hi] = hd["action_tokens"]

        # ── Figure: 4 heatmaps side by side ──
        fig, axes = plt.subplots(1, 4, figsize=(24, max(8, num_layers * 0.35)))
        fig.suptitle(
            f"{action_key} | ep{ep_id:03d} step{step_id:03d} — Per-Head Attention Breakdown\n"
            f"instruction: \"{data['instruction'][:60]}\"",
            fontsize=12, y=1.02,
        )

        titles = [
            "vision[0] (sink)",
            "vision[1:] (useful)",
            "text tokens",
            "action tokens",
        ]
        mats = [mat_v0, mat_v_other, mat_text, mat_action]
        cmaps = ["Reds", "Greens", "Blues", "Purples"]

        for ax, title, mat, cmap in zip(axes, titles, mats, cmaps):
            im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=max(0.01, mat.max()))
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Head", fontsize=8)
            ax.set_ylabel("Layer", fontsize=8)

            # Layer labels
            ax.set_yticks(range(0, num_layers, max(1, num_layers // 16)))
            layer_nums = [int(lk.split("_")[1]) for lk in layer_keys]
            ax.set_yticklabels(
                [str(layer_nums[i]) for i in range(0, num_layers, max(1, num_layers // 16))],
                fontsize=6,
            )
            ax.set_xticks(range(0, num_heads, max(1, num_heads // 8)))
            ax.set_xticklabels(
                [str(i) for i in range(0, num_heads, max(1, num_heads // 8))],
                fontsize=6,
            )
            plt.colorbar(im, ax=ax, shrink=0.8)

        plt.tight_layout()
        out_path = output_dir / f"{action_key}_perhead_sink.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ── Figure 2: Combined sink summary ──
        fig2, ax2 = plt.subplots(1, 1, figsize=(max(8, num_heads * 0.3), max(8, num_layers * 0.35)))
        # Combined: vision[0] + max_text + max_action = total "wasted" attention
        mat_combined = mat_v0 + mat_text * 0  # Only show vision[0] for clarity
        # Annotate cells with actual values
        im2 = ax2.imshow(mat_v0, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1.0)
        ax2.set_title(
            f"{action_key} — Vision Token 0 Attention per Head\n"
            f"(red = strong sink, white = no sink)",
            fontsize=10,
        )
        ax2.set_xlabel("Head", fontsize=9)
        ax2.set_ylabel("Layer", fontsize=9)
        ax2.set_yticks(range(num_layers))
        ax2.set_yticklabels([str(int(lk.split("_")[1])) for lk in layer_keys], fontsize=5)
        ax2.set_xticks(range(num_heads))
        ax2.set_xticklabels([str(i) for i in range(num_heads)], fontsize=5)
        plt.colorbar(im2, ax=ax2, label="Attention ratio to vision[0]")

        plt.tight_layout()
        out_path2 = output_dir / f"{action_key}_perhead_v0_heatmap.png"
        fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
        plt.close(fig2)

    print(f"Per-head visualizations saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Extract OpenVLA attention weights")
    parser.add_argument(
        "--episodes",
        type=str,
        default=None,
        help="Comma-separated episode IDs to process (default: all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--visualize-perhead",
        type=str,
        default=None,
        metavar="PATH",
        help="Standalone: generate per-head heatmaps from an existing _perhead.json file",
    )
    args = parser.parse_args()

    # Standalone per-head visualization mode
    if args.visualize_perhead:
        path = Path(args.visualize_perhead)
        if path.is_file():
            visualize_perhead_sink(path)
        elif path.is_dir():
            json_files = sorted(path.glob("*_perhead.json"))
            if not json_files:
                print(f"No *_perhead.json files found in {path}")
                return
            for jf in json_files:
                print(f"Processing {jf.name}...")
                visualize_perhead_sink(jf)
        else:
            print(f"Path not found: {path}")
        return

    episode_ids = None
    if args.episodes:
        episode_ids = [int(x) for x in args.episodes.split(",")]

    run_extraction(episode_ids=episode_ids, device=args.device)


if __name__ == "__main__":
    main()
