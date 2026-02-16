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
from transformers import AutoModelForVision2Seq, AutoProcessor

import config


# ═══════════════════════════════════════════════════════════════════════════
# Token boundary detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_token_boundaries(processor, model, sample_image, sample_instruction, device):
    """Detect vision/text token boundaries by running a forward pass.

    The <image> placeholder in input_ids gets expanded to N vision embeddings
    inside the model. We need to figure out the actual positions after expansion.

    Returns:
        dict with keys: vision_start, vision_end, text_start, text_end, total_seq_len
    """
    prompt = config.PROMPT_TEMPLATE.format(instruction=sample_instruction)
    inputs = processor(prompt, sample_image, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"][0]  # before expansion

    # Find <image> placeholder tokens in input_ids
    # OpenVLA/Prismatic uses a special token for image placeholder
    tokenizer = processor.tokenizer

    # Get the image token id (commonly 32000 for Prismatic/OpenVLA)
    image_token_str = "<image>"
    if image_token_str in tokenizer.get_vocab():
        image_token_id = tokenizer.convert_tokens_to_ids(image_token_str)
    else:
        # Fallback: search for common image token ids
        image_token_id = None
        for tok_str in ["<image>", "<img>", "<visual>"]:
            if tok_str in tokenizer.get_vocab():
                image_token_id = tokenizer.convert_tokens_to_ids(tok_str)
                break

    if image_token_id is not None:
        image_mask = (input_ids == image_token_id)
        num_image_placeholders = image_mask.sum().item()
        first_image_pos = image_mask.nonzero(as_tuple=True)[0][0].item() if num_image_placeholders > 0 else None
    else:
        num_image_placeholders = 0
        first_image_pos = None

    print(f"  Input IDs length (before expansion): {len(input_ids)}")
    print(f"  Image placeholders found: {num_image_placeholders}")

    # Run a forward pass to get the actual sequence length after vision embedding expansion
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            output_attentions=True,
            return_dict_in_generate=True,
            do_sample=False,
        )

    # The first generated token's attention tells us the full input seq_len
    # attentions[0][layer] → (batch, heads, 1, seq_len) for the first new token
    first_attn = outputs.attentions[0][0]  # first gen step, first layer
    full_seq_len = first_attn.shape[-1]  # includes all input tokens (expanded)

    print(f"  Full sequence length (after expansion): {full_seq_len}")

    # Calculate boundaries
    # Typical structure: [BOS] [vision_tokens...] [text_tokens...]
    # The number of non-image, non-special tokens before image
    pre_image_tokens = first_image_pos if first_image_pos is not None else 1  # BOS

    # Vision tokens = full_seq_len - (tokens before image) - (text tokens after image)
    # Text tokens after image = input_ids length - first_image_pos - num_image_placeholders
    if first_image_pos is not None:
        text_tokens_after_image = len(input_ids) - first_image_pos - num_image_placeholders
        num_vision_tokens = full_seq_len - pre_image_tokens - text_tokens_after_image
    else:
        # No image placeholder found; try to infer from seq_len difference
        num_vision_tokens = full_seq_len - len(input_ids)
        pre_image_tokens = 1  # BOS
        text_tokens_after_image = len(input_ids) - 1

    vision_start = pre_image_tokens
    vision_end = vision_start + num_vision_tokens
    text_start = vision_end
    text_end = full_seq_len

    boundaries = {
        "vision_start": vision_start,
        "vision_end": vision_end,
        "text_start": text_start,
        "text_end": text_end,
        "total_seq_len": full_seq_len,
        "num_vision_tokens": num_vision_tokens,
        "num_text_tokens": text_end - text_start,
        "pre_image_tokens": pre_image_tokens,
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
                self._make_hook(i), with_kwargs=True
            )
            self.hooks.append(hook)
        print("  Attention hooks registered on all layers.")

    def _make_hook(self, layer_idx):
        def hook_fn(module, args, output, **kwargs):
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


def extract_for_step(processor, model, image, instruction, boundaries, device):
    """Run inference for a single step and extract attention analysis.

    Returns:
        dict with predicted_action and attention_analysis
    """
    tokenizer = processor.tokenizer
    prompt = config.PROMPT_TEMPLATE.format(instruction=instruction)
    inputs = processor(prompt, image, return_tensors="pt").to(device)

    # ── Generate with attention ───────────────────────────────────────────
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.NUM_ACTION_TOKENS,
            output_attentions=True,
            return_dict_in_generate=True,
            do_sample=False,
        )

    # ── Check if attentions are available ─────────────────────────────────
    use_hook_fallback = False
    if outputs.attentions is None or len(outputs.attentions) == 0:
        print("  WARNING: output_attentions not returned. Using hook fallback.")
        use_hook_fallback = True

    if not use_hook_fallback:
        # Verify shape
        try:
            test_attn = outputs.attentions[0][0]  # first step, first layer
            if test_attn is None:
                use_hook_fallback = True
        except (IndexError, TypeError):
            use_hook_fallback = True

    if use_hook_fallback:
        return _extract_with_hooks(processor, model, image, instruction, boundaries, device)

    # ── Decode predicted action tokens ────────────────────────────────────
    generated_ids = outputs.sequences[0, inputs["input_ids"].shape[1]:]
    predicted_tokens = [tokenizer.decode([tid]) for tid in generated_ids]

    # ── Build input_ids in expanded form for text decoding ────────────────
    # During generation, the model internally expands <image> tokens to vision embeddings.
    # We need the expanded input_ids for text token decoding.
    # The attention seq_len tells us the expanded length.
    # For text token decoding, we reconstruct from the known boundaries.
    first_step_attn = outputs.attentions[0][0]  # (1, heads, 1, seq_len)
    expanded_seq_len = first_step_attn.shape[-1]

    # Build pseudo input_ids for text decoding:
    # We only need the text portion for decoding
    original_input_ids = inputs["input_ids"][0]
    # Find where text starts in original (after image placeholders)
    image_token_id = None
    for tok_str in ["<image>", "<img>", "<visual>"]:
        if tok_str in tokenizer.get_vocab():
            image_token_id = tokenizer.convert_tokens_to_ids(tok_str)
            break

    if image_token_id is not None:
        image_mask = (original_input_ids == image_token_id)
        num_placeholders = image_mask.sum().item()
        first_img_pos = image_mask.nonzero(as_tuple=True)[0][0].item() if num_placeholders > 0 else 0
        # Text tokens in original: everything after image placeholders
        text_ids_original = original_input_ids[first_img_pos + num_placeholders:]
        pre_image_ids = original_input_ids[:first_img_pos]
    else:
        pre_image_ids = original_input_ids[:1]  # BOS
        text_ids_original = original_input_ids[1:]

    # Build expanded input_ids: [pre_image] [dummy_vision * N] [text_ids]
    num_vision = boundaries["num_vision_tokens"]
    dummy_vision = torch.zeros(num_vision, dtype=torch.long)
    expanded_input_ids = torch.cat([pre_image_ids.cpu(), dummy_vision, text_ids_original.cpu()])

    # ── Extract top-K for each action token × each layer ──────────────────
    attention_analysis = {}
    num_action_steps = min(len(outputs.attentions), config.NUM_ACTION_TOKENS)

    for action_idx in range(num_action_steps):
        dim_name = config.ACTION_DIM_NAMES[action_idx]
        action_key = f"action_{action_idx}_{dim_name}"
        attention_analysis[action_key] = {}

        num_layers = len(outputs.attentions[action_idx])
        for layer_idx in range(num_layers):
            attn = outputs.attentions[action_idx][layer_idx]  # (1, heads, 1, seq_len)
            attn_weights = attn[0]  # (heads, 1, seq_len)

            layer_key = f"layer_{layer_idx:02d}"
            top_k_results = analyze_top_k(
                attn_weights, boundaries, tokenizer, expanded_input_ids
            )
            attention_analysis[action_key][layer_key] = {"top5": top_k_results}

    return {
        "predicted_tokens": predicted_tokens,
        "attention_analysis": attention_analysis,
    }


def _extract_with_hooks(processor, model, image, instruction, boundaries, device):
    """Fallback: extract attention using forward hooks."""
    tokenizer = processor.tokenizer
    prompt = config.PROMPT_TEMPLATE.format(instruction=instruction)
    inputs = processor(prompt, image, return_tensors="pt").to(device)

    hook_manager = AttentionHookManager(model)
    hook_manager.register_hooks()

    # We need to run generate step by step to capture per-step attention
    # Use a manual autoregressive loop
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs.get("pixel_values")

    generated_tokens = []
    attention_analysis = {}

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

            step_attns = hook_manager.get_attentions_for_step(0)
            if step_attns:
                for layer_idx in sorted(step_attns.keys()):
                    attn = step_attns[layer_idx]  # (1, heads, seq, seq)
                    # Take the last row (attention from last token)
                    attn_last = attn[0, :, -1:, :]  # (heads, 1, seq_len)

                    layer_key = f"layer_{layer_idx:02d}"
                    top_k_results = analyze_top_k(
                        attn_last, boundaries, tokenizer, None
                    )
                    attention_analysis[action_key][layer_key] = {"top5": top_k_results}

            # Append generated token for next step
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0) if next_token.dim() == 1 else next_token], dim=-1)
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(1, 1, device=device, dtype=attention_mask.dtype)
                ], dim=-1)
            # After first pass, don't send pixel_values again
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

    hook_manager.remove_hooks()

    predicted_tokens = [tokenizer.decode([tid]) for tid in generated_tokens]

    return {
        "predicted_tokens": predicted_tokens,
        "attention_analysis": attention_analysis,
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

            # Build output JSON
            output = {
                "episode_id": ep_id,
                "step_id": step_id,
                "instruction": instruction,
                "ground_truth_action": gt_action,
                "predicted_tokens": result["predicted_tokens"],
                "token_boundaries": boundaries,
                "attention_analysis": result["attention_analysis"],
            }

            # Save per-step JSON
            output_path = config.ATTENTION_RESULTS_DIR / f"ep{ep_id:03d}_step{step_id:03d}.json"
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

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
    args = parser.parse_args()

    episode_ids = None
    if args.episodes:
        episode_ids = [int(x) for x in args.episodes.split(",")]

    run_extraction(episode_ids=episode_ids, device=args.device)


if __name__ == "__main__":
    main()
