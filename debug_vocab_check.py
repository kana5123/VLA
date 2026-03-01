"""Check vocab_size mismatch between PeftModel and base model."""
import torch
from extract_attention import load_model_from_registry

device = "cuda:5"
processor, model, model_cfg = load_model_from_registry("openvla-7b", device)

print("=== Base Model ===")
cfg = model.config
pad = getattr(cfg, "pad_to_multiple_of", 0)
print(f"  pad_to_multiple_of: {pad}")
if hasattr(cfg, "text_config"):
    print(f"  text_config.vocab_size: {cfg.text_config.vocab_size}")
    print(f"  effective vocab: {cfg.text_config.vocab_size - pad}")
if hasattr(cfg, "vocab_size"):
    print(f"  config.vocab_size: {cfg.vocab_size}")
print(f"  n_action_bins: {getattr(cfg, 'n_action_bins', 'NOT SET')}")

from peft import PeftModel
peft_model = PeftModel.from_pretrained(model, "outputs/libero_ft/openvla-7b/libero_spatial/lora_adapter")

print("\n=== PeftModel ===")
cfg2 = peft_model.config
print(f"  type(config): {type(cfg2).__name__}")
print(f"  pad_to_multiple_of: {getattr(cfg2, 'pad_to_multiple_of', 'NOT SET')}")
if hasattr(cfg2, "text_config"):
    print(f"  text_config.vocab_size: {cfg2.text_config.vocab_size}")
    pad2 = getattr(cfg2, "pad_to_multiple_of", 0)
    print(f"  effective vocab: {cfg2.text_config.vocab_size - pad2}")
if hasattr(cfg2, "vocab_size"):
    print(f"  config.vocab_size: {cfg2.vocab_size}")
print(f"  n_action_bins: {getattr(cfg2, 'n_action_bins', 'NOT SET')}")

# Also check base_model path
base = peft_model.base_model.model if hasattr(peft_model, "base_model") else peft_model
cfg3 = base.config
print("\n=== PeftModel.base_model.model ===")
print(f"  type(config): {type(cfg3).__name__}")
print(f"  pad_to_multiple_of: {getattr(cfg3, 'pad_to_multiple_of', 'NOT SET')}")
if hasattr(cfg3, "text_config"):
    print(f"  text_config.vocab_size: {cfg3.text_config.vocab_size}")
    pad3 = getattr(cfg3, "pad_to_multiple_of", 0)
    print(f"  effective vocab: {cfg3.text_config.vocab_size - pad3}")
if hasattr(cfg3, "vocab_size"):
    print(f"  config.vocab_size: {cfg3.vocab_size}")
print(f"  n_action_bins: {getattr(cfg3, 'n_action_bins', 'NOT SET')}")

# Now test: encode action [0,0,0,0,0,0,0] → decode → check roundtrip
import numpy as np
from train_libero_lora import LiberoActionTokenizer
from extract_attention import detokenize_actions

# Training tokenizer uses base_model.model
train_tok = LiberoActionTokenizer(base)
test_action = np.array([0.1, -0.2, 0.3, -0.1, 0.05, -0.5, 0.0])
token_ids = train_tok.action_to_token_ids(test_action)
print(f"\n=== Roundtrip Test ===")
print(f"  Original action: {test_action}")
print(f"  Training token_ids: {token_ids}")
print(f"  Training decode: {train_tok.token_ids_to_action(token_ids)}")

# Inference detokenize uses model directly (PeftModel)
result = detokenize_actions(peft_model, token_ids)
print(f"  Inference decode (normalized_action): {result['normalized_action']}")
print(f"  Inference bin_indices: {result['bin_indices']}")

# Check if they match
train_decode = train_tok.token_ids_to_action(token_ids)
infer_decode = np.array(result['normalized_action'])
print(f"\n  MATCH: {np.allclose(train_decode, infer_decode)}")
if not np.allclose(train_decode, infer_decode):
    print(f"  DIFF: {train_decode - infer_decode}")
    print(f"  Train vocab_size={train_tok.vocab_size}")
