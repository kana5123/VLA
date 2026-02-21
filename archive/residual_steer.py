"""Residual stream steering for attention enhancement.

Instead of modifying attention weights (which breaks softmax normalization),
this adds a steering vector to the hidden states at a specific layer.
The steering direction = mean(object_patch_hidden) - mean(all_vision_hidden),
pushing the last token's representation toward object-focused features.
"""

from __future__ import annotations

from typing import Optional

import torch

import config


class ResidualSteerer:
    """Adds a steering vector to the residual stream at a target layer."""

    def __init__(self, alpha: float = config.RESIDUAL_STEER_ALPHA):
        self.alpha = alpha
        self.active = False
        self.object_patch_indices: list[int] = []
        self.vision_end: int = 256
        self.gripper_exempt: bool = False
        self.current_token_idx: int = 0
        self._handle: Optional[torch.utils.hooks.RemovableHook] = None

    def _hook_fn(self, module, args, output):
        if not self.active:
            return output
        if self.gripper_exempt and self.current_token_idx >= 6:
            return output

        # In transformers >=4.57, LlamaDecoderLayer returns a plain tensor;
        # in older versions it returns a tuple (hidden_states, ...).
        if isinstance(output, torch.Tensor):
            hidden_states = output
            is_tuple = False
        else:
            hidden_states = output[0]
            is_tuple = True

        if hidden_states.dim() != 3:
            return output

        seq_len = hidden_states.shape[1]

        valid_obj = [i for i in self.object_patch_indices if i < seq_len]
        vis_end = min(self.vision_end, seq_len)

        if not valid_obj or vis_end <= 0:
            return output

        obj_idx = torch.tensor(valid_obj, device=hidden_states.device)
        vis_idx = torch.arange(vis_end, device=hidden_states.device)

        obj_mean = hidden_states[:, obj_idx].mean(dim=1)  # (B, D)
        vis_mean = hidden_states[:, vis_idx].mean(dim=1)  # (B, D)
        steer_vec = obj_mean - vis_mean  # (B, D)

        modified = hidden_states.clone()
        modified[:, -1, :] = modified[:, -1, :] + self.alpha * steer_vec

        if is_tuple:
            return (modified,) + output[1:]
        return modified

    def install(self, model, layer_idx: int = config.RESIDUAL_STEER_LAYER) -> None:
        if self._handle is not None:
            return
        if hasattr(model, "language_model"):
            layers = model.language_model.model.layers
        else:
            layers = model.model.layers
        self._handle = layers[layer_idx].register_forward_hook(self._hook_fn)
        print(f"[residual_steer] Hook installed on layer {layer_idx}")

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
            print("[residual_steer] Hook removed")

    def configure(
        self,
        object_patches: list[int],
        vision_end: int,
        alpha: Optional[float] = None,
        gripper_exempt: bool = False,
    ) -> None:
        self.object_patch_indices = object_patches
        self.vision_end = vision_end
        if alpha is not None:
            self.alpha = alpha
        self.gripper_exempt = gripper_exempt
