"""SAM2 + GroundingDINO preprocessing for object-aware adapter v2.

Produces object_masks.dat — per-step binary masks indicating which
vision grid patches correspond to task-relevant objects.

Pipeline:
    1. Extract noun phrases from instructions (spaCy)
    2. GroundingDINO(image, noun_phrases) -> bounding boxes
    3. SAM2(image, boxes) -> pixel masks (256x256)
    4. Map pixel masks to vision grid (16x16) -> patch mask (V,)
    5. Store as memmap: object_masks.dat (total_steps, V) uint8

Usage:
    python sam_preprocess.py [--cache_dir PATH] [--vision_tokens 256] [--device cuda]
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import spacy

import config

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def extract_noun_phrases(instruction: str) -> list[str]:
    """Extract noun phrases from a robot instruction using spaCy.

    Args:
        instruction: e.g. "pick up the blue cup from the table"

    Returns:
        List of noun phrases, e.g. ["blue cup", "table"]
        Falls back to full instruction if no nouns found.
    """
    nlp = _get_nlp()
    doc = nlp(instruction)

    phrases = []
    for chunk in doc.noun_chunks:
        text = chunk.text.strip()
        if len(text) > 1 and chunk.root.pos_ != "PRON":
            phrases.append(text)

    if not phrases:
        phrases = [instruction.strip()]

    return phrases


def instruction_to_grounding_queries(instruction: str) -> list[str]:
    """Convert instruction to GroundingDINO query strings.

    Returns:
        List of individual query strings (each used separately for robustness).
    """
    return extract_noun_phrases(instruction)


def pixel_mask_to_patch_mask(
    pixel_mask: np.ndarray,
    grid_size: int = config.VISION_GRID_SIZE,
    threshold: float = config.SAM_PATCH_OVERLAP_THRESHOLD,
    vision_tokens: int = 256,
) -> np.ndarray:
    """Convert pixel-level mask (H, W) to vision grid patch mask (V,).

    Divides the image into grid_size x grid_size cells. A cell is marked
    as 'object' if the fraction of masked pixels exceeds threshold.

    Args:
        pixel_mask: (H, W) binary mask from SAM2
        grid_size: number of patches per side (16 for OpenVLA)
        threshold: minimum overlap fraction to mark a patch
        vision_tokens: 256 (single encoder) or 512 (dual encoder)

    Returns:
        patch_mask: (vision_tokens,) uint8 array, 1=object, 0=background
    """
    H, W = pixel_mask.shape
    cell_h = H // grid_size
    cell_w = W // grid_size

    patch_mask_2d = np.zeros((grid_size, grid_size), dtype=np.uint8)

    for r in range(grid_size):
        for c in range(grid_size):
            cell = pixel_mask[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
            overlap = cell.sum() / (cell_h * cell_w)
            if overlap >= threshold:
                patch_mask_2d[r, c] = 1

    patch_mask = patch_mask_2d.flatten()  # (grid_size^2,) = (256,)

    # Dual encoder: duplicate mask for both encoders
    if vision_tokens > len(patch_mask):
        patch_mask = np.concatenate([patch_mask, patch_mask])

    return patch_mask[:vision_tokens]


def process_single_image(
    image: np.ndarray,
    instruction: str,
    grounding_fn: Callable,
    sam_fn: Callable,
    vision_tokens: int = 256,
    max_area_fraction: float = config.GROUNDING_MAX_AREA_FRACTION,
) -> Optional[np.ndarray]:
    """Process a single image through GroundingDINO + SAM2 pipeline.

    Args:
        image: (H, W, 3) uint8 image
        instruction: robot instruction text
        grounding_fn: callable(image, query) -> {"boxes": (N,4), "scores": (N,)}
        sam_fn: callable(image, boxes) -> (H, W) binary mask
        vision_tokens: 256 or 512

    Returns:
        patch_mask: (vision_tokens,) uint8 or None if detection fails
    """
    H, W = image.shape[:2]
    img_area = H * W

    queries = instruction_to_grounding_queries(instruction)

    all_boxes = []
    all_scores = []

    for query in queries:
        try:
            result = grounding_fn(image, query)
            boxes = result["boxes"]
            scores = result["scores"]

            if len(boxes) == 0:
                continue

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                box_area = (x2 - x1) * (y2 - y1)
                if box_area / img_area <= max_area_fraction:
                    all_boxes.append(box)
                    all_scores.append(scores[i])
        except Exception:
            continue

    if len(all_boxes) == 0:
        return None

    all_boxes = np.array(all_boxes)

    try:
        pixel_mask = sam_fn(image, all_boxes)
    except Exception:
        return None

    if pixel_mask is None or pixel_mask.sum() == 0:
        return None

    return pixel_mask_to_patch_mask(pixel_mask, vision_tokens=vision_tokens)


def preprocess_all_steps(
    cache_dir: Path,
    grounding_fn: Callable,
    sam_fn: Callable,
    vision_tokens: int = 256,
) -> dict:
    """Run GroundingDINO + SAM2 on all cached images and save object_masks.dat.

    Args:
        cache_dir: Path containing images.dat, metadata.pkl, cache_info.json
        grounding_fn: callable(image, query) -> {"boxes": ..., "scores": ...}
        sam_fn: callable(image, boxes) -> (H, W) mask
        vision_tokens: 256 or 512

    Returns:
        dict with stats: {total, success, failure, failure_rate}
    """
    cache_dir = Path(cache_dir)

    with open(cache_dir / "cache_info.json") as f:
        info = json.load(f)
    total_steps = info["total_steps"]
    img_h, img_w = info["image_height"], info["image_width"]

    with open(cache_dir / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    images = np.memmap(
        str(cache_dir / "images.dat"), dtype=np.uint8, mode="r",
        shape=(total_steps, img_h, img_w, 3),
    )

    masks_path = cache_dir / config.SAM_MASKS_FILENAME
    masks = np.memmap(
        str(masks_path), dtype=np.uint8, mode="w+",
        shape=(total_steps, vision_tokens),
    )
    # Initialize all to failure marker
    masks[:] = config.SAM_FAILURE_MARKER

    success_count = 0
    failure_count = 0
    t0 = time.time()

    for idx in range(total_steps):
        image = np.array(images[idx])
        instruction = metadata[idx]["instruction"]

        result = process_single_image(
            image, instruction,
            grounding_fn=grounding_fn,
            sam_fn=sam_fn,
            vision_tokens=vision_tokens,
        )

        if result is not None:
            masks[idx] = result
            success_count += 1
        else:
            failure_count += 1

        if (idx + 1) % 10000 == 0:
            masks.flush()
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            print(
                f"  [{idx + 1}/{total_steps}] "
                f"success={success_count}, fail={failure_count} "
                f"({rate:.0f} steps/s)"
            )

    masks.flush()
    del masks

    stats = {
        "total": total_steps,
        "success": success_count,
        "failure": failure_count,
        "failure_rate": failure_count / max(total_steps, 1),
    }
    print(f"SAM preprocessing complete: {stats}")
    return stats


def load_grounding_and_sam(device: str = "cuda"):
    """Load GroundingDINO and SAM2 models.

    Returns:
        (grounding_fn, sam_fn) callables
    """
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    # GroundingDINO
    gd_processor = AutoProcessor.from_pretrained(config.GROUNDING_MODEL_ID)
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        config.GROUNDING_MODEL_ID
    ).to(device)

    def grounding_fn(image: np.ndarray, query: str) -> dict:
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(image)
        inputs = gd_processor(images=pil_img, text=query, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = gd_model(**inputs)
        results = gd_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=config.GROUNDING_BOX_THRESHOLD,
            text_threshold=config.GROUNDING_TEXT_THRESHOLD,
            target_sizes=[pil_img.size[::-1]],
        )[0]
        return {
            "boxes": results["boxes"].cpu().numpy(),
            "scores": results["scores"].cpu().numpy(),
        }

    # SAM2
    sam_predictor = SAM2ImagePredictor.from_pretrained(config.SAM2_MODEL_ID)
    sam_predictor.model.to(device)

    def sam_fn(image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        masks, _, _ = sam_predictor.predict(
            box=boxes, multimask_output=False,
        )
        combined = np.zeros(image.shape[:2], dtype=np.uint8)
        for mask in masks:
            if mask.ndim == 3:
                mask = mask[0]
            combined = np.maximum(combined, mask.astype(np.uint8))
        return combined

    return grounding_fn, sam_fn


if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser(description="SAM preprocessing for adapter v2")
    parser.add_argument("--cache_dir", type=str, default=str(config.DATA_CACHE_DIR))
    parser.add_argument("--vision_tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("Loading GroundingDINO + SAM2...")
    grounding_fn, sam_fn = load_grounding_and_sam(args.device)

    print("Starting SAM preprocessing...")
    stats = preprocess_all_steps(
        cache_dir=Path(args.cache_dir),
        grounding_fn=grounding_fn,
        sam_fn=sam_fn,
        vision_tokens=args.vision_tokens,
    )
    print(f"Done: {stats}")
