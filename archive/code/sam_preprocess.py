"""SAM2 + GroundingDINO preprocessing for object-aware adapter v2.

Produces object_masks.dat — per-step binary masks indicating which
vision grid patches correspond to task-relevant objects.

Pipeline:
    1. Extract noun phrases from instructions (spaCy)
    2. GroundingDINO(image, noun_phrases) -> bounding boxes  [BATCHED]
    3. SAM2(image, boxes) -> pixel masks (256x256)
    4. Map pixel masks to vision grid (16x16) -> patch mask (V,)
    5. Store as memmap: object_masks.dat (total_steps, V) uint8

Usage:
    # Single GPU:
    python sam_preprocess.py --cache_dir /path/to/cache --device cuda:1

    # Multi-GPU (parallel workers):
    python sam_preprocess.py --cache_dir /path/to/cache --gpu_ids 1,2,3,4,5,6,7

    # Custom batch size (default 64):
    python sam_preprocess.py --cache_dir /path/to/cache --gpu_ids 1,2,3,4,5,6,7 --batch_size 128
"""

from __future__ import annotations

import json
import os
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
    """Extract noun phrases from a robot instruction using spaCy."""
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


def instruction_to_grounding_query(instruction: str) -> str:
    """Convert instruction to a single GroundingDINO query string.

    GroundingDINO requires period-terminated phrases for reliable detection.
    """
    phrases = extract_noun_phrases(instruction)
    return " ".join(p.strip() + "." for p in phrases)


def pixel_mask_to_patch_mask(
    pixel_mask: np.ndarray,
    grid_size: int = config.VISION_GRID_SIZE,
    threshold: float = config.SAM_PATCH_OVERLAP_THRESHOLD,
    vision_tokens: int = 256,
) -> np.ndarray:
    """Convert pixel-level mask (H, W) to vision grid patch mask (V,)."""
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

    patch_mask = patch_mask_2d.flatten()

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
    """Process a single image through GroundingDINO + SAM2 pipeline."""
    H, W = image.shape[:2]
    img_area = H * W

    query = instruction_to_grounding_query(instruction)

    try:
        result = grounding_fn(image, query)
        boxes = result["boxes"]
        scores = result["scores"]
    except Exception:
        return None

    if len(boxes) == 0:
        return None

    keep = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        box_area = (x2 - x1) * (y2 - y1)
        if box_area / img_area <= max_area_fraction:
            keep.append(i)

    if len(keep) == 0:
        return None

    filtered_boxes = boxes[keep]

    try:
        pixel_mask = sam_fn(image, filtered_boxes)
    except Exception:
        return None

    if pixel_mask is None or pixel_mask.sum() == 0:
        return None

    return pixel_mask_to_patch_mask(pixel_mask, vision_tokens=vision_tokens)


# ── Batched model loading and inference ─────────────────────────────────


def load_models(device: str = "cuda"):
    """Load GroundingDINO and SAM2 models in fp16 for fast batched inference.

    Returns:
        (gd_processor, gd_model, sam_predictor)
    """
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    # GroundingDINO in fp16
    gd_processor = AutoProcessor.from_pretrained(config.GROUNDING_MODEL_ID)
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        config.GROUNDING_MODEL_ID,
    ).to(device=device, dtype=torch.float16).eval()

    # SAM2 in fp32 (fp16 causes dtype mismatch in set_image)
    sam_predictor = SAM2ImagePredictor.from_pretrained(config.SAM2_MODEL_ID)
    sam_predictor.model.to(device=device).eval()

    return gd_processor, gd_model, sam_predictor


def batched_grounding_detect(
    gd_processor, gd_model, pil_images: list, queries: list[str], device: str,
) -> list[dict]:
    """Run GroundingDINO on a batch of images at once.

    Returns:
        List of {"boxes": (N,4) np, "scores": (N,) np} per image.
    """
    import torch

    inputs = gd_processor(
        images=pil_images, text=queries, return_tensors="pt", padding=True,
    ).to(device=device, dtype=torch.float16)

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        outputs = gd_model(**inputs)

    target_sizes = [img.size[::-1] for img in pil_images]
    batch_results = gd_processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=config.GROUNDING_BOX_THRESHOLD,
        text_threshold=config.GROUNDING_TEXT_THRESHOLD,
        target_sizes=target_sizes,
    )

    results = []
    for r in batch_results:
        results.append({
            "boxes": r["boxes"].cpu().numpy(),
            "scores": r["scores"].cpu().numpy(),
        })
    return results


def sam_segment(sam_predictor, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Run SAM2 on one image with given boxes. Returns combined (H,W) mask."""
    sam_predictor.set_image(image)
    masks, _, _ = sam_predictor.predict(box=boxes, multimask_output=False)
    combined = np.zeros(image.shape[:2], dtype=np.uint8)
    for mask in masks:
        if mask.ndim == 3:
            mask = mask[0]
        combined = np.maximum(combined, mask.astype(np.uint8))
    return combined


# ── Batched preprocessing loop ──────────────────────────────────────────


def preprocess_batched(
    cache_dir: Path,
    device: str = "cuda",
    vision_tokens: int = 256,
    batch_size: int = 64,
    sam_sub_bs: int = 16,
    start_idx: int = 0,
    end_idx: int | None = None,
    worker_id: int = 0,
) -> dict:
    """Run batched GroundingDINO + batched SAM2 on cached images.

    GroundingDINO runs on full batches (batch_size images at once).
    SAM2 runs in sub-batches via set_image_batch() + predict_batch().
    """
    import torch
    from PIL import Image as PILImage

    # Limit CPU threads to prevent thrashing across workers
    torch.set_num_threads(8)

    cache_dir = Path(cache_dir)

    with open(cache_dir / "cache_info.json") as f:
        info = json.load(f)
    total_steps = info["total_steps"]
    img_h, img_w = info["image_height"], info["image_width"]

    if end_idx is None:
        end_idx = total_steps

    with open(cache_dir / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    images_mmap = np.memmap(
        str(cache_dir / "images.dat"), dtype=np.uint8, mode="r",
        shape=(total_steps, img_h, img_w, 3),
    )

    masks_path = cache_dir / config.SAM_MASKS_FILENAME
    masks_mmap = np.memmap(
        str(masks_path), dtype=np.uint8, mode="r+",
        shape=(total_steps, vision_tokens),
    )

    # Load models
    tag = f"[GPU {worker_id}]"
    print(f"{tag} Loading GroundingDINO (fp16) + SAM2 (fp32) on {device}...")
    gd_processor, gd_model, sam_predictor = load_models(device)
    print(f"{tag} Models loaded. GD batch={batch_size}, SAM sub-batch={sam_sub_bs}")

    chunk_size = end_idx - start_idx
    success_count = 0
    failure_count = 0
    t0 = time.time()
    max_area = config.GROUNDING_MAX_AREA_FRACTION
    img_area = img_h * img_w

    print(f"{tag} Processing steps {start_idx}..{end_idx} ({chunk_size} steps)")

    for batch_start_i in range(0, chunk_size, batch_size):
        batch_end_i = min(batch_start_i + batch_size, chunk_size)
        actual_bs = batch_end_i - batch_start_i

        # ── 1. Load batch of images + queries (CPU) ──────────────────
        batch_indices = list(range(start_idx + batch_start_i, start_idx + batch_end_i))
        pil_images = []
        queries = []
        np_images = []

        for idx in batch_indices:
            img_np = np.array(images_mmap[idx])
            np_images.append(img_np)
            pil_images.append(PILImage.fromarray(img_np))
            queries.append(instruction_to_grounding_query(metadata[idx]["instruction"]))

        # ── 2. Batched GroundingDINO detection (GPU) ─────────────────
        try:
            gd_results = batched_grounding_detect(
                gd_processor, gd_model, pil_images, queries, device,
            )
        except Exception:
            failure_count += actual_bs
            continue

        # ── 3. Collect images with valid detections for batched SAM2 ─
        sam_images = []
        sam_boxes = []
        sam_global_indices = []

        for j, (idx, gd_res) in enumerate(zip(batch_indices, gd_results)):
            boxes = gd_res["boxes"]
            if len(boxes) == 0:
                failure_count += 1
                continue

            keep = [k for k, box in enumerate(boxes)
                    if (box[2] - box[0]) * (box[3] - box[1]) / img_area <= max_area]
            if len(keep) == 0:
                failure_count += 1
                continue

            sam_images.append(np_images[j])
            sam_boxes.append(boxes[keep])
            sam_global_indices.append(idx)

        # ── 4. Batched SAM2 segmentation (sub-batches to avoid OOM) ──
        for sb_start in range(0, len(sam_images), sam_sub_bs):
            sb_end = min(sb_start + sam_sub_bs, len(sam_images))
            sb_imgs = sam_images[sb_start:sb_end]
            sb_boxes = sam_boxes[sb_start:sb_end]
            sb_indices = sam_global_indices[sb_start:sb_end]

            try:
                sam_predictor.set_image_batch(sb_imgs)
                masks_batch, _, _ = sam_predictor.predict_batch(
                    box_batch=sb_boxes, multimask_output=False,
                )
            except Exception:
                failure_count += len(sb_imgs)
                continue

            for mi, (idx, masks_per_img) in enumerate(zip(sb_indices, masks_batch)):
                combined = np.zeros(sb_imgs[mi].shape[:2], dtype=np.uint8)
                for mask in masks_per_img:
                    if mask.ndim == 3:
                        mask = mask[0]
                    combined = np.maximum(combined, mask.astype(np.uint8))

                if combined.sum() == 0:
                    failure_count += 1
                    continue

                patch_mask = pixel_mask_to_patch_mask(combined, vision_tokens=vision_tokens)
                masks_mmap[idx] = patch_mask
                success_count += 1

        # ── 5. Progress logging ──────────────────────────────────────
        done = batch_end_i
        if done % (batch_size * 10) < batch_size or done == chunk_size:
            masks_mmap.flush()
            elapsed = time.time() - t0
            rate = done / elapsed
            eta_s = (chunk_size - done) / max(rate, 1)
            eta_m = eta_s / 60
            print(
                f"  {tag} [{done}/{chunk_size}] "
                f"success={success_count}, fail={failure_count} "
                f"({rate:.1f} steps/s, ETA {eta_m:.0f}m)"
            )

    masks_mmap.flush()
    del masks_mmap

    stats = {
        "worker_id": worker_id,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "total": chunk_size,
        "success": success_count,
        "failure": failure_count,
        "failure_rate": failure_count / max(chunk_size, 1),
    }
    elapsed = time.time() - t0
    print(f"{tag} Done in {elapsed / 60:.0f}m: {stats}")
    return stats


# ── Legacy single-image interface (for tests) ──────────────────────────


def load_grounding_and_sam(device: str = "cuda"):
    """Load models and return (grounding_fn, sam_fn) callables.

    Legacy interface kept for tests and single-image use.
    """
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from sam2.sam2_image_predictor import SAM2ImagePredictor

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
            threshold=config.GROUNDING_BOX_THRESHOLD,
            text_threshold=config.GROUNDING_TEXT_THRESHOLD,
            target_sizes=[pil_img.size[::-1]],
        )[0]
        return {
            "boxes": results["boxes"].cpu().numpy(),
            "scores": results["scores"].cpu().numpy(),
        }

    sam_predictor = SAM2ImagePredictor.from_pretrained(config.SAM2_MODEL_ID)
    sam_predictor.model.to(device)

    def sam_fn(image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        masks, _, _ = sam_predictor.predict(box=boxes, multimask_output=False)
        combined = np.zeros(image.shape[:2], dtype=np.uint8)
        for mask in masks:
            if mask.ndim == 3:
                mask = mask[0]
            combined = np.maximum(combined, mask.astype(np.uint8))
        return combined

    return grounding_fn, sam_fn


# ── Keep legacy function for backward compat ────────────────────────────

def preprocess_all_steps(
    cache_dir: Path,
    grounding_fn: Callable,
    sam_fn: Callable,
    vision_tokens: int = 256,
    start_idx: int = 0,
    end_idx: int | None = None,
    worker_id: int = 0,
) -> dict:
    """Legacy sequential processing (kept for backward compat / tests)."""
    cache_dir = Path(cache_dir)

    with open(cache_dir / "cache_info.json") as f:
        info = json.load(f)
    total_steps = info["total_steps"]
    img_h, img_w = info["image_height"], info["image_width"]

    if end_idx is None:
        end_idx = total_steps

    with open(cache_dir / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    images = np.memmap(
        str(cache_dir / "images.dat"), dtype=np.uint8, mode="r",
        shape=(total_steps, img_h, img_w, 3),
    )

    masks_path = cache_dir / config.SAM_MASKS_FILENAME
    if not masks_path.exists():
        # Initialize masks file with failure markers (same as _init_masks_file)
        masks_init = np.memmap(
            str(masks_path), dtype=np.uint8, mode="w+",
            shape=(total_steps, vision_tokens),
        )
        masks_init[:] = config.SAM_FAILURE_MARKER
        masks_init.flush()
        del masks_init
    masks = np.memmap(
        str(masks_path), dtype=np.uint8, mode="r+",
        shape=(total_steps, vision_tokens),
    )

    chunk_size = end_idx - start_idx
    success_count = 0
    failure_count = 0
    t0 = time.time()
    tag = f"[GPU {worker_id}]"

    print(f"{tag} Processing steps {start_idx}..{end_idx} ({chunk_size} steps)")

    for i, idx in enumerate(range(start_idx, end_idx)):
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

        done = i + 1
        if done % 5000 == 0:
            masks.flush()
            elapsed = time.time() - t0
            rate = done / elapsed
            eta_s = (chunk_size - done) / max(rate, 1)
            eta_h = eta_s / 3600
            print(
                f"  {tag} [{done}/{chunk_size}] "
                f"success={success_count}, fail={failure_count} "
                f"({rate:.0f} steps/s, ETA {eta_h:.1f}h)"
            )

    masks.flush()
    del masks

    stats = {
        "worker_id": worker_id,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "total": chunk_size,
        "success": success_count,
        "failure": failure_count,
        "failure_rate": failure_count / max(chunk_size, 1),
    }
    elapsed = time.time() - t0
    print(f"{tag} Done in {elapsed / 3600:.1f}h: {stats}")
    return stats


# ── Mask file init ──────────────────────────────────────────────────────


def _init_masks_file(cache_dir: Path, vision_tokens: int):
    """Create and initialize object_masks.dat with failure markers."""
    with open(cache_dir / "cache_info.json") as f:
        info = json.load(f)
    total_steps = info["total_steps"]

    masks_path = cache_dir / config.SAM_MASKS_FILENAME
    masks = np.memmap(
        str(masks_path), dtype=np.uint8, mode="w+",
        shape=(total_steps, vision_tokens),
    )
    masks[:] = config.SAM_FAILURE_MARKER
    masks.flush()
    del masks
    print(f"Initialized {masks_path} ({total_steps} x {vision_tokens})")
    return total_steps


# ── CLI entry point ─────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse
    import subprocess
    import sys

    parser = argparse.ArgumentParser(description="SAM preprocessing for adapter v2")
    parser.add_argument("--cache_dir", type=str, default=str(config.DATA_CACHE_DIR))
    parser.add_argument("--vision_tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--gpu_ids", type=str, default=None,
        help="Comma-separated GPU IDs for multi-GPU, e.g. '1,2,3,4,5,6,7'",
    )
    # Worker-mode args (used by subprocess workers)
    parser.add_argument("--sam_sub_bs", type=int, default=16,
                        help="SAM2 sub-batch size for set_image_batch()")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--worker_id", type=int, default=0)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)

    if args.gpu_ids:
        # ── Multi-GPU: launch separate subprocesses per GPU ──────────
        gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
        num_workers = len(gpu_ids)

        total_steps = _init_masks_file(cache_dir, args.vision_tokens)

        chunk_size = total_steps // num_workers
        procs = []
        for i, gpu_id in enumerate(gpu_ids):
            start = i * chunk_size
            end = total_steps if i == num_workers - 1 else (i + 1) * chunk_size
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["PYTHONUNBUFFERED"] = "1"
            env["OMP_NUM_THREADS"] = "8"
            env["MKL_NUM_THREADS"] = "8"
            cmd = [
                sys.executable, __file__,
                "--cache_dir", str(cache_dir),
                "--vision_tokens", str(args.vision_tokens),
                "--batch_size", str(args.batch_size),
                "--sam_sub_bs", str(args.sam_sub_bs),
                "--device", "cuda",
                "--start_idx", str(start),
                "--end_idx", str(end),
                "--worker_id", str(gpu_id),
            ]
            log_path = cache_dir / f"sam_worker_gpu{gpu_id}.log"
            log_f = open(log_path, "w")
            p = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
            procs.append((p, log_f, gpu_id, start, end))
            print(f"Launched worker on GPU {gpu_id}: steps {start}..{end} (PID {p.pid}, log: {log_path})")

        print(f"\nAll {num_workers} workers launched. Waiting for completion...")
        print(f"Monitor with: tail -f {cache_dir}/sam_worker_gpu*.log")
        for p, log_f, gpu_id, start, end in procs:
            p.wait()
            log_f.close()
            status = "OK" if p.returncode == 0 else f"FAILED (rc={p.returncode})"
            print(f"  GPU {gpu_id}: {status}")
        print("All workers finished.")

    else:
        # ── Single GPU / Worker mode (batched) ───────────────────────
        import torch

        # Initialize masks file only if not already done (worker mode skips)
        masks_path = cache_dir / config.SAM_MASKS_FILENAME
        if not masks_path.exists():
            _init_masks_file(cache_dir, args.vision_tokens)

        with open(cache_dir / "cache_info.json") as f:
            total_steps = json.load(f)["total_steps"]
        end_idx = args.end_idx if args.end_idx is not None else total_steps

        stats = preprocess_batched(
            cache_dir=cache_dir,
            device=args.device,
            vision_tokens=args.vision_tokens,
            batch_size=args.batch_size,
            sam_sub_bs=args.sam_sub_bs,
            start_idx=args.start_idx,
            end_idx=end_idx,
            worker_id=args.worker_id,
        )
        print(f"Done: {stats}")
