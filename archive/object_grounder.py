"""Object grounding pipeline: instruction -> nouns -> GroundingDINO -> SAM2 -> patch indices.

V2: Grounded-SAM2 pipeline with Dream-inspired architecture.
- Noun filtering: removes abstract/positional nouns from SpaCy output
- Area filtering + NMS: from Dream's ObjectSegmentor (prevents oversized detections)
- SAM2 pixel masks: precise object contours instead of bounding boxes
- Mask-to-patch conversion: 16x16 grid overlap with configurable threshold

References:
- GroundingDINO (ECCV 2024): Open-set detection
- SAM2 (Meta, 2024): Segment Anything Model 2
- Dream pipeline: NMS + area filter + label merging architecture
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import numpy as np
from PIL import Image

import config


# ── Noun filtering ──────────────────────────────────────────────────

ABSTRACT_NOUNS = {
    "side", "top", "bottom", "middle", "left", "right",
    "front", "back", "end", "part", "piece", "area",
    "it", "them", "this", "that", "one", "thing",
    "way", "place", "spot", "point", "direction",
}


# ── Data class ──────────────────────────────────────────────────────

@dataclass
class GroundingResult:
    nouns: list[str]
    boxes_xyxy: list[list[float]]  # pixel coordinates [x1,y1,x2,y2]
    scores: list[float]
    labels: list[str]
    patch_indices: list[int]       # flat indices in vision token range
    masks: Optional[np.ndarray] = None  # (N, H, W) bool masks
    grid_size: int = 16
    image_wh: tuple[int, int] = (0, 0)
    patch_coverage: float = 0.0


# ── Main class ──────────────────────────────────────────────────────

class ObjectGrounder:
    """Grounded-SAM2 pipeline with Dream-style filtering.

    Pipeline:
        1. SpaCy noun extraction + filtering
        2. GroundingDINO detection
        3. NMS + area filtering (from Dream)
        4. SAM2 pixel-level segmentation
        5. Same-label merging (from Dream)
        6. Mask → 16x16 grid patch indices
    """

    def __init__(
        self,
        device: str = "cuda",
        box_threshold: float = config.GROUNDING_BOX_THRESHOLD,
        text_threshold: float = config.GROUNDING_TEXT_THRESHOLD,
        nms_iou_threshold: float = config.GROUNDING_NMS_IOU_THRESHOLD,
        max_area_fraction: float = config.GROUNDING_MAX_AREA_FRACTION,
        patch_overlap_threshold: float = config.SAM_PATCH_OVERLAP_THRESHOLD,
    ):
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_area_fraction = max_area_fraction
        self.patch_overlap_threshold = patch_overlap_threshold

        self._nlp = None
        self._gdino_processor = None
        self._gdino_model = None
        self._sam2_processor = None
        self._sam2_model = None

    # ── Lazy model loading ──────────────────────────────────────

    def _load_spacy(self):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")

    def _load_gdino(self):
        if self._gdino_model is None:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            self._gdino_processor = AutoProcessor.from_pretrained(
                config.GROUNDING_MODEL_ID
            )
            self._gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                config.GROUNDING_MODEL_ID
            ).to(self.device).eval()

    def _load_sam2(self):
        if self._sam2_model is None:
            from transformers import Sam2Processor, Sam2Model
            self._sam2_processor = Sam2Processor.from_pretrained(
                config.SAM2_MODEL_ID
            )
            self._sam2_model = Sam2Model.from_pretrained(
                config.SAM2_MODEL_ID
            ).to(self.device).eval()

    # ── Step 1: Noun extraction with filtering ──────────────────

    def extract_nouns(self, instruction: str) -> list[str]:
        """Extract physical object nouns from instruction.

        Filters out abstract nouns (side, top, piece), pronouns (it, them),
        and positional terms (middle, left, right).
        """
        if not instruction or not instruction.strip():
            return []

        self._load_spacy()
        doc = self._nlp(instruction)

        nouns = []
        for chunk in doc.noun_chunks:
            lemma = chunk.root.lemma_.lower()
            # Skip abstract/positional nouns
            if lemma in ABSTRACT_NOUNS:
                continue
            # Skip pronouns
            if chunk.root.pos_ == "PRON":
                continue
            nouns.append(lemma)

        # Fallback: individual NOUN tokens if no chunks found
        if not nouns:
            for t in doc:
                if t.pos_ in ("NOUN", "PROPN"):
                    lemma = t.lemma_.lower()
                    if lemma not in ABSTRACT_NOUNS:
                        nouns.append(lemma)

        # Deduplicate preserving order
        seen = set()
        unique = []
        for n in nouns:
            if n not in seen:
                seen.add(n)
                unique.append(n)
        return unique

    # ── Step 2: GroundingDINO detection ─────────────────────────

    @torch.no_grad()
    def detect_objects(
        self, image: Image.Image, nouns: list[str],
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Run GroundingDINO. Returns pixel-coordinate boxes, scores, labels."""
        if not nouns:
            return np.zeros((0, 4)), np.array([]), []

        self._load_gdino()
        text_prompt = " . ".join(nouns) + " ."

        inputs = self._gdino_processor(
            images=image, text=text_prompt, return_tensors="pt"
        ).to(self.device)

        outputs = self._gdino_model(**inputs)

        results = self._gdino_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],  # (H, W)
        )[0]

        boxes = results["boxes"].cpu().numpy()  # (N, 4) pixel coords
        scores = results["scores"].cpu().numpy()
        labels = list(results.get("text_labels", results.get("labels", [])))

        return boxes, scores, labels

    # ── Step 3: NMS + area filtering (from Dream) ──────────────

    def _filter_detections(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: list[str],
        image_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Filter oversized and duplicate detections (Dream architecture)."""
        if len(boxes) == 0:
            return boxes, scores, labels

        H, W = image_shape
        image_area = H * W

        # Remove boxes covering > max_area_fraction of image
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        area_mask = areas < (self.max_area_fraction * image_area)

        # Class-agnostic NMS
        keep_nms = self._nms(boxes, scores, self.nms_iou_threshold)
        nms_mask = np.zeros(len(boxes), dtype=bool)
        nms_mask[keep_nms] = True

        combined = area_mask & nms_mask
        filtered_labels = [l for l, m in zip(labels, combined) if m]
        return boxes[combined], scores[combined], filtered_labels

    @staticmethod
    def _nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float,
    ) -> list[int]:
        """Non-Maximum Suppression (from Dream)."""
        if len(boxes) == 0:
            return []

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    # ── Step 4: SAM2 segmentation ──────────────────────────────

    @torch.no_grad()
    def segment_with_sam2(
        self,
        image: Image.Image,
        boxes: np.ndarray,
    ) -> np.ndarray:
        """Run SAM2 segmentation using bounding boxes as prompts.

        Args:
            image: PIL image
            boxes: (N, 4) pixel-coordinate boxes [x1, y1, x2, y2]

        Returns:
            masks: (N, H, W) bool numpy array
        """
        if len(boxes) == 0:
            w, h = image.size
            return np.zeros((0, h, w), dtype=bool)

        self._load_sam2()

        # SAM2 expects input_boxes as list[list[list[float]]]
        # outer=batch, middle=boxes_per_image, inner=coords
        input_boxes = [boxes.tolist()]

        inputs = self._sam2_processor(
            images=image,
            input_boxes=input_boxes,
            return_tensors="pt",
        ).to(self.device)

        outputs = self._sam2_model(**inputs, multimask_output=False)

        # Post-process masks to original resolution
        masks = self._sam2_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"],
        )[0]  # first image in batch

        # masks shape: (num_boxes, 1, H, W) bool tensor
        # Squeeze the single-mask dimension
        masks_np = masks[:, 0].numpy().astype(bool)  # (N, H, W)

        return masks_np

    # ── Step 5: Same-label merging (from Dream) ────────────────

    @staticmethod
    def _merge_same_label(
        masks: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Merge detections with the same label into single instances."""
        if len(labels) == 0:
            return masks, boxes, scores, labels

        label_groups: dict[str, list[int]] = {}
        for i, label in enumerate(labels):
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(i)

        merged_masks = []
        merged_boxes = []
        merged_scores = []
        merged_labels = []

        for label, indices in label_groups.items():
            # OR-merge all masks with this label
            combined_mask = np.zeros_like(masks[0], dtype=bool)
            for idx in indices:
                combined_mask |= masks[idx]

            # Bounding box of merged mask
            ys, xs = np.where(combined_mask)
            if len(ys) > 0:
                merged_box = np.array(
                    [xs.min(), ys.min(), xs.max(), ys.max()],
                    dtype=boxes.dtype,
                )
            else:
                merged_box = boxes[indices[0]]

            best_score = max(scores[idx] for idx in indices)

            merged_masks.append(combined_mask)
            merged_boxes.append(merged_box)
            merged_scores.append(best_score)
            merged_labels.append(label)

        return (
            np.array(merged_masks),
            np.array(merged_boxes),
            np.array(merged_scores),
            merged_labels,
        )

    # ── Step 6: Mask → patch indices ───────────────────────────

    @staticmethod
    def masks_to_patch_indices(
        masks: np.ndarray,
        image_hw: tuple[int, int],
        grid_size: int,
        num_vision_tokens: int,
        overlap_threshold: float = 0.1,
    ) -> list[int]:
        """Convert SAM2 binary masks to 16x16 grid patch indices.

        For each grid cell, computes the fraction of pixels covered by any
        mask. If overlap >= threshold, that patch is selected.

        For dual-encoder (512 tokens), maps to both encoder_0 and encoder_1.
        """
        H, W = image_hw
        cell_h = H / grid_size
        cell_w = W / grid_size
        dual = (num_vision_tokens == grid_size * grid_size * 2)
        per_enc = grid_size * grid_size

        # Union all masks into single binary mask
        if len(masks) == 0:
            return []
        union_mask = np.zeros((H, W), dtype=bool)
        for m in masks:
            union_mask |= m

        patch_set = set()
        for r in range(grid_size):
            for c in range(grid_size):
                r_start = int(r * cell_h)
                r_end = int((r + 1) * cell_h)
                c_start = int(c * cell_w)
                c_end = int((c + 1) * cell_w)

                cell = union_mask[r_start:r_end, c_start:c_end]
                if cell.size == 0:
                    continue

                overlap = cell.sum() / cell.size
                if overlap >= overlap_threshold:
                    idx = r * grid_size + c
                    patch_set.add(idx)
                    if dual:
                        patch_set.add(idx + per_enc)

        return sorted(patch_set)

    # ── Legacy bbox-only method (backward compat) ──────────────

    @staticmethod
    def boxes_to_patch_indices(
        boxes_xyxy_norm: list[list[float]],
        grid_size: int,
        num_vision_tokens: int,
    ) -> list[int]:
        """Convert normalized bounding boxes to patch grid indices (legacy)."""
        patch_set = set()
        dual = (num_vision_tokens == grid_size * grid_size * 2)
        per_enc = grid_size * grid_size

        for box in boxes_xyxy_norm:
            x1, y1, x2, y2 = [max(0.0, min(1.0, v)) for v in box]
            col_s = int(x1 * grid_size)
            col_e = min(grid_size - 1, int(x2 * grid_size))
            row_s = int(y1 * grid_size)
            row_e = min(grid_size - 1, int(y2 * grid_size))

            for r in range(row_s, row_e + 1):
                for c in range(col_s, col_e + 1):
                    idx = r * grid_size + c
                    patch_set.add(idx)
                    if dual:
                        patch_set.add(idx + per_enc)

        return sorted(patch_set)

    # ── Full pipeline: Grounded-SAM2 ───────────────────────────

    def ground(
        self,
        image: Image.Image,
        instruction: str,
        num_vision_tokens: int,
        grid_size: int = config.VISION_GRID_SIZE,
        use_sam2: bool = True,
    ) -> GroundingResult:
        """Full pipeline: instruction -> nouns -> boxes -> SAM2 masks -> patch indices."""
        nouns = self.extract_nouns(instruction)

        # Detect objects
        boxes, scores, labels = self.detect_objects(image, nouns)

        w, h = image.size
        image_hw = (h, w)

        # Filter: NMS + area
        boxes, scores, labels = self._filter_detections(
            boxes, scores, labels, image_hw,
        )

        if use_sam2 and len(boxes) > 0:
            # SAM2 segmentation
            masks = self.segment_with_sam2(image, boxes)

            # Merge same-label instances
            masks, boxes, scores, labels = self._merge_same_label(
                masks, boxes, scores, labels,
            )

            # Mask → patch indices
            patch_indices = self.masks_to_patch_indices(
                masks, image_hw, grid_size, num_vision_tokens,
                self.patch_overlap_threshold,
            )
        else:
            masks = None
            # Fallback: bbox → normalized → legacy patch indices
            boxes_norm = []
            for box in boxes:
                x1, y1, x2, y2 = box
                boxes_norm.append([x1 / w, y1 / h, x2 / w, y2 / h])
            patch_indices = self.boxes_to_patch_indices(
                boxes_norm, grid_size, num_vision_tokens,
            )

        coverage = len(patch_indices) / num_vision_tokens if num_vision_tokens > 0 else 0

        return GroundingResult(
            nouns=nouns,
            boxes_xyxy=boxes.tolist() if isinstance(boxes, np.ndarray) and len(boxes) > 0 else [],
            scores=scores.tolist() if isinstance(scores, np.ndarray) else [],
            labels=labels,
            patch_indices=patch_indices,
            masks=masks,
            grid_size=grid_size,
            image_wh=image.size,
            patch_coverage=coverage,
        )


def precompute_grounding_for_episode(
    grounder: ObjectGrounder,
    episode: dict,
    num_vision_tokens: int,
    grid_size: int = config.VISION_GRID_SIZE,
    use_sam2: bool = True,
) -> dict:
    """Pre-run grounding for all steps. Returns step_id -> GroundingResult."""
    results = {}
    for step in episode["steps"]:
        image = Image.open(config.PROJECT_ROOT / step["image_path"]).convert("RGB")
        result = grounder.ground(
            image, step["instruction"], num_vision_tokens, grid_size,
            use_sam2=use_sam2,
        )
        results[step["step_id"]] = result
    return results
