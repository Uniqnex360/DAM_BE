"""
Production-ready embedded text removal.

Pipeline
--------
1. Multi-pass EasyOCR detection (standard + aggressive low-confidence pass)
2. Adaptive mask construction
   - Convex-hull polygon fill per detected word box
   - Dilation kernel scaled to image resolution
   - Connected-component merging of nearby blobs (gap ≤ 20 px)
   - Coverage guard: abort if mask > 40 % of image (false-positive protection)
3. LaMa AI inpainting for seamless background reconstruction
4. Graceful per-region fallback to border-color fill if LaMa is unavailable
"""

import logging
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from ..model_registry import get_lama, get_ocr_reader

logger = logging.getLogger(__name__)

# ── Tuning constants ──────────────────────────────────────────────────────────
_OCR_STD_CONF = 0.50          # Confidence threshold — standard pass
_OCR_AGG_CONF = 0.30          # Confidence threshold — aggressive pass (small/faint text)
_OCR_TEXT_THRESH_STD = 0.5    # EasyOCR text_threshold for standard pass
_OCR_TEXT_THRESH_AGG = 0.25   # EasyOCR text_threshold for aggressive pass
_OCR_LINK_THRESH = 0.3        # EasyOCR link_threshold (both passes)
_OCR_LOW_TEXT = 0.3           # EasyOCR low_text (both passes)
_MIN_BBOX_PX = 6              # Minimum bbox side-length to consider (px)
_DILATION_SCALE = 0.006       # Dilation kernel = max(h,w) × this, minimum 8 px
_MERGE_GAP_PX = 20            # Connected-component merge distance (px)
_MAX_MASK_COVERAGE = 0.40     # Abort if mask covers > 40 % of image
_LAMA_MAX_DIM = 2048          # Limit for LaMa inference (memory safety)
# ─────────────────────────────────────────────────────────────────────────────


def _build_dilation_kernel(h: int, w: int) -> np.ndarray:
    """Return a square dilation kernel whose size is proportional to the image."""
    side = max(8, int(max(h, w) * _DILATION_SCALE))
    # Ensure odd size so kernel has a well-defined centre
    side = side if side % 2 == 1 else side + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))


def _detections_to_mask(
    detections: List[Tuple],
    h: int,
    w: int,
    kernel: np.ndarray,
) -> np.ndarray:
    """
    Convert EasyOCR detections to a binary inpaint mask.

    Each detection is (bbox, text, prob) where bbox is a list of four
    (x, y) corner points.  We fill the convex hull of those points so the
    mask is tighter than a plain bounding rectangle.
    """
    mask = np.zeros((h, w), dtype=np.uint8)

    for bbox, _text, _prob in detections:
        points = np.array(bbox, dtype=np.float32)
        hull = cv2.convexHull(points.reshape(-1, 1, 2))
        hull_int = hull.astype(np.int32)
        cv2.fillPoly(mask, [hull_int], 255)

    # Dilate so the full character anti-aliasing fringe is covered
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


def _merge_nearby_components(mask: np.ndarray) -> np.ndarray:
    """
    Merge connected components that are within _MERGE_GAP_PX of each other.

    This ensures text on the same line (with small gaps between letters/words)
    is inpainted as a single region, which avoids ugly seam lines between calls.
    """
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (_MERGE_GAP_PX * 2 + 1, _MERGE_GAP_PX * 2 + 1),
    )
    dilated = cv2.dilate(mask, close_kernel, iterations=1)
    eroded = cv2.erode(dilated, close_kernel, iterations=1)
    return eroded


def _lama_inpaint(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Run LaMa inpainting on the full image with the given binary mask.

    Downscales if either dimension exceeds _LAMA_MAX_DIM to stay within
    memory limits, then upscales the result back to original size.
    """
    lama = get_lama()
    h, w = image_bgr.shape[:2]

    # --- Optional downscale for large images ---
    scale = 1.0
    if max(h, w) > _LAMA_MAX_DIM:
        scale = _LAMA_MAX_DIM / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        logger.debug(f"LaMa downscale: {w}×{h} → {new_w}×{new_h}")

    pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    pil_mask = Image.fromarray(mask)

    result_pil = lama(pil_img, pil_mask)
    result_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)

    # --- Upscale back if we downscaled ---
    if scale < 1.0:
        result_bgr = cv2.resize(result_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)

    return result_bgr


def _color_fill_fallback(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Simple fallback: fill masked regions with the median border colour.

    Used only when LaMa is unavailable.  Works well for plain/white backgrounds.
    """
    result = image_bgr.copy()
    h, w = image_bgr.shape[:2]

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]

        # Guard against tiny or zero-area ROIs
        if bw < _MIN_BBOX_PX or bh < _MIN_BBOX_PX:
            continue

        x1, y1 = max(0, x - 4), max(0, y - 4)
        x2, y2 = min(w, x + bw + 4), min(h, y + bh + 4)
        roi = image_bgr[y1:y2, x1:x2]

        roi_h, roi_w = roi.shape[:2]
        if roi_h < 4 or roi_w < 4:
            # ROI too small to sample a valid border
            avg_color = np.array([255, 255, 255], dtype=np.uint8)
        else:
            border_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
            border_mask[0, :] = 255
            border_mask[-1, :] = 255
            border_mask[:, 0] = 255
            border_mask[:, -1] = 255
            border_pixels = roi[border_mask > 0]
            avg_color = np.median(border_pixels, axis=0).astype(np.uint8)

        region_mask = (labels[y1:y2, x1:x2] == i).astype(np.uint8) * 255
        dilated_region = cv2.dilate(region_mask, np.ones((3, 3), np.uint8))
        result[y1:y2, x1:x2][dilated_region > 0] = avg_color

    return result


class TextRemovalStep:
    """
    Production-grade embedded text removal step.

    Detection  : EasyOCR (two passes — standard and aggressive)
    Mask       : Convex-hull polygon fill, adaptive dilation, component merging
    Inpainting : LaMa (SimpleLama) with full-resolution output
    Fallback   : Border-colour fill when LaMa is unavailable
    """

    def process(self, image: np.ndarray, original: np.ndarray) -> np.ndarray:
        try:
            logger.info("TextRemovalStep: starting production text removal …")
            h, w = image.shape[:2]
            reader = get_ocr_reader()

            # ── Stage 1: Multi-pass OCR detection ────────────────────────────
            # Standard pass — high-confidence text (headlines, labels, prices)
            std_results = reader.readtext(
                image,
                text_threshold=_OCR_TEXT_THRESH_STD,
                link_threshold=_OCR_LINK_THRESH,
                low_text=_OCR_LOW_TEXT,
                paragraph=False,
                batch_size=4,
            )
            std_detections = [
                (bbox, txt, prob)
                for (bbox, txt, prob) in std_results
                if prob >= _OCR_STD_CONF and _bbox_is_valid(bbox, _MIN_BBOX_PX)
            ]

            # Aggressive pass — catches small/faint embedded text
            agg_results = reader.readtext(
                image,
                text_threshold=_OCR_TEXT_THRESH_AGG,
                link_threshold=_OCR_LINK_THRESH,
                low_text=_OCR_LOW_TEXT,
                paragraph=False,
                batch_size=4,
            )
            agg_detections = [
                (bbox, txt, prob)
                for (bbox, txt, prob) in agg_results
                if _OCR_AGG_CONF <= prob < _OCR_STD_CONF
                and _bbox_is_valid(bbox, _MIN_BBOX_PX)
                and not _is_duplicate(bbox, std_detections)
            ]

            all_detections = std_detections + agg_detections

            if not all_detections:
                logger.info("TextRemovalStep: no text detected — image unchanged.")
                return image

            logger.info(
                f"TextRemovalStep: {len(std_detections)} standard + "
                f"{len(agg_detections)} aggressive detections = "
                f"{len(all_detections)} total."
            )

            # ── Stage 2: Mask construction ────────────────────────────────────
            kernel = _build_dilation_kernel(h, w)
            raw_mask = _detections_to_mask(all_detections, h, w, kernel)
            merged_mask = _merge_nearby_components(raw_mask)

            # Coverage guard — bail out if mask looks like a false positive
            coverage = float(np.count_nonzero(merged_mask)) / (h * w)
            logger.info(f"TextRemovalStep: mask coverage = {coverage * 100:.1f} %")
            if coverage > _MAX_MASK_COVERAGE:
                logger.warning(
                    f"TextRemovalStep: mask coverage {coverage*100:.1f}% exceeds "
                    f"{_MAX_MASK_COVERAGE*100:.0f}% threshold — aborting to avoid "
                    "destroying the image (likely false positives)."
                )
                return image

            if coverage == 0.0:
                logger.info("TextRemovalStep: empty mask after construction — image unchanged.")
                return image

            # ── Stage 3: LaMa AI inpainting ───────────────────────────────────
            try:
                result = _lama_inpaint(image, merged_mask)
                logger.info(
                    f"TextRemovalStep: LaMa inpainting complete "
                    f"({len(all_detections)} regions, {coverage*100:.1f}% coverage)."
                )
                return result

            except Exception as lama_err:
                logger.warning(
                    f"TextRemovalStep: LaMa inpainting failed ({lama_err}), "
                    "falling back to colour-fill method."
                )
                result = _color_fill_fallback(image, merged_mask)
                logger.info("TextRemovalStep: colour-fill fallback complete.")
                return result

        except Exception as e:
            logger.error(f"TextRemovalStep: unexpected error — {e}", exc_info=True)
            # Never crash the pipeline; return the original image
            return image


# ── Helper functions ──────────────────────────────────────────────────────────

def _bbox_is_valid(bbox, min_px: int) -> bool:
    """Return True only if the bounding box has sufficient size to process."""
    try:
        pts = np.array(bbox, dtype=np.float32)
        x, y, bw, bh = cv2.boundingRect(pts.reshape(-1, 1, 2).astype(np.int32))
        return bw >= min_px and bh >= min_px
    except Exception:
        return False


def _iou(bbox_a, bbox_b) -> float:
    """Compute approximate IoU between two EasyOCR bboxes."""
    try:
        pts_a = np.array(bbox_a, dtype=np.int32)
        pts_b = np.array(bbox_b, dtype=np.int32)
        xa, ya, wa, ha = cv2.boundingRect(pts_a.reshape(-1, 1, 2))
        xb, yb, wb, hb = cv2.boundingRect(pts_b.reshape(-1, 1, 2))

        ix1, iy1 = max(xa, xb), max(ya, yb)
        ix2, iy2 = min(xa + wa, xb + wb), min(ya + ha, yb + hb)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        inter = (ix2 - ix1) * (iy2 - iy1)
        union = wa * ha + wb * hb - inter
        return inter / max(union, 1)
    except Exception:
        return 0.0


def _is_duplicate(bbox, existing: list, iou_thresh: float = 0.4) -> bool:
    """Return True if bbox significantly overlaps any bbox in existing."""
    for (ex_bbox, _, _) in existing:
        if _iou(bbox, ex_bbox) >= iou_thresh:
            return True
    return False
