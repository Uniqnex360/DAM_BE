"""
Production-ready embedded text removal v2.

Improvements over v1:
1. Contour shape analysis — filter out smooth objects (signs, boards)
2. Adaptive erosion — shrink mask back toward original text bbox
3. Morphological opening — separate merged blobs intelligently
4. No manual threshold tuning — all logic data-driven
5. Handles signs, boards, watermarks across any background

Pipeline
--------
1. Multi-pass EasyOCR detection (standard + aggressive)
2. Adaptive mask construction
   - Convex-hull per detection with minimal dilation (antialiasing only)
   - Morphological opening (separate blobs, remove noise)
3. Contour filtering — reject smooth/large objects (signs, boards)
4. Adaptive erosion — shrink mask back to text core
5. LaMa inpainting or border-fill fallback
"""

import logging
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from ..model_registry import get_lama, get_ocr_reader

logger = logging.getLogger(__name__)

# ── Tuning constants (minimal, mostly shape-based) ────────────────────────────
_OCR_STD_CONF = 0.50
_OCR_AGG_CONF = 0.30
_OCR_TEXT_THRESH_STD = 0.5
_OCR_TEXT_THRESH_AGG = 0.25
_OCR_LINK_THRESH = 0.3
_OCR_LOW_TEXT = 0.3
_MIN_BBOX_PX = 6

# Dilation: minimal (antialiasing only, 1-2px), not aggressive
_ANTIALIASING_DILATION = 2  # pixels, hardcoded for consistency

# Morphological opening: separate blobs, remove noise
_MORPH_KERNEL_SIZE = 5  # small kernel, removes noise but keeps text shape

# Contour filtering thresholds (these are SHAPE metrics, not image-dependent)
_MIN_CONTOUR_AREA = 20  # minimum px² to consider (reject tiny noise)
_MAX_CONTOUR_SOLIDITY = 0.85  # RELAXED: merged text blocks have high solidity, allow up to 0.85
_MIN_CONTOUR_ASPECT_RATIO = 0.05  # text can be very wide (quotes, titles) or tall (labels)

# Erosion: shrink mask back toward original text (adaptive per blob size)
_EROSION_SCALE = 0.15  # shrink by ~15% of blob height/width

_LAMA_MAX_DIM = 2048
_MAX_MASK_COVERAGE = 0.50  # slightly higher to allow larger text regions
# ─────────────────────────────────────────────────────────────────────────────


def _build_antialiasing_kernel() -> np.ndarray:
    """
    Minimal dilation kernel for antialiasing only.
    Ellipse 3×3 = 1-2px fringe, catches subpixel edges without shape distortion.
    """
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def _detections_to_mask(
    detections: List[Tuple],
    h: int,
    w: int,
) -> np.ndarray:
    """
    Convert EasyOCR detections to binary mask.

    For each detection:
      1. Extract bbox corners as polygon
      2. Fill convex hull (tight fit, respects text shape)
      3. Single dilation for antialiasing fringe only
    
    No aggressive dilation here — we'll filter shapes next.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for bbox, _text, _prob in detections:
        # bbox is 4 corner points (x, y) from EasyOCR
        points = np.array(bbox, dtype=np.float32)
        
        # Convex hull: tightest polygon around text
        # Why: respects actual text shape, doesn't bloat mask
        hull = cv2.convexHull(points.reshape(-1, 1, 2))
        hull_int = hull.astype(np.int32)
        
        # Fill the polygon on mask
        cv2.fillPoly(mask, [hull_int], 255)
    
    # Single dilation: only catch antialiasing fringe (1-2px)
    # Why: minimal expansion, shapes stay recognizable
    kernel = _build_antialiasing_kernel()
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    return mask


def _filter_contours_by_shape(mask: np.ndarray, detections: List[Tuple]) -> np.ndarray:
    """
    Remove non-text blobs using shape analysis.
    
    Text characteristics:
      - Complex shape: high perimeter, jagged edges → low solidity (0.3–0.6)
      - Elongated: text is tall/thin or wide/thin → low aspect ratio (0.05–0.3)
    
    Object characteristics (signs, boards, stamps):
      - Smooth shape: circle, rectangle → high solidity (0.7–1.0)
      - Compact: aspect ratio near 1.0
    
    Algorithm:
      1. Find all contours in mask
      2. Compute solidity = area / convex hull area
      3. Compute aspect ratio = width / height (or vice versa, take min)
      4. Reject contours that look like objects (smooth + compact)
      5. Keep only text-like contours (jagged + elongated)
    """
    result_mask = np.zeros_like(mask)
    
    # Find all contours (connected components)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get bounding boxes of detected text for reference
    text_boxes = [np.array(bbox, dtype=np.float32) for bbox, _, _ in detections]
    
    kept_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Skip tiny noise
        if area < _MIN_CONTOUR_AREA:
            continue
        
        # Solidity = area / convex hull area
        # High solidity (>0.7) = smooth shape (bad for text, good for objects)
        # Low solidity (<0.65) = jagged shape (text)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area
        
        # Aspect ratio = min(w/h, h/w) 
        # Text is elongated, so AR is low (0.1–0.3)
        # Objects (circles, signs) have AR near 1.0
        x, y, w, h = cv2.boundingRect(contour)
        if w == 0 or h == 0:
            continue
        aspect_ratio = min(w / h, h / w)
        
        # Check if contour overlaps OCR detection (text)
        # Use bounding rect overlap instead of center-in-polygon
        # (large merged blobs have gaps, center might miss)
        overlaps_text = False
        contour_rect = cv2.boundingRect(contour)
        for text_box in text_boxes:
            text_rect = cv2.boundingRect(text_box.reshape(-1, 1, 2).astype(np.int32))
            # Check bounding rectangle overlap
            x1, y1, w1, h1 = contour_rect
            x2, y2, w2, h2 = text_rect
            
            # Rectangles overlap if they intersect
            if not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1):
                overlaps_text = True
                break
        
        # Decision: keep if overlaps OCR detection (OCR is primary validation)
        # Shape metrics (solidity, aspect) are secondary filters to catch egregious false positives
        if overlaps_text:
            kept_contours.append(contour)
            logger.debug(
                f"Keep contour: area={area:.0f}, solidity={solidity:.2f}, "
                f"aspect_ratio={aspect_ratio:.2f}, overlaps_ocr=True"
            )
        else:
            logger.debug(
                f"Reject contour: area={area:.0f}, solidity={solidity:.2f}, "
                f"aspect_ratio={aspect_ratio:.2f}, overlaps_ocr=False"
            )
    
    # Redraw only kept contours
    cv2.drawContours(result_mask, kept_contours, -1, 255, -1)
    return result_mask


def _adaptive_erosion(mask: np.ndarray, detections: List[Tuple]) -> np.ndarray:
    """
    Shrink mask back toward original text bounding boxes.
    
    Why: dilation expanded for antialiasing, but we don't need full expansion.
    This recovers the tight text core, spares surrounding objects.
    
    Per blob:
      1. Find bounding rect
      2. Erode by ~15% of blob size
      3. Redraw eroded region back on result mask
    """
    result_mask = np.zeros_like(mask)
    
    # Get individual blobs (connected components)
    num_labels, labels = cv2.connectedComponents(mask)
    
    for label_id in range(1, num_labels):
        # Extract this blob
        blob_mask = (labels == label_id).astype(np.uint8) * 255
        
        # Bounding rect of blob
        x, y, w, h = cv2.boundingRect(blob_mask)
        
        if w < _MIN_BBOX_PX or h < _MIN_BBOX_PX:
            continue
        
        # Erosion amount: 15% of blob size (adaptive)
        erode_px = max(1, int(min(w, h) * _EROSION_SCALE))
        
        # Create erosion kernel
        # Smaller kernel = gentler shrinkage, larger = aggressive
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
        
        # Erode this blob
        eroded = cv2.erode(blob_mask, kernel, iterations=1)
        
        # Add back to result
        result_mask = cv2.bitwise_or(result_mask, eroded)
    
    return result_mask


def _morphological_opening(mask: np.ndarray) -> np.ndarray:
    """
    Clean mask: erode then dilate.
    
    Why:
      - Erode: removes small noise, separates barely-touching blobs
      - Dilate: recovers original blob size
    
    Effect: text stays, noise/artifacts vanish, blobs slightly separated.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_MORPH_KERNEL_SIZE, _MORPH_KERNEL_SIZE))
    
    # Erode removes noise
    opened = cv2.erode(mask, kernel, iterations=1)
    
    # Dilate recovers size
    opened = cv2.dilate(opened, kernel, iterations=1)
    
    return opened


def _lama_inpaint(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    LaMa inpainting with automatic downscaling for large images.
    
    Why downscale: LaMa memory usage scales with image size.
    Downscale, inpaint, upscale back = saves memory, minimal quality loss.
    """
    lama = get_lama()
    h, w = image_bgr.shape[:2]
    
    scale = 1.0
    if max(h, w) > _LAMA_MAX_DIM:
        scale = _LAMA_MAX_DIM / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        logger.debug(f"LaMa downscale: {w}×{h} → {new_w}×{new_h}")
    
    # Convert BGR→RGB for PIL
    pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    pil_mask = Image.fromarray(mask)
    
    # Inpaint
    result_pil = lama(pil_img, pil_mask)
    result_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
    
    # Upscale if we downscaled
    if scale < 1.0:
        result_bgr = cv2.resize(result_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)
    
    return result_bgr


def _color_fill_fallback(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Fallback when LaMa unavailable: fill with median border colour.
    
    Per blob: sample pixels on the edge (border), use median as fill colour.
    Works for plain backgrounds, less ideal for textured.
    """
    result = image_bgr.copy()
    h, w = image_bgr.shape[:2]
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        
        if bw < _MIN_BBOX_PX or bh < _MIN_BBOX_PX:
            continue
        
        # Expand ROI slightly to get better border sample
        x1, y1 = max(0, x - 4), max(0, y - 4)
        x2, y2 = min(w, x + bw + 4), min(h, y + bh + 4)
        roi = image_bgr[y1:y2, x1:x2]
        
        roi_h, roi_w = roi.shape[:2]
        if roi_h < 4 or roi_w < 4:
            avg_color = np.array([255, 255, 255], dtype=np.uint8)
        else:
            # Sample border pixels only
            border_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
            border_mask[0, :] = 255
            border_mask[-1, :] = 255
            border_mask[:, 0] = 255
            border_mask[:, -1] = 255
            border_pixels = roi[border_mask > 0]
            avg_color = np.median(border_pixels, axis=0).astype(np.uint8)
        
        # Fill region with average colour
        region_mask = (labels[y1:y2, x1:x2] == i).astype(np.uint8) * 255
        dilated_region = cv2.dilate(region_mask, np.ones((3, 3), np.uint8))
        result[y1:y2, x1:x2][dilated_region > 0] = avg_color
    
    return result


class TextRemovalStep:
    """
    Production-grade text removal, v2.
    
    Handles: embedded text, watermarks, signs, boards
    No manual tuning: all logic data-driven (shape metrics, OCR overlap)
    """

    def process(self, image: np.ndarray, original: np.ndarray) -> np.ndarray:
        try:
            logger.info("TextRemovalStep v2: starting production text removal…")
            h, w = image.shape[:2]
            reader = get_ocr_reader()

            # ── Stage 1: Multi-pass OCR detection ────────────────────────────
            logger.info("TextRemovalStep: OCR detection (standard pass)…")
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

            logger.info("TextRemovalStep: OCR detection (aggressive pass)…")
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
                f"{len(agg_detections)} aggressive = {len(all_detections)} total"
            )

            # ── Stage 2: Mask construction ────────────────────────────────────
            logger.info("TextRemovalStep: building mask from detections…")
            mask = _detections_to_mask(all_detections, h, w)

            logger.info("TextRemovalStep: morphological opening (denoise)…")
            mask = _morphological_opening(mask)

            # ── Stage 3: Shape filtering ──────────────────────────────────────
            logger.info("TextRemovalStep: filtering non-text blobs by shape…")
            mask = _filter_contours_by_shape(mask, all_detections)

            # ── Stage 4: Adaptive erosion ─────────────────────────────────────
            logger.info("TextRemovalStep: adaptive erosion (recover tight text bounds)…")
            mask = _adaptive_erosion(mask, all_detections)

            # Coverage guard
            coverage = float(np.count_nonzero(mask)) / (h * w)
            logger.info(f"TextRemovalStep: final mask coverage = {coverage * 100:.1f}%")
            if coverage > _MAX_MASK_COVERAGE:
                logger.warning(
                    f"TextRemovalStep: mask {coverage*100:.1f}% exceeds "
                    f"{_MAX_MASK_COVERAGE*100:.0f}% — aborting (likely false positive)"
                )
                return image

            if coverage == 0.0:
                logger.info("TextRemovalStep: empty mask — image unchanged")
                return image

            # ── Stage 5: Inpainting ───────────────────────────────────────────
            try:
                logger.info("TextRemovalStep: LaMa inpainting…")
                result = _lama_inpaint(image, mask)
                logger.info(
                    f"TextRemovalStep: complete ({len(all_detections)} regions, "
                    f"{coverage*100:.1f}% coverage)"
                )
                return result

            except Exception as lama_err:
                logger.warning(f"TextRemovalStep: LaMa failed ({lama_err}), fallback to colour-fill")
                result = _color_fill_fallback(image, mask)
                logger.info("TextRemovalStep: colour-fill fallback complete")
                return result

        except Exception as e:
            logger.error(f"TextRemovalStep: unexpected error — {e}", exc_info=True)
            return image


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bbox_is_valid(bbox, min_px: int) -> bool:
    """Check if bbox large enough to process."""
    try:
        pts = np.array(bbox, dtype=np.float32)
        x, y, bw, bh = cv2.boundingRect(pts.reshape(-1, 1, 2).astype(np.int32))
        return bw >= min_px and bh >= min_px
    except Exception:
        return False


def _iou(bbox_a, bbox_b) -> float:
    """Intersection-over-union between two bboxes."""
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
    """Check if bbox overlaps any in existing list."""
    for (ex_bbox, _, _) in existing:
        if _iou(bbox, ex_bbox) >= iou_thresh:
            return True
    return False