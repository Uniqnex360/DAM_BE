
# import cv2
# import numpy as np
# from PIL import Image
# import io
# import logging
# from typing import Optional
# from app.services.image_processing.model_registry import get_rembg_session
# from rembg import remove



# logger = logging.getLogger(__name__)


# class ShadowRemovalStep:
    

#     def __init__(self, use_neural: bool = True, background_color: str = "#FFFFFF"):
        
#         self.use_neural = use_neural
#         self.background_color = background_color

#     def process(self, image: np.ndarray, original: np.ndarray = None) -> np.ndarray:
        
#         if self.use_neural:
#             try:
#                 logger.warning("Attempting neural shadow removal (ISNet)...")
#                 result = self._neural_remove(image)
#                 logger.info("Neural shadow removal succeeded ")
#                 return result
#             except Exception as e:
#                 logger.error(f"Neural model failed: {e}")
#                 logger.info("Falling back to classical shadow removal")

#         return self._classical_remove(image)

#     def _neural_remove(self, image: np.ndarray) -> np.ndarray:
       
#         session = get_rembg_session()

#         max_dim = 2000
#         h, w = image.shape[:2]
#         working_img = image.copy()

#         if max(h, w) > max_dim:
#             scale = max_dim / max(h, w)
#             working_img = cv2.resize(
#                 working_img,
#                 (int(w * scale), int(h * scale)),
#                 interpolation=cv2.INTER_AREA
#             )

#         success, buffer = cv2.imencode(".png", working_img)
#         if not success:
#             raise ValueError("Failed to encode image for rembg")

#         input_bytes = buffer.tobytes()

#         output_bytes = remove(input_bytes, session=session)

#         pil_img = Image.open(io.BytesIO(output_bytes)).convert("RGBA")

#         if self.background_color == "transparent":
#             return np.array(pil_img)


#         else:
#             rgb = tuple(int(self.background_color.lstrip('#')[i:i+2], 16)
#                         for i in (0, 2, 4))
#             bg = Image.new("RGB", pil_img.size, rgb)  

#         bg.paste(pil_img, mask=pil_img.split()[3])

#         result = cv2.cvtColor(np.array(bg), cv2.COLOR_RGB2BGR)

#         if max(h, w) > max_dim:
#             result = cv2.resize(result, (w, h), interpolation=cv2.INTER_CUBIC)

#         return result

#     def _classical_remove(self, image: np.ndarray) -> np.ndarray:
        
#         logger.info("Using classical CLAHE shadow removal")

#         lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#         l, a, b = cv2.split(lab)

#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         l = clahe.apply(l)

#         result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
#         return result
import cv2
import numpy as np
from PIL import Image
import io
import logging
import os
import time
from typing import Optional, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path

# External Dependencies
from skimage.segmentation import slic as sk_slic
from app.services.image_processing.model_registry import get_rembg_session
from rembg import remove

logger = logging.getLogger(__name__)

# ─── CONFIGURATION ───


@dataclass
class ShadowConfig:
    """Professional shadow removal settings."""
    # Adaptive thresholds
    delta_l_min_k: float = 1.2
    delta_l_min_floor_frac: float = 0.015
    delta_l_max_k: float = 0.50
    min_shadow_window: float = 40.0
    chroma_k: float = 3.5
    chroma_offset_frac: float = 0.06
    colored_bg_threshold: float = 8.0

    # Surface fit
    poly_degree: int = 2
    fit_subsample: int = 3

    # Corrective caps
    max_l_correction_frac: float = 0.35
    max_ab_correction_frac: float = 0.08
    mask_expand_frac: float = 0.04

    # Structural
    border_px: int = 25
    border_exclude_corners: int = 50
    n_superpixels: int = 400
    sp_compactness: float = 8.0
    morph_close_px: int = 11
    morph_open_px: int = 3
    conf_threshold: float = 0.12
    feather_px: int = 25

    # Backend choice: "surface" (recommended), "lama", or "classical"
    backend: str = "surface"
    max_side: int = 2000


@dataclass
class _Thresholds:
    bg_lab: np.ndarray
    bg_l_std: float
    bg_chroma_mag: float
    bg_chroma_std: float
    colored_bg: bool
    delta_l_min: float
    delta_l_max: float
    chroma_thresh: float
    product_delta_l: float
    product_chroma: float

# ─── CORE MATHEMATICAL FUNCTIONS ───


def _calibrate(img_bgr: np.ndarray, cfg: ShadowConfig) -> _Thresholds:
    h, w = img_bgr.shape[:2]
    bp = max(1, min(cfg.border_px, h // 8, w // 8))
    ec = cfg.border_exclude_corners

    top = img_bgr[:bp, ec:w-ec].reshape(-1, 3)
    bottom = img_bgr[-bp:, ec:w-ec].reshape(-1, 3)
    left = img_bgr[ec:h-ec, :bp].reshape(-1, 3)
    right = img_bgr[ec:h-ec, -bp:].reshape(-1, 3)
    strips = np.concatenate([top, bottom, left, right])

    gray = strips.mean(axis=1)
    thresh_10 = np.percentile(gray, 10)
    strips = strips[gray > thresh_10]
    if len(strips) < 10:
        strips = np.concatenate(
            [img_bgr[:bp, :].reshape(-1, 3), img_bgr[-bp:, :].reshape(-1, 3)])

    slab = cv2.cvtColor(strips.reshape(-1, 1, 3).astype(np.uint8),
                        cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    bg = np.median(slab, axis=0)
    bg_l_std = float(np.std(slab[:, 0]))
    chroma_vals = np.sqrt((slab[:, 1] - 128)**2 + (slab[:, 2] - 128)**2)
    bg_chroma_mag, bg_chroma_std = float(
        np.median(chroma_vals)), float(np.std(chroma_vals))

    delta_l_min = max(bg_l_std * cfg.delta_l_min_k,
                      bg[0] * cfg.delta_l_min_floor_frac)
    delta_l_max = float(
        np.clip(bg[0] * cfg.delta_l_max_k, delta_l_min + cfg.min_shadow_window, 100.0))
    chroma_thresh = (bg_chroma_std * cfg.chroma_k +
                     bg[0] * cfg.chroma_offset_frac + bg_chroma_mag * 0.3)

    return _Thresholds(
        bg_lab=bg, bg_l_std=bg_l_std, bg_chroma_mag=bg_chroma_mag, bg_chroma_std=bg_chroma_std,
        colored_bg=bg_chroma_mag > cfg.colored_bg_threshold,
        delta_l_min=delta_l_min, delta_l_max=delta_l_max, chroma_thresh=chroma_thresh,
        product_delta_l=delta_l_max + 5.0, product_chroma=chroma_thresh * 2.5
    )


def _fit_bg_surface(lab: np.ndarray, bg_mask: np.ndarray, cfg: ShadowConfig) -> np.ndarray:
    h, w = lab.shape[:2]
    ys = np.linspace(-1, 1, h, dtype=np.float32)
    xs = np.linspace(-1, 1, w, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    bg_pixels = bg_mask > 0

    if bg_pixels.sum() < 20:
        surface = np.empty((h, w, 3), np.float32)
        surface[:] = np.median(lab, axis=(0, 1))
        return surface

    # Polynomial expansion [1, x, y, x2, xy, y2...]
    def get_features(y, x):
        cols = []
        for d in range(cfg.poly_degree + 1):
            for xp in range(d + 1):
                cols.append((x**xp) * (y**(d - xp)))
        return np.column_stack(cols)

    A_fit = get_features(yy[bg_pixels][::cfg.fit_subsample],
                         xx[bg_pixels][::cfg.fit_subsample])
    lab_fit = lab[bg_pixels][::cfg.fit_subsample]
    A_full = get_features(yy.ravel(), xx.ravel())

    surface = np.empty((h * w, 3), np.float32)
    for ch in range(3):
        coeffs, _, _, _ = np.linalg.lstsq(A_fit, lab_fit[:, ch], rcond=None)
        surface[:, ch] = A_full @ coeffs
    return surface.reshape(h, w, 3)

# ─── MAIN STEP CLASS ───


class ShadowRemovalStep:
    # Update the __init__ to accept the old arguments
    def __init__(
        self,
        backend: str = "surface",
        max_side: int = 2000,
        background_color: str = "#FFFFFF",  # Add this back
        use_neural: bool = True            # Add this back
    ):
        # We store them to keep the Orchestrator happy
        self.background_color = background_color
        self.use_neural = use_neural

        # Initialize the professional config
        self.cfg = ShadowConfig(backend=backend, max_side=max_side)
        self.session = get_rembg_session()
    def process(self, image: np.ndarray, original: np.ndarray = None) -> np.ndarray:
        """
        Main pipeline: Detect Product -> Calibrate -> Detect Shadow -> Correct Surface.
        """
        try:
            h_orig, w_orig = image.shape[:2]

            # 1. Product Mask (Protection)
            logger.info("Detecting product mask (rembg)...")
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            p_mask = remove(img_rgb, session=self.session, only_mask=True)
            p_mask = (np.array(p_mask) > 128).astype(np.uint8) * 255

            # 2. Calibration
            t = _calibrate(image, self.cfg)

            # 3. Shadow Confidence Map
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
            dL = t.bg_lab[0] - lab[:, :, 0]
            da, db = lab[:, :, 1] - t.bg_lab[1], lab[:, :, 2] - t.bg_lab[2]
            chroma = np.sqrt(da**2 + db**2)  # Simplified for API

            in_shadow = (dL >= t.delta_l_min) & (
                dL < t.delta_l_max) & (chroma < t.chroma_thresh)
            conf = np.zeros_like(dL)
            if in_shadow.any():
                dL_n = np.clip((dL[in_shadow] - t.delta_l_min) /
                               (t.delta_l_max - t.delta_l_min + 1e-6), 0, 1)
                ch_n = 1.0 - \
                    np.clip(chroma[in_shadow] / (t.chroma_thresh + 1e-6), 0, 1)
                conf[in_shadow] = np.sqrt(dL_n * ch_n)

            # 4. Refine Shadow Mask
            s_mask = (conf > self.cfg.conf_threshold).astype(np.uint8) * 255
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.cfg.morph_close_px, self.cfg.morph_close_px))
            s_mask = cv2.morphologyEx(s_mask, cv2.MORPH_CLOSE, k)

            # Expansion
            exp = max(11, int(min(h_orig, w_orig) *
                      self.cfg.mask_expand_frac)) | 1
            s_mask = cv2.dilate(s_mask, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (exp, exp)))
            s_mask[p_mask > 0] = 0  # Protect product

            shadow_px = int((s_mask > 0).sum())
            if shadow_px < 50:
                logger.info(
                    "No significant shadow detected. Returning original.")
                return image

            # 5. Surface Correction
            logger.info(f"Correcting shadow surface ({shadow_px} px)...")
            confirmed_bg = ((s_mask == 0) & (p_mask == 0)
                            ).astype(np.uint8) * 255
            surface = _fit_bg_surface(lab, confirmed_bg, self.cfg)

            # Apply correction caps
            max_l = t.bg_lab[0] * self.cfg.max_l_correction_frac
            max_ab = t.bg_lab[0] * self.cfg.max_ab_correction_frac

            corr = np.empty_like(lab)
            corr[:, :, 0] = np.clip(
                surface[:, :, 0] - lab[:, :, 0], -max_l, max_l)
            corr[:, :, 1] = np.clip(
                surface[:, :, 1] - lab[:, :, 1], -max_ab, max_ab)
            corr[:, :, 2] = np.clip(
                surface[:, :, 2] - lab[:, :, 2], -max_ab, max_ab)

            # Blend
            feather = self.cfg.feather_px | 1
            alpha = cv2.GaussianBlur((s_mask > 0).astype(
                np.float32), (feather, feather), 0)[:, :, np.newaxis]

            corrected_lab = lab + corr * np.clip(alpha * 1.2, 0, 1)
            result = cv2.cvtColor(np.clip(corrected_lab, 0, 255).astype(
                np.uint8), cv2.COLOR_LAB2BGR)

            return result

        except Exception as e:
            logger.exception(f"Shadow removal failed: {e}")
            return self._classical_fallback(image)

    def _classical_fallback(self, image: np.ndarray) -> np.ndarray:
        """Original CLAHE logic as a safe fallback."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
