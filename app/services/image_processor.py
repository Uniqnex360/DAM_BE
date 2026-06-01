import cv2
import numpy as np
from PIL import Image
import io
import time
import logging
from typing import Tuple, Dict, List
from transparent_background import Remover
logger = logging.getLogger(__name__)
TARGET_SIZE = (2000, 2000)
CONFIDENCE_THRESHOLD = 0.6
_remover = None


def get_remover():
    global _remover
    if _remover is None:
        logger.info("Initializing background remover...")
        _remover = Remover(mode="fast")
        logger.info('Background remover ready!')
    return _remover


class ImageProcessor:
    def __init__(self, file_bytes: bytes, resize_dims: dict = None, operations: list = None, autoDetect: bool = False):
        nparr = np.frombuffer(file_bytes, np.uint8)
        self.img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if self.img is None:
            raise ValueError("Could not decode image bytes")
        self.original_h, self.original_w = self.img.shape[:2]
        self.resize_dims = resize_dims
        self.operations = operations or []
        self.original_img = self.img.copy()
        self.auto_detect = autoDetect
        logger.info(f"ImageProcessor initialized: dims={self.original_w}x{self.original_h}, "
                    f"resize_dims={self.resize_dims}, operations={self.operations}")

    def resize_ecom(self):
        h, w = self.img.shape[:2]
        print(
            f"resize_ecom called: current size={w}x{h}, resize_dims={self.resize_dims}")
        if not self.resize_dims:
            print(" resize_ecom: No resize_dims provided, returning")
            return
        if isinstance(self.resize_dims, list):
            results = []
            for config in self.resize_dims:
                result = self._apply_single_resize(self.original_img, config)
                results.append({
                    "id": config.get("id"),
                    "width": config.get("width"),
                    "height": config.get("height"),
                    "image_bytes": result
                })
            return results
        else:
            return self._apply_single_resize(self.img, self.resize_dims)

    def check_watermark_compliance(self, marketplace_id: str, confidence: float):
        policies = {
            "amazon-us": "reject", "amazon-uk": "reject",
            "walmart": "reject", "wayfair-us": "review", "wayfair-uk": "review",
            "ebay-us": "allowed", "ebay-uk": "allowed",
            "target-plus": "reject", "tiktok-shop": "reject", "homedepot": "reject"
        }
        policy = policies.get(marketplace_id, "reject")
        if policy == "reject" and confidence >= CONFIDENCE_THRESHOLD:
            return {"compliant": False, "action": "block", "message": "Watermarks not allowed"}
        elif policy == "review" and confidence >= CONFIDENCE_THRESHOLD:
            return {"compliant": False, "action": "flag", "message": "Manual review required"}
        elif policy == "allowed":
            return {"compliant": True, "action": "allow", "message": "Watermark allowed"}
        return {"compliant": True, "action": "allow", "message": "No watermark detected"}

    def _apply_single_resize(self, img, resize_config):
        target_w = resize_config.get("width")
        target_h = resize_config.get("height")
        if not target_w or not target_h:
            return None
        h, w = img.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pil = pil.resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
        offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
        canvas.paste(pil, offset)
        return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)

    def _foreground_mask(self, img) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        _, mask = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    def analyze(self) -> Dict[str, float]:
        h, w = self.img.shape[:2]
        print(f" analyze: Image size={w}x{h}, resize_dims={self.resize_dims}")
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        fg = self._foreground_mask(self.img)
        fg_ratio = np.sum(fg > 0) / (h * w)
        conf = {
            "bg_clean": 0.0, "shadow": 0.0, "crop": 0.0,
            "watermark": 0.0, "resize": 0.0
        }
        if self.resize_dims:
            needs_resize = False
            if isinstance(self.resize_dims, list):
                for dim in self.resize_dims:
                    target_w = dim.get("width")
                    target_h = dim.get("height")
                    if target_w and target_h and (w != target_w or h != target_h):
                        needs_resize = True
                        break
            else:
                target_w = self.resize_dims.get("width")
                target_h = self.resize_dims.get("height")
                if target_w and target_h and (w != target_w or h != target_h):
                    needs_resize = True
            conf["resize"] = 1.0 if needs_resize else 0.0
        else:
            conf["resize"] = 0.0
        if fg_ratio < 0.35:
            conf["crop"] = min(1.0, (0.5 - fg_ratio) * 3)
        corner_std = np.mean([
            np.std(gray[:80, :80]), np.std(gray[:80, -80:]),
            np.std(gray[-80:, :80]), np.std(gray[-80:, -80:])
        ])
        conf["bg_clean"] = np.clip((corner_std - 10) / 20, 0, 1)
        v = hsv[:, :, 2]
        mean_v = np.mean(v) if np.mean(v) > 0 else 1
        shadow_mask = (v < 0.35 * mean_v) & (fg > 0)
        shadow_ratio = np.sum(shadow_mask) / (h * w)
        conf["shadow"] = np.clip(shadow_ratio * 40, 0, 1)
        _, th = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(
            th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, cw, ch = cv2.boundingRect(c)
            if cw > w * 0.25 and ch < h * 0.12:
                conf["watermark"] = 0.85
                break
        print(f"analyze: Confidence scores = {conf}")
        return conf

    def retouch_image(self, mode: str = "auto"):
        """
        Professional retouch with mode detection
        mode: "auto" | "product" | "portrait"
        """
        try:
            height, width = self.img.shape[:2]
            logger.info(f"Retouch starting: {width}x{height}, mode={mode}")

            # Auto-detect mode based on image content
            if mode == "auto":
                mode = self._detect_image_type()
                logger.info(f"Auto-detected mode: {mode}")

            if mode == "product":
                return self._retouch_product()
            else:
                return self._retouch_portrait()

        except Exception as e:
            logger.error(f"Retouch failed: {e}")
            import traceback
            traceback.print_exc()
            return False


    def _detect_image_type(self) -> str:
        """Detect if image is product or portrait"""
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        
        # Check for skin tones
        lower_skin = np.array([0, 30, 60], dtype=np.uint8)
        upper_skin = np.array([20, 150, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_ratio = np.sum(skin_mask > 0) / skin_mask.size

        # Check background uniformity (products usually have clean BG)
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        corners = [
            gray[:50, :50], gray[:50, -50:],
            gray[-50:, :50], gray[-50:, -50:]
        ]
        corner_std = np.mean([np.std(c) for c in corners])

        # Heuristic: low skin + clean background = product
        if skin_ratio < 0.05 and corner_std < 20:
            return "product"
        return "portrait"


    def _retouch_product(self):
        """E-commerce product retouch - clean, crisp, natural"""
        logger.info("Applying PRODUCT retouch")

        # ────────────────────────────────────────
        # Stage 1: GENTLE DENOISE (preserve detail)
        # ────────────────────────────────────────
        # Light bilateral filter - removes noise without softening edges
        self.img = cv2.bilateralFilter(self.img, d=5, sigmaColor=35, sigmaSpace=35)
        logger.info("Gentle denoise complete")

        # ────────────────────────────────────────
        # Stage 2: WHITE BALANCE CORRECTION
        # ────────────────────────────────────────
        # Gray-world white balance algorithm
        result = self.img.copy().astype(np.float32)
        avg_b = np.mean(result[:, :, 0])
        avg_g = np.mean(result[:, :, 1])
        avg_r = np.mean(result[:, :, 2])
        avg_gray = (avg_b + avg_g + avg_r) / 3

        result[:, :, 0] = np.clip(result[:, :, 0] * (avg_gray / avg_b), 0, 255)
        result[:, :, 1] = np.clip(result[:, :, 1] * (avg_gray / avg_g), 0, 255)
        result[:, :, 2] = np.clip(result[:, :, 2] * (avg_gray / avg_r), 0, 255)
        self.img = result.astype(np.uint8)
        logger.info("White balance complete")

        # ────────────────────────────────────────
        # Stage 3: SUBTLE CONTRAST (LAB-based)
        # ────────────────────────────────────────
        lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # GENTLE CLAHE only on luminance
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # DON'T boost a/b channels - keeps colors natural
        self.img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        logger.info("Contrast enhancement complete")

        # ────────────────────────────────────────
        # Stage 4: PRECISION SHARPENING
        # ────────────────────────────────────────
        # Unsharp mask - the professional way
        gaussian = cv2.GaussianBlur(self.img, (0, 0), 1.5)
        self.img = cv2.addWeighted(self.img, 1.4, gaussian, -0.4, 0)
        self.img = np.clip(self.img, 0, 255).astype(np.uint8)
        logger.info("Precision sharpening complete")

        # ────────────────────────────────────────
        # Stage 5: SUBTLE SATURATION BOOST
        # ────────────────────────────────────────
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.1, 0, 255)  # Just 10% boost
        self.img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        logger.info("Color saturation boost complete")

        # ────────────────────────────────────────
        # Stage 6: GENTLE BRIGHTNESS LIFT
        # ────────────────────────────────────────
        # Very subtle - products shouldn't look washed out
        self.img = cv2.convertScaleAbs(self.img, alpha=1.03, beta=3)

        logger.info("✅ Product retouch complete")
        return True


    def _retouch_portrait(self):
        """Portrait/people retouch - skin smoothing, glow"""
        logger.info("Applying PORTRAIT retouch")

        # Denoise
        self.img = cv2.fastNlMeansDenoisingColored(self.img, None, 7, 7, 7, 21)

        # LAB enhancement
        lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        self.img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        # Skin smoothing
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 15, 50], dtype=np.uint8)
        upper_skin = np.array([25, 170, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_mask = cv2.GaussianBlur(skin_mask, (15, 15), 0)
        skin_mask_3ch = cv2.merge([skin_mask] * 3).astype(np.float32) / 255.0

        smoothed = cv2.bilateralFilter(self.img, 15, 80, 80)
        self.img = (
            smoothed.astype(np.float32) * skin_mask_3ch +
            self.img.astype(np.float32) * (1 - skin_mask_3ch)
        ).astype(np.uint8)

        # Sharpen non-skin areas
        gaussian = cv2.GaussianBlur(self.img, (0, 0), 1.5)
        sharpened = cv2.addWeighted(self.img, 1.3, gaussian, -0.3, 0)
        self.img = (
            self.img.astype(np.float32) * skin_mask_3ch +
            sharpened.astype(np.float32) * (1 - skin_mask_3ch)
        ).astype(np.uint8)

        # Final polish
        self.img = cv2.convertScaleAbs(self.img, alpha=1.05, beta=5)

        logger.info("✅ Portrait retouch complete")
        return True
    def clean_background(self):
        try:
            img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            remover = get_remover()
            out = remover.process(pil_img)
            bg = Image.new("RGB", out.size, (255, 255, 255))
            if out.mode == 'RGBA':
                bg.paste(out, mask=out.split()[3])
            else:
                bg.paste(out)
            self.img = cv2.cvtColor(np.array(bg), cv2.COLOR_RGB2BGR)
            logger.info('Background removal successful')
        except Exception as e:
            logger.error(f"BG removal failed: {e}")
            raise e

    def remove_shadow(self):
        """Hybrid approach - smart fallback"""
        try:
            img_rgb = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            remover = get_remover()
            cutout = remover.process(pil_img)
            if cutout.mode != 'RGBA':
                self._traditional_shadow_removal()
                return
            alpha = np.array(cutout.split()[3])
            edge_quality = self._check_alpha_quality(alpha)
            if edge_quality < 0.7:
                logger.warning(
                    "AI cutout quality low, using traditional method")
                self._traditional_shadow_removal()
                return
            rgba_array = np.array(cutout)
            alpha_float = alpha.astype(np.float32) / 255.0
            alpha_3ch = np.stack(
                [alpha_float, alpha_float, alpha_float], axis=2)
            rgb = rgba_array[:, :, :3].astype(np.float32)
            white = np.ones_like(rgb) * 255.0
            result = (alpha_3ch * rgb) + ((1 - alpha_3ch) * white)
            self.img = np.clip(result, 0, 255).astype(np.uint8)
            original_bgr = self.original_img
            result_bgr = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
            product_mask = alpha_float > 0.3
            if np.any(product_mask):
                orig_mean = cv2.mean(
                    original_bgr, mask=product_mask.astype(np.uint8))[:3]
                result_mean = cv2.mean(
                    result_bgr, mask=product_mask.astype(np.uint8))[:3]
                shift = np.array(orig_mean) - np.array(result_mean)
                corrected = result_bgr.copy()
                for c in range(3):
                    corrected[:, :, c] = np.where(
                        product_mask,
                        np.clip(result_bgr[:, :, c] + shift[c], 0, 255),
                        result_bgr[:, :, c]
                    )
                self.img = corrected
            else:
                self.img = result_bgr
            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            near_white = gray > 250
            was_background = alpha_float < 0.05
            safe_to_clean = near_white & was_background
            self.img[safe_to_clean] = [255, 255, 255]
            logger.info("AI shadow removal complete")
        except Exception as e:
            logger.error(
                f"AI shadow removal failed: {e}, falling back to traditional")
            self._traditional_shadow_removal()

    def _check_alpha_quality(self, alpha: np.ndarray) -> float:
        """Score alpha mask quality (0-1)"""
        edges = cv2.Canny(alpha, 50, 150)
        edge_ratio = np.sum(edges > 0) / alpha.size
        foreground_ratio = np.sum(alpha > 128) / alpha.size
        if edge_ratio > 0.15:
            return 0.3
        if foreground_ratio < 0.05 or foreground_ratio > 0.95:
            return 0.4
        return 0.9

    def _traditional_shadow_removal(self):
        """Non-AI shadow fix for when AI fails"""
        lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        self.img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        bg_mask = (s < 20) & (v > 230)
        self.img[bg_mask] = [255, 255, 255]
        logger.info("Traditional shadow removal complete")

    def remove_watermark(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        self.img = cv2.inpaint(self.img, mask, 5, cv2.INPAINT_TELEA)

    def smart_crop(self):
        h, w = self.img.shape[:2]
        fg = self._foreground_mask(self.img)
        cnts, _ = cv2.findContours(
            fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return
        image_area = h * w
        boxes = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < image_area * 0.01:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            boxes.append((x, y, cw, ch))
        if not boxes:
            return
        x_min = min(b[0] for b in boxes)
        y_min = min(b[1] for b in boxes)
        x_max = max(b[0] + b[2] for b in boxes)
        y_max = max(b[1] + b[3] for b in boxes)
        crop_w = x_max - x_min
        crop_h = y_max - y_min
        crop_area_ratio = (crop_w * crop_h) / image_area
        if crop_area_ratio < 0.4 or crop_area_ratio > 0.95:
            return
        pad = int(0.08 * max(crop_w, crop_h))
        x1, y1 = max(0, x_min - pad), max(0, y_min - pad)
        x2, y2 = min(w, x_max + pad), min(h, y_max + pad)
        self.img = self.img[y1:y2, x1:x2]

    def process(self) -> Dict:
        start_time = time.time()
        self.resize_results = None
        steps_applied = []
        print(
            f"process() called: auto_detect={self.auto_detect}, operations={self.operations}")
        confidence = self.analyze()
        if self.auto_detect:
            logger.info("Auto-detect mode activated")
            if confidence["shadow"] >= CONFIDENCE_THRESHOLD:
                self.remove_shadow()
                steps_applied.append("shadow_fix")
            if confidence["watermark"] >= CONFIDENCE_THRESHOLD:
                self.remove_watermark()
                steps_applied.append("watermark_removal")
            if confidence["crop"] >= CONFIDENCE_THRESHOLD:
                self.smart_crop()
                steps_applied.append("smart_crop")
            if confidence["bg_clean"] >= CONFIDENCE_THRESHOLD:
                self.clean_background()
                steps_applied.append("bg_removal")
            if confidence["resize"] >= CONFIDENCE_THRESHOLD:
                self.resize_ecom()
                steps_applied.append("resize")
        else:
            logger.info(f"User-selection mode: {self.operations}")
            if "retouch" in self.operations:
                self.retouch_image()
                steps_applied.append("retouch")
            if any(op in self.operations for op in ["shadow-remove", "shadow_fix", "shadow"]):
                self.remove_shadow()
                steps_applied.append("shadow_fix")
            if "watermark_removal" in self.operations or "watermark" in self.operations:
                self.remove_watermark()
                steps_applied.append("watermark_removal")
            if "smart_crop" in self.operations or "crop" in self.operations:
                self.smart_crop()
                steps_applied.append("smart_crop")
            if "bg-remove" in self.operations or "bg_removal" in self.operations:
                self.clean_background()
                steps_applied.append("bg_removal")
            if self.resize_dims:
                result = self.resize_ecom()
                if isinstance(result, list):
                    self.resize_results = result
                    steps_applied.append("resize_multiple")
                elif result is not None:
                    self.img = result
                    steps_applied.append("resize")
            else:
                if "resize" in self.operations:
                    logger.warning(
                        "Resize requested but no dimensions provided")
        print(f"process() complete: steps_applied={steps_applied}")
        end_time = time.time()
        duration_ms = int((end_time - start_time) * 1000)
        success, encoded_img = cv2.imencode(
            ".jpg", self.img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return {
            "image_bytes": encoded_img.tobytes(),
            "confidence": confidence,
            "steps_applied": steps_applied,
            "duration_ms": duration_ms,
            "resize_results": self.resize_results
        }
