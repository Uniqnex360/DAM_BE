import cv2
import numpy as np
from PIL import Image
from rembg import remove
import io
import time
import logging
from typing import Tuple, Dict, List

logger = logging.getLogger(__name__)

TARGET_SIZE = (2000, 2000)

# Original was 0.6
CONFIDENCE_THRESHOLD = 0.6


class ImageProcessor:
    def __init__(self, file_bytes: bytes, resize_dims: dict = None, operations: list = None, autoDetect: bool = False):
        nparr = np.frombuffer(file_bytes, np.uint8)
        self.img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if self.img is None:
            raise ValueError("Could not decode image bytes")
        self.original_h, self.original_w = self.img.shape[:2]
        self.resize_dims = resize_dims
        self.operations = operations or []
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

        # Get target dimensions - don't use 'or' with dict.get()
        target_w = self.resize_dims.get("width")
        target_h = self.resize_dims.get("height")
        print(f"🔍 DEBUG: target_w={target_w}, target_h={target_h}")
        if not target_w or not target_h:
            print(
                f"resize_ecom: Invalid dimensions width={target_w}, height={target_h}")
            return

        print(f"resize_ecom: Resizing to {target_w}x{target_h}")

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        print(f"resize_ecom: Scaled size will be {new_w}x{new_h}")

        pil = Image.fromarray(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        pil = pil.resize((new_w, new_h), Image.LANCZOS)

        canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
        offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
        canvas.paste(pil, offset)

        self.img = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)

        final_h, final_w = self.img.shape[:2]
        print(f"resize_ecom: Final size={final_w}x{final_h}")

    def _foreground_mask(self, img) -> np.ndarray:

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        _, mask = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    def analyze(self) -> Dict[str, float]:
        h, w = self.img.shape[:2]
        print(f"✅ analyze: Image size={w}x{h}, resize_dims={self.resize_dims}")
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        fg = self._foreground_mask(self.img)
        fg_ratio = np.sum(fg > 0) / (h * w)

        # Initialize conf dictionary FIRST
        conf = {
            "bg_clean": 0.0, "shadow": 0.0, "crop": 0.0,
            "watermark": 0.0, "resize": 0.0
        }

        # 1. Resize Confidence - Only resize if custom dimensions are provided
        if self.resize_dims:
            target_w = self.resize_dims.get("width")
            target_h = self.resize_dims.get("height")
            # Only resize if dimensions are different from current
            if target_w and target_h and (w != target_w or h != target_h):
                conf["resize"] = 1.0
            else:
                conf["resize"] = 0.0  # Skip resize - already correct size
        else:
            # If no custom dimensions provided, don't resize - keep original
            conf["resize"] = 0.0

        # 2. Crop Confidence (If foreground is tiny)
        if fg_ratio < 0.35:
            conf["crop"] = min(1.0, (0.5 - fg_ratio) * 3)

        # 3. Background Cleanliness (High Score = DIRTY background)
        corner_std = np.mean([
            np.std(gray[:80, :80]), np.std(gray[:80, -80:]),
            np.std(gray[-80:, :80]), np.std(gray[-80:, -80:])
        ])
        conf["bg_clean"] = np.clip((corner_std - 10) / 20, 0, 1)

        # 4. Shadow Confidence (High Score = Has Shadows)
        v = hsv[:, :, 2]
        mean_v = np.mean(v) if np.mean(v) > 0 else 1
        shadow_mask = (v < 0.35 * mean_v) & (fg > 0)
        shadow_ratio = np.sum(shadow_mask) / (h * w)
        conf["shadow"] = np.clip(shadow_ratio * 40, 0, 1)

        # 5. Watermark Confidence
        _, th = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(
            th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, cw, ch = cv2.boundingRect(c)
            if cw > w * 0.25 and ch < h * 0.12:
                conf["watermark"] = 0.85
                break
        print(f"✅ analyze: Confidence scores = {conf}")
        return conf

    def clean_background(self):
        try:
            img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            out = remove(pil_img)

            bg = Image.new("RGB", out.size, (255, 255, 255))
            if out.mode == "RGBA":
                bg.paste(out, mask=out.split()[3])
            else:
                bg.paste(out)
            self.img = cv2.cvtColor(np.array(bg), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"BG Removal failed: {e}")

    def remove_shadow(self):
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        mean_v = np.mean(v)
        shadow_mask = v < (0.4 * mean_v)

        shadow_mask = cv2.morphologyEx(
            shadow_mask.astype(np.uint8) * 255,
            cv2.MORPH_DILATE,
            np.ones((5, 5), np.uint8)
        )
        # Brighten pixels
        self.img[shadow_mask > 0] = cv2.add(
            self.img[shadow_mask > 0], np.array([25, 25, 25]))

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

    # def analyze(self) -> Dict[str, float]:
    #     h, w = self.img.shape[:2]
    #     gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
    #     hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

    #     fg = self._foreground_mask(self.img)
    #     fg_ratio = np.sum(fg > 0) / (h * w)

    #     # Initialize conf dictionary FIRST
    #     conf = {
    #         "bg_clean": 0.0, "shadow": 0.0, "crop": 0.0,
    #         "watermark": 0.0, "resize": 0.0
    #     }

    #     # 1. Resize Confidence - Only resize if custom dimensions are provided
    #     if self.resize_dims:
    #         target_w = self.resize_dims.get("width")
    #         target_h = self.resize_dims.get("height")
    #         # Only resize if dimensions are different from current
    #         if target_w and target_h and (w != target_w or h != target_h):
    #             conf["resize"] = 1.0
    #         else:
    #             conf["resize"] = 0.0  # Skip resize - already correct size
    #     else:
    #         # If no custom dimensions provided, don't resize - keep original
    #         conf["resize"] = 0.0

    #     # 2. Crop Confidence (If foreground is tiny)
    #     if fg_ratio < 0.35:
    #         conf["crop"] = min(1.0, (0.5 - fg_ratio) * 3)

    #     # 3. Background Cleanliness (High Score = DIRTY background)
    #     corner_std = np.mean([
    #         np.std(gray[:80, :80]), np.std(gray[:80, -80:]),
    #         np.std(gray[-80:, :80]), np.std(gray[-80:, -80:])
    #     ])
    #     conf["bg_clean"] = np.clip((corner_std - 10) / 20, 0, 1)

    #     # 4. Shadow Confidence (High Score = Has Shadows)
    #     v = hsv[:, :, 2]
    #     mean_v = np.mean(v) if np.mean(v) > 0 else 1
    #     shadow_mask = (v < 0.35 * mean_v) & (fg > 0)
    #     shadow_ratio = np.sum(shadow_mask) / (h * w)
    #     conf["shadow"] = np.clip(shadow_ratio * 40, 0, 1)

    #     # 5. Watermark Confidence
    #     _, th = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
    #     cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     for c in cnts:
    #         x, y, cw, ch = cv2.boundingRect(c)
    #         if cw > w * 0.25 and ch < h * 0.12:
    #             conf["watermark"] = 0.85
    #             break

    #     return conf

    def process(self) -> Tuple[bytes, Dict, List[str], int]:
        start_time = time.time()
        steps_applied = []
        print(
            f"✅ process() called: auto_detect={self.auto_detect}, operations={self.operations}")

        # Always analyze for telemetry
        confidence = self.analyze()

        if self.auto_detect:
            # AUTO-DETECT MODE: Apply all fixes based on confidence
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
            # USER-SELECTION MODE: Only apply user-selected operations
            logger.info(f"User-selection mode: {self.operations}")

            if "shadow_fix" in self.operations or "shadow" in self.operations:
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

            if "resize" in self.operations:
                if self.resize_dims:
                    self.resize_ecom()
                    steps_applied.append("resize")
                else:
                    logger.warning(
                        "Resize requested but no dimensions provided")
        print(f"✅ process() complete: steps_applied={steps_applied}")
        # Finalize and return
        end_time = time.time()
        duration_ms = int((end_time - start_time) * 1000)

        success, encoded_img = cv2.imencode(
            ".jpg", self.img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return encoded_img.tobytes(), confidence, steps_applied, duration_ms
