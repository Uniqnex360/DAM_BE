import cv2
import numpy as np
from PIL import Image
from rembg import remove
import io
import time
import logging
from typing import Tuple, Dict, List

# Configure Logging
logger = logging.getLogger(__name__)

# ==========================================
# CONSTANTS
# ==========================================
TARGET_SIZE = (2000, 2000)

# CHANGE THIS: Lower it to 0.1 temporarily to FORCE changes for testing
# Original was 0.6
CONFIDENCE_THRESHOLD = 0.6 

class ImageProcessor:
    def __init__(self, file_bytes: bytes):
        nparr = np.frombuffer(file_bytes, np.uint8)
        self.img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if self.img is None:
            raise ValueError("Could not decode image bytes")
        self.original_h, self.original_w = self.img.shape[:2]

    def _foreground_mask(self, img) -> np.ndarray:
        """Exact logic from your script: Get binary mask."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    def analyze(self) -> Dict[str, float]:
        """Exact logic from your script: Generate scores."""
        h, w = self.img.shape[:2]
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        
        fg = self._foreground_mask(self.img)
        fg_ratio = np.sum(fg > 0) / (h * w)

        conf = {
            "bg_clean": 0.0, "shadow": 0.0, "crop": 0.0, 
            "watermark": 0.0, "resize": 0.0
        }

        # 1. Resize Confidence (If image is small, we MUST resize)
        if min(h, w) < 2000: 
            conf["resize"] = 1.0

        # 2. Crop Confidence (If foreground is tiny)
        if fg_ratio < 0.35: 
            conf["crop"] = min(1.0, (0.5 - fg_ratio) * 3)

        # 3. Background Cleanliness (High Score = DIRTY background)
        corner_std = np.mean([
            np.std(gray[:80, :80]), np.std(gray[:80, -80:]),
            np.std(gray[-80:, :80]), np.std(gray[-80:, -80:])
        ])
        # Your logic: (std - 10) / 20. If std > 30, score is 1.0 (Dirty).
        conf["bg_clean"] = np.clip((corner_std - 10) / 20, 0, 1)

        # 4. Shadow Confidence (High Score = Has Shadows)
        v = hsv[:, :, 2]
        mean_v = np.mean(v) if np.mean(v) > 0 else 1
        shadow_mask = (v < 0.35 * mean_v) & (fg > 0)
        shadow_ratio = np.sum(shadow_mask) / (h * w)
        conf["shadow"] = np.clip(shadow_ratio * 40, 0, 1)

        # 5. Watermark Confidence
        _, th = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, cw, ch = cv2.boundingRect(c)
            if cw > w * 0.25 and ch < h * 0.12:
                conf["watermark"] = 0.85
                break

        return conf

    # ----------------------------------------
    # ENHANCEMENT METHODS
    # ----------------------------------------
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
        self.img[shadow_mask > 0] = cv2.add(self.img[shadow_mask > 0], np.array([25, 25, 25]))

    def remove_watermark(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        self.img = cv2.inpaint(self.img, mask, 5, cv2.INPAINT_TELEA)

    def smart_crop(self):
        h, w = self.img.shape[:2]
        fg = self._foreground_mask(self.img)

        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return

        image_area = h * w
        boxes = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < image_area * 0.01: continue
            x, y, cw, ch = cv2.boundingRect(c)
            boxes.append((x, y, cw, ch))

        if not boxes: return

        x_min = min(b[0] for b in boxes)
        y_min = min(b[1] for b in boxes)
        x_max = max(b[0] + b[2] for b in boxes)
        y_max = max(b[1] + b[3] for b in boxes)

        crop_w = x_max - x_min
        crop_h = y_max - y_min
        crop_area_ratio = (crop_w * crop_h) / image_area

        if crop_area_ratio < 0.4 or crop_area_ratio > 0.95: return

        pad = int(0.08 * max(crop_w, crop_h))
        x1, y1 = max(0, x_min - pad), max(0, y_min - pad)
        x2, y2 = min(w, x_max + pad), min(h, y_max + pad)

        self.img = self.img[y1:y2, x1:x2]

    def resize_ecom(self):
        h, w = self.img.shape[:2]
        scale = min(TARGET_SIZE[0] / w, TARGET_SIZE[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)

        pil = Image.fromarray(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        pil = pil.resize((new_w, new_h), Image.LANCZOS)

        canvas = Image.new("RGB", TARGET_SIZE, (255, 255, 255))
        offset = ((TARGET_SIZE[0] - new_w) // 2, (TARGET_SIZE[1] - new_h) // 2)
        canvas.paste(pil, offset)

        self.img = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)

    def process(self) -> Tuple[bytes, Dict, List[str], int]:
        start_time = time.time()
        steps_applied = []
        
        # 1. Analyze
        confidence = self.analyze()

        # 2. Apply Logic (Exact match to your script)
        # Note: Order matters. Usually Shadow/Watermark -> Crop -> BG -> Resize
        
        if confidence["shadow"] >= CONFIDENCE_THRESHOLD:
            self.remove_shadow()
            steps_applied.append("shadow_fix")

        if confidence["watermark"] >= CONFIDENCE_THRESHOLD:
            self.remove_watermark()
            steps_applied.append("watermark_removal")

        if confidence["crop"] >= CONFIDENCE_THRESHOLD:
            self.smart_crop()
            steps_applied.append("smart_crop")

        # Logic Fix: If score is high (DIRTY), we clean it.
        if confidence["bg_clean"] >= CONFIDENCE_THRESHOLD:
            self.clean_background()
            steps_applied.append("bg_removal")

        if confidence["resize"] >= CONFIDENCE_THRESHOLD:
            self.resize_ecom()
            steps_applied.append("resize")

        # 3. Finalize
        end_time = time.time()
        duration_ms = int((end_time - start_time) * 1000)

        success, encoded_img = cv2.imencode(".jpg", self.img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return encoded_img.tobytes(), confidence, steps_applied, duration_ms