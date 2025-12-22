import cv2
import numpy as np
from PIL import Image
from rembg import remove
import io
import time

TARGET_SIZE = (2000, 2000)
CONFIDENCE_THRESHOLD = 0.6

class ImageEnhancer:
    def __init__(self, file_bytes: bytes):
        nparr = np.frombuffer(file_bytes, np.uint8)
        self.img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        self.original_img = self.img.copy()
        self.conf = {}
        self.steps = []

    def foreground_mask(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        return mask

    def analyze(self):
        h, w = self.img.shape[:2]
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        fg = self.foreground_mask(self.img)
        fg_ratio = np.sum(fg > 0) / (h * w)

        self.conf = {
            "bg_clean": 0.0, "shadow": 0.0, "crop": 0.0, 
            "watermark": 0.0, "resize": 0.0
        }

        if min(h, w) < 2000: self.conf["resize"] = 1.0

        # Crop
        if fg_ratio < 0.35: self.conf["crop"] = min(1.0, (0.5 - fg_ratio) * 3)

        # Shadow
        v = hsv[:, :, 2]
        shadow_mask = (v < 0.35 * np.mean(v)) & (fg > 0)
        shadow_ratio = np.sum(shadow_mask) / (h * w)
        self.conf["shadow"] = np.clip(shadow_ratio * 40, 0, 1)

        return self.conf

    def clean_background(self):
        pil = Image.fromarray(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        out = remove(pil)
        bg = Image.new("RGB", out.size, (255, 255, 255))
        if out.mode == "RGBA":
            bg.paste(out, mask=out.split()[3])
        else:
            bg.paste(out)
        self.img = cv2.cvtColor(np.array(bg), cv2.COLOR_RGB2BGR)

    def process(self):
        start_time = time.time()
        self.analyze()
        
        if self.conf["bg_clean"] < CONFIDENCE_THRESHOLD: \
             # (Note: Your script logic was "if confidence HIGH, do it". Adjust as needed)
             pass
        
        # For demo, let's force background removal if user asks, or based on logic
        # Here we just run bg removal as an example step
        self.clean_background()
        self.steps.append("bg_removal")
        
        end_time = time.time()
        duration = int((end_time - start_time) * 1000)
        
        # Encode back to bytes
        is_success, buffer = cv2.imencode(".jpg", self.img)
        return buffer.tobytes(), self.conf, self.steps, duration