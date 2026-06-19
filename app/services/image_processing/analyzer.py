import numpy as np
import cv2
from .utils import foreground_mask


class ImageAnalyzer:
    def analyze(self, image: np.ndarray, original: np.ndarray, resize_dims, operations) -> dict:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        fg = foreground_mask(image)
        fg_ratio = np.sum(fg > 0) / (h * w)
        conf = {"bg_clean": 0.0, "shadow": 0.0,
                "crop": 0.0, "watermark": 0.0, "resize": 0.0}

        if resize_dims:
            conf["resize"] = 1.0
        if fg_ratio < 0.35:
            conf["crop"] = min(1.0, (0.5 - fg_ratio) * 3)

        corner_std = np.mean([np.std(gray[:80, :80]), np.std(gray[:80, -80:])])
        conf["bg_clean"] = np.clip((corner_std - 10) / 20, 0, 1)

        v = hsv[:, :, 2]
        mean_v = np.mean(v) if np.mean(v) > 0 else 1
        shadow_mask = (v < 0.35 * mean_v) & (fg > 0)
        shadow_ratio = np.sum(shadow_mask) / (h * w)
        conf["shadow"] = np.clip(shadow_ratio * 40, 0, 1)

        return conf
