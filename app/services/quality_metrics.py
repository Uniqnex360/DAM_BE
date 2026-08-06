import cv2
import numpy as np
from typing import Dict


def foreground_mask(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))


def analyze_confidence(img: np.ndarray, resize_dims=None) -> Dict[str, float]:
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    fg = foreground_mask(img)
    fg_ratio = np.sum(fg > 0) / (h * w)
    conf = {"bg_clean": 0.0, "shadow": 0.0, "crop": 0.0, "watermark": 0.0, "resize": 0.0}

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