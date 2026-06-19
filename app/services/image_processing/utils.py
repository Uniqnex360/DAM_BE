from typing import Optional
import cv2
import numpy as np
from PIL import Image


def decode_image(file_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")
    return img


def encode_image(image: np.ndarray, output_format: str = "jpg", quality: int = 95) -> bytes:
    ext = f".{output_format.lower().lstrip('.')}"
    success, encoded = cv2.imencode(
        ext, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise RuntimeError(f"Failed to encode image as {output_format}")
    return encoded.tobytes()


def crop_to_aspect_ratio(image: np.ndarray, ratio_str: str) -> np.ndarray:
    h, w = image.shape[:2]
    ratio_w, ratio_h = map(int, ratio_str.split(":"))
    target_ratio = ratio_w / ratio_h
    current_ratio = w / h

    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        offset = (w - new_w) // 2
        return image[:, offset: offset + new_w].copy()
    else:
        new_h = int(w / target_ratio)
        offset = (h - new_h) // 2
        return image[offset: offset + new_h, :].copy()


def foreground_mask(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))


def upscale_to_size(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize(
        (new_w, new_h), Image.LANCZOS
    )
    upscaled = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    canvas[y_off: y_off + new_h, x_off: x_off + new_w] = upscaled
    return canvas


def apply_single_resize(img: np.ndarray, resize_config: dict) -> Optional[np.ndarray]:
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
