import logging
import numpy as np
import cv2
from PIL import Image
from ..model_registry import get_wm_detector, get_lama

logger = logging.getLogger(__name__)


class WatermarkRemovalStep:
    def process(self, image: np.ndarray, original: np.ndarray) -> np.ndarray:
        try:
            logger.info("Starting watermark removal...")
            h, w = image.shape[:2]
            combined_mask = np.zeros((h, w), dtype=np.uint8)

            detector = get_wm_detector()
            if detector is None:
                logger.warning("Watermark detector unavailable, skipping.")
                return image

            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = detector(img_rgb, conf=0.25, verbose=False)

            detected = False
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    pad = 10
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(w, x2 + pad)
                    y2 = min(h, y2 + pad)
                    combined_mask[y1:y2, x1:x2] = 255
                    detected = True
                    logger.info(
                        f"Watermark detected: box=({x1},{y1},{x2},{y2}) conf={box.conf[0]:.2f}")

            if not detected:
                logger.info("No watermark detected by YOLO, skipping.")
                return image

            kernel = np.ones((15, 15), np.uint8)
            combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

            lama = get_lama()
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            pil_mask = Image.fromarray(combined_mask)
            result = lama(pil_img, pil_mask)
            return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

        except Exception as e:
            logger.error(f"Watermark removal failed: {e}")
            return image
