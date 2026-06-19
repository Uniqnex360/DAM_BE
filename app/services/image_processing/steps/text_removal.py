import logging
import numpy as np
import cv2
from ..model_registry import get_ocr_reader

logger = logging.getLogger(__name__)


class TextRemovalStep:
    def process(self, image: np.ndarray, original: np.ndarray) -> np.ndarray:
        try:
            logger.info("Starting clean-room text removal...")
            reader = get_ocr_reader()
            results = reader.readtext(
                image, text_threshold=0.3, link_threshold=0.2, low_text=0.2)

            if not results:
                logger.info("No text detected.")
                return image

            h, w = image.shape[:2]
            img = image.copy()

            for (bbox, text, prob) in results:
                points = np.array(bbox).astype(np.int32)
                x, y, bw, bh = cv2.boundingRect(points)

                x1, y1 = max(0, x - 2), max(0, y - 2)
                x2, y2 = min(w, x + bw + 2), min(h, y + bh + 2)

                roi = img[y1:y2, x1:x2]

                border_mask = np.ones(roi.shape[:2], dtype=np.uint8) * 255
                border_mask[2:-2, 2:-2] = 0

                bg_color = np.median(
                    roi[border_mask > 0], axis=0).astype(np.uint8)

                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                avg_bg_gray = np.mean(cv2.cvtColor(
                    np.uint8([[bg_color]]), cv2.COLOR_BGR2GRAY))

                if avg_bg_gray > 127:
                    _, text_mask = cv2.threshold(
                        gray_roi, avg_bg_gray - 30, 255, cv2.THRESH_BINARY_INV)
                else:
                    _, text_mask = cv2.threshold(
                        gray_roi, avg_bg_gray + 30, 255, cv2.THRESH_BINARY)

                text_mask = cv2.dilate(text_mask, np.ones(
                    (3, 3), np.uint8), iterations=1)

                roi[text_mask > 0] = bg_color

                roi_refined = cv2.GaussianBlur(roi, (3, 3), 0)
                mask_feather = cv2.GaussianBlur(
                    text_mask, (5, 5), 0).astype(np.float32) / 255.0

                for c in range(3):
                    img[y1:y2, x1:x2, c] = (
                        roi_refined[:, :, c] * mask_feather +
                        img[y1:y2, x1:x2, c] * (1 - mask_feather)
                    ).astype(np.uint8)

            logger.info(f"Surgically filled {len(results)} text areas.")
            return img
        except Exception as e:
            logger.error(f"Text removal failed: {e}")
            return image
