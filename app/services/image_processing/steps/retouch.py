import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class RetouchStep:
    def process(self, image: np.ndarray, original: np.ndarray) -> np.ndarray:
        try:
            mode = "auto"
            if mode == "auto":
                mode = self._detect_image_type(image)
            return self._retouch_product(image) if mode == "product" else self._retouch_portrait(image)
        except Exception as e:
            logger.error(f"Retouch failed: {e}")
            return image

    @staticmethod
    def _detect_image_type(image: np.ndarray) -> str:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 30, 60])
        upper_skin = np.array([20, 150, 255])
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        return "product" if (np.sum(skin_mask > 0) / skin_mask.size) < 0.05 else "portrait"

    @staticmethod
    def _retouch_product(image: np.ndarray) -> np.ndarray:
        logger.info("Applying PRO-WEBSITE clarity retouch")
        img = cv2.edgePreservingFilter(image, flags=1, sigma_s=30, sigma_r=0.4)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        l_float = l.astype(np.float32)
        blur = cv2.GaussianBlur(l_float, (0, 0), 3.0)
        high_pass = l_float - blur
        l = np.clip(l_float + (high_pass * 1.4), 0, 255).astype(np.uint8)
        gauss_fine = cv2.GaussianBlur(l, (0, 0), 0.8)
        l = cv2.addWeighted(l, 1.5, gauss_fine, -0.5, 0)
        img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * 0.03
        img = cv2.addWeighted(img, 1.0, cv2.filter2D(img, -1, kernel), 1.0, 0)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= 1.10
        img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(
            np.uint8), cv2.COLOR_HSV2BGR)
        return cv2.convertScaleAbs(img, alpha=1.04, beta=2)

    @staticmethod
    def _retouch_portrait(image: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoisingColored(image, None, 7, 7, 7, 21)
