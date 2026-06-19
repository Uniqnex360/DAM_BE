import logging
import cv2
import numpy as np
from ..model_registry import get_iopaint
from iopaint.schema import InpaintRequest, HDStrategy

logger = logging.getLogger(__name__)


class ImageRefillStep:
    def process(self, image: np.ndarray, original: np.ndarray) -> np.ndarray:
        try:
            logger.info("Auto-analyzing geometry for IOPaint refill...")
            h, w = image.shape[:2]

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            pad = int(max(h, w) * 0.15)
            img_padded = cv2.copyMakeBorder(
                image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            mask = np.zeros(img_padded.shape[:2], dtype=np.uint8)

            has_stubs = False

            if np.any(edges[0, :]):
                idx = np.where(edges[0, :] > 0)[0]
                cv2.rectangle(mask, (idx[0] + pad - 10, 0),
                              (idx[-1] + pad + 10, pad + 20), 255, -1)
                has_stubs = True

            if np.any(edges[-1, :]):
                idx = np.where(edges[-1, :] > 0)[0]
                cv2.rectangle(mask, (idx[0] + pad - 10, h + pad - 20),
                              (idx[-1] + pad + 10, h + 2 * pad), 255, -1)
                has_stubs = True

            if np.any(edges[:, 0]):
                idx = np.where(edges[:, 0] > 0)[0]
                cv2.rectangle(
                    mask, (0, idx[0] + pad - 10), (pad + 20, idx[-1] + pad + 10), 255, -1)
                has_stubs = True

            if np.any(edges[:, -1]):
                idx = np.where(edges[:, -1] > 0)[0]
                cv2.rectangle(
                    mask, (w + pad - 20, idx[0] + pad - 10), (w + 2 * pad, idx[-1] + pad + 10), 255, -1)
                has_stubs = True

            if not has_stubs:
                logger.info("No cut-off edges detected by AI analysis.")
                return image

            model = get_iopaint()
            img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)

            result_rgb = model.inpaint(
                image=img_rgb,
                mask=mask,
                config=InpaintRequest(
                    hd_strategy=HDStrategy.ORIGINAL,
                    hd_strategy_crop_margin=128,
                    prompt="seamless product surface extension, high resolution metal texture",
                ),
            )

            return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

        except Exception as e:
            logger.error(f"IOPaint Refill failed: {e}")
            return image
