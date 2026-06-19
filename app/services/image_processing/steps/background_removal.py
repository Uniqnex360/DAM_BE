import logging
import cv2
import numpy as np
from PIL import Image

from ..model_registry import get_remover

logger = logging.getLogger(__name__)


class BackgroundRemovalStep:
    def __init__(self, background_color: str = "#FFFFFF"):
        
        self.background_color = background_color

    def process(self, image: np.ndarray, original: np.ndarray) -> np.ndarray:
        try:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            remover = get_remover()
            out = remover.process(pil_img)

            if self.background_color == "transparent":
                return cv2.cvtColor(np.array(out), cv2.COLOR_RGBA2BGRA)
            else:
                rgb = tuple(int(self.background_color.lstrip('#')[i:i+2], 16)
                            for i in (0, 2, 4))
                bg = Image.new("RGB", out.size, rgb)

                if out.mode == "RGBA":
                    bg.paste(out, mask=out.split()[3])
                else:
                    bg.paste(out)

                return cv2.cvtColor(np.array(bg), cv2.COLOR_RGB2BGR)

        except Exception as e:
            logger.error(f"BG removal failed: {e}")
            return image
