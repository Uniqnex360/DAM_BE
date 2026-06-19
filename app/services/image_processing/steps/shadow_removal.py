
import cv2
import numpy as np
from PIL import Image
import io
import logging
from typing import Optional
from app.services.image_processing.model_registry import get_rembg_session
from rembg import remove



logger = logging.getLogger(__name__)


class ShadowRemovalStep:
    

    def __init__(self, use_neural: bool = True, background_color: str = "#FFFFFF"):
        
        self.use_neural = use_neural
        self.background_color = background_color

    def process(self, image: np.ndarray, original: np.ndarray = None) -> np.ndarray:
        
        if self.use_neural:
            try:
                logger.warning("Attempting neural shadow removal (ISNet)...")
                result = self._neural_remove(image)
                logger.info("Neural shadow removal succeeded ")
                return result
            except Exception as e:
                logger.error(f"Neural model failed: {e}")
                logger.info("Falling back to classical shadow removal")

        return self._classical_remove(image)

    def _neural_remove(self, image: np.ndarray) -> np.ndarray:
       
        session = get_rembg_session()

        max_dim = 2000
        h, w = image.shape[:2]
        working_img = image.copy()

        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            working_img = cv2.resize(
                working_img,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA
            )

        success, buffer = cv2.imencode(".png", working_img)
        if not success:
            raise ValueError("Failed to encode image for rembg")

        input_bytes = buffer.tobytes()

        output_bytes = remove(input_bytes, session=session)

        pil_img = Image.open(io.BytesIO(output_bytes)).convert("RGBA")

        if self.background_color == "transparent":
            return np.array(pil_img)


        else:
            rgb = tuple(int(self.background_color.lstrip('#')[i:i+2], 16)
                        for i in (0, 2, 4))
            bg = Image.new("RGB", pil_img.size, rgb)  

        bg.paste(pil_img, mask=pil_img.split()[3])

        result = cv2.cvtColor(np.array(bg), cv2.COLOR_RGB2BGR)

        if max(h, w) > max_dim:
            result = cv2.resize(result, (w, h), interpolation=cv2.INTER_CUBIC)

        return result

    def _classical_remove(self, image: np.ndarray) -> np.ndarray:
        
        logger.info("Using classical CLAHE shadow removal")

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        return result
