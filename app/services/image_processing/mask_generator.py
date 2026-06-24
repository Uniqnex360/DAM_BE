import torch
import numpy as np
import cv2
import logging
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

logger = logging.getLogger(__name__)

# B2 is good, but we will add classical CV refinement to make it perfect
MODEL_ID = "nvidia/segformer-b2-finetuned-ade-512-512"
processor = SegformerImageProcessor.from_pretrained(MODEL_ID)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)


class MaskGeneratorService:
    @staticmethod
    def generate_wall_mask(image_bgr: np.ndarray) -> np.ndarray:
        try:
            # 1. AI Inference (Rough Mask)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            inputs = processor(images=image_rgb, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)

            upsampled_logits = torch.nn.functional.interpolate(
                outputs.logits, size=image_bgr.shape[:2], mode="bilinear", align_corners=False
            )
            pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()

            # Create rough wall mask (Index 0 is wall)
            mask = np.where(pred_seg == 0, 255, 0).astype(np.uint8)

            # 2. EDGE REFINEMENT (The "Secret Sauce")
            # We use a Guided Filter. This uses the original high-res image
            # to "snap" the blurry AI mask to the sharp edges of the plants/furniture.
            gray_guide = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

            # radius=8 (how far to look for edges), eps=0.01^2 (edge sensitivity)
            refined_mask = cv2.ximgproc.guidedFilter(
                guide=gray_guide,
                src=mask,
                radius=10,
                eps=100
            )

            # 3. FURNITURE PROTECTION (Aggressive Erosion)
            # Shrink the wall mask slightly so paint "tucks behind" objects
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            refined_mask = cv2.erode(refined_mask, kernel, iterations=1)

            return refined_mask
        except Exception as e:
            logger.error(f"AI Mask Generation failed: {e}")
            raise e
