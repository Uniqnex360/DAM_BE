import cv2
import numpy as np
from PIL import ImageColor


class WallRecoloringService:
    @staticmethod
    def apply_color(room_bgr: np.ndarray, wall_mask: np.ndarray, hex_color: str) -> np.ndarray:
        # 1. Convert Hex to BGR
        rgb = ImageColor.getrgb(hex_color)
        target_bgr = np.array([rgb[2], rgb[1], rgb[0]], dtype=np.float32)

        # 2. Prepare float arrays for math
        room_float = room_bgr.astype(np.float32)
        color_layer = np.full(room_bgr.shape, target_bgr, dtype=np.float32)

        # 3. Blend: 50% original shadows/texture + 50% new color
        recolored = cv2.addWeighted(room_float, 0.5, color_layer, 0.5, 0)

        # 4. Prepare Mask
        mask_3d = cv2.merge([wall_mask, wall_mask, wall_mask]
                            ).astype(np.float32) / 255.0

        # 5. Composite: (Recolored * Mask) + (Original * InverseMask)
        result = (recolored * mask_3d) + (room_float * (1.0 - mask_3d))

        return np.clip(result, 0, 255).astype(np.uint8)
