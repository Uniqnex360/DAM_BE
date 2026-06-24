import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import io
import os
import logging
from typing import Optional, Dict, Any
from app.services.image_processing.model_registry import get_rembg_session
from rembg import remove

logger = logging.getLogger(__name__)

ROOM_REGISTRY = {
    "living_room": {
        "label": "Living Room",
        "emoji": "🛋️",
        "file": "living_room.jpg",
        "floor_y": 78,
        "scale_hint": 0.38
    },
    "bedroom": {
        "label": "Bedroom",
        "emoji": "🛏️",
        "file": "bedroom.jpg",
        "floor_y": 80,
        "scale_hint": 0.35
    },
    "living_room2": {
        "label": "Living2",
        "emoji": "🛏️",
        "file": "living_room2.jpg",
        "floor_y": 80,
        "scale_hint": 0.35
    },
    "office": {
        "label": "Office",
        "emoji": "💼",
        "file": "office.jpg",
        "floor_y": 75,
        "scale_hint": 0.30
    },
    "dining_room": {
        "label": "Dining Room",
        "emoji": "🍽️",
        "file": "dining_room.jpg",
        "floor_y": 80,
        "scale_hint": 0.36
    },
    "outdoor_patio": {
        "label": "Outdoor Patio",
        "emoji": "🌿",
        "file": "outdoor_patio.jpg",
        "floor_y": 82,
        "scale_hint": 0.40
    },
}


class RoomVisualizerStep:
    def __init__(
        self,
        room_id: str = "living_room",
        scale: float = 0.38,
        x_percent: float = 50.0,
        y_percent: Optional[float] = None
    ):
        self.room_id = room_id
        self.scale = scale
        self.x_percent = x_percent
        # Use provided y or fall back to registry default floor level
        self.y_percent = y_percent or ROOM_REGISTRY.get(
            room_id, {}).get("floor_y", 80)
        self.session = get_rembg_session()
        self.static_base = os.path.join("app", "static", "rooms")

    # app/services/image_processing/steps/room_visualizer.py

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Input: BGR Image (OpenCV)
        Output: BGR Image (OpenCV) composited into a room
        """
        try:
            # 1. Background Removal (Convert BGR to PIL for rembg compatibility)
            # Convert OpenCV (BGR) to PIL (RGB)
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_input = Image.fromarray(img_rgb)

            # Remove BG (When passed a PIL Image, rembg returns a PIL Image)
            logger.info("Running background removal...")
            pil_prod = remove(pil_input, session=self.session).convert("RGBA")

            # 2. Load Room Background
            room_info = ROOM_REGISTRY.get(self.room_id)
            if not room_info:
                raise ValueError(f"Room {self.room_id} not found in registry")

            # Use absolute path to ensure static file is found
            room_path = os.path.join(self.static_base, room_info["file"])
            if not os.path.exists(room_path):
                logger.error(f"Room file missing at: {room_path}")
                # Fallback: if room is missing, we can't composite. Return original.
                return image

            room_img = Image.open(room_path).convert("RGBA")
            rw, rh = room_img.size

            # 3. Scaling
            pw_target = int(rw * max(0.05, min(self.scale, 0.95)))
            # Handle empty images or errors
            if pil_prod.width == 0:
                return image

            ratio = pw_target / pil_prod.width
            ph_target = int(pil_prod.height * ratio)
            pil_prod_resized = pil_prod.resize(
                (pw_target, ph_target), Image.LANCZOS)

            # 4. Positioning (Anchor to bottom-center)
            cx = int(rw * self.x_percent / 100)
            floor_y = int(rh * self.y_percent / 100)
            paste_x = cx - pw_target // 2
            paste_y = floor_y - ph_target

            # 5. Create Shadow Layer
            canvas = room_img.copy()
            canvas = self._add_contact_shadow(
                canvas, pw_target, ph_target, paste_x, paste_y)

            # 6. Final Composite
            canvas.paste(pil_prod_resized,
                         (paste_x, paste_y), pil_prod_resized)

            # Convert back to OpenCV BGR
            res_rgb = np.array(canvas.convert("RGB"))
            return cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR)

        except Exception as e:
            logger.error(f"Room Visualizer failed: {e}")
            # If everything fails, return the original image so the API doesn't crash
            return image
    def _add_contact_shadow(self, canvas, pw, ph, px, py):
        """Draws a soft elliptical shadow at the base of the product."""
        shadow_w = int(pw * 0.9)
        shadow_h = int(pw * 0.15)
        sx = px + pw // 2 - shadow_w // 2
        sy = py + ph - shadow_h // 2

        ellipse_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(ellipse_layer)
        # Low opacity black ellipse
        draw.ellipse([sx, sy, sx + shadow_w, sy + shadow_h],
                     fill=(0, 0, 0, 80))
        # Blur the shadow
        blurred = ellipse_layer.filter(
            ImageFilter.GaussianBlur(radius=pw * 0.05))
        return Image.alpha_composite(canvas, blurred)
