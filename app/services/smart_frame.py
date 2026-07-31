"""
Smart Frame Fit Service
Automatically detects product boundaries, removes whitespace,
scales proportionally, and centers within a user-defined frame.
"""

import io
import logging
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)


class SmartFrameFit:
    """
    Smart Frame Fit processor.
    
    Flow:
    1. Detect product boundaries (remove whitespace)
    2. Calculate scale factor to fit within frame
    3. Scale image proportionally (maintaining aspect ratio)
    4. Center on output canvas with user-defined background
    """
    
    # Default frame configuration
    DEFAULT_OUTPUT_SIZE = (1200, 1200)
    DEFAULT_FRAME_INSET = 40  # pixels of padding inside the frame
    DEFAULT_BACKGROUND = "#FFFFFF"
    DEFAULT_WHITESPACE_THRESHOLD = 240  # pixels above this value are "white"
    
    def process(
        self,
        image_bytes: bytes,
        output_width: int = 1200,
        output_height: int = 1200,
        frame_inset: int = 40,
        background_color: str = "#FFFFFF",
        whitespace_threshold: int = 240,
        min_product_ratio: float = 0.3,  # product must fill at least 30% of frame
        max_product_ratio: float = 0.95,  # product fills at most 95% of frame
    ) -> bytes:
        """
        Process an image with Smart Frame Fit.
        
        Args:
            image_bytes: Raw image bytes
            output_width: Final canvas width
            output_height: Final canvas height
            frame_inset: Padding inside the frame (pixels)
            background_color: Canvas background color (hex)
            whitespace_threshold: Pixel value above which is "whitespace" (0-255)
            min_product_ratio: Minimum fill ratio (zoom in if below)
            max_product_ratio: Maximum fill ratio (zoom out if above)
            
        Returns:
            PNG bytes of the framed image
        """
        
        # 1. Load image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode == "RGBA":
            # Composite onto white background for boundary detection
            bg = Image.new("RGB", image.size, (255, 255, 255))
            bg.paste(image, mask=image.split()[3])
            original = image  # Keep original for final paste
            detection_image = bg
            has_alpha = True
        elif image.mode != "RGB":
            detection_image = image.convert("RGB")
            original = detection_image
            has_alpha = False
        else:
            detection_image = image
            original = image
            has_alpha = False
        
        logger.info(f"Smart Frame: Input size={detection_image.size}")
        
        # 2. Detect product boundaries (remove whitespace)
        bbox = self._detect_content_bounds(
            detection_image, 
            threshold=whitespace_threshold
        )
        
        if bbox is None:
            logger.warning("No product detected, using full image")
            bbox = (0, 0, detection_image.width, detection_image.height)
        
        left, top, right, bottom = bbox
        product_width = right - left
        product_height = bottom - top
        
        logger.info(
            f"Smart Frame: Detected product at ({left},{top})-({right},{bottom}), "
            f"size={product_width}x{product_height}"
        )
        
        # 3. Crop to product bounds
        if has_alpha:
            product_image = original.crop(bbox)
        else:
            product_image = original.crop(bbox)
        
        # 4. Calculate frame dimensions
        frame_width = output_width - (2 * frame_inset)
        frame_height = output_height - (2 * frame_inset)
        
        # 5. Calculate scale factor
        scale = self._calculate_scale(
            product_width, product_height,
            frame_width, frame_height,
            min_product_ratio, max_product_ratio
        )
        
        logger.info(f"Smart Frame: Scale factor={scale:.2f}")
        
        # 6. Scale product image
        new_product_width = int(product_width * scale)
        new_product_height = int(product_height * scale)
        
        # Use LANCZOS for high-quality resampling
        product_image = product_image.resize(
            (new_product_width, new_product_height),
            Image.LANCZOS
        )
        
        # 7. Create output canvas
        if has_alpha and background_color.lower() in ("transparent", "none", ""):
            canvas = Image.new("RGBA", (output_width, output_height), (0, 0, 0, 0))
        elif background_color.lower() in ("transparent", "none", ""):
            canvas = Image.new("RGBA", (output_width, output_height), (0, 0, 0, 0))
        else:
            canvas = Image.new("RGB", (output_width, output_height), background_color)
        
        # 8. Center product on canvas
        paste_x = (output_width - new_product_width) // 2
        paste_y = (output_height - new_product_height) // 2
        
        if product_image.mode == "RGBA":
            canvas.paste(product_image, (paste_x, paste_y), product_image)
        else:
            canvas.paste(product_image, (paste_x, paste_y))
        
        logger.info(
            f"Smart Frame: Output={output_width}x{output_height}, "
            f"Product pos=({paste_x},{paste_y}), "
            f"Product size={new_product_width}x{new_product_height}"
        )
        
        # 9. Return PNG bytes
        output = io.BytesIO()
        canvas.save(output, format="PNG", quality=95)
        return output.getvalue()
    
    def _detect_content_bounds(
        self,
        image: Image.Image,
        threshold: int = 240
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the bounding box of the actual product content.
        
        Uses edge detection + thresholding to find where the product starts/ends,
        ignoring the surrounding whitespace/background.
        """
        
        # Convert to numpy array
        pixels = np.array(image)
        
        # Convert to grayscale
        if len(pixels.shape) == 3:
            gray = np.mean(pixels, axis=2)
        else:
            gray = pixels
        
        # Create mask: pixels that are "not white" (product)
        # threshold: pixels below this are product, above are background
        mask = gray < threshold
        
        # Find bounding box of non-white pixels
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not rows.any() or not cols.any():
            return None  # No product detected
        
        # Get the first and last row/col with content
        top = np.argmax(rows)
        bottom = len(rows) - np.argmax(rows[::-1])
        left = np.argmax(cols)
        right = len(cols) - np.argmax(cols[::-1])
        
        # Add a small margin (2% of dimension) to avoid cutting too tight
        height = bottom - top
        width = right - left
        margin_x = int(width * 0.02)
        margin_y = int(height * 0.02)
        
        left = max(0, left - margin_x)
        top = max(0, top - margin_y)
        right = min(image.width, right + margin_x)
        bottom = min(image.height, bottom + margin_y)
        
        return (left, top, right, bottom)
    
    def _calculate_scale(
        self,
        product_width: int,
        product_height: int,
        frame_width: int,
        frame_height: int,
        min_product_ratio: float = 0.3,
        max_product_ratio: float = 0.95,
    ) -> float:
        """
        Calculate the scale factor to fit the product within the frame.
        
        - If product is too small (< min_product_ratio of frame), zoom IN
        - If product is too large (> max_product_ratio of frame), zoom OUT
        - Otherwise, keep current size
        
        Always maintains aspect ratio.
        """
        
        # Calculate current fill ratio (how much of the frame the product fills)
        width_ratio = product_width / frame_width
        height_ratio = product_height / frame_height
        current_fill = max(width_ratio, height_ratio)
        
        logger.info(
            f"Smart Frame: Product={product_width}x{product_height}, "
            f"Frame={frame_width}x{frame_height}, "
            f"Fill ratio={current_fill:.2f}"
        )
        
        if current_fill < min_product_ratio:
            # Product is too small → zoom IN
            # Scale so product fills at least min_product_ratio of frame
            target_fill = (min_product_ratio + max_product_ratio) / 2
            scale = target_fill / current_fill
            logger.info(f"Smart Frame: Zooming IN (fill={current_fill:.2f} < {min_product_ratio})")
            
        elif current_fill > max_product_ratio:
            # Product is too large → zoom OUT
            target_fill = (min_product_ratio + max_product_ratio) / 2
            scale = target_fill / current_fill
            logger.info(f"Smart Frame: Zooming OUT (fill={current_fill:.2f} > {max_product_ratio})")
            
        else:
            # Product is within acceptable range
            scale = 1.0
            logger.info(f"Smart Frame: No scaling needed")
        
        # Clamp scale to reasonable bounds
        scale = max(0.1, min(scale, 3.0))
        
        return scale


# Convenience function for direct use
def smart_frame_fit(
    image_bytes: bytes,
    output_width: int = 1200,
    output_height: int = 1200,
    frame_inset: int = 40,
    background_color: str = "#FFFFFF",
) -> bytes:
    """Quick one-liner for Smart Frame Fit."""
    processor = SmartFrameFit()
    return processor.process(
        image_bytes=image_bytes,
        output_width=output_width,
        output_height=output_height,
        frame_inset=frame_inset,
        background_color=background_color,
    )