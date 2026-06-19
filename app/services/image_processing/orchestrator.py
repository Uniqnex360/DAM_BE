import logging
import time
from typing import Dict, Optional

from .analyzer import ImageAnalyzer
from .exceptions import StepSkippedException
from .registry import StepRegistry
from .utils import (
    apply_single_resize,
    crop_to_aspect_ratio,
    decode_image,
    encode_image,
    upscale_to_size,
)

logger = logging.getLogger(__name__)
CONFIDENCE_THRESHOLD = 0.6


class   ImageProcessor:
    def __init__(
        self,
        file_bytes: bytes,
        resize_dims: dict = None,
        operations: list = None,
        autoDetect: bool = False,
        skip_crop: bool = False,
        crop_mode: Optional[str] = None,
        target_aspect_ratio: Optional[str] = None,
        target_dimensions: dict = None,
        background_color: str = "#FFFFFF",
        step_registry: Optional[StepRegistry] = None,
    ):
        self.img = decode_image(file_bytes)
        self.original_h, self.original_w = self.img.shape[:2]
        self.resize_dims = resize_dims
        self.operations = operations or []
        self.original_img = self.img.copy()
        self.auto_detect = autoDetect
        self.skip_crop = skip_crop
        self.crop_mode = crop_mode
        self.target_aspect_ratio = target_aspect_ratio
        self.background_color = background_color  
        self.last_ai_alpha = None

        if target_dimensions:
            self.target_w = target_dimensions.get("width", self.original_w)
            self.target_h = target_dimensions.get("height", self.original_h)
        else:
            self.target_w = self.original_w
            self.target_h = self.original_h

        self._analyzer = ImageAnalyzer()
        self._registry = step_registry if step_registry is not None else StepRegistry()

        logger.info(
            f"ImageProcessor: input={self.original_w}x{self.original_h}, "
            f"target={self.target_w}x{self.target_h}, "
            f"skip_crop={self.skip_crop}, operations={self.operations}"
        )

    def resize_ecom(self):
        h, w = self.img.shape[:2]
        if not self.resize_dims:
            return
        if isinstance(self.resize_dims, list):
            results = []
            for config in self.resize_dims:
                result = apply_single_resize(self.original_img, config)
                results.append({
                    "id": config.get("id"),
                    "width": config.get("width"),
                    "height": config.get("height"),
                    "image_bytes": result
                })
            return results
        else:
            return apply_single_resize(self.img, self.resize_dims)

    def process(self) -> Dict:
        start_time = time.time()
        self.resize_results = None
        steps_applied = []
        messages = []  # ← INITIALIZE HERE so it's always in scope

        confidence = self._analyzer.analyze(
            self.img, self.original_img, self.resize_dims, self.operations
        )

        if self.crop_mode == "preset" and self.target_aspect_ratio:
            self.img = crop_to_aspect_ratio(self.img, self.target_aspect_ratio)

        if self.auto_detect:
            if confidence["bg_clean"] > CONFIDENCE_THRESHOLD:
                step = self._registry.get_step("bg-remove")()
                self.img = step.process(self.img, self.original_img)
                steps_applied.append("bg_removal")
        else:
            _shadow_done = False

            if "text-remove" in self.operations:
                step = self._registry.get_step("text-remove")()
                self.img = step.process(self.img, self.original_img)
                steps_applied.append("text_removal")

            if "image-refill" in self.operations:
                step = self._registry.get_step("image-refill")()
                self.img = step.process(self.img, self.original_img)
                steps_applied.append("geometry_reconstruction")

            if "watermark-remove" in self.operations:
                step = self._registry.get_step("watermark-remove")()
                self.img = step.process(self.img, self.original_img)

            if "retouch" in self.operations:
                step = self._registry.get_step("retouch")()
                self.img = step.process(self.img, self.original_img)
                steps_applied.append("retouch")

            if any(op in self.operations for op in ("shadow-remove", "shadow_fix")):
                if not _shadow_done:
                    try:
                        step = self._registry.get_step("shadow-remove")(
                            background_color=self.background_color
                        )
                        self.img = step.process(self.img, self.original_img)
                        steps_applied.append("shadow_fix")
                        _shadow_done = True
                    except StepSkippedException as e:
                        messages.append(str(e))
                        logger.info(str(e))

            if "bg-remove" in self.operations:
                step = self._registry.get_step("bg-remove")(
                    background_color=self.background_color
                )
                self.img = step.process(self.img, self.original_img)
                steps_applied.append("bg_removal")

            if self.resize_dims:
                result = self.resize_ecom()
                if isinstance(result, list):
                    self.resize_results = result
                    steps_applied.append("resize_multiple")
                elif result is not None:
                    self.img = result
                    steps_applied.append("resize")

        if self.resize_results is None and "resize" not in steps_applied:
            if self.crop_mode != "preset":
                cur_h, cur_w = self.img.shape[:2]
                if (cur_w, cur_h) != (self.target_w, self.target_h):
                    self.img = upscale_to_size(
                        self.img, self.target_w, self.target_h)

        image_bytes = encode_image(self.img)

        return {
            "image_bytes": image_bytes,
            "confidence": confidence,
            "steps_applied": steps_applied,
            "messages": messages,
            "duration_ms": int((time.time() - start_time) * 1000),
            "resize_results": self.resize_results,
        }
