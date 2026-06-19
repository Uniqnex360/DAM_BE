from typing import Callable, Dict

from .protocols import ProcessingStep
from .steps import (
    BackgroundRemovalStep,
    ImageRefillStep,
    RetouchStep,
    ShadowRemovalStep,
    TextRemovalStep,
    WatermarkRemovalStep,
)


class StepRegistry:
    def __init__(self):
        self._steps: Dict[str, Callable[[], ProcessingStep]] = {
            "text-remove": TextRemovalStep,
            "watermark-remove": WatermarkRemovalStep,
            "bg-remove": BackgroundRemovalStep,
            "shadow-remove": ShadowRemovalStep,
            "shadow_fix": ShadowRemovalStep,
            "image-refill": ImageRefillStep,
            "retouch": RetouchStep,
        }

    def get_step(self, operation: str) -> Callable[[], ProcessingStep]:
        return self._steps.get(operation)

    def register(self, operation: str, factory: Callable[[], ProcessingStep]) -> None:
        self._steps[operation] = factory
