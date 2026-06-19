from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class ProcessingStep(Protocol):
    def process(self, image: np.ndarray, original: np.ndarray) -> np.ndarray:
        ...
