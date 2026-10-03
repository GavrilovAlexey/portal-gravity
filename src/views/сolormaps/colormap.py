from abc import ABC, abstractmethod
import cupy as cp
import cupy.typing as cpt


class Colormap(ABC):
    @abstractmethod
    def __init__(self, min_value: float, max_value: float):
        pass

    @abstractmethod
    def __call__(self, values: cpt.NDArray[cp.float32]) -> cpt.NDArray[cp.uint8]:
        pass
