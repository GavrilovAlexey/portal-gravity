from abc import ABC, abstractmethod
import cupy as cp


class Colormap(ABC):
    @abstractmethod
    def __init__(self, min_value: float, max_value: float):
        pass

    @abstractmethod
    def __call__(self, values: cp.ndarray) -> cp.ndarray:
        pass
