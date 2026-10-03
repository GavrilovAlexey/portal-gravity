from abc import ABC, abstractmethod
import cupy as cp
import cupy.typing as cpt
from src.models import Engine


class HeatmapQuantity(ABC):
    def __init__(self, engine: Engine):
        self.engine = engine

    @abstractmethod
    def __call__(self, grid: cpt.NDArray[cp.complex64]) -> cpt.NDArray[cp.float32]:
        pass
