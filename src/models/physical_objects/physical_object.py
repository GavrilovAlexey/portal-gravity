from abc import ABC, abstractmethod
from typing import Callable, Self
import cupy as cp


class PhysicalObject(ABC):
    @abstractmethod
    def complex_potentials(self, points: cp.ndarray) -> cp.ndarray:
        pass

    @abstractmethod
    def field_strengths(self, points: cp.ndarray) -> cp.ndarray:
        pass

    @abstractmethod
    def updated_object(self, field_strengths: Callable[[cp.ndarray], cp.ndarray], dt: float) -> Self:
        pass
