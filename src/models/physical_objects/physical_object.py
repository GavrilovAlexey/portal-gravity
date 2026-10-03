from abc import ABC, abstractmethod
import cupy as cp


class PhysicalObject(ABC):
    @abstractmethod
    def complex_potentials(self, points: cp.ndarray) -> cp.ndarray:
        pass

    @abstractmethod
    def field_strengths(self, points: cp.ndarray) -> cp.ndarray:
        pass
