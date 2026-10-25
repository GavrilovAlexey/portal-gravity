from abc import ABC, abstractmethod
import cupy as cp


class Curve(ABC):
    @abstractmethod
    def points(self, ts: cp.ndarray) -> cp.ndarray:
        pass
