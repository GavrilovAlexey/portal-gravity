from collections.abc import Callable
import cupy as cp
from .curve import Curve


class ContinuousCurve(Curve):
    def __init__(self, defining_function: Callable[[cp.ndarray], cp.ndarray]):
        self._defining_function = defining_function

    def points(self, ts: cp.ndarray) -> cp.ndarray:
        return self._defining_function(ts)
