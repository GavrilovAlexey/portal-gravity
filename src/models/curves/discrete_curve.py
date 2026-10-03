from typing import Union
import cupy as cp
from .curve import Curve


class DiscreteCurve(Curve):
    def __init__(self, vertices: cp.ndarray, density: Union[cp.ndarray, None] = None):
        self._vertices = vertices.copy()

        if density is None:
            self._density = cp.ones(len(self._vertices) - 1, cp.float32)
        self._density = self._density.copy()

        weights = cp.abs(cp.diff(self._vertices)) * self._density
        weights /= cp.sum(weights)

        self._cum_weights = cp.zeros(len(self._vertices), cp.float32)
        self._cum_weights[1:] = cp.cumsum(weights)

    def points(self, ts: cp.ndarray) -> cp.ndarray:
        return cp.interp(ts, self._cum_weights, self._vertices).astype(cp.complex64)
