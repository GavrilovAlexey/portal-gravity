import cupy as cp
from .curve import Curve


class DiscreteCurve(Curve):
    __slots__ = ["_vertices", "_weights"]

    def __init__(self, vertices: cp.ndarray):
        self._vertices = vertices.copy()

        distances = cp.abs(cp.diff(self._vertices))
        cumulative_lengths = cp.zeros_like(self._vertices, dtype=cp.float32)
        cumulative_lengths[1:] = cp.cumsum(distances)
        self._weights = cumulative_lengths / cumulative_lengths[-1]

    def points(self, ts: cp.ndarray) -> cp.ndarray:
        return cp.interp(ts, self._weights, self._vertices).astype(cp.complex64)
