import cupy as cp
from .colormap import Colormap


class BlueToRed(Colormap):
    _ANCHOR_VALUES = cp.linspace(0, 1, 11, dtype=cp.float32)
    _ANCHOR_COLORS = cp.array([
        (103, 0, 31), (178, 24, 43), (214, 96, 77), (244, 165, 130), (253, 219, 199),
        (247, 247, 247),
        (209, 229, 240), (146, 197, 222), (67, 147, 195), (33, 102, 172), (5, 48, 97)
    ], dtype=cp.uint8)

    def __init__(self, min_value: float, max_value: float):
        self._min_value = cp.float32(min_value)
        self._max_value = cp.float32(max_value)

    def __call__(self, values: cp.ndarray) -> cp.ndarray:
        values = cp.clip(values, self._min_value, self._max_value)
        values = (values - self._min_value) / (self._max_value - self._min_value)

        colors = cp.empty((values.size, 3))
        for i in range(3):
            colors[:, i] = cp.interp(values, self._ANCHOR_VALUES, self._ANCHOR_COLORS[:, i])
        return colors
