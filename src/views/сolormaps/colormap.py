import cupy as cp


class Colormap:
    __slots__ = ["_min_value", "_max_value"]

    _ANCHOR_VALUES: cp.ndarray
    _ANCHOR_COLORS: cp.ndarray

    def __init__(self, min_value: float, max_value: float):
        self._min_value = cp.float32(min_value)
        self._max_value = cp.float32(max_value)

    def __call__(self, values: cp.ndarray) -> cp.ndarray:
        values = cp.clip(values, self._min_value, self._max_value)
        values = (values - self._min_value) / (self._max_value - self._min_value)

        colors = cp.empty((values.size, 3), dtype=cp.uint8)
        for i in range(3):
            colors[:, i] = cp.interp(values, self._ANCHOR_VALUES, self._ANCHOR_COLORS[:, i])
        return colors
