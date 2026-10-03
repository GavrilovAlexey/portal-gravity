import cupy as cp
from .colormap import Colormap


class BlackToWhite(Colormap):
    def __init__(self, min_value: float, max_value: float):
        self._min_value = cp.float32(min_value)
        self._max_value = cp.float32(max_value)

    def __call__(self, values: cp.ndarray) -> cp.ndarray:
        brightness = cp.clip(values, self._min_value, self._max_value)
        brightness = (brightness - self._min_value) / (self._max_value - self._min_value)
        brightness = brightness.astype(cp.uint8)
        return cp.broadcast_to(brightness[:, None], (values.shape[0], 3))
