import cupy as cp
import cupy.typing as cpt
from .colormap import Colormap


class Grey(Colormap):
    def __init__(self, min_value: float, max_value: float):
        self._min_value = cp.float32(min_value)
        self._max_value = cp.float32(max_value)
        self._scale = cp.float32(255 / (self._max_value - self._min_value))

    def __call__(self, values: cpt.NDArray[cp.float32]) -> cpt.NDArray[cp.uint8]:
        brightness = cp.clip(values, self._min_value, self._max_value)
        brightness -= self._min_value
        brightness *= self._scale
        brightness = brightness.astype(cp.uint8)
        return cp.broadcast_to(brightness[:, None], (values.shape[0], 3))
