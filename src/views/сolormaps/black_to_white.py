import cupy as cp
from .colormap import Colormap


class BlackToWhite(Colormap):
    _ANCHOR_VALUES = cp.array([0, 1], dtype=cp.float32)
    _ANCHOR_COLORS = cp.array([[0, 0, 0], [255, 255, 255]], dtype=cp.uint8)
