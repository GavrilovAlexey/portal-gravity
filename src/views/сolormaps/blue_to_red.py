import cupy as cp
from .colormap import Colormap


class BlueToRed(Colormap):
    _ANCHOR_VALUES = cp.linspace(0, 1, 11, dtype=cp.float32)
    _ANCHOR_COLORS = cp.array([
        [5, 48, 97],
        [33, 102, 172],
        [67, 147, 195],
        [146, 197, 222],
        [209, 229, 240],
        [247, 247, 247],
        [253, 219, 199],
        [244, 165, 130],
        [214, 96, 77],
        [178, 24, 43],
        [103, 0, 31],
    ], dtype=cp.uint8)