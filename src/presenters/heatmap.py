import numpy as np
import numpy.typing as npt
import cupy as cp
from ._heatmap_quantities import HeatmapQuantity


class Heatmap:
    def __init__(
            self,
            displayed_quantity: HeatmapQuantity,
            virtual_screen_center: complex,
            virtual_screen_size: complex,
            screen_size: npt.NDArray[np.int32],
    ):
        self._displayed_quantity = displayed_quantity

        self._virtual_screen_center = virtual_screen_center
        self._virtual_screen_size = virtual_screen_size

        self._screen_size = screen_size
        self._generate_grid()

    def _generate_grid(self):
        simplified_x_axis = cp.linspace(self._left, self._right, self._screen_size[0], dtype=cp.float32)
        simplified_y_axis = cp.linspace(self._top, self._bottom, self._screen_size[1], dtype=cp.float32)
        xs, ys = cp.meshgrid(simplified_x_axis, simplified_y_axis, indexing="ij")
        self._grid = (xs + ys * 1j).ravel()

    @property
    def _left(self) -> float:
        return (self._virtual_screen_center - self._virtual_screen_size / 2).real

    @property
    def _right(self) -> float:
        return (self._virtual_screen_center + self._virtual_screen_size / 2).real

    @property
    def _bottom(self) -> float:
        return (self._virtual_screen_center - self._virtual_screen_size / 2).imag

    @property
    def _top(self) -> float:
        return (self._virtual_screen_center + self._virtual_screen_size / 2).imag

    @property
    def values(self) -> cp.ndarray:
        return self._displayed_quantity(self._grid)
