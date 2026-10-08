import numpy as np
import numpy.typing as npt
import cupy as cp
from ._heatmap_quantities import HeatmapQuantity


class Heatmap:
    def __init__(
            self,
            displayed_quantity: HeatmapQuantity,
            virtual_screen_center: npt.NDArray[np.float64],
            virtual_screen_size: npt.NDArray[np.float64],
            screen_size: npt.NDArray[np.int32],
    ):
        self._displayed_quantity = displayed_quantity

        self._virtual_screen_center = virtual_screen_center.copy()
        self._virtual_screen_size = virtual_screen_size.copy()

        self._screen_size = screen_size.copy()
        self._generate_grid()

    def _generate_grid(self):
        x_axis = cp.linspace(self._left, self._right, self._screen_size[0], dtype=cp.float32)
        y_axis = cp.linspace(self._top, self._bottom, self._screen_size[1], dtype=cp.float32)
        xs, ys = cp.meshgrid(x_axis, y_axis, indexing="ij")  # type: ignore
        self._grid = (xs + ys * 1j).ravel()

    def update_virtual_screen_center(self, virtual_screen_center: npt.NDArray[np.float64]):
        self._virtual_screen_center = virtual_screen_center.copy()
        self._generate_grid()

    def update_virtual_screen_size(self, virtual_screen_size: npt.NDArray[np.float64]):
        self._virtual_screen_size = virtual_screen_size.copy()
        self._generate_grid()

    @property
    def _left(self) -> float:
        return float((self._virtual_screen_center - self._virtual_screen_size / 2)[0])

    @property
    def _right(self) -> float:
        return float((self._virtual_screen_center + self._virtual_screen_size / 2)[0])

    @property
    def _bottom(self) -> float:
        return float((self._virtual_screen_center - self._virtual_screen_size / 2)[1])

    @property
    def _top(self) -> float:
        return float((self._virtual_screen_center + self._virtual_screen_size / 2)[1])

    @property
    def values(self) -> cp.ndarray:
        return self._displayed_quantity(self._grid)
