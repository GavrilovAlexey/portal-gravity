import numpy as np
import numpy.typing as npt
import cupy as cp
from ._virtual_screen import VirtualScreen
from ._heatmap_quantities import HeatmapQuantity


class Heatmap:
    def __init__(
            self,
            displayed_quantity: HeatmapQuantity,
            virtual_screen: VirtualScreen,
            hardware_screen_size: npt.NDArray[np.int32],
    ):
        self._displayed_quantity = displayed_quantity

        self._virtual_screen = virtual_screen

        self._hardware_screen_size = hardware_screen_size.copy()
        self._generate_grid()

    def _generate_grid(self):
        x_axis = cp.linspace(
            self._virtual_screen.left, self._virtual_screen.right, self._hardware_screen_size[0], dtype=cp.float32)
        y_axis = cp.linspace(
            self._virtual_screen.top, self._virtual_screen.bottom, self._hardware_screen_size[1], dtype=cp.float32)
        xs, ys = cp.meshgrid(x_axis, y_axis, indexing="ij")  # type: ignore
        self._grid = (xs + ys * 1j).ravel()

    def update(self):
        self._generate_grid()

    @property
    def values(self) -> cp.ndarray:
        return self._displayed_quantity.calculate(self._grid)
