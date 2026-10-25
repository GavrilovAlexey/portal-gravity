from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .main_window import MainWindow

import numpy as np
import numpy.typing as npt
import cupy as cp
import pygame


class Displayer:
    __slots__ = ["_main_window"]

    def __init__(self, main_window: "MainWindow"):
        self._main_window = main_window

    def __getattr__(self, item):
        return getattr(self._main_window, item)

    def display_heatmap_values(self, values: cp.ndarray):
        colors = self._colormap(values)
        heatmap = cp.reshape(colors, (*self.size, 3))
        heatmap = cp.asnumpy(heatmap)
        pygame.surfarray.blit_array(self._hardware_screen, heatmap)

    def display_point(
            self,
            normalized_point: npt.NDArray[np.float64],
            normalized_radius: float,
            color: tuple[int, int, int]
    ):
        point = self.to_hardware_coordinates(normalized_point)
        radius = max(1, self.to_hardware_dimension(normalized_radius))

        pygame.draw.circle(self._hardware_screen, color, point, radius)

    def display_curve(
            self,
            normalized_vertices: npt.NDArray[np.float64],
            normalized_width: float,
            color: tuple[int, int, int]
    ):
        vertices = self.to_hardware_coordinates(normalized_vertices)
        width = max(1, self.to_hardware_dimension(normalized_width))

        pygame.draw.lines(self._hardware_screen, color, False, vertices, width)
        pygame.draw.circle(self._hardware_screen, color, vertices[0], width / 2)
        pygame.draw.circle(self._hardware_screen, color, vertices[-1], width / 2)

    def to_hardware_dimension(self, normalized_dimension: float) -> int:
        return round(normalized_dimension * self.diagonal_length)

    def to_hardware_dimensions(self, normalized_dimensions: npt.NDArray[np.float64]) -> npt.NDArray[np.int32]:
        return np.round((normalized_dimensions * self.diagonal_length).astype(np.int32))

    def to_hardware_coordinates(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.int32]:
        return np.round((points * [1, -1] + 0.5) * self.size).astype(np.int32)