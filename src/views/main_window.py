import ctypes
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.presenters import InputHandler

import numpy as np
import numpy.typing as npt
import cupy as cp
import pygame
from config import SCREEN_SIZE, IS_FULL_SCREEN, FPS
from .mouse import Mouse
from .сolormaps import Colormap


class MainWindow:
    def __init__(self):
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # noqa
        pygame.init()

        if IS_FULL_SCREEN:
            self._hardware_screen = pygame.display.set_mode((0, 0), flags=pygame.FULLSCREEN)
        else:
            self._hardware_screen = pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption("Portal Gravity")

        self._mouse = Mouse()
        self._clock = pygame.time.Clock()

        self._colormap = None

    def handle_input(self, input_handler: "InputHandler"):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                input_handler.quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    input_handler.quit()
            self._mouse.handle_input(input_handler, event)

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

    def tick(self):
        print(f"\rFPS = {1000 / self._clock.tick(FPS)}", end='')

    @property
    def size(self) -> npt.NDArray[np.int32]:
        return np.asarray(self._hardware_screen.get_size(), dtype=np.int32)

    @property
    def diagonal_length(self) -> float:
        return float(np.hypot(self.size[0], self.size[1]))

    @property
    def colormap(self) -> Colormap:
        return self._colormap

    @colormap.setter
    def colormap(self, value: Colormap):
        self._colormap = value
