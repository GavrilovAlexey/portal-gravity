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
        heatmap = cp.reshape(colors, (*self.hardware_screen_size, 3))
        heatmap = cp.asnumpy(heatmap)
        pygame.surfarray.blit_array(self._hardware_screen, heatmap)

    def display_curve(self, normalized_vertices: npt.NDArray[np.float64], color: tuple[int, int, int]):
        vertices = self.to_hardware_coordinate(normalized_vertices)
        pygame.draw.lines(self._hardware_screen, color, False, vertices, 5)

    def to_hardware_coordinate(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.int32]:
        return np.round((points * [1, -1] + 0.5) * self.hardware_screen_size).astype(np.int32)

    def tick(self):
        print(f"\rFPS = {1000 / self._clock.tick(FPS)}", end='')

    @property
    def hardware_screen_size(self) -> npt.NDArray[np.int32]:
        return np.asarray(self._hardware_screen.get_size(), dtype=np.int32)

    @property
    def colormap(self) -> Colormap:
        return self._colormap

    @colormap.setter
    def colormap(self, value: Colormap):
        self._colormap = value
