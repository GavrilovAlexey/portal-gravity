import ctypes
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.presenters import AppPresenter

import numpy as np
import numpy.typing as npt
import cupy as cp
import pygame
from config import SCREEN_SIZE, IS_FULL_SCREEN, FPS
from .mouse import Mouse
from .сolormaps import Colormap


class MainWindow:
    def __init__(self):
        self._app_presenter = None

        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # noqa
        pygame.init()

        if IS_FULL_SCREEN:
            self._screen = pygame.display.set_mode((0, 0), flags=pygame.FULLSCREEN)
        else:
            self._screen = pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption("Portal Gravity")

        self._mouse = Mouse()
        self._clock = pygame.time.Clock()

        self._colormap = None

    def link_to_app_presenter(self, app_presenter: "AppPresenter"):
        self._app_presenter = app_presenter

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._app_presenter.quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._app_presenter.quit()
            self._mouse.handle_input(self._app_presenter, event)

    def display_heatmap_values(self, values: cp.ndarray):
        colors = self._colormap(values)
        heatmap = cp.reshape(colors, (*self.hardware_screen_size, 3))
        heatmap = cp.asnumpy(heatmap)
        pygame.surfarray.blit_array(self._screen, heatmap)

    def tick(self):
        print(f"\rFPS = {1000 / self._clock.tick(FPS)}", end='')

    @property
    def hardware_screen_size(self) -> npt.NDArray[np.int32]:
        return np.asarray(self._screen.get_size(), dtype=np.int32)

    @property
    def colormap(self) -> Colormap:
        return self._colormap

    @colormap.setter
    def colormap(self, value: Colormap):
        self._colormap = value
