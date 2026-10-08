from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.presenters import AppPresenter

import ctypes
import numpy as np
import numpy.typing as npt
import cupy as cp
import pygame
from config import SCREEN_SIZE, IS_FULL_SCREEN, FPS
from .сolormaps import Colormap


class MainWindow:
    def __init__(self):
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # noqa
        pygame.init()

        if IS_FULL_SCREEN:
            self._screen = pygame.display.set_mode((0, 0), flags=pygame.FULLSCREEN)
        else:
            self._screen = pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption("Portal Gravity")

        self._clock = pygame.time.Clock()

        self._is_dragging = False

    def handle_input(self, app_presenter: "AppPresenter"):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                app_presenter.quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    app_presenter.quit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._is_dragging = True
            elif event.type == pygame.MOUSEBUTTONUP:
                self._is_dragging = False
            elif event.type == pygame.MOUSEMOTION and self._is_dragging:
                app_presenter.move_virtual_screen_center(event.rel * np.array([1, -1]) / self.screen_size)
            elif event.type == pygame.MOUSEWHEEL:
                app_presenter.change_virtual_screen_size(np.sign(event.y))

    def display_heatmap_values(self, values: cp.ndarray, colormap: Colormap):
        colors = colormap(values)
        heatmap = cp.reshape(colors, (*self.screen_size, 3))
        heatmap = cp.asnumpy(heatmap)
        pygame.surfarray.blit_array(self._screen, heatmap)

    def tick(self) -> float:
        return self._clock.tick(FPS) / 1000

    @property
    def screen_size(self) -> npt.NDArray[np.int32]:
        return np.asarray(self._screen.get_size(), dtype=np.int32)
