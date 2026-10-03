import ctypes
import numpy as np
import numpy.typing as npt
import cupy as cp
import cupy.typing as cpt
import pygame
from config import SCREEN_SIZE, IS_FULL_SCREEN, FPS
from .сolormaps import Colormap


class MainWindow:
    def __init__(self):
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # noqa
        pygame.init()

        if IS_FULL_SCREEN:
            self._screen = pygame.display.set_mode((0, 0), flags=pygame.DOUBLEBUF | pygame.FULLSCREEN, vsync=True)
        else:
            self._screen = pygame.display.set_mode(SCREEN_SIZE, flags=pygame.DOUBLEBUF, vsync=True)
        pygame.display.set_caption("Portal Gravity")

        self._clock = pygame.time.Clock()

    def display_heatmap_values(self, values: cpt.NDArray[cp.float32], colormap: Colormap):
        colors = colormap(values)
        colors = cp.asnumpy(colors)
        heatmap = cp.reshape(colors, (*self.screen_size, 3))
        pygame.surfarray.blit_array(self._screen, heatmap)

    def tick(self):
        self._clock.tick(FPS)

    @property
    def screen_size(self) -> npt.NDArray[np.int32]:
        return np.asarray(self._screen.get_size(), dtype=np.int32)
