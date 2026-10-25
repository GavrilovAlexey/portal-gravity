import ctypes
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.presenters import InputHandler

import numpy as np
import numpy.typing as npt
import pygame
from config import HARDWARE_SCREEN_SIZE, IS_FULL_SCREEN, FPS
from .mouse import Mouse
from .displayer import Displayer
from .сolormaps import Viridis


class MainWindow:
    __slots__ = ["_displayer", "_hardware_screen", "_mouse", "_clock", "_colormap"]

    def __init__(self):
        self._displayer = Displayer(self)

        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # noqa
        pygame.init()

        if IS_FULL_SCREEN:
            self._hardware_screen = pygame.display.set_mode((0, 0), flags=pygame.FULLSCREEN)
        else:
            self._hardware_screen = pygame.display.set_mode(HARDWARE_SCREEN_SIZE)
        pygame.display.set_caption("Portal Gravity")

        self._mouse = Mouse()
        self._clock = pygame.time.Clock()

        self._colormap = Viridis(0, 20)

    def __getattr__(self, item):
        return getattr(self._displayer, item)

    def handle_input(self, input_handler: "InputHandler"):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                input_handler.quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    input_handler.quit()
            self._mouse.handle_input(input_handler, event)

    def tick(self):
        print(f"\rFPS = {1000 / self._clock.tick(FPS)}", end='')

    @property
    def size(self) -> npt.NDArray[np.int32]:
        return np.asarray(self._hardware_screen.get_size(), dtype=np.int32)

    @property
    def diagonal_length(self) -> float:
        return float(np.hypot(self.size[0], self.size[1]))
