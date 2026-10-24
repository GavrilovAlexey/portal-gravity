import ctypes
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.presenters import InputHandler

import numpy as np
import numpy.typing as npt
import pygame
from config import SCREEN_SIZE, IS_FULL_SCREEN, FPS
from .mouse import Mouse
from .displayer import Displayer
from .сolormaps import Viridis


class MainWindow:
    def __init__(self):
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # noqa
        pygame.init()

        if IS_FULL_SCREEN:
            self.hardware_screen = pygame.display.set_mode((0, 0), flags=pygame.FULLSCREEN)
        else:
            self.hardware_screen = pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption("Portal Gravity")

        self.mouse = Mouse()
        self._clock = pygame.time.Clock()

        self.colormap = Viridis(0, 20)

        self.displayer = Displayer(self)

    def handle_input(self, input_handler: "InputHandler"):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                input_handler.quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    input_handler.quit()
            self.mouse.handle_input(input_handler, event)

    def tick(self):
        print(f"\rFPS = {1000 / self._clock.tick(FPS)}", end='')

    @property
    def size(self) -> npt.NDArray[np.int32]:
        return np.asarray(self.hardware_screen.get_size(), dtype=np.int32)

    @property
    def diagonal_length(self) -> float:
        return float(np.hypot(self.size[0], self.size[1]))
