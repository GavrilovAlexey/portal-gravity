import sys
import numpy as np
import cupy as cp
import pygame

from config import TOLERANCE, FPS
from src.models import *
from src.views import *

from .input_handler import InputHandler
from .renderer import Renderer
from ._virtual_screen import VirtualScreen
from ._heatmap_quantities import *
from ._heatmap import Heatmap


class AppPresenter:
    __slots__ = ["_input_handler", "_renderer", "engine", "main_window", "virtual_screen", "heatmap", "clock"]

    def __init__(self, engine: Engine, main_window: MainWindow):
        self._input_handler = InputHandler(self)
        self._renderer = Renderer(self)

        self.engine = engine
        self.main_window = main_window

        self.virtual_screen = VirtualScreen(2 * self.main_window.size.astype(np.float64) * TOLERANCE)

        self.heatmap = Heatmap(
            MagnitudeOfField(self.engine),
            self.virtual_screen,
            self.main_window.size
        )

        self.clock = pygame.time.Clock()

        self.engine.add_physical_object(PointParticle(0, 0.5 + 0.5j, 5e10))
        self.engine.add_portal(P0Portal(
            DiscreteCurve(cp.array([-2 - 1j, -2 + 1j], dtype=cp.complex64)),
            DiscreteCurve(cp.array([2 - 1j, 2 + 1j], dtype=cp.complex64)),
            False, 100
        ))
        self.engine.solver.solve()

    def mainloop(self):
        while True:
            self._input_handler.handle_input()
            self.engine.update()
            self._renderer.render()
            self._tick()

    def _tick(self):
        print(f"\rFPS = {1000 / self.clock.tick(FPS)}", end='')

    @staticmethod
    def quit():
        pygame.quit()
        sys.exit(0)
