import sys
import numpy as np
import cupy as cp
import pygame
from config import TOLERANCE
from src.models import *
from src.views import *
from .input_handler import InputHandler
from .renderer import Renderer
from ._virtual_screen import VirtualScreen
from ._heatmap_quantities import *
from ._heatmap import Heatmap


class AppPresenter:
    __slots__ = ["_input_handler", "_renderer", "_engine", "_main_window", "_virtual_screen", "_heatmap"]

    def __init__(self, engine: Engine, main_window: MainWindow):
        self._input_handler = InputHandler(self)
        self._renderer = Renderer(self)

        self._engine = engine
        self._main_window = main_window

        self._virtual_screen = VirtualScreen(2 * self._main_window.size.astype(np.float64) * TOLERANCE)

        self._heatmap = Heatmap(
            MagnitudeOfField(self._engine),
            self._virtual_screen,
            self._main_window.size
        )

        self._engine.add_physical_object(PointParticle(0, 0.5 + 0.5j, 5e10))
        self._engine.add_portal(P0Portal(
            DiscreteCurve(cp.array([-2 - 1j, -2 + 1j], dtype=cp.complex64)),
            DiscreteCurve(cp.array([2 - 1j, 2 + 1j], dtype=cp.complex64)),
            False, 100
        ))
        self._engine.solve()

    def mainloop(self):
        while True:
            self._input_handler.handle_input()
            self._engine.update()
            self._renderer.render()
            self._tick()

    def _tick(self):
        self._main_window.tick()

    @staticmethod
    def quit():
        pygame.quit()
        sys.exit(0)
