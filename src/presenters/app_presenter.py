import sys
import cupy as cp
import pygame
from config import TOLERANCE, INITIAL_VIRTUAL_SCREEN_CENTER, INITIAL_VIRTUAL_SCREEN_SIZE
from src.models import *
from src.views import *
from .input_handler import InputHandler
from .renderer import Renderer
from ._virtual_screen import VirtualScreen
from ._heatmap_quantities import *
from ._heatmap import Heatmap


class AppPresenter:
    def __init__(self, engine: Engine, main_window: MainWindow):
        self.engine = engine
        self.main_window = main_window
        self.main_window.colormap = Viridis(0, 20)

        self.input_handler = InputHandler(self)
        self.displaying_presenter = Renderer(self)

        self.virtual_screen = VirtualScreen(INITIAL_VIRTUAL_SCREEN_CENTER, INITIAL_VIRTUAL_SCREEN_SIZE,
                                            2 * self.main_window.hardware_screen_size * TOLERANCE)

        self.heatmap = Heatmap(
            MagnitudeOfField(self.engine),
            self.virtual_screen,
            self.main_window.hardware_screen_size
        )

        self.engine.add_physical_object(PointParticle(0, 0.5 + 0.5j, 5e10))
        self.engine.add_portal(P0Portal(
            DiscreteCurve(cp.array([-2 - 1j, -2 + 1j], dtype=cp.complex64)),
            DiscreteCurve(cp.array([2 - 1j, 2 + 1j], dtype=cp.complex64)),
            False, 100
        ))
        self.engine.solve()

    def mainloop(self):
        while True:
            self.input_handler.handle_input()
            self.engine.update()
            self.displaying_presenter.render()
            self._tick()

    def _tick(self):
        self.main_window.tick()

    @staticmethod
    def quit():
        pygame.quit()
        sys.exit(0)
