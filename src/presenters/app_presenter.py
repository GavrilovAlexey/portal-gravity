import time
import cupy as cp
import pygame
from config import INITIAL_VIRTUAL_SCREEN_CENTER, INITIAL_VIRTUAL_SCREEN_SIZE
from src.models import *
from src.views import *
from ._heatmap_quantities import *
from .heatmap import Heatmap


class AppPresenter:
    def __init__(self, engine: Engine, main_window: MainWindow):
        self._engine = engine
        self._main_window = main_window
        self._is_running = True

        self._virtual_screen_center = INITIAL_VIRTUAL_SCREEN_CENTER
        self._virtual_screen_size = INITIAL_VIRTUAL_SCREEN_SIZE

        self._heatmap = Heatmap(
            Potential(self._engine),
            self._virtual_screen_center,
            self._virtual_screen_size,
            self._main_window.screen_size
        )
        self._colormap = BlueToRed(-20, 20)

        self._engine.external_field = -10j
        self._engine.add_portal(
            P0Portal(
                ContinuousCurve(lambda t: 1.5 * cp.exp(1j * cp.pi * (t - 0.5)) + 1),
                ContinuousCurve(lambda t: 1.5 * cp.exp(1j * cp.pi * (t + 0.5)) - 1),
                False, 500),
        )

        start = time.time()
        self._engine.solve()
        cp.cuda.Device().synchronize()
        print(f"The optimal parameters were found in {time.time() - start} seconds")

    def mainloop(self):
        while self._is_running:
            start_time = time.time()
            self._handle_input()
            self._update_view()

            self._tick()
            print(f"\rFPS = {1 / (time.time() - start_time)}", end='')
        self._quit()

    def _handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._is_running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._is_running = False

    def _update_view(self):
        self._main_window.display_heatmap_values(self._heatmap.values, self._colormap)
        pygame.display.flip()

    def _tick(self):
        self._main_window.tick()

    @staticmethod
    def _quit():
        pygame.quit()
