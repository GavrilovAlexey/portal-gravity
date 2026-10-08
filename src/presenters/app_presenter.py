import sys
import time
import numpy as np
import numpy.typing as npt
import cupy as cp
import pygame
from config import INITIAL_VIRTUAL_SCREEN_CENTER, INITIAL_VIRTUAL_SCREEN_SIZE, VELOCITY_OF_CHANGE_VIRTUAL_SCREEN_SIZE, \
    TOLERANCE
from src.models import *
from src.views import *
from ._heatmap_quantities import *
from .heatmap import Heatmap


class AppPresenter:
    def __init__(self, engine: Engine, main_window: MainWindow):
        self._engine = engine
        self._main_window = main_window
        self._is_running = True

        self._virtual_screen_center = INITIAL_VIRTUAL_SCREEN_CENTER.copy()
        self._virtual_screen_size = INITIAL_VIRTUAL_SCREEN_SIZE.copy()

        self._delta_time = 0

        self._heatmap = Heatmap(
            Potential(self._engine),
            self._virtual_screen_center,
            self._virtual_screen_size,
            self._main_window.screen_size,
        )
        self._colormap = BlueToRed(-10, 15)

        self._engine.add_physical_object(PointParticle(-2.5, 1e11))
        self._engine.add_portal(
            P0Portal(
                ContinuousCurve(lambda t: 1.5 + 0.5 * cp.cos(2 * cp.pi * t) + 3j * (t - 0.5)),
                ContinuousCurve(lambda t: cp.exp(1j * cp.pi * (t + 0.5)) - 1),
                False, 100),
        )
        self._engine.solve()
        cp.cuda.Device().synchronize()

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

    def move_virtual_screen_center(self, relative_mouse_movement: npt.NDArray[np.float64]):
        self._virtual_screen_center -= relative_mouse_movement * self._virtual_screen_size
        self._heatmap.virtual_screen_center = self._virtual_screen_center

    def change_virtual_screen_size(self, direction_of_change_virtual_screen_size: int):
        virtual_screen_size_scale = 1 - (direction_of_change_virtual_screen_size *
                                         VELOCITY_OF_CHANGE_VIRTUAL_SCREEN_SIZE * self._delta_time)
        self._virtual_screen_size *= virtual_screen_size_scale
        self._virtual_screen_size = np.maximum(self._virtual_screen_size, self._main_window.screen_size * TOLERANCE)
        self._heatmap.virtual_screen_size = self._virtual_screen_size

    def _handle_input(self):
        self._main_window.handle_input(self)

    def _update_view(self):
        self._main_window.display_heatmap_values(self._heatmap.values, self._colormap)
        pygame.display.flip()

    def _tick(self):
        self._delta_time = self._main_window.tick()

    def quit(self):
        self._is_running = False
        pygame.quit()
        sys.exit(0)
