import sys
import time
import numpy as np
import numpy.typing as npt
import cupy as cp
import pygame
from config import (TOLERANCE, INITIAL_SUBSTEPS_PER_FRAME, INITIAL_SUBSTEP_DT,
                    INITIAL_VIRTUAL_SCREEN_CENTER, INITIAL_VIRTUAL_SCREEN_SIZE, VELOCITY_OF_RESIZE_VIRTUAL_SCREEN)
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

        self._substeps_per_frame = INITIAL_SUBSTEPS_PER_FRAME
        self._substep_dt = INITIAL_SUBSTEP_DT

        self._engine.add_physical_object(PointParticle(0, 0.5 + 0.5j, 5e10))
        self._engine.add_portal(
            P0Portal(
                DiscreteCurve(cp.array([-2 + 1j, -2 - 1j], dtype=cp.complex64)),
                DiscreteCurve(cp.array([2 + 1j, 2 - 1j], dtype=cp.complex64)),
                False, 100),
        )
        self._engine.solve()
        cp.cuda.Device().synchronize()

        start = time.time()
        self._engine.solve()
        cp.cuda.Device().synchronize()
        print(f"The optimal parameters were found in {time.time() - start} seconds")

        self._heatmap = Heatmap(
            MagnitudeOfField(self._engine),
            self._virtual_screen_center,
            self._virtual_screen_size,
            self._main_window.screen_size,
        )
        self._colormap = Viridis(0, 20)

    def mainloop(self):
        while self._is_running:
            start_time = time.time()

            self._handle_input()
            self._update_engine()
            self._update_view()
            self._tick()

            print(f"\rFPS = {1 / (time.time() - start_time)}", end='')

    def move_virtual_screen(self, normalised_movement: npt.NDArray[np.float64]):
        self._virtual_screen_center -= normalised_movement * self._virtual_screen_size
        self._heatmap.update_virtual_screen_center(self._virtual_screen_center)

    def resize_virtual_screen(self, scroll_amount: float, normalised_position: npt.NDArray[np.float64]):
        scale = VELOCITY_OF_RESIZE_VIRTUAL_SCREEN ** (-scroll_amount)
        scale = max(scale,
                    2 * TOLERANCE * self._main_window.screen_size[0] / self._virtual_screen_size[0],
                    2 * TOLERANCE * self._main_window.screen_size[1] / self._virtual_screen_size[1])

        self._virtual_screen_size *= scale
        self.move_virtual_screen(normalised_position * (1 - 1 / scale))
        self._heatmap.update_virtual_screen_size(self._virtual_screen_size)

    def _handle_input(self):
        self._main_window.handle_input(self)

    def _update_engine(self):
        self._engine.update(self._substeps_per_frame, self._substep_dt)

    def _update_view(self):
        self._main_window.display_heatmap_values(self._heatmap.values, self._colormap)
        pygame.display.flip()

    def _tick(self):
        self._main_window.tick()

    def quit(self):
        self._is_running = False
        pygame.quit()
        sys.exit(0)
