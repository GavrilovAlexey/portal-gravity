import sys
import numpy as np
import numpy.typing as npt
import cupy as cp
import pygame
from config import (TOLERANCE, INITIAL_SUBSTEPS_PER_FRAME, INITIAL_SUBSTEP_DT,
                    INITIAL_VIRTUAL_SCREEN_CENTER, INITIAL_VIRTUAL_SCREEN_SIZE)
from src.models import *
from src.views import *
from ._virtual_screen import VirtualScreen
from ._heatmap_quantities import *
from ._heatmap import Heatmap


class AppPresenter:
    def __init__(self, engine: Engine, main_window: MainWindow):
        self._engine = engine
        self._main_window = main_window
        self._main_window.link_to_app_presenter(self)
        self._is_running = True

        self._virtual_screen = VirtualScreen(INITIAL_VIRTUAL_SCREEN_CENTER, INITIAL_VIRTUAL_SCREEN_SIZE,
                                             2 * self._main_window.hardware_screen_size * TOLERANCE)

        self._substeps_per_frame = INITIAL_SUBSTEPS_PER_FRAME
        self._substep_dt = INITIAL_SUBSTEP_DT

        self._engine.add_physical_object(PointParticle(0, 0.5 + 0.5j, 5e10))
        self._engine.add_portal(P0Portal(
            DiscreteCurve(cp.array([-2 - 1j, -2 + 1j], dtype=cp.complex64)),
            DiscreteCurve(cp.array([2 - 1j, 2 + 1j], dtype=cp.complex64)),
            False, 100
        ))
        self._engine.solve()

        self._heatmap = Heatmap(
            MagnitudeOfField(self._engine),
            self._virtual_screen,
            self._main_window.hardware_screen_size
        )
        self._main_window.colormap = Viridis(0, 20)

    def mainloop(self):
        while self._is_running:
            self._handle_input()
            self._update_engine()
            self._update_main_window()
            self._tick()

    def move_virtual_screen(self, movement: npt.NDArray[np.int32]):
        normalised_movement = movement / self._main_window.hardware_screen_size * np.array([1, -1])
        self._virtual_screen.move_virtual_screen(normalised_movement)
        self._heatmap.update()

    def resize_virtual_screen(self, scroll_amount: float, position: npt.NDArray[np.int32]):
        normalised_position = (position / self._main_window.hardware_screen_size - 0.5) * np.array([1, -1])
        self._virtual_screen.resize_virtual_screen(scroll_amount, normalised_position)
        self._heatmap.update()

    def _handle_input(self):
        self._main_window.handle_input()

    def _update_engine(self):
        self._engine.update(self._substeps_per_frame, self._substep_dt)

    def _update_main_window(self):
        self._main_window.display_heatmap_values(self._heatmap.values)
        pygame.display.flip()

    def _tick(self):
        self._main_window.tick()

    def quit(self):
        self._is_running = False
        pygame.quit()
        sys.exit(0)
