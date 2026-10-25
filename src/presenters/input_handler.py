from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.presenters import AppPresenter

import numpy as np
import numpy.typing as npt


class InputHandler:
    __slots__ = ["_app_presenter"]

    def __init__(self, app_presenter: "AppPresenter"):
        self._app_presenter = app_presenter

    def __getattr__(self, item):
        return getattr(self._app_presenter, item)

    def handle_input(self):
        self._main_window.handle_input(self)

    def move_virtual_screen(self, movement: npt.NDArray[np.int32]):
        normalised_movement = movement / self._main_window.size * np.array([1, -1])
        self._virtual_screen.move_virtual_screen(normalised_movement)
        self._heatmap.update()

    def resize_virtual_screen(self, scroll_amount: float, position: npt.NDArray[np.int32]):
        normalised_position = (position / self._main_window.size - 0.5) * np.array([1, -1])
        self._virtual_screen.resize_virtual_screen(scroll_amount, normalised_position)
        self._heatmap.update()
