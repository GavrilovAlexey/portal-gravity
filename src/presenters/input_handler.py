from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.presenters import AppPresenter

import numpy as np
import numpy.typing as npt
from ._heatmap import Heatmap


class InputHandler:
    def __init__(self, app_presenter: "AppPresenter"):
        self.app_presenter = app_presenter

        self.engine = app_presenter.engine
        self.main_window = app_presenter.main_window

        self.virtual_screen = app_presenter.virtual_screen

    def handle_input(self):
        self.main_window.handle_input(self)

    def move_virtual_screen(self, movement: npt.NDArray[np.int32]):
        normalised_movement = movement / self.main_window.size * np.array([1, -1])
        self.virtual_screen.move_virtual_screen(normalised_movement)
        self.heatmap.update()

    def resize_virtual_screen(self, scroll_amount: float, position: npt.NDArray[np.int32]):
        normalised_position = (position / self.main_window.size - 0.5) * np.array([1, -1])
        self.virtual_screen.resize_virtual_screen(scroll_amount, normalised_position)
        self.heatmap.update()

    def quit(self):
        self.app_presenter.quit()

    @property
    def heatmap(self) -> Heatmap:
        return self.app_presenter.heatmap
