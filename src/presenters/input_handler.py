from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.presenters import AppPresenter

import numpy as np
import numpy.typing as npt


class InputHandler:
    def __init__(self, app_presenter: "AppPresenter"):
        self.app_presenter = app_presenter

    def handle_input(self):
        self.app_presenter.main_window.handle_input(self)

    def move_virtual_screen(self, movement: npt.NDArray[np.int32]):
        normalised_movement = movement / self.app_presenter.main_window.hardware_screen_size * np.array([1, -1])
        self.app_presenter.virtual_screen.move_virtual_screen(normalised_movement)
        self.app_presenter.heatmap.update()

    def resize_virtual_screen(self, scroll_amount: float, position: npt.NDArray[np.int32]):
        normalised_position = (position / self.app_presenter.main_window.hardware_screen_size - 0.5) * np.array([1, -1])
        self.app_presenter.virtual_screen.resize_virtual_screen(scroll_amount, normalised_position)
        self.app_presenter.heatmap.update()
