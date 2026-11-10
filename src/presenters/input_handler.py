from typing import TYPE_CHECKING
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from src.presenters import AppPresenter


class InputHandler:
    __slots__ = ["_app_presenter"]

    def __init__(self, app_presenter: "AppPresenter"):
        self._app_presenter = app_presenter

    def handle_input(self):
        self._app_presenter.main_window.handle_input(self)

    def move_virtual_screen(self, movement: npt.NDArray[np.int32]):
        normalised_movement = movement / self._app_presenter.main_window.size * np.array([1, -1])
        self._app_presenter.virtual_screen.move_virtual_screen(normalised_movement)
        self._app_presenter.heatmap.update()

    def resize_virtual_screen(self, scroll_amount: float, position: npt.NDArray[np.int32]):
        normalised_position = (position / self._app_presenter.main_window.size - 0.5) * np.array([1, -1])
        self._app_presenter.virtual_screen.resize_virtual_screen(scroll_amount, normalised_position)
        self._app_presenter.heatmap.update()

    def quit(self):
        self._app_presenter.quit()
