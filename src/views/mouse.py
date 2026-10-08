from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.presenters import AppPresenter

import numpy as np
import numpy.typing as npt
import pygame


class Mouse:
    def __init__(self, screen_size):
        self._screen_size = screen_size

        self._position = (np.asarray(pygame.mouse.get_pos()) - screen_size / 2).astype(np.int32) * np.array([1, -1])
        self._movement = np.zeros(2, dtype=np.int32)
        self.is_LBM_pressed = pygame.mouse.get_pressed()[0]

    def handle_input(self, app_presenter: "AppPresenter", event: pygame.event.Event):
        if event.type in (pygame.WINDOWFOCUSGAINED, pygame.WINDOWENTER):
            self.__init__(self._screen_size)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.is_LBM_pressed = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.is_LBM_pressed = False
        elif event.type == pygame.MOUSEMOTION:
            self.move(np.asarray(event.rel))
            if self.is_LBM_pressed:
                app_presenter.move_virtual_screen(self.normalized_movement)
        elif event.type == pygame.MOUSEWHEEL:
            app_presenter.resize_virtual_screen(event.y)

    def move(self, movement: npt.NDArray[np.int32]):
        self._movement = movement * np.array([1, -1])
        self._position += self._movement

    @property
    def position(self) -> npt.NDArray[np.int32]:
        return self._position.copy()

    @property
    def movement(self) -> npt.NDArray[np.int32]:
        return self._movement.copy()

    @property
    def normalized_position(self) -> npt.NDArray[np.float64]:
        return self._position / self._screen_size

    @property
    def normalized_movement(self) -> npt.NDArray[np.float64]:
        return self._movement / self._screen_size
