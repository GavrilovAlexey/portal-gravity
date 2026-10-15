from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.presenters import InputHandler

import numpy as np
import numpy.typing as npt
import pygame


class Mouse:
    def __init__(self):
        self._position = None
        self._find_mouse_position()

        self._is_LBM_pressed = pygame.mouse.get_pressed()[0]

    def handle_input(self, input_handler: "InputHandler", event: pygame.event.Event):
        if event.type in (pygame.WINDOWFOCUSGAINED, pygame.WINDOWENTER):
            self._find_mouse_position()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._is_LBM_pressed = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._is_LBM_pressed = False
        elif event.type == pygame.MOUSEMOTION:
            self.move(event.rel)
            if self._is_LBM_pressed:
                input_handler.move_virtual_screen(event.rel)
        elif event.type == pygame.MOUSEWHEEL:
            input_handler.resize_virtual_screen(event.y, self.position)

    def move(self, movement: npt.NDArray[np.int32]):
        self._position += movement

    @property
    def position(self) -> npt.NDArray[np.int32]:
        return self._position.copy()

    def _find_mouse_position(self):
        self._position = np.asarray(pygame.mouse.get_pos(), dtype=np.int32)
