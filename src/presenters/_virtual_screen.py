import numpy as np
import numpy.typing as npt
from config import INITIAL_VIRTUAL_SCREEN_CENTER, INITIAL_VIRTUAL_SCREEN_SIZE, VELOCITY_OF_RESIZE_VIRTUAL_SCREEN


class VirtualScreen:
    def __init__(self, min_virtual_screen_size: npt.NDArray[np.float64]):
        self._virtual_screen_center = INITIAL_VIRTUAL_SCREEN_CENTER.copy()
        self._virtual_screen_size = INITIAL_VIRTUAL_SCREEN_SIZE.copy()
        self._min_virtual_screen_size = min_virtual_screen_size.copy()

    def to_normalized_dimension(self, dimension: float) -> float:
        return dimension / self.diagonal_length

    def to_normalized_dimensions(self, dimensions: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return dimensions / self.diagonal_length

    def to_normalized_coordinates(self, points: npt.NDArray[np.complex64]) -> npt.NDArray[np.float64]:
        points = np.column_stack((points.real, points.imag))
        return (points - self._virtual_screen_center) / self._virtual_screen_size

    def move_virtual_screen(self, normalised_movement: npt.NDArray[np.float64]):
        self._virtual_screen_center -= normalised_movement * self._virtual_screen_size

    def resize_virtual_screen(self, scroll_amount: float, normalised_position: npt.NDArray[np.float64]):
        scale = VELOCITY_OF_RESIZE_VIRTUAL_SCREEN ** (-scroll_amount)
        scale = max(scale, *(self._min_virtual_screen_size / self._virtual_screen_size))

        self._virtual_screen_size *= scale
        self.move_virtual_screen(normalised_position * (1 - 1 / scale))

    @property
    def size(self) -> npt.NDArray[np.float64]:
        return self._virtual_screen_size.copy()

    @property
    def diagonal_length(self) -> float:
        return float(np.hypot(self.size[0], self.size[1]))

    @property
    def left(self) -> float:
        return float((self._virtual_screen_center - self._virtual_screen_size / 2)[0])

    @property
    def right(self) -> float:
        return float((self._virtual_screen_center + self._virtual_screen_size / 2)[0])

    @property
    def bottom(self) -> float:
        return float((self._virtual_screen_center - self._virtual_screen_size / 2)[1])

    @property
    def top(self) -> float:
        return float((self._virtual_screen_center + self._virtual_screen_size / 2)[1])
