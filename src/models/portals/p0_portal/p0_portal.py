import cupy as cp

from config import TOLERANCE
from src.models.portals import Portal

from ._p0_kernels import *


class P0Portal(Portal):
    __slots__ = []

    NUMBER_OF_PARAMETERS_PER_ELEMENT = 2

    def complex_potential_coefficients(self, points: cp.ndarray) -> cp.ndarray:
        out = cp.zeros((points.size, self.number_of_parameters), dtype=cp.complex64)

        first_complex_potential_coefficients_kernel(
            points[:, cp.newaxis], *self._kernel_coefficient_arguments, out[:, :self._number_of_elements]
        )
        second_complex_potential_coefficients_kernel(
            points[:, cp.newaxis], *self._kernel_coefficient_arguments, out[:, self._number_of_elements:]
        )
        return out

    def field_strength_coefficients(self, points: cp.ndarray) -> cp.ndarray:
        out = cp.zeros((points.size, self.number_of_parameters), dtype=cp.complex64)

        first_field_strength_coefficients_kernel(
            points[:, cp.newaxis], *self._kernel_coefficient_arguments, out[:, :self._number_of_elements]
        )
        second_field_strength_coefficients_kernel(
            points[:, cp.newaxis], *self._kernel_coefficient_arguments, out[:, self._number_of_elements:]
        )
        return out

    def complex_potentials(self, points: cp.ndarray) -> cp.ndarray:
        out = cp.zeros_like(points, dtype=cp.complex64)
        complex_potentials_kernel(points, *self._kernel_quantity_arguments, out)
        return out

    def field_strengths(self, points: cp.ndarray) -> cp.ndarray:
        out = cp.zeros_like(points, dtype=cp.complex64)
        field_strengths_kernel(points, *self._kernel_quantity_arguments, out)
        return out

    @property
    def first_points(self) -> cp.ndarray:
        return (self._first_ends + self._first_starts) / 2

    @property
    def second_points(self) -> cp.ndarray:
        return (self._second_ends + self._second_starts) / 2

    @property
    def first_normals(self) -> cp.ndarray:
        return (self._first_ends - self._first_starts) * 1j

    @property
    def second_normals(self) -> cp.ndarray:
        return (self._second_ends - self._second_starts) * 1j * (-1 if self._is_inverted else 1)

    @property
    def _kernel_quantity_arguments(self) -> tuple:
        return (
            self._first_starts, self._first_ends,
            self._second_starts, self._second_ends,
            self._is_inverted,
            self._parameters,
            TOLERANCE
        )

    @property
    def _kernel_coefficient_arguments(self) -> tuple:
        return (
            self._first_starts[cp.newaxis, :], self._first_ends[cp.newaxis, :],
            self._second_starts[cp.newaxis, :], self._second_ends[cp.newaxis, :],
            self._is_inverted,
            TOLERANCE,
        )
