import cupy as cp
from config import TOLERANCE
from src.models import Curve
from src.models.portals import Portal
from ._p0_kernels import *


class P0Portal(Portal):
    NUMBER_OF_PARAMETERS_PER_ELEMENT = 2

    def __init__(self, first_curve: Curve, second_curve: Curve, is_inverted: bool, number_of_elements: int):
        super().__init__(first_curve, second_curve, is_inverted, number_of_elements)

        ts = cp.linspace(0.0, 1.0, self._number_of_elements + 1, dtype=cp.float32)

        first_vertices = self._first_curve.points(ts)
        self._first_starts = first_vertices[:-1]
        self._first_ends = first_vertices[1:]

        second_vertices = self._second_curve.points(ts)
        self._second_starts = second_vertices[:-1]
        self._second_ends = second_vertices[1:]

    def complex_potential_coefficients(self, points: cp.ndarray) -> cp.ndarray:
        complex_potential_coefficients = cp.zeros((points.size, self.number_of_parameters), dtype=cp.complex64)
        first_complex_potential_coefficients_kernel(
            points[:, cp.newaxis],
            self._first_starts[cp.newaxis, :], self._first_ends[cp.newaxis, :],
            self._second_starts[cp.newaxis, :], self._second_ends[cp.newaxis, :],
            self._is_inverted,
            TOLERANCE,
            complex_potential_coefficients[:, :self._number_of_elements]
        )
        second_complex_potential_coefficients_kernel(
            points[:, cp.newaxis],
            self._first_starts[cp.newaxis, :], self._first_ends[cp.newaxis, :],
            self._second_starts[cp.newaxis, :], self._second_ends[cp.newaxis, :],
            self._is_inverted,
            TOLERANCE,
            complex_potential_coefficients[:, self._number_of_elements:]
        )
        return complex_potential_coefficients

    def field_strength_coefficients(self, points: cp.ndarray) -> cp.ndarray:
        field_strength_coefficients = cp.zeros((points.size, self.number_of_parameters), dtype=cp.complex64)
        first_field_strength_coefficients_kernel(
            points[:, cp.newaxis],
            self._first_starts[cp.newaxis, :], self._first_ends[cp.newaxis, :],
            self._second_starts[cp.newaxis, :], self._second_ends[cp.newaxis, :],
            self._is_inverted,
            TOLERANCE,
            field_strength_coefficients[:, :self._number_of_elements]
        )
        second_field_strength_coefficients_kernel(
            points[:, cp.newaxis],
            self._first_starts[cp.newaxis, :], self._first_ends[cp.newaxis, :],
            self._second_starts[cp.newaxis, :], self._second_ends[cp.newaxis, :],
            self._is_inverted,
            TOLERANCE,
            field_strength_coefficients[:, self._number_of_elements:]
        )
        return field_strength_coefficients

    def complex_potentials(self, points: cp.ndarray) -> cp.ndarray:
        out = cp.zeros_like(points, dtype=cp.complex64)
        complex_potentials_kernel(
            points,
            self._first_starts, self._first_ends,
            self._second_starts, self._second_ends,
            self._is_inverted,
            self._parameters,
            TOLERANCE,
            out
        )
        return out

    def field_strengths(self, points: cp.ndarray) -> cp.ndarray:
        out = cp.zeros_like(points, dtype=cp.complex64)
        field_strengths_kernel(
            points,
            self._first_starts, self._first_ends,
            self._second_starts, self._second_ends,
            self._is_inverted,
            self._parameters,
            TOLERANCE,
            out
        )
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
