from typing import TYPE_CHECKING
import cupy as cp

if TYPE_CHECKING:
    from src.models import Engine


class Solver:
    __slots__ = ["_engine"]

    def __init__(self, engine: "Engine"):
        self._engine = engine

    def solve(self):
        if not self._engine.portals:
            return

        first_points = cp.concatenate([portal.first_points for portal in self._engine.portals])
        second_points = cp.concatenate([portal.second_points for portal in self._engine.portals])
        first_normals = cp.concatenate([portal.first_normals for portal in self._engine.portals])
        second_normals = cp.concatenate([portal.second_normals for portal in self._engine.portals])

        potential_differences_coefficients = cp.hstack([
            portal.potential_differences_coefficients(first_points, second_points) for portal in self._engine.portals
        ])
        potential_differences = -(self._engine.non_portal_complex_potentials(first_points).real -
                                  self._engine.non_portal_complex_potentials(second_points).real)

        flux_differences_coefficients = cp.hstack([
            portal.flux_differences_coefficients(
                first_points, second_points, first_normals, second_normals) for portal in self._engine.portals
        ])
        flux_differences = -((self._engine.non_portal_field_strengths(first_points) * cp.conj(first_normals)).real -
                             (self._engine.non_portal_field_strengths(second_points) * cp.conj(second_normals)).real)

        coefficient_matrix = cp.vstack((potential_differences_coefficients, flux_differences_coefficients))
        rhs_vector = cp.hstack((potential_differences, flux_differences))
        parameters = cp.linalg.solve(coefficient_matrix, rhs_vector)

        index = 0
        for portal in self._engine.portals:
            portal.parameters = parameters[index:index + portal.number_of_parameters]
            index += portal.number_of_parameters
