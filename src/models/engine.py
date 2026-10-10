import cupy as cp
from .physical_objects import PhysicalObject
from .portals import Portal


class Engine:
    def __init__(self):
        self._physical_objects: list[PhysicalObject] = []

        self._portals: list[Portal] = []
        self._splitting_indexes = cp.array([0], dtype=cp.int32)

        self._external_field = cp.complex64(0)

    def add_physical_object(self, physical_object: PhysicalObject):
        self._physical_objects.append(physical_object)

    def add_portal(self, portal: Portal):
        self._portals.append(portal)
        self._splitting_indexes = cp.append(self._splitting_indexes,
                                            self._splitting_indexes[-1] + portal.number_of_parameters)

    def solve(self):
        if not self._portals:
            return

        first_points = cp.concatenate([portal.first_points for portal in self._portals])
        second_points = cp.concatenate([portal.second_points for portal in self._portals])
        first_normals = cp.concatenate([portal.first_normals for portal in self._portals])
        second_normals = cp.concatenate([portal.second_normals for portal in self._portals])

        potential_differences_coefficients = cp.hstack([
            portal.potential_differences_coefficients(first_points, second_points) for portal in self._portals
        ])
        potential_differences = -(self.non_portal_complex_potentials(first_points).real -
                                  self.non_portal_complex_potentials(second_points).real)

        flux_differences_coefficients = cp.hstack([
            portal.flux_differences_coefficients(
                first_points, second_points, first_normals, second_normals) for portal in self._portals
        ])
        flux_differences = -((self.non_portal_field_strengths(first_points) * cp.conj(first_normals)).real -
                             (self.non_portal_field_strengths(second_points) * cp.conj(second_normals)).real)

        coefficient_matrix = cp.vstack((potential_differences_coefficients, flux_differences_coefficients))
        rhs_vector = cp.hstack((potential_differences, flux_differences))
        parameters = cp.linalg.solve(coefficient_matrix, rhs_vector)

        parameters = cp.split(parameters, self._splitting_indexes[1:-1].tolist())
        for portal, portal_parameters in zip(self._portals, parameters):
            portal.parameters = portal_parameters
        cp.get_default_memory_pool().free_all_blocks()

    def non_portal_complex_potentials(self, points: cp.ndarray) -> cp.ndarray:
        non_portal_complex_potentials = -cp.conj(self._external_field) * points
        for physical_object in self._physical_objects:
            non_portal_complex_potentials += physical_object.complex_potentials(points)
        return non_portal_complex_potentials

    def portal_complex_potentials(self, points: cp.ndarray) -> cp.ndarray:
        portal_complex_potentials = cp.zeros_like(points, dtype=cp.complex64)
        for portal in self._portals:
            portal_complex_potentials += portal.complex_potentials(points)
        return portal_complex_potentials

    def complex_potentials(self, points: cp.ndarray) -> cp.ndarray:
        return self.non_portal_complex_potentials(points) + self.portal_complex_potentials(points)

    def non_portal_field_strengths(self, points: cp.ndarray) -> cp.ndarray:
        non_portal_field_strengths = cp.full(points.shape, self._external_field, cp.complex64)
        for physical_object in self._physical_objects:
            non_portal_field_strengths += physical_object.field_strengths(points)
        return non_portal_field_strengths

    def portal_field_strengths(self, points: cp.ndarray) -> cp.ndarray:
        portal_field_strengths = cp.zeros_like(points, dtype=cp.complex64)
        for portal in self._portals:
            portal_field_strengths += portal.field_strengths(points)
        return portal_field_strengths

    def field_strengths(self, points: cp.ndarray) -> cp.ndarray:
        return self.non_portal_field_strengths(points) + self.portal_field_strengths(points)

    def update(self, number_of_substeps: int, substep_dt: float):
        for _ in range(number_of_substeps):
            updated_physical_objects = [physical_object.updated_object(self.field_strengths, substep_dt) for
                                        physical_object in self._physical_objects]
            self._physical_objects = updated_physical_objects
            self.solve()

    @property
    def external_field(self) -> cp.complex64:
        return self._external_field

    @external_field.setter
    def external_field(self, value: complex):
        self._external_field = cp.complex64(value)
