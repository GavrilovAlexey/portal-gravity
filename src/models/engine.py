import cupy as cp
from config import INITIAL_SUBSTEPS_PER_FRAME, INITIAL_SUBSTEP_DT
from .physical_objects import PhysicalObject
from .portals import Portal
from .solver import Solver


class Engine:
    __slots__ = ["_solver", "_physical_objects", "_portals", "_external_field", "_substeps_per_frame", "_substep_dt"]

    def __init__(self):
        self._solver = Solver(self)

        self._physical_objects: list[PhysicalObject] = []
        self._portals: list[Portal] = []

        self._external_field = cp.complex64(0)

        self._substeps_per_frame = INITIAL_SUBSTEPS_PER_FRAME
        self._substep_dt = INITIAL_SUBSTEP_DT

    def __getattr__(self, item):
        return getattr(self._solver, item)

    def add_physical_object(self, physical_object: PhysicalObject):
        self._physical_objects.append(physical_object)

    def add_portal(self, portal: Portal):
        self._portals.append(portal)

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

    def update(self):
        for _ in range(self._substeps_per_frame):
            updated_physical_objects = [physical_object.updated_object(self.field_strengths, self._substep_dt) for
                                        physical_object in self._physical_objects]
            self._physical_objects = updated_physical_objects
            self.solve()

    @property
    def external_field(self) -> cp.complex64:
        return self._external_field

    @external_field.setter
    def external_field(self, value: complex):
        self._external_field = cp.complex64(value)

    @property
    def physical_objects(self) -> list[PhysicalObject]:
        return self._physical_objects.copy()

    @property
    def portals(self) -> list[Portal]:
        return self._portals.copy()
