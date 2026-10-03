import cupy as cp
from config import G
from .physical_object import PhysicalObject


class PointParticle(PhysicalObject):
    def __init__(self, position: complex, mass: float):
        self._position = cp.complex64(position)
        self._mass = cp.float32(mass)

    def complex_potentials(self, points: cp.ndarray) -> cp.ndarray:
        return G * self._mass * cp.log(self._position - points)

    def field_strengths(self, points: cp.ndarray) -> cp.ndarray:
        return G * self._mass / cp.conj(self._position - points)
