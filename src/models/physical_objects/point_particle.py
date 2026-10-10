from typing import Callable, Self
import cupy as cp
from config import G
from .physical_object import PhysicalObject


class PointParticle(PhysicalObject):
    def __init__(self, position: complex | cp.ndarray, velocity: complex | cp.ndarray, mass: float | cp.ndarray):
        self._position = cp.asarray(position, dtype=cp.complex64)
        self._velocity = cp.asarray(velocity, dtype=cp.complex64)

        self._mass = cp.asarray(mass, dtype=cp.float32)

    def complex_potentials(self, points: cp.ndarray) -> cp.ndarray:
        return G * self._mass * cp.log(self._position - points)

    def field_strengths(self, points: cp.ndarray) -> cp.ndarray:
        return cp.where(cp.equal(points, self._position), cp.zeros_like(points),
                        G * self._mass / cp.conj(self._position - points))

    def updated_object(self, field_strengths: Callable[[cp.ndarray], cp.ndarray], dt: float) -> Self:
        acceleration = field_strengths(cp.array([self._position]))[0]
        updated_position = self._position + self._velocity * dt + acceleration * dt ** 2 / 2
        updated_velocity = self._velocity + acceleration * dt
        return PointParticle(updated_position, updated_velocity, self._mass)
