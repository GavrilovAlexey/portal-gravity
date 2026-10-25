from abc import ABC, abstractmethod
import cupy as cp


class Portal(ABC):
    __slots__ = ["_is_inverted", "_number_of_elements", "_number_of_parameters", "_parameters"]

    NUMBER_OF_PARAMETERS_PER_ELEMENT: int

    def __init__(self, is_inverted: bool, number_of_elements: int):
        self._is_inverted = cp.bool(is_inverted)

        self._number_of_elements = cp.int32(number_of_elements)
        self._number_of_parameters = self._number_of_elements * self.NUMBER_OF_PARAMETERS_PER_ELEMENT
        self._parameters = cp.zeros(self._number_of_parameters)

    @abstractmethod
    def complex_potential_coefficients(self, points: cp.ndarray) -> cp.ndarray:
        pass

    def potential_differences_coefficients(self, first_points: cp.ndarray, second_points: cp.ndarray) -> cp.ndarray:
        return (self.complex_potential_coefficients(first_points).real -
                self.complex_potential_coefficients(second_points).real)

    @abstractmethod
    def field_strength_coefficients(self, points: cp.ndarray) -> cp.ndarray:
        pass

    def flux_differences_coefficients(
            self,
            first_points: cp.ndarray, second_points: cp.ndarray,
            first_normals: cp.ndarray, second_normals: cp.ndarray
    ) -> cp.ndarray:
        first = (self.field_strength_coefficients(first_points) * cp.conj(first_normals)[:, cp.newaxis]).real
        second = (self.field_strength_coefficients(second_points) * cp.conj(second_normals)[:, cp.newaxis]).real
        return first - second

    @abstractmethod
    def complex_potentials(self, points: cp.ndarray) -> cp.ndarray:
        pass

    @abstractmethod
    def field_strengths(self, points: cp.ndarray) -> cp.ndarray:
        pass

    @property
    @abstractmethod
    def first_points(self) -> cp.ndarray:
        pass

    @property
    @abstractmethod
    def second_points(self) -> cp.ndarray:
        pass

    @property
    @abstractmethod
    def first_normals(self) -> cp.ndarray:
        pass

    @property
    @abstractmethod
    def second_normals(self) -> cp.ndarray:
        pass

    @property
    def number_of_parameters(self) -> cp.int32:
        return self._number_of_parameters

    @property
    def parameters(self) -> cp.ndarray:
        return self._parameters

    @parameters.setter
    def parameters(self, value: cp.ndarray):
        self._parameters = value

    @property
    @abstractmethod
    def first_vertices(self) -> cp.ndarray:
        pass

    @property
    @abstractmethod
    def second_vertices(self) -> cp.ndarray:
        pass
