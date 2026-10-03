import cupy as cp
import cupy.typing as cpt
from .heatmap_quantity import HeatmapQuantity


class StreamFunction(HeatmapQuantity):
    def __call__(self, grid: cpt.NDArray[cp.complex64]) -> cpt.NDArray[cp.float32]:
        return self.engine.complex_potentials(grid).imag
