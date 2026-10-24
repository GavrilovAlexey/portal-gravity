import cupy as cp
from .heatmap_quantity import HeatmapQuantity


class Potential(HeatmapQuantity):
    def calculate(self, grid: cp.ndarray) -> cp.ndarray:
        return self.engine.complex_potentials(grid).real
