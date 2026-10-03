import cupy as cp
import cupy.typing as cpt
from .heatmap_quantity import HeatmapQuantity


class MagnitudeOfField(HeatmapQuantity):
    def __call__(self, grid: cpt.NDArray[cp.complex64]) -> cpt.NDArray[cp.float32]:
        return cp.abs(self.engine.field_strengths(grid))
