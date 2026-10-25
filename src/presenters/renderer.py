from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.presenters import AppPresenter

import pygame
from src.models import PointParticle


class Renderer:
    __slots__ = ["_app_presenter"]

    def __init__(self, app_presenter: "AppPresenter"):
        self._app_presenter = app_presenter

    def __getattr__(self, item):
        return getattr(self._app_presenter, item)

    def render(self):
        self.render_heatmap()
        self.render_physical_objects()
        self.render_portals()

        pygame.display.flip()

    def render_heatmap(self):
        self._main_window.display_heatmap_values(self._heatmap.values)

    def render_physical_objects(self):
        for physical_object in self._engine.physical_objects:
            if isinstance(physical_object, PointParticle):
                point = physical_object.position.get()
                normalized_point = self._virtual_screen.to_normalized_coordinates(point)[0]
                self._main_window.display_point(
                    normalized_point, self._virtual_screen.to_normalized_dimension(0.05), (128, 128, 128)
                )

    def render_portals(self):
        for portal in self._engine.portals:
            vertices = portal.first_vertices.get()
            normalized_vertices = self._virtual_screen.to_normalized_coordinates(vertices)
            self._main_window.display_curve(
                normalized_vertices, self._virtual_screen.to_normalized_dimension(0.01), (255, 154, 0)
            )

            vertices = portal.second_vertices.get()
            normalized_vertices = self._virtual_screen.to_normalized_coordinates(vertices)
            self._main_window.display_curve(
                normalized_vertices, self._virtual_screen.to_normalized_dimension(0.01), (0, 101, 255)
            )
