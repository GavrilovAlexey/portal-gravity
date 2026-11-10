from typing import TYPE_CHECKING
import pygame

from src.models import PointParticle

if TYPE_CHECKING:
    from src.presenters import AppPresenter


class Renderer:
    __slots__ = ["_app_presenter"]

    def __init__(self, app_presenter: "AppPresenter"):
        self._app_presenter = app_presenter

    def render(self):
        self._render_heatmap()
        self._render_physical_objects()
        self._render_portals()

        pygame.display.flip()

    def _render_heatmap(self):
        self._app_presenter.main_window.displayer.display_heatmap_values(self._app_presenter.heatmap.values)

    def _render_physical_objects(self):
        for physical_object in self._app_presenter.engine.physical_objects:
            if isinstance(physical_object, PointParticle):
                point = physical_object.position.get()
                normalized_point = self._app_presenter.virtual_screen.to_normalized_coordinates(point)[0]
                self._app_presenter.main_window.displayer.display_point(
                    normalized_point, self._app_presenter.virtual_screen.to_normalized_dimension(0.05), (128, 128, 128)
                )

    def _render_portals(self):
        for portal in self._app_presenter.engine.portals:
            vertices = portal.first_vertices.get()
            normalized_vertices = self._app_presenter.virtual_screen.to_normalized_coordinates(vertices)
            self._app_presenter.main_window.displayer.display_curve(
                normalized_vertices, self._app_presenter.virtual_screen.to_normalized_dimension(0.01), (255, 154, 0)
            )

            vertices = portal.second_vertices.get()
            normalized_vertices = self._app_presenter.virtual_screen.to_normalized_coordinates(vertices)
            self._app_presenter.main_window.displayer.display_curve(
                normalized_vertices, self._app_presenter.virtual_screen.to_normalized_dimension(0.01), (0, 101, 255)
            )
