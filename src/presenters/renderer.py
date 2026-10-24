from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.presenters import AppPresenter

import pygame
from src.models import PointParticle


class Renderer:
    def __init__(self, app_presenter: "AppPresenter"):
        self.app_presenter = app_presenter

        self.engine = app_presenter.engine
        self.main_window = app_presenter.main_window

        self.virtual_screen = app_presenter.virtual_screen
        self.heatmap = app_presenter.heatmap

    def render(self):
        self.render_heatmap()
        self.render_physical_objects()
        self.render_portals()

        pygame.display.flip()

    def render_heatmap(self):
        self.main_window.display_heatmap_values(self.heatmap.values)

    def render_physical_objects(self):
        for physical_object in self.engine.physical_objects:
            if isinstance(physical_object, PointParticle):
                point = physical_object.position.get()
                normalized_point = self.virtual_screen.to_normalized_coordinates(point)[0]
                self.main_window.display_point(
                    normalized_point, self.virtual_screen.to_normalized_dimension(0.05), (128, 128, 128)
                )

    def render_portals(self):
        for portal in self.engine.portals:
            vertices = portal.first_vertices.get()
            normalized_vertices = self.virtual_screen.to_normalized_coordinates(vertices)
            self.main_window.display_curve(
                normalized_vertices, self.virtual_screen.to_normalized_dimension(0.01), (255, 154, 0)
            )

            vertices = portal.second_vertices.get()
            normalized_vertices = self.virtual_screen.to_normalized_coordinates(vertices)
            self.main_window.display_curve(
                normalized_vertices, self.virtual_screen.to_normalized_dimension(0.01), (0, 101, 255)
            )
