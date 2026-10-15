from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.presenters import AppPresenter

import pygame


class Renderer:
    def __init__(self, app_presenter: "AppPresenter"):
        self.app_presenter = app_presenter

    def render(self):
        self.render_heatmap()
        self.render_physical_objects()
        self.render_portals()

        pygame.display.flip()

    def render_heatmap(self):
        self.app_presenter.main_window.display_heatmap_values(self.app_presenter.heatmap.values)

    def render_physical_objects(self):
        pass

    def render_portals(self):
        for portal in self.app_presenter.engine.portals:
            vertices = portal.first_curve.vertices.get()
            normalized_vertices = self.app_presenter.virtual_screen.to_normalized_coordinate(vertices)
            self.app_presenter.main_window.display_curve(normalized_vertices, (255, 154, 0))

            vertices = portal.second_curve.vertices.get()
            normalized_vertices = self.app_presenter.virtual_screen.to_normalized_coordinate(vertices)
            self.app_presenter.main_window.display_curve(normalized_vertices, (0, 101, 255))
