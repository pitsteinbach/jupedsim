import numpy as np
from jupedsim.agent import Agent
from jupedsim.geometry import LineSegment
from jupedsim.models.custom_model import (
    CustomModelAgentUpdate,
    CustomOperationalModel,
)

from jupedsim.models.collision_free_speed import CollisionFreeSpeedModel
from jupedsim.point import Point
from shapely.geometry import LineString
import sys

flt_max = sys.float_info.max


class PythonColisionFreeSpeedModel(CustomOperationalModel):
    def __init__(self, model=CollisionFreeSpeedModel()):
        CustomOperationalModel.__init__(self)
        self.strength_neighbor_repulsion = model.strength_neighbor_repulsion
        self.range_neighbor_repulsion = model.range_neighbor_repulsion
        self.strength_geometry_repulsion = model.strength_geometry_repulsion
        self.range_geometry_repulsion = model.range_geometry_repulsion
        self.cutoff_radius = 3.0

    def remove_non_visible_neighbors(self, ped, neighbors, boundary):
        vis_neighbors = []
        for neigh in neighbors:
            if neigh.id == ped.id:
                continue
            ls_ag_neigh = LineString([ped.position, neigh.position])
            inter = False
            for boundary_ls in boundary:
                if boundary_ls.intersects(ls_ag_neigh):
                    inter = True
                    break
            if not inter:
                vis_neighbors.append(neigh)
        return vis_neighbors

    def neighbor_repulsion(self, ped1, ped2) -> Point:

        d12 = Point(ped2.position) - Point(ped1.position)
        distance = d12.norm()
        n_direction = d12.normalize()
        r1 = getattr(ped1.model, "radius", 0.3)
        r2 = getattr(ped2.model, "radius", 0.3)
        l = r1 + r2
        return Point(
            n_direction
            * -(
                self.strength_neighbor_repulsion
                * np.exp(l - distance)
                / self.range_neighbor_repulsion
            )
        )

    def boundary_repulsion(self, ped, element) -> Point:
        pt = Point(element.closest_point(ped.position))
        dist_vec = pt - ped.position
        distance = dist_vec.norm()
        n_direction = dist_vec.normalize()
        l = getattr(ped.model, "radius", 0.3)
        return (
            n_direction
            * self.strength_geometry_repulsion
            * np.exp((l - distance) / self.range_geometry_repulsion)
        )

    def get_spacing(self, ped1, ped2, direction: Point):

        d12 = Point(ped1.position) - Point(ped2.position)
        in_front = direction.scalar_product(d12)
        if in_front < 0.0:
            return flt_max
        left = direction.rotate_90_deg()
        r1 = getattr(ped1.model, "radius", 0.3)
        r2 = getattr(ped2.model, "radius", 0.3)
        l = r1 + r2
        in_corridor = np.abs(left.scalar_product(d12))
        if in_corridor > l:
            return flt_max
        return d12.norm() - l

    def optimal_speed(self, ped, spacing, time_gap):
        return np.min([np.max([spacing / time_gap, 0]), ped.model.v0])

    def compute_new_position(
        self, dt: float, ped, geometry, neighborhood_search
    ):
        """
        Compute new position with CFSM
        """
        pos = Point(ped.position)
        neighbors = neighborhood_search.get_neighboring_agents(
            pos, self.cutoff_radius
        )
        boundary = geometry.linesegments_close_to(pos)
        vis_neighbors = self.remove_non_visible_neighbors(
            ped, neighbors, boundary
        )
        neigh_repulsion = Point(0, 0)

        for neigh in vis_neighbors:
            neigh_repulsion = neigh_repulsion + self.neighbor_repulsion(
                ped, neigh
            )

        bound_repulsion = Point(0, 0)
        for element in boundary:
            bound_repulsion += self.boundary_repulsion(ped, element)

        desired_direction = (Point(ped.destination) - pos).normalize()
        direction = (
            desired_direction + neigh_repulsion + bound_repulsion
        ).normalize()

        spacing = flt_max
        for neigh in vis_neighbors:
            spacing += self.get_spacing(ped, neigh, direction)
        print(direction)
        optimal_speed = self.optimal_speed(ped, spacing, ped.model.time_gap)
        print(optimal_speed)
        velocity = direction * optimal_speed
        print(velocity)
        update = CustomModelAgentUpdate()
        update.position = pos + velocity * dt
        update.orientation = direction
        return update

    def check_model_constraint(self, ped, neighborhoodsearch, geometry):
        pass
