# SPDX-License-Identifier: LGPL-3.0-or-later

from typing import Any

import shapely

import jupedsim.native as py_jps
from jupedsim.geometry_utils import build_geometry

_GridData = dict  # {"width": int, "height": int, "origin": tuple, "cell_size": float, "data": list[float]}


class RoutingEngine:
    """RoutingEngine to compute the shortest paths with navigation meshes."""

    def __init__(
        self,
        geometry: (
            str
            | shapely.GeometryCollection
            | shapely.Polygon
            | shapely.MultiPolygon
            | shapely.MultiPoint
            | list[tuple[float, float]]
        ),
        **kwargs: Any,
    ) -> None:
        self._obj = py_jps.RoutingEngine(
            build_geometry(geometry, **kwargs)._obj
        )

    def compute_waypoints(
        self, frm: tuple[float, float], to: tuple[float, float]
    ) -> list[tuple[float, float]]:
        """Computes shortest path between specified points.

        Arguments:
            geometry: Data to create the geometry out of. Data may be supplied as:

                * list of 2d points describing the outer boundary, holes may be added with use of `excluded_areas` kw-argument

                * :class:`~shapely.GeometryCollection` consisting only out of :class:`Polygons <shapely.Polygon>`, :class:`MultiPolygons <shapely.MultiPolygon>` and :class:`MultiPoints <shapely.MultiPoint>`

                * :class:`~shapely.MultiPolygon`

                * :class:`~shapely.Polygon`

                * :class:`~shapely.MultiPoint` forming a "simple" polygon when points are interpreted as linear ring without repetition of the start/end point.

                * str with a valid Well Known Text. In this format the same WKT types as mentioned for the shapely types are supported: GEOMETRYCOLLETION, MULTIPOLYGON, POLYGON, MULTIPOINT. The same restrictions as mentioned for the shapely types apply.

            frm: point from which to find the shortest path
            to: point to which to find the shortest path

        Keyword Arguments:
            excluded_areas: describes exclusions
                from the walkable area. Only use this argument if `geometry` was
                provided as list[tuple[float, float]].

        Returns:
            List of points (path) from 'frm' to 'to' including from and to.

        """
        return self._obj.compute_waypoints(frm, to)

    def is_routable(self, p: tuple[float, float]) -> bool:
        """Tests if the supplied point is inside the underlying geometry.

        Returns:
            If the point is inside the geometry.

        """
        return self._obj.is_routable(p)

    def mesh(
        self,
    ) -> tuple[list[tuple[float, float]], list[list[int]]]:
        """Access the navigation mesh geometry.

        The navigation mesh is store as a collection of convex polygons in CCW order.

        The returned data is to be interpreted as:

        .. code::

            tuple[
                list[tuple[float, float]], # All vertices in this mesh.
                list[ # List of polygons
                    list[int] # List of indices into the vertices that compose this polygon in CCW order
                ]
            ]

        Returns:
            A tuple of vertices and list of polygons which in turn are a list of indices
            tuple[list[tuple[float, float]],list[list[int]]]
        """
        return self._obj.mesh()

    def edges_for(self, vertex_id: int):
        return self._obj.edges_for(vertex_id)


class Floorfield:
    """Floor-field router that solves the eikonal equation on a regular grid.

    The walkable domain is rasterised from the supplied geometry at construction
    time.  Travel times are computed on demand for each queried destination and
    cached so that repeated queries for the same destination are cheap.
    """

    def __init__(
        self,
        geometry: (
            str
            | shapely.GeometryCollection
            | shapely.Polygon
            | shapely.MultiPolygon
            | shapely.MultiPoint
            | list[tuple[float, float]]
        ),
        *,
        wall_influence_radius: float = 0.5,
        **kwargs: Any,
    ) -> None:
        """Create a floor-field router for the given geometry.

        Arguments:
            geometry: walkable area — same formats as :class:`RoutingEngine`.
            wall_influence_radius: cells within this distance (metres) of any
                boundary edge receive a proportionally reduced speed, so the
                eikonal naturally routes paths away from walls.  Default 0.5 m.

        Keyword Arguments:
            excluded_areas: holes subtracted from the walkable area (only when
                *geometry* is a ``list[tuple[float, float]]``).
        """
        self._obj = py_jps.Floorfield(
            build_geometry(geometry, **kwargs)._obj, wall_influence_radius
        )

    def compute_waypoints(
        self, frm: tuple[float, float], to: tuple[float, float]
    ) -> list[tuple[float, float]]:
        """Compute the path from *frm* to *to* via gradient descent on the floor field.

        The floor field for *to* is computed on the first call and cached.

        Arguments:
            frm: start point ``(x, y)``
            to:  destination point ``(x, y)``

        Returns:
            Ordered list of waypoints from *frm* to *to*, inclusive.
        """
        return self._obj.compute_waypoints(frm, to)

    def is_routable(self, p: tuple[float, float]) -> bool:
        """Return ``True`` if *p* lies inside the walkable domain."""
        return self._obj.is_routable(p)

    def speed_field(self) -> _GridData:
        """Return the walkability grid built from the geometry.

        The returned dict has the keys:

        * ``"width"``     – number of columns
        * ``"height"``    – number of rows
        * ``"origin"``    – ``(x, y)`` of the bottom-left corner of cell (0, 0)
        * ``"cell_size"`` – physical size of one cell in metres
        * ``"data"``      – flat row-major list; ``1.0`` = walkable, ``0.0`` = obstacle

        Reconstruct a 2-D array and an imshow extent with::

            import numpy as np
            d = ff.speed_field()
            grid = np.array(d["data"]).reshape(d["height"], d["width"])
            ox, oy = d["origin"]
            cs = d["cell_size"]
            extent = [ox, ox + d["width"] * cs, oy, oy + d["height"] * cs]
        """
        return self._obj.speed_field()

    def travel_times(self) -> _GridData:
        """Return the travel-time grid for the most recently queried destination.

        Has the same layout as :meth:`speed_field`.  Call
        :meth:`compute_waypoints` at least once before using this so the
        eikonal solution is populated.  Cells unreachable from the destination
        carry ``float("inf")``.
        """
        return self._obj.travel_times()

    def update_density(self, positions: list[tuple[float, float]]) -> None:
        """Update the density field from agent positions.

        Rebuilds the dynamic speed field every :meth:`set_recompute_interval`
        calls.  Has no effect until the rebuild threshold is reached.

        Arguments:
            positions: list of ``(x, y)`` agent positions.
        """
        self._obj.update_density(positions)

    def density_field(self) -> _GridData:
        """Return raw agent density (agents/m²) per cell.

        Has the same layout as :meth:`speed_field`.  Call
        :meth:`update_density` first to populate.
        """
        return self._obj.density_field()

    def dynamic_speed_field(self) -> _GridData:
        """Return the density-modulated speed field (0–1) per cell.

        Has the same layout as :meth:`speed_field`.  Values drop below the
        static :meth:`speed_field` in high-density cells.  Call
        :meth:`update_density` first to populate.
        """
        return self._obj.dynamic_speed_field()

    def set_recompute_interval(self, steps: int) -> None:
        """Set how many :meth:`update_density` calls trigger a full rebuild.

        A value of ``1`` rebuilds on every call; higher values amortise the
        cost across multiple frames at the expense of staleness.

        Arguments:
            steps: number of calls between rebuilds (default 200).
        """
        self._obj.set_recompute_interval(steps)
