# SPDX-License-Identifier: LGPL-3.0-or-later
import math

import jupedsim as jps
from jupedsim.internal.aabb import AABB
from vtkmodules.vtkCommonCore import vtkFloatArray, vtkLookupTable
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkRenderingAnnotation import vtkScalarBarActor
from vtkmodules.vtkRenderingCore import vtkActor, vtkDataSetMapper

from jupedsim_visualizer.config import ZLayers

_MODES = ("speed", "density", "dynamic_speed", "travel_time")


class FloorFieldViz:
    """VTK visualization of Floorfield fields for a given geometry.

    Modes
    -----
    "speed"         Static walkability speed (0–1). Default on show().
    "density"       Agent density (agents/m²) after update_density().
    "dynamic_speed" Density-modulated speed (0–1) after update_density().
    "travel_time"   Travel-time isochrones after set_destination().

    Usage (geometry viewer):
        viz = FloorFieldViz(shapely_geometry)
        renderer.AddActor(viz.get_actor())
        renderer.AddActor2D(viz.get_scalar_bar())
        viz.show(True)
        viz.set_destination(x, y)

    Usage (replay):
        viz = FloorFieldViz(shapely_geometry, mode="density")
        viz.set_recompute_interval(10)   # rebuild every 10 update_density() calls
        viz.show(True)
        # on each frame:
        viz.update_density([(x0,y0), (x1,y1), ...])
    """

    def __init__(self, geometry, mode: str = "speed") -> None:
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}")
        self._geometry = geometry
        self._mode = mode
        self._destination: tuple[float, float] | None = None
        self._ff: jps.Floorfield | None = None
        self._image: vtkImageData | None = None
        self._scalars: vtkFloatArray | None = None
        self._mapper: vtkDataSetMapper | None = None

        self._lut = vtkLookupTable()
        self._lut.SetHueRange(0.667, 0.0)  # blue (near) → red (far)
        self._lut.SetSaturationRange(1.0, 1.0)
        self._lut.SetValueRange(1.0, 1.0)
        self._lut.SetNanColor(
            0.25, 0.25, 0.25, 0.0
        )  # transparent for unreachable
        self._lut.SetTableRange(0.0, 1.0)
        self._lut.Build()

        self._actor = vtkActor()
        self._actor.SetVisibility(False)

        self._scalar_bar = vtkScalarBarActor()
        self._scalar_bar.SetLookupTable(self._lut)
        self._scalar_bar.SetTitle(self._bar_title())
        self._scalar_bar.SetNumberOfLabels(5)
        self._scalar_bar.SetPosition(0.87, 0.10)
        self._scalar_bar.SetWidth(0.10)
        self._scalar_bar.SetHeight(0.70)
        self._scalar_bar.SetVisibility(False)

    # ── internal ────────────────────────────────────────────────────────────

    def _bar_title(self) -> str:
        return {
            "speed": "Speed",
            "density": "Density (agents/m²)",
            "dynamic_speed": "Dynamic speed",
            "travel_time": "Travel time (s)",
        }[self._mode]

    def _ensure_floorfield(self) -> None:
        if self._ff is not None:
            return
        self._ff = jps.Floorfield(self._geometry, wall_influence_radius=0.5)
        sf = self._ff.speed_field()
        width: int = sf["width"]
        height: int = sf["height"]
        ox, oy = sf["origin"]
        cs: float = sf["cell_size"]

        self._image = vtkImageData()
        self._image.SetDimensions(width, height, 1)
        self._image.SetOrigin(ox, oy, ZLayers.floorfield)
        self._image.SetSpacing(cs, cs, 1.0)

        self._scalars = vtkFloatArray()
        self._scalars.SetNumberOfTuples(width * height)
        for i, v in enumerate(sf["data"]):
            self._scalars.SetValue(i, v)
        self._image.GetPointData().SetScalars(self._scalars)

        self._mapper = vtkDataSetMapper()
        self._mapper.SetInputData(self._image)
        self._mapper.SetLookupTable(self._lut)
        self._mapper.SetScalarRange(0.0, 1.0)
        self._mapper.SetUseLookupTableScalarRange(True)

        self._actor.SetMapper(self._mapper)
        self._actor.GetProperty().SetOpacity(0.75)

        self._refresh_field()

    def _refresh_field(self) -> None:
        """Redraw scalars from the currently active mode."""
        assert self._ff is not None and self._scalars is not None
        if self._mode == "speed":
            self._apply_speed_field(self._ff.speed_field())
        elif self._mode == "density":
            self._apply_density_field(self._ff.density_field())
        elif self._mode == "dynamic_speed":
            self._apply_dynamic_speed_field(self._ff.dynamic_speed_field())
        # travel_time is handled separately by set_destination()

    def _apply_speed_field(self, sf: dict) -> None:
        assert self._scalars is not None and self._mapper is not None
        for i, v in enumerate(sf["data"]):
            self._scalars.SetValue(i, float("nan") if v == 0.0 else v)
        self._scalars.Modified()
        self._lut.SetHueRange(0.667, 0.0)
        self._lut.SetTableRange(0.0, 1.0)
        self._lut.Build()
        self._mapper.SetScalarRange(0.0, 1.0)
        self._image.Modified()

    def _apply_density_field(self, df: dict) -> None:
        assert self._scalars is not None and self._mapper is not None
        data: list[float] = df["data"]
        finite_vals = [v for v in data if v > 0.0]
        vmax = max(finite_vals) if finite_vals else 1.0
        for i, v in enumerate(data):
            self._scalars.SetValue(i, float("nan") if v == 0.0 else v)
        self._scalars.Modified()
        # green (sparse) → red (crowded)
        self._lut.SetHueRange(0.333, 0.0)
        self._lut.SetTableRange(0.0, vmax)
        self._lut.Build()
        self._mapper.SetScalarRange(0.0, vmax)
        self._image.Modified()

    def _apply_dynamic_speed_field(self, dsf: dict) -> None:
        assert self._scalars is not None and self._mapper is not None
        for i, v in enumerate(dsf["data"]):
            self._scalars.SetValue(i, float("nan") if v == 0.0 else v)
        self._scalars.Modified()
        self._lut.SetHueRange(0.667, 0.0)
        self._lut.SetTableRange(0.0, 1.0)
        self._lut.Build()
        self._mapper.SetScalarRange(0.0, 1.0)
        self._image.Modified()

    def _show_speed_field(self, sf: dict) -> None:
        self._apply_speed_field(sf)

    # ── public API ──────────────────────────────────────────────────────────

    def set_mode(self, mode: str) -> None:
        """Switch the displayed field. One of 'speed', 'density', 'dynamic_speed', 'travel_time'."""
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}")
        if mode != "travel_time":
            self._destination = None
        self._mode = mode
        self._scalar_bar.SetTitle(self._bar_title())
        if self._ff is not None and mode != "travel_time":
            self._refresh_field()

    def set_recompute_interval(self, steps: int) -> None:
        """Rebuild the density/dynamic-speed fields every *steps* calls to update_density()."""
        self._ensure_floorfield()
        assert self._ff is not None
        self._ff.set_recompute_interval(steps)

    def update_density(self, positions: list[tuple[float, float]]) -> None:
        """Feed current agent positions; rebuilds density/dynamic-speed per recompute_interval."""
        self._ensure_floorfield()
        assert self._ff is not None
        self._ff.update_density(positions)
        if self._mode in ("density", "dynamic_speed"):
            self._refresh_field()
        elif self._mode == "travel_time" and self._destination is not None:
            self.set_destination(*self._destination)

    def set_destination(self, x: float, y: float) -> bool:
        """Compute travel times to (x, y) and update the overlay.

        Returns True if the destination is routable, False otherwise.
        """
        self._ensure_floorfield()
        assert self._ff is not None
        if not self._ff.is_routable((x, y)):
            return False
        self._destination = (x, y)
        # compute_waypoints triggers the eikonal solve for this destination.
        # Using the destination as both endpoints is enough to populate the grid.
        self._ff.compute_waypoints((x, y), (x, y))
        d = self._ff.travel_times()
        data: list[float] = d["data"]

        # Cells just inside a wall have speed ≈ 0, giving astronomically large
        # travel times that would compress the rest of the colour range to blue.
        # Use the 99th percentile of finite values as the colour ceiling instead
        # of the raw maximum, then clip displayed values at that ceiling.
        finite_vals = sorted(v for v in data if not math.isinf(v) and v >= 0)
        if not finite_vals:
            return False
        p99_idx = min(len(finite_vals) - 1, int(0.99 * len(finite_vals)))
        vmax = finite_vals[p99_idx] if finite_vals[p99_idx] > 0 else 1.0

        assert self._scalars is not None
        for i, v in enumerate(data):
            if math.isinf(v) or v < 0:
                self._scalars.SetValue(i, float("nan"))
            else:
                self._scalars.SetValue(i, min(v, vmax))
        self._scalars.Modified()
        self._lut.SetTableRange(0.0, vmax)
        self._lut.Build()
        self._mapper.SetScalarRange(0.0, vmax)
        self._image.Modified()
        return True

    def show(self, visible: bool) -> None:
        """Show or hide the floor field overlay and its colour bar."""
        if visible:
            self._ensure_floorfield()
        self._actor.SetVisibility(visible)
        self._scalar_bar.SetVisibility(visible)

    def get_actor(self) -> vtkActor:
        return self._actor

    def get_scalar_bar(self) -> vtkScalarBarActor:
        return self._scalar_bar

    def get_bounds(self) -> AABB:
        self._ensure_floorfield()
        assert self._ff is not None
        sf = self._ff.speed_field()
        ox, oy = sf["origin"]
        cs: float = sf["cell_size"]
        return AABB(
            xmin=ox,
            ymin=oy,
            xmax=ox + sf["width"] * cs,
            ymax=oy + sf["height"] * cs,
        )
