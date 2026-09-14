# SPDX-License-Identifier: LGPL-3.0-or-later
import math

import jupedsim as jps
from jupedsim.internal.aabb import AABB
from vtkmodules.vtkCommonCore import vtkFloatArray, vtkLookupTable, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkContourFilter, vtkGlyph3D
from vtkmodules.vtkFiltersSources import vtkArrowSource
from vtkmodules.vtkRenderingAnnotation import vtkScalarBarActor
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkDataSetMapper,
    vtkPolyDataMapper,
)

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

    Overlays (travel_time mode only)
    ---------------------------------
    set_gradient_arrows_visible(True)   arrows pointing toward destination
    set_isolines_visible(True)          travel-time isolines

    Export
    ------
    write_hdf5(path)   write speed_field + travel_times to HDF5 via Rust
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

        # Grid metadata (populated in _ensure_floorfield)
        self._grid_w: int = 0
        self._grid_h: int = 0
        self._grid_ox: float = 0.0
        self._grid_oy: float = 0.0
        self._grid_cs: float = 0.0
        self._current_vmax: float = 1.0

        self._lut = vtkLookupTable()
        self._lut.SetHueRange(0.667, 0.0)  # blue (near) → red (far)
        self._lut.SetSaturationRange(1.0, 1.0)
        self._lut.SetValueRange(1.0, 1.0)
        self._lut.SetNanColor(0.25, 0.25, 0.25, 0.0)
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

        # ── Gradient-arrow overlay pipeline ──────────────────────────────────
        self._show_arrows: bool = False
        self._arrow_stride: int = 5
        self._arrow_poly = vtkPolyData()
        self._arrow_poly.SetPoints(vtkPoints())
        _arr_vecs = vtkFloatArray()
        _arr_vecs.SetNumberOfComponents(3)
        _arr_vecs.SetName("direction")
        self._arrow_poly.GetPointData().SetVectors(_arr_vecs)

        _arrow_src = vtkArrowSource()
        _arrow_src.SetTipRadius(0.15)
        _arrow_src.SetShaftRadius(0.05)
        _arrow_src.SetTipLength(0.35)
        self._glyph = vtkGlyph3D()
        self._glyph.SetSourceConnection(_arrow_src.GetOutputPort())
        self._glyph.SetInputData(self._arrow_poly)
        self._glyph.SetVectorModeToUseVector()
        self._glyph.OrientOn()
        self._glyph.SetScaleModeToDataScalingOff()
        self._glyph.SetScaleFactor(1.0)

        _arrow_mapper = vtkPolyDataMapper()
        _arrow_mapper.SetInputConnection(self._glyph.GetOutputPort())
        self._arrow_actor = vtkActor()
        self._arrow_actor.SetMapper(_arrow_mapper)
        self._arrow_actor.GetProperty().SetColor(1.0, 0.5, 0.0)  # orange
        self._arrow_actor.GetProperty().SetAmbient(1.0)
        self._arrow_actor.GetProperty().SetDiffuse(0.0)
        self._arrow_actor.SetVisibility(False)

        # ── Isoline overlay pipeline ──────────────────────────────────────────
        self._show_isolines: bool = False
        self._n_iso_levels: int = 10
        self._contour = vtkContourFilter()
        self._contour_connected: bool = False

        _iso_mapper = vtkPolyDataMapper()
        _iso_mapper.SetInputConnection(self._contour.GetOutputPort())
        _iso_mapper.ScalarVisibilityOff()
        self._iso_actor = vtkActor()
        self._iso_actor.SetMapper(_iso_mapper)
        self._iso_actor.GetProperty().SetColor(0.9, 0.9, 0.9)  # near-white
        self._iso_actor.GetProperty().SetAmbient(1.0)
        self._iso_actor.GetProperty().SetDiffuse(0.0)
        self._iso_actor.GetProperty().SetLineWidth(1.5)
        self._iso_actor.GetProperty().SetOpacity(0.8)
        self._iso_actor.SetVisibility(False)

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

        self._grid_w = width
        self._grid_h = height
        self._grid_ox = ox
        self._grid_oy = oy
        self._grid_cs = cs

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

        # Connect contour filter to the image now that it exists
        self._contour.SetInputData(self._image)
        self._contour_connected = True

        self._refresh_field()

    def _refresh_field(self) -> None:
        assert self._ff is not None and self._scalars is not None
        if self._mode == "speed":
            self._apply_speed_field(self._ff.speed_field())
        elif self._mode == "density":
            self._apply_density_field(self._ff.density_field())
        elif self._mode == "dynamic_speed":
            self._apply_dynamic_speed_field(self._ff.dynamic_speed_field())

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

    def _rebuild_arrows(self, vmax: float) -> None:
        """Rebuild the gradient-arrow polydata from the Rust Sobel gradient."""
        assert self._ff is not None
        gd = self._ff.travel_time_gradient()
        grad_data: list[float] = gd["data"]
        w = self._grid_w
        h = self._grid_h
        ox = self._grid_ox
        oy = self._grid_oy
        cs = self._grid_cs
        stride = self._arrow_stride
        z = ZLayers.floorfield + 0.02

        pts = vtkPoints()
        vecs = vtkFloatArray()
        vecs.SetNumberOfComponents(3)
        vecs.SetName("direction")

        for row in range(stride // 2, h, stride):
            for col in range(stride // 2, w, stride):
                idx = row * w + col
                gx = grad_data[2 * idx]
                gy = grad_data[2 * idx + 1]
                mag = math.sqrt(gx * gx + gy * gy)
                if mag < 1e-9:
                    continue
                # Negate: arrow points toward lower T (toward destination)
                dx = -gx / mag
                dy = -gy / mag
                px = ox + (col + 0.5) * cs
                py = oy + (row + 0.5) * cs
                pts.InsertNextPoint(px, py, z)
                vecs.InsertNextTuple3(dx, dy, 0.0)

        self._arrow_poly.SetPoints(pts)
        self._arrow_poly.GetPointData().SetVectors(vecs)
        self._arrow_poly.Modified()
        self._glyph.SetScaleFactor(stride * cs * 0.8)
        self._arrow_actor.SetVisibility(self._show_arrows)

    def _rebuild_isolines(self, vmax: float) -> None:
        """Regenerate contour levels spaced across [vmax/n, vmax]."""
        n = self._n_iso_levels
        self._contour.GenerateValues(n, vmax / n, vmax)
        self._contour.Modified()
        self._iso_actor.SetVisibility(self._show_isolines)

    # ── public API ──────────────────────────────────────────────────────────

    def set_mode(self, mode: str) -> None:
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}")
        if mode != "travel_time":
            self._destination = None
        self._mode = mode
        self._scalar_bar.SetTitle(self._bar_title())
        if self._ff is not None and mode != "travel_time":
            self._refresh_field()

    def set_recompute_interval(self, steps: int) -> None:
        self._ensure_floorfield()
        assert self._ff is not None
        self._ff.set_recompute_interval(steps)

    def update_density(self, positions: list[tuple[float, float]]) -> None:
        self._ensure_floorfield()
        assert self._ff is not None
        self._ff.update_density(positions)
        if self._mode in ("density", "dynamic_speed"):
            self._refresh_field()
        elif self._mode == "travel_time" and self._destination is not None:
            self.set_destination(*self._destination)

    def set_destination(self, x: float, y: float) -> bool:
        """Compute travel times to (x, y) and update the overlay.

        Returns True if the destination is routable.
        """
        self._ensure_floorfield()
        assert self._ff is not None
        if not self._ff.is_routable((x, y)):
            return False
        self._destination = (x, y)
        self._ff.compute_waypoints((x, y), (x, y))
        d = self._ff.travel_times()
        data: list[float] = d["data"]

        finite_vals = sorted(v for v in data if not math.isinf(v) and v >= 0)
        if not finite_vals:
            return False
        p99_idx = min(len(finite_vals) - 1, int(0.99 * len(finite_vals)))
        vmax = finite_vals[p99_idx] if finite_vals[p99_idx] > 0 else 1.0
        self._current_vmax = vmax

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

        self._rebuild_arrows(vmax)
        self._rebuild_isolines(vmax)
        return True

    def set_gradient_arrows_visible(
        self, visible: bool, stride: int | None = None
    ) -> None:
        """Show or hide gradient-direction arrows.

        Args:
            visible: whether to show arrows.
            stride:  subsampling stride in grid cells (default 5).
                     Changing stride triggers a rebuild.
        """
        if stride is not None and stride != self._arrow_stride:
            self._arrow_stride = max(1, stride)
            if self._destination is not None and self._ff is not None:
                self._rebuild_arrows(self._current_vmax)
        self._show_arrows = visible
        self._arrow_actor.SetVisibility(visible)

    def set_isolines_visible(
        self, visible: bool, n_levels: int | None = None
    ) -> None:
        """Show or hide travel-time isolines.

        Args:
            visible:  whether to show isolines.
            n_levels: number of contour levels (default 10).
                      Changing n_levels triggers a rebuild.
        """
        if n_levels is not None and n_levels != self._n_iso_levels:
            self._n_iso_levels = max(2, n_levels)
            if self._destination is not None:
                self._rebuild_isolines(self._current_vmax)
        self._show_isolines = visible
        self._iso_actor.SetVisibility(visible)

    def write_hdf5(self, path: str) -> None:
        """Write speed_field + travel_times to *path* as HDF5.

        Raises RuntimeError if no destination has been set yet.
        Requires h5py and numpy.
        """
        if self._ff is None or self._destination is None:
            raise RuntimeError(
                "No travel-time field available; call set_destination() first."
            )
        self._ff.write_travel_times_hdf5(path)

    def show(self, visible: bool) -> None:
        if visible:
            self._ensure_floorfield()
        self._actor.SetVisibility(visible)
        self._scalar_bar.SetVisibility(visible)
        if not visible:
            self._arrow_actor.SetVisibility(False)
            self._iso_actor.SetVisibility(False)
        else:
            self._arrow_actor.SetVisibility(self._show_arrows)
            self._iso_actor.SetVisibility(self._show_isolines)

    def get_actor(self) -> vtkActor:
        return self._actor

    def get_arrow_actor(self) -> vtkActor:
        return self._arrow_actor

    def get_iso_actor(self) -> vtkActor:
        return self._iso_actor

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


class FloorFieldHdf5Viz:
    """VTK visualization of travel-time grids loaded from an HDF5 file.

    Handles two file layouts:

    * **Single-frame** (written by :py:meth:`~jupedsim.routing.Floorfield.write_travel_times_hdf5`):
      ``/speed_field`` + ``/travel_times`` datasets at the root — one snapshot.

    * **Time-series** (written by :class:`~jupedsim.FloorFieldHdf5Writer`):
      ``/floor_fields/`` group with ``travel_times[N, h, w]`` and
      ``frame_indices[N]`` — scrub through N snapshots via :py:meth:`set_frame`.

    Usage::

        viz = FloorFieldHdf5Viz("simulation_output.h5")
        renderer.AddActor(viz.get_actor())
        renderer.AddActor(viz.get_arrow_actor())
        renderer.AddActor(viz.get_iso_actor())
        renderer.AddActor2D(viz.get_scalar_bar())
        viz.show(True)

        # For time-series files:
        for i in range(viz.num_frames):
            viz.set_frame(i)
            render()
    """

    def __init__(self, path: str, dest_key: str | None = None) -> None:
        import h5py
        import numpy as np

        self._show_arrows: bool = False
        self._show_isolines: bool = False
        self._arrow_stride: int = 5
        self._n_iso_levels: int = 10

        # h5py file kept open for lazy multi-frame reads; closed in __del__
        self._h5file: object = h5py.File(path, "r")
        self._tt_ds = None  # h5py Dataset, set for time-series files

        f = self._h5file
        if "floor_fields" in f:
            # ── Time-series format written by FloorFieldHdf5Writer ─────────
            grp = f["floor_fields"]
            ox = float(grp.attrs["origin_x"])
            oy = float(grp.attrs["origin_y"])
            cs = float(grp.attrs["cell_size"])
            w = int(grp.attrs["width"])
            h_grid = int(grp.attrs["height"])
            self._tt_ds = grp["travel_times"]  # shape (N, h, w)
            n_frames = self._tt_ds.shape[0]
            if n_frames == 0:
                raise ValueError(f"{path!r} contains no floor-field frames.")
            self._num_frames: int = n_frames
            self._frame_sim_indices: list[int] = grp["frame_indices"][
                :
            ].tolist()
            data_np = self._tt_ds[0, :, :].astype(np.float64)
            data: list[float] = data_np.flatten().tolist()
        else:
            # ── Single-frame format ────────────────────────────────────────
            from jupedsim_visualizer.floorfield_io import load_travel_times

            all_fields = load_travel_times(path)
            if isinstance(all_fields, dict) and "data" in all_fields:
                field = all_fields
            else:
                if dest_key is None:
                    dest_key = next(iter(all_fields))
                field = all_fields[dest_key]
            data = field["data"]
            w = field["width"]
            h_grid = field["height"]
            ox, oy = field["origin"]
            cs = field["cell_size"]
            self._num_frames = 1
            self._frame_sim_indices = [0]

        self._w = w
        self._h = h_grid
        self._ox = ox
        self._oy = oy
        self._cs = cs

        # Colour range: 99th-percentile of finite values
        vmax = self._vmax_from(data)
        self._vmax = vmax

        # ── Colour map ────────────────────────────────────────────────────────
        self._lut = vtkLookupTable()
        self._lut.SetHueRange(0.667, 0.0)
        self._lut.SetSaturationRange(1.0, 1.0)
        self._lut.SetValueRange(1.0, 1.0)
        self._lut.SetNanColor(0.25, 0.25, 0.25, 0.0)
        self._lut.SetTableRange(0.0, vmax)
        self._lut.Build()

        # ── vtkImageData ──────────────────────────────────────────────────────
        self._image = vtkImageData()
        self._image.SetDimensions(w, h_grid, 1)
        self._image.SetOrigin(ox, oy, ZLayers.floorfield)
        self._image.SetSpacing(cs, cs, 1.0)

        self._scalars = vtkFloatArray()
        self._scalars.SetNumberOfTuples(w * h_grid)
        self._write_scalars(data, vmax)
        self._image.GetPointData().SetScalars(self._scalars)

        mapper = vtkDataSetMapper()
        mapper.SetInputData(self._image)
        mapper.SetLookupTable(self._lut)
        mapper.SetScalarRange(0.0, vmax)
        mapper.SetUseLookupTableScalarRange(True)
        self._mapper = mapper

        self._actor = vtkActor()
        self._actor.SetMapper(mapper)
        self._actor.GetProperty().SetOpacity(0.75)
        self._actor.SetVisibility(False)

        self._scalar_bar = vtkScalarBarActor()
        self._scalar_bar.SetLookupTable(self._lut)
        self._scalar_bar.SetTitle("Travel time (s)")
        self._scalar_bar.SetNumberOfLabels(5)
        self._scalar_bar.SetPosition(0.87, 0.10)
        self._scalar_bar.SetWidth(0.10)
        self._scalar_bar.SetHeight(0.70)
        self._scalar_bar.SetVisibility(False)

        # ── Gradient-arrow pipeline ───────────────────────────────────────────
        self._arrow_poly = vtkPolyData()
        self._arrow_poly.SetPoints(vtkPoints())
        _arr_vecs = vtkFloatArray()
        _arr_vecs.SetNumberOfComponents(3)
        _arr_vecs.SetName("direction")
        self._arrow_poly.GetPointData().SetVectors(_arr_vecs)

        _arrow_src = vtkArrowSource()
        _arrow_src.SetTipRadius(0.15)
        _arrow_src.SetShaftRadius(0.05)
        _arrow_src.SetTipLength(0.35)
        self._glyph = vtkGlyph3D()
        self._glyph.SetSourceConnection(_arrow_src.GetOutputPort())
        self._glyph.SetInputData(self._arrow_poly)
        self._glyph.SetVectorModeToUseVector()
        self._glyph.OrientOn()
        self._glyph.SetScaleModeToDataScalingOff()

        _arrow_mapper = vtkPolyDataMapper()
        _arrow_mapper.SetInputConnection(self._glyph.GetOutputPort())
        self._arrow_actor = vtkActor()
        self._arrow_actor.SetMapper(_arrow_mapper)
        self._arrow_actor.GetProperty().SetColor(1.0, 0.5, 0.0)  # orange
        self._arrow_actor.GetProperty().SetAmbient(1.0)
        self._arrow_actor.GetProperty().SetDiffuse(0.0)
        self._arrow_actor.SetVisibility(False)

        # ── Isoline pipeline ──────────────────────────────────────────────────
        self._contour = vtkContourFilter()
        self._contour.SetInputData(self._image)
        self._contour.GenerateValues(
            self._n_iso_levels, vmax / self._n_iso_levels, vmax
        )

        _iso_mapper = vtkPolyDataMapper()
        _iso_mapper.SetInputConnection(self._contour.GetOutputPort())
        _iso_mapper.ScalarVisibilityOff()
        self._iso_actor = vtkActor()
        self._iso_actor.SetMapper(_iso_mapper)
        self._iso_actor.GetProperty().SetColor(0.9, 0.9, 0.9)  # near-white
        self._iso_actor.GetProperty().SetAmbient(1.0)
        self._iso_actor.GetProperty().SetDiffuse(0.0)
        self._iso_actor.GetProperty().SetLineWidth(1.5)
        self._iso_actor.GetProperty().SetOpacity(0.8)
        self._iso_actor.SetVisibility(False)

        # Pre-compute gradient for arrows
        self._grad_flat = self._sobel_gradient(data)

    def __del__(self) -> None:
        if self._h5file is not None:
            try:
                self._h5file.close()  # type: ignore[union-attr]
            except Exception:
                pass
            self._h5file = None

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _vmax_from(data: list[float]) -> float:
        finite_vals = sorted(v for v in data if not math.isinf(v) and v >= 0)
        if not finite_vals:
            return 1.0
        vmax = finite_vals[
            min(len(finite_vals) - 1, int(0.99 * len(finite_vals)))
        ]
        return vmax if vmax > 0 else 1.0

    def _write_scalars(self, data: list[float], vmax: float) -> None:
        for i, v in enumerate(data):
            if math.isinf(v) or math.isnan(v) or v < 0:
                self._scalars.SetValue(i, float("nan"))
            else:
                self._scalars.SetValue(i, min(v, vmax))

    # ── internal ────────────────────────────────────────────────────────────

    def _sobel_gradient(self, data: list[float]) -> list[float]:
        """Vectorized 3×3 Sobel gradient; wall/inf neighbours → centre value."""
        import numpy as np

        w, h, cs = self._w, self._h, self._cs
        tt = np.array(data, dtype=np.float64).reshape(h, w)
        finite = np.isfinite(tt)
        tt_c = np.where(finite, tt, 0.0)  # zero for walls (approximation)

        # Pad with edge values so border cells are handled
        p = np.pad(tt_c, 1, mode="edge")
        # For wall neighbours we want centre value; use edge-padded finite val
        # (close enough for visualisation)

        gx = (
            (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:])
            - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
        ) / (8.0 * cs)
        gy = (
            (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:])
            - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
        ) / (8.0 * cs)

        gx[~finite] = 0.0
        gy[~finite] = 0.0

        # Interleaved (gx, gy) per cell, matching the Rust layout
        out = np.empty(2 * h * w, dtype=np.float64)
        out[0::2] = gx.ravel()
        out[1::2] = gy.ravel()
        return out.tolist()

    def _rebuild_arrows(self) -> None:
        w, h, ox, oy, cs = self._w, self._h, self._ox, self._oy, self._cs
        stride = self._arrow_stride
        z = ZLayers.floorfield + 0.02
        grad = self._grad_flat

        pts = vtkPoints()
        vecs = vtkFloatArray()
        vecs.SetNumberOfComponents(3)
        vecs.SetName("direction")

        for row in range(stride // 2, h, stride):
            for col in range(stride // 2, w, stride):
                idx = row * w + col
                gx = grad[2 * idx]
                gy = grad[2 * idx + 1]
                mag = math.sqrt(gx * gx + gy * gy)
                if mag < 1e-9:
                    continue
                pts.InsertNextPoint(
                    ox + (col + 0.5) * cs,
                    oy + (row + 0.5) * cs,
                    z,
                )
                vecs.InsertNextTuple3(-gx / mag, -gy / mag, 0.0)

        self._arrow_poly.SetPoints(pts)
        self._arrow_poly.GetPointData().SetVectors(vecs)
        self._arrow_poly.Modified()
        self._glyph.SetScaleFactor(stride * cs * 0.8)

    # ── public API ──────────────────────────────────────────────────────────

    @property
    def num_frames(self) -> int:
        """Number of time-series snapshots in the file (1 for single-frame files)."""
        return self._num_frames

    @property
    def frame_sim_indices(self) -> list[int]:
        """Simulation iteration number at which each snapshot was recorded."""
        return self._frame_sim_indices

    def frame_sim_index(self, i: int) -> int:
        """Simulation iteration number recorded for snapshot *i*."""
        return int(self._frame_sim_indices[i])

    def set_frame(self, i: int) -> None:
        """Switch the display to snapshot *i* (0-based).

        For single-frame files this is a no-op unless *i* == 0.
        """
        import numpy as np

        if i < 0 or i >= self._num_frames:
            raise IndexError(
                f"Frame index {i} out of range [0, {self._num_frames})"
            )
        if self._num_frames == 1:
            return

        data_np: np.ndarray = self._tt_ds[i, :, :].astype(np.float64)  # type: ignore[index]
        data: list[float] = data_np.flatten().tolist()

        vmax = self._vmax_from(data)
        self._vmax = vmax

        self._write_scalars(data, vmax)
        self._scalars.Modified()
        self._lut.SetTableRange(0.0, vmax)
        self._lut.Build()
        self._mapper.SetScalarRange(0.0, vmax)
        self._image.Modified()

        self._grad_flat = self._sobel_gradient(data)
        if self._show_arrows:
            self._rebuild_arrows()

        self._contour.GenerateValues(
            self._n_iso_levels, vmax / self._n_iso_levels, vmax
        )
        self._contour.Modified()

    def set_gradient_arrows_visible(
        self, visible: bool, stride: int | None = None
    ) -> None:
        if stride is not None and stride != self._arrow_stride:
            self._arrow_stride = max(1, stride)
            self._rebuild_arrows()
        elif visible and self._arrow_poly.GetNumberOfPoints() == 0:
            self._rebuild_arrows()
        self._show_arrows = visible
        self._arrow_actor.SetVisibility(visible)

    def set_isolines_visible(
        self, visible: bool, n_levels: int | None = None
    ) -> None:
        self._show_isolines = visible
        self._iso_actor.SetVisibility(visible)

    def show(self, visible: bool) -> None:
        self._actor.SetVisibility(visible)
        self._scalar_bar.SetVisibility(visible)
        if not visible:
            self._arrow_actor.SetVisibility(False)
            self._iso_actor.SetVisibility(False)
        else:
            self._arrow_actor.SetVisibility(self._show_arrows)
            self._iso_actor.SetVisibility(self._show_isolines)

    def get_actor(self) -> vtkActor:
        return self._actor

    def get_arrow_actor(self) -> vtkActor:
        return self._arrow_actor

    def get_iso_actor(self) -> vtkActor:
        return self._iso_actor

    def get_scalar_bar(self) -> vtkScalarBarActor:
        return self._scalar_bar

    def get_bounds(self) -> AABB:
        return AABB(
            xmin=self._ox,
            ymin=self._oy,
            xmax=self._ox + self._w * self._cs,
            ymax=self._oy + self._h * self._cs,
        )
