# SPDX-License-Identifier: LGPL-3.0-or-later
import sys
from pathlib import Path

import vtkmodules.qt
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleImage
from vtkmodules.vtkRenderingCore import vtkRenderer

from jupedsim_visualizer.config import Colors
from jupedsim_visualizer.floorfield_viz import FloorFieldHdf5Viz
from jupedsim_visualizer.grid import Grid

if sys.platform == "darwin":
    vtkmodules.qt.QVTKRWIBase = "QOpenGLWidget"
from vtkmodules.qt.QVTKRenderWindowInteractor import (
    QVTKRenderWindowInteractor,  # noqa: E402
)


class _VtkPane(QVTKRenderWindowInteractor):
    """Minimal VTK pane: parallel camera, image-style pan/zoom, optional grid."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ren = vtkRenderer()
        self.ren.SetBackground(*Colors.d)
        self.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.GetRenderWindow().GetInteractor()

        cam = self.ren.GetActiveCamera()
        cam.ParallelProjectionOn()

        style = vtkInteractorStyleImage()
        self.iren.SetInteractorStyle(style)
        self.iren.Initialize()

        self._grid = Grid(self.ren, cam)

    def show_grid(self, state: bool) -> None:
        self._grid.show(state)

    def render(self) -> None:
        self.iren.Render()

    def fit_bounds(
        self, xmin: float, ymin: float, xmax: float, ymax: float
    ) -> None:
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        w = xmax - xmin
        h = ymax - ymin
        cam = self.ren.GetActiveCamera()
        aw, ah = self.ren.GetAspect()
        vp_ratio = aw / ah if ah else 1.0
        scene_ratio = w / h if h else 1.0
        if vp_ratio > scene_ratio:
            scale = (h / 2) * 1.05
        else:
            scale = (w / 2) / vp_ratio * 1.05
        cam.SetParallelScale(scale)
        cam.SetFocalPoint(cx, cy, 0)
        cam.SetPosition(cx, cy, 100)
        cam.SetViewUp(0, 1, 0)
        cam.SetClippingRange(0, 200)
        self.iren.Render()


class FloorFieldHdf5Widget(QWidget):
    """Tab widget that displays a floor-field loaded from an HDF5 file.

    For time-series files (written by ``jps.FloorFieldHdf5Writer``) a frame
    slider appears below the controls so the user can scrub through snapshots.
    Single-frame files (written by ``Floorfield.write_travel_times_hdf5``)
    show no slider.
    """

    def __init__(self, path: str, dest_key: str | None = None, parent=None):
        super().__init__(parent)

        self._viz = FloorFieldHdf5Viz(path, dest_key)

        self.render_widget = _VtkPane(parent=self)
        ren = self.render_widget.ren
        ren.AddActor(self._viz.get_actor())
        ren.AddActor(self._viz.get_arrow_actor())
        ren.AddActor(self._viz.get_iso_actor())
        ren.AddActor2D(self._viz.get_scalar_bar())
        self._viz.show(True)

        # ── Top controls bar ──────────────────────────────────────────────────
        self._arrows_toggle = QCheckBox("Gradient arrows")
        self._iso_toggle = QCheckBox("Isolines")

        controls = QHBoxLayout()
        controls.addWidget(QLabel(Path(path).name))
        controls.addStretch()
        controls.addWidget(self._arrows_toggle)
        controls.addWidget(self._iso_toggle)

        # ── Frame slider (time-series files only) ─────────────────────────────
        n = self._viz.num_frames
        self._frame_row: QWidget | None = None
        if n > 1:
            self._frame_slider = QSlider(Qt.Orientation.Horizontal)
            self._frame_slider.setRange(0, n - 1)
            self._frame_slider.setValue(0)
            self._frame_slider.setTracking(True)
            self._frame_label = QLabel(self._frame_label_text(0))

            frame_row_layout = QHBoxLayout()
            frame_row_layout.addWidget(QLabel("Frame:"))
            frame_row_layout.addWidget(self._frame_slider, 1)
            frame_row_layout.addWidget(self._frame_label)
            self._frame_row = QWidget()
            self._frame_row.setLayout(frame_row_layout)

            self._frame_slider.valueChanged.connect(self._on_frame_changed)

        # ── Layout ────────────────────────────────────────────────────────────
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addLayout(controls)
        if self._frame_row is not None:
            layout.addWidget(self._frame_row)
        layout.addWidget(self.render_widget, 1)
        self.setLayout(layout)

        self._arrows_toggle.toggled.connect(self._on_arrows)
        self._iso_toggle.toggled.connect(self._on_iso)

        bounds = self._viz.get_bounds()
        self.render_widget.fit_bounds(
            bounds.xmin, bounds.ymin, bounds.xmax, bounds.ymax
        )

    # ── internal ──────────────────────────────────────────────────────────────

    def _frame_label_text(self, i: int) -> str:
        sim_frame = self._viz.frame_sim_index(i)
        return f"{i + 1}/{self._viz.num_frames}  (sim frame {sim_frame})"

    def _on_frame_changed(self, i: int) -> None:
        self._viz.set_frame(i)
        self._frame_label.setText(self._frame_label_text(i))
        self.render_widget.render()

    def _on_arrows(self, checked: bool) -> None:
        self._viz.set_gradient_arrows_visible(checked)
        self.render_widget.render()

    def _on_iso(self, checked: bool) -> None:
        self._viz.set_isolines_visible(checked)
        self.render_widget.render()

    def render(self) -> None:
        self.render_widget.render()
