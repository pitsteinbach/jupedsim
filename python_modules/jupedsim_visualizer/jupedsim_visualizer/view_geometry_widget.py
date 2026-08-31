# SPDX-License-Identifier: LGPL-3.0-or-later
import jupedsim as jps
from PySide6.QtCore import Qt
from PySide6.QtGui import QPaintEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from vtkmodules.vtkCommonCore import vtkCommand

from jupedsim_visualizer.floorfield_viz import FloorFieldViz
from jupedsim_visualizer.geometry import Geometry
from jupedsim_visualizer.geometry_widget import RenderWidget


class ViewGeometryWidget(QWidget):
    def __init__(
        self,
        navi: jps.RoutingEngine,
        geo: Geometry,
        geometry,
        name_text: str,
        info_text: str,
        parent=None,
    ):
        QWidget.__init__(self, parent)
        self.geo = geo
        self._navi = navi

        self._ff_viz = FloorFieldViz(geometry)

        bottom_layout = QHBoxLayout()
        geometry_label = QLabel(name_text)
        geometry_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        bottom_layout.addWidget(geometry_label, 1, Qt.AlignmentFlag.AlignLeft)

        properties_label = QLabel(info_text)
        properties_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        bottom_layout.addWidget(
            properties_label, 1, Qt.AlignmentFlag.AlignRight
        )

        layout = QVBoxLayout()

        controls = QHBoxLayout()
        reset_cam_bt = QPushButton("Reset Camera")
        controls.addWidget(reset_cam_bt)

        self._ff_toggle = QCheckBox("Floor field")
        self._ff_hint = QLabel("← click to set destination")
        self._ff_hint.setVisible(False)
        controls.addWidget(self._ff_toggle)
        controls.addWidget(self._ff_hint)
        controls.addStretch()
        layout.addLayout(controls)

        self.render_widget = RenderWidget(geo, navi, [geo], parent=self)

        # Register floor field actors with the renderer
        self.render_widget.ren.AddActor(self._ff_viz.get_actor())
        self.render_widget.ren.AddActor2D(self._ff_viz.get_scalar_bar())

        # Left-click on the VTK scene sets the floor field destination
        self.render_widget.style.AddObserver(
            vtkCommand.LeftButtonReleaseEvent, self._on_ff_click
        )

        layout.addWidget(self.render_widget)

        self.hover_label = QLabel("")
        self.hover_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.hover_label)

        layout.addLayout(bottom_layout)
        self.setLayout(layout)

        reset_cam_bt.clicked.connect(self.render_widget.reset_camera)
        self.render_widget.on_hover_triangle.connect(self.hover_label.setText)
        self._ff_toggle.toggled.connect(self._on_ff_toggled)

    def _on_ff_toggled(self, checked: bool) -> None:
        self._ff_hint.setVisible(checked)
        self._ff_viz.show(checked)
        # Disable path-drawing while the floor field overlay is active so that
        # left-click is used exclusively for setting the destination.
        self.render_widget.move_controller.set_navi(
            None if checked else self._navi
        )
        self.render_widget.render()

    def _on_ff_click(self, obj, evt) -> None:
        if not self._ff_toggle.isChecked():
            return
        interactor = obj.GetInteractor()
        pos = interactor.GetEventPosition()
        renderer = self.render_widget.ren
        renderer.SetDisplayPoint(pos[0], pos[1], 0)
        renderer.DisplayToWorld()
        world = renderer.GetWorldPoint()
        x = world[0] / world[3]
        y = world[1] / world[3]
        if self._ff_viz.set_destination(x, y):
            self.render_widget.render()

    def render(self):
        self.render_widget.render()

    def paintEvent(self, event: QPaintEvent) -> None:
        self.render()
        return super().paintEvent(event)
