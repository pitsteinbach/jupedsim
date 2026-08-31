# SPDX-License-Identifier: LGPL-3.0-or-later
import math

from jupedsim import RoutingEngine
from jupedsim.recording import Recording
from PySide6.QtCore import QSignalBlocker, Qt, QTimer
from PySide6.QtGui import QFont, QPaintEvent
from PySide6.QtStateMachine import QState, QStateMachine
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QStyle,
    QVBoxLayout,
    QWidget,
)
from vtkmodules.vtkCommonCore import vtkCommand

from jupedsim_visualizer.floorfield_viz import FloorFieldViz
from jupedsim_visualizer.geometry import Geometry
from jupedsim_visualizer.geometry_widget import RenderWidget
from jupedsim_visualizer.trajectory import Trajectory


class PlayerControlWidget(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.play = QPushButton(
            QApplication.style().standardIcon(
                QStyle.StandardPixmap.SP_MediaPlay
            ),
            "",
        )
        self.play.setCheckable(True)
        self.begin = QPushButton(
            QApplication.style().standardIcon(
                QStyle.StandardPixmap.SP_MediaSkipBackward
            ),
            "",
        )
        self.backward = QPushButton(
            QApplication.style().standardIcon(
                QStyle.StandardPixmap.SP_MediaSeekBackward
            ),
            "",
        )
        self.forward = QPushButton(
            QApplication.style().standardIcon(
                QStyle.StandardPixmap.SP_MediaSeekForward
            ),
            "",
        )
        self.end = QPushButton(
            QApplication.style().standardIcon(
                QStyle.StandardPixmap.SP_MediaSkipForward
            ),
            "",
        )
        self.speed_selector = QSpinBox()
        self.speed_selector.setRange(1, 10)
        self.speed_selector.setValue(1)
        self.speed_selector.setSuffix("x")
        self.slider = QSlider()
        self.slider.setOrientation(Qt.Orientation.Horizontal)
        self.slider.setMaximum(60)
        self.slider.setPageStep(1)
        self.slider.setTracking(True)
        self.replay_time = QLabel("00:00:00.000")
        font = QFont("monospace")
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.replay_time.setFont(font)
        row1 = QHBoxLayout()
        row1.addStretch()
        row1.addWidget(self.begin)
        row1.addWidget(self.backward)
        row1.addWidget(self.play)
        row1.addWidget(self.forward)
        row1.addWidget(self.end)
        row1.addWidget(self.speed_selector)
        row1.addStretch()
        row2 = QHBoxLayout()
        row2.addWidget(self.slider, 1)
        row2.addWidget(self.replay_time)
        layout = QVBoxLayout()
        layout.addLayout(row1)
        layout.addLayout(row2)
        self.setLayout(layout)
        self._build_state_machine()

    def _build_state_machine(self) -> None:
        sm = QStateMachine(self)

        replay_paused = QState()
        sm.addState(replay_paused)

        replay_playing = QState()
        replay_playing.entered.connect(lambda: self.play.setChecked(True))
        replay_playing.exited.connect(lambda: self.play.setChecked(False))

        sm.addState(replay_playing)

        sm.setInitialState(replay_paused)

        replay_paused.addTransition(self.play.clicked, replay_playing)

        replay_playing.addTransition(self.play.clicked, replay_paused)
        replay_playing.addTransition(self.forward.clicked, replay_paused)
        replay_playing.addTransition(self.backward.clicked, replay_paused)
        replay_playing.addTransition(self.begin.clicked, replay_paused)
        replay_playing.addTransition(self.end.clicked, replay_paused)
        replay_playing.addTransition(self.slider.valueChanged, replay_paused)

        sm.start()
        self.state_machine = sm
        self.replay_paused = replay_paused
        self.replay_playing = replay_playing

    def update_replay_time(self, time_in_seconds: float) -> None:
        hh = int(math.floor(time_in_seconds / 3600))
        time_in_seconds = time_in_seconds - hh * 3600
        mm = int(math.floor(time_in_seconds / 60))
        time_in_seconds = time_in_seconds - mm * 60
        ss = int(math.floor(time_in_seconds))
        time_in_seconds = time_in_seconds - ss
        ms = int(time_in_seconds * 1000)
        self.replay_time.setText(f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}")


class ReplayWidget(QWidget):
    def __init__(
        self,
        navi: RoutingEngine,
        rec: Recording,
        geo: Geometry,
        trajectory: Trajectory,
        parent=None,
    ):
        QWidget.__init__(self, parent)
        self.rec = rec
        self.trajectory = trajectory
        self.control = PlayerControlWidget(parent=self)
        self.render_widget = RenderWidget(
            geo, navi, [geo, trajectory], parent=self
        )
        self.geo = geo

        self._ff_viz = FloorFieldViz(rec.geometry(), mode="density")
        self.render_widget.ren.AddActor(self._ff_viz.get_actor())
        self.render_widget.ren.AddActor2D(self._ff_viz.get_scalar_bar())

        ff_controls = QHBoxLayout()
        self._ff_toggle = QCheckBox("Floor field")
        self._ff_mode = QComboBox()
        self._ff_mode.addItems(["density", "dynamic_speed", "travel_time"])
        self._ff_mode.setEnabled(False)
        self._ff_interval_label = QLabel("every")
        self._ff_interval_label.setEnabled(False)
        self._ff_interval = QSpinBox()
        self._ff_interval.setRange(1, 500)
        self._ff_interval.setValue(1)
        self._ff_interval.setSuffix(" frames")
        self._ff_interval.setEnabled(False)
        self._ff_dest_hint = QLabel("← click to set destination")
        self._ff_dest_hint.setVisible(False)
        ff_controls.addWidget(self._ff_toggle)
        ff_controls.addWidget(self._ff_mode)
        ff_controls.addWidget(self._ff_interval_label)
        ff_controls.addWidget(self._ff_interval)
        ff_controls.addWidget(self._ff_dest_hint)
        ff_controls.addStretch()

        layout = QVBoxLayout()
        layout.addLayout(ff_controls)
        layout.addWidget(self.render_widget, 1)
        layout.addWidget(self.control)
        self.setLayout(layout)

        self.render_widget.style.AddObserver(
            vtkCommand.LeftButtonReleaseEvent, self._on_ff_click
        )

        self._ff_toggle.toggled.connect(self._on_ff_toggled)
        self._ff_mode.currentTextChanged.connect(self._on_ff_mode_changed)
        self._ff_interval.valueChanged.connect(self._on_ff_interval_changed)

        self.control.play.toggled.connect(self.play)
        self.control.forward.clicked.connect(self.frame_forward)
        self.control.backward.clicked.connect(self.frame_backward)
        self.control.slider.setMaximum(self.rec.num_frames - 1)
        self.control.slider.valueChanged.connect(self.goto_frame)
        self.control.begin.clicked.connect(lambda: self.goto_frame(0))
        self.control.end.clicked.connect(
            lambda: self.goto_frame(self.trajectory.num_frames - 1)
        )

    def _current_positions(self) -> list[tuple[float, float]]:
        frame = self.rec.frame(self.trajectory.current_index)
        return [(a.position[0], a.position[1]) for a in frame.agents]

    def _on_ff_toggled(self, checked: bool) -> None:
        self._ff_mode.setEnabled(checked)
        self._ff_interval_label.setEnabled(checked)
        self._ff_interval.setEnabled(checked)
        if checked:
            self._ff_viz.set_recompute_interval(self._ff_interval.value())
            self._ff_viz.update_density(self._current_positions())
        self._ff_viz.show(checked)
        self.render_widget.render()

    def _on_ff_mode_changed(self, mode: str) -> None:
        self._ff_dest_hint.setVisible(mode == "travel_time")
        self._ff_viz.set_mode(mode)
        self.render_widget.render()

    def _on_ff_interval_changed(self, steps: int) -> None:
        self._ff_viz.set_recompute_interval(steps)

    def _on_ff_click(self, obj, evt) -> None:
        if not self._ff_toggle.isChecked():
            return
        if self._ff_mode.currentText() != "travel_time":
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

    def _maybe_update_ff(self) -> None:
        if self._ff_toggle.isChecked():
            self._ff_viz.update_density(self._current_positions())

    def frame_forward(self):
        self.trajectory.advance_frame(self.control.speed_selector.value())
        self.control.update_replay_time(
            self.trajectory.current_index * (1 / self.rec.fps)
        )
        self._maybe_update_ff()
        self.render_widget.render()
        with QSignalBlocker(self.control.slider):
            self.control.slider.setValue(self.trajectory.current_index)

    def frame_backward(self):
        self.trajectory.advance_frame(-self.control.speed_selector.value())
        self.control.update_replay_time(
            self.trajectory.current_index * (1 / self.rec.fps)
        )
        self._maybe_update_ff()
        self.render_widget.render()
        with QSignalBlocker(self.control.slider):
            self.control.slider.setValue(self.trajectory.current_index)

    def goto_frame(self, index: int):
        self.trajectory.goto_frame(index)
        self.control.update_replay_time(
            self.trajectory.current_index * (1 / self.rec.fps)
        )
        self._maybe_update_ff()
        self.render_widget.render()
        with QSignalBlocker(self.control.slider):
            self.control.slider.setValue(self.trajectory.current_index)

    def play(self, checked: bool):
        if checked:
            self.timer = QTimer()
            self.timer.setInterval(int(1000.0 / self.rec.fps))
            self.timer.timeout.connect(self.frame_forward)
            self.timer.start()
        else:
            if self.timer:
                self.timer.stop()

    def render(self):
        self.render_widget.render()

    def paintEvent(self, event: QPaintEvent) -> None:
        self.render()
        return super().paintEvent(event)
