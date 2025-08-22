from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton,
    QDialogButtonBox, QLabel, QComboBox, QDoubleSpinBox, QCheckBox,
    QGridLayout
)
from PyQt5.QtCore import Qt
import copy


class SpatialTrajectorySegmentDialog(QDialog):
    """Dialog to edit a single trajectory segment."""

    def __init__(self, parent=None, segment=None):
        super().__init__(parent)
        self.setWindowTitle("Trajectory Segment")
        self.segment = copy.deepcopy(segment) if segment else {}
        self._setup_ui()
        if segment:
            self._populate(self.segment)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        form = QGridLayout()

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["rotate", "oscillate", "rotating_arc"])
        self.mode_combo.currentTextChanged.connect(self._update_mode_fields)
        form.addWidget(QLabel("Mode:"), 0, 0)
        form.addWidget(self.mode_combo, 0, 1)

        self.start_label = QLabel("Start Deg:")
        self.start_spin = QDoubleSpinBox(); self.start_spin.setRange(-360.0, 360.0); self.start_spin.setDecimals(3)
        form.addWidget(self.start_label, 1, 0)
        form.addWidget(self.start_spin, 1, 1)

        self.speed_label = QLabel("Speed (deg/s):")
        self.speed_spin = QDoubleSpinBox(); self.speed_spin.setRange(-3600.0, 3600.0); self.speed_spin.setDecimals(3)
        form.addWidget(self.speed_label, 2, 0)
        form.addWidget(self.speed_spin, 2, 1)

        self.center_label = QLabel("Center Deg:")
        self.center_spin = QDoubleSpinBox(); self.center_spin.setRange(-360.0, 360.0); self.center_spin.setDecimals(3)
        form.addWidget(self.center_label, 3, 0)
        form.addWidget(self.center_spin, 3, 1)

        self.extent_label = QLabel("Extent Deg:")
        self.extent_spin = QDoubleSpinBox(); self.extent_spin.setRange(0.0, 360.0); self.extent_spin.setDecimals(3)
        form.addWidget(self.extent_label, 4, 0)
        form.addWidget(self.extent_spin, 4, 1)

        self.period_label = QLabel("Period (s):")
        self.period_spin = QDoubleSpinBox(); self.period_spin.setRange(0.001, 10000.0); self.period_spin.setDecimals(3)
        form.addWidget(self.period_label, 5, 0)
        form.addWidget(self.period_spin, 5, 1)

        self.rotate_freq_label = QLabel("Rotate Freq (Hz):")
        self.rotate_freq_spin = QDoubleSpinBox(); self.rotate_freq_spin.setRange(-1000.0, 1000.0); self.rotate_freq_spin.setDecimals(3)
        form.addWidget(self.rotate_freq_label, 6, 0)
        form.addWidget(self.rotate_freq_spin, 6, 1)

        # Distance controls
        self.dist_start_spin = QDoubleSpinBox(); self.dist_start_spin.setRange(0.0, 1000.0); self.dist_start_spin.setDecimals(3)
        self.dist_end_spin = QDoubleSpinBox(); self.dist_end_spin.setRange(0.0, 1000.0); self.dist_end_spin.setDecimals(3)
        self.dist_use_range = QCheckBox("Use range")
        dist_layout = QHBoxLayout()
        dist_layout.addWidget(self.dist_start_spin)
        dist_layout.addWidget(QLabel("to"))
        dist_layout.addWidget(self.dist_end_spin)
        dist_layout.addWidget(self.dist_use_range)
        form.addWidget(QLabel("Distance (m):"), 7, 0)
        form.addLayout(dist_layout, 7, 1)

        self.seconds_spin = QDoubleSpinBox(); self.seconds_spin.setRange(0.001, 10000.0); self.seconds_spin.setDecimals(3)
        form.addWidget(QLabel("Duration (s):"), 8, 0)
        form.addWidget(self.seconds_spin, 8, 1)

        self.easing_combo = QComboBox(); self.easing_combo.addItems(["linear", "sine"])
        form.addWidget(QLabel("Easing:"), 9, 0)
        form.addWidget(self.easing_combo, 9, 1)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._update_mode_fields(self.mode_combo.currentText())

    def _populate(self, seg):
        self.mode_combo.setCurrentText(seg.get("mode", "rotate"))
        self.start_spin.setValue(float(seg.get("start_deg", 0.0)))
        self.speed_spin.setValue(float(seg.get("speed_deg_per_s", 0.0)))
        self.center_spin.setValue(float(seg.get("center_deg", 0.0)))
        self.extent_spin.setValue(float(seg.get("extent_deg", 0.0)))
        self.period_spin.setValue(float(seg.get("period_s", 0.0)))
        self.rotate_freq_spin.setValue(float(seg.get("rotate_freq_hz", 0.0)))
        d = seg.get("distance_m", 1.0)
        if isinstance(d, (list, tuple)) and len(d) == 2:
            self.dist_use_range.setChecked(True)
            self.dist_start_spin.setValue(float(d[0]))
            self.dist_end_spin.setValue(float(d[1]))
        else:
            self.dist_use_range.setChecked(False)
            self.dist_start_spin.setValue(float(d))
            self.dist_end_spin.setValue(float(d))
        self.seconds_spin.setValue(float(seg.get("seconds", 0.0)))
        self.easing_combo.setCurrentText(seg.get("easing", "linear"))
        self._update_mode_fields(self.mode_combo.currentText())

    def _update_mode_fields(self, mode: str):
        def set_vis(show, *widgets):
            for w in widgets:
                w.setVisible(show)
                w.setEnabled(show)

        is_rotate = mode == "rotate"
        is_rotating_arc = mode == "rotating_arc"
        is_oscillate = not (is_rotate or is_rotating_arc)

        set_vis(is_rotate or is_rotating_arc, self.start_label, self.start_spin)
        set_vis(is_rotate, self.speed_label, self.speed_spin)
        set_vis(is_oscillate, self.center_label, self.center_spin)
        set_vis(is_rotating_arc or is_oscillate, self.extent_label, self.extent_spin)
        set_vis(is_oscillate, self.period_label, self.period_spin)
        set_vis(is_rotating_arc, self.rotate_freq_label, self.rotate_freq_spin)

    def get_segment(self) -> dict:
        seg = {"mode": self.mode_combo.currentText()}
        if seg["mode"] == "rotate":
            seg["start_deg"] = float(self.start_spin.value())
            seg["speed_deg_per_s"] = float(self.speed_spin.value())
        elif seg["mode"] == "rotating_arc":
            seg["start_deg"] = float(self.start_spin.value())
            seg["extent_deg"] = float(self.extent_spin.value())
            seg["rotate_freq_hz"] = float(self.rotate_freq_spin.value())
        else:
            seg["center_deg"] = float(self.center_spin.value())
            seg["extent_deg"] = float(self.extent_spin.value())
            seg["period_s"] = float(self.period_spin.value())
        if self.dist_use_range.isChecked():
            seg["distance_m"] = [
                float(self.dist_start_spin.value()),
                float(self.dist_end_spin.value()),
            ]
        else:
            seg["distance_m"] = float(self.dist_start_spin.value())
        seg["seconds"] = float(self.seconds_spin.value())
        seg["easing"] = self.easing_combo.currentText()
        return seg


class SpatialTrajectoryDialog(QDialog):
    """Dialog to manage a list of trajectory segments."""

    def __init__(self, parent=None, segments=None):
        super().__init__(parent)
        self.setWindowTitle("Spatial Trajectory")
        self.segments = copy.deepcopy(segments) if segments else []
        self._setup_ui()
        self._refresh_list()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add")
        edit_btn = QPushButton("Edit")
        remove_btn = QPushButton("Remove")
        add_btn.clicked.connect(self.add_segment)
        edit_btn.clicked.connect(self.edit_segment)
        remove_btn.clicked.connect(self.remove_segment)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(edit_btn)
        btn_row.addWidget(remove_btn)
        layout.addLayout(btn_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _refresh_list(self):
        self.list_widget.clear()
        for seg in self.segments:
            self.list_widget.addItem(self._format_segment(seg))

    def _format_segment(self, seg: dict) -> str:
        if seg.get("mode") == "rotate":
            desc = f"rotate start={seg.get('start_deg', 0)} speed={seg.get('speed_deg_per_s', 0)}"
        elif seg.get("mode") == "rotating_arc":
            desc = (
                f"rotating_arc start={seg.get('start_deg', 0)} extent={seg.get('extent_deg', 0)}"
                f" rot_freq={seg.get('rotate_freq_hz', 0)}"
            )
        else:
            desc = (
                f"oscillate center={seg.get('center_deg', 0)} extent={seg.get('extent_deg', 0)}"
                f" period={seg.get('period_s', 0)}"
            )
        d = seg.get("distance_m", 1.0)
        if isinstance(d, (list, tuple)) and len(d) == 2:
            d_str = f"dist={d[0]}→{d[1]}"
        else:
            d_str = f"dist={d}"
        desc += f" sec={seg.get('seconds', 0)} {d_str}"
        return desc

    def add_segment(self):
        dlg = SpatialTrajectorySegmentDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            self.segments.append(dlg.get_segment())
            self._refresh_list()

    def edit_segment(self):
        row = self.list_widget.currentRow()
        if row < 0 or row >= len(self.segments):
            return
        dlg = SpatialTrajectorySegmentDialog(self, self.segments[row])
        if dlg.exec_() == QDialog.Accepted:
            self.segments[row] = dlg.get_segment()
            self._refresh_list()

    def remove_segment(self):
        row = self.list_widget.currentRow()
        if row < 0 or row >= len(self.segments):
            return
        del self.segments[row]
        self._refresh_list()

    def get_segments(self):
        return self.segments
