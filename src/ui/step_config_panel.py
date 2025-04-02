from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QFormLayout,
                             QDoubleSpinBox, QLineEdit, QComboBox, QPushButton,
                             QScrollArea, QHBoxLayout, QLabel, QSpinBox, QCheckBox)
from PyQt5.QtCore import Qt, pyqtSignal
from sequence_model import PatternMode, Waveform

def clear_layout(layout):
    if layout is not None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

# Mapping for pattern modes.
PATTERN_MODE_LABELS = [
    ("NONE", PatternMode.NONE),
    ("SACRED_GEOMETRY", PatternMode.SACRED_GEOMETRY),
    ("FRACTAL_ARC", PatternMode.FRACTAL_ARC),
    ("PHI_SPIRAL", PatternMode.PHI_SPIRAL)
]

class StepConfigPanel(QWidget):
    submitClicked = pyqtSignal()  # Emitted when "Submit Step Settings" is clicked

    def __init__(self, parent=None):
        super().__init__(parent)
        # Use a scroll area for the step config.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        self.layout = QVBoxLayout(container)

        # Step Info Group.
        self.step_info_group = QGroupBox("Step Info")
        form_step_info = QFormLayout()
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.01, 6000)
        self.duration_spin.setValue(30)
        self.description_edit = QLineEdit()
        form_step_info.addRow("Duration (secs):", self.duration_spin)
        form_step_info.addRow("Description:", self.description_edit)
        self.step_info_group.setLayout(form_step_info)
        self.layout.addWidget(self.step_info_group)

        # Oscillator Mode Group.
        self.mode_group = QGroupBox("Oscillator Mode")
        mode_layout = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Combined", "Split", "Independent"])
        mode_layout.addWidget(QLabel("Select Mode:"))
        mode_layout.addWidget(self.mode_combo)
        self.mode_group.setLayout(mode_layout)
        self.layout.addWidget(self.mode_group)

        # Oscillator Settings container.
        self.osc_group = QGroupBox("Oscillator Settings")
        self.osc_layout = QVBoxLayout()
        self.osc_group.setLayout(self.osc_layout)
        self.layout.addWidget(self.osc_group)

        # Apply button for oscillator settings.
        self.btn_apply_all = QPushButton("Apply Oscillator 1 Settings to All")
        self.layout.addWidget(self.btn_apply_all)

        # Strobe Intensities container.
        self.strobe_group = QGroupBox("Strobe Intensities")
        self.strobe_layout = QHBoxLayout()
        self.strobe_group.setLayout(self.strobe_layout)
        self.layout.addWidget(self.strobe_group)

        # Submit button.
        self.btn_submit = QPushButton("Submit Step Settings")
        self.layout.addWidget(self.btn_submit)

        self.layout.addStretch()
        scroll.setWidget(container)
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll)

        # Containers for dynamically created oscillator and strobe controls.
        self.osc_controls = []
        self.strobe_controls = []

    def clear_oscillator_controls(self):
        clear_layout(self.osc_layout)
        self.osc_controls = []

    def clear_strobe_controls(self):
        clear_layout(self.strobe_layout)
        self.strobe_controls = []

    def build_oscillator_and_strobe_controls(self, mode):
        self.clear_oscillator_controls()
        self.clear_strobe_controls()

        if mode == "Combined":
            num_osc = 1
            osc_labels = ["Oscillator (All 6 LEDs)"]
            strobe_labels = ["Strobe (All 6 LEDs)"]
        elif mode == "Split":
            num_osc = 2
            osc_labels = ["Osc Group 1 (LEDs 0,2,4)", "Osc Group 2 (LEDs 1,3,5)"]
            strobe_labels = ["Strobe (LEDs 0,2,4)", "Strobe (LEDs 1,3,5)"]
        else:  # Independent
            num_osc = 6
            osc_labels = [f"Oscillator LED {i}" for i in range(6)]
            strobe_labels = [f"Strobe LED {i}" for i in range(6)]

        # Build oscillator controls.
        for i in range(num_osc):
            group = QGroupBox(osc_labels[i])
            form = QFormLayout()
            wave_combo = QComboBox()
            wave_combo.addItems(["Off", "Square", "Sine"])
            freq_start = QDoubleSpinBox()
            freq_start.setRange(0.01, 200.0)
            freq_start.setDecimals(2)
            freq_start.setValue(10.0)
            freq_end = QDoubleSpinBox()
            freq_end.setRange(0.01, 200.0)
            freq_end.setDecimals(2)
            freq_end.setValue(10.0)
            duty_start = QSpinBox()
            duty_start.setRange(1, 99)
            duty_start.setValue(50)
            duty_end = QSpinBox()
            duty_end.setRange(1, 99)
            duty_end.setValue(50)
            rfm_enable = QCheckBox("Enable RFM (LED)")
            rfm_range = QDoubleSpinBox()
            rfm_range.setRange(0.0, 5.0)
            rfm_range.setValue(0.5)
            rfm_speed = QDoubleSpinBox()
            rfm_speed.setRange(0.0, 2.0)
            rfm_speed.setValue(0.2)
            rfm_help_btn = QPushButton("?")
            rfm_help_btn.setMaximumWidth(30)
            rfm_layout = QHBoxLayout()
            rfm_layout.addWidget(rfm_enable)
            rfm_layout.addStretch()
            rfm_layout.addWidget(rfm_help_btn)

            form.addRow("Waveform:", wave_combo)
            form.addRow("Freq Start (Hz):", freq_start)
            form.addRow("Freq End (Hz):", freq_end)
            form.addRow("Duty Start (%):", duty_start)
            form.addRow("Duty End (%):", duty_end)
            form.addRow(rfm_layout)
            form.addRow("RFM Range (±Hz):", rfm_range)
            form.addRow("RFM Speed (Hz/s):", rfm_speed)

            # Pattern controls.
            phase_pattern_combo = QComboBox()
            brightness_pattern_combo = QComboBox()
            for label, _ in PATTERN_MODE_LABELS:
                phase_pattern_combo.addItem(label)
                brightness_pattern_combo.addItem(label)
            pattern_strength_spin = QDoubleSpinBox()
            pattern_strength_spin.setRange(0.0, 10.0)
            pattern_strength_spin.setValue(1.0)
            pattern_freq_spin = QDoubleSpinBox()
            pattern_freq_spin.setRange(0.0, 10.0)
            pattern_freq_spin.setValue(1.0)
            form.addRow("Phase Pattern:", phase_pattern_combo)
            form.addRow("Brightness Pattern:", brightness_pattern_combo)
            form.addRow("Pattern Strength:", pattern_strength_spin)
            form.addRow("Pattern Frequency:", pattern_freq_spin)

            group.setLayout(form)
            self.osc_layout.addWidget(group)

            self.osc_controls.append({
                "wave_combo": wave_combo,
                "freq_start": freq_start,
                "freq_end": freq_end,
                "duty_start": duty_start,
                "duty_end": duty_end,
                "rfm_enable": rfm_enable,
                "rfm_range": rfm_range,
                "rfm_speed": rfm_speed,
                "phase_pattern_combo": phase_pattern_combo,
                "brightness_pattern_combo": brightness_pattern_combo,
                "pattern_strength_spin": pattern_strength_spin,
                "pattern_freq_spin": pattern_freq_spin,
                "rfm_help_btn": rfm_help_btn
            })

        # Build strobe controls.
        for i in range(num_osc):
            group = QGroupBox(strobe_labels[i])
            form = QFormLayout()
            strobe_start = QSpinBox()
            strobe_start.setRange(0, 100)
            strobe_start.setValue(0)
            strobe_end = QSpinBox()
            strobe_end.setRange(0, 100)
            strobe_end.setValue(0)
            form.addRow("Start (%):", strobe_start)
            form.addRow("End (%):", strobe_end)
            group.setLayout(form)
            self.strobe_layout.addWidget(group)

            self.strobe_controls.append({
                "strobe_start": strobe_start,
                "strobe_end": strobe_end
            })

    def clear_step_fields(self):
        self.duration_spin.setValue(30)
        self.description_edit.clear()
        for ctrl in self.osc_controls:
            ctrl["wave_combo"].setCurrentIndex(0)
            ctrl["freq_start"].setValue(12.0)
            ctrl["freq_end"].setValue(12.0)
            ctrl["duty_start"].setValue(50)
            ctrl["duty_end"].setValue(50)
            ctrl["rfm_enable"].setChecked(False)
            ctrl["rfm_range"].setValue(0.5)
            ctrl["rfm_speed"].setValue(0.2)
            ctrl["phase_pattern_combo"].setCurrentIndex(0)
            ctrl["brightness_pattern_combo"].setCurrentIndex(0)
            ctrl["pattern_strength_spin"].setValue(1.0)
            ctrl["pattern_freq_spin"].setValue(1.0)
        for ctrl in self.strobe_controls:
            ctrl["strobe_start"].setValue(0)
            ctrl["strobe_end"].setValue(0)
