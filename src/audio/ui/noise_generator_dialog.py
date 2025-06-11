from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QHBoxLayout, QLineEdit,
    QPushButton, QFileDialog, QMessageBox, QLabel, QDoubleSpinBox,
    QSpinBox, QComboBox, QWidget
)
from PyQt5.QtCore import Qt

from synth_functions.noise_flanger import generate_swept_notch_pink_sound


class NoiseGeneratorDialog(QDialog):
    """Simple GUI for generating swept notch noise."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Noise Generator")
        self.resize(400, 0)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        # Output file
        file_layout = QHBoxLayout()
        self.file_edit = QLineEdit("swept_notch_noise.wav")
        self.file_edit.setToolTip("Where to save the generated audio file")
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(self.file_edit, 1)
        file_layout.addWidget(browse_btn)
        form.addRow("Output File:", file_layout)

        # Duration
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(1.0, 100000.0)
        self.duration_spin.setValue(60.0)
        self.duration_spin.setToolTip("Length of the output audio in seconds")
        form.addRow("Duration (s):", self.duration_spin)

        # Sample rate
        self.sample_rate_spin = QSpinBox()
        self.sample_rate_spin.setRange(8000, 192000)
        self.sample_rate_spin.setValue(44100)
        self.sample_rate_spin.setToolTip("Samples per second of the output file")
        form.addRow("Sample Rate:", self.sample_rate_spin)

        # Noise type
        self.noise_type_combo = QComboBox()
        self.noise_type_combo.addItems(["Pink", "Brown"])
        self.noise_type_combo.setToolTip("Base noise colour to generate")
        form.addRow("Noise Type:", self.noise_type_combo)

        # LFO waveform
        self.lfo_waveform_combo = QComboBox()
        self.lfo_waveform_combo.addItems(["Sine", "Triangle"])
        self.lfo_waveform_combo.setToolTip("Shape of the LFO controlling the sweep")
        form.addRow("LFO Waveform:", self.lfo_waveform_combo)

        # LFO freq
        self.lfo_spin = QDoubleSpinBox()
        self.lfo_spin.setRange(0.001, 10.0)
        self.lfo_spin.setDecimals(4)
        self.lfo_spin.setValue(1.0 / 12.0)
        self.lfo_spin.setToolTip("Rate of the sweeping notch movement")
        form.addRow("LFO Freq (Hz):", self.lfo_spin)

        # Number of sweeps
        self.num_sweeps_spin = QSpinBox()
        self.num_sweeps_spin.setRange(1, 3)
        self.num_sweeps_spin.setValue(1)
        self.num_sweeps_spin.setToolTip("How many independent sweeps to apply")
        self.num_sweeps_spin.valueChanged.connect(self.update_sweep_visibility)
        form.addRow("Num Sweeps:", self.num_sweeps_spin)

        # Sweep frequency ranges
        self.sweep_rows = []
        default_values = [(1000, 10000), (500, 1000), (1850, 3350)]
        for i in range(3):
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)

            min_spin = QSpinBox()
            min_spin.setRange(20, 20000)
            min_spin.setValue(default_values[i][0])
            min_spin.setToolTip(f"Minimum frequency for sweep {i+1}")

            max_spin = QSpinBox()
            max_spin.setRange(20, 22050)
            max_spin.setValue(default_values[i][1])
            max_spin.setToolTip(f"Maximum frequency for sweep {i+1}")

            row_layout.addWidget(QLabel("Min:"))
            row_layout.addWidget(min_spin)
            row_layout.addWidget(QLabel("Max:"))
            row_layout.addWidget(max_spin)

            form.addRow(f"Sweep {i+1}:", row_widget)
            self.sweep_rows.append((row_widget, min_spin, max_spin))

        self.update_sweep_visibility(self.num_sweeps_spin.value())

        # Notch Q factor
        self.notch_q_spin = QSpinBox()
        self.notch_q_spin.setRange(1, 1000)
        self.notch_q_spin.setValue(10)
        self.notch_q_spin.setToolTip("Q factor - higher values give narrower notches")
        form.addRow("Notch Q:", self.notch_q_spin)

        # Cascade count
        self.cascade_count_spin = QSpinBox()
        self.cascade_count_spin.setRange(1, 20)
        self.cascade_count_spin.setValue(5)
        self.cascade_count_spin.setToolTip("Number of times each notch filter is applied")
        form.addRow("Cascade Count:", self.cascade_count_spin)

        # LFO phase offset
        self.lfo_phase_spin = QSpinBox()
        self.lfo_phase_spin.setRange(0, 360)
        self.lfo_phase_spin.setValue(0)
        self.lfo_phase_spin.setToolTip("Phase offset for the right channel sweep")
        form.addRow("LFO Phase Offset (deg):", self.lfo_phase_spin)

        # Intra-channel offset
        self.intra_phase_spin = QSpinBox()
        self.intra_phase_spin.setRange(0, 360)
        self.intra_phase_spin.setValue(0)
        self.intra_phase_spin.setToolTip(
            "Phase offset between two swept filters in each channel"
        )
        form.addRow("Intra-Phase Offset (deg):", self.intra_phase_spin)

        # Optional input file
        input_layout = QHBoxLayout()
        self.input_file_edit = QLineEdit()
        self.input_file_edit.setToolTip("Optional file to process instead of generated noise")
        input_browse = QPushButton("Browse")
        input_browse.clicked.connect(self.browse_input_file)
        input_layout.addWidget(self.input_file_edit, 1)
        input_layout.addWidget(input_browse)
        form.addRow("Input Audio (optional):", input_layout)

        layout.addLayout(form)

        self.generate_btn = QPushButton("Generate")
        self.generate_btn.clicked.connect(self.on_generate)
        layout.addWidget(self.generate_btn, alignment=Qt.AlignRight)

    def update_sweep_visibility(self, count):
        for i, (row_widget, _min, _max) in enumerate(self.sweep_rows):
            row_widget.setVisible(i < count)

    def browse_file(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Audio", "", "WAV Files (*.wav)")
        if path:
            self.file_edit.setText(path)

    def browse_input_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Audio", "", "Audio Files (*.wav *.flac *.mp3)")
        if path:
            self.input_file_edit.setText(path)

    def on_generate(self):
        filename = self.file_edit.text() or "swept_notch_noise.wav"
        input_path = self.input_file_edit.text() or None
        try:
            sweeps = []
            for i in range(self.num_sweeps_spin.value()):
                _, min_spin, max_spin = self.sweep_rows[i]
                sweeps.append((int(min_spin.value()), int(max_spin.value())))

            generate_swept_notch_pink_sound(
                filename=filename,
                duration_seconds=float(self.duration_spin.value()),
                sample_rate=int(self.sample_rate_spin.value()),
                lfo_freq=float(self.lfo_spin.value()),
                filter_sweeps=sweeps,
                notch_q=int(self.notch_q_spin.value()),
                cascade_count=int(self.cascade_count_spin.value()),
                lfo_phase_offset_deg=int(self.lfo_phase_spin.value()),
                intra_phase_offset_deg=int(self.intra_phase_spin.value()),
                input_audio_path=input_path,
                noise_type=self.noise_type_combo.currentText().lower(),
                lfo_waveform=self.lfo_waveform_combo.currentText().lower(),
            )
            QMessageBox.information(self, "Success", f"Generated {filename}")
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

