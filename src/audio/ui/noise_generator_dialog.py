from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QHBoxLayout, QLineEdit,
    QPushButton, QFileDialog, QMessageBox, QLabel, QDoubleSpinBox,
    QSpinBox
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
        self.file_edit = QLineEdit("swept_notch_pink_sound.wav")
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(self.file_edit, 1)
        file_layout.addWidget(browse_btn)
        form.addRow("Output File:", file_layout)

        # Duration
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(1.0, 100000.0)
        self.duration_spin.setValue(60.0)
        form.addRow("Duration (s):", self.duration_spin)

        # Sample rate
        self.sample_rate_spin = QSpinBox()
        self.sample_rate_spin.setRange(8000, 192000)
        self.sample_rate_spin.setValue(44100)
        form.addRow("Sample Rate:", self.sample_rate_spin)

        # LFO freq
        self.lfo_spin = QDoubleSpinBox()
        self.lfo_spin.setRange(0.001, 10.0)
        self.lfo_spin.setDecimals(4)
        self.lfo_spin.setValue(1.0 / 12.0)
        form.addRow("LFO Freq (Hz):", self.lfo_spin)

        # Min freq
        self.min_freq_spin = QSpinBox()
        self.min_freq_spin.setRange(20, 20000)
        self.min_freq_spin.setValue(1000)
        form.addRow("Min Sweep Freq:", self.min_freq_spin)

        # Max freq
        self.max_freq_spin = QSpinBox()
        self.max_freq_spin.setRange(20, 22050)
        self.max_freq_spin.setValue(10000)
        form.addRow("Max Sweep Freq:", self.max_freq_spin)

        # Number of notches
        self.num_notches_spin = QSpinBox()
        self.num_notches_spin.setRange(1, 20)
        self.num_notches_spin.setValue(6)
        form.addRow("Num Notches:", self.num_notches_spin)

        # Notch spacing ratio
        self.notch_spacing_spin = QDoubleSpinBox()
        self.notch_spacing_spin.setRange(1.0, 2.0)
        self.notch_spacing_spin.setDecimals(3)
        self.notch_spacing_spin.setSingleStep(0.01)
        self.notch_spacing_spin.setValue(1.1)
        form.addRow("Notch Spacing Ratio:", self.notch_spacing_spin)

        # Notch Q factor
        self.notch_q_spin = QSpinBox()
        self.notch_q_spin.setRange(1, 1000)
        self.notch_q_spin.setValue(100)
        form.addRow("Notch Q:", self.notch_q_spin)

        # Cascade count
        self.cascade_count_spin = QSpinBox()
        self.cascade_count_spin.setRange(1, 20)
        self.cascade_count_spin.setValue(3)
        form.addRow("Cascade Count:", self.cascade_count_spin)

        # LFO phase offset
        self.lfo_phase_spin = QSpinBox()
        self.lfo_phase_spin.setRange(0, 360)
        self.lfo_phase_spin.setValue(90)
        form.addRow("LFO Phase Offset (deg):", self.lfo_phase_spin)

        # Optional input file
        input_layout = QHBoxLayout()
        self.input_file_edit = QLineEdit()
        input_browse = QPushButton("Browse")
        input_browse.clicked.connect(self.browse_input_file)
        input_layout.addWidget(self.input_file_edit, 1)
        input_layout.addWidget(input_browse)
        form.addRow("Input Audio (optional):", input_layout)

        layout.addLayout(form)

        self.generate_btn = QPushButton("Generate")
        self.generate_btn.clicked.connect(self.on_generate)
        layout.addWidget(self.generate_btn, alignment=Qt.AlignRight)

    def browse_file(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Audio", "", "WAV Files (*.wav)")
        if path:
            self.file_edit.setText(path)

    def browse_input_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Audio", "", "Audio Files (*.wav *.flac *.mp3)")
        if path:
            self.input_file_edit.setText(path)

    def on_generate(self):
        filename = self.file_edit.text() or "swept_notch_pink_sound.wav"
        input_path = self.input_file_edit.text() or None
        try:
            generate_swept_notch_pink_sound(
                filename=filename,
                duration_seconds=float(self.duration_spin.value()),
                sample_rate=int(self.sample_rate_spin.value()),
                lfo_freq=float(self.lfo_spin.value()),
                min_freq=int(self.min_freq_spin.value()),
                max_freq=int(self.max_freq_spin.value()),
                num_notches=int(self.num_notches_spin.value()),
                notch_spacing_ratio=float(self.notch_spacing_spin.value()),
                notch_q=int(self.notch_q_spin.value()),
                cascade_count=int(self.cascade_count_spin.value()),
                lfo_phase_offset_deg=int(self.lfo_phase_spin.value()),
                input_audio_path=input_path,
            )
            QMessageBox.information(self, "Success", f"Generated {filename}")
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

