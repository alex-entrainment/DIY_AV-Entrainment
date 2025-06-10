from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QHBoxLayout, QLineEdit,
    QPushButton, QFileDialog, QMessageBox, QLabel, QDoubleSpinBox,
    QSpinBox
)
from PyQt5.QtCore import Qt

from .synth_functions.noise_flanger import generate_swept_notch_pink_sound


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

        layout.addLayout(form)

        self.generate_btn = QPushButton("Generate")
        self.generate_btn.clicked.connect(self.on_generate)
        layout.addWidget(self.generate_btn, alignment=Qt.AlignRight)

    def browse_file(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Audio", "", "WAV Files (*.wav)")
        if path:
            self.file_edit.setText(path)

    def on_generate(self):
        filename = self.file_edit.text() or "swept_notch_pink_sound.wav"
        try:
            generate_swept_notch_pink_sound(
                filename=filename,
                duration_seconds=float(self.duration_spin.value()),
                sample_rate=int(self.sample_rate_spin.value()),
                lfo_freq=float(self.lfo_spin.value()),
                min_freq=int(self.min_freq_spin.value()),
                max_freq=int(self.max_freq_spin.value()),
            )
            QMessageBox.information(self, "Success", f"Generated {filename}")
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

