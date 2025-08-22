from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QDoubleSpinBox,
    QSpinBox,
    QFileDialog,
    QMessageBox,
)

import soundfile as sf

from utils.colored_noise import ColoredNoiseGenerator, plot_spectrogram


class ColoredNoiseDialog(QDialog):
    """Dialog for generating customizable colored noise."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Colored Noise Generator")
        self.resize(400, 0)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        file_layout = QHBoxLayout()
        self.file_edit = QLineEdit("colored_noise.wav")
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(self.file_edit, 1)
        file_layout.addWidget(browse_btn)
        form.addRow("Output File:", file_layout)

        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.1, 3600.0)
        self.duration_spin.setValue(60.0)
        form.addRow("Duration (s):", self.duration_spin)

        self.sample_rate_spin = QSpinBox()
        self.sample_rate_spin.setRange(8000, 192000)
        self.sample_rate_spin.setValue(44100)
        form.addRow("Sample Rate:", self.sample_rate_spin)

        self.exponent_spin = QDoubleSpinBox()
        self.exponent_spin.setRange(-3.0, 3.0)
        self.exponent_spin.setValue(1.0)
        form.addRow("Exponent:", self.exponent_spin)

        self.lowcut_spin = QDoubleSpinBox()
        self.lowcut_spin.setRange(0.0, 20000.0)
        self.lowcut_spin.setValue(0.0)
        form.addRow("Low Cut (Hz):", self.lowcut_spin)

        self.highcut_spin = QDoubleSpinBox()
        self.highcut_spin.setRange(0.0, 20000.0)
        self.highcut_spin.setValue(0.0)
        form.addRow("High Cut (Hz):", self.highcut_spin)

        self.amplitude_spin = QDoubleSpinBox()
        self.amplitude_spin.setRange(0.0, 10.0)
        self.amplitude_spin.setValue(1.0)
        form.addRow("Amplitude:", self.amplitude_spin)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(-1, 2**31 - 1)
        self.seed_spin.setValue(-1)
        form.addRow("Seed:", self.seed_spin)

        layout.addLayout(form)

        button_layout = QHBoxLayout()
        self.spectro_btn = QPushButton("Spectrogram")
        self.spectro_btn.clicked.connect(self.on_spectrogram)
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.clicked.connect(self.on_generate)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.spectro_btn)
        button_layout.addStretch(1)
        button_layout.addWidget(close_btn)
        button_layout.addWidget(self.generate_btn)
        layout.addLayout(button_layout)

    def browse_file(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Audio", "", "WAV Files (*.wav)")
        if path:
            self.file_edit.setText(path)

    def _collect_params(self) -> ColoredNoiseGenerator:
        lowcut = self.lowcut_spin.value() or None
        highcut = self.highcut_spin.value() or None
        seed_val = self.seed_spin.value()
        seed = seed_val if seed_val != -1 else None
        return ColoredNoiseGenerator(
            sample_rate=int(self.sample_rate_spin.value()),
            duration=float(self.duration_spin.value()),
            exponent=float(self.exponent_spin.value()),
            lowcut=lowcut,
            highcut=highcut,
            amplitude=float(self.amplitude_spin.value()),
            seed=seed,
        )

    def on_generate(self) -> None:
        try:
            gen = self._collect_params()
            noise = gen.generate()
            sf.write(self.file_edit.text() or "colored_noise.wav", noise, gen.sample_rate)
            QMessageBox.information(self, "Success", f"Generated {self.file_edit.text()}")
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

    def on_spectrogram(self) -> None:
        try:
            gen = self._collect_params()
            noise = gen.generate()
            plot_spectrogram(noise, gen.sample_rate)
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))
