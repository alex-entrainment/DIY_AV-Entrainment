from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QLabel,
    QDoubleSpinBox,
    QMessageBox,
    QComboBox,
)
from PyQt5.QtCore import Qt


class SubliminalDialog(QDialog):
    """Dialog to add a subliminal audio voice to a step."""

    def __init__(self, parent=None, app_ref=None, step_index=None):
        super().__init__(parent)
        self.app = app_ref
        self.step_index = step_index
        self.setWindowTitle("Add Subliminal Voice")
        self.resize(400, 0)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()

        file_layout = QHBoxLayout()
        self.file_edit = QLineEdit()
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(self.file_edit, 1)
        file_layout.addWidget(browse_btn)
        form.addRow("Audio File(s):", file_layout)

        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(15000.0, 20000.0)
        self.freq_spin.setValue(17500.0)
        self.freq_spin.setDecimals(1)
        form.addRow("Carrier Freq (Hz):", self.freq_spin)

        self.amp_spin = QDoubleSpinBox()
        self.amp_spin.setRange(0.0, 1.0)
        self.amp_spin.setSingleStep(0.05)
        self.amp_spin.setValue(0.5)
        form.addRow("Amplitude:", self.amp_spin)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["sequence", "stack"])
        form.addRow("Mode:", self.mode_combo)

        layout.addLayout(form)

        btn_row = QHBoxLayout()
        self.ok_btn = QPushButton("Add")
        self.ok_btn.clicked.connect(self.on_accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addStretch(1)
        btn_row.addWidget(self.ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

    def browse_file(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Audio", "", "Audio Files (*.wav *.flac *.mp3)"
        )
        if paths:
            self.file_edit.setText(";".join(paths))

    def on_accept(self):
        raw_paths = self.file_edit.text().strip()
        if not raw_paths:
            QMessageBox.warning(self, "Input Required", "Please select an audio file.")
            return
        paths = [p.strip() for p in raw_paths.split(";") if p.strip()]
        try:
            step = self.app.track_data["steps"][self.step_index]
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Invalid step: {exc}")
            return
        voice_data = {
            "synth_function_name": "subliminal_encode",
            "is_transition": False,
            "params": {
                "carrierFreq": float(self.freq_spin.value()),
                "amp": float(self.amp_spin.value()),
                "mode": self.mode_combo.currentText(),
            },
            "volume_envelope": None,
            "description": "Subliminal",
        }
        if len(paths) == 1:
            voice_data["params"]["audio_path"] = paths[0]
        else:
            voice_data["params"]["audio_paths"] = paths
        step.setdefault("voices", []).append(voice_data)
        self.accept()
