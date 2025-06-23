from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QHBoxLayout,
    QLineEdit, QPushButton, QFileDialog, QLabel,
    QDoubleSpinBox, QMessageBox
)
from PyQt5.QtCore import Qt
import soundfile as sf


class OverlayClipDialog(QDialog):
    """Dialog to add or edit an overlay clip entry."""

    def __init__(self, parent=None, clip_data=None):
        super().__init__(parent)
        self.setWindowTitle("Overlay Clip")
        self.clip_data = clip_data or {}

        self._setup_ui()
        if clip_data:
            self._populate_from_data(clip_data)

    def _get_clip_duration(self, path: str) -> float:
        """Return the duration of the audio clip at ``path`` in seconds."""
        try:
            info = sf.info(path)
            if info.samplerate > 0:
                return info.frames / float(info.samplerate)
        except Exception:
            pass
        return 0.0

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()

        file_layout = QHBoxLayout()
        self.file_edit = QLineEdit()
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(self.file_edit, 1)
        file_layout.addWidget(browse_btn)
        form.addRow("Audio File:", file_layout)

        self.desc_edit = QLineEdit()
        form.addRow("Description:", self.desc_edit)

        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(0.0, 99999.0)
        self.start_spin.setDecimals(3)
        form.addRow("Start Time (s):", self.start_spin)

        self.amp_spin = QDoubleSpinBox()
        self.amp_spin.setRange(0.0, 10.0)
        self.amp_spin.setSingleStep(0.1)
        self.amp_spin.setValue(1.0)
        form.addRow("Amplitude:", self.amp_spin)

        self.pan_spin = QDoubleSpinBox()
        self.pan_spin.setRange(-1.0, 1.0)
        self.pan_spin.setSingleStep(0.1)
        form.addRow("Pan:", self.pan_spin)

        self.fade_in_spin = QDoubleSpinBox()
        self.fade_in_spin.setRange(0.0, 60.0)
        self.fade_in_spin.setDecimals(3)
        form.addRow("Fade In (s):", self.fade_in_spin)

        self.fade_out_spin = QDoubleSpinBox()
        self.fade_out_spin.setRange(0.0, 60.0)
        self.fade_out_spin.setDecimals(3)
        form.addRow("Fade Out (s):", self.fade_out_spin)

        layout.addLayout(form)

        btn_row = QHBoxLayout()
        self.ok_btn = QPushButton("OK")
        self.ok_btn.setDefault(True)
        self.ok_btn.clicked.connect(self.on_accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addStretch(1)
        btn_row.addWidget(self.ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

    def _populate_from_data(self, data):
        self.file_edit.setText(data.get("file_path", ""))
        self.start_spin.setValue(float(data.get("start", 0.0)))
        self.amp_spin.setValue(float(data.get("amp", 1.0)))
        self.pan_spin.setValue(float(data.get("pan", 0.0)))
        self.fade_in_spin.setValue(float(data.get("fade_in", 0.0)))
        self.fade_out_spin.setValue(float(data.get("fade_out", 0.0)))
        self.desc_edit.setText(data.get("description", ""))

    def browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", "Audio Files (*.wav *.flac *.mp3)"
        )
        if path:
            self.file_edit.setText(path)

    def on_accept(self):
        path = self.file_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "Input Required", "Please select an audio file.")
            return
        duration = self._get_clip_duration(path)
        self.clip_data = {
            "file_path": path,
            "start": float(self.start_spin.value()),
            "duration": duration,
            "amp": float(self.amp_spin.value()),
            "pan": float(self.pan_spin.value()),
            "fade_in": float(self.fade_in_spin.value()),
            "fade_out": float(self.fade_out_spin.value()),
            "description": self.desc_edit.text().strip(),
        }
        self.accept()

    def get_clip_data(self):
        return self.clip_data
