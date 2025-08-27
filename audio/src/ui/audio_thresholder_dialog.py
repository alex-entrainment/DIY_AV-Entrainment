from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QDoubleSpinBox,
    QPushButton,
    QDialogButtonBox,
    QMessageBox,
)
from PyQt5.QtCore import QBuffer, QIODevice
try:
    from PyQt5.QtMultimedia import (
        QAudioOutput,
        QAudioFormat,
        QAudioDeviceInfo,
    )
    QT_MULTIMEDIA_AVAILABLE = True
except Exception as e:  # noqa: PIE786 - broad for missing backends
    print(
        "WARNING: PyQt5.QtMultimedia could not be imported.\n"
        "AudioThresholderDialog will be disabled.\n"
        f"Original error: {e}"
    )
    QT_MULTIMEDIA_AVAILABLE = False

import numpy as np

try:
    from utils.preferences import Preferences
except ImportError:  # running standalone
    from utils.preferences import Preferences
from utils.amp_utils import db_to_amplitude, MIN_DB


class AudioThresholderDialog(QDialog):
    """Determine the user's audio threshold and compute target output level."""

    def __init__(self, prefs: Preferences, parent=None):
        super().__init__(parent)
        self.prefs = prefs or Preferences()
        self.setWindowTitle("Audio Thresholder")

        self.audio_output = None
        self.audio_buffer = None
        self.threshold_db = None

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        level_row = QHBoxLayout()
        level_row.addWidget(QLabel("Test Level:"))
        self.level_spin = QDoubleSpinBox()
        self.level_spin.setRange(MIN_DB, 0.0)
        self.level_spin.setDecimals(1)
        self.level_spin.setSingleStep(1.0)
        self.level_spin.setSuffix(" dB")
        self.level_spin.setValue(-40.0)
        level_row.addWidget(self.level_spin)
        self.play_btn = QPushButton("Play Test Tone")
        self.play_btn.clicked.connect(self.play_test_tone)
        level_row.addWidget(self.play_btn)
        set_btn = QPushButton("Set as Threshold")
        set_btn.clicked.connect(self.set_threshold)
        level_row.addWidget(set_btn)
        layout.addLayout(level_row)

        self.threshold_label = QLabel("Threshold: not set")
        layout.addWidget(self.threshold_label)

        offset_row = QHBoxLayout()
        offset_row.addWidget(QLabel("Offset Above Threshold:"))
        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(0.0, 120.0)
        self.offset_spin.setDecimals(1)
        self.offset_spin.setSingleStep(1.0)
        self.offset_spin.setSuffix(" dB")
        self.offset_spin.setValue(50.0)
        offset_row.addWidget(self.offset_spin)
        layout.addLayout(offset_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        if not QT_MULTIMEDIA_AVAILABLE:
            self.play_btn.setEnabled(False)

    def play_test_tone(self):
        if not QT_MULTIMEDIA_AVAILABLE:
            QMessageBox.critical(
                self,
                "PyQt5 Multimedia Missing",
                "PyQt5.QtMultimedia is required for audio playback, but it could not be loaded.",
            )
            return
        amp = db_to_amplitude(self.level_spin.value())
        sr = int(getattr(self.prefs, "sample_rate", 44100))
        duration = 1.0
        t = np.arange(int(sr * duration)) / sr
        tone = np.sin(2 * np.pi * 1000.0 * t) * amp
        stereo = np.column_stack([tone, tone])
        data = (np.clip(stereo, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

        fmt = QAudioFormat()
        fmt.setCodec("audio/pcm")
        fmt.setSampleRate(sr)
        fmt.setSampleSize(16)
        fmt.setChannelCount(2)
        fmt.setByteOrder(QAudioFormat.LittleEndian)
        fmt.setSampleType(QAudioFormat.SignedInt)
        device_info = QAudioDeviceInfo.defaultOutputDevice()
        if not device_info.isFormatSupported(fmt):
            QMessageBox.warning(self, "Audio Format", "Default output device does not support the required format")
            return

        if self.audio_output:
            self.audio_output.stop()
            self.audio_output = None
        self.audio_output = QAudioOutput(fmt, self)
        self.audio_buffer = QBuffer()
        self.audio_buffer.setData(data)
        self.audio_buffer.open(QIODevice.ReadOnly)
        self.audio_output.start(self.audio_buffer)

    def set_threshold(self):
        self.threshold_db = self.level_spin.value()
        self.threshold_label.setText(f"Threshold: {self.threshold_db:.1f} dBFS")

    def get_target_amplitude(self) -> float:
        thresh = self.threshold_db if self.threshold_db is not None else self.level_spin.value()
        target_db = thresh + self.offset_spin.value()
        if target_db > 0.0:
            target_db = 0.0
        return db_to_amplitude(target_db)
