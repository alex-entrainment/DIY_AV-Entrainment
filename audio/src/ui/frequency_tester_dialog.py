from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QWidget,
    QLabel,
    QCheckBox,
    QDoubleSpinBox,
    QPushButton,
    QMessageBox,
)
from PyQt5.QtCore import Qt, QBuffer, QIODevice
try:
    from PyQt5.QtMultimedia import (
        QAudioOutput,
        QAudioFormat,
        QAudioDeviceInfo,
        QAudio,
    )
    QT_MULTIMEDIA_AVAILABLE = True
except Exception as e:  # noqa: PIE786 - broad for missing backends
    print(
        "WARNING: PyQt5.QtMultimedia could not be imported.\n"
        "FrequencyTesterDialog will be disabled.\n"
        f"Original error: {e}"
    )
    QT_MULTIMEDIA_AVAILABLE = False

import numpy as np

from audio.synth_functions.binaural_beat import binaural_beat
from audio.utils.preferences import Preferences


class FrequencyTesterDialog(QDialog):
    """Dialog for quickly previewing multiple binaural voices."""

    MAX_VOICES = 10
    SEGMENT_DURATION_S = 60

    def __init__(self, parent=None, prefs: Preferences = None):
        super().__init__(parent)
        self.prefs = prefs or Preferences()
        self.audio_output = None
        self.audio_buffer = None

        self.setWindowTitle("Frequency Tester")
        self.resize(600, 0)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.voice_controls = []  # list of dicts
        for i in range(self.MAX_VOICES):
            enabled = QCheckBox(f"Voice {i + 1}")
            base_spin = QDoubleSpinBox()
            base_spin.setRange(20.0, 20000.0)
            base_spin.setValue(200.0)
            base_spin.setDecimals(2)
            beat_spin = QDoubleSpinBox()
            beat_spin.setRange(0.0, 40.0)
            beat_spin.setValue(4.0)
            beat_spin.setDecimals(2)
            amp_spin = QDoubleSpinBox()
            if getattr(self.prefs, "amplitude_display_mode", "absolute") == "dB":
                from audio.utils.amp_utils import amplitude_to_db, MIN_DB
                amp_spin.setRange(MIN_DB, 0.0)
                amp_spin.setDecimals(1)
                amp_spin.setSingleStep(1.0)
                amp_spin.setSuffix(" dB")
                amp_spin.setValue(amplitude_to_db(0.5))
            else:
                amp_spin.setRange(0.0, 1.0)
                amp_spin.setSingleStep(0.05)
                amp_spin.setValue(0.5)
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(enabled)
            row_layout.addWidget(QLabel("Base Freq"))
            row_layout.addWidget(base_spin)
            row_layout.addWidget(QLabel("Beat Freq"))
            row_layout.addWidget(beat_spin)
            row_layout.addWidget(QLabel("Amp"))
            row_layout.addWidget(amp_spin)
            form.addRow(row)
            self.voice_controls.append({
                "enable": enabled,
                "base": base_spin,
                "beat": beat_spin,
                "amp": amp_spin,
            })

        layout.addLayout(form)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.on_start)
        self.stop_btn.clicked.connect(self.on_stop)
        btn_row.addStretch(1)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        layout.addLayout(btn_row)

        if not QT_MULTIMEDIA_AVAILABLE:
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)

    def generate_audio(self):
        sample_rate = int(self.prefs.sample_rate) if hasattr(self.prefs, "sample_rate") else 44100
        duration = self.SEGMENT_DURATION_S
        total_samples = int(duration * sample_rate)
        mix = np.zeros((total_samples, 2), dtype=np.float32)

        any_enabled = False
        for vc in self.voice_controls:
            if vc["enable"].isChecked():
                any_enabled = True
                base = vc["base"].value()
                beat = vc["beat"].value()
                amp = vc["amp"].value()
                if getattr(self.prefs, "amplitude_display_mode", "absolute") == "dB":
                    from audio.utils.amp_utils import db_to_amplitude
                    amp = db_to_amplitude(amp)
                voice = binaural_beat(duration, sample_rate=sample_rate,
                                       ampL=amp, ampR=amp,
                                       baseFreq=base, beatFreq=beat)
                if voice.shape[0] < total_samples:
                    voice = np.pad(voice, ((0, total_samples - voice.shape[0]), (0, 0)))
                mix += voice
        if not any_enabled:
            return None, sample_rate

        peak = np.max(np.abs(mix))
        if peak > 1.0:
            mix /= peak
        audio_int16 = (np.clip(mix, -1.0, 1.0) * 32767).astype(np.int16)
        return audio_int16.tobytes(), sample_rate

    def on_start(self):
        if not QT_MULTIMEDIA_AVAILABLE:
            QMessageBox.critical(
                self,
                "PyQt5 Multimedia Missing",
                "PyQt5.QtMultimedia is required for audio playback, but it "
                "could not be loaded."
            )
            return
        data, sr = self.generate_audio()
        if data is None:
            QMessageBox.warning(self, "Frequency Tester", "No voices enabled")
            return
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
        self.audio_output.stateChanged.connect(self._handle_state_change)

        self.audio_buffer = QBuffer()
        self.audio_buffer.setData(data)
        self.audio_buffer.open(QIODevice.ReadOnly)

        self.audio_output.start(self.audio_buffer)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _handle_state_change(self, state):
        if not QT_MULTIMEDIA_AVAILABLE:
            return
        if state == QAudio.IdleState and self.audio_output and self.audio_buffer:
            # Loop playback
            self.audio_buffer.seek(0)
            self.audio_output.start(self.audio_buffer)

    def on_stop(self):
        if QT_MULTIMEDIA_AVAILABLE and self.audio_output:
            self.audio_output.stop()
            self.audio_output = None
        if self.audio_buffer:
            self.audio_buffer.close()
            self.audio_buffer = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def closeEvent(self, event):
        self.on_stop()
        super().closeEvent(event)
