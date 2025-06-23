from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QLabel,
    QDoubleSpinBox,
    QSpinBox,
    QComboBox,
    QWidget,
    QCheckBox,
    QGridLayout,
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
        "NoiseGeneratorDialog will have audio preview disabled.\n"
        f"Original error: {e}"
    )
    QT_MULTIMEDIA_AVAILABLE = False

import numpy as np

from synth_functions.noise_flanger import (
    generate_swept_notch_pink_sound,
    generate_swept_notch_pink_sound_transition,
    _generate_swept_notch_arrays,
    _generate_swept_notch_arrays_transition,
)

from utils.noise_file import (
    NoiseParams,
    save_noise_params,
    load_noise_params,
    NOISE_FILE_EXTENSION,
)


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

        # Transition enable
        self.transition_check = QCheckBox("Enable Transition")
        form.addRow("Transition:", self.transition_check)

        # LFO waveform
        self.lfo_waveform_combo = QComboBox()
        self.lfo_waveform_combo.addItems(["Sine", "Triangle"])
        self.lfo_waveform_combo.setToolTip("Shape of the LFO controlling the sweep")
        form.addRow("LFO Waveform:", self.lfo_waveform_combo)

        # LFO freq start/end
        lfo_layout = QHBoxLayout()
        self.lfo_start_spin = QDoubleSpinBox()
        self.lfo_start_spin.setRange(0.001, 10.0)
        self.lfo_start_spin.setDecimals(4)
        self.lfo_start_spin.setValue(1.0 / 12.0)
        self.lfo_start_spin.setToolTip("Start LFO frequency")
        self.lfo_end_spin = QDoubleSpinBox()
        self.lfo_end_spin.setRange(0.001, 10.0)
        self.lfo_end_spin.setDecimals(4)
        self.lfo_end_spin.setValue(1.0 / 12.0)
        self.lfo_end_spin.setToolTip("End LFO frequency")
        lfo_layout.addWidget(QLabel("Start:"))
        lfo_layout.addWidget(self.lfo_start_spin)
        lfo_layout.addWidget(QLabel("End:"))
        lfo_layout.addWidget(self.lfo_end_spin)
        form.addRow("LFO Freq (Hz):", lfo_layout)

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
            row_layout = QGridLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)

            s_min = QSpinBox(); s_min.setRange(20, 20000); s_min.setValue(default_values[i][0])
            e_min = QSpinBox(); e_min.setRange(20, 20000); e_min.setValue(default_values[i][0])
            s_max = QSpinBox(); s_max.setRange(20, 22050); s_max.setValue(default_values[i][1])
            e_max = QSpinBox(); e_max.setRange(20, 22050); e_max.setValue(default_values[i][1])
            s_q = QSpinBox(); s_q.setRange(1, 1000); s_q.setValue(25)
            e_q = QSpinBox(); e_q.setRange(1, 1000); e_q.setValue(25)
            s_casc = QSpinBox(); s_casc.setRange(1, 20); s_casc.setValue(10)
            e_casc = QSpinBox(); e_casc.setRange(1, 20); e_casc.setValue(10)

            row_layout.addWidget(QLabel("Start Min:"), 0, 0)
            row_layout.addWidget(s_min, 0, 1)
            row_layout.addWidget(QLabel("End Min:"), 0, 2)
            row_layout.addWidget(e_min, 0, 3)
            row_layout.addWidget(QLabel("Start Max:"), 1, 0)
            row_layout.addWidget(s_max, 1, 1)
            row_layout.addWidget(QLabel("End Max:"), 1, 2)
            row_layout.addWidget(e_max, 1, 3)
            row_layout.addWidget(QLabel("Start Q:"), 2, 0)
            row_layout.addWidget(s_q, 2, 1)
            row_layout.addWidget(QLabel("End Q:"), 2, 2)
            row_layout.addWidget(e_q, 2, 3)
            row_layout.addWidget(QLabel("Start Casc:"), 3, 0)
            row_layout.addWidget(s_casc, 3, 1)
            row_layout.addWidget(QLabel("End Casc:"), 3, 2)
            row_layout.addWidget(e_casc, 3, 3)

            form.addRow(f"Sweep {i+1}:", row_widget)
            self.sweep_rows.append(
                (row_widget, s_min, e_min, s_max, e_max, s_q, e_q, s_casc, e_casc)
            )

        self.update_sweep_visibility(self.num_sweeps_spin.value())


        # LFO phase offset start/end
        phase_layout = QHBoxLayout()
        self.lfo_phase_start_spin = QSpinBox(); self.lfo_phase_start_spin.setRange(0, 360); self.lfo_phase_start_spin.setValue(0)
        self.lfo_phase_end_spin = QSpinBox(); self.lfo_phase_end_spin.setRange(0, 360); self.lfo_phase_end_spin.setValue(0)
        phase_layout.addWidget(QLabel("Start:"))
        phase_layout.addWidget(self.lfo_phase_start_spin)
        phase_layout.addWidget(QLabel("End:"))
        phase_layout.addWidget(self.lfo_phase_end_spin)
        form.addRow("LFO Phase Offset (deg):", phase_layout)

        # Intra-channel offset start/end
        intra_layout = QHBoxLayout()
        self.intra_phase_start_spin = QSpinBox(); self.intra_phase_start_spin.setRange(0, 360); self.intra_phase_start_spin.setValue(0)
        self.intra_phase_end_spin = QSpinBox(); self.intra_phase_end_spin.setRange(0, 360); self.intra_phase_end_spin.setValue(0)
        intra_layout.addWidget(QLabel("Start:"))
        intra_layout.addWidget(self.intra_phase_start_spin)
        intra_layout.addWidget(QLabel("End:"))
        intra_layout.addWidget(self.intra_phase_end_spin)
        form.addRow("Intra-Phase Offset (deg):", intra_layout)

        # Initial/Post transition offsets
        offset_layout = QHBoxLayout()
        self.initial_offset_spin = QDoubleSpinBox()
        self.initial_offset_spin.setRange(0.0, 10000.0)
        self.initial_offset_spin.setDecimals(3)
        self.initial_offset_spin.setValue(0.0)
        self.initial_offset_spin.setToolTip("Time before transition starts")
        self.post_offset_spin = QDoubleSpinBox()
        self.post_offset_spin.setRange(0.0, 10000.0)
        self.post_offset_spin.setDecimals(3)
        self.post_offset_spin.setValue(0.0)
        self.post_offset_spin.setToolTip("Time after transition ends")
        offset_layout.addWidget(QLabel("Init:"))
        offset_layout.addWidget(self.initial_offset_spin)
        offset_layout.addWidget(QLabel("Post:"))
        offset_layout.addWidget(self.post_offset_spin)
        form.addRow("Offsets (s):", offset_layout)

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

        button_row = QHBoxLayout()
        self.load_btn = QPushButton("Load")
        self.load_btn.clicked.connect(self.load_settings)
        self.save_btn = QPushButton("Save")
        self.save_btn.setDefault(True)
        self.save_btn.clicked.connect(self.save_settings)
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.clicked.connect(self.on_generate)
        self.test_btn = QPushButton("Test")
        self.test_btn.clicked.connect(self.on_test)
        self.audio_output = None
        self.audio_buffer = None
        button_row.addWidget(self.load_btn)
        button_row.addWidget(self.save_btn)
        button_row.addStretch(1)
        button_row.addWidget(self.test_btn)
        button_row.addWidget(self.generate_btn)
        layout.addLayout(button_row)

        if not QT_MULTIMEDIA_AVAILABLE:
            self.test_btn.setEnabled(False)

    def update_sweep_visibility(self, count):
        for i, (row_widget, *_rest) in enumerate(self.sweep_rows):
            row_widget.setVisible(i < count)

    def browse_file(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Audio", "", "WAV Files (*.wav)")
        if path:
            self.file_edit.setText(path)

    def browse_input_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Audio", "", "Audio Files (*.wav *.flac *.mp3)")
        if path:
            self.input_file_edit.setText(path)

    def get_noise_params(self) -> NoiseParams:
        """Collect the current UI values into a :class:`NoiseParams`."""
        params = NoiseParams(
            duration_seconds=float(self.duration_spin.value()),
            sample_rate=int(self.sample_rate_spin.value()),
            noise_type=self.noise_type_combo.currentText().lower(),
            lfo_waveform=self.lfo_waveform_combo.currentText().lower(),
            transition=self.transition_check.isChecked(),
            lfo_freq=float(self.lfo_start_spin.value()),
            start_lfo_freq=float(self.lfo_start_spin.value()),
            end_lfo_freq=float(self.lfo_end_spin.value()),
            start_lfo_phase_offset_deg=int(self.lfo_phase_start_spin.value()),
            end_lfo_phase_offset_deg=int(self.lfo_phase_end_spin.value()),
            start_intra_phase_offset_deg=int(self.intra_phase_start_spin.value()),
            end_intra_phase_offset_deg=int(self.intra_phase_end_spin.value()),
            initial_offset=float(self.initial_offset_spin.value()),
            post_offset=float(self.post_offset_spin.value()),
            input_audio_path=self.input_file_edit.text(),
        )
        sweeps = []
        for i in range(self.num_sweeps_spin.value()):
            (
                _, s_min, e_min, s_max, e_max, s_q, e_q, s_casc, e_casc
            ) = self.sweep_rows[i]
            sweeps.append(
                {
                    "start_min": int(s_min.value()),
                    "end_min": int(e_min.value()),
                    "start_max": int(s_max.value()),
                    "end_max": int(e_max.value()),
                    "start_q": int(s_q.value()),
                    "end_q": int(e_q.value()),
                    "start_casc": int(s_casc.value()),
                    "end_casc": int(e_casc.value()),
                }
            )
        params.sweeps = sweeps
        return params

    def set_noise_params(self, params: NoiseParams) -> None:
        """Apply ``params`` to the UI widgets."""
        self.duration_spin.setValue(params.duration_seconds)
        self.sample_rate_spin.setValue(params.sample_rate)
        idx = self.noise_type_combo.findText(params.noise_type.capitalize())
        if idx != -1:
            self.noise_type_combo.setCurrentIndex(idx)
        idx = self.lfo_waveform_combo.findText(params.lfo_waveform.capitalize())
        if idx != -1:
            self.lfo_waveform_combo.setCurrentIndex(idx)
        self.transition_check.setChecked(params.transition)
        start_freq = params.start_lfo_freq if params.transition else params.lfo_freq
        self.lfo_start_spin.setValue(start_freq)
        self.lfo_end_spin.setValue(params.end_lfo_freq)
        self.num_sweeps_spin.setValue(max(1, len(params.sweeps)))
        for i, (
            _, s_min, e_min, s_max, e_max, s_q, e_q, s_casc, e_casc
        ) in enumerate(self.sweep_rows):
            if i < len(params.sweeps):
                sweep = params.sweeps[i]
                s_min.setValue(sweep.get("start_min", s_min.value()))
                e_min.setValue(sweep.get("end_min", e_min.value()))
                s_max.setValue(sweep.get("start_max", s_max.value()))
                e_max.setValue(sweep.get("end_max", e_max.value()))
                s_q.setValue(sweep.get("start_q", s_q.value()))
                e_q.setValue(sweep.get("end_q", e_q.value()))
                s_casc.setValue(sweep.get("start_casc", s_casc.value()))
                e_casc.setValue(sweep.get("end_casc", e_casc.value()))
        self.lfo_phase_start_spin.setValue(params.start_lfo_phase_offset_deg)
        self.lfo_phase_end_spin.setValue(params.end_lfo_phase_offset_deg)
        self.intra_phase_start_spin.setValue(params.start_intra_phase_offset_deg)
        self.intra_phase_end_spin.setValue(params.end_intra_phase_offset_deg)
        self.initial_offset_spin.setValue(params.initial_offset)
        self.post_offset_spin.setValue(params.post_offset)
        self.input_file_edit.setText(params.input_audio_path or "")

    def save_settings(self):
        params = self.get_noise_params()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Noise Settings",
            "",
            f"Noise Files (*{NOISE_FILE_EXTENSION})",
        )
        if not path:
            return
        try:
            save_noise_params(params, path)
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

    def load_settings(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Noise Settings",
            "",
            f"Noise Files (*{NOISE_FILE_EXTENSION})",
        )
        if not path:
            return
        try:
            params = load_noise_params(path)
            self.set_noise_params(params)
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

    def on_generate(self):
        filename = self.file_edit.text() or "swept_notch_noise.wav"
        input_path = self.input_file_edit.text() or None
        try:
            start_sweeps = []
            end_sweeps = []
            start_q_vals = []
            end_q_vals = []
            start_casc = []
            end_casc = []
            for i in range(self.num_sweeps_spin.value()):
                (
                    _, s_min, e_min, s_max, e_max, s_q, e_q, s_casc, e_casc
                ) = self.sweep_rows[i]
                start_sweeps.append((int(s_min.value()), int(s_max.value())))
                end_sweeps.append((int(e_min.value()), int(e_max.value())))
                start_q_vals.append(int(s_q.value()))
                end_q_vals.append(int(e_q.value()))
                start_casc.append(int(s_casc.value()))
                end_casc.append(int(e_casc.value()))

            if self.transition_check.isChecked():
                generate_swept_notch_pink_sound_transition(
                    filename=filename,
                    duration_seconds=float(self.duration_spin.value()),
                    sample_rate=int(self.sample_rate_spin.value()),
                    start_lfo_freq=float(self.lfo_start_spin.value()),
                    end_lfo_freq=float(self.lfo_end_spin.value()),
                    start_filter_sweeps=start_sweeps,
                    end_filter_sweeps=end_sweeps,
                    start_notch_q=start_q_vals if len(start_q_vals) > 1 else start_q_vals[0],
                    end_notch_q=end_q_vals if len(end_q_vals) > 1 else end_q_vals[0],
                    start_cascade_count=start_casc if len(start_casc) > 1 else start_casc[0],
                    end_cascade_count=end_casc if len(end_casc) > 1 else end_casc[0],
                    start_lfo_phase_offset_deg=int(self.lfo_phase_start_spin.value()),
                    end_lfo_phase_offset_deg=int(self.lfo_phase_end_spin.value()),
                    start_intra_phase_offset_deg=int(self.intra_phase_start_spin.value()),
                    end_intra_phase_offset_deg=int(self.intra_phase_end_spin.value()),
                  
                    initial_offset=float(self.initial_offset_spin.value()),
                    post_offset=float(self.post_offset_spin.value()),

                    input_audio_path=input_path,
                    noise_type=self.noise_type_combo.currentText().lower(),
                    lfo_waveform=self.lfo_waveform_combo.currentText().lower(),
                )
            else:
                generate_swept_notch_pink_sound(
                    filename=filename,
                    duration_seconds=float(self.duration_spin.value()),
                    sample_rate=int(self.sample_rate_spin.value()),
                    lfo_freq=float(self.lfo_start_spin.value()),
                    filter_sweeps=start_sweeps,
                    notch_q=start_q_vals if len(start_q_vals) > 1 else start_q_vals[0],
                    cascade_count=start_casc if len(start_casc) > 1 else start_casc[0],
                    lfo_phase_offset_deg=int(self.lfo_phase_start_spin.value()),
                    intra_phase_offset_deg=int(self.intra_phase_start_spin.value()),
                    input_audio_path=input_path,
                    noise_type=self.noise_type_combo.currentText().lower(),
                    lfo_waveform=self.lfo_waveform_combo.currentText().lower(),
                )
            QMessageBox.information(self, "Success", f"Generated {filename}")
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

    def _generate_noise_array(self, params: NoiseParams):
        start_sweeps = []
        end_sweeps = []
        start_q_vals = []
        end_q_vals = []
        start_casc = []
        end_casc = []
        for sw in params.sweeps:
            start_sweeps.append((int(sw.get("start_min", 1000)), int(sw.get("start_max", 10000))))
            end_sweeps.append((int(sw.get("end_min", 1000)), int(sw.get("end_max", 10000))))
            start_q_vals.append(int(sw.get("start_q", 25)))
            end_q_vals.append(int(sw.get("end_q", 25)))
            start_casc.append(int(sw.get("start_casc", 10)))
            end_casc.append(int(sw.get("end_casc", 10)))

        if params.transition:
            audio, _ = _generate_swept_notch_arrays_transition(
                params.duration_seconds,
                params.sample_rate,
                params.start_lfo_freq,
                params.end_lfo_freq,
                start_sweeps,
                end_sweeps,
                start_q_vals if len(start_q_vals) > 1 else start_q_vals[0],
                end_q_vals if len(end_q_vals) > 1 else end_q_vals[0],
                start_casc if len(start_casc) > 1 else start_casc[0],
                end_casc if len(end_casc) > 1 else end_casc[0],
                params.start_lfo_phase_offset_deg,
                params.end_lfo_phase_offset_deg,
                params.start_intra_phase_offset_deg,
                params.end_intra_phase_offset_deg,
                params.input_audio_path or None,
                params.noise_type,
                params.lfo_waveform,
                params.initial_offset,
                params.post_offset,
                "linear",
                False,
                2,
            )
        else:
            audio, _ = _generate_swept_notch_arrays(
                params.duration_seconds,
                params.sample_rate,
                params.lfo_freq,
                start_sweeps,
                start_q_vals if len(start_q_vals) > 1 else start_q_vals[0],
                start_casc if len(start_casc) > 1 else start_casc[0],
                params.start_lfo_phase_offset_deg,
                params.start_intra_phase_offset_deg,
                params.input_audio_path or None,
                params.noise_type,
                params.lfo_waveform,
                False,
                2,
            )
        return audio

    def on_test(self):
        if not QT_MULTIMEDIA_AVAILABLE:
            QMessageBox.critical(
                self,
                "PyQt5 Multimedia Missing",
                "PyQt5.QtMultimedia is required for audio preview, but it "
                "could not be loaded."
            )
            return
        params = self.get_noise_params()
        params.duration_seconds = 30.0
        try:
            stereo = self._generate_noise_array(params)
            audio_int16 = (np.clip(stereo, -1.0, 1.0) * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()

            fmt = QAudioFormat()
            fmt.setCodec("audio/pcm")
            fmt.setSampleRate(int(params.sample_rate))
            fmt.setSampleSize(16)
            fmt.setChannelCount(2)
            fmt.setByteOrder(QAudioFormat.LittleEndian)
            fmt.setSampleType(QAudioFormat.SignedInt)

            device_info = QAudioDeviceInfo.defaultOutputDevice()
            if not device_info.isFormatSupported(fmt):
                QMessageBox.warning(self, "Noise Test", "Default output device does not support the required format")
                return

            if self.audio_output:
                self.audio_output.stop()
                self.audio_output = None

            self.audio_output = QAudioOutput(fmt, self)
            self.audio_buffer = QBuffer()
            self.audio_buffer.setData(audio_bytes)
            self.audio_buffer.open(QIODevice.ReadOnly)
            self.audio_output.start(self.audio_buffer)
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

