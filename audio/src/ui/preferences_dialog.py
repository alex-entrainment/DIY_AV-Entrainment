from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QFontComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QPushButton,
    QFileDialog, QCheckBox, QComboBox, QDialogButtonBox, QLabel
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt


try:
    from utils.preferences import Preferences
except ImportError:  # Running as a script without packages
    from utils.preferences import Preferences
from . import themes  # reuse themes from audio package

class PreferencesDialog(QDialog):
    def __init__(self, prefs: Preferences, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self._prefs = prefs
        layout = QVBoxLayout(self)

        # Font settings
        font_group = QGroupBox("Font")
        form = QFormLayout()
        self.font_combo = QFontComboBox()
        if prefs.font_family:
            self.font_combo.setCurrentFont(QFont(prefs.font_family))
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(6, 48)
        self.font_size_spin.setValue(prefs.font_size)
        form.addRow("Family:", self.font_combo)
        form.addRow("Size:", self.font_size_spin)
        font_group.setLayout(form)
        layout.addWidget(font_group)

        # Theme
        theme_group = QGroupBox("Theme")
        theme_layout = QHBoxLayout()
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(themes.THEMES.keys())
        idx = self.theme_combo.findText(prefs.theme)
        if idx != -1:
            self.theme_combo.setCurrentIndex(idx)
        theme_layout.addWidget(QLabel("Theme:"))
        theme_layout.addWidget(self.theme_combo)
        theme_group.setLayout(theme_layout)
        layout.addWidget(theme_group)

        # Export directory
        export_group = QGroupBox("Export")
        export_layout = QHBoxLayout()
        self.export_edit = QLineEdit(prefs.export_dir)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_dir)
        export_layout.addWidget(self.export_edit)
        export_layout.addWidget(browse_btn)
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        # Sample rate, test step duration, and target amplitude
        audio_group = QGroupBox("Audio/Test")
        audio_form = QFormLayout()
        self.sample_rate_spin = QSpinBox()
        self.sample_rate_spin.setRange(8000, 192000)
        self.sample_rate_spin.setValue(prefs.sample_rate)
        self.test_duration_spin = QDoubleSpinBox()
        self.test_duration_spin.setRange(0.1, 600.0)
        self.test_duration_spin.setDecimals(1)
        self.test_duration_spin.setValue(prefs.test_step_duration)
        self.target_amp_spin = QDoubleSpinBox()
        self._amp_mode = getattr(prefs, "amplitude_display_mode", "absolute")
        if self._amp_mode == "dB":
            from utils.amp_utils import amplitude_to_db, MIN_DB
            self.target_amp_spin.setRange(MIN_DB, 0.0)
            self.target_amp_spin.setDecimals(1)
            self.target_amp_spin.setSingleStep(1.0)
            self.target_amp_spin.setSuffix(" dB")
            self.target_amp_spin.setValue(amplitude_to_db(prefs.target_output_amplitude))
        else:
            self.target_amp_spin.setRange(0.0, 1.0)
            self.target_amp_spin.setDecimals(3)
            self.target_amp_spin.setSingleStep(0.05)
            self.target_amp_spin.setSuffix("")
            self.target_amp_spin.setValue(prefs.target_output_amplitude)
        self.crossfade_curve_combo = QComboBox()
        self.crossfade_curve_combo.addItems(["linear", "equal_power"])
        idx_curve = self.crossfade_curve_combo.findText(getattr(prefs, "crossfade_curve", "linear"))
        if idx_curve != -1:
            self.crossfade_curve_combo.setCurrentIndex(idx_curve)
        self.track_metadata_chk = QCheckBox("Include track export metadata")
        self.track_metadata_chk.setChecked(prefs.track_metadata)
        self.apply_target_amp_chk = QCheckBox("Apply Target Amplitude")
        self.apply_target_amp_chk.setChecked(getattr(prefs, "apply_target_amplitude", True))
        self.amp_mode_combo = QComboBox()
        self.amp_mode_combo.addItems(["absolute", "dB"])
        idx_mode = self.amp_mode_combo.findText(self._amp_mode)
        if idx_mode != -1:
            self.amp_mode_combo.setCurrentIndex(idx_mode)
        self.amp_mode_combo.currentTextChanged.connect(self._on_amp_mode_change)
        self.convert_amps_btn = QPushButton("Convert Amplitudes to dB")
        self.convert_amps_btn.clicked.connect(self.convert_amplitudes_to_db)
        audio_form.addRow("Sample Rate (Hz):", self.sample_rate_spin)
        audio_form.addRow("Test Step Duration (s):", self.test_duration_spin)
        audio_form.addRow("Target Output Amplitude:", self.target_amp_spin)
        audio_form.addRow("Amplitude Display:", self.amp_mode_combo)
        audio_form.addRow("Crossfade Curve:", self.crossfade_curve_combo)
        audio_form.addRow(self.track_metadata_chk)
        audio_form.addRow(self.apply_target_amp_chk)
        audio_form.addRow(self.convert_amps_btn)
        audio_group.setLayout(audio_form)
        layout.addWidget(audio_group)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)

        ok_button = buttons.button(QDialogButtonBox.Ok)
        if ok_button is not None:
            ok_button.setDefault(True)

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def browse_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Export Directory", self.export_edit.text())
        if directory:
            self.export_edit.setText(directory)

    def _on_amp_mode_change(self, mode: str):
        """Handle switching between absolute and dB amplitude display."""
        from utils.amp_utils import db_to_amplitude, amplitude_to_db, MIN_DB
        value = self.target_amp_spin.value()
        if self._amp_mode == "dB" and mode == "absolute":
            value = db_to_amplitude(value)
        elif self._amp_mode == "absolute" and mode == "dB":
            value = amplitude_to_db(value)
        self._amp_mode = mode
        if mode == "dB":
            self.target_amp_spin.setRange(MIN_DB, 0.0)
            self.target_amp_spin.setDecimals(1)
            self.target_amp_spin.setSingleStep(1.0)
            self.target_amp_spin.setSuffix(" dB")
        else:
            self.target_amp_spin.setRange(0.0, 1.0)
            self.target_amp_spin.setDecimals(3)
            self.target_amp_spin.setSingleStep(0.05)
            self.target_amp_spin.setSuffix("")
        self.target_amp_spin.setValue(value)

    def convert_amplitudes_to_db(self):
        """Convert current amplitude values in the project to dB."""
        from utils.amp_utils import amplitude_to_db, MIN_DB

        if self._amp_mode != "dB":
            self._on_amp_mode_change("dB")
            idx = self.amp_mode_combo.findText("dB")
            if idx != -1:
                self.amp_mode_combo.setCurrentIndex(idx)

        parent = self.parent()
        if parent and hasattr(parent, "track_data"):
            def convert_dict(d: dict):
                for k, v in d.items():
                    if isinstance(v, (int, float)) and "amp" in k.lower():
                        d[k] = amplitude_to_db(v)

            td = parent.track_data
            bg = td.get("background_noise", {})
            if isinstance(bg, dict):
                convert_dict(bg)

            for clip in td.get("clips", []):
                if isinstance(clip, dict):
                    convert_dict(clip)

            for step in td.get("steps", []):
                for voice in step.get("voices", []):
                    params = voice.get("params", {})
                    if isinstance(params, dict):
                        convert_dict(params)

        dv = getattr(self._prefs, "default_voice", {})
        if isinstance(dv, dict):
            params = dv.get("params", {})
            if isinstance(params, dict):
                for k, v in params.items():
                    if isinstance(v, (int, float)) and "amp" in k.lower():
                        params[k] = amplitude_to_db(v)

    def get_preferences(self) -> Preferences:
        from utils.amp_utils import db_to_amplitude
        amp_mode = self.amp_mode_combo.currentText()
        amp_value = self.target_amp_spin.value()
        if amp_mode == "dB":
            amp_value = db_to_amplitude(amp_value)
        p = Preferences(
            font_family=self.font_combo.currentFont().family(),
            font_size=self.font_size_spin.value(),
            theme=self.theme_combo.currentText(),
            export_dir=self.export_edit.text(),
            sample_rate=self.sample_rate_spin.value(),
            test_step_duration=self.test_duration_spin.value(),
            track_metadata=self.track_metadata_chk.isChecked(),
            target_output_amplitude=amp_value,
            crossfade_curve=self.crossfade_curve_combo.currentText(),
            amplitude_display_mode=amp_mode,
            apply_target_amplitude=self.apply_target_amp_chk.isChecked(),
        )
        return p
