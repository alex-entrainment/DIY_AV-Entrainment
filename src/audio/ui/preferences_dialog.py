from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QFontComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QPushButton,
    QFileDialog, QCheckBox, QComboBox, QDialogButtonBox, QLabel
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt


try:
    from ..preferences import Preferences
except ImportError:  # Running as a script without packages
    from preferences import Preferences
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

        # Sample rate and test step duration
        audio_group = QGroupBox("Audio/Test")
        audio_form = QFormLayout()
        self.sample_rate_spin = QSpinBox()
        self.sample_rate_spin.setRange(8000, 192000)
        self.sample_rate_spin.setValue(prefs.sample_rate)
        self.test_duration_spin = QDoubleSpinBox()
        self.test_duration_spin.setRange(0.1, 600.0)
        self.test_duration_spin.setDecimals(1)
        self.test_duration_spin.setValue(prefs.test_step_duration)
        self.track_metadata_chk = QCheckBox("Include track export metadata")
        self.track_metadata_chk.setChecked(prefs.track_metadata)
        audio_form.addRow("Sample Rate (Hz):", self.sample_rate_spin)
        audio_form.addRow("Test Step Duration (s):", self.test_duration_spin)
        audio_form.addRow(self.track_metadata_chk)
        audio_group.setLayout(audio_form)
        layout.addWidget(audio_group)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def browse_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Export Directory", self.export_edit.text())
        if directory:
            self.export_edit.setText(directory)

    def get_preferences(self) -> Preferences:
        p = Preferences(
            font_family=self.font_combo.currentFont().family(),
            font_size=self.font_size_spin.value(),
            theme=self.theme_combo.currentText(),
            export_dir=self.export_edit.text(),
            sample_rate=self.sample_rate_spin.value(),
            test_step_duration=self.test_duration_spin.value(),
            track_metadata=self.track_metadata_chk.isChecked(),
        )
        return p
