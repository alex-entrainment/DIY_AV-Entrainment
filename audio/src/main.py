import sys
import json
from synth_functions import sound_creator  # Updated import path
import soundfile as sf
import os
import copy  # For deep copying voice data
import traceback  # For error reporting

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QTreeView,
    QTreeWidget,
    QTreeWidgetItem,
    QTextEdit,
    QGroupBox,
    QSplitter,
    QFileDialog,
    QMessageBox,
    QDialog,
    QSizePolicy,
    QInputDialog,
    QHeaderView,
    QSlider,
    QDoubleSpinBox,
    QAbstractItemView,
    QAction,
    QProgressBar,
)

from PyQt5.QtCore import Qt, pyqtSlot, QTimer, QBuffer, QIODevice, QItemSelectionModel

from PyQt5.QtGui import QIntValidator, QDoubleValidator, QFont
try:
    from PyQt5.QtMultimedia import (
        QAudioFormat,
        QAudioOutput,
        QAudioDeviceInfo,
        QAudio,
    )
except Exception as e:  # noqa: PIE786 - generic Exception to capture missing Qt deps
    print(
        "ERROR: Failed to import PyQt5.QtMultimedia modules.\n"\
        "Ensure PyQt5 with multimedia extras is installed and system libraries are available.\n"\
        f"Original error: {e}"
    )
    QAudioFormat = QAudioOutput = QAudioDeviceInfo = QAudio = None

from functools import partial
from ui.themes import THEMES, apply_theme
from utils.preferences import Preferences
from utils.settings_file import load_settings, save_settings
from ui.preferences_dialog import PreferencesDialog
from ui.noise_generator_dialog import NoiseGeneratorDialog
from ui.frequency_tester_dialog import FrequencyTesterDialog
from ui.subliminal_dialog import SubliminalDialog
from utils.timeline_visualizer import visualize_track_timeline
from ui.overlay_clip_dialog import OverlayClipDialog
from ui.collapsible_box import CollapsibleBox
from models import StepModel, VoiceModel
from utils.voice_file import (
    VoicePreset,
    save_voice_preset_list,
    load_voice_preset_list,
    VOICES_FILE_EXTENSION,
)

# Disable Windows action sounds by overriding QMessageBox helpers
def _silent_message_box(parent, title, text, buttons=QMessageBox.Ok, default_button=QMessageBox.NoButton):
    box = QMessageBox(parent)
    box.setWindowTitle(title)
    box.setText(text)
    box.setStandardButtons(buttons)
    box.setDefaultButton(default_button)
    box.setIcon(QMessageBox.NoIcon)
    return box.exec_()


def _patch_qmessagebox():
    """Replace QMessageBox helpers to prevent system notification sounds."""

    def _wrap(default_buttons):
        def wrapper(parent, title, text, buttons=default_buttons, default_button=QMessageBox.NoButton):
            return _silent_message_box(parent, title, text, buttons, default_button)

        return wrapper

    QMessageBox.information = _wrap(QMessageBox.Ok)
    QMessageBox.warning = _wrap(QMessageBox.Ok)
    QMessageBox.critical = _wrap(QMessageBox.Ok)

    def question_wrapper(
        parent,
        title,
        text,
        buttons=QMessageBox.StandardButtons(QMessageBox.Yes | QMessageBox.No),
        default_button=QMessageBox.NoButton,
    ):
        return _silent_message_box(parent, title, text, buttons, default_button)

    QMessageBox.question = question_wrapper


_patch_qmessagebox()

# Attempt to import VoiceEditorDialog. Handle if ui/voice_editor_dialog.py is not found.
try:
    from ui.voice_editor_dialog import VoiceEditorDialog
    VOICE_EDITOR_DIALOG_AVAILABLE = True
except ImportError:
    VOICE_EDITOR_DIALOG_AVAILABLE = False
    # Create a dummy VoiceEditorDialog if the real one is not available
    # This allows the main application to run, but voice editing will be non-functional.
    class VoiceEditorDialog(QDialog):
        def __init__(self, parent=None, app_ref=None, step_index=None, voice_index=None, voice_data_override=None):
            super().__init__(parent)
            self.setWindowTitle("Voice Editor (Unavailable)")
            layout = QVBoxLayout(self)
            label = QLabel("VoiceEditorDialog is not available. Please ensure 'ui/voice_editor_dialog.py' exists.")
            layout.addWidget(label)
            ok_button = QPushButton("OK")
            ok_button.clicked.connect(self.reject)
            layout.addWidget(ok_button)
            print("WARNING: ui.voice_editor_dialog.VoiceEditorDialog not found. Voice editing will be limited/non-functional.")

        def exec_(self):
            # Override exec_ to prevent showing a non-functional dialog,
            # or show a simple message.
            QMessageBox.critical(self.parent(), "Voice Editor Error", "VoiceEditorDialog component is missing.\nCannot add or edit voices.")
            return QDialog.Rejected


# Numpy is used in _generate_test_step_audio
import numpy as np

# Common stylesheet for a slightly more modern dark look
GLOBAL_STYLE_SHEET = """
QWidget {
    font-size: 10pt;
}
QGroupBox {
    border: 1px solid #444444;
    border-radius: 4px;
    margin-top: 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 3px 0 3px;
}
QPushButton {
    padding: 4px 12px;
}
QTreeWidget {
    background-color: #2b2b2b;

    color: #ffffff;

}
QLineEdit, QComboBox, QSlider {
    background-color: #202020;
    border: 1px solid #555555;
    color: #ffffff;
}
"""

# Material global style with subtle shadows and rounded cards
GLOBAL_STYLE_SHEET_MATERIAL = """
QWidget {
    font-size: 10pt;
}
QGroupBox {
    background-color: #ffffff;
    border: 1px solid #d0d0d0;
    border-radius: 8px;
    margin-top: 12px;
    padding: 8px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 4px;
}
QPushButton {
    padding: 6px 16px;
    border-radius: 4px;
}
QLineEdit, QComboBox, QSlider {
    background-color: #ffffff;
    border: 1px solid #bdbdbd;
    color: #212121;
    border-radius: 4px;
    padding: 2px 4px;
}
"""

# --- Constants ---
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_CROSSFADE = 1.0 # Ensure float for consistency
MAX_VOICES_PER_STEP = 100 # From previous version
ENVELOPE_TYPE_NONE = "None" # From previous, useful if VoiceEditorDialog uses it
ENVELOPE_TYPE_LINEAR = "linear_fade" # From previous
SUPPORTED_ENVELOPE_TYPES = [ENVELOPE_TYPE_NONE, ENVELOPE_TYPE_LINEAR] # From previous


# Updated import path for sound_creator
try:
    from synth_functions.sound_creator import generate_single_step_audio_segment  # Used for test preview
    AUDIO_GENERATION_AVAILABLE = True # For test preview specifically
except ImportError as e:
    generate_single_step_audio_segment = None
    AUDIO_GENERATION_AVAILABLE = False
    print(
        f"Warning: Could not import 'generate_single_step_audio_segment' from 'synth_functions.sound_creator': {e}. "
        "Test step audio generation will be non-functional."
    )

# --- Main Application Class ---
class TrackEditorApp(QMainWindow):
    def __init__(self, prefs: Preferences = None):
        super().__init__()
        self.prefs: Preferences = prefs or load_settings()
        self.apply_preferences()
        self.setWindowTitle("Binaural Track Editor (PyQt5)")
        self.setMinimumSize(950, 600)
        self.resize(1200, 800)

        self.track_data = self._get_default_track_data()
        self.current_json_path = None

        # Validators (reusable)
        self.int_validator_positive = QIntValidator(1, 999999, self)
        self.double_validator_non_negative = QDoubleValidator(0.0, 999999.0, 6, self)
        self.double_validator_zero_to_one = QDoubleValidator(0.0, 1.0, 6, self)
        self.double_validator = QDoubleValidator(-999999.0, 999999.0, 6, self)
        self.int_validator = QIntValidator(-999999, 999999, self)

        # Audio test attributes
        self.test_audio_output = None
        self.test_audio_io_device = None
        self.test_step_raw_audio = None
        self.is_step_test_playing = False
        self.is_step_test_paused = False
        self.current_test_step_index = -1 # Index of step whose audio is loaded, or -1
        self.test_audio_timer = QTimer(self)
        self.test_audio_timer.timeout.connect(self._update_test_audio_progress)
        self.test_audio_format = None
        self.total_test_audio_duration_ms = 0 # Used by test player UI

        self.test_step_duration = self.prefs.test_step_duration

        # Overlay clip preview attributes
        self.clip_audio_output = None
        self.clip_audio_buffer = None
        self.is_clip_playing = False

        # Initialize history and update flags early to avoid race conditions
        # when UI signals fire during setup.
        self.history = []
        self.history_index = -1
        self._voices_tree_updating = False
        self._steps_tree_updating = False
        self._clips_tree_updating = False
        self._copied_voices = []

        self._setup_ui()
        self.setStyleSheet(GLOBAL_STYLE_SHEET)
        self._update_ui_from_global_settings()
        self.refresh_steps_tree()
        self.refresh_clips_tree()
        self._update_step_actions_state()
        self._update_voice_actions_state()
        self._update_clip_actions_state()

        self.statusBar()
        self._create_menu()

        self._push_history_state()

    def _get_default_track_data(self):
        return {
            "global_settings": {
                "sample_rate": self.prefs.sample_rate if hasattr(self, "prefs") else DEFAULT_SAMPLE_RATE,
                "crossfade_duration": DEFAULT_CROSSFADE,
                "crossfade_curve": getattr(self.prefs, "crossfade_curve", "linear"),
                "output_filename": "my_track.flac",
            },
            "background_noise": {
                "file_path": "",
                "amp": 0.0,
                "pan": 0.0,
                "start_time": 0.0,
                "fade_in": 0.0,
                "fade_out": 0.0,
                "amp_envelope": [],
            },
            "clips": [],
            "steps": []
        }

    def _get_clip_duration(self, path: str) -> float:
        """Return duration of ``path`` in seconds or ``0.0`` on error."""
        try:
            info = sf.info(path)
            if info.samplerate > 0:
                return info.frames / float(info.samplerate)
        except Exception:
            pass
        return 0.0

    def _create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        new_act = QAction("New", self)
        new_act.triggered.connect(self.new_file)
        file_menu.addAction(new_act)

        open_act = QAction("Open", self)
        open_act.triggered.connect(self.load_json)
        file_menu.addAction(open_act)

        save_act = QAction("Save", self)
        save_act.triggered.connect(self.save_json)
        file_menu.addAction(save_act)

        save_as_act = QAction("Save As", self)
        save_as_act.triggered.connect(self.save_json_as)
        file_menu.addAction(save_as_act)

        pref_act = QAction("Preferences", self)
        pref_act.triggered.connect(self.open_preferences)
        file_menu.addAction(pref_act)

        defaults_act = QAction("Configure Defaults", self)
        defaults_act.triggered.connect(self.open_default_voice_config)
        file_menu.addAction(defaults_act)

        file_menu.addSeparator()
        theme_menu = file_menu.addMenu("Theme")
        for name in sorted(THEMES.keys()):
            act = QAction(name, self)
            act.triggered.connect(partial(self.set_theme, name))
            theme_menu.addAction(act)

        edit_menu = menubar.addMenu("Edit")
        self.undo_act = QAction("Undo", self)
        self.undo_act.setShortcut("Ctrl+Z")
        self.undo_act.triggered.connect(self.undo)
        edit_menu.addAction(self.undo_act)

        self.redo_act = QAction("Redo", self)
        self.redo_act.setShortcut("Ctrl+Y")
        self.redo_act.triggered.connect(self.redo)
        edit_menu.addAction(self.redo_act)
        self._update_undo_redo_actions_state()


    def set_theme(self, name):
        apply_theme(QApplication.instance(), name)
        if name == "Material":
            self.setStyleSheet(GLOBAL_STYLE_SHEET_MATERIAL)
        else:
            self.setStyleSheet(GLOBAL_STYLE_SHEET)
        if hasattr(self, "prefs"):
            self.prefs.theme = name
            save_settings(self.prefs)

    def open_preferences(self):
        dialog = PreferencesDialog(self.prefs, self)
        if dialog.exec_() == QDialog.Accepted:
            self.prefs = dialog.get_preferences()
            save_settings(self.prefs)
            self.apply_preferences()
            # Update runtime values from new preferences
            self.test_step_duration = self.prefs.test_step_duration
            # Update sample rate field if preferences changed
            if hasattr(self, "sr_entry"):
                self.sr_entry.setText(str(self.prefs.sample_rate))
                self._update_global_settings_from_ui()
            # Sync crossfade curve into current track settings
            if "global_settings" in self.track_data:
                self.track_data["global_settings"]["crossfade_curve"] = self.prefs.crossfade_curve

    def open_default_voice_config(self):
        from ui.default_voice_dialog import DefaultVoiceDialog
        dialog = DefaultVoiceDialog(self.prefs, self)
        if dialog.exec_() == QDialog.Accepted:
            self.prefs.default_voice = dialog.get_default_voice()
            save_settings(self.prefs)

    def open_noise_generator(self):
        dialog = NoiseGeneratorDialog(self)
        dialog.exec_()

    def open_frequency_tester(self):
        dialog = FrequencyTesterDialog(self, self.prefs)
        dialog.exec_()

    def open_audio_thresholder(self):
        from ui.audio_thresholder_dialog import AudioThresholderDialog
        dialog = AudioThresholderDialog(self.prefs, self)
        if dialog.exec_() == QDialog.Accepted:
            self.prefs.target_output_amplitude = dialog.get_target_amplitude()
            save_settings(self.prefs)

    def open_subliminal_dialog(self):
        selected_step_index = self.get_selected_step_index()
        if selected_step_index is None or len(self.get_selected_step_indices()) != 1:
            QMessageBox.warning(self, "Subliminal", "Please select exactly one step first.")
            return
        dialog = SubliminalDialog(self, app_ref=self, step_index=selected_step_index)
        if dialog.exec_() == QDialog.Accepted:
            self.refresh_steps_tree()
            if 0 <= selected_step_index < self.step_model.rowCount():
                idx = self.step_model.index(selected_step_index, 0)
                self.steps_tree.setCurrentIndex(idx)
                QTimer.singleShot(0, lambda: self._select_last_voice_in_current_step())
            self._push_history_state()

    def _setup_ui(self):
        """Setup the widget-based interface."""
        self._setup_widget_ui()

    def open_timeline_visualizer(self):
        """Display the timeline visualizer for the current track data."""
        try:
            visualize_track_timeline(self.track_data)
        except Exception as e:
            QMessageBox.warning(self, "Timeline Error", f"Failed to display timeline: {e}")

    def apply_preferences(self):
        app = QApplication.instance()
        if self.prefs.font_family or self.prefs.font_size:
            font = QFont(self.prefs.font_family or app.font().family(), self.prefs.font_size)
            app.setFont(font)
        apply_theme(app, self.prefs.theme)
        self.setStyleSheet(GLOBAL_STYLE_SHEET)

    def _setup_widget_ui(self):
        # Central Widget and Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Control Frame (Collapsible) ---
        control_groupbox = CollapsibleBox("Controls")
        control_layout = QHBoxLayout()
        control_groupbox.setContentLayout(control_layout)

        # We'll place the controls and the rest of the UI in a vertical splitter
        # so the user can resize the top controls to occupy less space.
        vertical_splitter = QSplitter(Qt.Vertical)
        vertical_splitter.addWidget(control_groupbox)
        main_layout.addWidget(vertical_splitter, 1)

        # Tool Buttons (Noise Generator, Frequency Tester, Subliminal Voice)
        tools_groupbox = QGroupBox("Tools")
        tools_layout = QHBoxLayout()
        tools_groupbox.setLayout(tools_layout)
        tools_left_layout = QVBoxLayout()
        tools_right_layout = QVBoxLayout()
        tools_layout.addLayout(tools_left_layout)
        tools_layout.addLayout(tools_right_layout)

        self.open_noise_button = QPushButton("Open Noise Generator")
        self.open_noise_button.clicked.connect(self.open_noise_generator)
        tools_left_layout.addWidget(self.open_noise_button)

        self.open_freq_tester_button = QPushButton("Frequency Tester")
        self.open_freq_tester_button.clicked.connect(self.open_frequency_tester)
        tools_left_layout.addWidget(self.open_freq_tester_button)

        self.open_thresholder_button = QPushButton("Audio Thresholder")
        self.open_thresholder_button.clicked.connect(self.open_audio_thresholder)
        tools_left_layout.addWidget(self.open_thresholder_button)

        self.open_subliminal_button = QPushButton("Add Subliminal Voice")
        self.open_subliminal_button.clicked.connect(self.open_subliminal_dialog)
        tools_left_layout.addWidget(self.open_subliminal_button)

        self.open_timeline_button = QPushButton("View Timeline")
        self.open_timeline_button.clicked.connect(self.open_timeline_visualizer)
        tools_left_layout.addWidget(self.open_timeline_button)
        tools_left_layout.addStretch(1)

        self.new_file_button = QPushButton("New File")
        self.new_file_button.clicked.connect(self.new_file)
        tools_right_layout.addWidget(self.new_file_button)

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_json)
        tools_right_layout.addWidget(self.save_button)

        self.save_as_button = QPushButton("Save As")
        self.save_as_button.clicked.connect(self.save_json_as)
        tools_right_layout.addWidget(self.save_as_button)

        self.load_button = QPushButton("Load File")
        self.load_button.clicked.connect(self.load_json)
        tools_right_layout.addWidget(self.load_button)
        tools_right_layout.addStretch(1)
        control_layout.addWidget(tools_groupbox)

        # Global Settings
        globals_groupbox = QGroupBox("Global Settings")
        globals_layout = QGridLayout()
        globals_groupbox.setLayout(globals_layout)
        globals_layout.addWidget(QLabel("Sample Rate:"), 0, 0)
        self.sr_entry = QLineEdit(str(DEFAULT_SAMPLE_RATE))
        self.sr_entry.setValidator(self.int_validator_positive)
        self.sr_entry.setMaximumWidth(80)
        globals_layout.addWidget(self.sr_entry, 0, 1)
        globals_layout.addWidget(QLabel("Crossfade (s):"), 1, 0)
        self.cf_entry = QLineEdit(str(DEFAULT_CROSSFADE))
        self.cf_entry.setValidator(self.double_validator_non_negative)
        self.cf_entry.setMaximumWidth(80)
        globals_layout.addWidget(self.cf_entry, 1, 1)
        globals_layout.addWidget(QLabel("Output File:"), 2, 0)
        self.outfile_entry = QLineEdit("my_track.wav")
        globals_layout.addWidget(self.outfile_entry, 2, 1)
        self.browse_outfile_button = QPushButton("Browse...")
        self.browse_outfile_button.clicked.connect(self.browse_outfile)
        globals_layout.addWidget(self.browse_outfile_button, 2, 2)

        globals_layout.addWidget(QLabel("Noise Preset:"), 3, 0)
        self.noise_file_entry = QLineEdit()
        globals_layout.addWidget(self.noise_file_entry, 3, 1)
        self.browse_noise_button = QPushButton("Browse...")
        self.browse_noise_button.clicked.connect(self.browse_noise_file)
        globals_layout.addWidget(self.browse_noise_button, 3, 2)

        globals_layout.addWidget(QLabel("Noise Amp:"), 4, 0)
        self.noise_amp_entry = QLineEdit("0.0")
        self.noise_amp_entry.setValidator(self.double_validator)
        self.noise_amp_entry.setMaximumWidth(80)
        globals_layout.addWidget(self.noise_amp_entry, 4, 1)

        globals_layout.setColumnStretch(1, 1)
        control_layout.addWidget(globals_groupbox, 1)
        control_layout.addStretch(1)

        # Generate Button
        generate_frame = QWidget()
        generate_layout = QHBoxLayout(generate_frame)
        generate_layout.addStretch(1)
        self.generate_button = QPushButton("Generate Audio")
        self.generate_button.setStyleSheet("QPushButton { background-color: #0078D7; color: white; padding: 8px; font-weight: bold; border-radius: 3px; } QPushButton:hover { background-color: #005A9E; } QPushButton:pressed { background-color: #003C6A; } QPushButton:disabled { background-color: #AAAAAA; color: #666666; }")
        self.generate_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.generate_button.clicked.connect(self.generate_audio_action)
        generate_layout.addWidget(self.generate_button)

        self.generate_selected_button = QPushButton("Generate Selected Steps")
        self.generate_selected_button.setStyleSheet(self.generate_button.styleSheet())
        self.generate_selected_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.generate_selected_button.clicked.connect(self.generate_selected_audio_action)
        generate_layout.addWidget(self.generate_selected_button)

        self.generate_progress_bar = QProgressBar()
        self.generate_progress_bar.setRange(0, 100)
        self.generate_progress_bar.setValue(0)
        self.generate_progress_bar.setVisible(False)
        generate_layout.addWidget(self.generate_progress_bar)
        generate_layout.setContentsMargins(0,0,0,0)
        control_layout.addWidget(generate_frame)

        # --- Main Paned Window (Splitter) ---
        main_splitter = QSplitter(Qt.Horizontal)
        vertical_splitter.addWidget(main_splitter)

        # --- Steps Frame Widgets ---
        steps_outer_widget = QWidget()
        steps_outer_layout = QVBoxLayout(steps_outer_widget)
        steps_outer_layout.setContentsMargins(0,0,0,0)
        main_splitter.addWidget(steps_outer_widget)
        # Allow steps pane to shrink well below half the window width
        steps_outer_widget.setMinimumWidth(150)
        steps_groupbox = QGroupBox("Steps")
        steps_groupbox_layout = QVBoxLayout(steps_groupbox)
        steps_outer_layout.addWidget(steps_groupbox)
        self.step_model = StepModel(self.track_data.get("steps", []))
        self.steps_tree = QTreeView()
        self.steps_tree.setModel(self.step_model)
        self.steps_tree.setRootIsDecorated(False)
        self.steps_tree.setUniformRowHeights(True)
        self.steps_tree.setColumnWidth(0, 80)
        self.steps_tree.setColumnWidth(1, 150)
        self.steps_tree.setColumnWidth(2, 60)
        self.steps_tree.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self.steps_tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.steps_tree.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
        )
        self.steps_tree.selectionModel().selectionChanged.connect(
            lambda *_: self.on_step_select()
        )
        self.step_model.dataChanged.connect(lambda *_: self._push_history_state())

        steps_content_layout = QHBoxLayout()
        steps_button_layout = QVBoxLayout()

        self.add_step_button = QPushButton("Add Step")
        self.load_external_step_button = QPushButton("Load External Step")
        self.duplicate_step_button = QPushButton("Duplicate Step")
        self.create_end_state_button = QPushButton("Create End State Step")
        self.remove_step_button = QPushButton("Remove Step(s)")
        self.add_step_button.clicked.connect(self.add_step)
        self.load_external_step_button.clicked.connect(self.load_external_step)
        self.duplicate_step_button.clicked.connect(self.duplicate_step)
        self.create_end_state_button.clicked.connect(self.create_end_state_step)
        self.remove_step_button.clicked.connect(self.remove_step)

        self.edit_duration_button = QPushButton("Edit Duration")
        self.edit_description_button = QPushButton("Edit Description")
        self.move_step_up_button = QPushButton("Move Up")
        self.move_step_down_button = QPushButton("Move Down")
        self.edit_duration_button.clicked.connect(self.edit_step_duration)
        self.edit_description_button.clicked.connect(self.edit_step_description)
        self.move_step_up_button.clicked.connect(lambda: self.move_step(-1))
        self.move_step_down_button.clicked.connect(lambda: self.move_step(1))

        steps_button_layout.addWidget(self.add_step_button)
        steps_button_layout.addWidget(self.load_external_step_button)
        steps_button_layout.addWidget(self.duplicate_step_button)
        steps_button_layout.addWidget(self.create_end_state_button)
        steps_button_layout.addWidget(self.remove_step_button)
        steps_button_layout.addWidget(self.edit_duration_button)
        steps_button_layout.addWidget(self.edit_description_button)
        steps_button_layout.addWidget(self.move_step_up_button)
        steps_button_layout.addWidget(self.move_step_down_button)
        steps_button_layout.addStretch(1)

        steps_content_layout.addLayout(steps_button_layout)
        steps_content_layout.addWidget(self.steps_tree, 1)

        steps_groupbox_layout.addLayout(steps_content_layout)
        self.total_duration_label = QLabel("Total Duration: 0.00 s")
        steps_groupbox_layout.addWidget(self.total_duration_label)

        # --- Test Step Preview Section ---
        test_step_groupbox = QGroupBox("Test Step Preview")
        test_step_main_layout = QVBoxLayout(test_step_groupbox)

        test_controls_top_layout = QHBoxLayout() # For buttons
        self.test_step_play_pause_button = QPushButton("Play")
        self.test_step_play_pause_button.clicked.connect(self.on_play_pause_step_test)
        test_controls_top_layout.addWidget(self.test_step_play_pause_button)

        self.test_step_stop_button = QPushButton("Stop")
        self.test_step_stop_button.clicked.connect(self.on_stop_step_test) # Connect to the base on_stop_step_test
        test_controls_top_layout.addWidget(self.test_step_stop_button)

        self.test_step_reset_button = QPushButton("Reset Tester") # New button
        self.test_step_reset_button.clicked.connect(self.on_reset_step_test)
        test_controls_top_layout.addWidget(self.test_step_reset_button)

        test_controls_top_layout.addWidget(QLabel("Duration (s):"))
        self.test_step_duration_spin = QDoubleSpinBox()
        self.test_step_duration_spin.setDecimals(2)
        self.test_step_duration_spin.setRange(0.0, 9999.0)
        self.test_step_duration_spin.setValue(self.test_step_duration)
        self.test_step_duration_spin.setMaximumWidth(80)
        self.test_step_duration_spin.valueChanged.connect(self.on_test_duration_changed)
        test_controls_top_layout.addWidget(self.test_step_duration_spin)
        test_controls_top_layout.addStretch()
        test_step_main_layout.addLayout(test_controls_top_layout)

        self.test_step_loaded_label = QLabel("No step loaded for preview.") # New label
        self.test_step_loaded_label.setWordWrap(True)
        self.test_step_loaded_label.setAlignment(Qt.AlignCenter) 
        test_step_main_layout.addWidget(self.test_step_loaded_label)

        self.test_step_time_slider = QSlider(Qt.Horizontal)
        self.test_step_time_slider.sliderMoved.connect(self.on_test_slider_moved)
        self.test_step_time_slider.sliderReleased.connect(self.on_test_slider_released)
        test_step_main_layout.addWidget(self.test_step_time_slider)

        self.test_step_time_label = QLabel("00:00 / 00:00")
        self.test_step_time_label.setAlignment(Qt.AlignCenter)
        test_step_main_layout.addWidget(self.test_step_time_label)
        steps_groupbox_layout.addWidget(test_step_groupbox)


        # --- Right Pane (Splitter) ---
        right_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(right_splitter)

        # --- Voices Frame Widgets ---
        voices_outer_widget = QWidget()
        voices_outer_layout = QVBoxLayout(voices_outer_widget)
        voices_outer_layout.setContentsMargins(0,0,0,0)
        right_splitter.addWidget(voices_outer_widget)
        self.voices_groupbox = QGroupBox("Voices for Selected Step")
        # Use an HBox layout so buttons can be stacked on the left side
        voices_groupbox_layout = QHBoxLayout(self.voices_groupbox)
        voices_outer_layout.addWidget(self.voices_groupbox)
        self.voice_model = VoiceModel([])
        self.voices_tree = QTreeView()
        self.voices_tree.setModel(self.voice_model)
        self.voices_tree.setRootIsDecorated(False)
        self.voices_tree.setUniformRowHeights(True)
        self.voices_tree.setColumnWidth(0, 220)
        self.voices_tree.setColumnWidth(1, 100)
        self.voices_tree.setColumnWidth(2, 80)
        self.voices_tree.setColumnWidth(3, 80)
        self.voices_tree.setColumnWidth(4, 80)
        self.voices_tree.setColumnWidth(5, 80)
        self.voices_tree.setColumnWidth(6, 150)
        self.voices_tree.header().setStretchLastSection(True)
        self.voices_tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.voices_tree.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
        )
        self.voices_tree.selectionModel().selectionChanged.connect(
            lambda *_: self.on_voice_select()
        )
        self.voice_model.dataChanged.connect(lambda *_: self._push_history_state())
        # Buttons stacked vertically on the left
        voices_buttons_layout = QVBoxLayout()
        self.add_voice_button = QPushButton("Add Voice")
        self.edit_voice_button = QPushButton("Edit Voice")
        self.group_edit_button = QPushButton("Group Edit")
        self.group_swap_button = QPushButton("Group Swap Transitions")
        self.duplicate_voice_button = QPushButton("Duplicate Voice")
        self.remove_voice_button = QPushButton("Remove Voice(s)")
        self.add_voice_button.clicked.connect(self.add_voice)
        self.edit_voice_button.clicked.connect(self.edit_voice)
        self.group_edit_button.clicked.connect(self.group_edit_voices)
        self.group_swap_button.clicked.connect(self.group_swap_transition_parameters)
        self.duplicate_voice_button.clicked.connect(self.duplicate_voice)
        self.remove_voice_button.clicked.connect(self.remove_voice)
        voices_buttons_layout.addWidget(self.add_voice_button)
        voices_buttons_layout.addWidget(self.edit_voice_button)
        voices_buttons_layout.addWidget(self.group_edit_button)
        voices_buttons_layout.addWidget(self.group_swap_button)
        voices_buttons_layout.addWidget(self.duplicate_voice_button)
        voices_buttons_layout.addWidget(self.remove_voice_button)
        self.copy_voices_button = QPushButton("Copy Voice(s)")
        self.paste_voices_button = QPushButton("Paste Voice(s)")
        self.copy_voices_button.clicked.connect(self.copy_selected_voices)
        self.paste_voices_button.clicked.connect(self.paste_copied_voices)
        voices_buttons_layout.addWidget(self.copy_voices_button)
        voices_buttons_layout.addWidget(self.paste_voices_button)
        self.save_voices_button = QPushButton("Save Voices")
        self.load_voices_button = QPushButton("Load Voices")
        self.save_voices_button.clicked.connect(self.save_selected_voices)
        self.load_voices_button.clicked.connect(self.load_voices_from_file)
        voices_buttons_layout.addWidget(self.save_voices_button)
        voices_buttons_layout.addWidget(self.load_voices_button)
        self.move_voice_up_button = QPushButton("Move Up")
        self.move_voice_down_button = QPushButton("Move Down")
        self.move_voice_up_button.clicked.connect(lambda: self.move_voice(-1))
        self.move_voice_down_button.clicked.connect(lambda: self.move_voice(1))
        voices_buttons_layout.addWidget(self.move_voice_up_button)
        voices_buttons_layout.addWidget(self.move_voice_down_button)
        voices_buttons_layout.addStretch(1)
        voices_groupbox_layout.addLayout(voices_buttons_layout)
        voices_groupbox_layout.addWidget(self.voices_tree, 1)

        # --- Voice Details Frame Widgets ---
        voice_details_outer_widget = QWidget()
        voice_details_outer_layout = QVBoxLayout(voice_details_outer_widget)
        voice_details_outer_layout.setContentsMargins(0,0,0,0)
        right_splitter.addWidget(voice_details_outer_widget)
        # Splitter to allow resizing between voice details and overlay clips
        details_splitter = QSplitter(Qt.Vertical)
        voice_details_outer_layout.addWidget(details_splitter)
        self.voice_details_groupbox = QGroupBox("Selected Voice Details")
        voice_details_groupbox_layout = QVBoxLayout(self.voice_details_groupbox)
        details_splitter.addWidget(self.voice_details_groupbox)
        self.voice_details_text = QTextEdit()
        self.voice_details_text.setReadOnly(True)
        self.voice_details_text.setFont(QFont("Consolas", 9))
        self.voice_details_text.setLineWrapMode(QTextEdit.WidgetWidth)
        voice_details_groupbox_layout.addWidget(self.voice_details_text)

        # --- Overlay Clips Widgets ---
        self.clips_groupbox = QGroupBox("Overlay Clips")
        # Layout with buttons stacked vertically on the left
        clips_groupbox_layout = QHBoxLayout(self.clips_groupbox)
        details_splitter.addWidget(self.clips_groupbox)

        self.clips_tree = QTreeWidget()
        self.clips_tree.setColumnCount(9)
        self.clips_tree.setHeaderLabels([
            "File",
            "Description",
            "Start",
            "Duration",
            "Finish",
            "Amp",
            "Pan",
            "FadeIn",
            "FadeOut",
        ])
        self.clips_tree.setSelectionMode(QTreeWidget.ExtendedSelection)
        self.clips_tree.itemSelectionChanged.connect(self.on_clip_select)
        self.clips_tree.itemChanged.connect(self.on_clip_item_changed)
        self.clips_tree.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
        )
        clips_btn_layout = QVBoxLayout()
        self.add_clip_button = QPushButton("Add Clip")
        self.edit_clip_button = QPushButton("Edit Clip")
        self.remove_clip_button = QPushButton("Remove Clip(s)")
        self.add_clip_button.clicked.connect(self.add_clip)
        self.edit_clip_button.clicked.connect(self.edit_clip)
        self.remove_clip_button.clicked.connect(self.remove_clip)
        self.play_clip_button = QPushButton("Start Clip")
        self.play_clip_button.clicked.connect(self.on_start_stop_clip)
        clips_btn_layout.addWidget(self.add_clip_button)
        clips_btn_layout.addWidget(self.edit_clip_button)
        clips_btn_layout.addWidget(self.remove_clip_button)
        clips_btn_layout.addWidget(self.play_clip_button)
        clips_btn_layout.addStretch(1)
        clips_groupbox_layout.addLayout(clips_btn_layout)
        clips_groupbox_layout.addWidget(self.clips_tree, 1)

        main_splitter.setSizes([400, 700])
        # Provide a reasonable default width split between voices and details
        right_splitter.setSizes([500, 300])
        # Give the control section a relatively small initial height so that the
        # main editor gets more space by default. Users can resize it as needed.
        vertical_splitter.setSizes([control_groupbox.sizeHint().height(), 1])

    # --- Button State Management ---
    def _update_step_actions_state(self):
        selected_indices = self.get_selected_step_indices()
        num_selected = len(selected_indices)
        current_idx = self.get_selected_step_index()  # focused item

        is_single_selection = (num_selected == 1) and (current_idx is not None)
        has_selection = num_selected > 0
        num_steps = len(self.track_data["steps"])

        self.add_step_button.setEnabled(True)
        self.load_external_step_button.setEnabled(True)
        self.remove_step_button.setEnabled(has_selection)
        self.duplicate_step_button.setEnabled(is_single_selection)
        self.create_end_state_button.setEnabled(is_single_selection)
        self.edit_duration_button.setEnabled(is_single_selection)
        self.edit_description_button.setEnabled(is_single_selection)
        # Enable Add Voice if a single step is selected. Availability of the
        # actual editor will be checked when the button is pressed so that the
        # user can receive an explanatory message if the dialog failed to load.
        self.add_voice_button.setEnabled(is_single_selection)
        self.open_subliminal_button.setEnabled(is_single_selection)

        can_move_up = has_selection and min(selected_indices) > 0
        can_move_down = has_selection and max(selected_indices) < (num_steps - 1)
        self.move_step_up_button.setEnabled(can_move_up)
        self.move_step_down_button.setEnabled(can_move_down)

        # --- Test Step Preview Button States ---
        can_test_selected_step = is_single_selection and current_idx is not None and \
                        0 <= current_idx < len(self.track_data["steps"]) and \
                        len(self.track_data["steps"][current_idx].get("voices", [])) > 0 and \
                        AUDIO_GENERATION_AVAILABLE
        
        self.test_step_play_pause_button.setEnabled(can_test_selected_step)
        
        self.test_step_stop_button.setEnabled(
            self.is_step_test_playing or self.is_step_test_paused or (self.test_step_raw_audio is not None)
        )
        
        self.test_step_reset_button.setEnabled(self.test_step_raw_audio is not None or self.current_test_step_index != -1)
        
        self.test_step_time_slider.setEnabled(self.test_step_raw_audio is not None)

        # If selection changes away from the actively playing/paused step, reset the tester.
        if self.current_test_step_index != -1 and self.current_test_step_index != current_idx:
            if self.is_step_test_playing or self.is_step_test_paused:
                print(f"Selection changed from {self.current_test_step_index} to {current_idx} while tester active. Resetting.")
                self.on_reset_step_test() 

        # If the currently loaded step becomes untestable (e.g., voices removed), reset tester.
        if not can_test_selected_step and self.current_test_step_index == current_idx and current_idx is not None:
             print(f"Currently loaded step {current_idx} became untestable. Resetting tester.")
             self.on_reset_step_test()
        
        # If no single step is selected, clear voice-related UI and potentially reset tester if it was idle.
        if not is_single_selection:
            self.voice_model.refresh([])
            self.clear_voice_details()
            self.voices_groupbox.setTitle("Voices for Selected Step")
            if self.current_test_step_index == -1 and self.test_step_raw_audio is None:
                self.test_step_loaded_label.setText("No step loaded for preview.")


    def _update_voice_actions_state(self):
        selected_voice_indices = self.get_selected_voice_indices()
        num_selected_voices = len(selected_voice_indices)
        current_voice_idx = self.get_selected_voice_index()
        current_step_idx = self.get_selected_step_index()

        is_single_step_selected = len(self.get_selected_step_indices()) == 1 and current_step_idx is not None

        # Always allow the button click when a single step is selected. If the
        # editor dialog failed to import, the handler will inform the user when
        # they attempt to add a voice.
        self.add_voice_button.setEnabled(is_single_step_selected)

        if not is_single_step_selected:
            self.edit_voice_button.setEnabled(False)
            self.group_edit_button.setEnabled(False)
            self.group_swap_button.setEnabled(False)
            self.duplicate_voice_button.setEnabled(False)
            self.remove_voice_button.setEnabled(False)
            self.save_voices_button.setEnabled(False)
            self.load_voices_button.setEnabled(False)
            self.move_voice_up_button.setEnabled(False)
            self.move_voice_down_button.setEnabled(False)
            self.copy_voices_button.setEnabled(False)
            self.paste_voices_button.setEnabled(False)
            return

        has_voice_selection = num_selected_voices > 0
        is_single_voice_selection = num_selected_voices == 1

        # Enable editing whenever a single voice is selected. The click handler
        # will report an error if the dialog cannot be loaded.
        self.edit_voice_button.setEnabled(is_single_voice_selection)
        self.group_edit_button.setEnabled(num_selected_voices > 1)
        has_transition_selection = False
        if current_step_idx is not None and 0 <= current_step_idx < len(self.track_data["steps"]):
            voices = self.track_data["steps"][current_step_idx].get("voices", [])
            for vi in selected_voice_indices:
                if 0 <= vi < len(voices) and voices[vi].get("is_transition"):
                    has_transition_selection = True
                    break
        self.group_swap_button.setEnabled(has_transition_selection)
        self.duplicate_voice_button.setEnabled(is_single_voice_selection)
        self.remove_voice_button.setEnabled(has_voice_selection)
        self.save_voices_button.setEnabled(has_voice_selection)
        self.load_voices_button.setEnabled(True)

        self.copy_voices_button.setEnabled(has_voice_selection)
        self.paste_voices_button.setEnabled(bool(self._copied_voices))

        num_voices_in_current_step = 0
        current_step_idx = self.get_selected_step_index()
        if current_step_idx is not None and 0 <= current_step_idx < len(self.track_data["steps"]):
            num_voices_in_current_step = len(self.track_data["steps"][current_step_idx].get("voices", []))


        can_move_voice_up = has_voice_selection and min(selected_voice_indices) > 0
        can_move_voice_down = has_voice_selection and max(selected_voice_indices) < (num_voices_in_current_step - 1)
        self.move_voice_up_button.setEnabled(can_move_voice_up)
        self.move_voice_down_button.setEnabled(can_move_voice_down)

    def _update_clip_actions_state(self):
        selected_items = self.clips_tree.selectedItems()
        has_selection = len(selected_items) > 0
        is_single = len(selected_items) == 1
        self.edit_clip_button.setEnabled(is_single)
        self.remove_clip_button.setEnabled(has_selection)
        self.play_clip_button.setEnabled(is_single or self.is_clip_playing)

    @pyqtSlot()
    def on_clip_select(self):
        self._update_clip_actions_state()

    # --- Internal Data Handling ---
    def _update_global_settings_from_ui(self):
        try:
            sr_str = self.sr_entry.text()
            cf_str = self.cf_entry.text()
            outfile = self.outfile_entry.text().strip()
            noise_file = self.noise_file_entry.text().strip()
            noise_amp_str = self.noise_amp_entry.text()
            if not sr_str: raise ValueError("Sample rate cannot be empty.")
            sr = int(sr_str)
            if sr <= 0: raise ValueError("Sample rate must be positive.")
            self.track_data["global_settings"]["sample_rate"] = sr
            if not cf_str: raise ValueError("Crossfade duration cannot be empty.")
            cf_str_safe = cf_str.replace(',', '.')
            cf = float(cf_str_safe)
            if cf < 0: raise ValueError("Crossfade duration cannot be negative.")
            self.track_data["global_settings"]["crossfade_duration"] = cf
            # Ensure step start times reflect any change in crossfade duration
            self._recalculate_step_start_times()
            if not outfile: raise ValueError("Output filename cannot be empty.")
            if any(c in outfile for c in '<>:"/\\|?*'):
                raise ValueError("Output filename contains invalid characters.")
            self.track_data["global_settings"]["output_filename"] = outfile

            self.track_data.setdefault(
                "background_noise",
                {
                    "file_path": "",
                    "amp": 0.0,
                    "pan": 0.0,
                    "start_time": 0.0,
                    "fade_in": 0.0,
                    "fade_out": 0.0,
                    "amp_envelope": [],
                },
            )
            self.track_data["background_noise"]["file_path"] = noise_file
            try:
                noise_amp = float(noise_amp_str) if noise_amp_str else 0.0
            except ValueError:
                raise ValueError("Invalid noise amplitude")
            self.track_data["background_noise"]["amp"] = noise_amp
            # Noise pan parameter removed from UI; value remains unchanged
        except ValueError as e:
            QMessageBox.critical(self, "Input Error", f"Invalid global settings:\n{e}")
            return False
        except Exception as e:
            QMessageBox.critical(self, "Input Error", f"Unexpected error reading global settings:\n{e}")
            return False
        return True

    def _update_ui_from_global_settings(self):
        settings = self.track_data.get("global_settings", {})
        self.sr_entry.setText(str(settings.get("sample_rate", DEFAULT_SAMPLE_RATE)))
        self.cf_entry.setText(str(settings.get("crossfade_duration", DEFAULT_CROSSFADE)))
        self.outfile_entry.setText(settings.get("output_filename", "my_track.wav"))
        noise = self.track_data.get("background_noise", {})
        self.noise_file_entry.setText(noise.get("file_path", ""))
        self.noise_amp_entry.setText(str(noise.get("amp", 0.0)))

    def _recalculate_step_start_times(self):
        crossfade = float(self.track_data.get("global_settings", {}).get("crossfade_duration", 0.0))
        current_time = 0.0
        for step in self.track_data.get("steps", []):
            step["start"] = current_time
            advance = float(step.get("duration", 0.0))
            if crossfade > 0.0:
                advance = max(0.0, advance - crossfade)
            current_time += advance

    def _update_total_duration_label(self):
        total = 0.0
        for step in self.track_data.get("steps", []):
            try:
                total += float(step.get("duration", 0.0))
            except (TypeError, ValueError):
                continue
        self.total_duration_label.setText(f"Total Duration: {total:.2f} s")

    # --- UI Refresh Functions ---
    def refresh_steps_tree(self):
        self._steps_tree_updating = True
        current_row = self.get_selected_step_index()
        selected_rows = self.get_selected_step_indices()
        self._recalculate_step_start_times()
        self.step_model.refresh(self.track_data.get("steps", []))
        sel_model = self.steps_tree.selectionModel()
        sel_model.clearSelection()
        for row in selected_rows:
            if 0 <= row < self.step_model.rowCount():
                idx = self.step_model.index(row, 0)
                sel_model.select(idx, QItemSelectionModel.Select | QItemSelectionModel.Rows)
        if current_row is not None and 0 <= current_row < self.step_model.rowCount():
            idx = self.step_model.index(current_row, 0)
            self.steps_tree.setCurrentIndex(idx)
            self.steps_tree.scrollTo(idx, QAbstractItemView.PositionAtCenter)
        self.on_step_select()
        self._update_total_duration_label()
        self._steps_tree_updating = False

    def refresh_voices_tree(self):
        self._voices_tree_updating = True
        current_row = self.get_selected_voice_index()
        selected_rows = self.get_selected_voice_indices()
        self.clear_voice_details()
        selected_step_idx = self.get_selected_step_index()
        if selected_step_idx is None or len(self.get_selected_step_indices()) != 1:
            self.voice_model.refresh([])
            self.voices_groupbox.setTitle("Voices for Selected Step")
            self._update_voice_actions_state()
            self._voices_tree_updating = False
            return
        self.voices_groupbox.setTitle(f"Voices for Step {selected_step_idx + 1}")
        try:
            step_data = self.track_data["steps"][selected_step_idx]
            voices = step_data.get("voices", [])
        except (IndexError, KeyError):
            voices = []
        self.voice_model.refresh(voices)
        sel_model = self.voices_tree.selectionModel()
        sel_model.clearSelection()
        for row in selected_rows:
            if 0 <= row < self.voice_model.rowCount():
                idx = self.voice_model.index(row, 0)
                sel_model.select(idx, QItemSelectionModel.Select | QItemSelectionModel.Rows)
        if current_row is not None and 0 <= current_row < self.voice_model.rowCount():
            idx = self.voice_model.index(current_row, 0)
            self.voices_tree.setCurrentIndex(idx)
            self.voices_tree.scrollTo(idx, QAbstractItemView.PositionAtCenter)
        self.on_voice_select()
        self._voices_tree_updating = False

    def refresh_clips_tree(self):
        self._clips_tree_updating = True
        current_data = None
        current_item = self.clips_tree.currentItem()
        if current_item:
            current_data = current_item.data(0, Qt.UserRole)
        selected = set()
        for item in self.clips_tree.selectedItems():
            data = item.data(0, Qt.UserRole)
            if data is not None:
                selected.add(data)
        self.clips_tree.clear()
        clips = self.track_data.get("clips", [])
        for i, clip in enumerate(clips):
            item = QTreeWidgetItem(self.clips_tree)
            item.setText(0, os.path.basename(clip.get("file_path", "")))
            item.setText(1, clip.get("description", ""))
            start = float(clip.get('start', 0.0))
            duration = float(clip.get('duration', 0.0))
            if duration <= 0 and clip.get('file_path'):
                duration = self._get_clip_duration(clip['file_path'])
                clip['duration'] = duration
            finish = start + duration
            item.setText(2, f"{start:.2f}")
            item.setText(3, f"{duration:.2f}")
            item.setText(4, f"{finish:.2f}")
            item.setText(5, str(clip.get('amp', 1.0)))
            item.setText(6, str(clip.get('pan', 0.0)))
            item.setText(7, str(clip.get('fade_in', 0.0)))
            item.setText(8, str(clip.get('fade_out', 0.0)))
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            item.setData(0, Qt.UserRole, i)
            if i in selected:
                item.setSelected(True)
                if i == current_data:
                    self.clips_tree.setCurrentItem(item)
        self._update_clip_actions_state()
        self._clips_tree_updating = False

    def clear_voice_details(self):
        self.voice_details_text.clear()
        self.voice_details_groupbox.setTitle("Selected Voice Details")

    def update_voice_details(self):
        self.clear_voice_details()
        if len(self.get_selected_step_indices()) != 1 or len(self.get_selected_voice_indices()) != 1:
            return
        selected_step_idx = self.get_selected_step_index()
        selected_voice_idx = self.get_selected_voice_index()
        if selected_step_idx is None or selected_voice_idx is None: return
        try:
            voice_data = self.track_data["steps"][selected_step_idx]["voices"][selected_voice_idx]
            func_name = voice_data.get('synth_function_name', 'N/A')
            self.voice_details_groupbox.setTitle(f"Details: Step {selected_step_idx+1}, Voice {selected_voice_idx+1} ({func_name})")
            params = voice_data.get("params", {})
            details = f"Function: {func_name}\n"
            details += f"Transition: {'Yes' if voice_data.get('is_transition', False) else 'No'}\n"
            desc = voice_data.get('description', '')
            if desc:
                details += f"Description: {desc}\n"
            details += "Parameters:\n"
            if params:
                for key, value in sorted(params.items()):
                    if isinstance(value, float): details += f"  {key}: {value:.4g}\n"
                    else: details += f"  {key}: {value}\n"
            else: details += "  (No parameters defined)\n"
            env_data = voice_data.get("volume_envelope")
            if env_data and isinstance(env_data, dict):
                env_type = env_data.get('type', 'N/A')
                details += f"\nEnvelope Type: {env_type}\n"
                env_params = env_data.get('params', {})
                if env_params:
                    details += "  Envelope Params:\n"
                    for key, value in sorted(env_params.items()):
                        if isinstance(value, float): details += f"    {key}: {value:.4g}\n"
                        else: details += f"    {key}: {value}\n"
                else: details += "  (No envelope parameters defined)\n"
            else: details += "\nEnvelope Type: None\n"
            self.voice_details_text.setPlainText(details)
        except (IndexError, KeyError) as e:
            print(f"Error accessing voice data for details: Step {selected_step_idx}, Voice {selected_voice_idx}. Error: {e}")
        except Exception as e:
            print(f"Unexpected error updating voice details: {e}")
            traceback.print_exc()

    # --- Event Handlers (Slots) ---
    @pyqtSlot()
    def on_step_select(self):
        current_selected_idx = self.get_selected_step_index()

        if len(self.get_selected_step_indices()) == 1 and current_selected_idx is not None:
            self.refresh_voices_tree() # Also calls on_voice_select -> _update_voice_actions_state
            # Update the test preview label if the tester is currently idle
            if not self.is_step_test_playing and not self.is_step_test_paused and self.test_step_raw_audio is None:
                try:
                    step_data = self.track_data["steps"][current_selected_idx]
                    try:
                        step_duration = float(step_data.get("duration", 0.0))
                    except (TypeError, ValueError):
                        step_duration = 0.0
                    if hasattr(self, "test_step_duration_spin"):
                        max_val = step_duration if step_duration > 0.0 else 9999.0
                        self.test_step_duration_spin.setMaximum(max_val)
                        if step_duration > 0.0 and self.test_step_duration_spin.value() > step_duration:
                            self.test_step_duration_spin.setValue(step_duration)
                    step_desc = step_data.get("description", "N/A")
                    is_testable = len(step_data.get("voices", [])) > 0 and AUDIO_GENERATION_AVAILABLE
                    short_desc = f"{step_desc[:30]}{'...' if len(step_desc) > 30 else ''}"
                    if is_testable:
                        self.test_step_loaded_label.setText(f"Ready: {short_desc}")
                    else:
                        self.test_step_loaded_label.setText(f"Info: {short_desc} (not testable)")
                except IndexError:
                    self.test_step_loaded_label.setText("Error: Step not found.")
        else:
            # No single step selected (0 or multiple)
            self.voice_model.refresh([])
            self.clear_voice_details()
            self.voices_groupbox.setTitle("Voices for Selected Step")
            if not self.is_step_test_playing and not self.is_step_test_paused and self.test_step_raw_audio is None:
                self.test_step_loaded_label.setText("No step loaded for preview.")
            if hasattr(self, "test_step_duration_spin"):
                self.test_step_duration_spin.setMaximum(9999.0)
            self._update_voice_actions_state() # Update voice buttons if step selection changes to non-single

        self._update_step_actions_state() # Update all step and test preview button states

    @pyqtSlot()
    def on_voice_select(self):
        self._update_voice_actions_state()
        self.update_voice_details()



    @pyqtSlot(QTreeWidgetItem, int)
    def on_clip_item_changed(self, item, column):
        if self._clips_tree_updating or column != 1:
            return
        idx = item.data(0, Qt.UserRole)
        if idx is None:
            return
        try:
            new_desc = item.text(1).strip()
            self.track_data["clips"][idx]["description"] = new_desc
            self._push_history_state()
        except Exception as e:
            print(f"Error updating clip description: {e}")

    # --- Action Methods (File Ops, Step/Voice Ops) ---
    @pyqtSlot()
    def new_file(self):
        if self.is_step_test_playing or self.is_step_test_paused or self.test_step_raw_audio:
            self.on_reset_step_test() # Reset tester before loading new file content
        if self.is_clip_playing:
            self._stop_clip_playback()
        self.track_data = self._get_default_track_data()
        self.current_json_path = None
        self._update_ui_from_global_settings()
        self.refresh_clips_tree()
        self.refresh_steps_tree() # This calls on_step_select -> _update_step_actions_state
        self.setWindowTitle("Binaural Track Editor (PyQt5) - New File")
        QMessageBox.information(self, "New File", "New track created.")
        self._push_history_state()

    @pyqtSlot()
    def load_json(self):
        if self.is_step_test_playing or self.is_step_test_paused or self.test_step_raw_audio:
            self.on_reset_step_test() # Reset tester before loading new file content
        if self.is_clip_playing:
            self._stop_clip_playback()
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Track JSON", "", "JSON files (*.json);;All files (*.*)")
        if not filepath: return
        try:
            loaded_data = sound_creator.load_track_from_json(filepath)
            if loaded_data and isinstance(loaded_data.get("steps"), list) and isinstance(loaded_data.get("global_settings"), dict):
                self.track_data = loaded_data
                self.track_data.setdefault(
                    "background_noise",
                    {
                        "file_path": "",
                        "amp": 0.0,
                        "pan": 0.0,
                        "start_time": 0.0,
                        "fade_in": 0.0,
                        "fade_out": 0.0,
                        "amp_envelope": [],
                    },
                )
                self.track_data.setdefault("clips", [])
                self.current_json_path = filepath
                self.setWindowTitle(f"Binaural Track Editor (PyQt5) - {os.path.basename(filepath)}")
                self._update_ui_from_global_settings()
                self.refresh_steps_tree() # This calls on_step_select -> _update_step_actions_state
                self.refresh_clips_tree()
                QMessageBox.information(self, "Load Success", f"Track loaded from\n{filepath}")
                self._push_history_state()
            elif loaded_data is not None:
                QMessageBox.critical(self, "Load Error", "Invalid JSON structure for track data.")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load or process JSON file:\n{e}")
            traceback.print_exc()
        # No explicit call to _update_step/voice_actions_state here, refresh_steps_tree handles it.

    @pyqtSlot()
    def save_json(self):
        if not self.current_json_path:
            self.save_json_as()
        else:
            if not self._update_global_settings_from_ui(): return
            try:
                if sound_creator.save_track_to_json(self.track_data, self.current_json_path):
                    QMessageBox.information(self, "Save Success", f"Track saved to\n{self.current_json_path}")
                    self.setWindowTitle(f"Binaural Track Editor (PyQt5) - {os.path.basename(self.current_json_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save JSON file:\n{e}")
                traceback.print_exc()

    @pyqtSlot()
    def save_json_as(self):
        if not self._update_global_settings_from_ui(): return
        initial_filename = "track_definition.json"
        initial_dir = os.path.dirname(self.current_json_path) if self.current_json_path else "."
        try:
            wav_filename = self.track_data["global_settings"].get("output_filename", "my_track.wav")
            base, _ = os.path.splitext(wav_filename)
            initial_filename = base + ".json"
        except Exception: pass
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Track JSON As", os.path.join(initial_dir, initial_filename), "JSON files (*.json);;All files (*.*)")
        if not filepath: return
        try:
            if sound_creator.save_track_to_json(self.track_data, filepath):
                self.current_json_path = filepath
                self.setWindowTitle(f"Binaural Track Editor (PyQt5) - {os.path.basename(filepath)}")
                QMessageBox.information(self, "Save Success", f"Track saved to\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save JSON file as:\n{e}")
            traceback.print_exc()

    @pyqtSlot()
    def browse_outfile(self):
        initial_filename = self.outfile_entry.text()
        initial_dir = os.path.dirname(self.current_json_path) if self.current_json_path else "."
        suggested_path = os.path.join(initial_dir, initial_filename if initial_filename else "my_track.wav")
        filepath, _ = QFileDialog.getSaveFileName(self, "Select Output WAV File", suggested_path, "WAV files (*.wav);;All files (*.*)")
        if filepath:
            self.outfile_entry.setText(filepath)
            self._update_global_settings_from_ui()

    @pyqtSlot()
    def browse_noise_file(self):
        initial_dir = os.path.dirname(self.current_json_path) if self.current_json_path else "."
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Noise Preset",
            initial_dir,
            f"Noise Files (*{'.noise'});;All files (*.*)"
        )
        if path:
            self.noise_file_entry.setText(path)
            self._update_global_settings_from_ui()

    @pyqtSlot()
    def load_external_step(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load External Steps from JSON",
            "",
            "JSON files (*.json);;All files (*.*)"
        )
        if not filepath:
            return
        try:
            external_data = sound_creator.load_track_from_json(filepath)
            if not external_data or "steps" not in external_data or not isinstance(external_data["steps"], list):
                QMessageBox.critical(
                    self,
                    "Load Error",
                    "Invalid JSON structure: 'steps' key missing or not a list."
                )
                return
            external_steps = external_data["steps"]
            if not external_steps:
                QMessageBox.information(self, "Load Info", "The selected file contains no steps to load.")
                return
            current_step_count = len(self.track_data["steps"])
            loaded_count = 0
            for step_data in external_steps:
                if isinstance(step_data, dict) and "duration" in step_data and "voices" in step_data:
                    self.track_data["steps"].append(copy.deepcopy(step_data))
                    loaded_count +=1
                else:
                    QMessageBox.warning(self, "Load Warning", f"Skipping an invalid step structure: {str(step_data)[:100]}")
            if loaded_count > 0:
                self.refresh_steps_tree()
                QMessageBox.information(self, "Load Success", f"{loaded_count} step(s) loaded and added from\n{filepath}")
                if current_step_count < len(self.track_data["steps"]):
                    idx = self.step_model.index(current_step_count, 0)
                    self.steps_tree.selectionModel().clearSelection()
                    self.steps_tree.setCurrentIndex(idx)
                    self.steps_tree.selectionModel().select(idx, QItemSelectionModel.Select | QItemSelectionModel.Rows)
                    self.steps_tree.scrollTo(idx, QAbstractItemView.PositionAtCenter)
                self._push_history_state()
            else:
                QMessageBox.information(self, "Load Info", "No valid steps found to load.")
        except json.JSONDecodeError:
            QMessageBox.critical(self, "Load Error", f"Failed to decode JSON from file:\n{filepath}")
            traceback.print_exc()
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"An unexpected error occurred:\n{e}")
            traceback.print_exc()
        # refresh_steps_tree calls on_step_select which calls _update_step_actions_state

    @pyqtSlot()
    def add_step(self):
        new_step = {"duration": 10.0, "description": "New Step", "voices": []}
        selected_focused_index = self.get_selected_step_index()
        insert_index = selected_focused_index + 1 if selected_focused_index is not None else len(self.track_data["steps"])
        self.track_data["steps"].insert(insert_index, new_step)
        self.refresh_steps_tree()
        if 0 <= insert_index < self.step_model.rowCount():
            idx = self.step_model.index(insert_index, 0)
            self.steps_tree.selectionModel().clearSelection()
            self.steps_tree.setCurrentIndex(idx)
            self.steps_tree.selectionModel().select(idx, QItemSelectionModel.Select | QItemSelectionModel.Rows)
            self.steps_tree.scrollTo(idx, QAbstractItemView.PositionAtCenter)
        self._push_history_state()
        # refresh_steps_tree calls on_step_select which calls _update_step_actions_state

    @pyqtSlot()
    def duplicate_step(self):
        selected_index = self.get_selected_step_index()
        if selected_index is None or len(self.get_selected_step_indices()) != 1:
            QMessageBox.warning(self, "Duplicate Step", "Please select exactly one step to duplicate.")
            return
        try:
            original_step_data = self.track_data["steps"][selected_index]
            duplicated_step_data = copy.deepcopy(original_step_data)
            insert_index = selected_index + 1
            self.track_data["steps"].insert(insert_index, duplicated_step_data)
            self.refresh_steps_tree()
            if 0 <= insert_index < self.step_model.rowCount():
                idx = self.step_model.index(insert_index, 0)
                self.steps_tree.selectionModel().clearSelection()
                self.steps_tree.setCurrentIndex(idx)
                self.steps_tree.selectionModel().select(idx, QItemSelectionModel.Select | QItemSelectionModel.Rows)
                self.steps_tree.scrollTo(idx, QAbstractItemView.PositionAtCenter)
            self._push_history_state()
        except IndexError: QMessageBox.critical(self, "Error", "Failed to duplicate step (index out of range).")
        except Exception as e: QMessageBox.critical(self, "Error", f"Failed to duplicate step:\n{e}"); traceback.print_exc()
        # refresh_steps_tree calls on_step_select which calls _update_step_actions_state

    @pyqtSlot()
    def create_end_state_step(self):
        selected_index = self.get_selected_step_index()
        if selected_index is None or len(self.get_selected_step_indices()) != 1:
            QMessageBox.warning(self, "Create End State Step", "Please select exactly one step to use as the previous step.")
            return
        try:
            prev_step = self.track_data["steps"][selected_index]
        except IndexError:
            QMessageBox.critical(self, "Error", "Failed to access selected step.")
            return

        new_step = copy.deepcopy(prev_step)
        for voice in new_step.get("voices", []):
            if voice.get("is_transition", False):
                params = voice.get("params", {})
                new_params = {}
                for k, v in params.items():
                    base = None
                    if k.startswith("end_"):
                        base = k[4:]
                    elif k.startswith("end"):
                        base = k[3:]
                    elif k.startswith("start_") or k.startswith("start"):
                        continue
                    if base is not None:
                        if base and base[0].isupper():
                            base = base[0].lower() + base[1:]
                        new_params[base] = v
                    else:
                        new_params[k] = v

                func_name = voice.get("synth_function_name", "")
                if func_name.endswith("_transition"):
                    func_name = func_name[:-11]
                voice["synth_function_name"] = func_name
                voice["is_transition"] = False
                voice["params"] = new_params

        insert_index = selected_index + 1
        self.track_data["steps"].insert(insert_index, new_step)
        self.refresh_steps_tree()
        if 0 <= insert_index < self.step_model.rowCount():
            idx = self.step_model.index(insert_index, 0)
            self.steps_tree.selectionModel().clearSelection()
            self.steps_tree.setCurrentIndex(idx)
            self.steps_tree.selectionModel().select(idx, QItemSelectionModel.Select | QItemSelectionModel.Rows)
            self.steps_tree.scrollTo(idx, QAbstractItemView.PositionAtCenter)
        self._push_history_state()

    @pyqtSlot()
    def remove_step(self):
        selected_indices = self.get_selected_step_indices()
        if not selected_indices:
            QMessageBox.warning(self, "Remove Step(s)", "Please select one or more steps to remove.")
            return
        
        # If any of the steps to be removed is the one currently loaded in tester, reset tester
        if self.current_test_step_index in selected_indices:
            self.on_reset_step_test()

        reply = QMessageBox.question(self, "Confirm Remove", f"Are you sure you want to remove {len(selected_indices)} selected step(s)?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            try:
                for index in sorted(selected_indices, reverse=True):
                    if 0 <= index < len(self.track_data["steps"]):
                        del self.track_data["steps"][index]
                self.refresh_steps_tree()
                self._push_history_state()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to remove step(s):\n{e}"); traceback.print_exc()
        # refresh_steps_tree calls on_step_select which calls _update_step_actions_state

    @pyqtSlot()
    def edit_step_duration(self):
        selected_index = self.get_selected_step_index()
        if selected_index is None or len(self.get_selected_step_indices()) != 1:
            QMessageBox.warning(self, "Edit Duration", "Please select exactly one step to edit.")
            return
        try: current_duration = float(self.track_data["steps"][selected_index].get("duration", 0.0))
        except (IndexError, ValueError, TypeError) as e: QMessageBox.critical(self, "Error", f"Failed to get current duration (index {selected_index}):\n{e}"); return
        new_duration, ok = QInputDialog.getDouble(self, f"Edit Step {selected_index + 1} Duration", "New Duration (s):", current_duration, 0.001, 99999.0, 3)
        if ok and new_duration is not None:
            if new_duration <= 0: QMessageBox.warning(self, "Invalid Input", "Duration must be positive."); return
            try:
                self.track_data["steps"][selected_index]["duration"] = new_duration
                self.refresh_steps_tree()
                self._push_history_state()
            except IndexError: QMessageBox.critical(self, "Error", "Failed to set duration (index out of range after edit).")
            except Exception as e: QMessageBox.critical(self, "Error", f"Failed to set duration:\n{e}")

    @pyqtSlot()
    def edit_step_description(self):
        selected_index = self.get_selected_step_index()
        if selected_index is None or len(self.get_selected_step_indices()) != 1:
            QMessageBox.warning(self, "Edit Description", "Please select exactly one step to edit.")
            return
        try: current_description = str(self.track_data["steps"][selected_index].get("description", ""))
        except IndexError as e: QMessageBox.critical(self, "Error", f"Failed to get current description (index {selected_index}):\n{e}"); return
        new_description, ok = QInputDialog.getText(self, f"Edit Step {selected_index + 1} Description", "Description:", QLineEdit.Normal, current_description)
        if ok and new_description is not None:
            try:
                self.track_data["steps"][selected_index]["description"] = new_description.strip()
                self.refresh_steps_tree() # This will update the display
                self._push_history_state()
                # If this was the currently loaded step in the tester, update its label
                if self.current_test_step_index == selected_index and self.test_step_raw_audio:
                    short_desc = f"{new_description.strip()[:30]}{'...' if len(new_description.strip()) > 30 else ''}"
                    if self.is_step_test_playing:
                        self.test_step_loaded_label.setText(f"Playing: {short_desc}")
                    elif self.is_step_test_paused:
                        self.test_step_loaded_label.setText(f"Paused: {short_desc}")
                    else: # Stopped but loaded
                        self.test_step_loaded_label.setText(f"Ready: {short_desc}")
                elif self.current_test_step_index == -1 and self.test_step_raw_audio is None and self.get_selected_step_index() == selected_index:
                    # Tester is idle, and the edited step is the currently selected one
                    self.on_step_select() # Re-evaluate "Ready" label

            except IndexError: QMessageBox.critical(self, "Error", "Failed to set description (index out of range after edit).")
            except Exception as e: QMessageBox.critical(self, "Error", f"Failed to set description:\n{e}")

    @pyqtSlot(int)
    def move_step(self, direction):
        selected_indices = self.get_selected_step_indices()
        if not selected_indices:
            QMessageBox.warning(self, "Move Step", "Please select one or more steps to move.")
            return

        if self.current_test_step_index in selected_indices:
            self.on_reset_step_test()

        num_steps = len(self.track_data["steps"])
        if direction < 0:
            if min(selected_indices) == 0:
                return
            iter_indices = selected_indices
        else:
            if max(selected_indices) >= num_steps - 1:
                return
            iter_indices = sorted(selected_indices, reverse=True)

        try:
            steps = self.track_data["steps"]
            for idx in iter_indices:
                steps[idx], steps[idx + direction] = steps[idx + direction], steps[idx]
            self.refresh_steps_tree()
            sel_model = self.steps_tree.selectionModel()
            sel_model.clearSelection()
            for idx in [i + direction for i in selected_indices]:
                if 0 <= idx < self.step_model.rowCount():
                    qidx = self.step_model.index(idx, 0)
                    sel_model.select(qidx, QItemSelectionModel.Select | QItemSelectionModel.Rows)
            anchor = min(selected_indices) + direction if direction < 0 else max(selected_indices) + direction
            if 0 <= anchor < self.step_model.rowCount():
                self.steps_tree.scrollTo(self.step_model.index(anchor, 0), QAbstractItemView.PositionAtCenter)
            self._push_history_state()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to move step:\n{e}"); traceback.print_exc()

    @pyqtSlot()
    def add_voice(self):
        # If the dialog could not be loaded, the fallback implementation will
        # display an informative message when executed. Therefore we allow the
        # user to click the button even when VOICE_EDITOR_DIALOG_AVAILABLE is
        # False.
        selected_step_index = self.get_selected_step_index()
        if selected_step_index is None or len(self.get_selected_step_indices()) != 1:
            QMessageBox.warning(self, "Add Voice", "Please select exactly one step first.")
            return
        try:
            current_voices = self.track_data["steps"][selected_step_index].get("voices", [])
            if len(current_voices) >= MAX_VOICES_PER_STEP:
                QMessageBox.warning(self, "Add Voice", f"Maximum voices per step ({MAX_VOICES_PER_STEP}) reached.")
                return
        except IndexError: QMessageBox.critical(self, "Error", "Cannot add voice (selected step index out of range)."); return
        dialog = VoiceEditorDialog(parent=self, app_ref=self, step_index=selected_step_index, voice_index=None)
        if dialog.exec_() == QDialog.Accepted:
            self.refresh_steps_tree() # This updates step voice count and calls on_step_select
            # If the modified step was the one loaded in tester and it's now testable (or was already)
            # and tester is idle, on_step_select will update the "Ready" label.
            if 0 <= selected_step_index < self.step_model.rowCount():
                idx = self.step_model.index(selected_step_index, 0)
                self.steps_tree.setCurrentIndex(idx)
                QTimer.singleShot(0, lambda: self._select_last_voice_in_current_step())
            self._push_history_state()

    def _select_last_voice_in_current_step(self):
        voice_count = self.voice_model.rowCount()
        if voice_count > 0:
            idx = self.voice_model.index(voice_count - 1, 0)
            self.voices_tree.selectionModel().clearSelection()
            self.voices_tree.setCurrentIndex(idx)
            self.voices_tree.selectionModel().select(idx, QItemSelectionModel.Select | QItemSelectionModel.Rows)
            self.voices_tree.scrollTo(idx, QAbstractItemView.PositionAtCenter)
        self._update_voice_actions_state()

    @pyqtSlot()
    def edit_voice(self):
        # If the editor dialog failed to load we still allow the action so the
        # fallback dialog can inform the user about the issue.
        selected_step_index = self.get_selected_step_index()
        selected_voice_index = self.get_selected_voice_index()
        if selected_step_index is None or selected_voice_index is None or \
           len(self.get_selected_step_indices()) != 1 or len(self.get_selected_voice_indices()) != 1:
            QMessageBox.warning(self, "Edit Voice", "Please select exactly one step and one voice to edit.")
            return
        
        # If editing a voice in the currently loaded step, it might change its test audio.
        # For simplicity, reset the tester. The user can then replay.
        if self.current_test_step_index == selected_step_index and self.test_step_raw_audio:
            self.on_reset_step_test()

        dialog = VoiceEditorDialog(parent=self, app_ref=self, step_index=selected_step_index, voice_index=selected_voice_index)
        if dialog.exec_() == QDialog.Accepted:
            self.refresh_steps_tree() # This updates step voice count and calls on_step_select
            if 0 <= selected_step_index < self.step_model.rowCount():
                step_idx = self.step_model.index(selected_step_index, 0)
                self.steps_tree.setCurrentIndex(step_idx)
                if 0 <= selected_voice_index < self.voice_model.rowCount():
                    voice_idx = self.voice_model.index(selected_voice_index, 0)
                    self.voices_tree.setCurrentIndex(voice_idx)
                    self.voices_tree.scrollTo(voice_idx, QAbstractItemView.PositionAtCenter)
            self._update_voice_actions_state()
            self._push_history_state()

    @pyqtSlot()
    def group_edit_voices(self):
        selected_step_idx = self.get_selected_step_index()
        selected_voice_indices = self.get_selected_voice_indices()
        if selected_step_idx is None or len(self.get_selected_step_indices()) != 1:
            QMessageBox.warning(self, "Group Edit", "Please select exactly one step first.")
            return
        if len(selected_voice_indices) < 2:
            QMessageBox.warning(self, "Group Edit", "Select two or more voices to edit.")
            return
        try:
            voices_list = self.track_data["steps"][selected_step_idx]["voices"]
            base_func = voices_list[selected_voice_indices[0]].get("synth_function_name", "")
            for idx in selected_voice_indices[1:]:
                if voices_list[idx].get("synth_function_name", "") != base_func:
                    QMessageBox.warning(self, "Group Edit", "Selected voices must use the same synth function.")
                    return
        except Exception as exc:
            QMessageBox.critical(self, "Group Edit", f"Failed to load voices: {exc}")
            return

        if self.current_test_step_index == selected_step_idx and self.test_step_raw_audio:
            self.on_reset_step_test()

        voices = self.track_data["steps"][selected_step_idx]["voices"]
        original = copy.deepcopy(voices[selected_voice_indices[0]])

        dialog = VoiceEditorDialog(parent=self, app_ref=self, step_index=selected_step_idx,
                                   voice_index=selected_voice_indices[0])
        if dialog.exec_() == QDialog.Accepted:
            try:
                new_data = copy.deepcopy(voices[selected_voice_indices[0]])

                # Determine which top-level fields changed
                changed_fields = {
                    k: new_data.get(k)
                    for k in ["synth_function_name", "is_transition", "volume_envelope", "description"]
                    if new_data.get(k) != original.get(k)
                }

                # Determine which individual parameters changed
                old_params = original.get("params", {})
                new_params = new_data.get("params", {})
                changed_params = {
                    k: new_params.get(k)
                    for k in set(new_params.keys()) | set(old_params.keys())
                    if new_params.get(k) != old_params.get(k)
                }

                for idx in selected_voice_indices[1:]:
                    target = voices[idx]
                    for f, val in changed_fields.items():
                        target[f] = copy.deepcopy(val)
                    if changed_params:
                        target.setdefault("params", {})
                        for p, val in changed_params.items():
                            if val is None:
                                target["params"].pop(p, None)
                            else:
                                target["params"][p] = copy.deepcopy(val)

                self.refresh_steps_tree()
                if 0 <= selected_step_idx < self.step_model.rowCount():
                    step_idx = self.step_model.index(selected_step_idx, 0)
                    self.steps_tree.setCurrentIndex(step_idx)
                sel_model = self.voices_tree.selectionModel()
                sel_model.clearSelection()
                for vi in selected_voice_indices:
                    if 0 <= vi < self.voice_model.rowCount():
                        idx_obj = self.voice_model.index(vi, 0)
                        sel_model.select(idx_obj, QItemSelectionModel.Select | QItemSelectionModel.Rows)
                self.voices_tree.scrollTo(self.voice_model.index(selected_voice_indices[0], 0), QAbstractItemView.PositionAtCenter)
                self._update_voice_actions_state()
                self._push_history_state()
            except Exception as exc:
                QMessageBox.critical(self, "Group Edit", f"Failed to apply changes: {exc}")

    @pyqtSlot()
    def group_swap_transition_parameters(self):
        selected_step_idx = self.get_selected_step_index()
        selected_voice_indices = self.get_selected_voice_indices()
        if selected_step_idx is None or len(self.get_selected_step_indices()) != 1:
            QMessageBox.warning(self, "Group Swap", "Please select exactly one step first.")
            return
        if not selected_voice_indices:
            QMessageBox.warning(self, "Group Swap", "Select one or more voices to swap.")
            return

        try:
            voices = self.track_data["steps"][selected_step_idx]["voices"]
        except Exception as exc:
            QMessageBox.critical(self, "Group Swap", f"Failed to load voices: {exc}")
            return

        swapped_any = False
        for idx in selected_voice_indices:
            if idx < 0 or idx >= len(voices):
                continue
            voice = voices[idx]
            if not voice.get("is_transition"):
                continue
            params = voice.get("params", {})
            for name in list(params.keys()):
                if name.startswith("start_"):
                    base = name[6:]
                    end_name = "end_" + base
                elif name.startswith("start"):
                    base = name[5:]
                    end_name = "end" + base
                else:
                    continue
                if end_name in params:
                    params[name], params[end_name] = params[end_name], params[name]
                    swapped_any = True

        if not swapped_any:
            QMessageBox.warning(self, "Group Swap", "No transition voices selected.")
            return

        self.refresh_steps_tree()
        if 0 <= selected_step_idx < self.step_model.rowCount():
            step_idx = self.step_model.index(selected_step_idx, 0)
            self.steps_tree.setCurrentIndex(step_idx)

        sel_model = self.voices_tree.selectionModel()
        sel_model.clearSelection()
        for vi in selected_voice_indices:
            if 0 <= vi < self.voice_model.rowCount():
                idx_obj = self.voice_model.index(vi, 0)
                sel_model.select(idx_obj, QItemSelectionModel.Select | QItemSelectionModel.Rows)

        if selected_voice_indices:
            self.voices_tree.scrollTo(self.voice_model.index(selected_voice_indices[0], 0), QAbstractItemView.PositionAtCenter)

        self._update_voice_actions_state()
        self._push_history_state()

    @pyqtSlot()
    def duplicate_voice(self):
        selected_step_idx = self.get_selected_step_index()
        selected_voice_idx = self.get_selected_voice_index()
        if selected_step_idx is None or selected_voice_idx is None or \
           len(self.get_selected_step_indices()) != 1 or len(self.get_selected_voice_indices()) != 1:
            QMessageBox.warning(self, "Duplicate Voice", "Please select exactly one step and one voice to duplicate.")
            return

        if self.current_test_step_index == selected_step_idx and self.test_step_raw_audio:
            self.on_reset_step_test()

        try:
            voices_list = self.track_data["steps"][selected_step_idx]["voices"]
            original_voice = voices_list[selected_voice_idx]
            duplicated_voice = copy.deepcopy(original_voice)
            insert_idx = selected_voice_idx + 1
            voices_list.insert(insert_idx, duplicated_voice)
            self.refresh_voices_tree()
            if 0 <= insert_idx < self.voice_model.rowCount():
                idx = self.voice_model.index(insert_idx, 0)
                self.voices_tree.selectionModel().clearSelection()
                self.voices_tree.setCurrentIndex(idx)
                self.voices_tree.selectionModel().select(idx, QItemSelectionModel.Select | QItemSelectionModel.Rows)
                self.voices_tree.scrollTo(idx, QAbstractItemView.PositionAtCenter)
            self._push_history_state()
        except IndexError:
            QMessageBox.critical(self, "Error", "Failed to duplicate voice (index out of range).")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to duplicate voice:\n{e}")
            traceback.print_exc()
        self._update_voice_actions_state()

    @pyqtSlot()
    def remove_voice(self):
        selected_step_idx = self.get_selected_step_index()
        selected_voice_indices = self.get_selected_voice_indices()
        if selected_step_idx is None or len(self.get_selected_step_indices()) != 1:
            QMessageBox.warning(self, "Remove Voice(s)", "Please select exactly one step first.")
            return
        if not selected_voice_indices:
            QMessageBox.warning(self, "Remove Voice(s)", "Please select one or more voices to remove.")
            return
        
        # If removing voices from the currently loaded step, it might become untestable or change audio.
        # Reset the tester.
        if self.current_test_step_index == selected_step_idx and self.test_step_raw_audio:
            self.on_reset_step_test()
            
        try:
            voices_list = self.track_data["steps"][selected_step_idx].get("voices")
            if not voices_list: return
            reply = QMessageBox.question(self, "Confirm Remove", f"Remove {len(selected_voice_indices)} selected voice(s) from Step {selected_step_idx + 1}?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                for voice_idx in sorted(selected_voice_indices, reverse=True):
                    if 0 <= voice_idx < len(voices_list): del voices_list[voice_idx]
                self.refresh_steps_tree() # This calls on_step_select, which will update tester label if idle
                self._push_history_state()
        except IndexError: QMessageBox.critical(self, "Error", "Failed to remove voice(s) (step index out of range).")
        except Exception as e: QMessageBox.critical(self, "Error", f"Failed to remove voice(s):\n{e}"); traceback.print_exc()

    @pyqtSlot()
    def copy_selected_voices(self):
        selected_step_idx = self.get_selected_step_index()
        selected_voice_indices = self.get_selected_voice_indices()
        if selected_step_idx is None or len(self.get_selected_step_indices()) != 1:
            QMessageBox.warning(self, "Copy Voice(s)", "Please select exactly one step first.")
            return
        if not selected_voice_indices:
            QMessageBox.warning(self, "Copy Voice(s)", "Please select one or more voices to copy.")
            return
        try:
            voices_list = self.track_data["steps"][selected_step_idx]["voices"]
            self._copied_voices = [copy.deepcopy(voices_list[i]) for i in selected_voice_indices if 0 <= i < len(voices_list)]
        except Exception as e:
            QMessageBox.critical(self, "Copy Voice(s)", f"Failed to copy voices:\n{e}")
            self._copied_voices = []
        self._update_voice_actions_state()

    @pyqtSlot()
    def paste_copied_voices(self):
        if not self._copied_voices:
            QMessageBox.warning(self, "Paste Voice(s)", "No voices have been copied.")
            return
        selected_step_idx = self.get_selected_step_index()
        selected_voice_indices = self.get_selected_voice_indices()
        if selected_step_idx is None or len(self.get_selected_step_indices()) != 1:
            QMessageBox.warning(self, "Paste Voice(s)", "Please select exactly one step to paste into.")
            return
        try:
            voices_list = self.track_data["steps"][selected_step_idx].setdefault("voices", [])
            insert_idx = max(selected_voice_indices) + 1 if selected_voice_indices else len(voices_list)
            for voice in self._copied_voices:
                voices_list.insert(insert_idx, copy.deepcopy(voice))
                insert_idx += 1
            self.refresh_voices_tree()
            sel_model = self.voices_tree.selectionModel()
            sel_model.clearSelection()
            start_idx = insert_idx - len(self._copied_voices)
            for i in range(start_idx, insert_idx):
                if 0 <= i < self.voice_model.rowCount():
                    idx = self.voice_model.index(i, 0)
                    sel_model.select(idx, QItemSelectionModel.Select | QItemSelectionModel.Rows)
            if 0 <= insert_idx - 1 < self.voice_model.rowCount():
                self.voices_tree.scrollTo(self.voice_model.index(insert_idx - 1, 0), QAbstractItemView.PositionAtCenter)
            self._push_history_state()
        except Exception as e:
            QMessageBox.critical(self, "Paste Voice(s)", f"Failed to paste voices:\n{e}")
        self._update_voice_actions_state()

    @pyqtSlot(int)
    def move_voice(self, direction):
        selected_step_idx = self.get_selected_step_index()
        selected_voice_indices = self.get_selected_voice_indices()
        if selected_step_idx is None or not selected_voice_indices or len(self.get_selected_step_indices()) != 1:
            QMessageBox.warning(self, "Move Voice", "Please select one step and one or more voices to move.")
            return
        
        # Moving voices might change audio characteristics. Reset if it's the loaded step.
        if self.current_test_step_index == selected_step_idx and self.test_step_raw_audio:
            self.on_reset_step_test()

        try:
            voices_list = self.track_data["steps"][selected_step_idx]["voices"]
            num_voices = len(voices_list)

            if direction < 0:
                if min(selected_voice_indices) == 0:
                    return
                iter_indices = selected_voice_indices
            else:
                if max(selected_voice_indices) >= num_voices - 1:
                    return
                iter_indices = sorted(selected_voice_indices, reverse=True)

            for idx in iter_indices:
                voices_list[idx], voices_list[idx + direction] = voices_list[idx + direction], voices_list[idx]

            self.refresh_voices_tree()  # Calls on_voice_select -> _update_voice_actions_state

            sel_model = self.voices_tree.selectionModel()
            sel_model.clearSelection()
            for idx in [i + direction for i in selected_voice_indices]:
                if 0 <= idx < self.voice_model.rowCount():
                    qidx = self.voice_model.index(idx, 0)
                    sel_model.select(qidx, QItemSelectionModel.Select | QItemSelectionModel.Rows)
            anchor = min(selected_voice_indices) + direction if direction < 0 else max(selected_voice_indices) + direction
            if 0 <= anchor < self.voice_model.rowCount():
                self.voices_tree.scrollTo(self.voice_model.index(anchor, 0), QAbstractItemView.PositionAtCenter)

            self._push_history_state()
        except IndexError:
            QMessageBox.critical(self, "Error", "Failed to move voice (index out of range).")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred while moving voice:\n{e}")
        self._update_voice_actions_state()

    @pyqtSlot()
    def save_selected_voices(self):
        step_idx = self.get_selected_step_index()
        selected_voice_indices = self.get_selected_voice_indices()
        if step_idx is None or len(self.get_selected_step_indices()) != 1:
            QMessageBox.warning(self, "Save Voices", "Please select exactly one step first.")
            return
        if not selected_voice_indices:
            QMessageBox.warning(self, "Save Voices", "Please select one or more voices to save.")
            return
        try:
            voices = [
                copy.deepcopy(self.track_data["steps"][step_idx]["voices"][i])
                for i in selected_voice_indices
            ]
        except Exception as exc:
            QMessageBox.critical(self, "Save Voices", f"Failed to access selected voices:\n{exc}")
            return
        presets = [
            VoicePreset(
                synth_function_name=v.get("synth_function_name", ""),
                is_transition=v.get("is_transition", False),
                params=v.get("params", {}),
                volume_envelope=v.get("volume_envelope"),
                description=v.get("description", ""),
            )
            for v in voices
        ]
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Voices",
            "",
            f"Voice Lists (*{VOICES_FILE_EXTENSION})",
        )
        if not path:
            return
        try:
            save_voice_preset_list(presets, path)
        except Exception as exc:
            QMessageBox.critical(self, "Save Voices", f"Failed to save voices:\n{exc}")

    @pyqtSlot()
    def load_voices_from_file(self):
        step_idx = self.get_selected_step_index()
        if step_idx is None or len(self.get_selected_step_indices()) != 1:
            QMessageBox.warning(self, "Load Voices", "Please select exactly one step first.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Voices",
            "",
            f"Voice Lists (*{VOICES_FILE_EXTENSION})",
        )
        if not path:
            return
        try:
            presets = load_voice_preset_list(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load Voices", f"Failed to load voices:\n{exc}")
            return
        if not presets:
            QMessageBox.information(self, "Load Voices", "No voices found in file.")
            return
        voices_list = self.track_data["steps"][step_idx].setdefault("voices", [])
        for preset in presets:
            voices_list.append(
                {
                    "synth_function_name": preset.synth_function_name,
                    "is_transition": preset.is_transition,
                    "params": copy.deepcopy(preset.params),
                    "volume_envelope": copy.deepcopy(preset.volume_envelope),
                    "description": preset.description,
                }
            )
        self.refresh_steps_tree()
        if voices_list:
            last = len(voices_list) - 1
            sel_model = self.voices_tree.selectionModel()
            sel_model.clearSelection()
            for i in range(last - len(presets) + 1, last + 1):
                if 0 <= i < self.voice_model.rowCount():
                    idx = self.voice_model.index(i, 0)
                    sel_model.select(idx, QItemSelectionModel.Select | QItemSelectionModel.Rows)
            if 0 <= last < self.voice_model.rowCount():
                self.voices_tree.scrollTo(self.voice_model.index(last, 0), QAbstractItemView.PositionAtCenter)
        self._push_history_state()

    @pyqtSlot()
    def add_clip(self):
        dialog = OverlayClipDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            clip = dialog.get_clip_data()
            self.track_data.setdefault("clips", []).append(clip)
            self.refresh_clips_tree()
            self._push_history_state()

    @pyqtSlot()
    def edit_clip(self):
        idx = self.get_selected_clip_index()
        if idx is None or len(self.clips_tree.selectedItems()) != 1:
            QMessageBox.warning(self, "Edit Clip", "Please select one clip to edit.")
            return
        try:
            clip_data = self.track_data.get("clips", [])[idx]
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Invalid clip: {exc}")
            return
        dialog = OverlayClipDialog(self, clip_data)
        if dialog.exec_() == QDialog.Accepted:
            self.track_data["clips"][idx] = dialog.get_clip_data()
            self.refresh_clips_tree()
            self._push_history_state()

    @pyqtSlot()
    def remove_clip(self):
        indices = self.get_selected_clip_indices()
        if not indices:
            QMessageBox.warning(self, "Remove Clip", "Please select clip(s) to remove.")
            return
        reply = QMessageBox.question(
            self,
            "Confirm Remove",
            f"Remove {len(indices)} selected clip(s)?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            for i in sorted(indices, reverse=True):
                if 0 <= i < len(self.track_data.get("clips", [])):
                    del self.track_data["clips"][i]
            self.refresh_clips_tree()
            self._push_history_state()


    @pyqtSlot()
    def on_start_stop_clip(self):
        if self.is_clip_playing:
            self._stop_clip_playback()
        else:
            self._start_clip_playback()

    def _start_clip_playback(self):
        idx = self.get_selected_clip_index()
        if idx is None or len(self.clips_tree.selectedItems()) != 1:
            QMessageBox.warning(self, "Play Clip", "Please select one clip to play.")
            return
        try:
            clip_data = self.track_data.get("clips", [])[idx]
            path = clip_data.get("file_path")
            if not path or not os.path.isfile(path):
                QMessageBox.warning(self, "Play Clip", "Clip file not found.")
                return
            data, sr = sf.read(path, always_2d=True)
            if data.shape[1] == 1:
                data = np.repeat(data, 2, axis=1)
            data = np.clip(data, -1.0, 1.0)
            audio_int16 = (data * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()

            fmt = QAudioFormat()
            fmt.setCodec("audio/pcm")
            fmt.setSampleRate(int(sr))
            fmt.setSampleSize(16)
            fmt.setChannelCount(2)
            fmt.setByteOrder(QAudioFormat.LittleEndian)
            fmt.setSampleType(QAudioFormat.SignedInt)
            device_info = QAudioDeviceInfo.defaultOutputDevice()
            if not device_info.isFormatSupported(fmt):
                QMessageBox.warning(self, "Play Clip", "Output device does not support the required format.")
                return

            if self.clip_audio_output:
                self.clip_audio_output.stop()
                self.clip_audio_output = None

            self.clip_audio_output = QAudioOutput(fmt, self)
            self.clip_audio_output.stateChanged.connect(self._handle_clip_audio_state_change)

            self.clip_audio_buffer = QBuffer()
            self.clip_audio_buffer.setData(audio_bytes)
            self.clip_audio_buffer.open(QIODevice.ReadOnly)

            self.clip_audio_output.start(self.clip_audio_buffer)
            self.is_clip_playing = True
            self.play_clip_button.setText("Stop Clip")
            self._update_clip_actions_state()
        except Exception as e:
            QMessageBox.critical(self, "Play Clip", f"Failed to play clip:\n{e}")
            self._stop_clip_playback()

    def _stop_clip_playback(self):
        if self.clip_audio_output:
            self.clip_audio_output.stop()
            self.clip_audio_output = None
        if self.clip_audio_buffer:
            self.clip_audio_buffer.close()
            self.clip_audio_buffer = None
        self.is_clip_playing = False
        self.play_clip_button.setText("Start Clip")
        self._update_clip_actions_state()

    def _handle_clip_audio_state_change(self, state):
        if state in (QAudio.IdleState, QAudio.StoppedState):
            self._stop_clip_playback()

    # --- generate_audio_action ---
    @pyqtSlot()
    def generate_audio_action(self):
        if not self._update_global_settings_from_ui(): return
        current_track_data = self.track_data
        output_filepath = current_track_data["global_settings"].get("output_filename")
        if output_filepath and self.prefs.export_dir and not os.path.isabs(output_filepath):
            final_output_path = os.path.join(self.prefs.export_dir, output_filepath)
        else:
            final_output_path = output_filepath
        if not final_output_path:
            QMessageBox.critical(self, "Output Error", "Output filename is not specified in global settings. Please set it and try again.")
            return
        if not hasattr(sound_creator, 'generate_audio'):
            QMessageBox.critical(
                self,
                "Audio Engine Error",
                "The 'generate_audio' function is missing from 'synth_functions.sound_creator'. Cannot generate the final track."
            )
            return
        reply = QMessageBox.question(self, 'Confirm Generation', f"This will generate the audio file: {os.path.basename(output_filepath)}\nBased on the current track configuration.\n\nProceed?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.No: return
        try:
            self.generate_button.setEnabled(False)
            self.generate_progress_bar.setValue(0)
            self.generate_progress_bar.setVisible(True)
            self.statusBar().showMessage("Generating audio file, please wait...")
            QApplication.processEvents()

            def progress_cb(progress):
                self.generate_progress_bar.setValue(int(progress * 100))
                QApplication.processEvents()

            print(f"Initiating audio generation for: {final_output_path}")
            target_level = self.prefs.target_output_amplitude if getattr(self.prefs, "apply_target_amplitude", True) else 1.0
            success = sound_creator.generate_audio(
                current_track_data,
                output_filename=final_output_path,
                target_level=target_level,
                progress_callback=progress_cb,
            )
            if success:
                abs_path = os.path.abspath(final_output_path)
                QMessageBox.information(self, "Generation Complete", f"Audio file '{os.path.basename(final_output_path)}' generated successfully!\nFull path: {abs_path}")
            else:
                QMessageBox.critical(self, "Generation Failed", "Failed to generate audio file. Please check the console output for more details and error messages from the sound engine.")
        except Exception as e:
            QMessageBox.critical(self, "Audio Generation Error", f"An unexpected error occurred during the audio generation process:\n{str(e)}\n\nPlease check the console for a detailed traceback.")
            traceback.print_exc()
        finally:
            self.generate_button.setEnabled(True)
            self.generate_progress_bar.setVisible(False)
            self.statusBar().clearMessage()
            QApplication.processEvents()

    @pyqtSlot()
    def generate_selected_audio_action(self):
        if not self._update_global_settings_from_ui():
            return

        selection = sorted({idx.row() for idx in self.steps_tree.selectionModel().selectedRows()})
        if not selection:
            QMessageBox.warning(self, "No Steps Selected", "Please select one or more steps to generate.")
            return

        selected_track = copy.deepcopy(self.track_data)
        selected_track["steps"] = [copy.deepcopy(self.track_data["steps"][i]) for i in selection]

        # Re-base start times so the earliest selected step begins at 0.
        # This prevents leading silence when generating a subset of steps.
        first_start = float(
            selected_track["steps"][0].get(
                "start", selected_track["steps"][0].get("start_time", 0)
            )
        )
        for step in selected_track["steps"]:
            step_start = float(step.get("start", step.get("start_time", 0))) - first_start
            step["start"] = max(0.0, step_start)
            step.pop("start_time", None)

        output_filepath = selected_track["global_settings"].get("output_filename")
        if output_filepath and self.prefs.export_dir and not os.path.isabs(output_filepath):
            final_output_path = os.path.join(self.prefs.export_dir, output_filepath)
        else:
            final_output_path = output_filepath
        if not final_output_path:
            QMessageBox.critical(self, "Output Error", "Output filename is not specified in global settings. Please set it and try again.")
            return

        base, ext = os.path.splitext(final_output_path)
        final_output_path = f"{base}_selection{ext}"

        if not hasattr(sound_creator, 'generate_audio'):
            QMessageBox.critical(
                self,
                "Audio Engine Error",
                "The 'generate_audio' function is missing from 'synth_functions.sound_creator'. Cannot generate the final track.",
            )
            return

        reply = QMessageBox.question(
            self,
            'Confirm Generation',
            f"This will generate the audio file: {os.path.basename(final_output_path)}\nUsing only the selected steps.\n\nProceed?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.No:
            return

        try:
            self.generate_button.setEnabled(False)
            self.generate_selected_button.setEnabled(False)
            self.generate_progress_bar.setValue(0)
            self.generate_progress_bar.setVisible(True)
            self.statusBar().showMessage("Generating audio file, please wait...")
            QApplication.processEvents()

            def progress_cb(progress):
                self.generate_progress_bar.setValue(int(progress * 100))
                QApplication.processEvents()

            print(f"Initiating audio generation for selected steps: {final_output_path}")
            target_level = self.prefs.target_output_amplitude if getattr(self.prefs, "apply_target_amplitude", True) else 1.0
            success = sound_creator.generate_audio(
                selected_track,
                output_filename=final_output_path,
                target_level=target_level,
                progress_callback=progress_cb,
            )
            if success:
                abs_path = os.path.abspath(final_output_path)
                QMessageBox.information(
                    self,
                    "Generation Complete",
                    f"Audio file '{os.path.basename(final_output_path)}' generated successfully!\nFull path: {abs_path}",
                )
            else:
                QMessageBox.critical(
                    self,
                    "Generation Failed",
                    "Failed to generate audio file. Please check the console output for more details and error messages from the sound engine.",
                )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Audio Generation Error",
                f"An unexpected error occurred during the audio generation process:\n{str(e)}\n\nPlease check the console for a detailed traceback.",
            )
            traceback.print_exc()
        finally:
            self.generate_button.setEnabled(True)
            self.generate_selected_button.setEnabled(True)
            self.generate_progress_bar.setVisible(False)
            self.statusBar().clearMessage()
            QApplication.processEvents()

    # --- Test Step Preview Logic ---
    def _generate_test_step_audio(self, step_index):
        if not AUDIO_GENERATION_AVAILABLE or generate_single_step_audio_segment is None:
            QMessageBox.warning(self, "Audio Engine Error", "Audio generation function (generate_single_step_audio_segment) not available. Cannot generate test audio.")
            return False
        try:
            step_data = self.track_data["steps"][step_index]
            global_settings = self.track_data["global_settings"]
            sample_rate = global_settings["sample_rate"]
            
            # Determine preview duration based on step length
            step_duration = 0.0
            try:
                step_duration = float(step_data.get("duration", 0.0))
            except (TypeError, ValueError):
                step_duration = 0.0

            if step_duration > 0.0:
                test_duration = min(self.test_step_duration, step_duration)
            else:
                test_duration = self.test_step_duration

            # Generate audio (float32, stereo)
            audio_data_np_float32 = generate_single_step_audio_segment(
                step_data,
                global_settings,
                test_duration,
                test_duration,
            )
            
            if audio_data_np_float32 is None or audio_data_np_float32.size == 0:
                QMessageBox.critical(self, "Audio Generation Error", "Failed to generate test audio data (empty).")
                self.test_step_raw_audio = None; return False
            if not isinstance(audio_data_np_float32, np.ndarray) or audio_data_np_float32.dtype != np.float32:
                QMessageBox.critical(self, "Audio Format Error", "Generated audio is not float32 NumPy array.")
                self.test_step_raw_audio = None; return False
            
            # Ensure stereo
            if audio_data_np_float32.ndim == 1: # Mono to stereo
                audio_data_np_float32 = np.column_stack((audio_data_np_float32, audio_data_np_float32))
            elif audio_data_np_float32.ndim == 2 and audio_data_np_float32.shape[1] == 1: # Mono (column vector) to stereo
                audio_data_np_float32 = np.hstack((audio_data_np_float32, audio_data_np_float32))
            
            if not (audio_data_np_float32.ndim == 2 and audio_data_np_float32.shape[1] == 2):
                QMessageBox.critical(self, "Audio Format Error", f"Generated audio is not stereo (shape: {audio_data_np_float32.shape}).")
                self.test_step_raw_audio = None; return False
            
            # Convert to int16 bytes, applying target output amplitude if enabled
            amp_factor = self.prefs.target_output_amplitude if getattr(self.prefs, "apply_target_amplitude", True) else 1.0
            scaled = audio_data_np_float32 * amp_factor
            audio_data_scaled_int16 = (
                np.clip(scaled, -1.0, 1.0) * 32767
            ).astype(np.int16)
            self.test_step_raw_audio = audio_data_scaled_int16.tobytes()
            
            if not self.test_step_raw_audio:
                QMessageBox.critical(self, "Audio Generation Error", "Failed to convert audio to byte format.")
                return False

            # Setup QAudioFormat
            self.test_audio_format = QAudioFormat()
            self.test_audio_format.setCodec("audio/pcm")
            self.test_audio_format.setSampleRate(sample_rate)
            self.test_audio_format.setSampleSize(16) # For int16
            self.test_audio_format.setChannelCount(2) # Stereo
            self.test_audio_format.setByteOrder(QAudioFormat.LittleEndian) # Common
            self.test_audio_format.setSampleType(QAudioFormat.SignedInt) # For int16
            
            device_info = QAudioDeviceInfo.defaultOutputDevice()
            if not device_info.isFormatSupported(self.test_audio_format):
                QMessageBox.warning(self, "Audio Format Error", "Default audio output device does not support format for test playback.")
                self.test_audio_format = None; self.test_step_raw_audio = None; return False
            
            # Setup QAudioOutput
            if self.test_audio_output:
                self.test_audio_output.stop()
                try: self.test_audio_output.stateChanged.disconnect()
                except TypeError: pass 
                self.test_audio_output = None 
            self.test_audio_output = QAudioOutput(self.test_audio_format, self)
            self.test_audio_output.stateChanged.connect(self._handle_test_audio_state_change)
            
            # Setup QBuffer for audio data
            self.test_audio_io_device = QBuffer()
            self.test_audio_io_device.setData(self.test_step_raw_audio)
            self.test_audio_io_device.open(QIODevice.ReadOnly)
            
            # Calculate total duration for slider and label
            num_frames = len(self.test_step_raw_audio) // (self.test_audio_format.channelCount() * (self.test_audio_format.sampleSize() // 8))
            self.total_test_audio_duration_ms = (num_frames / sample_rate) * 1000

            self.test_step_time_slider.setRange(0, int(self.total_test_audio_duration_ms))
            self.test_step_time_slider.setValue(0)
            self._update_time_label(0, self.total_test_audio_duration_ms)
            return True
        except Exception as e:
            QMessageBox.critical(self, "Test Audio Error", f"Error generating/preparing test audio for step {step_index}: {e}")
            traceback.print_exc()
            self.test_step_raw_audio = None
            if self.test_audio_output: self.test_audio_output.stop()
            self.test_audio_output = None; self.test_audio_io_device = None
            return False

    @pyqtSlot()
    def on_reset_step_test(self):
        print("on_reset_step_test called")
        self.on_stop_step_test(clear_loaded_audio=True) 

        self.test_step_raw_audio = None
        self.current_test_step_index = -1
        self.total_test_audio_duration_ms = 0
        
        self.test_step_play_pause_button.setText("Play")
        self.test_step_time_slider.setRange(0, 1) 
        self.test_step_time_slider.setValue(0)
        self._update_time_label(0, 0)
        
        # After resetting, update label based on current tree selection
        current_selected_idx_tree = self.get_selected_step_index()
        if current_selected_idx_tree is not None and len(self.get_selected_step_indices()) == 1:
            try:
                step_data = self.track_data["steps"][current_selected_idx_tree]
                step_desc = step_data.get("description", "N/A")
                is_testable = len(step_data.get("voices", [])) > 0 and AUDIO_GENERATION_AVAILABLE
                short_desc = f"{step_desc[:30]}{'...' if len(step_desc) > 30 else ''}"
                if is_testable:
                    self.test_step_loaded_label.setText(f"Ready: {short_desc}")
                else:
                    self.test_step_loaded_label.setText(f"Info: {short_desc} (not testable)")
            except IndexError:
                self.test_step_loaded_label.setText("Error: Step not found.")
        else:
            self.test_step_loaded_label.setText("No step loaded for preview.")
        
        self._update_step_actions_state()


    @pyqtSlot(float)
    def on_test_duration_changed(self, value: float):
        """Update test preview duration ensuring it does not exceed step length."""
        step_idx = self.current_test_step_index
        if step_idx < 0:
            step_idx = self.get_selected_step_index()

        max_dur = float('inf')
        if step_idx is not None and 0 <= step_idx < len(self.track_data.get("steps", [])):
            try:
                step_dur = float(self.track_data["steps"][step_idx].get("duration", 0.0))
                if step_dur > 0.0:
                    max_dur = step_dur
            except (TypeError, ValueError):
                pass

        clamped = max(0.0, min(value, max_dur))
        if clamped != value:
            self.test_step_duration_spin.blockSignals(True)
            self.test_step_duration_spin.setValue(clamped)
            self.test_step_duration_spin.blockSignals(False)
        self.test_step_duration = clamped
        if hasattr(self, "prefs"):
            self.prefs.test_step_duration = clamped
            save_settings(self.prefs)


    @pyqtSlot()
    def on_play_pause_step_test(self):
        selected_step_idx_tree = self.get_selected_step_index() # Currently focused/selected in tree
        if selected_step_idx_tree is None:
            QMessageBox.warning(self, "Test Step", "Please select a step to test.")
            return

        # If a different step is selected in tree than what's loaded, or nothing is loaded, then load new.
        if self.current_test_step_index != selected_step_idx_tree or self.test_step_raw_audio is None:
            print(f"Play/Pause: New step ({selected_step_idx_tree}) or no audio. Current loaded: {self.current_test_step_index}")
            self.on_stop_step_test(clear_loaded_audio=True) # Fully reset before loading new
            
            step_desc_load = "Step"
            try:
                step_desc_load = self.track_data["steps"][selected_step_idx_tree].get("description", "N/A")
                short_desc_load = f"{step_desc_load[:25]}{'...' if len(step_desc_load) > 25 else ''}"
                self.test_step_loaded_label.setText(f"Loading: {short_desc_load}")
                QApplication.processEvents() 
            except IndexError:
                self.test_step_loaded_label.setText("Error: Step not found for loading.")
                self._update_step_actions_state(); return

            if not self._generate_test_step_audio(selected_step_idx_tree):
                self.test_step_loaded_label.setText(f"Failed to load: {short_desc_load}")
                self.current_test_step_index = -1 
                self.test_step_raw_audio = None
                self._update_step_actions_state(); return
            
            self.current_test_step_index = selected_step_idx_tree 
        
        # At this point, self.current_test_step_index IS selected_step_idx_tree and audio is loaded

        if not self.test_audio_output or not self.test_step_raw_audio:
            QMessageBox.critical(self, "Test Error", "Audio output not ready or no audio data.")
            self._update_step_actions_state(); return

        active_step_desc = "Step"
        if self.current_test_step_index != -1:
            try: active_step_desc = self.track_data["steps"][self.current_test_step_index].get("description", "N/A")
            except IndexError: pass
        short_active_desc = f"{active_step_desc[:25]}{'...' if len(active_step_desc) > 25 else ''}"

        if self.is_step_test_playing: # PAUSE action
            self.test_audio_output.suspend()
            self.test_audio_timer.stop()
            self.is_step_test_playing = False
            self.is_step_test_paused = True
            self.test_step_play_pause_button.setText("Play")
            self.test_step_loaded_label.setText(f"Paused: {short_active_desc}")
        else: # PLAY action 
            if self.test_audio_io_device.atEnd(): 
                self.test_audio_io_device.seek(0)
                self.test_step_time_slider.setValue(0)
            
            current_slider_pos_ms = self.test_step_time_slider.value()
            if self.test_audio_output.state() == QAudio.SuspendedState: 
                self.test_audio_output.resume()
            elif self.test_audio_output.state() == QAudio.StoppedState or self.test_audio_output.state() == QAudio.IdleState :
                if not self.test_audio_io_device.isOpen(): self.test_audio_io_device.open(QIODevice.ReadOnly)
                
                # Calculate byte position from slider milliseconds
                if self.test_audio_format and self.test_audio_format.sampleRate() > 0: # Check format is valid
                    sample_rate = self.test_audio_format.sampleRate()
                    channels = self.test_audio_format.channelCount()
                    bytes_per_sample_frame = channels * (self.test_audio_format.sampleSize() // 8)
                    if bytes_per_sample_frame > 0:
                        byte_position = int((current_slider_pos_ms / 1000.0) * sample_rate * bytes_per_sample_frame)
                        # Align to frame boundary
                        byte_position = (byte_position // bytes_per_sample_frame) * bytes_per_sample_frame
                        if byte_position >= self.test_audio_io_device.size():
                             byte_position = self.test_audio_io_device.size() - bytes_per_sample_frame if self.test_audio_io_device.size() > bytes_per_sample_frame else 0
                        if byte_position < 0: byte_position = 0
                        self.test_audio_io_device.seek(byte_position)
                    else:
                        self.test_audio_io_device.seek(0) # Fallback
                else:
                     self.test_audio_io_device.seek(0) # Fallback if format not ready

                self.test_audio_output.start(self.test_audio_io_device)
            
            self.test_audio_timer.start(100)
            self.is_step_test_playing = True
            self.is_step_test_paused = False
            self.test_step_play_pause_button.setText("Pause")
            self.test_step_loaded_label.setText(f"Playing: {short_active_desc}")
            
        self._update_step_actions_state()

    @pyqtSlot()
    def on_stop_step_test(self, clear_loaded_audio=False): 
        print(f"on_stop_step_test called, clear_loaded_audio={clear_loaded_audio}")
        if self.test_audio_output: self.test_audio_output.stop()
        # Do not seek here if just stopping, allow resume from current slider pos via play
        # if self.test_audio_io_device: self.test_audio_io_device.seek(0) 
        
        self.test_audio_timer.stop()
        self.is_step_test_playing = False
        self.is_step_test_paused = False
        self.test_step_play_pause_button.setText("Play")
        # Don't reset slider value on stop, only on full reset or end of playback.
        # self.test_step_time_slider.setValue(0) 

        if clear_loaded_audio:
            if self.test_audio_io_device: self.test_audio_io_device.seek(0) # Seek to 0 on full clear
            self.test_step_raw_audio = None
            self.current_test_step_index = -1
            self.total_test_audio_duration_ms = 0
            # Label will be set by on_reset_step_test or on_step_select after this
            self._update_time_label(0, 0)
            self.test_step_time_slider.setRange(0,1) 
            self.test_step_time_slider.setValue(0)
        elif self.test_step_raw_audio and self.current_test_step_index != -1:
            try:
                step_desc = self.track_data["steps"][self.current_test_step_index].get("description", "N/A")
                short_desc = f"{step_desc[:30]}{'...' if len(step_desc) > 30 else ''}"
                self.test_step_loaded_label.setText(f"Ready: {short_desc}")
                # Update time label to current slider position / total
                self._update_time_label(self.test_step_time_slider.value(), self.total_test_audio_duration_ms)
            except IndexError:
                self.test_step_loaded_label.setText("Error: Step data missing.")
                self._update_time_label(0, 0)
        else:
            self._update_time_label(0, self.total_test_audio_duration_ms if self.test_step_raw_audio else 0)
            if not clear_loaded_audio : # If not a full clear, but no audio was loaded
                 self.test_step_loaded_label.setText("No step loaded for preview.")


        self._update_step_actions_state()


    @pyqtSlot(int)
    def on_test_slider_moved(self, position_ms):
        if not self.test_step_raw_audio: return
        self._update_time_label(position_ms, self.total_test_audio_duration_ms)
        # No need to suspend here, on_test_slider_released will handle seek and resume if playing

    @pyqtSlot()
    def on_test_slider_released(self):
        if not self.test_audio_output or not self.test_audio_io_device or not self.test_step_raw_audio or not self.test_audio_format: return
        
        position_ms = self.test_step_time_slider.value()
        sample_rate = self.test_audio_format.sampleRate()
        channels = self.test_audio_format.channelCount()
        bytes_per_sample_frame = channels * (self.test_audio_format.sampleSize() // 8)

        if bytes_per_sample_frame == 0: return # Avoid division by zero

        byte_position = int((position_ms / 1000.0) * sample_rate * bytes_per_sample_frame)
        byte_position = (byte_position // bytes_per_sample_frame) * bytes_per_sample_frame # Align to frame
        
        if byte_position >= self.test_audio_io_device.size():
            byte_position = self.test_audio_io_device.size() - bytes_per_sample_frame
            if byte_position < 0: byte_position = 0
        
        self.test_audio_io_device.seek(byte_position)
        
        if self.is_step_test_playing: # If it was playing, stop current, then restart from new position
            self.test_audio_output.stop() # Stop to ensure buffer is flushed before restart
            self.test_audio_output.start(self.test_audio_io_device) # Restart from new position
            if not self.test_audio_timer.isActive(): self.test_audio_timer.start(100)
        elif self.is_step_test_paused: # If paused, just update position. Play will pick it up.
             pass # IO device position is set, resume will use it.
        
        self._update_time_label(position_ms, self.total_test_audio_duration_ms)


    def _update_test_audio_progress(self):
        if self.is_step_test_playing and self.test_audio_output and self.test_audio_format:
            if self.test_audio_output.state() == QAudio.ActiveState:
                # elapsed_ms based on bytes processed is more reliable than processedUSecs with seeks
                bytes_processed = self.test_audio_io_device.pos() # Current read position in bytes
                sample_rate = self.test_audio_format.sampleRate()
                bytes_per_frame = self.test_audio_format.channelCount() * (self.test_audio_format.sampleSize() // 8)
                
                if bytes_per_frame > 0 and sample_rate > 0:
                    current_ms = (bytes_processed / bytes_per_frame / sample_rate) * 1000
                    if not self.test_step_time_slider.isSliderDown():
                        self.test_step_time_slider.setValue(int(current_ms))
                    self._update_time_label(int(current_ms), self.total_test_audio_duration_ms)
        else: 
            self.test_audio_timer.stop()


    def _update_time_label(self, current_ms, total_ms): 
        current_s_total = current_ms // 1000
        total_s_total = total_ms // 1000
        current_minutes, current_seconds = divmod(current_s_total, 60)
        total_minutes, total_seconds = divmod(total_s_total, 60)
        # Display as SS.MS / SS.MS, to one decimal place (seconds.milliseconds)
        current_seconds_total = current_ms / 1000.0
        total_seconds_total = total_ms / 1000.0
        self.test_step_time_label.setText(f"{current_seconds_total:05.1f} / {total_seconds_total:05.1f}")


    def _handle_test_audio_state_change(self, state):
        if state == QAudio.IdleState: # Playback finished or stopped and buffer empty
            if self.is_step_test_playing and self.test_audio_io_device and self.test_audio_io_device.atEnd():
                # Audio finished playing naturally
                self.is_step_test_playing = False 
                self.is_step_test_paused = False
                self.test_step_play_pause_button.setText("Play")
                self.test_audio_timer.stop()
                
                self.test_step_time_slider.setValue(self.test_step_time_slider.maximum()) # Move slider to end
                self._update_time_label(self.total_test_audio_duration_ms, self.total_test_audio_duration_ms)
                
                if self.current_test_step_index != -1 and self.test_step_raw_audio:
                    try:
                        step_desc = self.track_data["steps"][self.current_test_step_index].get("description", "N/A")
                        short_desc = f"{step_desc[:30]}{'...' if len(step_desc) > 30 else ''}"
                        self.test_step_loaded_label.setText(f"Finished: {short_desc}")
                    except IndexError:
                        self.test_step_loaded_label.setText("Playback finished.")
                self._update_step_actions_state() 
        elif state == QAudio.StoppedState:
            if self.test_audio_output and self.test_audio_output.error() != QAudio.NoError:
                QMessageBox.warning(self, "Audio Playback Error", f"Audio error: {self.test_audio_output.error()}")
                self.on_reset_step_test() # Full reset on error
        self._update_step_actions_state() # Update button states on any state change

    # --- History Management ---
    def _push_history_state(self):
        # Trim forward history if undo was used
        if self.history_index < len(self.history) - 1:
            self.history = self.history[: self.history_index + 1]
        self.history.append(copy.deepcopy(self.track_data))
        self.history_index += 1
        self._update_undo_redo_actions_state()

    def undo(self):
        if self.history_index > 0:
            self.history_index -= 1
            self.track_data = copy.deepcopy(self.history[self.history_index])
            self.refresh_steps_tree()
            self._update_undo_redo_actions_state()

    def redo(self):
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.track_data = copy.deepcopy(self.history[self.history_index])
            self.refresh_steps_tree()
            self._update_undo_redo_actions_state()

    def _update_undo_redo_actions_state(self):
        if hasattr(self, 'undo_act'):
            self.undo_act.setEnabled(self.history_index > 0)
        if hasattr(self, 'redo_act'):
            self.redo_act.setEnabled(self.history_index < len(self.history) - 1)

    # --- Utility Methods ---
    def get_selected_step_index(self):
        indexes = self.steps_tree.selectionModel().selectedRows()
        if indexes:
            return indexes[0].row()
        return None

    def get_selected_step_indices(self):
        indexes = self.steps_tree.selectionModel().selectedRows()
        return sorted({idx.row() for idx in indexes})

    def get_selected_voice_index(self):
        indexes = self.voices_tree.selectionModel().selectedRows()
        if indexes:
            return indexes[0].row()
        return None

    def get_selected_voice_indices(self):
        indexes = self.voices_tree.selectionModel().selectedRows()
        return sorted({idx.row() for idx in indexes})

    def get_selected_clip_index(self):
        current_item = self.clips_tree.currentItem()
        if current_item:
            return current_item.data(0, Qt.UserRole)
        return None

    def get_selected_clip_indices(self):
        selected_items = self.clips_tree.selectedItems()
        indices = []
        for item in selected_items:
            data = item.data(0, Qt.UserRole)
            if data is not None:
                indices.append(int(data))
        return sorted(indices)

    def closeEvent(self, event):
        if self.test_audio_output:
            self.test_audio_output.stop()
        if self.is_clip_playing:
            self._stop_clip_playback()
        super().closeEvent(event)

# --- Run the Application ---
if __name__ == "__main__":

    app = QApplication(sys.argv)
    prefs = load_settings()
    if prefs.font_family or prefs.font_size:
        font = QFont(prefs.font_family or app.font().family(), prefs.font_size)
        app.setFont(font)
    apply_theme(app, prefs.theme)
    app.setStyle("Fusion")

    window = TrackEditorApp(prefs)
    window.show()
    sys.exit(app.exec_())
