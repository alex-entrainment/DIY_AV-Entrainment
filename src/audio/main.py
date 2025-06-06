import sys
from collections import OrderedDict
import json
from synth_functions import sound_creator  # Updated import path
import os
import copy # For deep copying voice data
import math # For default values like pi
import traceback # For error reporting

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
    QAbstractItemView,
    QAction,
)
from PyQt5.QtCore import Qt, pyqtSlot, QTimer, QBuffer, QIODevice
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QFont, QPalette, QColor
from PyQt5.QtMultimedia import (
    QAudioFormat,
    QAudioOutput,
    QAudioDeviceInfo,
    QAudio,
)

from functools import partial
from ui import themes
from preferences import load_preferences, save_preferences, Preferences
from ui.preferences_dialog import PreferencesDialog

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
    color: #dddddd;
}
QLineEdit, QComboBox, QSlider {
    background-color: #202020;
    border: 1px solid #555555;
    color: #dddddd;
}
"""

# --- Constants ---
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_CROSSFADE = 1.0 # Ensure float for consistency
TEST_AUDIO_DURATION_S = 30 # Duration for step test preview in seconds
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
        self.prefs: Preferences = prefs or load_preferences()
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

        self._setup_ui()
        self.setStyleSheet(GLOBAL_STYLE_SHEET)
        self._update_ui_from_global_settings()
        self.refresh_steps_tree()
        self._update_step_actions_state()
        self._update_voice_actions_state()
        self.statusBar()
        self._create_menu()

        # Flag to prevent handling itemChanged signals while refreshing
        self._voices_tree_updating = False
        self._steps_tree_updating = False

    def _get_default_track_data(self):
        return {
            "global_settings": {
                "sample_rate": DEFAULT_SAMPLE_RATE,
                "crossfade_duration": DEFAULT_CROSSFADE,
                "output_filename": "my_track.flac"
            },
            "steps": []
        }

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

        file_menu.addSeparator()
        theme_menu = file_menu.addMenu("Theme")
        for name in themes.THEMES.keys():
            act = QAction(name, self)
            act.triggered.connect(partial(self.set_theme, name))
            theme_menu.addAction(act)

    def set_theme(self, name):
        themes.apply_theme(QApplication.instance(), name)
        if hasattr(self, "prefs"):
            self.prefs.theme = name
            save_preferences(self.prefs)

    def open_preferences(self):
        dialog = PreferencesDialog(self.prefs, self)
        if dialog.exec_() == QDialog.Accepted:
            self.prefs = dialog.get_preferences()
            save_preferences(self.prefs)
            self.apply_preferences()

    def apply_preferences(self):
        app = QApplication.instance()
        if self.prefs.font_family or self.prefs.font_size:
            font = QFont(self.prefs.font_family or app.font().family(), self.prefs.font_size)
            app.setFont(font)
        themes.apply_theme(app, self.prefs.theme)

    def _setup_ui(self):
        # Central Widget and Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Control Frame ---
        control_groupbox = QGroupBox("Controls")
        control_layout = QHBoxLayout()
        control_groupbox.setLayout(control_layout)
        main_layout.addWidget(control_groupbox)

        # File Operations
        file_ops_groupbox = QGroupBox("File")
        file_ops_layout = QVBoxLayout()
        file_ops_groupbox.setLayout(file_ops_layout)
        self.new_file_button = QPushButton("New File")
        self.new_file_button.clicked.connect(self.new_file)
        file_ops_layout.addWidget(self.new_file_button)
        self.load_button = QPushButton("Load JSON")
        self.save_button = QPushButton("Save JSON")
        self.save_as_button = QPushButton("Save JSON As...")
        self.load_button.clicked.connect(self.load_json)
        self.save_button.clicked.connect(self.save_json)
        self.save_as_button.clicked.connect(self.save_json_as)
        file_ops_layout.addWidget(self.load_button)
        file_ops_layout.addWidget(self.save_button)
        file_ops_layout.addWidget(self.save_as_button)
        file_ops_layout.addStretch(1)
        control_layout.addWidget(file_ops_groupbox)

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
        generate_layout.setContentsMargins(0,0,0,0)
        control_layout.addWidget(generate_frame)

        # --- Main Paned Window (Splitter) ---
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter, 1)

        # --- Steps Frame Widgets ---
        steps_outer_widget = QWidget()
        steps_outer_layout = QVBoxLayout(steps_outer_widget)
        steps_outer_layout.setContentsMargins(0,0,0,0)
        main_splitter.addWidget(steps_outer_widget)
        steps_groupbox = QGroupBox("Steps")
        steps_groupbox_layout = QVBoxLayout(steps_groupbox)
        steps_outer_layout.addWidget(steps_groupbox)
        self.steps_tree = QTreeWidget()
        self.steps_tree.setColumnCount(3)
        self.steps_tree.setHeaderLabels(["Duration (s)", "Description", "# Voices"])
        self.steps_tree.setColumnWidth(0, 80)
        self.steps_tree.setColumnWidth(1, 150)
        self.steps_tree.setColumnWidth(2, 60)
        self.steps_tree.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self.steps_tree.setSelectionMode(QTreeWidget.ExtendedSelection)
        self.steps_tree.itemSelectionChanged.connect(self.on_step_select)
        self.steps_tree.itemChanged.connect(self.on_step_item_changed)
        self.steps_tree.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
        )
        steps_groupbox_layout.addWidget(self.steps_tree, 1)

        steps_button_layout_1 = QHBoxLayout()
        self.add_step_button = QPushButton("Add Step")
        self.load_external_step_button = QPushButton("Load External Step")
        self.duplicate_step_button = QPushButton("Duplicate Step")
        self.remove_step_button = QPushButton("Remove Step(s)")
        self.add_step_button.clicked.connect(self.add_step)
        self.load_external_step_button.clicked.connect(self.load_external_step)
        self.duplicate_step_button.clicked.connect(self.duplicate_step)
        self.remove_step_button.clicked.connect(self.remove_step)
        steps_button_layout_1.addWidget(self.add_step_button)
        steps_button_layout_1.addWidget(self.load_external_step_button)
        steps_button_layout_1.addWidget(self.duplicate_step_button)
        steps_button_layout_1.addWidget(self.remove_step_button)
        steps_button_layout_1.addStretch(1)
        steps_groupbox_layout.addLayout(steps_button_layout_1)

        steps_button_layout_2 = QHBoxLayout()
        self.edit_duration_button = QPushButton("Edit Duration")
        self.edit_description_button = QPushButton("Edit Description")
        self.move_step_up_button = QPushButton("Move Up")
        self.move_step_down_button = QPushButton("Move Down")
        self.edit_duration_button.clicked.connect(self.edit_step_duration)
        self.edit_description_button.clicked.connect(self.edit_step_description)
        self.move_step_up_button.clicked.connect(lambda: self.move_step(-1))
        self.move_step_down_button.clicked.connect(lambda: self.move_step(1))
        steps_button_layout_2.addWidget(self.edit_duration_button)
        steps_button_layout_2.addWidget(self.edit_description_button)
        steps_button_layout_2.addWidget(self.move_step_up_button)
        steps_button_layout_2.addWidget(self.move_step_down_button)
        steps_button_layout_2.addStretch(1)
        steps_groupbox_layout.addLayout(steps_button_layout_2)

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
        right_splitter = QSplitter(Qt.Vertical)
        main_splitter.addWidget(right_splitter)

        # --- Voices Frame Widgets ---
        voices_outer_widget = QWidget()
        voices_outer_layout = QVBoxLayout(voices_outer_widget)
        voices_outer_layout.setContentsMargins(0,0,0,0)
        right_splitter.addWidget(voices_outer_widget)
        self.voices_groupbox = QGroupBox("Voices for Selected Step")
        voices_groupbox_layout = QVBoxLayout(self.voices_groupbox)
        voices_outer_layout.addWidget(self.voices_groupbox)
        self.voices_tree = QTreeWidget()
        self.voices_tree.setColumnCount(4)
        self.voices_tree.setHeaderLabels([
            "Synth Function",
            "Carrier Freq",
            "Transition?",
            "Description",
        ])
        self.voices_tree.setColumnWidth(0, 220)
        self.voices_tree.setColumnWidth(1, 100)
        self.voices_tree.setColumnWidth(2, 80)
        self.voices_tree.setColumnWidth(3, 150)
        self.voices_tree.header().setStretchLastSection(True)
        self.voices_tree.setSelectionMode(QTreeWidget.ExtendedSelection)
        self.voices_tree.itemSelectionChanged.connect(self.on_voice_select)
        self.voices_tree.itemChanged.connect(self.on_voice_item_changed)
        self.voices_tree.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
        )
        voices_groupbox_layout.addWidget(self.voices_tree, 1)

        voices_button_layout_1 = QHBoxLayout()
        self.add_voice_button = QPushButton("Add Voice")
        self.edit_voice_button = QPushButton("Edit Voice")
        self.remove_voice_button = QPushButton("Remove Voice(s)")
        self.add_voice_button.clicked.connect(self.add_voice)
        self.edit_voice_button.clicked.connect(self.edit_voice)
        self.remove_voice_button.clicked.connect(self.remove_voice)
        voices_button_layout_1.addWidget(self.add_voice_button)
        voices_button_layout_1.addWidget(self.edit_voice_button)
        voices_button_layout_1.addWidget(self.remove_voice_button)
        voices_button_layout_1.addStretch(1)
        voices_groupbox_layout.addLayout(voices_button_layout_1)

        voices_button_layout_2 = QHBoxLayout()
        self.move_voice_up_button = QPushButton("Move Up")
        self.move_voice_down_button = QPushButton("Move Down")
        self.move_voice_up_button.clicked.connect(lambda: self.move_voice(-1))
        self.move_voice_down_button.clicked.connect(lambda: self.move_voice(1))
        voices_button_layout_2.addWidget(self.move_voice_up_button)
        voices_button_layout_2.addWidget(self.move_voice_down_button)
        voices_button_layout_2.addStretch(1)
        voices_groupbox_layout.addLayout(voices_button_layout_2)

        # --- Voice Details Frame Widgets ---
        voice_details_outer_widget = QWidget()
        voice_details_outer_layout = QVBoxLayout(voice_details_outer_widget)
        voice_details_outer_layout.setContentsMargins(0,0,0,0)
        right_splitter.addWidget(voice_details_outer_widget)
        self.voice_details_groupbox = QGroupBox("Selected Voice Details")
        voice_details_groupbox_layout = QVBoxLayout(self.voice_details_groupbox)
        voice_details_outer_layout.addWidget(self.voice_details_groupbox)
        self.voice_details_text = QTextEdit()
        self.voice_details_text.setReadOnly(True)
        self.voice_details_text.setFont(QFont("Consolas", 9))
        self.voice_details_text.setLineWrapMode(QTextEdit.WidgetWidth)
        voice_details_groupbox_layout.addWidget(self.voice_details_text)

        main_splitter.setSizes([400, 700])
        right_splitter.setSizes([500, 200])

    # --- Button State Management ---
    def _update_step_actions_state(self):
        selected_step_items = self.steps_tree.selectedItems()
        num_selected = len(selected_step_items)
        current_item = self.steps_tree.currentItem()
        current_idx = self.get_selected_step_index() # This is the *focused* item

        is_single_selection = (num_selected == 1) and (current_item is not None)
        has_selection = (num_selected > 0)
        num_steps = len(self.track_data["steps"])

        self.add_step_button.setEnabled(True)
        self.load_external_step_button.setEnabled(True)
        self.remove_step_button.setEnabled(has_selection)
        self.duplicate_step_button.setEnabled(is_single_selection)
        self.edit_duration_button.setEnabled(is_single_selection)
        self.edit_description_button.setEnabled(is_single_selection)
        # Enable Add Voice if a single step is selected. Availability of the
        # actual editor will be checked when the button is pressed so that the
        # user can receive an explanatory message if the dialog failed to load.
        self.add_voice_button.setEnabled(is_single_selection)

        can_move_up = is_single_selection and current_idx is not None and current_idx > 0
        can_move_down = is_single_selection and current_idx is not None and current_idx < (num_steps - 1)
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
                 self.voices_tree.clear()
                 self.clear_voice_details()
                 self.voices_groupbox.setTitle("Voices for Selected Step")
                 # If tester was idle and selection is lost, ensure label is "No step loaded"
                 if self.current_test_step_index == -1 and self.test_step_raw_audio is None:
                     self.test_step_loaded_label.setText("No step loaded for preview.")


    def _update_voice_actions_state(self):
        selected_voice_items = self.voices_tree.selectedItems()
        num_selected_voices = len(selected_voice_items)
        current_voice_idx = self.get_selected_voice_index()

        is_single_step_selected = len(self.steps_tree.selectedItems()) == 1 and self.get_selected_step_index() is not None

        # Always allow the button click when a single step is selected. If the
        # editor dialog failed to import, the handler will inform the user when
        # they attempt to add a voice.
        self.add_voice_button.setEnabled(is_single_step_selected)

        if not is_single_step_selected:
            self.edit_voice_button.setEnabled(False)
            self.remove_voice_button.setEnabled(False)
            self.move_voice_up_button.setEnabled(False)
            self.move_voice_down_button.setEnabled(False)
            return

        has_voice_selection = num_selected_voices > 0
        is_single_voice_selection = num_selected_voices == 1

        # Enable editing whenever a single voice is selected. The click handler
        # will report an error if the dialog cannot be loaded.
        self.edit_voice_button.setEnabled(is_single_voice_selection)
        self.remove_voice_button.setEnabled(has_voice_selection)

        num_voices_in_current_step = 0
        current_step_idx = self.get_selected_step_index()
        if current_step_idx is not None and 0 <= current_step_idx < len(self.track_data["steps"]):
            num_voices_in_current_step = len(self.track_data["steps"][current_step_idx].get("voices", []))

        can_move_voice_up = is_single_voice_selection and current_voice_idx is not None and current_voice_idx > 0
        can_move_voice_down = is_single_voice_selection and current_voice_idx is not None and current_voice_idx < (num_voices_in_current_step - 1)
        self.move_voice_up_button.setEnabled(can_move_voice_up)
        self.move_voice_down_button.setEnabled(can_move_voice_down)

    # --- Internal Data Handling ---
    def _update_global_settings_from_ui(self):
        try:
            sr_str = self.sr_entry.text()
            cf_str = self.cf_entry.text()
            outfile = self.outfile_entry.text().strip()
            if not sr_str: raise ValueError("Sample rate cannot be empty.")
            sr = int(sr_str)
            if sr <= 0: raise ValueError("Sample rate must be positive.")
            self.track_data["global_settings"]["sample_rate"] = sr
            if not cf_str: raise ValueError("Crossfade duration cannot be empty.")
            cf_str_safe = cf_str.replace(',', '.')
            cf = float(cf_str_safe)
            if cf < 0: raise ValueError("Crossfade duration cannot be negative.")
            self.track_data["global_settings"]["crossfade_duration"] = cf
            if not outfile: raise ValueError("Output filename cannot be empty.")
            if any(c in outfile for c in '<>:"/\\|?*'):
                raise ValueError("Output filename contains invalid characters.")
            self.track_data["global_settings"]["output_filename"] = outfile
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

    # --- UI Refresh Functions ---
    def refresh_steps_tree(self):
        self._steps_tree_updating = True
        current_focused_item_data = None
        current_item = self.steps_tree.currentItem()
        if current_item:
            current_focused_item_data = current_item.data(0, Qt.UserRole)
        selected_items_data = set()
        for item in self.steps_tree.selectedItems():
            data = item.data(0, Qt.UserRole)
            if data is not None: selected_items_data.add(data)
        self.steps_tree.clear()
        for i, step in enumerate(self.track_data.get("steps", [])):
            duration = step.get("duration", 0.0)
            description = step.get("description", "")
            num_voices = len(step.get("voices", []))
            item = QTreeWidgetItem(self.steps_tree)
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            item.setText(0, f"{duration:.2f}")
            item.setText(1, description)
            item.setText(2, str(num_voices))
            item.setData(0, Qt.UserRole, i)
        new_focused_item = None
        for i in range(self.steps_tree.topLevelItemCount()):
            item = self.steps_tree.topLevelItem(i)
            item_data = item.data(0, Qt.UserRole)
            if item_data in selected_items_data:
                item.setSelected(True)
                if item_data == current_focused_item_data: new_focused_item = item
        if new_focused_item:
            self.steps_tree.setCurrentItem(new_focused_item)
            self.steps_tree.scrollToItem(new_focused_item, QTreeWidget.PositionAtCenter)
        elif self.steps_tree.topLevelItemCount() > 0 and selected_items_data:
            first_selected_restored = None
            for i in range(self.steps_tree.topLevelItemCount()):
                item = self.steps_tree.topLevelItem(i)
                if item.isSelected(): first_selected_restored = item; break
            if first_selected_restored: self.steps_tree.setCurrentItem(first_selected_restored)
        self.on_step_select() # This will also trigger _update_step_actions_state
        self._steps_tree_updating = False

    def refresh_voices_tree(self):
        self._voices_tree_updating = True
        current_focused_voice_item_data = None
        current_voice_item = self.voices_tree.currentItem()
        if current_voice_item:
            current_focused_voice_item_data = current_voice_item.data(0, Qt.UserRole)
        selected_voice_items_data = set()
        for item in self.voices_tree.selectedItems():
            data = item.data(0, Qt.UserRole)
            if data is not None: selected_voice_items_data.add(data)
        self.voices_tree.clear()
        self.clear_voice_details()
        selected_step_idx = self.get_selected_step_index()
        if selected_step_idx is None or len(self.steps_tree.selectedItems()) != 1:
            self.voices_groupbox.setTitle("Voices for Selected Step")
            self._update_voice_actions_state()
            return
        self.voices_groupbox.setTitle(f"Voices for Step {selected_step_idx + 1}")
        try:
            step_data = self.track_data["steps"][selected_step_idx]
            voices = step_data.get("voices", [])
            for i, voice in enumerate(voices):
                func_name = voice.get("synth_function_name", "N/A")
                params = voice.get("params", {})
                is_transition = voice.get("is_transition", False)
                description = voice.get("description", "")
                carrier_freq_str = 'N/A'
                if 'baseFreq' in params: carrier_freq = params['baseFreq']
                elif 'frequency' in params: carrier_freq = params['frequency']
                elif 'carrierFreq' in params: carrier_freq = params['carrierFreq']
                else:
                    freq_keys = [k for k in params if ('Freq' in k or 'Frequency' in k) and not k.startswith(('start','end','target'))]
                    if is_transition:
                        freq_keys = [k for k in params if k.startswith('start') and ('Freq' in k or 'Frequency' in k)] or freq_keys
                    carrier_freq = params.get(freq_keys[0]) if freq_keys else 'N/A'
                try:
                    if carrier_freq is not None and carrier_freq != 'N/A': carrier_freq_str = f"{float(carrier_freq):.2f}"
                    else: carrier_freq_str = str(carrier_freq)
                except (ValueError, TypeError): carrier_freq_str = str(carrier_freq)
                transition_str = "Yes" if is_transition else "No"
                item = QTreeWidgetItem(self.voices_tree)
                item.setText(0, func_name)
                item.setText(1, carrier_freq_str)
                item.setText(2, transition_str)
                item.setText(3, description)
                item.setFlags(item.flags() | Qt.ItemIsEditable)
                item.setData(0, Qt.UserRole, i)
            new_focused_voice_item = None
            for i in range(self.voices_tree.topLevelItemCount()):
                item = self.voices_tree.topLevelItem(i)
                item_data = item.data(0, Qt.UserRole)
                if item_data in selected_voice_items_data:
                    item.setSelected(True)
                    if item_data == current_focused_voice_item_data: new_focused_voice_item = item
            if new_focused_voice_item:
                self.voices_tree.setCurrentItem(new_focused_voice_item)
                self.voices_tree.scrollToItem(new_focused_voice_item, QTreeWidget.PositionAtCenter)
            elif self.voices_tree.topLevelItemCount() > 0 and selected_voice_items_data:
                first_selected_restored_voice = None
                for i in range(self.voices_tree.topLevelItemCount()):
                    item = self.voices_tree.topLevelItem(i)
                    if item.isSelected(): first_selected_restored_voice = item; break
                if first_selected_restored_voice: self.voices_tree.setCurrentItem(first_selected_restored_voice)
        except IndexError:
            print(f"Error: Selected step index {selected_step_idx} out of range for voice display.")
            self.voices_groupbox.setTitle("Voices for Selected Step")
        except Exception as e:
            print(f"Error refreshing voices tree: {e}")
            traceback.print_exc()
        self.on_voice_select()
        self._voices_tree_updating = False

    def clear_voice_details(self):
        self.voice_details_text.clear()
        self.voice_details_groupbox.setTitle("Selected Voice Details")

    def update_voice_details(self):
        self.clear_voice_details()
        if len(self.steps_tree.selectedItems()) != 1 or len(self.voices_tree.selectedItems()) != 1: return
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
        selected_step_items = self.steps_tree.selectedItems()
        current_selected_idx = self.get_selected_step_index() 

        if len(selected_step_items) == 1 and current_selected_idx is not None:
            self.refresh_voices_tree() # Also calls on_voice_select -> _update_voice_actions_state
            # Update the test preview label if the tester is currently idle
            if not self.is_step_test_playing and not self.is_step_test_paused and self.test_step_raw_audio is None:
                try:
                    step_data = self.track_data["steps"][current_selected_idx]
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
            self.voices_tree.clear()
            self.clear_voice_details()
            self.voices_groupbox.setTitle("Voices for Selected Step")
            if not self.is_step_test_playing and not self.is_step_test_paused and self.test_step_raw_audio is None:
                self.test_step_loaded_label.setText("No step loaded for preview.")
            self._update_voice_actions_state() # Update voice buttons if step selection changes to non-single

        self._update_step_actions_state() # Update all step and test preview button states

    @pyqtSlot()
    def on_voice_select(self):
        self._update_voice_actions_state()
        self.update_voice_details()

    @pyqtSlot(QTreeWidgetItem, int)
    def on_step_item_changed(self, item, column):
        if self._steps_tree_updating:
            return
        step_idx = item.data(0, Qt.UserRole)
        if step_idx is None:
            return
        try:
            if column == 1:
                new_desc = item.text(1).strip()
                self.track_data["steps"][step_idx]["description"] = new_desc
            else:
                # Revert edits to non-editable columns
                self._steps_tree_updating = True
                step = self.track_data["steps"][step_idx]
                if column == 0:
                    item.setText(0, f"{step.get('duration', 0.0):.2f}")
                elif column == 2:
                    item.setText(2, str(len(step.get('voices', []))))
                self._steps_tree_updating = False
        except Exception as e:
            print(f"Error updating step item: {e}")

    @pyqtSlot(QTreeWidgetItem, int)
    def on_voice_item_changed(self, item, column):
        if self._voices_tree_updating or column != 3:
            return
        step_idx = self.get_selected_step_index()
        voice_idx = item.data(0, Qt.UserRole)
        if step_idx is None or voice_idx is None:
            return
        try:
            new_desc = item.text(3).strip()
            self.track_data["steps"][step_idx]["voices"][voice_idx][
                "description"
            ] = new_desc
        except Exception as e:
            print(f"Error updating voice description: {e}")

    # --- Action Methods (File Ops, Step/Voice Ops) ---
    @pyqtSlot()
    def new_file(self):
        if self.is_step_test_playing or self.is_step_test_paused or self.test_step_raw_audio:
            self.on_reset_step_test() # Reset tester before loading new file content
        self.track_data = self._get_default_track_data()
        self.current_json_path = None
        self._update_ui_from_global_settings()
        self.refresh_steps_tree() # This calls on_step_select -> _update_step_actions_state
        self.setWindowTitle("Binaural Track Editor (PyQt5) - New File")
        QMessageBox.information(self, "New File", "New track created.")

    @pyqtSlot()
    def load_json(self):
        if self.is_step_test_playing or self.is_step_test_paused or self.test_step_raw_audio:
            self.on_reset_step_test() # Reset tester before loading new file content
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Track JSON", "", "JSON files (*.json);;All files (*.*)")
        if not filepath: return
        try:
            loaded_data = sound_creator.load_track_from_json(filepath)
            if loaded_data and isinstance(loaded_data.get("steps"), list) and isinstance(loaded_data.get("global_settings"), dict):
                self.track_data = loaded_data
                self.current_json_path = filepath
                self.setWindowTitle(f"Binaural Track Editor (PyQt5) - {os.path.basename(filepath)}")
                self._update_ui_from_global_settings()
                self.refresh_steps_tree() # This calls on_step_select -> _update_step_actions_state
                QMessageBox.information(self, "Load Success", f"Track loaded from\n{filepath}")
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
    def load_external_step(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Load External Steps from JSON", "", "JSON files (*.json);;All files (*.*)")
        if not filepath: return
        try:
            with open(filepath, 'r') as f: external_data = json.load(f)
            if "steps" not in external_data or not isinstance(external_data["steps"], list):
                QMessageBox.critical(self, "Load Error", "Invalid JSON structure: 'steps' key missing or not a list.")
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
                    new_item = self.steps_tree.topLevelItem(current_step_count)
                    if new_item:
                        self.steps_tree.setCurrentItem(new_item)
                        self.steps_tree.scrollToItem(new_item, QTreeWidget.PositionAtCenter)
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
        if 0 <= insert_index < self.steps_tree.topLevelItemCount():
            new_item = self.steps_tree.topLevelItem(insert_index)
            self.steps_tree.clearSelection()
            self.steps_tree.setCurrentItem(new_item)
            new_item.setSelected(True)
            self.steps_tree.scrollToItem(new_item, QTreeWidget.PositionAtCenter)
        # refresh_steps_tree calls on_step_select which calls _update_step_actions_state

    @pyqtSlot()
    def duplicate_step(self):
        selected_index = self.get_selected_step_index()
        if selected_index is None or len(self.steps_tree.selectedItems()) != 1 :
            QMessageBox.warning(self, "Duplicate Step", "Please select exactly one step to duplicate.")
            return
        try:
            original_step_data = self.track_data["steps"][selected_index]
            duplicated_step_data = copy.deepcopy(original_step_data)
            insert_index = selected_index + 1
            self.track_data["steps"].insert(insert_index, duplicated_step_data)
            self.refresh_steps_tree()
            if 0 <= insert_index < self.steps_tree.topLevelItemCount():
                new_item = self.steps_tree.topLevelItem(insert_index)
                self.steps_tree.clearSelection()
                self.steps_tree.setCurrentItem(new_item)
                new_item.setSelected(True)
                self.steps_tree.scrollToItem(new_item, QTreeWidget.PositionAtCenter)
        except IndexError: QMessageBox.critical(self, "Error", "Failed to duplicate step (index out of range).")
        except Exception as e: QMessageBox.critical(self, "Error", f"Failed to duplicate step:\n{e}"); traceback.print_exc()
        # refresh_steps_tree calls on_step_select which calls _update_step_actions_state

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
                    if 0 <= index < len(self.track_data["steps"]): del self.track_data["steps"][index]
                self.refresh_steps_tree()
            except Exception as e: QMessageBox.critical(self, "Error", f"Failed to remove step(s):\n{e}"); traceback.print_exc()
        # refresh_steps_tree calls on_step_select which calls _update_step_actions_state

    @pyqtSlot()
    def edit_step_duration(self):
        selected_index = self.get_selected_step_index()
        if selected_index is None or len(self.steps_tree.selectedItems()) != 1:
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
            except IndexError: QMessageBox.critical(self, "Error", "Failed to set duration (index out of range after edit).")
            except Exception as e: QMessageBox.critical(self, "Error", f"Failed to set duration:\n{e}")

    @pyqtSlot()
    def edit_step_description(self):
        selected_index = self.get_selected_step_index()
        if selected_index is None or len(self.steps_tree.selectedItems()) != 1:
            QMessageBox.warning(self, "Edit Description", "Please select exactly one step to edit.")
            return
        try: current_description = str(self.track_data["steps"][selected_index].get("description", ""))
        except IndexError as e: QMessageBox.critical(self, "Error", f"Failed to get current description (index {selected_index}):\n{e}"); return
        new_description, ok = QInputDialog.getText(self, f"Edit Step {selected_index + 1} Description", "Description:", QLineEdit.Normal, current_description)
        if ok and new_description is not None:
            try:
                self.track_data["steps"][selected_index]["description"] = new_description.strip()
                self.refresh_steps_tree() # This will update the display
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
        selected_index = self.get_selected_step_index()
        if selected_index is None or len(self.steps_tree.selectedItems()) != 1:
            QMessageBox.warning(self, "Move Step", "Please select exactly one step to move.")
            return
        
        # If the moved step was the one loaded in tester, update current_test_step_index
        # or reset tester if it becomes complex to track. For simplicity, reset.
        if self.current_test_step_index == selected_index:
            self.on_reset_step_test() 
            # After reset, the new selection logic in on_reset_step_test will handle the label for the moved item if it's selected.

        num_steps = len(self.track_data["steps"])
        new_index = selected_index + direction
        if 0 <= new_index < num_steps:
            try:
                steps = self.track_data["steps"]
                steps[selected_index], steps[new_index] = steps[new_index], steps[selected_index]
                self.refresh_steps_tree()
                if new_index < self.steps_tree.topLevelItemCount():
                    moved_item = self.steps_tree.topLevelItem(new_index)
                    self.steps_tree.clearSelection()
                    self.steps_tree.setCurrentItem(moved_item)
                    moved_item.setSelected(True)
                    self.steps_tree.scrollToItem(moved_item, QTreeWidget.PositionAtCenter)
            except Exception as e: QMessageBox.critical(self, "Error", f"Failed to move step:\n{e}"); traceback.print_exc()

    @pyqtSlot()
    def add_voice(self):
        # If the dialog could not be loaded, the fallback implementation will
        # display an informative message when executed. Therefore we allow the
        # user to click the button even when VOICE_EDITOR_DIALOG_AVAILABLE is
        # False.
        selected_step_index = self.get_selected_step_index()
        if selected_step_index is None or len(self.steps_tree.selectedItems()) != 1:
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
            if selected_step_index < self.steps_tree.topLevelItemCount():
                step_item = self.steps_tree.topLevelItem(selected_step_index)
                self.steps_tree.setCurrentItem(step_item) # Keep focus on the step
                QTimer.singleShot(0, lambda: self._select_last_voice_in_current_step())

    def _select_last_voice_in_current_step(self): 
        voice_count = self.voices_tree.topLevelItemCount()
        if voice_count > 0:
            new_voice_item = self.voices_tree.topLevelItem(voice_count - 1)
            self.voices_tree.clearSelection()
            self.voices_tree.setCurrentItem(new_voice_item)
            new_voice_item.setSelected(True)
            self.voices_tree.scrollToItem(new_voice_item, QTreeWidget.PositionAtCenter)
        self._update_voice_actions_state()

    @pyqtSlot()
    def edit_voice(self):
        # If the editor dialog failed to load we still allow the action so the
        # fallback dialog can inform the user about the issue.
        selected_step_index = self.get_selected_step_index()
        selected_voice_index = self.get_selected_voice_index()
        if selected_step_index is None or selected_voice_index is None or \
           len(self.steps_tree.selectedItems()) != 1 or len(self.voices_tree.selectedItems()) != 1:
            QMessageBox.warning(self, "Edit Voice", "Please select exactly one step and one voice to edit.")
            return
        
        # If editing a voice in the currently loaded step, it might change its test audio.
        # For simplicity, reset the tester. The user can then replay.
        if self.current_test_step_index == selected_step_index and self.test_step_raw_audio:
            self.on_reset_step_test()

        dialog = VoiceEditorDialog(parent=self, app_ref=self, step_index=selected_step_index, voice_index=selected_voice_index)
        if dialog.exec_() == QDialog.Accepted:
            self.refresh_steps_tree() # This updates step voice count and calls on_step_select
            if selected_step_index < self.steps_tree.topLevelItemCount():
                step_item = self.steps_tree.topLevelItem(selected_step_index)
                self.steps_tree.setCurrentItem(step_item)
                if selected_voice_index < self.voices_tree.topLevelItemCount():
                    voice_item = self.voices_tree.topLevelItem(selected_voice_index)
                    self.voices_tree.setCurrentItem(voice_item)
                    self.voices_tree.scrollToItem(voice_item, QTreeWidget.PositionAtCenter)
            self._update_voice_actions_state()

    @pyqtSlot()
    def remove_voice(self):
        selected_step_idx = self.get_selected_step_index()
        selected_voice_indices = self.get_selected_voice_indices()
        if selected_step_idx is None or len(self.steps_tree.selectedItems()) != 1:
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
        except IndexError: QMessageBox.critical(self, "Error", "Failed to remove voice(s) (step index out of range).")
        except Exception as e: QMessageBox.critical(self, "Error", f"Failed to remove voice(s):\n{e}"); traceback.print_exc()

    @pyqtSlot(int)
    def move_voice(self, direction):
        selected_step_idx = self.get_selected_step_index()
        selected_voice_idx = self.get_selected_voice_index()
        if selected_step_idx is None or selected_voice_idx is None or \
           len(self.steps_tree.selectedItems()) != 1 or len(self.voices_tree.selectedItems()) != 1:
            QMessageBox.warning(self, "Move Voice", "Please select exactly one step and one voice to move.")
            return
        
        # Moving voices might change audio characteristics. Reset if it's the loaded step.
        if self.current_test_step_index == selected_step_idx and self.test_step_raw_audio:
            self.on_reset_step_test()

        try:
            voices_list = self.track_data["steps"][selected_step_idx]["voices"]
            num_voices = len(voices_list)
            new_voice_idx = selected_voice_idx + direction
            if 0 <= new_voice_idx < num_voices:
                voices_list[selected_voice_idx], voices_list[new_voice_idx] = voices_list[new_voice_idx], voices_list[selected_voice_idx]
                self.refresh_voices_tree() # Calls on_voice_select -> _update_voice_actions_state
                if new_voice_idx < self.voices_tree.topLevelItemCount():
                    moved_item = self.voices_tree.topLevelItem(new_voice_idx)
                    self.voices_tree.clearSelection()
                    self.voices_tree.setCurrentItem(moved_item)
                    moved_item.setSelected(True)
                    self.voices_tree.scrollToItem(moved_item, QTreeWidget.PositionAtCenter)
        except IndexError: QMessageBox.critical(self, "Error", "Failed to move voice (index out of range).")
        except Exception as e: QMessageBox.critical(self, "Error", f"An unexpected error occurred while moving voice:\n{e}")
        self._update_voice_actions_state()

    # --- generate_audio_action ---
    @pyqtSlot()
    def generate_audio_action(self):
        if not self._update_global_settings_from_ui(): return
        current_track_data = self.track_data
        output_filepath = current_track_data["global_settings"].get("output_filename")
        if not output_filepath:
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
            self.statusBar().showMessage("Generating audio file, please wait...")
            QApplication.processEvents()
            print(f"Initiating audio generation for: {output_filepath}")
            success = sound_creator.generate_audio(current_track_data, output_filename=output_filepath)
            if success:
                abs_path = os.path.abspath(output_filepath)
                QMessageBox.information(self, "Generation Complete", f"Audio file '{os.path.basename(output_filepath)}' generated successfully!\nFull path: {abs_path}")
            else:
                QMessageBox.critical(self, "Generation Failed", "Failed to generate audio file. Please check the console output for more details and error messages from the sound engine.")
        except Exception as e:
            QMessageBox.critical(self, "Audio Generation Error", f"An unexpected error occurred during the audio generation process:\n{str(e)}\n\nPlease check the console for a detailed traceback.")
            traceback.print_exc()
        finally:
            self.generate_button.setEnabled(True)
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
            
            # Generate audio (float32, stereo)
            audio_data_np_float32 = generate_single_step_audio_segment(step_data, global_settings, self.test_step_duration, self.test_step_duration)
            
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
            
            # Convert to int16 bytes
            audio_data_scaled_int16 = (np.clip(audio_data_np_float32, -1.0, 1.0) * 32767).astype(np.int16)
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
        if current_selected_idx_tree is not None and len(self.steps_tree.selectedItems()) == 1:
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

    # --- Utility Methods ---
    def get_selected_step_index(self):
        current_item = self.steps_tree.currentItem()
        if current_item: return current_item.data(0, Qt.UserRole)
        return None

    def get_selected_step_indices(self):
        selected_items = self.steps_tree.selectedItems()
        indices = []
        if selected_items:
            for item in selected_items:
                data = item.data(0, Qt.UserRole)
                if data is not None: indices.append(int(data))
        return sorted(indices)

    def get_selected_voice_index(self):
        current_item = self.voices_tree.currentItem()
        if current_item: return current_item.data(0, Qt.UserRole)
        return None

    def get_selected_voice_indices(self):
        selected_items = self.voices_tree.selectedItems()
        indices = []
        if selected_items:
            for item in selected_items:
                data = item.data(0, Qt.UserRole)
                if data is not None: indices.append(int(data))
        return sorted(indices)

    def closeEvent(self, event):
        if self.test_audio_output: self.test_audio_output.stop()
        super().closeEvent(event)

# --- Run the Application ---
if __name__ == "__main__":
    if not hasattr(sound_creator, 'SYNTH_FUNCTIONS'): 
        # temp_app_for_error_msg = QApplication.instance() 
        # if not temp_app_for_error_msg:
        #     temp_app_for_error_msg = QApplication(sys.argv)
        # mbox = QMessageBox()
        # mbox.setIcon(QMessageBox.Critical)
        # mbox.setText(
        #     "Critical Error: Sound creator module is missing vital components (SYNTH_FUNCTIONS).\n"
        #     "The application cannot start.\nPlease check the 'synth_functions/sound_creator.py' file."
        # )
        # mbox.setWindowTitle("Application Startup Error")
        # mbox.setStandardButtons(QMessageBox.Ok)
        # mbox.exec_()
        print(
            "Critical Error: sound_creator.SYNTH_FUNCTIONS not found. Ensure synth_functions/sound_creator.py is correct and accessible."
        )
        sys.exit(1)

    app = QApplication(sys.argv)
    prefs = load_preferences()
    if prefs.font_family or prefs.font_size:
        font = QFont(prefs.font_family or app.font().family(), prefs.font_size)
        app.setFont(font)
    themes.apply_theme(app, prefs.theme)

    window = TrackEditorApp(prefs)
    window.show()
    sys.exit(app.exec_())
