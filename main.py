import sys
import json
# Make sure sound_creator.py is accessible (in the same directory or Python path)
import sound_creator # Import the refactored sound generation script
import inspect
import os
import copy # For deep copying voice data
import math # For default values like pi
import traceback # For error reporting
import re # For parsing source code
import ast # For safely evaluating default values from source

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QLineEdit, QTreeWidget, QTreeWidgetItem, QTextEdit,
    QComboBox, QCheckBox, QGroupBox, QSplitter, QFileDialog, QMessageBox,
    QDialog, QScrollArea, QSizePolicy, QInputDialog, QSpacerItem, QHeaderView,
)
from PyQt5.QtCore import Qt, pyqtSlot, QSize, QTimer # Added QTimer
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QFont

# --- Constants ---
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_CROSSFADE = 1.0
MAX_VOICES_PER_STEP = 30
ENVELOPE_TYPE_NONE = "None"
ENVELOPE_TYPE_LINEAR = "linear_fade"
# Add more envelope types here if needed in the future
SUPPORTED_ENVELOPE_TYPES = [ENVELOPE_TYPE_NONE, ENVELOPE_TYPE_LINEAR]

# --- Main Application Class ---
class TrackEditorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Binaural Track Editor (PyQt5)")
        self.setMinimumSize(950, 600)
        self.resize(1200, 800) # Increased size slightly

        self.track_data = self._get_default_track_data()
        self.current_json_path = None

        # Validators (reusable)
        self.int_validator_positive = QIntValidator(1, 999999, self) # Sample Rate > 0
        self.double_validator_non_negative = QDoubleValidator(0.0, 999999.0, 6, self) # Crossfade >= 0, Duration > 0, Env Duration >=0
        self.double_validator_zero_to_one = QDoubleValidator(0.0, 1.0, 6, self) # Amplitude (0-1 range typical)
        self.double_validator = QDoubleValidator(-999999.0, 999999.0, 6, self) # General float params
        self.int_validator = QIntValidator(-999999, 999999, self) # General int params

        self._setup_ui()
        self._update_ui_from_global_settings()
        self.refresh_steps_tree()
        self._update_step_actions_state() # Initial button states
        self._update_voice_actions_state() # Initial button states

    def _get_default_track_data(self):
        """Returns the default structure for track data."""
        return {
            "global_settings": {
                "sample_rate": DEFAULT_SAMPLE_RATE,
                "crossfade_duration": DEFAULT_CROSSFADE,
                "output_filename": "my_track.wav"
            },
            "steps": []
        }
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
        globals_layout.setColumnStretch(1, 1) # Allow entry to expand
        control_layout.addWidget(globals_groupbox, 1) # Add stretch factor

        control_layout.addStretch(1)

        # Generate Button (aligned right)
        generate_frame = QWidget() # Use a widget for alignment
        generate_layout = QHBoxLayout(generate_frame)
        generate_layout.addStretch(1)
        self.generate_button = QPushButton("Generate WAV")
        self.generate_button.setStyleSheet("QPushButton { background-color: #0078D7; color: white; padding: 8px; font-weight: bold; border-radius: 3px; } QPushButton:hover { background-color: #005A9E; } QPushButton:pressed { background-color: #003C6A; } QPushButton:disabled { background-color: #AAAAAA; color: #666666; }")
        self.generate_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.generate_button.clicked.connect(self.generate_wav_action)
        generate_layout.addWidget(self.generate_button)
        generate_layout.setContentsMargins(0,0,0,0)
        control_layout.addWidget(generate_frame)


        # --- Main Paned Window (Splitter) ---
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter, 1) # Give more stretch factor

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
        self.steps_tree.setSelectionMode(QTreeWidget.ExtendedSelection) # CHANGED
        self.steps_tree.itemSelectionChanged.connect(self.on_step_select)
        # self.steps_tree.itemDoubleClicked.connect(self.edit_step_description) # Can be used if desired

        steps_groupbox_layout.addWidget(self.steps_tree, 1)

        steps_button_layout_1 = QHBoxLayout() # First row of step buttons
        self.add_step_button = QPushButton("Add Step")
        self.load_external_step_button = QPushButton("Load External Step") # NEW
        self.duplicate_step_button = QPushButton("Duplicate Step")
        self.remove_step_button = QPushButton("Remove Step(s)")

        self.add_step_button.clicked.connect(self.add_step)
        self.load_external_step_button.clicked.connect(self.load_external_step) # NEW
        self.duplicate_step_button.clicked.connect(self.duplicate_step)
        self.remove_step_button.clicked.connect(self.remove_step)

        steps_button_layout_1.addWidget(self.add_step_button)
        steps_button_layout_1.addWidget(self.load_external_step_button) # NEW
        steps_button_layout_1.addWidget(self.duplicate_step_button)
        steps_button_layout_1.addWidget(self.remove_step_button)
        steps_button_layout_1.addStretch(1)
        steps_groupbox_layout.addLayout(steps_button_layout_1)

        steps_button_layout_2 = QHBoxLayout() # Second row of step buttons
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
        self.voices_tree.setColumnCount(3)
        self.voices_tree.setHeaderLabels(["Synth Function", "Carrier Freq", "Transition?"])
        self.voices_tree.setColumnWidth(0, 220)
        self.voices_tree.setColumnWidth(1, 100)
        self.voices_tree.setColumnWidth(2, 80)
        self.voices_tree.header().setStretchLastSection(True)
        self.voices_tree.setSelectionMode(QTreeWidget.ExtendedSelection) # CHANGED
        self.voices_tree.itemSelectionChanged.connect(self.on_voice_select)
        # self.voices_tree.itemDoubleClicked.connect(self.edit_voice) # Can be used if desired
        voices_groupbox_layout.addWidget(self.voices_tree, 1)

        voices_button_layout_1 = QHBoxLayout() # First row of voice buttons
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

        voices_button_layout_2 = QHBoxLayout() # Second row of voice buttons
        self.move_voice_up_button = QPushButton("Move Up") # NEW
        self.move_voice_down_button = QPushButton("Move Down") # NEW

        self.move_voice_up_button.clicked.connect(lambda: self.move_voice(-1)) # NEW
        self.move_voice_down_button.clicked.connect(lambda: self.move_voice(1)) # NEW
        voices_button_layout_2.addWidget(self.move_voice_up_button) # NEW
        voices_button_layout_2.addWidget(self.move_voice_down_button) # NEW
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

        # Set Splitter Sizes (initial proportions)
        main_splitter.setSizes([400, 700]) # Adjusted
        right_splitter.setSizes([500, 200])

    # --- Button State Management ---
    def _update_step_actions_state(self):
        """Updates the enabled state of step-related buttons based on selection."""
        selected_step_items = self.steps_tree.selectedItems()
        num_selected = len(selected_step_items)
        current_item = self.steps_tree.currentItem() # The focused item
        current_idx = self.get_selected_step_index() # Index of focused item

        is_single_selection = (num_selected == 1)
        has_selection = (num_selected > 0)
        num_steps = len(self.track_data["steps"])

        # Add Step, Load External Step are always enabled (or based on other app logic)
        self.add_step_button.setEnabled(True)
        self.load_external_step_button.setEnabled(True)

        # Actions for one or more selected steps
        self.remove_step_button.setEnabled(has_selection)

        # Actions for exactly one selected step
        self.duplicate_step_button.setEnabled(is_single_selection)
        self.edit_duration_button.setEnabled(is_single_selection)
        self.edit_description_button.setEnabled(is_single_selection)
        self.add_voice_button.setEnabled(is_single_selection) # Add voice to the focused step

        can_move_up = is_single_selection and current_idx is not None and current_idx > 0
        can_move_down = is_single_selection and current_idx is not None and current_idx < (num_steps - 1)
        self.move_step_up_button.setEnabled(can_move_up)
        self.move_step_down_button.setEnabled(can_move_down)

        # If multiple steps are selected, the voice panel might be cleared or reflect the 'current' item.
        # The voice actions will be updated based on the voice selection within that 'current' step.
        if not is_single_selection:
             self.voices_tree.clear()
             self.clear_voice_details()
             self.voices_groupbox.setTitle("Voices for Selected Step")


    def _update_voice_actions_state(self):
        """Updates the enabled state of voice-related buttons."""
        selected_voice_items = self.voices_tree.selectedItems()
        num_selected_voices = len(selected_voice_items)
        current_voice_idx = self.get_selected_voice_index() # Index of focused voice

        # Check if there's a single valid step selected that voices could belong to
        is_single_step_selected = len(self.steps_tree.selectedItems()) == 1 and self.get_selected_step_index() is not None

        self.add_voice_button.setEnabled(is_single_step_selected) # Re-check, this is mainly step-dependent

        # Voice actions depend on both step selection and voice selection
        if not is_single_step_selected: # If no single step is focused, disable all voice-specific actions
            self.edit_voice_button.setEnabled(False)
            self.remove_voice_button.setEnabled(False)
            self.move_voice_up_button.setEnabled(False)
            self.move_voice_down_button.setEnabled(False)
            return

        # Now, specific voice actions based on voice selection
        has_voice_selection = num_selected_voices > 0
        is_single_voice_selection = num_selected_voices == 1

        self.edit_voice_button.setEnabled(is_single_voice_selection)
        self.remove_voice_button.setEnabled(has_voice_selection)

        num_voices_in_current_step = 0
        current_step_idx = self.get_selected_step_index() # Focused step
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
        current_focused_item_data = None
        current_item = self.steps_tree.currentItem()
        if current_item:
            current_focused_item_data = current_item.data(0, Qt.UserRole) # Store original index of focused item

        # Storing all selected items' original indices to try and restore selection
        selected_items_data = set()
        for item in self.steps_tree.selectedItems():
            data = item.data(0, Qt.UserRole)
            if data is not None:
                selected_items_data.add(data)

        self.steps_tree.clear()
        for i, step in enumerate(self.track_data.get("steps", [])):
            duration = step.get("duration", 0.0)
            description = step.get("description", "")
            num_voices = len(step.get("voices", []))
            item = QTreeWidgetItem(self.steps_tree)
            item.setText(0, f"{duration:.2f}")
            item.setText(1, description)
            item.setText(2, str(num_voices))
            item.setData(0, Qt.UserRole, i) # Store original index

        # Restore selection and focus
        new_focused_item = None
        for i in range(self.steps_tree.topLevelItemCount()):
            item = self.steps_tree.topLevelItem(i)
            item_data = item.data(0, Qt.UserRole)
            if item_data in selected_items_data:
                item.setSelected(True)
                if item_data == current_focused_item_data: # Restore focus
                    new_focused_item = item
        
        if new_focused_item:
            self.steps_tree.setCurrentItem(new_focused_item)
            self.steps_tree.scrollToItem(new_focused_item, QTreeWidget.PositionAtCenter)
        elif self.steps_tree.topLevelItemCount() > 0 and selected_items_data: # if focus was lost but there were selections
            # Try to set current item to the first one that was previously selected
            first_selected_restored = None
            for i in range(self.steps_tree.topLevelItemCount()):
                item = self.steps_tree.topLevelItem(i)
                if item.isSelected():
                    first_selected_restored = item
                    break
            if first_selected_restored:
                 self.steps_tree.setCurrentItem(first_selected_restored)


        # Refreshing steps inherently requires refreshing voices based on new (or lack of) focus
        self.on_step_select() # This will call refresh_voices_tree and update button states


    def refresh_voices_tree(self):
        current_focused_voice_item_data = None
        current_voice_item = self.voices_tree.currentItem()
        if current_voice_item:
            current_focused_voice_item_data = current_voice_item.data(0, Qt.UserRole)

        selected_voice_items_data = set()
        for item in self.voices_tree.selectedItems():
            data = item.data(0, Qt.UserRole)
            if data is not None:
                selected_voice_items_data.add(data)

        self.voices_tree.clear()
        self.clear_voice_details()

        selected_step_idx = self.get_selected_step_index() # Focused step

        if selected_step_idx is None or len(self.steps_tree.selectedItems()) != 1:
            self.voices_groupbox.setTitle("Voices for Selected Step")
            self._update_voice_actions_state() # Update buttons as no valid single step is selected
            return

        self.voices_groupbox.setTitle(f"Voices for Step {selected_step_idx + 1}")
        try:
            step_data = self.track_data["steps"][selected_step_idx]
            voices = step_data.get("voices", [])
            for i, voice in enumerate(voices):
                func_name = voice.get("synth_function_name", "N/A")
                params = voice.get("params", {})
                is_transition = voice.get("is_transition", False)
                carrier_freq_str = 'N/A'
                if 'baseFreq' in params: carrier_freq = params['baseFreq']
                else:
                    freq_keys = [k for k in params if ('Freq' in k or 'Frequency' in k) and not k.startswith(('start','end','target'))]
                    if is_transition:
                        freq_keys = [k for k in params if k.startswith('start') and ('Freq' in k or 'Frequency' in k)] or freq_keys
                    carrier_freq = params.get(freq_keys[0]) if freq_keys else 'N/A'
                try:
                    if carrier_freq is not None and carrier_freq != 'N/A': carrier_freq_str = f"{float(carrier_freq):.2f}"
                    else: carrier_freq_str = str(carrier_freq) # Display 'N/A' or variable name
                except (ValueError, TypeError): carrier_freq_str = str(carrier_freq)

                transition_str = "Yes" if is_transition else "No"
                item = QTreeWidgetItem(self.voices_tree)
                item.setText(0, func_name)
                item.setText(1, carrier_freq_str)
                item.setText(2, transition_str)
                item.setData(0, Qt.UserRole, i)

            new_focused_voice_item = None
            for i in range(self.voices_tree.topLevelItemCount()):
                item = self.voices_tree.topLevelItem(i)
                item_data = item.data(0, Qt.UserRole)
                if item_data in selected_voice_items_data:
                    item.setSelected(True)
                    if item_data == current_focused_voice_item_data:
                        new_focused_voice_item = item
            
            if new_focused_voice_item:
                self.voices_tree.setCurrentItem(new_focused_voice_item)
                self.voices_tree.scrollToItem(new_focused_voice_item, QTreeWidget.PositionAtCenter)
            elif self.voices_tree.topLevelItemCount() > 0 and selected_voice_items_data:
                first_selected_restored_voice = None
                for i in range(self.voices_tree.topLevelItemCount()):
                    item = self.voices_tree.topLevelItem(i)
                    if item.isSelected():
                        first_selected_restored_voice = item
                        break
                if first_selected_restored_voice:
                    self.voices_tree.setCurrentItem(first_selected_restored_voice)


        except IndexError:
            print(f"Error: Selected step index {selected_step_idx} out of range for voice display.")
            self.voices_groupbox.setTitle("Voices for Selected Step")
        except Exception as e:
            print(f"Error refreshing voices tree: {e}")
            traceback.print_exc()
        
        self.on_voice_select() # Update details and button states


    def clear_voice_details(self):
        self.voice_details_text.clear()
        self.voice_details_groupbox.setTitle("Selected Voice Details")

    def update_voice_details(self):
        self.clear_voice_details()
        # Details should only show if exactly one voice in a single step is selected and focused
        if len(self.steps_tree.selectedItems()) != 1 or len(self.voices_tree.selectedItems()) != 1:
            return

        selected_step_idx = self.get_selected_step_index() # Focused step
        selected_voice_idx = self.get_selected_voice_index() # Focused voice

        if selected_step_idx is None or selected_voice_idx is None: return

        try:
            voice_data = self.track_data["steps"][selected_step_idx]["voices"][selected_voice_idx]
            func_name = voice_data.get('synth_function_name', 'N/A')
            self.voice_details_groupbox.setTitle(f"Details: Step {selected_step_idx+1}, Voice {selected_voice_idx+1} ({func_name})")
            params = voice_data.get("params", {})
            details = f"Function: {func_name}\n"
            details += f"Transition: {'Yes' if voice_data.get('is_transition', False) else 'No'}\n"
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
        self._update_step_actions_state()
        # Refresh voices only if a single step is selected/focused
        if len(self.steps_tree.selectedItems()) == 1 and self.steps_tree.currentItem() is not None:
            self.refresh_voices_tree() # This will also call on_voice_select -> _update_voice_actions_state
        else:
            self.voices_tree.clear()
            self.clear_voice_details()
            self._update_voice_actions_state() # Explicitly update voice button states if no single step is focused


    @pyqtSlot()
    def on_voice_select(self):
        self._update_voice_actions_state()
        self.update_voice_details() # Update details based on current (single) voice selection


    # --- Action Methods (Slots for Buttons) ---
    @pyqtSlot()
    def load_json(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Track JSON", "", "JSON files (*.json);;All files (*.*)")
        if not filepath: return
        try:
            loaded_data = sound_creator.load_track_from_json(filepath)
            if loaded_data and isinstance(loaded_data.get("steps"), list) and isinstance(loaded_data.get("global_settings"), dict):
                self.track_data = loaded_data
                self.current_json_path = filepath
                self.setWindowTitle(f"Binaural Track Editor (PyQt5) - {os.path.basename(filepath)}")
                self._update_ui_from_global_settings()
                self.refresh_steps_tree()
                QMessageBox.information(self, "Load Success", f"Track loaded from\n{filepath}")
            elif loaded_data is not None:
                QMessageBox.critical(self, "Load Error", "Invalid JSON structure.")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load or parse JSON file:\n{e}")
            traceback.print_exc()
        self._update_step_actions_state()
        self._update_voice_actions_state()

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
        suggested_path = os.path.join(initial_dir, initial_filename)
        filepath, _ = QFileDialog.getSaveFileName(self, "Select Output WAV File", suggested_path, "WAV files (*.wav);;All files (*.*)")
        if filepath:
            self.outfile_entry.setText(filepath)

    @pyqtSlot()
    def load_external_step(self): # NEW
        filepath, _ = QFileDialog.getOpenFileName(self, "Load External Steps from JSON", "",
                                                  "JSON files (*.json);;All files (*.*)")
        if not filepath:
            return

        try:
            with open(filepath, 'r') as f:
                external_data = json.load(f)

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
                QMessageBox.information(self, "Load Success",
                                        f"{loaded_count} step(s) loaded and added from\n{filepath}")
                # Select the first of the newly added steps if focus makes sense
                if current_step_count < len(self.track_data["steps"]):
                    new_item = self.steps_tree.topLevelItem(current_step_count)
                    if new_item: # Check if item exists
                        self.steps_tree.setCurrentItem(new_item) # This triggers on_step_select
                        self.steps_tree.scrollToItem(new_item, QTreeWidget.PositionAtCenter)
            else:
                 QMessageBox.information(self, "Load Info", "No valid steps found to load.")


        except json.JSONDecodeError:
            QMessageBox.critical(self, "Load Error", f"Failed to decode JSON from file:\n{filepath}")
            traceback.print_exc()
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"An unexpected error occurred:\n{e}")
            traceback.print_exc()
        self._update_step_actions_state() # Ensure buttons are updated
        self._update_voice_actions_state()


    @pyqtSlot()
    def add_step(self):
        new_step = {"duration": 10.0, "description": "New Step", "voices": []}
        selected_focused_index = self.get_selected_step_index() # Index of focused item
        insert_index = selected_focused_index + 1 if selected_focused_index is not None else len(self.track_data["steps"])

        self.track_data["steps"].insert(insert_index, new_step)
        self.refresh_steps_tree()
        if 0 <= insert_index < self.steps_tree.topLevelItemCount():
            new_item = self.steps_tree.topLevelItem(insert_index)
            # Clear existing selection before setting new current item to make it the only selected one
            self.steps_tree.clearSelection()
            self.steps_tree.setCurrentItem(new_item)
            new_item.setSelected(True)
            self.steps_tree.scrollToItem(new_item, QTreeWidget.PositionAtCenter)
        self._update_step_actions_state()
        self._update_voice_actions_state()

    @pyqtSlot()
    def duplicate_step(self):
        selected_index = self.get_selected_step_index() # Focused item
        if selected_index is None or len(self.steps_tree.selectedItems()) != 1 : # Only if one item is focused and selected
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
        except IndexError:
             QMessageBox.critical(self, "Error", "Failed to duplicate step (index out of range).")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to duplicate step:\n{e}")
            traceback.print_exc()
        self._update_step_actions_state()

    @pyqtSlot()
    def remove_step(self): # MODIFIED for multi-select
        selected_indices = self.get_selected_step_indices()
        if not selected_indices:
            QMessageBox.warning(self, "Remove Step(s)", "Please select one or more steps to remove.")
            return
        reply = QMessageBox.question(self, "Confirm Remove", f"Are you sure you want to remove {len(selected_indices)} selected step(s)?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            try:
                for index in sorted(selected_indices, reverse=True):
                    if 0 <= index < len(self.track_data["steps"]):
                        del self.track_data["steps"][index]
                self.refresh_steps_tree() # This calls on_step_select -> updates buttons and voices
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to remove step(s):\n{e}")
                traceback.print_exc()
        # Button states updated by refresh_steps_tree -> on_step_select cascade

    @pyqtSlot()
    def edit_step_duration(self):
        selected_index = self.get_selected_step_index() # Focused item
        if selected_index is None or len(self.steps_tree.selectedItems()) != 1:
            QMessageBox.warning(self, "Edit Duration", "Please select exactly one step to edit.")
            return
        try:
            current_duration = float(self.track_data["steps"][selected_index].get("duration", 0.0))
        except (IndexError, ValueError, TypeError) as e:
            QMessageBox.critical(self, "Error", f"Failed to get current duration (index {selected_index}):\n{e}")
            return
        new_duration, ok = QInputDialog.getDouble(self, f"Edit Step {selected_index + 1} Duration", "New Duration (s):", current_duration, 0.001, 99999.0, 3)
        if ok and new_duration is not None:
             if new_duration <= 0:
                 QMessageBox.warning(self, "Invalid Input", "Duration must be positive.")
                 return
             try:
                 self.track_data["steps"][selected_index]["duration"] = new_duration
                 self.refresh_steps_tree() # Will re-select and update buttons
             except IndexError:
                 QMessageBox.critical(self, "Error", "Failed to set duration (index out of range after edit).")
             except Exception as e:
                 QMessageBox.critical(self, "Error", f"Failed to set duration:\n{e}")
        # Button states updated by refresh_steps_tree

    @pyqtSlot()
    def edit_step_description(self):
        selected_index = self.get_selected_step_index() # Focused item
        if selected_index is None or len(self.steps_tree.selectedItems()) != 1:
            QMessageBox.warning(self, "Edit Description", "Please select exactly one step to edit.")
            return
        try:
            current_description = str(self.track_data["steps"][selected_index].get("description", ""))
        except IndexError as e:
            QMessageBox.critical(self, "Error", f"Failed to get current description (index {selected_index}):\n{e}")
            return
        new_description, ok = QInputDialog.getText(self, f"Edit Step {selected_index + 1} Description", "Description:", QLineEdit.Normal, current_description)
        if ok and new_description is not None:
             try:
                 self.track_data["steps"][selected_index]["description"] = new_description.strip()
                 self.refresh_steps_tree() # Will re-select and update buttons
             except IndexError:
                 QMessageBox.critical(self, "Error", "Failed to set description (index out of range after edit).")
             except Exception as e:
                 QMessageBox.critical(self, "Error", f"Failed to set description:\n{e}")
        # Button states updated by refresh_steps_tree

    @pyqtSlot()
    def move_step(self, direction):
        selected_index = self.get_selected_step_index() # Focused item
        if selected_index is None or len(self.steps_tree.selectedItems()) != 1:
            QMessageBox.warning(self, "Move Step", "Please select exactly one step to move.")
            return
        num_steps = len(self.track_data["steps"])
        new_index = selected_index + direction
        if 0 <= new_index < num_steps:
            try:
                steps = self.track_data["steps"]
                steps[selected_index], steps[new_index] = steps[new_index], steps[selected_index]
                self.refresh_steps_tree()
                if new_index < self.steps_tree.topLevelItemCount(): # Ensure new_index is valid
                    moved_item = self.steps_tree.topLevelItem(new_index)
                    self.steps_tree.clearSelection() # Clear old selections
                    self.steps_tree.setCurrentItem(moved_item) # Set new current item
                    moved_item.setSelected(True) # Explicitly select it
                    self.steps_tree.scrollToItem(moved_item, QTreeWidget.PositionAtCenter)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to move step:\n{e}")
                traceback.print_exc()
        # Button states updated by refresh_steps_tree

    @pyqtSlot()
    def add_voice(self):
        selected_step_index = self.get_selected_step_index() # Focused step
        if selected_step_index is None or len(self.steps_tree.selectedItems()) != 1:
            QMessageBox.warning(self, "Add Voice", "Please select exactly one step first.")
            return
        try:
            current_voices = self.track_data["steps"][selected_step_index].get("voices", [])
            if len(current_voices) >= MAX_VOICES_PER_STEP:
                QMessageBox.warning(self, "Add Voice", f"Maximum voices per step ({MAX_VOICES_PER_STEP}) reached.")
                return
        except IndexError:
            QMessageBox.critical(self, "Error", "Cannot add voice (selected step index out of range).")
            return
        dialog = VoiceEditorDialog(parent=self, app_ref=self, step_index=selected_step_index, voice_index=None)
        if dialog.exec_() == QDialog.Accepted:
             self.refresh_steps_tree() # Update step voice count
             # Select the parent step and the new voice if the step is still valid
             if selected_step_index < self.steps_tree.topLevelItemCount():
                 step_item = self.steps_tree.topLevelItem(selected_step_index)
                 self.steps_tree.setCurrentItem(step_item) # This will trigger voice refresh via on_step_select
                 # Now select the last voice in the refreshed list
                 QTimer.singleShot(0, lambda: self._select_last_voice_in_current_step()) # Use QTimer for reliability

    def _select_last_voice_in_current_step(self):
        voice_count = self.voices_tree.topLevelItemCount()
        if voice_count > 0:
            new_voice_item = self.voices_tree.topLevelItem(voice_count - 1)
            self.voices_tree.clearSelection()
            self.voices_tree.setCurrentItem(new_voice_item)
            new_voice_item.setSelected(True)
            self.voices_tree.scrollToItem(new_voice_item, QTreeWidget.PositionAtCenter)
        self._update_voice_actions_state() # Ensure button states are correct


    @pyqtSlot()
    def edit_voice(self):
        selected_step_index = self.get_selected_step_index() # Focused step
        selected_voice_index = self.get_selected_voice_index() # Focused voice

        if selected_step_index is None or selected_voice_index is None or \
           len(self.steps_tree.selectedItems()) != 1 or len(self.voices_tree.selectedItems()) != 1:
            QMessageBox.warning(self, "Edit Voice", "Please select exactly one step and one voice to edit.")
            return
        dialog = VoiceEditorDialog(parent=self, app_ref=self, step_index=selected_step_index, voice_index=selected_voice_index)
        if dialog.exec_() == QDialog.Accepted:
            self.refresh_steps_tree() # This will cascade to refresh voices and update buttons
            # Try to re-select the edited voice
            if selected_step_index < self.steps_tree.topLevelItemCount():
                step_item = self.steps_tree.topLevelItem(selected_step_index)
                self.steps_tree.setCurrentItem(step_item) # Focus the step
                if selected_voice_index < self.voices_tree.topLevelItemCount():
                     voice_item = self.voices_tree.topLevelItem(selected_voice_index)
                     self.voices_tree.setCurrentItem(voice_item) # Focus the voice
                     self.voices_tree.scrollToItem(voice_item, QTreeWidget.PositionAtCenter)
            self._update_voice_actions_state()


    @pyqtSlot()
    def remove_voice(self): # MODIFIED for multi-select
        selected_step_idx = self.get_selected_step_index() # Focused step
        selected_voice_indices = self.get_selected_voice_indices()

        if selected_step_idx is None or len(self.steps_tree.selectedItems()) != 1:
            QMessageBox.warning(self, "Remove Voice(s)", "Please select exactly one step first.")
            return
        if not selected_voice_indices:
            QMessageBox.warning(self, "Remove Voice(s)", "Please select one or more voices to remove.")
            return
        try:
            voices_list = self.track_data["steps"][selected_step_idx].get("voices")
            if not voices_list: return

            reply = QMessageBox.question(self, "Confirm Remove", f"Remove {len(selected_voice_indices)} selected voice(s) from Step {selected_step_idx + 1}?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                for voice_idx in sorted(selected_voice_indices, reverse=True):
                    if 0 <= voice_idx < len(voices_list):
                        del voices_list[voice_idx]
                self.refresh_steps_tree() # Update voice count for the step, and this will refresh voice tree & buttons
        except IndexError:
            QMessageBox.critical(self, "Error", "Failed to remove voice(s) (step index out of range).")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to remove voice(s):\n{e}")
            traceback.print_exc()
        # Button states updated by refresh cascade

    @pyqtSlot(int) # NEW method from previous plan
    def move_voice(self, direction):
        selected_step_idx = self.get_selected_step_index() # Focused step
        selected_voice_idx = self.get_selected_voice_index() # Focused voice

        if selected_step_idx is None or selected_voice_idx is None or \
           len(self.steps_tree.selectedItems()) != 1 or len(self.voices_tree.selectedItems()) != 1:
            QMessageBox.warning(self, "Move Voice", "Please select exactly one step and one voice to move.")
            return

        try:
            voices_list = self.track_data["steps"][selected_step_idx]["voices"]
            num_voices = len(voices_list)
            new_voice_idx = selected_voice_idx + direction

            if 0 <= new_voice_idx < num_voices:
                voices_list[selected_voice_idx], voices_list[new_voice_idx] = \
                    voices_list[new_voice_idx], voices_list[selected_voice_idx]
                self.refresh_voices_tree()
                if new_voice_idx < self.voices_tree.topLevelItemCount():
                    moved_item = self.voices_tree.topLevelItem(new_voice_idx)
                    self.voices_tree.clearSelection()
                    self.voices_tree.setCurrentItem(moved_item)
                    moved_item.setSelected(True)
                    self.voices_tree.scrollToItem(moved_item, QTreeWidget.PositionAtCenter)
            # self.update_voice_details() # update_voice_details is called by on_voice_select from refresh_voices_tree
        except IndexError:
            QMessageBox.critical(self, "Error", "Failed to move voice (index out of range).")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred while moving voice:\n{e}")
        self._update_voice_actions_state() # Ensure button states are correct


    @pyqtSlot()
    def generate_wav_action(self):
        if not self._update_global_settings_from_ui(): return
        output_filename = self.track_data["global_settings"]["output_filename"]
        if not output_filename:
            QMessageBox.critical(self, "Generate Error", "Please specify an output filename.")
            return
        if os.path.exists(output_filename):
            reply = QMessageBox.question(self, "Confirm Overwrite", f"Output file '{os.path.basename(output_filename)}' exists.\nOverwrite?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No: return

        self.generate_button.setEnabled(False)
        self.generate_button.setText("Generating...")
        QApplication.processEvents()
        try:
            success = sound_creator.generate_wav(self.track_data, output_filename)
            if success:
                QMessageBox.information(self, "Generation Complete", f"WAV file generated successfully:\n{output_filename}")
            else:
                QMessageBox.critical(self, "Generation Failed", "Error during WAV generation. Check console/logs.")
        except Exception as e:
            QMessageBox.critical(self, "Generation Error", f"An unexpected error occurred:\n{e}")
            traceback.print_exc()
        finally:
            self.generate_button.setEnabled(True)
            self.generate_button.setText("Generate WAV")

    # --- Utility Methods ---
    def get_selected_step_index(self): # MODIFIED: Gets focused item's index
        """Gets the original index of the CURRENTLY FOCUSED step."""
        current_item = self.steps_tree.currentItem()
        if current_item:
            data = current_item.data(0, Qt.UserRole)
            if data is not None:
                return int(data)
        return None

    def get_selected_step_indices(self): # NEW
        """Gets the original indices of all selected steps, sorted."""
        selected_items = self.steps_tree.selectedItems()
        indices = []
        if selected_items:
            for item in selected_items:
                data = item.data(0, Qt.UserRole)
                if data is not None:
                    indices.append(int(data))
        return sorted(indices)

    def get_selected_voice_index(self): # MODIFIED: Gets focused item's index
        """Gets the original index of the CURRENTLY FOCUSED voice."""
        current_item = self.voices_tree.currentItem()
        if current_item:
            data = current_item.data(0, Qt.UserRole)
            if data is not None:
                return int(data)
        return None

    def get_selected_voice_indices(self): # NEW
        """Gets the original indices of all selected voices, sorted."""
        selected_items = self.voices_tree.selectedItems()
        indices = []
        if selected_items:
            for item in selected_items:
                data = item.data(0, Qt.UserRole)
                if data is not None:
                    indices.append(int(data))
        return sorted(indices)

    def closeEvent(self, event):
        super().closeEvent(event)


# --- Voice Editor Dialog Class (Unchanged from previous version, but included for completeness) ---
class VoiceEditorDialog(QDialog):
    DEFAULT_WIDTH = 900
    DEFAULT_HEIGHT = 700

    def __init__(self, parent, app_ref, step_index, voice_index=None):
        super().__init__(parent)
        self.app = app_ref
        self.step_index = step_index
        self.voice_index = voice_index
        self.is_new_voice = (voice_index is None)

        self.double_validator_non_negative = QDoubleValidator(0.0, 999999.0, 6, self)
        self.double_validator_zero_to_one = QDoubleValidator(0.0, 1.0, 6, self)
        self.double_validator = QDoubleValidator(-999999.0, 999999.0, 6, self)
        self.int_validator = QIntValidator(-999999, 999999, self)

        self.setWindowTitle(f"{'Add' if self.is_new_voice else 'Edit'} Voice for Step {step_index + 1}")
        self.resize(self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT)
        self.setMinimumSize(700, 600)
        self.setModal(True)

        self.param_widgets = {}
        self.envelope_param_widgets = {}

        self._load_initial_data()
        self._setup_ui()
        self.populate_parameters()
        self._populate_envelope_controls()
        self._populate_reference_step_combo()

        initial_ref_step = self.app.get_selected_step_index() # Focused step in main window
        initial_ref_voice = self.app.get_selected_voice_index() # Focused voice in main window
        if initial_ref_step is not None:
             step_combo_index = self.reference_step_combo.findData(initial_ref_step)
             if step_combo_index != -1:
                 self.reference_step_combo.setCurrentIndex(step_combo_index)
                 if initial_ref_voice is not None:
                     QTimer.singleShot(50, lambda: self._select_initial_reference_voice(initial_ref_voice))
             elif self.reference_step_combo.count() > 0:
                 self.reference_step_combo.setCurrentIndex(0)
        elif self.reference_step_combo.count() > 0:
             self.reference_step_combo.setCurrentIndex(0)


    def _load_initial_data(self):
        if self.is_new_voice:
            available_funcs = sorted(sound_creator.SYNTH_FUNCTIONS.keys())
            first_func_name = available_funcs[0] if available_funcs else ""
            is_trans = first_func_name.endswith("_transition")
            default_params = self._get_default_params(first_func_name, is_trans)
            self.current_voice_data = {
                "synth_function_name": first_func_name,
                "is_transition": is_trans,
                "params": default_params,
                "volume_envelope": None
            }
        else:
            try:
                original_voice = self.app.track_data["steps"][self.step_index]["voices"][self.voice_index]
                self.current_voice_data = copy.deepcopy(original_voice)
                if "params" not in self.current_voice_data: self.current_voice_data["params"] = {}
                if "volume_envelope" not in self.current_voice_data: self.current_voice_data["volume_envelope"] = None
                if "is_transition" not in self.current_voice_data:
                    self.current_voice_data["is_transition"] = self.current_voice_data.get("synth_function_name","").endswith("_transition")
            except (IndexError, KeyError) as e:
                QMessageBox.critical(self.parent(), "Error", f"Could not load voice data for editing:\n{e}")
                self.current_voice_data = {"params": {}, "synth_function_name": "Error", "is_transition": False, "volume_envelope": None}
                QTimer.singleShot(0, self.reject)


    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        top_frame = QWidget()
        top_layout = QHBoxLayout(top_frame)
        top_layout.setContentsMargins(0,0,0,0)
        top_layout.addWidget(QLabel("Synth Function:"))
        self.synth_func_combo = QComboBox()
        func_names = sorted(sound_creator.SYNTH_FUNCTIONS.keys())
        self.synth_func_combo.addItems(func_names)
        current_func_name = self.current_voice_data.get("synth_function_name", "")
        if current_func_name in func_names: self.synth_func_combo.setCurrentText(current_func_name)
        self.synth_func_combo.currentIndexChanged.connect(self.on_synth_function_change)
        top_layout.addWidget(self.synth_func_combo, 1)
        self.transition_check = QCheckBox("Is Transition?")
        self.transition_check.setChecked(self.current_voice_data.get("is_transition", False))
        self.transition_check.stateChanged.connect(self.on_transition_toggle)
        top_layout.addWidget(self.transition_check)
        main_layout.addWidget(top_frame)

        h_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(h_splitter, 1)
        self.params_groupbox = QGroupBox("Synth Parameters (Editing)")
        params_groupbox_layout = QVBoxLayout(self.params_groupbox)
        self.params_scroll_area = QScrollArea()
        self.params_scroll_area.setWidgetResizable(True)
        self.params_scroll_content = QWidget()
        self.params_scroll_layout = QVBoxLayout(self.params_scroll_content)
        self.params_scroll_layout.setAlignment(Qt.AlignTop)
        self.params_scroll_area.setWidget(self.params_scroll_content)
        params_groupbox_layout.addWidget(self.params_scroll_area)
        h_splitter.addWidget(self.params_groupbox)

        reference_groupbox = QGroupBox("Select Voice for Reference")
        reference_layout = QVBoxLayout(reference_groupbox)
        ref_select_layout = QHBoxLayout()
        ref_select_layout.addWidget(QLabel("Step:"))
        self.reference_step_combo = QComboBox()
        self.reference_step_combo.setMinimumWidth(100)
        self.reference_step_combo.currentIndexChanged.connect(self._update_reference_voice_combo)
        ref_select_layout.addWidget(self.reference_step_combo)
        ref_select_layout.addWidget(QLabel("Voice:"))
        self.reference_voice_combo = QComboBox()
        self.reference_voice_combo.setMinimumWidth(150)
        self.reference_voice_combo.currentIndexChanged.connect(self._update_reference_display)
        ref_select_layout.addWidget(self.reference_voice_combo, 1)
        reference_layout.addLayout(ref_select_layout)
        self.reference_details_text = QTextEdit()
        self.reference_details_text.setReadOnly(True)
        self.reference_details_text.setFont(QFont("Consolas", 9))
        reference_layout.addWidget(self.reference_details_text, 1)
        h_splitter.addWidget(reference_groupbox)
        h_splitter.setSizes([500, 350])

        self.env_groupbox = QGroupBox("Volume Envelope")
        env_layout = QVBoxLayout(self.env_groupbox)
        env_type_layout = QHBoxLayout()
        env_type_layout.addWidget(QLabel("Type:"))
        self.env_type_combo = QComboBox()
        self.env_type_combo.addItems(SUPPORTED_ENVELOPE_TYPES)
        self.env_type_combo.currentIndexChanged.connect(self._on_envelope_type_change)
        env_type_layout.addWidget(self.env_type_combo)
        env_type_layout.addStretch(1)
        env_layout.addLayout(env_type_layout)
        self.env_params_widget = QWidget()
        self.env_params_layout = QGridLayout(self.env_params_widget)
        self.env_params_layout.setContentsMargins(10, 5, 5, 5)
        self.env_params_layout.setAlignment(Qt.AlignTop)
        env_layout.addWidget(self.env_params_widget)
        env_layout.addStretch(1)
        main_layout.addWidget(self.env_groupbox)

        button_frame = QWidget()
        button_layout = QHBoxLayout(button_frame)
        button_layout.addStretch(1)
        self.cancel_button = QPushButton("Cancel")
        self.save_button = QPushButton("Save Voice")
        self.save_button.setStyleSheet("QPushButton { background-color: #0078D7; color: white; padding: 6px; font-weight: bold; border-radius: 3px; } QPushButton:hover { background-color: #005A9E; } QPushButton:pressed { background-color: #003C6A; }")
        self.cancel_button.clicked.connect(self.reject)
        self.save_button.clicked.connect(self.save_voice)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.save_button)
        main_layout.addWidget(button_frame)

    def _clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None: widget.deleteLater()
                else:
                    sub_layout = item.layout()
                    if sub_layout is not None: self._clear_layout(sub_layout)

    def populate_parameters(self):
        self._clear_layout(self.params_scroll_layout)
        self.param_widgets = {}
        func_name = self.synth_func_combo.currentText()
        is_transition = self.transition_check.isChecked()
        default_params = self._get_default_params(func_name, is_transition)
        if not default_params:
            self.params_scroll_layout.addWidget(QLabel(f"Warning: Could not determine parameters for '{func_name}'.", self))
            return

        current_saved_params = self.current_voice_data.get("params", {})
        params_to_display = default_params.copy()
        for key, value in current_saved_params.items():
            if key in params_to_display: params_to_display[key] = value

        processed_end_params = set()
        transition_pairs = {}
        if is_transition:
            for name in default_params.keys():
                if name.startswith('start'):
                    base_name = name[len('start'):]
                    end_name = 'end' + base_name
                    if end_name in default_params:
                        transition_pairs[base_name] = {'start': name, 'end': end_name}
                        processed_end_params.add(end_name)
                elif name.startswith('end'):
                    base_name = name[len('end'):]
                    if ('start' + base_name) in default_params and base_name not in transition_pairs:
                        processed_end_params.add(name)
        
        param_names_sorted = sorted(default_params.keys())
        for name in param_names_sorted:
            if name in processed_end_params: continue
            default_value = default_params[name]
            current_value = params_to_display.get(name, default_value)
            base_name_for_pair = name[len('start'):] if is_transition and name.startswith('start') else None
            is_pair_start = base_name_for_pair is not None and base_name_for_pair in transition_pairs

            frame = QWidget()
            row_layout = QGridLayout(frame)
            row_layout.setContentsMargins(2,2,2,2)
            param_storage_type = 'str'; param_type_hint = "any"
            range_hint = self._get_param_range_hint(name if not is_pair_start else base_name_for_pair)

            if isinstance(default_value, bool): param_type_hint = 'bool'
            elif isinstance(default_value, int): param_type_hint = 'float' # Treat as float for QDoubleValidator
            elif isinstance(default_value, float): param_type_hint = 'float'
            elif isinstance(default_value, str): param_type_hint = 'str'
            elif default_value is None: # Heuristics
                if 'bool' in name.lower(): param_type_hint = 'bool'
                elif any(s in name.lower() for s in ['freq', 'depth', 'dur', 'amp', 'pan', 'rate', 'gain', 'level']): param_type_hint = 'float'
                else: param_type_hint = 'int'


            if is_pair_start:
                start_name = name; end_name = transition_pairs[base_name_for_pair]['end']
                end_val = params_to_display.get(end_name, default_params.get(end_name, current_value))
                validator = QDoubleValidator(-999999.0, 999999.0, 6, self) if param_type_hint != 'int' else self.int_validator
                if isinstance(validator, QDoubleValidator): validator.setNotation(QDoubleValidator.StandardNotation)
                param_storage_type = 'float' if param_type_hint != 'int' else 'int'
                hint_text = f"({param_storage_type}{', ' + range_hint if range_hint else ''})"

                row_layout.addWidget(QLabel(f"{base_name_for_pair}:"), 0, 0, Qt.AlignLeft)
                row_layout.addWidget(QLabel("Start:"), 0, 1, Qt.AlignRight)
                start_entry = QLineEdit(str(current_value) if current_value is not None else "")
                start_entry.setValidator(validator); start_entry.setMinimumWidth(60)
                row_layout.addWidget(start_entry, 0, 2)
                self.param_widgets[start_name] = {'widget': start_entry, 'type': param_storage_type}
                row_layout.addWidget(QLabel("End:"), 0, 3, Qt.AlignRight)
                end_entry = QLineEdit(str(end_val) if end_val is not None else "")
                end_entry.setValidator(validator); end_entry.setMinimumWidth(60)
                row_layout.addWidget(end_entry, 0, 4)
                self.param_widgets[end_name] = {'widget': end_entry, 'type': param_storage_type}
                row_layout.addWidget(QLabel(hint_text), 0, 5, Qt.AlignLeft)
                row_layout.setColumnStretch(2, 1); row_layout.setColumnStretch(4, 1); row_layout.setColumnStretch(5, 2)
            else: # Single parameter
                widget = None
                row_layout.addWidget(QLabel(f"{name}:"), 0, 0, Qt.AlignLeft)
                if 'bool' in param_type_hint.lower():
                    widget = QCheckBox(); widget.setChecked(bool(current_value) if current_value is not None else False)
                    row_layout.addWidget(widget, 0, 1, 1, 2, Qt.AlignLeft); param_storage_type = 'bool'
                elif name == 'noiseType' and param_type_hint == 'int':
                    widget = QComboBox(); widget.addItems(['1', '2', '3']) # White, Pink, Brown
                    widget.setCurrentText(str(int(current_value)) if current_value in [1,2,3] else '1')
                    widget.setMaximumWidth(100); row_layout.addWidget(widget, 0, 1, 1, 1, Qt.AlignLeft); param_storage_type = 'int'
                elif name == 'pathShape' and param_type_hint == 'str' and hasattr(sound_creator, 'VALID_SAM_PATHS'):
                    widget = QComboBox(); widget.addItems(sound_creator.VALID_SAM_PATHS)
                    widget.setCurrentText(str(current_value) if current_value in sound_creator.VALID_SAM_PATHS else sound_creator.VALID_SAM_PATHS[0])
                    widget.setMinimumWidth(120); row_layout.addWidget(widget, 0, 1, 1, 2, Qt.AlignLeft); param_storage_type = 'str'
                else: # General QLineEdit
                    widget = QLineEdit(str(current_value) if current_value is not None else "")
                    validator = None; entry_width = 150
                    if param_type_hint == 'int': validator = self.int_validator; param_storage_type = 'int'; entry_width = 80
                    elif param_type_hint == 'float':
                        validator = QDoubleValidator(-999999.0, 999999.0, 6, self)
                        validator.setNotation(QDoubleValidator.StandardNotation)
                        param_storage_type = 'float'; entry_width = 80
                    else: param_storage_type = 'str'; entry_width = 200
                    if validator: widget.setValidator(validator)
                    widget.setMinimumWidth(entry_width); row_layout.addWidget(widget, 0, 1, 1, 4)
                hint_text = f"({param_storage_type}{', ' + range_hint if range_hint else ''})"
                row_layout.addWidget(QLabel(hint_text), 0, 5, Qt.AlignLeft)
                row_layout.setColumnStretch(1,1); row_layout.setColumnStretch(5,1)
                if widget is not None: self.param_widgets[name] = {'widget': widget, 'type': param_storage_type}
            self.params_scroll_layout.addWidget(frame)
        self.params_scroll_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

    def _populate_envelope_controls(self):
        env_data = self.current_voice_data.get("volume_envelope")
        env_type = ENVELOPE_TYPE_NONE
        if isinstance(env_data, dict) and "type" in env_data: env_type = env_data["type"]
        self.env_type_combo.blockSignals(True)
        self.env_type_combo.setCurrentText(env_type if env_type in SUPPORTED_ENVELOPE_TYPES else ENVELOPE_TYPE_NONE)
        self.env_type_combo.blockSignals(False)
        self._on_envelope_type_change()

    @pyqtSlot()
    def _on_envelope_type_change(self):
        self._clear_layout(self.env_params_layout)
        self.envelope_param_widgets = {}
        selected_type = self.env_type_combo.currentText()
        env_data = self.current_voice_data.get("volume_envelope")
        current_env_params = {}
        if isinstance(env_data, dict) and env_data.get("type") == selected_type:
             current_env_params = env_data.get("params", {})
        row = 0
        if selected_type == ENVELOPE_TYPE_LINEAR:
            params_def = [
                ("Fade Duration (s):", "fade_duration", 0.1, self.double_validator_non_negative, "float"),
                ("Start Amplitude:", "start_amp", 0.0, self.double_validator_zero_to_one, "float"),
                ("End Amplitude:", "end_amp", 1.0, self.double_validator_zero_to_one, "float")
            ]
            for label_text, param_name, default_val, validator, val_type in params_def:
                label = QLabel(label_text); entry = QLineEdit()
                entry.setValidator(validator)
                entry.setText(str(current_env_params.get(param_name, default_val)))
                self.env_params_layout.addWidget(label, row, 0)
                self.env_params_layout.addWidget(entry, row, 1)
                self.envelope_param_widgets[param_name] = {'widget': entry, 'type': val_type}
                row += 1
            self.env_params_layout.setColumnStretch(1, 1)
        if row == 0: self.env_params_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding), row, 0, 1, 2)
        self.env_params_widget.setVisible(selected_type != ENVELOPE_TYPE_NONE)

    def _populate_reference_step_combo(self):
        self.reference_step_combo.blockSignals(True); self.reference_step_combo.clear()
        steps = self.app.track_data.get("steps", [])
        if not steps: self.reference_step_combo.addItem("No Steps Available", -1); self.reference_step_combo.setEnabled(False)
        else:
            self.reference_step_combo.setEnabled(True)
            for i, _ in enumerate(steps): self.reference_step_combo.addItem(f"Step {i+1}", i)
        self.reference_step_combo.blockSignals(False)

    @pyqtSlot(int)
    def _update_reference_voice_combo(self, _combo_idx=-1):
        self.reference_voice_combo.blockSignals(True); self.reference_voice_combo.clear()
        selected_step_index = self.reference_step_combo.currentData()
        if selected_step_index is None or selected_step_index < 0:
            self.reference_voice_combo.addItem("No Voices Available", -1); self.reference_voice_combo.setEnabled(False)
        else:
            try:
                voices = self.app.track_data["steps"][selected_step_index].get("voices", [])
                if not voices: self.reference_voice_combo.addItem("No Voices in Step", -1); self.reference_voice_combo.setEnabled(False)
                else:
                    self.reference_voice_combo.setEnabled(True)
                    for i, voice in enumerate(voices): self.reference_voice_combo.addItem(f"Voice {i+1} ({voice.get('synth_function_name', 'N/A')})", i)
            except IndexError: self.reference_voice_combo.addItem("Error loading voices", -1); self.reference_voice_combo.setEnabled(False)
        self.reference_voice_combo.blockSignals(False)
        if self.reference_voice_combo.count() > 0 and self.reference_voice_combo.itemData(0) != -1: self.reference_voice_combo.setCurrentIndex(0)
        self._update_reference_display()

    @pyqtSlot(int)
    def _update_reference_display(self, _combo_idx=-1):
        self.reference_details_text.clear()
        ref_step_idx = self.reference_step_combo.currentData(); ref_voice_idx = self.reference_voice_combo.currentData()
        details = "Select a Step and Voice for reference."
        if ref_step_idx is not None and ref_step_idx >= 0 and ref_voice_idx is not None and ref_voice_idx >= 0:
            is_editing_same = (not self.is_new_voice and self.step_index == ref_step_idx and self.voice_index == ref_voice_idx)
            if is_editing_same: details = "Reference is the voice currently being edited.\nDetails reflect saved state."
            else:
                try:
                    voice_data = self.app.track_data["steps"][ref_step_idx]["voices"][ref_voice_idx]
                    details = f"Ref: Step {ref_step_idx+1}, Voice {ref_voice_idx+1}\n------------------------------------\n"
                    details += f"Function: {voice_data.get('synth_function_name', 'N/A')}\nTransition: {'Yes' if voice_data.get('is_transition', False) else 'No'}\nParameters:\n"
                    params = voice_data.get("params", {})
                    if params:
                        for k, v in sorted(params.items()): details += f"  {k}: {v:.4g if isinstance(v, float) else v}\n"
                    else: details += "  (No parameters defined)\n"
                    env_data = voice_data.get("volume_envelope")
                    if env_data and isinstance(env_data, dict):
                        details += f"\nEnvelope Type: {env_data.get('type', 'N/A')}\n  Envelope Params:\n"
                        env_params = env_data.get('params', {})
                        if env_params:
                            for k,v in sorted(env_params.items()): details += f"    {k}: {v:.4g if isinstance(v,float) else v}\n"
                        else: details += "  (No envelope parameters defined)\n"
                    else: details += "\nEnvelope Type: None\n"
                except IndexError: details = "Error: Invalid Step or Voice index for reference."
        elif ref_step_idx is not None and ref_step_idx >= 0: details = "Select a Voice from the selected Step."
        elif self.reference_step_combo.count() > 0 and self.reference_step_combo.itemData(0) == -1: details = "No steps available."
        self.reference_details_text.setPlainText(details)

    def _select_initial_reference_voice(self, voice_index_to_select):
        voice_combo_index = self.reference_voice_combo.findData(voice_index_to_select)
        if voice_combo_index != -1: self.reference_voice_combo.setCurrentIndex(voice_combo_index)
        elif self.reference_voice_combo.count() > 0 and self.reference_voice_combo.itemData(0) != -1: self.reference_voice_combo.setCurrentIndex(0)

    @pyqtSlot()
    def on_synth_function_change(self):
        selected_func = self.synth_func_combo.currentText()
        if not selected_func: return
        is_potentially_transition = selected_func.endswith("_transition")
        self.transition_check.blockSignals(True); self.transition_check.setChecked(is_potentially_transition); self.transition_check.blockSignals(False)
        self.current_voice_data["synth_function_name"] = selected_func
        self.current_voice_data["is_transition"] = is_potentially_transition
        new_defaults = self._get_default_params(selected_func, is_potentially_transition)
        existing_params = self.current_voice_data.get("params", {})
        merged_params = {k: (existing_params[k] if k in existing_params else v) for k, v in new_defaults.items()}
        self.current_voice_data["params"] = merged_params
        self.populate_parameters()

    @pyqtSlot(int)
    def on_transition_toggle(self, state):
        is_transition = bool(state == Qt.Checked)
        self.current_voice_data["is_transition"] = is_transition
        func_name = self.synth_func_combo.currentText()
        new_defaults = self._get_default_params(func_name, is_transition)
        existing_params = self.current_voice_data.get("params", {})
        merged_params = {k: (existing_params[k] if k in existing_params else v) for k, v in new_defaults.items()}
        self.current_voice_data["params"] = merged_params
        self.populate_parameters()

    def _get_param_range_hint(self, param_name):
        name_lower = param_name.lower()
        if any(s in name_lower for s in ['amp', 'gain', 'level', 'depth']): return '(0.0-1.0+)'
        if 'pan' in name_lower: return '(-1 L to 1 R)'
        if any(s in name_lower for s in ['freq', 'frequency', 'rate']): return '(Hz, >0)'
        if 'rq' == name_lower or 'q' == name_lower: return '(>0, ~0.1-20)'
        if any(s in name_lower for s in ['dur', 'attack', 'decay', 'release']): return '(secs, >=0)'
        return ''

    def _get_default_params(self, func_name, is_transition):
        params = {}
        target_func_name = func_name
        if is_transition and not func_name.endswith("_transition"):
            if (func_name + "_transition") in sound_creator.SYNTH_FUNCTIONS: target_func_name = func_name + "_transition"
        elif not is_transition and func_name.endswith("_transition"):
            base_name = func_name.replace("_transition", "")
            if base_name in sound_creator.SYNTH_FUNCTIONS: target_func_name = base_name
        if target_func_name not in sound_creator.SYNTH_FUNCTIONS: return {}
        
        target_func = sound_creator.SYNTH_FUNCTIONS[target_func_name]
        try:
            source_code = inspect.getsource(target_func)
            regex = r"params\.get\(\s*['\"]([^'\"]+)['\"]\s*,\s*(.*?)\s*\)"
            for match in re.finditer(regex, source_code):
                param_name, default_value_str = match.group(1), match.group(2).strip().rstrip(',')
                try:
                    if default_value_str.lower() == 'true': default_value = True
                    elif default_value_str.lower() == 'false': default_value = False
                    elif default_value_str.lower() == 'none': default_value = None
                    elif default_value_str == 'math.pi/2': default_value = math.pi / 2
                    elif default_value_str == 'math.pi': default_value = math.pi
                    else: default_value = ast.literal_eval(default_value_str)
                except Exception: default_value = 0.0 # Fallback
                params[param_name] = default_value
            if 'spatial_angle_modulation' in target_func_name and 'pathShape' not in params and hasattr(sound_creator, 'VALID_SAM_PATHS') and sound_creator.VALID_SAM_PATHS:
                params['pathShape'] = sound_creator.VALID_SAM_PATHS[0]
        except Exception as e: print(f"Error parsing source for '{target_func_name}': {e}")
        return params

    @pyqtSlot()
    def save_voice(self):
        new_synth_params = {}; new_envelope_data = None; error_occurred = False; validation_errors = []
        try: # Synth Params
            for name, data in self.param_widgets.items():
                widget, param_type = data['widget'], data['type']; value = None
                if isinstance(widget, QCheckBox): value = widget.isChecked()
                elif isinstance(widget, QComboBox):
                    value_str = widget.currentText()
                    if name == 'noiseType' and param_type == 'int': value = int(value_str)
                    else: value = value_str # pathShape is str
                elif isinstance(widget, QLineEdit):
                    value_str = widget.text().strip()
                    if not value_str: value = None # Treat empty as None
                    else:
                        try:
                            if param_type == 'int': value = int(value_str)
                            elif param_type == 'float': value = float(value_str.replace(',', '.'))
                            else: value = value_str
                        except ValueError: error_occurred=True; validation_errors.append(f"Invalid {param_type} for '{name}': {value_str}"); widget.setStyleSheet("border: 1px solid red;")
                if value is not None: new_synth_params[name] = value
        except Exception as e: QMessageBox.critical(self, "Error", f"Error gathering synth params:\n{e}"); return

        selected_env_type = self.env_type_combo.currentText() # Envelope Params
        if selected_env_type != ENVELOPE_TYPE_NONE:
            new_env_params = {}
            try:
                for name, data in self.envelope_param_widgets.items():
                    widget, param_type = data['widget'], data['type']; value = None
                    if isinstance(widget, QLineEdit):
                        value_str = widget.text().strip()
                        if not value_str: error_occurred=True; validation_errors.append(f"Env param '{name}' empty."); widget.setStyleSheet("border: 1px solid red;")
                        else:
                            try:
                                if param_type == 'float': value = float(value_str.replace(',', '.'))
                                elif param_type == 'int': value = int(value_str)
                                else: value = value_str
                                if 'amp' in name.lower() and not (0.0 <= value <= 1.0): pass # Warning, not error
                            except ValueError: error_occurred=True; validation_errors.append(f"Invalid env {param_type} for '{name}': {value_str}"); widget.setStyleSheet("border: 1px solid red;")
                    if value is not None: new_env_params[name] = value
                if not any(f"env param '{n}'" in e for n in self.envelope_param_widgets for e in validation_errors):
                    new_envelope_data = {"type": selected_env_type, "params": new_env_params}
            except Exception as e: QMessageBox.critical(self, "Error", f"Error gathering env params:\n{e}"); return
        
        if error_occurred: QMessageBox.critical(self, "Parameter Error", "Please correct highlighted fields:\n\n" + "\n".join(validation_errors)); return

        final_voice_data = {
             "synth_function_name": self.synth_func_combo.currentText(),
             "is_transition": self.transition_check.isChecked(),
             "params": new_synth_params, "volume_envelope": new_envelope_data
        }
        try:
            target_step = self.app.track_data["steps"][self.step_index]
            if "voices" not in target_step: target_step["voices"] = []
            if self.is_new_voice: target_step["voices"].append(final_voice_data)
            elif 0 <= self.voice_index < len(target_step["voices"]): target_step["voices"][self.voice_index] = final_voice_data
            else: QMessageBox.critical(self.app, "Error", "Voice index out of bounds."); self.reject(); return
            self.accept()
        except IndexError: QMessageBox.critical(self.app, "Error", "Failed to save voice (step index issue)."); self.reject()
        except Exception as e: QMessageBox.critical(self.app, "Error", f"Failed to save voice data:\n{e}"); self.reject()


# --- Run the Application ---
if __name__ == "__main__":
    if not hasattr(sound_creator, 'SYNTH_FUNCTIONS'):
        print("Critical Error: sound_creator.SYNTH_FUNCTIONS not found. Ensure sound_creator.py is correct and accessible.")
        # Consider a QMessageBox here for GUI users before sys.exit
        mbox = QMessageBox()
        mbox.setIcon(QMessageBox.Critical)
        mbox.setText("Critical Error: Sound creator module is missing vital components (SYNTH_FUNCTIONS).\nThe application cannot start.\nPlease check the 'sound_creator.py' file.")
        mbox.setWindowTitle("Application Startup Error")
        mbox.setStandardButtons(QMessageBox.Ok)
        mbox.exec_()
        sys.exit(1) # Exit if critical component is missing

    app = QApplication(sys.argv)
    app.setStyle('Fusion') # Optional styling
    window = TrackEditorApp()
    window.show()
    sys.exit(app.exec_())
