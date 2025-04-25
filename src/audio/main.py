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
    QDialog, QScrollArea, QSizePolicy, QInputDialog, QSpacerItem
)
from PyQt5.QtCore import Qt, pyqtSlot, QSize, QTimer # Added QTimer
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QFont

# --- Constants ---
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_CROSSFADE = 1.0
MAX_VOICES_PER_STEP = 16
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
        self.resize(1100, 750)

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
        # self.generate_button.setMinimumHeight(35)
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
        self.steps_tree.setColumnCount(2)
        self.steps_tree.setHeaderLabels(["Duration (s)", "# Voices"])
        self.steps_tree.setColumnWidth(0, 100)
        self.steps_tree.setColumnWidth(1, 80)
        self.steps_tree.header().setStretchLastSection(True)
        self.steps_tree.setSelectionMode(QTreeWidget.SingleSelection)
        self.steps_tree.itemSelectionChanged.connect(self.on_step_select)
        steps_groupbox_layout.addWidget(self.steps_tree, 1)

        steps_button_layout = QHBoxLayout()
        self.add_step_button = QPushButton("Add Step")
        # --- NEW: Duplicate Step Button ---
        self.duplicate_step_button = QPushButton("Duplicate Step")
        self.remove_step_button = QPushButton("Remove Step")
        self.edit_duration_button = QPushButton("Edit Duration")
        self.move_step_up_button = QPushButton("Move Up")
        self.move_step_down_button = QPushButton("Move Down")
        self.add_step_button.clicked.connect(self.add_step)
        # --- NEW: Connect Duplicate Button ---
        self.duplicate_step_button.clicked.connect(self.duplicate_step)
        self.remove_step_button.clicked.connect(self.remove_step)
        self.edit_duration_button.clicked.connect(self.edit_step_duration)
        self.move_step_up_button.clicked.connect(lambda: self.move_step(-1))
        self.move_step_down_button.clicked.connect(lambda: self.move_step(1))
        steps_button_layout.addWidget(self.add_step_button)
        # --- NEW: Add Duplicate Button to Layout ---
        steps_button_layout.addWidget(self.duplicate_step_button)
        steps_button_layout.addWidget(self.remove_step_button)
        steps_button_layout.addWidget(self.edit_duration_button)
        steps_button_layout.addWidget(self.move_step_up_button)
        steps_button_layout.addWidget(self.move_step_down_button)
        steps_button_layout.addStretch(1)
        steps_groupbox_layout.addLayout(steps_button_layout)

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
        # self.voices_tree.header().setStretchLastSection(False) # Prevent last column stretch
        # self.voices_tree.header().setSectionResizeMode(0, QHeaderView.Stretch) # Stretch first column instead
        self.voices_tree.header().setStretchLastSection(True) # Stretch last column
        self.voices_tree.setSelectionMode(QTreeWidget.SingleSelection)
        self.voices_tree.itemSelectionChanged.connect(self.on_voice_select)
        self.voices_tree.itemDoubleClicked.connect(self.edit_voice)
        voices_groupbox_layout.addWidget(self.voices_tree, 1)

        voices_button_layout = QHBoxLayout()
        self.add_voice_button = QPushButton("Add Voice")
        self.edit_voice_button = QPushButton("Edit Voice")
        self.remove_voice_button = QPushButton("Remove Voice")
        self.add_voice_button.clicked.connect(self.add_voice)
        self.edit_voice_button.clicked.connect(self.edit_voice)
        self.remove_voice_button.clicked.connect(self.remove_voice)
        voices_button_layout.addWidget(self.add_voice_button)
        voices_button_layout.addWidget(self.edit_voice_button)
        voices_button_layout.addWidget(self.remove_voice_button)
        voices_button_layout.addStretch(1)
        voices_groupbox_layout.addLayout(voices_button_layout)

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
        self.voice_details_text.setFont(QFont("Consolas", 9)) # Monospaced font
        self.voice_details_text.setLineWrapMode(QTextEdit.WidgetWidth) # Equivalent to wrap=tk.WORD
        voice_details_groupbox_layout.addWidget(self.voice_details_text)

        # Set Splitter Sizes (initial proportions)
        main_splitter.setSizes([300, 700]) # Adjust as needed
        right_splitter.setSizes([500, 200]) # Adjust as needed

    # --- Internal Data Handling ---
    def _update_global_settings_from_ui(self):
        """Reads global settings from UI elements and updates self.track_data."""
        try:
            sr_str = self.sr_entry.text()
            cf_str = self.cf_entry.text()
            outfile = self.outfile_entry.text().strip()

            if not sr_str: raise ValueError("Sample rate cannot be empty.")
            sr = int(sr_str)
            if sr <= 0: raise ValueError("Sample rate must be positive.")
            self.track_data["global_settings"]["sample_rate"] = sr

            if not cf_str: raise ValueError("Crossfade duration cannot be empty.")
            # Use locale-independent conversion for float
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
        """Updates UI elements with values from self.track_data['global_settings']."""
        settings = self.track_data.get("global_settings", {})
        self.sr_entry.setText(str(settings.get("sample_rate", DEFAULT_SAMPLE_RATE)))
        self.cf_entry.setText(str(settings.get("crossfade_duration", DEFAULT_CROSSFADE)))
        self.outfile_entry.setText(settings.get("output_filename", "my_track.wav"))

    # --- UI Refresh Functions ---
    def refresh_steps_tree(self):
        """Refreshes the steps treeview based on self.track_data."""
        current_selection = self.steps_tree.currentItem()
        current_index = self.steps_tree.indexOfTopLevelItem(current_selection) if current_selection else -1

        self.steps_tree.clear()
        for i, step in enumerate(self.track_data.get("steps", [])):
            duration = step.get("duration", 0.0)
            num_voices = len(step.get("voices", []))
            item = QTreeWidgetItem(self.steps_tree)
            item.setText(0, f"{duration:.2f}")
            item.setText(1, str(num_voices))
            item.setData(0, Qt.UserRole, i) # Store original index

        if current_index != -1 and current_index < self.steps_tree.topLevelItemCount():
            new_item_to_select = self.steps_tree.topLevelItem(current_index)
            self.steps_tree.setCurrentItem(new_item_to_select)
            self.steps_tree.scrollToItem(new_item_to_select, QTreeWidget.PositionAtCenter)
        # Important: Refreshing steps inherently requires refreshing voices
        self.refresh_voices_tree()


    def refresh_voices_tree(self):
        """Refreshes the voices treeview based on the selected step."""
        current_voice_selection = self.voices_tree.currentItem()
        current_voice_index = self.voices_tree.indexOfTopLevelItem(current_voice_selection) if current_voice_selection else -1

        self.voices_tree.clear() # Clear previous items first
        self.clear_voice_details() # Clear details pane when voices are refreshed

        selected_step_idx = self.get_selected_step_index()

        if selected_step_idx is None:
            self.voices_groupbox.setTitle("Voices for Selected Step")
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

                # Heuristic to find primary frequency (same logic as Tkinter version)
                carrier_freq = 'N/A'
                if is_transition:
                    freq_keys = [k for k in params if k.startswith('start') and ('Freq' in k or 'Frequency' in k)]
                    if freq_keys: carrier_freq = params.get(freq_keys[0])
                    else:
                        freq_keys = [k for k in params if ('Freq' in k or 'Frequency' in k) and not k.startswith('end')]
                        carrier_freq = params.get(freq_keys[0]) if freq_keys else 'N/A'
                else:
                    freq_keys = [k for k in params if ('Freq' in k or 'Frequency' in k) and not k.startswith(('start','end'))]
                    carrier_freq = params.get(freq_keys[0]) if freq_keys else 'N/A'

                # Format frequency
                try:
                    if carrier_freq is not None and carrier_freq != 'N/A': carrier_freq_str = f"{float(carrier_freq):.2f}"
                    else: carrier_freq_str = 'N/A'
                except (ValueError, TypeError): carrier_freq_str = str(carrier_freq)

                transition_str = "Yes" if is_transition else "No"
                item = QTreeWidgetItem(self.voices_tree)
                item.setText(0, func_name)
                item.setText(1, carrier_freq_str)
                item.setText(2, transition_str)
                item.setData(0, Qt.UserRole, i) # Store original index

        except IndexError:
            print(f"Error: Selected step index {selected_step_idx} out of range.")
            self.voices_groupbox.setTitle("Voices for Selected Step")
            self.clear_voice_details()
            return
        except Exception as e:
            print(f"Error refreshing voices tree: {e}")
            traceback.print_exc()
            self.clear_voice_details()
            return

        # Restore selection if possible
        if current_voice_index != -1 and current_voice_index < self.voices_tree.topLevelItemCount():
            new_item_to_select = self.voices_tree.topLevelItem(current_voice_index)
            self.voices_tree.setCurrentItem(new_item_to_select)
            self.voices_tree.scrollToItem(new_item_to_select, QTreeWidget.PositionAtCenter)
            self.update_voice_details() # Update details for restored selection
        else:
             self.clear_voice_details() # Clear if no selection or selection invalid


    def clear_voice_details(self):
        """Clears the voice details text area."""
        self.voice_details_text.clear()
        self.voice_details_groupbox.setTitle("Selected Voice Details")

    def update_voice_details(self):
        """Updates the details text area with selected voice parameters and envelope."""
        self.clear_voice_details()
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
            details += "Parameters:\n"
            if params:
                for key, value in sorted(params.items()):
                    if isinstance(value, float): details += f"  {key}: {value:.4g}\n"
                    else: details += f"  {key}: {value}\n"
            else: details += "  (No parameters defined)\n"

            # --- Display Envelope Details ---
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
                else:
                    details += "  (No envelope parameters defined)\n"
            else:
                details += "\nEnvelope Type: None\n"
            # --- End Envelope Details ---

            self.voice_details_text.setPlainText(details)
        except (IndexError, KeyError) as e:
            print(f"Error accessing voice data for details view: Step {selected_step_idx}, Voice {selected_voice_idx}. Error: {e}")
            self.clear_voice_details()
        except Exception as e:
            print(f"Unexpected error updating voice details: {e}")
            traceback.print_exc()
            self.clear_voice_details()

    # --- Event Handlers (Slots) ---
    @pyqtSlot()
    def on_step_select(self):
        # Selection change automatically triggers refresh now
        self.refresh_voices_tree()

    @pyqtSlot()
    def on_voice_select(self):
         # Selection change automatically triggers update now
        self.update_voice_details()

    # --- Action Methods (Slots for Buttons) ---
    @pyqtSlot()
    def load_json(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Track JSON", "", "JSON files (*.json);;All files (*.*)")
        if not filepath: return
        try:
            loaded_data = sound_creator.load_track_from_json(filepath) # Assuming this function handles errors and returns None/dict
            if loaded_data and isinstance(loaded_data.get("steps"), list) and isinstance(loaded_data.get("global_settings"), dict):
                self.track_data = loaded_data
                self.current_json_path = filepath
                self.setWindowTitle(f"Binaural Track Editor (PyQt5) - {os.path.basename(filepath)}")
                self._update_ui_from_global_settings()
                self.refresh_steps_tree() # This will also refresh voices
                QMessageBox.information(self, "Load Success", f"Track loaded from\n{filepath}")
            elif loaded_data is not None: # sound_creator returned something, but invalid structure
                QMessageBox.critical(self, "Load Error", "Invalid JSON structure.")
            # If loaded_data is None, assume sound_creator already showed an error via printing/logging
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load or parse JSON file:\n{e}")
            traceback.print_exc()

    @pyqtSlot()
    def save_json(self):
        if not self.current_json_path:
            self.save_json_as()
        else:
            if not self._update_global_settings_from_ui(): return
            try:
                # Assuming sound_creator.save_track_to_json returns True/False or raises Exception
                if sound_creator.save_track_to_json(self.track_data, self.current_json_path):
                    QMessageBox.information(self, "Save Success", f"Track saved to\n{self.current_json_path}")
                    self.setWindowTitle(f"Binaural Track Editor (PyQt5) - {os.path.basename(self.current_json_path)}")
                # else: # Optionally handle False return if sound_creator doesn't raise Exception
                #     QMessageBox.warning(self, "Save Warning", f"Track data might not have been saved correctly to\n{self.current_json_path}")
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
            # else: # Optional
            #     QMessageBox.warning(self, "Save Warning", f"Track data might not have been saved correctly to\n{filepath}")
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
    def add_step(self):
        new_step = {"duration": 10.0, "voices": []}
        # If a step is selected, insert after it, otherwise append
        selected_index = self.get_selected_step_index()
        insert_index = selected_index + 1 if selected_index is not None else len(self.track_data["steps"])

        self.track_data["steps"].insert(insert_index, new_step)
        self.refresh_steps_tree()
        # Select the newly added step
        if 0 <= insert_index < self.steps_tree.topLevelItemCount():
            new_item = self.steps_tree.topLevelItem(insert_index)
            self.steps_tree.setCurrentItem(new_item)
            self.steps_tree.scrollToItem(new_item, QTreeWidget.PositionAtCenter)
        # refresh_steps_tree already calls refresh_voices_tree

    # --- NEW: Duplicate Step Method ---
    @pyqtSlot()
    def duplicate_step(self):
        """Duplicates the selected step and inserts it after the original."""
        selected_index = self.get_selected_step_index()
        if selected_index is None:
            QMessageBox.warning(self, "Duplicate Step", "Please select a step to duplicate.")
            return

        try:
            original_step_data = self.track_data["steps"][selected_index]
            # Use deepcopy to ensure nested lists/dicts (like voices and params) are independent
            duplicated_step_data = copy.deepcopy(original_step_data)
            insert_index = selected_index + 1

            self.track_data["steps"].insert(insert_index, duplicated_step_data)
            self.refresh_steps_tree() # Refresh the tree

            # Select the newly duplicated step
            if 0 <= insert_index < self.steps_tree.topLevelItemCount():
                new_item = self.steps_tree.topLevelItem(insert_index)
                self.steps_tree.setCurrentItem(new_item)
                self.steps_tree.scrollToItem(new_item, QTreeWidget.PositionAtCenter)

        except IndexError:
             QMessageBox.critical(self, "Error", "Failed to duplicate step (index out of range).")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to duplicate step:\n{e}")
            traceback.print_exc()

    @pyqtSlot()
    def remove_step(self):
        selected_index = self.get_selected_step_index()
        if selected_index is None:
            QMessageBox.warning(self, "Remove Step", "Please select a step to remove.")
            return

        reply = QMessageBox.question(self, "Confirm Remove", f"Are you sure you want to remove Step {selected_index + 1}?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            try:
                if 0 <= selected_index < len(self.track_data["steps"]):
                    del self.track_data["steps"][selected_index]
                    self.refresh_steps_tree() # Will refresh voices too
                else:
                    QMessageBox.critical(self, "Error", "Failed to remove step (index out of range).")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to remove step:\n{e}")
                traceback.print_exc()

    @pyqtSlot()
    def edit_step_duration(self):
        selected_index = self.get_selected_step_index()
        if selected_index is None:
            QMessageBox.warning(self, "Edit Duration", "Please select a step to edit.")
            return

        try:
            current_duration = float(self.track_data["steps"][selected_index].get("duration", 0.0))
        except (IndexError, ValueError, TypeError) as e:
            QMessageBox.critical(self, "Error", f"Failed to get current duration (index {selected_index}):\n{e}")
            return

        new_duration, ok = QInputDialog.getDouble(self, f"Edit Step {selected_index + 1} Duration", "New Duration (s):", current_duration, 0.001, 99999.0, 3) # min, max, decimals

        if ok and new_duration is not None:
             if new_duration <= 0:
                 QMessageBox.warning(self, "Invalid Input", "Duration must be positive.")
                 return
             try:
                 self.track_data["steps"][selected_index]["duration"] = new_duration
                 self.refresh_steps_tree()
                 # Re-select the edited item (refresh might change item objects)
                 if selected_index < self.steps_tree.topLevelItemCount():
                     edited_item = self.steps_tree.topLevelItem(selected_index)
                     self.steps_tree.setCurrentItem(edited_item) # Select it again
                     self.steps_tree.scrollToItem(edited_item, QTreeWidget.PositionAtCenter)

             except IndexError:
                 QMessageBox.critical(self, "Error", "Failed to set duration (index out of range after edit).")
             except Exception as e:
                 QMessageBox.critical(self, "Error", f"Failed to set duration:\n{e}")


    @pyqtSlot()
    def move_step(self, direction):
        selected_index = self.get_selected_step_index()
        if selected_index is None:
            QMessageBox.warning(self, "Move Step", "Please select a step to move.")
            return

        num_steps = len(self.track_data["steps"])
        new_index = selected_index + direction

        if 0 <= new_index < num_steps:
            try:
                steps = self.track_data["steps"]
                steps[selected_index], steps[new_index] = steps[new_index], steps[selected_index]

                self.refresh_steps_tree()
                # Select the moved item at its new position
                if new_index < self.steps_tree.topLevelItemCount():
                    moved_item = self.steps_tree.topLevelItem(new_index)
                    self.steps_tree.setCurrentItem(moved_item)
                    self.steps_tree.scrollToItem(moved_item, QTreeWidget.PositionAtCenter)
                # refresh_steps_tree calls refresh_voices_tree

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to move step:\n{e}")
                traceback.print_exc()

    @pyqtSlot()
    def add_voice(self):
        selected_step_index = self.get_selected_step_index()
        if selected_step_index is None:
            QMessageBox.warning(self, "Add Voice", "Please select a step first.")
            return

        try:
            current_voices = self.track_data["steps"][selected_step_index].get("voices", [])
            if len(current_voices) >= MAX_VOICES_PER_STEP:
                QMessageBox.warning(self, "Add Voice", f"Maximum voices per step ({MAX_VOICES_PER_STEP}) reached.")
                return
        except IndexError:
            QMessageBox.critical(self, "Error", "Cannot add voice (selected step index out of range).")
            return
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error checking voice count:\n{e}")
            return

        dialog = VoiceEditorDialog(parent=self, app_ref=self, step_index=selected_step_index, voice_index=None)
        if dialog.exec_() == QDialog.Accepted: # Check if Save was clicked
             self.refresh_steps_tree() # Update step voice count
             # Select the parent step and the new voice
             if selected_step_index < self.steps_tree.topLevelItemCount():
                 step_item = self.steps_tree.topLevelItem(selected_step_index)
                 self.steps_tree.setCurrentItem(step_item) # Select step first (triggers voice refresh)
                 self.refresh_voices_tree() # Explicit refresh might be needed depending on timing
                 # Now select the last voice in the refreshed list
                 voice_count = self.voices_tree.topLevelItemCount()
                 if voice_count > 0:
                     new_voice_item = self.voices_tree.topLevelItem(voice_count - 1)
                     self.voices_tree.setCurrentItem(new_voice_item)
                     self.voices_tree.scrollToItem(new_voice_item, QTreeWidget.PositionAtCenter)
                     self.update_voice_details() # Update details view

    @pyqtSlot()
    def edit_voice(self):
        selected_step_index = self.get_selected_step_index()
        selected_voice_index = self.get_selected_voice_index()

        if selected_step_index is None or selected_voice_index is None:
            QMessageBox.warning(self, "Edit Voice", "Please select a step and a voice to edit.")
            return

        dialog = VoiceEditorDialog(parent=self, app_ref=self, step_index=selected_step_index, voice_index=selected_voice_index)
        if dialog.exec_() == QDialog.Accepted:
            # Refresh trees and restore selection
            self.refresh_steps_tree() # Update step voice count potentially
            if selected_step_index < self.steps_tree.topLevelItemCount():
                step_item = self.steps_tree.topLevelItem(selected_step_index)
                self.steps_tree.setCurrentItem(step_item)
                self.refresh_voices_tree() # Refresh voices for the selected step
                if selected_voice_index < self.voices_tree.topLevelItemCount():
                    voice_item = self.voices_tree.topLevelItem(selected_voice_index)
                    self.voices_tree.setCurrentItem(voice_item)
                    self.voices_tree.scrollToItem(voice_item, QTreeWidget.PositionAtCenter)
                    self.update_voice_details()


    @pyqtSlot()
    def remove_voice(self):
        selected_step_index = self.get_selected_step_index()
        selected_voice_index = self.get_selected_voice_index()

        if selected_step_index is None or selected_voice_index is None:
            QMessageBox.warning(self, "Remove Voice", "Please select a step and a voice to remove.")
            return

        try:
            voices_list = self.track_data["steps"][selected_step_index].get("voices")
            if not voices_list or not (0 <= selected_voice_index < len(voices_list)):
                QMessageBox.critical(self, "Error", "Selected voice index is out of bounds.")
                return

            voice_name = voices_list[selected_voice_index].get("synth_function_name", "N/A")
            reply = QMessageBox.question(self, "Confirm Remove", f"Remove Voice {selected_voice_index + 1} ({voice_name}) from Step {selected_step_index + 1}?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.Yes:
                del voices_list[selected_voice_index]
                # Refresh and maintain step selection
                self.refresh_steps_tree()
                if selected_step_index < self.steps_tree.topLevelItemCount():
                    step_item = self.steps_tree.topLevelItem(selected_step_index)
                    self.steps_tree.setCurrentItem(step_item)
                    # refresh_steps_tree calls refresh_voices_tree, which clears voice selection

        except IndexError:
            QMessageBox.critical(self, "Error", "Failed to remove voice (step index out of range).")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to remove voice:\n{e}")
            traceback.print_exc()

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

        print("\nStarting WAV generation process...")
        self.generate_button.setEnabled(False)
        self.generate_button.setText("Generating...")
        QApplication.processEvents() # Allow UI to update

        try:
            # *** Crossfade Handling Assumption ***
            # It is assumed that the sound_creator.generate_wav function will internally
            # use the 'crossfade_duration' value from track_data["global_settings"]
            # to apply crossfades between the generated audio segments for each step.
            # This UI code only ensures the setting is correctly passed.
            success = sound_creator.generate_wav(self.track_data, output_filename)
            if success:
                QMessageBox.information(self, "Generation Complete", f"WAV file generated successfully:\n{output_filename}")
            else:
                QMessageBox.critical(self, "Generation Failed", "Error during WAV generation. Check console/logs.")
        except Exception as e:
            QMessageBox.critical(self, "Generation Error", f"An unexpected error occurred during WAV generation:\n{e}")
            traceback.print_exc()
        finally:
            self.generate_button.setEnabled(True)
            self.generate_button.setText("Generate WAV")

    # --- Utility Methods ---
    def get_selected_step_index(self):
        """Gets the original index of the selected step."""
        selected_items = self.steps_tree.selectedItems()
        if selected_items:
            # Retrieve the original index stored in UserRole
            data = selected_items[0].data(0, Qt.UserRole)
            if data is not None:
                return int(data)
        return None

    def get_selected_voice_index(self):
        """Gets the original index of the selected voice."""
        selected_items = self.voices_tree.selectedItems()
        if selected_items:
            # Retrieve the original index stored in UserRole
            data = selected_items[0].data(0, Qt.UserRole)
            if data is not None:
                return int(data)
        return None

    def closeEvent(self, event):
        # Optional: Add confirmation dialog before closing if needed
        # reply = QMessageBox.question(self, 'Confirm Exit',
        #                                 "Are you sure you want to exit?",
        #                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        # if reply == QMessageBox.Yes:
        #     event.accept()
        # else:
        #     event.ignore()
        super().closeEvent(event)


# --- Voice Editor Dialog Class ---
class VoiceEditorDialog(QDialog):
    DEFAULT_WIDTH = 900
    DEFAULT_HEIGHT = 700 # Increased height slightly for reference controls

    def __init__(self, parent, app_ref, step_index, voice_index=None):
        super().__init__(parent)
        self.app = app_ref # Reference to the main TrackEditorApp instance
        self.step_index = step_index
        self.voice_index = voice_index
        self.is_new_voice = (voice_index is None)

        # Validators (use from parent or create new)
        self.double_validator_non_negative = QDoubleValidator(0.0, 999999.0, 6, self)
        self.double_validator_zero_to_one = QDoubleValidator(0.0, 1.0, 6, self) # For Amplitudes
        self.double_validator = QDoubleValidator(-999999.0, 999999.0, 6, self)
        self.int_validator = QIntValidator(-999999, 999999, self)

        self.setWindowTitle(f"{'Add' if self.is_new_voice else 'Edit'} Voice for Step {step_index + 1}")
        self.resize(self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT)
        self.setMinimumSize(700, 600) # Increased min height
        self.setModal(True) # Act like a modal dialog

        # --- Widgets Storage ---
        self.param_widgets = {} # For synth function parameters
        self.envelope_param_widgets = {} # For envelope parameters

        # --- Data ---
        self._load_initial_data() # Loads self.current_voice_data

        # --- Widgets ---
        self._setup_ui() # Creates UI elements
        self.populate_parameters() # Populates synth parameters based on loaded data
        self._populate_envelope_controls() # Populate envelope controls based on loaded data

        # --- NEW: Populate Reference Selector Combos ---
        self._populate_reference_step_combo()
        # Initial population of voice combo and details will be triggered by the step combo signal

        # --- NEW: Set initial reference selection (optional, try matching main window) ---
        initial_ref_step = self.app.get_selected_step_index()
        initial_ref_voice = self.app.get_selected_voice_index()
        if initial_ref_step is not None:
             step_combo_index = self.reference_step_combo.findData(initial_ref_step)
             if step_combo_index != -1:
                 self.reference_step_combo.setCurrentIndex(step_combo_index)
                 # The step combo's signal will populate the voice combo
                 # Now try to select the voice
                 # Need a slight delay for the voice combo to populate? Use QTimer.
                 if initial_ref_voice is not None:
                     QTimer.singleShot(50, lambda: self._select_initial_reference_voice(initial_ref_voice))
             else:
                 # Default to first step if main selection not found or invalid
                 if self.reference_step_combo.count() > 0:
                     self.reference_step_combo.setCurrentIndex(0)
        elif self.reference_step_combo.count() > 0:
             # Default to first step if nothing selected in main window
             self.reference_step_combo.setCurrentIndex(0)


    def _load_initial_data(self):
        """Loads or creates the initial voice data dictionary for the editor."""
        if self.is_new_voice:
            available_funcs = sorted(sound_creator.SYNTH_FUNCTIONS.keys())
            first_func_name = available_funcs[0] if available_funcs else ""
            # Determine initial transition state based on name heuristic
            is_trans = first_func_name.endswith("_transition")
            default_params = self._get_default_params(first_func_name, is_trans)
            self.current_voice_data = {
                "synth_function_name": first_func_name,
                "is_transition": is_trans,
                "params": default_params,
                "volume_envelope": None # Default to no envelope
            }
        else:
            try:
                original_voice = self.app.track_data["steps"][self.step_index]["voices"][self.voice_index]
                # Deep copy to avoid modifying the original data until save
                self.current_voice_data = copy.deepcopy(original_voice)
                # Ensure essential keys exist
                if "params" not in self.current_voice_data:
                    self.current_voice_data["params"] = {}
                if "volume_envelope" not in self.current_voice_data:
                    self.current_voice_data["volume_envelope"] = None # Ensure envelope key exists
                # Add is_transition if missing, inferring from name
                if "is_transition" not in self.current_voice_data:
                    self.current_voice_data["is_transition"] = self.current_voice_data.get("synth_function_name","").endswith("_transition")

            except (IndexError, KeyError) as e:
                QMessageBox.critical(self.parent(), "Error", f"Could not load voice data for editing:\n{e}")
                # Let's make current_voice_data empty to prevent further errors in UI setup
                self.current_voice_data = {
                    "params": {},
                    "synth_function_name": "Error",
                    "is_transition": False,
                    "volume_envelope": None
                }
                # Schedule reject after init finishes? Better to handle gracefully.
                # Using QTimer to call reject after the constructor finishes
                QTimer.singleShot(0, self.reject)


    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        # --- Top Frame ---
        top_frame = QWidget()
        top_layout = QHBoxLayout(top_frame)
        top_layout.setContentsMargins(0,0,0,0)
        top_layout.addWidget(QLabel("Synth Function:"))
        self.synth_func_combo = QComboBox()
        func_names = sorted(sound_creator.SYNTH_FUNCTIONS.keys())
        self.synth_func_combo.addItems(func_names)
        current_func_name = self.current_voice_data.get("synth_function_name", "")
        if current_func_name in func_names:
            self.synth_func_combo.setCurrentText(current_func_name)
        self.synth_func_combo.currentIndexChanged.connect(self.on_synth_function_change)
        top_layout.addWidget(self.synth_func_combo, 1) # Stretch combobox

        self.transition_check = QCheckBox("Is Transition?")
        self.transition_check.setChecked(self.current_voice_data.get("is_transition", False))
        self.transition_check.stateChanged.connect(self.on_transition_toggle)
        top_layout.addWidget(self.transition_check)
        main_layout.addWidget(top_frame)

        # --- Splitter for Params and Reference ---
        h_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(h_splitter, 1) # Stretch splitter

        # --- Parameter Editing Frame (Left) ---
        self.params_groupbox = QGroupBox("Synth Parameters (Editing)")
        params_groupbox_layout = QVBoxLayout(self.params_groupbox)
        self.params_scroll_area = QScrollArea()
        self.params_scroll_area.setWidgetResizable(True)
        self.params_scroll_content = QWidget() # Widget to hold the actual parameters layout
        self.params_scroll_layout = QVBoxLayout(self.params_scroll_content) # Layout for parameters
        self.params_scroll_layout.setAlignment(Qt.AlignTop) # Parameters align top
        self.params_scroll_area.setWidget(self.params_scroll_content)
        params_groupbox_layout.addWidget(self.params_scroll_area)
        h_splitter.addWidget(self.params_groupbox)


        # --- Reference Details Frame (Right) ---
        # --- MODIFIED: Reference Pane ---
        reference_groupbox = QGroupBox("Select Voice for Reference") # New Title
        reference_layout = QVBoxLayout(reference_groupbox)

        # --- NEW: Reference Selection Controls ---
        ref_select_layout = QHBoxLayout()
        ref_select_layout.addWidget(QLabel("Step:"))
        self.reference_step_combo = QComboBox()
        self.reference_step_combo.setMinimumWidth(100)
        self.reference_step_combo.currentIndexChanged.connect(self._update_reference_voice_combo)
        ref_select_layout.addWidget(self.reference_step_combo)

        ref_select_layout.addWidget(QLabel("Voice:"))
        self.reference_voice_combo = QComboBox()
        self.reference_voice_combo.setMinimumWidth(150)
        self.reference_voice_combo.currentIndexChanged.connect(self._update_reference_display) # Connect to update display
        ref_select_layout.addWidget(self.reference_voice_combo, 1) # Stretch voice combo

        reference_layout.addLayout(ref_select_layout) # Add selector layout first

        # Reference Details Display
        self.reference_details_text = QTextEdit()
        self.reference_details_text.setReadOnly(True)
        self.reference_details_text.setFont(QFont("Consolas", 9))
        reference_layout.addWidget(self.reference_details_text, 1) # Give text area stretch factor

        h_splitter.addWidget(reference_groupbox)
        # --- End Reference Pane Modifications ---

        h_splitter.setSizes([500, 350]) # Initial sizes

        # --- Envelope Frame ---
        self.env_groupbox = QGroupBox("Volume Envelope")
        env_layout = QVBoxLayout(self.env_groupbox)

        # Envelope Type Selection
        env_type_layout = QHBoxLayout()
        env_type_layout.addWidget(QLabel("Type:"))
        self.env_type_combo = QComboBox()
        self.env_type_combo.addItems(SUPPORTED_ENVELOPE_TYPES)
        self.env_type_combo.currentIndexChanged.connect(self._on_envelope_type_change)
        env_type_layout.addWidget(self.env_type_combo)
        env_type_layout.addStretch(1)
        env_layout.addLayout(env_type_layout)

        # Envelope Parameter Area (Dynamically populated)
        self.env_params_widget = QWidget() # Container for dynamic params
        self.env_params_layout = QGridLayout(self.env_params_widget) # Use Grid for params
        self.env_params_layout.setContentsMargins(10, 5, 5, 5) # Indent params slightly
        self.env_params_layout.setAlignment(Qt.AlignTop)
        env_layout.addWidget(self.env_params_widget)
        env_layout.addStretch(1) # Push params to top if space allows

        main_layout.addWidget(self.env_groupbox)

        # --- Button Frame ---
        button_frame = QWidget()
        button_layout = QHBoxLayout(button_frame)
        button_layout.addStretch(1)
        self.cancel_button = QPushButton("Cancel")
        self.save_button = QPushButton("Save Voice")
        self.save_button.setStyleSheet("QPushButton { background-color: #0078D7; color: white; padding: 6px; font-weight: bold; border-radius: 3px; } QPushButton:hover { background-color: #005A9E; } QPushButton:pressed { background-color: #003C6A; }")
        self.cancel_button.clicked.connect(self.reject) # QDialog convention
        self.save_button.clicked.connect(self.save_voice)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.save_button)
        main_layout.addWidget(button_frame)

    def _clear_layout(self, layout):
        """Removes all widgets from a layout."""
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    # Recursively clear sub-layouts
                    sub_layout = item.layout()
                    if sub_layout is not None:
                        self._clear_layout(sub_layout)

    def populate_parameters(self):
        """Populates synth parameter widgets based on the selected synth function."""
        # Clear existing parameter widgets
        self._clear_layout(self.params_scroll_layout)
        self.param_widgets = {} # Reset storage

        func_name = self.synth_func_combo.currentText()
        is_transition = self.transition_check.isChecked()
        print(f"Populating parameters for: Function='{func_name}', Transition={is_transition}")

        default_params = self._get_default_params(func_name, is_transition)

        if not default_params:
            warning_label = QLabel(f"Warning: Could not determine parameters for '{func_name}'.\nCheck console/source parsing.")
            warning_label.setStyleSheet("color: red;")
            self.params_scroll_layout.addWidget(warning_label)
            return

        current_saved_params = self.current_voice_data.get("params", {})
        # Use default_params as the base, update with currently saved values for this voice
        params_to_display = default_params.copy()
        # Only update with keys that actually exist in the defaults for the *current* function/transition state
        for key, value in current_saved_params.items():
            if key in params_to_display:
                 params_to_display[key] = value # Keep existing valid value


        processed_end_params = set()
        widgets_created_count = 0

        # --- Determine Transition Pairs ---
        transition_pairs = {} # base_name -> {'start': start_name, 'end': end_name}
        if is_transition:
            for name in default_params.keys():
                if name.startswith('start'):
                    base_name = name[len('start'):]
                    end_name = 'end' + base_name
                    if end_name in default_params:
                        transition_pairs[base_name] = {'start': name, 'end': end_name}
                        processed_end_params.add(end_name) # Mark end param as handled by its start pair
                elif name.startswith('end'):
                    # If start wasn't found first, ensure end param is marked if pair exists
                    base_name = name[len('end'):]
                    start_name = 'start' + base_name
                    if start_name in default_params and base_name not in transition_pairs:
                        processed_end_params.add(name)

        # --- Create Widgets ---
        param_names_sorted = sorted(default_params.keys())

        for name in param_names_sorted:
            if name in processed_end_params:
                print(f"            --> Skipping already processed end parameter: '{name}'")
                continue

            default_value = default_params[name]
            current_value = params_to_display.get(name, default_value) # Value to display initially
            print(f"            Processing parameter: '{name}' (Default: {default_value}, Current: {current_value})")

            # Determine if this is the start of a transition pair
            base_name_for_pair = name[len('start'):] if is_transition and name.startswith('start') else None
            is_pair_start = base_name_for_pair is not None and base_name_for_pair in transition_pairs

            # --- Create Frame for Row ---
            frame = QWidget()
            row_layout = QGridLayout(frame) # Use Grid for better alignment
            row_layout.setContentsMargins(2,2,2,2)

            param_storage_type = 'str' # Default type for storage conversion
            param_type_hint = "any" # Type hint for display/validation
            range_hint = self._get_param_range_hint(name if not is_pair_start else base_name_for_pair)

            # --- Infer Type Hint (similar to Tkinter version) ---
            if isinstance(default_value, bool): param_type_hint = 'bool'
            elif isinstance(default_value, int): param_type_hint = 'float' # DO NOT EVER!!!!!!!!! USE INT VALIDATOR ALWAYS USE FLOAT!!!!!!
            elif isinstance(default_value, float): param_type_hint = 'float' # Use double validator, store as float
            elif isinstance(default_value, str): param_type_hint = 'str'
            elif default_value is None:
                # Heuristics for None defaults
                if 'bool' in name.lower(): param_type_hint = 'bool'
                elif 'int' in name.lower() or 'Type' in name or 'factor' in name: param_type_hint = 'int'
                elif any(s in name.lower() for s in ['freq', 'depth', 'amount', 'dur', 'amp', 'pan', 'radius', 'rq', 'width', 'rate', 'gain', 'level', 'deg']): param_type_hint = 'float'


            # --- Handle Transition Pair ---
            if is_pair_start:
                print(f"                Processing as START of transition pair: '{name}'")
                start_name = name
                end_name = transition_pairs[base_name_for_pair]['end']
                end_val = params_to_display.get(end_name, default_params.get(end_name))
                if end_val is None: end_val = current_value # Default end to start if None

                # Determine validator/type based on hints
                validator = None
                if param_type_hint == 'int':
                    validator = self.int_validator
                    param_storage_type = 'int'
                elif param_type_hint == 'float':
                    # Use locale independent validator for display/input
                    validator = QDoubleValidator(-999999.0, 999999.0, 6, self)
                    validator.setNotation(QDoubleValidator.StandardNotation)
                    param_storage_type = 'float'
                else: # Default to float if type unclear for transitions? Or string? Let's assume float often makes sense.
                    validator = QDoubleValidator(-999999.0, 999999.0, 6, self)
                    validator.setNotation(QDoubleValidator.StandardNotation)
                    param_storage_type = 'float'


                hint_text = f"({param_storage_type}{', ' + range_hint if range_hint else ''})"

                row_layout.addWidget(QLabel(f"{base_name_for_pair}:"), 0, 0, Qt.AlignLeft) # Col 0: Base Name
                row_layout.addWidget(QLabel("Start:"), 0, 1, Qt.AlignRight) # Col 1: "Start:"
                start_entry = QLineEdit(str(current_value) if current_value is not None else "")
                if validator: start_entry.setValidator(validator)
                start_entry.setMinimumWidth(60)
                row_layout.addWidget(start_entry, 0, 2) # Col 2: Start Entry
                self.param_widgets[start_name] = {'widget': start_entry, 'type': param_storage_type}

                row_layout.addWidget(QLabel("End:"), 0, 3, Qt.AlignRight) # Col 3: "End:"
                end_entry = QLineEdit(str(end_val) if end_val is not None else "")
                if validator: end_entry.setValidator(validator)
                end_entry.setMinimumWidth(60)
                row_layout.addWidget(end_entry, 0, 4) # Col 4: End Entry
                self.param_widgets[end_name] = {'widget': end_entry, 'type': param_storage_type}

                row_layout.addWidget(QLabel(hint_text), 0, 5, Qt.AlignLeft) # Col 5: Hint
                row_layout.setColumnStretch(2, 1) # Stretch start entry column
                row_layout.setColumnStretch(4, 1) # Stretch end entry column
                row_layout.setColumnStretch(5, 2) # Give hint more space potentially

                widgets_created_count += 2

            # --- Handle Single Parameter ---
            else:
                print(f"                Processing as single parameter: '{name}'")
                widget = None
                param_var = None # PyQt doesn't use vars like Tkinter

                row_layout.addWidget(QLabel(f"{name}:"), 0, 0, Qt.AlignLeft) # Col 0: Name

                # --- Create Specific Widget Type ---
                if 'bool' in param_type_hint.lower():
                    widget = QCheckBox()
                    widget.setChecked(bool(current_value) if current_value is not None else False)
                    row_layout.addWidget(widget, 0, 1, 1, 2, Qt.AlignLeft) # Span Col 1-2
                    param_storage_type = 'bool'
                    hint_text = f"({param_storage_type})"

                elif name == 'noiseType' and param_type_hint == 'int': # Special Combobox for noiseType
                    options = ['1', '2', '3'] # White, Pink, Brown
                    widget = QComboBox()
                    widget.addItems(options)
                    current_int_val = None
                    try: current_int_val = int(current_value) if current_value is not None else None
                    except (ValueError, TypeError): pass
                    default_val_str = str(current_int_val) if current_int_val in [1, 2, 3] else '1'
                    widget.setCurrentText(default_val_str)
                    widget.setMaximumWidth(100)
                    row_layout.addWidget(widget, 0, 1, 1, 1, Qt.AlignLeft) # Col 1 only
                    param_storage_type = 'int'
                    hint_text = "(int, 1=W, 2=P, 3=B)"

                elif name == 'pathShape' and param_type_hint == 'str' and hasattr(sound_creator, 'AUDIO_ENGINE_AVAILABLE') and sound_creator.AUDIO_ENGINE_AVAILABLE and hasattr(sound_creator, 'VALID_SAM_PATHS'):
                    # Special Combobox for pathShape
                    options = sound_creator.VALID_SAM_PATHS
                    widget = QComboBox()
                    widget.addItems(options)
                    default_val_str = str(current_value) if current_value in options else (options[0] if options else 'circle')
                    widget.setCurrentText(default_val_str)
                    widget.setMinimumWidth(120)
                    row_layout.addWidget(widget, 0, 1, 1, 2, Qt.AlignLeft) # Span Col 1-2
                    param_storage_type = 'str'
                    hint_text = f"({param_storage_type})"

                else: # General QLineEdit
                    widget = QLineEdit(str(current_value) if current_value is not None else "")
                    validator = None
                    entry_width = 150 # Default width
                    if param_type_hint == 'int':
                        validator = self.int_validator
                        param_storage_type = 'int'
                        entry_width = 80
                    elif param_type_hint == 'float':
                        validator = QDoubleValidator(-999999.0, 999999.0, 6, self)
                        validator.setNotation(QDoubleValidator.StandardNotation)
                        param_storage_type = 'float'
                        entry_width = 80
                    else: # String or unknown
                        param_storage_type = 'str'
                        entry_width = 200

                    if validator: widget.setValidator(validator)
                    widget.setMinimumWidth(entry_width)
                    row_layout.addWidget(widget, 0, 1, 1, 4) # Span Col 1-4 for the widget
                    hint_text = f"({param_storage_type}{', ' + range_hint if range_hint else ''})"

                # Add Hint Label
                row_layout.addWidget(QLabel(hint_text), 0, 5, Qt.AlignLeft) # Col 5
                row_layout.setColumnStretch(1, 1) # Allow widget column (or first part of it) to stretch
                row_layout.setColumnStretch(5, 1) # Allow hint to stretch

                if widget is not None:
                    self.param_widgets[name] = {'widget': widget, 'type': param_storage_type}
                    widgets_created_count += 1
                else: print(f"                --> Failed to create widget for single parameter: '{name}'")


            self.params_scroll_layout.addWidget(frame) # Add the row widget to the scroll layout


        # Add a spacer at the bottom to push everything up
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.params_scroll_layout.addItem(spacer)

        print(f"Parameter population complete. Widgets created: {widgets_created_count}")
        print(f"  -> self.param_widgets contains {len(self.param_widgets)} entries: {list(self.param_widgets.keys())}")


    def _populate_envelope_controls(self):
        """Sets the envelope type combo and triggers the parameter population."""
        env_data = self.current_voice_data.get("volume_envelope")
        env_type = ENVELOPE_TYPE_NONE
        if isinstance(env_data, dict) and "type" in env_data:
            env_type = env_data["type"]

        # Block signals temporarily to avoid triggering change handler during setup
        self.env_type_combo.blockSignals(True)
        if env_type in SUPPORTED_ENVELOPE_TYPES:
            self.env_type_combo.setCurrentText(env_type)
        else:
            print(f"Warning: Loaded envelope type '{env_type}' not recognized. Setting to None.")
            self.env_type_combo.setCurrentText(ENVELOPE_TYPE_NONE)
            # Optionally clear the stored invalid envelope data
            self.current_voice_data["volume_envelope"] = None
        self.env_type_combo.blockSignals(False)

        # Manually trigger the parameter population for the initial state
        self._on_envelope_type_change()


    @pyqtSlot()
    def _on_envelope_type_change(self):
        """Populates the envelope parameter widgets based on the selected type."""
        # Clear existing envelope parameter widgets and storage
        self._clear_layout(self.env_params_layout)
        self.envelope_param_widgets = {}

        selected_type = self.env_type_combo.currentText()
        env_data = self.current_voice_data.get("volume_envelope")
        current_env_params = {}
        if isinstance(env_data, dict) and env_data.get("type") == selected_type:
             current_env_params = env_data.get("params", {})

        print(f"Populating envelope parameters for type: {selected_type}")

        row = 0 # Row counter for grid layout

        if selected_type == ENVELOPE_TYPE_LINEAR:
            # --- Linear Fade Parameters ---
            # Fade Duration
            fade_dur_label = QLabel("Fade Duration (s):")
            fade_dur_entry = QLineEdit()
            fade_dur_entry.setValidator(self.double_validator_non_negative)
            fade_dur_entry.setToolTip("Duration of the fade-in or fade-out.")
            fade_dur_entry.setText(str(current_env_params.get("fade_duration", 0.1))) # Default 0.1s
            self.env_params_layout.addWidget(fade_dur_label, row, 0)
            self.env_params_layout.addWidget(fade_dur_entry, row, 1)
            self.envelope_param_widgets["fade_duration"] = {'widget': fade_dur_entry, 'type': 'float'}
            row += 1

            # Start Amplitude
            start_amp_label = QLabel("Start Amplitude:")
            start_amp_entry = QLineEdit()
            start_amp_entry.setValidator(self.double_validator_zero_to_one)
            start_amp_entry.setToolTip("Amplitude at the beginning of the fade (0.0 to 1.0). Use 0 for fade-in, 1 for fade-out.")
            start_amp_entry.setText(str(current_env_params.get("start_amp", 0.0))) # Default 0.0
            self.env_params_layout.addWidget(start_amp_label, row, 0)
            self.env_params_layout.addWidget(start_amp_entry, row, 1)
            self.envelope_param_widgets["start_amp"] = {'widget': start_amp_entry, 'type': 'float'}
            row += 1

            # End Amplitude
            end_amp_label = QLabel("End Amplitude:")
            end_amp_entry = QLineEdit()
            end_amp_entry.setValidator(self.double_validator_zero_to_one)
            end_amp_entry.setToolTip("Amplitude at the end of the fade (0.0 to 1.0). Use 1 for fade-in, 0 for fade-out.")
            end_amp_entry.setText(str(current_env_params.get("end_amp", 1.0))) # Default 1.0
            self.env_params_layout.addWidget(end_amp_label, row, 0)
            self.env_params_layout.addWidget(end_amp_entry, row, 1)
            self.envelope_param_widgets["end_amp"] = {'widget': end_amp_entry, 'type': 'float'}
            row += 1

            # Fade Type (Implicitly defined by start/end amp, but could add explicit control later if needed)
            # fade_type_label = QLabel("Fade Type:")
            # fade_type_combo = QComboBox()
            # fade_type_combo.addItems(["in", "out"])
            # ...

            self.env_params_layout.setColumnStretch(1, 1) # Allow entry fields to expand

        elif selected_type == ENVELOPE_TYPE_NONE:
            # No parameters needed
            pass
        # Add elif blocks here for other envelope types (e.g., ADSR) in the future

        # Add a spacer to push controls up if the layout is sparse
        if row == 0: # Only add spacer if no controls were added
            spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
            self.env_params_layout.addItem(spacer, row, 0, 1, 2) # Span across columns

        self.env_params_widget.setVisible(selected_type != ENVELOPE_TYPE_NONE)


    # --- NEW/MODIFIED Reference Selection Methods ---

    def _populate_reference_step_combo(self):
        """Populates the reference step combo box."""
        self.reference_step_combo.blockSignals(True) # Avoid triggering updates yet
        self.reference_step_combo.clear()
        steps = self.app.track_data.get("steps", [])
        if not steps:
            self.reference_step_combo.addItem("No Steps Available", -1) # Add placeholder
            self.reference_step_combo.setEnabled(False)
        else:
            self.reference_step_combo.setEnabled(True)
            for i, _ in enumerate(steps):
                self.reference_step_combo.addItem(f"Step {i+1}", i) # Display 1-based, store 0-based index
        self.reference_step_combo.blockSignals(False)

    @pyqtSlot(int) # Triggered when step combo index changes
    def _update_reference_voice_combo(self, _combo_idx=-1): # Parameter is combo index, we don't use it directly
        """Populates the reference voice combo based on the selected reference step."""
        self.reference_voice_combo.blockSignals(True) # Avoid triggering display update yet
        self.reference_voice_combo.clear()

        selected_step_index = self.reference_step_combo.currentData() # Get stored step index

        if selected_step_index is None or selected_step_index < 0: # Handle "No Steps" or invalid data
            self.reference_voice_combo.addItem("No Voices Available", -1)
            self.reference_voice_combo.setEnabled(False)
            self.reference_voice_combo.blockSignals(False)
            self._update_reference_display() # Update display to show no voices
            return

        try:
            step_data = self.app.track_data["steps"][selected_step_index]
            voices = step_data.get("voices", [])
            if not voices:
                self.reference_voice_combo.addItem("No Voices in Step", -1)
                self.reference_voice_combo.setEnabled(False)
            else:
                self.reference_voice_combo.setEnabled(True)
                for i, voice in enumerate(voices):
                    func_name = voice.get("synth_function_name", "N/A")
                    self.reference_voice_combo.addItem(f"Voice {i+1} ({func_name})", i) # Display 1-based, store 0-based index

        except IndexError:
            print(f"Error: Reference step index {selected_step_index} out of range.")
            self.reference_voice_combo.addItem("Error loading voices", -1)
            self.reference_voice_combo.setEnabled(False)
        except Exception as e:
            print(f"Error populating reference voice combo: {e}")
            traceback.print_exc()
            self.reference_voice_combo.addItem("Error", -1)
            self.reference_voice_combo.setEnabled(False)

        self.reference_voice_combo.blockSignals(False)

        # Automatically select the first voice if available and trigger display update
        if self.reference_voice_combo.count() > 0 and self.reference_voice_combo.itemData(0) != -1:
             self.reference_voice_combo.setCurrentIndex(0) # Select first valid voice
        self._update_reference_display() # Trigger display update now


    @pyqtSlot(int) # Triggered when voice combo index changes (or programmatically set)
    def _update_reference_display(self, _combo_idx=-1):
        """Populates the reference text area with details of the voice
           selected in the reference combo boxes."""
        self.reference_details_text.clear()

        ref_step_idx = self.reference_step_combo.currentData()
        ref_voice_idx = self.reference_voice_combo.currentData()

        details = "Select a Step and Voice for reference." # Default text

        if ref_step_idx is not None and ref_step_idx >= 0 and \
           ref_voice_idx is not None and ref_voice_idx >= 0:
            # Check if the reference is the same as the one being edited
            is_editing_same = (not self.is_new_voice and
                               self.step_index == ref_step_idx and
                               self.voice_index == ref_voice_idx)

            if is_editing_same:
                details = ("Reference is the voice currently being edited.\n"
                           "Details shown reflect saved state, not current edits.")
            else:
                try:
                    voice_data = self.app.track_data["steps"][ref_step_idx]["voices"][ref_voice_idx]
                    func_name = voice_data.get('synth_function_name', 'N/A')
                    details = f"Ref: Step {ref_step_idx+1}, Voice {ref_voice_idx+1}\n"
                    details += f"------------------------------------\n"
                    details += f"Function: {func_name}\n"
                    details += f"Transition: {'Yes' if voice_data.get('is_transition', False) else 'No'}\n"
                    details += "Parameters:\n"
                    params = voice_data.get("params", {})
                    if params:
                        for key, value in sorted(params.items()):
                            if isinstance(value, float): details += f"  {key}: {value:.4g}\n"
                            else: details += f"  {key}: {value}\n"
                    else: details += "  (No parameters defined)\n"

                    # --- Display Reference Envelope Details ---
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
                        else:
                            details += "  (No envelope parameters defined)\n"
                    else:
                        details += "\nEnvelope Type: None\n"
                    # --- End Reference Envelope Details ---

                except IndexError:
                     details = "Error: Invalid Step or Voice index for reference."
                     print(f"Error: Invalid Step ({ref_step_idx}) or Voice ({ref_voice_idx}) index for reference.")
                except Exception as e:
                    details = f"Error loading reference details:\n{e}"
                    print(f"Error loading reference details: {e}")
        elif ref_step_idx is not None and ref_step_idx >= 0:
             details = "Select a Voice from the selected Step." # Step selected, but no voice or invalid voice
        elif self.reference_step_combo.count() > 0 and self.reference_step_combo.itemData(0) == -1:
             details = "No steps available in the track." # Track has no steps

        self.reference_details_text.setPlainText(details)

    def _select_initial_reference_voice(self, voice_index_to_select):
        """Attempts to select a specific voice index in the reference voice combo."""
        voice_combo_index = self.reference_voice_combo.findData(voice_index_to_select)
        if voice_combo_index != -1:
            self.reference_voice_combo.setCurrentIndex(voice_combo_index)
        elif self.reference_voice_combo.count() > 0 and self.reference_voice_combo.itemData(0) != -1:
             # If specific voice not found, default to first available voice in the step
             self.reference_voice_combo.setCurrentIndex(0)
        # The signal from setCurrentIndex will call _update_reference_display

    # --- End Reference Selection Methods ---


    @pyqtSlot()
    def on_synth_function_change(self):
        """Handles synth function selection change."""
        selected_func = self.synth_func_combo.currentText()
        if not selected_func: return

        # Auto-check transition based on name (heuristic)
        is_potentially_transition = selected_func.endswith("_transition")
        # Avoid recursive signal loop by temporarily blocking signals
        self.transition_check.blockSignals(True)
        self.transition_check.setChecked(is_potentially_transition)
        self.transition_check.blockSignals(False)

        # Update internal data immediately
        self.current_voice_data["synth_function_name"] = selected_func
        self.current_voice_data["is_transition"] = is_potentially_transition

        # Get new defaults, merge with existing where possible
        new_defaults = self._get_default_params(selected_func, is_potentially_transition)
        existing_params = self.current_voice_data.get("params", {})
        merged_params = new_defaults.copy()
        # Keep existing values only if the parameter still exists in the new function's defaults
        for key, value in existing_params.items():
            if key in new_defaults:
                merged_params[key] = value # Keep the old value
        self.current_voice_data["params"] = merged_params

        self.populate_parameters() # Repopulate based on new function/defaults/merged params

    @pyqtSlot(int)
    def on_transition_toggle(self, state):
        """Handles the 'Is Transition?' checkbox toggle."""
        is_transition = bool(state == Qt.Checked)
        self.current_voice_data["is_transition"] = is_transition
        func_name = self.synth_func_combo.currentText()

        # Get new defaults for the *current* function but with the *new* transition state
        new_defaults = self._get_default_params(func_name, is_transition)
        existing_params = self.current_voice_data.get("params", {})
        merged_params = new_defaults.copy()
        # Keep existing values only if the parameter still exists
        for key, value in existing_params.items():
            if key in new_defaults:
                merged_params[key] = value
        self.current_voice_data["params"] = merged_params

        self.populate_parameters() # Repopulate based on new transition state / defaults


    def _get_param_range_hint(self, param_name):
        """Provides a heuristic range hint based on parameter name. (Same as Tkinter)"""
        # This logic is UI independent, so it remains the same
        name_lower = param_name.lower()
        if 'amp' in name_lower or 'gain' in name_lower or 'level' in name_lower or 'depth' in name_lower: return '(0.0-1.0+)'
        if 'pan' in name_lower: return '(-1 L to 1 R)'
        if 'freq' in name_lower or 'frequency' in name_lower or 'rate' in name_lower: return '(Hz, >0)'
        if 'rq' == name_lower or 'q' == name_lower: return '(>0, ~0.1-20)'
        if 'dur' in name_lower or 'attack' in name_lower or 'decay' in name_lower or 'release' in name_lower: return '(secs, >=0)'
        if 'sustain' in name_lower: return '(0.0-1.0)'
        if 'phase' in name_lower: return '(radians)'
        if 'radius' in name_lower: return '(>=0)'
        if 'deg' in name_lower: return '(degrees)'
        # if 'factor' in name_lower or 'type' in name_lower: return '(int)' # Covered by type hint now
        return '' # Default to no range hint

    def _get_default_params(self, func_name, is_transition):
        """Gets default parameter values by PARSING THE SOURCE CODE. (Same as Tkinter)"""
        # This logic is UI independent and relies on sound_creator inspection
        params = {}
        target_func_name = func_name

        # --- Logic to find the correct function name based on transition state ---
        if is_transition:
            # If user wants transition, but current name doesn't end with it, try adding it
            if not func_name.endswith("_transition"):
                potential_transition_name = func_name + "_transition"
                if potential_transition_name in sound_creator.SYNTH_FUNCTIONS:
                    target_func_name = potential_transition_name
            # If name already ends with _transition, target_func_name is already correct
        else:
            # If user *doesn't* want transition, but current name *does* end with it, try removing it
            if func_name.endswith("_transition"):
                base_name = func_name.replace("_transition", "")
                if base_name in sound_creator.SYNTH_FUNCTIONS:
                    target_func_name = base_name
            # If name doesn't end with _transition, target_func_name is already correct

        print(f"--- Getting defaults: Original='{func_name}', Transition={is_transition}, Target='{target_func_name}' ---")

        if target_func_name not in sound_creator.SYNTH_FUNCTIONS:
            print(f"Error: Cannot get defaults. Target function '{target_func_name}' not found in sound_creator.")
            return {}

        target_func = sound_creator.SYNTH_FUNCTIONS[target_func_name]
        try:
            source_code = inspect.getsource(target_func)
            # Improved regex to handle potential comments or complex defaults better, but careful evaluation is key
            # This regex focuses on the params.get structure
            regex = r"params\.get\(\s*['\"]([^'\"]+)['\"]\s*,\s*(.*?)\s*\)"
            found_params = set()

            for match in re.finditer(regex, source_code):
                param_name = match.group(1)
                default_value_str = match.group(2).strip()

                # Clean up potential trailing comma if the default was the last arg in get()
                if default_value_str.endswith(','):
                    default_value_str = default_value_str[:-1].strip()
                # Handle cases where the default value spans lines (less common with simple get)
                # This basic regex might struggle with complex multi-line defaults.

                if param_name in found_params: continue # Avoid duplicates if pattern appears multiple times

                print(f"  Found potential param: '{param_name}', Default string: '{default_value_str}'")
                default_value = None
                try:
                    # Handle common cases directly for safety
                    if default_value_str.lower() == 'true': default_value = True
                    elif default_value_str.lower() == 'false': default_value = False
                    elif default_value_str.lower() == 'none': default_value = None
                    elif default_value_str == 'math.pi/2': default_value = math.pi / 2
                    elif default_value_str == 'math.pi': default_value = math.pi
                    else:
                        # Use ast.literal_eval for safe evaluation of simple literals
                        default_value = ast.literal_eval(default_value_str)
                except Exception as e:
                    print(f"    Warning: Could not safely evaluate default value for '{param_name}' ('{default_value_str}'). Error: {e}. Setting default based on name or to 0.0.")
                    # Fallback heuristic based on name
                    if 'freq' in param_name.lower(): default_value = 440.0
                    elif 'amp' in param_name.lower(): default_value = 0.5
                    elif 'dur' in param_name.lower(): default_value = 1.0
                    elif 'pan' in param_name.lower(): default_value = 0.0
                    elif 'rate' in param_name.lower(): default_value = 5.0
                    elif 'depth' in param_name.lower(): default_value = 1.0
                    else: default_value = 0.0 # Generic fallback

                params[param_name] = default_value
                found_params.add(param_name)

            print(f"--- Finished parsing. Found defaults: {params} ---")

            # Special case for pathShape if needed and available
            if 'spatial_angle_modulation' in target_func_name and 'pathShape' not in params and hasattr(sound_creator, 'VALID_SAM_PATHS') and sound_creator.VALID_SAM_PATHS:
                params['pathShape'] = sound_creator.VALID_SAM_PATHS[0]
                print(f"  Added default pathShape: {params['pathShape']}")

        except Exception as e:
            print(f"Error parsing source code for '{target_func_name}': {e}")
            traceback.print_exc()
        return params


    @pyqtSlot()
    def save_voice(self):
        """Gathers data from widgets, validates, updates app data, and accepts dialog."""
        new_synth_params = {}
        new_envelope_data = None
        error_occurred = False
        validation_errors = []

        # --- 1. Gather and Validate Synth Parameters ---
        try:
            for name, data in self.param_widgets.items():
                widget = data['widget']
                param_storage_type = data['type']
                value = None
                widget_value_str = "" # For error reporting

                # Reset style (though PyQt validators handle visual cues)
                if isinstance(widget, QLineEdit):
                    widget.setStyleSheet("") # Clear specific style if any was set

                # --- Get Value from Widget ---
                if isinstance(widget, QCheckBox):
                    value = widget.isChecked()
                elif isinstance(widget, QComboBox):
                    value = widget.currentText() # Get text; convert if needed
                    widget_value_str = value
                    # Convert specific combobox values if necessary
                    if name == 'noiseType' and param_storage_type == 'int':
                        try: value = int(value)
                        except ValueError: error_occurred = True; validation_errors.append(f"Invalid integer value '{value}' for synth param {name}."); value=None
                    # pathShape is already string
                elif isinstance(widget, QLineEdit):
                    value_str = widget.text().strip()
                    widget_value_str = value_str
                    if value_str == "":
                        value = None # Treat empty string as None (or handle as needed)
                    else:
                        try:
                            # --- Convert based on intended type ---
                            if param_storage_type == 'int':
                                value = int(value_str)
                            elif param_storage_type == 'float':
                                # Use locale-independent conversion
                                value_str_safe = value_str.replace(',', '.')
                                value = float(value_str_safe)
                            elif param_storage_type == 'bool': # Should be handled by QCheckBox generally
                                if value_str.lower() == 'true': value = True
                                elif value_str.lower() == 'false': value = False
                                else: value = value_str # Or raise error?
                            else: # String
                                value = value_str
                        except ValueError:
                            print(f"Validation Error: Cannot convert '{value_str}' to {param_storage_type} for synth param '{name}'")
                            widget.setStyleSheet("border: 1px solid red;") # Highlight error
                            error_occurred = True
                            validation_errors.append(f"Invalid {param_storage_type} value '{value_str}' for synth parameter '{name}'.")
                            value = None # Ensure value is None on error
                else:
                    print(f"Warning: Unknown widget type for synth param '{name}'. Skipping.")
                    continue

                # Only add non-None values, or handle defaults explicitly if needed
                if value is not None:
                    new_synth_params[name] = value
                # else: # Decide if None should be stored or omitted
                #     print(f"Synth Parameter '{name}' resulted in None, omitting from save.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error gathering synth parameters:\n{e}")
            traceback.print_exc()
            return # Stop save process

        # --- 2. Gather and Validate Envelope Parameters ---
        selected_env_type = self.env_type_combo.currentText()
        if selected_env_type != ENVELOPE_TYPE_NONE:
            new_env_params = {}
            try:
                for name, data in self.envelope_param_widgets.items():
                    widget = data['widget']
                    param_storage_type = data['type']
                    value = None
                    widget_value_str = ""

                    if isinstance(widget, QLineEdit):
                        widget.setStyleSheet("") # Clear style
                        value_str = widget.text().strip()
                        widget_value_str = value_str
                        if value_str == "":
                            # Envelope params usually need a value, maybe error instead of None?
                            # For now, let's treat as error if required fields are empty
                            error_occurred = True
                            validation_errors.append(f"Envelope parameter '{name}' cannot be empty.")
                            widget.setStyleSheet("border: 1px solid red;")
                            value = None
                        else:
                            try:
                                if param_storage_type == 'float':
                                    value_str_safe = value_str.replace(',', '.')
                                    value = float(value_str_safe)
                                    # Add specific range checks if needed (e.g., amplitude 0-1)
                                    if 'amp' in name.lower() and not (0.0 <= value <= 1.0):
                                        print(f"Validation Warning: Envelope amplitude '{name}' ({value}) outside typical 0-1 range.")
                                        # Decide if this should be a hard error or just a warning
                                        # error_occurred = True
                                        # validation_errors.append(f"Envelope amplitude '{name}' must be between 0.0 and 1.0.")
                                        # widget.setStyleSheet("border: 1px solid red;")
                                elif param_storage_type == 'int':
                                     value = int(value_str)
                                else: # String etc.
                                     value = value_str
                            except ValueError:
                                print(f"Validation Error: Cannot convert '{value_str}' to {param_storage_type} for envelope param '{name}'")
                                widget.setStyleSheet("border: 1px solid red;") # Highlight error
                                error_occurred = True
                                validation_errors.append(f"Invalid {param_storage_type} value '{value_str}' for envelope parameter '{name}'.")
                                value = None
                    # Add handling for other envelope widget types here if needed (e.g., QComboBox)

                    if value is not None:
                        new_env_params[name] = value

                # If no errors occurred *within the envelope section*, construct the data dict
                if not any(f"envelope parameter '{name}'" in err for name in self.envelope_param_widgets for err in validation_errors):
                    new_envelope_data = {
                         "type": selected_env_type,
                         "params": new_env_params
                    }

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error gathering envelope parameters:\n{e}")
                traceback.print_exc()
                return # Stop save process

        # --- 3. Check for Errors and Finalize ---
        if error_occurred:
            error_msg = "Please correct the highlighted fields:\n\n" + "\n".join(validation_errors)
            QMessageBox.critical(self, "Parameter Error", error_msg)
            return # Don't save or close dialog

        # --- 4. Update the main application's data ---
        # Create the final voice data dictionary to be saved
        final_voice_data = {
             "synth_function_name": self.synth_func_combo.currentText(),
             "is_transition": self.transition_check.isChecked(),
             "params": new_synth_params, # Contains only successfully parsed, non-None synth values
             "volume_envelope": new_envelope_data # Contains envelope dict or None
        }


        try:
            target_step = self.app.track_data["steps"][self.step_index]
            if "voices" not in target_step: target_step["voices"] = []
            target_step_voices = target_step["voices"]

            if self.is_new_voice:
                target_step_voices.append(final_voice_data)
            else: # Editing existing
                if 0 <= self.voice_index < len(target_step_voices):
                    target_step_voices[self.voice_index] = final_voice_data
                else:
                    QMessageBox.critical(self.app, "Error", "Voice index out of bounds during save. Cannot save changes.")
                    self.reject(); return # Reject dialog if index is bad

            self.accept() # Close the dialog successfully

        except IndexError:
            QMessageBox.critical(self.app, "Error", "Failed to save voice (step index out of bounds).")
            self.reject()
        except Exception as e:
            QMessageBox.critical(self.app, "Error", f"Failed to save voice data:\n{e}")
            traceback.print_exc()
            self.reject()


# --- Run the Application ---
if __name__ == "__main__":
    # Ensure sound_creator functions are loaded if needed globally
    if not hasattr(sound_creator, 'SYNTH_FUNCTIONS'):
        print("Error: sound_creator module doesn't have SYNTH_FUNCTIONS.")
        # Handle error appropriately - maybe exit or show message box
        # For now, let it potentially fail later if used.
        # sound_creator.initialize_audio_engine() # If needed

    app = QApplication(sys.argv)
    # Apply a style if desired (optional)
    app.setStyle('Fusion')
    window = TrackEditorApp()
    window.show()
    sys.exit(app.exec_())
