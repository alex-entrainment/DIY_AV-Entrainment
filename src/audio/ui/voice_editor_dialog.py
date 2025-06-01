from collections import OrderedDict
import copy
import math
import sound_creator
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QComboBox, QCheckBox, QSplitter, QGroupBox, QScrollArea, QGridLayout, QPushButton, QTextEdit, QMessageBox, QSpacerItem, QSizePolicy, QLineEdit
from PyQt5.QtCore import Qt, pyqtSlot, QTimer
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QFont

ENVELOPE_TYPE_NONE = "None"
ENVELOPE_TYPE_LINEAR = "linear_fade"
SUPPORTED_ENVELOPE_TYPES = [ENVELOPE_TYPE_NONE, ENVELOPE_TYPE_LINEAR]

class VoiceEditorDialog(QDialog):
    def __init__(self, voice_data, track_idx, voice_idx, parent=None, main_app_instance=None):
        super().__init__(parent)
        self.main_app_instance = main_app_instance  # Store the main app instance
        self.setWindowTitle(f"Edit Voice {track_idx}.{voice_idx}")
        self.setMinimumSize(800, 700)  # Increased minimum size

        self.voice_data = copy.deepcopy(voice_data)  # Work on a copy
        self.track_idx = track_idx
        self.voice_idx = voice_idx

        self.original_voice_data = copy.deepcopy(voice_data) # For checking if changes were made

        self.layout = QVBoxLayout(self)

        # Top part for basic voice settings
        self.basic_settings_group = QGroupBox("Basic Settings")
        self.basic_settings_layout = QGridLayout()

        # Sound Type
        self.basic_settings_layout.addWidget(QLabel("Sound Type:"), 0, 0)
        self.sound_type_combo = QComboBox()
        self.sound_type_combo.addItems(sound_creator.SUPPORTED_SOUND_TYPES)
        self.sound_type_combo.setCurrentText(self.voice_data.get("sound_type", sound_creator.SOUND_TYPE_SINE))
        self.sound_type_combo.currentTextChanged.connect(self.update_sound_type_params)
        self.basic_settings_layout.addWidget(self.sound_type_combo, 0, 1)

        # Duration
        self.basic_settings_layout.addWidget(QLabel("Duration (s):"), 0, 2)
        self.duration_edit = QLineEdit(str(self.voice_data.get("duration", 1.0)))
        self.duration_edit.setValidator(QDoubleValidator(0.01, 1000.0, 2))
        self.basic_settings_layout.addWidget(self.duration_edit, 0, 3)
        
        # Amplitude
        self.basic_settings_layout.addWidget(QLabel("Amplitude (0-1):"), 0, 4)
        self.amplitude_edit = QLineEdit(str(self.voice_data.get("amplitude", 0.5)))
        self.amplitude_edit.setValidator(QDoubleValidator(0.0, 1.0, 2))
        self.basic_settings_layout.addWidget(self.amplitude_edit, 0, 5)

        # Enabled Checkbox
        self.enabled_checkbox = QCheckBox("Enabled")
        self.enabled_checkbox.setChecked(self.voice_data.get("enabled", True))
        self.basic_settings_layout.addWidget(self.enabled_checkbox, 0, 6)


        self.basic_settings_group.setLayout(self.basic_settings_layout)
        self.layout.addWidget(self.basic_settings_group)

        # Splitter for parameters and advanced settings
        self.splitter = QSplitter(Qt.Horizontal)
        self.layout.addWidget(self.splitter)

        # Left side: Sound Type Parameters
        self.sound_params_group = QGroupBox("Sound Type Parameters")
        self.sound_params_scroll = QScrollArea()
        self.sound_params_scroll.setWidgetResizable(True)
        self.sound_params_widget = QWidget()
        self.sound_params_layout = QGridLayout(self.sound_params_widget)
        self.sound_params_scroll.setWidget(self.sound_params_widget)
        self.sound_params_group.setLayout(QVBoxLayout()) # Set a layout for the group box
        self.sound_params_group.layout().addWidget(self.sound_params_scroll) # Add scroll area to group box layout
        self.splitter.addWidget(self.sound_params_group)
        
        # Right side: Envelope and Modulation
        self.adv_settings_widget = QWidget()
        self.adv_settings_layout = QVBoxLayout(self.adv_settings_widget)
        self.splitter.addWidget(self.adv_settings_widget)

        # Envelope Settings
        self.envelope_group = QGroupBox("Envelope")
        self.envelope_layout = QGridLayout()

        self.envelope_layout.addWidget(QLabel("Envelope Type:"), 0, 0)
        self.env_type_combo = QComboBox()
        self.env_type_combo.addItems(SUPPORTED_ENVELOPE_TYPES)
        self.env_type_combo.currentTextChanged.connect(self.update_envelope_params_visibility)
        self.envelope_layout.addWidget(self.env_type_combo, 0, 1)
        
        # Placeholder for envelope parameters
        self.env_params_widget = QWidget()
        self.env_params_layout = QGridLayout(self.env_params_widget)
        self.envelope_layout.addWidget(self.env_params_widget, 1, 0, 1, 2) # Span across two columns
        self.env_params_widget.setVisible(False) # Initially hidden

        self.envelope_group.setLayout(self.envelope_layout)
        self.adv_settings_layout.addWidget(self.envelope_group)

        # Modulation Settings - Placeholder for now
        self.modulation_group = QGroupBox("Modulation (Future)")
        # self.modulation_layout = QVBoxLayout()
        # self.modulation_group.setLayout(self.modulation_layout)
        # self.adv_settings_layout.addWidget(self.modulation_group)
        # Add a stretch to push modulation group to the top if other elements are added later
        self.adv_settings_layout.addStretch(1)


        # Bottom part for raw JSON display and buttons
        self.json_display_group = QGroupBox("Raw Voice Data (JSON)")
        self.json_display_layout = QVBoxLayout()
        self.json_text_edit = QTextEdit()
        self.json_text_edit.setReadOnly(True)
        self.json_text_edit.setFont(QFont("Courier New", 10))
        self.json_display_layout.addWidget(self.json_text_edit)
        self.json_display_group.setLayout(self.json_display_layout)
        self.layout.addWidget(self.json_display_group)

        # Buttons
        self.button_layout = QHBoxLayout()
        self.test_button = QPushButton("Test Voice")
        self.test_button.clicked.connect(self.test_voice)
        self.button_layout.addWidget(self.test_button)
        
        self.button_layout.addStretch() # Add stretch to push buttons to the right

        self.save_button = QPushButton("Save Changes")
        self.save_button.clicked.connect(self.save_changes)
        self.button_layout.addWidget(self.save_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject) # QDialog's reject() method
        self.button_layout.addWidget(self.cancel_button)
        
        self.layout.addLayout(self.button_layout)

        # Initialize UI based on current voice_data
        self.update_sound_type_params(self.sound_type_combo.currentText())
        self.load_voice_data_to_ui()
        self.update_json_display() # Initial display

        # Connect signals for live JSON update
        self._connect_live_update_signals()
        
        # Set initial splitter sizes (adjust as needed)
        self.splitter.setSizes([self.width() // 2, self.width() // 2])


    def _connect_live_update_signals(self):
        """Connects UI elements to update the JSON display live."""
        self.sound_type_combo.currentTextChanged.connect(self.update_json_on_change)
        self.duration_edit.textChanged.connect(self.update_json_on_change)
        self.amplitude_edit.textChanged.connect(self.update_json_on_change)
        self.enabled_checkbox.stateChanged.connect(self.update_json_on_change)
        self.env_type_combo.currentTextChanged.connect(self.update_json_on_change)
        # For dynamically created QLineEdit, connect them in update_sound_type_params and update_envelope_params_ui

    def _disconnect_live_update_signals(self):
        """Disconnects UI elements to prevent unwanted updates, e.g., during programmatic changes."""
        try:
            self.sound_type_combo.currentTextChanged.disconnect(self.update_json_on_change)
            self.duration_edit.textChanged.disconnect(self.update_json_on_change)
            self.amplitude_edit.textChanged.disconnect(self.update_json_on_change)
            self.enabled_checkbox.stateChanged.disconnect(self.update_json_on_change)
            self.env_type_combo.currentTextChanged.disconnect(self.update_json_on_change)

            # Disconnect dynamically created QLineEdit for sound params
            for i in range(self.sound_params_layout.count()):
                widget = self.sound_params_layout.itemAt(i).widget()
                if isinstance(widget, QLineEdit):
                    try:
                        widget.textChanged.disconnect(self.update_json_on_change)
                    except TypeError: # Already disconnected or never connected
                        pass
            
            # Disconnect dynamically created QLineEdit for envelope params
            for i in range(self.env_params_layout.count()):
                widget = self.env_params_layout.itemAt(i).widget()
                if isinstance(widget, QLineEdit):
                    try:
                        widget.textChanged.disconnect(self.update_json_on_change)
                    except TypeError:
                        pass
        except Exception as e:
            print(f"Error disconnecting signals: {e}")


    @pyqtSlot()
    def update_json_on_change(self, *args):
        """Slot to update JSON display when any relevant UI element changes."""
        # QTimer.singleShot(0, self.update_json_display) # Update after current event processing
        self.update_json_display()


    def load_voice_data_to_ui(self):
        """Loads the self.voice_data into the UI fields."""
        self._disconnect_live_update_signals() # Disconnect to prevent immediate feedback loops

        self.sound_type_combo.setCurrentText(self.voice_data.get("sound_type", sound_creator.SOUND_TYPE_SINE))
        self.duration_edit.setText(str(self.voice_data.get("duration", 1.0)))
        self.amplitude_edit.setText(str(self.voice_data.get("amplitude", 0.5)))
        self.enabled_checkbox.setChecked(self.voice_data.get("enabled", True))

        # Load sound parameters
        self.update_sound_type_params_ui(self.voice_data.get("sound_type_params", {}))

        # Load envelope
        env_type = self.voice_data.get("envelope", {}).get("type", ENVELOPE_TYPE_NONE)
        self.env_type_combo.setCurrentText(env_type if env_type in SUPPORTED_ENVELOPE_TYPES else ENVELOPE_TYPE_NONE)
        self.update_envelope_params_visibility(env_type) # This will also call update_envelope_params_ui

        self._connect_live_update_signals() # Reconnect signals
        self.update_json_display()


    def update_sound_type_params(self, sound_type):
        """Updates the parameters section based on the selected sound type."""
        # Clear previous params
        for i in reversed(range(self.sound_params_layout.count())): 
            self.sound_params_layout.itemAt(i).widget().setParent(None)

        params_config = sound_creator.SOUND_TYPE_PARAMS_CONFIG.get(sound_type, {})
        current_params = self.voice_data.get("sound_type_params", {})
        row = 0
        for param_name, config in params_config.items():
            self.sound_params_layout.addWidget(QLabel(f"{config['label']}:"), row, 0)
            edit = QLineEdit(str(current_params.get(param_name, config['default'])))
            if config['type'] == 'int':
                edit.setValidator(QIntValidator(config.get('min', 0), config.get('max', 100000)))
            elif config['type'] == 'float':
                edit.setValidator(QDoubleValidator(config.get('min', 0.0), config.get('max', 100000.0), config.get('decimals', 2)))
            
            edit.setProperty("param_name", param_name) # Store param_name for later retrieval
            edit.textChanged.connect(self.update_json_on_change) # Connect live update
            self.sound_params_layout.addWidget(edit, row, 1)
            if 'unit' in config:
                self.sound_params_layout.addWidget(QLabel(config['unit']), row, 2)
            row += 1
        
        self.sound_params_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding), row, 0, 1, 3)
        self.update_json_display() # Update JSON after params change


    def update_sound_type_params_ui(self, params_data):
        """Fills the sound type parameter QLineEdits with data."""
        for i in range(self.sound_params_layout.count()):
            widget = self.sound_params_layout.itemAt(i).widget()
            if isinstance(widget, QLineEdit):
                param_name = widget.property("param_name")
                if param_name and param_name in params_data:
                    widget.setText(str(params_data[param_name]))


    def update_envelope_params_visibility(self, selected_type):
        """Shows or hides envelope parameters based on selected type."""
        self.env_params_widget.setVisible(selected_type != ENVELOPE_TYPE_NONE)
        if selected_type != ENVELOPE_TYPE_NONE:
            self.update_envelope_params_ui(selected_type)
        self.update_json_display()


    def update_envelope_params_ui(self, env_type):
        """Updates the UI for envelope parameters."""
        # Clear previous params
        for i in reversed(range(self.env_params_layout.count())):
            self.env_params_layout.itemAt(i).widget().setParent(None)

        current_env_params = self.voice_data.get("envelope", {}).get("params", {})
        row = 0
        if env_type == ENVELOPE_TYPE_LINEAR:
            # Fade In
            self.env_params_layout.addWidget(QLabel("Fade In Duration (s):"), row, 0)
            fade_in_edit = QLineEdit(str(current_env_params.get("fade_in_duration", 0.1)))
            fade_in_edit.setValidator(QDoubleValidator(0.0, 100.0, 2))
            fade_in_edit.setProperty("env_param_name", "fade_in_duration")
            fade_in_edit.textChanged.connect(self.update_json_on_change)
            self.env_params_layout.addWidget(fade_in_edit, row, 1)
            row += 1
            # Fade Out
            self.env_params_layout.addWidget(QLabel("Fade Out Duration (s):"), row, 0)
            fade_out_edit = QLineEdit(str(current_env_params.get("fade_out_duration", 0.1)))
            fade_out_edit.setValidator(QDoubleValidator(0.0, 100.0, 2))
            fade_out_edit.setProperty("env_param_name", "fade_out_duration")
            fade_out_edit.textChanged.connect(self.update_json_on_change)
            self.env_params_layout.addWidget(fade_out_edit, row, 1)
            row += 1
        # Add other envelope types here if needed

        self.env_params_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding), row, 0, 1, 2)
        
        # Load existing values if any
        for i in range(self.env_params_layout.count()):
            widget = self.env_params_layout.itemAt(i).widget()
            if isinstance(widget, QLineEdit):
                param_name = widget.property("env_param_name")
                if param_name and param_name in current_env_params:
                    widget.setText(str(current_env_params[param_name]))
        self.update_json_display()


    def _collect_data_from_ui(self):
        """Collects all data from UI fields and returns it as a dictionary."""
        data = OrderedDict() # Use OrderedDict to maintain key order for JSON display
        data["sound_type"] = self.sound_type_combo.currentText()
        try:
            data["duration"] = float(self.duration_edit.text())
        except ValueError:
            data["duration"] = 1.0 # Default or handle error
        try:
            data["amplitude"] = float(self.amplitude_edit.text())
        except ValueError:
            data["amplitude"] = 0.5 # Default or handle error
        data["enabled"] = self.enabled_checkbox.isChecked()
        
        # Collect sound type parameters
        sound_params = OrderedDict()
        params_config = sound_creator.SOUND_TYPE_PARAMS_CONFIG.get(data["sound_type"], {})
        for i in range(self.sound_params_layout.count()):
            widget = self.sound_params_layout.itemAt(i).widget()
            if isinstance(widget, QLineEdit):
                param_name = widget.property("param_name")
                if param_name:
                    try:
                        if params_config[param_name]['type'] == 'int':
                            sound_params[param_name] = int(widget.text())
                        elif params_config[param_name]['type'] == 'float':
                            sound_params[param_name] = float(widget.text())
                        else:
                            sound_params[param_name] = widget.text() # Should not happen with current config
                    except ValueError:
                        # Use default if conversion fails
                        sound_params[param_name] = params_config[param_name]['default']
        data["sound_type_params"] = sound_params

        # Collect envelope parameters
        selected_env_type = self.env_type_combo.currentText()
        if selected_env_type != ENVELOPE_TYPE_NONE:
            env_data = OrderedDict()
            env_data["type"] = selected_env_type
            env_params = OrderedDict()
            if selected_env_type == ENVELOPE_TYPE_LINEAR:
                for i in range(self.env_params_layout.count()):
                    widget = self.env_params_layout.itemAt(i).widget()
                    if isinstance(widget, QLineEdit):
                        param_name = widget.property("env_param_name")
                        if param_name == "fade_in_duration":
                            try:
                                env_params["fade_in_duration"] = float(widget.text())
                            except ValueError:
                                env_params["fade_in_duration"] = 0.1
                        elif param_name == "fade_out_duration":
                            try:
                                env_params["fade_out_duration"] = float(widget.text())
                            except ValueError:
                                env_params["fade_out_duration"] = 0.1
            env_data["params"] = env_params
            data["envelope"] = env_data
        else:
            data["envelope"] = {"type": ENVELOPE_TYPE_NONE, "params": {}} # Ensure envelope key exists

        return data

    def update_json_display(self):
        """Updates the QTextEdit with the current voice data as JSON."""
        current_data = self._collect_data_from_ui()
        try:
            # Convert OrderedDict to regular dict for consistent JSON output with sound_creator
            # but keep the order for display if possible (json.dumps doesn't guarantee for dicts < 3.7)
            display_data = dict(current_data)
            if "sound_type_params" in display_data:
                display_data["sound_type_params"] = dict(display_data["sound_type_params"])
            if "envelope" in display_data and "params" in display_data["envelope"]:
                 display_data["envelope"]["params"] = dict(display_data["envelope"]["params"])


            json_str = sound_creator.json_dumps_ordered(display_data, indent=4)
            self.json_text_edit.setText(json_str)
        except Exception as e:
            self.json_text_edit.setText(f"Error generating JSON: {e}")


    def test_voice(self):
        """Generates and plays the sound based on current dialog settings."""
        temp_voice_data = self._collect_data_from_ui()
        if not temp_voice_data.get("enabled", False):
            QMessageBox.information(self, "Voice Disabled", "This voice is currently disabled. Enable it to test.")
            return

        try:
            # Use the main_app_instance's sound_manager if available
            if self.main_app_instance and hasattr(self.main_app_instance, 'sound_manager'):
                self.main_app_instance.sound_manager.play_voice_data_async(temp_voice_data)
            else:
                # Fallback or error if no sound_manager is found
                QMessageBox.warning(self, "Playback Error", "Sound manager not available for testing.")
                # As a more direct fallback, though less ideal as it bypasses central control:
                # temp_sound = sound_creator.SoundCreator().create_sound_from_voice(temp_voice_data)
                # if temp_sound:
                #     sound_creator.SoundPlayer().play_sound_async(temp_sound) # Needs a SoundPlayer instance
                # else:
                #     QMessageBox.warning(self, "Sound Creation Error", "Could not create sound for testing.")

        except Exception as e:
            QMessageBox.critical(self, "Test Error", f"""Error testing voice: {e}

Voice Data:
{sound_creator.json_dumps_ordered(temp_voice_data, indent=2)}""")


    def save_changes(self):
        """Saves the changes and closes the dialog."""
        self.voice_data = self._collect_data_from_ui()
        
        # Basic validation (example: duration must be positive)
        if self.voice_data["duration"] <= 0:
            QMessageBox.warning(self, "Invalid Data", "Duration must be a positive value.")
            return
        if not (0 <= self.voice_data["amplitude"] <= 1):
            QMessageBox.warning(self, "Invalid Data", "Amplitude must be between 0 and 1.")
            return

        # More specific validation for sound_type_params based on SOUND_TYPE_PARAMS_CONFIG
        sound_type = self.voice_data["sound_type"]
        params_config = sound_creator.SOUND_TYPE_PARAMS_CONFIG.get(sound_type, {})
        for param_name, config in params_config.items():
            value = self.voice_data["sound_type_params"].get(param_name)
            if value is None: # Should have a default from _collect_data_from_ui
                QMessageBox.warning(self, "Invalid Data", f"Parameter \'{config['label']}\' for {sound_type} is missing.")
                return
            
            min_val = config.get('min')
            max_val = config.get('max')

            if min_val is not None and value < min_val:
                QMessageBox.warning(self, "Invalid Data", f"\'{config['label']}\' ({value}) is below minimum ({min_val}).")
                return
            if max_val is not None and value > max_val:
                QMessageBox.warning(self, "Invalid Data", f"\'{config['label']}\' ({value}) is above maximum ({max_val}).")
                return

        # Envelope parameter validation
        env_data = self.voice_data.get("envelope", {})
        if env_data.get("type") == ENVELOPE_TYPE_LINEAR:
            env_params = env_data.get("params", {})
            fade_in = env_params.get("fade_in_duration", 0)
            fade_out = env_params.get("fade_out_duration", 0)
            duration = self.voice_data["duration"]
            if fade_in < 0 or fade_out < 0:
                QMessageBox.warning(self, "Invalid Envelope", "Fade durations cannot be negative.")
                return
            if fade_in + fade_out > duration:
                QMessageBox.warning(self, "Invalid Envelope", f"Sum of fade in ({fade_in}s) and fade out ({fade_out}s) exceeds voice duration ({duration}s).")
                return


        self.accept() # QDialog's accept() method

    def get_voice_data(self):
        """Returns the edited voice data."""
        return self.voice_data

    def has_changes(self):
        """Checks if the voice data has been modified."""
        return self._collect_data_from_ui() != self.original_voice_data

    def reject(self):
        """Overrides reject to check for changes before closing."""
        if self.has_changes():
            reply = QMessageBox.question(self, 'Unsaved Changes',
                                       "You have unsaved changes. Are you sure you want to cancel?",
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                super().reject()
            # else: do nothing, stay in dialog
        else:
            super().reject()

    def closeEvent(self, event):
        """Handle the dialog being closed by the window's X button."""
        if self.has_changes():
            reply = QMessageBox.question(self, 'Unsaved Changes',
                                       "You have unsaved changes. Are you sure you want to close?",
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
