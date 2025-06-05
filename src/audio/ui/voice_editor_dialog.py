from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QWidget, QLabel,
                             QComboBox, QCheckBox, QSplitter, QGroupBox, QScrollArea,
                             QGridLayout, QPushButton, QTextEdit, QMessageBox,
                             QSpacerItem, QSizePolicy, QLineEdit)
from PyQt5.QtCore import Qt, pyqtSlot, QTimer
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QFont
import copy
from collections import OrderedDict
import math
import inspect
import traceback
import sound_creator # Dialog directly imports sound_creator as per this "original" structure

# Constants from your original dialog structure for envelopes
ENVELOPE_TYPE_NONE = "None"
ENVELOPE_TYPE_LINEAR = "linear_fade" # Corresponds to create_linear_fade_envelope in sound_creator
SUPPORTED_ENVELOPE_TYPES = [ENVELOPE_TYPE_NONE, ENVELOPE_TYPE_LINEAR]

# Synth functions that should be available for parameter lookup but hidden
# from the UI drop-down list. These are typically transition variants that
# are selected automatically when the "Is Transition?" box is checked.
UI_EXCLUDED_FUNCTION_NAMES = [
    'rhythmic_waveshaping_transition',
    'stereo_am_independent_transition',
    'wave_shape_stereo_am_transition',
    'binaural_beat_transition',
    'isochronic_tone_transition',
    'monaural_beat_stereo_amps_transition',
    'qam_beat_transition',
    'hybrid_qam_monaural_beat_transition',
    'spatial_angle_modulation_transition',
    'spatial_angle_modulation_monaural_beat_transition',
]


class VoiceEditorDialog(QDialog): # Standard class name

    DEFAULT_WIDTH = 900
    DEFAULT_HEIGHT = 700

    def __init__(self, parent, app_ref, step_index, voice_index=None):
        super().__init__(parent)
        self.app = app_ref # Main application reference
        self.step_index = step_index
        self.voice_index = voice_index
        self.is_new_voice = (voice_index is None)
        if parent:
            self.setPalette(parent.palette())
            self.setStyleSheet(parent.styleSheet())

        # Validators
        self.double_validator_non_negative = QDoubleValidator(0.0, 999999.0, 6, self)
        self.double_validator_zero_to_one = QDoubleValidator(0.0, 1.0, 6, self)
        self.double_validator = QDoubleValidator(-999999.0, 999999.0, 6, self)
        self.int_validator = QIntValidator(-999999, 999999, self)

        self.setWindowTitle(f"{'Add' if self.is_new_voice else 'Edit'} Voice for Step {step_index + 1}")
        self.resize(self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT)
        self.setMinimumSize(700, 600)
        self.setModal(True)

        self.param_widgets = {}  # To store {'param_name': {'widget': widget, 'type': type_str}}
        self.envelope_param_widgets = {} # Similar for envelope params

        self._load_initial_data() # Loads or creates self.current_voice_data
        self._setup_ui()          # Creates UI elements
        
        # Initial population after UI setup
        if self.current_voice_data.get("synth_function_name") != "Error": # Check if data load failed
            self.populate_parameters() # Populates synth parameters based on current_voice_data
            self._populate_envelope_controls() # Populates envelope controls
        
        self._populate_reference_step_combo()

        # Set initial reference selection (if possible)
        initial_ref_step = self.app.get_selected_step_index() 
        initial_ref_voice = self.app.get_selected_voice_index()
        if initial_ref_step is not None:
            step_combo_index = self.reference_step_combo.findData(initial_ref_step)
            if step_combo_index != -1:
                self.reference_step_combo.setCurrentIndex(step_combo_index)
                # Defer voice selection slightly to ensure voice combo is populated
                if initial_ref_voice is not None:
                    QTimer.singleShot(50, lambda: self._select_initial_reference_voice(initial_ref_voice))
            elif self.reference_step_combo.count() > 0: # Fallback to first step if specific not found
                self.reference_step_combo.setCurrentIndex(0)
        elif self.reference_step_combo.count() > 0: # Fallback if no initial step selected in main
            self.reference_step_combo.setCurrentIndex(0)


    def _load_initial_data(self):
        if self.is_new_voice:
            available_funcs = sorted(
                name for name in sound_creator.SYNTH_FUNCTIONS.keys()
                if name not in UI_EXCLUDED_FUNCTION_NAMES
            )
            if not available_funcs:
                available_funcs = sorted(sound_creator.SYNTH_FUNCTIONS.keys())
            first_func_name = available_funcs[0] if available_funcs else "default_sine"
            
            is_trans = first_func_name.endswith("_transition")
            default_params = self._get_default_params(first_func_name, is_trans)
            
            self.current_voice_data = {
                "synth_function_name": first_func_name,
                "is_transition": is_trans,
                "params": default_params,
                "volume_envelope": None,  # Or {"type": ENVELOPE_TYPE_NONE, "params": {}}
                "description": "",
            }
        else:
            try:
                original_voice = self.app.track_data["steps"][self.step_index]["voices"][self.voice_index]
                self.current_voice_data = copy.deepcopy(original_voice)
                
                # Ensure essential keys exist
                if "params" not in self.current_voice_data:
                    self.current_voice_data["params"] = {}
                if "volume_envelope" not in self.current_voice_data: # Default to no envelope
                    self.current_voice_data["volume_envelope"] = None # Or {"type": ENVELOPE_TYPE_NONE, "params": {}}
                if "is_transition" not in self.current_voice_data: # Infer if missing
                    self.current_voice_data["is_transition"] = self.current_voice_data.get("synth_function_name","").endswith("_transition")
                if "description" not in self.current_voice_data:
                    self.current_voice_data["description"] = ""

            except (IndexError, KeyError, AttributeError) as e: # Added AttributeError for self.app.track_data access
                QMessageBox.critical(self.parent(), "Error", f"Could not load voice data for editing:\n{e}")
                # Fallback to a clearly erroneous state or a very basic default
                self.current_voice_data = {
                    "synth_function_name": "Error", 
                    "is_transition": False, 
                    "params": {}, 
                    "volume_envelope": None
                }
                QTimer.singleShot(0, self.reject) # Close dialog if data load fails critically


    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        # Top: Synth Function and Transition Check
        top_frame = QWidget()
        top_layout = QHBoxLayout(top_frame)
        top_layout.setContentsMargins(0,0,0,0)
        top_layout.addWidget(QLabel("Synth Function:"))
        self.synth_func_combo = QComboBox()
        try:
            # Populate from sound_creator.SYNTH_FUNCTIONS but hide functions
            # that are intended for internal use only.
            func_names = sorted(
                name for name in sound_creator.SYNTH_FUNCTIONS.keys()
                if name not in UI_EXCLUDED_FUNCTION_NAMES
            )
            if not func_names:
                raise ValueError("No synth functions found in sound_creator.SYNTH_FUNCTIONS")
            self.synth_func_combo.addItems(func_names)
        except Exception as e:
            print(f"Error populating synth_func_combo: {e}")
            self.synth_func_combo.addItem("Error: Functions unavailable")
            self.synth_func_combo.setEnabled(False)

        current_func_name = self.current_voice_data.get("synth_function_name", "")
        if current_func_name and self.synth_func_combo.findText(current_func_name) != -1:
            self.synth_func_combo.setCurrentText(current_func_name)
        elif self.synth_func_combo.count() > 0:
             self.synth_func_combo.setCurrentIndex(0) # Select first if current not found or invalid

        self.synth_func_combo.currentIndexChanged.connect(self.on_synth_function_change)
        top_layout.addWidget(self.synth_func_combo, 1) # Give combo box more stretch

        self.transition_check = QCheckBox("Is Transition?")
        self.transition_check.setChecked(self.current_voice_data.get("is_transition", False))
        self.transition_check.stateChanged.connect(self.on_transition_toggle)
        top_layout.addWidget(self.transition_check)
        main_layout.addWidget(top_frame)

        # Main Horizontal Splitter
        h_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(h_splitter, 1) # Allow splitter to take up most space

        # Left side: Synth Parameters
        self.params_groupbox = QGroupBox("Synth Parameters") # Removed "(Editing)" for cleaner UI
        params_groupbox_layout = QVBoxLayout(self.params_groupbox)
        self.params_scroll_area = QScrollArea()
        self.params_scroll_area.setWidgetResizable(True)
        self.params_scroll_content = QWidget()
        self.params_scroll_layout = QVBoxLayout(self.params_scroll_content)
        self.params_scroll_layout.setAlignment(Qt.AlignTop) # Important for parameter rows
        self.params_scroll_area.setWidget(self.params_scroll_content)
        params_groupbox_layout.addWidget(self.params_scroll_area)
        h_splitter.addWidget(self.params_groupbox)

        # Right side: Reference Voice Viewer
        reference_groupbox = QGroupBox("Reference Voice Details") # Changed title
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
        ref_select_layout.addWidget(self.reference_voice_combo, 1) # Give voice combo more stretch
        reference_layout.addLayout(ref_select_layout)
        self.reference_details_text = QTextEdit()
        self.reference_details_text.setReadOnly(True)
        self.reference_details_text.setFont(QFont("Consolas", 9)) # Good for code/data display
        reference_layout.addWidget(self.reference_details_text, 1) # Allow text edit to stretch
        h_splitter.addWidget(reference_groupbox)
        h_splitter.setSizes([int(self.DEFAULT_WIDTH * 0.6), int(self.DEFAULT_WIDTH * 0.4)]) # Adjust initial split

        # Bottom: Envelope Settings
        self.env_groupbox = QGroupBox("Volume Envelope")
        env_layout = QVBoxLayout(self.env_groupbox)
        env_type_layout = QHBoxLayout()
        env_type_layout.addWidget(QLabel("Type:"))
        self.env_type_combo = QComboBox()
        self.env_type_combo.addItems(SUPPORTED_ENVELOPE_TYPES) # Uses dialog's constants
        self.env_type_combo.currentIndexChanged.connect(self._on_envelope_type_change)
        env_type_layout.addWidget(self.env_type_combo)
        env_type_layout.addStretch(1)
        env_layout.addLayout(env_type_layout)
        self.env_params_widget = QWidget() # This widget will hold the dynamic envelope parameters
        self.env_params_layout = QGridLayout(self.env_params_widget)
        self.env_params_layout.setContentsMargins(10, 5, 5, 5)
        self.env_params_layout.setAlignment(Qt.AlignTop)
        env_layout.addWidget(self.env_params_widget)
        env_layout.addStretch(1) # Push envelope params to the top
        main_layout.addWidget(self.env_groupbox)

        # Dialog Buttons (Save, Cancel)
        button_frame = QWidget()
        button_layout = QHBoxLayout(button_frame)
        button_layout.addStretch(1) # Push buttons to the right
        self.cancel_button = QPushButton("Cancel")
        self.save_button = QPushButton("Save Voice")
        self.save_button.setStyleSheet("QPushButton { background-color: #0078D7; color: white; padding: 6px; font-weight: bold; border-radius: 3px; } QPushButton:hover { background-color: #005A9E; } QPushButton:pressed { background-color: #003C6A; }")
        self.cancel_button.clicked.connect(self.reject) # QDialog's built-in reject
        self.save_button.clicked.connect(self.save_voice)
        self.save_button.setDefault(True)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.save_button)
        main_layout.addWidget(button_frame)

    def _clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                if item is None: continue
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater() # Recommended way
                else: # If item is a layout
                    sub_layout = item.layout()
                    if sub_layout is not None:
                        self._clear_layout(sub_layout) # Recurse

    def populate_parameters(self):
        self._clear_layout(self.params_scroll_layout)
        self.param_widgets = {} 
        
        func_name = self.synth_func_combo.currentText()
        is_transition = self.transition_check.isChecked()
        
        default_params_ordered = self._get_default_params(func_name, is_transition)

        if not default_params_ordered and func_name != "Error":
            self.params_scroll_layout.addWidget(QLabel(f"Warning: Could not determine parameters for '{func_name}'.\nEnsure it's defined in _get_default_params.", self))
            return

        current_saved_params = self.current_voice_data.get("params", {})
        params_to_display = OrderedDict()
        for name, default_value in default_params_ordered.items():
            params_to_display[name] = current_saved_params.get(name, default_value)

        processed_end_params = set()
        transition_pairs = {}

        if is_transition:
            for name in default_params_ordered.keys():
                if name.startswith('start'):
                    base_name = name[len('start'):]
                    end_name = 'end' + base_name
                    if end_name in default_params_ordered:
                        transition_pairs[base_name] = {'start': name, 'end': end_name}
                        processed_end_params.add(end_name)
                elif name.startswith('end') and name not in processed_end_params:
                    base_name = name[len('end'):]
                    if ('start' + base_name) in default_params_ordered and base_name not in transition_pairs:
                        pass

        for name in default_params_ordered.keys():
            if name in processed_end_params:
                is_part_of_pair = False
                for base, pair_names in transition_pairs.items():
                    if pair_names['end'] == name:
                        is_part_of_pair = True
                        break
                if is_part_of_pair:
                    continue

            default_value = default_params_ordered[name]
            current_value = params_to_display.get(name, default_value)
            base_name_for_pair = None
            is_pair_start = False
            if is_transition and name.startswith('start'):
                base_name_for_pair_candidate = name[len('start'):]
                if base_name_for_pair_candidate in transition_pairs and transition_pairs[base_name_for_pair_candidate]['start'] == name:
                    base_name_for_pair = base_name_for_pair_candidate
                    is_pair_start = True
            
            frame = QWidget()
            row_layout = QGridLayout(frame)
            row_layout.setContentsMargins(2,2,2,2)
            
            param_storage_type = 'str'
            param_type_hint = "any"
            range_hint = self._get_param_range_hint(name if not is_pair_start else base_name_for_pair)

            if default_value is not None:
                if isinstance(default_value, bool): param_type_hint = 'bool'
                elif isinstance(default_value, int): param_type_hint = 'int'
                elif isinstance(default_value, float): param_type_hint = 'float'
                elif isinstance(default_value, str): param_type_hint = 'str'
            else:
                name_lower = name.lower()
                if 'bool' in name_lower or 'enable' in name_lower: param_type_hint = 'bool'
                elif any(s in name_lower for s in ['freq', 'depth', 'dur', 'amp', 'pan', 'rate', 'gain', 'level', 'radius', 'width', 'ratio', 'amount', 'offset', 'range', 'interval']): param_type_hint = 'float'
                elif any(s in name_lower for s in ['count', 'factor', 'index', 'type']): param_type_hint = 'int'
                else: param_type_hint = 'str'

            if is_pair_start:
                start_name = name
                end_name = transition_pairs[base_name_for_pair]['end']
                end_default_value = default_params_ordered.get(end_name, default_value)
                end_current_value = params_to_display.get(end_name, end_default_value)

                current_validator = None # Create a new validator instance for each pair or reuse type
                if param_type_hint == 'int':
                    current_validator = QIntValidator(-999999, 999999, self) # New instance
                    param_storage_type = 'int'
                elif param_type_hint == 'float':
                    current_validator = QDoubleValidator(-999999.0, 999999.0, 6, self) # New instance
                    current_validator.setNotation(QDoubleValidator.StandardNotation)
                    param_storage_type = 'float'
                # else: validator remains None for string types
                
                hint_text = f"({param_storage_type}{', ' + range_hint if range_hint else ''})"

                row_layout.addWidget(QLabel(f"{base_name_for_pair}:"), 0, 0, Qt.AlignLeft)
                row_layout.addWidget(QLabel("Start:"), 0, 1, Qt.AlignRight)
                start_entry = QLineEdit(str(current_value) if current_value is not None else "")
                if current_validator: start_entry.setValidator(current_validator) # Assign directly
                start_entry.setMinimumWidth(70); start_entry.setMaximumWidth(100)
                row_layout.addWidget(start_entry, 0, 2)
                self.param_widgets[start_name] = {'widget': start_entry, 'type': param_storage_type}
                
                row_layout.addWidget(QLabel("End:"), 0, 3, Qt.AlignRight)
                end_entry = QLineEdit(str(end_current_value) if end_current_value is not None else "")
                if current_validator: end_entry.setValidator(current_validator) # Assign directly (can reuse if settings are same, or make another new one)
                # If you need truly independent validators for start/end (e.g. different ranges), create another new one here.
                # For simplicity, if they share the same type/range, reusing is fine.
                end_entry.setMinimumWidth(70); end_entry.setMaximumWidth(100)
                row_layout.addWidget(end_entry, 0, 4)
                self.param_widgets[end_name] = {'widget': end_entry, 'type': param_storage_type}
                
                row_layout.addWidget(QLabel(hint_text), 0, 5, Qt.AlignLeft)
                row_layout.setColumnStretch(0,1); row_layout.setColumnStretch(2,1)
                row_layout.setColumnStretch(4,1); row_layout.setColumnStretch(5,1)
            else: # Single parameter
                widget = None
                row_layout.addWidget(QLabel(f"{name}:"), 0, 0, Qt.AlignLeft)

                if param_type_hint == 'bool':
                    widget = QCheckBox(); widget.setChecked(bool(current_value) if current_value is not None else False)
                    row_layout.addWidget(widget, 0, 1, 1, 2, Qt.AlignLeft); param_storage_type = 'bool'
                elif name == 'noiseType' and param_type_hint == 'int':
                    widget = QComboBox(); widget.addItems(['1', '2', '3'])
                    val_to_set = str(int(current_value)) if isinstance(current_value, (int, float)) and int(current_value) in [1,2,3] else '1'
                    widget.setCurrentText(val_to_set)
                    widget.setMaximumWidth(100); row_layout.addWidget(widget, 0, 1, 1, 1, Qt.AlignLeft); param_storage_type = 'int'
                elif name == 'pathShape' and param_type_hint == 'str' and hasattr(sound_creator, 'VALID_SAM_PATHS'):
                    widget = QComboBox(); widget.addItems(sound_creator.VALID_SAM_PATHS)
                    val_to_set = str(current_value) if current_value in sound_creator.VALID_SAM_PATHS else sound_creator.VALID_SAM_PATHS[0]
                    widget.setCurrentText(val_to_set)
                    widget.setMinimumWidth(120); row_layout.addWidget(widget, 0, 1, 1, 2, Qt.AlignLeft); param_storage_type = 'str'
                else: 
                    widget = QLineEdit(str(current_value) if current_value is not None else "")
                    entry_width = 150
                    current_validator_instance = None # Create a new validator for this specific widget
                    if param_type_hint == 'int':
                        current_validator_instance = QIntValidator(-999999, 999999, self) # New instance
                        param_storage_type = 'int'; entry_width = 80
                    elif param_type_hint == 'float':
                        current_validator_instance = QDoubleValidator(-999999.0, 999999.0, 6, self) # New instance
                        current_validator_instance.setNotation(QDoubleValidator.StandardNotation)
                        param_storage_type = 'float'; entry_width = 80
                    else: param_storage_type = 'str'; entry_width = 200
                    
                    if current_validator_instance: widget.setValidator(current_validator_instance) # Assign new instance directly
                    widget.setMinimumWidth(entry_width); widget.setMaximumWidth(entry_width + 50)
                    row_layout.addWidget(widget, 0, 1, 1, 1)
                    
                    hint_text_label = QLabel(f"({param_storage_type}{', ' + range_hint if range_hint else ''})")
                    row_layout.addWidget(hint_text_label, 0, 2, Qt.AlignLeft)
                    row_layout.setColumnStretch(2,1)

                if widget is not None: self.param_widgets[name] = {'widget': widget, 'type': param_storage_type}
            
            self.params_scroll_layout.addWidget(frame)
        self.params_scroll_layout.addStretch(1)

    def _populate_envelope_controls(self):
        env_data = self.current_voice_data.get("volume_envelope") # main.py uses "volume_envelope"
        env_type = ENVELOPE_TYPE_NONE
        
        if isinstance(env_data, dict) and "type" in env_data:
            env_type_from_data = env_data["type"]
            if env_type_from_data in SUPPORTED_ENVELOPE_TYPES: # Ensure it's a supported type
                env_type = env_type_from_data
        
        self.env_type_combo.blockSignals(True)
        self.env_type_combo.setCurrentText(env_type)
        self.env_type_combo.blockSignals(False)
        
        self._on_envelope_type_change() # This will build and populate params for the selected type

    @pyqtSlot()
    def _on_envelope_type_change(self):
        self._clear_layout(self.env_params_layout)
        self.envelope_param_widgets = {} # Reset
        selected_type = self.env_type_combo.currentText()
        
        env_data = self.current_voice_data.get("volume_envelope") # From main.py format
        current_env_params = {}
        if isinstance(env_data, dict) and env_data.get("type") == selected_type:
            current_env_params = env_data.get("params", {})

        row = 0
        if selected_type == ENVELOPE_TYPE_LINEAR:
            # Definition from your original VoiceEditorDialogue for linear envelope
            params_def = [
                ("Fade Duration (s):", "fade_duration", 0.1, self.double_validator_non_negative, "float"), # Corresponds to 'create_linear_fade_envelope'
                ("Start Amplitude:", "start_amp", 0.0, self.double_validator_zero_to_one, "float"),
                ("End Amplitude:", "end_amp", 1.0, self.double_validator_zero_to_one, "float")
            ]
            # If your main.py "linear_fade" uses "fade_in_duration" and "fade_out_duration" like my previous dialog suggestion, adjust here.
            # Assuming the above params_def is what sound_creator.create_linear_fade_envelope expects.
            
            for label_text, param_name, default_val, validator_type, val_type in params_def:
                label = QLabel(label_text)
                entry = QLineEdit()
                entry.setValidator(copy.deepcopy(validator_type)) # Use a copy of the validator
                entry.setText(str(current_env_params.get(param_name, default_val)))
                
                self.env_params_layout.addWidget(label, row, 0)
                self.env_params_layout.addWidget(entry, row, 1)
                self.envelope_param_widgets[param_name] = {'widget': entry, 'type': val_type}
                row += 1
            self.env_params_layout.setColumnStretch(1, 1) # Allow entry fields to expand
        
        # Add other envelope types (ADSR, Linen) here if you support them
        # elif selected_type == "adsr": ...
            
        if row == 0 and selected_type != ENVELOPE_TYPE_NONE : # If params were expected but none defined
             self.env_params_layout.addWidget(QLabel(f"No parameters defined for '{selected_type}' envelope."), 0,0)

        self.env_params_layout.addItem(QSpacerItem(20,10, QSizePolicy.Minimum, QSizePolicy.Expanding), row +1, 0)
        self.env_params_widget.setVisible(selected_type != ENVELOPE_TYPE_NONE and row > 0)


    def _populate_reference_step_combo(self):
        self.reference_step_combo.blockSignals(True)
        self.reference_step_combo.clear()
        steps = self.app.track_data.get("steps", [])
        if not steps:
            self.reference_step_combo.addItem("No Steps Available", -1)
            self.reference_step_combo.setEnabled(False)
        else:
            self.reference_step_combo.setEnabled(True)
            for i, step_data in enumerate(steps):
                desc = step_data.get("description","")
                item_text = f"Step {i+1}"
                if desc: item_text += f": {desc[:20]}{'...' if len(desc)>20 else ''}"
                self.reference_step_combo.addItem(item_text, i) # Store original index as data
        self.reference_step_combo.blockSignals(False)
        # Trigger update for voice combo if items were added
        if self.reference_step_combo.count() > 0:
            self._update_reference_voice_combo(self.reference_step_combo.currentIndex())


    @pyqtSlot(int)
    def _update_reference_voice_combo(self, combo_idx=-1): # combo_idx is from signal, not used directly if data is used
        self.reference_voice_combo.blockSignals(True)
        self.reference_voice_combo.clear()
        selected_step_index = self.reference_step_combo.currentData() # Get original step index

        if selected_step_index is None or selected_step_index < 0:
            self.reference_voice_combo.addItem("No Voices Available", -1)
            self.reference_voice_combo.setEnabled(False)
        else:
            try:
                voices = self.app.track_data["steps"][selected_step_index].get("voices", [])
                if not voices:
                    self.reference_voice_combo.addItem("No Voices in Step", -1)
                    self.reference_voice_combo.setEnabled(False)
                else:
                    self.reference_voice_combo.setEnabled(True)
                    for i, voice in enumerate(voices):
                        self.reference_voice_combo.addItem(f"Voice {i+1} ({voice.get('synth_function_name', 'N/A')[:25]})", i) # Store original voice index
            except IndexError:
                self.reference_voice_combo.addItem("Error loading voices", -1)
                self.reference_voice_combo.setEnabled(False)
        self.reference_voice_combo.blockSignals(False)
        # Trigger display update if items were added
        if self.reference_voice_combo.count() > 0 :
             self._update_reference_display(self.reference_voice_combo.currentIndex())


    @pyqtSlot(int)
    def _update_reference_display(self, combo_idx=-1): # combo_idx is from signal
        self.reference_details_text.clear()
        ref_step_idx = self.reference_step_combo.currentData()
        ref_voice_idx = self.reference_voice_combo.currentData()
        
        details = "Select a Step and Voice to see its details for reference."
        if ref_step_idx is not None and ref_step_idx >= 0 and \
           ref_voice_idx is not None and ref_voice_idx >= 0: # Valid selection
            
            is_editing_same_voice = (not self.is_new_voice and 
                                     self.step_index == ref_step_idx and 
                                     self.voice_index == ref_voice_idx)
            
            if is_editing_same_voice:
                # Show current UI state of the voice being edited for "reference"
                details = "Reference is the voice currently being edited.\nDetails reflect current UI settings (unsaved):\n------------------------------------\n"
                current_ui_data = self._collect_data_for_main_app() # Get data from current dialog UI
                details += f"Function: {current_ui_data.get('synth_function_name', 'N/A')}\n"
                details += f"Transition: {'Yes' if current_ui_data.get('is_transition', False) else 'No'}\n"
                details += "Parameters:\n"
                params = current_ui_data.get("params", {})
            else:
                # Show saved data of the selected reference voice
                try:
                    voice_data = self.app.track_data["steps"][ref_step_idx]["voices"][ref_voice_idx]
                    details = f"Reference: Step {ref_step_idx+1}, Voice {ref_voice_idx+1} (Saved State)\n------------------------------------\n"
                    details += f"Function: {voice_data.get('synth_function_name', 'N/A')}\n"
                    details += f"Transition: {'Yes' if voice_data.get('is_transition', False) else 'No'}\n"
                    details += "Parameters:\n"
                    params = voice_data.get("params", {})
                except IndexError:
                    details = "Error: Invalid Step or Voice index for reference."
                    params = {} # Ensure params is defined
            
            if params:
                for k, v in sorted(params.items()):
                    details += "  {}: {}\n".format(k, f"{v:.4g}" if isinstance(v, float) else v)
            else:
                if "Function:" in details: # Only if not an error message
                    details += "  (No parameters defined)\n"

            # Envelope details (either current UI if editing same, or saved for other ref)
            if is_editing_same_voice:
                env_data_collected = self._collect_data_from_ui().get("volume_envelope") # Dialog's internal structure uses "envelope"
                env_data_to_display = env_data_collected if env_data_collected else self.current_voice_data.get("volume_envelope")
            else:
                try:
                     env_data_to_display = self.app.track_data["steps"][ref_step_idx]["voices"][ref_voice_idx].get("volume_envelope")
                except: env_data_to_display = None

            if env_data_to_display and isinstance(env_data_to_display, dict):
                details += f"\nEnvelope Type: {env_data_to_display.get('type', 'N/A')}\n  Envelope Params:\n"
                env_params = env_data_to_display.get('params', {})
                if env_params:
                    for k,v in sorted(env_params.items()): details += f"    {k}: {v:.4g if isinstance(v,float) else v}\n"
                else: details += "  (No envelope parameters defined)\n"
            elif "Function:" in details: # Only if not an error message
                 details += "\nEnvelope Type: None\n"

        elif ref_step_idx is not None and ref_step_idx >= 0:
            details = "Select a Voice from the chosen Step."
        elif self.reference_step_combo.count() > 0 and self.reference_step_combo.itemData(0) == -1:
            details = "No steps available in the track to reference."
            
        self.reference_details_text.setPlainText(details)


    def _select_initial_reference_voice(self, voice_index_to_select):
        """Tries to select a specific voice index in the reference voice combo."""
        if self.reference_voice_combo.isEnabled(): # Only if combo is enabled (has voices)
            voice_combo_index = self.reference_voice_combo.findData(voice_index_to_select)
            if voice_combo_index != -1:
                self.reference_voice_combo.setCurrentIndex(voice_combo_index)
            elif self.reference_voice_combo.count() > 0 and self.reference_voice_combo.itemData(0) != -1: # Not "No voices"
                self.reference_voice_combo.setCurrentIndex(0) # Fallback to first actual voice


    @pyqtSlot()
    def on_synth_function_change(self):
        selected_func = self.synth_func_combo.currentText()
        if not selected_func or selected_func.startswith("Error:"): return

        # Auto-update transition checkbox based on function name convention
        is_transition_by_name = selected_func.endswith("_transition")
        self.transition_check.blockSignals(True)
        self.transition_check.setChecked(is_transition_by_name)
        self.transition_check.blockSignals(False)

        # Update current_voice_data to reflect selection and new defaults
        self.current_voice_data["synth_function_name"] = selected_func
        self.current_voice_data["is_transition"] = is_transition_by_name # Reflect UI
        
        new_default_params = self._get_default_params(selected_func, is_transition_by_name)
        # Keep existing param values if they exist for the new set of params, otherwise use new defaults
        updated_params = OrderedDict()
        current_params_in_data = self.current_voice_data.get("params",{})
        for name, default_val in new_default_params.items():
            updated_params[name] = current_params_in_data.get(name, default_val)
        self.current_voice_data["params"] = updated_params
        
        self.populate_parameters() # Rebuild UI with (potentially new) params and (potentially updated) values

    @pyqtSlot(int)
    def on_transition_toggle(self, state):
        is_transition = bool(state == Qt.Checked)
        self.current_voice_data["is_transition"] = is_transition

        # Refresh parameters UI as the set of params might change (e.g. startX/endX vs X)
        func_name = self.synth_func_combo.currentText()
        new_default_params = self._get_default_params(func_name, is_transition)
        current_params_in_data = self.current_voice_data.get("params", {})

        def _norm(key: str) -> str:
            return key.replace("_", "").lower()

        updated_params = OrderedDict()
        if is_transition:
            base_map = { _norm(k): v for k, v in current_params_in_data.items() }
            for name, default_val in new_default_params.items():
                if name.startswith("start"):
                    base_key = _norm(name[len("start"):])
                    if name in current_params_in_data:
                        updated_params[name] = current_params_in_data[name]
                    elif base_key in base_map:
                        updated_params[name] = base_map[base_key]
                    else:
                        updated_params[name] = default_val
                else:
                    updated_params[name] = current_params_in_data.get(name, default_val)
        else:
            start_map = { _norm(k[len("start"):]): v for k, v in current_params_in_data.items() if k.startswith("start") }
            for name, default_val in new_default_params.items():
                base_key = _norm(name)
                if base_key in start_map:
                    updated_params[name] = start_map[base_key]
                else:
                    updated_params[name] = current_params_in_data.get(name, default_val)

        self.current_voice_data["params"] = updated_params

        self.populate_parameters()

    # --- Helper methods for collecting UI data ---
    def _collect_data_for_main_app(self):
        """
        Collects current UI data and returns it in the format expected by main app.
        This is similar to save_voice() but doesn't validate or save, just collects.
        """
        # Collect synth parameters
        synth_params = {}
        for name, data in self.param_widgets.items():
            widget, param_type = data['widget'], data['type']
            value = None
            
            if isinstance(widget, QCheckBox):
                value = widget.isChecked()
            elif isinstance(widget, QComboBox):
                value_str = widget.currentText()
                if name == 'noiseType' and param_type == 'int':
                    try: 
                        value = int(value_str)
                    except ValueError: 
                        value = 1  # Default fallback
                else:
                    value = value_str
            elif isinstance(widget, QLineEdit):
                value_str = widget.text().strip()
                if value_str:
                    try:
                        if param_type == 'int': 
                            value = int(value_str)
                        elif param_type == 'float': 
                            value = float(value_str.replace(',', '.'))
                        else: 
                            value = value_str
                    except ValueError:
                        value = None  # Will use default
                else:
                    value = None
            
            if value is not None:
                synth_params[name] = value
        
        # Collect envelope data
        envelope_data = None
        selected_env_type = self.env_type_combo.currentText()
        if selected_env_type != ENVELOPE_TYPE_NONE:
            env_params = {}
            for name, data in self.envelope_param_widgets.items():
                widget, param_type = data['widget'], data['type']
                if isinstance(widget, QLineEdit):
                    value_str = widget.text().strip()
                    if value_str:
                        try:
                            if param_type == 'float': 
                                env_params[name] = float(value_str.replace(',', '.'))
                            elif param_type == 'int': 
                                env_params[name] = int(value_str)
                            else: 
                                env_params[name] = value_str
                        except ValueError:
                            pass  # Skip invalid values
            
            if env_params:
                envelope_data = {"type": selected_env_type, "params": env_params}
        
        return {
            "synth_function_name": self.synth_func_combo.currentText(),
            "is_transition": self.transition_check.isChecked(),
            "params": synth_params,
            "volume_envelope": envelope_data
        }
    
    def _collect_data_from_ui(self):
        """
        Alias for _collect_data_for_main_app() for consistency with existing code.
        """
        return self._collect_data_for_main_app()

    def _get_param_range_hint(self, param_name_base): # param_name_base is without start/end
        name_lower = param_name_base.lower()
        # More specific first
        if 'pan' in name_lower: return '-1 L to 1 R'
        if 'noiseType' in name_lower: return '1:W,2:P,3:B'
        if 'shape' in name_lower and 'path' not in name_lower : return 'e.g. 0-10'
        if 'pathShape' in name_lower: return 'e.g. circle, line'


        if any(s in name_lower for s in ['amp', 'gain', 'level', 'depth']) and 'mod' in name_lower : return 'e.g. 0.0-1.0'
        if any(s in name_lower for s in ['amp', 'gain', 'level']) : return 'e.g. 0.0-1.0+' # Amplitudes can exceed 1 prior to mixing
        
        if any(s in name_lower for s in ['freq', 'frequency', 'rate']): return 'Hz, >0'
        if 'q' == name_lower or 'rq' == name_lower : return '>0, e.g. 0.1-20' # Quality factor
        if any(s in name_lower for s in ['dur', 'attack', 'decay', 'release', 'delay', 'interval']): return 'secs, >=0'
        if 'phase' in name_lower and 'offset' not in name_lower: return 'radians, e.g. 0-2pi'
        if 'radius' in name_lower : return '>=0'
        if 'width' in name_lower : return 'Hz or ratio'
        if 'ratio' in name_lower : return '>0'
        if 'amount' in name_lower : return 'varies'
        if 'factor' in name_lower : return 'varies'


        return '' # No specific hint

    def _get_default_params(self, func_name_from_combo: str, is_transition_mode: bool) -> OrderedDict:
        """
        Retrieves an OrderedDict of default parameters for a given synth function.
        Uses the internal param_definitions structure.
        """
        # This map should align QComboBox text with keys in param_definitions
        internal_func_key_map = {
            "Rhythmic Waveshaping": "rhythmic_waveshaping",
            "Rhythmic Waveshaping Transition": "rhythmic_waveshaping_transition",
            "Stereo AM Independent": "stereo_am_independent",
            "Stereo AM Independent Transition": "stereo_am_independent_transition",
            "Wave Shape Stereo AM": "wave_shape_stereo_am",
            "Wave Shape Stereo AM Transition": "wave_shape_stereo_am_transition",
            "Spatial Angle Modulation (SAM Engine)": "spatial_angle_modulation_engine", # Uses Node/SAMVoice directly
            "Spatial Angle Modulation (SAM Engine Transition)": "spatial_angle_modulation_engine_transition",
            "Binaural Beat": "binaural_beat",
            "Binaural Beat Transition": "binaural_beat_transition",
            "Monaural Beat Stereo Amps": "monaural_beat_stereo_amps",
            "Monaural Beat Stereo Amps Transition": "monaural_beat_stereo_amps_transition",
            "Spatial Angle Modulation (Monaural Core)": "spatial_angle_modulation_monaural", # Uses monaural_beat as core
            "Spatial Angle Modulation (Monaural Core Transition)": "spatial_angle_modulation_monaural_transition",
            "Isochronic Tone": "isochronic_tone",
            "Isochronic Tone Transition": "isochronic_tone_transition",
            "QAM Beat": "qam_beat", # Ensure this mapping is correct for your UIAdd commentMore actions
            "QAM Beat Transition": "qam_beat_transition",
            "Hybrid QAM Monaural Beat": "hybrid_qam_monaural_beat",
            "Hybrid QAM Monaural Beat Transition": "hybrid_qam_monaural_beat_transition",
            # Add other mappings if your UI names differ from these examples
        }

        base_func_key = internal_func_key_map.get(func_name_from_combo, func_name_from_combo)


        param_definitions = {
            "rhythmic_waveshaping": { # This is an example, ensure it's correctAdd commentMore actions
                "standard": [
                    ('amp', 0.25), ('carrierFreq', 200), ('modFreq', 4),
                    ('modDepth', 1.0), ('shapeAmount', 5.0), ('pan', 0)
                ],
                "transition": [ 
                    ('amp', 0.25), ('startCarrierFreq', 200), ('endCarrierFreq', 80),
                    ('startModFreq', 12), ('endModFreq', 7.83),
                    ('startModDepth', 1.0), ('endModDepth', 1.0),
                    ('startShapeAmount', 5.0), ('endShapeAmount', 5.0), ('pan', 0)
                ]
            },
            "stereo_am_independent": { # This is an example, ensure it's correct
                "standard": [
                    ('amp', 0.25), ('carrierFreq', 200.0), ('modFreqL', 4.0),
                    ('modDepthL', 0.8), ('modPhaseL', 0), ('modFreqR', 4.0),
                    ('modDepthR', 0.8), ('modPhaseR', 0), ('stereo_width_hz', 0.2)
                ],
                "transition": [
                    ('amp', 0.25), ('startCarrierFreq', 200), ('endCarrierFreq', 250),
                    ('startModFreqL', 4), ('endModFreqL', 6),
                    ('startModDepthL', 0.8), ('endModDepthL', 0.8),
                    ('startModPhaseL', 0),
                    ('startModFreqR', 4.1), ('endModFreqR', 5.9),
                    ('startModDepthR', 0.8), ('endModDepthR', 0.8),
                    ('startModPhaseR', 0),
                    ('startStereoWidthHz', 0.2), ('endStereoWidthHz', 0.2)
                ]
            },
            "wave_shape_stereo_am": { # This is an example, ensure it's correct
                "standard": [
                    ('amp', 0.15), ('carrierFreq', 200), ('shapeModFreq', 4),
                    ('shapeModDepth', 0.8), ('shapeAmount', 0.5),
                    ('stereoModFreqL', 4.1), ('stereoModDepthL', 0.8),
                    ('stereoModPhaseL', 0), ('stereoModFreqR', 4.0),
                    ('stereoModDepthR', 0.8), ('stereoModPhaseR', math.pi / 2)
                ],
                "transition": [
                    ('amp', 0.15), ('startCarrierFreq', 200), ('endCarrierFreq', 100),
                    ('startShapeModFreq', 4), ('endShapeModFreq', 8),
                    ('startShapeModDepth', 0.8), ('endShapeModDepth', 0.8),
                    ('startShapeAmount', 0.5), ('endShapeAmount', 0.5),
                    ('startStereoModFreqL', 4.1), ('endStereoModFreqL', 6.0),
                    ('startStereoModDepthL', 0.8), ('endStereoModDepthL', 0.8),
                    ('startStereoModPhaseL', 0),
                    ('startStereoModFreqR', 4.0), ('endStereoModFreqR', 6.1),
                    ('startStereoModDepthR', 0.9), ('endStereoModDepthR', 0.9),
                    ('startStereoModPhaseR', math.pi / 2)
                ]
            },
            "spatial_angle_modulation_engine": { # This is an example, ensure it's correct
                "standard": [
                    ('amp', 0.7), ('carrierFreq', 440.0), ('beatFreq', 4.0),
                    ('pathShape', 'circle'), ('pathRadius', 1.0),
                    ('arcStartDeg', 0.0), ('arcEndDeg', 360.0),
                    ('frame_dur_ms', 46.4), ('overlap_factor', 8)
                ],
                "transition": [
                    ('amp', 0.7),
                    ('startCarrierFreq', 440.0), ('endCarrierFreq', 440.0),
                    ('startBeatFreq', 4.0), ('endBeatFreq', 4.0),
                    ('startPathShape', 'circle'), ('endPathShape', 'circle'),
                    ('startPathRadius', 1.0), ('endPathRadius', 1.0),
                    ('startArcStartDeg', 0.0), ('endArcStartDeg', 0.0),
                    ('startArcEndDeg', 360.0), ('endArcEndDeg', 360.0),
                    ('frame_dur_ms', 46.4), ('overlap_factor', 8)
                ]
            },
            "binaural_beat": { # This is an example, ensure it's correct
                "standard": [
                    ('ampL', 0.5), ('ampR', 0.5), ('baseFreq', 200.0), ('beatFreq', 4.0),
                    ('forceMono', False), ('startPhaseL', 0.0), ('startPhaseR', 0.0),
                    ('ampOscDepthL', 0.0), ('ampOscFreqL', 0.0),
                    ('ampOscDepthR', 0.0), ('ampOscFreqR', 0.0),
                    ('freqOscRangeL', 0.0), ('freqOscFreqL', 0.0),
                    ('freqOscRangeR', 0.0), ('freqOscFreqR', 0.0),
                    ('ampOscPhaseOffsetL', 0.0), ('ampOscPhaseOffsetR', 0.0),
                    ('phaseOscFreq', 0.0), ('phaseOscRange', 0.0),
                    ('glitchInterval', 0.0), ('glitchDur', 0.0), 
                    ('glitchNoiseLevel', 0.0), ('glitchFocusWidth', 0.0), ('glitchFocusExp', 0.0)
                ],
                "transition": [ 
                    ('startAmpL', 0.5), ('endAmpL', 0.5),
                    ('startAmpR', 0.5), ('endAmpR', 0.5),
                    ('startBaseFreq', 200.0), ('endBaseFreq', 200.0),
                    ('startBeatFreq', 4.0), ('endBeatFreq', 4.0),
                    ('startForceMono', 0.0), ('endForceMono', 0.0), # Should be bool if possible, or handled as 0/1
                    ('startStartPhaseL', 0.0), ('endStartPhaseL', 0.0),
                    ('startStartPhaseR', 0.0), ('endStartPhaseR', 0.0),
                    ('startPhaseOscFreq', 0.0), ('endPhaseOscFreq', 0.0),
                    ('startPhaseOscRange', 0.0), ('endPhaseOscRange', 0.0),
                    ('startAmpOscDepthL', 0.0), ('endAmpOscDepthL', 0.0),
                    ('startAmpOscFreqL', 0.0), ('endAmpOscFreqL', 0.0),
                    ('startAmpOscDepthR', 0.0), ('endAmpOscDepthR', 0.0),
                    ('startAmpOscFreqR', 0.0), ('endAmpOscFreqR', 0.0),
                    ('startAmpOscPhaseOffsetL', 0.0), ('endAmpOscPhaseOffsetL', 0.0),
                    ('startAmpOscPhaseOffsetR', 0.0), ('endAmpOscPhaseOffsetR', 0.0),
                    ('startFreqOscRangeL', 0.0), ('endFreqOscRangeL', 0.0),
                    ('startFreqOscFreqL', 0.0), ('endFreqOscFreqL', 0.0),
                    ('startFreqOscRangeR', 0.0), ('endFreqOscRangeR', 0.0),
                    ('startFreqOscFreqR', 0.0), ('endFreqOscFreqR', 0.0),
                    ('startGlitchInterval', 0.0), ('endGlitchInterval', 0.0),
                    ('startGlitchDur', 0.0), ('endGlitchDur', 0.0),
                    ('startGlitchNoiseLevel', 0.0), ('endGlitchNoiseLevel', 0.0),
                    ('startGlitchFocusWidth', 0.0), ('endGlitchFocusWidth', 0.0),
                    ('startGlitchFocusExp', 0.0), ('endGlitchFocusExp', 0.0)
                ]
            },
            "monaural_beat_stereo_amps": { # This is an example, ensure it's correct
                "standard": [
                    ('amp_lower_L', 0.5), ('amp_upper_L', 0.5),
                    ('amp_lower_R', 0.5), ('amp_upper_R', 0.5),
                    ('baseFreq', 200.0), ('beatFreq', 4.0),
                    ('startPhaseL', 0.0), ('startPhaseR', 0.0),
                    ('phaseOscFreq', 0.0), ('phaseOscRange', 0.0),
                    ('ampOscDepth', 0.0), ('ampOscFreq', 0.0), ('ampOscPhaseOffset', 0.0)
                ],
                "transition": [ 
                    ('start_amp_lower_L', 0.5), ('end_amp_lower_L', 0.5),
                    ('start_amp_upper_L', 0.5), ('end_amp_upper_L', 0.5),
                    ('start_amp_lower_R', 0.5), ('end_amp_lower_R', 0.5),
                    ('start_amp_upper_R', 0.5), ('end_amp_upper_R', 0.5),
                    ('startBaseFreq', 200.0), ('endBaseFreq', 200.0),
                    ('startBeatFreq', 4.0), ('endBeatFreq', 4.0),
                    ('startStartPhaseL', 0.0), ('endStartPhaseL', 0.0),
                    ('startStartPhaseU', 0.0), ('endStartPhaseU', 0.0), 
                    ('startPhaseOscFreq', 0.0), ('endPhaseOscFreq', 0.0),
                    ('startPhaseOscRange', 0.0), ('endPhaseOscRange', 0.0),
                    ('startAmpOscDepth', 0.0), ('endAmpOscDepth', 0.0),
                    ('startAmpOscFreq', 0.0), ('endAmpOscFreq', 0.0),
                    ('startAmpOscPhaseOffset', 0.0), ('endAmpOscPhaseOffset', 0.0)
                ]
            },
            "spatial_angle_modulation_monaural": { # This is an example, ensure it's correct
                "standard": [ 
                    ('sam_ampOscDepth', 0.0), ('sam_ampOscFreq', 0.0), ('sam_ampOscPhaseOffset', 0.0),
                    ('amp_lower_L', 0.5), ('amp_upper_L', 0.5),
                    ('amp_lower_R', 0.5), ('amp_upper_R', 0.5),
                    ('baseFreq', 200.0), ('beatFreq', 4.0),
                    ('startPhaseL', 0.0), ('startPhaseR', 0.0),
                    ('phaseOscFreq', 0.0), ('phaseOscRange', 0.0),
                    ('monaural_ampOscDepth', 0.0), ('monaural_ampOscFreq', 0.0),
                    ('monaural_ampOscPhaseOffset', 0.0),
                    ('spatialBeatFreq', 4.0), ('spatialPhaseOffset', 0.0),
                    ('amp', 0.7), ('pathRadius', 1.0),
                    ('frame_dur_ms', 46.4), ('overlap_factor', 8)
                ],
                "transition": [ 
                    ('start_amp_lower_L', 0.5), ('end_amp_lower_L', 0.5),
                    ('start_amp_upper_L', 0.5), ('end_amp_upper_L', 0.5),
                    ('start_amp_lower_R', 0.5), ('end_amp_lower_R', 0.5),
                    ('start_amp_upper_R', 0.5), ('end_amp_upper_R', 0.5),
                    ('startBaseFreq', 200.0), ('endBaseFreq', 200.0),
                    ('startBeatFreq', 4.0), ('endBeatFreq', 4.0),
                    ('startStartPhaseL_monaural', 0.0), ('endStartPhaseL_monaural', 0.0),
                    ('startStartPhaseU_monaural', 0.0), ('endStartPhaseU_monaural', 0.0),
                    ('startPhaseOscFreq_monaural', 0.0), ('endPhaseOscFreq_monaural', 0.0),
                    ('startPhaseOscRange_monaural', 0.0), ('endPhaseOscRange_monaural', 0.0),
                    ('startAmpOscDepth_monaural', 0.0), ('endAmpOscDepth_monaural', 0.0),
                    ('startAmpOscFreq_monaural', 0.0), ('endAmpOscFreq_monaural', 0.0),
                    ('startAmpOscPhaseOffset_monaural', 0.0), ('endAmpOscPhaseOffset_monaural', 0.0),
                    ('start_sam_ampOscDepth', 0.0), ('end_sam_ampOscDepth', 0.0),
                    ('start_sam_ampOscFreq', 0.0), ('end_sam_ampOscFreq', 0.0),
                    ('start_sam_ampOscPhaseOffset', 0.0), ('end_sam_ampOscPhaseOffset', 0.0),
                    ('startSpatialBeatFreq', 4.0), ('endSpatialBeatFreq', 4.0),
                    ('startSpatialPhaseOffset', 0.0), ('endSpatialPhaseOffset', 0.0),
                    ('startPathRadius', 1.0), ('endPathRadius', 1.0),
                    ('startAmp', 0.7), ('endAmp', 0.7),
                    ('frame_dur_ms', 46.4), ('overlap_factor', 8)
                ]
            },
            "isochronic_tone": { # This is an example, ensure it's correct
                "standard": [
                    ('amp', 0.5), ('baseFreq', 200.0), ('beatFreq', 4.0),
                    ('rampPercent', 0.2), ('gapPercent', 0.15), ('pan', 0.0)
                ],
                "transition": [ 
                    ('amp', 0.5), ('startBaseFreq', 200.0), ('endBaseFreq', 200.0), 
                    ('startBeatFreq', 4.0), ('endBeatFreq', 4.0), 
                    ('rampPercent', 0.2), ('gapPercent', 0.15), ('pan', 0.0)
                ]
            },
            "qam_beat": { # CORRECTED AND COMPLETED for qam_beat based on qam_beat.py
                "standard": [
                    ('ampL', 0.5), ('ampR', 0.5),
                    ('baseFreqL', 200.0), ('baseFreqR', 204.0),
                    ('qamAmFreqL', 4.0), ('qamAmDepthL', 0.5), ('qamAmPhaseOffsetL', 0.0),
                    ('qamAmFreqR', 4.0), ('qamAmDepthR', 0.5), ('qamAmPhaseOffsetR', 0.0),
                    ('qamAm2FreqL', 0.0), ('qamAm2DepthL', 0.0), ('qamAm2PhaseOffsetL', 0.0),
                    ('qamAm2FreqR', 0.0), ('qamAm2DepthR', 0.0), ('qamAm2PhaseOffsetR', 0.0),
                    ('modShapeL', 1.0), ('modShapeR', 1.0),
                    ('crossModDepth', 0.0), ('crossModDelay', 0.0),
                    ('harmonicDepth', 0.0), ('harmonicRatio', 2.0),
                    ('subHarmonicFreq', 0.0), ('subHarmonicDepth', 0.0),
                    ('startPhaseL', 0.0), ('startPhaseR', 0.0),
                    ('phaseOscFreq', 0.0), ('phaseOscRange', 0.0), ('phaseOscPhaseOffset', 0.0),
                    ('beatingSidebands', False), ('sidebandOffset', 1.0), ('sidebandDepth', 0.1),
                    ('attackTime', 0.0), ('releaseTime', 0.0)
                ],
                "transition": [
                    ('startAmpL', 0.5), ('endAmpL', 0.5),
                    ('startAmpR', 0.5), ('endAmpR', 0.5),
                    ('startBaseFreqL', 200.0), ('endBaseFreqL', 200.0),
                    ('startBaseFreqR', 204.0), ('endBaseFreqR', 204.0),
                    ('startQamAmFreqL', 4.0), ('endQamAmFreqL', 4.0),
                    ('startQamAmDepthL', 0.5), ('endQamAmDepthL', 0.5),
                    ('startQamAmPhaseOffsetL', 0.0), ('endQamAmPhaseOffsetL', 0.0),
                    ('startQamAmFreqR', 4.0), ('endQamAmFreqR', 4.0),
                    ('startQamAmDepthR', 0.5), ('endQamAmDepthR', 0.5),
                    ('startQamAmPhaseOffsetR', 0.0), ('endQamAmPhaseOffsetR', 0.0),
                    ('startQamAm2FreqL', 0.0), ('endQamAm2FreqL', 0.0),
                    ('startQamAm2DepthL', 0.0), ('endQamAm2DepthL', 0.0),
                    ('startQamAm2PhaseOffsetL', 0.0), ('endQamAm2PhaseOffsetL', 0.0),
                    ('startQamAm2FreqR', 0.0), ('endQamAm2FreqR', 0.0),
                    ('startQamAm2DepthR', 0.0), ('endQamAm2DepthR', 0.0),
                    ('startQamAm2PhaseOffsetR', 0.0), ('endQamAm2PhaseOffsetR', 0.0),
                    ('startModShapeL', 1.0), ('endModShapeL', 1.0),
                    ('startModShapeR', 1.0), ('endModShapeR', 1.0),
                    ('startCrossModDepth', 0.0), ('endCrossModDepth', 0.0),
                    ('startHarmonicDepth', 0.0), ('endHarmonicDepth', 0.0),
                    ('startSubHarmonicFreq', 0.0), ('endSubHarmonicFreq', 0.0),
                    ('startSubHarmonicDepth', 0.0), ('endSubHarmonicDepth', 0.0),
                    ('startStartPhaseL', 0.0), ('endStartPhaseL', 0.0), # Corresponds to 'startPhaseL' in qam_beat
                    ('startStartPhaseR', 0.0), ('endStartPhaseR', 0.0), # Corresponds to 'startPhaseR' in qam_beat
                    ('startPhaseOscFreq', 0.0), ('endPhaseOscFreq', 0.0),
                    ('startPhaseOscRange', 0.0), ('endPhaseOscRange', 0.0),
                    # Static parameters for transition mode (values are fixed, not interpolated)
                    ('crossModDelay', 0.0),
                    ('harmonicRatio', 2.0),
                    ('phaseOscPhaseOffset', 0.0),
                    ('beatingSidebands', False),
                    ('sidebandOffset', 1.0),
                    ('sidebandDepth', 0.1),
                    ('attackTime', 0.0),
                    ('releaseTime', 0.0)
                ]
            },
            "hybrid_qam_monaural_beat": { # This is an example, ensure it's correct
                "standard": [
                    ('ampL', 0.5), ('ampR', 0.5),
                    ('qamCarrierFreqL', 100.0), ('qamAmFreqL', 4.0), ('qamAmDepthL', 0.5),
                    ('qamAmPhaseOffsetL', 0.0), ('qamStartPhaseL', 0.0),
                    ('monoCarrierFreqR', 100.0), ('monoBeatFreqInChannelR', 4.0),
                    ('monoAmDepthR', 0.0), ('monoAmFreqR', 0.0), ('monoAmPhaseOffsetR', 0.0),
                    ('monoFmRangeR', 0.0), ('monoFmFreqR', 0.0), ('monoFmPhaseOffsetR', 0.0),
                    ('monoStartPhaseR_Tone1', 0.0), ('monoStartPhaseR_Tone2', 0.0),
                    ('monoPhaseOscFreqR', 0.0), ('monoPhaseOscRangeR', 0.0), ('monoPhaseOscPhaseOffsetR', 0.0)
                ],
                "transition": [ 
                    ('startAmpL', 0.5), ('endAmpL', 0.5),
                    ('startAmpR', 0.5), ('endAmpR', 0.5),
                    ('startQamCarrierFreqL', 100.0), ('endQamCarrierFreqL', 100.0),
                    ('startQamAmFreqL', 4.0), ('endQamAmFreqL', 4.0),
                    ('startQamAmDepthL', 0.5), ('endQamAmDepthL', 0.5),
                    ('startQamAmPhaseOffsetL', 0.0), ('endQamAmPhaseOffsetL', 0.0),
                    ('startQamStartPhaseL', 0.0), ('endQamStartPhaseL', 0.0),
                    ('startMonoCarrierFreqR', 100.0), ('endMonoCarrierFreqR', 100.0),
                    ('startMonoBeatFreqInChannelR', 4.0), ('endMonoBeatFreqInChannelR', 4.0),
                    ('startMonoAmDepthR', 0.0), ('endMonoAmDepthR', 0.0),
                    ('startMonoAmFreqR', 0.0), ('endMonoAmFreqR', 0.0),
                    ('startMonoAmPhaseOffsetR', 0.0), ('endMonoAmPhaseOffsetR', 0.0),
                    ('startMonoFmRangeR', 0.0), ('endMonoFmRangeR', 0.0),
                    ('startMonoFmFreqR', 0.0), ('endMonoFmFreqR', 0.0),
                    ('startMonoFmPhaseOffsetR', 0.0), ('endMonoFmPhaseOffsetR', 0.0),
                    ('startMonoStartPhaseR_Tone1', 0.0), ('endMonoStartPhaseR_Tone1', 0.0),
                    ('startMonoStartPhaseR_Tone2', 0.0), ('endMonoStartPhaseR_Tone2', 0.0),
                    ('startMonoPhaseOscFreqR', 0.0), ('endMonoPhaseOscFreqR', 0.0),
                    ('startMonoPhaseOscRangeR', 0.0), ('endMonoPhaseOscRangeR', 0.0),
                    ('startMonoPhaseOscPhaseOffsetR', 0.0), ('endMonoPhaseOscPhaseOffsetR', 0.0)
                ]
            }
        }
        # --- End of param_definitions ---

        ordered_params = OrderedDict()
        definition_set = param_definitions.get(base_func_key)
        
        # If func_name_from_combo itself is a transition func (e.g. "binaural_beat_transition")
        # then is_transition_mode might be overridden by this fact.
        # The self.transition_check usually dictates the mode.
        effective_func_name_for_lookup = func_name_from_combo # The name from combo
        effective_is_transition = is_transition_mode

        # If the selected combo item ALREADY implies transition (e.g., "func_transition")
        # then we should look for its base definition and use "transition" params.
        if effective_func_name_for_lookup.endswith("_transition"):
            potential_base_key = effective_func_name_for_lookup[:-len("_transition")]
            if potential_base_key in param_definitions:
                definition_set = param_definitions.get(potential_base_key)
                effective_is_transition = True # Force transition mode if function name implies it
            else: # No base key found, try the full name if it exists
                definition_set = param_definitions.get(effective_func_name_for_lookup)
        else: # Not a name ending with _transition
            definition_set = param_definitions.get(effective_func_name_for_lookup)


        if not definition_set:
            print(f"Warning: No parameter definitions found for function '{effective_func_name_for_lookup}' (derived from combo '{func_name_from_combo}').")
            # Try to get params by direct introspection as a fallback
            if hasattr(sound_creator, 'get_synth_params'):
                 raw_params = sound_creator.get_synth_params(effective_func_name_for_lookup) # This gets signature defaults
                 for name, default_val in raw_params.items():
                     if default_val == inspect.Parameter.empty: # No default in signature
                         # Try a sensible default based on type hint or name
                         if any(s in name.lower() for s in ['freq','rate']): default_val = 440.0
                         elif any(s in name.lower() for s in ['amp','depth','gain','level']): default_val = 0.5
                         elif any(s in name.lower() for s in ['pan']): default_val = 0.0
                         elif 'bool' in name.lower() or 'enable' in name.lower(): default_val = False
                         else: default_val = 0 # Generic numeric default
                     ordered_params[name] = default_val
                 if ordered_params:
                     print(f"Note: Using introspected params for '{effective_func_name_for_lookup}' as fallback.")
                     return ordered_params

            return ordered_params # Return empty if no definition and no introspection fallback

        selected_mode_key = "transition" if effective_is_transition else "standard"
        params_list = definition_set.get(selected_mode_key)
        
        if not params_list: # Fallback if specific mode (e.g. "transition") is missing for a base key
            if selected_mode_key == "transition" and "standard" in definition_set:
                print(f"Warning: No 'transition' parameters for '{base_func_key}'. Using 'standard' parameters instead even if transition is checked.")
                params_list = definition_set.get("standard")
            if not params_list:
                print(f"Warning: No parameters found for '{base_func_key}' in mode '{selected_mode_key}'.")
                return ordered_params

        for name, default_val in params_list:
            ordered_params[name] = default_val
            
        return ordered_params

    @pyqtSlot()
    def save_voice(self):
        new_synth_params = {}
        error_occurred = False
        validation_errors = []

        # Collect Synth Parameters
        for name, data in self.param_widgets.items():
            widget, param_type = data['widget'], data['type']
            value = None
            widget.setStyleSheet("") # Clear previous error styles

            if isinstance(widget, QCheckBox):
                value = widget.isChecked()
            elif isinstance(widget, QComboBox):
                value_str = widget.currentText()
                if name == 'noiseType' and param_type == 'int': # Specific handling
                    try: value = int(value_str)
                    except ValueError: error_occurred=True; validation_errors.append(f"Invalid int for '{name}': {value_str}"); widget.setStyleSheet("border: 1px solid red;")
                else: # For other combos like pathShape (string)
                    value = value_str
            elif isinstance(widget, QLineEdit):
                value_str = widget.text().strip()
                if not value_str and param_type != 'str': # Allow empty strings if type is str, otherwise might be error or None
                     # Check if param can be None or needs a value
                     # For now, let's assume None is okay if empty and not string.
                     # Or, find default for this param:
                     # default_val_for_name = self._get_default_params(self.synth_func_combo.currentText(), self.transition_check.isChecked()).get(name)
                     # if default_val_for_name is not None: value = default_val_for_name
                     # else: # param cannot be None and is empty
                     #    error_occurred=True; validation_errors.append(f"Parameter '{name}' cannot be empty."); widget.setStyleSheet("border: 1px solid red;")
                     value = None # Allow None if field is empty (synth function should handle None with defaults)
                elif value_str: # Only parse if not empty
                    try:
                        if param_type == 'int': value = int(value_str)
                        elif param_type == 'float': value = float(value_str.replace(',', '.')) # Allow comma for float
                        else: value = value_str # String type
                    except ValueError:
                        error_occurred=True; validation_errors.append(f"Invalid {param_type} for '{name}': {value_str}"); widget.setStyleSheet("border: 1px solid red;")
            
            if value is not None: # Only add if value is set (allows params to be omitted if they are None)
                new_synth_params[name] = value
            elif value_str == "" and param_type == 'str': # explicitly save empty string if it's a string type
                new_synth_params[name] = ""


        # Collect Envelope Parameters
        new_envelope_data = None
        selected_env_type = self.env_type_combo.currentText()
        if selected_env_type != ENVELOPE_TYPE_NONE:
            new_env_params = {}
            for name, data in self.envelope_param_widgets.items():
                widget, param_type = data['widget'], data['type']
                value = None
                widget.setStyleSheet("") # Clear error styles
                if isinstance(widget, QLineEdit):
                    value_str = widget.text().strip()
                    if not value_str:
                        error_occurred=True; validation_errors.append(f"Envelope parameter '{name}' cannot be empty."); widget.setStyleSheet("border: 1px solid red;")
                    else:
                        try:
                            if param_type == 'float': value = float(value_str.replace(',', '.'))
                            elif param_type == 'int': value = int(value_str) # If any int env params
                            else: value = value_str
                            # Example validation for envelope params
                            if 'amp' in name.lower() and not (0.0 <= value <= 1.0):
                                validation_errors.append(f"Envelope amp '{name}' ({value}) out of 0-1 range (warning).") # Non-blocking
                            if 'dur' in name.lower() and value < 0:
                                error_occurred=True; validation_errors.append(f"Envelope duration '{name}' ({value}) cannot be negative."); widget.setStyleSheet("border: 1px solid red;")

                        except ValueError:
                            error_occurred=True; validation_errors.append(f"Invalid envelope {param_type} for '{name}': {value_str}"); widget.setStyleSheet("border: 1px solid red;")
                if value is not None: new_env_params[name] = value
            
            if not any(f"Envelope parameter '{n}'" in e for n in self.envelope_param_widgets for e in validation_errors if "cannot be empty" in e): # check if fatal error occurred
                new_envelope_data = {"type": selected_env_type, "params": new_env_params}

        if error_occurred:
            QMessageBox.warning(self, "Input Error", "Please correct highlighted fields:\n\n" + "\n".join(validation_errors))
            return

        # Final voice data structure for main.py
        final_voice_data = {
            "synth_function_name": self.synth_func_combo.currentText(),
            "is_transition": self.transition_check.isChecked(),
            "params": new_synth_params,
            "volume_envelope": new_envelope_data,  # This is the correct key for main.py
            "description": self.current_voice_data.get("description", ""),
        }
        
        # Update the main application's track_data
        try:
            target_step = self.app.track_data["steps"][self.step_index]
            if "voices" not in target_step: # Should exist if step exists
                target_step["voices"] = []
            
            if self.is_new_voice:
                target_step["voices"].append(final_voice_data)
            elif 0 <= self.voice_index < len(target_step["voices"]):
                target_step["voices"][self.voice_index] = final_voice_data
            else: # Should not happen if initialized correctly
                QMessageBox.critical(self.app, "Save Error", "Voice index out of bounds during save.")
                self.reject() # Close without saving
                return
            
            self.accept() # Signal to main.py that dialog was accepted (and data is ready)
        except IndexError:
            QMessageBox.critical(self.app, "Save Error", "Failed to save voice (step index issue). Check track data integrity.")
            self.reject()
        except Exception as e:
            QMessageBox.critical(self.app, "Save Error", f"An unexpected error occurred while saving voice data:\n{e}")
            traceback.print_exc()
            self.reject()

    #get_voice_data is not strictly needed if main.py updates on accept() by re-reading from track_data
    #but if main.py explicitly calls dialog.get_voice_data(), it should return the final structure.
    def get_voice_data(self):
        # This would be called by main.py if it needs the data explicitly after accept()
        # For this version, main.py modifies its own track_data directly in save_voice() before accept()
        # So, this method can just return the self.current_voice_data which was updated.
        # To be more robust, it should return what was actually decided to be saved.
        # However, the original save_voice directly modifies self.app.track_data.
        # Let's assume main.py will refresh from its track_data.
        # If main.py *does* call this, current_voice_data is what was loaded/being edited.
        # A better pattern for dialogs is to return the *new* data, not modify external data directly.
        # But sticking to "original version" structure for save_voice.
        return self.current_voice_data # This reflects data at dialog open or after UI changes if not saved yet.
                                      # For the "saved" data, main.py should re-read its own track_data.