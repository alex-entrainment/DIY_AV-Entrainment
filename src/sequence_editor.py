
import sys
import os
import json
import copy

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QAction,
    QFileDialog,
    QMessageBox,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QFormLayout,
    QSpinBox,
    QDoubleSpinBox,
    QLineEdit,
    QComboBox,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QLabel,
    QCheckBox,
    QStatusBar,
    QTabWidget,
    QSlider,
    QScrollArea
)
from PyQt5.QtCore import Qt

from sequence_model import (
    Step,
    Oscillator,
    StrobeSet,
    Waveform,
    AudioSettings,
    AudioCarrier,
    PatternMode
)

from audio_generator import generate_audio_file_for_steps_offline_rfm

# A mapping from PatternMode enum -> human-readable text
PATTERN_MODE_LABELS = [
    ("NONE", PatternMode.NONE),
    ("SACRED_GEOMETRY", PatternMode.SACRED_GEOMETRY),
    ("FRACTAL_ARC", PatternMode.FRACTAL_ARC),
    ("PHI_SPIRAL", PatternMode.PHI_SPIRAL)
]

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("6-LED Sequence Editor with Stepwise Audio Sync")
        self.resize(1300, 700)

        self.steps = []
        self.currentFile = None
        self.audio_settings = AudioSettings()

        # Create central widget and main layout
        central = QWidget(self)
        main_layout = QHBoxLayout(central)
        self.setCentralWidget(central)

        # ---------------------------
        # LEFT: Step List & Buttons
        # ---------------------------
        left_side_layout = QVBoxLayout()

        self.step_list = QListWidget()
        self.step_list.setStyleSheet("""
            QListWidget::item {
                border: 1px solid gray;
                margin: 1px;
                padding: 3px;
            }
            QListWidget::item:selected {
                background-color: #b0d4f1;
            }
        """)
        self.step_list.currentRowChanged.connect(self.on_step_selected)
        left_side_layout.addWidget(self.step_list)

        # Buttons for step management
        hlayout1 = QHBoxLayout()
        self.btn_add_step = QPushButton("Add Step")
        self.btn_add_step.clicked.connect(self.add_step)
        hlayout1.addWidget(self.btn_add_step)

        self.btn_duplicate_step = QPushButton("Duplicate Step")
        self.btn_duplicate_step.clicked.connect(self.duplicate_step)
        hlayout1.addWidget(self.btn_duplicate_step)
        left_side_layout.addLayout(hlayout1)

        hlayout2 = QHBoxLayout()
        self.btn_remove_step = QPushButton("Remove Step")
        self.btn_remove_step.clicked.connect(self.remove_step)
        hlayout2.addWidget(self.btn_remove_step)
        left_side_layout.addLayout(hlayout2)

        hlayout3 = QHBoxLayout()
        self.btn_move_up = QPushButton("Move Up")
        self.btn_move_up.clicked.connect(self.move_step_up)
        hlayout3.addWidget(self.btn_move_up)
        self.btn_move_down = QPushButton("Move Down")
        self.btn_move_down.clicked.connect(self.move_step_down)
        hlayout3.addWidget(self.btn_move_down)
        left_side_layout.addLayout(hlayout3)

        left_side_layout.addStretch()

        # Add the left-side layout to main layout
        main_layout.addLayout(left_side_layout, 2)

        # ---------------------------
        # RIGHT: Step Config (Scroll)
        # ---------------------------
        # We'll embed everything in a QScrollArea so the user can scroll if it doesn't fit.
        self.step_config_scroll_area = QScrollArea()
        self.step_config_scroll_area.setWidgetResizable(True)

        # This widget will hold the step_config_layout
        self.step_config_container = QWidget()
        self.step_config_layout = QVBoxLayout(self.step_config_container)

        # 1) Step Info
        self.step_info_group = QGroupBox("Step Info")
        form_step_info = QFormLayout()
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(1, 5400)
        self.duration_spin.setValue(30)

        self.description_edit = QLineEdit()

        form_step_info.addRow("Duration (secs):", self.duration_spin)
        form_step_info.addRow("Description:", self.description_edit)
        self.step_info_group.setLayout(form_step_info)
        self.step_config_layout.addWidget(self.step_info_group)

        # 2) Oscillator Mode
        mode_group = QGroupBox("Oscillator Mode")
        hlayout_mode = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Combined", "Split", "Independent"])
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        hlayout_mode.addWidget(QLabel("Select Mode:"))
        hlayout_mode.addWidget(self.mode_combo)
        mode_group.setLayout(hlayout_mode)
        self.step_config_layout.addWidget(mode_group)

        # 3) Oscillator Controls container
        self.osc_group = QGroupBox("Oscillator Settings")
        self.osc_layout = QVBoxLayout()
        self.osc_group.setLayout(self.osc_layout)
        self.step_config_layout.addWidget(self.osc_group)

        self.btn_apply_all = QPushButton("Apply Oscillator 1 Settings to All")
        self.btn_apply_all.clicked.connect(self.apply_osc1_to_all)
        self.step_config_layout.addWidget(self.btn_apply_all)

        # 4) Strobe Intensities
        self.strobe_group = QGroupBox("Strobe Intensities")
        self.strobe_layout = QHBoxLayout()
        self.strobe_group.setLayout(self.strobe_layout)
        self.step_config_layout.addWidget(self.strobe_group)

        # 5) Submit Step
        self.btn_submit_step = QPushButton("Submit Step Settings")
        self.btn_submit_step.clicked.connect(self.on_submit_step)
        self.step_config_layout.addWidget(self.btn_submit_step)

        # Add a stretch so we can push everything up
        self.step_config_layout.addStretch()

        # Put the step_config_container into the scroll area
        self.step_config_scroll_area.setWidget(self.step_config_container)

        # Add the scroll area to main layout
        main_layout.addWidget(self.step_config_scroll_area, 3)

        # ---------------------------
        # GLOBAL AUDIO SETTINGS
        # ---------------------------
        # We'll keep your existing logic or place it below in a QTabWidget, etc.
        # For brevity, let's place it in a separate QVBoxLayout to the right or add to main_layout.
        # But you can do the same approach (scroll area) if it also can become tall.

        audio_side_layout = QVBoxLayout()

        self.audio_tabs = QTabWidget()
        self.carrier_panels = []
        for i in range(3):
            panel = self.create_carrier_panel(i+1)
            self.audio_tabs.addTab(panel, f"Carrier {i+1}")
            self.carrier_panels.append(panel)

        self.global_audio_panel = self.create_global_audio_panel()
        self.audio_tabs.addTab(self.global_audio_panel, "Global Settings")

        audio_side_layout.addWidget(self.audio_tabs)
        main_layout.addLayout(audio_side_layout, 3)

        # Menubar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        new_act = QAction("New", self)
        new_act.triggered.connect(self.new_sequence)
        file_menu.addAction(new_act)

        open_act = QAction("Open", self)
        open_act.triggered.connect(self.load_sequence)
        file_menu.addAction(open_act)

        save_act = QAction("Save", self)
        save_act.triggered.connect(self.save_sequence)
        file_menu.addAction(save_act)

        save_as_act = QAction("Save As", self)
        save_as_act.triggered.connect(self.save_sequence_as)
        file_menu.addAction(save_as_act)

        delete_act = QAction("Delete", self)
        delete_act.triggered.connect(self.delete_sequence_file)
        file_menu.addAction(delete_act)

        # Status Bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.update_sequence_duration()

        # Initialize containers for oscillator/strobe controls
        self.osc_controls = []
        self.strobe_controls = []

        self.on_mode_changed()

        # Start with a default step
        self.add_step()

    # ---------------------------------------------------------------
    # Create the carriers & global audio panel
    # ---------------------------------------------------------------
    def create_carrier_panel(self, carrier_num):
        carrier_group = QGroupBox(f"Carrier {carrier_num} Settings")
        carrier_form = QFormLayout()
    
        enabled_cb = QCheckBox("Enable Carrier")
        enabled_cb.setChecked(carrier_num == 1)  # first carrier enabled by default
        carrier_form.addRow(enabled_cb)
    
        # Base frequency settings (for backward compatibility)
        freq_label = QLabel("Base Frequency (Hz):")
        freq_label.setToolTip("Legacy setting - used if channel frequencies aren't specified")
        carrier_form.addRow(freq_label)
        
        start_freq_spin = QDoubleSpinBox()
        start_freq_spin.setRange(20.0, 1000.0)
        start_freq_spin.setValue(200.0)
        carrier_form.addRow("Base Start Freq:", start_freq_spin)
    
        end_freq_spin = QDoubleSpinBox()
        end_freq_spin.setRange(20.0, 1000.0)
        end_freq_spin.setValue(200.0)
        carrier_form.addRow("Base End Freq:", end_freq_spin)
    
        # Add a separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        carrier_form.addRow(separator)
    
        # Channel-specific frequency settings
        channel_label = QLabel("Channel Frequencies (Hz):")
        channel_label.setToolTip("Independent left/right channel frequencies for precise binaural control")
        carrier_form.addRow(channel_label)
        
        # Left channel
        start_freq_left_spin = QDoubleSpinBox()
        start_freq_left_spin.setRange(20.0, 1000.0)
        start_freq_left_spin.setValue(205.0 if carrier_num == 1 else 200.0)  # Default slightly higher for left
        carrier_form.addRow("Left Start Freq:", start_freq_left_spin)
    
        end_freq_left_spin = QDoubleSpinBox()
        end_freq_left_spin.setRange(20.0, 1000.0)
        end_freq_left_spin.setValue(205.0 if carrier_num == 1 else 200.0)
        carrier_form.addRow("Left End Freq:", end_freq_left_spin)
    
        # Right channel
        start_freq_right_spin = QDoubleSpinBox()
        start_freq_right_spin.setRange(20.0, 1000.0)
        start_freq_right_spin.setValue(195.0 if carrier_num == 1 else 200.0)  # Default slightly lower for right
        carrier_form.addRow("Right Start Freq:", start_freq_right_spin)
    
        end_freq_right_spin = QDoubleSpinBox()
        end_freq_right_spin.setRange(20.0, 1000.0)
        end_freq_right_spin.setValue(195.0 if carrier_num == 1 else 200.0)
        carrier_form.addRow("Right End Freq:", end_freq_right_spin)
    
        # Calculate and display the binaural beat frequency
        beat_label = QLabel("Binaural: 10.0 Hz")
        
        def update_beat_label():
            left_start = start_freq_left_spin.value()
            right_start = start_freq_right_spin.value()
            left_end = end_freq_left_spin.value()
            right_end = end_freq_right_spin.value()
            
            start_diff = abs(left_start - right_start)
            end_diff = abs(left_end - right_end)
            
            if start_diff == end_diff:
                beat_label.setText(f"Binaural: {start_diff:.1f} Hz")
            else:
                beat_label.setText(f"Binaural: {start_diff:.1f} → {end_diff:.1f} Hz")
        
        start_freq_left_spin.valueChanged.connect(update_beat_label)
        end_freq_left_spin.valueChanged.connect(update_beat_label)
        start_freq_right_spin.valueChanged.connect(update_beat_label)
        end_freq_right_spin.valueChanged.connect(update_beat_label)
        
        carrier_form.addRow("Beat Frequency:", beat_label)
    
        # Add another separator
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setFrameShadow(QFrame.Sunken)
        carrier_form.addRow(separator2)
    
        # Volume control
        volume_slider = QSlider(Qt.Horizontal)
        volume_slider.setRange(0, 100)
        volume_slider.setValue(100 if carrier_num == 1 else 50)
        volume_label = QLabel("100%" if carrier_num == 1 else "50%")
        volume_slider.valueChanged.connect(lambda val: volume_label.setText(f"{val}%"))
    
        volume_layout = QHBoxLayout()
        volume_layout.addWidget(volume_slider)
        volume_layout.addWidget(volume_label)
        carrier_form.addRow("Volume:", volume_layout)
    
        # RFM settings
        rfm_cb = QCheckBox("Enable RFM")
        rfm_cb.setChecked(False)
        
        rfm_help_btn = QPushButton("?")
        rfm_help_btn.setMaximumWidth(30)
        rfm_help_btn.clicked.connect(self.show_rfm_help)
    
        rfm_layout = QHBoxLayout()
        rfm_layout.addWidget(rfm_cb)
        rfm_layout.addStretch()
        rfm_layout.addWidget(rfm_help_btn)
        
        carrier_form.addRow(rfm_layout)
    
        rfm_range_spin = QDoubleSpinBox()
        rfm_range_spin.setRange(0.0, 20.0)
        rfm_range_spin.setValue(0.5)
        carrier_form.addRow("RFM Range (±Hz):", rfm_range_spin)
    
        rfm_speed_spin = QDoubleSpinBox()
        rfm_speed_spin.setRange(0.0, 5.0)
        rfm_speed_spin.setValue(0.2)
        carrier_form.addRow("RFM Speed (Hz/s):", rfm_speed_spin)
    
        # Apply to panel
        carrier_group.setLayout(carrier_form)
    
        # Store references to the controls
        carrier_group.enabled = enabled_cb
        carrier_group.start_freq = start_freq_spin
        carrier_group.end_freq = end_freq_spin
        
        # New channel-specific controls
        carrier_group.start_freq_left = start_freq_left_spin
        carrier_group.end_freq_left = end_freq_left_spin
        carrier_group.start_freq_right = start_freq_right_spin
        carrier_group.end_freq_right = end_freq_right_spin
        carrier_group.beat_label = beat_label
        
        carrier_group.volume = volume_slider
        carrier_group.volume_label = volume_label
        carrier_group.rfm_enable = rfm_cb
        carrier_group.rfm_range = rfm_range_spin
        carrier_group.rfm_speed = rfm_speed_spin
    
        return carrier_group

    def create_global_audio_panel(self):
        global_group = QGroupBox("Global Audio Settings")
        global_form = QFormLayout()

        enabled_cb = QCheckBox("Enable Audio")
        global_form.addRow(enabled_cb)

        beat_spin = QDoubleSpinBox()
        beat_spin.setRange(0.1, 100.0)
        beat_spin.setValue(10.0)
        global_form.addRow("Beat Freq (Hz) (fallback):", beat_spin)

        binaural_cb = QCheckBox("Binaural")
        binaural_cb.setChecked(True)
        global_form.addRow(binaural_cb)

        isochronic_cb = QCheckBox("Isochronic")
        isochronic_cb.setChecked(False)
        global_form.addRow(isochronic_cb)

        def update_binaural(state):
            if state == Qt.Checked:
                isochronic_cb.setChecked(False)
        def update_isochronic(state):
            if state == Qt.Checked:
                binaural_cb.setChecked(False)
        binaural_cb.stateChanged.connect(update_binaural)
        isochronic_cb.stateChanged.connect(update_isochronic)

        global_rfm_cb = QCheckBox("Enable Global RFM")
        global_form.addRow(global_rfm_cb)

        global_rfm_range = QDoubleSpinBox()
        global_rfm_range.setRange(0.0, 10.0)
        global_rfm_range.setValue(0.5)
        global_form.addRow("Global RFM Range (±Hz):", global_rfm_range)

        global_rfm_speed = QDoubleSpinBox()
        global_rfm_speed.setRange(0.0, 2.0)
        global_rfm_speed.setValue(0.2)
        global_form.addRow("Global RFM Speed (Hz/s):", global_rfm_speed)

        pink_noise_cb = QCheckBox("Enable Pink Noise")
        global_form.addRow(pink_noise_cb)

        pink_noise_slider = QSlider(Qt.Horizontal)
        pink_noise_slider.setRange(0, 100)
        pink_noise_slider.setValue(10)
        pink_noise_label = QLabel("10%")
        pink_noise_slider.valueChanged.connect(lambda val: pink_noise_label.setText(f"{val}%"))
        pn_layout = QHBoxLayout()
        pn_layout.addWidget(pink_noise_slider)
        pn_layout.addWidget(pink_noise_label)
        global_form.addRow("Pink Noise Volume:", pn_layout)

        sample_rate_spin = QSpinBox()
        sample_rate_spin.setRange(8000, 192000)
        sample_rate_spin.setValue(44100)
        sample_rate_spin.setSingleStep(1000)
        global_form.addRow("Sample Rate (Hz):", sample_rate_spin)

        global_group.setLayout(global_form)

        global_group.enabled = enabled_cb
        global_group.beat_freq = beat_spin
        global_group.binaural = binaural_cb
        global_group.isochronic = isochronic_cb
        global_group.global_rfm_enable = global_rfm_cb
        global_group.global_rfm_range = global_rfm_range
        global_group.global_rfm_speed = global_rfm_speed
        global_group.pink_noise_enable = pink_noise_cb
        global_group.pink_noise_volume = pink_noise_slider
        global_group.pink_noise_label = pink_noise_label
        global_group.sample_rate = sample_rate_spin

        return global_group

    # ---------------------------------------------------------------
    # Oscillator mode & UI
    # ---------------------------------------------------------------

    def show_rfm_help(self):
        msg = QMessageBox()
        msg.setWindowTitle("RFM Parameter Guidelines")
        msg.setText("""<h3>Random Frequency Modulation Tips</h3>
        <p>RFM adds natural variation to frequencies to prevent habituation.</p>
        
        <p><b>Guidelines for stable operation:</b></p>
        <ul>
            <li><b>RFM Range</b>: Keep under 25% of your lowest frequency</li>
            <li><b>RFM Speed</b>: Lower values (0.1-0.3) work best for most purposes</li>
            <li><b>During Transitions</b>: Consider using smaller RFM ranges for steps 
            that include frequency transitions</li>
        </ul>
        
        <p>The system automatically adjusts RFM impact during rapid frequency 
        changes to prevent irregular patterns.</p>""")
        msg.exec_()

    def on_mode_changed(self):
        """Rebuild the oscillator and strobe controls when the user changes the mode."""
        mode = self.mode_combo.currentText()

        # Clear old controls
        self.clear_layout(self.osc_layout)
        self.clear_layout(self.strobe_layout)
        self.osc_controls = []
        self.strobe_controls = []

        if mode == "Combined":
            num_osc = 1
            osc_labels = ["Oscillator (All 6 LEDs)"]
            strobe_labels = ["Strobe (All 6 LEDs)"]
        elif mode == "Split":
            num_osc = 2
            osc_labels = [
                "Osc Group 1 (LEDs 0,2,4)",
                "Osc Group 2 (LEDs 1,3,5)"
            ]
            strobe_labels = [
                "Strobe (LEDs 0,2,4)",
                "Strobe (LEDs 1,3,5)"
            ]
        else:  # Independent
            num_osc = 6
            osc_labels = [f"Oscillator LED {i}" for i in range(6)]
            strobe_labels = [f"Strobe LED {i}" for i in range(6)]

        # Build oscillator controls
        for i in range(num_osc):
            group = QGroupBox(osc_labels[i])
            form = QFormLayout()

            wave_combo = QComboBox()
            wave_combo.addItems(["Off", "Square", "Sine"])

            freq_start = QDoubleSpinBox()
            freq_start.setRange(0.1, 200.0)
            freq_start.setDecimals(2)
            freq_start.setValue(12.0)

            freq_end = QDoubleSpinBox()
            freq_end.setRange(0.1, 200.0)
            freq_end.setDecimals(2)
            freq_end.setValue(12.0)

            duty_start = QSpinBox()
            duty_start.setRange(1, 99)
            duty_start.setValue(50)

            duty_end = QSpinBox()
            duty_end.setRange(1, 99)
            duty_end.setValue(50)

            rfm_enable = QCheckBox("Enable RFM (LED)")
            rfm_enable.setChecked(False)
            rfm_range = QDoubleSpinBox()
            rfm_range.setRange(0.0, 5.0)
            rfm_range.setValue(0.5)
            rfm_speed = QDoubleSpinBox()
            rfm_speed.setRange(0.0, 2.0)
            rfm_speed.setValue(0.2)

            rfm_help_btn = QPushButton("?")
            rfm_help_btn.setMaximumWidth(30)
            rfm_help_btn.clicked.connect(self.show_rfm_help)

            rfm_layout = QHBoxLayout()
            rfm_layout.addWidget(rfm_enable)
            rfm_layout.addStretch()
            rfm_layout.addWidget(rfm_help_btn)

            form.addRow("Waveform:", wave_combo)
            form.addRow("Freq Start (Hz):", freq_start)
            form.addRow("Freq End (Hz):", freq_end)
            form.addRow("Duty Start (%):", duty_start)
            form.addRow("Duty End (%):", duty_end)
            form.addRow(rfm_layout)
            form.addRow("RFM Range (±Hz):", rfm_range)
            form.addRow("RFM Speed (Hz/s):", rfm_speed)

            # Pattern controls
            phase_pattern_combo = QComboBox()
            brightness_pattern_combo = QComboBox()
            for label, pmode in PATTERN_MODE_LABELS:
                phase_pattern_combo.addItem(label)
                brightness_pattern_combo.addItem(label)

            pattern_strength_spin = QDoubleSpinBox()
            pattern_strength_spin.setRange(0.0, 10.0)
            pattern_strength_spin.setValue(1.0)

            pattern_freq_spin = QDoubleSpinBox()
            pattern_freq_spin.setRange(0.0, 10.0)
            pattern_freq_spin.setValue(1.0)

            form.addRow("Phase Pattern:", phase_pattern_combo)
            form.addRow("Brightness Pattern:", brightness_pattern_combo)
            form.addRow("Pattern Strength:", pattern_strength_spin)
            form.addRow("Pattern Frequency:", pattern_freq_spin)

            group.setLayout(form)
            self.osc_layout.addWidget(group)

            self.osc_controls.append({
                "wave_combo": wave_combo,
                "freq_start": freq_start,
                "freq_end": freq_end,
                "duty_start": duty_start,
                "duty_end": duty_end,
                "rfm_enable": rfm_enable,
                "rfm_range": rfm_range,
                "rfm_speed": rfm_speed,

                "phase_pattern_combo": phase_pattern_combo,
                "brightness_pattern_combo": brightness_pattern_combo,
                "pattern_strength_spin": pattern_strength_spin,
                "pattern_freq_spin": pattern_freq_spin
            })

        # Strobe controls
        for i in range(num_osc):
            group = QGroupBox(strobe_labels[i])
            form = QFormLayout()
            strobe_start = QSpinBox()
            strobe_start.setRange(0, 100)
            strobe_start.setValue(0)
            strobe_end = QSpinBox()
            strobe_end.setRange(0, 100)
            strobe_end.setValue(0)
            form.addRow("Start (%):", strobe_start)
            form.addRow("End (%):", strobe_end)
            group.setLayout(form)
            self.strobe_layout.addWidget(group)

            self.strobe_controls.append({
                "strobe_start": strobe_start,
                "strobe_end": strobe_end
            })

        self.btn_apply_all.setEnabled(mode == "Independent")

    def clear_layout(self, layout):
        """Utility to clear all widgets within a layout."""
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()

    # ---------------------------------------------------------------
    # Add, remove, duplicate steps
    # ---------------------------------------------------------------
    def duplicate_step(self):
        row = self.step_list.currentRow()
        if row < 0 or row >= len(self.steps):
            return
        original_step = self.steps[row]
        new_step = copy.deepcopy(original_step)
        new_step.description = f"{original_step.description} (Copy)"
        self.steps.insert(row + 1, new_step)
        item = QListWidgetItem(new_step.description)
        self.step_list.insertItem(row + 1, item)
        self.step_list.setCurrentRow(row + 1)
        self.update_sequence_duration()

    def add_step(self):
        if self.steps:
            # We create a new step that references the last step's end frequencies, etc.
            prev = self.steps[-1]
            mode = self.mode_combo.currentText()
            if mode == "Combined":
                if prev.oscillators:
                    po = prev.oscillators[0]
                else:
                    po = Oscillator(12.0,12.0,Waveform.SQUARE)
                no = Oscillator(
                    start_freq=po.end_freq,
                    end_freq=po.end_freq,
                    waveform=po.waveform,
                    start_duty=po.end_duty,
                    end_duty=po.end_duty,
                    enable_rfm=po.enable_rfm,
                    rfm_range=po.rfm_range,
                    rfm_speed=po.rfm_speed,
                    phase_pattern=po.phase_pattern,
                    brightness_pattern=po.brightness_pattern,
                    pattern_strength=po.pattern_strength,
                    pattern_freq=po.pattern_freq
                )
                oscillators = [no]
                if prev.strobe_sets:
                    ps = prev.strobe_sets[0]
                else:
                    ps = StrobeSet(list(range(6)),50,50,[1])
                ns = StrobeSet(
                    channels=list(range(6)),
                    start_intensity=ps.end_intensity,
                    end_intensity=ps.end_intensity,
                    oscillator_weights=[1.0]
                )
                strobe_sets = [ns]

            elif mode == "Split":
                oscillators = []
                for i in range(2):
                    if i < len(prev.oscillators):
                        po = prev.oscillators[i]
                    else:
                        po = Oscillator(12.0,12.0,Waveform.SQUARE)
                    no = Oscillator(
                        start_freq=po.end_freq,
                        end_freq=po.end_freq,
                        waveform=po.waveform,
                        start_duty=po.end_duty,
                        end_duty=po.end_duty,
                        enable_rfm=po.enable_rfm,
                        rfm_range=po.rfm_range,
                        rfm_speed=po.rfm_speed,
                        phase_pattern=po.phase_pattern,
                        brightness_pattern=po.brightness_pattern,
                        pattern_strength=po.pattern_strength,
                        pattern_freq=po.pattern_freq
                    )
                    oscillators.append(no)
                if len(prev.strobe_sets) >= 2:
                    ps1 = prev.strobe_sets[0]
                    ps2 = prev.strobe_sets[1]
                else:
                    ps1 = StrobeSet([0,2,4],50,50,[1,0])
                    ps2 = StrobeSet([1,3,5],50,50,[0,1])
                ns1 = StrobeSet([0,2,4], ps1.end_intensity, ps1.end_intensity,[1,0])
                ns2 = StrobeSet([1,3,5], ps2.end_intensity, ps2.end_intensity,[0,1])
                strobe_sets = [ns1, ns2]

            else:  # Independent
                oscillators = []
                strobe_sets = []
                for i in range(6):
                    if i < len(prev.oscillators):
                        po = prev.oscillators[i]
                    else:
                        po = Oscillator(12.0,12.0,Waveform.SQUARE)
                    no = Oscillator(
                        start_freq=po.end_freq,
                        end_freq=po.end_freq,
                        waveform=po.waveform,
                        start_duty=po.end_duty,
                        end_duty=po.end_duty,
                        enable_rfm=po.enable_rfm,
                        rfm_range=po.rfm_range,
                        rfm_speed=po.rfm_speed,
                        phase_pattern=po.phase_pattern,
                        brightness_pattern=po.brightness_pattern,
                        pattern_strength=po.pattern_strength,
                        pattern_freq=po.pattern_freq
                    )
                    oscillators.append(no)
                    weights = [0]*6
                    weights[i] = 1
                    if i < len(prev.strobe_sets):
                        ps = prev.strobe_sets[i]
                        ns = StrobeSet([i], ps.end_intensity, ps.end_intensity, weights)
                    else:
                        ns = StrobeSet([i],50,50,weights)
                    strobe_sets.append(ns)

            step = Step(30, "New Step", oscillators, strobe_sets)
        else:
            # no steps yet
            mode = self.mode_combo.currentText()
            if mode == "Combined":
                oscillators = [Oscillator(12.0,12.0,Waveform.SQUARE)]
                strobe_sets = [StrobeSet(list(range(6)),50,50,[1])]
            elif mode == "Split":
                osc1 = Oscillator(12.0,12.0,Waveform.SQUARE)
                osc2 = Oscillator(12.0,12.0,Waveform.SQUARE)
                oscillators = [osc1, osc2]
                s1 = StrobeSet([0,2,4],50,50,[1,0])
                s2 = StrobeSet([1,3,5],50,50,[0,1])
                strobe_sets = [s1, s2]
            else:
                oscillators = []
                strobe_sets = []
                for i in range(6):
                    osc = Oscillator(12.0,12.0,Waveform.SQUARE)
                    oscillators.append(osc)
                    weights = [0]*6
                    weights[i] = 1
                    s = StrobeSet([i],50,50,weights)
                    strobe_sets.append(s)
            step = Step(30, "New Step", oscillators, strobe_sets)

        self.steps.append(step)
        item = QListWidgetItem(step.description)
        self.step_list.addItem(item)
        self.step_list.setCurrentRow(self.step_list.count()-1)
        self.update_sequence_duration()

    def remove_step(self):
        row = self.step_list.currentRow()
        if row < 0:
            return
        self.steps.pop(row)
        self.step_list.takeItem(row)
        if self.steps:
            self.step_list.setCurrentRow(min(row, len(self.steps)-1))
        else:
            # If we've removed all steps, add a default step
            self.clear_step_fields()
            self.add_step()
        self.update_sequence_duration()

    def move_step_up(self):
        row = self.step_list.currentRow()
        if row > 0:
            self.steps[row], self.steps[row-1] = self.steps[row-1], self.steps[row]
            item = self.step_list.takeItem(row)
            self.step_list.insertItem(row-1, item)
            self.step_list.item(row-1).setText(self.steps[row-1].description)
            self.step_list.item(row).setText(self.steps[row].description)
            self.step_list.setCurrentRow(row-1)

    def move_step_down(self):
        row = self.step_list.currentRow()
        if row >= 0 and row < len(self.steps)-1:
            self.steps[row], self.steps[row+1] = self.steps[row+1], self.steps[row]
            item = self.step_list.takeItem(row)
            self.step_list.insertItem(row+1, item)
            self.step_list.item(row+1).setText(self.steps[row+1].description)
            self.step_list.item(row).setText(self.steps[row].description)
            self.step_list.setCurrentRow(row+1)

    def clear_step_fields(self):
        self.duration_spin.setValue(30)
        self.description_edit.clear()
        for ctrl in self.osc_controls:
            ctrl["wave_combo"].setCurrentIndex(0)
            ctrl["freq_start"].setValue(12.0)
            ctrl["freq_end"].setValue(12.0)
            ctrl["duty_start"].setValue(50)
            ctrl["duty_end"].setValue(50)
            ctrl["rfm_enable"].setChecked(False)
            ctrl["rfm_range"].setValue(0.5)
            ctrl["rfm_speed"].setValue(0.2)
            ctrl["phase_pattern_combo"].setCurrentIndex(0)
            ctrl["brightness_pattern_combo"].setCurrentIndex(0)
            ctrl["pattern_strength_spin"].setValue(1.0)
            ctrl["pattern_freq_spin"].setValue(1.0)
        for ctrl in self.strobe_controls:
            ctrl["strobe_start"].setValue(0)
            ctrl["strobe_end"].setValue(0)

    def on_step_selected(self, index):
        if index < 0 or index >= len(self.steps):
            return
        step = self.steps[index]
        self.duration_spin.setValue(step.duration)
        self.description_edit.setText(step.description)

        n_osc = len(step.oscillators)
        if n_osc == 1:
            mode = "Combined"
        elif n_osc == 2:
            mode = "Split"
        else:
            mode = "Independent"

        self.mode_combo.setCurrentText(mode)
        self.on_mode_changed()

        for i, osc in enumerate(step.oscillators):
            if i < len(self.osc_controls):
                ctrl = self.osc_controls[i]
                ctrl["wave_combo"].setCurrentIndex(osc.waveform.value)
                ctrl["freq_start"].setValue(osc.start_freq)
                ctrl["freq_end"].setValue(osc.end_freq)
                ctrl["duty_start"].setValue(int(osc.start_duty))
                ctrl["duty_end"].setValue(int(osc.end_duty))
                ctrl["rfm_enable"].setChecked(osc.enable_rfm)
                ctrl["rfm_range"].setValue(osc.rfm_range)
                ctrl["rfm_speed"].setValue(osc.rfm_speed)

                # Patterns
                ctrl["phase_pattern_combo"].setCurrentIndex(osc.phase_pattern.value)
                ctrl["brightness_pattern_combo"].setCurrentIndex(osc.brightness_pattern.value)
                ctrl["pattern_strength_spin"].setValue(osc.pattern_strength)
                ctrl["pattern_freq_spin"].setValue(osc.pattern_freq)

        for i, sset in enumerate(step.strobe_sets):
            if i < len(self.strobe_controls):
                ctrl = self.strobe_controls[i]
                ctrl["strobe_start"].setValue(sset.start_intensity)
                ctrl["strobe_end"].setValue(sset.end_intensity)

    def apply_osc1_to_all(self):
        mode = self.mode_combo.currentText()
        if mode != "Independent" or not self.osc_controls:
            return
        first = self.osc_controls[0]
        for ctrl in self.osc_controls[1:]:
            ctrl["wave_combo"].setCurrentIndex(first["wave_combo"].currentIndex())
            ctrl["freq_start"].setValue(first["freq_start"].value())
            ctrl["freq_end"].setValue(first["freq_end"].value())
            ctrl["duty_start"].setValue(first["duty_start"].value())
            ctrl["duty_end"].setValue(first["duty_end"].value())
            ctrl["rfm_enable"].setChecked(first["rfm_enable"].isChecked())
            ctrl["rfm_range"].setValue(first["rfm_range"].value())
            ctrl["rfm_speed"].setValue(first["rfm_speed"].value())

            ctrl["phase_pattern_combo"].setCurrentIndex(first["phase_pattern_combo"].currentIndex())
            ctrl["brightness_pattern_combo"].setCurrentIndex(first["brightness_pattern_combo"].currentIndex())
            ctrl["pattern_strength_spin"].setValue(first["pattern_strength_spin"].value())
            ctrl["pattern_freq_spin"].setValue(first["pattern_freq_spin"].value())

    def on_submit_step(self):
        row = self.step_list.currentRow()
        if row < 0 or row >= len(self.steps):
            return
        step = self.steps[row]
        step.duration = self.duration_spin.value()
        step.description = self.description_edit.text()

        mode = self.mode_combo.currentText()
        if mode == "Combined":
            num_osc = 1
        elif mode == "Split":
            num_osc = 2
        else:
            num_osc = 6

        oscillators = []
        for i in range(num_osc):
            ctrl = self.osc_controls[i]
            wave_idx = ctrl["wave_combo"].currentIndex()
            wave = Waveform(wave_idx)
            freq_s = ctrl["freq_start"].value()
            freq_e = ctrl["freq_end"].value()
            duty_s = ctrl["duty_start"].value()
            duty_e = ctrl["duty_end"].value()
            rfm_en = ctrl["rfm_enable"].isChecked()
            rfm_rng = ctrl["rfm_range"].value()
            rfm_spd = ctrl["rfm_speed"].value()

            phase_idx = ctrl["phase_pattern_combo"].currentIndex()
            bright_idx = ctrl["brightness_pattern_combo"].currentIndex()
            pattern_str = ctrl["pattern_strength_spin"].value()
            pattern_fq = ctrl["pattern_freq_spin"].value()

            osc_obj = Oscillator(
                start_freq=freq_s,
                end_freq=freq_e,
                waveform=wave,
                start_duty=duty_s,
                end_duty=duty_e,
                enable_rfm=rfm_en,
                rfm_range=rfm_rng,
                rfm_speed=rfm_spd,
                phase_pattern=PatternMode(phase_idx),
                brightness_pattern=PatternMode(bright_idx),
                pattern_strength=pattern_str,
                pattern_freq=pattern_fq
            )
            oscillators.append(osc_obj)

        strobe_sets = []
        if mode == "Combined":
            sctrl = self.strobe_controls[0]
            s_start = sctrl["strobe_start"].value()
            s_end   = sctrl["strobe_end"].value()
            strobe_sets = [
                StrobeSet(list(range(6)), s_start, s_end, [1.0])
            ]
        elif mode == "Split":
            # two strobe sets
            for i in range(2):
                sctrl = self.strobe_controls[i]
                s_start = sctrl["strobe_start"].value()
                s_end   = sctrl["strobe_end"].value()
                if i == 0:
                    strobe_sets.append(
                        StrobeSet([0,2,4], s_start, s_end, [1.0, 0.0])
                    )
                else:
                    strobe_sets.append(
                        StrobeSet([1,3,5], s_start, s_end, [0.0, 1.0])
                    )
        else:
            for i in range(6):
                sctrl = self.strobe_controls[i]
                s_start = sctrl["strobe_start"].value()
                s_end   = sctrl["strobe_end"].value()
                weights = [0.0]*6
                weights[i] = 1.0
                strobe_sets.append(
                    StrobeSet([i], s_start, s_end, weights)
                )

        step.oscillators = oscillators
        step.strobe_sets = strobe_sets
        self.step_list.item(row).setText(step.description)
        self.update_sequence_duration()
        QMessageBox.information(self, "Step Updated", f"Updated step '{step.description}'.")

    def update_sequence_duration(self):
        total = sum(step.duration for step in self.steps)
        hrs = total // 3600
        mins = (total % 3600) // 60
        secs = total % 60
        self.status.showMessage(f"Sequence Duration: {int(hrs):02d}:{int(mins):02d}:{int(secs):02d}")

    # ---------------------------------------------------------------
    # File menu actions
    # ---------------------------------------------------------------
    def new_sequence(self):
        self.steps.clear()
        self.step_list.clear()
        self.clear_step_fields()
        self.currentFile = None
        self.audio_settings = AudioSettings()
        self._update_audio_ui_from_settings()
        self.update_sequence_duration()
        self.add_step()

    def load_sequence(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Load Sequence", "", "Sequence Files (*.json);;All Files (*)")
        if not fname:
            return
        try:
            with open(fname, "r") as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{e}")
            return
        self.steps.clear()
        self.step_list.clear()
        self.currentFile = fname

        for step_dict in data.get("steps", []):
            s = Step.from_dict(step_dict)
            self.steps.append(s)
            item = QListWidgetItem(s.description)
            self.step_list.addItem(item)

        audio_dict = data.get("audio_settings", {})
        self.audio_settings = AudioSettings.from_dict(audio_dict)
        self._update_audio_ui_from_settings()

        if self.steps:
            self.step_list.setCurrentRow(0)
        else:
            self.add_step()
        self.update_sequence_duration()
        QMessageBox.information(self, "Loaded", f"Sequence loaded: {fname}")

    def save_sequence(self):
        if not self.currentFile:
            self.save_sequence_as()
        else:
            self.save_to_file(self.currentFile)

    def save_sequence_as(self):
        fname, _ = QFileDialog.getSaveFileName(self, "Save Sequence As", "",
                                               "Sequence Files (*.json);;All Files (*)")
        if not fname:
            return
        self.currentFile = fname
        self.save_to_file(fname)

    def save_to_file(self, fname):
        self._gather_audio_settings_from_ui()

        data = {
            "steps": [s.to_dict() for s in self.steps],
            "audio_settings": self.audio_settings.to_dict()
        }
        try:
            with open(fname, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save:\n{e}")
            return

        # If audio is enabled, generate audio
        if self.audio_settings.enabled:
            base, _ = os.path.splitext(fname)
            audio_filename = base + ".wav"
            audio_dict = data["audio_settings"]
            audio_dict["sample_rate"] = self.global_audio_panel.sample_rate.value()

            generate_audio_file_for_steps_offline_rfm(self.steps, audio_filename, audio_dict)

        QMessageBox.information(self, "Saved", f"Sequence saved to {fname}")

    def delete_sequence_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Delete Sequence File", "",
                                               "Sequence Files (*.json);;All Files (*)")
        if not fname:
            return
        reply = QMessageBox.question(self, "Confirm Delete",
                                     f"Delete file?\n{fname}",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            try:
                os.remove(fname)
                QMessageBox.information(self, "Deleted", f"{fname} removed.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not delete:\n{e}")

    # ---------------------------------------------------------------
    # Audio settings helpers
    # ---------------------------------------------------------------
    def _update_audio_ui_from_settings(self):
    panel = self.global_audio_panel
    panel.enabled.setChecked(self.audio_settings.enabled)
    panel.beat_freq.setValue(self.audio_settings.beat_freq)
    panel.binaural.setChecked(self.audio_settings.is_binaural)
    panel.isochronic.setChecked(self.audio_settings.is_isochronic)
    panel.global_rfm_enable.setChecked(self.audio_settings.enable_rfm)
    panel.global_rfm_range.setValue(self.audio_settings.rfm_range)
    panel.global_rfm_speed.setValue(self.audio_settings.rfm_speed)
    panel.pink_noise_enable.setChecked(self.audio_settings.enable_pink_noise)
    panel.pink_noise_volume.setValue(int(self.audio_settings.pink_noise_volume * 100))
    panel.pink_noise_label.setText(f"{int(self.audio_settings.pink_noise_volume*100)}%")
    panel.sample_rate.setValue(44100) # or store a user-chosen value

    # carriers
    for i, carrier in enumerate(self.audio_settings.carriers):
        if i < len(self.carrier_panels):
            cpanel = self.carrier_panels[i]
            cpanel.enabled.setChecked(carrier.enabled)
            
            # Update base frequencies
            cpanel.start_freq.setValue(carrier.start_freq)
            cpanel.end_freq.setValue(carrier.end_freq)
            
            # Update channel-specific frequencies if they exist in the carrier
            if hasattr(carrier, 'start_freq_left') and carrier.start_freq_left is not None:
                cpanel.start_freq_left.setValue(carrier.start_freq_left)
            else:
                # Default to base frequency + half of beat frequency
                cpanel.start_freq_left.setValue(carrier.start_freq + self.audio_settings.beat_freq/2)
                
            if hasattr(carrier, 'end_freq_left') and carrier.end_freq_left is not None:
                cpanel.end_freq_left.setValue(carrier.end_freq_left)
            else:
                cpanel.end_freq_left.setValue(carrier.end_freq + self.audio_settings.beat_freq/2)
                
            if hasattr(carrier, 'start_freq_right') and carrier.start_freq_right is not None:
                cpanel.start_freq_right.setValue(carrier.start_freq_right)
            else:
                cpanel.start_freq_right.setValue(carrier.start_freq - self.audio_settings.beat_freq/2)
                
            if hasattr(carrier, 'end_freq_right') and carrier.end_freq_right is not None:
                cpanel.end_freq_right.setValue(carrier.end_freq_right)
            else:
                cpanel.end_freq_right.setValue(carrier.end_freq - self.audio_settings.beat_freq/2)
            
            # Update the beat frequency label
            left_start = cpanel.start_freq_left.value()
            right_start = cpanel.start_freq_right.value()
            left_end = cpanel.end_freq_left.value()
            right_end = cpanel.end_freq_right.value()
            
            start_diff = abs(left_start - right_start)
            end_diff = abs(left_end - right_end)
            
            if start_diff == end_diff:
                cpanel.beat_label.setText(f"Binaural: {start_diff:.1f} Hz")
            else:
                cpanel.beat_label.setText(f"Binaural: {start_diff:.1f} → {end_diff:.1f} Hz")
            
            # Update remaining carrier settings
            cpanel.volume.setValue(int(carrier.volume*100))
            cpanel.volume_label.setText(f"{int(carrier.volume*100)}%")
            cpanel.rfm_enable.setChecked(carrier.enable_rfm)
            cpanel.rfm_range.setValue(carrier.rfm_range)
            cpanel.rfm_speed.setValue(carrier.rfm_speed)

    def _gather_audio_settings_from_ui(self):
        panel = self.global_audio_panel
        self.audio_settings.enabled = panel.enabled.isChecked()
        self.audio_settings.beat_freq = panel.beat_freq.value()
        self.audio_settings.is_binaural = panel.binaural.isChecked()
        self.audio_settings.is_isochronic = panel.isochronic.isChecked()
        self.audio_settings.enable_rfm = panel.global_rfm_enable.isChecked()
        self.audio_settings.rfm_range = panel.global_rfm_range.value()
        self.audio_settings.rfm_speed = panel.global_rfm_speed.value()
        self.audio_settings.enable_pink_noise = panel.pink_noise_enable.isChecked()
        self.audio_settings.pink_noise_volume = panel.pink_noise_volume.value()/100.0
    
        carriers = []
        for i in range(len(self.carrier_panels)):
            cpanel = self.carrier_panels[i]
            carrier = AudioCarrier(
                enabled=cpanel.enabled.isChecked(),
                start_freq=cpanel.start_freq.value(),
                end_freq=cpanel.end_freq.value(),
                start_freq_left=cpanel.start_freq_left.value(),
                end_freq_left=cpanel.end_freq_left.value(),
                start_freq_right=cpanel.start_freq_right.value(),
                end_freq_right=cpanel.end_freq_right.value(),
                volume=cpanel.volume.value()/100.0,
                enable_rfm=cpanel.rfm_enable.isChecked(),
                rfm_range=cpanel.rfm_range.value(),
                rfm_speed=cpanel.rfm_speed.value()
            )
            carriers.append(carrier)
        self.audio_settings.carriers = carriers


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

