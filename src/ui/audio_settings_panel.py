from PyQt5.QtWidgets import (QTabWidget, QWidget, QGroupBox, QFormLayout, QCheckBox,
                             QDoubleSpinBox, QSpinBox, QSlider, QLabel, QHBoxLayout, QPushButton, QComboBox)
from PyQt5.QtCore import Qt

class CarrierPanel(QGroupBox):
    def __init__(self, carrier_num, parent=None):
        super().__init__(f"Carrier {carrier_num} Settings", parent)
        self.form = QFormLayout(self)
        
        # Enable carrier checkbox
        self.enabled = QCheckBox("Enable Carrier")
        self.enabled.setChecked(carrier_num == 1)
        self.form.addRow(self.enabled)
        
        # Tone Mode selection per carrier
        self.tone_mode_combo = QComboBox()
        self.tone_mode_combo.addItems(["Binaural", "Isochronic", "Monaural"])
        self.form.addRow("Tone Mode:", self.tone_mode_combo)
        
        # Connect the mode change signal to update UI elements
        self.tone_mode_combo.currentTextChanged.connect(self.update_ui_for_mode)
        
        # Initialize frequency controls container (to be populated based on mode)
        self.freq_controls_container = QWidget()
        self.freq_controls_layout = QFormLayout(self.freq_controls_container)
        self.freq_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.form.addRow(self.freq_controls_container)
        
        # Volume control with fixed label updating
        self.volume = QSlider(Qt.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setValue(100 if carrier_num == 1 else 50)
        self.volume_label = QLabel(f"{self.volume.value()}%")
        
        # Connect the value change signal to update the label
        self.volume.valueChanged.connect(self.update_volume_label)
        
        vol_layout = QHBoxLayout()
        vol_layout.addWidget(self.volume)
        vol_layout.addWidget(self.volume_label)
        self.form.addRow("Volume:", vol_layout)
        
        # RFM settings
        self.rfm_enable = QCheckBox("Enable RFM")
        self.rfm_range = QDoubleSpinBox()
        self.rfm_range.setRange(0.0, 20.0)
        self.rfm_range.setValue(0.5)
        self.rfm_speed = QDoubleSpinBox()
        self.rfm_speed.setRange(0.0, 5.0)
        self.rfm_speed.setValue(0.2)
        rfm_help_btn = QPushButton("?")
        rfm_help_btn.setMaximumWidth(30)
        rfm_layout = QHBoxLayout()
        rfm_layout.addWidget(self.rfm_enable)
        rfm_layout.addStretch()
        rfm_layout.addWidget(rfm_help_btn)
        self.form.addRow(rfm_layout)
        self.form.addRow("RFM Range (±Hz):", self.rfm_range)
        self.form.addRow("RFM Speed (Hz/s):", self.rfm_speed)
        
        # Initialize frequency controls for the default mode (Binaural)
        self.create_binaural_controls()
        
    def update_volume_label(self, value):
        """Update the volume label when slider value changes"""
        self.volume_label.setText(f"{value}%")
        
    def update_ui_for_mode(self, mode):
        """Update frequency controls based on selected tone mode"""
        # Clear existing frequency controls
        while self.freq_controls_layout.count():
            item = self.freq_controls_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Create appropriate controls based on mode
        if mode == "Binaural":
            self.create_binaural_controls()
        elif mode == "Isochronic":
            self.create_isochronic_controls()
        else:  # Monaural
            self.create_monaural_controls()
            
    def create_binaural_controls(self):
        """Create frequency controls for Binaural mode"""
        # Left channel frequencies
        self.start_freq_left = QDoubleSpinBox()
        self.start_freq_left.setRange(20.0, 1000.0)
        self.start_freq_left.setValue(205.0)
        self.freq_controls_layout.addRow("Left Start Freq:", self.start_freq_left)
        
        self.end_freq_left = QDoubleSpinBox()
        self.end_freq_left.setRange(20.0, 1000.0)
        self.end_freq_left.setValue(205.0)
        self.freq_controls_layout.addRow("Left End Freq:", self.end_freq_left)
        
        # Right channel frequencies
        self.start_freq_right = QDoubleSpinBox()
        self.start_freq_right.setRange(20.0, 1000.0)
        self.start_freq_right.setValue(195.0)
        self.freq_controls_layout.addRow("Right Start Freq:", self.start_freq_right)
        
        self.end_freq_right = QDoubleSpinBox()
        self.end_freq_right.setRange(20.0, 1000.0)
        self.end_freq_right.setValue(195.0)
        self.freq_controls_layout.addRow("Right End Freq:", self.end_freq_right)
        
    def create_isochronic_controls(self):
        """Create frequency controls for Isochronic mode"""
        # Carrier frequency (the tone that gets modulated)
        self.start_carrier_freq = QDoubleSpinBox()
        self.start_carrier_freq.setRange(20.0, 1000.0)
        self.start_carrier_freq.setValue(200.0)
        self.freq_controls_layout.addRow("Start Carrier Freq:", self.start_carrier_freq)
        
        self.end_carrier_freq = QDoubleSpinBox()
        self.end_carrier_freq.setRange(20.0, 1000.0)
        self.end_carrier_freq.setValue(200.0)
        self.freq_controls_layout.addRow("End Carrier Freq:", self.end_carrier_freq)
        
        # Entrainment frequency (the rate at which the tone pulses)
        self.start_entrainment_freq = QDoubleSpinBox()
        self.start_entrainment_freq.setRange(0.5, 50.0)
        self.start_entrainment_freq.setValue(10.0)
        self.freq_controls_layout.addRow("Start Entrainment Freq:", self.start_entrainment_freq)
        
        self.end_entrainment_freq = QDoubleSpinBox()
        self.end_entrainment_freq.setRange(0.5, 50.0)
        self.end_entrainment_freq.setValue(10.0)
        self.freq_controls_layout.addRow("End Entrainment Freq:", self.end_entrainment_freq)
        
        # Pulse shape controls
        self.pulse_shape_combo = QComboBox()
        self.pulse_shape_combo.addItems(["Square", "Soft", "Sine"])
        self.freq_controls_layout.addRow("Pulse Shape:", self.pulse_shape_combo)
        
    def create_monaural_controls(self):
        """Create frequency controls for Monaural mode"""
        # For monaural, we just need a single set of frequencies
        self.start_freq = QDoubleSpinBox()
        self.start_freq.setRange(20.0, 1000.0)
        self.start_freq.setValue(200.0)
        self.freq_controls_layout.addRow("Start Freq:", self.start_freq)
        
        self.end_freq = QDoubleSpinBox()
        self.end_freq.setRange(20.0, 1000.0)
        self.end_freq.setValue(200.0)
        self.freq_controls_layout.addRow("End Freq:", self.end_freq)
        
    def get_frequency_values(self):
        """Get frequency values based on current mode"""
        mode = self.tone_mode_combo.currentText()
        
        if mode == "Binaural":
            return {
                "start_freq_left": self.start_freq_left.value(),
                "end_freq_left": self.end_freq_left.value(),
                "start_freq_right": self.start_freq_right.value(),
                "end_freq_right": self.end_freq_right.value(),
                "mode": "binaural"
            }
        elif mode == "Isochronic":
            return {
                "start_carrier_freq": self.start_carrier_freq.value(),
                "end_carrier_freq": self.end_carrier_freq.value(),
                "start_entrainment_freq": self.start_entrainment_freq.value(),
                "end_entrainment_freq": self.end_entrainment_freq.value(),
                "pulse_shape": self.pulse_shape_combo.currentText().lower(),
                "mode": "isochronic"
            }
        else:  # Monaural
            return {
                "start_freq": self.start_freq.value(),
                "end_freq": self.end_freq.value(),
                "mode": "monaural"
            }

class GlobalAudioPanel(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Global Audio Settings", parent)
        form = QFormLayout(self)
        self.enabled = QCheckBox("Enable Audio")
        form.addRow(self.enabled)
        # Global RFM settings.
        self.global_rfm_enable = QCheckBox("Enable Global RFM")
        form.addRow(self.global_rfm_enable)
        self.global_rfm_range = QDoubleSpinBox()
        self.global_rfm_range.setRange(0.0, 10.0)
        self.global_rfm_range.setValue(0.5)
        form.addRow("Global RFM Range (±Hz):", self.global_rfm_range)
        self.global_rfm_speed = QDoubleSpinBox()
        self.global_rfm_speed.setRange(0.0, 2.0)
        self.global_rfm_speed.setValue(0.2)
        form.addRow("Global RFM Speed (Hz/s):", self.global_rfm_speed)
        # Pink noise settings.
        self.pink_noise_enable = QCheckBox("Enable Pink Noise")
        form.addRow(self.pink_noise_enable)
        self.pink_noise_volume = QSlider(Qt.Horizontal)
        self.pink_noise_volume.setRange(0, 100)
        self.pink_noise_volume.setValue(10)
        self.pink_noise_label = QLabel("10%")
        
        # Connect the value change signal to update the label
        self.pink_noise_volume.valueChanged.connect(self.update_pink_noise_label)
        
        pn_layout = QHBoxLayout()
        pn_layout.addWidget(self.pink_noise_volume)
        pn_layout.addWidget(self.pink_noise_label)
        form.addRow("Pink Noise Volume:", pn_layout)
        # Sample rate.
        self.sample_rate = QSpinBox()
        self.sample_rate.setRange(8000, 192000)
        self.sample_rate.setValue(44100)
        self.sample_rate.setSingleStep(1000)
        form.addRow("Sample Rate (Hz):", self.sample_rate)
        
    def update_pink_noise_label(self, value):
        """Update the pink noise volume label when slider value changes"""
        self.pink_noise_label.setText(f"{value}%")
class AudioSettingsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tabs = QTabWidget()
        # Create three carrier panels.
        self.carrier_panels = [CarrierPanel(i+1) for i in range(3)]
        for i, panel in enumerate(self.carrier_panels):
            self.tabs.addTab(panel, f"Carrier {i+1}")
        # Global audio panel.
        self.global_audio_panel = GlobalAudioPanel()
        self.tabs.addTab(self.global_audio_panel, "Global Settings")
        layout = QHBoxLayout(self)
        layout.addWidget(self.tabs)

