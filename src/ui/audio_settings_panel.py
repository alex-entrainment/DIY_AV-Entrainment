from PyQt5.QtWidgets import (QWidget, QTabWidget, QGroupBox, QFormLayout, QCheckBox,
                             QDoubleSpinBox, QSlider, QLabel, QHBoxLayout, QSpinBox, QPushButton)
from PyQt5.QtCore import Qt

class CarrierPanel(QGroupBox):
    def __init__(self, carrier_num, parent=None):
        super().__init__(f"Carrier {carrier_num} Settings", parent)
        form = QFormLayout(self)
        # Enable carrier.
        self.enabled = QCheckBox("Enable Carrier")
        self.enabled.setChecked(carrier_num == 1)
        form.addRow(self.enabled)
        # Base frequency settings.
        self.start_freq = QDoubleSpinBox()
        self.start_freq.setRange(20.0, 1000.0)
        self.start_freq.setValue(200.0)
        form.addRow("Base Start Freq:", self.start_freq)
        self.end_freq = QDoubleSpinBox()
        self.end_freq.setRange(20.0, 1000.0)
        self.end_freq.setValue(200.0)
        form.addRow("Base End Freq:", self.end_freq)
        # Channel-specific frequency settings.
        self.start_freq_left = QDoubleSpinBox()
        self.start_freq_left.setRange(20.0, 1000.0)
        self.start_freq_left.setValue(205.0 if carrier_num == 1 else 200.0)
        form.addRow("Left Start Freq:", self.start_freq_left)
        self.end_freq_left = QDoubleSpinBox()
        self.end_freq_left.setRange(20.0, 1000.0)
        self.end_freq_left.setValue(205.0 if carrier_num == 1 else 200.0)
        form.addRow("Left End Freq:", self.end_freq_left)
        self.start_freq_right = QDoubleSpinBox()
        self.start_freq_right.setRange(20.0, 1000.0)
        self.start_freq_right.setValue(195.0 if carrier_num == 1 else 200.0)
        form.addRow("Right Start Freq:", self.start_freq_right)
        self.end_freq_right = QDoubleSpinBox()
        self.end_freq_right.setRange(20.0, 1000.0)
        self.end_freq_right.setValue(195.0 if carrier_num == 1 else 200.0)
        form.addRow("Right End Freq:", self.end_freq_right)
        # Binaural beat label.
        self.beat_label = QLabel("Binaural: 10.0 Hz")
        form.addRow("Beat Frequency:", self.beat_label)
        # Volume control.
        self.volume = QSlider(Qt.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setValue(100 if carrier_num == 1 else 50)
        self.volume_label = QLabel(f"{self.volume.value()}%")
        vol_layout = QHBoxLayout()
        vol_layout.addWidget(self.volume)
        vol_layout.addWidget(self.volume_label)
        form.addRow("Volume:", vol_layout)
        # RFM settings.
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
        form.addRow(rfm_layout)
        form.addRow("RFM Range (±Hz):", self.rfm_range)
        form.addRow("RFM Speed (Hz/s):", self.rfm_speed)

class GlobalAudioPanel(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Global Audio Settings", parent)
        form = QFormLayout(self)
        self.enabled = QCheckBox("Enable Audio")
        form.addRow(self.enabled)
        self.beat_freq = QDoubleSpinBox()
        self.beat_freq.setRange(0.1, 100.0)
        self.beat_freq.setValue(10.0)
        form.addRow("Beat Freq (Hz) (fallback):", self.beat_freq)
        self.binaural = QCheckBox("Binaural")
        self.binaural.setChecked(True)
        form.addRow(self.binaural)
        self.isochronic = QCheckBox("Isochronic")
        self.isochronic.setChecked(False)
        form.addRow(self.isochronic)
        def update_binaural(state):
            if state == Qt.Checked:
                self.isochronic.setChecked(False)
        def update_isochronic(state):
            if state == Qt.Checked:
                self.binaural.setChecked(False)
        self.binaural.stateChanged.connect(update_binaural)
        self.isochronic.stateChanged.connect(update_isochronic)
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
        self.pink_noise_enable = QCheckBox("Enable Pink Noise")
        form.addRow(self.pink_noise_enable)
        self.pink_noise_volume = QSlider(Qt.Horizontal)
        self.pink_noise_volume.setRange(0, 100)
        self.pink_noise_volume.setValue(10)
        self.pink_noise_label = QLabel("10%")
        pn_layout = QHBoxLayout()
        pn_layout.addWidget(self.pink_noise_volume)
        pn_layout.addWidget(self.pink_noise_label)
        form.addRow("Pink Noise Volume:", pn_layout)
        self.sample_rate = QSpinBox()
        self.sample_rate.setRange(8000, 192000)
        self.sample_rate.setValue(44100)
        self.sample_rate.setSingleStep(1000)
        form.addRow("Sample Rate (Hz):", self.sample_rate)

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
