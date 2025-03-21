import math
import sys
import json
import numpy as np
import sounddevice as sd
import traceback

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QAbstractItemView, QFileDialog,
    QLineEdit, QMessageBox, QAction, QMenuBar, QDialog, QDialogButtonBox,
    QFormLayout, QSlider, QSplitter
)
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg

# Import the integrated audio engine instead of the original
import audio_engine as audio_engine
from audio_engine import SoundPath


########################################################################
# Data Model Classes
########################################################################

class NodeData:
    def __init__(self, duration, base_freq, beat_freq, volume_left, volume_right,
                 phase_deviation=0.7, left_phase_offset=0.0, right_phase_offset=0.0,
                 sound_path=SoundPath.CIRCULAR):
        self.duration = duration
        self.base_freq = base_freq
        self.beat_freq = beat_freq
        self.volume_left = volume_left
        self.volume_right = volume_right
        
        # SAM-specific parameters (updated to match new implementation)
        self.phase_deviation = phase_deviation
        self.left_phase_offset = left_phase_offset
        self.right_phase_offset = right_phase_offset
        self.sound_path = sound_path
        
    def to_dict(self):
        return {
            "duration": self.duration,
            "base_freq": self.base_freq,
            "beat_freq": self.beat_freq,
            "volume_left": self.volume_left,
            "volume_right": self.volume_right,
            "phase_deviation": self.phase_deviation,
            "left_phase_offset": self.left_phase_offset,
            "right_phase_offset": self.right_phase_offset,
            "sound_path": self.sound_path.value if isinstance(self.sound_path, SoundPath) else self.sound_path
        }
    
    @staticmethod
    def from_dict(d):
        # Convert string to SoundPath enum if needed
        sound_path = d.get("sound_path", SoundPath.CIRCULAR)
        if isinstance(sound_path, str):
            try:
                sound_path = SoundPath[sound_path.upper()]
            except (KeyError, AttributeError):
                sound_path = SoundPath.CIRCULAR
                
        return NodeData(
            d["duration"],
            d["base_freq"],
            d["beat_freq"],
            d["volume_left"],
            d["volume_right"],
            d.get("phase_deviation", 0.7),
            d.get("left_phase_offset", 0.0),
            d.get("right_phase_offset", 0.0),
            sound_path
        )


class VoiceData:
    def __init__(self, voice_type, voice_name="Voice",
                 ramp_percent=0.2, gap_percent=0.15, amplitude=1.0):
        """
        Generic voice data for any voice type:
        - nodes: a list of NodeData
        - For isochronic: ramp/gap/amplitude
        - For SAM: spatial modulation parameters
        """
        self.voice_type = voice_type
        self.voice_name = voice_name
        self.ramp_percent = ramp_percent
        self.gap_percent = gap_percent
        self.amplitude = amplitude
        self.nodes = []
        self.muted = False
        self.view_enabled = True

        # SAM parameters (updated to match new implementation)
        self.phase_deviation = 0.7
        self.left_phase_offset = 0.0
        self.right_phase_offset = 0.0
        self.sound_path = SoundPath.CIRCULAR
        
        # Add support for MultiSAMBinauralVoice
        self.secondary_freq_ratio = 1.5
        self.secondary_spatial_ratio = 0.7
        self.secondary_volume = 0.4
        self.use_secondary_source = False

    def to_dict(self):
        return {
            "voice_type": self.voice_type,
            "voice_name": self.voice_name,
            "ramp_percent": self.ramp_percent,
            "gap_percent": self.gap_percent,
            "amplitude": self.amplitude,
            "muted": self.muted,
            "view_enabled": self.view_enabled,
            "nodes": [n.to_dict() for n in self.nodes],

            # SAM parameters (updated)
            "phase_deviation": self.phase_deviation,
            "left_phase_offset": self.left_phase_offset,
            "right_phase_offset": self.right_phase_offset,
            "sound_path": self.sound_path.value if isinstance(self.sound_path, SoundPath) else self.sound_path,
            
            # Multi-source parameters
            "secondary_freq_ratio": self.secondary_freq_ratio,
            "secondary_spatial_ratio": self.secondary_spatial_ratio,
            "secondary_volume": self.secondary_volume,
            "use_secondary_source": self.use_secondary_source
        }

    @staticmethod
    def from_dict(d):
        v = VoiceData(
            d.get("voice_type", "BinauralBeat"),
            d.get("voice_name", "Voice"),
            d.get("ramp_percent", 0.2),
            d.get("gap_percent", 0.15),
            d.get("amplitude", 1.0)
        )
        v.muted = d.get("muted", False)
        v.view_enabled = d.get("view_enabled", True)

        # Load NodeData
        for nd in d.get("nodes", []):
            v.nodes.append(NodeData.from_dict(nd))

        # SAM parameters (updated)
        v.phase_deviation = d.get("phase_deviation", 0.7)
        v.left_phase_offset = d.get("left_phase_offset", 0.0)
        v.right_phase_offset = d.get("right_phase_offset", 0.0)
        
        # Convert sound_path string to enum
        sound_path = d.get("sound_path", SoundPath.CIRCULAR)
        if isinstance(sound_path, str):
            try:
                v.sound_path = SoundPath[sound_path.upper()]
            except (KeyError, AttributeError):
                v.sound_path = SoundPath.CIRCULAR
        else:
            v.sound_path = sound_path
            
        # Multi-source parameters
        v.secondary_freq_ratio = d.get("secondary_freq_ratio", 1.5)
        v.secondary_spatial_ratio = d.get("secondary_spatial_ratio", 0.7)
        v.secondary_volume = d.get("secondary_volume", 0.4)
        v.use_secondary_source = d.get("use_secondary_source", False)

        return v


class Track:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.voices = []

    def to_dict(self):
        return {
            "sample_rate": self.sample_rate,
            "voices": [v.to_dict() for v in self.voices]
        }

    @staticmethod
    def from_dict(d):
        t = Track(sample_rate=d.get("sample_rate", 44100))
        for vdict in d.get("voices", []):
            t.voices.append(VoiceData.from_dict(vdict))
        return t


########################################################################
# Dialogs
########################################################################

class VoiceCreationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add a New Voice")
        self.voice_type_box = QComboBox()
        self.voice_type_box.addItems([
            "BinauralBeat",
            "Isochronic",
            "AltIsochronic",
            "AltIsochronic2",
            "PinkNoise",
            "ExternalAudio",
            "SAMBinaural",
            "MultiSAMBinaural"
        ])
        form = QFormLayout()
        form.addRow("Voice Type:", self.voice_type_box)
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(buttonBox)
        self.setLayout(layout)


class NodeEditDialog(QDialog):
    """
    Allows user to edit the NodeData for either SAM or other voices.
    The same node structure is used: (duration, base_freq, beat_freq, vol_left, vol_right).
    Additional parameters are shown when editing a node in a SAM voice.
    """
    def __init__(self, node, voice_type="BinauralBeat", parent=None):
        super().__init__(parent)
        self.node = node
        self.voice_type = voice_type
        self.setWindowTitle("Edit Node")

        # Create basic node parameters
        self.spin_duration = QDoubleSpinBox()
        self.spin_duration.setRange(0.0, 9999.0)
        self.spin_duration.setSingleStep(0.5)
        self.spin_duration.setValue(self.node.duration)

        self.spin_base_freq = QDoubleSpinBox()
        self.spin_base_freq.setRange(0.0, 20000.0)
        self.spin_base_freq.setSingleStep(1.0)
        self.spin_base_freq.setValue(self.node.base_freq)

        self.spin_beat_freq = QDoubleSpinBox()
        self.spin_beat_freq.setRange(0.0, 20000.0)
        self.spin_beat_freq.setSingleStep(0.1)
        self.spin_beat_freq.setValue(self.node.beat_freq)
        self.spin_beat_freq.setToolTip("For SAM voices, this controls spatial oscillation frequency (Hz)")

        self.spin_vol_left = QDoubleSpinBox()
        self.spin_vol_left.setRange(0.0, 1.0)
        self.spin_vol_left.setSingleStep(0.1)
        self.spin_vol_left.setValue(self.node.volume_left)

        self.spin_vol_right = QDoubleSpinBox()
        self.spin_vol_right.setRange(0.0, 1.0)
        self.spin_vol_right.setSingleStep(0.1)
        self.spin_vol_right.setValue(self.node.volume_right)

        form = QFormLayout()
        form.addRow("Duration (sec):", self.spin_duration)
        form.addRow("Base Frequency:", self.spin_base_freq)
        form.addRow("Beat/Spatial Frequency:", self.spin_beat_freq)
        form.addRow("Volume Left:", self.spin_vol_left)
        form.addRow("Volume Right:", self.spin_vol_right)

        # Add SAM-specific controls if this is a SAM voice
        if voice_type in ["SAMBinaural", "MultiSAMBinaural"]:
            # Create controls for SAM parameters
            self.spin_phase_deviation = QDoubleSpinBox()
            self.spin_phase_deviation.setRange(0.0, 5.0)
            self.spin_phase_deviation.setSingleStep(0.1)
            self.spin_phase_deviation.setValue(getattr(self.node, 'phase_deviation', 0.7))
            self.spin_phase_deviation.setToolTip("Controls perceived spatial width")
            
            self.spin_left_phase_offset = QDoubleSpinBox()
            self.spin_left_phase_offset.setRange(-3.14, 3.14)
            self.spin_left_phase_offset.setSingleStep(0.1)
            self.spin_left_phase_offset.setValue(getattr(self.node, 'left_phase_offset', 0.0))
            self.spin_left_phase_offset.setToolTip("Phase offset for left channel (radians)")
            
            self.spin_right_phase_offset = QDoubleSpinBox()
            self.spin_right_phase_offset.setRange(-3.14, 3.14)
            self.spin_right_phase_offset.setSingleStep(0.1)
            self.spin_right_phase_offset.setValue(getattr(self.node, 'right_phase_offset', 0.0))
            self.spin_right_phase_offset.setToolTip("Phase offset for right channel (radians)")
            
            self.combo_sound_path = QComboBox()
            for path in SoundPath:
                self.combo_sound_path.addItem(path.value.title(), path)
                
            # Set current sound path if it exists
            current_path = getattr(self.node, 'sound_path', SoundPath.CIRCULAR)
            for i in range(self.combo_sound_path.count()):
                if self.combo_sound_path.itemData(i) == current_path:
                    self.combo_sound_path.setCurrentIndex(i)
                    break
            
            self.combo_sound_path.setToolTip("Type of perceived movement pattern")
            
            # Add to form layout with a separator
            form.addRow(QLabel("--- SAM Parameters ---"))
            form.addRow("Phase Deviation:", self.spin_phase_deviation)
            form.addRow("Left Phase Offset:", self.spin_left_phase_offset)
            form.addRow("Right Phase Offset:", self.spin_right_phase_offset)
            form.addRow("Sound Path:", self.combo_sound_path)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.accept_data)
        buttonBox.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(buttonBox)
        self.setLayout(layout)

    def accept_data(self):
        # Update basic node parameters
        self.node.duration = self.spin_duration.value()
        self.node.base_freq = self.spin_base_freq.value()
        self.node.beat_freq = self.spin_beat_freq.value()
        self.node.volume_left = self.spin_vol_left.value()
        self.node.volume_right = self.spin_vol_right.value()

        # Update SAM parameters if they exist
        if hasattr(self, 'spin_phase_deviation'):
            self.node.phase_deviation = self.spin_phase_deviation.value()
            self.node.left_phase_offset = self.spin_left_phase_offset.value()
            self.node.right_phase_offset = self.spin_right_phase_offset.value()
            self.node.sound_path = self.combo_sound_path.currentData()
            
        self.accept()


########################################################################
# Main Window
########################################################################

class MainWindow(QMainWindow):
    color_list = [
        (255, 0, 0),       # red
        (0, 255, 0),       # green
        (0, 0, 255),       # blue
        (255, 255, 0),     # yellow
        (255, 0, 255),     # magenta
        (0, 255, 255),     # cyan
        (200, 200, 200)    # light gray
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM-enabled UI")
        self.resize(1400, 800)

        self.track = Track()
        self.final_audio = None
        self.is_playing = False
        self.play_start_time = 0
        self.current_play_offset = 0.0

        # Keep track of selected nodes for the graph
        self.selected_nodes = set()

        self.create_menu()
        self.setup_ui()
        self.connect_node_table_selection()  # Add this line

        self.play_timer = QTimer()
        self.play_timer.setInterval(200)
        self.play_timer.timeout.connect(self.update_playhead)

    ####################################################################
    # Menu
    ####################################################################
    def create_menu(self):
        menubar = QMenuBar(self)
        self.setMenuBar(menubar)

        file_menu = menubar.addMenu("File")
        save_action = QAction("Save JSON", self)
        save_action.triggered.connect(self.save_json)
        load_action = QAction("Load JSON", self)
        load_action.triggered.connect(self.load_json)
        file_menu.addAction(save_action)
        file_menu.addAction(load_action)

        export_menu = menubar.addMenu("Export")
        export_wav_action = QAction("Export WAV", self)
        export_wav_action.triggered.connect(self.export_wav)
        export_flac_action = QAction("Export FLAC", self)
        export_flac_action.triggered.connect(self.export_flac)
        export_mp3_action = QAction("Export MP3", self)
        export_mp3_action.triggered.connect(self.export_mp3)

        export_menu.addAction(export_wav_action)
        export_menu.addAction(export_flac_action)
        export_menu.addAction(export_mp3_action)

    ####################################################################
    # UI Setup
    ####################################################################
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)  # Changed to QVBoxLayout for main container

        # Create main splitter for horizontal arrangement
        self.main_splitter = QSplitter(Qt.Horizontal)
        
        # Voice table
        self.voice_table = QTableWidget()
        self.voice_table.setColumnCount(5)
        self.voice_table.setHorizontalHeaderLabels(["Color","Name","Type","Mute","View"])
        self.voice_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.voice_table.setEditTriggers(QAbstractItemView.AllEditTriggers)
        self.voice_table.itemChanged.connect(self.on_voice_table_item_changed)
        self.voice_table.currentCellChanged.connect(self.on_voice_table_selected)

        self.btn_add_voice = QPushButton("Add Voice")
        self.btn_add_voice.clicked.connect(self.add_voice)
        self.btn_remove_voice = QPushButton("Remove Voice")
        self.btn_remove_voice.clicked.connect(self.remove_voice)

        voice_layout = QVBoxLayout()
        voice_layout.addWidget(QLabel("Voices:"))
        voice_layout.addWidget(self.voice_table)
        voice_btn_layout = QHBoxLayout()
        voice_btn_layout.addWidget(self.btn_add_voice)
        voice_btn_layout.addWidget(self.btn_remove_voice)
        voice_layout.addLayout(voice_btn_layout)

        # Voice container widget
        voices_container = QWidget()
        voices_container.setLayout(voice_layout)
        
        # Node table
        self.node_table = QTableWidget()
        self.node_table.setColumnCount(5)
        self.node_table.setHorizontalHeaderLabels(["Duration","BaseFreq","BeatFreq","VolLeft","VolRight"])
        self.node_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.node_table.setSelectionMode(QAbstractItemView.ExtendedSelection)  # Changed to ExtendedSelection
        self.node_table.itemChanged.connect(self.on_node_table_item_changed)

        self.btn_add_node = QPushButton("Add Node")
        self.btn_add_node.clicked.connect(self.add_node)
        self.btn_remove_node = QPushButton("Remove Node")
        self.btn_remove_node.clicked.connect(self.remove_node)
        
        # Add button for batch editing multiple nodes
        self.btn_edit_multiple_nodes = QPushButton("Edit Selected Nodes")
        self.btn_edit_multiple_nodes.clicked.connect(self.edit_multiple_nodes)
        self.btn_edit_multiple_nodes.setToolTip("Edit parameters for all selected nodes at once")

        node_layout = QVBoxLayout()
        node_layout.addWidget(QLabel("Nodes:"))
        node_layout.addWidget(self.node_table)
        node_btn_layout = QHBoxLayout()
        node_btn_layout.addWidget(self.btn_add_node)
        node_btn_layout.addWidget(self.btn_remove_node)
        node_btn_layout.addWidget(self.btn_edit_multiple_nodes)
        node_layout.addLayout(node_btn_layout)

        # Iso param controls
        self.label_ramp = QLabel("Ramp%")
        self.spin_ramp = QDoubleSpinBox()
        self.spin_ramp.setRange(0.0, 0.5)
        self.spin_ramp.valueChanged.connect(self.on_iso_params_changed)

        self.label_gap = QLabel("Gap%")
        self.spin_gap = QDoubleSpinBox()
        self.spin_gap.setRange(0.0, 0.5)
        self.spin_gap.valueChanged.connect(self.on_iso_params_changed)

        self.label_amp = QLabel("Amplitude")
        self.spin_amp = QDoubleSpinBox()
        self.spin_amp.setRange(0.0, 10.0)
        self.spin_amp.setValue(1.0)
        self.spin_amp.valueChanged.connect(self.on_iso_params_changed)

        iso_layout = QGridLayout()
        iso_layout.addWidget(self.label_ramp, 0, 0)
        iso_layout.addWidget(self.spin_ramp, 0, 1)
        iso_layout.addWidget(self.label_gap, 1, 0)
        iso_layout.addWidget(self.spin_gap, 1, 1)
        iso_layout.addWidget(self.label_amp, 2, 0)
        iso_layout.addWidget(self.spin_amp, 2, 1)

        middle_layout = QVBoxLayout()
        middle_layout.addLayout(node_layout, stretch=2)
        middle_layout.addLayout(iso_layout, stretch=1)

        # SAM param layout
        self.sam_group_label = QLabel("Spatial Angle Modulation (SAM) Parameters")

        self.label_phase_deviation = QLabel("Phase Deviation")
        self.spin_phase_deviation = QDoubleSpinBox()
        self.spin_phase_deviation.setRange(0.0, 5.0)
        self.spin_phase_deviation.setSingleStep(0.1)
        self.spin_phase_deviation.setValue(0.7)
        self.spin_phase_deviation.setToolTip("Controls perceived spatial width")

        self.label_left_phase_offset = QLabel("Left Phase Offset")
        self.spin_left_phase_offset = QDoubleSpinBox()
        self.spin_left_phase_offset.setRange(-3.14, 3.14)
        self.spin_left_phase_offset.setSingleStep(0.1)
        self.spin_left_phase_offset.setValue(0.0)
        self.spin_left_phase_offset.setToolTip("Phase offset for left channel (radians)")

        self.label_right_phase_offset = QLabel("Right Phase Offset")
        self.spin_right_phase_offset = QDoubleSpinBox()
        self.spin_right_phase_offset.setRange(-3.14, 3.14)
        self.spin_right_phase_offset.setSingleStep(0.1)
        self.spin_right_phase_offset.setValue(0.0)
        self.spin_right_phase_offset.setToolTip("Phase offset for right channel (radians)")

        self.label_sound_path = QLabel("Sound Path")
        self.combo_sound_path = QComboBox()
        for path in SoundPath:
            self.combo_sound_path.addItem(path.value.title(), path)
        self.combo_sound_path.setToolTip("Type of perceived movement pattern")

        # MultiSAM specific controls
        self.label_use_secondary = QLabel("Use Secondary Source")
        self.check_use_secondary = QPushButton("Secondary Source")
        self.check_use_secondary.setCheckable(True)
        self.check_use_secondary.setChecked(False)
        self.check_use_secondary.clicked.connect(self.on_secondary_source_toggled)

        self.label_secondary_freq_ratio = QLabel("Secondary Freq Ratio")
        self.spin_secondary_freq_ratio = QDoubleSpinBox()
        self.spin_secondary_freq_ratio.setRange(0.5, 3.0)
        self.spin_secondary_freq_ratio.setSingleStep(0.1)
        self.spin_secondary_freq_ratio.setValue(1.5)
        self.spin_secondary_freq_ratio.setToolTip("Ratio of secondary source frequency to primary")

        self.label_secondary_spatial_ratio = QLabel("Secondary Spatial Ratio")
        self.spin_secondary_spatial_ratio = QDoubleSpinBox()
        self.spin_secondary_spatial_ratio.setRange(0.1, 2.0)
        self.spin_secondary_spatial_ratio.setSingleStep(0.1)
        self.spin_secondary_spatial_ratio.setValue(0.7)
        self.spin_secondary_spatial_ratio.setToolTip("Ratio of secondary spatial frequency to primary")

        self.label_secondary_volume = QLabel("Secondary Volume")
        self.spin_secondary_volume = QDoubleSpinBox()
        self.spin_secondary_volume.setRange(0.0, 1.0)
        self.spin_secondary_volume.setSingleStep(0.1)
        self.spin_secondary_volume.setValue(0.4)
        self.spin_secondary_volume.setToolTip("Volume of secondary source relative to primary")

        # Node selection help text
        self.node_selection_help = QLabel(
            "Selection Tips: Shift+Click node to edit. Ctrl+Click to select multiple nodes."
        )
        self.node_selection_help.setWordWrap(True)
        self.node_selection_help.setStyleSheet("color: gray; font-style: italic;")

        self.sam_layout = QGridLayout()
        self.sam_layout.addWidget(self.sam_group_label, 0, 0, 1, 2)
        self.sam_layout.addWidget(self.label_phase_deviation, 1, 0)
        self.sam_layout.addWidget(self.spin_phase_deviation, 1, 1)
        self.sam_layout.addWidget(self.label_left_phase_offset, 2, 0)
        self.sam_layout.addWidget(self.spin_left_phase_offset, 2, 1)
        self.sam_layout.addWidget(self.label_right_phase_offset, 3, 0)
        self.sam_layout.addWidget(self.spin_right_phase_offset, 3, 1)
        self.sam_layout.addWidget(self.label_sound_path, 4, 0)
        self.sam_layout.addWidget(self.combo_sound_path, 4, 1)
        
        # Secondary source controls
        self.sam_layout.addWidget(QLabel("--- Secondary Source ---"), 5, 0, 1, 2)
        self.sam_layout.addWidget(self.label_use_secondary, 6, 0)
        self.sam_layout.addWidget(self.check_use_secondary, 6, 1)
        self.sam_layout.addWidget(self.label_secondary_freq_ratio, 7, 0)
        self.sam_layout.addWidget(self.spin_secondary_freq_ratio, 7, 1)
        self.sam_layout.addWidget(self.label_secondary_spatial_ratio, 8, 0)
        self.sam_layout.addWidget(self.spin_secondary_spatial_ratio, 8, 1)
        self.sam_layout.addWidget(self.label_secondary_volume, 9, 0)
        self.sam_layout.addWidget(self.spin_secondary_volume, 9, 1)
        
        # Add node selection help
        self.sam_layout.addWidget(self.node_selection_help, 10, 0, 1, 2)

        self.sam_container = QWidget()
        self.sam_container.setLayout(self.sam_layout)
        self.sam_container.hide()

        # Hook up signals for SAM controls
        self.spin_phase_deviation.valueChanged.connect(self.on_sam_params_changed)
        self.spin_left_phase_offset.valueChanged.connect(self.on_sam_params_changed)
        self.spin_right_phase_offset.valueChanged.connect(self.on_sam_params_changed)
        self.combo_sound_path.currentIndexChanged.connect(self.on_sam_params_changed)
        self.spin_secondary_freq_ratio.valueChanged.connect(self.on_sam_params_changed)
        self.spin_secondary_spatial_ratio.valueChanged.connect(self.on_sam_params_changed)
        self.spin_secondary_volume.valueChanged.connect(self.on_sam_params_changed)

        middle_layout.addWidget(self.sam_container)

        # Graph
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("bottom","Time (s)")
        self.plot_widget.setLabel("left","Parameter Value")

        self.plot_param_combo = QComboBox()
        self.plot_param_combo.addItems(["base_freq","beat_freq","volume","stereo_balance"])
        self.plot_param_combo.currentIndexChanged.connect(self.update_graph)

        top_graph_layout = QHBoxLayout()
        top_graph_layout.addWidget(QLabel("Graph Param:"))
        top_graph_layout.addWidget(self.plot_param_combo)

        # Playback
        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self.play_track)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_track)

        self.lbl_play_time = QLabel("Time: 0.00 / 0.00")
        self.scrub_slider = QSlider(Qt.Horizontal)
        self.scrub_slider.setRange(0, 100)
        self.scrub_slider.valueChanged.connect(self.on_scrub_changed)

        pb_top_layout = QHBoxLayout()
        pb_top_layout.addWidget(self.btn_play)
        pb_top_layout.addWidget(self.btn_stop)
        pb_top_layout.addWidget(self.lbl_play_time)

        pb_layout = QVBoxLayout()
        pb_layout.addLayout(pb_top_layout)
        pb_layout.addWidget(QLabel("Scrub:"))
        pb_layout.addWidget(self.scrub_slider)

        right_graph_layout = QVBoxLayout()
        right_graph_layout.addLayout(top_graph_layout)
        right_graph_layout.addWidget(self.plot_widget, stretch=2)
        right_graph_layout.addLayout(pb_layout, stretch=1)

        middle_container = QWidget()
        middle_container.setLayout(middle_layout)

        right_container = QWidget()
        right_container.setLayout(right_graph_layout)

        # Add the widgets to the main horizontal splitter
        self.main_splitter.addWidget(voices_container)
        self.main_splitter.addWidget(middle_container)
        self.main_splitter.addWidget(right_container)
        
        # Set initial sizes for the splitter
        self.main_splitter.setSizes([200, 300, 500])
        
        # Add the main splitter to the layout
        main_layout.addWidget(self.main_splitter)

    def edit_multiple_nodes(self):
        """Edit parameters for multiple selected nodes at once."""
        voice_idx = self.voice_table.currentRow()
        if voice_idx < 0 or voice_idx >= len(self.track.voices):
            return
        
        voice = self.track.voices[voice_idx]
        
        # Get all selected rows from node table
        selected_rows = sorted(set([item.row() for item in self.node_table.selectedItems()]))
        if not selected_rows:
            QMessageBox.information(self, "Selection", "Please select at least one node to edit.")
            return
            
        # Create a dialog for batch editing
        dlg = QDialog(self)
        dlg.setWindowTitle("Batch Edit Nodes")
        layout = QVBoxLayout()
        
        form = QFormLayout()
        
        # Create checkboxes and spinboxes for each parameter
        chk_duration = QCheckBox("Edit Duration")
        spin_duration = QDoubleSpinBox()
        spin_duration.setRange(0.0, 9999.0)
        spin_duration.setSingleStep(0.5)
        spin_duration.setValue(voice.nodes[selected_rows[0]].duration)
        spin_duration.setEnabled(False)
        chk_duration.toggled.connect(spin_duration.setEnabled)
        
        chk_base_freq = QCheckBox("Edit Base Frequency")
        spin_base_freq = QDoubleSpinBox()
        spin_base_freq.setRange(0.0, 20000.0)
        spin_base_freq.setSingleStep(1.0)
        spin_base_freq.setValue(voice.nodes[selected_rows[0]].base_freq)
        spin_base_freq.setEnabled(False)
        chk_base_freq.toggled.connect(spin_base_freq.setEnabled)
        
        chk_beat_freq = QCheckBox("Edit Beat Frequency")
        spin_beat_freq = QDoubleSpinBox()
        spin_beat_freq.setRange(0.0, 20000.0)
        spin_beat_freq.setSingleStep(0.1)
        spin_beat_freq.setValue(voice.nodes[selected_rows[0]].beat_freq)
        spin_beat_freq.setEnabled(False)
        chk_beat_freq.toggled.connect(spin_beat_freq.setEnabled)
        
        chk_vol_left = QCheckBox("Edit Left Volume")
        spin_vol_left = QDoubleSpinBox()
        spin_vol_left.setRange(0.0, 1.0)
        spin_vol_left.setSingleStep(0.1)
        spin_vol_left.setValue(voice.nodes[selected_rows[0]].volume_left)
        spin_vol_left.setEnabled(False)
        chk_vol_left.toggled.connect(spin_vol_left.setEnabled)
        
        chk_vol_right = QCheckBox("Edit Right Volume")
        spin_vol_right = QDoubleSpinBox()
        spin_vol_right.setRange(0.0, 1.0)
        spin_vol_right.setSingleStep(0.1)
        spin_vol_right.setValue(voice.nodes[selected_rows[0]].volume_right)
        spin_vol_right.setEnabled(False)
        chk_vol_right.toggled.connect(spin_vol_right.setEnabled)
        
        # Add SAM-specific parameters if applicable
        sam_controls = []
        if voice.voice_type in ["SAMBinaural", "MultiSAMBinaural"]:
            # Phase deviation
            chk_phase_dev = QCheckBox("Edit Phase Deviation")
            spin_phase_dev = QDoubleSpinBox()
            spin_phase_dev.setRange(0.0, 5.0)
            spin_phase_dev.setSingleStep(0.1)
            spin_phase_dev.setValue(getattr(voice.nodes[selected_rows[0]], 'phase_deviation', 0.7))
            spin_phase_dev.setEnabled(False)
            chk_phase_dev.toggled.connect(spin_phase_dev.setEnabled)
            
            # Left phase offset
            chk_left_offset = QCheckBox("Edit Left Phase Offset")
            spin_left_offset = QDoubleSpinBox()
            spin_left_offset.setRange(-3.14, 3.14)
            spin_left_offset.setSingleStep(0.1)
            spin_left_offset.setValue(getattr(voice.nodes[selected_rows[0]], 'left_phase_offset', 0.0))
            spin_left_offset.setEnabled(False)
            chk_left_offset.toggled.connect(spin_left_offset.setEnabled)
            
            # Right phase offset
            chk_right_offset = QCheckBox("Edit Right Phase Offset")
            spin_right_offset = QDoubleSpinBox()
            spin_right_offset.setRange(-3.14, 3.14)
            spin_right_offset.setSingleStep(0.1)
            spin_right_offset.setValue(getattr(voice.nodes[selected_rows[0]], 'right_phase_offset', 0.0))
            spin_right_offset.setEnabled(False)
            chk_right_offset.toggled.connect(spin_right_offset.setEnabled)
            
            # Sound path
            chk_sound_path = QCheckBox("Edit Sound Path")
            combo_sound_path = QComboBox()
            for path in SoundPath:
                combo_sound_path.addItem(path.value.title(), path)
            
            # Set current sound path
            current_path = getattr(voice.nodes[selected_rows[0]], 'sound_path', SoundPath.CIRCULAR)
            for i in range(combo_sound_path.count()):
                if combo_sound_path.itemData(i) == current_path:
                    combo_sound_path.setCurrentIndex(i)
                    break
            
            combo_sound_path.setEnabled(False)
            chk_sound_path.toggled.connect(combo_sound_path.setEnabled)
            
            # Add to controls list for later reference
            sam_controls = [
                (chk_phase_dev, spin_phase_dev, 'phase_deviation'),
                (chk_left_offset, spin_left_offset, 'left_phase_offset'),
                (chk_right_offset, spin_right_offset, 'right_phase_offset'),
                (chk_sound_path, combo_sound_path, 'sound_path')
            ]

        # Add basic parameters to form
        form.addRow(chk_duration, spin_duration)
        form.addRow(chk_base_freq, spin_base_freq)
        form.addRow(chk_beat_freq, spin_beat_freq)
        form.addRow(chk_vol_left, spin_vol_left)
        form.addRow(chk_vol_right, spin_vol_right)
        
        # Add SAM parameters if applicable
        if voice.voice_type in ["SAMBinaural", "MultiSAMBinaural"]:
            form.addRow(QLabel("--- SAM Parameters ---"))
            for chk, ctrl, _ in sam_controls:
                form.addRow(chk, ctrl)
        
        # Button box
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        
        layout.addLayout(form)
        layout.addWidget(buttonBox)
        dlg.setLayout(layout)
        
        # Connect button signals
        buttonBox.accepted.connect(dlg.accept)
        buttonBox.rejected.connect(dlg.reject)
        
        # Show the dialog
        if dlg.exec_() == QDialog.Accepted:
            # Apply changes to all selected nodes
            for idx in selected_rows:
                if idx < len(voice.nodes):
                    node = voice.nodes[idx]
                    
                    # Update basic parameters if checked
                    if chk_duration.isChecked():
                        node.duration = spin_duration.value()
                    if chk_base_freq.isChecked():
                        node.base_freq = spin_base_freq.value()
                    if chk_beat_freq.isChecked():
                        node.beat_freq = spin_beat_freq.value()
                    if chk_vol_left.isChecked():
                        node.volume_left = spin_vol_left.value()
                    if chk_vol_right.isChecked():
                        node.volume_right = spin_vol_right.value()
                    
                    # Update SAM parameters if checked
                    if voice.voice_type in ["SAMBinaural", "MultiSAMBinaural"]:
                        for chk, ctrl, param_name in sam_controls:
                            if chk.isChecked():
                                if param_name == 'sound_path':
                                    setattr(node, param_name, ctrl.currentData())
                                else:
                                    setattr(node, param_name, ctrl.value())
            
            # Refresh the node table and graph
            self.on_voice_table_selected(voice_idx, 0, 0, 0)
            self.update_graph()
    def on_secondary_source_toggled(self, checked):
        """Handle toggling the secondary source checkbox"""
        row = self.voice_table.currentRow()
        if row < 0 or row >= len(self.track.voices):
            return
        v = self.track.voices[row]
        v.use_secondary_source = checked

        # Enable or disable secondary controls
        self.spin_secondary_freq_ratio.setEnabled(checked)
        self.spin_secondary_spatial_ratio.setEnabled(checked)
        self.spin_secondary_volume.setEnabled(checked)

    ####################################################################
    # Voice Table / Node Table
    ####################################################################

    def add_voice(self):
        dlg = VoiceCreationDialog(self)
        if dlg.exec_() == dlg.Accepted:
            vtype = dlg.voice_type_box.currentText()
            v = VoiceData(voice_type=vtype, voice_name=f"{vtype}")
            
            # Provide a default node with appropriate parameters
            if vtype in ["SAMBinaural", "MultiSAMBinaural"]:
                # Create node with SAM parameters
                node = NodeData(
                    5.0, 250.0, 0.5, 0.8, 0.8,  # Basic parameters
                    phase_deviation=0.7,
                    left_phase_offset=0.0,
                    right_phase_offset=0.0,
                    sound_path=SoundPath.CIRCULAR
                )
                
                # Set appropriate defaults for MultiSAMBinaural
                if vtype == "MultiSAMBinaural":
                    v.use_secondary_source = True
                    v.secondary_freq_ratio = 1.5
                    v.secondary_spatial_ratio = 0.7
                    v.secondary_volume = 0.4
            else:
                # Standard node for other voice types
                node = NodeData(5.0, 100.0, 0.0, 1.0, 1.0)
                
            v.nodes.append(node)
            self.track.voices.append(v)
            self.refresh_voice_table()

    def remove_voice(self):
        row = self.voice_table.currentRow()
        if row < 0 or row >= len(self.track.voices):
            return
        self.track.voices.pop(row)
        self.refresh_voice_table()

    def refresh_voice_table(self):
        self.voice_table.blockSignals(True)
        self.voice_table.setRowCount(len(self.track.voices))
        for i, v in enumerate(self.track.voices):
            color_item = QTableWidgetItem()
            c = self.color_list[i % len(self.color_list)]
            color_item.setBackground(pg.mkColor(c[0], c[1], c[2], 255))
            color_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            self.voice_table.setItem(i, 0, color_item)

            name_item = QTableWidgetItem(v.voice_name)
            self.voice_table.setItem(i, 1, name_item)

            type_item = QTableWidgetItem(v.voice_type)
            type_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            self.voice_table.setItem(i, 2, type_item)

            mute_item = QTableWidgetItem()
            mute_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            mute_item.setCheckState(Qt.Checked if v.muted else Qt.Unchecked)
            self.voice_table.setItem(i, 3, mute_item)

            view_item = QTableWidgetItem()
            view_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            view_item.setCheckState(Qt.Checked if v.view_enabled else Qt.Unchecked)
            self.voice_table.setItem(i, 4, view_item)
        self.voice_table.blockSignals(False)

        # Move selection to last voice
        if len(self.track.voices) > 0:
            idx = len(self.track.voices) - 1
            self.voice_table.setCurrentCell(idx, 0)
            self.on_voice_table_selected(idx, 0, 0, 0)
        else:
            self.node_table.setRowCount(0)

    
    def on_voice_table_selected(self, currentRow, currentColumn, previousRow, previousColumn):
        if currentRow < 0 or currentRow >= len(self.track.voices):
            self.node_table.setRowCount(0)
            return

        v = self.track.voices[currentRow]
        # iso param
        self.spin_ramp.setValue(v.ramp_percent)
        self.spin_gap.setValue(v.gap_percent)
        self.spin_amp.setValue(v.amplitude)

        # If it's SAM, show the SAM param area
        if v.voice_type in ["SAMBinaural", "MultiSAMBinaural"]:
            self.sam_container.show()
            
            # Set SAM parameters
            self.spin_phase_deviation.setValue(v.phase_deviation)
            self.spin_left_phase_offset.setValue(v.left_phase_offset)
            self.spin_right_phase_offset.setValue(v.right_phase_offset)
            
            # Set sound path combobox
            for i in range(self.combo_sound_path.count()):
                if self.combo_sound_path.itemData(i) == v.sound_path:
                    self.combo_sound_path.setCurrentIndex(i)
                    break
            
            # Show/hide secondary source controls based on voice type
            if v.voice_type == "MultiSAMBinaural":
                self.check_use_secondary.setVisible(True)
                self.label_use_secondary.setVisible(True)
                self.label_secondary_freq_ratio.setVisible(True)
                self.spin_secondary_freq_ratio.setVisible(True)
                self.label_secondary_spatial_ratio.setVisible(True)
                self.spin_secondary_spatial_ratio.setVisible(True)
                self.label_secondary_volume.setVisible(True)
                self.spin_secondary_volume.setVisible(True)
                
                # Set secondary source parameters
                self.check_use_secondary.setChecked(v.use_secondary_source)
                self.spin_secondary_freq_ratio.setValue(v.secondary_freq_ratio)
                self.spin_secondary_spatial_ratio.setValue(v.secondary_spatial_ratio)
                self.spin_secondary_volume.setValue(v.secondary_volume)
                
                # Enable/disable based on use_secondary_source
                self.spin_secondary_freq_ratio.setEnabled(v.use_secondary_source)
                self.spin_secondary_spatial_ratio.setEnabled(v.use_secondary_source)
                self.spin_secondary_volume.setEnabled(v.use_secondary_source)
            else:
                # Hide secondary source controls for regular SAM
                self.check_use_secondary.setVisible(False)
                self.label_use_secondary.setVisible(False)
                self.label_secondary_freq_ratio.setVisible(False)
                self.spin_secondary_freq_ratio.setVisible(False)
                self.label_secondary_spatial_ratio.setVisible(False)
                self.spin_secondary_spatial_ratio.setVisible(False)
                self.label_secondary_volume.setVisible(False)
                self.spin_secondary_volume.setVisible(False)
        else:
            self.sam_container.hide()

        # Rebuild node table for all voice types
        self.node_table.blockSignals(True)
        self.node_table.setRowCount(len(v.nodes))
        for r, nd in enumerate(v.nodes):
            self.node_table.setItem(r, 0, QTableWidgetItem(str(nd.duration)))
            self.node_table.setItem(r, 1, QTableWidgetItem(str(nd.base_freq)))
            self.node_table.setItem(r, 2, QTableWidgetItem(str(nd.beat_freq)))
            self.node_table.setItem(r, 3, QTableWidgetItem(str(nd.volume_left)))
            self.node_table.setItem(r, 4, QTableWidgetItem(str(nd.volume_right)))
        self.node_table.blockSignals(False)

        self.update_graph()

    def on_voice_table_item_changed(self, item):
        row = item.row()
        col = item.column()
        if row < 0 or row >= len(self.track.voices):
            return
        v = self.track.voices[row]

        if col == 1:
            # name
            v.voice_name = item.text()
        elif col == 3:
            # mute
            v.muted = (item.checkState() == Qt.Checked)
        elif col == 4:
            # view
            v.view_enabled = (item.checkState() == Qt.Checked)

        self.update_graph()

    def add_node(self):
        row = self.voice_table.currentRow()
        if row < 0 or row >= len(self.track.voices):
            return
        v = self.track.voices[row]
        
        # Add appropriate node based on voice type
        if v.voice_type in ["SAMBinaural", "MultiSAMBinaural"]:
            new_node = NodeData(
                5.0, 250.0, 0.5, 0.8, 0.8,  # Basic parameters
                phase_deviation=v.phase_deviation,
                left_phase_offset=v.left_phase_offset,
                right_phase_offset=v.right_phase_offset,
                sound_path=v.sound_path
            )
        else:
            # Standard node for other voice types
            new_node = NodeData(5.0, 100.0, 0.0, 1.0, 1.0)
            
        v.nodes.append(new_node)
        self.on_voice_table_selected(row, 0, 0, 0)

    def remove_node(self):
        row = self.voice_table.currentRow()
        if row < 0 or row >= len(self.track.voices):
            return
        v = self.track.voices[row]
        sel_rows = sorted(set([r.row() for r in self.node_table.selectedItems()]), reverse=True)
        for r in sel_rows:
            if 0 <= r < len(v.nodes):
                v.nodes.pop(r)
        self.on_voice_table_selected(row, 0, 0, 0)

    def connect_node_table_selection(self):
        """Connect node table selection to update the graph selection."""
        self.node_table.itemSelectionChanged.connect(self.on_node_table_selection_changed)

    def on_node_table_selection_changed(self):
        """Update graph selection when node table selection changes."""
        voice_idx = self.voice_table.currentRow()
        if voice_idx < 0 or voice_idx >= len(self.track.voices):
            return
        
        # Clear current selection
        self.selected_nodes.clear()
        
        # Add all selected nodes
        for item in self.node_table.selectedItems():
            node_idx = item.row()
            self.selected_nodes.add((voice_idx, node_idx))
        
        # Update graph
        self.highlight_selected_nodes()

    def on_node_table_item_changed(self, item):
        vrow = self.voice_table.currentRow()
        if vrow < 0 or vrow >= len(self.track.voices):
            return
        v = self.track.voices[vrow]
        row = item.row()
        if row < 0 or row >= len(v.nodes):
            return

        nd = v.nodes[row]
        val = item.text()
        try:
            fval = float(val)
        except ValueError:
            return
        col = item.column()
        if col == 0:
            nd.duration = fval
        elif col == 1:
            nd.base_freq = fval
        elif col == 2:
            nd.beat_freq = fval
        elif col == 3:
            nd.volume_left = fval
        elif col == 4:
            nd.volume_right = fval

        self.update_graph()

    ####################################################################
    # Iso Param
    ####################################################################
    def on_iso_params_changed(self):
        row = self.voice_table.currentRow()
        if row < 0 or row >= len(self.track.voices):
            return
        v = self.track.voices[row]
        v.ramp_percent = self.spin_ramp.value()
        v.gap_percent = self.spin_gap.value()
        v.amplitude = self.spin_amp.value()

    ####################################################################
    # SAM Param
    ####################################################################
    def on_sam_params_changed(self):
        row = self.voice_table.currentRow()
        if row < 0 or row >= len(self.track.voices):
            return
        v = self.track.voices[row]
        if v.voice_type not in ["SAMBinaural", "MultiSAMBinaural"]:
            return

        # Update voice-level parameters
        v.phase_deviation = self.spin_phase_deviation.value()
        v.left_phase_offset = self.spin_left_phase_offset.value()
        v.right_phase_offset = self.spin_right_phase_offset.value()
        v.sound_path = self.combo_sound_path.currentData()
        
        # Update secondary source parameters if MultiSAM
        if v.voice_type == "MultiSAMBinaural":
            v.secondary_freq_ratio = self.spin_secondary_freq_ratio.value()
            v.secondary_spatial_ratio = self.spin_secondary_spatial_ratio.value()
            v.secondary_volume = self.spin_secondary_volume.value()

        # Update each node with the same parameters (if user wants global control)
        # You might want to add a checkbox in the UI to toggle this behavior
        update_nodes = True  # Add UI control for this if needed
        if update_nodes:
            for node in v.nodes:
                node.phase_deviation = v.phase_deviation
                node.left_phase_offset = v.left_phase_offset
                node.right_phase_offset = v.right_phase_offset
                node.sound_path = v.sound_path

        self.update_graph() 
    ####################################################################
    # Graph & Node Click
    ####################################################################
    def update_graph(self):
        self.plot_widget.clear()
        self.selected_nodes.clear()

        param = self.plot_param_combo.currentText()
        color_count = 0

        for i, v in enumerate(self.track.voices):
            if not v.view_enabled:
                continue
            c = self.color_list[color_count % len(self.color_list)]
            color_count += 1
            pen_color = pg.mkColor(c[0], c[1], c[2], 200)

            times, vals = self.get_voice_plot_data(v, param)
            self.plot_widget.plot(times, vals, pen=pen_color)

            scatter = pg.ScatterPlotItem(size=10, brush=pen_color)
            spots = []
            for idx in range(len(v.nodes)):
                if idx < len(times):
                    spots.append(dict(pos=(times[idx], vals[idx]), data=(i, idx)))
            scatter.addPoints(spots)
            scatter.sigClicked.connect(self.on_scatter_clicked)
            self.plot_widget.addItem(scatter)

        self.plot_widget.enableAutoRange()

    def get_voice_plot_data(self, v, param):
        """
        Build times, values from node durations & the chosen param.
        """
        times = []
        vals = []
        t = 0.0
        for nd in v.nodes:
            times.append(t)
            vals.append(self.node_param_value(nd, param))
            t += nd.duration
        if v.nodes:
            times.append(t)
            vals.append(self.node_param_value(v.nodes[-1], param))
        return times, vals

    def node_param_value(self, nd, param):
        if param == "base_freq":
            return nd.base_freq
        elif param == "beat_freq":
            return nd.beat_freq
        elif param == "volume":
            return (nd.volume_left + nd.volume_right)/2.0
        elif param == "stereo_balance":
            return nd.volume_right - nd.volume_left
        return 0.0

    def on_scatter_clicked(self, scatter, points):
        if not points:
            return
        pt = points[0]
        (voice_idx, node_idx) = pt.data()

        mods = QApplication.keyboardModifiers()
        if mods & Qt.ControlModifier:
            # toggle selection
            if (voice_idx, node_idx) in self.selected_nodes:
                self.selected_nodes.remove((voice_idx, node_idx))
            else:
                self.selected_nodes.add((voice_idx, node_idx))
                
            # Update node table selection to match
            self.voice_table.setCurrentCell(voice_idx, 0)
            self.node_table.clearSelection()
            for _, n_idx in self.selected_nodes:
                if n_idx < self.node_table.rowCount():
                    self.node_table.selectRow(n_idx)
                    
        elif mods & Qt.ShiftModifier:
            # Single node edit
            self.edit_single_node(voice_idx, node_idx)
        else:
            # Clear selection and select just this node
            self.selected_nodes.clear()
            self.selected_nodes.add((voice_idx, node_idx))
            
            # Update node table selection
            self.voice_table.setCurrentCell(voice_idx, 0)
            self.node_table.clearSelection()
            self.node_table.selectRow(node_idx)

        # highlight selections
        self.highlight_selected_nodes()

    def highlight_selected_nodes(self):
        self.plot_widget.clear()
        param = self.plot_param_combo.currentText()
        color_count = 0

        for i, v in enumerate(self.track.voices):
            if not v.view_enabled:
                continue
            c = self.color_list[color_count % len(self.color_list)]
            color_count += 1
            pen_color = pg.mkColor(c[0], c[1], c[2], 200)

            times, vals = self.get_voice_plot_data(v, param)
            self.plot_widget.plot(times, vals, pen=pen_color)

            scatter = pg.ScatterPlotItem(size=10)
            spots = []
            for idx in range(len(v.nodes)):
                if idx < len(times):
                    is_sel = ((i, idx) in self.selected_nodes)
                    if is_sel:
                        # highlight
                        spots.append(dict(pos=(times[idx], vals[idx]),
                                          brush=pg.mkBrush(255,255,255),
                                          pen=pg.mkPen('k', width=1),
                                          size=12,
                                          data=(i, idx)))
                    else:
                        spots.append(dict(pos=(times[idx], vals[idx]),
                                          brush=pen_color, pen=None,
                                          data=(i, idx)))
            scatter.addPoints(spots)
            scatter.sigClicked.connect(self.on_scatter_clicked)
            self.plot_widget.addItem(scatter)

        self.plot_widget.enableAutoRange()

    def edit_single_node(self, voice_idx, node_idx):
        if voice_idx < 0 or voice_idx >= len(self.track.voices):
            return
        v = self.track.voices[voice_idx]
        if node_idx < 0 or node_idx >= len(v.nodes):
            return
        nd = v.nodes[node_idx]
        dlg = NodeEditDialog(nd, voice_type=v.voice_type, parent=self)
        if dlg.exec_() == dlg.Accepted:
            self.refresh_voice_table()

    ####################################################################
    # Audio Generation
    ####################################################################
    def generate_full_audio_if_needed(self):
        if self.is_playing:
            return
        
        print("Generating full audio...")
        voices = []
        for i, v in enumerate(self.track.voices):
            print(f"Processing voice {i}: {v.voice_name} ({v.voice_type})")
            voice_obj = self.build_voice_from_data(v)
            voices.append(voice_obj)
            
        print(f"Generating audio from {len(voices)} voices...")
        self.final_audio = audio_engine.generate_track_audio(voices, self.track.sample_rate)
        print(f"Generated audio shape: {self.final_audio.shape if self.final_audio is not None else 'None'}")
        print(f"Audio max value: {np.max(np.abs(self.final_audio)) if self.final_audio is not None and self.final_audio.size > 0 else 'N/A'}")

    def build_voice_from_data(self, vdata):
        if vdata.muted or not vdata.nodes:
            total_dur = sum(nd.duration for nd in vdata.nodes) or 5.0
            return audio_engine.PinkNoiseVoice(
                [audio_engine.Node(total_dur, 0, 0, 0, 0)],
                self.track.sample_rate
            )
            
        # Convert NodeData -> audio_engine.Node
        nodes = []
        for nd in vdata.nodes:
            # SAM-specific parameters need to be included for SAM voices
            node = audio_engine.Node(
                nd.duration,
                nd.base_freq,
                nd.beat_freq,
                nd.volume_left,
                nd.volume_right,
                phase_deviation=getattr(nd, 'phase_deviation', 0.7),
                left_phase_offset=getattr(nd, 'left_phase_offset', 0.0),
                right_phase_offset=getattr(nd, 'right_phase_offset', 0.0),
                sound_path=getattr(nd, 'sound_path', SoundPath.CIRCULAR)
            )
            nodes.append(node)

        vt = vdata.voice_type
        if vt == "BinauralBeat":
            return audio_engine.BinauralBeatVoice(nodes, self.track.sample_rate)
        elif vt == "Isochronic":
            return audio_engine.IsochronicVoice(
                nodes, self.track.sample_rate,
                ramp_percent=vdata.ramp_percent,
                gap_percent=vdata.gap_percent,
                amplitude=vdata.amplitude
            )
        elif vt == "AltIsochronic":
            return audio_engine.AltIsochronicVoice(
                nodes, self.track.sample_rate,
                ramp_percent=vdata.ramp_percent,
                gap_percent=vdata.gap_percent,
                amplitude=vdata.amplitude
            )
        elif vt == "AltIsochronic2":
            return audio_engine.AltIsochronic2Voice(
                nodes, self.track.sample_rate,
                ramp_percent=vdata.ramp_percent,
                gap_percent=vdata.gap_percent,
                amplitude=vdata.amplitude
            )
        elif vt == "PinkNoise":
            return audio_engine.PinkNoiseVoice(nodes, self.track.sample_rate)
        elif vt == "ExternalAudio":
            return audio_engine.ExternalAudioVoice(
                nodes, file_path="some_file.wav",
                sample_rate=self.track.sample_rate
            )
        elif vt == "SAMBinaural":
            print(f"Creating SAM Binaural voice with {len(nodes)} nodes")
            try:
                # Create the SAM voice with node-level parameters
                sam_voice = audio_engine.ImprovedSAMVoice(
                    nodes,
                    sample_rate=self.track.sample_rate
                )
                return sam_voice
            except Exception as e:
                print(f"Error creating SAM voice: {e}")
                traceback.print_exc()
                # Fall back to a simpler voice type
                return audio_engine.BinauralBeatVoice(nodes, self.track.sample_rate)
        elif vt == "MultiSAMBinaural":
            print(f"Creating Multi-SAM Binaural voice with {len(nodes)} nodes")
            try:
                # Only use MultiSAMBinauralVoice if secondary source is enabled
                if vdata.use_secondary_source:
                    multi_sam_voice = audio_engine.MultiSAMVoice(
                        nodes,
                        sample_rate=self.track.sample_rate,
                        secondary_freq_ratio=vdata.secondary_freq_ratio,
                        secondary_spatial_ratio=vdata.secondary_spatial_ratio,
                        secondary_volume=vdata.secondary_volume
                    )
                    return multi_sam_voice
                else:
                    # Use regular SAMBinauralVoice if secondary is disabled
                    return audio_engine.ImprovedSAMVoice(nodes, self.track.sample_rate)
            except Exception as e:
                print(f"Error creating Multi-SAM voice: {e}")
                traceback.print_exc()
                # Fall back to a simpler voice type
                return audio_engine.BinauralBeatVoice(nodes, self.track.sample_rate)
        else:
            return audio_engine.BinauralBeatVoice(nodes, self.track.sample_rate)

    def play_track(self):
        if self.is_playing:
            return
        self.generate_full_audio_if_needed()
        if self.final_audio is None:
            return
        total_len_s = self.final_audio.shape[0] / self.track.sample_rate
        if self.current_play_offset >= total_len_s:
            self.current_play_offset = 0.0

        start_idx = int(self.current_play_offset * self.track.sample_rate)
        if start_idx < 0: start_idx = 0
        if start_idx >= self.final_audio.shape[0]:
            start_idx = 0

        sd.play(self.final_audio[start_idx:], samplerate=self.track.sample_rate)
        self.is_playing = True
        self.play_start_time = sd.get_stream().time
        self.update_playhead()
        self.play_timer.start()

    def stop_track(self):
        sd.stop()
        self.is_playing = False
        self.play_timer.stop()
        self.lbl_play_time.setText("Time: 0.00 / 0.00")

    def update_playhead(self):
        if not self.is_playing:
            return
        d = sd.get_stream().time - self.play_start_time
        self.play_start_time = sd.get_stream().time
        self.current_play_offset += d

        track_len = 0.0
        if self.final_audio is not None:
            track_len = self.final_audio.shape[0]/self.track.sample_rate
        if self.current_play_offset > track_len:
            self.stop_track()
            self.current_play_offset = 0.0
            return

        self.lbl_play_time.setText(f"Time: {self.current_play_offset:.2f} / {track_len:.2f}")
        if track_len > 0:
            ratio = self.current_play_offset / track_len
            self.scrub_slider.blockSignals(True)
            self.scrub_slider.setValue(int(ratio * self.scrub_slider.maximum()))
            self.scrub_slider.blockSignals(False)

    def on_scrub_changed(self, val):
        if self.final_audio is None:
            self.generate_full_audio_if_needed()
        track_len = 0.0
        if self.final_audio is not None:
            track_len = self.final_audio.shape[0]/self.track.sample_rate
        ratio = val / self.scrub_slider.maximum()
        self.current_play_offset = ratio * track_len

        if self.is_playing:
            self.stop_track()
            self.play_track()
        else:
            self.lbl_play_time.setText(f"Time: {self.current_play_offset:.2f} / {track_len:.2f}")

    ####################################################################
    # Save / Load
    ####################################################################
    def save_json(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Track JSON", "", "JSON Files (*.json)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.track.to_dict(), f, indent=2)

    def load_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Track JSON", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.track = Track.from_dict(data)
            self.final_audio = None
            self.current_play_offset = 0.0
            self.refresh_voice_table()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load JSON:\n{e}")    # ----------------------------------------------------------------
    # Export
    # ----------------------------------------------------------------
    def export_wav(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export WAV", "", "WAV Files (*.wav)")
        if path:
            self.export_generic(path, "wav")

    def export_flac(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export FLAC", "", "FLAC Files (*.flac)")
        if path:
            self.export_generic(path, "flac")

    def export_mp3(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export MP3", "", "MP3 Files (*.mp3)")
        if path:
            self.export_generic(path, "mp3")

    def export_generic(self, path, fmt):
        self.generate_full_audio_if_needed()
        if self.final_audio is None:
            QMessageBox.warning(self, "Export Error", "No audio data to export")
            return
        
        # Add verification
        max_val = np.max(np.abs(self.final_audio)) if self.final_audio.size > 0 else 0
        print(f"Audio verification - Shape: {self.final_audio.shape}, Max value: {max_val}")
        
        if max_val < 1e-6:
            print("Warning: Audio appears to be silent")
        if fmt == "wav":
            audio_engine.export_wav(self.final_audio, self.track.sample_rate, path)
        elif fmt == "flac":
            audio_engine.export_flac(self.final_audio, self.track.sample_rate, path)
        elif fmt == "mp3":
            audio_engine.export_mp3(self.final_audio, self.track.sample_rate, path)
        QMessageBox.information(self, "Export", f"Exported track as {fmt.upper()}:\n{path}")


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
