
import sys
import json
import numpy as np
import sounddevice as sd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QAbstractItemView, QFileDialog,
    QLineEdit, QMessageBox, QAction, QMenuBar, QDialog, QDialogButtonBox,
    QFormLayout, QSlider
)
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg

import audio_engine


# --------------------------------------------------------------------
# Data Models (Added 'voice_name' to VoiceData for user description)
# --------------------------------------------------------------------

class NodeData:
    def __init__(self, duration, base_freq, beat_freq, volume_left, volume_right):
        self.duration = duration
        self.base_freq = base_freq
        self.beat_freq = beat_freq
        self.volume_left = volume_left
        self.volume_right = volume_right

    def to_dict(self):
        return {
            "duration": self.duration,
            "base_freq": self.base_freq,
            "beat_freq": self.beat_freq,
            "volume_left": self.volume_left,
            "volume_right": self.volume_right
        }

    @staticmethod
    def from_dict(d):
        return NodeData(
            d["duration"], d["base_freq"], d["beat_freq"],
            d["volume_left"], d["volume_right"]
        )

class VoiceData:
    def __init__(self, voice_type, voice_name="Voice", ramp_percent=0.2,
                 gap_percent=0.15, amplitude=1.0):
        self.voice_type = voice_type
        self.voice_name = voice_name
        self.ramp_percent = ramp_percent
        self.gap_percent = gap_percent
        self.amplitude = amplitude

        # Existing
        self.nodes = []
        self.muted = False
        self.view_enabled = True

        # NEW: SAM-specific parameters
        # Some initial defaults; feel free to tweak
        self.sam_carrier_from_nodes = False
        self.sam_carrier_freq = 200.0
        self.sam_arc_freq_left = 10.0
        self.sam_arc_freq_right = 10.0
        self.sam_arc_peak_left = 1.57   # ~ pi/2
        self.sam_arc_peak_right = 1.57
        self.sam_phase_offset_left = 0.0
        self.sam_phase_offset_right = 0.0
        self.sam_arc_function = "sin"   # or "triangle"

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

            # NEW SAM fields:
            "sam_carrier_from_nodes": self.sam_carrier_from_nodes,
            "sam_carrier_freq": self.sam_carrier_freq,
            "sam_arc_freq_left": self.sam_arc_freq_left,
            "sam_arc_freq_right": self.sam_arc_freq_right,
            "sam_arc_peak_left": self.sam_arc_peak_left,
            "sam_arc_peak_right": self.sam_arc_peak_right,
            "sam_phase_offset_left": self.sam_phase_offset_left,
            "sam_phase_offset_right": self.sam_phase_offset_right,
            "sam_arc_function": self.sam_arc_function,
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

        # Load nodes
        for nd in d.get("nodes", []):
            v.nodes.append(NodeData.from_dict(nd))

        # Load SAM fields (if present), else fallback to default
        v.sam_carrier_from_nodes = d.get("sam_carrier_from_nodes", False)
        v.sam_carrier_freq = d.get("sam_carrier_freq", 200.0)
        v.sam_arc_freq_left = d.get("sam_arc_freq_left", 10.0)
        v.sam_arc_freq_right = d.get("sam_arc_freq_right", 10.0)
        v.sam_arc_peak_left = d.get("sam_arc_peak_left", 1.57)
        v.sam_arc_peak_right = d.get("sam_arc_peak_right", 1.57)
        v.sam_phase_offset_left = d.get("sam_phase_offset_left", 0.0)
        v.sam_phase_offset_right = d.get("sam_phase_offset_right", 0.0)
        v.sam_arc_function = d.get("sam_arc_function", "sin")

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


# --------------------------------------------------------------------
# Popup Dialogs
# --------------------------------------------------------------------

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
            "SpatialAngleMod"
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
    def __init__(self, node, parent=None):
        super().__init__(parent)
        self.node = node
        self.setWindowTitle("Edit Node")

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
        self.spin_beat_freq.setSingleStep(1.0)
        self.spin_beat_freq.setValue(self.node.beat_freq)

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
        form.addRow("Beat Frequency:", self.spin_beat_freq)
        form.addRow("Volume Left:", self.spin_vol_left)
        form.addRow("Volume Right:", self.spin_vol_right)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.accept_data)
        buttonBox.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(buttonBox)
        self.setLayout(layout)

    def accept_data(self):
        self.node.duration = self.spin_duration.value()
        self.node.base_freq = self.spin_base_freq.value()
        self.node.beat_freq = self.spin_beat_freq.value()
        self.node.volume_left = self.spin_vol_left.value()
        self.node.volume_right = self.spin_vol_right.value()
        self.accept()


# --------------------------------------------------------------------
# Main Window
# --------------------------------------------------------------------
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
        self.setWindowTitle("Gnaural Py - Color & Name UI")
        self.resize(1400, 800)

        self.track = Track()
        self.final_audio = None
        self.is_playing = False
        self.play_start_time = 0
        self.current_play_offset = 0.0

        # Store multi-node selection for the graph
        self.selected_nodes = set()

        self.create_menu()
        self.setup_ui()

        self.play_timer = QTimer()
        self.play_timer.setInterval(200)
        self.play_timer.timeout.connect(self.update_playhead)

    # ----------------------------------------------------------------
    # UI Layout
    # ----------------------------------------------------------------
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

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # --------------------------------------------------
        # Voice Table
        # --------------------------------------------------
        self.voice_table = QTableWidget()
        self.voice_table.setColumnCount(5)
        self.voice_table.setHorizontalHeaderLabels(["Color","Name","Type","Mute","View"])
        self.voice_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.voice_table.setEditTriggers(QAbstractItemView.AllEditTriggers)
        # We'll track changes
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

        # Node Table
        self.node_table = QTableWidget()
        self.node_table.setColumnCount(5)
        self.node_table.setHorizontalHeaderLabels(["Duration","BaseFreq","BeatFreq","VolLeft","VolRight"])
        self.node_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.node_table.setSelectionMode(QAbstractItemView.MultiSelection)
        self.node_table.itemChanged.connect(self.on_node_table_item_changed)

        self.btn_add_node = QPushButton("Add Node")
        self.btn_add_node.clicked.connect(self.add_node)
        self.btn_remove_node = QPushButton("Remove Node")
        self.btn_remove_node.clicked.connect(self.remove_node)

        node_layout = QVBoxLayout()
        node_layout.addWidget(QLabel("Nodes:"))
        node_layout.addWidget(self.node_table)
        node_btn_layout = QHBoxLayout()
        node_btn_layout.addWidget(self.btn_add_node)
        node_btn_layout.addWidget(self.btn_remove_node)
        node_layout.addLayout(node_btn_layout)

        # Isochronic controls
        self.label_ramp = QLabel("Ramp%")
        self.spin_ramp = QDoubleSpinBox()
        self.spin_ramp.setRange(0.0, 0.5)
        self.spin_ramp.setSingleStep(0.01)
        self.spin_ramp.valueChanged.connect(self.on_iso_params_changed)

        self.label_gap = QLabel("Gap%")
        self.spin_gap = QDoubleSpinBox()
        self.spin_gap.setRange(0.0, 0.5)
        self.spin_gap.setSingleStep(0.01)
        self.spin_gap.valueChanged.connect(self.on_iso_params_changed)

        self.label_amp = QLabel("Amplitude")
        self.spin_amp = QDoubleSpinBox()
        self.spin_amp.setRange(0.0, 10.0)
        self.spin_amp.setSingleStep(0.1)
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

        # SAM
        self.sam_group_label = QLabel("Spatial Angle Modulation (SAM) Params")
        self.check_carrier_from_nodes = QComboBox()  
        self.check_carrier_from_nodes.addItems(["False", "True"])  # or use a QCheckBox if you prefer
        self.label_sam_carrier = QLabel("Carrier Freq")
        self.spin_sam_carrier = QDoubleSpinBox()
        self.spin_sam_carrier.setRange(0.0, 20000.0)
        self.spin_sam_carrier.setValue(200.0)

        self.label_arc_freq_left = QLabel("Arc Freq Left")
        self.spin_arc_freq_left = QDoubleSpinBox()
        self.spin_arc_freq_left.setRange(0.0, 200.0)
        self.spin_arc_freq_left.setValue(10.0)

        self.label_arc_freq_right = QLabel("Arc Freq Right")
        self.spin_arc_freq_right = QDoubleSpinBox()
        self.spin_arc_freq_right.setRange(0.0, 200.0)
        self.spin_arc_freq_right.setValue(10.0)

        self.label_arc_peak_left = QLabel("Arc Peak Left (rad)")
        self.spin_arc_peak_left = QDoubleSpinBox()
        self.spin_arc_peak_left.setRange(0.0, 6.28)
        self.spin_arc_peak_left.setValue(1.57)

        self.label_arc_peak_right = QLabel("Arc Peak Right (rad)")
        self.spin_arc_peak_right = QDoubleSpinBox()
        self.spin_arc_peak_right.setRange(0.0, 6.28)
        self.spin_arc_peak_right.setValue(1.57)

        self.label_phase_offset_left = QLabel("Phase Offset Left (rad)")
        self.spin_phase_offset_left = QDoubleSpinBox()
        self.spin_phase_offset_left.setRange(-6.28, 6.28)
        self.spin_phase_offset_left.setValue(0.0)

        self.label_phase_offset_right = QLabel("Phase Offset Right (rad)")
        self.spin_phase_offset_right = QDoubleSpinBox()
        self.spin_phase_offset_right.setRange(-6.28, 6.28)
        self.spin_phase_offset_right.setValue(0.0)

        self.label_arc_function = QLabel("Arc Function")
        self.combo_arc_function = QComboBox()
        self.combo_arc_function.addItems(["sin", "triangle"])

        # We'll arrange them in a grid
        self.sam_layout = QGridLayout()
        self.sam_layout.addWidget(self.sam_group_label,      0, 0, 1, 2)
        self.sam_layout.addWidget(QLabel("Use Node Carriers?"), 1, 0)
        self.sam_layout.addWidget(self.check_carrier_from_nodes,1, 1)
        self.sam_layout.addWidget(self.label_sam_carrier,     2, 0)
        self.sam_layout.addWidget(self.spin_sam_carrier,      2, 1)
        self.sam_layout.addWidget(self.label_arc_freq_left,    3, 0)
        self.sam_layout.addWidget(self.spin_arc_freq_left,     3, 1)
        self.sam_layout.addWidget(self.label_arc_freq_right,   4, 0)
        self.sam_layout.addWidget(self.spin_arc_freq_right,    4, 1)
        self.sam_layout.addWidget(self.label_arc_peak_left,    5, 0)
        self.sam_layout.addWidget(self.spin_arc_peak_left,     5, 1)
        self.sam_layout.addWidget(self.label_arc_peak_right,   6, 0)
        self.sam_layout.addWidget(self.spin_arc_peak_right,    6, 1)
        self.sam_layout.addWidget(self.label_phase_offset_left,7, 0)
        self.sam_layout.addWidget(self.spin_phase_offset_left, 7, 1)
        self.sam_layout.addWidget(self.label_phase_offset_right,8,0)
        self.sam_layout.addWidget(self.spin_phase_offset_right,8,1)
        self.sam_layout.addWidget(self.label_arc_function,     9, 0)
        self.sam_layout.addWidget(self.combo_arc_function,     9, 1)

        # Hook up signals
        self.check_carrier_from_nodes.currentIndexChanged.connect(self.on_sam_params_changed)
        self.spin_sam_carrier.valueChanged.connect(self.on_sam_params_changed)
        self.spin_arc_freq_left.valueChanged.connect(self.on_sam_params_changed)
        self.spin_arc_freq_right.valueChanged.connect(self.on_sam_params_changed)
        self.spin_arc_peak_left.valueChanged.connect(self.on_sam_params_changed)
        self.spin_arc_peak_right.valueChanged.connect(self.on_sam_params_changed)
        self.spin_phase_offset_left.valueChanged.connect(self.on_sam_params_changed)
        self.spin_phase_offset_right.valueChanged.connect(self.on_sam_params_changed)
        self.combo_arc_function.currentIndexChanged.connect(self.on_sam_params_changed)

        # If you want, you can hide these by default, or keep them visible
        # We'll add them below your iso_layout or in the same "middle_layout"
        self.sam_container = QWidget()
        self.sam_container.setLayout(self.sam_layout)

        # Suppose your 'middle_layout' is a QVBoxLayout that has
        #   node_layout, iso_layout, etc. We'll just add the sam_container below:
        middle_layout.addWidget(self.sam_container)  # end of the code snippet

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
        self.scrub_slider = QSlider(Qt.Horizontal)
        self.scrub_slider.setRange(0, 100)
        self.scrub_slider.setValue(0)
        self.scrub_slider.valueChanged.connect(self.on_scrub_changed)

        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self.play_track)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_track)
        self.lbl_play_time = QLabel("Time: 0.00 / 0.00")

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

        # Combine
        voices_container = QWidget()
        voices_container.setLayout(voice_layout)

        middle_container = QWidget()
        middle_container.setLayout(middle_layout)

        right_container = QWidget()
        right_container.setLayout(right_graph_layout)

        main_layout.addWidget(voices_container, stretch=2)
        main_layout.addWidget(middle_container, stretch=3)
        main_layout.addWidget(right_container, stretch=4)

    # ----------------------------------------------------------------
    # Voice Table Management
    # ----------------------------------------------------------------
    def add_voice(self):
        dlg = VoiceCreationDialog(self)
        if dlg.exec_() == dlg.Accepted:
            vtype = dlg.voice_type_box.currentText()
            v = VoiceData(voice_type=vtype, voice_name=f"{vtype}")
            # Provide one default node
            v.nodes.append(NodeData(5.0, 100.0, 5.0, 1.0, 1.0))
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
            # color
            color_item = QTableWidgetItem()
            c = self.color_list[i % len(self.color_list)]
            color_item.setBackground(pg.mkColor(c[0], c[1], c[2], 255))
            # not editable
            color_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            self.voice_table.setItem(i, 0, color_item)

            # name
            name_item = QTableWidgetItem(v.voice_name)
            self.voice_table.setItem(i, 1, name_item)

            # type
            type_item = QTableWidgetItem(v.voice_type)
            type_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            self.voice_table.setItem(i, 2, type_item)

            # mute
            mute_item = QTableWidgetItem()
            mute_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            mute_item.setCheckState(Qt.Checked if v.muted else Qt.Unchecked)
            self.voice_table.setItem(i, 3, mute_item)

            # view
            view_item = QTableWidgetItem()
            view_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            view_item.setCheckState(Qt.Checked if v.view_enabled else Qt.Unchecked)
            self.voice_table.setItem(i, 4, view_item)

        self.voice_table.blockSignals(False)
        self.voice_table.setCurrentCell(len(self.track.voices)-1, 0)
        self.on_voice_table_selected(len(self.track.voices)-1, 0, 0, 0)

    def on_voice_table_selected(self, currentRow, currentColumn, previousRow, previousColumn):
        if currentRow < 0 or currentRow >= len(self.track.voices):
            self.node_table.setRowCount(0)
            return
        v = self.track.voices[currentRow]
        self.spin_ramp.setValue(v.ramp_percent)
        self.spin_gap.setValue(v.gap_percent)
        self.spin_amp.setValue(v.amplitude)
        if v.voice_type == "SpatialAngleMod":
            self.sam_container.show()

            # Populate each field
            idx_bool = 1 if v.sam_carrier_from_nodes else 0
            self.check_carrier_from_nodes.setCurrentIndex(idx_bool)
            self.spin_sam_carrier.setValue(v.sam_carrier_freq)
            self.spin_arc_freq_left.setValue(v.sam_arc_freq_left)
            self.spin_arc_freq_right.setValue(v.sam_arc_freq_right)
            self.spin_arc_peak_left.setValue(v.sam_arc_peak_left)
            self.spin_arc_peak_right.setValue(v.sam_arc_peak_right)
            self.spin_phase_offset_left.setValue(v.sam_phase_offset_left)
            self.spin_phase_offset_right.setValue(v.sam_phase_offset_right)

            # arc_function
            # If you have 2 items: "sin"=0, "triangle"=1, ...
            idx_func = 0 if v.sam_arc_function=="sin" else 1
            self.combo_arc_function.setCurrentIndex(idx_func)
        else:
            self.sam_container.hide()
            # Populate node table
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
            state = item.checkState()
            v.muted = (state == Qt.Checked)
        elif col == 4:
            # view
            state = item.checkState()
            v.view_enabled = (state == Qt.Checked)

        self.update_graph()

    def on_sam_params_changed(self):
        row = self.voice_table.currentRow()
        if row < 0 or row >= len(self.track.voices):
            return
        v = self.track.voices[row]
        if v.voice_type != "SpatialAngleMod":
            return  # do nothing if not SAM

        # read values back from UI
        v.sam_carrier_from_nodes = (self.check_carrier_from_nodes.currentIndex() == 1)
        v.sam_carrier_freq = self.spin_sam_carrier.value()
        v.sam_arc_freq_left = self.spin_arc_freq_left.value()
        v.sam_arc_freq_right = self.spin_arc_freq_right.value()
        v.sam_arc_peak_left = self.spin_arc_peak_left.value()
        v.sam_arc_peak_right = self.spin_arc_peak_right.value()
        v.sam_phase_offset_left = self.spin_phase_offset_left.value()
        v.sam_phase_offset_right = self.spin_phase_offset_right.value()
        v.sam_arc_function = self.combo_arc_function.currentText()

        # If you want to auto-update the preview graph:
        self.update_graph()

    # ----------------------------------------------------------------
    # Node Table Management
    # ----------------------------------------------------------------
    def add_node(self):
        row = self.voice_table.currentRow()
        if row < 0 or row >= len(self.track.voices):
            return
        v = self.track.voices[row]
        v.nodes.append(NodeData(5.0, 100.0, 5.0, 1.0, 1.0))
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

    # ----------------------------------------------------------------
    # Isochronic Param UI
    # ----------------------------------------------------------------
    def on_iso_params_changed(self):
        row = self.voice_table.currentRow()
        if row < 0 or row >= len(self.track.voices):
            return
        v = self.track.voices[row]
        v.ramp_percent = self.spin_ramp.value()
        v.gap_percent = self.spin_gap.value()
        v.amplitude = self.spin_amp.value()

    # ----------------------------------------------------------------
    # Graph & Node Selection
    # ----------------------------------------------------------------
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
        times = []
        vals = []
        t = 0.0
        for nd in v.nodes:
            times.append(t)
            vals.append(self.node_param_value(nd, param))
            t += nd.duration
        # repeat last for a continuous line
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
            # toggle
            if (voice_idx, node_idx) in self.selected_nodes:
                self.selected_nodes.remove((voice_idx, node_idx))
            else:
                self.selected_nodes.add((voice_idx, node_idx))
        else:
            self.selected_nodes.clear()
            self.selected_nodes.add((voice_idx, node_idx))

        # SHIFT => open editor
        if mods & Qt.ShiftModifier:
            self.edit_single_node(voice_idx, node_idx)

        # set the voice_table selection & node_table selection
        self.voice_table.setCurrentCell(voice_idx, 0)
        self.node_table.setCurrentCell(node_idx, 0)

        # highlight
        self.highlight_selected_nodes()

    def highlight_selected_nodes(self):
        # We re‐draw the entire graph so selected nodes appear differently
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

            # Scatter with possible highlight
            scatter = pg.ScatterPlotItem(size=10)
            spots = []
            for idx in range(len(v.nodes)):
                if idx < len(times):
                    is_selected = ((i, idx) in self.selected_nodes)
                    if is_selected:
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
        dlg = NodeEditDialog(nd, self)
        if dlg.exec_() == dlg.Accepted:
            self.refresh_voice_table()

    # ----------------------------------------------------------------
    # Playback
    # ----------------------------------------------------------------
    def generate_full_audio_if_needed(self):
        voices = []
        for v in self.track.voices:
            voices.append(self.build_voice_from_data(v))
        self.final_audio = audio_engine.generate_track_audio(voices, self.track.sample_rate)

    def build_voice_from_data(self, vdata):
        if vdata.muted or not vdata.nodes:
            total_dur = sum(nd.duration for nd in vdata.nodes) or 5.0
            return audio_engine.PinkNoiseVoice(
                [audio_engine.Node(total_dur, 0, 0, 0, 0)],
                self.track.sample_rate
            )
        nodes = []
        for nd in vdata.nodes:
            node = audio_engine.Node(nd.duration, nd.base_freq, nd.beat_freq,
                                     nd.volume_left, nd.volume_right)
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
            return audio_engine.ExternalAudioVoice(nodes, "some_file.wav", self.track.sample_rate)
        elif vt == "SpatialAngleMod":
            # NEW: read the SAM parameters from vdata
            return audio_engine.SpatialAngleModVoice(
                nodes=nodes,
                sample_rate=self.track.sample_rate,
                carrier_from_nodes=vdata.sam_carrier_from_nodes,
                carrier_freq=vdata.sam_carrier_freq,
                arc_freq_left=vdata.sam_arc_freq_left,
                arc_freq_right=vdata.sam_arc_freq_right,
                arc_peak_left=vdata.sam_arc_peak_left,
                arc_peak_right=vdata.sam_arc_peak_right,
                phase_offset_left=vdata.sam_phase_offset_left,
                phase_offset_right=vdata.sam_phase_offset_right,
                arc_function=vdata.sam_arc_function
            )        
        else:
            return audio_engine.BinauralBeatVoice(nodes, self.track.sample_rate)

    def play_track(self):
        if self.is_playing:
            return
        self.generate_full_audio_if_needed()
        if self.final_audio is None:
            return
        total_len_s = self.final_audio.shape[0]/self.track.sample_rate
        if self.current_play_offset >= total_len_s:
            self.current_play_offset = 0.0
        start_sample = int(self.current_play_offset * self.track.sample_rate)
        if start_sample < 0:
            start_sample = 0
        if start_sample >= self.final_audio.shape[0]:
            start_sample = 0

        sd.play(self.final_audio[start_sample:], samplerate=self.track.sample_rate)
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
        delta = sd.get_stream().time - self.play_start_time
        self.play_start_time = sd.get_stream().time
        self.current_play_offset += delta

        total_len_s = 0.0
        if self.final_audio is not None:
            total_len_s = self.final_audio.shape[0]/self.track.sample_rate

        if self.current_play_offset > total_len_s:
            self.stop_track()
            self.current_play_offset = 0.0
            return

        self.lbl_play_time.setText(f"Time: {self.current_play_offset:.2f} / {total_len_s:.2f}")

        if total_len_s > 0:
            ratio = self.current_play_offset / total_len_s
            self.scrub_slider.blockSignals(True)
            self.scrub_slider.setValue(int(ratio*self.scrub_slider.maximum()))
            self.scrub_slider.blockSignals(False)

    def on_scrub_changed(self, val):
        if self.final_audio is None:
            self.generate_full_audio_if_needed()
        total_len_s = 0.0
        if self.final_audio is not None:
            total_len_s = self.final_audio.shape[0]/self.track.sample_rate
        ratio = val/self.scrub_slider.maximum()
        self.current_play_offset = ratio*total_len_s
        if self.is_playing:
            self.stop_track()
            self.play_track()
        else:
            self.lbl_play_time.setText(f"Time: {self.current_play_offset:.2f} / {total_len_s:.2f}")

    # ----------------------------------------------------------------
    # Save / Load JSON
    # ----------------------------------------------------------------
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
                d = json.load(f)
            self.track = Track.from_dict(d)
            self.final_audio = None
            self.current_play_offset = 0.0
            self.refresh_voice_table()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load JSON:\n{e}")

    # ----------------------------------------------------------------
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
            return
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

