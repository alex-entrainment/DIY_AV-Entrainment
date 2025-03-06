
import sys
from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QApplication,
                             QStatusBar, QAction, QFileDialog, QMessageBox)
from ui.step_list_panel import StepListPanel
from ui.step_config_panel import StepConfigPanel
from ui.audio_settings_panel import AudioSettingsPanel
from controllers.step_controller import StepController
from controllers.file_controller import FileController
from sequence_model import Oscillator, StrobeSet, Waveform, PatternMode
from ui.simulator import SimulatorWindow

def clear_layout(layout):
    if layout is not None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("6-LED Sequence Editor with Stepwise Audio Sync")
        self.resize(1300, 700)
        
        self.simulator_window = None  # to hold the simulator instance

        # Controllers.
        self.step_controller = StepController()
        self.file_controller = FileController()
        self.currentFile = None
        self.audio_settings = {}  # global audio settings (for fallback/global parameters)

        # Create central widget and layout.
        central = QWidget(self)
        main_layout = QHBoxLayout(central)
        self.setCentralWidget(central)

        # Create panels.
        self.step_list_panel = StepListPanel()
        self.step_config_panel = StepConfigPanel()
        self.audio_settings_panel = AudioSettingsPanel()

        # Add panels to main layout.
        main_layout.addWidget(self.step_list_panel, 2)
        main_layout.addWidget(self.step_config_panel, 3)
        main_layout.addWidget(self.audio_settings_panel, 3)

        # Status bar.
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.update_sequence_duration()

        # Menu actions.
        self._create_menu()

        # Wire up signals.
        self.step_list_panel.addStepClicked.connect(self.handle_add_step)
        self.step_list_panel.duplicateStepClicked.connect(self.handle_duplicate_step)
        self.step_list_panel.removeStepClicked.connect(self.handle_remove_step)
        self.step_list_panel.moveUpClicked.connect(self.handle_move_step_up)
        self.step_list_panel.moveDownClicked.connect(self.handle_move_step_down)
        self.step_list_panel.stepSelectionChanged.connect(self.handle_step_selected)
        self.step_config_panel.btn_submit.clicked.connect(self.handle_submit_step)
        self.step_config_panel.btn_apply_all.clicked.connect(self.handle_apply_osc1_to_all)
        self.step_config_panel.mode_combo.currentTextChanged.connect(self.handle_mode_changed)

        # Initialize oscillator/strobe controls.
        self.step_config_panel.build_oscillator_and_strobe_controls(self.step_config_panel.mode_combo.currentText())

        # Start with a default step.
        self.handle_add_step()

    def _create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        new_act = QAction("New", self)
        new_act.triggered.connect(self.handle_new_sequence)
        file_menu.addAction(new_act)
        open_act = QAction("Open", self)
        open_act.triggered.connect(self.handle_load_sequence)
        file_menu.addAction(open_act)
        save_act = QAction("Save", self)
        save_act.triggered.connect(self.handle_save_sequence)
        file_menu.addAction(save_act)
        save_as_act = QAction("Save As", self)
        save_as_act.triggered.connect(self.handle_save_sequence_as)
        file_menu.addAction(save_as_act)
        delete_act = QAction("Delete", self)
        delete_act.triggered.connect(self.handle_delete_sequence_file)
        file_menu.addAction(delete_act)
        simulator_menu = menubar.addMenu("Simulator")
        simulator_act = QAction("Open Simulator", self)
        simulator_act.triggered.connect(self.open_simulator)
        simulator_menu.addAction(simulator_act)

    def open_simulator(self):
        if self.simulator_window is None:
            self.simulator_window = SimulatorWindow(self)
        self.simulator_window.show()
        self.simulator_window.raise_()
        self.simulator_window.activateWindow()

    def update_sequence_duration(self):
        duration = self.step_controller.update_sequence_duration()
        self.status.showMessage(f"Sequence Duration: {duration}")

    def handle_add_step(self):
        mode = self.step_config_panel.mode_combo.currentText()
        new_step = self.step_controller.add_default_step(mode)
        self.step_list_panel.add_step_item(new_step.description)
        self.step_list_panel.set_current_row(len(self.step_controller.steps) - 1)
        self.update_sequence_duration()

    def handle_duplicate_step(self):
        index = self.step_list_panel.step_list.currentRow()
        new_step = self.step_controller.duplicate_step(index)
        if new_step:
            self.step_list_panel.insert_step_item(index + 1, new_step.description)
            self.step_list_panel.set_current_row(index + 1)
            self.update_sequence_duration()

    def handle_remove_step(self):
        index = self.step_list_panel.step_list.currentRow()
        self.step_controller.remove_step(index)
        self.step_list_panel.remove_current_item()
        if not self.step_controller.steps:
            self.step_config_panel.clear_step_fields()
            self.handle_add_step()
        else:
            self.step_list_panel.set_current_row(min(index, len(self.step_controller.steps) - 1))
        self.update_sequence_duration()

    def handle_move_step_up(self):
        index = self.step_list_panel.step_list.currentRow()
        if self.step_controller.move_step_up(index):
            self._refresh_step_list()
            self.step_list_panel.set_current_row(index - 1)

    def handle_move_step_down(self):
        index = self.step_list_panel.step_list.currentRow()
        if self.step_controller.move_step_down(index):
            self._refresh_step_list()
            self.step_list_panel.set_current_row(index + 1)

    def _refresh_step_list(self):
        self.step_list_panel.clear()
        for step in self.step_controller.steps:
            self.step_list_panel.add_step_item(step.description)

    def handle_step_selected(self, index):
        if index < 0 or index >= len(self.step_controller.steps):
            return
        step = self.step_controller.steps[index]
        self.step_config_panel.duration_spin.setValue(step.duration)
        self.step_config_panel.description_edit.setText(step.description)
        if len(step.oscillators) == 1:
            mode = "Combined"
        elif len(step.oscillators) == 2:
            mode = "Split"
        else:
            mode = "Independent"
        self.step_config_panel.mode_combo.setCurrentText(mode)
        self.step_config_panel.build_oscillator_and_strobe_controls(mode)
        for i, osc in enumerate(step.oscillators):
            if i < len(self.step_config_panel.osc_controls):
                ctrl = self.step_config_panel.osc_controls[i]
                ctrl["wave_combo"].setCurrentIndex(osc.waveform.value)
                ctrl["freq_start"].setValue(osc.start_freq)
                ctrl["freq_end"].setValue(osc.end_freq)
                ctrl["duty_start"].setValue(int(osc.start_duty))
                ctrl["duty_end"].setValue(int(osc.end_duty))
                ctrl["rfm_enable"].setChecked(osc.enable_rfm)
                ctrl["rfm_range"].setValue(osc.rfm_range)
                ctrl["rfm_speed"].setValue(osc.rfm_speed)
                ctrl["phase_pattern_combo"].setCurrentIndex(osc.phase_pattern.value)
                ctrl["brightness_pattern_combo"].setCurrentIndex(osc.brightness_pattern.value)
                ctrl["pattern_strength_spin"].setValue(osc.pattern_strength)
                ctrl["pattern_freq_spin"].setValue(osc.pattern_freq)
        for i, sset in enumerate(step.strobe_sets):
            if i < len(self.step_config_panel.strobe_controls):
                ctrl = self.step_config_panel.strobe_controls[i]
                # Convert strobe intensities to int before calling setValue
                ctrl["strobe_start"].setValue(int(sset.start_intensity))
                ctrl["strobe_end"].setValue(int(sset.end_intensity))
        # Update the audio settings panel with the step's audio settings.
        audio_settings = step.audio_settings or {"carriers": []}
        carriers = audio_settings.get("carriers", [])
        for i, cpanel in enumerate(self.audio_settings_panel.carrier_panels):
            if i < len(carriers):
                carrier = carriers[i]
                cpanel.start_freq_left.setValue(carrier.get("start_freq_left", 205.0))
                cpanel.end_freq_left.setValue(carrier.get("end_freq_left", 205.0))
                cpanel.start_freq_right.setValue(carrier.get("start_freq_right", 195.0))
                cpanel.end_freq_right.setValue(carrier.get("end_freq_right", 195.0))
                cpanel.tone_mode_combo.setCurrentText(carrier.get("tone_mode", "Binaural"))
                cpanel.volume.setValue(int(carrier.get("volume", 1.0)*100))
                cpanel.rfm_enable.setChecked(carrier.get("enable_rfm", False))
                cpanel.rfm_range.setValue(carrier.get("rfm_range", 0.5))
                cpanel.rfm_speed.setValue(carrier.get("rfm_speed", 0.2))
            else:
                cpanel.start_freq_left.setValue(205.0 if i == 0 else 200.0)
                cpanel.end_freq_left.setValue(205.0 if i == 0 else 200.0)
                cpanel.start_freq_right.setValue(195.0 if i == 0 else 200.0)
                cpanel.end_freq_right.setValue(195.0 if i == 0 else 200.0)
                cpanel.tone_mode_combo.setCurrentText("Binaural")
                cpanel.volume.setValue(100 if i == 0 else 50)
                cpanel.rfm_enable.setChecked(False)
                cpanel.rfm_range.setValue(0.5)
                cpanel.rfm_speed.setValue(0.2)

    def handle_apply_osc1_to_all(self):
        mode = self.step_config_panel.mode_combo.currentText()
        if mode != "Independent" or not self.step_config_panel.osc_controls:
            return
        first = self.step_config_panel.osc_controls[0]
        for ctrl in self.step_config_panel.osc_controls[1:]:
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

    def handle_mode_changed(self, mode):
        self.step_config_panel.build_oscillator_and_strobe_controls(mode)

    def handle_submit_step(self):
        index = self.step_list_panel.step_list.currentRow()
        if index < 0 or index >= len(self.step_controller.steps):
            return
        step = self.step_controller.steps[index]
        step.duration = self.step_config_panel.duration_spin.value()
        step.description = self.step_config_panel.description_edit.text()
        mode = self.step_config_panel.mode_combo.currentText()
        num_osc = {"Combined": 1, "Split": 2, "Independent": 6}[mode]
        oscillators = []
        for i in range(num_osc):
            ctrl = self.step_config_panel.osc_controls[i]
            wave_idx = ctrl["wave_combo"].currentIndex()
            wave = Waveform(wave_idx)
            osc_obj = Oscillator(
                start_freq=ctrl["freq_start"].value(),
                end_freq=ctrl["freq_end"].value(),
                waveform=wave,
                start_duty=ctrl["duty_start"].value(),
                end_duty=ctrl["duty_end"].value(),
                enable_rfm=ctrl["rfm_enable"].isChecked(),
                rfm_range=ctrl["rfm_range"].value(),
                rfm_speed=ctrl["rfm_speed"].value(),
                phase_pattern=PatternMode(ctrl["phase_pattern_combo"].currentIndex()),
                brightness_pattern=PatternMode(ctrl["brightness_pattern_combo"].currentIndex()),
                pattern_strength=ctrl["pattern_strength_spin"].value(),
                pattern_freq=ctrl["pattern_freq_spin"].value()
            )
            oscillators.append(osc_obj)
        strobe_sets = []
        if mode == "Combined":
            sctrl = self.step_config_panel.strobe_controls[0]
            strobe_sets = [StrobeSet(list(range(6)), sctrl["strobe_start"].value(),
                                     sctrl["strobe_end"].value(), [1.0])]
        elif mode == "Split":
            for i in range(2):
                sctrl = self.step_config_panel.strobe_controls[i]
                if i == 0:
                    strobe_sets.append(StrobeSet([0, 2, 4], sctrl["strobe_start"].value(),
                                                 sctrl["strobe_end"].value(), [1.0, 0.0]))
                else:
                    strobe_sets.append(StrobeSet([1, 3, 5], sctrl["strobe_start"].value(),
                                                 sctrl["strobe_end"].value(), [0.0, 1.0]))
        else:  # Independent
            for i in range(6):
                sctrl = self.step_config_panel.strobe_controls[i]
                weights = [0.0] * 6
                weights[i] = 1.0
                strobe_sets.append(StrobeSet([i], sctrl["strobe_start"].value(),
                                             sctrl["strobe_end"].value(), weights))
        step.oscillators = oscillators
        step.strobe_sets = strobe_sets

        # Save per-step audio settings from the audio settings panel.
        carriers = []
        for cpanel in self.audio_settings_panel.carrier_panels:
            carrier = {
                "enabled": cpanel.enabled.isChecked(),
                "start_freq_left": cpanel.start_freq_left.value(),
                "end_freq_left": cpanel.end_freq_left.value(),
                "start_freq_right": cpanel.start_freq_right.value(),
                "end_freq_right": cpanel.end_freq_right.value(),
                "tone_mode": cpanel.tone_mode_combo.currentText(),
                "volume": cpanel.volume.value() / 100.0,
                "enable_rfm": cpanel.rfm_enable.isChecked(),
                "rfm_range": cpanel.rfm_range.value(),
                "rfm_speed": cpanel.rfm_speed.value()
            }
            carriers.append(carrier)
        step.audio_settings = {"carriers": carriers}
        self.step_list_panel.update_step_item(index, step.description)
        self.update_sequence_duration()
        QMessageBox.information(self, "Step Updated", f"Updated step '{step.description}'.")

    def handle_new_sequence(self):
        self.step_controller.steps.clear()
        self.step_list_panel.clear()
        self.step_config_panel.clear_step_fields()
        self.currentFile = None
        self.audio_settings = {}
        self.update_sequence_duration()
        self.handle_add_step()

    def handle_load_sequence(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Load Sequence", "", "Sequence Files (*.json);;All Files (*)")
        if not fname:
            return
        steps, audio_settings = self.file_controller.load_sequence(fname)
        if steps is None:
            return
        self.step_controller.steps = steps
        self.step_list_panel.clear()
        for step in steps:
            self.step_list_panel.add_step_item(step.description)
        self.audio_settings = audio_settings
        if self.step_controller.steps:
            self.step_list_panel.set_current_row(0)
            self.handle_step_selected(0)
        else:
            self.handle_add_step()
        self.currentFile = fname
        self.update_sequence_duration()
        QMessageBox.information(self, "Loaded", f"Sequence loaded: {fname}")

    def handle_save_sequence(self):
        if not self.currentFile:
            self.handle_save_sequence_as()
        else:
            self._save_to_file(self.currentFile)

    def handle_save_sequence_as(self):
        fname, _ = QFileDialog.getSaveFileName(self, "Save Sequence As", "", "Sequence Files (*.json);;All Files (*)")
        if not fname:
            return
        self.currentFile = fname
        self._save_to_file(fname)

    def _save_to_file(self, fname):
        panel = self.audio_settings_panel.global_audio_panel
        self.audio_settings["enabled"] = panel.enabled.isChecked()
        self.audio_settings["enable_pink_noise"] = panel.pink_noise_enable.isChecked()
        self.audio_settings["pink_noise_volume"] = panel.pink_noise_volume.value() / 100.0
        self.audio_settings["global_rfm_enable"] = panel.global_rfm_enable.isChecked()
        self.audio_settings["global_rfm_range"] = panel.global_rfm_range.value()
        self.audio_settings["global_rfm_speed"] = panel.global_rfm_speed.value()
        self.audio_settings["sample_rate"] = panel.sample_rate.value()
        carriers = []
        for cpanel in self.audio_settings_panel.carrier_panels:
            carrier = {
                "enabled": cpanel.enabled.isChecked(),
                "start_freq_left": cpanel.start_freq_left.value(),
                "end_freq_left": cpanel.end_freq_left.value(),
                "start_freq_right": cpanel.start_freq_right.value(),
                "end_freq_right": cpanel.end_freq_right.value(),
                "tone_mode": cpanel.tone_mode_combo.currentText(),
                "volume": cpanel.volume.value() / 100.0,
                "enable_rfm": cpanel.rfm_enable.isChecked(),
                "rfm_range": cpanel.rfm_range.value(),
                "rfm_speed": cpanel.rfm_speed.value()
            }
            carriers.append(carrier)
        self.audio_settings["carriers"] = carriers
        if self.file_controller.save_sequence(fname, self.step_controller.steps,
                                              self.audio_settings,
                                              panel):
            QMessageBox.information(self, "Saved", f"Sequence saved to {fname}")
        self.update_sequence_duration()

    def handle_delete_sequence_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Delete Sequence File", "", "Sequence Files (*.json);;All Files (*)")
        if not fname:
            return
        reply = QMessageBox.question(self, "Confirm Delete", f"Delete file?\n{fname}",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.file_controller.delete_sequence_file(fname)

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

