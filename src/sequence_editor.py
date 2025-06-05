import sys

from PyQt5 import sip
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QApplication,
    QStatusBar,
    QAction,
    QFileDialog,
    QMessageBox,
)
from functools import partial
from ui import themes
from ui.step_list_panel import StepListPanel
from ui.step_config_panel import StepConfigPanel
# REMOVED: from ui.audio_settings_panel import AudioSettingsPanel
from controllers.step_controller import StepController
from controllers.file_controller import FileController
from sequence_model import Oscillator, StrobeSet, Waveform, PatternMode, Step # Added Step import for clarity
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
        self.setWindowTitle("6-LED Sequence Editor") # Updated title
        self.resize(1000, 700) # Adjusted default size might be needed

        self.simulator_window = None  # to hold the simulator instance

        # Controllers.
        self.step_controller = StepController()
        self.file_controller = FileController()
        self.currentFile = None
        # REMOVED: self.audio_settings = {} - No longer needed for UI interaction

        # Create central widget and layout.
        central = QWidget(self)
        main_layout = QHBoxLayout(central)
        self.setCentralWidget(central)

        # Create panels.
        self.step_list_panel = StepListPanel()
        self.step_config_panel = StepConfigPanel()
        # REMOVED: self.audio_settings_panel = AudioSettingsPanel()

        # Add panels to main layout.
        main_layout.addWidget(self.step_list_panel, 2)
        main_layout.addWidget(self.step_config_panel, 3)
        # REMOVED: main_layout.addWidget(self.audio_settings_panel, 3)

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

        theme_menu = menubar.addMenu("Theme")
        for name in themes.THEMES.keys():
            act = QAction(name, self)
            act.triggered.connect(partial(self.set_theme, name))
            theme_menu.addAction(act)

    def open_simulator(self):
        if self.simulator_window is None:
            self.simulator_window = SimulatorWindow(self)
        self.simulator_window.show()
        self.simulator_window.raise_()
        self.simulator_window.activateWindow()

    def set_theme(self, name):
        themes.apply_theme(QApplication.instance(), name)

    def update_sequence_duration(self):
        duration = self.step_controller.update_sequence_duration()
        self.status.showMessage(f"Sequence Duration: {duration}")

    def handle_add_step(self):
        mode = self.step_config_panel.mode_combo.currentText()
        # Pass the audio settings from the previous step if available, otherwise default
        previous_audio_settings = {}
        if self.step_controller.steps:
            previous_audio_settings = self.step_controller.steps[-1].audio_settings
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
            self.handle_add_step() # Add a default one back if list is empty
        else:
            # Select the previous item, or the first if the first was deleted
            new_index = max(0, min(index, len(self.step_controller.steps) - 1))
            self.step_list_panel.set_current_row(new_index)
            # Explicitly trigger selection change if index didn't change but content did
            if new_index == index and self.step_controller.steps:
                 self.handle_step_selected(new_index)

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
             # Handle case where list might be empty or index is invalid
            self.step_config_panel.clear_step_fields()
            # Disable config panel elements if no step is selected
            self.step_config_panel.setEnabled(False)
            return

        # Enable config panel if a valid step is selected
        self.step_config_panel.setEnabled(True)
        step = self.step_controller.steps[index]

        # Block signals temporarily to prevent unwanted rebuilds during population
        self.step_config_panel.mode_combo.blockSignals(True)
        self.step_config_panel.duration_spin.blockSignals(True)
        self.step_config_panel.description_edit.blockSignals(True)

        self.step_config_panel.duration_spin.setValue(step.duration)
        self.step_config_panel.description_edit.setText(step.description)

        # Determine mode based on loaded step data
        if len(step.oscillators) == 1 and len(step.strobe_sets) == 1:
             mode = "Combined"
        elif len(step.oscillators) == 2 and len(step.strobe_sets) == 2:
             mode = "Split"
        elif len(step.oscillators) == 6 and len(step.strobe_sets) == 6:
             mode = "Independent"
        else:
             # Fallback or handle potential inconsistency - default to Combined?
             print(f"Warning: Inconsistent oscillator/strobe counts for step {index}. Defaulting mode to Combined.")
             mode = "Combined" # Or choose another sensible default/error handling

        # Set the mode combo WITHOUT triggering its change handler yet
        current_mode_index = self.step_config_panel.mode_combo.findText(mode)
        if current_mode_index != -1:
            self.step_config_panel.mode_combo.setCurrentIndex(current_mode_index)

        # Now, ensure the controls match the selected mode
        # Check if controls need rebuilding *before* populating them
        # A simple check could be the number of expected vs existing controls
        num_expected_controls = {"Combined": 1, "Split": 2, "Independent": 6}[mode]
        if len(self.step_config_panel.osc_controls) != num_expected_controls:
             self.step_config_panel.build_oscillator_and_strobe_controls(mode)


        # Populate Oscillator Controls
        for i, osc in enumerate(step.oscillators):
             # Ensure we don't try to access controls that don't exist for the current mode
            if i < len(self.step_config_panel.osc_controls):
                ctrl = self.step_config_panel.osc_controls[i]
                try:
                    # Check if controls are valid before accessing
                    if ctrl and not sip.isdeleted(ctrl["wave_combo"]):
                        ctrl["wave_combo"].blockSignals(True)
                        ctrl["wave_combo"].setCurrentIndex(osc.waveform.value)
                        ctrl["wave_combo"].blockSignals(False)

                    if ctrl and not sip.isdeleted(ctrl["freq_start"]):
                        ctrl["freq_start"].blockSignals(True)
                        ctrl["freq_start"].setValue(osc.start_freq)
                        ctrl["freq_start"].blockSignals(False)

                    if ctrl and not sip.isdeleted(ctrl["freq_end"]):
                         ctrl["freq_end"].blockSignals(True)
                         ctrl["freq_end"].setValue(osc.end_freq)
                         ctrl["freq_end"].blockSignals(False)

                    if ctrl and not sip.isdeleted(ctrl["duty_start"]):
                        ctrl["duty_start"].blockSignals(True)
                        ctrl["duty_start"].setValue(int(osc.start_duty))
                        ctrl["duty_start"].blockSignals(False)

                    if ctrl and not sip.isdeleted(ctrl["duty_end"]):
                        ctrl["duty_end"].blockSignals(True)
                        ctrl["duty_end"].setValue(int(osc.end_duty))
                        ctrl["duty_end"].blockSignals(False)

                    if ctrl and not sip.isdeleted(ctrl["rfm_enable"]):
                        ctrl["rfm_enable"].blockSignals(True)
                        ctrl["rfm_enable"].setChecked(osc.enable_rfm)
                        ctrl["rfm_enable"].blockSignals(False)

                    if ctrl and not sip.isdeleted(ctrl["rfm_range"]):
                        ctrl["rfm_range"].blockSignals(True)
                        ctrl["rfm_range"].setValue(osc.rfm_range)
                        ctrl["rfm_range"].blockSignals(False)

                    if ctrl and not sip.isdeleted(ctrl["rfm_speed"]):
                         ctrl["rfm_speed"].blockSignals(True)
                         ctrl["rfm_speed"].setValue(osc.rfm_speed)
                         ctrl["rfm_speed"].blockSignals(False)

                    if ctrl and not sip.isdeleted(ctrl["phase_pattern_combo"]):
                         ctrl["phase_pattern_combo"].blockSignals(True)
                         ctrl["phase_pattern_combo"].setCurrentIndex(osc.phase_pattern.value)
                         ctrl["phase_pattern_combo"].blockSignals(False)

                    if ctrl and not sip.isdeleted(ctrl["brightness_pattern_combo"]):
                         ctrl["brightness_pattern_combo"].blockSignals(True)
                         ctrl["brightness_pattern_combo"].setCurrentIndex(osc.brightness_pattern.value)
                         ctrl["brightness_pattern_combo"].blockSignals(False)

                    if ctrl and not sip.isdeleted(ctrl["pattern_strength_spin"]):
                        ctrl["pattern_strength_spin"].blockSignals(True)
                        ctrl["pattern_strength_spin"].setValue(osc.pattern_strength)
                        ctrl["pattern_strength_spin"].blockSignals(False)

                    if ctrl and not sip.isdeleted(ctrl["pattern_freq_spin"]):
                         ctrl["pattern_freq_spin"].blockSignals(True)
                         ctrl["pattern_freq_spin"].setValue(osc.pattern_freq)
                         ctrl["pattern_freq_spin"].blockSignals(False)

                except Exception as e:
                    print(f"Error updating oscillator control {i} for step {index}: {e}")
            else:
                 print(f"Warning: Mismatch between loaded oscillators ({len(step.oscillators)}) and UI controls ({len(self.step_config_panel.osc_controls)}) for mode '{mode}'.")


        # Populate Strobe Controls
        for i, sset in enumerate(step.strobe_sets):
             # Ensure we don't try to access controls that don't exist for the current mode
            if i < len(self.step_config_panel.strobe_controls):
                ctrl = self.step_config_panel.strobe_controls[i]
                try:
                    if ctrl and not sip.isdeleted(ctrl["strobe_start"]):
                        ctrl["strobe_start"].blockSignals(True)
                        ctrl["strobe_start"].setValue(int(sset.start_intensity))
                        ctrl["strobe_start"].blockSignals(False)

                    if ctrl and not sip.isdeleted(ctrl["strobe_end"]):
                         ctrl["strobe_end"].blockSignals(True)
                         ctrl["strobe_end"].setValue(int(sset.end_intensity))
                         ctrl["strobe_end"].blockSignals(False)

                except Exception as e:
                    print(f"Error updating strobe control {i} for step {index}: {e}")
            else:
                 print(f"Warning: Mismatch between loaded strobe sets ({len(step.strobe_sets)}) and UI controls ({len(self.step_config_panel.strobe_controls)}) for mode '{mode}'.")


        # REMOVED: Block that populated the audio_settings_panel

        # Unblock signals
        self.step_config_panel.mode_combo.blockSignals(False)
        self.step_config_panel.duration_spin.blockSignals(False)
        self.step_config_panel.description_edit.blockSignals(False)


    def handle_apply_osc1_to_all(self):
        mode = self.step_config_panel.mode_combo.currentText()
        if mode != "Independent" or not self.step_config_panel.osc_controls:
            QMessageBox.warning(self, "Mode Error", "This function only applies in 'Independent' mode.")
            return

        if len(self.step_config_panel.osc_controls) < 1:
             return # Should not happen if mode is Independent, but safe check

        first = self.step_config_panel.osc_controls[0]
        for ctrl in self.step_config_panel.osc_controls[1:]:
             # Block signals during update
            ctrl["wave_combo"].blockSignals(True)
            ctrl["freq_start"].blockSignals(True)
            ctrl["freq_end"].blockSignals(True)
            ctrl["duty_start"].blockSignals(True)
            ctrl["duty_end"].blockSignals(True)
            ctrl["rfm_enable"].blockSignals(True)
            ctrl["rfm_range"].blockSignals(True)
            ctrl["rfm_speed"].blockSignals(True)
            ctrl["phase_pattern_combo"].blockSignals(True)
            ctrl["brightness_pattern_combo"].blockSignals(True)
            ctrl["pattern_strength_spin"].blockSignals(True)
            ctrl["pattern_freq_spin"].blockSignals(True)

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

            # Unblock signals
            ctrl["wave_combo"].blockSignals(False)
            ctrl["freq_start"].blockSignals(False)
            ctrl["freq_end"].blockSignals(False)
            ctrl["duty_start"].blockSignals(False)
            ctrl["duty_end"].blockSignals(False)
            ctrl["rfm_enable"].blockSignals(False)
            ctrl["rfm_range"].blockSignals(False)
            ctrl["rfm_speed"].blockSignals(False)
            ctrl["phase_pattern_combo"].blockSignals(False)
            ctrl["brightness_pattern_combo"].blockSignals(False)
            ctrl["pattern_strength_spin"].blockSignals(False)
            ctrl["pattern_freq_spin"].blockSignals(False)


    def handle_mode_changed(self, mode):
        # Rebuild controls ONLY if the mode actually changes the structure needed
        num_expected_controls = {"Combined": 1, "Split": 2, "Independent": 6}[mode]
        if len(self.step_config_panel.osc_controls) != num_expected_controls:
            print(f"Mode changed to {mode}, rebuilding controls.")
            self.step_config_panel.build_oscillator_and_strobe_controls(mode)
        else:
            print(f"Mode changed to {mode}, controls structure is the same, not rebuilding.")
        # After potentially rebuilding, ensure the correct step data is displayed if a step is selected
        current_index = self.step_list_panel.step_list.currentRow()
        if current_index >= 0:
            self.handle_step_selected(current_index) # Refresh the display with current step data


    def handle_submit_step(self):
        index = self.step_list_panel.step_list.currentRow()
        if index < 0 or index >= len(self.step_controller.steps):
            QMessageBox.warning(self, "No Step Selected", "Please select a step to update.")
            return

        step = self.step_controller.steps[index]
        # Keep existing audio settings when submitting LED changes
        existing_audio_settings = step.audio_settings

        # Update basic step info
        step.duration = self.step_config_panel.duration_spin.value()
        step.description = self.step_config_panel.description_edit.text()

        mode = self.step_config_panel.mode_combo.currentText()
        num_osc = {"Combined": 1, "Split": 2, "Independent": 6}[mode]

        oscillators = []
        strobe_sets = []

        # Check if the number of controls matches the expected number for the mode
        if len(self.step_config_panel.osc_controls) != num_osc or len(self.step_config_panel.strobe_controls) != num_osc:
             QMessageBox.critical(self, "Internal Error", f"UI control count mismatch for mode '{mode}'. Cannot submit.")
             # Attempt to fix by rebuilding and selecting again?
             self.handle_mode_changed(mode) # This will rebuild and call handle_step_selected
             return


        # Read Oscillator Data
        for i in range(num_osc):
            ctrl = self.step_config_panel.osc_controls[i]
            try:
                # Check controls validity
                if sip.isdeleted(ctrl["wave_combo"]): raise RuntimeError("Wave Combo deleted")
                if sip.isdeleted(ctrl["freq_start"]): raise RuntimeError("Freq Start deleted")
                # ... check others ...

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
            except Exception as e:
                 QMessageBox.critical(self, "Error Reading Oscillator", f"Failed to read oscillator {i+1} settings: {e}")
                 return # Stop submission if reading fails

        # Read Strobe Data and construct StrobeSet based on mode
        try:
            if mode == "Combined":
                sctrl = self.step_config_panel.strobe_controls[0]
                if sip.isdeleted(sctrl["strobe_start"]): raise RuntimeError("Strobe Start deleted")
                if sip.isdeleted(sctrl["strobe_end"]): raise RuntimeError("Strobe End deleted")
                strobe_sets = [StrobeSet(list(range(6)), sctrl["strobe_start"].value(),
                                         sctrl["strobe_end"].value(), [1.0])] # Single weight for single oscillator
            elif mode == "Split":
                for i in range(2):
                    sctrl = self.step_config_panel.strobe_controls[i]
                    if sip.isdeleted(sctrl["strobe_start"]): raise RuntimeError(f"Strobe Start {i} deleted")
                    if sip.isdeleted(sctrl["strobe_end"]): raise RuntimeError(f"Strobe End {i} deleted")
                    if i == 0:
                        strobe_sets.append(StrobeSet([0, 2, 4], sctrl["strobe_start"].value(),
                                                     sctrl["strobe_end"].value(), [1.0, 0.0])) # Weight for Osc 1
                    else:
                        strobe_sets.append(StrobeSet([1, 3, 5], sctrl["strobe_start"].value(),
                                                     sctrl["strobe_end"].value(), [0.0, 1.0])) # Weight for Osc 2
            else:  # Independent
                for i in range(6):
                    sctrl = self.step_config_panel.strobe_controls[i]
                    if sip.isdeleted(sctrl["strobe_start"]): raise RuntimeError(f"Strobe Start {i} deleted")
                    if sip.isdeleted(sctrl["strobe_end"]): raise RuntimeError(f"Strobe End {i} deleted")
                    weights = [0.0] * 6
                    weights[i] = 1.0 # Each strobe set linked only to its corresponding oscillator
                    strobe_sets.append(StrobeSet([i], sctrl["strobe_start"].value(),
                                                 sctrl["strobe_end"].value(), weights))
        except Exception as e:
             QMessageBox.critical(self, "Error Reading Strobe", f"Failed to read strobe settings: {e}")
             return # Stop submission

        # Update the step object
        step.oscillators = oscillators
        step.strobe_sets = strobe_sets
        step.audio_settings = existing_audio_settings # Put back the original audio settings

        # REMOVED: Block that read audio settings from the (now removed) audio panel

        # Update the list display and duration
        self.step_list_panel.update_step_item(index, step.description)
        self.update_sequence_duration()
        QMessageBox.information(self, "Step Updated", f"Updated step '{step.description}'.")


    def handle_new_sequence(self):
        reply = QMessageBox.question(self, "New Sequence",
                                     "Discard current unsaved changes and start a new sequence?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.step_controller.steps.clear()
            self.step_list_panel.clear()
            self.step_config_panel.clear_step_fields()
            self.currentFile = None
            # REMOVED: self.audio_settings = {}
            self.update_sequence_duration()
             # Set config panel to a default state and add one step
            self.step_config_panel.mode_combo.setCurrentIndex(0) # Set to Combined
            self.handle_mode_changed("Combined") # Ensure controls match
            self.handle_add_step()
            self.step_config_panel.setEnabled(True) # Ensure it's enabled


    def handle_load_sequence(self):
        # Optional: Check for unsaved changes before loading
        # reply = QMessageBox.question(...)
        # if reply == QMessageBox.No: return

        fname, _ = QFileDialog.getOpenFileName(self, "Load Sequence", "", "Sequence Files (*.json);;All Files (*)")
        if not fname:
            return

        steps, audio_settings_loaded = self.file_controller.load_sequence(fname) # Keep loading audio settings dict
        if steps is None: # Error occurred during loading
            # Error message already shown by FileController
            return

        self.step_controller.steps = steps
        # REMOVED: self.audio_settings = audio_settings_loaded - No longer needed at this level

        self._refresh_step_list() # Use the refresh helper

        if self.step_controller.steps:
            self.step_list_panel.set_current_row(0) # Select first step triggers handle_step_selected
        else:
             # If loaded file had no steps, add a default one
            self.step_config_panel.clear_step_fields()
            self.step_config_panel.mode_combo.setCurrentIndex(0)
            self.handle_mode_changed("Combined")
            self.handle_add_step()
            self.step_config_panel.setEnabled(True)


        self.currentFile = fname
        self.update_sequence_duration()
        QMessageBox.information(self, "Loaded", f"Sequence loaded from: {fname}")


    def handle_save_sequence(self):
        if not self.currentFile:
            self.handle_save_sequence_as()
        else:
            self._save_to_file(self.currentFile)

    def handle_save_sequence_as(self):
        fname, _ = QFileDialog.getSaveFileName(self, "Save Sequence As", "", "Sequence Files (*.json);;All Files (*)")
        if not fname:
            return
        # Ensure file has .json extension if user didn't provide one
        if not fname.lower().endswith('.json'):
            fname += '.json'
        self.currentFile = fname
        self._save_to_file(fname)

    def _save_to_file(self, fname):
        # NOTE: We no longer read global/carrier settings from a panel.
        # We just save the step data, including any audio_settings dict
        # that might have been loaded or copied within each step.
        # The FileController's save_sequence no longer generates audio.

        # Create a dictionary containing just the steps for saving.
        # The file format will implicitly contain the per-step audio_settings.
        # If global settings were needed *in the file*, they'd have to be managed differently now.
        # Assuming the important part is saving the steps as they are.
        sequence_data = {
            "steps": [s.to_dict() for s in self.step_controller.steps]
            # No top-level "audio_settings" needed unless specifically required by external tools.
            # If needed, it would have to be managed programmatically now.
        }

        if self.file_controller.save_sequence(fname, sequence_data): # Pass the dict directly
            QMessageBox.information(self, "Saved", f"Sequence saved to {fname}")
            # Update title or status bar if needed to reflect saved state
        # else: Error message handled by FileController

        self.update_sequence_duration()


    def handle_delete_sequence_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Delete Sequence File", "", "Sequence Files (*.json);;All Files (*)")
        if not fname:
            return
        reply = QMessageBox.question(self, "Confirm Delete", f"Are you sure you want to delete this file?\n{fname}\nThis action cannot be undone.",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            deleted = self.file_controller.delete_sequence_file(fname)
            # If the deleted file was the currently loaded one, reset the state
            if deleted and fname == self.currentFile:
                 self.handle_new_sequence() # Or just clear fields without confirmation
                 self.setWindowTitle("6-LED Sequence Editor") # Reset title


def main():
    app = QApplication(sys.argv)
    themes.apply_theme(app, "Dark")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
