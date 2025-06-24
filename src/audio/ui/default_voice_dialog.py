"""Dialog for configuring default voice parameters."""

from PyQt5.QtWidgets import QDialog, QGroupBox

from voice_editor_dialog import VoiceEditorDialog
from utils.preferences import Preferences


class DefaultVoiceDialog(VoiceEditorDialog):
    """Reuse :class:`VoiceEditorDialog` to edit default voice settings."""

    def __init__(self, prefs: Preferences, parent=None):
        # Create a minimal dummy app with prefs and empty track data
        dummy_app = type("DummyApp", (), {"track_data": {"steps": []}, "prefs": prefs})
        super().__init__(parent=parent, app_ref=dummy_app, step_index=0, voice_index=None)
        self._prefs = prefs
        self.setWindowTitle("Configure Default Voice")
        self.save_button.setText("Save Defaults")

        # Hide reference viewer as it is not useful here
        for g in self.findChildren(QGroupBox):
            if g.title().startswith("Reference Voice"):
                g.hide()
                break

    def save_voice(self):
        """Collect data and store into ``prefs.default_voice``."""
        data = self._collect_data_for_main_app()
        self._prefs.default_voice = {
            "synth_function_name": data.get("synth_function_name", ""),
            "is_transition": data.get("is_transition", False),
            "params": data.get("params", {}),
            "volume_envelope": data.get("volume_envelope"),
        }
        self.accept()

    def get_default_voice(self):
        return self._prefs.default_voice

