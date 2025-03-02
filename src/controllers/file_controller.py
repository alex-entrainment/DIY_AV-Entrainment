import json
import os
from PyQt5.QtWidgets import QMessageBox
from sequence_model import Step
from audio_generator import generate_audio_file_for_steps_offline_rfm

class FileController:
    def load_sequence(self, fname):
        try:
            with open(fname, "r") as f:
                data = json.load(f)
            steps = [Step.from_dict(s) for s in data.get("steps", [])]
            audio_settings = data.get("audio_settings", {})
            return steps, audio_settings
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to load file:\n{e}")
            return None, None

    def save_sequence(self, fname, steps, audio_settings, global_audio_panel):
        data = {
            "steps": [s.to_dict() for s in steps],
            "audio_settings": audio_settings
        }
        try:
            with open(fname, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Could not save:\n{e}")
            return False

        if audio_settings.get("enabled", False):
            base, _ = os.path.splitext(fname)
            audio_filename = base + ".wav"
            audio_settings["sample_rate"] = global_audio_panel.sample_rate.value()
            generate_audio_file_for_steps_offline_rfm(steps, audio_filename, audio_settings)
        return True

    def delete_sequence_file(self, fname):
        try:
            os.remove(fname)
            QMessageBox.information(None, "Deleted", f"{fname} removed.")
            return True
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Could not delete:\n{e}")
            return False
