import json
import os
from PyQt5.QtWidgets import QMessageBox
from common.sequence_model import Step, Sequence # Import Sequence if using it for structure

# REMOVED: from audio_generator import generate_audio_file_for_steps_offline_rfm

class FileController:
    def load_sequence(self, fname):
        """ Loads sequence data from a JSON file.
            Returns a tuple: (list_of_steps, audio_settings_dict)
            Returns (None, None) on error.
        """
        try:
            with open(fname, "r") as f:
                data = json.load(f)

            # Load steps using Step.from_dict for validation and defaults
            steps_data = data.get("steps", [])
            if not isinstance(steps_data, list):
                 raise ValueError("'steps' data is not a list")
            steps = [Step.from_dict(s) for s in steps_data]

            # Load audio_settings as a dictionary (kept for compatibility)
            audio_settings = data.get("audio_settings", {})
            if not isinstance(audio_settings, dict):
                 print(f"Warning: 'audio_settings' data in {fname} is not a dictionary. Ignoring.")
                 audio_settings = {} # Default to empty dict if format is wrong

            return steps, audio_settings

        except json.JSONDecodeError as e:
             QMessageBox.critical(None, "Load Error", f"Failed to parse JSON file:\n{fname}\n\nError: {e}")
             return None, None
        except Exception as e:
            QMessageBox.critical(None, "Load Error", f"Failed to load sequence from file:\n{fname}\n\nError: {e}")
            return None, None

    def save_sequence(self, fname, sequence_data):
        """ Saves the sequence data (dictionary containing steps) to a JSON file.
            Audio generation is removed.
            sequence_data: A dictionary expected to have a "steps" key.
        """
        # Basic validation of input data
        if not isinstance(sequence_data, dict) or "steps" not in sequence_data:
             QMessageBox.critical(None, "Save Error", "Invalid data format provided for saving.")
             return False

        try:
            with open(fname, "w") as f:
                json.dump(sequence_data, f, indent=2, ensure_ascii=False) # Use ensure_ascii=False for wider character support
            return True # Indicate success
        except Exception as e:
            QMessageBox.critical(None, "Save Error", f"Could not save sequence to file:\n{fname}\n\nError: {e}")
            return False

        # REMOVED: Block that checked audio_settings["enabled"] and called generate_audio_file...

    def delete_sequence_file(self, fname):
        """ Deletes the specified sequence file and its corresponding .wav (if exists) """
        deleted_json = False
        try:
            if os.path.exists(fname):
                os.remove(fname)
                deleted_json = True
            else:
                 QMessageBox.warning(None, "File Not Found", f"File does not exist:\n{fname}")
                 return False # Indicate file wasn't found to delete

        except Exception as e:
            QMessageBox.critical(None, "Delete Error", f"Could not delete sequence file:\n{fname}\n\nError: {e}")
            # Even if JSON deletion fails, try deleting WAV
            # return False # Return false if JSON delete failed

        # Also try to delete the corresponding .wav file if it exists
        base, _ = os.path.splitext(fname)
        audio_filename = base + ".wav"
        try:
             if os.path.exists(audio_filename):
                 os.remove(audio_filename)
                 print(f"Also removed associated audio file: {audio_filename}")
        except Exception as e:
             # Don't necessarily fail the whole operation if only WAV deletion fails,
             # but notify the user.
             QMessageBox.warning(None, "Delete Warning", f"Sequence file deleted, but could not delete associated audio file:\n{audio_filename}\n\nError: {e}")


        if deleted_json:
             QMessageBox.information(None, "Deleted", f"Sequence file removed:\n{fname}")
             return True
        else:
             # This path might be reached if JSON didn't exist but WAV potentially did
             # Or if JSON deletion failed but we continued. Return based on JSON deletion status primarily.
             return False
