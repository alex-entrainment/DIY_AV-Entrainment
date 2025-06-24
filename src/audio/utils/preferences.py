from dataclasses import dataclass, asdict, field
from pathlib import Path
import json

@dataclass
class Preferences:
    font_family: str = ""
    font_size: int = 10
    theme: str = "Dark"
    export_dir: str = ""
    sample_rate: int = 44100
    test_step_duration: float = 30.0
    track_metadata: bool = False
    # Peak amplitude for the final exported audio (0-1.0)
    target_output_amplitude: float = 0.25
    crossfade_curve: str = "linear"
    amplitude_display_mode: str = "absolute"  # or "dB"
    apply_target_amplitude: bool = True
    # Default voice configuration used when adding new voices
    default_voice: dict = field(default_factory=dict)

PREF_FILE = Path.home() / ".entrainment_prefs.json"

def load_preferences() -> Preferences:
    if PREF_FILE.is_file():
        try:
            with open(PREF_FILE, "r") as f:
                data = json.load(f)
            prefs = Preferences()
            for k, v in data.items():
                if hasattr(prefs, k):
                    setattr(prefs, k, v)
            return prefs
        except Exception as e:
            print(f"Failed to load preferences: {e}")
    return Preferences()


def save_preferences(prefs: Preferences):
    try:
        with open(PREF_FILE, "w") as f:
            json.dump(asdict(prefs), f, indent=2)
    except Exception as e:
        print(f"Failed to save preferences: {e}")
