import json
from dataclasses import asdict
from pathlib import Path

from .preferences import Preferences

# Settings file path located in the user's home directory
SETTINGS_FILE = Path.home() / ".entrainment_prefs.json"


def load_settings() -> Preferences:
    """Load preferences from ``SETTINGS_FILE``.

    Returns a :class:`Preferences` instance populated with the values found in
    the JSON file. If the file does not exist or cannot be read, defaults are
    returned.
    """
    if SETTINGS_FILE.is_file():
        try:
            with open(SETTINGS_FILE, "r") as f:
                data = json.load(f)
            prefs = Preferences()
            for k, v in data.items():
                if hasattr(prefs, k):
                    setattr(prefs, k, v)
            return prefs
        except Exception as e:
            print(f"Failed to load preferences: {e}")
    return Preferences()


def save_settings(prefs: Preferences) -> None:
    """Save the given ``Preferences`` instance to ``SETTINGS_FILE``."""
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(asdict(prefs), f, indent=2)
    except Exception as e:
        print(f"Failed to save preferences: {e}")
