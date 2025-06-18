import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Any, Optional

VOICE_FILE_EXTENSION = ".voice"

@dataclass
class VoicePreset:
    """Container for a voice's synthesis settings."""

    synth_function_name: str = ""
    is_transition: bool = False
    params: Dict[str, Any] = field(default_factory=dict)
    volume_envelope: Optional[Dict[str, Any]] = None
    description: str = ""


def save_voice_preset(preset: VoicePreset, filepath: str) -> None:
    """Save ``preset`` to ``filepath`` as JSON inside a ``.voice`` file."""
    path = Path(filepath)
    if path.suffix != VOICE_FILE_EXTENSION:
        path = path.with_suffix(VOICE_FILE_EXTENSION)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(preset), f, indent=2)


def load_voice_preset(filepath: str) -> VoicePreset:
    """Load a voice preset from ``filepath``."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Voice preset not found: {filepath}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    preset = VoicePreset()
    for k, v in data.items():
        if hasattr(preset, k):
            setattr(preset, k, v)
    return preset

__all__ = [
    "VoicePreset",
    "save_voice_preset",
    "load_voice_preset",
    "VOICE_FILE_EXTENSION",
]
