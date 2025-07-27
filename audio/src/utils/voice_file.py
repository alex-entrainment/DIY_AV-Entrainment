import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Any, Optional, List

VOICE_FILE_EXTENSION = ".voice"
VOICES_FILE_EXTENSION = ".voices"

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


def save_voice_preset_list(presets: List[VoicePreset], filepath: str) -> None:
    """Save a list of ``VoicePreset`` objects to ``filepath``."""
    path = Path(filepath)
    if path.suffix != VOICES_FILE_EXTENSION:
        path = path.with_suffix(VOICES_FILE_EXTENSION)
    data = [asdict(p) for p in presets]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_voice_preset_list(filepath: str) -> List[VoicePreset]:
    """Load a list of voice presets from ``filepath``."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Voice preset list not found: {filepath}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Voice preset list must be a JSON array")
    presets: List[VoicePreset] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        preset = VoicePreset()
        for k, v in item.items():
            if hasattr(preset, k):
                setattr(preset, k, v)
        presets.append(preset)
    return presets

__all__ = [
    "VoicePreset",
    "save_voice_preset",
    "load_voice_preset",
    "VOICE_FILE_EXTENSION",
    "VOICES_FILE_EXTENSION",
    "save_voice_preset_list",
    "load_voice_preset_list",
]
