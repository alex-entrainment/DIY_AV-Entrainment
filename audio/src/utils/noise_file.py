"""Helper for saving and loading noise generator parameters."""

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Any, Optional

# Default file extension for noise parameter files
NOISE_FILE_EXTENSION = ".noise"

@dataclass
class NoiseParams:
    """Representation of parameters used for noise generation."""
    duration_seconds: float = 60.0
    sample_rate: int = 44100
    noise_type: str = "pink"
    lfo_waveform: str = "sine"
    transition: bool = False
    # Non-transition mode uses ``lfo_freq`` and ``sweeps``
    lfo_freq: float = 1.0 / 12.0
    # Transition mode
    start_lfo_freq: float = 1.0 / 12.0
    end_lfo_freq: float = 1.0 / 12.0
    sweeps: List[Dict[str, Any]] = field(default_factory=list)
    start_lfo_phase_offset_deg: int = 0
    end_lfo_phase_offset_deg: int = 0
    start_intra_phase_offset_deg: int = 0
    end_intra_phase_offset_deg: int = 0
    initial_offset: float = 0.0
    transition_duration: Optional[float] = None
    input_audio_path: str = ""
    start_time: float = 0.0
    fade_in: float = 0.0
    fade_out: float = 0.0
    amp_envelope: List[Dict[str, Any]] = field(default_factory=list)


def save_noise_params(params: NoiseParams, filepath: str) -> None:
    """Save ``params`` to ``filepath`` using JSON inside a ``.noise`` file."""
    path = Path(filepath)
    if path.suffix != NOISE_FILE_EXTENSION:
        path = path.with_suffix(NOISE_FILE_EXTENSION)
    data = asdict(params)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_noise_params(filepath: str) -> NoiseParams:
    """Load noise parameters from ``filepath`` and return a :class:`NoiseParams`."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Noise parameter file not found: {filepath}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    params = NoiseParams()
    for k, v in data.items():
        if hasattr(params, k):
            setattr(params, k, v)
        elif k == "post_offset" and not hasattr(params, k):
            # Backwards compatibility: derive the new transition duration from
            # the legacy post offset value if possible.
            total = float(data.get("duration_seconds", params.duration_seconds))
            initial = float(data.get("initial_offset", params.initial_offset))
            params.transition_duration = max(0.0, total - initial - float(v))
    max_transition = max(0.0, params.duration_seconds - params.initial_offset)
    if params.transition_duration is None:
        params.transition_duration = max_transition
    else:
        params.transition_duration = max(0.0, min(params.transition_duration, max_transition))
    return params

__all__ = ["NoiseParams", "save_noise_params", "load_noise_params", "NOISE_FILE_EXTENSION"]

