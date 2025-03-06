#
# sequence_model.py
#
# Contains the core data classes for a 6-LED strobe sequence with optional audio entrainment.
# Now ENHANCED to handle advanced "visual patterns" in each Oscillator for complex
# brightness/phase modulation.
#

from enum import Enum

#
# 1) Waveform
#
class Waveform(Enum):
    OFF = 0
    SQUARE = 1
    SINE = 2
    # Additional waveforms can be added if needed (e.g., SAW, TRIANGLE).

#
# 1A) PatternMode
#
class PatternMode(Enum):
    NONE = 0
    SACRED_GEOMETRY = 1
    FRACTAL_ARC = 2
    PHI_SPIRAL = 3
    # Feel free to add more enumerations here.

#
# 2) Oscillator
#
class Oscillator:
    """
    Represents a single LED oscillator within a step.

    Attributes:
        start_freq (float):  The initial frequency (Hz) at the beginning of the step.
        end_freq (float):    The final frequency (Hz) at the end of the step.
        waveform (Waveform): OFF, SQUARE, or SINE.

        start_duty (float):  The initial duty cycle (%) for square wave usage.
        end_duty (float):    The final duty cycle (%) for square wave usage.

        enable_rfm (bool):   If True, applies a random frequency modulation offset.
        rfm_range (float):   Maximum ± range in Hz for the random walk.
        rfm_speed (float):   How fast the random offset changes (Hz/sec).

        phase_pattern (PatternMode): The kind of complex offset pattern to apply.
        brightness_pattern (PatternMode): The kind of brightness modulation pattern.
        pattern_strength (float): For each pattern mode, how strongly to apply it.
        pattern_freq (float): A frequency or speed factor for the pattern.
    """
    def __init__(
        self,
        start_freq: float,
        end_freq: float,
        waveform: Waveform,
        start_duty: float = 50.0,
        end_duty: float   = 50.0,
        enable_rfm: bool  = False,
        rfm_range: float  = 0.5,
        rfm_speed: float  = 0.2,
        phase_pattern = PatternMode.NONE,
        brightness_pattern = PatternMode.NONE,
        pattern_strength: float = 1.0,
        pattern_freq: float = 1.0
    ):
        # Clamp frequency and duty cycle values
        self.start_freq = max(0.1, min(200.0, start_freq))
        self.end_freq   = max(0.1, min(200.0, end_freq))
        self.waveform   = waveform

        self.start_duty = max(1,   min(99, start_duty))
        self.end_duty   = max(1,   min(99, end_duty))

        self.enable_rfm = enable_rfm
        self.rfm_range  = max(0.0, rfm_range)
        self.rfm_speed  = max(0.0, rfm_speed)

        # Advanced pattern fields
        self.phase_pattern = phase_pattern
        self.brightness_pattern = brightness_pattern
        self.pattern_strength = max(0.0, pattern_strength)
        self.pattern_freq = max(0.0, pattern_freq)

    def to_dict(self) -> dict:
        """
        Serialize fields to a dictionary (suitable for JSON).
        """
        return {
            "start_freq": self.start_freq,
            "end_freq":   self.end_freq,
            "waveform":   self.waveform.value,
            "start_duty": self.start_duty,
            "end_duty":   self.end_duty,
            "enable_rfm": self.enable_rfm,
            "rfm_range":  self.rfm_range,
            "rfm_speed":  self.rfm_speed,
            "phase_pattern": self.phase_pattern.value,
            "brightness_pattern": self.brightness_pattern.value,
            "pattern_strength": self.pattern_strength,
            "pattern_freq": self.pattern_freq
        }

    @classmethod
    def from_dict(cls, data: dict):
        """
        Create an Oscillator instance from a dictionary.
        """
        wf_value = data.get("waveform", 0)  # default to OFF if missing

        from_dict_phase_mode  = PatternMode(data.get("phase_pattern", 0))
        from_dict_bright_mode = PatternMode(data.get("brightness_pattern", 0))

        return cls(
            start_freq = data.get("start_freq", 12.0),
            end_freq   = data.get("end_freq",   12.0),
            waveform   = Waveform(wf_value),
            start_duty = data.get("start_duty", 50.0),
            end_duty   = data.get("end_duty",   50.0),
            enable_rfm = data.get("enable_rfm", False),
            rfm_range  = data.get("rfm_range",  0.5),
            rfm_speed  = data.get("rfm_speed",  0.2),
            phase_pattern      = from_dict_phase_mode,
            brightness_pattern = from_dict_bright_mode,
            pattern_strength   = data.get("pattern_strength", 1.0),
            pattern_freq       = data.get("pattern_freq", 1.0)
        )

#
# 3) StrobeSet
#
class StrobeSet:
    """
    Maps a set of LED channels to one or more oscillator outputs with weights.
    Each channel in 'channels' will share the same intensity, determined by
    (weighted sum of oscillator outputs) * (interpolated intensity).
    """
    def __init__(
        self,
        channels: list,
        start_intensity: float,
        end_intensity: float,
        oscillator_weights: list
    ):
        self.channels = channels
        self.start_intensity = max(0, min(100, start_intensity))
        self.end_intensity   = max(0, min(100, end_intensity))

        total = sum(oscillator_weights)
        if total > 0:
            self.oscillator_weights = [w / total for w in oscillator_weights]
        else:
            self.oscillator_weights = [0] * len(oscillator_weights)

    def to_dict(self) -> dict:
        return {
            "channels":           self.channels,
            "start_intensity":    self.start_intensity,
            "end_intensity":      self.end_intensity,
            "oscillator_weights": self.oscillator_weights
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            channels           = data.get("channels", []),
            start_intensity    = data.get("start_intensity", 0),
            end_intensity      = data.get("end_intensity", 0),
            oscillator_weights = data.get("oscillator_weights", [])
        )

#
# 4) Step
#
class Step:
    def __init__(self, duration: float, description: str, oscillators: list, strobe_sets: list, audio_settings: dict = {}):
        self.duration = max(1, min(5400, duration))
        self.description = description
        self.oscillators = oscillators
        self.strobe_sets = strobe_sets
        # New per-step audio settings (e.g. carrier parameters)
        self.audio_settings = audio_settings or {}

    def to_dict(self) -> dict:
        return {
            "duration":    self.duration,
            "description": self.description,
            "oscillators": [osc.to_dict() for osc in self.oscillators],
            "strobe_sets": [st.to_dict() for st in self.strobe_sets],
            "audio_settings": self.audio_settings
        }

    @classmethod
    def from_dict(cls, data: dict):
        duration    = data.get("duration", 30)
        description = data.get("description", "Untitled Step")
        osc_list    = data.get("oscillators", [])
        strobe_list = data.get("strobe_sets", [])
        audio_settings = data.get("audio_settings", {})
        oscillators = [Oscillator.from_dict(o) for o in osc_list]
        strobe_sets = [StrobeSet.from_dict(s) for s in strobe_list]
        return cls(duration, description, oscillators, strobe_sets, audio_settings)

#
# 5) AudioCarrier
#
class AudioCarrier:
    """
    Represents a single carrier frequency in the audio generation.
    Multiple carriers can be combined to create complex entrainment patterns.
    Now supports independent left and right channel frequencies for precise
    binaural beat control.

    Attributes:
        enabled (bool): if False, this carrier is not used in audio generation
        
        start_freq (float): base carrier frequency at the beginning (for backward compatibility)
        end_freq (float): base carrier frequency at the end (for backward compatibility)
        
        start_freq_left (float): left channel frequency at the beginning of the step
        end_freq_left (float): left channel frequency at the end of the step
        start_freq_right (float): right channel frequency at the beginning of the step
        end_freq_right (float): right channel frequency at the end of the step
        
        volume (float): volume of this carrier (0.0 to 1.0)
        enable_rfm (bool): apply random frequency modulation to this carrier
        rfm_range (float): maximum range of frequency modulation
        rfm_speed (float): speed of frequency modulation
    """
    def __init__(
        self,
        enabled: bool = True,
        start_freq: float = 200.0,
        end_freq: float = 200.0,
        start_freq_left: float = None,
        end_freq_left: float = None,
        start_freq_right: float = None,
        end_freq_right: float = None,
        volume: float = 1.0,
        enable_rfm: bool = False,
        rfm_range: float = 0.5,
        rfm_speed: float = 0.2
    ):
        self.enabled = enabled
        self.start_freq = max(20.0, min(1000.0, start_freq))
        self.end_freq = max(20.0, min(1000.0, end_freq))
        
        # Initialize channel-specific frequencies, using the base frequency if not specified
        self.start_freq_left = max(20.0, min(1000.0, start_freq_left if start_freq_left is not None else start_freq))
        self.end_freq_left = max(20.0, min(1000.0, end_freq_left if end_freq_left is not None else end_freq))
        self.start_freq_right = max(20.0, min(1000.0, start_freq_right if start_freq_right is not None else start_freq))
        self.end_freq_right = max(20.0, min(1000.0, end_freq_right if end_freq_right is not None else end_freq))
        
        self.volume = max(0.0, min(1.0, volume))
        self.enable_rfm = enable_rfm
        self.rfm_range = max(0.0, rfm_range)
        self.rfm_speed = max(0.0, rfm_speed)
    
    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "start_freq": self.start_freq,
            "end_freq": self.end_freq,
            "start_freq_left": self.start_freq_left,
            "end_freq_left": self.end_freq_left,
            "start_freq_right": self.start_freq_right,
            "end_freq_right": self.end_freq_right,
            "volume": self.volume,
            "enable_rfm": self.enable_rfm,
            "rfm_range": self.rfm_range,
            "rfm_speed": self.rfm_speed
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            enabled = data.get("enabled", True),
            start_freq = data.get("start_freq", 200.0),
            end_freq = data.get("end_freq", 200.0),
            start_freq_left = data.get("start_freq_left", None),
            end_freq_left = data.get("end_freq_left", None),
            start_freq_right = data.get("start_freq_right", None),
            end_freq_right = data.get("end_freq_right", None),
            volume = data.get("volume", 1.0),
            enable_rfm = data.get("enable_rfm", False),
            rfm_range = data.get("rfm_range", 0.5),
            rfm_speed = data.get("rfm_speed", 0.2)
        )

#
# 6) AudioSettings
#
class AudioSettings:
    """
    A container for global audio parameters for the entire sequence (not per step).

    Attributes:
        enabled (bool): if False, no audio is generated or played.
        beat_freq (float): difference or modulation frequency, e.g. 10.0 Hz.
        is_binaural (bool): if True, produce left/right freq offset (binaural).
        is_isochronic (bool): if True, produce pulses at the beat frequency (isochronic).
        enable_rfm (bool): random freq modulation for the audio track.
        rfm_range (float): ± range for random drift of the carrier freq, e.g. 0.5 Hz.
        rfm_speed (float): how quickly the random drift changes, e.g. 0.2 Hz/sec.
        carriers (list[AudioCarrier]): up to 3 audio carriers for complex tones.
        enable_pink_noise (bool): if True, add pink noise to the background
        pink_noise_volume (float): volume of the pink noise (0.0 to 1.0)
    """
    def __init__(
        self,
        enabled: bool = False,
        beat_freq: float = 10.0,
        is_binaural: bool = True,
        is_isochronic: bool = False,
        enable_rfm: bool = False,
        rfm_range: float = 0.5,
        rfm_speed: float = 0.2,
        carriers: list = None,
        enable_pink_noise: bool = False,
        pink_noise_volume: float = 0.1
    ):
        self.enabled = enabled
        self.beat_freq = max(0.1, min(100.0, beat_freq))
        self.is_binaural = is_binaural
        self.is_isochronic = is_isochronic
        self.enable_rfm = enable_rfm
        self.rfm_range = max(0.0, rfm_range)
        self.rfm_speed = max(0.0, rfm_speed)
        
        # Initialize with up to 3 carriers
        if carriers is None or len(carriers) == 0:
            self.carriers = [AudioCarrier()]
        else:
            self.carriers = carriers[:3]
        
        # Fill out to exactly 3 carriers
        while len(self.carriers) < 3:
            self.carriers.append(AudioCarrier(enabled=False))
        
        # Pink noise settings
        self.enable_pink_noise = enable_pink_noise
        self.pink_noise_volume = max(0.0, min(1.0, pink_noise_volume))

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "beat_freq": self.beat_freq,
            "is_binaural": self.is_binaural,
            "is_isochronic": self.is_isochronic,
            "enable_rfm": self.enable_rfm,
            "rfm_range": self.rfm_range,
            "rfm_speed": self.rfm_speed,
            "carriers": [c.to_dict() for c in self.carriers],
            "enable_pink_noise": self.enable_pink_noise,
            "pink_noise_volume": self.pink_noise_volume
        }

    @classmethod
    def from_dict(cls, data: dict):
        carriers_data = data.get("carriers", [])
        carriers = []

        # If legacy format with "carrier_freq", convert to new format
        if "carrier_freq" in data and len(carriers_data) == 0:
            legacy_carrier = AudioCarrier(
                enabled=True,
                start_freq=data.get("carrier_freq", 200.0),
                end_freq=data.get("carrier_freq", 200.0)
            )
            carriers.append(legacy_carrier)
        else:
            for carrier_data in carriers_data:
                carriers.append(AudioCarrier.from_dict(carrier_data))

        return cls(
            enabled = data.get("enabled", False),
            beat_freq = data.get("beat_freq", 10.0),
            is_binaural = data.get("is_binaural", True),
            is_isochronic = data.get("is_isochronic", False),
            enable_rfm = data.get("enable_rfm", False),
            rfm_range = data.get("rfm_range", 0.5),
            rfm_speed = data.get("rfm_speed", 0.2),
            carriers = carriers,
            enable_pink_noise = data.get("enable_pink_noise", False),
            pink_noise_volume = data.get("pink_noise_volume", 0.1)
        )

#
# 7) Sequence (Optional)
#
class Sequence:
    """
    Optional container for the entire sequence: steps + audio settings.
    This can simplify load/save if you prefer a single object to handle.
    """
    def __init__(self, steps=None, audio_settings=None):
        self.steps = steps if steps else []
        self.audio_settings = audio_settings if audio_settings else AudioSettings()

    def to_dict(self) -> dict:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "audio_settings": self.audio_settings.to_dict()
        }

    @classmethod
    def from_dict(cls, data: dict):
        steps_data = data.get("steps", [])
        steps = [Step.from_dict(s) for s in steps_data]

        as_data = data.get("audio_settings", {})
        audio_settings = AudioSettings.from_dict(as_data)

        return cls(steps, audio_settings)
