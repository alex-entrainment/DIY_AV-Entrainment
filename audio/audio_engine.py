import numpy as np
import librosa
import soundfile as sf
import scipy.signal
from enum import Enum
from typing import List, Tuple, Optional, Union


# -----------------------------------------------------------
# Enums and Configuration Classes
# -----------------------------------------------------------

class SoundPath(Enum):
    """Enum for different types of sound paths"""
    CIRCULAR = "circular"
    SEMI_CIRCULAR = "semi-circular"
    OPEN = "open"
    DISCONTINUOUS = "discontinuous"
    LINEAR = "linear"


class BrainwaveState(Enum):
    """Enum for different brainwave states with their frequency ranges"""
    DELTA = "delta"  # Deep sleep (0.1-4 Hz)
    THETA = "theta"  # Meditation, drowsiness (4-8 Hz)
    ALPHA = "alpha"  # Relaxed, calm (8-13 Hz)
    BETA = "beta"    # Alert, focused (13-30 Hz)
    GAMMA = "gamma"  # Higher mental activity, perception (30-100 Hz)


# -----------------------------------------------------------
# Data Classes
# -----------------------------------------------------------

class Node:
    def __init__(self, duration, base_freq, beat_freq, volume_left, volume_right,
                 phase_deviation=0.05, left_phase_offset=0.0, right_phase_offset=0.0,
                 sound_path=SoundPath.CIRCULAR):
        """
        Initialize a node for audio generation.
        
        Args:
            duration: Length of this node in seconds
            base_freq: Carrier frequency in Hz
            beat_freq: Beat/spatial frequency in Hz (used differently by different voices)
            volume_left: Volume for left channel (0.0 to 1.0)
            volume_right: Volume for right channel (0.0 to 1.0)
            phase_deviation: Maximum phase deviation (reduced from original to prevent buzzing)
            left_phase_offset: Static phase offset for left channel
            right_phase_offset: Static phase offset for right channel
            sound_path: Type of spatial sound movement
        """
        self.duration = float(duration)
        self.base_freq = float(base_freq)
        self.beat_freq = float(beat_freq)
        self.volume_left = float(volume_left)
        self.volume_right = float(volume_right)
        # SAM-specific parameters (with reduced phase_deviation)
        self.phase_deviation = min(float(phase_deviation), 0.1)  # Cap at 0.1 to prevent buzzing
        self.left_phase_offset = float(left_phase_offset)
        self.right_phase_offset = float(right_phase_offset)
        self.sound_path = sound_path if isinstance(sound_path, SoundPath) else SoundPath.CIRCULAR
    
    def to_dict(self):
        """Convert node to dictionary for serialization."""
        return {
            "duration": self.duration,
            "base_freq": self.base_freq,
            "beat_freq": self.beat_freq,
            "volume_left": self.volume_left,
            "volume_right": self.volume_right,
            "phase_deviation": self.phase_deviation,
            "left_phase_offset": self.left_phase_offset,
            "right_phase_offset": self.right_phase_offset,
            "sound_path": self.sound_path.value if isinstance(self.sound_path, Enum) else self.sound_path
        }
    
    @staticmethod
    def from_dict(d):
        """Create a node from a dictionary."""
        # Convert string to enum if needed
        sound_path = d.get("sound_path", SoundPath.CIRCULAR)
        if isinstance(sound_path, str):
            try:
                sound_path = SoundPath[sound_path.upper()]
            except (KeyError, AttributeError):
                sound_path = SoundPath.CIRCULAR
        
        return Node(
            d["duration"],
            d["base_freq"],
            d["beat_freq"],
            d["volume_left"],
            d["volume_right"],
            d.get("phase_deviation", 0.05),  # Default to safer value
            d.get("left_phase_offset", 0.0),
            d.get("right_phase_offset", 0.0),
            sound_path
        )


class Voice:
    """
    Base voice class. Subclasses should override `generate_samples` with
    their specific audio synthesis logic. This version provides:

      1) Timeline building: total duration, sample count, and node start times.
      2) Automatic creation of interpolation arrays for all node parameters.
    """
    def __init__(self, nodes, sample_rate=44100):
        self.nodes = nodes
        self.sample_rate = sample_rate
        self._build_timeline()

    def _build_timeline(self):
        """Build the timeline and pre-compute arrays for interpolation."""
        self.start_times = [0.0]
        for i in range(len(self.nodes) - 1):
            self.start_times.append(self.start_times[i] + self.nodes[i].duration)
        self.total_duration = sum(n.duration for n in self.nodes)
        self.num_samples = int(self.total_duration * self.sample_rate)

        # Pre-build arrays for interpolation
        self.node_times = np.array(self.start_times, dtype=np.float64)
        self.base_freq_values = np.array([n.base_freq for n in self.nodes], dtype=np.float64)
        self.beat_freq_values = np.array([n.beat_freq for n in self.nodes], dtype=np.float64)
        self.vol_left_values = np.array([n.volume_left for n in self.nodes], dtype=np.float64)
        self.vol_right_values = np.array([n.volume_right for n in self.nodes], dtype=np.float64)
        
        # SAM-specific parameter arrays
        self.phase_deviation_values = np.array([n.phase_deviation for n in self.nodes], dtype=np.float64)
        self.left_phase_offset_values = np.array([n.left_phase_offset for n in self.nodes], dtype=np.float64)
        self.right_phase_offset_values = np.array([n.right_phase_offset for n in self.nodes], dtype=np.float64)
        
        # We can't interpolate enums, so we store the sound paths separately
        self.sound_paths = [n.sound_path for n in self.nodes]

    def _get_param_arrays(self):
        """
        Generate arrays for all parameters interpolated across the timeline.
        
        Returns a tuple of arrays for all parameters, interpolated across node boundaries.
        """
        if self.num_samples == 0:
            # Edge case: no samples
            empty_array = np.array([], dtype=np.float32)
            return (empty_array,) * 8

        # Create time array
        t_array = np.arange(self.num_samples, dtype=np.float64) / self.sample_rate

        # Use np.interp to do linear interpolation across node boundaries
        base_freq_array = np.interp(t_array, self.node_times, self.base_freq_values)
        beat_freq_array = np.interp(t_array, self.node_times, self.beat_freq_values)
        vol_left_array = np.interp(t_array, self.node_times, self.vol_left_values)
        vol_right_array = np.interp(t_array, self.node_times, self.vol_right_values)
        
        # SAM-specific parameter interpolation
        phase_dev_array = np.interp(t_array, self.node_times, self.phase_deviation_values)
        left_phase_offset_array = np.interp(t_array, self.node_times, self.left_phase_offset_values)
        right_phase_offset_array = np.interp(t_array, self.node_times, self.right_phase_offset_values)

        return (
            t_array,
            base_freq_array,
            beat_freq_array,
            vol_left_array,
            vol_right_array,
            phase_dev_array,
            left_phase_offset_array,
            right_phase_offset_array,
        )

    def generate_samples(self):
        """
        Generate audio samples. Subclasses must override this method.
        
        Returns a NumPy array of shape (num_samples, 2) containing stereo audio data.
        """
        raise NotImplementedError()


# -----------------------------------------------------------
# Spatial Angle Modulation (SAM) Voice - Improved Version
# -----------------------------------------------------------

def generate_spatial_modulator(t, spatial_freq, phase_deviation, sound_path, phase_offset=0.0):
    """
    Generate a phase modulation signal based on the specified sound path type.
    
    Args:
        t: Time array
        spatial_freq: Frequency of spatial modulation (Hz)
        phase_deviation: Maximum phase deviation (radians)
        sound_path: Type of perceived movement
        phase_offset: Static phase offset (radians)
        
    Returns:
        Phase modulation array
    """
    # Base angular frequency
    omega = 2.0 * np.pi * spatial_freq
    
    # Generate the modulation based on sound path type
    if sound_path == SoundPath.CIRCULAR:
        mod = phase_deviation * np.sin(omega * t)
    elif sound_path == SoundPath.SEMI_CIRCULAR:
        mod = phase_deviation * 0.5 * (np.sin(omega * t) + 1)
    elif sound_path == SoundPath.OPEN:
        # Smoother sawtooth for open path
        t_mod = np.mod(omega * t, 2 * np.pi) / np.pi
        mod = phase_deviation * (t_mod - 1)
    elif sound_path == SoundPath.DISCONTINUOUS:
        # Smoother square-like wave using tanh
        t_mod = np.mod(omega * t, 2 * np.pi) / (2 * np.pi)
        mod = phase_deviation * np.tanh(np.sin(2 * np.pi * t_mod) / 0.2)
    elif sound_path == SoundPath.LINEAR:
        # Triangular wave for linear path
        t_mod = np.mod(omega * t, 2 * np.pi) / (2 * np.pi)
        mod = phase_deviation * (2 * np.abs(2 * t_mod - 1) - 1)
    else:
        # Default to circular
        mod = phase_deviation * np.sin(omega * t)
    
    # Add phase offset
    return mod + phase_offset


def apply_fade(signal, fade_samples=1000):
    """
    Apply fade-in and fade-out to a signal to avoid clicks and pops.
    
    Args:
        signal: Input signal array
        fade_samples: Number of samples for fading
    
    Returns:
        Signal with fades applied
    """
    if len(signal) <= 2 * fade_samples:
        # Signal too short for fading, return as is
        return signal
    
    # Create a copy to avoid modifying the input
    result = signal.copy()
    
    # Create fade-in and fade-out windows (using raised cosine for smoothness)
    fade_in = 0.5 * (1 - np.cos(np.pi * np.arange(fade_samples) / fade_samples))
    fade_out = 0.5 * (1 + np.cos(np.pi * np.arange(fade_samples) / fade_samples))
    
    # Apply fades
    result[:fade_samples] *= fade_in
    result[-fade_samples:] *= fade_out
    
    return result


class ImprovedSAMVoice(Voice):
    """
    Improved Spatial Angle Modulation voice that produces clean, non-buzzy audio.
    This class generates binaural audio with spatial movement based on the specified
    sound path, using gentle phase modulation to maintain audio quality.
    """
    def __init__(self, nodes, sample_rate=44100, apply_fades=True):
        super().__init__(nodes, sample_rate)
        self.apply_fades = apply_fades
    
    def generate_samples(self):
        """Generate high-quality SAM binaural audio samples."""
        # Get parameter arrays
        params = self._get_param_arrays()
        t_array, base_freq_array, beat_freq_array, vol_left_array, vol_right_array, \
        phase_dev_array, left_phase_offset_array, right_phase_offset_array = params
        
        num_samples = t_array.size
        if num_samples == 0:
            return np.zeros((0, 2), dtype=np.float32)
        
        # Initialize output arrays
        audio = np.zeros((num_samples, 2), dtype=np.float32)
        
        # Process in segments based on node boundaries to handle sound path changes
        i_start = 0
        current_node_idx = 0
        
        while i_start < num_samples and current_node_idx < len(self.nodes):
            # Find the end of the current node
            node_end_time = self.node_times[current_node_idx] + self.nodes[current_node_idx].duration
            i_end = min(np.searchsorted(t_array, node_end_time, side='left'), num_samples)
            
            # Get the slice for this node
            slice_indices = slice(i_start, i_end)
            segment_samples = i_end - i_start
            
            if segment_samples <= 0:
                current_node_idx += 1
                i_start = i_end
                continue
            
            # Extract parameters for this segment
            t_slice = t_array[slice_indices]
            base_freq_slice = base_freq_array[slice_indices]
            beat_freq_slice = beat_freq_array[slice_indices]  # Used as spatial frequency
            phase_dev_slice = phase_dev_array[slice_indices]
            left_offset_slice = left_phase_offset_array[slice_indices]
            right_offset_slice = right_phase_offset_array[slice_indices]
            vol_left_slice = vol_left_array[slice_indices]
            vol_right_slice = vol_right_array[slice_indices]
            
            # Use the sound path for this node
            sound_path = self.sound_paths[current_node_idx]
            
            # Generate phase modulations with opposite phases for left/right
            # For left channel, use the specified sound path
            left_mod = generate_spatial_modulator(
                t_slice,
                beat_freq_slice[0],  # Use first value for consistency within segment
                phase_dev_slice[0],  # Use first value for consistency
                sound_path,
                left_offset_slice[0]
            )
            
            # For right channel, use the same parameters but with π phase shift when appropriate
            if sound_path in [SoundPath.CIRCULAR, SoundPath.SEMI_CIRCULAR]:
                # For circular patterns, we want opposite phase
                right_mod = generate_spatial_modulator(
                    t_slice,
                    beat_freq_slice[0],
                    phase_dev_slice[0],
                    sound_path,
                    right_offset_slice[0] + np.pi  # Add π for opposite phase
                )
            else:
                # For other patterns, use complementary modulation
                right_mod = generate_spatial_modulator(
                    t_slice,
                    beat_freq_slice[0],
                    phase_dev_slice[0],
                    sound_path,
                    right_offset_slice[0]
                )
                # Invert the modulation for complementary effect
                right_mod = -left_mod + 2 * right_offset_slice[0]
            
            # Generate carrier waves with phase modulation
            carrier_phase = 2.0 * np.pi * base_freq_slice * t_slice
            left_wave = np.sin(carrier_phase + left_mod) * vol_left_slice
            right_wave = np.sin(carrier_phase + right_mod) * vol_right_slice
            
            # Apply fades to segment if needed to prevent clicks at node boundaries
            if self.apply_fades and segment_samples > 2000:
                fade_len = min(1000, segment_samples // 10)
                left_wave = apply_fade(left_wave, fade_len)
                right_wave = apply_fade(right_wave, fade_len)
            
            # Copy to output array
            audio[slice_indices, 0] = left_wave
            audio[slice_indices, 1] = right_wave
            
            # Move to next segment
            current_node_idx += 1
            i_start = i_end
        
        return audio.astype(np.float32)


# -----------------------------------------------------------
# Binaural Beats Voice
# -----------------------------------------------------------

class BinauralBeatVoice(Voice):
    """
    Classic binaural beat generator.
    Produces a slightly different frequency in each ear to create a beat frequency.
    """
    def generate_samples(self):
        # Get parameter arrays (excluding SAM-specific ones)
        t_array, base_freq_array, beat_freq_array, vol_left_array, vol_right_array, *_ = self._get_param_arrays()
        num_samples = t_array.size
        if num_samples == 0:
            return np.zeros((0, 2), dtype=np.float32)

        # Calculate frequencies for left and right channels
        # left_freq = base - (beat/2), right_freq = base + (beat/2)
        left_freq_array = base_freq_array - beat_freq_array / 2.0
        right_freq_array = base_freq_array + beat_freq_array / 2.0

        # Calculate phases
        phase_left = 2.0 * np.pi * left_freq_array * t_array
        phase_right = 2.0 * np.pi * right_freq_array * t_array

        # Generate sine waves
        s_left = np.sin(phase_left) * vol_left_array
        s_right = np.sin(phase_right) * vol_right_array

        # Combine into stereo output
        audio = np.column_stack((s_left, s_right)).astype(np.float32)
        return audio


# -----------------------------------------------------------
# Multi-Source SAM Voice
# -----------------------------------------------------------

class MultiSAMVoice(ImprovedSAMVoice):
    """
    Enhanced SAM voice that generates multiple sound sources with different
    frequency relationships for a richer auditory experience.
    """
    def __init__(self, nodes, sample_rate=44100, secondary_freq_ratio=1.5, 
                 secondary_spatial_ratio=0.7, secondary_volume=0.5, apply_fades=True):
        super().__init__(nodes, sample_rate, apply_fades)
        self.secondary_freq_ratio = secondary_freq_ratio
        self.secondary_spatial_ratio = secondary_spatial_ratio
        self.secondary_volume = secondary_volume
    
    def generate_samples(self):
        # Get parameter arrays
        params = self._get_param_arrays()
        t_array, base_freq_array, beat_freq_array, vol_left_array, vol_right_array, \
        phase_dev_array, left_phase_offset_array, right_phase_offset_array = params
        
        num_samples = t_array.size
        if num_samples == 0:
            return np.zeros((0, 2), dtype=np.float32)
        
        # Create secondary source parameters
        secondary_base_freq_array = base_freq_array * self.secondary_freq_ratio
        secondary_spatial_freq_array = beat_freq_array * self.secondary_spatial_ratio
        secondary_vol_left_array = vol_left_array * self.secondary_volume
        secondary_vol_right_array = vol_right_array * self.secondary_volume
        
        # Initialize output arrays
        audio = np.zeros((num_samples, 2), dtype=np.float32)
        
        # Process in segments based on node boundaries
        i_start = 0
        current_node_idx = 0
        
        while i_start < num_samples and current_node_idx < len(self.nodes):
            # Find the end of the current node
            node_end_time = self.node_times[current_node_idx] + self.nodes[current_node_idx].duration
            i_end = min(np.searchsorted(t_array, node_end_time, side='left'), num_samples)
            
            # Get the slice for this node
            slice_indices = slice(i_start, i_end)
            segment_samples = i_end - i_start
            
            if segment_samples <= 0:
                current_node_idx += 1
                i_start = i_end
                continue
            
            # Extract parameters for this segment
            t_slice = t_array[slice_indices]
            
            # Primary source parameters
            p_base_freq = base_freq_array[slice_indices]
            p_spatial_freq = beat_freq_array[slice_indices]
            p_phase_dev = phase_dev_array[slice_indices][0]
            p_left_offset = left_phase_offset_array[slice_indices][0]
            p_right_offset = right_phase_offset_array[slice_indices][0]
            p_vol_left = vol_left_array[slice_indices]
            p_vol_right = vol_right_array[slice_indices]
            
            # Secondary source parameters
            s_base_freq = secondary_base_freq_array[slice_indices]
            s_spatial_freq = secondary_spatial_freq_array[slice_indices]
            s_phase_dev = p_phase_dev * 0.8  # Slightly reduced deviation
            s_vol_left = secondary_vol_left_array[slice_indices]
            s_vol_right = secondary_vol_right_array[slice_indices]
            
            # Primary sound path
            primary_path = self.sound_paths[current_node_idx]
            
            # Secondary sound path (use a different one for variation)
            if primary_path == SoundPath.CIRCULAR:
                secondary_path = SoundPath.OPEN
            elif primary_path == SoundPath.OPEN:
                secondary_path = SoundPath.CIRCULAR
            else:
                secondary_path = SoundPath.SEMI_CIRCULAR
            
            # Generate primary source modulations
            p_left_mod = generate_spatial_modulator(
                t_slice, 
                p_spatial_freq[0],
                p_phase_dev,
                primary_path,
                p_left_offset
            )
            
            if primary_path in [SoundPath.CIRCULAR, SoundPath.SEMI_CIRCULAR]:
                p_right_mod = generate_spatial_modulator(
                    t_slice,
                    p_spatial_freq[0],
                    p_phase_dev,
                    primary_path,
                    p_right_offset + np.pi
                )
            else:
                p_right_mod = -p_left_mod + 2 * p_right_offset
            
            # Generate secondary source modulations with complementary movement
            s_left_mod = generate_spatial_modulator(
                t_slice, 
                s_spatial_freq[0],
                s_phase_dev,
                secondary_path,
                p_right_offset  # Swap offsets for more spatial richness
            )
            
            if secondary_path in [SoundPath.CIRCULAR, SoundPath.SEMI_CIRCULAR]:
                s_right_mod = generate_spatial_modulator(
                    t_slice,
                    s_spatial_freq[0],
                    s_phase_dev,
                    secondary_path,
                    p_left_offset + np.pi
                )
            else:
                s_right_mod = -s_left_mod + 2 * p_left_offset
            
            # Generate carrier waves
            p_carrier_phase = 2.0 * np.pi * p_base_freq * t_slice
            s_carrier_phase = 2.0 * np.pi * s_base_freq * t_slice
            
            # Primary waves
            p_left_wave = np.sin(p_carrier_phase + p_left_mod) * p_vol_left
            p_right_wave = np.sin(p_carrier_phase + p_right_mod) * p_vol_right
            
            # Secondary waves
            s_left_wave = np.sin(s_carrier_phase + s_left_mod) * s_vol_left
            s_right_wave = np.sin(s_carrier_phase + s_right_mod) * s_vol_right
            
            # Mix primary and secondary
            left_mix = p_left_wave + s_left_wave
            right_mix = p_right_wave + s_right_wave
            
            # Apply fades to prevent clicks at node boundaries
            if self.apply_fades and segment_samples > 2000:
                fade_len = min(1000, segment_samples // 10)
                left_mix = apply_fade(left_mix, fade_len)
                right_mix = apply_fade(right_mix, fade_len)
            
            # Copy to output array
            audio[slice_indices, 0] = left_mix
            audio[slice_indices, 1] = right_mix
            
            # Move to next segment
            current_node_idx += 1
            i_start = i_end
        
        # Normalize if needed to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio /= max_val
        
        return audio.astype(np.float32)


# -----------------------------------------------------------
# Isochronic Pulse Generator
# -----------------------------------------------------------

def trapezoid_envelope_vectorized(t_in_cycle, cycle_len, ramp_percent, gap_percent):
    """
    Generate a trapezoidal envelope for isochronic tones.
    
    Args:
        t_in_cycle: Time within the current cycle (0 to cycle_len)
        cycle_len: Total cycle length in seconds
        ramp_percent: Percentage of audible portion used for ramps
        gap_percent: Percentage of cycle that is silent
    
    Returns:
        Envelope values in range [0, 1]
    """
    # Initialize output array
    env = np.zeros_like(t_in_cycle, dtype=np.float64)
    
    # Skip invalid cycles
    valid_mask = cycle_len > 0
    if not np.any(valid_mask):
        return env
    
    # Calculate segment durations
    audible_len = (1.0 - gap_percent) * cycle_len
    ramp_total = audible_len * ramp_percent * 2.0
    stable_len = audible_len - ramp_total
    
    # Handle edge cases
    stable_len_clamped = np.maximum(stable_len, 0.0)
    ramp_total_clamped = audible_len - stable_len_clamped
    ramp_up_len_clamped = ramp_total_clamped / 2.0
    stable_end = ramp_up_len_clamped + stable_len_clamped
    
    # Create masks for different envelope regions
    in_gap_mask = (t_in_cycle >= audible_len)
    ramp_up_mask = (t_in_cycle < ramp_up_len_clamped) & (~in_gap_mask) & valid_mask
    ramp_down_mask = (t_in_cycle >= stable_end) & (t_in_cycle < audible_len) & valid_mask
    stable_mask = (t_in_cycle >= ramp_up_len_clamped) & (t_in_cycle < stable_end) & (~in_gap_mask) & valid_mask
    
    # Calculate envelope values for each region
    env[ramp_up_mask] = t_in_cycle[ramp_up_mask] / ramp_up_len_clamped[ramp_up_mask]
    env[stable_mask] = 1.0
    
    # Ramp down calculation
    if np.any(ramp_down_mask):
        time_into_down = (t_in_cycle[ramp_down_mask] - stable_end[ramp_down_mask])
        down_len = (audible_len[ramp_down_mask] - stable_end[ramp_down_mask])
        env[ramp_down_mask] = 1.0 - (time_into_down / down_len)
    
    return env


class IsochronicVoice(Voice):
    """
    Isochronic tone generator with trapezoidal envelope.
    Creates pulses at the beat frequency with smooth ramps to reduce clicks.
    """
    def __init__(self, nodes, sample_rate=44100, ramp_percent=0.2, 
                 gap_percent=0.15, amplitude=1.0):
        super().__init__(nodes, sample_rate)
        self.ramp_percent = float(ramp_percent)
        self.gap_percent = float(gap_percent)
        self.amplitude = float(amplitude)
    
    def generate_samples(self):
        # Get parameter arrays
        t_array, base_freq_array, beat_freq_array, vol_left_array, vol_right_array, *_ = self._get_param_arrays()
        num_samples = t_array.size
        if num_samples == 0:
            return np.zeros((0, 2), dtype=np.float32)
        
        # Calculate cycle lengths from beat frequency
        cycle_len_array = np.zeros_like(beat_freq_array, dtype=np.float64)
        valid_beat_mask = (beat_freq_array > 0)
        cycle_len_array[valid_beat_mask] = 1.0 / beat_freq_array[valid_beat_mask]
        
        # Calculate time within each cycle
        t_in_cycle = np.mod(t_array, cycle_len_array, where=valid_beat_mask, out=np.zeros_like(t_array))
        
        # Generate envelope
        env = trapezoid_envelope_vectorized(
            t_in_cycle, 
            cycle_len_array,
            self.ramp_percent, 
            self.gap_percent
        )
        
        # Apply envelope to carrier wave
        carrier = np.sin(2.0 * np.pi * base_freq_array * t_array) * env * self.amplitude
        
        # Generate stereo output
        left = carrier * vol_left_array
        right = carrier * vol_right_array
        audio = np.column_stack((left, right)).astype(np.float32)
        
        return audio


class AltIsochronicVoice(Voice):
    """
    Alternating isochronic tone generator that switches between left and right channels
    for each complete cycle. Creates a perceived movement between ears.
    """
    def __init__(self, nodes, sample_rate=44100, ramp_percent=0.2, 
                 gap_percent=0.15, amplitude=1.0):
        super().__init__(nodes, sample_rate)
        self.ramp_percent = float(ramp_percent)
        self.gap_percent = float(gap_percent)
        self.amplitude = float(amplitude)
    
    def generate_samples(self):
        # Get parameter arrays
        t_array, base_freq_array, beat_freq_array, vol_left_array, vol_right_array, *_ = self._get_param_arrays()
        num_samples = t_array.size
        if num_samples == 0:
            return np.zeros((0, 2), dtype=np.float32)
        
        # Initialize output array
        audio = np.zeros((num_samples, 2), dtype=np.float32)
        
        # Process each beat cycle separately to handle alternation
        current_left = True
        i_start = 0
        
        while i_start < num_samples:
            # Get beat frequency at this point
            beat = beat_freq_array[i_start]
            if beat <= 0:
                i_start += 1
                continue
            
            # Calculate cycle duration and samples
            cycle_duration = 1.0 / beat
            t0 = t_array[i_start]
            t_end = t0 + cycle_duration
            
            # Find end index for this cycle
            i_end = np.searchsorted(t_array, t_end, side='left')
            i_end = min(i_end, num_samples)
            
            # Get slice for this cycle
            slice_indices = slice(i_start, i_end)
            sub_t = t_array[slice_indices]
            sub_base = base_freq_array[slice_indices]
            sub_vol_left = vol_left_array[slice_indices]
            sub_vol_right = vol_right_array[slice_indices]
            
            # Time within this cycle
            sub_t_in_cycle = sub_t - t0
            
            # Generate envelope
            env = trapezoid_envelope_vectorized(
                sub_t_in_cycle,
                np.full_like(sub_t_in_cycle, cycle_duration),
                self.ramp_percent,
                self.gap_percent
            )
            
            # Apply envelope to carrier
            carrier = np.sin(2.0 * np.pi * sub_base * sub_t) * env * self.amplitude
            
            # Route to appropriate channel
            if current_left:
                audio[slice_indices, 0] = carrier * sub_vol_left
                audio[slice_indices, 1] = 0.0
            else:
                audio[slice_indices, 0] = 0.0
                audio[slice_indices, 1] = carrier * sub_vol_right
            
            # Toggle channel for next cycle
            current_left = not current_left
            i_start = i_end
        
        return audio


class AltIsochronic2Voice(Voice):
    """
    Enhanced alternating isochronic tone generator that puts the first half-cycle
    in the left ear and the second half-cycle in the right ear. Creates a
    more rapid alternation for enhanced entrainment.
    """
    def __init__(self, nodes, sample_rate=44100, ramp_percent=0.2, 
                 gap_percent=0.15, amplitude=1.0):
        super().__init__(nodes, sample_rate)
        self.ramp_percent = float(ramp_percent)
        self.gap_percent = float(gap_percent)
        self.amplitude = float(amplitude)
    
    def generate_samples(self):
        # Get parameter arrays
        t_array, base_freq_array, beat_freq_array, vol_left_array, vol_right_array, *_ = self._get_param_arrays()
        num_samples = t_array.size
        if num_samples == 0:
            return np.zeros((0, 2), dtype=np.float32)
        
        # Initialize output array
        audio = np.zeros((num_samples, 2), dtype=np.float32)
        
        # Process each beat cycle
        i_start = 0
        
        while i_start < num_samples:
            # Get beat frequency at this point
            beat = beat_freq_array[i_start]
            if beat <= 0:
                i_start += 1
                continue
            
            # Calculate cycle duration and end time
            cycle_duration = 1.0 / beat
            t0 = t_array[i_start]
            t_end = t0 + cycle_duration
            
            # Find end index for this cycle
            i_end = np.searchsorted(t_array, t_end, side='left')
            i_end = min(i_end, num_samples)
            
            # Get slice for this cycle
            slice_indices = slice(i_start, i_end)
            sub_t = t_array[slice_indices]
            sub_base = base_freq_array[slice_indices]
            sub_vol_left = vol_left_array[slice_indices]
            sub_vol_right = vol_right_array[slice_indices]
            
            # Calculate half-cycle duration
            half_duration = cycle_duration / 2.0
            
            # Time within this cycle
            sub_t_in_cycle = sub_t - t0
            
            # Create masks for left and right halves
            left_mask = (sub_t_in_cycle < half_duration)
            right_mask = ~left_mask
            
            # Process left half
            if np.any(left_mask):
                t_in_left = sub_t_in_cycle[left_mask]
                env_left = trapezoid_envelope_vectorized(
                    t_in_left,
                    np.full_like(t_in_left, half_duration),
                    self.ramp_percent,
                    self.gap_percent
                )
                carrier_left = np.sin(2.0 * np.pi * sub_base[left_mask] * sub_t[left_mask]) * env_left * self.amplitude
                audio[slice_indices][left_mask, 0] = carrier_left * sub_vol_left[left_mask]
            
            # Process right half
            if np.any(right_mask):
                t_in_right = sub_t_in_cycle[right_mask] - half_duration
                env_right = trapezoid_envelope_vectorized(
                    t_in_right,
                    np.full_like(t_in_right, half_duration),
                    self.ramp_percent,
                    self.gap_percent
                )
                carrier_right = np.sin(2.0 * np.pi * sub_base[right_mask] * sub_t[right_mask]) * env_right * self.amplitude
                audio[slice_indices][right_mask, 1] = carrier_right * sub_vol_right[right_mask]
            
            # Move to next cycle
            i_start = i_end
        
        return audio


# -----------------------------------------------------------
# Pink Noise Voice
# -----------------------------------------------------------

class PinkNoiseVoice(Voice):
    """
    Pink noise generator with 1/f spectral character.
    Creates a soothing background noise that can mask distractions.
    """
    def generate_samples(self):
        # Get parameter arrays (we only need t_array and volumes)
        t_array, _, _, vol_left_array, vol_right_array, *_ = self._get_param_arrays()
        num_samples = t_array.size
        if num_samples == 0:
            return np.zeros((0, 2), dtype=np.float32)
        
        # Generate white noise
        white = np.random.normal(0, 1, num_samples).astype(np.float32)
        
        # Apply simple 1-pole filter to approximate pink noise
        alpha = 0.98  # Filter coefficient (controls steepness of 1/f falloff)
        pink = np.zeros(num_samples, dtype=np.float32)
        
        # Initial condition
        pink[0] = white[0]
        
        # IIR filter loop - not easily vectorizable
        for i in range(1, num_samples):
            pink[i] = alpha * pink[i-1] + (1.0 - alpha) * white[i]
        
        # Scale down for comfortable listening level
        scale = 0.1
        
        # Apply volumes and create stereo output
        audio = np.zeros((num_samples, 2), dtype=np.float32)
        audio[:, 0] = pink * vol_left_array * scale
        audio[:, 1] = pink * vol_right_array * scale
        
        return audio


# -----------------------------------------------------------
# External Audio Voice
# -----------------------------------------------------------

class ExternalAudioVoice(Voice):
    """
    Voice that loads external audio from a file and applies volume envelopes.
    Useful for adding background sounds or guided meditation vocals.
    """
    def __init__(self, nodes, file_path, sample_rate=44100):
        super().__init__(nodes, sample_rate)
        self.file_path = file_path
        
        try:
            # Load the audio file
            data, sr = librosa.load(self.file_path, sr=sample_rate, mono=False)
            
            # Handle mono input (convert to stereo)
            if data.ndim == 1:
                # Convert mono to stereo by duplicating the channel
                data = np.vstack((data, data))
            
            self.ext_audio = data
            self.ext_length = self.ext_audio.shape[1]
        except Exception as e:
            print(f"Error loading external audio: {e}")
            # Create placeholder audio in case of error
            self.ext_audio = np.zeros((2, 1))
            self.ext_length = 1
    
    def generate_samples(self):
        # Get parameter arrays (we only need t_array and volumes)
        t_array, _, _, vol_left_array, vol_right_array, *_ = self._get_param_arrays()
        num_samples = t_array.size
        if num_samples == 0 or self.ext_length <= 1:
            return np.zeros((0, 2), dtype=np.float32)
        
        # Initialize output array
        audio = np.zeros((num_samples, 2), dtype=np.float32)
        
        # Create array of indices into the external audio
        idx_array = np.arange(num_samples)
        
        # Clamp to the external audio length (loop if needed)
        idx_array = np.mod(idx_array, self.ext_length)
        
        # Extract samples from external audio
        left_samples = self.ext_audio[0][idx_array]
        right_samples = self.ext_audio[1][idx_array]
        
        # Apply volume envelopes
        audio[:, 0] = left_samples * vol_left_array
        audio[:, 1] = right_samples * vol_right_array
        
        return audio


# -----------------------------------------------------------
# Preset Generators
# -----------------------------------------------------------

def get_preset_nodes_for_state(state: BrainwaveState, duration: float = 300.0) -> List[Node]:
    """
    Create nodes for a specific brainwave state.
    
    Args:
        state: Target brainwave state
        duration: Total duration in seconds
        
    Returns:
        List of Node objects configured for the requested state
    """
    # Define presets for each brainwave state
    presets = {
        BrainwaveState.DELTA: {
            'base_freq': 100.0,  # Lower carrier for deeper states
            'beat_freq': 2.0,    # 0.1-4 Hz beat range, using 2Hz for SAM spatial freq
            'phase_deviation': 0.05,
            'sound_path': SoundPath.CIRCULAR,
            'volume': 0.8
        },
        BrainwaveState.THETA: {
            'base_freq': 200.0,
            'beat_freq': 5.0,    # 4-8 Hz beat range, using 5Hz for SAM spatial freq
            'phase_deviation': 0.05,
            'sound_path': SoundPath.CIRCULAR,
            'volume': 0.8
        },
        BrainwaveState.ALPHA: {
            'base_freq': 440.0,  # Standard A note
            'beat_freq': 10.0,   # 8-13 Hz beat range, using 10Hz for SAM spatial freq
            'phase_deviation': 0.05,
            'sound_path': SoundPath.CIRCULAR,
            'volume': 0.8
        },
        BrainwaveState.BETA: {
            'base_freq': 528.0,  # Solfeggio "miracle" frequency
            'beat_freq': 20.0,   # 13-30 Hz beat range, using 20Hz for SAM spatial freq
            'phase_deviation': 0.05,
            'sound_path': SoundPath.SEMI_CIRCULAR,
            'volume': 0.7
        },
        BrainwaveState.GAMMA: {
            'base_freq': 528.0,
            'beat_freq': 40.0,   # 30-100 Hz beat range, using 40Hz for SAM spatial freq
            'phase_deviation': 0.05,
            'sound_path': SoundPath.OPEN,
            'volume': 0.7
        }
    }
    
    # Get preset for requested state
    preset = presets[state]
    
    # Create a single node with the preset parameters
    node = Node(
        duration=duration,
        base_freq=preset['base_freq'],
        beat_freq=preset['beat_freq'],
        volume_left=preset['volume'],
        volume_right=preset['volume'],
        phase_deviation=preset['phase_deviation'],
        sound_path=preset['sound_path']
    )
    
    return [node]


def create_meditation_session_nodes(total_duration: float = 1200.0) -> List[Node]:
    """
    Create a complete meditation session that gradually transitions from 
    alert beta to relaxed alpha to meditative theta and back.
    
    Args:
        total_duration: Total session duration in seconds
        
    Returns:
        List of Node objects defining the meditation journey
    """
    # Calculate durations for each segment
    beta_time = total_duration * 0.1      # Starting alert state
    transition1 = total_duration * 0.05   # Transition to alpha
    alpha_time = total_duration * 0.2     # Relaxed state
    transition2 = total_duration * 0.05   # Transition to theta
    theta_time = total_duration * 0.4     # Meditation state
    transition3 = total_duration * 0.05   # Transition back to alpha
    alpha_end = total_duration * 0.15     # Return to relaxed
    
    # Create nodes list
    nodes = []
    
    # Beta state (alert, focused)
    nodes.append(Node(
        duration=beta_time,
        base_freq=528.0,
        beat_freq=20.0,
        volume_left=0.7,
        volume_right=0.7,
        phase_deviation=0.05,
        sound_path=SoundPath.SEMI_CIRCULAR
    ))
    
    # Transition to Alpha
    nodes.append(Node(
        duration=transition1,
        base_freq=484.0,
        beat_freq=15.0,
        volume_left=0.75,
        volume_right=0.75,
        phase_deviation=0.05,
        sound_path=SoundPath.SEMI_CIRCULAR
    ))
    
    # Alpha state (relaxed)
    nodes.append(Node(
        duration=alpha_time,
        base_freq=440.0,
        beat_freq=10.0,
        volume_left=0.8,
        volume_right=0.8,
        phase_deviation=0.05,
        sound_path=SoundPath.CIRCULAR
    ))
    
    # Transition to Theta
    nodes.append(Node(
        duration=transition2,
        base_freq=320.0,
        beat_freq=7.5,
        volume_left=0.85,
        volume_right=0.85,
        phase_deviation=0.05,
        sound_path=SoundPath.CIRCULAR
    ))
    
    # Theta state (meditative)
    nodes.append(Node(
        duration=theta_time,
        base_freq=200.0,
        beat_freq=5.0,
        volume_left=0.9,
        volume_right=0.9,
        phase_deviation=0.05,
        sound_path=SoundPath.CIRCULAR
    ))
    
    # Transition back to Alpha
    nodes.append(Node(
        duration=transition3,
        base_freq=320.0,
        beat_freq=7.5,
        volume_left=0.85,
        volume_right=0.85,
        phase_deviation=0.05,
        sound_path=SoundPath.CIRCULAR
    ))
    
    # Alpha state (return to relaxed)
    nodes.append(Node(
        duration=alpha_end,
        base_freq=440.0,
        beat_freq=10.0,
        volume_left=0.8,
        volume_right=0.8,
        phase_deviation=0.05,
        sound_path=SoundPath.CIRCULAR
    ))
    
    return nodes


def create_sleep_session_nodes(total_duration: float = 1800.0) -> List[Node]:
    """
    Create a sleep induction session that progressively guides the brain
    from alpha through theta to delta waves.
    
    Args:
        total_duration: Total session duration in seconds
        
    Returns:
        List of Node objects defining the sleep journey
    """
    # Calculate durations for each segment
    alpha_time = total_duration * 0.15    # Starting relaxed state
    transition1 = total_duration * 0.05   # Transition to theta
    theta_time = total_duration * 0.2     # Light sleep preparation
    transition2 = total_duration * 0.05   # Transition to delta
    delta_time = total_duration * 0.55    # Deep sleep state
    
    # Create nodes list
    nodes = []
    
    # Alpha state (relaxed)
    nodes.append(Node(
        duration=alpha_time,
        base_freq=440.0,
        beat_freq=10.0,
        volume_left=0.8,
        volume_right=0.8,
        phase_deviation=0.05,
        sound_path=SoundPath.CIRCULAR
    ))
    
    # Transition to Theta
    nodes.append(Node(
        duration=transition1,
        base_freq=320.0,
        beat_freq=7.5,
        volume_left=0.85,
        volume_right=0.85,
        phase_deviation=0.05,
        sound_path=SoundPath.CIRCULAR
    ))
    
    # Theta state (drowsy)
    nodes.append(Node(
        duration=theta_time,
        base_freq=200.0,
        beat_freq=5.0,
        volume_left=0.9,
        volume_right=0.9,
        phase_deviation=0.05,
        sound_path=SoundPath.CIRCULAR
    ))
    
    # Transition to Delta
    nodes.append(Node(
        duration=transition2,
        base_freq=150.0,
        beat_freq=3.5,
        volume_left=0.9,
        volume_right=0.9,
        phase_deviation=0.05,
        sound_path=SoundPath.CIRCULAR
    ))
    
    # Delta state (deep sleep)
    nodes.append(Node(
        duration=delta_time,
        base_freq=100.0,
        beat_freq=2.0,
        volume_left=0.9,
        volume_right=0.9,
        phase_deviation=0.05,
        sound_path=SoundPath.CIRCULAR
    ))
    
    return nodes


def create_focus_session_nodes(total_duration: float = 900.0) -> List[Node]:
    """
    Create a focus enhancement session that establishes an alert, 
    concentrated state through beta wave entrainment.
    
    Args:
        total_duration: Total session duration in seconds
        
    Returns:
        List of Node objects defining the focus session
    """
    # Calculate durations for each segment
    alpha_time = total_duration * 0.2     # Starting relaxed state
    transition1 = total_duration * 0.05   # Transition to beta
    beta_time = total_duration * 0.7      # Focused state main period
    transition2 = total_duration * 0.05   # Short transition to alpha
    
    # Create nodes list
    nodes = []
    
    # Alpha state (relaxed)
    nodes.append(Node(
        duration=alpha_time,
        base_freq=440.0,
        beat_freq=10.0,
        volume_left=0.8,
        volume_right=0.8,
        phase_deviation=0.05,
        sound_path=SoundPath.CIRCULAR
    ))
    
    # Transition to Beta
    nodes.append(Node(
        duration=transition1,
        base_freq=484.0,
        beat_freq=15.0,
        volume_left=0.75,
        volume_right=0.75,
        phase_deviation=0.05,
        sound_path=SoundPath.OPEN
    ))
    
    # Beta state (focused)
    nodes.append(Node(
        duration=beta_time,
        base_freq=528.0,
        beat_freq=20.0,
        volume_left=0.7,
        volume_right=0.7,
        phase_deviation=0.05,
        sound_path=SoundPath.OPEN
    ))
    
    return nodes


# -----------------------------------------------------------
# Track Generation and Export
# -----------------------------------------------------------

def generate_track_audio(voices, sample_rate=44100):
    """
    Mix multiple voices together into a single audio track.
    
    Args:
        voices: List of Voice objects to mix
        sample_rate: Audio sample rate in Hz
        
    Returns:
        NumPy array of shape (num_samples, 2) containing the mixed audio
    """
    if not voices:
        return np.zeros((0, 2), dtype=np.float32)
    
    # Find maximum track length
    track_length_seconds = max(v.total_duration for v in voices)
    total_samples = int(track_length_seconds * sample_rate)
    
    # Initialize mix buffer
    mixed = np.zeros((total_samples, 2), dtype=np.float32)
    
    # Mix each voice into the buffer
    for v in voices:
        try:
            buf = v.generate_samples()
            length = buf.shape[0]
            if length > 0:
                # Only add up to the end of the buffer
                buf_to_add = buf[:min(length, total_samples)]
                mixed[:buf_to_add.shape[0]] += buf_to_add
        except Exception as e:
            print(f"Error generating voice samples: {e}")
            continue
    
    # Normalize if needed to prevent clipping
    max_val = np.max(np.abs(mixed))
    if max_val > 1.0:
        mixed /= max_val
    
    return mixed


def export_wav(audio_data, sample_rate, file_path):
    """
    Export audio data to WAV file.
    
    Args:
        audio_data: NumPy array of shape (num_samples, 2)
        sample_rate: Audio sample rate in Hz
        file_path: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    if audio_data is None or audio_data.size == 0:
        print(f"Error: No audio data to export to {file_path}")
        return False
    
    # Check for silent audio
    max_amplitude = np.max(np.abs(audio_data))
    print(f"Exporting WAV file to {file_path}, max amplitude: {max_amplitude}")
    
    try:
        sf.write(file_path, audio_data, sample_rate, format='WAV')
        print(f"Successfully wrote WAV file: {file_path}, shape: {audio_data.shape}")
        return True
    except Exception as e:
        print(f"Error writing WAV file: {e}")
        return False


def export_flac(audio_data, sample_rate, file_path):
    """
    Export audio data to FLAC file.
    
    Args:
        audio_data: NumPy array of shape (num_samples, 2)
        sample_rate: Audio sample rate in Hz
        file_path: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    if audio_data is None or audio_data.size == 0:
        print(f"Error: No audio data to export to {file_path}")
        return False
    
    # Check for silent audio
    max_amplitude = np.max(np.abs(audio_data))
    print(f"Exporting FLAC file to {file_path}, max amplitude: {max_amplitude}")
    
    try:
        sf.write(file_path, audio_data, sample_rate, format='FLAC')
        print(f"Successfully wrote FLAC file: {file_path}, shape: {audio_data.shape}")
        return True
    except Exception as e:
        print(f"Error writing FLAC file: {e}")
        return False


def export_mp3(audio_data, sample_rate, file_path):
    """
    Export audio data to MP3 file.
    
    Args:
        audio_data: NumPy array of shape (num_samples, 2)
        sample_rate: Audio sample rate in Hz
        file_path: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from pydub import AudioSegment
        import os
    except ImportError:
        print("Error: pydub module not found. Install it with 'pip install pydub'")
        return False
    
    if audio_data is None or audio_data.size == 0:
        print(f"Error: No audio data to export to {file_path}")
        return False
    
    # Check for silent audio
    max_amplitude = np.max(np.abs(audio_data))
    print(f"Exporting MP3 file to {file_path}, max amplitude: {max_amplitude}")
    
    try:
        # First save as temporary WAV file
        temp_wav = file_path + ".temp.wav"
        sf.write(temp_wav, audio_data, sample_rate, format='WAV')
        
        # Convert to MP3 using pydub
        seg = AudioSegment.from_wav(temp_wav)
        seg.export(file_path, format="mp3")
        
        # Remove temporary file
        os.remove(temp_wav)
        
        print(f"Successfully wrote MP3 file: {file_path}, shape: {audio_data.shape}")
        return True
    except Exception as e:
        print(f"Error writing MP3 file: {e}")
        return False


# -----------------------------------------------------------
# Example Usage
# -----------------------------------------------------------

def create_example_audio():
    """
    Create a simple example audio file demonstrating the improved implementation.
    """
    # Create nodes for a 30-second alpha relaxation
    nodes = get_preset_nodes_for_state(BrainwaveState.ALPHA, duration=30.0)
    
    # Create different voice types
    sam_voice = ImprovedSAMVoice(nodes)
    binaural_voice = BinauralBeatVoice(nodes)
    
    # Generate audio with SAM voice
    sam_audio = sam_voice.generate_samples()
    export_wav(sam_audio, 44100, "sam_example.wav")
    
    # Generate audio with binaural voice
    binaural_audio = binaural_voice.generate_samples()
    export_wav(binaural_audio, 44100, "binaural_example.wav")
    
    # Create a multi-voice track with background pink noise
    noise_voice = PinkNoiseVoice(nodes)
    
    # Mix voices together with relative volumes
    mixed_voices = [
        sam_voice,       # Main entrainment
        noise_voice      # Background noise
    ]
    
    mixed_audio = generate_track_audio(mixed_voices)
    export_wav(mixed_audio, 44100, "mixed_example.wav")
    
    print("Created example audio files.")


if __name__ == "__main__":
 
