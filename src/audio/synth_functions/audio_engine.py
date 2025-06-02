import numpy as np
import librosa
import soundfile as sf
from enum import Enum
from typing import List, Tuple, Optional, Union
import time
import os

# Attempt to import slab for SAM voices
try:
    import slab
    SLAB_AVAILABLE = True
    # Try loading HRTF once at import to check availability and get default SR
    try:
        _DEFAULT_HRTF = slab.HRTF.kemar()
        DEFAULT_HRTF_SR = _DEFAULT_HRTF.samplerate
        print(f"SLAB: KEMAR HRTF loaded successfully. Default SR: {DEFAULT_HRTF_SR} Hz")
        del _DEFAULT_HRTF # Don't keep it loaded globally, load per instance
    except Exception as e:
        print(f"SLAB WARNING: Could not load KEMAR HRTF at import: {e}")
        print("SLAB WARNING: SAMVoice/SAMEffectVoice may fail or use fallback SR.")
        DEFAULT_HRTF_SR = 44100 # Fallback SR if HRTF load fails initially
        SLAB_AVAILABLE = False # Treat as unavailable if HRTF fails
except ImportError:
    print("SLAB WARNING: 'slab' library not found (pip install psychacoustics-laboratory).")
    print("SLAB WARNING: SAMVoice and SAMEffectVoice will not be available.")
    SLAB_AVAILABLE = False
    DEFAULT_HRTF_SR = 44100 # Define fallback SR

# -----------------------------------------------------------
# Enums and Configuration Classes
# -----------------------------------------------------------

class BrainwaveState(Enum):
    """Enum for different brainwave states with their frequency ranges"""
    DELTA = "delta"  # Deep sleep (0.1-4 Hz)
    THETA = "theta"  # Meditation, drowsiness (4-8 Hz)
    ALPHA = "alpha"  # Relaxed, calm (8-13 Hz)
    BETA = "beta"    # Alert, focused (13-30 Hz)
    GAMMA = "gamma"  # Higher mental activity, perception (30-100 Hz)

# Define valid path shapes for SAM voices (used instead of Enum)
VALID_SAM_PATHS = ['circle', 'square', 'linear', 'arc', 'diamond', 'oscillating_arc']

# -----------------------------------------------------------
# Data Classes
# -----------------------------------------------------------

class Node:
    def __init__(self, duration, base_freq, beat_freq, volume_left, volume_right):
        """
        Initialize a node for audio generation.

        Args:
            duration (float): Length of this node in seconds
            base_freq (float): Carrier frequency in Hz (used as source freq for SAMVoice)
            beat_freq (float): Beat/Pulse/Modulation frequency in Hz (used as SAM speed)
            volume_left (float): Volume for left channel (0.0 to 1.0)
            volume_right (float): Volume for right channel (0.0 to 1.0)
        """
        self.duration = float(duration)
        self.base_freq = float(base_freq)
        self.beat_freq = float(beat_freq)
        self.volume_left = float(volume_left)
        self.volume_right = float(volume_right)

    def to_dict(self):
        """Convert node to dictionary for serialization."""
        return {
            "duration": self.duration,
            "base_freq": self.base_freq,
            "beat_freq": self.beat_freq,
            "volume_left": self.volume_left,
            "volume_right": self.volume_right,
        }

    @staticmethod
    def from_dict(d):
        """Create a node from a dictionary."""
        return Node(
            d["duration"],
            d["base_freq"],
            d["beat_freq"],
            d["volume_left"],
            d["volume_right"],
        )

class Voice:
    """
    Base voice class. Subclasses override `generate_samples`.

    Handles timeline building and parameter interpolation based on Nodes.
    Forces sample rate to match KEMAR HRTF if a SAM voice is used or
    if SLAB is available and a default rate isn't specified matching it.
    Accepts optional sam_node_params for SAM voices.
    """
    def __init__(self, nodes: List['Node'], sample_rate: int = 44100, **kwargs): # Accept kwargs
        # Basic Node validation
        if not isinstance(nodes, list) or not all(isinstance(n, Node) for n in nodes):
             raise ValueError("Nodes must be a list of audio_engine.Node objects.")

        # --- Sample Rate Handling ---
        # Determine target SR based on SLAB availability and provided rate
        kamar_sr = int(DEFAULT_HRTF_SR) # Use global from top of file
        if SLAB_AVAILABLE:
            if int(sample_rate) != kamar_sr:
                print(f"Voice Warning: Provided SR {sample_rate}Hz != KEMAR SR {kamar_sr}Hz. Forcing KEMAR SR.")
                self.sample_rate = kamar_sr
            else:
                self.sample_rate = int(sample_rate)
        else:
             # If SLAB is not available, use the provided rate
             self.sample_rate = int(sample_rate)

        self.nodes = nodes

        # --- CORRECTED: Only set sam_node_params if passed and valid ---
        self.sam_node_params = kwargs.get('sam_node_params', None) # Store if passed, else None
        if self.sam_node_params is not None:
             if not isinstance(self.sam_node_params, list) or len(self.sam_node_params) != len(self.nodes):
                  print(f"Voice Warning: Length mismatch or invalid type for sam_node_params ({len(self.sam_node_params) if isinstance(self.sam_node_params, list) else 'Invalid'}) vs nodes ({len(self.nodes)}). Discarding SAM params for this voice.")
                  self.sam_node_params = None # Invalidate if length mismatch or not list

        # Initialize timeline and parameter storage
        self._build_timeline()


    def _build_timeline(self):
        """Build the timeline and pre-compute node value arrays."""
        self.start_times = [0.0]
        # Handle empty nodes case
        if not self.nodes:
            self.total_duration = 0.0
            self.num_samples = 0
            self.node_times = np.array([], dtype=np.float64)
            self.base_freq_values = np.array([], dtype=np.float64)
            self.beat_freq_values = np.array([], dtype=np.float64)
            self.vol_left_values = np.array([], dtype=np.float64)
            self.vol_right_values = np.array([], dtype=np.float64)
            # Initialize SAM value arrays as empty
            self.path_shape_values = []
            self.path_radius_values = np.array([], dtype=np.float64)
            self.arc_start_deg_values = np.array([], dtype=np.float64)
            self.arc_end_deg_values = np.array([], dtype=np.float64)
            return

        # Filter out zero-duration nodes for calculation, but keep for param arrays
        node_durations = [n.duration for n in self.nodes]
        valid_indices = [i for i, d in enumerate(node_durations) if d > 0]

        if not valid_indices: # All nodes have zero duration
             self.total_duration = 0.0
             self.num_samples = 0
             # Keep parameter arrays corresponding to original nodes list length
             self.node_times = np.zeros(len(self.nodes), dtype=np.float64) # All start at 0
        else:
             # Calculate start times based on positive durations only
             current_time = 0.0
             self.start_times = []
             node_map_idx = 0
             for i in range(len(self.nodes)):
                 self.start_times.append(current_time)
                 if i in valid_indices:
                     current_time += max(0.0, node_durations[i]) # Use max just in case

             self.total_duration = current_time # Total duration is end time of last node
             self.num_samples = int(self.total_duration * self.sample_rate)


        # Store parameters for ALL original nodes
        self.node_times = np.array(self.start_times[:len(self.nodes)], dtype=np.float64) # Ensure length matches nodes
        self.base_freq_values = np.array([n.base_freq for n in self.nodes], dtype=np.float64)
        self.beat_freq_values = np.array([n.beat_freq for n in self.nodes], dtype=np.float64)
        self.vol_left_values = np.array([n.volume_left for n in self.nodes], dtype=np.float64)
        self.vol_right_values = np.array([n.volume_right for n in self.nodes], dtype=np.float64)


        # --- CORRECTED: Add storage for SAM node params ONLY if available AND valid ---
        self.path_shape_values = []
        self.path_radius_values = np.array([], dtype=np.float64)
        self.arc_start_deg_values = np.array([], dtype=np.float64)
        self.arc_end_deg_values = np.array([], dtype=np.float64)

        # Check existence AND validity (not None, correct length) before processing
        if hasattr(self, 'sam_node_params') and self.sam_node_params is not None:
            # Double check length against original node list
            if len(self.sam_node_params) == len(self.nodes):
                try:
                    self.path_shape_values = [p['path_shape'] for p in self.sam_node_params]
                    self.path_radius_values = np.array([p['path_radius'] for p in self.sam_node_params], dtype=np.float64)
                    self.arc_start_deg_values = np.array([p['arc_start_deg'] for p in self.sam_node_params], dtype=np.float64)
                    self.arc_end_deg_values = np.array([p['arc_end_deg'] for p in self.sam_node_params], dtype=np.float64)
                except (KeyError, TypeError, AttributeError) as e:
                    print(f"Error processing sam_node_params in _build_timeline: {e}. Discarding SAM params.")
                    # Reset arrays if params are malformed
                    self.path_shape_values = []; self.path_radius_values = np.array([])
                    self.arc_start_deg_values = np.array([]); self.arc_end_deg_values = np.array([])
                    self.sam_node_params = None # Ensure invalid state is marked
            else:
                # Length mismatch caught in init, but redundant check here
                print("Error in _build_timeline: Mismatch len(sam_node_params) != len(nodes).")
                self.sam_node_params = None # Ensure invalid state is marked


    def _get_param_arrays(self) -> Tuple:
        """
        Generate arrays for all parameters interpolated across the timeline.
        Includes basic params and potentially SAM params if available.
        Returns a tuple containing time array and all parameter arrays.
        """
        num_samples = self.num_samples
        # Basic check for valid number of samples
        if num_samples <= 0 or not self.nodes:
            empty_array = np.array([], dtype=np.float32)
            # Return empty arrays matching the maximum possible tuple size (basic + sam)
            return (empty_array,) * 9

        # Initialize basic arrays
        t_array = np.arange(num_samples, dtype=np.float64) / self.sample_rate
        base_freq_array = np.zeros(num_samples, dtype=np.float64)
        beat_freq_array = np.zeros(num_samples, dtype=np.float64)
        vol_left_array = np.zeros(num_samples, dtype=np.float64)
        vol_right_array = np.zeros(num_samples, dtype=np.float64)

        # Check if SAM parameters are validly stored (correct length)
        has_sam_params = (hasattr(self, 'path_shape_values') and
                          isinstance(self.path_shape_values, list) and
                          len(self.path_shape_values) == len(self.nodes))

        # Initialize SAM arrays
        path_shape_array = np.full(num_samples, '', dtype=object)
        path_radius_array = np.zeros(num_samples, dtype=np.float64)
        arc_start_deg_array = np.zeros(num_samples, dtype=np.float64)
        arc_end_deg_array = np.zeros(num_samples, dtype=np.float64)

        # --- Interpolation Loop ---
        last_node_end_time = self.total_duration # Use calculated total duration

        for i in range(len(self.nodes)):
            # Get basic params
            current_params = { 'base_freq': self.base_freq_values[i], 'beat_freq': self.beat_freq_values[i],
                               'vol_left': self.vol_left_values[i], 'vol_right': self.vol_right_values[i], }
            if i < len(self.nodes) - 1:
                next_params = { 'base_freq': self.base_freq_values[i+1], 'beat_freq': self.beat_freq_values[i+1],
                                'vol_left': self.vol_left_values[i+1], 'vol_right': self.vol_right_values[i+1], }
                end_time = self.start_times[i+1]
            else:
                next_params = current_params # Hold last value
                end_time = last_node_end_time

            # Get SAM params if available
            current_sam_params = {}; next_sam_params = {}
            if has_sam_params:
                current_sam_params = { 'path_shape': self.path_shape_values[i], 'path_radius': self.path_radius_values[i],
                                       'arc_start': self.arc_start_deg_values[i], 'arc_end': self.arc_end_deg_values[i] }
                if i < len(self.nodes) - 1:
                    next_sam_params = { 'path_shape': self.path_shape_values[i+1], 'path_radius': self.path_radius_values[i+1],
                                        'arc_start': self.arc_start_deg_values[i+1], 'arc_end': self.arc_end_deg_values[i+1] }
                else:
                    next_sam_params = current_sam_params

            # Calculate indices for this segment
            start_time = self.start_times[i]
            # Add tolerance for float comparisons
            node_duration = end_time - start_time
            if node_duration < 1e-9: continue # Skip zero duration nodes in interpolation loop

            start_idx = max(0, int(start_time * self.sample_rate))
            # Ensure end_idx doesn't exceed num_samples
            end_idx = min(num_samples, int(end_time * self.sample_rate))
            segment_length = end_idx - start_idx

            if segment_length > 0:
                t_slice = slice(start_idx, end_idx)
                # Linspace excluding endpoint: generates segment_length points from 0 up to (not including) 1
                t_progress = np.linspace(0, 1, segment_length, endpoint=False)

                # Interpolate basic params
                base_freq_array[t_slice] = current_params['base_freq'] + t_progress * (next_params['base_freq'] - current_params['base_freq'])
                beat_freq_array[t_slice] = current_params['beat_freq'] + t_progress * (next_params['beat_freq'] - current_params['beat_freq'])
                vol_left_array[t_slice] = current_params['vol_left'] + t_progress * (next_params['vol_left'] - current_params['vol_left'])
                vol_right_array[t_slice] = current_params['vol_right'] + t_progress * (next_params['vol_right'] - current_params['vol_right'])

                # Interpolate/Set SAM params if they exist
                if has_sam_params:
                    path_shape_array[t_slice] = current_sam_params['path_shape'] # Hold shape
                    path_radius_array[t_slice] = current_sam_params['path_radius'] + t_progress * (next_sam_params['path_radius'] - current_sam_params['path_radius'])
                    arc_start_deg_array[t_slice] = current_sam_params['arc_start'] + t_progress * (next_sam_params['arc_start'] - current_sam_params['arc_start'])
                    arc_end_deg_array[t_slice] = current_sam_params['arc_end'] + t_progress * (next_sam_params['arc_end'] - current_sam_params['arc_end'])

        # Ensure the very last sample holds the final node's value
        if num_samples > 0:
            last_idx = num_samples - 1
            if self.nodes: # Ensure nodes list is not empty
                 base_freq_array[last_idx] = self.base_freq_values[-1]
                 beat_freq_array[last_idx] = self.beat_freq_values[-1]
                 vol_left_array[last_idx] = self.vol_left_values[-1]
                 vol_right_array[last_idx] = self.vol_right_values[-1]
                 if has_sam_params:
                      path_shape_array[last_idx] = self.path_shape_values[-1]
                      path_radius_array[last_idx] = self.path_radius_values[-1]
                      arc_start_deg_array[last_idx] = self.arc_start_deg_values[-1]
                      arc_end_deg_array[last_idx] = self.arc_end_deg_values[-1]

        # Construct return tuple conditionally
        result = (t_array, base_freq_array, beat_freq_array, vol_left_array, vol_right_array)
        if has_sam_params:
             result += (path_shape_array, path_radius_array, arc_start_deg_array, arc_end_deg_array)
        # Else: SAM arrays remain empty and are not added to the tuple
        # The receiving function must check the length of the received tuple

        return result


    def generate_samples(self) -> np.ndarray:
        """
        Generate audio samples. Subclasses must override this method.
        Returns a NumPy array of shape (num_samples, 2) containing stereo audio data.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must override generate_samples()")
# -----------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------

def apply_fade(signal: np.ndarray, fade_samples: int = 1000) -> np.ndarray:
    """
    Apply fade-in and fade-out to a signal to avoid clicks and pops.

    Args:
        signal: Input signal array (can be 1D or 2D)
        fade_samples: Number of samples for fading

    Returns:
        Signal with fades applied
    """
    if signal.ndim == 0 or signal.shape[0] <= 2 * fade_samples:
        # Signal too short for fading, return as is
        return signal

    num_samples = signal.shape[0]
    fade_samples = min(fade_samples, num_samples // 2) # Ensure fade doesn't exceed half length

    # Create a copy to avoid modifying the input
    result = signal.copy()

    # Create fade-in and fade-out windows (using Hanning for smoothness)
    # fade_in = 0.5 * (1 - np.cos(np.pi * np.arange(fade_samples) / fade_samples)) # Raised Cosine
    # fade_out = 0.5 * (1 + np.cos(np.pi * np.arange(fade_samples) / fade_samples))
    hanning_window = np.hanning(2 * fade_samples)
    fade_in = hanning_window[:fade_samples]
    fade_out = hanning_window[fade_samples:]


    # Apply fades (broadcast if signal is stereo)
    if result.ndim == 1:
        result[:fade_samples] *= fade_in
        result[-fade_samples:] *= fade_out
    elif result.ndim == 2:
        result[:fade_samples, :] *= fade_in[:, np.newaxis]
        result[-fade_samples:, :] *= fade_out[:, np.newaxis]

    return result

# -----------------------------------------------------------
# Spatial Angle Modulation (SAM) Voice Implementation
# -----------------------------------------------------------

class SAMVoice(Voice):
    """
    Spatial Angle Modulation (SAM) voice using KEMAR HRTFs via 'slab'.

    Generates a pure tone based on `base_freq` from nodes and moves it
    spatially according to path parameters defined per-node. The speed of
    movement along the path is determined by `beat_freq` from the nodes.

    Requires 'slab' library and KEMAR HRTFs.
    Node-level SAM parameters are passed via `sam_node_params` kwarg during init.
    Voice-level parameters control OLA processing and source amplitude.
    """
    def __init__(self,
                 nodes: List['Node'],
                 sample_rate: int = 44100,
                 # Voice-level OLA / Source Amp parameters:
                 frame_dur_ms: float = 46.4,
                 overlap_factor: int = 8, # Default increased
                 source_amp: float = 0.7,
                 # Kwargs to catch sam_node_params for base class
                 **kwargs
                ):
        if not SLAB_AVAILABLE:
            raise ImportError("SAMVoice requires the 'slab' library and KEMAR HRTFs.")

        # Pass nodes, SR, and potentially sam_node_params to base init
        super().__init__(nodes, sample_rate, **kwargs)

        # Store only voice-level OLA/Amp params
        self.frame_dur_ms = float(frame_dur_ms)
        self.overlap_factor = int(overlap_factor)
        self.source_amp = np.clip(float(source_amp), 0.0, 1.0)
        self.hrtf = None # Loaded below

        # Validate voice-level params
        if not isinstance(self.overlap_factor, int) or self.overlap_factor < 2:
             raise ValueError("overlap_factor must be an integer >= 2.")
        if self.frame_dur_ms <= 0:
             raise ValueError("frame_dur_ms must be positive.")

        # Load HRTF instance for this voice
        try:
            print(f"SAMVoice: Loading KEMAR HRTFs...")
            self.hrtf = slab.HRTF.kemar()
            # Ensure voice sample rate matches HRTF rate after potential base class adjustment
            if self.hrtf.samplerate != self.sample_rate:
                 print(f"SAMVoice CRITICAL WARNING: HRTF SR ({self.hrtf.samplerate}) != Voice SR ({self.sample_rate}). Forcing SR.")
                 self.sample_rate = int(self.hrtf.samplerate)
                 # Rebuild timeline if SR was forced *here* (should ideally happen in base)
                 self._build_timeline()
            print(f"SAMVoice: KEMAR HRTFs loaded (SR={self.sample_rate}Hz).")
        except Exception as e:
             raise RuntimeError(f"SAMVoice: Failed to load KEMAR HRTFs: {e}")


    def _calculate_paths(self, t_array: np.ndarray, modulation_frequency_array: np.ndarray,
                         path_shape_array: np.ndarray, path_radius_array: np.ndarray,
                         arc_start_deg_array: np.ndarray, arc_end_deg_array: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates azimuth and elevation degrees based on time and interpolated
        node parameters for modulation frequency and SAM path/shape/angles.
        """
        num_samples = len(t_array)
        if num_samples == 0: return np.array([]), np.array([])

        # --- Calculate Modulation Phase ---
        mod_freq = np.maximum(0, modulation_frequency_array)
        # Corrected cumulative phase calculation (radians)
        instantaneous_phase = np.cumsum(2 * np.pi * mod_freq / self.sample_rate)

        # --- Calculate Path Coordinates based on time-varying parameters ---
        x = np.zeros_like(t_array)
        y = np.zeros_like(t_array)
        static_mask = (mod_freq < 1e-6) # Use threshold for static check

        # Iterate through unique shapes present in the array
        # This handles transitions between shapes at node boundaries
        unique_shapes, shape_indices = np.unique(path_shape_array, return_inverse=True)

        for shape_idx, shape in enumerate(unique_shapes):
             mask = (shape_indices == shape_idx) # Samples where this shape applies
             if not np.any(mask): continue

             # Get slices of arrays corresponding to this shape's active period
             current_phase = instantaneous_phase[mask]
             current_radius = path_radius_array[mask]
             current_arc_start = arc_start_deg_array[mask]
             current_arc_end = arc_end_deg_array[mask]
             current_x = np.zeros_like(current_phase)
             current_y = np.zeros_like(current_phase)

             # Calculate coordinates based on the current shape
             if shape == 'circle':
                 angle_rad = current_phase
                 current_x = current_radius * np.sin(angle_rad)
                 current_y = current_radius * np.cos(angle_rad)
             elif shape == 'linear': # Front-back oscillation along Y axis
                 current_y = current_radius * np.sin(current_phase)
             elif shape == 'arc': # Sinusoidal sweep based on phase
                 arc_progress = (1 - np.cos(current_phase)) / 2 # 0 -> 1 -> 0
                 current_angle_deg = current_arc_start + arc_progress * (current_arc_end - current_arc_start)
                 current_angle_rad = np.radians(current_angle_deg)
                 current_x = current_radius * np.sin(current_angle_rad)
                 current_y = current_radius * np.cos(current_angle_rad)
             elif shape == 'oscillating_arc': # Linear (triangle wave) sweep
                 norm_phase_mod_1 = np.mod(current_phase / (2 * np.pi), 1.0) # Phase 0-1 repeating
                 triangle_norm = 1.0 - (2 * np.abs(norm_phase_mod_1 - 0.5)) # 0->1->0

                 start_le_end = current_arc_start <= current_arc_end
                 diff = np.abs(current_arc_end - current_arc_start)
                 # Vectorized conditional assignment using np.where
                 current_angle_deg = np.where(start_le_end,
                                              current_arc_start + triangle_norm * diff,
                                              current_arc_start - triangle_norm * diff)

                 current_angle_rad = np.radians(current_angle_deg)
                 current_x = current_radius * np.sin(current_angle_rad)
                 current_y = current_radius * np.cos(current_angle_rad)
             elif shape == 'square' or shape == 'diamond': # Approximation
                  cos_phase = np.cos(current_phase); sin_phase = np.sin(current_phase)
                  abs_cos = np.abs(cos_phase); abs_sin = np.abs(sin_phase)
                  max_comp = np.maximum(abs_cos, abs_sin)
                  # Avoid division by zero safely
                  inv_max_comp = np.zeros_like(max_comp)
                  valid_max = max_comp > 1e-9
                  inv_max_comp[valid_max] = 1.0 / max_comp[valid_max]

                  x_sq = current_radius * cos_phase * inv_max_comp
                  y_sq = current_radius * sin_phase * inv_max_comp
                  if shape == 'square':
                       current_x, current_y = x_sq, y_sq
                  else: # Diamond (rotated square)
                       cos45, sin45 = np.cos(np.pi / 4), np.sin(np.pi / 4)
                       current_x = x_sq * cos45 + y_sq * sin45
                       current_y = -x_sq * sin45 + y_sq * cos45
             else:
                  # Default to circle if shape unrecognized (shouldn't happen with validation)
                  print(f"Warning: Unrecognized path shape '{shape}' encountered. Defaulting to circle.")
                  angle_rad = current_phase
                  current_x = current_radius * np.sin(angle_rad)
                  current_y = current_radius * np.cos(angle_rad)


             # Place calculated values back into the main arrays using the mask
             x[mask] = current_x
             y[mask] = current_y

        # --- Handle Static Case ---
        # Where modulation frequency is effectively zero, place source in front
        x[static_mask] = 0.0
        # Use the radius defined by the node at that static point
        y[static_mask] = path_radius_array[static_mask]

        # --- Convert to Azimuth/Elevation ---
        # slab: 0=front, +90=right | cartesian: +x=right, +y=front | angle = atan2(x,y)
        azimuth_deg = np.degrees(np.arctan2(x, y))
        elevation_deg = np.zeros_like(azimuth_deg) # Keep horizontal

        return azimuth_deg, elevation_deg


    def generate_samples(self) -> np.ndarray:
        """Generates the spatialized pure tone audio."""
        if not self.hrtf:
             print("SAMVoice Error: HRTF not loaded. Cannot generate samples.")
             return np.zeros((0, 2), dtype=np.float32)

        # --- 1. Get ALL Interpolated Parameter Arrays ---
        param_arrays = self._get_param_arrays()
        # Check length to see if SAM params are included
        if len(param_arrays) == 9:
             t_array, base_freq_array, beat_freq_array, vol_left_array, vol_right_array, \
             path_shape_array, path_radius_array, arc_start_deg_array, arc_end_deg_array = param_arrays
        elif len(param_arrays) == 5: # Only basic params returned
             # This implies sam_node_params was missing or invalid
             if self.nodes: # Check if nodes actually exist
                  raise ValueError("SAMVoice initialized without valid sam_node_params or they failed processing.")
             else: # No nodes, return silence
                  print("SAMVoice: No nodes, generating silence.")
                  return np.zeros((self.num_samples, 2), dtype=np.float32)
        else: # Unexpected tuple length
             raise ValueError(f"SAMVoice: _get_param_arrays returned unexpected tuple length {len(param_arrays)}")

        num_samples = len(t_array)
        if num_samples == 0: return np.zeros((0, 2), dtype=np.float32)

        # --- 2. Generate Mono Source Signal ---
        instantaneous_freq = np.maximum(0.0, base_freq_array)
        # Corrected phase calculation
        carrier_phase = np.cumsum(2 * np.pi * instantaneous_freq / self.sample_rate)
        mono_source_data = self.source_amp * np.sin(carrier_phase)

        # --- 3. Calculate Spatial Path using interpolated arrays ---
        azimuth_deg, elevation_deg = self._calculate_paths(
             t_array, beat_freq_array, path_shape_array, path_radius_array,
             arc_start_deg_array, arc_end_deg_array
        )

        # --- 4. Overlap-Add Processing Setup ---
        frame_len = slab.Signal.in_samples(self.frame_dur_ms / 1000.0, self.sample_rate)
        if frame_len <= 0:
            print(f"SAMVoice Error: Frame duration -> frame length {frame_len}. Returning silence.")
            return np.zeros((num_samples, 2), dtype=np.float32)
        step_size = frame_len // self.overlap_factor
        if step_size <= 0:
            print(f"SAMVoice Error: Step size {step_size}. Returning silence.")
            return np.zeros((num_samples, 2), dtype=np.float32)

        # Window selection based on voice-level overlap factor
        if self.overlap_factor == 2: window_coeffs = np.sqrt(np.hanning(frame_len))
        elif self.overlap_factor == 4: window_coeffs = np.sqrt(np.hanning(frame_len))
        else: window_coeffs = np.hanning(frame_len) # Standard Hanning for 8x or others
        window = window_coeffs[:, np.newaxis]

        buffer_padding = frame_len
        binaural_output_data = np.zeros((num_samples + buffer_padding, 2), dtype=np.float64)

        print(f"SAMVoice: OLA - frames={num_samples // step_size}, frame={frame_len}, step={step_size}, overlap={(1.0 - 1.0/self.overlap_factor)*100:.1f}% using {'sqrt(Hanning)' if self.overlap_factor in [2,4] else 'Hanning'} window")
        n_frames_ola = 0

        # --- 5. Overlap-Add Loop ---
        start_ola_time = time.time()
        for i in range(0, num_samples, step_size): # Process full input
            start = i
            end = start + frame_len
            n_frames_ola += 1

            frame_data_1d = mono_source_data[start:end]
            actual_frame_len = len(frame_data_1d)
            if actual_frame_len == 0: continue

            current_window = window[:actual_frame_len]
            windowed_frame_data = frame_data_1d[:, np.newaxis] * current_window

            if actual_frame_len < frame_len:
                padding_len = frame_len - actual_frame_len
                windowed_frame_data = np.pad(windowed_frame_data, ((0, padding_len), (0, 0)), 'constant')

            # Position for this frame (midpoint of step)
            mid_idx = min(start + step_size // 2, num_samples - 1)
            azi = azimuth_deg[mid_idx]
            ele = elevation_deg[mid_idx]

            try:
                windowed_frame_sound = slab.Sound(windowed_frame_data, samplerate=self.sample_rate)
                binaural_filter = self.hrtf.interpolate(azimuth=azi, elevation=ele, method='nearest')
                filtered_frame = binaural_filter.apply(windowed_frame_sound)
                output_frame_data = filtered_frame.data

                len_to_add = min(len(output_frame_data), binaural_output_data.shape[0] - start)
                binaural_output_data[start : start + len_to_add] += output_frame_data[:len_to_add]
            except Exception as e:
                print(f"\nSAMVoice Warning: OLA frame {n_frames_ola} failed. Az={azi:.1f}, El={ele:.1f}. Err: {e}")
                # traceback.print_exc() # Optionally print full traceback for debug

        end_ola_time = time.time()
        print(f"SAMVoice: OLA finished {n_frames_ola} frames in {end_ola_time - start_ola_time:.2f}s.")

        # --- 6. Finalize Output ---
        final_signal_data = binaural_output_data[:num_samples] # Trim

        if np.isnan(final_signal_data).any():
             print("SAMVoice Warning: NaNs detected. Replacing with 0.")
             final_signal_data = np.nan_to_num(final_signal_data)

        # Apply final volume envelopes from nodes
        if len(vol_left_array) == num_samples and len(vol_right_array) == num_samples:
             final_signal_data[:, 0] *= vol_left_array
             final_signal_data[:, 1] *= vol_right_array
        else:
             print("SAMVoice Warning: Volume array length mismatch. Skipping final volume application.")

        # Normalize post-volume (conservative)
        max_abs_val = np.max(np.abs(final_signal_data)) if final_signal_data.size > 0 else 0
        if max_abs_val > 1.0:
             print(f"SAMVoice Warning: Clipping detected (max={max_abs_val:.3f}). Normalizing to 0.98.")
             final_signal_data /= (max_abs_val / 0.98)
        elif max_abs_val < 1e-9:
             print("SAMVoice Warning: Output is silent.")

        return final_signal_data.astype(np.float32)
# -----------------------------------------------------------
# SAM Effect Wrapper Voice Implementation
# -----------------------------------------------------------

class SAMEffectVoice(Voice):
    """
    Applies Spatial Angle Modulation (SAM) as an effect to the audio
    generated by another Voice instance.

    Takes a `source_voice` instance during initialization. Its audio is
    converted to mono and spatialized using HRTFs.

    The parameters for the SAM effect itself (modulation speed via `beat_freq`,
    path shape, final volume via `vol_left`/`vol_right`) are controlled by
    the `nodes` passed to *this* SAMEffectVoice instance (via `sam_node_params`).
    Voice-level parameters control OLA processing and normalization target peak.
    """
    def __init__(self,
                 nodes: List['Node'], # Effect's basic nodes
                 source_voice: Voice, # The voice instance providing audio input
                 sample_rate: int = 44100,
                 # Voice-level OLA / Peak parameters:
                 frame_dur_ms: float = 46.4,
                 overlap_factor: int = 8, # Default increased
                 target_peak: float = 0.95,
                 # Kwargs to catch sam_node_params for base class
                 **kwargs
                ):
        if not SLAB_AVAILABLE:
            raise ImportError("SAMEffectVoice requires the 'slab' library.")
        if not isinstance(source_voice, Voice):
            raise TypeError("source_voice must be an instance of a Voice subclass.")

        # Pass effect's nodes, SR, and potentially sam_node_params to base init
        super().__init__(nodes, sample_rate, **kwargs)

        self.source_voice = source_voice
        # Store only voice-level OLA/Peak params
        self.frame_dur_ms = float(frame_dur_ms)
        self.overlap_factor = int(overlap_factor)
        self.target_peak = np.clip(float(target_peak), 0.0, 1.0)
        self.hrtf = None

        # Validate voice-level params
        if not isinstance(self.overlap_factor, int) or self.overlap_factor < 2:
              raise ValueError("overlap_factor must be an integer >= 2.")
        if self.frame_dur_ms <= 0:
              raise ValueError("frame_dur_ms must be positive.")

        # --- Sample Rate Check & HRTF Load ---
        if self.source_voice.sample_rate != self.sample_rate:
             print(f"SAMEffectVoice Warning: Source voice SR ({self.source_voice.sample_rate}Hz) != Effect SR ({self.sample_rate}Hz). Will resample.")
        try:
             print(f"SAMEffectVoice: Loading KEMAR HRTFs...")
             self.hrtf = slab.HRTF.kemar()
             # Ensure voice SR matches HRTF SR
             if self.hrtf.samplerate != self.sample_rate:
                  print(f"SAMEffectVoice CRITICAL WARNING: HRTF SR ({self.hrtf.samplerate}) != Effect SR ({self.sample_rate}). Forcing SR.")
                  self.sample_rate = int(self.hrtf.samplerate)
                  # Rebuild timeline if SR forced here
                  self._build_timeline()
             print(f"SAMEffectVoice: KEMAR HRTFs loaded (SR={self.sample_rate}Hz).")
        except Exception as e:
              raise RuntimeError(f"SAMEffectVoice: Failed to load KEMAR HRTFs: {e}")

        # Match duration/samples to source voice AFTER potential SR changes
        self.total_duration = self.source_voice.total_duration
        self.num_samples = int(self.total_duration * self.sample_rate) # Use potentially updated SR
        print(f"SAMEffectVoice: Duration/samples matched to source ({self.total_duration:.2f}s / {self.num_samples} samples @ {self.sample_rate}Hz).")


    # Use SAMVoice's _calculate_paths method (shared logic)
    _calculate_paths = SAMVoice._calculate_paths

    def generate_samples(self) -> np.ndarray:
        """Generates the audio by applying SAM effect to the source voice."""
        if not self.hrtf:
             print("SAMEffectVoice Error: HRTF not loaded.")
             return np.zeros((0, 2), dtype=np.float32)

        # --- 1. Generate & Prepare Source Audio ---
        print(f"SAMEffectVoice: Generating source audio from {type(self.source_voice).__name__}...")
        source_audio_stereo = self.source_voice.generate_samples()
        if source_audio_stereo.shape[0] == 0:
             print("SAMEffectVoice Warning: Source voice generated no audio.")
             return np.zeros((0, 2), dtype=np.float32)

        # --- 2. Resample source if necessary ---
        if self.source_voice.sample_rate != self.sample_rate:
             print(f"SAMEffectVoice: Resampling source audio from {self.source_voice.sample_rate}Hz to {self.sample_rate}Hz...")
             try:
                  source_audio_stereo = librosa.resample(source_audio_stereo.T,
                                                         orig_sr=self.source_voice.sample_rate,
                                                         target_sr=self.sample_rate, res_type='kaiser_fast').T
                  print(f"SAMEffectVoice: Resampling complete, new shape {source_audio_stereo.shape}.")
             except Exception as e:
                  print(f"SAMEffectVoice Error during resampling: {e}. Returning silence.")
                  traceback.print_exc()
                  return np.zeros((self.num_samples, 2), dtype=np.float32)


        # --- 3. Convert Source to Mono ---
        if source_audio_stereo.ndim == 2 and source_audio_stereo.shape[1] >= 2:
             mono_source_data = np.mean(source_audio_stereo[:, :2], axis=1) # Avg L/R if stereo or more
        elif source_audio_stereo.ndim == 1:
             mono_source_data = source_audio_stereo
        else:
             print(f"SAMEffectVoice Error: Unexpected source audio shape: {source_audio_stereo.shape}.")
             return np.zeros((self.num_samples, 2), dtype=np.float32)

        num_samples_source = len(mono_source_data)

        # --- 4. Get Effect Parameter Arrays (including SAM) ---
        # Use the effect's own timeline/nodes via _get_param_arrays
        param_arrays = self._get_param_arrays()
        if len(param_arrays) == 9:
             t_array, _, beat_freq_array, vol_left_array, vol_right_array, \
             path_shape_array, path_radius_array, arc_start_deg_array, arc_end_deg_array = param_arrays
        elif len(param_arrays) == 5:
             if self.nodes: raise ValueError("SAMEffectVoice requires node-level SAM parameters.")
             else: print("SAMEffectVoice: No nodes, cannot generate effect."); return np.zeros((num_samples_source, 2), dtype=np.float32)
        else: raise ValueError(f"SAMEffectVoice: _get_param_arrays returned unexpected length {len(param_arrays)}")

        num_samples_effect = len(t_array)

        # --- 5. Adjust Lengths - Make effect params match source length ---
        # Determine the final number of samples based on the processed source audio
        num_samples = num_samples_source
        mono_source_data = mono_source_data[:num_samples] # Ensure source isn't longer

        if num_samples_effect != num_samples:
             print(f"SAMEffectVoice Warning: Effect param length ({num_samples_effect}) differs from source length ({num_samples}). Adjusting params.")
             # Resample or trim/pad parameter arrays? Trimming/padding is simpler. Pad with last value.
             if num_samples_effect < num_samples:
                 pad_width = num_samples - num_samples_effect
                 beat_freq_array = np.pad(beat_freq_array, (0, pad_width), mode='edge')
                 vol_left_array = np.pad(vol_left_array, (0, pad_width), mode='edge')
                 vol_right_array = np.pad(vol_right_array, (0, pad_width), mode='edge')
                 path_shape_array = np.pad(path_shape_array, (0, pad_width), mode='edge')
                 path_radius_array = np.pad(path_radius_array, (0, pad_width), mode='edge')
                 arc_start_deg_array = np.pad(arc_start_deg_array, (0, pad_width), mode='edge')
                 arc_end_deg_array = np.pad(arc_end_deg_array, (0, pad_width), mode='edge')
             else: # Effect params longer, trim
                 beat_freq_array = beat_freq_array[:num_samples]
                 vol_left_array = vol_left_array[:num_samples]
                 vol_right_array = vol_right_array[:num_samples]
                 path_shape_array = path_shape_array[:num_samples]
                 path_radius_array = path_radius_array[:num_samples]
                 arc_start_deg_array = arc_start_deg_array[:num_samples]
                 arc_end_deg_array = arc_end_deg_array[:num_samples]
             # Regenerate t_array to match final num_samples
             t_array = np.arange(num_samples, dtype=np.float64) / self.sample_rate


        # --- 6. Calculate Spatial Path using effect's interpolated arrays ---
        azimuth_deg, elevation_deg = self._calculate_paths(
             t_array, beat_freq_array, path_shape_array, path_radius_array,
             arc_start_deg_array, arc_end_deg_array
        )

        # --- 7. Overlap-Add Processing Setup & Loop ---
        # Uses self.frame_dur_ms, self.overlap_factor
        frame_len = slab.Signal.in_samples(self.frame_dur_ms / 1000.0, self.sample_rate)
        if frame_len <= 0: return np.zeros((num_samples, 2), dtype=np.float32)
        step_size = frame_len // self.overlap_factor
        if step_size <= 0: return np.zeros((num_samples, 2), dtype=np.float32)

        # Window selection
        if self.overlap_factor == 2: window_coeffs = np.sqrt(np.hanning(frame_len))
        elif self.overlap_factor == 4: window_coeffs = np.sqrt(np.hanning(frame_len))
        else: window_coeffs = np.hanning(frame_len)
        window = window_coeffs[:, np.newaxis]

        buffer_padding = frame_len
        # Use num_samples based on source data length
        binaural_output_data = np.zeros((num_samples + buffer_padding, 2), dtype=np.float64)

        print(f"SAMEffectVoice: OLA - frames={num_samples // step_size}, frame={frame_len}, step={step_size}, overlap={(1.0 - 1.0/self.overlap_factor)*100:.1f}% using {'sqrt(Hanning)' if self.overlap_factor in [2,4] else 'Hanning'} window")
        n_frames_ola = 0
        start_ola_time = time.time()

        # Loop over the MONO source data
        for i in range(0, num_samples, step_size):
            start = i
            end = start + frame_len
            n_frames_ola += 1

            frame_data_1d = mono_source_data[start:end]
            actual_frame_len = len(frame_data_1d)
            if actual_frame_len == 0: continue

            current_window = window[:actual_frame_len]
            windowed_frame_data = frame_data_1d[:, np.newaxis] * current_window

            if actual_frame_len < frame_len:
                padding_len = frame_len - actual_frame_len
                windowed_frame_data = np.pad(windowed_frame_data, ((0, padding_len), (0, 0)), 'constant')

            # Position for this frame
            mid_idx = min(start + step_size // 2, num_samples - 1)
            azi = azimuth_deg[mid_idx]
            ele = elevation_deg[mid_idx]

            try:
                windowed_frame_sound = slab.Sound(windowed_frame_data, samplerate=self.sample_rate)
                binaural_filter = self.hrtf.interpolate(azimuth=azi, elevation=ele, method='nearest')
                filtered_frame = binaural_filter.apply(windowed_frame_sound)
                output_frame_data = filtered_frame.data

                len_to_add = min(len(output_frame_data), binaural_output_data.shape[0] - start)
                binaural_output_data[start : start + len_to_add] += output_frame_data[:len_to_add]
            except Exception as e:
                print(f"\nSAMEffectVoice Warning: OLA frame {n_frames_ola} failed. Az={azi:.1f}, El={ele:.1f}. Err: {e}")

        end_ola_time = time.time()
        print(f"SAMEffectVoice: OLA finished {n_frames_ola} frames in {end_ola_time - start_ola_time:.2f}s.")

        # --- 8. Finalize Output ---
        final_signal_data = binaural_output_data[:num_samples] # Trim to source length

        if np.isnan(final_signal_data).any():
             print("SAMEffectVoice Warning: NaNs detected. Replacing with 0.")
             final_signal_data = np.nan_to_num(final_signal_data)

        # Normalize post-OLA to target peak *before* applying effect volumes
        max_abs_ola = np.max(np.abs(final_signal_data)) if final_signal_data.size > 0 else 0
        if max_abs_ola > self.target_peak and max_abs_ola > 1e-9:
             print(f"SAMEffectVoice: Normalizing post-OLA peak from {max_abs_ola:.3f} to {self.target_peak:.3f}...")
             norm_factor = self.target_peak / max_abs_ola
             final_signal_data *= norm_factor
        elif max_abs_ola < 1e-9:
             print("SAMEffectVoice Warning: Post-OLA signal is silent.")

        # Apply final volume envelopes from the *effect's* nodes
        if len(vol_left_array) == num_samples and len(vol_right_array) == num_samples:
             final_signal_data[:, 0] *= vol_left_array
             final_signal_data[:, 1] *= vol_right_array
        else:
             print("SAMEffectVoice Warning: Effect volume array length mismatch. Skipping final volume application.")


        # Final check/clip for safety
        max_final_val = np.max(np.abs(final_signal_data)) if final_signal_data.size > 0 else 0
        if max_final_val > 1.0:
             print(f"SAMEffectVoice Warning: Clipping detected after volume (max={max_final_val:.3f}). Clipping to [-1, 1].")
             final_signal_data = np.clip(final_signal_data, -1.0, 1.0)

        return final_signal_data.astype(np.float32)

# -----------------------------------------------------------
# Standard Beat Voices (Unchanged from original, but ensure compatibility)
# -----------------------------------------------------------

class MonauralBeatVoice(Voice):
    """
    Monaural beat generator. Same carrier in both channels, amplitude modulated
    at the beat frequency. Physically present beat.
    """
    def __init__(self, nodes, sample_rate=44100, modulation_depth=0.8):
        super().__init__(nodes, sample_rate)
        self.modulation_depth = np.clip(float(modulation_depth), 0.0, 1.0)

    def generate_samples(self):
        t_array, base_freq_array, beat_freq_array, vol_left_array, vol_right_array = self._get_param_arrays()
        num_samples = t_array.size
        if num_samples == 0: return np.zeros((0, 2), dtype=np.float32)

        # Integrate frequency for phase - carrier
        instantaneous_carrier_freq = np.maximum(0.0, base_freq_array)
        carrier_phase = 2 * np.pi * np.cumsum(instantaneous_carrier_freq) / self.sample_rate
        carrier = np.sin(carrier_phase)

        # Integrate frequency for phase - modulation
        instantaneous_beat_freq = np.maximum(0.0, beat_freq_array)
        mod_phase = 2 * np.pi * np.cumsum(instantaneous_beat_freq) / self.sample_rate

        min_amp = 1.0 - self.modulation_depth
        # Use cosine for modulation to start at peak? Or sine shifted? Let's use sine.
        mod = min_amp + (1.0 - min_amp) * (0.5 + 0.5 * np.sin(mod_phase))

        s_left = carrier * mod * vol_left_array
        s_right = carrier * mod * vol_right_array

        # Combine into stereo output and apply fade
        audio = np.column_stack((s_left, s_right)).astype(np.float32)
        audio = apply_fade(audio, fade_samples=min(1000, num_samples // 10)) # Apply fade

        return audio

class BinauralBeatVoice(Voice):
    """
    Classic binaural beat generator. Produces a slightly different frequency
    in each ear to create a perceived beat frequency.
    Uses cumulative sum for phase calculation to handle varying frequencies.
    """
    def __init__(self, nodes: List['Node'], sample_rate: int = 44100, **kwargs): # Added **kwargs
        # Pass kwargs to base class (might include sam_node_params, though unused here)
        super().__init__(nodes, sample_rate, **kwargs)
        # No voice-specific parameters for standard binaural beats

    def generate_samples(self) -> np.ndarray:
        # Use length of tuple returned by _get_param_arrays to check for SAM params
        param_arrays = self._get_param_arrays()
        if len(param_arrays) == 9: # Includes SAM params (ignore them)
             t_array, base_freq_array, beat_freq_array, vol_left_array, vol_right_array, \
             _, _, _, _ = param_arrays # Unpack and ignore SAM arrays
        elif len(param_arrays) == 5: # Only basic params
             t_array, base_freq_array, beat_freq_array, vol_left_array, vol_right_array = param_arrays
        else: # Unexpected tuple length
             raise ValueError(f"BinauralBeatVoice: _get_param_arrays returned unexpected tuple length {len(param_arrays)}")

        num_samples = t_array.size
        if num_samples == 0: return np.zeros((0, 2), dtype=np.float32)

        # Calculate instantaneous frequencies for left and right channels
        half_beat_freq = beat_freq_array / 2.0
        # Ensure frequencies don't go below zero
        left_freq_array = np.maximum(0.0, base_freq_array - half_beat_freq)
        right_freq_array = np.maximum(0.0, base_freq_array + half_beat_freq)

        # --- Corrected Phase Calculation using Cumsum ---
        # Calculate phase by integrating frequencies (cumulative sum)
        # delta_phase = 2 * pi * freq / sample_rate
        # phase = cumsum(delta_phase)
        phase_left = np.cumsum(2 * np.pi * left_freq_array / self.sample_rate)
        phase_right = np.cumsum(2 * np.pi * right_freq_array / self.sample_rate)
        # --- End Correction ---

        # Generate sine waves using the calculated phases
        s_left = np.sin(phase_left) * vol_left_array
        s_right = np.sin(phase_right) * vol_right_array

        # Combine into stereo output
        audio = np.column_stack((s_left, s_right)).astype(np.float32)

        # Optional: Apply fade (often not desired for pure binaural beats)
        # if 'apply_fade' in globals():
        #     audio = apply_fade(audio, fade_samples=min(1000, num_samples // 10))

        return audio# -----------------------------------------------------------
# Isochronic Voices (Unchanged from original, adapted for phase integration)
# -----------------------------------------------------------

def trapezoid_envelope_vectorized(t_in_cycle, cycle_len, ramp_percent, gap_percent):
    """Vectorized trapezoidal envelope generation."""
    env = np.zeros_like(t_in_cycle, dtype=np.float64)
    valid_mask = cycle_len > 0
    if not np.any(valid_mask): return env

    audible_len = (1.0 - gap_percent) * cycle_len
    ramp_total = np.clip(audible_len * ramp_percent * 2.0, 0.0, audible_len) # Ensure ramp doesn't exceed audible
    stable_len = audible_len - ramp_total
    ramp_up_len = ramp_total / 2.0
    stable_end = ramp_up_len + stable_len

    # Masks
    in_gap_mask = (t_in_cycle >= audible_len) & valid_mask
    ramp_up_mask = (t_in_cycle < ramp_up_len) & (~in_gap_mask) & valid_mask
    ramp_down_mask = (t_in_cycle >= stable_end) & (t_in_cycle < audible_len) & (~in_gap_mask) & valid_mask
    stable_mask = (t_in_cycle >= ramp_up_len) & (t_in_cycle < stable_end) & (~in_gap_mask) & valid_mask

    # Calculations with division checks
    with np.errstate(divide='ignore', invalid='ignore'):
        # Ramp up
        div_ramp_up = ramp_up_len[ramp_up_mask]
        ramp_up_vals = np.full_like(div_ramp_up, 0.0)
        valid_div_up = div_ramp_up > 0
        ramp_up_vals[valid_div_up] = t_in_cycle[ramp_up_mask][valid_div_up] / div_ramp_up[valid_div_up]
        env[ramp_up_mask] = np.nan_to_num(ramp_up_vals)

        # Stable
        env[stable_mask] = 1.0

        # Ramp down
        time_into_down = (t_in_cycle[ramp_down_mask] - stable_end[ramp_down_mask])
        down_len = ramp_up_len[ramp_down_mask] # Should be same as ramp up length
        ramp_down_vals = np.full_like(down_len, 0.0)
        valid_div_down = down_len > 0
        ramp_down_vals[valid_div_down] = 1.0 - (time_into_down[valid_div_down] / down_len[valid_div_down])
        env[ramp_down_mask] = np.nan_to_num(ramp_down_vals)

    return np.clip(env, 0.0, 1.0) # Ensure output is [0, 1]


class IsochronicVoice(Voice):
    """
    Isochronic tone generator with trapezoidal envelope.
    Uses cumulative sum for phase calculation.
    """
    def __init__(self, nodes: List['Node'], sample_rate: int = 44100, ramp_percent=0.2,
                 gap_percent=0.15, amplitude=1.0, **kwargs): # Added **kwargs
        super().__init__(nodes, sample_rate, **kwargs)
        self.ramp_percent = float(ramp_percent)
        self.gap_percent = float(gap_percent)
        self.amplitude = float(amplitude)

    def generate_samples(self) -> np.ndarray:
        param_arrays = self._get_param_arrays()
        # Ignore potential SAM params
        t_array, base_freq_array, beat_freq_array, vol_left_array, vol_right_array = param_arrays[:5]

        num_samples = t_array.size
        if num_samples == 0: return np.zeros((0, 2), dtype=np.float32)

        # --- Corrected Carrier Phase Calculation ---
        instantaneous_carrier_freq = np.maximum(0.0, base_freq_array)
        carrier_phase = np.cumsum(2 * np.pi * instantaneous_carrier_freq / self.sample_rate)
        carrier = np.sin(carrier_phase)
        # --- End Correction ---

        # --- Corrected Envelope Timing Calculation ---
        instantaneous_beat_freq = np.maximum(0.0, beat_freq_array)
        cycle_len_array = np.zeros_like(instantaneous_beat_freq)
        valid_beat_mask = instantaneous_beat_freq > 1e-9 # Use small threshold for valid frequency
        with np.errstate(divide='ignore'):
             cycle_len_array[valid_beat_mask] = 1.0 / instantaneous_beat_freq[valid_beat_mask]

        # Calculate cumulative phase in *cycles* (not radians) for modulo operation
        # delta_phase_cycles = freq / sample_rate
        # Cumulative sum gives total cycles elapsed
        beat_phase_cycles = np.cumsum(instantaneous_beat_freq / self.sample_rate)
        # Time within the current cycle (0 to cycle_len)
        t_in_cycle = np.mod(beat_phase_cycles, 1.0) * cycle_len_array
        # Ensure t_in_cycle is 0 where beat freq is invalid
        t_in_cycle[~valid_beat_mask] = 0.0
        # --- End Correction ---

        # Generate envelope using the trapezoid helper function
        # Make sure trapezoid_envelope_vectorized is defined in the file
        env = trapezoid_envelope_vectorized(
            t_in_cycle, cycle_len_array, self.ramp_percent, self.gap_percent
        )

        # Apply envelope and amplitude
        mono_signal = carrier * env * self.amplitude

        # Generate stereo output and apply fade
        audio = np.column_stack((mono_signal * vol_left_array,
                                 mono_signal * vol_right_array)).astype(np.float32)
        if 'apply_fade' in globals():
            audio = apply_fade(audio, fade_samples=min(100, num_samples // 20)) # Short fade for clicks

        return audio


class AltIsochronicVoice(Voice):
    """
    Alternating isochronic (switches channel each full cycle).
    Uses cumulative sum for phase calculation.
    """
    def __init__(self, nodes: List['Node'], sample_rate: int = 44100, ramp_percent=0.2,
                 gap_percent=0.15, amplitude=1.0, **kwargs): # Added **kwargs
        super().__init__(nodes, sample_rate, **kwargs)
        self.ramp_percent = float(ramp_percent)
        self.gap_percent = float(gap_percent)
        self.amplitude = float(amplitude)

    def generate_samples(self) -> np.ndarray:
        param_arrays = self._get_param_arrays()
        t_array, base_freq_array, beat_freq_array, vol_left_array, vol_right_array = param_arrays[:5]

        num_samples = t_array.size
        if num_samples == 0: return np.zeros((0, 2), dtype=np.float32)

        # --- Corrected Carrier Phase Calculation ---
        instantaneous_carrier_freq = np.maximum(0.0, base_freq_array)
        carrier_phase = np.cumsum(2 * np.pi * instantaneous_carrier_freq / self.sample_rate)
        carrier = np.sin(carrier_phase)
        # --- End Correction ---

        # --- Corrected Envelope Timing and Cycle Calculation ---
        instantaneous_beat_freq = np.maximum(0.0, beat_freq_array)
        cycle_len_array = np.zeros_like(instantaneous_beat_freq)
        valid_beat_mask = instantaneous_beat_freq > 1e-9
        with np.errstate(divide='ignore'):
             cycle_len_array[valid_beat_mask] = 1.0 / instantaneous_beat_freq[valid_beat_mask]

        # Calculate cumulative phase in cycles
        beat_phase_cycles = np.cumsum(instantaneous_beat_freq / self.sample_rate)
        # Time within the current cycle (0 to cycle_len)
        t_in_cycle = np.mod(beat_phase_cycles, 1.0) * cycle_len_array
        t_in_cycle[~valid_beat_mask] = 0.0
        # --- End Correction ---

        # Generate envelope (same as IsochronicVoice)
        env = trapezoid_envelope_vectorized(
            t_in_cycle, cycle_len_array, self.ramp_percent, self.gap_percent
        )

        mono_signal = carrier * env * self.amplitude

        # --- Determine channel based on cycle number ---
        # Integer part of cumulative cycles determines the cycle number
        cycle_number = np.floor(beat_phase_cycles).astype(int)
        # Mask for samples belonging to the left channel (even cycles 0, 2, 4...)
        is_left_channel = (cycle_number % 2 == 0)
        # Also ensure beat frequency is valid for channel assignment
        is_left_channel &= valid_beat_mask
        is_right_channel = (~is_left_channel) & valid_beat_mask
        # --- End Correction ---

        audio = np.zeros((num_samples, 2), dtype=np.float32)
        audio[is_left_channel, 0] = mono_signal[is_left_channel] * vol_left_array[is_left_channel]
        audio[is_right_channel, 1] = mono_signal[is_right_channel] * vol_right_array[is_right_channel]

        if 'apply_fade' in globals():
             audio = apply_fade(audio, fade_samples=min(100, num_samples // 20))

        return audio


class AltIsochronic2Voice(Voice):
    """
    Alternating isochronic (switches channel each half cycle).
    Uses cumulative sum for phase calculation.
    """
    def __init__(self, nodes: List['Node'], sample_rate: int = 44100, ramp_percent=0.2,
                 gap_percent=0.15, amplitude=1.0, **kwargs): # Added **kwargs
        super().__init__(nodes, sample_rate, **kwargs)
        self.ramp_percent = float(ramp_percent)
        self.gap_percent = float(gap_percent)
        self.amplitude = float(amplitude)

    def generate_samples(self) -> np.ndarray:
        param_arrays = self._get_param_arrays()
        t_array, base_freq_array, beat_freq_array, vol_left_array, vol_right_array = param_arrays[:5]

        num_samples = t_array.size
        if num_samples == 0: return np.zeros((0, 2), dtype=np.float32)

        # --- Corrected Carrier Phase Calculation ---
        instantaneous_carrier_freq = np.maximum(0.0, base_freq_array)
        carrier_phase = np.cumsum(2 * np.pi * instantaneous_carrier_freq / self.sample_rate)
        carrier = np.sin(carrier_phase)
        # --- End Correction ---

        # --- Corrected Envelope Timing Calculation (based on half-cycle) ---
        instantaneous_beat_freq = np.maximum(0.0, beat_freq_array)
        # Calculate half-cycle length
        half_cycle_len_array = np.zeros_like(instantaneous_beat_freq)
        valid_beat_mask = instantaneous_beat_freq > 1e-9
        with np.errstate(divide='ignore'):
            # half_cycle_len = 1.0 / (2.0 * beat_freq)
            half_cycle_len_array[valid_beat_mask] = 0.5 / instantaneous_beat_freq[valid_beat_mask]

        # Calculate cumulative phase in *half-cycles*
        # delta_phase_half_cycles = (freq / 2.0) / sample_rate
        beat_phase_half_cycles = np.cumsum(instantaneous_beat_freq / (2.0 * self.sample_rate))

        # Time within the current *half* cycle (0 to half_cycle_len)
        t_in_half_cycle = np.mod(beat_phase_half_cycles, 1.0) * half_cycle_len_array
        t_in_half_cycle[~valid_beat_mask] = 0.0
        # --- End Correction ---

        # Generate envelope based on half cycle length
        env = trapezoid_envelope_vectorized(
            t_in_half_cycle, half_cycle_len_array, self.ramp_percent, self.gap_percent
        )

        mono_signal = carrier * env * self.amplitude

        # --- Determine channel based on half-cycle number ---
        # Integer part of cumulative half-cycles determines the half-cycle number
        half_cycle_number = np.floor(beat_phase_half_cycles).astype(int)
        # First half-cycle (0, 2, 4...) -> Left channel
        is_first_half = (half_cycle_number % 2 == 0)
        is_left_channel = is_first_half & valid_beat_mask
        is_right_channel = (~is_first_half) & valid_beat_mask
        # --- End Correction ---

        audio = np.zeros((num_samples, 2), dtype=np.float32)
        audio[is_left_channel, 0] = mono_signal[is_left_channel] * vol_left_array[is_left_channel]
        audio[is_right_channel, 1] = mono_signal[is_right_channel] * vol_right_array[is_right_channel]

        if 'apply_fade' in globals():
            audio = apply_fade(audio, fade_samples=min(100, num_samples // 20))

        return audio
# -----------------------------------------------------------
# Noise and External Audio Voices (Unchanged from original)
# -----------------------------------------------------------

class PinkNoiseVoice(Voice):
    """Pink noise generator (1/f)."""
    def generate_samples(self):
        t_array, _, _, vol_left_array, vol_right_array = self._get_param_arrays()
        num_samples = t_array.size
        if num_samples == 0: return np.zeros((0, 2), dtype=np.float32)

        # Voss-McCartney algorithm approximation (summing octaves of white noise)
        # More computationally intensive but better approximation than simple filter
        num_octaves = 8 # Adjust for quality vs performance
        white_sources = [np.random.normal(0, 1, num_samples) for _ in range(num_octaves)]
        pink = np.zeros(num_samples, dtype=np.float32)
        total_amplitude = 0.0

        for i in range(num_octaves):
             amplitude = 1 / (2**i)
             total_amplitude += amplitude
             # Resample white noise source to simulate lower frequency octaves
             num_steps = 2**i
             indices = np.floor(np.arange(num_samples) / num_steps).astype(int) % num_samples # Wrap indices
             # Use advanced indexing carefully
             current_octave = np.take(white_sources[i], indices)
             pink += current_octave * amplitude

        # Normalize based on expected amplitude sum
        if total_amplitude > 0:
             pink /= total_amplitude

        # Apply low-pass filter to smooth highest frequencies slightly (optional)
        # b, a = scipy.signal.butter(1, 0.9) # Simple 1st order butterworth
        # pink = scipy.signal.lfilter(b, a, pink)

        scale = 0.15 # Adjust overall loudness

        audio = np.zeros((num_samples, 2), dtype=np.float32)
        audio[:, 0] = pink * vol_left_array * scale
        audio[:, 1] = pink * vol_right_array * scale

        # Apply fade to prevent clicks
        audio = apply_fade(audio, fade_samples=min(1000, num_samples // 10))

        return audio


class ExternalAudioVoice(Voice):
    """Loads external audio file and applies volume envelopes."""
    def __init__(self, nodes, file_path, sample_rate=44100):
        # Determine SR first based on SLAB availability
        target_sr = DEFAULT_HRTF_SR if SLAB_AVAILABLE else sample_rate
        if SLAB_AVAILABLE and sample_rate != target_sr:
             print(f"ExternalAudioVoice: Forcing sample rate to KEMAR default {target_sr}Hz.")

        super().__init__(nodes, target_sr) # Use the determined SR
        self.file_path = file_path
        self.ext_audio = np.zeros((2, 1)) # Placeholder
        self.ext_length = 1

        try:
            print(f"ExternalAudioVoice: Loading '{self.file_path}' at SR={self.sample_rate}...")
            # Load the audio file at the voice's sample rate
            data, sr = librosa.load(self.file_path, sr=self.sample_rate, mono=False)

            if data.ndim == 1:
                data = np.vstack((data, data)) # Convert mono to stereo
            elif data.ndim > 2:
                 print(f"ExternalAudioVoice Warning: Loaded audio has unexpected shape {data.shape}. Using first two channels.")
                 data = data[:2,:]
            elif data.shape[0] > 2 :
                 print(f"ExternalAudioVoice Warning: Loaded audio has >2 channels ({data.shape[0]}). Using first two channels.")
                 data = data[:2,:]
            elif data.shape[0] == 1:
                 print(f"ExternalAudioVoice: Loaded audio is mono, duplicating channel.")
                 data = np.vstack((data, data))


            self.ext_audio = data.astype(np.float32)
            self.ext_length = self.ext_audio.shape[1]
            print(f"ExternalAudioVoice: Loaded successfully, shape={self.ext_audio.shape}.")
        except Exception as e:
            print(f"ExternalAudioVoice Error loading '{self.file_path}': {e}")
            # Keep placeholder audio

    def generate_samples(self):
        t_array, _, _, vol_left_array, vol_right_array = self._get_param_arrays()
        num_samples = t_array.size
        if num_samples == 0 or self.ext_length <= 1:
            return np.zeros((0, 2), dtype=np.float32)

        audio = np.zeros((num_samples, 2), dtype=np.float32)
        idx_array = np.mod(np.arange(num_samples), self.ext_length)

        # Extract samples and apply volume
        audio[:, 0] = self.ext_audio[0][idx_array] * vol_left_array
        audio[:, 1] = self.ext_audio[1][idx_array] * vol_right_array

        return audio

# -----------------------------------------------------------
# Preset Generators (Unchanged Logic, using updated Node)
# -----------------------------------------------------------

def get_preset_nodes_for_state(state: BrainwaveState, duration: float = 300.0) -> List[Node]:
    presets = {
        BrainwaveState.DELTA: {'base_freq': 100.0, 'beat_freq': 2.0, 'volume': 0.8},
        BrainwaveState.THETA: {'base_freq': 200.0, 'beat_freq': 5.0, 'volume': 0.8},
        BrainwaveState.ALPHA: {'base_freq': 440.0, 'beat_freq': 10.0, 'volume': 0.8},
        BrainwaveState.BETA: {'base_freq': 528.0, 'beat_freq': 20.0, 'volume': 0.7},
        BrainwaveState.GAMMA: {'base_freq': 528.0, 'beat_freq': 40.0, 'volume': 0.7}
    }
    preset = presets[state]
    return [Node(duration=duration, base_freq=preset['base_freq'], beat_freq=preset['beat_freq'],
                 volume_left=preset['volume'], volume_right=preset['volume'])]

def create_meditation_session_nodes(total_duration: float = 1200.0) -> List[Node]:
    beta_time, t1, alpha_time, t2, theta_time, t3, alpha_end = (
        total_duration * p for p in [0.1, 0.05, 0.2, 0.05, 0.4, 0.05, 0.15] )
    return [
        Node(beta_time, 528.0, 20.0, 0.7, 0.7), Node(t1, 484.0, 15.0, 0.75, 0.75),
        Node(alpha_time, 440.0, 10.0, 0.8, 0.8), Node(t2, 320.0, 7.5, 0.85, 0.85),
        Node(theta_time, 200.0, 5.0, 0.9, 0.9), Node(t3, 320.0, 7.5, 0.85, 0.85),
        Node(alpha_end, 440.0, 10.0, 0.8, 0.8) ]

def create_sleep_session_nodes(total_duration: float = 1800.0) -> List[Node]:
    alpha_time, t1, theta_time, t2, delta_time = (
        total_duration * p for p in [0.15, 0.05, 0.2, 0.05, 0.55] )
    return [
        Node(alpha_time, 440.0, 10.0, 0.8, 0.8), Node(t1, 320.0, 7.5, 0.85, 0.85),
        Node(theta_time, 200.0, 5.0, 0.9, 0.9), Node(t2, 150.0, 3.5, 0.9, 0.9),
        Node(delta_time, 100.0, 2.0, 0.9, 0.9) ]

def create_focus_session_nodes(total_duration: float = 900.0) -> List[Node]:
    alpha_time, t1, beta_time, _ = ( total_duration * p for p in [0.2, 0.05, 0.7, 0.05] )
    return [
        Node(alpha_time, 440.0, 10.0, 0.8, 0.8), Node(t1, 484.0, 15.0, 0.75, 0.75),
        Node(beta_time, 528.0, 20.0, 0.7, 0.7) ]

# -----------------------------------------------------------
# Track Generation and Export (Unchanged Logic)
# -----------------------------------------------------------

def generate_track_audio(voices: List[Voice], sample_rate: Optional[int] = None) -> np.ndarray:
    """
    Mix multiple voices together. Uses the sample rate of the first voice
    if not specified, or the forced KEMAR rate if SLAB is used.
    """
    if not voices:
        return np.zeros((0, 2), dtype=np.float32)

    # Determine target sample rate - use first voice's SR
    target_sr = voices[0].sample_rate
    if sample_rate is not None and sample_rate != target_sr:
         print(f"Warning: Specified sample_rate {sample_rate} differs from first voice's rate {target_sr}. Using voice rate.")
         # Or should we resample everything? For now, use voice rate.

    # Find maximum track length based on voices' durations
    track_length_seconds = 0
    for v in voices:
        if v.sample_rate != target_sr:
             print(f"Warning: Voice {type(v).__name__} has different SR ({v.sample_rate}) from target ({target_sr}). Mixing may be incorrect without resampling.")
             # Ideally, all voices should be initialized with the same target SR.
        track_length_seconds = max(track_length_seconds, v.total_duration)

    total_samples = int(track_length_seconds * target_sr)
    mixed = np.zeros((total_samples, 2), dtype=np.float32)
    print(f"Generating track: {track_length_seconds:.2f}s, {total_samples} samples, SR={target_sr}Hz")

    for i, v in enumerate(voices):
        print(f"Mixing voice {i+1}/{len(voices)}: {type(v).__name__}...")
        try:
            # Ensure voice generates at the target SR (should happen at init)
            if v.sample_rate != target_sr:
                 print(f"  SKIPPING voice {type(v).__name__} due to sample rate mismatch.")
                 continue # Skip if SR doesn't match, proper fix is init consistency

            buf = v.generate_samples()
            length = buf.shape[0]
            if length > 0:
                # Add voice output, ensuring length doesn't exceed mix buffer
                add_len = min(length, total_samples)
                mixed[:add_len] += buf[:add_len] # Add only valid part
        except Exception as e:
            print(f"  Error generating/mixing voice {type(v).__name__}: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            continue

    # Normalize final mix to prevent clipping
    max_val = np.max(np.abs(mixed))
    if max_val == 0:
         print("Warning: Final mix is silent.")
    elif max_val > 1.0:
        print(f"Normalizing final mix (max val: {max_val:.3f})")
        mixed /= max_val
    else:
        print(f"Final mix peak: {max_val:.3f}")


    return mixed

def export_wav(audio_data, sample_rate, file_path):
    if audio_data is None or audio_data.size == 0:
        print(f"Error: No audio data to export to {file_path}"); return False
    max_amp = np.max(np.abs(audio_data)) if audio_data.size > 0 else 0
    print(f"Exporting WAV: {file_path}, SR={sample_rate}, Samples={audio_data.shape[0]}, Peak={max_amp:.4f}")
    try:
        sf.write(file_path, audio_data, int(sample_rate), format='WAV', subtype='PCM_16') # Use 16-bit PCM
        print(f"Successfully wrote WAV: {file_path}"); return True
    except Exception as e: print(f"Error writing WAV: {e}"); return False

def export_flac(audio_data, sample_rate, file_path):
    if audio_data is None or audio_data.size == 0:
        print(f"Error: No audio data to export to {file_path}"); return False
    max_amp = np.max(np.abs(audio_data)) if audio_data.size > 0 else 0
    print(f"Exporting FLAC: {file_path}, SR={sample_rate}, Samples={audio_data.shape[0]}, Peak={max_amp:.4f}")
    try:
        sf.write(file_path, audio_data, int(sample_rate), format='FLAC') # Default subtype is usually 16-bit
        print(f"Successfully wrote FLAC: {file_path}"); return True
    except Exception as e: print(f"Error writing FLAC: {e}"); return False

def export_mp3(audio_data, sample_rate, file_path, bitrate="192k"):
    try: from pydub import AudioSegment
    except ImportError: print("Error: pydub not found (pip install pydub). Ensure ffmpeg/libav installed."); return False
    if audio_data is None or audio_data.size == 0: print(f"Error: No audio data for MP3: {file_path}"); return False

    max_amp = np.max(np.abs(audio_data)) if audio_data.size > 0 else 0
    print(f"Exporting MP3: {file_path}, SR={sample_rate}, Samples={audio_data.shape[0]}, Peak={max_amp:.4f}, Bitrate={bitrate}")
    try:
        int_data = np.clip(audio_data * 32767, -32767, 32767).astype(np.int16)
        seg = AudioSegment(int_data.tobytes(), frame_rate=int(sample_rate), sample_width=2, channels=2)
        seg.export(file_path, format="mp3", bitrate=bitrate)
        print(f"Successfully wrote MP3: {file_path}"); return True
    except Exception as e: print(f"Error writing MP3: {e}"); return False


# -----------------------------------------------------------
# Example Usage
# -----------------------------------------------------------
if __name__ == "__main__":

    # --- Configuration ---
    OUTPUT_FILENAME_BASE = "audio_engine_test"
    SESSION_TYPE = "SAM_EFFECT_TEST" # Options: "MEDITATION", "SLEEP", "FOCUS", "PRESET", "SAM_TEST", "SAM_EFFECT_TEST", "EXTERNAL_TEST"
    TARGET_STATE = BrainwaveState.ALPHA # Used only if SESSION_TYPE == "PRESET"
    TOTAL_DURATION = 20 # seconds
    EXPORT_FORMATS = ["WAV", "MP3"] # "WAV", "FLAC", "MP3"

    # --- Determine Sample Rate ---
    # Use KEMAR default SR if SLAB is available, otherwise 44100
    master_sample_rate = DEFAULT_HRTF_SR if SLAB_AVAILABLE else 44100
    print(f"\n--- Master Sample Rate set to: {master_sample_rate} Hz ---\n")

    # --- Create Voices ---
    voices_to_mix: List[Voice] = []

    if SESSION_TYPE == "MEDITATION":
        print("Creating Meditation Session...")
        nodes = create_meditation_session_nodes(TOTAL_DURATION)
        voices_to_mix.append(BinauralBeatVoice(nodes, sample_rate=master_sample_rate))
        # Add background noise
        noise_nodes = [Node(TOTAL_DURATION, 0, 0, 0.1, 0.1)] # Low volume pink noise
        voices_to_mix.append(PinkNoiseVoice(noise_nodes, sample_rate=master_sample_rate))

    elif SESSION_TYPE == "SLEEP":
        print("Creating Sleep Session...")
        nodes = create_sleep_session_nodes(TOTAL_DURATION)
        voices_to_mix.append(IsochronicVoice(nodes, sample_rate=master_sample_rate, amplitude=0.7))

    elif SESSION_TYPE == "FOCUS":
        print("Creating Focus Session...")
        nodes = create_focus_session_nodes(TOTAL_DURATION)
        voices_to_mix.append(MonauralBeatVoice(nodes, sample_rate=master_sample_rate, modulation_depth=0.7))

    elif SESSION_TYPE == "PRESET":
        print(f"Creating Preset Session for {TARGET_STATE.name}...")
        nodes = get_preset_nodes_for_state(TARGET_STATE, TOTAL_DURATION)
        voices_to_mix.append(BinauralBeatVoice(nodes, sample_rate=master_sample_rate))

    elif SESSION_TYPE == "SAM_TEST":
        if SLAB_AVAILABLE:
            print("Creating SAMVoice Test Session (Circular Path)...")
            # Nodes define base freq, modulation speed (beat_freq), and volume over time
            sam_nodes = [
                Node(duration=TOTAL_DURATION/2, base_freq=200, beat_freq=0.2, volume_left=0.8, volume_right=0.8), # Slow circle
                Node(duration=TOTAL_DURATION/2, base_freq=300, beat_freq=0.8, volume_left=0.6, volume_right=0.6), # Faster circle, lower volume
            ]
            voices_to_mix.append(SAMVoice(
                nodes=sam_nodes,
                sample_rate=master_sample_rate, # SR is forced anyway
                path_shape='circle'
                # Other SAM params use defaults
            ))
        else:
            print("Cannot run SAM_TEST: SLAB library not available.")

    elif SESSION_TYPE == "SAM_EFFECT_TEST":
         if SLAB_AVAILABLE:
            print("Creating SAMEffectVoice Test Session (Oscillating Arc on Binaural Beats)...")
            # 1. Create the source voice (Binaural Beats)
            source_nodes = get_preset_nodes_for_state(BrainwaveState.ALPHA, TOTAL_DURATION)
            source_voice = BinauralBeatVoice(source_nodes, sample_rate=master_sample_rate)

            # 2. Create nodes for the SAM *effect* (controls effect speed and volume)
            effect_nodes = [
                Node(duration=TOTAL_DURATION, base_freq=0, beat_freq=0.3, volume_left=1.0, volume_right=1.0), # Constant slow sweep, full volume applied by effect
            ]

            # 3. Create the SAMEffectVoice, passing the source voice
            sam_effect = SAMEffectVoice(
                nodes=effect_nodes,
                source_voice=source_voice,
                sample_rate=master_sample_rate, # SR forced anyway
                path_shape='oscillating_arc',
                arc_start_deg=-80,
                arc_end_deg=80,
                frame_dur_ms=50,
                overlap_factor=4
            )
            voices_to_mix.append(sam_effect)

            # Optional: Add background noise separately
            # noise_nodes = [Node(TOTAL_DURATION, 0, 0, 0.08, 0.08)]
            # voices_to_mix.append(PinkNoiseVoice(noise_nodes, sample_rate=master_sample_rate))

         else:
            print("Cannot run SAM_EFFECT_TEST: SLAB library not available.")


    elif SESSION_TYPE == "EXTERNAL_TEST":
        print("Creating External Audio Test Session...")
        # Ensure you have an audio file named "background.wav" or change the path
        external_file = "background.wav"
        if not os.path.exists(external_file):
             print(f"Warning: External audio file '{external_file}' not found. Skipping ExternalAudioVoice.")
             # Add a placeholder voice if needed
             nodes = get_preset_nodes_for_state(BrainwaveState.THETA, TOTAL_DURATION)
             voices_to_mix.append(BinauralBeatVoice(nodes, sample_rate=master_sample_rate))
        else:
             # Add binaural beats
             nodes_bb = get_preset_nodes_for_state(BrainwaveState.THETA, TOTAL_DURATION)
             voices_to_mix.append(BinauralBeatVoice(nodes_bb, sample_rate=master_sample_rate))
             # Add external audio with volume envelope
             nodes_ext = [
                 Node(duration=TOTAL_DURATION*0.1, base_freq=0, beat_freq=0, volume_left=0.0, volume_right=0.0), # Fade in
                 Node(duration=TOTAL_DURATION*0.8, base_freq=0, beat_freq=0, volume_left=0.3, volume_right=0.3), # Play at low volume
                 Node(duration=TOTAL_DURATION*0.1, base_freq=0, beat_freq=0, volume_left=0.0, volume_right=0.0), # Fade out
             ]
             voices_to_mix.append(ExternalAudioVoice(nodes_ext, external_file, sample_rate=master_sample_rate))

    else:
        print(f"Unknown SESSION_TYPE: {SESSION_TYPE}")

    # --- Generate and Export ---
    if voices_to_mix:
        print("\nGenerating final track...")
        final_audio = generate_track_audio(voices_to_mix, sample_rate=master_sample_rate)

        if final_audio.size > 0:
            print("\nExporting audio...")
            timestamp = int(time.time())
            if "WAV" in EXPORT_FORMATS:
                export_wav(final_audio, master_sample_rate, f"{OUTPUT_FILENAME_BASE}_{SESSION_TYPE.lower()}_{timestamp}.wav")
            if "FLAC" in EXPORT_FORMATS:
                export_flac(final_audio, master_sample_rate, f"{OUTPUT_FILENAME_BASE}_{SESSION_TYPE.lower()}_{timestamp}.flac")
            if "MP3" in EXPORT_FORMATS:
                export_mp3(final_audio, master_sample_rate, f"{OUTPUT_FILENAME_BASE}_{SESSION_TYPE.lower()}_{timestamp}.mp3")
            print("\nDone.")
        else:
            print("\nSkipping export: No audio was generated.")
    else:
        print("\nNo voices were created for the selected session type.")
