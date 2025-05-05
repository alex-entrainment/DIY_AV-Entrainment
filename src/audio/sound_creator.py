import numpy as np
from scipy.signal import butter, lfilter
from scipy.io.wavfile import write
import soundfile as sf # FLAC conversion
from pydub import AudioSegment # MP3 conversion
import math
import json
import inspect # Needed to inspect function parameters for GUI
import os # Needed for path checks in main example
import traceback # For detailed error printing

# Placeholder for the missing audio_engine module
# If you have the 'audio_engine.py' file, place it in the same directory.
# Otherwise, the SAM functions will not work.
try:
    # Attempt to import the real audio_engine if available
    from audio_engine import Node, SAMVoice, VALID_SAM_PATHS
    AUDIO_ENGINE_AVAILABLE = True
    print("INFO: audio_engine module loaded successfully.")
except ImportError:
    AUDIO_ENGINE_AVAILABLE = False
    print("WARNING: audio_engine module not found. Spatial Angle Modulation (SAM) functions will not be available.")
    # Define dummy classes/variables if audio_engine is missing
    class Node:
        def __init__(self, *args, **kwargs):
            # print("WARNING: Using dummy Node class. SAM functionality disabled.")
            # Store args needed for generate_samples duration calculation
            # Simplified: Just store duration if provided
            self.duration = args[0] if args else kwargs.get('duration', 0)
            pass
    class SAMVoice:
        def __init__(self, *args, **kwargs):
            # print("WARNING: Using dummy SAMVoice class. SAM functionality disabled.")
            # Store args needed for generate_samples duration calculation
            self._nodes = kwargs.get('nodes', [])
            self._sample_rate = kwargs.get('sample_rate', 44100)
            pass
        def generate_samples(self):
            print("WARNING: SAM generate_samples called on dummy class. Returning silence.")
            # Calculate duration from stored nodes
            duration = 0
            if hasattr(self, '_nodes'):
                # Access duration attribute correctly from dummy Node
                duration = sum(node.duration for node in self._nodes if hasattr(node, 'duration'))
            sample_rate = getattr(self, '_sample_rate', 44100)
            N = int(duration * sample_rate) if duration > 0 else int(1.0 * sample_rate) # Default 1 sec if no duration found
            return np.zeros((N, 2))

    VALID_SAM_PATHS = ['circle', 'line', 'lissajous', 'figure_eight', 'arc'] # Example paths

# -----------------------------------------------------------------------------
# Helper functions (Copied from original sc.txt, minor adjustments if needed)
# -----------------------------------------------------------------------------

def sine_wave(freq, t, phase: float=0):
    """Generate a sine wave at a constant frequency using an absolute time base."""
    freq = float(freq)
    phase = float(phase)
    # Ensure freq is not zero or negative if used in log etc later, though not here.
    freq = np.maximum(freq, 1e-9) # Avoid potential issues with zero freq
    return np.sin(2 * np.pi * freq * t + phase)

def sine_wave_varying(freq_array, t, sample_rate=44100):
    """
    Generate a sine wave whose instantaneous frequency is defined by freq_array.
    The phase is computed as the cumulative sum (integral) of the frequency
    over the absolute time vector t. An initial phase is added so that for constant
    frequency the result is sin(2π f t).
    """
    # Ensure freq_array does not contain non-positive values before cumsum
    freq_array = np.maximum(np.asarray(freq_array, dtype=float), 1e-9)
    t = np.asarray(t, dtype=float)
    sample_rate = float(sample_rate)

    if len(t) <= 1:
        return np.zeros_like(t) # Handle edge case of single point or empty array

    # Calculate dt, ensuring it has the same length as t and freq_array
    dt = np.diff(t, prepend=t[0]) # Use first element to calculate first dt assuming t[0] is start time
    # Alternative if t doesn't start at 0: dt = np.diff(t, prepend=t[0] - (t[1]-t[0]) if len(t)>1 else 0)

    # Add an initial phase corresponding to integration from t=0 to t[0]
    initial_phase = 2 * np.pi * freq_array[0] * t[0] if len(t) > 0 else 0
    # Calculate phase using cumulative sum
    phase = initial_phase + 2 * np.pi * np.cumsum(freq_array * dt)
    return np.sin(phase)

def adsr_envelope(t, attack=0.01, decay=0.1, sustain_level=0.8, release=0.1):
    """
    Generate an ADSR envelope over t. Assumes that t starts at 0.
    """
    t = np.asarray(t, dtype=float)
    if len(t) <= 1: return np.zeros_like(t) # Handle edge case

    duration = t[-1] - t[0] + (t[1]-t[0]) if len(t) > 1 else 0
    sr = len(t) / duration if duration > 0 else 44100
    total_samples = len(t)

    # Ensure non-negative durations and valid sustain level
    attack = max(0, float(attack))
    decay = max(0, float(decay))
    release = max(0, float(release))
    sustain_level = np.clip(float(sustain_level), 0, 1)

    attack_samples = int(attack * sr)
    decay_samples = int(decay * sr)
    release_samples = int(release * sr)

    # Ensure samples are not negative
    attack_samples = max(0, attack_samples)
    decay_samples = max(0, decay_samples)
    release_samples = max(0, release_samples)

    # Adjust if total duration is less than ADSR times
    total_adsr_samples = attack_samples + decay_samples + release_samples
    if total_adsr_samples > total_samples and total_adsr_samples > 0:
        scale_factor = total_samples / total_adsr_samples
        attack_samples = int(attack_samples * scale_factor)
        decay_samples = int(decay_samples * scale_factor)
        # Ensure release always has at least one sample if requested > 0 and space allows
        release_samples = total_samples - attack_samples - decay_samples
        release_samples = max(0, release_samples)


    sustain_samples = total_samples - (attack_samples + decay_samples + release_samples)
    sustain_samples = max(0, sustain_samples) # Ensure sustain is not negative

    # Generate curves
    attack_curve = np.linspace(0, 1, attack_samples, endpoint=False) if attack_samples > 0 else np.array([])
    decay_curve = np.linspace(1, sustain_level, decay_samples, endpoint=False) if decay_samples > 0 else np.array([])
    sustain_curve = np.full(sustain_samples, sustain_level) if sustain_samples > 0 else np.array([])
    # Ensure release ends exactly at 0
    release_curve = np.linspace(sustain_level, 0, release_samples, endpoint=True) if release_samples > 0 else np.array([])

    # Concatenate, ensuring correct length
    env = np.concatenate([attack_curve, decay_curve, sustain_curve, release_curve])

    # Pad or truncate to ensure exact length
    if len(env) < total_samples:
        # Pad with the end value of the release curve (should be 0)
        pad_value = 0.0 if release_samples > 0 else sustain_level # Pad with sustain if no release
        env = np.pad(env, (0, total_samples - len(env)), mode='constant', constant_values=pad_value)
    elif len(env) > total_samples:
        env = env[:total_samples]

    return env


def create_linear_fade_envelope(total_duration, sample_rate, fade_duration, start_amp, end_amp, fade_type='in'):
    """
    Create a linear fade envelope over a total duration.
    Handles fade_type 'in' (start) and 'out' (end).
    """
    total_duration = float(total_duration)
    sample_rate = int(sample_rate)
    fade_duration = float(fade_duration)
    start_amp = float(start_amp)
    end_amp = float(end_amp)

    total_samples = int(total_duration * sample_rate)
    fade_duration_samples = int(fade_duration * sample_rate)

    if total_samples <= 0: return np.array([])

    # Ensure fade duration isn't longer than total duration
    fade_samples = min(fade_duration_samples, total_samples)
    fade_samples = max(0, fade_samples) # Ensure non-negative

    if fade_samples <= 0: # No fade needed or possible
        # If fade in, hold end_amp; if fade out, hold start_amp
        hold_value = end_amp if fade_type == 'in' else start_amp
        return np.full(total_samples, hold_value)

    envelope = np.ones(total_samples) # Initialize

    if fade_type == 'in':
        # Generate ramp from start_amp to end_amp over fade_samples
        ramp = np.linspace(start_amp, end_amp, fade_samples, endpoint=True)
        envelope[:fade_samples] = ramp
        # Hold the end amplitude for the rest of the duration
        envelope[fade_samples:] = end_amp
    elif fade_type == 'out':
        sustain_samples = total_samples - fade_samples
        # Hold the start amplitude until the fade begins
        envelope[:sustain_samples] = start_amp
        # Generate ramp from start_amp to end_amp over fade_samples at the end
        ramp = np.linspace(start_amp, end_amp, fade_samples, endpoint=True)
        envelope[sustain_samples:] = ramp
    else:
        raise ValueError("fade_type must be 'in' or 'out'")

    return envelope

def linen_envelope(t, attack=0.1, release=0.1):
    """
    Generate a linear envelope that ramps up then down across t.
    Assumes t is relative (starts at 0).
    """
    t = np.asarray(t, dtype=float)
    if len(t) <= 1: return np.zeros_like(t) # Handle edge case

    duration = t[-1] - t[0] + (t[1]-t[0]) if len(t) > 1 else 0
    sr = len(t) / duration if duration > 0 else 44100
    total_samples = len(t)

    # Ensure non-negative durations
    attack = max(0, float(attack))
    release = max(0, float(release))

    attack_samples = int(attack * sr)
    release_samples = int(release * sr)

    # Ensure samples are not negative
    attack_samples = max(0, attack_samples)
    release_samples = max(0, release_samples)

    # Adjust if total duration is less than attack + release
    total_ar_samples = attack_samples + release_samples
    if total_ar_samples > total_samples and total_ar_samples > 0:
        scale_factor = total_samples / total_ar_samples
        attack_samples = int(attack_samples * scale_factor)
        # Ensure release gets remaining samples, minimum 0
        release_samples = max(0, total_samples - attack_samples)

    sustain_samples = total_samples - (attack_samples + release_samples)
    sustain_samples = max(0, sustain_samples)

    # Generate curves
    attack_curve = np.linspace(0, 1, attack_samples, endpoint=False) if attack_samples > 0 else np.array([])
    sustain_curve = np.ones(sustain_samples) if sustain_samples > 0 else np.array([])
    # Ensure release ends exactly at 0
    release_curve = np.linspace(1, 0, release_samples, endpoint=True) if release_samples > 0 else np.array([])

    # Concatenate
    env = np.concatenate([attack_curve, sustain_curve, release_curve])

    # Pad or truncate
    if len(env) < total_samples:
        # Pad with end value (should be 0 if release exists)
        pad_value = 0.0 if release_samples > 0 else 1.0
        env = np.pad(env, (0, total_samples - len(env)), mode='constant', constant_values=pad_value)
    elif len(env) > total_samples:
        env = env[:total_samples]

    return env


def pan2(signal, pan=0):
    """
    Pan a mono signal into stereo.
    pan: -1 => full left, 0 => center, 1 => full right.
    Uses simple sine/cosine panning (equal power).
    """
    pan = float(pan)
    pan = np.clip(pan, -1, 1) # Ensure pan is within range
    # Map pan [-1, 1] to angle [0, pi/2]
    angle = (pan + 1) * np.pi / 4
    left_gain = np.cos(angle)
    right_gain = np.sin(angle)
    left = left_gain * signal
    right = right_gain * signal
    # Ensure output is 2D array (N, 2)
    if signal.ndim == 1:
        return np.vstack([left, right]).T
    elif signal.ndim == 2 and signal.shape[1] == 1: # If input is already (N, 1)
        return np.hstack([left, right])
    else: # If input is already stereo, apply gains? Or assume mono input?
        # Assuming mono input is intended based on original code.
        # If stereo input needs panning, logic would differ.
        print("Warning: pan2 received unexpected input shape. Assuming mono.")
        # Attempt to take the first channel if stereo input received unexpectedly
        if signal.ndim == 2 and signal.shape[1] > 1:
            signal = signal[:, 0]
            left = left_gain * signal
            right = right_gain * signal
        return np.vstack([left, right]).T


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    # Ensure lowcut < highcut and both are positive and less than Nyquist
    lowcut = max(1e-6, float(lowcut)) # Avoid zero/negative freq, ensure float
    highcut = max(lowcut + 1e-6, float(highcut)) # Ensure high > low, ensure float
    # Clip frequencies to be slightly below Nyquist
    lowcut = min(lowcut, nyq * 0.9999)
    highcut = min(highcut, nyq * 0.9999)

    low = lowcut / nyq
    high = highcut / nyq
    # Ensure low < high after normalization, prevent edge cases
    if low >= high:
        low = high * 0.999 # Adjust if rounding caused issues
    low = max(1e-9, low) # Ensure > 0
    high = min(1.0 - 1e-9, high) # Ensure < 1

    # Check for invalid frequency range after clipping
    if low >= high:
         raise ValueError(f"Bandpass filter frequency range is invalid after Nyquist clipping: low={low}, high={high} (normalized)")


    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, center, Q, fs):
    """
    Apply a bandpass filter to the data. Q = center/bandwidth.
    """
    center = float(center)
    Q = float(Q)
    fs = float(fs)
    if Q <= 0: Q = 0.1 # Avoid zero or negative Q
    if center <= 0: center = 1.0 # Avoid zero or negative center frequency

    bandwidth = center / Q
    lowcut = center - bandwidth / 2
    highcut = center + bandwidth / 2
    # Ensure lowcut is positive
    lowcut = max(1e-6, lowcut)
    if lowcut >= highcut: # Handle cases where bandwidth is too large or center too low
        # Recalculate highcut based on positive lowcut and Q factor
        highcut = lowcut * (1 + 1/Q) # Simple approximation
        highcut = max(highcut, lowcut + 1e-6) # Ensure distinct

    try:
        b, a = butter_bandpass(lowcut, highcut, fs, order=2) # Lower order often more stable
        return lfilter(b, a, data)
    except ValueError as e:
        print(f"Error in bandpass filter calculation: {e}")
        print(f"  center={center}, Q={Q}, fs={fs} -> lowcut={lowcut}, highcut={highcut}")
        # Return original data or zeros? Returning original for now.
        return data


def butter_bandstop(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    # Input validation similar to bandpass
    lowcut = max(1e-6, float(lowcut))
    highcut = max(lowcut + 1e-6, float(highcut))
    lowcut = min(lowcut, nyq * 0.9999)
    highcut = min(highcut, nyq * 0.9999)

    low = lowcut / nyq
    high = highcut / nyq
    if low >= high: low = high * 0.999
    low = max(1e-9, low)
    high = min(1.0 - 1e-9, high)

    if low >= high:
         raise ValueError(f"Bandstop filter frequency range is invalid after Nyquist clipping: low={low}, high={high} (normalized)")

    b, a = butter(order, [low, high], btype='bandstop')
    return b, a

def bandreject_filter(data, center, Q, fs):
    """
    Apply a band-reject (notch) filter.
    """
    center = float(center)
    Q = float(Q)
    fs = float(fs)
    if Q <= 0: Q = 0.1
    if center <= 0: center = 1.0

    bandwidth = center / Q
    lowcut = center - bandwidth / 2
    highcut = center + bandwidth / 2
    lowcut = max(1e-6, lowcut)
    if lowcut >= highcut:
        highcut = lowcut * (1 + 1/Q)
        highcut = max(highcut, lowcut + 1e-6)

    try:
        b, a = butter_bandstop(lowcut, highcut, fs, order=2)
        return lfilter(b, a, data)
    except ValueError as e:
        print(f"Error in bandreject filter calculation: {e}")
        print(f"  center={center}, Q={Q}, fs={fs} -> lowcut={lowcut}, highcut={highcut}")
        return data


def lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    # Ensure cutoff is positive and below Nyquist
    cutoff = max(1e-6, float(cutoff))
    cutoff = min(cutoff, nyq * 0.9999)
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

def pink_noise(n):
    """
    Generate approximate pink noise using filtering method (more common).
    Filters white noise with a filter whose magnitude response is proportional to 1/sqrt(f).
    """
    n = int(n)
    if n <= 0: return np.array([])
    # Voss-McCartney algorithm approximation (simpler than FFT filtering)
    # Or use a library if available: pip install colorednoise
    try:
        import colorednoise as cn
        # Generate noise with exponent 1 for pink noise
        pink = cn.powerlaw_psd_gaussian(1, n)
        # Normalize to approx [-1, 1] range (optional, depends on desired amplitude)
        max_abs = np.max(np.abs(pink))
        if max_abs > 1e-9:
           pink = pink / max_abs
        return pink
    except ImportError:
        # print("WARNING: 'colorednoise' library not found. Using FFT filtering for pink noise.")
        # FFT filtering method
        white_noise = np.random.randn(n)
        fft_white = np.fft.rfft(white_noise)
        frequencies = np.fft.rfftfreq(n, d=1.0/44100) # Use sample rate if known, else d=1.0

        # Create filter: amplitude proportional to 1/sqrt(f)
        # Handle f=0 case (DC component)
        scale = np.ones_like(frequencies)
        non_zero_freq_indices = frequencies > 0
        # Avoid division by zero or sqrt of zero
        valid_freqs = frequencies[non_zero_freq_indices]
        scale[non_zero_freq_indices] = 1.0 / np.sqrt(np.maximum(valid_freqs, 1e-9)) # Ensure positive sqrt argument
        # Optional: Set DC component to 0 or a small value
        scale[0] = 0 # Typically remove DC

        # Apply filter in frequency domain
        pink_spectrum = fft_white * scale

        # Inverse FFT to get time domain signal
        pink = np.fft.irfft(pink_spectrum, n=n)

        # Normalize
        max_abs = np.max(np.abs(pink))
        if max_abs > 1e-9:
           pink = pink / max_abs
        return pink


def brown_noise(n):
    """
    Generate brown (red) noise by cumulatively summing white noise.
    """
    n = int(n)
    if n <= 0: return np.array([])
    wn = np.random.randn(n)
    brown = np.cumsum(wn)
    # Normalize to prevent excessive drift/amplitude
    max_abs = np.max(np.abs(brown))
    return brown / max_abs if max_abs > 1e-9 else brown

# --- NEW: Helper for Isochronic Tones (Copied from audio_engine.py) ---
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


# --- NEW: Helper function for Flanger (adapted from flanger.py) ---

def _flanger_effect_stereo_continuous(
        x,
        sr=44100,
        max_delay_ms=15.0, # Adjusted default
        min_delay_ms=1.0,  # Adjusted default
        rate_hz=0.2,       # Adjusted default for noticeable continuous sweep
        lfo_start_phase_rad=0.0, # New parameter
        skip_below_freq=30.0,
        lowest_freq_mode='notch',
        dry_mix=0.5,
        wet_mix=0.5
    ):
    """
    Internal helper to apply a continuously modulating stereo flanger effect.
    Uses a sine LFO based on rate_hz.
    """
    num_samples = x.shape[0]
    duration_s = num_samples / sr

    # Ensure input is float32 for processing consistency
    x = x.astype(np.float32)

    #----------------------------------------------------------
    # (1) Figure out max allowed delay if skipping low comb freqs
    if skip_below_freq is not None and skip_below_freq > 0:
        if lowest_freq_mode.lower() == 'peak':
            max_allowed_delay_s = 1.0 / skip_below_freq
        else: # 'notch'
            max_allowed_delay_s = 1.0 / (2.0 * skip_below_freq)
        max_allowed_delay_ms = 1000.0 * max_allowed_delay_s
        # Apply the limit: the actual max delay cannot exceed this calculated value
        actual_max_delay_ms = min(max_delay_ms, max_allowed_delay_ms)
    else:
        actual_max_delay_ms = max_delay_ms

    # Ensure min delay is not greater than max delay
    min_delay_ms = min(min_delay_ms, actual_max_delay_ms)
    # Prevent zero or negative range
    actual_max_delay_ms = max(min_delay_ms, actual_max_delay_ms)


    #----------------------------------------------------------
    # (2) Create the continuous LFO PHASE array
    t_abs = np.linspace(0, duration_s, num_samples, endpoint=False)
    # Phase = 2 * pi * rate * time + start_phase
    full_phase_array = 2 * np.pi * rate_hz * t_abs + lfo_start_phase_rad

    #----------------------------------------------------------
    # (3) Process sample-by-sample with FRACTIONAL DELAY interpolation
    y = np.zeros_like(x, dtype=np.float32) # Ensure output is float32

    # Pre-calculate delay range and constants
    delay_range_ms = actual_max_delay_ms - min_delay_ms
    delay_range_samples = (delay_range_ms / 1000.0) * sr
    min_delay_samples = (min_delay_ms / 1000.0) * sr

    for n in range(num_samples):
        ph = full_phase_array[n]
        # Convert sine wave [-1, 1] to fraction [0, 1] for delay modulation
        # (sin(ph) + 1) / 2 maps phase to [0, 1] range
        lfo_mod_signal = (np.sin(ph) + 1.0) * 0.5

        # Calculate delay in samples using pre-calculated ranges
        delay_in_samps = min_delay_samples + lfo_mod_signal * delay_range_samples

        # Fractional "tap" position in the past
        tap_float = n - delay_in_samps

        if tap_float < 0:
            # Not enough history at the start, use dry signal for the delayed part
            delayed_sample = x[n] # Simplification: feed dry through wet path initially
        else:
            # Linear interpolation
            i0 = int(np.floor(tap_float))
            frac = tap_float - i0
            i1 = i0 + 1

            # Boundary check for i1 (ensure lookback doesn't exceed current sample 'n')
            if i1 > n:
                 # If i1 goes beyond current sample 'n', clamp to i0
                 x0 = x[i0]
                 x1 = x0 # Use x[i0] when i1 is out of bounds
            else:
                 # Ensure indices are within bounds of x
                 i0 = max(0, i0)
                 i1 = max(0, i1) # Should already be >= i0
                 i1 = min(n, i1) # Clamp i1 to n as well
                 x0 = x[i0]
                 x1 = x[i1]


            # Interpolate for both channels
            delayed_sample = (1.0 - frac) * x0 + frac * x1

        # Mix dry + wet
        y[n] = dry_mix * x[n] + wet_mix * delayed_sample

    return y

# -----------------------------------------------------------------------------
# Synth Def translations – Updated to accept duration, sr, and params dict
# -----------------------------------------------------------------------------

# Each function now takes duration, sample_rate, and **params
# It calculates N, t_rel, t_abs internally.
# It returns a stereo (N x 2) NumPy array.

# --- Synth Function Definitions ---

# Example: Basic AM


def basic_am(duration, sample_rate=44100, **params):
    """Basic Amplitude Modulation synth."""
    # Extract parameters with defaults and ensure correct type
    amp = float(params.get('amp', 0.25))
    carrierFreq = float(params.get('carrierFreq', 200))
    modFreq = float(params.get('modFreq', 4))
    modDepth = float(params.get('modDepth', 0.8))
    pan = float(params.get('pan', 0))  # Added pan parameter

    N = int(sample_rate * duration)
    if N <= 0:
        return np.zeros((0, 2))
    t_rel = np.linspace(0, duration, N, endpoint=False)
    t_abs = t_rel  # For constant params, relative == absolute for phase

    # Note: ADSR envelope is applied *within* generate_voice_audio if specified there.
    # This function just generates the core sound.
    carrier = sine_wave(carrierFreq, t_abs)
    lfo = sine_wave(modFreq, t_abs)
    # Modulator maps LFO [-1, 1] to [1 - modDepth, 1]
    # Original: modulator = np.interp(lfo, [-1, 1], [1 - modDepth, 1])
    # Correct element-wise approach:
    modulator = 1.0 - modDepth * (1.0 - lfo) * 0.5
    output_mono = carrier * modulator * amp # Apply base amp here
    return pan2(output_mono, pan=pan)


def basic_am_transition(duration, sample_rate=44100, **params):
    """Basic Amplitude Modulation synth with parameter transitions."""
    # Extract parameters with defaults and ensure correct type
    amp = float(params.get('amp', 0.25))
    startCarrierFreq = float(params.get('startCarrierFreq', 200))
    endCarrierFreq = float(params.get('endCarrierFreq', 100))
    startModFreq = float(params.get('startModFreq', 4))
    endModFreq = float(params.get('endModFreq', 8))
    startModDepth = float(params.get('startModDepth', 0.8))
    endModDepth = float(params.get('endModDepth', 0.2))
    pan = float(params.get('pan', 0))

    N = int(sample_rate * duration)
    if N <= 0:
        return np.zeros((0, 2))
    t_rel = np.linspace(0, duration, N, endpoint=False)
    t_abs = t_rel

    # Linearly interpolate parameters over the duration
    currentCarrierFreq = np.linspace(startCarrierFreq, endCarrierFreq, N)
    currentModFreq = np.linspace(startModFreq, endModFreq, N)
    currentModDepth = np.linspace(startModDepth, endModDepth, N)

    carrier = sine_wave_varying(currentCarrierFreq, t_abs, sample_rate)
    lfo = sine_wave_varying(currentModFreq, t_abs, sample_rate)

    # --- FIX APPLIED ---
    # Original: modulator = np.interp(lfo, [-1, 1], [1 - currentModDepth, 1])
    modulator = 1.0 - currentModDepth * (1.0 - lfo) * 0.5
    # --------------------

    # Note: Envelope is applied *within* generate_voice_audio if specified there.
    output_mono = carrier * modulator * amp # Apply base amp here
    return pan2(output_mono, pan=pan)


def fsam_filter_bank(duration, sample_rate=44100, **params):
    """Frequency-Selective Amplitude Modulation using filter bank approach."""
    amp = float(params.get('amp', 0.15))
    noiseType = int(params.get('noiseType', 1))  # 1=white, 2=pink, 3=brown
    filterCenterFreq = float(params.get('filterCenterFreq', 1000))
    filterRQ = float(params.get('filterRQ', 0.5)) # Q = 1/RQ
    modFreq = float(params.get('modFreq', 4))
    modDepth = float(params.get('modDepth', 0.8))
    pan = float(params.get('pan', 0))

    N = int(sample_rate * duration)
    if N <= 0:
        return np.zeros((0, 2))
    t_rel = np.linspace(0, duration, N, endpoint=False)
    t_abs = t_rel

    if noiseType == 1:
        source = np.random.randn(N)
    elif noiseType == 2:
        source = pink_noise(N)
    elif noiseType == 3:
        source = brown_noise(N)
    else:
        source = np.random.randn(N)

    # Ensure filter params are valid
    filterCenterFreq = max(20.0, filterCenterFreq)  # Min center freq
    filterQ = 1.0 / max(0.01, filterRQ) if filterRQ > 0 else 100.0 # Calculate Q, avoid division by zero

    bandSignal = bandpass_filter(source, filterCenterFreq, filterQ, sample_rate)
    restSignal = bandreject_filter(source, filterCenterFreq, filterQ, sample_rate)

    lfo = sine_wave(modFreq, t_abs)
    # Correct element-wise approach:
    modulator = 1.0 - modDepth * (1.0 - lfo) * 0.5
    modulatedBand = bandSignal * modulator
    output_mono = modulatedBand + restSignal

    # Note: Envelope is applied *within* generate_voice_audio if specified there.
    output_mono = output_mono * amp # Apply base amp here
    return pan2(output_mono, pan=pan)


def fsam_filter_bank_transition(duration, sample_rate=44100, **params):
    """Frequency-Selective AM with parameter transitions."""
    amp = float(params.get('amp', 0.15))
    noiseType = int(params.get('noiseType', 1))
    startFilterCenterFreq = float(params.get('startFilterCenterFreq', 1000))
    endFilterCenterFreq = float(params.get('endFilterCenterFreq', 3000))
    startFilterRQ = float(params.get('startFilterRQ', 0.5))
    endFilterRQ = float(params.get('endFilterRQ', 0.1))
    startModFreq = float(params.get('startModFreq', 4))
    endModFreq = float(params.get('endModFreq', 12))
    startModDepth = float(params.get('startModDepth', 0.8))
    endModDepth = float(params.get('endModDepth', 0.5))
    pan = float(params.get('pan', 0))

    N = int(sample_rate * duration)
    if N <= 0:
        return np.zeros((0, 2))
    t_rel = np.linspace(0, duration, N, endpoint=False)
    t_abs = t_rel

    # Generate noise source
    if noiseType == 1:
        source = np.random.randn(N)
    elif noiseType == 2:
        source = pink_noise(N)
    elif noiseType == 3:
        source = brown_noise(N)
    else:
        source = np.random.randn(N)

    # Interpolate parameters
    currentFilterCenterFreq = np.linspace(startFilterCenterFreq, endFilterCenterFreq, N)
    currentFilterRQ = np.linspace(startFilterRQ, endFilterRQ, N)
    currentModFreq = np.linspace(startModFreq, endModFreq, N)
    currentModDepth = np.linspace(startModDepth, endModDepth, N)

    # --- Filtering with time-varying parameters is complex. ---
    # --- Simplification: Filter with START parameters only. ---
    # --- A more accurate approach would use time-varying filters (e.g., state-variable filter). ---
    startCenter = max(20.0, startFilterCenterFreq)
    startQ = 1.0 / max(0.01, startFilterRQ) if startFilterRQ > 0 else 100.0
    print("Warning: fsam_filter_bank_transition uses filter parameters from the START of the transition only.")
    bandSignal = bandpass_filter(source, startCenter, startQ, sample_rate)
    restSignal = bandreject_filter(source, startCenter, startQ, sample_rate)
    # --- End Simplification ---

    lfo = sine_wave_varying(currentModFreq, t_abs, sample_rate)

    # --- FIX APPLIED ---
    modulator = 1.0 - currentModDepth * (1.0 - lfo) * 0.5
    # --------------------

    modulatedBand = bandSignal * modulator
    output_mono = modulatedBand + restSignal

    # Note: Envelope is applied *within* generate_voice_audio if specified there.
    output_mono = output_mono * amp # Apply base amp here
    return pan2(output_mono, pan=pan)


def rhythmic_waveshaping(duration, sample_rate=44100, **params):
    """Rhythmic waveshaping using tanh function."""
    amp = float(params.get('amp', 0.25))
    carrierFreq = float(params.get('carrierFreq', 200))
    modFreq = float(params.get('modFreq', 4))
    modDepth = float(params.get('modDepth', 1.0))
    shapeAmount = float(params.get('shapeAmount', 5.0))
    pan = float(params.get('pan', 0))

    N = int(sample_rate * duration)
    if N <= 0:
        return np.zeros((0, 2))
    t_rel = np.linspace(0, duration, N, endpoint=False)
    t_abs = t_rel

    carrier = sine_wave(carrierFreq, t_abs)
    lfo = sine_wave(modFreq, t_abs)
    # Correct element-wise approach:
    shapeLFO = 1.0 - modDepth * (1.0 - lfo) * 0.5 # Modulates amplitude before shaping
    modulatedInput = carrier * shapeLFO

    # Apply waveshaping (tanh)
    shapeAmount = max(1e-6, shapeAmount)  # Avoid division by zero
    tanh_shape_amount = np.tanh(shapeAmount)
    # Use np.divide for safe division
    shapedSignal = np.divide(np.tanh(modulatedInput * shapeAmount), tanh_shape_amount,
                                 out=np.zeros_like(modulatedInput), where=tanh_shape_amount > 1e-6)

    # Note: Envelope is applied *within* generate_voice_audio if specified there.
    output_mono = shapedSignal * amp # Apply base amp here
    return pan2(output_mono, pan=pan)


def rhythmic_waveshaping_transition(duration, sample_rate=44100, **params):
    """Rhythmic waveshaping with parameter transitions."""
    amp = float(params.get('amp', 0.25))
    startCarrierFreq = float(params.get('startCarrierFreq', 200))
    endCarrierFreq = float(params.get('endCarrierFreq', 80))
    startModFreq = float(params.get('startModFreq', 12))
    endModFreq = float(params.get('endModFreq', 7.83))
    startModDepth = float(params.get('startModDepth', 1.0))
    endModDepth = float(params.get('endModDepth', 1.0))  # Allow transition
    startShapeAmount = float(params.get('startShapeAmount', 5.0))
    endShapeAmount = float(params.get('endShapeAmount', 5.0))  # Allow transition
    pan = float(params.get('pan', 0))

    N = int(sample_rate * duration)
    if N <= 0:
        return np.zeros((0, 2))
    t_rel = np.linspace(0, duration, N, endpoint=False)
    t_abs = t_rel

    # Interpolate parameters
    currentCarrierFreq = np.linspace(startCarrierFreq, endCarrierFreq, N)
    currentModFreq = np.linspace(startModFreq, endModFreq, N)
    currentModDepth = np.linspace(startModDepth, endModDepth, N)
    currentShapeAmount = np.linspace(startShapeAmount, endShapeAmount, N)

    carrier = sine_wave_varying(currentCarrierFreq, t_abs, sample_rate)
    lfo = sine_wave_varying(currentModFreq, t_abs, sample_rate)

    # --- FIX APPLIED ---
    shapeLFO = 1.0 - currentModDepth * (1.0 - lfo) * 0.5
    # --------------------

    modulatedInput = carrier * shapeLFO

    # Apply time-varying waveshaping
    currentShapeAmount = np.maximum(1e-6, currentShapeAmount)  # Avoid zero
    tanh_shape_amount = np.tanh(currentShapeAmount)
    # Use np.divide for safe division
    shapedSignal = np.divide(np.tanh(modulatedInput * currentShapeAmount), tanh_shape_amount,
                                 out=np.zeros_like(modulatedInput), where=tanh_shape_amount > 1e-6)

    # Note: Envelope is applied *within* generate_voice_audio if specified there.
    output_mono = shapedSignal * amp # Apply base amp here
    return pan2(output_mono, pan=pan)


# Removed: rhythmic_granular_rate


# Removed: additive_phase_mod


# Removed: additive_phase_mod_transition


def stereo_am_independent(duration, sample_rate=44100, **params):
    """Stereo Amplitude Modulation with independent L/R modulators."""
    amp = float(params.get('amp', 0.25))
    carrierFreq = float(params.get('carrierFreq', 200.0))
    modFreqL = float(params.get('modFreqL', 4.0))
    modDepthL = float(params.get('modDepthL', 0.8))
    modPhaseL = float(params.get('modPhaseL', 0))  # Phase in radians
    modFreqR = float(params.get('modFreqR', 4.0))
    modDepthR = float(params.get('modDepthR', 0.8))
    modPhaseR = float(params.get('modPhaseR', 0))  # Phase in radians
    stereo_width_hz = float(params.get('stereo_width_hz', 0.2))  # Freq difference

    N = int(sample_rate * duration)
    if N <= 0:
        return np.zeros((0, 2))
    t_rel = np.linspace(0, duration, N, endpoint=False)
    t_abs = t_rel

    # Carriers with slight detuning
    carrierL = sine_wave(carrierFreq - stereo_width_hz / 2, t_abs)
    carrierR = sine_wave(carrierFreq + stereo_width_hz / 2, t_abs)

    # Independent LFOs
    lfoL = sine_wave(modFreqL, t_abs, phase=modPhaseL)
    lfoR = sine_wave(modFreqR, t_abs, phase=modPhaseR)

    # Modulators (Correct element-wise approach)
    modulatorL = 1.0 - modDepthL * (1.0 - lfoL) * 0.5
    modulatorR = 1.0 - modDepthR * (1.0 - lfoR) * 0.5

    # Note: Envelope is applied *within* generate_voice_audio if specified there.
    outputL = carrierL * modulatorL * amp # Apply base amp here
    outputR = carrierR * modulatorR * amp

    return np.vstack([outputL, outputR]).T


def stereo_am_independent_transition(duration, sample_rate=44100, **params):
    """Stereo AM Independent with parameter transitions."""
    amp = float(params.get('amp', 0.25))
    startCarrierFreq = float(params.get('startCarrierFreq', 200))
    endCarrierFreq = float(params.get('endCarrierFreq', 250))
    startModFreqL = float(params.get('startModFreqL', 4))
    endModFreqL = float(params.get('endModFreqL', 6))
    startModDepthL = float(params.get('startModDepthL', 0.8))
    endModDepthL = float(params.get('endModDepthL', 0.8))  # Allow transition
    startModPhaseL = float(params.get('startModPhaseL', 0))  # Constant phase
    startModFreqR = float(params.get('startModFreqR', 4.1))
    endModFreqR = float(params.get('endModFreqR', 5.9))
    startModDepthR = float(params.get('startModDepthR', 0.8))
    endModDepthR = float(params.get('endModDepthR', 0.8))  # Allow transition
    startModPhaseR = float(params.get('startModPhaseR', 0))  # Constant phase
    startStereoWidthHz = float(params.get('startStereoWidthHz', 0.2))
    endStereoWidthHz = float(params.get('endStereoWidthHz', 0.2))  # Allow transition

    N = int(sample_rate * duration)
    if N <= 0:
        return np.zeros((0, 2))
    t_rel = np.linspace(0, duration, N, endpoint=False)
    t_abs = t_rel

    # Interpolate parameters
    currentCarrierFreq = np.linspace(startCarrierFreq, endCarrierFreq, N)
    currentModFreqL = np.linspace(startModFreqL, endModFreqL, N)
    currentModDepthL = np.linspace(startModDepthL, endModDepthL, N)
    currentModFreqR = np.linspace(startModFreqR, endModFreqR, N)
    currentModDepthR = np.linspace(startModDepthR, endModDepthR, N)
    currentStereoWidthHz = np.linspace(startStereoWidthHz, endStereoWidthHz, N)

    # Varying frequency carriers
    carrierL = sine_wave_varying(currentCarrierFreq - currentStereoWidthHz / 2, t_abs, sample_rate)
    carrierR = sine_wave_varying(currentCarrierFreq + currentStereoWidthHz / 2, t_abs, sample_rate)

    # Varying frequency LFOs (assuming constant phase during transition)
    lfoL = sine_wave_varying(currentModFreqL, t_abs, sample_rate) # Phase ignored by sine_wave_varying
    lfoR = sine_wave_varying(currentModFreqR, t_abs, sample_rate) # Phase ignored by sine_wave_varying

    # Modulators with varying depth
    # --- FIX APPLIED (L) ---
    modulatorL = 1.0 - currentModDepthL * (1.0 - lfoL) * 0.5
    # -----------------------
    # --- FIX APPLIED (R) ---
    modulatorR = 1.0 - currentModDepthR * (1.0 - lfoR) * 0.5
    # -----------------------

    # Note: Envelope is applied *within* generate_voice_audio if specified there.
    outputL = carrierL * modulatorL * amp # Apply base amp here
    outputR = carrierR * modulatorR * amp

    return np.vstack([outputL, outputR]).T


# Removed: noise_am

# Removed: noise_am_transition

def wave_shape_stereo_am(duration, sample_rate=44100, **params):
    """Combines rhythmic waveshaping and stereo AM."""
    amp = float(params.get('amp', 0.15))
    carrierFreq = float(params.get('carrierFreq', 200))
    shapeModFreq = float(params.get('shapeModFreq', 4))
    shapeModDepth = float(params.get('shapeModDepth', 0.8))
    shapeAmount = float(params.get('shapeAmount', 0.5))
    stereoModFreqL = float(params.get('stereoModFreqL', 4.1))
    stereoModDepthL = float(params.get('stereoModDepthL', 0.8))
    stereoModPhaseL = float(params.get('stereoModPhaseL', 0))
    stereoModFreqR = float(params.get('stereoModFreqR', 4.0))
    stereoModDepthR = float(params.get('stereoModDepthR', 0.8))
    stereoModPhaseR = float(params.get('stereoModPhaseR', math.pi * 2)) # Original had 2pi, likely meant pi/2 or pi? Using pi/2 for quadrature.
    stereoModPhaseR = float(params.get('stereoModPhaseR', math.pi / 2)) # Changed default

    N = int(sample_rate * duration)
    if N <= 0:
        return np.zeros((0, 2))
    t_rel = np.linspace(0, duration, N, endpoint=False)
    t_abs = t_rel

    # Rhythmic waveshaping part (mono)
    carrier = sine_wave(carrierFreq, t_abs)
    shapeLFO_wave = sine_wave(shapeModFreq, t_abs)
    # Correct element-wise approach:
    shapeLFO_amp = 1.0 - shapeModDepth * (1.0 - shapeLFO_wave) * 0.5
    modulatedInput = carrier * shapeLFO_amp
    shapeAmount = max(1e-6, shapeAmount)
    tanh_shape_amount = np.tanh(shapeAmount)
    shapedSignal = np.divide(np.tanh(modulatedInput * shapeAmount), tanh_shape_amount,
                                 out=np.zeros_like(modulatedInput), where=tanh_shape_amount > 1e-6)

    # Stereo AM part
    stereoLFO_L = sine_wave(stereoModFreqL, t_abs, phase=stereoModPhaseL)
    stereoLFO_R = sine_wave(stereoModFreqR, t_abs, phase=stereoModPhaseR)
    # Correct element-wise approach:
    modulatorL = 1.0 - stereoModDepthL * (1.0 - stereoLFO_L) * 0.5
    modulatorR = 1.0 - stereoModDepthR * (1.0 - stereoLFO_R) * 0.5

    # Apply stereo modulators
    outputL = shapedSignal * modulatorL
    outputR = shapedSignal * modulatorR

    # Apply overall amplitude (envelope applied later)
    outputL = outputL * amp
    outputR = outputR * amp

    return np.vstack([outputL, outputR]).T


def wave_shape_stereo_am_transition(duration, sample_rate=44100, **params):
    """Combined waveshaping and stereo AM with parameter transitions."""
    amp = float(params.get('amp', 0.15))
    startCarrierFreq = float(params.get('startCarrierFreq', 200))
    endCarrierFreq = float(params.get('endCarrierFreq', 100))
    startShapeModFreq = float(params.get('startShapeModFreq', 4))
    endShapeModFreq = float(params.get('endShapeModFreq', 8))
    startShapeModDepth = float(params.get('startShapeModDepth', 0.8))
    endShapeModDepth = float(params.get('endShapeModDepth', 0.8))
    startShapeAmount = float(params.get('startShapeAmount', 0.5))
    endShapeAmount = float(params.get('endShapeAmount', 0.5))
    startStereoModFreqL = float(params.get('startStereoModFreqL', 4.1))
    endStereoModFreqL = float(params.get('endStereoModFreqL', 6.0))
    startStereoModDepthL = float(params.get('startStereoModDepthL', 0.8))
    endStereoModDepthL = float(params.get('endStereoModDepthL', 0.8))
    startStereoModPhaseL = float(params.get('startStereoModPhaseL', 0))  # Constant
    startStereoModFreqR = float(params.get('startStereoModFreqR', 4.0))
    endStereoModFreqR = float(params.get('endStereoModFreqR', 6.1))
    startStereoModDepthR = float(params.get('startStereoModDepthR', 0.9))
    endStereoModDepthR = float(params.get('endStereoModDepthR', 0.9))
    startStereoModPhaseR = float(params.get('startStereoModPhaseR', math.pi / 2)) # Constant (using corrected default)

    N = int(sample_rate * duration)
    if N <= 0:
        return np.zeros((0, 2))
    t_rel = np.linspace(0, duration, N, endpoint=False)
    t_abs = t_rel

    # Interpolate parameters
    currentCarrierFreq = np.linspace(startCarrierFreq, endCarrierFreq, N)
    currentShapeModFreq = np.linspace(startShapeModFreq, endShapeModFreq, N)
    currentShapeModDepth = np.linspace(startShapeModDepth, endShapeModDepth, N)
    currentShapeAmount = np.linspace(startShapeAmount, endShapeAmount, N)
    currentStereoModFreqL = np.linspace(startStereoModFreqL, endStereoModFreqL, N)
    currentStereoModDepthL = np.linspace(startStereoModDepthL, endStereoModDepthL, N)
    currentStereoModFreqR = np.linspace(startStereoModFreqR, endStereoModFreqR, N)
    currentStereoModDepthR = np.linspace(startStereoModDepthR, endStereoModDepthR, N)

    # Rhythmic waveshaping part (mono)
    carrier = sine_wave_varying(currentCarrierFreq, t_abs, sample_rate)
    shapeLFO_wave = sine_wave_varying(currentShapeModFreq, t_abs, sample_rate)
    # --- FIX APPLIED (shapeLFO) ---
    shapeLFO_amp = 1.0 - currentShapeModDepth * (1.0 - shapeLFO_wave) * 0.5
    # -----------------------------
    modulatedInput = carrier * shapeLFO_amp
    currentShapeAmount = np.maximum(1e-6, currentShapeAmount)
    tanh_shape_amount = np.tanh(currentShapeAmount)
    shapedSignal = np.divide(np.tanh(modulatedInput * currentShapeAmount), tanh_shape_amount,
                                 out=np.zeros_like(modulatedInput), where=tanh_shape_amount > 1e-6)

    # Stereo AM part (assuming constant phase)
    stereoLFO_L = sine_wave_varying(currentStereoModFreqL, t_abs, sample_rate)
    stereoLFO_R = sine_wave_varying(currentStereoModFreqR, t_abs, sample_rate)
    # --- FIX APPLIED (modulatorL) ---
    modulatorL = 1.0 - currentStereoModDepthL * (1.0 - stereoLFO_L) * 0.5
    # --------------------------------
    # --- FIX APPLIED (modulatorR) ---
    modulatorR = 1.0 - currentStereoModDepthR * (1.0 - stereoLFO_R) * 0.5
    # --------------------------------

    # Apply stereo modulators
    outputL = shapedSignal * modulatorL
    outputR = shapedSignal * modulatorR

    # Apply overall amplitude (envelope applied later)
    outputL = outputL * amp
    outputR = outputR * amp

    return np.vstack([outputL, outputR]).T

# --- NEW: Flanged Voice Synth Definition ---
def flanged_voice(duration, sample_rate=44100, **params):
    """
    Generates a continuously modulating flanging effect on a noise source.
    Uses a sine LFO based on rate_hz.
    """
    # Extract parameters
    amp = float(params.get('amp', 0.5)) # Overall amplitude
    noiseType = int(params.get('noiseType', 2))  # 1=white, 2=pink, 3=brown
    max_delay_ms = float(params.get('max_delay_ms', 15.0)) # Max LFO delay
    min_delay_ms = float(params.get('min_delay_ms', 1.0))  # Min LFO delay
    rate_hz = float(params.get('rate_hz', 0.2)) # Flanger LFO rate (speed)
    lfo_start_phase_rad = float(params.get('lfo_start_phase_rad', 0.0)) # Starting phase of LFO
    skip_below_freq = float(params.get('skip_below_freq', 30.0)) # Skip comb frequencies below this
    lowest_freq_mode = str(params.get('lowest_freq_mode', 'notch')) # 'notch' or 'peak'
    dry_mix = float(params.get('dry_mix', 0.5)) # Mix level of original signal
    wet_mix = float(params.get('wet_mix', 0.5)) # Mix level of flanged signal

    N = int(sample_rate * duration)
    if N <= 0:
        return np.zeros((0, 2))

    # Generate noise source (mono)
    if noiseType == 1:
        source_mono = np.random.randn(N)
    elif noiseType == 2:
        source_mono = pink_noise(N)
    elif noiseType == 3:
        source_mono = brown_noise(N)
    else:
        print(f"Warning: Unknown noiseType {noiseType}. Defaulting to pink noise.")
        source_mono = pink_noise(N) # Default to pink

    # Ensure source is not completely silent
    if np.max(np.abs(source_mono)) < 1e-9:
         print("Warning: Generated noise source is silent for flanger.")
         return np.zeros((N, 2))

    # Convert mono noise to stereo for the flanger effect input
    source_stereo = np.vstack([source_mono, source_mono]).T

    # Apply the flanger effect using the internal helper
    try:
        flanged_output = _flanger_effect_stereo_continuous(
            x=source_stereo,
            sr=sample_rate,
            max_delay_ms=max_delay_ms,
            min_delay_ms=min_delay_ms,
            rate_hz=rate_hz,
            lfo_start_phase_rad=lfo_start_phase_rad,
            skip_below_freq=skip_below_freq,
            lowest_freq_mode=lowest_freq_mode,
            dry_mix=dry_mix,
            wet_mix=wet_mix
        )
    except Exception as e:
        print(f"Error during continuous flanger effect processing:")
        traceback.print_exc()
        return np.zeros((N, 2)) # Return silence on error

    # Apply overall amplitude AFTER the effect
    output_stereo = flanged_output * amp

    # Note: Volume envelope (like ADSR/Linen) is applied later in generate_voice_audio.
    return output_stereo.astype(np.float32) # Ensure final output is float32


# --- NEW: Flanged Voice Transition Synth Definition ---
def flanged_voice_transition(duration, sample_rate=44100, **params):
    """
    Generates a flanging effect with linearly transitioning parameters.
    Uses a continuously modulating sine LFO where rate can also transition.
    """
    # Extract start/end parameters
    startAmp = float(params.get('startAmp', 0.5))
    endAmp = float(params.get('endAmp', startAmp)) # Default end to start if not provided

    startNoiseType = int(params.get('startNoiseType', 2)) # Use start value only

    startMaxDelayMs = float(params.get('startMaxDelayMs', 15.0))
    endMaxDelayMs = float(params.get('endMaxDelayMs', startMaxDelayMs))
    startMinDelayMs = float(params.get('startMinDelayMs', 1.0))
    endMinDelayMs = float(params.get('endMinDelayMs', startMinDelayMs))

    startRateHz = float(params.get('startRateHz', 0.2))
    endRateHz = float(params.get('endRateHz', startRateHz))

    startLfoPhaseRad = float(params.get('startLfoPhaseRad', 0.0)) # Use start value only

    startSkipBelowFreq = float(params.get('startSkipBelowFreq', 30.0))
    endSkipBelowFreq = float(params.get('endSkipBelowFreq', startSkipBelowFreq))
    startLowestFreqMode = str(params.get('startLowestFreqMode', 'notch')) # Use start value only

    startDryMix = float(params.get('startDryMix', 0.5))
    endDryMix = float(params.get('endDryMix', startDryMix))
    startWetMix = float(params.get('startWetMix', 0.5))
    endWetMix = float(params.get('endWetMix', startWetMix))

    N = int(sample_rate * duration)
    if N <= 0:
        return np.zeros((0, 2))

    # --- Generate Noise Source (based on start type) ---
    if startNoiseType == 1:
        source_mono = np.random.randn(N)
    elif startNoiseType == 2:
        source_mono = pink_noise(N)
    elif startNoiseType == 3:
        source_mono = brown_noise(N)
    else:
        print(f"Warning: Unknown startNoiseType {startNoiseType}. Defaulting to pink noise.")
        source_mono = pink_noise(N)

    if np.max(np.abs(source_mono)) < 1e-9:
        print("Warning: Generated noise source is silent for flanger transition.")
        return np.zeros((N, 2))

    source_stereo = np.vstack([source_mono, source_mono]).T.astype(np.float32)

    # --- Interpolate Parameters ---
    currentAmp = np.linspace(startAmp, endAmp, N)
    currentMaxDelayMs = np.linspace(startMaxDelayMs, endMaxDelayMs, N)
    currentMinDelayMs = np.linspace(startMinDelayMs, endMinDelayMs, N)
    currentRateHz = np.linspace(startRateHz, endRateHz, N)
    currentSkipBelowFreq = np.linspace(startSkipBelowFreq, endSkipBelowFreq, N)
    currentDryMix = np.linspace(startDryMix, endDryMix, N)
    currentWetMix = np.linspace(startWetMix, endWetMix, N)

    # Ensure min_delay <= max_delay and handle skip_freq limit dynamically
    # Vectorize calculations for efficiency
    currentMinDelayMs = np.minimum(currentMinDelayMs, currentMaxDelayMs)
    currentMaxDelayMs = np.maximum(currentMinDelayMs, currentMaxDelayMs) # Ensure max >= min after potential adjustments

    max_allowed_delay_ms = np.full(N, np.inf) # Initialize with no limit
    valid_skip_mask = currentSkipBelowFreq > 0
    if np.any(valid_skip_mask):
        if startLowestFreqMode.lower() == 'peak':
             max_allowed_delay_ms[valid_skip_mask] = 1000.0 / currentSkipBelowFreq[valid_skip_mask]
        else: # 'notch'
             max_allowed_delay_ms[valid_skip_mask] = 1000.0 / (2.0 * currentSkipBelowFreq[valid_skip_mask])

    actualMaxDelayMs = np.minimum(currentMaxDelayMs, max_allowed_delay_ms)
    # Re-ensure min <= max after applying skip limit
    actualMinDelayMs = np.minimum(currentMinDelayMs, actualMaxDelayMs)
    actualMaxDelayMs = np.maximum(actualMinDelayMs, actualMaxDelayMs)


    # --- Calculate Time-Varying LFO Phase ---
    # Phase is integral of angular frequency (2*pi*rate)
    # Use cumsum for discrete integration: phase[n] = start_phase + sum(2*pi*rate[i]*dt for i=0 to n-1)
    dt = 1.0 / sample_rate
    inst_angular_freq = 2 * np.pi * currentRateHz
    phase_increment = inst_angular_freq * dt
    # Add start phase to the first element, then cumsum includes it
    phase_lfo = np.cumsum(phase_increment) + startLfoPhaseRad - phase_increment[0] # Adjust start phase offset for cumsum


    # --- Sample-by-Sample Processing (Inlined & Adapted) ---
    y = np.zeros_like(source_stereo, dtype=np.float32)

    # Pre-calculate delay ranges in samples (now arrays)
    delay_range_samples = (actualMaxDelayMs - actualMinDelayMs) / 1000.0 * sample_rate
    min_delay_samples = actualMinDelayMs / 1000.0 * sample_rate

    # Ensure non-negative ranges
    delay_range_samples = np.maximum(0, delay_range_samples)
    min_delay_samples = np.maximum(0, min_delay_samples)


    for n in range(N):
        ph = phase_lfo[n]
        lfo_mod_signal = (np.sin(ph) + 1.0) * 0.5

        # Calculate delay in samples for this time step n
        delay_in_samps = min_delay_samples[n] + lfo_mod_signal * delay_range_samples[n]

        # Fractional "tap" position in the past
        tap_float = n - delay_in_samps

        if tap_float < 0:
            delayed_sample = source_stereo[n] # Use dry signal when history is insufficient
        else:
            # Linear interpolation
            i0 = int(np.floor(tap_float))
            frac = tap_float - i0
            i1 = i0 + 1

            if i1 > n:
                x0 = source_stereo[max(0, i0)] # Ensure index >= 0
                x1 = x0
            else:
                i0 = max(0, i0)
                i1 = max(0, i1)
                x0 = source_stereo[i0]
                x1 = source_stereo[i1]

            delayed_sample = (1.0 - frac) * x0 + frac * x1

        # Mix dry + wet using interpolated values for this time step
        y[n] = currentDryMix[n] * source_stereo[n] + currentWetMix[n] * delayed_sample

    # Apply overall interpolated amplitude
    output_stereo = y * currentAmp[:, np.newaxis] # Apply amp using broadcasting

    # Note: Volume envelope is applied later in generate_voice_audio.
    return output_stereo.astype(np.float32)

def spatial_angle_modulation(duration, sample_rate=44100, **params):
    """Spatial Angle Modulation using external audio_engine module."""
    if not AUDIO_ENGINE_AVAILABLE:
        print("Error: SAM function called, but audio_engine module is missing.")
        N = int(sample_rate * duration)
        return np.zeros((N, 2))

    amp = float(params.get('amp', 0.7))
    carrierFreq = float(params.get('carrierFreq', 440.0))
    beatFreq = float(params.get('beatFreq', 4.0))
    pathShape = str(params.get('pathShape', 'circle'))
    pathRadius = float(params.get('pathRadius', 1.0))
    arcStartDeg = float(params.get('arcStartDeg', 0.0))
    arcEndDeg = float(params.get('arcEndDeg', 360.0))
    frame_dur_ms = float(params.get('frame_dur_ms', 46.4))
    overlap_factor = int(params.get('overlap_factor', 8))

    if pathShape not in VALID_SAM_PATHS:
        print(f"Warning: Invalid pathShape '{pathShape}'. Defaulting to 'circle'. Valid: {VALID_SAM_PATHS}")
        pathShape = 'circle'

    try:
        node = Node(duration, carrierFreq, beatFreq, 1.0, 1.0) # Using real Node now
    except Exception as e:
        print(f"Error creating Node for SAM: {e}")
        N = int(sample_rate * duration)
        return np.zeros((N, 2))

    sam_params_dict = {
        'path_shape': pathShape,
        'path_radius': pathRadius,
        'arc_start_deg': arcStartDeg,
        'arc_end_deg': arcEndDeg
    }

    try:
        voice = SAMVoice( # Using real SAMVoice now
            nodes=[node],
            sample_rate=sample_rate,
            frame_dur_ms=frame_dur_ms,
            overlap_factor=overlap_factor,
            source_amp=amp, # Apply amp within SAMVoice
            sam_node_params=[sam_params_dict]
        )
        # Note: Envelope is applied *within* generate_voice_audio if specified there.
        return voice.generate_samples()
    except Exception as e:
        print(f"Error during SAMVoice generation: {e}")
        traceback.print_exc()
        N = int(sample_rate * duration)
        return np.zeros((N, 2))


def spatial_angle_modulation_transition(duration, sample_rate=44100, **params):
    """Spatial Angle Modulation with parameter transitions."""
    if not AUDIO_ENGINE_AVAILABLE:
        print("Error: SAM transition function called, but audio_engine module is missing.")
        N = int(sample_rate * duration)
        return np.zeros((N, 2))

    amp = float(params.get('amp', 0.7))
    startCarrierFreq = float(params.get('startCarrierFreq', 440.0))
    endCarrierFreq = float(params.get('endCarrierFreq', 440.0))
    startBeatFreq = float(params.get('startBeatFreq', 4.0))
    endBeatFreq = float(params.get('endBeatFreq', 4.0))
    startPathShape = str(params.get('startPathShape', 'circle'))
    endPathShape = str(params.get('endPathShape', 'circle'))
    startPathRadius = float(params.get('startPathRadius', 1.0))
    endPathRadius = float(params.get('endPathRadius', 1.0))
    startArcStartDeg = float(params.get('startArcStartDeg', 0.0))
    endArcStartDeg = float(params.get('endArcStartDeg', 0.0))
    startArcEndDeg = float(params.get('startArcEndDeg', 360.0))
    endArcEndDeg = float(params.get('endArcEndDeg', 360.0))
    frame_dur_ms = float(params.get('frame_dur_ms', 46.4))
    overlap_factor = int(params.get('overlap_factor', 8))

    if startPathShape not in VALID_SAM_PATHS:
        print(f"Warning: Invalid startPathShape '{startPathShape}'. Defaulting to 'circle'.")
        startPathShape = 'circle'
    if endPathShape not in VALID_SAM_PATHS:
        print(f"Warning: Invalid endPathShape '{endPathShape}'. Defaulting to 'circle'.")
        endPathShape = 'circle'

    try:
        # Define start and end nodes for transition
        node_start = Node(duration, startCarrierFreq, startBeatFreq, 1.0, 1.0)
        node_end = Node(0.0, endCarrierFreq, endBeatFreq, 1.0, 1.0) # End node has 0 duration
    except Exception as e:
        print(f"Error creating Nodes for SAM transition: {e}")
        N = int(sample_rate * duration)
        return np.zeros((N, 2))

    sam_params_list = [
        { # Parameters for the start node
            'path_shape': startPathShape,
            'path_radius': startPathRadius,
            'arc_start_deg': startArcStartDeg,
            'arc_end_deg': startArcEndDeg
        },
        { # Parameters for the (conceptual) end node
            'path_shape': endPathShape,
            'path_radius': endPathRadius,
            'arc_start_deg': endArcStartDeg,
            'arc_end_deg': endArcEndDeg
        }
    ]

    try:
        voice = SAMVoice( # Using real SAMVoice
            nodes=[node_start, node_end], # Pass both nodes for transition
            sample_rate=sample_rate,
            frame_dur_ms=frame_dur_ms,
            overlap_factor=overlap_factor,
            source_amp=amp, # Apply amp within SAMVoice
            sam_node_params=sam_params_list # Pass list of params
        )
        # Note: Envelope is applied *within* generate_voice_audio if specified there.
        return voice.generate_samples()
    except Exception as e:
        print(f"Error during SAMVoice transition generation: {e}")
        traceback.print_exc()
        N = int(sample_rate * duration)
        return np.zeros((N, 2))


def binaural_beat(duration, sample_rate=44100, **params):
    """
    Generates a binaural beat signal with optional phase, independent L/R
    base amplitude, amplitude modulation, and frequency oscillation.

    Args:
        duration (float): Duration of the signal in seconds.
        sample_rate (int): Sampling rate in Hz.
        **params: Dictionary of parameters:
            ampL (float): Base amplitude for LEFT channel (0 to 1). Default: 0.5.
            ampR (float): Base amplitude for RIGHT channel (0 to 1). Default: 0.5.
            baseFreq (float): Center frequency in Hz before beat/oscillation. Default: 200.0.
            beatFreq (float): Beat frequency in Hz (difference between L/R mean freqs). Default: 4.0.
            startPhaseL (float): Starting phase for the left channel in radians. Default: 0.0.
            startPhaseR (float): Starting phase for the right channel in radians. Default: 0.0.
            phaseOscFreq (float): Frequency of phase offset oscillation in Hz. Default: 0.0.
            phaseOscRange (float): Range of phase offset oscillation in radians. Default: 0.0.
            ampOscDepthL (float): Depth of LEFT amplitude modulation (0 to 1). Default: 0.0.
            ampOscFreqL (float): Frequency of LEFT amplitude modulation in Hz. Default: 0.0.
            ampOscDepthR (float): Depth of RIGHT amplitude modulation (0 to 1). Default: 0.0.
            ampOscFreqR (float): Frequency of RIGHT amplitude modulation in Hz. Default: 0.0.
            freqOscRangeL (float): Range of LEFT frequency oscillation in Hz. Default: 0.0.
            freqOscFreqL (float): Frequency of LEFT frequency oscillation in Hz. Default: 0.0.
            freqOscRangeR (float): Range of RIGHT frequency oscillation in Hz. Default: 0.0.
            freqOscFreqR (float): Frequency of RIGHT frequency oscillation in Hz. Default: 0.0.


    Returns:
        np.ndarray: Stereo audio signal as a numpy array (N, 2).
    """
    # --- Get Parameters ---
    ampL = float(params.get('ampL', 0.5))
    ampR = float(params.get('ampR', 0.5))
    baseFreq = float(params.get('baseFreq', 200.0))
    beatFreq = float(params.get('beatFreq', 4.0))
    startPhaseL_rad = float(params.get('startPhaseL', 0.0))
    startPhaseR_rad = float(params.get('startPhaseR', 0.0))
    phaseOscFreq = float(params.get('phaseOscFreq', 0.0))
    phaseOscRange_rad = float(params.get('phaseOscRange', 0.0))
    ampOscDepthL = float(params.get('ampOscDepthL', 0.0))
    ampOscFreqL = float(params.get('ampOscFreqL', 0.0))
    ampOscDepthR = float(params.get('ampOscDepthR', 0.0))
    ampOscFreqR = float(params.get('ampOscFreqR', 0.0))
    # New frequency oscillation parameters
    freqOscRangeL = float(params.get('freqOscRangeL', 0.0))
    freqOscFreqL = float(params.get('freqOscFreqL', 0.0))
    freqOscRangeR = float(params.get('freqOscRangeR', 0.0))
    freqOscFreqR = float(params.get('freqOscFreqR', 0.0))

    # --- Basic Setup ---
    N = int(sample_rate * duration)
    if N <= 0: return np.zeros((0, 2))
    t_abs = np.linspace(0, duration, N, endpoint=False)

    # --- Frequency Calculation (Now includes Oscillation) ---
    half_beat_freq = beatFreq / 2.0
    # Calculate base frequencies for each channel
    baseFreqL = baseFreq - half_beat_freq
    baseFreqR = baseFreq + half_beat_freq

    # Calculate frequency oscillation terms (LFOs for frequency)
    # Oscillation is centered around 0, range is peak-to-peak
    freq_osc_term_l = (freqOscRangeL / 2.0) * np.sin(2 * np.pi * freqOscFreqL * t_abs)
    freq_osc_term_r = (freqOscRangeR / 2.0) * np.sin(2 * np.pi * freqOscFreqR * t_abs)

    # Calculate instantaneous frequency arrays for each channel
    # Add the oscillation term to the base frequency for each channel
    # Ensure frequency doesn't go below zero
    left_freq_array = np.maximum(0.0, baseFreqL + freq_osc_term_l)
    right_freq_array = np.maximum(0.0, baseFreqR + freq_osc_term_r)

    # --- Phase Calculation (Integrates the instantaneous frequency) ---
    # Calculate time delta between samples for integration
    if N > 1: dt = np.diff(t_abs, prepend=t_abs[0]) # More robust dt calculation
    elif N == 1: dt = np.array([duration])
    else: dt = np.array([])

    if N > 0:
      # Use cumulative sum for phase integration: phase = integral(2*pi*frequency(t) dt)
      phase_left_base = np.cumsum(2 * np.pi * left_freq_array * dt)
      phase_right_base = np.cumsum(2 * np.pi * right_freq_array * dt)
    else:
      phase_left_base = np.array([])
      phase_right_base = np.array([])

    # Add starting phase and phase offset oscillation (if any)
    phase_osc_term = (phaseOscRange_rad / 2.0) * np.sin(2 * np.pi * phaseOscFreq * t_abs)
    phase_left_total = phase_left_base + startPhaseL_rad - phase_osc_term
    phase_right_total = phase_right_base + startPhaseR_rad + phase_osc_term

    # --- Independent Amplitude Modulation (Tremolo) ---
    # Amplitude LFOs: vary between (1 - depth) and 1 ? No, previous was better
    # Corrected LFO: (1.0 - depth / 2.0) + (depth / 2.0) * sin -> range [1-depth, 1] Seems wrong
    # Let's try Amplitude * (1 + depth * sin): Range [Amp*(1-depth), Amp*(1+depth)] centered at Amp. Needs clipping/scaling.
    # Let's stick to the previous formula: multiplies base signal by a value in [1-depth, 1]
    # Correct calculation: Modulator varies from 0 to 1: (1 + sin)/2
    # Modulated Amplitude = BaseAmp * (1 - Depth * Modulator) = BaseAmp * (1 - Depth * (1+sin)/2 )
    # This ranges from BaseAmp*(1-Depth) to BaseAmp*(1 - Depth*0) = BaseAmp
    amp_mod_l = (1.0 - ampOscDepthL * (0.5 * (1 + np.sin(2 * np.pi * ampOscFreqL * t_abs))))
    amp_mod_r = (1.0 - ampOscDepthR * (0.5 * (1 + np.sin(2 * np.pi * ampOscFreqR * t_abs))))

    # --- Signal Generation ---
    s_left_base = np.sin(phase_left_total)
    s_right_base = np.sin(phase_right_total)

    # Apply independent amplitude modulation AND independent base amplitude
    s_left = s_left_base * amp_mod_l * ampL
    s_right = s_right_base * amp_mod_r * ampR

    # Combine into stereo signal
    audio = np.column_stack((s_left, s_right)).astype(np.float32)
    return audio

def monaural_beat_stereo_amps(duration, sample_rate=44100, **params):
    """
    Generates a potentially stereo beat signal by summing two tones with
    independent amplitude control for each tone in each channel.

    This allows for creating variations beyond a strict monaural beat
    (where L and R channels are identical). Amplitude modulation (tremolo)
    can still be applied to the resulting signal in each channel.

    Args:
        duration (float): Duration of the signal in seconds.
        sample_rate (int): Sampling rate in Hz.
        **params: Dictionary of parameters:
            amp_lower_L (float): Amplitude (0 to 1) for the lower frequency tone
                                 (baseFreq - beatFreq/2) in the Left channel. Default: 0.5.
            amp_upper_L (float): Amplitude (0 to 1) for the upper frequency tone
                                 (baseFreq + beatFreq/2) in the Left channel. Default: 0.5.
            amp_lower_R (float): Amplitude (0 to 1) for the lower frequency tone
                                 (baseFreq - beatFreq/2) in the Right channel. Default: 0.5.
            amp_upper_R (float): Amplitude (0 to 1) for the upper frequency tone
                                 (baseFreq + beatFreq/2) in the Right channel. Default: 0.5.
            baseFreq (float): Center frequency in Hz. Default: 200.0.
            beatFreq (float): Beat frequency in Hz (difference between the two tones). Default: 4.0.
            startPhaseL (float): Starting phase for the lower frequency tone in radians. Default: 0.0.
            startPhaseR (float): Starting phase for the upper frequency tone in radians. Default: 0.0.
            phaseOscFreq (float): Frequency of phase offset oscillation in Hz. Default: 0.0.
            phaseOscRange (float): Range of phase offset oscillation in radians. Default: 0.0.
            ampOscDepth (float): Depth of amplitude modulation (tremolo) applied to
                                 the summed signal in each channel (0 to 2). Default: 0.0.
            ampOscFreq (float): Frequency of amplitude modulation (tremolo) in Hz. Default: 0.0.

    Returns:
        np.ndarray: Stereo audio signal (N, 2) as float32.
    """
    # --- Get Parameters ---
    # Amplitudes for each tone component in each channel
    amp_lower_L = float(params.get('amp_lower_L', 0.5))
    amp_upper_L = float(params.get('amp_upper_L', 0.5))
    amp_lower_R = float(params.get('amp_lower_R', 0.5))
    amp_upper_R = float(params.get('amp_upper_R', 0.5))

    # Frequency and Phase parameters
    baseFreq = float(params.get('baseFreq', 200.0))
    beatFreq = float(params.get('beatFreq', 4.0))
    startPhaseLower_rad = float(params.get('startPhaseL', 0.0)) # Renamed for clarity
    startPhaseUpper_rad = float(params.get('startPhaseR', 0.0)) # Renamed for clarity
    phaseOscFreq = float(params.get('phaseOscFreq', 0.0))
    phaseOscRange_rad = float(params.get('phaseOscRange', 0.0))

    # Amplitude Modulation (Tremolo) parameters
    ampOscDepth = float(params.get('ampOscDepth', 0.0))
    ampOscFreq = float(params.get('ampOscFreq', 0.0))

    # --- Basic Setup ---
    N = int(sample_rate * duration)
    if N <= 0: return np.zeros((0, 2), dtype=np.float32)
    t_abs = np.linspace(0, duration, N, endpoint=False)

    # --- Frequency Calculation ---
    # Ensure frequencies are non-negative
    freq_lower = np.maximum(0.0, baseFreq - beatFreq / 2.0)
    freq_upper = np.maximum(0.0, baseFreq + beatFreq / 2.0)
    freq_lower_array = np.full(N, freq_lower)
    freq_upper_array = np.full(N, freq_upper)

    # --- Phase Calculation ---
    # Calculate instantaneous phase for each frequency component
    if N > 1:
        dt = np.diff(t_abs, prepend=t_abs[0]) # Time difference between samples
    elif N == 1:
        dt = np.array([duration])
    else: # N == 0
        dt = np.array([])

    phase_lower_base = np.cumsum(2 * np.pi * freq_lower_array * dt)
    phase_upper_base = np.cumsum(2 * np.pi * freq_upper_array * dt)

    # Optional phase oscillation between the two tones
    phase_osc_term = (phaseOscRange_rad / 2.0) * np.sin(2 * np.pi * phaseOscFreq * t_abs)

    # Total phase for each tone, including start phase and oscillation
    phase_lower_total = phase_lower_base + startPhaseLower_rad - phase_osc_term
    phase_upper_total = phase_upper_base + startPhaseUpper_rad + phase_osc_term

    # --- Generate Individual Sine Waves ---
    s_lower = np.sin(phase_lower_total)
    s_upper = np.sin(phase_upper_total)

    # --- Apply Individual Amplitudes and Sum for Each Channel ---
    # Note: No division by 2 here, as amplitudes control the mix.
    s_L = (s_lower * amp_lower_L) + (s_upper * amp_upper_L)
    s_R = (s_lower * amp_lower_R) + (s_upper * amp_upper_R)

    # --- Amplitude Modulation (Tremolo) ---
    # Applied to the combined signal in each channel
    # LFO form: (1 - depth/2 + (depth/2)*sin) ensures max amp is 1 when depth=0
    # and max amp is 1.5 when depth=1, max amp is 2 when depth=2.
    # Clamp depth to avoid negative modulation values.
    clamped_ampOscDepth = np.clip(ampOscDepth, 0.0, 2.0)
    amp_mod = (1.0 - clamped_ampOscDepth / 2.0) + \
              (clamped_ampOscDepth / 2.0) * np.sin(2 * np.pi * ampOscFreq * t_abs)

    # Apply modulation to each channel
    s_L_mod = s_L * amp_mod
    s_R_mod = s_R * amp_mod

    # --- Create Stereo Output ---
    # Combine left and right channels
    # Clip final signal to [-1, 1] to prevent potential clipping issues
    # due to summing and modulation.
    audio_L = np.clip(s_L_mod, -1.0, 1.0)
    audio_R = np.clip(s_R_mod, -1.0, 1.0)

    audio = np.column_stack((audio_L, audio_R)).astype(np.float32)

    return audio

def binaural_beat_transition(duration, sample_rate=44100, **params):
    """
    Generates a binaural beat with linearly transitioning parameters,
    including independent L/R base amplitude, phase modulation,
    and amplitude modulation.
    """
    # --- Get Start/End Parameters ---
    startAmpL = float(params.get('startAmpL', 0.5))
    endAmpL = float(params.get('endAmpL', startAmpL))
    startAmpR = float(params.get('startAmpR', 0.5))
    endAmpR = float(params.get('endAmpR', startAmpR))

    startBaseFreq = float(params.get('startBaseFreq', 200.0))
    endBaseFreq = float(params.get('endBaseFreq', startBaseFreq))
    startBeatFreq = float(params.get('startBeatFreq', 4.0))
    endBeatFreq = float(params.get('endBeatFreq', startBeatFreq))

    startStartPhaseL = float(params.get('startStartPhaseL', 0.0))
    endStartPhaseL = float(params.get('endStartPhaseL', startStartPhaseL))
    startStartPhaseR = float(params.get('startStartPhaseR', 0.0))
    endStartPhaseR = float(params.get('endStartPhaseR', startStartPhaseR))

    startPhaseOscFreq = float(params.get('startPhaseOscFreq', 0.0))
    endPhaseOscFreq = float(params.get('endPhaseOscFreq', startPhaseOscFreq))
    startPhaseOscRange = float(params.get('startPhaseOscRange', 0.0))
    endPhaseOscRange = float(params.get('endPhaseOscRange', startPhaseOscRange))

    startAmpOscDepthL = float(params.get('startAmpOscDepthL', 0.0))
    endAmpOscDepthL = float(params.get('endAmpOscDepthL', startAmpOscDepthL))
    startAmpOscFreqL = float(params.get('startAmpOscFreqL', 0.0))
    endAmpOscFreqL = float(params.get('endAmpOscFreqL', startAmpOscFreqL))

    startAmpOscDepthR = float(params.get('startAmpOscDepthR', 0.0))
    endAmpOscDepthR = float(params.get('endAmpOscDepthR', startAmpOscDepthR))
    startAmpOscFreqR = float(params.get('startAmpOscFreqR', 0.0))
    endAmpOscFreqR = float(params.get('endAmpOscFreqR', startAmpOscFreqR))

    # --- Basic Setup ---
    N = int(sample_rate * duration)
    if N <= 0: return np.zeros((0, 2))
    t_abs = np.linspace(0, duration, N, endpoint=False)

    # --- Interpolate Parameters ---
    ampL_array = np.linspace(startAmpL, endAmpL, N) # Base Amp L
    ampR_array = np.linspace(startAmpR, endAmpR, N) # Base Amp R
    base_freq_array = np.linspace(startBaseFreq, endBaseFreq, N)
    beat_freq_array = np.linspace(startBeatFreq, endBeatFreq, N)
    startPhaseL_array = np.linspace(startStartPhaseL, endStartPhaseL, N) # Initial phase offset for L
    startPhaseR_array = np.linspace(startStartPhaseR, endStartPhaseR, N) # Initial phase offset for R
    phaseOscFreq_array = np.linspace(startPhaseOscFreq, endPhaseOscFreq, N)
    phaseOscRange_array = np.linspace(startPhaseOscRange, endPhaseOscRange, N)
    ampOscDepthL_array = np.linspace(startAmpOscDepthL, endAmpOscDepthL, N)
    ampOscFreqL_array = np.linspace(startAmpOscFreqL, endAmpOscFreqL, N)
    ampOscDepthR_array = np.linspace(startAmpOscDepthR, endAmpOscDepthR, N)
    ampOscFreqR_array = np.linspace(startAmpOscFreqR, endAmpOscFreqR, N)

    # --- Frequency Calculation (Time-Varying) ---
    half_beat_freq_array = beat_freq_array / 2.0
    left_freq_array = np.maximum(0.0, base_freq_array - half_beat_freq_array)
    right_freq_array = np.maximum(0.0, base_freq_array + half_beat_freq_array)

    # --- Phase Calculation (Time-Varying) ---
    if N > 1: dt = np.diff(t_abs, prepend=t_abs[0])
    elif N == 1: dt = np.array([duration])
    else: dt = np.array([])

    # Integrate instantaneous frequency for base phase
    phase_left_base = np.cumsum(2 * np.pi * left_freq_array * dt)
    phase_right_base = np.cumsum(2 * np.pi * right_freq_array * dt)

    # Integrate instantaneous phase oscillation frequency for its phase argument
    phase_osc_phase_arg = np.cumsum(2 * np.pi * phaseOscFreq_array * dt)
    phase_osc_term = (phaseOscRange_array / 2.0) * np.sin(phase_osc_phase_arg)

    # Combine: base phase + initial offset + oscillation term
    phase_left_total = phase_left_base + startPhaseL_array - phase_osc_term
    phase_right_total = phase_right_base + startPhaseR_array + phase_osc_term

    # --- Independent Amplitude Modulation (Time-Varying) ---
    # Integrate instantaneous amp oscillation frequency for LFO phase arguments
    amp_osc_phase_arg_l = np.cumsum(2 * np.pi * ampOscFreqL_array * dt)
    amp_osc_phase_arg_r = np.cumsum(2 * np.pi * ampOscFreqR_array * dt)

    # Calculate time-varying LFO using the simpler form
    amp_mod_l = (1.0 - ampOscDepthL_array / 2.0) + (ampOscDepthL_array / 2.0) * np.sin(amp_osc_phase_arg_l)
    amp_mod_r = (1.0 - ampOscDepthR_array / 2.0) + (ampOscDepthR_array / 2.0) * np.sin(amp_osc_phase_arg_r)

    # --- Signal Generation ---
    s_left_base = np.sin(phase_left_total)
    s_right_base = np.sin(phase_right_total)

    # Apply independent amplitude modulation AND independent interpolated base amplitude
    s_left = s_left_base * amp_mod_l * ampL_array
    s_right = s_right_base * amp_mod_r * ampR_array

    # Combine into stereo signal
    audio = np.column_stack((s_left, s_right)).astype(np.float32)
    return audio


def isochronic_tone(duration, sample_rate=44100, **params):
    amp = float(params.get('amp', 0.5))
    baseFreq = float(params.get('baseFreq', 200.0))
    beatFreq = float(params.get('beatFreq', 4.0)) # Note: For isochronic, this is the pulse rate
    rampPercent = float(params.get('rampPercent', 0.2)) # Duration of ramp up/down as % of pulse ON time
    gapPercent = float(params.get('gapPercent', 0.15)) # Duration of silence as % of total cycle time
    pan = float(params.get('pan', 0.0)) # Panning (-1 L, 0 C, 1 R)

    N = int(sample_rate * duration)
    if N <= 0:
        return np.zeros((0, 2))

    t_abs = np.linspace(0, duration, N, endpoint=False)

    # --- Carrier Wave ---
    instantaneous_carrier_freq = np.maximum(0.0, baseFreq) # Ensure non-negative
    carrier_freq_array = np.full(N, instantaneous_carrier_freq)

    if N > 1: dt = np.diff(t_abs, prepend=t_abs[0])
    elif N == 1: dt = np.array([duration])
    else: dt = np.array([])

    carrier_phase = np.cumsum(2 * np.pi * carrier_freq_array * dt)
    carrier = np.sin(carrier_phase)

    # --- Isochronic Envelope ---
    instantaneous_beat_freq = np.maximum(0.0, beatFreq) # Pulse rate, ensure non-negative
    beat_freq_array = np.full(N, instantaneous_beat_freq)

    # Calculate cycle length (period) of the pulse rate
    cycle_len_array = np.zeros_like(beat_freq_array)
    valid_beat_mask = beat_freq_array > 1e-9 # Avoid division by zero

    # Use np.errstate to suppress division by zero warnings for the masked elements
    with np.errstate(divide='ignore', invalid='ignore'):
        cycle_len_array[valid_beat_mask] = 1.0 / beat_freq_array[valid_beat_mask]
        # For beat_freq == 0, cycle_len_array remains 0

    # Calculate phase within the isochronic cycle (0 to 1 represents one full cycle)
    beat_phase_cycles = np.cumsum(beat_freq_array * dt)

    # Calculate time within the current cycle (handling beat_freq=0 where cycle_len=0)
    # Use modulo 1.0 on the cycle phase, then scale by cycle length
    t_in_cycle = np.mod(beat_phase_cycles, 1.0) * cycle_len_array
    t_in_cycle[~valid_beat_mask] = 0.0 # Set time to 0 if beat freq is zero

    # Generate the trapezoid envelope based on time within cycle
    iso_env = trapezoid_envelope_vectorized(t_in_cycle, cycle_len_array, rampPercent, gapPercent)

    # Apply envelope to carrier
    mono_signal = carrier * iso_env

    # Apply overall amplitude
    output_mono = mono_signal * amp

    # Pan the result
    audio = pan2(output_mono, pan=pan)

    # Note: Volume envelope (like ADSR/Linen) is applied *within* generate_voice_audio if specified there.
    # The trapezoid envelope is inherent to the isochronic tone generation itself.

    return audio.astype(np.float32)


def isochronic_tone_transition(duration, sample_rate=44100, **params):
    amp = float(params.get('amp', 0.5)) # Constant amplitude for the voice
    startBaseFreq = float(params.get('startBaseFreq', 200.0))
    endBaseFreq = float(params.get('endBaseFreq', startBaseFreq)) # Default end to start
    startBeatFreq = float(params.get('startBeatFreq', 4.0)) # Start pulse rate
    endBeatFreq = float(params.get('endBeatFreq', startBeatFreq)) # End pulse rate
    rampPercent = float(params.get('rampPercent', 0.2)) # Constant ramp %
    gapPercent = float(params.get('gapPercent', 0.15)) # Constant gap %
    pan = float(params.get('pan', 0.0)) # Constant panning

    N = int(sample_rate * duration)
    if N <= 0:
        return np.zeros((0, 2))

    t_abs = np.linspace(0, duration, N, endpoint=False)

    # --- Interpolate Frequencies ---
    base_freq_array = np.linspace(startBaseFreq, endBaseFreq, N)
    beat_freq_array = np.linspace(startBeatFreq, endBeatFreq, N) # Pulse rate array

    # Ensure frequencies are non-negative
    instantaneous_carrier_freq_array = np.maximum(0.0, base_freq_array)
    instantaneous_beat_freq_array = np.maximum(0.0, beat_freq_array)

    # --- Carrier Wave (Time-Varying Frequency) ---
    if N > 1: dt = np.diff(t_abs, prepend=t_abs[0])
    elif N == 1: dt = np.array([duration])
    else: dt = np.array([])

    carrier_phase = np.cumsum(2 * np.pi * instantaneous_carrier_freq_array * dt)
    carrier = np.sin(carrier_phase)

    # --- Isochronic Envelope (Time-Varying Pulse Rate) ---
    # Calculate time-varying cycle length
    cycle_len_array = np.zeros_like(instantaneous_beat_freq_array)
    valid_beat_mask = instantaneous_beat_freq_array > 1e-9

    with np.errstate(divide='ignore', invalid='ignore'):
        cycle_len_array[valid_beat_mask] = 1.0 / instantaneous_beat_freq_array[valid_beat_mask]

    # Calculate phase within the isochronic cycle (using time-varying pulse rate)
    beat_phase_cycles = np.cumsum(instantaneous_beat_freq_array * dt)

    # Calculate time within the current cycle (using time-varying cycle length)
    t_in_cycle = np.mod(beat_phase_cycles, 1.0) * cycle_len_array
    t_in_cycle[~valid_beat_mask] = 0.0

    # Generate the trapezoid envelope (using constant ramp/gap percentages but time-varying cycle length)
    iso_env = trapezoid_envelope_vectorized(t_in_cycle, cycle_len_array, rampPercent, gapPercent)

    # Apply envelope to carrier
    mono_signal = carrier * iso_env

    # Apply overall amplitude
    output_mono = mono_signal * amp

    # Pan the result
    audio = pan2(output_mono, pan=pan)

    # Note: Volume envelope (like ADSR/Linen) is applied *within* generate_voice_audio if specified there.

    return audio.astype(np.float32)


# --- Flanger (Example - Assuming it exists and is complex) ---
# Placeholder - Replace with your actual flanger implementation if needed
def _flanger_effect_stereo_continuous(*args, **kwargs):
     print("Warning: _flanger_effect_stereo_continuous is a placeholder.")
     # Need N samples to return correct shape silence
     duration = kwargs.get('duration', 1.0)
     sample_rate = kwargs.get('sample_rate', 44100)
     N = int(duration * sample_rate)
     return np.zeros((N, 2))

def flanged_voice(duration, sample_rate=44100, **params):
    print("Warning: flanged_voice is using a placeholder effect.")
    # Example: Generate noise and apply placeholder flanger
    amp = float(params.get('amp', 0.5))
    noiseType = int(params.get('noiseType', 1)) # 1=W, 2=P, 3=B (example)

    N = int(duration * sample_rate)
    if N <= 0: return np.zeros((0, 2))

    # Generate base noise (simplified placeholders)
    if noiseType == 1: # White
        noise_mono = (np.random.rand(N) * 2 - 1) * amp
    elif noiseType == 2: # Pink (very basic approximation)
        # This is NOT a proper pink noise generator, just placeholder
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        # Pad white noise for filter transient
        wn = np.random.randn(N + len(a))
        # Apply filter (requires scipy.signal.lfilter)
        # For now, just use white noise as placeholder if scipy not imported
        try:
           from scipy.signal import lfilter
           filtered_noise = lfilter(b, a, wn)[len(a):] # Apply filter and remove transient
           noise_mono = filtered_noise / np.max(np.abs(filtered_noise)) * amp if np.max(np.abs(filtered_noise)) > 0 else np.zeros(N)
        except ImportError:
            print("Warning: SciPy not found, using white noise for pink noise placeholder.")
            noise_mono = (np.random.rand(N) * 2 - 1) * amp

    elif noiseType == 3: # Brown (basic approximation)
        wn = (np.random.rand(N) * 2 - 1)
        noise_mono = np.cumsum(wn)
        noise_mono = noise_mono / np.max(np.abs(noise_mono)) * amp if np.max(np.abs(noise_mono)) > 0 else np.zeros(N)
    else:
        noise_mono = np.zeros(N)

    # Apply placeholder flanger effect
    # Pass necessary params from the input 'params' dict
    flanger_params = {k: v for k, v in params.items() if k not in ['amp', 'noiseType']} # Pass other params
    flanger_params['duration'] = duration # Ensure duration/sr are passed if needed
    flanger_params['sample_rate'] = sample_rate
    audio_stereo = _flanger_effect_stereo_continuous(noise_mono, **flanger_params)

    # Ensure output is stereo float32
    if audio_stereo.ndim == 1:
        audio_stereo = np.column_stack((audio_stereo, audio_stereo))
    elif audio_stereo.shape[1] == 1:
         audio_stereo = np.column_stack((audio_stereo[:,0], audio_stereo[:,0]))

    return audio_stereo.astype(np.float32)


# -----------------------------------------------------------------------------
# Crossfade and Assembly Logic
# -----------------------------------------------------------------------------

def crossfade_signals(signal_a, signal_b, sample_rate, transition_duration):
    """
    Crossfades two stereo signals. Assumes signal_a fades out, signal_b fades in.
    Operates on the initial segments of the signals up to transition_duration.
    Returns the blended segment.
    """
    n_samples = int(transition_duration * sample_rate)
    if n_samples <= 0:
        # No crossfade duration, return silence or handle appropriately
        return np.zeros((0, 2))

    # Determine the actual number of samples available for crossfade
    len_a = signal_a.shape[0]
    len_b = signal_b.shape[0]
    actual_crossfade_samples = min(n_samples, len_a, len_b)

    if actual_crossfade_samples <= 0:
        print(f"Warning: Crossfade not possible or zero length. Samples: {n_samples}, SigA: {len_a}, SigB: {len_b}")
        # Return an empty array matching the expected dimensions if no crossfade happens
        return np.zeros((0, 2))

    # Ensure signals are 2D stereo (N, 2) before slicing
    def ensure_stereo(sig):
        if sig.ndim == 1: sig = np.column_stack((sig, sig)) # Mono to Stereo
        elif sig.shape[1] == 1: sig = np.column_stack((sig[:,0], sig[:,0])) # (N, 1) to (N, 2)
        if sig.shape[1] != 2: raise ValueError("Signal must be stereo (N, 2) for crossfade.")
        return sig

    try:
        signal_a = ensure_stereo(signal_a)
        signal_b = ensure_stereo(signal_b)
    except ValueError as e:
        print(f"Error in crossfade_signals: {e}")
        return np.zeros((0, 2)) # Return empty on error

    # Take only the required number of samples for crossfade
    signal_a_seg = signal_a[:actual_crossfade_samples]
    signal_b_seg = signal_b[:actual_crossfade_samples]

    # Linear crossfade ramp (can be replaced with equal power: np.sqrt(fade))
    fade_out = np.linspace(1, 0, actual_crossfade_samples)[:, None] # Column vector for broadcasting
    fade_in = np.linspace(0, 1, actual_crossfade_samples)[:, None]

    # Apply fades and sum
    blended_segment = signal_a_seg * fade_out + signal_b_seg * fade_in
    return blended_segment


# Dictionary mapping function names (strings) to actual functions
# --- UPDATED SYNTH_FUNCTIONS DICTIONARY ---
# Exclude helper/internal functions explicitly
_EXCLUDED_FUNCTION_NAMES = [
    'validate_float', 'validate_int', 'butter_bandpass', 'bandpass_filter',
    'butter_bandstop', 'bandreject_filter', 'lowpass_filter', 'pink_noise',
    'brown_noise', 'sine_wave', 'sine_wave_varying', 'adsr_envelope',
    'create_linear_fade_envelope', 'linen_envelope', 'pan2', 'safety_limiter',
    'crossfade_signals', 'assemble_track_from_data', 'generate_voice_audio',
    'load_track_from_json', 'save_track_to_json', 'generate_audio', 'get_synth_params',
    'trapezoid_envelope_vectorized', '_flanger_effect_stereo_continuous', 'butter',
    'lfilter', 'write', 'ensure_stereo',
    # Standard library functions that might be imported
    'json', 'inspect', 'os', 'traceback', 'math', 'copy'
]

SYNTH_FUNCTIONS = {}
try:
    current_module = __import__(__name__)
    for name, obj in inspect.getmembers(current_module):
        if inspect.isfunction(obj) and name not in _EXCLUDED_FUNCTION_NAMES and not name.startswith('_'):
            # Further check if the function is defined in *this* module, not imported
            if inspect.getmodule(obj) == current_module:
                SYNTH_FUNCTIONS[name] = obj
except Exception as e:
    print(f"Error inspecting functions: {e}")

print(f"Detected Synth Functions: {list(SYNTH_FUNCTIONS.keys())}")


def get_synth_params(func_name):
    """Gets parameter names and default values for a synth function by inspecting its signature."""
    if func_name not in SYNTH_FUNCTIONS:
        print(f"Warning: Function '{func_name}' not found in SYNTH_FUNCTIONS.")
        return {}

    func = SYNTH_FUNCTIONS[func_name]
    params = {}
    try:
        sig = inspect.signature(func)
        for name, param in sig.parameters.items():
            # Skip standard args and the catch-all kwargs
            if name in ['duration', 'sample_rate'] or param.kind == inspect.Parameter.VAR_KEYWORD:
                continue

            # Store the default value if it exists, otherwise store inspect._empty
            params[name] = param.default # Keep _empty to distinguish from None default

    except Exception as e:
        print(f"Error inspecting signature for '{func_name}': {e}")
        # Fallback to trying source code parsing if signature fails? Or just return empty?
        # For now, return empty on inspection error. Source parsing is done in GUI.
        return {}

    return params


def generate_voice_audio(voice_data, duration, sample_rate, global_start_time):
    """Generates audio for a single voice based on its definition."""
    func_name = voice_data.get("synth_function_name")
    params = voice_data.get("params", {})
    is_transition = voice_data.get("is_transition", False) # Check if this step IS a transition

    # --- Select the correct function (static or transition) ---
    actual_func_name = func_name
    selected_func_is_transition_type = func_name and func_name.endswith("_transition")

    # Determine the function to actually call based on 'is_transition' flag
    if is_transition:
        if not selected_func_is_transition_type:
            transition_func_name = func_name + "_transition"
            if transition_func_name in SYNTH_FUNCTIONS:
                actual_func_name = transition_func_name
                print(f"Note: Step marked as transition, using '{actual_func_name}' instead of base '{func_name}'.")
            else:
                print(f"Warning: Step marked as transition, but transition function '{transition_func_name}' not found for base '{func_name}'. Using static version '{func_name}'. Parameters might mismatch.")
                # Keep actual_func_name as func_name (the static one)
    else: # Not a transition step
        if selected_func_is_transition_type:
            base_func_name = func_name.replace("_transition", "")
            if base_func_name in SYNTH_FUNCTIONS:
                actual_func_name = base_func_name
                print(f"Note: Step not marked as transition, using base function '{actual_func_name}' instead of selected '{func_name}'.")
            else:
                print(f"Warning: Step not marked as transition, selected '{func_name}', but base function '{base_func_name}' not found. Using selected '{func_name}'. Parameters might mismatch.")
                # Keep actual_func_name as func_name (the transition one user selected)

    if not actual_func_name or actual_func_name not in SYNTH_FUNCTIONS:
        print(f"Error: Synth function '{actual_func_name}' (derived from '{func_name}') not found or invalid.")
        N = int(duration * sample_rate)
        return np.zeros((N, 2))

    synth_func = SYNTH_FUNCTIONS[actual_func_name]

    # Clean params: remove None values before passing to function, as functions use .get() with defaults
    cleaned_params = {k: v for k, v in params.items() if v is not None}

    # --- Generate base audio ---
    try:
        print(f"  Calling: {actual_func_name}(duration={duration}, sample_rate={sample_rate}, **{cleaned_params})")
        audio = synth_func(duration=duration, sample_rate=sample_rate, **cleaned_params)
    except Exception as e:
        print(f"Error calling synth function '{actual_func_name}' with params {cleaned_params}:")
        traceback.print_exc()
        N = int(duration * sample_rate)
        return np.zeros((N, 2))

    if audio is None:
        print(f"Error: Synth function '{actual_func_name}' returned None.")
        N = int(duration * sample_rate)
        return np.zeros((N, 2))

    # --- Apply volume envelope if defined ---
    envelope_data = voice_data.get("volume_envelope")
    N_audio = audio.shape[0]
    # Ensure t_rel matches audio length, especially if N calculation differs slightly
    t_rel = np.linspace(0, duration, N_audio, endpoint=False) if N_audio > 0 else np.array([])
    env = np.ones(N_audio) # Default flat envelope

    if envelope_data and isinstance(envelope_data, dict) and N_audio > 0:
        env_type = envelope_data.get("type")
        env_params = envelope_data.get("params", {})
        cleaned_env_params = {k: v for k, v in env_params.items() if v is not None}

        try:
            # Pass duration and sample_rate if needed by envelope func
            if 'duration' not in cleaned_env_params: cleaned_env_params['duration'] = duration
            if 'sample_rate' not in cleaned_env_params: cleaned_env_params['sample_rate'] = sample_rate
            
            print(f"Debugging: cleaned_env_params = {cleaned_env_params}")
            if env_type == "adsr":
                env = adsr_envelope(t_rel, **cleaned_env_params)
            elif env_type == "linen":
                 env = linen_envelope(t_rel, cleaned_env_params['attack'], cleaned_env_params['release'])
            elif env_type == "linear_fade":
                 # This function uses duration/sr internally, ensure they are passed if needed
                 required = ['fade_duration', 'start_amp', 'end_amp']
                 # Check params specific to linear_fade
                 specific_env_params = {k: v for k, v in cleaned_env_params.items() if k in required}
                 if all(p in specific_env_params for p in required):
                      # Pass the main duration and sample rate, not t_rel
                      env = create_linear_fade_envelope(duration, sample_rate, **specific_env_params)
                      # Resample envelope if its length doesn't match audio
                      if len(env) != N_audio:
                            print(f"Warning: Resampling '{env_type}' envelope from {len(env)} to {N_audio} samples.")
                            if len(env) > 0:
                                 env = np.interp(np.linspace(0, 1, N_audio), np.linspace(0, 1, len(env)), env)
                            else:
                                 env = np.ones(N_audio) # Fallback
                 else:
                      print(f"Warning: Missing parameters for 'linear_fade' envelope. Using flat envelope. Got: {specific_env_params}")
            # Add other envelope types here
            # elif env_type == "other_env":
            #    env = other_env_function(t_rel, **cleaned_env_params)
            else:
                print(f"Warning: Unknown envelope type '{env_type}'. Using flat envelope.")

            # Ensure envelope is broadcastable (N,)
            if env.shape != (N_audio,):
                 print(f"Warning: Envelope shape mismatch ({env.shape} vs {(N_audio,)}). Attempting reshape.")
                 if len(env) == N_audio: env = env.reshape(N_audio)
                 else:
                      print("Error: Cannot reshape envelope. Using flat envelope.")
                      env = np.ones(N_audio) # Fallback

        except Exception as e:
            print(f"Error creating envelope type '{env_type}':")
            traceback.print_exc()
            env = np.ones(N_audio) # Fallback

    # Apply the calculated envelope
    try:
        if audio.ndim == 2 and audio.shape[1] == 2 and len(env) == audio.shape[0]:
             audio = audio * env[:, np.newaxis] # Apply envelope element-wise to stereo
        elif audio.ndim == 1 and len(env) == len(audio): # Handle potential mono output from synth
             audio = audio * env
        elif N_audio == 0:
             pass # No audio to apply envelope to
        else:
             print(f"Error: Envelope length ({len(env)}) or audio shape ({audio.shape}) mismatch. Skipping envelope application.")
    except Exception as e:
        print(f"Error applying envelope to audio:")
        traceback.print_exc()


    # --- Ensure output is stereo ---
    if audio.ndim == 1:
        print(f"Note: Synth function '{actual_func_name}' resulted in mono audio. Panning.")
        pan_val = cleaned_params.get('pan', 0.0) # Assume pan param exists or default to center
        audio = pan2(audio, pan_val)
    elif audio.ndim == 2 and audio.shape[1] == 1:
        print(f"Note: Synth function '{actual_func_name}' resulted in mono audio (N, 1). Panning.")
        pan_val = cleaned_params.get('pan', 0.0)
        audio = pan2(audio[:,0], pan_val) # Extract the single column before panning

    # Final check for shape (N, 2)
    if not (audio.ndim == 2 and audio.shape[1] == 2):
          if N_audio == 0: return np.zeros((0, 2)) # Handle zero duration case gracefully
          else:
                print(f"Error: Final audio shape for voice is incorrect ({audio.shape}). Returning silence.")
                N_expected = int(duration * sample_rate)
                return np.zeros((N_expected, 2))


    # Add a small default fade if no specific envelope was requested and audio exists
    # This helps prevent clicks when steps are concatenated without crossfade
    if not envelope_data and N_audio > 0:
        fade_len = min(N_audio, int(0.01 * sample_rate)) # 10ms fade or audio length
        if fade_len > 1:
             fade_in = np.linspace(0, 1, fade_len)
             fade_out = np.linspace(1, 0, fade_len)
             # Apply fade using broadcasting
             audio[:fade_len] *= fade_in[:, np.newaxis]
             audio[-fade_len:] *= fade_out[:, np.newaxis]

    return audio.astype(np.float32) # Ensure float32 output


def assemble_track_from_data(track_data, sample_rate, crossfade_duration):
    """
    Assembles a track from a track_data dictionary.
    Uses crossfading between steps by overlapping their placement.
    Includes per-step normalization to prevent excessive peaks before final mix.
    """
    steps_data = track_data.get("steps", [])
    if not steps_data:
        print("Warning: No steps found in track data.")
        return np.zeros((sample_rate, 2)) # Return 1 second silence

    # --- Calculate Track Length Estimation ---
    estimated_total_duration = sum(float(step.get("duration", 0)) for step in steps_data)
    if estimated_total_duration <= 0:
        print("Warning: Track has zero or negative estimated total duration.")
        return np.zeros((sample_rate, 2))

    # Add buffer for potential rounding errors and final sample
    estimated_total_samples = int(estimated_total_duration * sample_rate) + sample_rate
    track = np.zeros((estimated_total_samples, 2), dtype=np.float32) # Use float32

    # --- Time and Sample Tracking ---
    current_time = 0.0 # Start time for the *next* step to be placed
    last_step_end_sample_in_track = 0 # Tracks the actual last sample index used

    crossfade_samples = int(crossfade_duration * sample_rate)
    if crossfade_samples < 0: crossfade_samples = 0

    print(f"Assembling track: {len(steps_data)} steps, Est. Max Duration: {estimated_total_duration:.2f}s, Crossfade: {crossfade_duration:.2f}s ({crossfade_samples} samples)")

    for i, step_data in enumerate(steps_data):
        step_duration = float(step_data.get("duration", 0))
        if step_duration <= 0:
            print(f"Skipping step {i+1} with zero or negative duration.")
            continue

        # --- Calculate Placement Indices ---
        step_start_sample_abs = int(current_time * sample_rate)
        N_step = int(step_duration * sample_rate)
        step_end_sample_abs = step_start_sample_abs + N_step

        print(f"  Processing Step {i+1}: Place Start: {current_time:.2f}s ({step_start_sample_abs}), Duration: {step_duration:.2f}s, Samples: {N_step}")

        # Generate audio for all voices in this step and mix them
        step_audio_mix = np.zeros((N_step, 2), dtype=np.float32) # Use float32
        voices_data = step_data.get("voices", [])

        if not voices_data:
            print(f"        Warning: Step {i+1} has no voices.")
        else:
            num_voices_in_step = len(voices_data)
            print(f"        Mixing {num_voices_in_step} voice(s) for Step {i+1}...")
            for j, voice_data in enumerate(voices_data):
                func_name_short = voice_data.get('synth_function_name', 'UnknownFunc')
                print(f"          Generating Voice {j+1}/{num_voices_in_step}: {func_name_short}")
                voice_audio = generate_voice_audio(voice_data, step_duration, sample_rate, current_time)

                # Add generated audio if valid
                if voice_audio is not None and voice_audio.shape[0] == N_step and voice_audio.ndim == 2 and voice_audio.shape[1] == 2:
                    step_audio_mix += voice_audio # Sum voices
                elif voice_audio is not None:
                    print(f"          Error: Voice {j+1} ({func_name_short}) generated audio shape mismatch ({voice_audio.shape} vs {(N_step, 2)}). Skipping voice.")

            # --- *** NEW: Per-Step Normalization/Limiting *** ---
            # Check the peak of the mixed step audio
            step_peak = np.max(np.abs(step_audio_mix))
            # Define a threshold slightly above 1.0 to allow headroom but prevent extreme peaks
            step_normalization_threshold = 1.0
            if step_peak > step_normalization_threshold:
                print(f"        Normalizing Step {i+1} mix (peak={step_peak:.3f}) down to {step_normalization_threshold:.2f}")
                step_audio_mix *= (step_normalization_threshold / step_peak)
            # --- *** End Per-Step Normalization *** ---


        # --- Placement and Crossfading ---
        # Clip placement indices to the allocated track buffer boundaries
        safe_place_start = max(0, step_start_sample_abs)
        safe_place_end = min(estimated_total_samples, step_end_sample_abs)
        segment_len_in_track = safe_place_end - safe_place_start

        if segment_len_in_track <= 0:
            print(f"        Skipping Step {i+1} placement (no valid range in track buffer).")
            continue

        # Determine the portion of step_audio_mix to use
        audio_to_use = step_audio_mix[:segment_len_in_track]

        # Double check length (should normally match)
        if audio_to_use.shape[0] != segment_len_in_track:
            print(f"        Warning: Step {i+1} audio length adjustment needed ({audio_to_use.shape[0]} vs {segment_len_in_track}). Padding/Truncating.")
            if audio_to_use.shape[0] < segment_len_in_track:
                audio_to_use = np.pad(audio_to_use, ((0, segment_len_in_track - audio_to_use.shape[0]), (0,0)), 'constant')
            else:
                audio_to_use = audio_to_use[:segment_len_in_track]

        # --- Actual Crossfade Logic ---
        overlap_start_sample_in_track = safe_place_start
        overlap_end_sample_in_track = min(safe_place_end, last_step_end_sample_in_track)
        overlap_samples = overlap_end_sample_in_track - overlap_start_sample_in_track

        can_crossfade = i > 0 and overlap_samples > 0 and crossfade_samples > 0

        if can_crossfade:
            actual_crossfade_samples = min(overlap_samples, crossfade_samples)
            print(f"        Crossfading Step {i+1} with previous. Overlap: {overlap_samples / sample_rate:.3f}s, Actual CF: {actual_crossfade_samples / sample_rate:.3f}s")

            if actual_crossfade_samples > 0:
                # Get segments for crossfading
                prev_segment = track[overlap_start_sample_in_track : overlap_start_sample_in_track + actual_crossfade_samples]
                new_segment = audio_to_use[:actual_crossfade_samples]

                # Perform crossfade
                blended_segment = crossfade_signals(prev_segment, new_segment, sample_rate, actual_crossfade_samples / sample_rate)

                # Place blended segment (overwrite previous tail)
                track[overlap_start_sample_in_track : overlap_start_sample_in_track + actual_crossfade_samples] = blended_segment

                # Add the remainder of the new step (after the crossfaded part)
                remaining_start_index_in_step_audio = actual_crossfade_samples
                remaining_start_index_in_track = overlap_start_sample_in_track + actual_crossfade_samples
                remaining_end_index_in_track = safe_place_end

                if remaining_start_index_in_track < remaining_end_index_in_track:
                    num_remaining_samples_to_add = remaining_end_index_in_track - remaining_start_index_in_track
                    if remaining_start_index_in_step_audio < audio_to_use.shape[0]:
                        remaining_audio_from_step = audio_to_use[remaining_start_index_in_step_audio : remaining_start_index_in_step_audio + num_remaining_samples_to_add]
                        # Add the remaining part (use += as it might overlap with the next step's fade-in region)
                        track[remaining_start_index_in_track : remaining_start_index_in_track + remaining_audio_from_step.shape[0]] += remaining_audio_from_step

            else: # Overlap existed but calculated crossfade samples was zero
                 print(f"        Placing Step {i+1} without crossfade (actual_crossfade_samples=0). Adding.")
                 track[safe_place_start:safe_place_end] += audio_to_use # Add instead of overwrite

        else: # No crossfade (first step or no overlap)
            print(f"        Placing Step {i+1} without crossfade. Adding.")
            # Add the audio (use += because the space might be overlapped by the *next* step's fade)
            track[safe_place_start:safe_place_end] += audio_to_use

        # --- Update Markers for Next Loop ---
        last_step_end_sample_in_track = max(last_step_end_sample_in_track, safe_place_end)
        # Advance current_time for the START of the next step, pulling back by crossfade duration
        effective_advance_duration = max(0.0, step_duration - crossfade_duration) if crossfade_samples > 0 else step_duration
        current_time += effective_advance_duration


    # --- Final Trimming ---
    final_track_samples = last_step_end_sample_in_track
    if final_track_samples <= 0:
        print("Warning: Final track assembly resulted in zero length.")
        return np.zeros((sample_rate, 2))

    if final_track_samples < track.shape[0]:
        track = track[:final_track_samples]
    elif final_track_samples > estimated_total_samples:
         print(f"Warning: Final track samples ({final_track_samples}) exceeded initial estimate ({estimated_total_samples}).")

    print(f"Track assembly finished. Final Duration: {track.shape[0] / sample_rate:.2f}s")
    return track


# -----------------------------------------------------------------------------
# JSON Loading/Saving
# -----------------------------------------------------------------------------

# Custom JSON encoder to handle numpy types (if needed)
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def load_track_from_json(filepath):
    """Loads track definition from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            track_data = json.load(f)
        print(f"Track data loaded successfully from {filepath}")
        # Basic validation
        if not isinstance(track_data, dict) or \
           "global_settings" not in track_data or \
           "steps" not in track_data or \
           not isinstance(track_data["steps"], list) or \
           not isinstance(track_data["global_settings"], dict):
            print("Error: Invalid JSON structure. Missing 'global_settings' dict or 'steps' list.")
            return None
        # Further validation could check step/voice structure if needed
        return track_data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {filepath}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred loading {filepath}:")
        traceback.print_exc()
        return None

def save_track_to_json(track_data, filepath):
    """Saves track definition to a JSON file."""
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(track_data, f, indent=4, cls=NumpyEncoder)
        print(f"Track data saved successfully to {filepath}")
        return True
    except IOError as e:
        print(f"Error writing file to {filepath}: {e}")
        return False
    except TypeError as e:
        print(f"Error serializing track data to JSON: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"An unexpected error occurred saving to {filepath}:")
        traceback.print_exc()
        return False

# -----------------------------------------------------------------------------
# Main Generation Function
# -----------------------------------------------------------------------------

def generate_audio(track_data, output_format:str, output_filename=None):
    """Generates and saves the audio file based on the track_data."""
    """
    output_format: "wav", "flac", or "mp3"
    """
    if not track_data:
        print(f"Error: Cannot generate {output_format.upper()}, track data is missing.")
        return False

    global_settings = track_data.get("global_settings", {})
    try:
        sample_rate = int(global_settings.get("sample_rate", 44100))
        crossfade_duration = float(global_settings.get("crossfade_duration", 1.0))
    except (ValueError, TypeError) as e:
         print(f"Error: Invalid global settings (sample_rate or crossfade_duration): {e}")
         return False

    output_filename = output_filename or global_settings.get("output_filename", f"generated_track.{output_format.lower()}")
    if not output_filename or not isinstance(output_filename, str):
         print(f"Error: Invalid output filename: {output_filename}")
         return False

    output_filename = output_filename.replace(f".{output_format.lower()}", "") + f".{output_format.lower()}" # Ensure .wav extension

    # Ensure output directory exists before assembly
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory for audio: {output_dir}")
        except OSError as e:
            print(f"Error creating output directory '{output_dir}': {e}")
            return False


    print(f"\n--- Starting {output_format.upper()} Generation ---")
    print(f"Sample Rate: {sample_rate} Hz")
    print(f"Crossfade Duration: {crossfade_duration} s")
    print(f"Output File: {output_filename}")

    # Assemble the track (includes per-step normalization now)
    track_audio = assemble_track_from_data(track_data, sample_rate, crossfade_duration)

    if track_audio is None or track_audio.size == 0:
        print("Error: Track assembly failed or resulted in empty audio.")
        return False

    # --- Final Normalization ---
    max_abs_val = np.max(np.abs(track_audio))

    if max_abs_val > 1e-9: # Avoid division by zero for silent tracks
        # --- *** CHANGED: Increase target level *** ---
        target_level = 0.2 # Normalize closer to full scale (e.g., -0.4 dBFS)
        # --- *** End Change *** ---
        scaling_factor = target_level / max_abs_val
        print(f"Normalizing final track (peak value: {max_abs_val:.4f}) to target level: {target_level}")
        normalized_track = track_audio * scaling_factor
        # Optional: Apply a limiter after normalization as a final safety net
        # normalized_track = np.clip(normalized_track, -target_level, target_level)
    else:
        print("Track is silent or near-silent. Skipping final normalization.")
        normalized_track = track_audio # Already silent or zero

    # Convert normalized float audio to 16-bit PCM
    if not np.issubdtype(normalized_track.dtype, np.floating):
         print(f"Warning: Normalized track data type is not float ({normalized_track.dtype}). Attempting conversion.")
         try: normalized_track = normalized_track.astype(np.float64) # Use float64 for precision before scaling
         except Exception as e:
              print(f"Error converting normalized track to float: {e}")
              return False

    # Scale to 16-bit integer range and clip just in case
    track_int16 = np.int16(np.clip(normalized_track * 32767, -32768, 32767))

    # Write WAV file
    try:
        if output_format.lower() == "wav":
            write(output_filename, sample_rate, track_int16)
        elif output_format.lower() == "flac":
            sf.write(output_filename, track_int16, sample_rate, format='FLAC')
        elif output_format.lower() == "mp3":
            # Convert to MP3 using pydub
            audio_segment = AudioSegment(
            track_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2, # 16-bit PCM = 2 bytes per sample
            channels=2
            )
            audio_segment.export(output_filename, format="mp3", bitrate="320k")
        
        print(f"--- {output_format.upper()} Generation Complete ---")
        print(f"Track successfully written to {output_filename}")
        return True
    except Exception as e:
        print(f"Error writing {output_format.upper()} file {output_filename}:")
        traceback.print_exc()
        return False

# -----------------------------------------------------------------------------
# Example Usage (if script is run directly)
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    print("Running sound_creator.py directly.")
    print("This script defines audio generation functions and logic.")
    print("Run track_editor_gui.py to use the graphical interface.")

    # Example: Create and save a default JSON if it doesn't exist
    default_json_file = 'default_track_with_flanger.json' # New name for testing
    if not os.path.exists(default_json_file):
        print(f"Creating a simple default JSON file: {default_json_file}")
        default_data = {
            "global_settings": {
                 "sample_rate": 44100,
                 "crossfade_duration": 1.0, # Longer crossfade for smoother flanger test
                 "output_filename": "test_flanger_output.wav"
            },
            "steps": [
                 {
                     "duration": 10.0, # Longer duration to hear flanger sweep
                     "voices": [
                         {
                             "synth_function_name": "flanged_voice", # Test new flanger
                             "is_transition": False,
                             "params": {
                                 "amp": 0.6,
                                 "noiseType": 2, # Pink noise
                                 "max_delay_ms": 15.0, # Shorter max delay
                                 "min_delay_ms": 0.5,
                                 "rate_hz": 0.1, # Faster LFO for testing
                                 "start_frac": 0.25, # Peak
                                 "end_frac": 0.75, # Trough
                                 "begin_hold_time_s": 1.0, # Hold at start
                                 "end_hold_time_s": 1.0, # Hold at end
                                 "skip_below_freq": 30.0,
                                 "lowest_freq_mode": 'notch',
                                 "dry_mix": 0.6,
                                 "wet_mix": 0.4,
                                 "reverse": False
                             },
                             "volume_envelope": {"type": "linen", "params": {"attack": 1.0, "release": 1.0}}
                         }
                     ]
                 },
                 {
                     "duration": 5.0,
                     "voices": [
                         {
                             "synth_function_name": "binaural_beat",
                             "is_transition": False,
                             "params": {"amp": 0.4, "baseFreq": 100, "beatFreq": 3},
                             "volume_envelope": None
                         }
                     ]
                 }
            ]
        }
        if save_track_to_json(default_data, default_json_file):
             print("\nAttempting to load and generate the test track...")
             track_definition = load_track_from_json(default_json_file)
             if track_definition:
                  generate_wav(track_definition)
             else:
                  print("Failed to load track definition after saving.")
    else:
        print(f"Default file '{default_json_file}' already exists.")
        # Optionally load and generate here for testing
        print("\nAttempting to load and generate existing test track...")
        track_definition = load_track_from_json(default_json_file)
        if track_definition:
           generate_wav(track_definition)
        else:
           print("Failed to load existing track definition.")
