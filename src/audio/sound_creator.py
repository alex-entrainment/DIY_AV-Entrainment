import numpy as np
import slab
from scipy.signal import butter, lfilter, sosfiltfilt
from scipy.io.wavfile import write
import math
import json
import inspect # Needed to inspect function parameters for GUI
import os # Needed for path checks in main example
import traceback # For detailed error printing
import numba 
from numba import jit, prange

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


# -----------------------------------------------------------------------------
# Synth Def translations – Updated to accept duration, sr, and params dict
# -----------------------------------------------------------------------------

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
    # --- Unpack synthesis parameters ---
    ampL = float(params.get('ampL', 0.5))
    ampR = float(params.get('ampR', 0.5))
    baseF = float(params.get('baseFreq', 200.0))
    beatF = float(params.get('beatFreq', 4.0))
    force_mono = bool(params.get('forceMono', False))
    startL = float(params.get('startPhaseL', 0.0)) # in radians
    startR = float(params.get('startPhaseR', 0.0)) # in radians
    aODL = float(params.get('ampOscDepthL', 0.0))
    aOFL = float(params.get('ampOscFreqL', 0.0))
    aODR = float(params.get('ampOscDepthR', 0.0))
    aOFR = float(params.get('ampOscFreqR', 0.0))
    fORL = float(params.get('freqOscRangeL', 0.0))
    fOFL = float(params.get('freqOscFreqL', 0.0))
    fORR = float(params.get('freqOscRangeR', 0.0))
    fOFR = float(params.get('freqOscFreqR', 0.0))
    ampOscPhaseOffsetL = float(params.get('ampOscPhaseOffsetL', 0.0))
    ampOscPhaseOffsetR = float(params.get('ampOscPhaseOffsetR', 0.0))
    pOF = float(params.get('phaseOscFreq', 0.0))
    pOR = float(params.get('phaseOscRange', 0.0)) # in radians

    # --- UNCOMMENT IF "GLITCH/CHIRP" EFFECT NEEDED OR DESIRED
    # --- Unpack glitch parameters --- 
    # glitchInterval   = float(params.get('glitchInterval',   0.0))
    # glitchDur        = float(params.get('glitchDur',        0.0))
    # glitchNoiseLevel = float(params.get('glitchNoiseLevel', 0.0))
    # glitchFocusWidth = float(params.get('glitchFocusWidth', 0.0))
    # glitchFocusExp   = float(params.get('glitchFocusExp',   0.0))

    N = int(duration * sample_rate)
    #
    # # --- Precompute glitch bursts in NumPy ---
    # positions = []
    # bursts = []
    # if glitchInterval > 0 and glitchDur > 0 and glitchNoiseLevel > 0 and N > 0:
    #     full_n = int(glitchDur * sample_rate)
    #     if full_n > 0:
    #         repeats = int(duration / glitchInterval)
    #         for k in range(1, repeats + 1):
    #             t_end   = k * glitchInterval
    #             t_start = max(0.0, t_end - glitchDur)
    #             i0 = int(t_start * sample_rate)
    #             i1 = i0 + full_n
    #             if i1 > N:
    #                 continue
    #
    #             white = np.random.standard_normal(full_n)
    #             S = np.fft.rfft(white)
    #             freqs_denominator = full_n if full_n != 0 else 1
    #             freqs = np.arange(S.size) * (sample_rate / freqs_denominator)
    #
    #             if glitchFocusWidth == 0:
    #                 gauss = np.ones_like(freqs)
    #             else:
    #                 gauss = np.exp(-0.5 * ((freqs - baseF) / glitchFocusWidth) ** glitchFocusExp)
    #
    #             shaped = np.fft.irfft(S * gauss, n=full_n)
    #             max_abs_shaped = np.max(np.abs(shaped))
    #             if max_abs_shaped < 1e-16:
    #                 shaped_normalized = np.zeros_like(shaped)
    #             else:
    #                 shaped_normalized = shaped / max_abs_shaped
    #             ramp = np.linspace(0.0, 1.0, full_n, endpoint=True)
    #             burst_samples = (shaped_normalized * ramp * glitchNoiseLevel).astype(np.float32)
    #
    #             positions.append(i0)
    #             bursts.append(burst_samples)
    #
    # if bursts:
    #     pos_arr   = np.array(positions, dtype=np.int32)
    #     # Ensure all burst_samples have the same length if concatenating or handle individually
    #     # For simplicity, assuming all burst_samples are concatenated correctly if generated
    #     if any(b.ndim == 0 or b.size == 0 for b in bursts): # check for empty or scalar arrays
    #         burst_arr = np.empty(0, dtype=np.float32)
    #         pos_arr = np.empty(0, dtype=np.int32) # Clear positions if bursts are problematic
    #     else:
    #         try:
    #             burst_arr = np.concatenate(bursts)
    #         except ValueError: # Handle cases where concatenation might fail (e.g. varying dimensions beyond the first)
    #              # Fallback: if concatenation fails, create empty burst array or handle error appropriately
    #             burst_arr = np.empty(0, dtype=np.float32)
    #             pos_arr = np.empty(0, dtype=np.int32)
    #

    pos_arr   = np.empty(0, dtype=np.int32)
    burst_arr = np.empty(0, dtype=np.float32)

    return _binaural_beat_core(
        N,
        float(duration),
        float(sample_rate),
        ampL, ampR, baseF, beatF, force_mono,
        startL, startR, pOF, pOR,
        aODL, aOFL, aODR, aOFR,
        ampOscPhaseOffsetL, ampOscPhaseOffsetR,
        fORL, fOFL, fORR, fOFR,
        pos_arr, burst_arr
    )

@numba.njit(parallel=True, fastmath=True)
def _binaural_beat_core(
    N, duration, sample_rate,
    ampL, ampR, baseF, beatF, force_mono,
    startL, startR, pOF, pOR,
    aODL, aOFL, aODR, aOFR,
    ampOscPhaseOffsetL, ampOscPhaseOffsetR,
    fORL, fOFL, fORR, fOFR,
    pos,   # int32[:] start indices
    burst  # float32[:] concatenated glitch samples
):
    if N <= 0 :
        return np.zeros((0,2), dtype=np.float32)
        
    t = np.empty(N, dtype=np.float64)
    dt = duration / N if N > 0 else 0.0
    for i in numba.prange(N): # Use prange for parallel
        t[i] = i * dt

    halfB = beatF / 2.0
    fL_base = baseF - halfB
    fR_base = baseF + halfB
    instL = np.empty(N, dtype=np.float64)
    instR = np.empty(N, dtype=np.float64)
    for i in numba.prange(N): # Use prange
        vibL = (fORL/2.0) * np.sin(2*np.pi*fOFL*t[i])
        vibR = (fORR/2.0) * np.sin(2*np.pi*fOFR*t[i])
        instL[i] = max(0.0, fL_base + vibL)
        instR[i] = max(0.0, fR_base + vibR)
    
    if force_mono or beatF == 0.0:
        for i in numba.prange(N): # Use prange
            instL[i] = baseF
            instR[i] = baseF

    phL = np.empty(N, dtype=np.float64)
    phR = np.empty(N, dtype=np.float64)
    curL = startL
    curR = startR
    for i in range(N): # Sequential loop
        curL += 2 * np.pi * instL[i] * dt
        curR += 2 * np.pi * instR[i] * dt
        phL[i] = curL
        phR[i] = curR

    if pOF != 0.0 or pOR != 0.0:
        for i in numba.prange(N): # Use prange
            dphi = (pOR/2.0) * np.sin(2*np.pi*pOF*t[i])
            phL[i] -= dphi
            phR[i] += dphi

    envL = np.empty(N, dtype=np.float64)
    envR = np.empty(N, dtype=np.float64)
    for i in numba.prange(N): # Use prange
        envL[i] = 1.0 - aODL * (0.5*(1.0 + np.sin(2*np.pi*aOFL*t[i] + ampOscPhaseOffsetL)))
        envR[i] = 1.0 - aODR * (0.5*(1.0 + np.sin(2*np.pi*aOFR*t[i] + ampOscPhaseOffsetR)))

    out = np.empty((N,2), dtype=np.float32)
    for i in numba.prange(N): # Use prange
        out[i,0] = np.float32(np.sin(phL[i]) * envL[i] * ampL)
        out[i,1] = np.float32(np.sin(phR[i]) * envR[i] * ampR)

    num_bursts = pos.shape[0]
    if num_bursts > 0 and burst.size > 0: # ensure burst is not empty
        # Ensure burst.size is divisible by num_bursts.
        # This check implies burst is not empty and pos is not empty.
        if burst.size % num_bursts == 0 :
            L = burst.size // num_bursts
            if L > 0: # Ensure segment length L is positive
                idx = 0
                # This loop should be sequential if bursts can overlap or write to the same output indices.
                # If pos guarantees no overlap, then outer loop b can be prange.
                # Assuming pos are sorted and bursts don't overlap for safety in parallel context.
                # However, a sequential loop is safer unless overlap is explicitly managed.
                # For now, using sequential for the outer loop over bursts.
                for b in range(num_bursts): # Changed from prange to range for safety with additions
                    start_idx = pos[b]
                    current_burst_segment = burst[idx : idx + L]
                    for j in range(L):
                        p = start_idx + j
                        if p < N: # Boundary check
                            # Numba does not automatically make += atomic in all contexts.
                            # For safety, this part is tricky with prange on `b` if `pos` allows overlaps.
                            out[p,0] += current_burst_segment[j]
                            out[p,1] += current_burst_segment[j]
                    idx += L
    return out

# -----------------------------------------------------------------------------
# Transition BB version - UPDATED
# -----------------------------------------------------------------------------
def binaural_beat_transition(duration, sample_rate=44100, **params):
    # --- Unpack start/end parameters ---
    startAmpL = float(params.get('startAmpL', params.get('ampL', 0.5)))
    endAmpL = float(params.get('endAmpL', startAmpL))
    startAmpR = float(params.get('startAmpR', params.get('ampR', 0.5)))
    endAmpR = float(params.get('endAmpR', startAmpR))
    startBaseF = float(params.get('startBaseFreq', params.get('baseFreq', 200.0)))
    endBaseF = float(params.get('endBaseFreq', startBaseF))
    startBeatF = float(params.get('startBeatFreq', params.get('beatFreq', 4.0)))
    endBeatF = float(params.get('endBeatFreq', startBeatF))

    # New transitional parameters from binaural_beat
    startForceMono = float(params.get('startForceMono', params.get('forceMono', 0.0))) # 0.0 for False, 1.0 for True
    endForceMono = float(params.get('endForceMono', startForceMono))
    startStartPhaseL = float(params.get('startStartPhaseL', params.get('startPhaseL', 0.0)))
    endStartPhaseL = float(params.get('endStartPhaseL', startStartPhaseL))
    startStartPhaseR = float(params.get('startStartPhaseR', params.get('startPhaseR', 0.0)))
    endStartPhaseR = float(params.get('endStartPhaseR', startStartPhaseR))
    
    startPOF = float(params.get('startPhaseOscFreq', params.get('phaseOscFreq', 0.0)))
    endPOF = float(params.get('endPhaseOscFreq', startPOF))
    startPOR = float(params.get('startPhaseOscRange', params.get('phaseOscRange', 0.0)))
    endPOR = float(params.get('endPhaseOscRange', startPOR))

    startAODL = float(params.get('startAmpOscDepthL', params.get('ampOscDepthL', 0.0)))
    endAODL = float(params.get('endAmpOscDepthL', startAODL))
    startAOFL = float(params.get('startAmpOscFreqL', params.get('ampOscFreqL', 0.0)))
    endAOFL = float(params.get('endAmpOscFreqL', startAOFL))
    startAODR = float(params.get('startAmpOscDepthR', params.get('ampOscDepthR', 0.0)))
    endAODR = float(params.get('endAmpOscDepthR', startAODR))
    startAOFR = float(params.get('startAmpOscFreqR', params.get('ampOscFreqR', 0.0)))
    endAOFR = float(params.get('endAmpOscFreqR', startAOFR))

    startAmpOscPhaseOffsetL = float(params.get('startAmpOscPhaseOffsetL', params.get('ampOscPhaseOffsetL', 0.0)))
    endAmpOscPhaseOffsetL = float(params.get('endAmpOscPhaseOffsetL', startAmpOscPhaseOffsetL))
    startAmpOscPhaseOffsetR = float(params.get('startAmpOscPhaseOffsetR', params.get('ampOscPhaseOffsetR', 0.0)))
    endAmpOscPhaseOffsetR = float(params.get('endAmpOscPhaseOffsetR', startAmpOscPhaseOffsetR))

    startFORL = float(params.get('startFreqOscRangeL', params.get('freqOscRangeL', 0.0)))
    endFORL = float(params.get('endFreqOscRangeL', startFORL))
    startFOFL = float(params.get('startFreqOscFreqL', params.get('freqOscFreqL', 0.0)))
    endFOFL = float(params.get('endFreqOscFreqL', startFOFL))
    startFORR = float(params.get('startFreqOscRangeR', params.get('freqOscRangeR', 0.0)))
    endFORR = float(params.get('endFreqOscRangeR', startFORR))
    startFOFR = float(params.get('startFreqOscFreqR', params.get('freqOscFreqR', 0.0)))
    endFOFR = float(params.get('endFreqOscFreqR', startFOFR))

    # Glitch parameters (using average for pre-computation)
    s_glitchInterval = float(params.get('startGlitchInterval', params.get('glitchInterval', 0.0)))
    e_glitchInterval = float(params.get('endGlitchInterval', s_glitchInterval))
    avg_glitchInterval = (s_glitchInterval + e_glitchInterval) / 2.0

    s_glitchDur = float(params.get('startGlitchDur', params.get('glitchDur', 0.0)))
    e_glitchDur = float(params.get('endGlitchDur', s_glitchDur))
    avg_glitchDur = (s_glitchDur + e_glitchDur) / 2.0
    
    s_glitchNoiseLevel = float(params.get('startGlitchNoiseLevel', params.get('glitchNoiseLevel', 0.0)))
    e_glitchNoiseLevel = float(params.get('endGlitchNoiseLevel', s_glitchNoiseLevel))
    avg_glitchNoiseLevel = (s_glitchNoiseLevel + e_glitchNoiseLevel) / 2.0

    s_glitchFocusWidth = float(params.get('startGlitchFocusWidth', params.get('glitchFocusWidth', 0.0)))
    e_glitchFocusWidth = float(params.get('endGlitchFocusWidth', s_glitchFocusWidth))
    avg_glitchFocusWidth = (s_glitchFocusWidth + e_glitchFocusWidth) / 2.0
    
    s_glitchFocusExp = float(params.get('startGlitchFocusExp', params.get('glitchFocusExp', 0.0)))
    e_glitchFocusExp = float(params.get('endGlitchFocusExp', s_glitchFocusExp))
    avg_glitchFocusExp = (s_glitchFocusExp + e_glitchFocusExp) / 2.0

    N = int(duration * sample_rate)

    # --- Precompute glitch bursts using average parameters ---
    positions = []
    bursts = []
    # Use the average base frequency for glitch shaping if it's also transitional
    # For simplicity, using startBaseF here, or could use (startBaseF + endBaseF) / 2
    glitch_shaping_base_freq = (startBaseF + endBaseF) / 2.0

    if avg_glitchInterval > 0 and avg_glitchDur > 0 and avg_glitchNoiseLevel > 0 and N > 0:
        full_n = int(avg_glitchDur * sample_rate)
        if full_n > 0:
            repeats = int(duration / avg_glitchInterval) if avg_glitchInterval > 0 else 0
            for k in range(1, repeats + 1):
                t_end   = k * avg_glitchInterval
                t_start = max(0.0, t_end - avg_glitchDur)
                i0 = int(t_start * sample_rate)
                i1 = i0 + full_n
                if i1 > N:
                    continue

                white = np.random.standard_normal(full_n)
                S = np.fft.rfft(white)
                freqs_denominator = full_n if full_n != 0 else 1
                freqs = np.arange(S.size) * (sample_rate / freqs_denominator)

                if avg_glitchFocusWidth == 0:
                    gauss = np.ones_like(freqs)
                else:
                    gauss = np.exp(-0.5 * ((freqs - glitch_shaping_base_freq) / avg_glitchFocusWidth) ** avg_glitchFocusExp)
                
                shaped = np.fft.irfft(S * gauss, n=full_n)
                max_abs_shaped = np.max(np.abs(shaped))
                if max_abs_shaped < 1e-16:
                    shaped_normalized = np.zeros_like(shaped)
                else:
                    shaped_normalized = shaped / max_abs_shaped
                ramp = np.linspace(0.0, 1.0, full_n, endpoint=True)
                burst_samples = (shaped_normalized * ramp * avg_glitchNoiseLevel).astype(np.float32)
                
                positions.append(i0)
                bursts.append(burst_samples)
    
    if bursts:
        pos_arr   = np.array(positions, dtype=np.int32)
        if any(b.ndim == 0 or b.size == 0 for b in bursts):
            burst_arr = np.empty(0, dtype=np.float32)
            pos_arr = np.empty(0, dtype=np.int32)
        else:
            try:
                burst_arr = np.concatenate(bursts)
            except ValueError:
                burst_arr = np.empty(0, dtype=np.float32)
                pos_arr = np.empty(0, dtype=np.int32)
    else:
        pos_arr   = np.empty(0, dtype=np.int32)
        burst_arr = np.empty(0, dtype=np.float32)

    return _binaural_beat_transition_core(
        N, float(duration), float(sample_rate),
        startAmpL, endAmpL, startAmpR, endAmpR,
        startBaseF, endBaseF, startBeatF, endBeatF,
        startForceMono, endForceMono,
        startStartPhaseL, endStartPhaseL, startStartPhaseR, endStartPhaseR,
        startPOF, endPOF, startPOR, endPOR,
        startAODL, endAODL, startAOFL, endAOFL,
        startAODR, endAODR, startAOFR, endAOFR,
        startAmpOscPhaseOffsetL, endAmpOscPhaseOffsetL,
        startAmpOscPhaseOffsetR, endAmpOscPhaseOffsetR,
        startFORL, endFORL, startFOFL, endFOFL,
        startFORR, endFORR, startFOFR, endFOFR,
        pos_arr, burst_arr # Pass pre-calculated glitches
    )

@numba.njit(parallel=True, fastmath=True)
def _binaural_beat_transition_core(
    N, duration, sample_rate,
    startAmpL, endAmpL, startAmpR, endAmpR,
    startBaseF, endBaseF, startBeatF, endBeatF,
    startForceMono, endForceMono,
    startStartPhaseL, endStartPhaseL, startStartPhaseR, endStartPhaseR,
    startPOF, endPOF, startPOR, endPOR,
    startAODL, endAODL, startAOFL, endAOFL,
    startAODR, endAODR, startAOFR, endAOFR,
    startAmpOscPhaseOffsetL, endAmpOscPhaseOffsetL,
    startAmpOscPhaseOffsetR, endAmpOscPhaseOffsetR,
    startFORL, endFORL, startFOFL, endFOFL,
    startFORR, endFORR, startFOFR, endFOFR,
    pos, burst # Glitch arrays (static for this core run)
):
    if N <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    dt = duration / N
    
    t_arr = np.empty(N, np.float64)
    ampL_arr = np.empty(N, np.float64)
    ampR_arr = np.empty(N, np.float64)
    baseF_arr = np.empty(N, np.float64)
    beatF_arr = np.empty(N, np.float64)
    force_mono_arr = np.empty(N, np.float64) # Interpolate as float

    pOF_arr = np.empty(N, np.float64)
    pOR_arr = np.empty(N, np.float64)
    aODL_arr = np.empty(N, np.float64)
    aOFL_arr = np.empty(N, np.float64)
    aODR_arr = np.empty(N, np.float64)
    aOFR_arr = np.empty(N, np.float64)
    ampOscPhaseOffsetL_arr = np.empty(N, np.float64)
    ampOscPhaseOffsetR_arr = np.empty(N, np.float64)
    fORL_arr = np.empty(N, np.float64)
    fOFL_arr = np.empty(N, np.float64)
    fORR_arr = np.empty(N, np.float64)
    fOFR_arr = np.empty(N, np.float64)
    
    instL = np.empty(N, np.float64)
    instR = np.empty(N, np.float64)

    # Linear interpolation of parameters
    for i in numba.prange(N):
        alpha = i / (N - 1) if N > 1 else 0.0
        t_arr[i] = i * dt
        
        ampL_arr[i] = startAmpL + (endAmpL - startAmpL) * alpha
        ampR_arr[i] = startAmpR + (endAmpR - startAmpR) * alpha
        baseF_arr[i] = startBaseF + (endBaseF - startBaseF) * alpha
        beatF_arr[i] = startBeatF + (endBeatF - startBeatF) * alpha
        force_mono_arr[i] = startForceMono + (endForceMono - startForceMono) * alpha

        pOF_arr[i] = startPOF + (endPOF - startPOF) * alpha
        pOR_arr[i] = startPOR + (endPOR - startPOR) * alpha
        aODL_arr[i] = startAODL + (endAODL - startAODL) * alpha
        aOFL_arr[i] = startAOFL + (endAOFL - startAOFL) * alpha
        aODR_arr[i] = startAODR + (endAODR - startAODR) * alpha
        aOFR_arr[i] = startAOFR + (endAOFR - startAOFR) * alpha
        ampOscPhaseOffsetL_arr[i] = startAmpOscPhaseOffsetL + (endAmpOscPhaseOffsetL - startAmpOscPhaseOffsetL) * alpha
        ampOscPhaseOffsetR_arr[i] = startAmpOscPhaseOffsetR + (endAmpOscPhaseOffsetR - startAmpOscPhaseOffsetR) * alpha
        fORL_arr[i] = startFORL + (endFORL - startFORL) * alpha
        fOFL_arr[i] = startFOFL + (endFOFL - startFOFL) * alpha
        fORR_arr[i] = startFORR + (endFORR - startFORR) * alpha
        fOFR_arr[i] = startFOFR + (endFOFR - startFOFR) * alpha

        # Instantaneous frequencies
        halfB_i = beatF_arr[i] * 0.5
        fL_base_i = baseF_arr[i] - halfB_i
        fR_base_i = baseF_arr[i] + halfB_i

        vibL_i = (fORL_arr[i]/2.0) * np.sin(2*np.pi*fOFL_arr[i]*t_arr[i])
        vibR_i = (fORR_arr[i]/2.0) * np.sin(2*np.pi*fOFR_arr[i]*t_arr[i])
        
        instL_candidate = fL_base_i + vibL_i
        instR_candidate = fR_base_i + vibR_i

        instL[i] = instL_candidate if instL_candidate > 0.0 else 0.0
        instR[i] = instR_candidate if instR_candidate > 0.0 else 0.0
        
        if force_mono_arr[i] > 0.5 or beatF_arr[i] == 0.0: # Apply force_mono
            instL[i] = baseF_arr[i]
            instR[i] = baseF_arr[i]
            if instL[i] < 0.0: instL[i] = 0.0 # Ensure baseF is not negative
            if instR[i] < 0.0: instR[i] = 0.0


    # Phase accumulation (sequential)
    phL = np.empty(N, np.float64)
    phR = np.empty(N, np.float64)
    # Interpolate start phases (initial phase for the accumulation)
    # For phase, the start/end is for the *initial* phase value, not a rate of change of start phase.
    curL = startStartPhaseL # Use the start of the transition for the initial phase value
    curR = startStartPhaseR # Use the start of the transition for the initial phase value

    current_start_phase_L = startStartPhaseL
    current_start_phase_R = startStartPhaseR
    curL = startStartPhaseL
    curR = startStartPhaseR

    for i in range(N): # Sequential loop
        curL += 2.0 * np.pi * instL[i] * dt
        curR += 2.0 * np.pi * instR[i] * dt
        phL[i] = curL
        phR[i] = curR

    # Phase modulation
    for i in numba.prange(N): # Parallel (as it's per sample based on already calculated t_arr and pOF/pOR_arr)
        if pOF_arr[i] != 0.0 or pOR_arr[i] != 0.0:
            dphi = (pOR_arr[i]/2.0) * np.sin(2*np.pi*pOF_arr[i]*t_arr[i])
            phL[i] -= dphi
            phR[i] += dphi
            
    # Amplitude envelopes
    envL = np.empty(N, np.float64)
    envR = np.empty(N, np.float64)
    for i in numba.prange(N): # Parallel
        envL[i] = 1.0 - aODL_arr[i] * (0.5*(1.0 + np.sin(2*np.pi*aOFL_arr[i]*t_arr[i] + ampOscPhaseOffsetL_arr[i])))
        envR[i] = 1.0 - aODR_arr[i] * (0.5*(1.0 + np.sin(2*np.pi*aOFR_arr[i]*t_arr[i] + ampOscPhaseOffsetR_arr[i])))

    # Generate output
    out = np.empty((N, 2), dtype=np.float32)
    for i in numba.prange(N): # Parallel
        out[i, 0] = np.float32(np.sin(phL[i]) * envL[i] * ampL_arr[i])
        out[i, 1] = np.float32(np.sin(phR[i]) * envR[i] * ampR_arr[i])

    # Add glitch bursts (using the static pos and burst arrays)
    num_bursts = pos.shape[0]
    if num_bursts > 0 and burst.size > 0:
        if burst.size % num_bursts == 0:
            L_glitch = burst.size // num_bursts
            if L_glitch > 0:
                idx = 0
                for b in range(num_bursts): # Sequential for safety
                    start_idx = pos[b]
                    current_burst_segment = burst[idx : idx + L_glitch]
                    for j in range(L_glitch):
                        p = start_idx + j
                        if p < N:
                            out[p,0] += current_burst_segment[j]
                            out[p,1] += current_burst_segment[j]
                    idx += L_glitch
    return out


# -----------------------------------------------------------------------------
# Monaural Beats Stereo Amplitudes control
# -----------------------------------------------------------------------------
def monaural_beat_stereo_amps(duration, sample_rate=44100, **params):
    amp_l_L  = float(params.get('amp_lower_L', 0.5))
    amp_u_L  = float(params.get('amp_upper_L', 0.5))
    amp_l_R  = float(params.get('amp_lower_R', 0.5))
    amp_u_R  = float(params.get('amp_upper_R', 0.5))
    baseF    = float(params.get('baseFreq',    200.0))
    beatF    = float(params.get('beatFreq',      4.0))
    start_l  = float(params.get('startPhaseL',  0.0)) # Corresponds to lower frequency component
    start_u  = float(params.get('startPhaseR',  0.0)) # Corresponds to upper frequency component (was startPhaseR)
    phiF     = float(params.get('phaseOscFreq', 0.0))
    phiR     = float(params.get('phaseOscRange',0.0))
    aOD      = float(params.get('ampOscDepth',  0.0))
    aOF      = float(params.get('ampOscFreq',   0.0))
    # New: ampOscPhaseOffset for monaural (applies to both L and R mixed signal)
    aOP      = float(params.get('ampOscPhaseOffset', 0.0))


    N = int(duration * sample_rate)
    return _monaural_beat_stereo_amps_core(
        N, float(duration), float(sample_rate),
        amp_l_L, amp_u_L, amp_l_R, amp_u_R,
        baseF, beatF,
        start_l, start_u,
        phiF, phiR,
        aOD, aOF, aOP
    )

@numba.njit(parallel=True, fastmath=True)
def _monaural_beat_stereo_amps_core(
    N, duration, sample_rate,
    amp_l_L, amp_u_L, amp_l_R, amp_u_R,
    baseF, beatF,
    start_l, start_u, # start phases for lower and upper components
    phiF, phiR,       # phase oscillation freq and range
    aOD, aOF, aOP     # amplitude oscillation depth, freq, phase offset
):
    if N <= 0:
        return np.zeros((0,2), dtype=np.float32)

    t = np.empty(N, dtype=np.float64)
    dt = duration / N if N > 0 else 0.0
    for i in numba.prange(N):
        t[i] = i * dt

    halfB = beatF / 2.0
    f_l = baseF - halfB
    f_u = baseF + halfB
    if f_l < 0.0: f_l = 0.0
    if f_u < 0.0: f_u = 0.0
    
    ph_l = np.empty(N, dtype=np.float64)
    ph_u = np.empty(N, dtype=np.float64)
    cur_l = start_l
    cur_u = start_u
    for i in range(N): # Sequential
        cur_l += 2 * np.pi * f_l * dt
        cur_u += 2 * np.pi * f_u * dt
        ph_l[i] = cur_l
        ph_u[i] = cur_u

    if phiF != 0.0 or phiR != 0.0:
        for i in numba.prange(N):
            dphi = (phiR/2.0) * np.sin(2*np.pi*phiF*t[i])
            ph_l[i] -= dphi
            ph_u[i] += dphi

    s_l = np.empty(N, dtype=np.float64)
    s_u = np.empty(N, dtype=np.float64)
    for i in numba.prange(N):
        s_l[i] = np.sin(ph_l[i])
        s_u[i] = np.sin(ph_u[i])

    mix_L = np.empty(N, dtype=np.float64)
    mix_R = np.empty(N, dtype=np.float64)
    for i in numba.prange(N):
        mix_L[i] = s_l[i] * amp_l_L + s_u[i] * amp_u_L
        mix_R[i] = s_l[i] * amp_l_R + s_u[i] * amp_u_R

    # Amplitude modulation (using aOD, aOF, and new aOP)
    # The original code had depth clamping, let's make aOD behave like ampOscDepthL/R (0-1 usual range for depth)
    # (1.0 - aOD * (0.5 * (1 + sin(...)))) means aOD=0 -> factor 1, aOD=1 -> factor 0.5 to 1.5 around 1
    # Let's use: env_factor = 1.0 - aOD * (0.5 * (1.0 + np.sin(2*np.pi*aOF*t[i] + aOP)))
    # Or, if aOD is meant like the previous `depth` (0-2):
    # m = (1.0 - aOD/2.0) + (aOD/2.0)*np.sin(2*np.pi*aOF*t[i] + aOP)
    # Let's use the latter form as it was in the original code for aOD.
    # Clamping aOD (depth) to 0-2 as before
    current_aOD = aOD
    if current_aOD < 0.0: current_aOD = 0.0
    if current_aOD > 2.0: current_aOD = 2.0

    if current_aOD != 0.0 and aOF != 0.0:
        for i in numba.prange(N):
            # Original form: m = (1.0 - depth/2.0) + (depth/2.0)*np.sin(2*np.pi*aOF*t[i])
            # Adding aOP:
            mod_val = (1.0 - current_aOD/2.0) + (current_aOD/2.0) * np.sin(2*np.pi*aOF*t[i] + aOP)
            mix_L[i] *= mod_val
            mix_R[i] *= mod_val
            
    out = np.empty((N,2), dtype=np.float32)
    for i in numba.prange(N):
        l_val = mix_L[i]
        if l_val > 1.0: l_val = 1.0
        elif l_val < -1.0: l_val = -1.0
        r_val = mix_R[i]
        if r_val > 1.0: r_val = 1.0
        elif r_val < -1.0: r_val = -1.0
        out[i,0] = np.float32(l_val)
        out[i,1] = np.float32(r_val)

    return out

# -----------------------------------------------------------------------------
# Monaural Beats Stereo Amps Transition - UPDATED
# -----------------------------------------------------------------------------
def monaural_beat_stereo_amps_transition(duration, sample_rate=44100, **params):
    s_ll = float(params.get('start_amp_lower_L', params.get('amp_lower_L', 0.5)))
    e_ll = float(params.get('end_amp_lower_L',   s_ll))
    s_ul = float(params.get('start_amp_upper_L', params.get('amp_upper_L', 0.5)))
    e_ul = float(params.get('end_amp_upper_L',   s_ul))
    s_lr = float(params.get('start_amp_lower_R', params.get('amp_lower_R', 0.5)))
    e_lr = float(params.get('end_amp_lower_R',   s_lr))
    s_ur = float(params.get('start_amp_upper_R', params.get('amp_upper_R', 0.5)))
    e_ur = float(params.get('end_amp_upper_R',   s_ur))
    sBF  = float(params.get('startBaseFreq',     params.get('baseFreq',    200.0)))
    eBF  = float(params.get('endBaseFreq',       sBF))
    sBt  = float(params.get('startBeatFreq',     params.get('beatFreq',      4.0)))
    eBt  = float(params.get('endBeatFreq',       sBt))

    # New transitional parameters from monaural_beat_stereo_amps
    sStartPhaseL = float(params.get('startStartPhaseL', params.get('startPhaseL', 0.0)))
    eStartPhaseL = float(params.get('endStartPhaseL', sStartPhaseL))
    sStartPhaseU = float(params.get('startStartPhaseU', params.get('startPhaseR', 0.0))) # Was startPhaseR
    eStartPhaseU = float(params.get('endStartPhaseU', sStartPhaseU))

    sPhiF = float(params.get('startPhaseOscFreq', params.get('phaseOscFreq', 0.0)))
    ePhiF = float(params.get('endPhaseOscFreq', sPhiF))
    sPhiR = float(params.get('startPhaseOscRange', params.get('phaseOscRange', 0.0)))
    ePhiR = float(params.get('endPhaseOscRange', sPhiR))

    sAOD = float(params.get('startAmpOscDepth', params.get('ampOscDepth', 0.0)))
    eAOD = float(params.get('endAmpOscDepth', sAOD))
    sAOF = float(params.get('startAmpOscFreq', params.get('ampOscFreq', 0.0)))
    eAOF = float(params.get('endAmpOscFreq', sAOF))
    sAOP = float(params.get('startAmpOscPhaseOffset', params.get('ampOscPhaseOffset', 0.0))) # New
    eAOP = float(params.get('endAmpOscPhaseOffset', sAOP))


    N = int(duration * sample_rate)
    return _monaural_beat_stereo_amps_transition_core(
        N, float(duration), float(sample_rate),
        s_ll, e_ll, s_ul, e_ul, s_lr, e_lr, s_ur, e_ur,
        sBF, eBF, sBt, eBt,
        sStartPhaseL, eStartPhaseL, sStartPhaseU, eStartPhaseU,
        sPhiF, ePhiF, sPhiR, ePhiR,
        sAOD, eAOD, sAOF, eAOF, sAOP, eAOP
    )

@numba.njit(parallel=True, fastmath=True)
def _monaural_beat_stereo_amps_transition_core(
    N, duration, sample_rate,
    s_ll, e_ll, s_ul, e_ul, s_lr, e_lr, s_ur, e_ur, # Amplitudes
    sBF, eBF, sBt, eBt,                             # Frequencies
    sSPL, eSPL, sSPU, eSPU,                         # Start Phases (lower, upper)
    sPhiF, ePhiF, sPhiR, ePhiR,                     # Phase Osc
    sAOD, eAOD, sAOF, eAOF, sAOP, eAOP              # Amp Osc
):
    if N <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    dt = duration / N
    
    t_arr = np.empty(N, np.float64)
    amp_ll_arr = np.empty(N, np.float64)
    amp_ul_arr = np.empty(N, np.float64)
    amp_lr_arr = np.empty(N, np.float64)
    amp_ur_arr = np.empty(N, np.float64)
    baseF_arr = np.empty(N, np.float64)
    beatF_arr = np.empty(N, np.float64)
    
    f_lower_arr = np.empty(N, np.float64)
    f_upper_arr = np.empty(N, np.float64)

    phiF_arr = np.empty(N, np.float64)
    phiR_arr = np.empty(N, np.float64)
    aOD_arr = np.empty(N, np.float64)
    aOF_arr = np.empty(N, np.float64)
    aOP_arr = np.empty(N, np.float64)


    for i in numba.prange(N):
        t_arr[i] = i * dt
        alpha = i / (N - 1) if N > 1 else 0.0
        
        amp_ll_arr[i] = s_ll + (e_ll - s_ll) * alpha
        amp_ul_arr[i] = s_ul + (e_ul - s_ul) * alpha
        amp_lr_arr[i] = s_lr + (e_lr - s_lr) * alpha
        amp_ur_arr[i] = s_ur + (e_ur - s_ur) * alpha
        baseF_arr[i] = sBF + (eBF - sBF) * alpha
        beatF_arr[i] = sBt + (eBt - sBt) * alpha
        
        halfB_i = beatF_arr[i] * 0.5
        f_l_cand = baseF_arr[i] - halfB_i
        f_u_cand = baseF_arr[i] + halfB_i
        
        f_lower_arr[i] = f_l_cand if f_l_cand > 0.0 else 0.0
        f_upper_arr[i] = f_u_cand if f_u_cand > 0.0 else 0.0

        phiF_arr[i] = sPhiF + (ePhiF - sPhiF) * alpha
        phiR_arr[i] = sPhiR + (ePhiR - sPhiR) * alpha
        aOD_arr[i] = sAOD + (eAOD - sAOD) * alpha
        aOF_arr[i] = sAOF + (eAOF - sAOF) * alpha
        aOP_arr[i] = sAOP + (eAOP - sAOP) * alpha


    ph_l = np.empty(N, np.float64)
    ph_u = np.empty(N, np.float64)
    # Interpolate initial phases. For now, use the start values as fixed initial phases.
    cur_l = sSPL # Start phase for lower component at the beginning of this transition segment
    cur_u = sSPU # Start phase for upper component
    
    for i in range(N): # Sequential
        cur_l += 2.0 * np.pi * f_lower_arr[i] * dt
        cur_u += 2.0 * np.pi * f_upper_arr[i] * dt
        ph_l[i] = cur_l
        ph_u[i] = cur_u

    for i in numba.prange(N): # Parallel
        if phiF_arr[i] != 0.0 or phiR_arr[i] != 0.0:
            dphi = (phiR_arr[i]/2.0) * np.sin(2*np.pi*phiF_arr[i]*t_arr[i])
            ph_l[i] -= dphi
            ph_u[i] += dphi

    s_l_wav = np.empty(N, np.float64)
    s_u_wav = np.empty(N, np.float64)
    for i in numba.prange(N):
        s_l_wav[i] = np.sin(ph_l[i])
        s_u_wav[i] = np.sin(ph_u[i])

    mix_L = np.empty(N, dtype=np.float64)
    mix_R = np.empty(N, dtype=np.float64)
    for i in numba.prange(N):
        mix_L[i] = s_l_wav[i] * amp_ll_arr[i] + s_u_wav[i] * amp_ul_arr[i]
        mix_R[i] = s_l_wav[i] * amp_lr_arr[i] + s_u_wav[i] * amp_ur_arr[i]

    for i in numba.prange(N):
        current_aOD_i = aOD_arr[i]
        if current_aOD_i < 0.0: current_aOD_i = 0.0
        if current_aOD_i > 2.0: current_aOD_i = 2.0

        if current_aOD_i != 0.0 and aOF_arr[i] != 0.0:
            mod_val = (1.0 - current_aOD_i/2.0) + \
                      (current_aOD_i/2.0) * np.sin(2*np.pi*aOF_arr[i]*t_arr[i] + aOP_arr[i])
            mix_L[i] *= mod_val
            mix_R[i] *= mod_val

    out = np.empty((N, 2), dtype=np.float32)
    for i in numba.prange(N):
        l_val = mix_L[i]
        if l_val > 1.0: l_val = 1.0
        elif l_val < -1.0: l_val = -1.0
        r_val = mix_R[i]
        if r_val > 1.0: r_val = 1.0
        elif r_val < -1.0: r_val = -1.0
        out[i, 0] = np.float32(l_val)
        out[i, 1] = np.float32(r_val)
        
    return out

# -----------------------------------------------------------------------------
# Spatial Angle Modulation - _prepare_beats_and_angles (Corrected)
# -----------------------------------------------------------------------------
@numba.njit(parallel=True, fastmath=True)
def _prepare_beats_and_angles( # Corrected version
    mono: np.ndarray,
    sample_rate: float,
    aOD: float, aOF: float, aOP: float,      # AM for this stage
    spatial_freq: float,
    path_radius: float,
    spatial_phase_off: float                 # Initial phase offset for spatial rotation
):
    N = mono.shape[0]
    if N == 0:
        return (np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32))

    mod_beat  = np.empty(N, dtype=np.float32)
    azimuth   = np.empty(N, dtype=np.float32)
    elevation = np.zeros(N, dtype=np.float32) # Elevation is kept at zero
    
    dt = 1.0 / sample_rate

    # 1) Apply custom AM envelope + build mod_beat
    for i in numba.prange(N):
        t = i * dt
        if aOD > 0.0 and aOF > 0.0: # Original condition was aOD > 0 and aOF > 0
            # Using (1-D/2) + (D/2)*sin form for AM to be consistent if aOD is 0-2
            # If aOD is 0-1 (like other ampOscDepths), then:
            # env = 1.0 - aOD * (0.5 * (1.0 + math.sin(2*math.pi*aOF*t + aOP)))
            # Assuming aOD here is also depth 0-2 like in monaural
            clamped_aOD = min(max(aOD, 0.0), 2.0)
            env = (1.0 - clamped_aOD/2.0) + (clamped_aOD/2.0) * math.sin(2*math.pi*aOF*t + aOP)
        else:
            env = 1.0
        mod_beat[i] = mono[i] * env # mono is already float32

    # 2) Compute circular path (radius from mod_beat) → azimuth
    # Corrected phase calculation for parallel loop (spatial_freq is constant here)
    for i in numba.prange(N):
        t_i = i * dt
        # Calculate phase at time t_i directly
        current_spatial_phase_at_t = spatial_phase_off + (2 * math.pi * spatial_freq * t_i)
        
        r = path_radius * (0.5 * (mod_beat[i] + 1.0)) # mod_beat is [-1, 1], so (mod_beat+1)/2 is [0,1]
        
        # Cartesian coordinates for HRTF lookup (y is often 'forward' in HRTF)
        # X = R * sin(angle) (side)
        # Y = R * cos(angle) (front/back)
        # atan2(x,y) means angle relative to positive Y axis, clockwise if X is positive.
        x_coord = r * math.sin(current_spatial_phase_at_t)
        y_coord = r * math.cos(current_spatial_phase_at_t)
        
        azimuth[i] = math.degrees(math.atan2(x_coord, y_coord))

    return mod_beat, azimuth, elevation

# -----------------------------------------------------------------------------
# Spatial Angle Modulation - Non-Transition Version (uses corrected _prepare_beats_and_angles)
# -----------------------------------------------------------------------------
def spatial_angle_modulation_monaural_beat(
    duration,
    sample_rate=44100,
    **params
):
    # --- unpack AM params for this specific stage ---
    # These are applied *after* the monaural beat generation's own AM
    sam_aOD = float(params.get('sam_ampOscDepth', params.get('ampOscDepth', 0.0))) # prefix with sam_ to avoid conflict
    sam_aOF = float(params.get('sam_ampOscFreq', params.get('ampOscFreq', 0.0)))
    sam_aOP = float(params.get('sam_ampOscPhaseOffset', params.get('ampOscPhaseOffset', 0.0)))

    # --- prepare core beat args (can have its own AM) ---
    beat_params = {
        'amp_lower_L':   float(params.get('amp_lower_L',   0.5)),
        'amp_upper_L':   float(params.get('amp_upper_L',   0.5)),
        'amp_lower_R':   float(params.get('amp_lower_R',   0.5)),
        'amp_upper_R':   float(params.get('amp_upper_R',   0.5)),
        'baseFreq':      float(params.get('baseFreq',      200.0)),
        'beatFreq':      float(params.get('beatFreq',        4.0)),
        'startPhaseL':   float(params.get('startPhaseL',    0.0)),
        'startPhaseR':   float(params.get('startPhaseR',    0.0)), # for upper component
        'phaseOscFreq':  float(params.get('phaseOscFreq',  0.0)),
        'phaseOscRange': float(params.get('phaseOscRange', 0.0)),
        'ampOscDepth':   float(params.get('monaural_ampOscDepth', 0.0)), # AM for monaural beat
        'ampOscFreq':    float(params.get('monaural_ampOscFreq', 0.0)),
        'ampOscPhaseOffset': float(params.get('monaural_ampOscPhaseOffset', 0.0)) # AM Phase for monaural
    }
    beat_freq         = beat_params['beatFreq']
    spatial_freq      = float(params.get('spatialBeatFreq', beat_freq)) # Default to beatFreq
    spatial_phase_off = float(params.get('spatialPhaseOffset', 0.0)) # Radians

    # --- SAM controls ---
    amp               = float(params.get('amp', 0.7)) # Overall amplitude for HRTF input
    path_radius       = float(params.get('pathRadius', 1.0)) # Normalized radius factor
    frame_dur_ms      = float(params.get('frame_dur_ms', 46.4))
    overlap_fac       = int(params.get('overlap_factor',   8))

    N = int(duration * sample_rate)
    if N <= 0:
        return np.zeros((0,2), dtype=np.float32)

    # Generate core stereo beat & collapse to mono
    beat_stereo = monaural_beat_stereo_amps(duration, sample_rate, **beat_params)
    mono_beat   = np.mean(beat_stereo, axis=1).astype(np.float32)

    # Call Numba helper to get mod_beat + az/el arrays
    mod_beat, azimuth_deg, elevation_deg = _prepare_beats_and_angles(
        mono_beat, float(sample_rate),
        sam_aOD, sam_aOF, sam_aOP, # SAM-specific AM params
        spatial_freq, path_radius,
        spatial_phase_off
    )

    # --- OLA + HRTF (assuming slab and HRTF are available and configured) ---
    stereo_out = np.zeros((N, 2), dtype=np.float32)
    stereo_out[:, 0] = mod_beat * amp
    stereo_out[:, 1] = mod_beat * amp # Simple mono duplicate
    max_val = np.max(np.abs(stereo_out))
    if max_val > 1.0:
       stereo_out /= (max_val / 0.98)
    return stereo_out


# -----------------------------------------------------------------------------
# Spatial Angle Modulation - Core for Transition (_prepare_beats_and_angles_transition_core) - NEW
# -----------------------------------------------------------------------------
@numba.njit(parallel=True, fastmath=True)
def _prepare_beats_and_angles_transition_core(
    mono_input: np.ndarray, # Already transitional mono beat
    sample_rate: float,
    sAOD: float, eAOD: float,       # Start/End SAM Amplitude Osc Depth
    sAOF: float, eAOF: float,       # Start/End SAM Amplitude Osc Freq
    sAOP: float, eAOP: float,       # Start/End SAM Amplitude Osc Phase Offset
    sSpatialFreq: float, eSpatialFreq: float,
    sPathRadius: float, ePathRadius: float,
    sSpatialPhaseOff: float, eSpatialPhaseOff: float # Start/End for initial spatial phase
):
    N = mono_input.shape[0]
    if N == 0:
        return (np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32))

    mod_beat    = np.empty(N, dtype=np.float32)
    azimuth_deg = np.empty(N, dtype=np.float32)
    elevation_deg = np.zeros(N, dtype=np.float32) # Elevation is kept at zero
    
    actual_spatial_phase = np.empty(N, dtype=np.float64) # For storing accumulated phase

    dt = 1.0 / sample_rate

    # Loop 1 (parallel): Interpolate AM params and calculate mod_beat
    for i in numba.prange(N):
        alpha = i / (N - 1) if N > 1 else 0.0
        t_i = i * dt

        current_aOD = sAOD + (eAOD - sAOD) * alpha
        current_aOF = sAOF + (eAOF - sAOF) * alpha
        current_aOP = sAOP + (eAOP - sAOP) * alpha
        
        env_factor = 1.0
        if current_aOD > 0.0 and current_aOF > 0.0: # Assuming depth 0-2
            clamped_aOD_i = min(max(current_aOD, 0.0), 2.0)
            env_factor = (1.0 - clamped_aOD_i/2.0) + \
                         (clamped_aOD_i/2.0) * math.sin(2*math.pi*current_aOF*t_i + current_aOP)
        mod_beat[i] = mono_input[i] * env_factor

    # Loop 2 (sequential): Interpolate spatial freq and accumulate spatial phase
    # The 'spatial_phase_off' transition is for the initial phase offset value.
    initial_phase_offset_val = sSpatialPhaseOff + (eSpatialPhaseOff - sSpatialPhaseOff) * 0.0 # Value at alpha=0
    
    current_phase_val = initial_phase_offset_val 
    if N > 0:
      actual_spatial_phase[0] = current_phase_val

    for i in range(N): # Must be sequential due to phase accumulation
        alpha = i / (N - 1) if N > 1 else 0.0
        current_sf_i = sSpatialFreq + (eSpatialFreq - sSpatialFreq) * alpha
        
        if i > 0: # Accumulate phase
            current_phase_val += (2 * math.pi * current_sf_i * dt)
            actual_spatial_phase[i] = current_phase_val
        elif i == 0: # Already set for i=0 if N>0
             actual_spatial_phase[i] = current_phase_val


    # Loop 3 (parallel): Interpolate path radius and calculate azimuth
    for i in numba.prange(N):
        alpha = i / (N - 1) if N > 1 else 0.0
        current_path_r = sPathRadius + (ePathRadius - sPathRadius) * alpha
        
        r_factor = current_path_r * (0.5 * (mod_beat[i] + 1.0))
        
        x_coord = r_factor * math.sin(actual_spatial_phase[i])
        y_coord = r_factor * math.cos(actual_spatial_phase[i])
        azimuth_deg[i] = math.degrees(math.atan2(x_coord, y_coord))
        
    return mod_beat, azimuth_deg, elevation_deg


# -----------------------------------------------------------------------------
# Spatial Angle Modulation - Transition Version - NEW
# -----------------------------------------------------------------------------
def spatial_angle_modulation_monaural_beat_transition(
    duration,
    sample_rate=44100,
    **params
):
    N = int(duration * sample_rate)
    if N <= 0:
        return np.zeros((0,2), dtype=np.float32)

    # --- Parameters for the underlying monaural_beat_stereo_amps_transition ---
    s_ll = float(params.get('start_amp_lower_L', params.get('amp_lower_L', 0.5)))
    e_ll = float(params.get('end_amp_lower_L',   s_ll))
    s_ul = float(params.get('start_amp_upper_L', params.get('amp_upper_L', 0.5)))
    e_ul = float(params.get('end_amp_upper_L',   s_ul))
    s_lr = float(params.get('start_amp_lower_R', params.get('amp_lower_R', 0.5)))
    e_lr = float(params.get('end_amp_lower_R',   s_lr))
    s_ur = float(params.get('start_amp_upper_R', params.get('amp_upper_R', 0.5)))
    e_ur = float(params.get('end_amp_upper_R',   s_ur))
    sBF  = float(params.get('startBaseFreq',     params.get('baseFreq',    200.0)))
    eBF  = float(params.get('endBaseFreq',       sBF))
    sBt  = float(params.get('startBeatFreq',     params.get('beatFreq',      4.0)))
    eBt  = float(params.get('endBeatFreq',       sBt))
    sSPL_mono = float(params.get('startStartPhaseL_monaural', params.get('startPhaseL', 0.0)))
    eSPL_mono = float(params.get('endStartPhaseL_monaural', sSPL_mono))
    sSPU_mono = float(params.get('startStartPhaseU_monaural', params.get('startPhaseR', 0.0)))
    eSPU_mono = float(params.get('endStartPhaseU_monaural', sSPU_mono))
    sPhiF_mono = float(params.get('startPhaseOscFreq_monaural', params.get('phaseOscFreq', 0.0)))
    ePhiF_mono = float(params.get('endPhaseOscFreq_monaural', sPhiF_mono))
    sPhiR_mono = float(params.get('startPhaseOscRange_monaural', params.get('phaseOscRange', 0.0)))
    ePhiR_mono = float(params.get('endPhaseOscRange_monaural', sPhiR_mono))
    sAOD_mono = float(params.get('startAmpOscDepth_monaural', params.get('monaural_ampOscDepth', 0.0)))
    eAOD_mono = float(params.get('endAmpOscDepth_monaural', sAOD_mono))
    sAOF_mono = float(params.get('startAmpOscFreq_monaural', params.get('monaural_ampOscFreq', 0.0)))
    eAOF_mono = float(params.get('endAmpOscFreq_monaural', sAOF_mono))
    sAOP_mono = float(params.get('startAmpOscPhaseOffset_monaural', params.get('monaural_ampOscPhaseOffset', 0.0)))
    eAOP_mono = float(params.get('endAmpOscPhaseOffset_monaural', sAOP_mono))


    monaural_trans_params = {
        'start_amp_lower_L': s_ll, 'end_amp_lower_L': e_ll,
        'start_amp_upper_L': s_ul, 'end_amp_upper_L': e_ul,
        'start_amp_lower_R': s_lr, 'end_amp_lower_R': e_lr,
        'start_amp_upper_R': s_ur, 'end_amp_upper_R': e_ur,
        'startBaseFreq': sBF, 'endBaseFreq': eBF,
        'startBeatFreq': sBt, 'endBeatFreq': eBt,
        'startStartPhaseL': sSPL_mono, 'endStartPhaseL': eSPL_mono,
        'startStartPhaseU': sSPU_mono, 'endStartPhaseU': eSPU_mono,
        'startPhaseOscFreq': sPhiF_mono, 'endPhaseOscFreq': ePhiF_mono,
        'startPhaseOscRange': sPhiR_mono, 'endPhaseOscRange': ePhiR_mono,
        'startAmpOscDepth': sAOD_mono, 'endAmpOscDepth': eAOD_mono,
        'startAmpOscFreq': sAOF_mono, 'endAmpOscFreq': eAOF_mono,
        'startAmpOscPhaseOffset': sAOP_mono, 'endAmpOscPhaseOffset': eAOP_mono,
    }

    # --- Parameters for the SAM stage AM and spatialization (transitional) ---
    sSamAOD = float(params.get('start_sam_ampOscDepth', params.get('sam_ampOscDepth', 0.0)))
    eSamAOD = float(params.get('end_sam_ampOscDepth', sSamAOD))
    sSamAOF = float(params.get('start_sam_ampOscFreq', params.get('sam_ampOscFreq', 0.0)))
    eSamAOF = float(params.get('end_sam_ampOscFreq', sSamAOF))
    sSamAOP = float(params.get('start_sam_ampOscPhaseOffset', params.get('sam_ampOscPhaseOffset', 0.0)))
    eSamAOP = float(params.get('end_sam_ampOscPhaseOffset', sSamAOP))

    default_spatial_freq = (sBt + eBt) / 2.0 # Default to average beatFreq
    sSpatialFreq = float(params.get('startSpatialBeatFreq', params.get('spatialBeatFreq', default_spatial_freq)))
    eSpatialFreq = float(params.get('endSpatialBeatFreq', sSpatialFreq))
    
    sSpatialPhaseOff = float(params.get('startSpatialPhaseOffset', params.get('spatialPhaseOffset', 0.0)))
    eSpatialPhaseOff = float(params.get('endSpatialPhaseOffset', sSpatialPhaseOff))

    sPathRadius = float(params.get('startPathRadius', params.get('pathRadius', 1.0)))
    ePathRadius = float(params.get('endPathRadius', sPathRadius))

    # --- SAM controls (non-transitional for OLA) ---
    sAmp = float(params.get('startAmp', params.get('amp', 0.7))) # Overall amplitude for HRTF input
    eAmp = float(params.get('endAmp', sAmp))
    # For OLA, amp is applied per frame. We can interpolate it if needed, or use an average.
    # For now, let's use interpolated amp for mono_src
    
    frame_dur_ms = float(params.get('frame_dur_ms', 46.4))
    overlap_fac  = int(params.get('overlap_factor',   8))


    # 1. Generate transitional monaural beat
    trans_beat_stereo = monaural_beat_stereo_amps_transition(duration, sample_rate, **monaural_trans_params)
    trans_mono_beat   = np.mean(trans_beat_stereo, axis=1).astype(np.float32)

    # 2. Call the new transitional _prepare_beats_and_angles_transition_core
    trans_mod_beat, trans_azimuth_deg, trans_elevation_deg = \
        _prepare_beats_and_angles_transition_core(
            trans_mono_beat, float(sample_rate),
            sSamAOD, eSamAOD, sSamAOF, eSamAOF, sSamAOP, eSamAOP,
            sSpatialFreq, eSpatialFreq,
            sPathRadius, ePathRadius,
            sSpatialPhaseOff, eSpatialPhaseOff
        )
    
    # 3. OLA + HRTF processing (using transitional mod_beat and azimuth)
    # This part remains Python-based.
    # Placeholder for slab integration - replace with actual calls.
    # try:
    #     import slab
    # except ImportError:
    #     print("Slab library not found. HRTF processing will be skipped for transition function.")
    # ...etc

    print("spatial_angle_modulation_monaural_beat_transition: HRTF processing part is illustrative.")
    # Fallback: return a simple stereo mix of trans_mod_beat with interpolated amp
    final_amp_coeffs = np.linspace(sAmp, eAmp, N, dtype=np.float32) if N > 0 else np.array([], dtype=np.float32)
    
    temp_out = np.zeros((N, 2), dtype=np.float32)
    if N > 0:
       temp_out[:, 0] = trans_mod_beat * final_amp_coeffs
       temp_out[:, 1] = trans_mod_beat * final_amp_coeffs
    
    max_v = np.max(np.abs(temp_out)) if N > 0 and temp_out.size > 0 else 0.0
    if max_v > 1.0: temp_out /= (max_v / 0.98)
    return temp_out


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

# -----------------------------------------------------------------------------
# Quadrature Amplitude Modulation (QAM) 
# -----------------------------------------------------------------------------


# --- Filter design and application (from your qam.py, not Numba compatible) ---
def design_filter(filter_type, cutoff, fs, order=4):
    """
    Designs a Butterworth filter.
    Args:
        filter_type (str): 'highpass', 'lowpass', 'bandpass', 'bandstop'.
        cutoff (float or list): Cutoff frequency or frequencies.
        fs (float): Sampling frequency.
        order (int): Filter order.
    Returns:
        ndarray: Second-order sections representation of the filter.
    """
    nyq = 0.5 * fs
    norm_cutoff = np.array(cutoff) / nyq if isinstance(cutoff, (list, tuple)) else cutoff / nyq
    return butter(order, norm_cutoff, btype=filter_type, output='sos')

def apply_filters(signal_segment, fs):
    """
    Applies pre-defined high-pass and low-pass filters to a signal segment.
    Args:
        signal_segment (np.ndarray): Input signal (1D).
        fs (float): Sampling frequency.
    Returns:
        np.ndarray: Filtered signal.
    """
    if signal_segment.ndim == 0 or signal_segment.size == 0:
        return signal_segment # Return empty or scalar if no data to filter
    if fs <= 0: # Invalid sampling rate
        return signal_segment
    
    # Define cutoff frequencies ensuring they are valid
    hp_cutoff = 30.0
    lp_cutoff = fs * 0.5 * 0.9 

    if hp_cutoff <= 0 or hp_cutoff >= fs / 2: # Invalid high-pass cutoff
        pass # Skip high-pass if invalid
    else:
        sos_hp = design_filter('highpass', hp_cutoff, fs)
        signal_segment = sosfiltfilt(sos_hp, signal_segment)

    if lp_cutoff <= 0 or lp_cutoff >= fs / 2: # Invalid low-pass cutoff
        pass # Skip low-pass if invalid
    else:
        sos_lp = design_filter('lowpass', lp_cutoff, fs)
        signal_segment = sosfiltfilt(sos_lp, signal_segment)
        
    return signal_segment

# -----------------------------------------------------------------------------
# QAM Beat
# -----------------------------------------------------------------------------
def qam_beat(duration, sample_rate=44100, **params):
    """
    Generates a QAM-based binaural beat.
    Each channel's carrier is amplitude modulated.
    Optional phase oscillation can be applied between channels.
    """
    # --- Unpack synthesis parameters ---
    ampL = float(params.get('ampL', 0.5))
    ampR = float(params.get('ampR', 0.5))
    
    baseFreqL = float(params.get('baseFreqL', 200.0)) 
    baseFreqR = float(params.get('baseFreqR', baseFreqL + 4.0)) 

    qamAmFreqL = float(params.get('qamAmFreqL', 4.0)) 
    qamAmDepthL = float(params.get('qamAmDepthL', 0.5)) 
    qamAmPhaseOffsetL = float(params.get('qamAmPhaseOffsetL', 0.0))

    qamAmFreqR = float(params.get('qamAmFreqR', qamAmFreqL)) 
    qamAmDepthR = float(params.get('qamAmDepthR', qamAmDepthL)) 
    qamAmPhaseOffsetR = float(params.get('qamAmPhaseOffsetR', qamAmPhaseOffsetL))

    startPhaseL = float(params.get('startPhaseL', 0.0)) 
    startPhaseR = float(params.get('startPhaseR', 0.0)) 

    phaseOscFreq = float(params.get('phaseOscFreq', 0.0))
    phaseOscRange = float(params.get('phaseOscRange', 0.0)) 
    phaseOscPhaseOffset = float(params.get('phaseOscPhaseOffset', 0.0))

    N = int(duration * sample_rate)
    if N <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    # Call Numba core function
    raw_signal = _qam_beat_core(
        N, duration, float(sample_rate),
        ampL, ampR,
        baseFreqL, baseFreqR,
        qamAmFreqL, qamAmDepthL, qamAmPhaseOffsetL,
        qamAmFreqR, qamAmDepthR, qamAmPhaseOffsetR,
        startPhaseL, startPhaseR,
        phaseOscFreq, phaseOscRange, phaseOscPhaseOffset
    )

    if raw_signal.size > 0 :
        filtered_L = apply_filters(raw_signal[:, 0].copy(), float(sample_rate))
        filtered_R = apply_filters(raw_signal[:, 1].copy(), float(sample_rate))
        return np.ascontiguousarray(np.vstack((filtered_L, filtered_R)).T.astype(np.float32))
    else:
        return raw_signal


@numba.njit(parallel=True, fastmath=True)
def _qam_beat_core(
    N, duration_float, sample_rate_float,
    ampL, ampR,
    baseFreqL, baseFreqR,
    qamAmFreqL, qamAmDepthL, qamAmPhaseOffsetL,
    qamAmFreqR, qamAmDepthR, qamAmPhaseOffsetR,
    startPhaseL, startPhaseR,
    phaseOscFreq, phaseOscRange, phaseOscPhaseOffset
):
    if N <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    t_arr = np.empty(N, dtype=np.float64)
    dt = duration_float / N
    for i in numba.prange(N):
        t_arr[i] = i * dt

    phL_carrier = np.empty(N, dtype=np.float64)
    phR_carrier = np.empty(N, dtype=np.float64)
    
    currentPhaseL = startPhaseL
    currentPhaseR = startPhaseR
    for i in range(N): 
        phL_carrier[i] = currentPhaseL
        phR_carrier[i] = currentPhaseR
        currentPhaseL += 2 * np.pi * baseFreqL * dt
        currentPhaseR += 2 * np.pi * baseFreqR * dt
        
    if phaseOscFreq != 0.0 or phaseOscRange != 0.0:
        for i in numba.prange(N):
            d_phi = (phaseOscRange / 2.0) * np.sin(2 * np.pi * phaseOscFreq * t_arr[i] + phaseOscPhaseOffset)
            phL_carrier[i] -= d_phi
            phR_carrier[i] += d_phi

    envL = np.empty(N, dtype=np.float64)
    envR = np.empty(N, dtype=np.float64)
    for i in numba.prange(N):
        if qamAmFreqL != 0.0 and qamAmDepthL != 0.0:
            envL[i] = 1.0 + qamAmDepthL * np.cos(2 * np.pi * qamAmFreqL * t_arr[i] + qamAmPhaseOffsetL)
        else:
            envL[i] = 1.0
        
        if qamAmFreqR != 0.0 and qamAmDepthR != 0.0:
            envR[i] = 1.0 + qamAmDepthR * np.cos(2 * np.pi * qamAmFreqR * t_arr[i] + qamAmPhaseOffsetR)
        else:
            envR[i] = 1.0

    out = np.empty((N, 2), dtype=np.float32)
    for i in numba.prange(N):
        sigL = envL[i] * np.cos(phL_carrier[i])
        sigR = envR[i] * np.cos(phR_carrier[i])
        
        out[i, 0] = np.float32(sigL * ampL)
        out[i, 1] = np.float32(sigR * ampR)

    return out


# -----------------------------------------------------------------------------
# QAM Beat Transition
# -----------------------------------------------------------------------------
def qam_beat_transition(duration, sample_rate=44100, **params):
    """
    Generates a QAM-based binaural beat with parameters linearly interpolated over the duration.
    """
    s_ampL = float(params.get('startAmpL', params.get('ampL', 0.5)))
    e_ampL = float(params.get('endAmpL', s_ampL))
    s_ampR = float(params.get('startAmpR', params.get('ampR', 0.5)))
    e_ampR = float(params.get('endAmpR', s_ampR))

    s_baseFreqL = float(params.get('startBaseFreqL', params.get('baseFreqL', 200.0)))
    e_baseFreqL = float(params.get('endBaseFreqL', s_baseFreqL))
    s_baseFreqR = float(params.get('startBaseFreqR', params.get('baseFreqR', s_baseFreqL + 4.0)))
    e_baseFreqR = float(params.get('endBaseFreqR', s_baseFreqR))

    s_qamAmFreqL = float(params.get('startQamAmFreqL', params.get('qamAmFreqL', 4.0)))
    e_qamAmFreqL = float(params.get('endQamAmFreqL', s_qamAmFreqL))
    s_qamAmFreqR = float(params.get('startQamAmFreqR', params.get('qamAmFreqR', s_qamAmFreqL)))
    e_qamAmFreqR = float(params.get('endQamAmFreqR', s_qamAmFreqR))

    s_qamAmDepthL = float(params.get('startQamAmDepthL', params.get('qamAmDepthL', 0.5)))
    e_qamAmDepthL = float(params.get('endQamAmDepthL', s_qamAmDepthL))
    s_qamAmDepthR = float(params.get('startQamAmDepthR', params.get('qamAmDepthR', s_qamAmDepthL)))
    e_qamAmDepthR = float(params.get('endQamAmDepthR', s_qamAmDepthR))

    s_qamAmPhaseOffsetL = float(params.get('startQamAmPhaseOffsetL', params.get('qamAmPhaseOffsetL', 0.0)))
    e_qamAmPhaseOffsetL = float(params.get('endQamAmPhaseOffsetL', s_qamAmPhaseOffsetL))
    s_qamAmPhaseOffsetR = float(params.get('startQamAmPhaseOffsetR', params.get('qamAmPhaseOffsetR', s_qamAmPhaseOffsetL)))
    e_qamAmPhaseOffsetR = float(params.get('endQamAmPhaseOffsetR', s_qamAmPhaseOffsetR))

    s_startPhaseL = float(params.get('startStartPhaseL', params.get('startPhaseL', 0.0)))
    e_startPhaseL = float(params.get('endStartPhaseL', s_startPhaseL)) 
    s_startPhaseR = float(params.get('startStartPhaseR', params.get('startPhaseR', 0.0)))
    e_startPhaseR = float(params.get('endStartPhaseR', s_startPhaseR))

    s_phaseOscFreq = float(params.get('startPhaseOscFreq', params.get('phaseOscFreq', 0.0)))
    e_phaseOscFreq = float(params.get('endPhaseOscFreq', s_phaseOscFreq))
    s_phaseOscRange = float(params.get('startPhaseOscRange', params.get('phaseOscRange', 0.0)))
    e_phaseOscRange = float(params.get('endPhaseOscRange', s_phaseOscRange))
    s_phaseOscPhaseOffset = float(params.get('startPhaseOscPhaseOffset', params.get('phaseOscPhaseOffset', 0.0)))
    e_phaseOscPhaseOffset = float(params.get('endPhaseOscPhaseOffset', s_phaseOscPhaseOffset))

    N = int(duration * sample_rate)
    if N <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    raw_signal = _qam_beat_transition_core(
        N, float(duration), float(sample_rate),
        s_ampL, e_ampL, s_ampR, e_ampR,
        s_baseFreqL, e_baseFreqL, s_baseFreqR, e_baseFreqR,
        s_qamAmFreqL, e_qamAmFreqL, s_qamAmDepthL, e_qamAmDepthL, s_qamAmPhaseOffsetL, e_qamAmPhaseOffsetL,
        s_qamAmFreqR, e_qamAmFreqR, s_qamAmDepthR, e_qamAmDepthR, s_qamAmPhaseOffsetR, e_qamAmPhaseOffsetR,
        s_startPhaseL, e_startPhaseL, s_startPhaseR, e_startPhaseR, 
        s_phaseOscFreq, e_phaseOscFreq, s_phaseOscRange, e_phaseOscRange, s_phaseOscPhaseOffset, e_phaseOscPhaseOffset
    )
    
    if raw_signal.size > 0:
        filtered_L = apply_filters(raw_signal[:, 0].copy(), float(sample_rate))
        filtered_R = apply_filters(raw_signal[:, 1].copy(), float(sample_rate))
        return np.ascontiguousarray(np.vstack((filtered_L, filtered_R)).T.astype(np.float32))
    else:
        return raw_signal


@numba.njit(parallel=True, fastmath=True)
def _qam_beat_transition_core(
    N, duration_float, sample_rate_float,
    s_ampL, e_ampL, s_ampR, e_ampR,
    s_baseFreqL, e_baseFreqL, s_baseFreqR, e_baseFreqR,
    s_qamAmFreqL, e_qamAmFreqL, s_qamAmDepthL, e_qamAmDepthL, s_qamAmPhaseOffsetL, e_qamAmPhaseOffsetL,
    s_qamAmFreqR, e_qamAmFreqR, s_qamAmDepthR, e_qamAmDepthR, s_qamAmPhaseOffsetR, e_qamAmPhaseOffsetR,
    s_startPhaseL_init, e_startPhaseL_init, s_startPhaseR_init, e_startPhaseR_init, 
    s_phaseOscFreq, e_phaseOscFreq, s_phaseOscRange, e_phaseOscRange, s_phaseOscPhaseOffset, e_phaseOscPhaseOffset
):
    if N <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    t_arr = np.empty(N, dtype=np.float64)
    alpha_arr = np.empty(N, dtype=np.float64) 
    dt = duration_float / N
    for i in numba.prange(N):
        t_arr[i] = i * dt
        alpha_arr[i] = i / (N - 1) if N > 1 else 0.0

    ampL_arr = np.empty(N, dtype=np.float64)
    ampR_arr = np.empty(N, dtype=np.float64)
    baseFreqL_arr = np.empty(N, dtype=np.float64)
    baseFreqR_arr = np.empty(N, dtype=np.float64)
    qamAmFreqL_arr = np.empty(N, dtype=np.float64)
    qamAmDepthL_arr = np.empty(N, dtype=np.float64)
    qamAmPhaseOffsetL_arr = np.empty(N, dtype=np.float64)
    qamAmFreqR_arr = np.empty(N, dtype=np.float64)
    qamAmDepthR_arr = np.empty(N, dtype=np.float64)
    qamAmPhaseOffsetR_arr = np.empty(N, dtype=np.float64)
    phaseOscFreq_arr = np.empty(N, dtype=np.float64)
    phaseOscRange_arr = np.empty(N, dtype=np.float64)
    phaseOscPhaseOffset_arr = np.empty(N, dtype=np.float64)
    
    for i in numba.prange(N):
        alpha = alpha_arr[i]
        ampL_arr[i] = s_ampL + (e_ampL - s_ampL) * alpha
        ampR_arr[i] = s_ampR + (e_ampR - s_ampR) * alpha
        baseFreqL_arr[i] = s_baseFreqL + (e_baseFreqL - s_baseFreqL) * alpha
        baseFreqR_arr[i] = s_baseFreqR + (e_baseFreqR - s_baseFreqR) * alpha
        qamAmFreqL_arr[i] = s_qamAmFreqL + (e_qamAmFreqL - s_qamAmFreqL) * alpha
        qamAmDepthL_arr[i] = s_qamAmDepthL + (e_qamAmDepthL - s_qamAmDepthL) * alpha
        qamAmPhaseOffsetL_arr[i] = s_qamAmPhaseOffsetL + (e_qamAmPhaseOffsetL - s_qamAmPhaseOffsetL) * alpha
        qamAmFreqR_arr[i] = s_qamAmFreqR + (e_qamAmFreqR - s_qamAmFreqR) * alpha
        qamAmDepthR_arr[i] = s_qamAmDepthR + (e_qamAmDepthR - s_qamAmDepthR) * alpha
        qamAmPhaseOffsetR_arr[i] = s_qamAmPhaseOffsetR + (e_qamAmPhaseOffsetR - s_qamAmPhaseOffsetR) * alpha
        phaseOscFreq_arr[i] = s_phaseOscFreq + (e_phaseOscFreq - s_phaseOscFreq) * alpha
        phaseOscRange_arr[i] = s_phaseOscRange + (e_phaseOscRange - s_phaseOscRange) * alpha
        phaseOscPhaseOffset_arr[i] = s_phaseOscPhaseOffset + (e_phaseOscPhaseOffset - s_phaseOscPhaseOffset) * alpha

    phL_carrier = np.empty(N, dtype=np.float64)
    phR_carrier = np.empty(N, dtype=np.float64)
    
    currentPhaseL = s_startPhaseL_init 
    currentPhaseR = s_startPhaseR_init
    for i in range(N): 
        phL_carrier[i] = currentPhaseL
        phR_carrier[i] = currentPhaseR
        currentPhaseL += 2 * np.pi * baseFreqL_arr[i] * dt
        currentPhaseR += 2 * np.pi * baseFreqR_arr[i] * dt
        
    for i in numba.prange(N):
        if phaseOscFreq_arr[i] != 0.0 or phaseOscRange_arr[i] != 0.0:
            d_phi = (phaseOscRange_arr[i] / 2.0) * np.sin(2 * np.pi * phaseOscFreq_arr[i] * t_arr[i] + phaseOscPhaseOffset_arr[i])
            phL_carrier[i] -= d_phi
            phR_carrier[i] += d_phi

    envL_arr = np.empty(N, dtype=np.float64)
    envR_arr = np.empty(N, dtype=np.float64)
    for i in numba.prange(N):
        if qamAmFreqL_arr[i] != 0.0 and qamAmDepthL_arr[i] != 0.0:
            envL_arr[i] = 1.0 + qamAmDepthL_arr[i] * np.cos(2 * np.pi * qamAmFreqL_arr[i] * t_arr[i] + qamAmPhaseOffsetL_arr[i])
        else:
            envL_arr[i] = 1.0
        
        if qamAmFreqR_arr[i] != 0.0 and qamAmDepthR_arr[i] != 0.0:
            envR_arr[i] = 1.0 + qamAmDepthR_arr[i] * np.cos(2 * np.pi * qamAmFreqR_arr[i] * t_arr[i] + qamAmPhaseOffsetR_arr[i])
        else:
            envR_arr[i] = 1.0

    out = np.empty((N, 2), dtype=np.float32)
    for i in numba.prange(N):
        sigL = envL_arr[i] * np.cos(phL_carrier[i])
        sigR = envR_arr[i] * np.cos(phR_carrier[i])
        
        out[i, 0] = np.float32(sigL * ampL_arr[i])
        out[i, 1] = np.float32(sigR * ampR_arr[i])

    return out


# -----------------------------------------------------------------------------
# Hybrid QAM (Left) - Monaural (Right) Beat
# -----------------------------------------------------------------------------
def hybrid_qam_monaural_beat(duration, sample_rate=44100, **params):
    """
    Generates a hybrid signal:
    Left Channel: QAM-style modulated signal.
    Right Channel: Monaural beat with its own AM, FM, and phase oscillation.
    """
    ampL = float(params.get('ampL', 0.5)) 
    ampR = float(params.get('ampR', 0.5)) 

    qam_L_carrierFreq = float(params.get('qamCarrierFreqL', 100.0)) 
    qam_L_amFreq = float(params.get('qamAmFreqL', 4.0))
    qam_L_amDepth = float(params.get('qamAmDepthL', 0.5)) 
    qam_L_amPhaseOffset = float(params.get('qamAmPhaseOffsetL', 0.0)) 
    qam_L_startPhase = float(params.get('qamStartPhaseL', 0.0)) 

    mono_R_carrierFreq = float(params.get('monoCarrierFreqR', 100.0)) 
    mono_R_beatFreqInChannel = float(params.get('monoBeatFreqInChannelR', 4.0)) 

    mono_R_amDepth = float(params.get('monoAmDepthR', 0.0)) 
    mono_R_amFreq = float(params.get('monoAmFreqR', 0.0))
    mono_R_amPhaseOffset = float(params.get('monoAmPhaseOffsetR', 0.0)) 

    mono_R_fmRange = float(params.get('monoFmRangeR', 0.0)) 
    mono_R_fmFreq = float(params.get('monoFmFreqR', 0.0)) 
    mono_R_fmPhaseOffset = float(params.get('monoFmPhaseOffsetR', 0.0)) 

    mono_R_startPhaseTone1 = float(params.get('monoStartPhaseR_Tone1', 0.0)) 
    mono_R_startPhaseTone2 = float(params.get('monoStartPhaseR_Tone2', 0.0)) 
    
    mono_R_phaseOscFreq = float(params.get('monoPhaseOscFreqR', 0.0))
    mono_R_phaseOscRange = float(params.get('monoPhaseOscRangeR', 0.0)) 
    mono_R_phaseOscPhaseOffset = float(params.get('monoPhaseOscPhaseOffsetR', 0.0)) 

    N = int(duration * sample_rate)
    if N <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    raw_signal = _hybrid_qam_monaural_beat_core(
        N, float(duration), float(sample_rate),
        ampL, ampR,
        qam_L_carrierFreq, qam_L_amFreq, qam_L_amDepth, qam_L_amPhaseOffset, qam_L_startPhase,
        mono_R_carrierFreq, mono_R_beatFreqInChannel,
        mono_R_amDepth, mono_R_amFreq, mono_R_amPhaseOffset,
        mono_R_fmRange, mono_R_fmFreq, mono_R_fmPhaseOffset,
        mono_R_startPhaseTone1, mono_R_startPhaseTone2,
        mono_R_phaseOscFreq, mono_R_phaseOscRange, mono_R_phaseOscPhaseOffset
    )

    if raw_signal.size > 0:
        filtered_L = apply_filters(raw_signal[:, 0].copy(), float(sample_rate))
        filtered_R = apply_filters(raw_signal[:, 1].copy(), float(sample_rate))
        return np.ascontiguousarray(np.vstack((filtered_L, filtered_R)).T.astype(np.float32))
    else:
        return raw_signal


@numba.njit(parallel=True, fastmath=True)
def _hybrid_qam_monaural_beat_core(
    N, duration_float, sample_rate_float,
    ampL, ampR,
    qam_L_carrierFreq, qam_L_amFreq, qam_L_amDepth, qam_L_amPhaseOffset, qam_L_startPhase,
    mono_R_carrierFreq_base, mono_R_beatFreqInChannel, 
    mono_R_amDepth, mono_R_amFreq, mono_R_amPhaseOffset,
    mono_R_fmRange, mono_R_fmFreq, mono_R_fmPhaseOffset,
    mono_R_startPhaseTone1, mono_R_startPhaseTone2,
    mono_R_phaseOscFreq, mono_R_phaseOscRange, mono_R_phaseOscPhaseOffset
):
    if N <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    t_arr = np.empty(N, dtype=np.float64)
    dt = duration_float / N
    for i in numba.prange(N):
        t_arr[i] = i * dt

    out = np.empty((N, 2), dtype=np.float32)

    ph_qam_L_carrier = np.empty(N, dtype=np.float64)
    currentPhaseQAM_L = qam_L_startPhase
    for i in range(N): 
        ph_qam_L_carrier[i] = currentPhaseQAM_L
        currentPhaseQAM_L += 2 * np.pi * qam_L_carrierFreq * dt 

    env_qam_L = np.empty(N, dtype=np.float64)
    for i in numba.prange(N):
        if qam_L_amFreq != 0.0 and qam_L_amDepth != 0.0:
            env_qam_L[i] = 1.0 + qam_L_amDepth * np.cos(2 * np.pi * qam_L_amFreq * t_arr[i] + qam_L_amPhaseOffset)
        else:
            env_qam_L[i] = 1.0
        
        sig_qam_L = env_qam_L[i] * np.cos(ph_qam_L_carrier[i])
        out[i, 0] = np.float32(sig_qam_L * ampL)

    mono_R_carrierFreq_inst = np.empty(N, dtype=np.float64)
    for i in numba.prange(N):
        if mono_R_fmFreq != 0.0 and mono_R_fmRange != 0.0:
            fm_mod_signal = (mono_R_fmRange / 2.0) * np.sin(2 * np.pi * mono_R_fmFreq * t_arr[i] + mono_R_fmPhaseOffset)
            mono_R_carrierFreq_inst[i] = mono_R_carrierFreq_base + fm_mod_signal
        else:
            mono_R_carrierFreq_inst[i] = mono_R_carrierFreq_base
        if mono_R_carrierFreq_inst[i] < 0: mono_R_carrierFreq_inst[i] = 0.0

    half_mono_beat_R = mono_R_beatFreqInChannel / 2.0
    mono_R_freqTone1_inst = np.empty(N, dtype=np.float64)
    mono_R_freqTone2_inst = np.empty(N, dtype=np.float64)
    for i in numba.prange(N):
        mono_R_freqTone1_inst[i] = mono_R_carrierFreq_inst[i] - half_mono_beat_R
        mono_R_freqTone2_inst[i] = mono_R_carrierFreq_inst[i] + half_mono_beat_R
        if mono_R_freqTone1_inst[i] < 0: mono_R_freqTone1_inst[i] = 0.0
        if mono_R_freqTone2_inst[i] < 0: mono_R_freqTone2_inst[i] = 0.0

    ph_mono_R_tone1 = np.empty(N, dtype=np.float64)
    ph_mono_R_tone2 = np.empty(N, dtype=np.float64)
    currentPhaseMonoR1 = mono_R_startPhaseTone1
    currentPhaseMonoR2 = mono_R_startPhaseTone2
    for i in range(N): 
        ph_mono_R_tone1[i] = currentPhaseMonoR1
        ph_mono_R_tone2[i] = currentPhaseMonoR2
        currentPhaseMonoR1 += 2 * np.pi * mono_R_freqTone1_inst[i] * dt
        currentPhaseMonoR2 += 2 * np.pi * mono_R_freqTone2_inst[i] * dt

    if mono_R_phaseOscFreq != 0.0 or mono_R_phaseOscRange != 0.0:
        for i in numba.prange(N):
            d_phi_mono = (mono_R_phaseOscRange / 2.0) * np.sin(2 * np.pi * mono_R_phaseOscFreq * t_arr[i] + mono_R_phaseOscPhaseOffset)
            ph_mono_R_tone1[i] -= d_phi_mono
            ph_mono_R_tone2[i] += d_phi_mono
            
    s_mono_R_tone1 = np.empty(N, dtype=np.float64)
    s_mono_R_tone2 = np.empty(N, dtype=np.float64)
    for i in numba.prange(N):
        s_mono_R_tone1[i] = np.sin(ph_mono_R_tone1[i])
        s_mono_R_tone2[i] = np.sin(ph_mono_R_tone2[i])
    
    summed_mono_R = np.empty(N, dtype=np.float64)
    for i in numba.prange(N):
        summed_mono_R[i] = s_mono_R_tone1[i] + s_mono_R_tone2[i] 

    env_mono_R = np.empty(N, dtype=np.float64)
    for i in numba.prange(N):
        if mono_R_amFreq != 0.0 and mono_R_amDepth != 0.0:
            clamped_am_depth_R = max(0.0, min(1.0, mono_R_amDepth)) 
            env_mono_R[i] = 1.0 - clamped_am_depth_R * (0.5 * (1.0 + np.sin(2 * np.pi * mono_R_amFreq * t_arr[i] + mono_R_amPhaseOffset)))
        else:
            env_mono_R[i] = 1.0
        
        out[i, 1] = np.float32(summed_mono_R[i] * env_mono_R[i] * ampR)

    return out


# -----------------------------------------------------------------------------
# Hybrid QAM-Monaural Beat Transition
# -----------------------------------------------------------------------------
def hybrid_qam_monaural_beat_transition(duration, sample_rate=44100, **params):
    """
    Generates a hybrid QAM-Monaural beat with parameters linearly interpolated.
    """
    s_ampL = float(params.get('startAmpL', params.get('ampL', 0.5)))
    e_ampL = float(params.get('endAmpL', s_ampL))
    s_ampR = float(params.get('startAmpR', params.get('ampR', 0.5)))
    e_ampR = float(params.get('endAmpR', s_ampR))

    s_qam_L_carrierFreq = float(params.get('startQamCarrierFreqL', params.get('qamCarrierFreqL', 100.0)))
    e_qam_L_carrierFreq = float(params.get('endQamCarrierFreqL', s_qam_L_carrierFreq))
    s_qam_L_amFreq = float(params.get('startQamAmFreqL', params.get('qamAmFreqL', 4.0)))
    e_qam_L_amFreq = float(params.get('endQamAmFreqL', s_qam_L_amFreq))
    s_qam_L_amDepth = float(params.get('startQamAmDepthL', params.get('qamAmDepthL', 0.5)))
    e_qam_L_amDepth = float(params.get('endQamAmDepthL', s_qam_L_amDepth))
    s_qam_L_amPhaseOffset = float(params.get('startQamAmPhaseOffsetL', params.get('qamAmPhaseOffsetL', 0.0)))
    e_qam_L_amPhaseOffset = float(params.get('endQamAmPhaseOffsetL', s_qam_L_amPhaseOffset))
    s_qam_L_startPhase = float(params.get('startQamStartPhaseL', params.get('qamStartPhaseL', 0.0)))
    e_qam_L_startPhase = float(params.get('endQamStartPhaseL', s_qam_L_startPhase)) 

    s_mono_R_carrierFreq = float(params.get('startMonoCarrierFreqR', params.get('monoCarrierFreqR', 100.0)))
    e_mono_R_carrierFreq = float(params.get('endMonoCarrierFreqR', s_mono_R_carrierFreq))
    s_mono_R_beatFreqInChannel = float(params.get('startMonoBeatFreqInChannelR', params.get('monoBeatFreqInChannelR', 4.0)))
    e_mono_R_beatFreqInChannel = float(params.get('endMonoBeatFreqInChannelR', s_mono_R_beatFreqInChannel))
    
    s_mono_R_amDepth = float(params.get('startMonoAmDepthR', params.get('monoAmDepthR', 0.0)))
    e_mono_R_amDepth = float(params.get('endMonoAmDepthR', s_mono_R_amDepth))
    s_mono_R_amFreq = float(params.get('startMonoAmFreqR', params.get('monoAmFreqR', 0.0)))
    e_mono_R_amFreq = float(params.get('endMonoAmFreqR', s_mono_R_amFreq))
    s_mono_R_amPhaseOffset = float(params.get('startMonoAmPhaseOffsetR', params.get('monoAmPhaseOffsetR', 0.0)))
    e_mono_R_amPhaseOffset = float(params.get('endMonoAmPhaseOffsetR', s_mono_R_amPhaseOffset))

    s_mono_R_fmRange = float(params.get('startMonoFmRangeR', params.get('monoFmRangeR', 0.0)))
    e_mono_R_fmRange = float(params.get('endMonoFmRangeR', s_mono_R_fmRange))
    s_mono_R_fmFreq = float(params.get('startMonoFmFreqR', params.get('monoFmFreqR', 0.0)))
    e_mono_R_fmFreq = float(params.get('endMonoFmFreqR', s_mono_R_fmFreq))
    s_mono_R_fmPhaseOffset = float(params.get('startMonoFmPhaseOffsetR', params.get('monoFmPhaseOffsetR', 0.0)))
    e_mono_R_fmPhaseOffset = float(params.get('endMonoFmPhaseOffsetR', s_mono_R_fmPhaseOffset))

    s_mono_R_startPhaseTone1 = float(params.get('startMonoStartPhaseR_Tone1', params.get('monoStartPhaseR_Tone1', 0.0)))
    e_mono_R_startPhaseTone1 = float(params.get('endMonoStartPhaseR_Tone1', s_mono_R_startPhaseTone1))
    s_mono_R_startPhaseTone2 = float(params.get('startMonoStartPhaseR_Tone2', params.get('monoStartPhaseR_Tone2', 0.0)))
    e_mono_R_startPhaseTone2 = float(params.get('endMonoStartPhaseR_Tone2', s_mono_R_startPhaseTone2))

    s_mono_R_phaseOscFreq = float(params.get('startMonoPhaseOscFreqR', params.get('monoPhaseOscFreqR', 0.0)))
    e_mono_R_phaseOscFreq = float(params.get('endMonoPhaseOscFreqR', s_mono_R_phaseOscFreq))
    s_mono_R_phaseOscRange = float(params.get('startMonoPhaseOscRangeR', params.get('monoPhaseOscRangeR', 0.0)))
    e_mono_R_phaseOscRange = float(params.get('endMonoPhaseOscRangeR', s_mono_R_phaseOscRange))
    s_mono_R_phaseOscPhaseOffset = float(params.get('startMonoPhaseOscPhaseOffsetR', params.get('monoPhaseOscPhaseOffsetR', 0.0)))
    e_mono_R_phaseOscPhaseOffset = float(params.get('endMonoPhaseOscPhaseOffsetR', s_mono_R_phaseOscPhaseOffset))

    N = int(duration * sample_rate)
    if N <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    raw_signal = _hybrid_qam_monaural_beat_transition_core(
        N, float(duration), float(sample_rate),
        s_ampL, e_ampL, s_ampR, e_ampR,
        s_qam_L_carrierFreq, e_qam_L_carrierFreq, s_qam_L_amFreq, e_qam_L_amFreq, 
        s_qam_L_amDepth, e_qam_L_amDepth, s_qam_L_amPhaseOffset, e_qam_L_amPhaseOffset, 
        s_qam_L_startPhase, e_qam_L_startPhase,
        s_mono_R_carrierFreq, e_mono_R_carrierFreq, s_mono_R_beatFreqInChannel, e_mono_R_beatFreqInChannel,
        s_mono_R_amDepth, e_mono_R_amDepth, s_mono_R_amFreq, e_mono_R_amFreq, 
        s_mono_R_amPhaseOffset, e_mono_R_amPhaseOffset,
        s_mono_R_fmRange, e_mono_R_fmRange, s_mono_R_fmFreq, e_mono_R_fmFreq, 
        s_mono_R_fmPhaseOffset, e_mono_R_fmPhaseOffset,
        s_mono_R_startPhaseTone1, e_mono_R_startPhaseTone1, s_mono_R_startPhaseTone2, e_mono_R_startPhaseTone2,
        s_mono_R_phaseOscFreq, e_mono_R_phaseOscFreq, s_mono_R_phaseOscRange, e_mono_R_phaseOscRange,
        s_mono_R_phaseOscPhaseOffset, e_mono_R_phaseOscPhaseOffset
    )

    if raw_signal.size > 0:
        filtered_L = apply_filters(raw_signal[:, 0].copy(), float(sample_rate))
        filtered_R = apply_filters(raw_signal[:, 1].copy(), float(sample_rate))
        return np.ascontiguousarray(np.vstack((filtered_L, filtered_R)).T.astype(np.float32))
    else:
        return raw_signal


@numba.njit(parallel=True, fastmath=True)
def _hybrid_qam_monaural_beat_transition_core(
    N, duration_float, sample_rate_float,
    s_ampL, e_ampL, s_ampR, e_ampR,
    s_qam_L_carrierFreq, e_qam_L_carrierFreq, s_qam_L_amFreq, e_qam_L_amFreq, 
    s_qam_L_amDepth, e_qam_L_amDepth, s_qam_L_amPhaseOffset, e_qam_L_amPhaseOffset, 
    s_qam_L_startPhase_init, e_qam_L_startPhase_init, 
    s_mono_R_carrierFreq_base, e_mono_R_carrierFreq_base, s_mono_R_beatFreqInChannel, e_mono_R_beatFreqInChannel,
    s_mono_R_amDepth, e_mono_R_amDepth, s_mono_R_amFreq, e_mono_R_amFreq, 
    s_mono_R_amPhaseOffset, e_mono_R_amPhaseOffset,
    s_mono_R_fmRange, e_mono_R_fmRange, s_mono_R_fmFreq, e_mono_R_fmFreq, 
    s_mono_R_fmPhaseOffset, e_mono_R_fmPhaseOffset,
    s_mono_R_startPhaseTone1_init, e_mono_R_startPhaseTone1_init, 
    s_mono_R_startPhaseTone2_init, e_mono_R_startPhaseTone2_init,
    s_mono_R_phaseOscFreq, e_mono_R_phaseOscFreq, s_mono_R_phaseOscRange, e_mono_R_phaseOscRange,
    s_mono_R_phaseOscPhaseOffset, e_mono_R_phaseOscPhaseOffset
):
    if N <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    t_arr = np.empty(N, dtype=np.float64)
    alpha_arr = np.empty(N, dtype=np.float64)
    dt = duration_float / N
    for i in numba.prange(N):
        t_arr[i] = i * dt
        alpha_arr[i] = i / (N - 1) if N > 1 else 0.0

    out = np.empty((N, 2), dtype=np.float32)

    ampL_val_arr = np.empty(N, dtype=np.float64)
    qam_L_carrierFreq_arr = np.empty(N, dtype=np.float64)
    qam_L_amFreq_arr = np.empty(N, dtype=np.float64)
    qam_L_amDepth_arr = np.empty(N, dtype=np.float64)
    qam_L_amPhaseOffset_arr = np.empty(N, dtype=np.float64)
    
    for i in numba.prange(N):
        alpha = alpha_arr[i]
        ampL_val_arr[i] = s_ampL + (e_ampL - s_ampL) * alpha
        qam_L_carrierFreq_arr[i] = s_qam_L_carrierFreq + (e_qam_L_carrierFreq - s_qam_L_carrierFreq) * alpha
        qam_L_amFreq_arr[i] = s_qam_L_amFreq + (e_qam_L_amFreq - s_qam_L_amFreq) * alpha
        qam_L_amDepth_arr[i] = s_qam_L_amDepth + (e_qam_L_amDepth - s_qam_L_amDepth) * alpha
        qam_L_amPhaseOffset_arr[i] = s_qam_L_amPhaseOffset + (e_qam_L_amPhaseOffset - s_qam_L_amPhaseOffset) * alpha

    ph_qam_L_carrier = np.empty(N, dtype=np.float64)
    currentPhaseQAM_L = s_qam_L_startPhase_init 
    for i in range(N): 
        ph_qam_L_carrier[i] = currentPhaseQAM_L
        currentPhaseQAM_L += 2 * np.pi * qam_L_carrierFreq_arr[i] * dt

    env_qam_L_arr = np.empty(N, dtype=np.float64)
    for i in numba.prange(N):
        if qam_L_amFreq_arr[i] != 0.0 and qam_L_amDepth_arr[i] != 0.0:
            env_qam_L_arr[i] = 1.0 + qam_L_amDepth_arr[i] * np.cos(2 * np.pi * qam_L_amFreq_arr[i] * t_arr[i] + qam_L_amPhaseOffset_arr[i])
        else:
            env_qam_L_arr[i] = 1.0
        sig_qam_L = env_qam_L_arr[i] * np.cos(ph_qam_L_carrier[i])
        out[i, 0] = np.float32(sig_qam_L * ampL_val_arr[i])

    ampR_val_arr = np.empty(N, dtype=np.float64)
    mono_R_carrierFreq_base_arr = np.empty(N, dtype=np.float64)
    mono_R_beatFreqInChannel_arr = np.empty(N, dtype=np.float64)
    mono_R_amDepth_arr = np.empty(N, dtype=np.float64)
    mono_R_amFreq_arr = np.empty(N, dtype=np.float64)
    mono_R_amPhaseOffset_arr = np.empty(N, dtype=np.float64)
    mono_R_fmRange_arr = np.empty(N, dtype=np.float64)
    mono_R_fmFreq_arr = np.empty(N, dtype=np.float64)
    mono_R_fmPhaseOffset_arr = np.empty(N, dtype=np.float64)
    mono_R_phaseOscFreq_arr = np.empty(N, dtype=np.float64)
    mono_R_phaseOscRange_arr = np.empty(N, dtype=np.float64)
    mono_R_phaseOscPhaseOffset_arr = np.empty(N, dtype=np.float64)

    for i in numba.prange(N):
        alpha = alpha_arr[i]
        ampR_val_arr[i] = s_ampR + (e_ampR - s_ampR) * alpha
        mono_R_carrierFreq_base_arr[i] = s_mono_R_carrierFreq_base + (e_mono_R_carrierFreq_base - s_mono_R_carrierFreq_base) * alpha
        mono_R_beatFreqInChannel_arr[i] = s_mono_R_beatFreqInChannel + (e_mono_R_beatFreqInChannel - s_mono_R_beatFreqInChannel) * alpha
        mono_R_amDepth_arr[i] = s_mono_R_amDepth + (e_mono_R_amDepth - s_mono_R_amDepth) * alpha
        mono_R_amFreq_arr[i] = s_mono_R_amFreq + (e_mono_R_amFreq - s_mono_R_amFreq) * alpha
        mono_R_amPhaseOffset_arr[i] = s_mono_R_amPhaseOffset + (e_mono_R_amPhaseOffset - s_mono_R_amPhaseOffset) * alpha
        mono_R_fmRange_arr[i] = s_mono_R_fmRange + (e_mono_R_fmRange - s_mono_R_fmRange) * alpha
        mono_R_fmFreq_arr[i] = s_mono_R_fmFreq + (e_mono_R_fmFreq - s_mono_R_fmFreq) * alpha
        mono_R_fmPhaseOffset_arr[i] = s_mono_R_fmPhaseOffset + (e_mono_R_fmPhaseOffset - s_mono_R_fmPhaseOffset) * alpha
        mono_R_phaseOscFreq_arr[i] = s_mono_R_phaseOscFreq + (e_mono_R_phaseOscFreq - s_mono_R_phaseOscFreq) * alpha
        mono_R_phaseOscRange_arr[i] = s_mono_R_phaseOscRange + (e_mono_R_phaseOscRange - s_mono_R_phaseOscRange) * alpha
        mono_R_phaseOscPhaseOffset_arr[i] = s_mono_R_phaseOscPhaseOffset + (e_mono_R_phaseOscPhaseOffset - s_mono_R_phaseOscPhaseOffset) * alpha
        
    mono_R_carrierFreq_inst_arr = np.empty(N, dtype=np.float64)
    for i in numba.prange(N):
        if mono_R_fmFreq_arr[i] != 0.0 and mono_R_fmRange_arr[i] != 0.0:
            fm_mod = (mono_R_fmRange_arr[i] / 2.0) * np.sin(2 * np.pi * mono_R_fmFreq_arr[i] * t_arr[i] + mono_R_fmPhaseOffset_arr[i])
            mono_R_carrierFreq_inst_arr[i] = mono_R_carrierFreq_base_arr[i] + fm_mod
        else:
            mono_R_carrierFreq_inst_arr[i] = mono_R_carrierFreq_base_arr[i]
        if mono_R_carrierFreq_inst_arr[i] < 0: mono_R_carrierFreq_inst_arr[i] = 0.0

    mono_R_freqTone1_inst_arr = np.empty(N, dtype=np.float64)
    mono_R_freqTone2_inst_arr = np.empty(N, dtype=np.float64)
    for i in numba.prange(N):
        half_beat = mono_R_beatFreqInChannel_arr[i] / 2.0
        mono_R_freqTone1_inst_arr[i] = mono_R_carrierFreq_inst_arr[i] - half_beat
        mono_R_freqTone2_inst_arr[i] = mono_R_carrierFreq_inst_arr[i] + half_beat
        if mono_R_freqTone1_inst_arr[i] < 0: mono_R_freqTone1_inst_arr[i] = 0.0
        if mono_R_freqTone2_inst_arr[i] < 0: mono_R_freqTone2_inst_arr[i] = 0.0

    ph_mono_R_tone1 = np.empty(N, dtype=np.float64)
    ph_mono_R_tone2 = np.empty(N, dtype=np.float64)
    currentPhaseMonoR1 = s_mono_R_startPhaseTone1_init
    currentPhaseMonoR2 = s_mono_R_startPhaseTone2_init
    for i in range(N): 
        ph_mono_R_tone1[i] = currentPhaseMonoR1
        ph_mono_R_tone2[i] = currentPhaseMonoR2
        currentPhaseMonoR1 += 2 * np.pi * mono_R_freqTone1_inst_arr[i] * dt
        currentPhaseMonoR2 += 2 * np.pi * mono_R_freqTone2_inst_arr[i] * dt

    if np.any(mono_R_phaseOscFreq_arr != 0.0) or np.any(mono_R_phaseOscRange_arr != 0.0): 
        for i in numba.prange(N):
            if mono_R_phaseOscFreq_arr[i] !=0.0 or mono_R_phaseOscRange_arr[i] != 0.0:
                d_phi_mono = (mono_R_phaseOscRange_arr[i] / 2.0) * np.sin(2 * np.pi * mono_R_phaseOscFreq_arr[i] * t_arr[i] + mono_R_phaseOscPhaseOffset_arr[i])
                ph_mono_R_tone1[i] -= d_phi_mono
                ph_mono_R_tone2[i] += d_phi_mono
    
    summed_mono_R_arr = np.empty(N, dtype=np.float64)
    for i in numba.prange(N):
        summed_mono_R_arr[i] = np.sin(ph_mono_R_tone1[i]) + np.sin(ph_mono_R_tone2[i])

    env_mono_R_arr = np.empty(N, dtype=np.float64)
    for i in numba.prange(N):
        if mono_R_amFreq_arr[i] != 0.0 and mono_R_amDepth_arr[i] != 0.0:
            clamped_depth = max(0.0, min(1.0, mono_R_amDepth_arr[i]))
            env_mono_R_arr[i] = 1.0 - clamped_depth * (0.5 * (1.0 + np.sin(2 * np.pi * mono_R_amFreq_arr[i] * t_arr[i] + mono_R_amPhaseOffset_arr[i])))
        else:
            env_mono_R_arr[i] = 1.0
        out[i, 1] = np.float32(summed_mono_R_arr[i] * env_mono_R_arr[i] * ampR_val_arr[i])

    return out


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
    'load_track_from_json', 'save_track_to_json', 'generate_wav', 'get_synth_params',
    'trapezoid_envelope_vectorized', '_flanger_effect_stereo_continuous',
    'butter', 'lfilter', 'write', 'ensure_stereo', 'apply_filters', 'design_filter', 
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

            if env_type == "adsr":
                env = adsr_envelope(t_rel, **cleaned_env_params)
            elif env_type == "linen":
                 env = linen_envelope(t_rel, **cleaned_env_params)
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

def generate_wav(track_data, output_filename=None):
    """Generates and saves the WAV file based on the track_data."""
    if not track_data:
        print("Error: Cannot generate WAV, track data is missing.")
        return False

    global_settings = track_data.get("global_settings", {})
    try:
        sample_rate = int(global_settings.get("sample_rate", 44100))
        crossfade_duration = float(global_settings.get("crossfade_duration", 1.0))
    except (ValueError, TypeError) as e:
         print(f"Error: Invalid global settings (sample_rate or crossfade_duration): {e}")
         return False

    output_filename = output_filename or global_settings.get("output_filename", "generated_track.wav")
    if not output_filename or not isinstance(output_filename, str):
         print(f"Error: Invalid output filename: {output_filename}")
         return False

    # Ensure output directory exists before assembly
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory for WAV: {output_dir}")
        except OSError as e:
            print(f"Error creating output directory '{output_dir}': {e}")
            return False


    print(f"\n--- Starting WAV Generation ---")
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
        write(output_filename, sample_rate, track_int16)
        print(f"--- WAV Generation Complete ---")
        print(f"Track successfully written to {output_filename}")
        return True
    except Exception as e:
        print(f"Error writing WAV file {output_filename}:")
        traceback.print_exc()
        return False

