
# sound_creator.py
import numpy as np
from scipy.signal import butter, lfilter
from scipy.io.wavfile import write
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
                 duration = sum(node.duration for node in self._nodes if hasattr(node, 'duration'))
            sample_rate = getattr(self, '_sample_rate', 44100)
            N = int(duration * sample_rate) if duration > 0 else int(1.0 * sample_rate) # Default 1 sec if no duration found
            return np.zeros((N, 2))

    VALID_SAM_PATHS = ['circle', 'line', 'lissajous', 'figure_eight', 'arc'] # Example paths

# -----------------------------------------------------------------------------
# Helper functions (Copied from original sc.txt, minor adjustments if needed)
# -----------------------------------------------------------------------------

def sine_wave(freq, t, phase=0):
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

    dt = np.diff(t, prepend=t[0])
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
    pan = float(params.get('pan', 0)) # Added pan parameter

    N = int(sample_rate * duration)
    if N <= 0: return np.zeros((0, 2))
    t_rel = np.linspace(0, duration, N, endpoint=False)
    t_abs = t_rel # For constant params, relative == absolute for phase

    env = adsr_envelope(t_rel) # Use default ADSR
    carrier = sine_wave(carrierFreq, t_abs)
    lfo = sine_wave(modFreq, t_abs)
    # Modulator maps LFO [-1, 1] to [1 - modDepth, 1]
    modulator = np.interp(lfo, [-1, 1], [1 - modDepth, 1])
    output_mono = carrier * modulator * env * amp
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
    if N <= 0: return np.zeros((0, 2))
    t_rel = np.linspace(0, duration, N, endpoint=False)
    t_abs = t_rel

    # Linearly interpolate parameters over the duration
    currentCarrierFreq = np.linspace(startCarrierFreq, endCarrierFreq, N)
    currentModFreq = np.linspace(startModFreq, endModFreq, N)
    currentModDepth = np.linspace(startModDepth, endModDepth, N)

    carrier = sine_wave_varying(currentCarrierFreq, t_abs, sample_rate)
    lfo = sine_wave_varying(currentModFreq, t_abs, sample_rate)
    # Modulator maps LFO [-1, 1] to [1 - currentModDepth, 1]
    modulator = np.interp(lfo, [-1, 1], [1 - currentModDepth, 1])
    # Use a simple linear fade in/out envelope for transitions
    env = linen_envelope(t_rel, attack=0.01, release=0.01) # Short attack/release
    output_mono = carrier * modulator * env * amp
    return pan2(output_mono, pan=pan)

def fsam_filter_bank(duration, sample_rate=44100, **params):
    """Frequency-Selective Amplitude Modulation using filter bank approach."""
    amp = float(params.get('amp', 0.15))
    noiseType = int(params.get('noiseType', 1)) # 1=white, 2=pink, 3=brown
    filterCenterFreq = float(params.get('filterCenterFreq', 1000))
    filterRQ = float(params.get('filterRQ', 0.5))
    modFreq = float(params.get('modFreq', 4))
    modDepth = float(params.get('modDepth', 0.8))
    pan = float(params.get('pan', 0))

    N = int(sample_rate * duration)
    if N <= 0: return np.zeros((0, 2))
    t_rel = np.linspace(0, duration, N, endpoint=False)
    t_abs = t_rel

    if noiseType == 1: source = np.random.randn(N)
    elif noiseType == 2: source = pink_noise(N)
    elif noiseType == 3: source = brown_noise(N)
    else: source = np.random.randn(N)

    # Ensure filter params are valid
    filterCenterFreq = max(20.0, filterCenterFreq) # Min center freq
    filterRQ = max(0.01, filterRQ) # Min Q to avoid extreme bandwidth

    bandSignal = bandpass_filter(source, filterCenterFreq, filterRQ, sample_rate)
    restSignal = bandreject_filter(source, filterCenterFreq, filterRQ, sample_rate)

    lfo = sine_wave(modFreq, t_abs)
    modulator = np.interp(lfo, [-1, 1], [1 - modDepth, 1])
    modulatedBand = bandSignal * modulator
    output_mono = modulatedBand + restSignal

    env = adsr_envelope(t_rel, attack=0.05, decay=0.2, sustain_level=0.8, release=0.5)
    output_mono = output_mono * env * amp
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
    if N <= 0: return np.zeros((0, 2))
    t_rel = np.linspace(0, duration, N, endpoint=False)
    t_abs = t_rel

    # Generate noise source
    if noiseType == 1: source = np.random.randn(N)
    elif noiseType == 2: source = pink_noise(N)
    elif noiseType == 3: source = brown_noise(N)
    else: source = np.random.randn(N)

    # Interpolate parameters
    currentFilterCenterFreq = np.linspace(startFilterCenterFreq, endFilterCenterFreq, N)
    currentFilterRQ = np.linspace(startFilterRQ, endFilterRQ, N)
    currentModFreq = np.linspace(startModFreq, endModFreq, N)
    currentModDepth = np.linspace(startModDepth, endModDepth, N)

    # Filter with start parameters (simplification)
    # A time-varying filter would be much more complex (e.g., frame-based filtering)
    startCenter = max(20.0, startFilterCenterFreq)
    startRQ = max(0.01, startFilterRQ)
    bandSignal = bandpass_filter(source, startCenter, startRQ, sample_rate)
    restSignal = bandreject_filter(source, startCenter, startRQ, sample_rate)

    lfo = sine_wave_varying(currentModFreq, t_abs, sample_rate)
    modulator = np.interp(lfo, [-1, 1], [1 - currentModDepth, 1])
    modulatedBand = bandSignal * modulator
    output_mono = modulatedBand + restSignal

    env = linen_envelope(t_rel, attack=0.01, release=0.01)
    output_mono = output_mono * env * amp
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
    if N <= 0: return np.zeros((0, 2))
    t_rel = np.linspace(0, duration, N, endpoint=False)
    t_abs = t_rel

    carrier = sine_wave(carrierFreq, t_abs)
    lfo = sine_wave(modFreq, t_abs)
    shapeLFO = np.interp(lfo, [-1, 1], [1 - modDepth, 1]) # Modulates amplitude before shaping
    modulatedInput = carrier * shapeLFO

    # Apply waveshaping (tanh)
    shapeAmount = max(1e-6, shapeAmount) # Avoid division by zero
    tanh_shape_amount = np.tanh(shapeAmount)
    shapedSignal = np.divide(np.tanh(modulatedInput * shapeAmount), tanh_shape_amount,
                              out=np.zeros_like(modulatedInput), where=tanh_shape_amount > 1e-6)

    env = adsr_envelope(t_rel)
    output_mono = shapedSignal * env * amp
    return pan2(output_mono, pan=pan)

def rhythmic_waveshaping_transition(duration, sample_rate=44100, **params):
    """Rhythmic waveshaping with parameter transitions."""
    amp = float(params.get('amp', 0.25))
    startCarrierFreq = float(params.get('startCarrierFreq', 200))
    endCarrierFreq = float(params.get('endCarrierFreq', 80))
    startModFreq = float(params.get('startModFreq', 12))
    endModFreq = float(params.get('endModFreq', 7.83))
    startModDepth = float(params.get('startModDepth', 1.0))
    endModDepth = float(params.get('endModDepth', 1.0)) # Allow transition
    startShapeAmount = float(params.get('startShapeAmount', 5.0))
    endShapeAmount = float(params.get('endShapeAmount', 5.0)) # Allow transition
    pan = float(params.get('pan', 0))

    N = int(sample_rate * duration)
    if N <= 0: return np.zeros((0, 2))
    t_rel = np.linspace(0, duration, N, endpoint=False)
    t_abs = t_rel

    # Interpolate parameters
    currentCarrierFreq = np.linspace(startCarrierFreq, endCarrierFreq, N)
    currentModFreq = np.linspace(startModFreq, endModFreq, N)
    currentModDepth = np.linspace(startModDepth, endModDepth, N)
    currentShapeAmount = np.linspace(startShapeAmount, endShapeAmount, N)

    carrier = sine_wave_varying(currentCarrierFreq, t_abs, sample_rate)
    lfo = sine_wave_varying(currentModFreq, t_abs, sample_rate)
    shapeLFO = np.interp(lfo, [-1, 1], [1 - currentModDepth, 1])
    modulatedInput = carrier * shapeLFO

    # Apply time-varying waveshaping
    currentShapeAmount = np.maximum(1e-6, currentShapeAmount) # Avoid zero
    tanh_shape_amount = np.tanh(currentShapeAmount)
    shapedSignal = np.divide(np.tanh(modulatedInput * currentShapeAmount), tanh_shape_amount,
                              out=np.zeros_like(modulatedInput), where=tanh_shape_amount > 1e-6)

    env = linen_envelope(t_rel, attack=0.01, release=0.01)
    output_mono = shapedSignal * env * amp
    return pan2(output_mono, pan=pan)

def rhythmic_granular_rate(duration, sample_rate=44100, **params):
    """Simplified rhythmic granular synthesis (placeholder)."""
    print("WARNING: rhythmic_granular_rate not fully implemented for GUI yet.")
    N = int(sample_rate * duration)
    amp = float(params.get('amp', 0.5))
    pan = float(params.get('pan', 0))
    output_mono = np.zeros(N)
    return pan2(output_mono * amp, pan=pan) # Apply amp and pan to silence

def additive_phase_mod(duration, sample_rate=44100, **params):
    """Additive synthesis with phase modulation on a harmonic."""
    amp = float(params.get('amp', 0.15))
    fundFreq = float(params.get('fundFreq', 200))
    modFreq = float(params.get('modFreq', 4))
    modDepth = float(params.get('modDepth', 1.0)) # Scaled phase offset factor
    h2Amp = float(params.get('h2Amp', 0.5))
    pan = float(params.get('pan', 0))

    N = int(sample_rate * duration)
    if N <= 0: return np.zeros((0, 2))
    t_rel = np.linspace(0, duration, N, endpoint=False)
    t_abs = t_rel

    fundamental = sine_wave(fundFreq, t_abs)
    lfo = sine_wave(modFreq, t_abs)
    phaseModOffset = lfo * modDepth * np.pi

    # Generate 2nd harmonic with phase modulation
    harmonic2_freq = fundFreq * 2
    harmonic2_phase = 2 * np.pi * harmonic2_freq * t_abs + phaseModOffset
    harmonic2 = h2Amp * np.sin(harmonic2_phase)

    env = adsr_envelope(t_rel)
    output_mono = (fundamental + harmonic2) * env * amp
    return pan2(output_mono, pan=pan)

def additive_phase_mod_transition(duration, sample_rate=44100, **params):
    """Additive phase modulation with parameter transitions."""
    amp = float(params.get('amp', 0.15))
    startFundFreq = float(params.get('startFundFreq', 200))
    endFundFreq = float(params.get('endFundFreq', 150))
    startModFreq = float(params.get('startModFreq', 4))
    endModFreq = float(params.get('endModFreq', 10))
    startModDepth = float(params.get('startModDepth', 1.0)) # Scaled by Pi later
    endModDepth = float(params.get('endModDepth', 3.0))
    starth2Amp = float(params.get('starth2Amp', 0.5))
    endh2Amp = float(params.get('endh2Amp', 0.5)) # Allow transition
    pan = float(params.get('pan', 0))

    N = int(sample_rate * duration)
    if N <= 0: return np.zeros((0, 2))
    t_rel = np.linspace(0, duration, N, endpoint=False)
    t_abs = t_rel

    # Interpolate parameters
    currentFundFreq = np.linspace(startFundFreq, endFundFreq, N)
    currentModFreq = np.linspace(startModFreq, endModFreq, N)
    currentModDepth = np.linspace(startModDepth, endModDepth, N)
    currenth2Amp = np.linspace(starth2Amp, endh2Amp, N)

    # Generate fundamental with varying frequency
    fundamental = sine_wave_varying(currentFundFreq, t_abs, sample_rate)

    # Generate LFO for phase mod
    lfo = sine_wave_varying(currentModFreq, t_abs, sample_rate)
    phaseModOffset = lfo * currentModDepth * np.pi

    # Generate 2nd harmonic with varying freq and phase mod
    harmonic2_freq = currentFundFreq * 2
    if N > 1:
        dt = np.diff(t_abs, prepend=t_abs[0])
        harmonic2_inst_phase_change = 2 * np.pi * harmonic2_freq * dt
        initial_phase = 2 * np.pi * harmonic2_freq[0] * t_abs[0] if N > 0 else 0
        harmonic2_cumulative_phase = initial_phase + np.cumsum(harmonic2_inst_phase_change)
    elif N == 1:
         harmonic2_cumulative_phase = 2 * np.pi * harmonic2_freq[0] * t_abs[0]
    else: # N == 0
         harmonic2_cumulative_phase = np.array([])

    harmonic2 = currenth2Amp * np.sin(harmonic2_cumulative_phase + phaseModOffset)

    env = linen_envelope(t_rel, attack=0.01, release=0.01)
    output_mono = (fundamental + harmonic2) * env * amp
    return pan2(output_mono, pan=pan)

def stereo_am_independent(duration, sample_rate=44100, **params):
    """Stereo Amplitude Modulation with independent L/R modulators."""
    amp = float(params.get('amp', 0.25))
    carrierFreq = float(params.get('carrierFreq', 200.0))
    modFreqL = float(params.get('modFreqL', 4.0))
    modDepthL = float(params.get('modDepthL', 0.8))
    modPhaseL = float(params.get('modPhaseL', 0)) # Phase in radians
    modFreqR = float(params.get('modFreqR', 4.0))
    modDepthR = float(params.get('modDepthR', 0.8))
    modPhaseR = float(params.get('modPhaseR', 0)) # Phase in radians
    stereo_width_hz = float(params.get('stereo_width_hz', 0.2)) # Freq difference

    N = int(sample_rate * duration)
    if N <= 0: return np.zeros((0, 2))
    t_rel = np.linspace(0, duration, N, endpoint=False)
    t_abs = t_rel

    # Carriers with slight detuning
    carrierL = sine_wave(carrierFreq - stereo_width_hz / 2, t_abs)
    carrierR = sine_wave(carrierFreq + stereo_width_hz / 2, t_abs)

    # Independent LFOs
    lfoL = sine_wave(modFreqL, t_abs, phase=modPhaseL)
    lfoR = sine_wave(modFreqR, t_abs, phase=modPhaseR)

    # Modulators
    modulatorL = np.interp(lfoL, [-1, 1], [1 - modDepthL, 1])
    modulatorR = np.interp(lfoR, [-1, 1], [1 - modDepthR, 1])

    env = adsr_envelope(t_rel) # Apply envelope equally

    outputL = carrierL * modulatorL * env * amp
    outputR = carrierR * modulatorR * env * amp

    return np.vstack([outputL, outputR]).T

def stereo_am_independent_transition(duration, sample_rate=44100, **params):
    """Stereo AM Independent with parameter transitions."""
    amp = float(params.get('amp', 0.25))
    startCarrierFreq = float(params.get('startCarrierFreq', 200))
    endCarrierFreq = float(params.get('endCarrierFreq', 250))
    startModFreqL = float(params.get('startModFreqL', 4))
    endModFreqL = float(params.get('endModFreqL', 6))
    startModDepthL = float(params.get('startModDepthL', 0.8))
    endModDepthL = float(params.get('endModDepthL', 0.8)) # Allow transition
    startModPhaseL = float(params.get('startModPhaseL', 0)) # Constant phase
    startModFreqR = float(params.get('startModFreqR', 4.1))
    endModFreqR = float(params.get('endModFreqR', 5.9))
    startModDepthR = float(params.get('startModDepthR', 0.8))
    endModDepthR = float(params.get('endModDepthR', 0.8)) # Allow transition
    startModPhaseR = float(params.get('startModPhaseR', 0)) # Constant phase
    startStereoWidthHz = float(params.get('startStereoWidthHz', 0.2))
    endStereoWidthHz = float(params.get('endStereoWidthHz', 0.2)) # Allow transition

    N = int(sample_rate * duration)
    if N <= 0: return np.zeros((0, 2))
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

    # Varying frequency LFOs (use constant start phase)
    lfoL = sine_wave_varying(currentModFreqL, t_abs, sample_rate) # Phase evolves based on freq integral
    lfoR = sine_wave_varying(currentModFreqR, t_abs, sample_rate) # Phase evolves based on freq integral

    # Modulators with varying depth
    modulatorL = np.interp(lfoL, [-1, 1], [1 - currentModDepthL, 1])
    modulatorR = np.interp(lfoR, [-1, 1], [1 - currentModDepthR, 1])

    env = linen_envelope(t_rel, attack=0.01, release=0.01) # Transition envelope

    outputL = carrierL * modulatorL * env * amp
    outputR = carrierR * modulatorR * env * amp

    return np.vstack([outputL, outputR]).T

def noise_am(duration, sample_rate=44100, **params):
    """Amplitude modulation using filtered noise as the modulator."""
    amp = float(params.get('amp', 0.25))
    carrierFreq = float(params.get('carrierFreq', 200))
    noiseType = int(params.get('noiseType', 1)) # 1=white, 2=pink, 3=brown
    modFilterFreq = float(params.get('modFilterFreq', 2)) # Cutoff
    modDepth = float(params.get('modDepth', 0.8))
    pan = float(params.get('pan', 0))

    N = int(sample_rate * duration)
    if N <= 0: return np.zeros((0, 2))
    t_rel = np.linspace(0, duration, N, endpoint=False)
    t_abs = t_rel

    carrier = sine_wave(carrierFreq, t_abs)

    # Generate noise modulator source
    if noiseType == 1: noise_mod_src = np.random.randn(N)
    elif noiseType == 2: noise_mod_src = pink_noise(N)
    elif noiseType == 3: noise_mod_src = brown_noise(N)
    else: noise_mod_src = np.random.randn(N)

    # Filter the noise
    modFilterFreq = max(0.1, modFilterFreq) # Ensure positive cutoff
    filteredMod = lowpass_filter(noise_mod_src, modFilterFreq, sample_rate)

    # Scale filtered noise
    min_fm, max_fm = np.min(filteredMod), np.max(filteredMod)
    if max_fm - min_fm > 1e-6: # Avoid division by zero
        scaledMod = np.interp(filteredMod, [min_fm, max_fm], [1 - modDepth, 1])
    else:
        scaledMod = np.ones(N) # Constant modulator if noise is flat

    env = adsr_envelope(t_rel)
    output_mono = carrier * scaledMod * env * amp
    return pan2(output_mono, pan=pan)

def noise_am_transition(duration, sample_rate=44100, **params):
    """Noise AM with parameter transitions."""
    amp = float(params.get('amp', 0.25))
    startCarrierFreq = float(params.get('startCarrierFreq', 200))
    endCarrierFreq = float(params.get('endCarrierFreq', 300))
    noiseType = int(params.get('noiseType', 1))
    startModFilterFreq = float(params.get('startModFilterFreq', 2))
    endModFilterFreq = float(params.get('endModFilterFreq', 0.5))
    startModDepth = float(params.get('startModDepth', 0.8))
    endModDepth = float(params.get('endModDepth', 0.3))
    pan = float(params.get('pan', 0))

    N = int(sample_rate * duration)
    if N <= 0: return np.zeros((0, 2))
    t_rel = np.linspace(0, duration, N, endpoint=False)
    t_abs = t_rel

    # Interpolate parameters
    currentCarrierFreq = np.linspace(startCarrierFreq, endCarrierFreq, N)
    currentModFilterFreq = np.linspace(startModFilterFreq, endModFilterFreq, N)
    currentModDepth = np.linspace(startModDepth, endModDepth, N)

    # Generate carrier
    carrier = sine_wave_varying(currentCarrierFreq, t_abs, sample_rate)

    # Generate noise source
    if noiseType == 1: noise_mod_src = np.random.randn(N)
    elif noiseType == 2: noise_mod_src = pink_noise(N)
    elif noiseType == 3: noise_mod_src = brown_noise(N)
    else: noise_mod_src = np.random.randn(N)

    # Filter with start frequency (simplification)
    startFiltFreq = max(0.1, startModFilterFreq)
    filteredMod = lowpass_filter(noise_mod_src, startFiltFreq, sample_rate)

    # Scale modulator with varying depth
    min_fm, max_fm = np.min(filteredMod), np.max(filteredMod)
    if max_fm - min_fm > 1e-6:
        scaledMod = np.interp(filteredMod, [min_fm, max_fm], [1 - currentModDepth, 1])
    else:
        scaledMod = np.ones(N)

    env = linen_envelope(t_rel, attack=0.01, release=0.01)
    output_mono = carrier * scaledMod * env * amp
    return pan2(output_mono, pan=pan)

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
    stereoModPhaseR = float(params.get('stereoModPhaseR', math.pi * 2))

    N = int(sample_rate * duration)
    if N <= 0: return np.zeros((0, 2))
    t_rel = np.linspace(0, duration, N, endpoint=False)
    t_abs = t_rel

    # Rhythmic waveshaping part (mono)
    carrier = sine_wave(carrierFreq, t_abs)
    shapeLFO_wave = sine_wave(shapeModFreq, t_abs)
    shapeLFO_amp = np.interp(shapeLFO_wave, [-1, 1], [1 - shapeModDepth, 1])
    modulatedInput = carrier * shapeLFO_amp
    shapeAmount = max(1e-6, shapeAmount)
    tanh_shape_amount = np.tanh(shapeAmount)
    shapedSignal = np.divide(np.tanh(modulatedInput * shapeAmount), tanh_shape_amount,
                              out=np.zeros_like(modulatedInput), where=tanh_shape_amount > 1e-6)

    # Stereo AM part
    stereoLFO_L = sine_wave(stereoModFreqL, t_abs, phase=stereoModPhaseL)
    stereoLFO_R = sine_wave(stereoModFreqR, t_abs, phase=stereoModPhaseR)
    modulatorL = np.interp(stereoLFO_L, [-1, 1], [1 - stereoModDepthL, 1])
    modulatorR = np.interp(stereoLFO_R, [-1, 1], [1 - stereoModDepthR, 1])

    # Apply stereo modulators
    outputL = shapedSignal * modulatorL
    outputR = shapedSignal * modulatorR

    # Apply overall envelope and amplitude
    env = adsr_envelope(t_rel, attack=0.05, decay=0.2, sustain_level=0.8, release=0.5)
    outputL = outputL * env * amp
    outputR = outputR * env * amp

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
    startStereoModPhaseL = float(params.get('startStereoModPhaseL', 0)) # Constant
    startStereoModFreqR = float(params.get('startStereoModFreqR', 4.0))
    endStereoModFreqR = float(params.get('endStereoModFreqR', 6.1))
    startStereoModDepthR = float(params.get('startStereoModDepthR', 0.9))
    endStereoModDepthR = float(params.get('endStereoModDepthR', 0.9))
    startStereoModPhaseR = float(params.get('startStereoModPhaseR', math.pi * 2)) # Constant

    N = int(sample_rate * duration)
    if N <= 0: return np.zeros((0, 2))
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
    shapeLFO_amp = np.interp(shapeLFO_wave, [-1, 1], [1 - currentShapeModDepth, 1])
    modulatedInput = carrier * shapeLFO_amp
    currentShapeAmount = np.maximum(1e-6, currentShapeAmount)
    tanh_shape_amount = np.tanh(currentShapeAmount)
    shapedSignal = np.divide(np.tanh(modulatedInput * currentShapeAmount), tanh_shape_amount,
                              out=np.zeros_like(modulatedInput), where=tanh_shape_amount > 1e-6)

    # Stereo AM part
    stereoLFO_L = sine_wave_varying(currentStereoModFreqL, t_abs, sample_rate)
    stereoLFO_R = sine_wave_varying(currentStereoModFreqR, t_abs, sample_rate)
    modulatorL = np.interp(stereoLFO_L, [-1, 1], [1 - currentStereoModDepthL, 1])
    modulatorR = np.interp(stereoLFO_R, [-1, 1], [1 - currentStereoModDepthR, 1])

    # Apply stereo modulators
    outputL = shapedSignal * modulatorL
    outputR = shapedSignal * modulatorR

    # Apply overall envelope
    env = linen_envelope(t_rel, attack=0.01, release=0.01)
    outputL = outputL * env * amp
    outputR = outputR * env * amp

    return np.vstack([outputL, outputR]).T

def spatial_angle_modulation(duration, sample_rate=44100, **params):
    """Spatial Angle Modulation using external audio_engine module."""
    if not AUDIO_ENGINE_AVAILABLE:
        print("Error: SAM function called, but audio_engine module is missing.")
        N = int(sample_rate * duration)
        return np.zeros((N, 2))

    # Extract parameters for SAMVoice
    amp = float(params.get('amp', 0.7))
    carrierFreq = float(params.get('carrierFreq', 440.0))
    beatFreq = float(params.get('beatFreq', 4.0))
    pathShape = str(params.get('pathShape', 'circle'))
    pathRadius = float(params.get('pathRadius', 1.0))
    arcStartDeg = float(params.get('arcStartDeg', 0.0))
    arcEndDeg = float(params.get('arcEndDeg', 360.0))
    frame_dur_ms = float(params.get('frame_dur_ms', 46.4))
    overlap_factor = int(params.get('overlap_factor', 8))

    # Validate pathShape
    if pathShape not in VALID_SAM_PATHS:
        print(f"Warning: Invalid pathShape '{pathShape}'. Defaulting to 'circle'. Valid: {VALID_SAM_PATHS}")
        pathShape = 'circle'

    # Build a single-node timeline for SAMVoice
    try:
        node = Node(duration, carrierFreq, beatFreq, 1.0, 1.0)
    except Exception as e:
        print(f"Error creating Node for SAM: {e}")
        N = int(sample_rate * duration)
        return np.zeros((N, 2))

    sam_params_dict = {
        'path_shape':     pathShape,
        'path_radius':    pathRadius,
        'arc_start_deg':  arcStartDeg,
        'arc_end_deg':    arcEndDeg
    }

    try:
        voice = SAMVoice(
            nodes=[node],
            sample_rate=sample_rate,
            frame_dur_ms=frame_dur_ms,
            overlap_factor=overlap_factor,
            source_amp=amp,
            sam_node_params=[sam_params_dict] # Expects a list of dicts
        )
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

    # Extract start and end parameters
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

    # Validate path shapes
    if startPathShape not in VALID_SAM_PATHS:
        print(f"Warning: Invalid startPathShape '{startPathShape}'. Defaulting to 'circle'.")
        startPathShape = 'circle'
    if endPathShape not in VALID_SAM_PATHS:
        print(f"Warning: Invalid endPathShape '{endPathShape}'. Defaulting to 'circle'.")
        endPathShape = 'circle'

    # Create two nodes for interpolation
    try:
        node_start = Node(duration, startCarrierFreq, startBeatFreq, 1.0, 1.0)
        node_end = Node(0.0, endCarrierFreq, endBeatFreq, 1.0, 1.0) # Zero duration holds end state
    except Exception as e:
        print(f"Error creating Nodes for SAM transition: {e}")
        N = int(sample_rate * duration)
        return np.zeros((N, 2))

    sam_params_list = [
        {'path_shape': startPathShape, 'path_radius': startPathRadius, 'arc_start_deg': startArcStartDeg, 'arc_end_deg': startArcEndDeg},
        {'path_shape': endPathShape, 'path_radius': endPathRadius, 'arc_start_deg': endArcStartDeg, 'arc_end_deg': endArcEndDeg}
    ]

    try:
        voice = SAMVoice(
            nodes=[node_start, node_end],
            sample_rate=sample_rate,
            frame_dur_ms=frame_dur_ms,
            overlap_factor=overlap_factor,
            source_amp=amp,
            sam_node_params=sam_params_list
        )
        return voice.generate_samples()
    except Exception as e:
        print(f"Error during SAMVoice transition generation: {e}")
        traceback.print_exc()
        N = int(sample_rate * duration)
        return np.zeros((N, 2))

# --- NEW: Binaural Beat Functions ---

def binaural_beat(duration, sample_rate=44100, **params):
    """
    Classic binaural beat generator. Produces a slightly different frequency
    in each ear to create a perceived beat frequency.
    Uses cumulative sum for phase calculation to handle varying frequencies.
    """
    amp = float(params.get('amp', 0.5))
    baseFreq = float(params.get('baseFreq', 200.0))
    beatFreq = float(params.get('beatFreq', 4.0))
    # Pan is ignored for binaural beats as L/R channels are inherently separate.

    N = int(sample_rate * duration)
    if N <= 0: return np.zeros((0, 2))
    t_abs = np.linspace(0, duration, N, endpoint=False) # Used for phase calculation

    # Calculate instantaneous frequencies for left and right channels
    half_beat_freq = beatFreq / 2.0
    # Ensure frequencies don't go below zero
    left_freq = np.maximum(0.0, baseFreq - half_beat_freq)
    right_freq = np.maximum(0.0, baseFreq + half_beat_freq)

    # --- Phase Calculation using Cumsum ---
    # Calculate phase by integrating frequencies (cumulative sum)
    # delta_phase = 2 * pi * freq / sample_rate
    # phase = cumsum(delta_phase)
    # Need to create arrays for frequency if they are constant
    left_freq_array = np.full(N, left_freq)
    right_freq_array = np.full(N, right_freq)

    # Calculate dt (time difference between samples)
    if N > 1:
        dt = np.diff(t_abs, prepend=t_abs[0])
    elif N == 1:
        dt = np.array([duration]) # Single sample duration
    else: # N == 0
        dt = np.array([])

    phase_left = np.cumsum(2 * np.pi * left_freq_array * dt)
    phase_right = np.cumsum(2 * np.pi * right_freq_array * dt)
    # --- End Phase Calculation ---

    # Generate sine waves using the calculated phases
    s_left = np.sin(phase_left) * amp
    s_right = np.sin(phase_right) * amp

    # Combine into stereo output
    audio = np.column_stack((s_left, s_right)).astype(np.float32)

    # Apply a short fade-in/out to avoid clicks
    env = linen_envelope(t_abs, attack=0.01, release=0.01)
    audio = audio * env[:, np.newaxis]

    return audio

def binaural_beat_transition(duration, sample_rate=44100, **params):
    """
    Binaural beat generator with transitions for base and beat frequencies.
    """
    amp = float(params.get('amp', 0.5))
    startBaseFreq = float(params.get('startBaseFreq', 200.0))
    endBaseFreq = float(params.get('endBaseFreq', 180.0))
    startBeatFreq = float(params.get('startBeatFreq', 4.0))
    endBeatFreq = float(params.get('endBeatFreq', 8.0))
    # Pan is ignored.

    N = int(sample_rate * duration)
    if N <= 0: return np.zeros((0, 2))
    t_abs = np.linspace(0, duration, N, endpoint=False)

    # Linearly interpolate parameters
    base_freq_array = np.linspace(startBaseFreq, endBaseFreq, N)
    beat_freq_array = np.linspace(startBeatFreq, endBeatFreq, N)

    # Calculate instantaneous frequencies for left and right channels
    half_beat_freq_array = beat_freq_array / 2.0
    # Ensure frequencies don't go below zero
    left_freq_array = np.maximum(0.0, base_freq_array - half_beat_freq_array)
    right_freq_array = np.maximum(0.0, base_freq_array + half_beat_freq_array)

    # --- Phase Calculation using Cumsum ---
    # Calculate dt (time difference between samples)
    if N > 1:
        dt = np.diff(t_abs, prepend=t_abs[0])
    elif N == 1:
        dt = np.array([duration])
    else: # N == 0
        dt = np.array([])

    phase_left = np.cumsum(2 * np.pi * left_freq_array * dt)
    phase_right = np.cumsum(2 * np.pi * right_freq_array * dt)
    # --- End Phase Calculation ---

    # Generate sine waves using the calculated phases
    s_left = np.sin(phase_left) * amp
    s_right = np.sin(phase_right) * amp

    # Combine into stereo output
    audio = np.column_stack((s_left, s_right)).astype(np.float32)

    # Apply a short fade-in/out to avoid clicks
    env = linen_envelope(t_abs, attack=0.01, release=0.01)
    audio = audio * env[:, np.newaxis]

    return audio


# --- NEW: Isochronic Tone Functions ---

def isochronic_tone(duration, sample_rate=44100, **params):
    """
    Isochronic tone generator with trapezoidal envelope.
    Uses cumulative sum for phase calculation.
    """
    amp = float(params.get('amp', 0.5))
    baseFreq = float(params.get('baseFreq', 200.0))
    beatFreq = float(params.get('beatFreq', 4.0))
    rampPercent = float(params.get('rampPercent', 0.2))
    gapPercent = float(params.get('gapPercent', 0.15))
    pan = float(params.get('pan', 0.0))

    N = int(sample_rate * duration)
    if N <= 0: return np.zeros((0, 2))
    t_abs = np.linspace(0, duration, N, endpoint=False)

    # --- Carrier Phase Calculation ---
    instantaneous_carrier_freq = np.maximum(0.0, baseFreq)
    carrier_freq_array = np.full(N, instantaneous_carrier_freq)

    if N > 1: dt = np.diff(t_abs, prepend=t_abs[0])
    elif N==1: dt = np.array([duration])
    else: dt = np.array([])

    carrier_phase = np.cumsum(2 * np.pi * carrier_freq_array * dt)
    carrier = np.sin(carrier_phase)
    # --- End Carrier Calculation ---

    # --- Envelope Timing Calculation ---
    instantaneous_beat_freq = np.maximum(0.0, beatFreq)
    beat_freq_array = np.full(N, instantaneous_beat_freq)

    cycle_len_array = np.zeros_like(beat_freq_array)
    valid_beat_mask = beat_freq_array > 1e-9 # Use small threshold for valid frequency
    with np.errstate(divide='ignore'):
        cycle_len_array[valid_beat_mask] = 1.0 / beat_freq_array[valid_beat_mask]

    # Calculate cumulative phase in *cycles* (not radians) for modulo operation
    beat_phase_cycles = np.cumsum(beat_freq_array * dt)
    # Time within the current cycle (0 to cycle_len)
    t_in_cycle = np.mod(beat_phase_cycles, 1.0) * cycle_len_array
    # Ensure t_in_cycle is 0 where beat freq is invalid
    t_in_cycle[~valid_beat_mask] = 0.0
    # --- End Envelope Timing Calculation ---

    # Generate envelope using the trapezoid helper function
    env = trapezoid_envelope_vectorized(
        t_in_cycle, cycle_len_array, rampPercent, gapPercent
    )

    # Apply envelope and amplitude
    mono_signal = carrier * env

    # Apply overall amp and pan
    output_mono = mono_signal * amp
    audio = pan2(output_mono, pan=pan)

    # Apply a short fade-in/out to avoid clicks (optional, but good practice)
    fade_env = linen_envelope(t_abs, attack=0.01, release=0.01)
    audio = audio * fade_env[:, np.newaxis]

    return audio.astype(np.float32)

def isochronic_tone_transition(duration, sample_rate=44100, **params):
    """
    Isochronic tone generator with transitions for base and beat frequencies.
    """
    amp = float(params.get('amp', 0.5))
    startBaseFreq = float(params.get('startBaseFreq', 200.0))
    endBaseFreq = float(params.get('endBaseFreq', 180.0))
    startBeatFreq = float(params.get('startBeatFreq', 4.0))
    endBeatFreq = float(params.get('endBeatFreq', 8.0))
    rampPercent = float(params.get('rampPercent', 0.2)) # Constant for simplicity
    gapPercent = float(params.get('gapPercent', 0.15)) # Constant for simplicity
    pan = float(params.get('pan', 0.0))

    N = int(sample_rate * duration)
    if N <= 0: return np.zeros((0, 2))
    t_abs = np.linspace(0, duration, N, endpoint=False)

    # Interpolate frequencies
    base_freq_array = np.linspace(startBaseFreq, endBaseFreq, N)
    beat_freq_array = np.linspace(startBeatFreq, endBeatFreq, N)

    # --- Carrier Phase Calculation ---
    instantaneous_carrier_freq_array = np.maximum(0.0, base_freq_array)

    if N > 1: dt = np.diff(t_abs, prepend=t_abs[0])
    elif N==1: dt = np.array([duration])
    else: dt = np.array([])

    carrier_phase = np.cumsum(2 * np.pi * instantaneous_carrier_freq_array * dt)
    carrier = np.sin(carrier_phase)
    # --- End Carrier Calculation ---

    # --- Envelope Timing Calculation ---
    instantaneous_beat_freq_array = np.maximum(0.0, beat_freq_array)

    cycle_len_array = np.zeros_like(instantaneous_beat_freq_array)
    valid_beat_mask = instantaneous_beat_freq_array > 1e-9
    with np.errstate(divide='ignore'):
        cycle_len_array[valid_beat_mask] = 1.0 / instantaneous_beat_freq_array[valid_beat_mask]

    beat_phase_cycles = np.cumsum(instantaneous_beat_freq_array * dt)
    t_in_cycle = np.mod(beat_phase_cycles, 1.0) * cycle_len_array
    t_in_cycle[~valid_beat_mask] = 0.0
    # --- End Envelope Timing Calculation ---

    # Generate envelope using the trapezoid helper function
    env = trapezoid_envelope_vectorized(
        t_in_cycle, cycle_len_array, rampPercent, gapPercent
    )

    # Apply envelope and amplitude
    mono_signal = carrier * env

    # Apply overall amp and pan
    output_mono = mono_signal * amp
    audio = pan2(output_mono, pan=pan)

    # Apply a short fade-in/out to avoid clicks
    fade_env = linen_envelope(t_abs, attack=0.01, release=0.01)
    audio = audio * fade_env[:, np.newaxis]

    return audio.astype(np.float32)

# -----------------------------------------------------------------------------
# Safety Limiter
# -----------------------------------------------------------------------------
def safety_limiter(signal, threshold=0.9):
    """Clips the signal to prevent exceeding the threshold."""
    threshold = float(threshold)
    return np.clip(signal, -threshold, threshold)

# -----------------------------------------------------------------------------
# Crossfade and Assembly Logic (Adapted for JSON structure)
# -----------------------------------------------------------------------------

def crossfade_signals(signal_a, signal_b, sample_rate, transition_duration):
    """
    Crossfades two stereo signals of the same length (transition_duration).
    Assumes signal_a fades out, signal_b fades in.
    """
    n_samples = int(transition_duration * sample_rate)
    # Check if signals have enough samples for the crossfade
    len_a = signal_a.shape[0]
    len_b = signal_b.shape[0]

    # Use minimum available length if signals are too short
    actual_crossfade_samples = min(n_samples, len_a, len_b)

    if actual_crossfade_samples <= 0:
        print(f"Warning: Crossfade not possible. Samples: {n_samples}, SigA: {len_a}, SigB: {len_b}")
        # If no overlap or length mismatch, just return the second signal's segment intended for overlap
        return signal_b[:min(len_b, n_samples)] if n_samples > 0 else np.zeros((0,2))

    # Ensure signals are 2D (N, 2)
    if signal_a.ndim == 1: signal_a = np.vstack([signal_a, signal_a]).T
    if signal_b.ndim == 1: signal_b = np.vstack([signal_b, signal_b]).T
    if signal_a.shape[1] == 1: signal_a = np.hstack([signal_a, signal_a])
    if signal_b.shape[1] == 1: signal_b = np.hstack([signal_b, signal_b])

    # Take only the required number of samples for crossfade
    signal_a_seg = signal_a[:actual_crossfade_samples]
    signal_b_seg = signal_b[:actual_crossfade_samples]

    # Linear crossfade (equal power can also be used: sqrt(fade))
    fade_out = np.linspace(1, 0, actual_crossfade_samples)[:, None] # Column vector
    fade_in = np.linspace(0, 1, actual_crossfade_samples)[:, None]

    # Apply fades and sum
    return signal_a_seg * fade_out + signal_b_seg * fade_in


# Dictionary mapping function names (strings) to actual functions
# --- UPDATED SYNTH_FUNCTIONS DICTIONARY ---
SYNTH_FUNCTIONS = {name: obj for name, obj in inspect.getmembers(__import__(__name__)) if inspect.isfunction(obj) and name not in [
    # Exclude helper/internal functions explicitly
    'validate_float', 'validate_int', 'butter_bandpass', 'bandpass_filter',
    'butter_bandstop', 'bandreject_filter', 'lowpass_filter', 'pink_noise',
    'brown_noise', 'sine_wave', 'sine_wave_varying', 'adsr_envelope',
    'create_linear_fade_envelope', 'linen_envelope', 'pan2', 'safety_limiter',
    'crossfade_signals', 'assemble_track_from_data', 'generate_voice_audio',
    'load_track_from_json', 'save_track_to_json', 'generate_wav', 'get_synth_params',
    'trapezoid_envelope_vectorized' # Exclude the new helper
]}
print(f"Detected Synth Functions: {list(SYNTH_FUNCTIONS.keys())}")


# Function to get parameters for a given synth function name
def get_synth_params(func_name):
    """Gets parameter names and default values for a synth function."""
    if func_name not in SYNTH_FUNCTIONS:
        return {}
    func = SYNTH_FUNCTIONS[func_name]
    sig = inspect.signature(func)
    params = {}
    for name, param in sig.parameters.items():
        # Skip duration, sample_rate, and **params catch-all
        if name in ['duration', 'sample_rate', 'params', 'kwargs']:
            continue
        params[name] = param.default if param.default != inspect.Parameter.empty else None
    return params


def generate_voice_audio(voice_data, duration, sample_rate, global_start_time):
    """Generates audio for a single voice based on its definition."""
    func_name = voice_data.get("synth_function_name")
    params = voice_data.get("params", {})
    is_transition = voice_data.get("is_transition", False)

    # --- Select the correct function (static or transition) ---
    actual_func_name = func_name
    if is_transition:
        if not func_name.endswith("_transition"):
            transition_func_name = func_name + "_transition"
            if transition_func_name in SYNTH_FUNCTIONS:
                actual_func_name = transition_func_name
            else:
                print(f"Warning: Transition checked, but '{transition_func_name}' not found for base '{func_name}'. Using static version '{func_name}'.")
                actual_func_name = func_name
    else:
        if func_name.endswith("_transition"):
            base_func_name = func_name.replace("_transition", "")
            if base_func_name in SYNTH_FUNCTIONS:
                 actual_func_name = base_func_name
                 print(f"Note: Transition unchecked for '{func_name}'. Using base function '{base_func_name}'.")
            else:
                 print(f"Warning: Transition unchecked for '{func_name}', base '{base_func_name}' not found. Using '{func_name}' but parameters might be misinterpreted.")
                 actual_func_name = func_name # Stick with selected func

    if actual_func_name not in SYNTH_FUNCTIONS:
        print(f"Error: Synth function '{actual_func_name}' not found.")
        N = int(duration * sample_rate)
        return np.zeros((N, 2))

    synth_func = SYNTH_FUNCTIONS[actual_func_name]

    # Clean params: remove None values before passing to function
    # This allows function defaults to take effect if a param wasn't set in JSON/GUI
    cleaned_params = {k: v for k, v in params.items() if v is not None}

    # --- Generate base audio ---
    try:
        # Pass duration and sample_rate explicitly, others via cleaned_params
        audio = synth_func(duration=duration, sample_rate=sample_rate, **cleaned_params)
    except Exception as e:
        print(f"Error calling synth function '{actual_func_name}' with params {cleaned_params}:")
        traceback.print_exc()
        N = int(duration * sample_rate)
        return np.zeros((N, 2))

    if audio is None: # Check if synth function failed silently
         print(f"Error: Synth function '{actual_func_name}' returned None.")
         N = int(duration * sample_rate)
         return np.zeros((N, 2))

    # --- Apply volume envelope if defined ---
    envelope_data = voice_data.get("volume_envelope")
    if envelope_data and isinstance(envelope_data, dict):
        env_type = envelope_data.get("type")
        env_params = envelope_data.get("params", {})
        cleaned_env_params = {k: v for k, v in env_params.items() if v is not None}

        N_audio = audio.shape[0]
        t_rel = np.linspace(0, duration, N_audio, endpoint=False) # Use audio length for time vector
        env = np.ones(N_audio) # Default envelope

        try:
            if env_type == "adsr":
                env = adsr_envelope(t_rel, **cleaned_env_params)
            elif env_type == "linen":
                 env = linen_envelope(t_rel, **cleaned_env_params)
            elif env_type == "linear_fade":
                 required = ['fade_duration', 'start_amp', 'end_amp', 'fade_type']
                 if all(p in cleaned_env_params for p in required):
                     # Pass actual duration for envelope calculation, not just t_rel length
                     env = create_linear_fade_envelope(duration, sample_rate, **cleaned_env_params)
                     # Resample envelope if its length doesn't match audio (e.g., due to rounding)
                     if len(env) != N_audio:
                          print(f"Warning: Resampling '{env_type}' envelope from {len(env)} to {N_audio} samples.")
                          env = np.interp(np.linspace(0, 1, N_audio), np.linspace(0, 1, len(env)), env)
                 else:
                     print(f"Warning: Missing parameters for 'linear_fade' envelope. Using flat envelope. Got: {cleaned_env_params}")
            else:
                print(f"Warning: Unknown envelope type '{env_type}'. Using flat envelope.")

            # Apply envelope
            if audio.ndim == 2 and audio.shape[1] == 2:
                 if len(env) == audio.shape[0]:
                      audio = audio * env[:, np.newaxis]
                 else:
                      print(f"Error: Envelope length ({len(env)}) mismatch with audio length ({audio.shape[0]}). Skipping envelope.")
            elif audio.ndim == 1:
                 if len(env) == len(audio):
                      audio = audio * env
                 else:
                      print(f"Error: Envelope length ({len(env)}) mismatch with mono audio length ({len(audio)}). Skipping envelope.")
            else:
                 print(f"Warning: Audio shape ({audio.shape}) not suitable for envelope application.")

        except Exception as e:
            print(f"Error applying envelope type '{env_type}':")
            traceback.print_exc()

    # --- Ensure output is stereo ---
    if audio.ndim == 1:
        print(f"Note: Synth function '{actual_func_name}' returned mono audio. Panning.")
        audio = pan2(audio, cleaned_params.get('pan', 0))
    elif audio.ndim == 2 and audio.shape[1] == 1:
        print(f"Note: Synth function '{actual_func_name}' returned mono audio (N, 1). Panning.")
        audio = pan2(audio[:,0], cleaned_params.get('pan', 0))

    # Final check for shape (N, 2)
    if not (audio.ndim == 2 and audio.shape[1] == 2):
         print(f"Error: Final audio shape for voice is incorrect ({audio.shape}). Returning silence.")
         N = int(duration * sample_rate)
         return np.zeros((N, 2))

    return audio


def assemble_track_from_data(track_data, sample_rate, crossfade_duration):
    """
    Assembles a track from a track_data dictionary (loaded from JSON).
    Uses crossfading between steps.
    """
    steps_data = track_data.get("steps", [])
    if not steps_data:
        print("Warning: No steps found in track data.")
        return np.zeros((sample_rate, 2)) # Return 1 second silence

    # Calculate total duration and sample count from steps
    total_duration = sum(float(step.get("duration", 0)) for step in steps_data)
    if total_duration <= 0:
        print("Warning: Track has zero or negative total duration.")
        return np.zeros((sample_rate, 2))

    total_samples = int(total_duration * sample_rate)
    track = np.zeros((total_samples, 2))
    current_time = 0.0
    last_step_end_sample_in_track = 0 # Track the end sample of the previous step's audio IN THE FINAL TRACK

    crossfade_samples = int(crossfade_duration * sample_rate)

    print(f"Assembling track: {len(steps_data)} steps, Total Duration: {total_duration:.2f}s, Crossfade: {crossfade_duration:.2f}s ({crossfade_samples} samples)")

    for i, step_data in enumerate(steps_data):
        step_duration = float(step_data.get("duration", 0))
        if step_duration <= 0:
            print(f"Skipping step {i+1} with zero or negative duration.")
            continue

        step_start_time = current_time
        step_start_sample_abs = int(step_start_time * sample_rate) # Absolute start sample in timeline
        step_end_sample_abs = step_start_sample_abs + int(step_duration * sample_rate)
        N_step = int(step_duration * sample_rate) # Samples in this step

        print(f"  Processing Step {i+1}: Start: {step_start_time:.2f}s, Duration: {step_duration:.2f}s, Samples: {N_step}")

        # Generate audio for all voices in this step and mix them
        step_audio_mix = np.zeros((N_step, 2))
        voices_data = step_data.get("voices", [])

        if not voices_data:
             print(f"    Warning: Step {i+1} has no voices.")
        else:
            for j, voice_data in enumerate(voices_data):
                func_name_short = voice_data.get('synth_function_name', 'UnknownFunc')
                print(f"      Generating Voice {j+1}: {func_name_short}")
                # Pass absolute start time for potential phase coherence needs
                voice_audio = generate_voice_audio(voice_data, step_duration, sample_rate, step_start_time)

                # Add generated audio if valid
                if voice_audio is not None and voice_audio.shape[0] == N_step and voice_audio.ndim == 2 and voice_audio.shape[1] == 2:
                    step_audio_mix += voice_audio
                elif voice_audio is not None:
                     print(f"      Error: Voice {j+1} ({func_name_short}) generated audio shape mismatch ({voice_audio.shape} vs {(N_step, 2)}). Skipping voice.")
                # else: generate_voice_audio already printed error


        # --- Placement and Crossfading ---
        place_start_sample = step_start_sample_abs
        place_end_sample = step_end_sample_abs # Nominal end based on duration

        # Clip placement indices to track boundaries
        safe_place_start = max(0, place_start_sample)
        safe_place_end = min(total_samples, place_end_sample)
        segment_len_in_track = safe_place_end - safe_place_start

        if segment_len_in_track <= 0:
             print(f"    Skipping Step {i+1} placement (no valid range in track).")
             continue # Skip if no valid placement range

        # Determine the portion of step_audio_mix to use
        # Start from beginning of step audio, take samples corresponding to duration within track bounds
        audio_start_idx = 0
        audio_end_idx = min(N_step, segment_len_in_track)
        step_audio_to_place = step_audio_mix[audio_start_idx:audio_end_idx]

        # Ensure step_audio_to_place has the correct length for the track segment
        if step_audio_to_place.shape[0] != segment_len_in_track:
             print(f"    Warning: Step {i+1} audio length ({step_audio_to_place.shape[0]}) doesn't match track segment length ({segment_len_in_track}). Adjusting.")
             # This might happen if step_end_sample_abs > total_samples
             # We already sliced step_audio_to_place correctly above, this check might be redundant
             # If it's too short (shouldn't happen often), pad it
             if step_audio_to_place.shape[0] < segment_len_in_track:
                  step_audio_to_place = np.pad(step_audio_to_place, ((0, segment_len_in_track - step_audio_to_place.shape[0]), (0,0)), 'constant')


        # Determine overlap region with the previously placed audio in the track
        overlap_start_sample_in_track = safe_place_start
        overlap_end_sample_in_track = min(safe_place_end, last_step_end_sample_in_track)
        overlap_samples = overlap_end_sample_in_track - overlap_start_sample_in_track

        # Check if there's a valid overlap region for crossfading
        can_crossfade = i > 0 and overlap_samples > 0 and crossfade_samples > 0

        if can_crossfade:
            # Ensure we don't try to crossfade more samples than available in the overlap
            # or more than the specified crossfade duration
            actual_crossfade_samples = min(overlap_samples, crossfade_samples)
            print(f"    Crossfading Step {i+1} with previous. Overlap: {overlap_samples / sample_rate:.2f}s, Crossfade: {actual_crossfade_samples / sample_rate:.2f}s")

            # Get segments for crossfading
            prev_segment = track[overlap_start_sample_in_track : overlap_start_sample_in_track + actual_crossfade_samples]
            new_segment = step_audio_to_place[:actual_crossfade_samples]

            # Perform crossfade
            blended_segment = crossfade_signals(prev_segment, new_segment, sample_rate, actual_crossfade_samples / sample_rate)

            # Place blended segment into the track (overwrite the previous tail)
            track[overlap_start_sample_in_track : overlap_start_sample_in_track + actual_crossfade_samples] = blended_segment

            # Add the remainder of the new step (after the crossfaded part)
            remaining_start_index_in_step = actual_crossfade_samples
            remaining_start_index_in_track = overlap_start_sample_in_track + actual_crossfade_samples
            remaining_end_index_in_track = safe_place_end # Use safe end index

            if remaining_start_index_in_track < remaining_end_index_in_track:
                num_remaining_samples = remaining_end_index_in_track - remaining_start_index_in_track
                # Ensure we don't read past the end of step_audio_to_place
                if remaining_start_index_in_step + num_remaining_samples <= step_audio_to_place.shape[0]:
                    track[remaining_start_index_in_track : remaining_end_index_in_track] += step_audio_to_place[remaining_start_index_in_step : remaining_start_index_in_step + num_remaining_samples]
                else:
                     # This case indicates a potential logic error or unexpected audio length
                     available_samples = step_audio_to_place.shape[0] - remaining_start_index_in_step
                     if available_samples > 0:
                          print(f"    Warning: Mismatch adding remaining samples for step {i+1}. Adding {available_samples} samples.")
                          track[remaining_start_index_in_track : remaining_start_index_in_track + available_samples] += step_audio_to_place[remaining_start_index_in_step : remaining_start_index_in_step + available_samples]

        else:
            # No crossfade (first step or no overlap), just add the audio
            print(f"    Placing Step {i+1} without crossfade.")
            track[safe_place_start:safe_place_end] += step_audio_to_place

        # Update the end marker for the next iteration's overlap check
        # Use the actual end sample placed in the track
        last_step_end_sample_in_track = max(last_step_end_sample_in_track, safe_place_end)
        current_time += step_duration # Move time forward by the step's duration

    # Final clip just in case rounding caused slight over/under run
    if track.shape[0] > total_samples:
        track = track[:total_samples]
    elif track.shape[0] < total_samples:
         print(f"Warning: Final track length ({track.shape[0]}) is shorter than expected ({total_samples}). Padding end with silence.")
         track = np.pad(track, ((0, total_samples - track.shape[0]), (0,0)), 'constant')

    print("Track assembly finished.")
    return track


# -----------------------------------------------------------------------------
# JSON Loading/Saving
# -----------------------------------------------------------------------------

# Custom JSON encoder to handle numpy types (if needed, though GUI should convert)
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist() # Convert arrays to lists
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
        # Further validation could check step/voice structure
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
        # GUI should provide data with standard python types already
        # Using NumpyEncoder just in case numpy types slipped through
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
# Main Generation Function (called by GUI or script)
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

    print(f"\n--- Starting WAV Generation ---")
    print(f"Sample Rate: {sample_rate} Hz")
    print(f"Crossfade Duration: {crossfade_duration} s")
    print(f"Output File: {output_filename}")

    # Assemble the track
    track_audio = assemble_track_from_data(track_data, sample_rate, crossfade_duration)

    if track_audio is None or track_audio.size == 0:
        print("Error: Track assembly failed or resulted in empty audio.")
        return False

    # Apply safety limiter
    track_audio = safety_limiter(track_audio, threshold=0.9)

    # Normalize and convert to 16-bit PCM
    max_abs_val = np.max(np.abs(track_audio))
    if max_abs_val > 1e-9: # Avoid division by zero if silent
        normalized_track = track_audio / max_abs_val
    else:
        normalized_track = track_audio # Already silent or zero

    # Ensure the data type is suitable for int16 conversion
    if not np.issubdtype(normalized_track.dtype, np.floating):
         print(f"Warning: Normalized track data type is not float ({normalized_track.dtype}). Attempting conversion.")
         try:
             normalized_track = normalized_track.astype(np.float64)
         except Exception as e:
              print(f"Error converting normalized track to float: {e}")
              return False

    track_int16 = np.int16(normalized_track * 32767)

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

# -----------------------------------------------------------------------------
# Example Usage (if script is run directly)
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    print("Running sound_creator.py directly.")
    print("This script defines audio generation functions and logic.")
    print("Run track_editor_gui.py to use the graphical interface.")

    # Example: Create and save a default JSON if it doesn't exist
    default_json_file = 'default_track.json'
    if not os.path.exists(default_json_file):
         print(f"Creating a simple default JSON file: {default_json_file}")
         default_data = {
             "global_settings": {
                 "sample_rate": 44100,
                 "crossfade_duration": 0.5,
                 "output_filename": "test_output.wav"
             },
             "steps": [
                 {
                     "duration": 5.0,
                     "voices": [
                         {
                             "synth_function_name": "binaural_beat", # Test new func
                             "is_transition": False,
                             "params": {"amp": 0.4, "baseFreq": 150, "beatFreq": 5},
                             "volume_envelope": None
                         },
                         {
                             "synth_function_name": "isochronic_tone", # Test new func
                             "is_transition": False,
                             "params": {"amp": 0.6, "baseFreq": 300, "beatFreq": 7, "rampPercent": 0.1, "gapPercent": 0.2, "pan": 0.5},
                             "volume_envelope": {"type": "linen", "params": {"attack": 0.5, "release": 1.0}}
                         }
                     ]
                 },
                 {
                     "duration": 5.0,
                     "voices": [
                         {
                             "synth_function_name": "binaural_beat", # Test transition
                             "is_transition": True,
                             "params": {
                                 "amp": 0.5, "startBaseFreq": 150, "endBaseFreq": 140,
                                 "startBeatFreq": 5, "endBeatFreq": 10
                             },
                             "volume_envelope": None
                         }
                     ]
                 }
             ]
         }
         save_track_to_json(default_data, default_json_file)
         print(f"Default file created. Run track_editor_gui.py to load and generate.")
    else:
         print(f"Default file '{default_json_file}' already exists.")
         # Optionally load and generate here for testing
         # print("\nAttempting to load and generate default track...")
         # track_definition = load_track_from_json(default_json_file)
         # if track_definition:
         #     generate_wav(track_definition)
         # else:
         #     print("Could not load default track definition.")


