"""Isochronic tone synthesis functions."""

import numpy as np
from .common import pan2, trapezoid_envelope_vectorized, calculate_transition_alpha


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


def isochronic_tone_transition(duration, sample_rate=44100, initial_offset=0.0, post_offset=0.0, **params):
    """Isochronic tone where every parameter can transition over time."""
    startAmp = float(params.get('startAmp', params.get('amp', 0.5)))
    endAmp = float(params.get('endAmp', startAmp))
    startBaseFreq = float(params.get('startBaseFreq', params.get('baseFreq', 200.0)))
    endBaseFreq = float(params.get('endBaseFreq', startBaseFreq))
    startBeatFreq = float(params.get('startBeatFreq', params.get('beatFreq', 4.0)))
    endBeatFreq = float(params.get('endBeatFreq', startBeatFreq))
    startRampPercent = float(params.get('startRampPercent', params.get('rampPercent', 0.2)))
    endRampPercent = float(params.get('endRampPercent', startRampPercent))
    startGapPercent = float(params.get('startGapPercent', params.get('gapPercent', 0.15)))
    endGapPercent = float(params.get('endGapPercent', startGapPercent))
    startPan = float(params.get('startPan', params.get('pan', 0.0)))
    endPan = float(params.get('endPan', startPan))

    N = int(sample_rate * duration)
    if N <= 0:
        return np.zeros((0, 2))

    t_abs = np.linspace(0, duration, N, endpoint=False)

    curve = params.get('transition_curve', 'linear')
    alpha = calculate_transition_alpha(duration, sample_rate, initial_offset, post_offset, curve)

    # --- Interpolate Parameters ---
    base_freq_array = startBaseFreq + (endBaseFreq - startBaseFreq) * alpha
    beat_freq_array = startBeatFreq + (endBeatFreq - startBeatFreq) * alpha  # Pulse rate
    amp_array = startAmp + (endAmp - startAmp) * alpha
    ramp_percent_array = startRampPercent + (endRampPercent - startRampPercent) * alpha
    gap_percent_array = startGapPercent + (endGapPercent - startGapPercent) * alpha
    pan_array = startPan + (endPan - startPan) * alpha

    # Ensure frequencies are non-negative
    instantaneous_carrier_freq_array = np.maximum(0.0, base_freq_array)
    instantaneous_beat_freq_array = np.maximum(0.0, beat_freq_array)
    ramp_percent_array = np.clip(ramp_percent_array, 0.0, 1.0)
    gap_percent_array = np.clip(gap_percent_array, 0.0, 1.0)
    pan_array = np.clip(pan_array, -1.0, 1.0)

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

    # Generate the trapezoid envelope using time-varying ramp/gap settings
    iso_env = trapezoid_envelope_vectorized(t_in_cycle, cycle_len_array, ramp_percent_array, gap_percent_array)

    # Apply envelope to carrier
    mono_signal = carrier * iso_env

    # Apply overall amplitude
    output_mono = mono_signal * amp_array

    # Variable panning
    angle = (pan_array + 1.0) * np.pi / 4.0
    left_gain = np.cos(angle)
    right_gain = np.sin(angle)
    audio = np.column_stack((output_mono * left_gain, output_mono * right_gain))

    # Note: Volume envelope (like ADSR/Linen) is applied *within* generate_voice_audio if specified there.

    return audio.astype(np.float32)
