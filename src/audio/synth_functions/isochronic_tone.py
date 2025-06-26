"""Isochronic tone synthesis functions."""

import numpy as np
from .common import pan2, trapezoid_envelope_vectorized, calculate_transition_alpha


def isochronic_tone(duration, sample_rate=44100, **params):
    """Generate an isochronic tone with extended modulation options."""

    # Legacy amplitude/pan parameters
    base_amp = float(params.get('amp', 0.5))
    pan = float(params.get('pan', 0.0))

    # Extended parameters matching ``binaural_beat`` (glitch parameters omitted)
    ampL = float(params.get('ampL', base_amp))
    ampR = float(params.get('ampR', base_amp))
    baseFreq = float(params.get('baseFreq', 200.0))
    beatFreq = float(params.get('beatFreq', 4.0))  # Pulse rate
    force_mono = bool(params.get('forceMono', False))
    startPhaseL = float(params.get('startPhaseL', 0.0))
    startPhaseR = float(params.get('startPhaseR', 0.0))
    ampOscDepthL = float(params.get('ampOscDepthL', 0.0))
    ampOscFreqL = float(params.get('ampOscFreqL', 0.0))
    ampOscDepthR = float(params.get('ampOscDepthR', 0.0))
    ampOscFreqR = float(params.get('ampOscFreqR', 0.0))
    freqOscRangeL = float(params.get('freqOscRangeL', 0.0))
    freqOscFreqL = float(params.get('freqOscFreqL', 0.0))
    freqOscRangeR = float(params.get('freqOscRangeR', 0.0))
    freqOscFreqR = float(params.get('freqOscFreqR', 0.0))
    ampOscPhaseOffsetL = float(params.get('ampOscPhaseOffsetL', 0.0))
    ampOscPhaseOffsetR = float(params.get('ampOscPhaseOffsetR', 0.0))
    phaseOscFreq = float(params.get('phaseOscFreq', 0.0))
    phaseOscRange = float(params.get('phaseOscRange', 0.0))

    rampPercent = float(params.get('rampPercent', 0.2))
    gapPercent = float(params.get('gapPercent', 0.15))

    N = int(sample_rate * duration)
    if N <= 0:
        return np.zeros((0, 2))

    t = np.linspace(0, duration, N, endpoint=False)
    dt = 1.0 / sample_rate if N > 1 else duration

    # --- Instantaneous carrier frequencies (with vibrato) ---
    vibL = (freqOscRangeL / 2.0) * np.sin(2 * np.pi * freqOscFreqL * t)
    vibR = (freqOscRangeR / 2.0) * np.sin(2 * np.pi * freqOscFreqR * t)

    inst_freq_L = baseFreq + vibL
    inst_freq_R = baseFreq + vibR

    if force_mono:
        inst_freq_L[:] = baseFreq
        inst_freq_R[:] = baseFreq

    inst_freq_L = np.maximum(0.0, inst_freq_L)
    inst_freq_R = np.maximum(0.0, inst_freq_R)

    # --- Phase accumulation ---
    phase_inc_L = 2 * np.pi * inst_freq_L * dt
    phase_inc_R = 2 * np.pi * inst_freq_R * dt
    phase_L = startPhaseL + np.cumsum(np.concatenate(([0.0], phase_inc_L[:-1])))
    phase_R = startPhaseR + np.cumsum(np.concatenate(([0.0], phase_inc_R[:-1])))

    # --- Phase modulation ---
    if phaseOscFreq != 0.0 or phaseOscRange != 0.0:
        dphi = (phaseOscRange / 2.0) * np.sin(2 * np.pi * phaseOscFreq * t)
        phase_L -= dphi
        phase_R += dphi

    carrier_L = np.sin(phase_L)
    carrier_R = np.sin(phase_R)

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

    # Apply envelope to carriers
    env_amp_L = 1.0 - ampOscDepthL * (0.5 * (1.0 + np.sin(2 * np.pi * ampOscFreqL * t + ampOscPhaseOffsetL)))
    env_amp_R = 1.0 - ampOscDepthR * (0.5 * (1.0 + np.sin(2 * np.pi * ampOscFreqR * t + ampOscPhaseOffsetR)))

    left = carrier_L * iso_env * env_amp_L * ampL
    right = carrier_R * iso_env * env_amp_R * ampR

    if pan != 0.0:
        # Apply additional panning if requested
        stereo = np.column_stack((left, right))
        stereo = pan2(stereo.mean(axis=1), pan=pan)
        left, right = stereo[:, 0], stereo[:, 1]

    audio = np.column_stack((left, right))

    # Note: Volume envelope (like ADSR/Linen) is applied *within* generate_voice_audio if specified there.
    # The trapezoid envelope is inherent to the isochronic tone generation itself.

    return audio.astype(np.float32)


def isochronic_tone_transition(duration, sample_rate=44100, initial_offset=0.0, post_offset=0.0, **params):

    """Transitioning version of :func:`isochronic_tone`."""

    base_amp = float(params.get('amp', 0.5))
    startAmpL = float(params.get('startAmpL', params.get('ampL', base_amp)))
    endAmpL = float(params.get('endAmpL', startAmpL))
    startAmpR = float(params.get('startAmpR', params.get('ampR', base_amp)))
    endAmpR = float(params.get('endAmpR', startAmpR))

    startBaseFreq = float(params.get('startBaseFreq', 200.0))
    endBaseFreq = float(params.get('endBaseFreq', startBaseFreq))
    startBeatFreq = float(params.get('startBeatFreq', 4.0))
    endBeatFreq = float(params.get('endBeatFreq', startBeatFreq))

    startForceMono = float(params.get('startForceMono', params.get('forceMono', 0.0)))
    endForceMono = float(params.get('endForceMono', startForceMono))
    startStartPhaseL = float(params.get('startStartPhaseL', params.get('startPhaseL', 0.0)))
    endStartPhaseL = float(params.get('endStartPhaseL', startStartPhaseL))
    startStartPhaseR = float(params.get('startStartPhaseR', params.get('startPhaseR', 0.0)))
    endStartPhaseR = float(params.get('endStartPhaseR', startStartPhaseR))

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

    startPOF = float(params.get('startPhaseOscFreq', params.get('phaseOscFreq', 0.0)))
    endPOF = float(params.get('endPhaseOscFreq', startPOF))
    startPOR = float(params.get('startPhaseOscRange', params.get('phaseOscRange', 0.0)))
    endPOR = float(params.get('endPhaseOscRange', startPOR))

    rampPercent = float(params.get('rampPercent', 0.2))
    gapPercent = float(params.get('gapPercent', 0.15))
    pan = float(params.get('pan', 0.0))


    N = int(sample_rate * duration)
    if N <= 0:
        return np.zeros((0, 2))

    t = np.linspace(0, duration, N, endpoint=False)

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

    # --- Carrier Wave (Time-Varying Frequency with vibrato) ---
    dt = 1.0 / sample_rate if N > 1 else duration

    vibL = (startFORL + (endFORL - startFORL) * alpha) / 2.0
    vibL *= np.sin(2 * np.pi * (startFOFL + (endFOFL - startFOFL) * alpha) * t)
    vibR = (startFORR + (endFORR - startFORR) * alpha) / 2.0
    vibR *= np.sin(2 * np.pi * (startFOFR + (endFOFR - startFOFR) * alpha) * t)

    inst_freq_L = instantaneous_carrier_freq_array + vibL
    inst_freq_R = instantaneous_carrier_freq_array + vibR

    force_mono_arr = startForceMono + (endForceMono - startForceMono) * alpha
    mono_mask = force_mono_arr > 0.5
    inst_freq_L[mono_mask] = instantaneous_carrier_freq_array[mono_mask]
    inst_freq_R[mono_mask] = instantaneous_carrier_freq_array[mono_mask]

    inst_freq_L = np.maximum(0.0, inst_freq_L)
    inst_freq_R = np.maximum(0.0, inst_freq_R)

    phase_inc_L = 2 * np.pi * inst_freq_L * dt
    phase_inc_R = 2 * np.pi * inst_freq_R * dt
    phase_L = (startStartPhaseL + (endStartPhaseL - startStartPhaseL) * alpha)
    phase_R = (startStartPhaseR + (endStartPhaseR - startStartPhaseR) * alpha)
    phase_L = phase_L + np.cumsum(np.concatenate(([0.0], phase_inc_L[:-1])))
    phase_R = phase_R + np.cumsum(np.concatenate(([0.0], phase_inc_R[:-1])))

    pOF_arr = startPOF + (endPOF - startPOF) * alpha
    pOR_arr = startPOR + (endPOR - startPOR) * alpha
    if np.any(pOF_arr != 0.0) or np.any(pOR_arr != 0.0):
        dphi = (pOR_arr / 2.0) * np.sin(2 * np.pi * pOF_arr * t)
        phase_L -= dphi
        phase_R += dphi

    carrier_L = np.sin(phase_L)
    carrier_R = np.sin(phase_R)

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

    # Generate the trapezoid envelope
    iso_env = trapezoid_envelope_vectorized(t_in_cycle, cycle_len_array, rampPercent, gapPercent)


    env_amp_L = 1.0 - (startAODL + (endAODL - startAODL) * alpha) * (
        0.5 * (1.0 + np.sin(2 * np.pi * (startAOFL + (endAOFL - startAOFL) * alpha) * t +
                           (startAmpOscPhaseOffsetL + (endAmpOscPhaseOffsetL - startAmpOscPhaseOffsetL) * alpha)))
    )
    env_amp_R = 1.0 - (startAODR + (endAODR - startAODR) * alpha) * (
        0.5 * (1.0 + np.sin(2 * np.pi * (startAOFR + (endAOFR - startAOFR) * alpha) * t +
                           (startAmpOscPhaseOffsetR + (endAmpOscPhaseOffsetR - startAmpOscPhaseOffsetR) * alpha)))
    )

    ampL_arr = startAmpL + (endAmpL - startAmpL) * alpha
    ampR_arr = startAmpR + (endAmpR - startAmpR) * alpha

    left = carrier_L * iso_env * env_amp_L * ampL_arr
    right = carrier_R * iso_env * env_amp_R * ampR_arr

    if pan != 0.0:
        stereo = np.column_stack((left, right))
        stereo = pan2(stereo.mean(axis=1), pan=pan)
        left, right = stereo[:, 0], stereo[:, 1]

    audio = np.column_stack((left, right))

    # Note: Volume envelope (like ADSR/Linen) is applied *within* generate_voice_audio if specified there.

    return audio.astype(np.float32)
