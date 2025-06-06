"""Stereo amplitude modulation with independent L/R modulators."""

import numpy as np
from .common import sine_wave, sine_wave_varying, calculate_transition_alpha


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


def stereo_am_independent_transition(duration, sample_rate=44100, initial_offset=0.0, post_offset=0.0, **params):
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

    alpha = calculate_transition_alpha(duration, sample_rate, initial_offset, post_offset)

    # Interpolate parameters using alpha
    currentCarrierFreq = startCarrierFreq + (endCarrierFreq - startCarrierFreq) * alpha
    currentModFreqL = startModFreqL + (endModFreqL - startModFreqL) * alpha
    currentModDepthL = startModDepthL + (endModDepthL - startModDepthL) * alpha
    currentModFreqR = startModFreqR + (endModFreqR - startModFreqR) * alpha
    currentModDepthR = startModDepthR + (endModDepthR - startModDepthR) * alpha
    currentStereoWidthHz = startStereoWidthHz + (endStereoWidthHz - startStereoWidthHz) * alpha

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
