"""Rhythmic waveshaping synthesis functions."""

import numpy as np
from .common import sine_wave, sine_wave_varying, pan2, calculate_transition_alpha


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



def rhythmic_waveshaping_transition(duration, sample_rate=44100, initial_offset=0.0, post_offset=0.0, **params):
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

    curve = params.get('transition_curve', 'linear')
    alpha = calculate_transition_alpha(duration, sample_rate, initial_offset, post_offset, curve)

    # Interpolate parameters using alpha
    currentCarrierFreq = startCarrierFreq + (endCarrierFreq - startCarrierFreq) * alpha
    currentModFreq = startModFreq + (endModFreq - startModFreq) * alpha
    currentModDepth = startModDepth + (endModDepth - startModDepth) * alpha
    currentShapeAmount = startShapeAmount + (endShapeAmount - startShapeAmount) * alpha

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
