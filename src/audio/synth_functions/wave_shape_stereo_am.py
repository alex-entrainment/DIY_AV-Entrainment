"""Combines rhythmic waveshaping and stereo AM."""

import numpy as np
import math
from .common import sine_wave, sine_wave_varying


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
