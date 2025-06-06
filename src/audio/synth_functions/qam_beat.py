"""
Enhanced QAM Beat synthesis functions.

Generates QAM-based binaural beats with advanced modulation capabilities.
Includes multiple modulation sources, cross-channel interactions, and
psychoacoustic enhancements for entrainment applications.
"""

import numpy as np
import numba
from .common import apply_filters


def qam_beat(duration, sample_rate=44100, **params):
    """
    Generates an enhanced QAM-based binaural beat.
    
    Features:
    - Multiple simultaneous AM components per channel
    - Cross-channel modulation coupling
    - Harmonic relationships between modulations
    - Psychoacoustic enhancements (beating harmonics, phantom tones)
    - Non-linear modulation shapes
    """
    # --- Core parameters ---
    ampL = float(params.get('ampL', 0.5))
    ampR = float(params.get('ampR', 0.5))
    
    baseFreqL = float(params.get('baseFreqL', 200.0)) 
    baseFreqR = float(params.get('baseFreqR', 204.0))  # Default 4Hz beat
    
    # --- Primary QAM modulation ---
    qamAmFreqL = float(params.get('qamAmFreqL', 4.0)) 
    qamAmDepthL = float(params.get('qamAmDepthL', 0.5)) 
    qamAmPhaseOffsetL = float(params.get('qamAmPhaseOffsetL', 0.0))
    
    qamAmFreqR = float(params.get('qamAmFreqR', 4.0)) 
    qamAmDepthR = float(params.get('qamAmDepthR', 0.5)) 
    qamAmPhaseOffsetR = float(params.get('qamAmPhaseOffsetR', 0.0))
    
    # --- Secondary QAM modulation ---
    qamAm2FreqL = float(params.get('qamAm2FreqL', 0.0))
    qamAm2DepthL = float(params.get('qamAm2DepthL', 0.0))
    qamAm2PhaseOffsetL = float(params.get('qamAm2PhaseOffsetL', 0.0))
    
    qamAm2FreqR = float(params.get('qamAm2FreqR', 0.0))
    qamAm2DepthR = float(params.get('qamAm2DepthR', 0.0))
    qamAm2PhaseOffsetR = float(params.get('qamAm2PhaseOffsetR', 0.0))
    
    # --- Modulation shape control ---
    modShapeL = float(params.get('modShapeL', 1.0))  # 1=sine, >1=sharper, <1=smoother
    modShapeR = float(params.get('modShapeR', 1.0))
    
    # --- Cross-channel coupling ---
    crossModDepth = float(params.get('crossModDepth', 0.0))  # Amount of cross-channel AM coupling
    crossModDelay = float(params.get('crossModDelay', 0.0))  # Phase delay for cross coupling
    
    # --- Harmonic enhancement ---
    harmonicDepth = float(params.get('harmonicDepth', 0.0))  # Add carrier harmonics
    harmonicRatio = float(params.get('harmonicRatio', 2.0))  # Harmonic frequency ratio
    
    # --- Sub-harmonic modulation ---
    subHarmonicFreq = float(params.get('subHarmonicFreq', 0.0))  # Sub-harmonic of beat freq
    subHarmonicDepth = float(params.get('subHarmonicDepth', 0.0))
    
    # --- Phase parameters ---
    startPhaseL = float(params.get('startPhaseL', 0.0)) 
    startPhaseR = float(params.get('startPhaseR', 0.0))
    
    phaseOscFreq = float(params.get('phaseOscFreq', 0.0))
    phaseOscRange = float(params.get('phaseOscRange', 0.0)) 
    phaseOscPhaseOffset = float(params.get('phaseOscPhaseOffset', 0.0))
    
    # --- Psychoacoustic enhancement ---
    beatingSidebands = bool(params.get('beatingSidebands', False))  # Add beating sidebands
    sidebandOffset = float(params.get('sidebandOffset', 1.0))  # Hz offset from carrier
    sidebandDepth = float(params.get('sidebandDepth', 0.1))
    
    # --- Envelope shaping ---
    attackTime = float(params.get('attackTime', 0.0))  # Fade-in time
    releaseTime = float(params.get('releaseTime', 0.0))  # Fade-out time
    
    N = int(duration * sample_rate)
    if N <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    
    # Call enhanced Numba core function
    raw_signal = _qam_beat_core(
        N, duration, float(sample_rate),
        ampL, ampR,
        baseFreqL, baseFreqR,
        qamAmFreqL, qamAmDepthL, qamAmPhaseOffsetL,
        qamAmFreqR, qamAmDepthR, qamAmPhaseOffsetR,
        qamAm2FreqL, qamAm2DepthL, qamAm2PhaseOffsetL,
        qamAm2FreqR, qamAm2DepthR, qamAm2PhaseOffsetR,
        modShapeL, modShapeR,
        crossModDepth, crossModDelay,
        harmonicDepth, harmonicRatio,
        subHarmonicFreq, subHarmonicDepth,
        startPhaseL, startPhaseR,
        phaseOscFreq, phaseOscRange, phaseOscPhaseOffset,
        beatingSidebands, sidebandOffset, sidebandDepth,
        attackTime, releaseTime
    )
    
    if raw_signal.size > 0:
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
    qamAm2FreqL, qamAm2DepthL, qamAm2PhaseOffsetL,
    qamAm2FreqR, qamAm2DepthR, qamAm2PhaseOffsetR,
    modShapeL, modShapeR,
    crossModDepth, crossModDelay,
    harmonicDepth, harmonicRatio,
    subHarmonicFreq, subHarmonicDepth,
    startPhaseL, startPhaseR,
    phaseOscFreq, phaseOscRange, phaseOscPhaseOffset,
    beatingSidebands, sidebandOffset, sidebandDepth,
    attackTime, releaseTime,
    alpha_arr
):
    if N <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    
    t_arr = np.empty(N, dtype=np.float64)
    dt = duration_float / N
    for i in numba.prange(N):
        t_arr[i] = i * dt
    
    # Generate carrier phases with phase oscillation
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
    
    # Generate complex envelopes
    envL = np.ones(N, dtype=np.float64)
    envR = np.ones(N, dtype=np.float64)
    
    # Primary modulation with shape control
    for i in numba.prange(N):
        if qamAmFreqL != 0.0 and qamAmDepthL != 0.0:
            if modShapeL == 1.0:
                modL1 = np.cos(2 * np.pi * qamAmFreqL * t_arr[i] + qamAmPhaseOffsetL)
            else:
                # Use power shaping for non-sinusoidal modulation
                phase = 2 * np.pi * qamAmFreqL * t_arr[i] + qamAmPhaseOffsetL
                cos_val = np.cos(phase)
                modL1 = np.sign(cos_val) * np.abs(cos_val) ** (1.0 / modShapeL)
            envL[i] *= (1.0 + qamAmDepthL * modL1)
        
        if qamAmFreqR != 0.0 and qamAmDepthR != 0.0:
            if modShapeR == 1.0:
                modR1 = np.cos(2 * np.pi * qamAmFreqR * t_arr[i] + qamAmPhaseOffsetR)
            else:
                phase = 2 * np.pi * qamAmFreqR * t_arr[i] + qamAmPhaseOffsetR
                cos_val = np.cos(phase)
                modR1 = np.sign(cos_val) * np.abs(cos_val) ** (1.0 / modShapeR)
            envR[i] *= (1.0 + qamAmDepthR * modR1)
    
    # Secondary modulation
    if qamAm2FreqL != 0.0 and qamAm2DepthL != 0.0:
        for i in numba.prange(N):
            modL2 = np.cos(2 * np.pi * qamAm2FreqL * t_arr[i] + qamAm2PhaseOffsetL)
            envL[i] *= (1.0 + qamAm2DepthL * modL2)
    
    if qamAm2FreqR != 0.0 and qamAm2DepthR != 0.0:
        for i in numba.prange(N):
            modR2 = np.cos(2 * np.pi * qamAm2FreqR * t_arr[i] + qamAm2PhaseOffsetR)
            envR[i] *= (1.0 + qamAm2DepthR * modR2)
    
    # Cross-channel modulation coupling
    if crossModDepth != 0.0 and crossModDelay != 0.0:
        crossEnvL = np.copy(envL)
        crossEnvR = np.copy(envR)
        delay_samples = int(crossModDelay * sample_rate_float)
        
        for i in numba.prange(N):
            if i >= delay_samples:
                # Cross-modulate with delayed envelope from opposite channel
                envL[i] *= (1.0 + crossModDepth * (crossEnvR[i - delay_samples] - 1.0))
                envR[i] *= (1.0 + crossModDepth * (crossEnvL[i - delay_samples] - 1.0))
    
    # Sub-harmonic modulation
    if subHarmonicFreq != 0.0 and subHarmonicDepth != 0.0:
        for i in numba.prange(N):
            subMod = np.cos(2 * np.pi * subHarmonicFreq * t_arr[i])
            envL[i] *= (1.0 + subHarmonicDepth * subMod)
            envR[i] *= (1.0 + subHarmonicDepth * subMod)
    
    # Generate output with optional harmonics and sidebands
    out = np.empty((N, 2), dtype=np.float32)
    
    for i in numba.prange(N):
        # Main carriers
        sigL = envL[i] * np.cos(phL_carrier[i])
        sigR = envR[i] * np.cos(phR_carrier[i])
        
        # Add harmonics
        if harmonicDepth != 0.0:
            sigL += harmonicDepth * envL[i] * np.cos(harmonicRatio * phL_carrier[i])
            sigR += harmonicDepth * envR[i] * np.cos(harmonicRatio * phR_carrier[i])
        
        # Add beating sidebands for psychoacoustic enhancement
        if beatingSidebands and sidebandDepth != 0.0:
            # Lower sideband
            sigL += sidebandDepth * envL[i] * np.cos(phL_carrier[i] - 2 * np.pi * sidebandOffset * t_arr[i])
            sigR += sidebandDepth * envR[i] * np.cos(phR_carrier[i] - 2 * np.pi * sidebandOffset * t_arr[i])
            # Upper sideband
            sigL += sidebandDepth * envL[i] * np.cos(phL_carrier[i] + 2 * np.pi * sidebandOffset * t_arr[i])
            sigR += sidebandDepth * envR[i] * np.cos(phR_carrier[i] + 2 * np.pi * sidebandOffset * t_arr[i])
        
        # Apply envelope shaping
        envelope_mult = 1.0
        if attackTime > 0.0 and t_arr[i] < attackTime:
            envelope_mult *= t_arr[i] / attackTime
        if releaseTime > 0.0 and t_arr[i] > (duration_float - releaseTime):
            envelope_mult *= (duration_float - t_arr[i]) / releaseTime
        
        out[i, 0] = np.float32(sigL * ampL * envelope_mult)
        out[i, 1] = np.float32(sigR * ampR * envelope_mult)
    
    return out


from .common import calculate_transition_alpha


def qam_beat_transition(duration, sample_rate=44100, initial_offset=0.0, post_offset=0.0, **params):
    """
    Enhanced QAM-based binaural beat with parameter transitions.
    Includes all enhanced features with smooth interpolation.
    """
    # Start/end parameters for all features
    startAmpL = float(params.get('startAmpL', params.get('ampL', 0.5)))
    endAmpL = float(params.get('endAmpL', startAmpL))
    startAmpR = float(params.get('startAmpR', params.get('ampR', 0.5)))
    endAmpR = float(params.get('endAmpR', startAmpR))
    
    startBaseFreqL = float(params.get('startBaseFreqL', params.get('baseFreqL', 200.0)))
    endBaseFreqL = float(params.get('endBaseFreqL', startBaseFreqL))
    startBaseFreqR = float(params.get('startBaseFreqR', params.get('baseFreqR', 204.0)))
    endBaseFreqR = float(params.get('endBaseFreqR', startBaseFreqR))
    
    # Primary modulation transitions
    startQamAmFreqL = float(params.get('startQamAmFreqL', params.get('qamAmFreqL', 4.0)))
    endQamAmFreqL = float(params.get('endQamAmFreqL', startQamAmFreqL))
    startQamAmFreqR = float(params.get('startQamAmFreqR', params.get('qamAmFreqR', 4.0)))
    endQamAmFreqR = float(params.get('endQamAmFreqR', startQamAmFreqR))
    
    startQamAmDepthL = float(params.get('startQamAmDepthL', params.get('qamAmDepthL', 0.5)))
    endQamAmDepthL = float(params.get('endQamAmDepthL', startQamAmDepthL))
    startQamAmDepthR = float(params.get('startQamAmDepthR', params.get('qamAmDepthR', 0.5)))
    endQamAmDepthR = float(params.get('endQamAmDepthR', startQamAmDepthR))
    
    startQamAmPhaseOffsetL = float(params.get('startQamAmPhaseOffsetL', params.get('qamAmPhaseOffsetL', 0.0)))
    endQamAmPhaseOffsetL = float(params.get('endQamAmPhaseOffsetL', startQamAmPhaseOffsetL))
    startQamAmPhaseOffsetR = float(params.get('startQamAmPhaseOffsetR', params.get('qamAmPhaseOffsetR', 0.0)))
    endQamAmPhaseOffsetR = float(params.get('endQamAmPhaseOffsetR', startQamAmPhaseOffsetR))
    
    # Secondary modulation transitions
    startQamAm2FreqL = float(params.get('startQamAm2FreqL', params.get('qamAm2FreqL', 0.0)))
    endQamAm2FreqL = float(params.get('endQamAm2FreqL', startQamAm2FreqL))
    startQamAm2FreqR = float(params.get('startQamAm2FreqR', params.get('qamAm2FreqR', 0.0)))
    endQamAm2FreqR = float(params.get('endQamAm2FreqR', startQamAm2FreqR))
    
    startQamAm2DepthL = float(params.get('startQamAm2DepthL', params.get('qamAm2DepthL', 0.0)))
    endQamAm2DepthL = float(params.get('endQamAm2DepthL', startQamAm2DepthL))
    startQamAm2DepthR = float(params.get('startQamAm2DepthR', params.get('qamAm2DepthR', 0.0)))
    endQamAm2DepthR = float(params.get('endQamAm2DepthR', startQamAm2DepthR))
    
    startQamAm2PhaseOffsetL = float(params.get('startQamAm2PhaseOffsetL', params.get('qamAm2PhaseOffsetL', 0.0)))
    endQamAm2PhaseOffsetL = float(params.get('endQamAm2PhaseOffsetL', startQamAm2PhaseOffsetL))
    startQamAm2PhaseOffsetR = float(params.get('startQamAm2PhaseOffsetR', params.get('qamAm2PhaseOffsetR', 0.0)))
    endQamAm2PhaseOffsetR = float(params.get('endQamAm2PhaseOffsetR', startQamAm2PhaseOffsetR))
    
    # Enhanced feature transitions
    startModShapeL = float(params.get('startModShapeL', params.get('modShapeL', 1.0)))
    endModShapeL = float(params.get('endModShapeL', startModShapeL))
    startModShapeR = float(params.get('startModShapeR', params.get('modShapeR', 1.0)))
    endModShapeR = float(params.get('endModShapeR', startModShapeR))
    
    startCrossModDepth = float(params.get('startCrossModDepth', params.get('crossModDepth', 0.0)))
    endCrossModDepth = float(params.get('endCrossModDepth', startCrossModDepth))
    
    startHarmonicDepth = float(params.get('startHarmonicDepth', params.get('harmonicDepth', 0.0)))
    endHarmonicDepth = float(params.get('endHarmonicDepth', startHarmonicDepth))
    
    startSubHarmonicFreq = float(params.get('startSubHarmonicFreq', params.get('subHarmonicFreq', 0.0)))
    endSubHarmonicFreq = float(params.get('endSubHarmonicFreq', startSubHarmonicFreq))
    startSubHarmonicDepth = float(params.get('startSubHarmonicDepth', params.get('subHarmonicDepth', 0.0)))
    endSubHarmonicDepth = float(params.get('endSubHarmonicDepth', startSubHarmonicDepth))
    
    # Phase modulation transitions
    startPhaseOscFreq = float(params.get('startPhaseOscFreq', params.get('phaseOscFreq', 0.0)))
    endPhaseOscFreq = float(params.get('endPhaseOscFreq', startPhaseOscFreq))
    startPhaseOscRange = float(params.get('startPhaseOscRange', params.get('phaseOscRange', 0.0)))
    endPhaseOscRange = float(params.get('endPhaseOscRange', startPhaseOscRange))
    
    startStartPhaseL = float(params.get('startStartPhaseL', params.get('startPhaseL', 0.0)))
    endStartPhaseL = float(params.get('endStartPhaseL', startStartPhaseL))
    startStartPhaseR = float(params.get('startStartPhaseR', params.get('startPhaseR', 0.0)))
    endStartPhaseR = float(params.get('endStartPhaseR', startStartPhaseR))
    
    # Static parameters (don't transition)
    crossModDelay = float(params.get('crossModDelay', 0.0))
    harmonicRatio = float(params.get('harmonicRatio', 2.0))
    phaseOscPhaseOffset = float(params.get('phaseOscPhaseOffset', 0.0))
    beatingSidebands = bool(params.get('beatingSidebands', False))
    sidebandOffset = float(params.get('sidebandOffset', 1.0))
    sidebandDepth = float(params.get('sidebandDepth', 0.1))
    attackTime = float(params.get('attackTime', 0.0))
    releaseTime = float(params.get('releaseTime', 0.0))
    
    N = int(duration * sample_rate)
    if N <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    
    alpha_arr = calculate_transition_alpha(duration, sample_rate, initial_offset, post_offset)

    raw_signal = _qam_beat_transition_core(
        N, float(duration), float(sample_rate),
        startAmpL, endAmpL, startAmpR, endAmpR,
        startBaseFreqL, endBaseFreqL, startBaseFreqR, endBaseFreqR,
        startQamAmFreqL, endQamAmFreqL, startQamAmDepthL, endQamAmDepthL,
        startQamAmFreqR, endQamAmFreqR, startQamAmDepthR, endQamAmDepthR,
        startQamAmPhaseOffsetL, endQamAmPhaseOffsetL,
        startQamAmPhaseOffsetR, endQamAmPhaseOffsetR,
        startQamAm2FreqL, endQamAm2FreqL, startQamAm2DepthL, endQamAm2DepthL,
        startQamAm2FreqR, endQamAm2FreqR, startQamAm2DepthR, endQamAm2DepthR,
        startQamAm2PhaseOffsetL, endQamAm2PhaseOffsetL,
        startQamAm2PhaseOffsetR, endQamAm2PhaseOffsetR,
        startModShapeL, endModShapeL, startModShapeR, endModShapeR,
        startCrossModDepth, endCrossModDepth, crossModDelay,
        startHarmonicDepth, endHarmonicDepth, harmonicRatio,
        startSubHarmonicFreq, endSubHarmonicFreq, startSubHarmonicDepth, endSubHarmonicDepth,
        startPhaseOscFreq, endPhaseOscFreq, startPhaseOscRange, endPhaseOscRange,
        startStartPhaseL, endStartPhaseL, startStartPhaseR, endStartPhaseR,
        phaseOscPhaseOffset,
        beatingSidebands, sidebandOffset, sidebandDepth,
        attackTime, releaseTime,
        alpha_arr
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
    s_qamAmFreqL, e_qamAmFreqL, s_qamAmDepthL, e_qamAmDepthL,
    s_qamAmFreqR, e_qamAmFreqR, s_qamAmDepthR, e_qamAmDepthR,
    s_qamAmPhaseOffsetL, e_qamAmPhaseOffsetL,
    s_qamAmPhaseOffsetR, e_qamAmPhaseOffsetR,
    s_qamAm2FreqL, e_qamAm2FreqL, s_qamAm2DepthL, e_qamAm2DepthL,
    s_qamAm2FreqR, e_qamAm2FreqR, s_qamAm2DepthR, e_qamAm2DepthR,
    s_qamAm2PhaseOffsetL, e_qamAm2PhaseOffsetL,
    s_qamAm2PhaseOffsetR, e_qamAm2PhaseOffsetR,
    s_modShapeL, e_modShapeL, s_modShapeR, e_modShapeR,
    s_crossModDepth, e_crossModDepth, crossModDelay,
    s_harmonicDepth, e_harmonicDepth, harmonicRatio,
    s_subHarmonicFreq, e_subHarmonicFreq, s_subHarmonicDepth, e_subHarmonicDepth,
    s_phaseOscFreq, e_phaseOscFreq, s_phaseOscRange, e_phaseOscRange,
    s_startPhaseL_init, e_startPhaseL_init, s_startPhaseR_init, e_startPhaseR_init,
    phaseOscPhaseOffset,
    beatingSidebands, sidebandOffset, sidebandDepth,
    attackTime, releaseTime
):
    if N <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    
    t_arr = np.empty(N, dtype=np.float64)
    dt = duration_float / N
    for i in numba.prange(N):
        t_arr[i] = i * dt
        alpha = alpha_arr[i] if alpha_arr.size == N else (i / (N - 1) if N > 1 else 0.0)
        alpha_arr[i] = alpha
    
    # Interpolate all transitioning parameters
    ampL_arr = np.empty(N, dtype=np.float64)
    ampR_arr = np.empty(N, dtype=np.float64)
    baseFreqL_arr = np.empty(N, dtype=np.float64)
    baseFreqR_arr = np.empty(N, dtype=np.float64)
    qamAmFreqL_arr = np.empty(N, dtype=np.float64)
    qamAmDepthL_arr = np.empty(N, dtype=np.float64)
    qamAmFreqR_arr = np.empty(N, dtype=np.float64)
    qamAmDepthR_arr = np.empty(N, dtype=np.float64)
    qamAmPhaseOffsetL_arr = np.empty(N, dtype=np.float64)
    qamAmPhaseOffsetR_arr = np.empty(N, dtype=np.float64)
    qamAm2FreqL_arr = np.empty(N, dtype=np.float64)
    qamAm2DepthL_arr = np.empty(N, dtype=np.float64)
    qamAm2FreqR_arr = np.empty(N, dtype=np.float64)
    qamAm2DepthR_arr = np.empty(N, dtype=np.float64)
    qamAm2PhaseOffsetL_arr = np.empty(N, dtype=np.float64)
    qamAm2PhaseOffsetR_arr = np.empty(N, dtype=np.float64)
    modShapeL_arr = np.empty(N, dtype=np.float64)
    modShapeR_arr = np.empty(N, dtype=np.float64)
    crossModDepth_arr = np.empty(N, dtype=np.float64)
    harmonicDepth_arr = np.empty(N, dtype=np.float64)
    subHarmonicFreq_arr = np.empty(N, dtype=np.float64)
    subHarmonicDepth_arr = np.empty(N, dtype=np.float64)
    phaseOscFreq_arr = np.empty(N, dtype=np.float64)
    phaseOscRange_arr = np.empty(N, dtype=np.float64)
    
    for i in numba.prange(N):
        alpha = alpha_arr[i]
        ampL_arr[i] = s_ampL + (e_ampL - s_ampL) * alpha
        ampR_arr[i] = s_ampR + (e_ampR - s_ampR) * alpha
        baseFreqL_arr[i] = s_baseFreqL + (e_baseFreqL - s_baseFreqL) * alpha
        baseFreqR_arr[i] = s_baseFreqR + (e_baseFreqR - s_baseFreqR) * alpha
        qamAmFreqL_arr[i] = s_qamAmFreqL + (e_qamAmFreqL - s_qamAmFreqL) * alpha
        qamAmDepthL_arr[i] = s_qamAmDepthL + (e_qamAmDepthL - s_qamAmDepthL) * alpha
        qamAmFreqR_arr[i] = s_qamAmFreqR + (e_qamAmFreqR - s_qamAmFreqR) * alpha
        qamAmDepthR_arr[i] = s_qamAmDepthR + (e_qamAmDepthR - s_qamAmDepthR) * alpha
        qamAmPhaseOffsetL_arr[i] = s_qamAmPhaseOffsetL + (e_qamAmPhaseOffsetL - s_qamAmPhaseOffsetL) * alpha
        qamAmPhaseOffsetR_arr[i] = s_qamAmPhaseOffsetR + (e_qamAmPhaseOffsetR - s_qamAmPhaseOffsetR) * alpha
        qamAm2FreqL_arr[i] = s_qamAm2FreqL + (e_qamAm2FreqL - s_qamAm2FreqL) * alpha
        qamAm2DepthL_arr[i] = s_qamAm2DepthL + (e_qamAm2DepthL - s_qamAm2DepthL) * alpha
        qamAm2FreqR_arr[i] = s_qamAm2FreqR + (e_qamAm2FreqR - s_qamAm2FreqR) * alpha
        qamAm2DepthR_arr[i] = s_qamAm2DepthR + (e_qamAm2DepthR - s_qamAm2DepthR) * alpha
        qamAm2PhaseOffsetL_arr[i] = s_qamAm2PhaseOffsetL + (e_qamAm2PhaseOffsetL - s_qamAm2PhaseOffsetL) * alpha
        qamAm2PhaseOffsetR_arr[i] = s_qamAm2PhaseOffsetR + (e_qamAm2PhaseOffsetR - s_qamAm2PhaseOffsetR) * alpha
        modShapeL_arr[i] = s_modShapeL + (e_modShapeL - s_modShapeL) * alpha
        modShapeR_arr[i] = s_modShapeR + (e_modShapeR - s_modShapeR) * alpha
        crossModDepth_arr[i] = s_crossModDepth + (e_crossModDepth - s_crossModDepth) * alpha
        harmonicDepth_arr[i] = s_harmonicDepth + (e_harmonicDepth - s_harmonicDepth) * alpha
        subHarmonicFreq_arr[i] = s_subHarmonicFreq + (e_subHarmonicFreq - s_subHarmonicFreq) * alpha
        subHarmonicDepth_arr[i] = s_subHarmonicDepth + (e_subHarmonicDepth - s_subHarmonicDepth) * alpha
        phaseOscFreq_arr[i] = s_phaseOscFreq + (e_phaseOscFreq - s_phaseOscFreq) * alpha
        phaseOscRange_arr[i] = s_phaseOscRange + (e_phaseOscRange - s_phaseOscRange) * alpha
    
    # Generate carrier phases
    phL_carrier = np.empty(N, dtype=np.float64)
    phR_carrier = np.empty(N, dtype=np.float64)
    
    currentPhaseL = s_startPhaseL_init
    currentPhaseR = s_startPhaseR_init
    for i in range(N):
        phL_carrier[i] = currentPhaseL
        phR_carrier[i] = currentPhaseR
        currentPhaseL += 2 * np.pi * baseFreqL_arr[i] * dt
        currentPhaseR += 2 * np.pi * baseFreqR_arr[i] * dt
    
    # Apply phase oscillation
    for i in numba.prange(N):
        if phaseOscFreq_arr[i] != 0.0 or phaseOscRange_arr[i] != 0.0:
            d_phi = (phaseOscRange_arr[i] / 2.0) * np.sin(2 * np.pi * phaseOscFreq_arr[i] * t_arr[i] + phaseOscPhaseOffset)
            phL_carrier[i] -= d_phi
            phR_carrier[i] += d_phi
    
    # Generate complex envelopes with transitions
    envL_arr = np.ones(N, dtype=np.float64)
    envR_arr = np.ones(N, dtype=np.float64)
    
    # Primary modulation with shape transitions
    for i in numba.prange(N):
        if qamAmFreqL_arr[i] != 0.0 and qamAmDepthL_arr[i] != 0.0:
            if modShapeL_arr[i] == 1.0:
                modL1 = np.cos(2 * np.pi * qamAmFreqL_arr[i] * t_arr[i] + qamAmPhaseOffsetL_arr[i])
            else:
                phase = 2 * np.pi * qamAmFreqL_arr[i] * t_arr[i] + qamAmPhaseOffsetL_arr[i]
                cos_val = np.cos(phase)
                modL1 = np.sign(cos_val) * np.abs(cos_val) ** (1.0 / modShapeL_arr[i])
            envL_arr[i] *= (1.0 + qamAmDepthL_arr[i] * modL1)
        
        if qamAmFreqR_arr[i] != 0.0 and qamAmDepthR_arr[i] != 0.0:
            if modShapeR_arr[i] == 1.0:
                modR1 = np.cos(2 * np.pi * qamAmFreqR_arr[i] * t_arr[i] + qamAmPhaseOffsetR_arr[i])
            else:
                phase = 2 * np.pi * qamAmFreqR_arr[i] * t_arr[i] + qamAmPhaseOffsetR_arr[i]
                cos_val = np.cos(phase)
                modR1 = np.sign(cos_val) * np.abs(cos_val) ** (1.0 / modShapeR_arr[i])
            envR_arr[i] *= (1.0 + qamAmDepthR_arr[i] * modR1)
    
    # Secondary modulation
    for i in numba.prange(N):
        if qamAm2FreqL_arr[i] != 0.0 and qamAm2DepthL_arr[i] != 0.0:
            modL2 = np.cos(2 * np.pi * qamAm2FreqL_arr[i] * t_arr[i] + qamAm2PhaseOffsetL_arr[i])
            envL_arr[i] *= (1.0 + qamAm2DepthL_arr[i] * modL2)
        
        if qamAm2FreqR_arr[i] != 0.0 and qamAm2DepthR_arr[i] != 0.0:
            modR2 = np.cos(2 * np.pi * qamAm2FreqR_arr[i] * t_arr[i] + qamAm2PhaseOffsetR_arr[i])
            envR_arr[i] *= (1.0 + qamAm2DepthR_arr[i] * modR2)
    
    # Cross-channel modulation coupling with transitions
    if crossModDelay != 0.0:
        crossEnvL = np.copy(envL_arr)
        crossEnvR = np.copy(envR_arr)
        delay_samples = int(crossModDelay * sample_rate_float)
        
        for i in numba.prange(N):
            if i >= delay_samples and crossModDepth_arr[i] != 0.0:
                envL_arr[i] *= (1.0 + crossModDepth_arr[i] * (crossEnvR[i - delay_samples] - 1.0))
                envR_arr[i] *= (1.0 + crossModDepth_arr[i] * (crossEnvL[i - delay_samples] - 1.0))
    
    # Sub-harmonic modulation with transitions
    for i in numba.prange(N):
        if subHarmonicFreq_arr[i] != 0.0 and subHarmonicDepth_arr[i] != 0.0:
            subMod = np.cos(2 * np.pi * subHarmonicFreq_arr[i] * t_arr[i])
            envL_arr[i] *= (1.0 + subHarmonicDepth_arr[i] * subMod)
            envR_arr[i] *= (1.0 + subHarmonicDepth_arr[i] * subMod)
    
    # Generate output with all features
    out = np.empty((N, 2), dtype=np.float32)
    
    for i in numba.prange(N):
        # Main carriers
        sigL = envL_arr[i] * np.cos(phL_carrier[i])
        sigR = envR_arr[i] * np.cos(phR_carrier[i])
        
        # Add harmonics with transitions
        if harmonicDepth_arr[i] != 0.0:
            sigL += harmonicDepth_arr[i] * envL_arr[i] * np.cos(harmonicRatio * phL_carrier[i])
            sigR += harmonicDepth_arr[i] * envR_arr[i] * np.cos(harmonicRatio * phR_carrier[i])
        
        # Add beating sidebands
        if beatingSidebands and sidebandDepth != 0.0:
            sigL += sidebandDepth * envL_arr[i] * np.cos(phL_carrier[i] - 2 * np.pi * sidebandOffset * t_arr[i])
            sigR += sidebandDepth * envR_arr[i] * np.cos(phR_carrier[i] - 2 * np.pi * sidebandOffset * t_arr[i])
            sigL += sidebandDepth * envL_arr[i] * np.cos(phL_carrier[i] + 2 * np.pi * sidebandOffset * t_arr[i])
            sigR += sidebandDepth * envR_arr[i] * np.cos(phR_carrier[i] + 2 * np.pi * sidebandOffset * t_arr[i])
        
        # Apply envelope shaping
        envelope_mult = 1.0
        if attackTime > 0.0 and t_arr[i] < attackTime:
            envelope_mult *= t_arr[i] / attackTime
        if releaseTime > 0.0 and t_arr[i] > (duration_float - releaseTime):
            envelope_mult *= (duration_float - t_arr[i]) / releaseTime
        
        out[i, 0] = np.float32(sigL * ampL_arr[i] * envelope_mult)
        out[i, 1] = np.float32(sigR * ampR_arr[i] * envelope_mult)
    
    return out
