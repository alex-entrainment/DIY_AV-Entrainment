"""
QAM Beat synthesis functions.

Generates QAM-based binaural beats where each channel's carrier is amplitude modulated.
Optional phase oscillation can be applied between channels.
"""

import numpy as np
import numba
from .common import apply_filters


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
