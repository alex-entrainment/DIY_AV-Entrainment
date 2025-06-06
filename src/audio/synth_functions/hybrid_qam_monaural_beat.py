"""
Hybrid QAM-Monaural Beat synthesis functions.

Generates a hybrid signal:
Left Channel: QAM-style modulated signal.
Right Channel: Monaural beat with its own AM, FM, and phase oscillation.
"""

import numpy as np
import numba
from .common import apply_filters, calculate_transition_alpha


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


def hybrid_qam_monaural_beat_transition(duration, sample_rate=44100, initial_offset=0.0, post_offset=0.0, **params):
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

    alpha_arr = calculate_transition_alpha(duration, sample_rate, initial_offset, post_offset)

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
        s_mono_R_phaseOscPhaseOffset, e_mono_R_phaseOscPhaseOffset,
        alpha_arr
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
    s_mono_R_phaseOscPhaseOffset, e_mono_R_phaseOscPhaseOffset,
    alpha_arr
):
    if N <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    t_arr = np.empty(N, dtype=np.float64)
    dt = duration_float / N
    for i in numba.prange(N):
        t_arr[i] = i * dt
        alpha = alpha_arr[i] if alpha_arr.size == N else (i / (N - 1) if N > 1 else 0.0)
        alpha_arr[i] = alpha

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
