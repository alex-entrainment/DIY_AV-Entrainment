"""Monaural beat with stereo amplitude control synthesis functions."""

import numpy as np
import numba
from .common import calculate_transition_alpha


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


def monaural_beat_stereo_amps_transition(duration, sample_rate=44100, initial_offset=0.0, post_offset=0.0, **params):
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
    alpha_arr = calculate_transition_alpha(duration, sample_rate, initial_offset, post_offset)
    return _monaural_beat_stereo_amps_transition_core(
        N, float(duration), float(sample_rate),
        s_ll, e_ll, s_ul, e_ul, s_lr, e_lr, s_ur, e_ur,
        sBF, eBF, sBt, eBt,
        sStartPhaseL, eStartPhaseL, sStartPhaseU, eStartPhaseU,
        sPhiF, ePhiF, sPhiR, ePhiR,
        sAOD, eAOD, sAOF, eAOF, sAOP, eAOP,
        alpha_arr
    )


@numba.njit(parallel=True, fastmath=True)
def _monaural_beat_stereo_amps_transition_core(
    N, duration, sample_rate,
    s_ll, e_ll, s_ul, e_ul, s_lr, e_lr, s_ur, e_ur, # Amplitudes
    sBF, eBF, sBt, eBt,                             # Frequencies
    sSPL, eSPL, sSPU, eSPU,                         # Start Phases (lower, upper)
    sPhiF, ePhiF, sPhiR, ePhiR,                     # Phase Osc
    sAOD, eAOD, sAOF, eAOF, sAOP, eAOP,              # Amp Osc
    alpha_arr
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
        alpha = alpha_arr[i] if alpha_arr.size == N else (i / (N - 1) if N > 1 else 0.0)
        
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
