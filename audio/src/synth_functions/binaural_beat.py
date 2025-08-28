"""Binaural beat synthesis functions."""

import numpy as np
import numba
from .common import skewed_sine_phase, skewed_triangle_phase, _frac
from .spatial_ambi2d import spatialize_binaural_mid_only, generate_azimuth_trajectory


def binaural_beat(duration, sample_rate=44100, **params):
    # --- Unpack synthesis parameters ---
    ampL = float(params.get('ampL', 0.5))
    ampR = float(params.get('ampR', 0.5))
    baseF = float(params.get('baseFreq', 200.0))
    beatF = float(params.get('beatFreq', 4.0))
    leftHigh = bool(params.get('leftHigh', False))
    startL = float(params.get('startPhaseL', 0.0)) # in radians
    startR = float(params.get('startPhaseR', 0.0)) # in radians
    aODL = float(params.get('ampOscDepthL', 0.0))
    aOFL = float(params.get('ampOscFreqL', 0.0))
    aODR = float(params.get('ampOscDepthR', 0.0))
    aOFR = float(params.get('ampOscFreqR', 0.0))
    fORL = float(params.get('freqOscRangeL', 0.0))
    fOFL = float(params.get('freqOscFreqL', 0.0))
    fORR = float(params.get('freqOscRangeR', 0.0))
    fOFR = float(params.get('freqOscFreqR', 0.0))
    freqOscSkewL = float(params.get('freqOscSkewL', 0.0))
    freqOscSkewR = float(params.get('freqOscSkewR', 0.0))
    freqOscPhaseOffsetL = float(params.get('freqOscPhaseOffsetL', 0.0))
    freqOscPhaseOffsetR = float(params.get('freqOscPhaseOffsetR', 0.0))
    ampOscPhaseOffsetL = float(params.get('ampOscPhaseOffsetL', 0.0))
    ampOscPhaseOffsetR = float(params.get('ampOscPhaseOffsetR', 0.0))
    ampOscSkewL = float(params.get('ampOscSkewL', 0.0))
    ampOscSkewR = float(params.get('ampOscSkewR', 0.0))
    freqOscShape = str(params.get('freqOscShape', 'sine')).lower()
    shape_int = 1 if freqOscShape == 'triangle' else 0
    pOF = float(params.get('phaseOscFreq', 0.0))
    pOR = float(params.get('phaseOscRange', 0.0)) # in radians

    # --- UNCOMMENT IF "GLITCH/CHIRP" EFFECT NEEDED OR DESIRED
    # --- Unpack glitch parameters --- 
    # glitchInterval   = float(params.get('glitchInterval',   0.0))
    # glitchDur        = float(params.get('glitchDur',        0.0))
    # glitchNoiseLevel = float(params.get('glitchNoiseLevel', 0.0))
    # glitchFocusWidth = float(params.get('glitchFocusWidth', 0.0))
    # glitchFocusExp   = float(params.get('glitchFocusExp',   0.0))

    N = int(duration * sample_rate)
    #
    # # --- Precompute glitch bursts in NumPy ---
    # positions = []
    # bursts = []
    # if glitchInterval > 0 and glitchDur > 0 and glitchNoiseLevel > 0 and N > 0:
    #     full_n = int(glitchDur * sample_rate)
    #     if full_n > 0:
    #         repeats = int(duration / glitchInterval)
    #         for k in range(1, repeats + 1):
    #             t_end   = k * glitchInterval
    #             t_start = max(0.0, t_end - glitchDur)
    #             i0 = int(t_start * sample_rate)
    #             i1 = i0 + full_n
    #             if i1 > N:
    #                 continue
    #
    #             white = np.random.standard_normal(full_n)
    #             S = np.fft.rfft(white)
    #             freqs_denominator = full_n if full_n != 0 else 1
    #             freqs = np.arange(S.size) * (sample_rate / freqs_denominator)
    #
    #             if glitchFocusWidth == 0:
    #                 gauss = np.ones_like(freqs)
    #             else:
    #                 gauss = np.exp(-0.5 * ((freqs - baseF) / glitchFocusWidth) ** glitchFocusExp)
    #
    #             shaped = np.fft.irfft(S * gauss, n=full_n)
    #             max_abs_shaped = np.max(np.abs(shaped))
    #             if max_abs_shaped < 1e-16:
    #                 shaped_normalized = np.zeros_like(shaped)
    #             else:
    #                 shaped_normalized = shaped / max_abs_shaped
    #             ramp = np.linspace(0.0, 1.0, full_n, endpoint=True)
    #             burst_samples = (shaped_normalized * ramp * glitchNoiseLevel).astype(np.float32)
    #
    #             positions.append(i0)
    #             bursts.append(burst_samples)
    #
    # if bursts:
    #     pos_arr   = np.array(positions, dtype=np.int32)
    #     # Ensure all burst_samples have the same length if concatenating or handle individually
    #     # For simplicity, assuming all burst_samples are concatenated correctly if generated
    #     if any(b.ndim == 0 or b.size == 0 for b in bursts): # check for empty or scalar arrays
    #         burst_arr = np.empty(0, dtype=np.float32)
    #         pos_arr = np.empty(0, dtype=np.int32) # Clear positions if bursts are problematic
    #     else:
    #         try:
    #             burst_arr = np.concatenate(bursts)
    #         except ValueError: # Handle cases where concatenation might fail (e.g. varying dimensions beyond the first)
    #              # Fallback: if concatenation fails, create empty burst array or handle error appropriately
    #             burst_arr = np.empty(0, dtype=np.float32)
    #             pos_arr = np.empty(0, dtype=np.int32)
    #

    pos_arr   = np.empty(0, dtype=np.int32)
    burst_arr = np.empty(0, dtype=np.float32)

    audio = _binaural_beat_core(
        N,
        float(duration),
        float(sample_rate),
        leftHigh,
        ampL, ampR, baseF, beatF,
        startL, startR, pOF, pOR,
        aODL, aOFL, aODR, aOFR,
        ampOscPhaseOffsetL, ampOscPhaseOffsetR,
        ampOscSkewL, ampOscSkewR,
        fORL, fOFL, fORR, fOFR,
        freqOscSkewL, freqOscSkewR,
        freqOscPhaseOffsetL, freqOscPhaseOffsetR,
        shape_int,
        pos_arr, burst_arr
    )

    if bool(params.get("spatialEnable", False)):
        theta_deg, distance_m = generate_azimuth_trajectory(
            duration, sample_rate,
            segments=params.get(
                "spatialTrajectory",
                [{
                    "mode": "oscillate",
                    "center_deg": 0,
                    "extent_deg": 75,
                    "period_s": 20.0,
                    "distance_m": [1.0, 1.4],
                    "seconds": duration,
                }],
            ),
        )
        audio = spatialize_binaural_mid_only(
            audio.astype(np.float32), float(sample_rate),
            theta_deg, distance_m,
            ild_enable=int(params.get("spatialUseIld", 1)),
            ear_angle_deg=float(params.get("spatialEarAngleDeg", 30.0)),
            head_radius_m=float(params.get("spatialHeadRadiusM", 0.0875)),
            itd_scale=float(params.get("spatialItdScale", 1.0)),
            ild_max_db=float(params.get("spatialIldMaxDb", 1.5)),
            ild_xover_hz=float(params.get("spatialIldXoverHz", 700.0)),
            ref_distance_m=float(params.get("spatialRefDistanceM", 1.0)),
            rolloff=float(params.get("spatialRolloff", 1.0)),
            hf_roll_db_per_m=float(params.get("spatialHfRollDbPerM", 0.0)),
            dz_theta_ms=float(params.get("spatialDezipperThetaMs", 60.0)),
            dz_dist_ms=float(params.get("spatialDezipperDistMs", 80.0)),
            decoder=0 if str(params.get("spatialDecoder", "itd_head")).lower() != "foa_cardioid" else 1,
            min_distance_m=float(params.get("spatialMinDistanceM", 0.1)),
            max_deg_per_s=float(params.get("spatialMaxDegPerS", 90.0)),
            max_delay_step_samples=float(params.get("spatialMaxDelayStepSamples", 0.02)),
            interp_mode=int(params.get("spatialInterp", 1)),
        )

    return audio


@numba.njit(parallel=True, fastmath=True)
def _binaural_beat_core(
    N, duration, sample_rate,
    leftHigh,
    ampL, ampR, baseF, beatF,
    startL, startR, pOF, pOR,
    aODL, aOFL, aODR, aOFR,
    ampOscPhaseOffsetL, ampOscPhaseOffsetR,
    ampOscSkewL, ampOscSkewR,
    fORL, fOFL, fORR, fOFR,
    freqOscSkewL, freqOscSkewR,
    freqOscPhaseOffsetL, freqOscPhaseOffsetR,
    freqOscShape,
    pos,   # int32[:] start indices
    burst  # float32[:] concatenated glitch samples
):
    if N <= 0 :
        return np.zeros((0,2), dtype=np.float32)
        
    t = np.empty(N, dtype=np.float64)
    dt = duration / N if N > 0 else 0.0
    for i in numba.prange(N): # Use prange for parallel
        t[i] = i * dt

    halfB = beatF / 2.0
    if leftHigh:
        fL_base = baseF + halfB
        fR_base = baseF - halfB
    else:
        fL_base = baseF - halfB
        fR_base = baseF + halfB
    instL = np.empty(N, dtype=np.float64)
    instR = np.empty(N, dtype=np.float64)
    for i in numba.prange(N): # Use prange
        phaseL = fOFL * t[i] + freqOscPhaseOffsetL/(2*np.pi)
        phaseR = fOFR * t[i] + freqOscPhaseOffsetR/(2*np.pi)
        if freqOscShape == 1:
            vibL = (fORL/2.0) * skewed_triangle_phase(_frac(phaseL), freqOscSkewL)
            vibR = (fORR/2.0) * skewed_triangle_phase(_frac(phaseR), freqOscSkewR)
        else:
            vibL = (fORL/2.0) * skewed_sine_phase(_frac(phaseL), freqOscSkewL)
            vibR = (fORR/2.0) * skewed_sine_phase(_frac(phaseR), freqOscSkewR)
        instL[i] = max(0.0, fL_base + vibL)
        instR[i] = max(0.0, fR_base + vibR)
    
    # If beat frequency is zero we still want any frequency oscillations to
    # apply.  The original code overwrote the instantaneous frequency with the
    # base value, ignoring modulation.  Dropping that behaviour keeps both
    # channels identical while respecting `freqOsc*` parameters.

    phL = np.empty(N, dtype=np.float64)
    phR = np.empty(N, dtype=np.float64)
    curL = startL
    curR = startR
    for i in range(N): # Sequential loop
        curL += 2 * np.pi * instL[i] * dt
        curR += 2 * np.pi * instR[i] * dt
        phL[i] = curL
        phR[i] = curR

    if pOF != 0.0 or pOR != 0.0:
        for i in numba.prange(N): # Use prange
            dphi = (pOR/2.0) * np.sin(2*np.pi*pOF*t[i])
            phL[i] -= dphi
            phR[i] += dphi

    envL = np.empty(N, dtype=np.float64)
    envR = np.empty(N, dtype=np.float64)
    for i in numba.prange(N): # Use prange
        phaseL = aOFL*t[i] + ampOscPhaseOffsetL/(2*np.pi)
        phaseR = aOFR*t[i] + ampOscPhaseOffsetR/(2*np.pi)
        envL[i] = 1.0 - aODL * (0.5*(1.0 + skewed_sine_phase(_frac(phaseL), ampOscSkewL)))
        envR[i] = 1.0 - aODR * (0.5*(1.0 + skewed_sine_phase(_frac(phaseR), ampOscSkewR)))

    out = np.empty((N,2), dtype=np.float32)
    for i in numba.prange(N): # Use prange
        out[i,0] = np.float32(np.sin(phL[i]) * envL[i] * ampL)
        out[i,1] = np.float32(np.sin(phR[i]) * envR[i] * ampR)

    num_bursts = pos.shape[0]
    if num_bursts > 0 and burst.size > 0: # ensure burst is not empty
        # Ensure burst.size is divisible by num_bursts.
        # This check implies burst is not empty and pos is not empty.
        if burst.size % num_bursts == 0 :
            L = burst.size // num_bursts
            if L > 0: # Ensure segment length L is positive
                idx = 0
                # This loop should be sequential if bursts can overlap or write to the same output indices.
                # If pos guarantees no overlap, then outer loop b can be prange.
                # Assuming pos are sorted and bursts don't overlap for safety in parallel context.
                # However, a sequential loop is safer unless overlap is explicitly managed.
                # For now, using sequential for the outer loop over bursts.
                for b in range(num_bursts): # Changed from prange to range for safety with additions
                    start_idx = pos[b]
                    current_burst_segment = burst[idx : idx + L]
                    for j in range(L):
                        p = start_idx + j
                        if p < N: # Boundary check
                            # Numba does not automatically make += atomic in all contexts.
                            # For safety, this part is tricky with prange on `b` if `pos` allows overlaps.
                            out[p,0] += current_burst_segment[j]
                            out[p,1] += current_burst_segment[j]
                    idx += L
    return out


from .common import calculate_transition_alpha


def binaural_beat_transition(duration, sample_rate=44100, initial_offset=0.0, post_offset=0.0, **params):
    # --- Unpack start/end parameters ---
    startAmpL = float(params.get('startAmpL', params.get('ampL', 0.5)))
    endAmpL = float(params.get('endAmpL', startAmpL))
    startAmpR = float(params.get('startAmpR', params.get('ampR', 0.5)))
    endAmpR = float(params.get('endAmpR', startAmpR))
    startBaseF = float(params.get('startBaseFreq', params.get('baseFreq', 200.0)))
    endBaseF = float(params.get('endBaseFreq', startBaseF))
    startBeatF = float(params.get('startBeatFreq', params.get('beatFreq', 4.0)))
    endBeatF = float(params.get('endBeatFreq', startBeatF))
    leftHigh = bool(params.get('leftHigh', False))

    startStartPhaseL = float(params.get('startStartPhaseL', params.get('startPhaseL', 0.0)))
    endStartPhaseL = float(params.get('endStartPhaseL', startStartPhaseL))
    startStartPhaseR = float(params.get('startStartPhaseR', params.get('startPhaseR', 0.0)))
    endStartPhaseR = float(params.get('endStartPhaseR', startStartPhaseR))
    
    startPOF = float(params.get('startPhaseOscFreq', params.get('phaseOscFreq', 0.0)))
    endPOF = float(params.get('endPhaseOscFreq', startPOF))
    startPOR = float(params.get('startPhaseOscRange', params.get('phaseOscRange', 0.0)))
    endPOR = float(params.get('endPhaseOscRange', startPOR))

    startAODL = float(params.get('startAmpOscDepthL', params.get('ampOscDepthL', 0.0)))
    endAODL = float(params.get('endAmpOscDepthL', startAODL))
    startAOFL = float(params.get('startAmpOscFreqL', params.get('ampOscFreqL', 0.0)))
    endAOFL = float(params.get('endAmpOscFreqL', startAOFL))
    startAODR = float(params.get('startAmpOscDepthR', params.get('ampOscDepthR', 0.0)))
    endAODR = float(params.get('endAmpOscDepthR', startAODR))
    startAOFR = float(params.get('startAmpOscFreqR', params.get('ampOscFreqR', 0.0)))
    endAOFR = float(params.get('endAmpOscFreqR', startAOFR))
    startAmpOscSkewL = float(params.get('startAmpOscSkewL', params.get('ampOscSkewL', 0.0)))
    endAmpOscSkewL = float(params.get('endAmpOscSkewL', startAmpOscSkewL))
    startAmpOscSkewR = float(params.get('startAmpOscSkewR', params.get('ampOscSkewR', 0.0)))
    endAmpOscSkewR = float(params.get('endAmpOscSkewR', startAmpOscSkewR))

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
    startFreqOscSkewL = float(params.get('startFreqOscSkewL', params.get('freqOscSkewL', 0.0)))
    endFreqOscSkewL = float(params.get('endFreqOscSkewL', startFreqOscSkewL))
    startFreqOscSkewR = float(params.get('startFreqOscSkewR', params.get('freqOscSkewR', 0.0)))
    endFreqOscSkewR = float(params.get('endFreqOscSkewR', startFreqOscSkewR))
    startFreqOscPhaseOffsetL = float(params.get('startFreqOscPhaseOffsetL', params.get('freqOscPhaseOffsetL', 0.0)))
    endFreqOscPhaseOffsetL = float(params.get('endFreqOscPhaseOffsetL', startFreqOscPhaseOffsetL))
    startFreqOscPhaseOffsetR = float(params.get('startFreqOscPhaseOffsetR', params.get('freqOscPhaseOffsetR', 0.0)))
    endFreqOscPhaseOffsetR = float(params.get('endFreqOscPhaseOffsetR', startFreqOscPhaseOffsetR))

    freqOscShape = str(params.get('freqOscShape', 'sine')).lower()
    shape_int = 1 if freqOscShape == 'triangle' else 0

    # Glitch parameters (using average for pre-computation)
    s_glitchInterval = float(params.get('startGlitchInterval', params.get('glitchInterval', 0.0)))
    e_glitchInterval = float(params.get('endGlitchInterval', s_glitchInterval))
    avg_glitchInterval = (s_glitchInterval + e_glitchInterval) / 2.0

    s_glitchDur = float(params.get('startGlitchDur', params.get('glitchDur', 0.0)))
    e_glitchDur = float(params.get('endGlitchDur', s_glitchDur))
    avg_glitchDur = (s_glitchDur + e_glitchDur) / 2.0
    
    s_glitchNoiseLevel = float(params.get('startGlitchNoiseLevel', params.get('glitchNoiseLevel', 0.0)))
    e_glitchNoiseLevel = float(params.get('endGlitchNoiseLevel', s_glitchNoiseLevel))
    avg_glitchNoiseLevel = (s_glitchNoiseLevel + e_glitchNoiseLevel) / 2.0

    s_glitchFocusWidth = float(params.get('startGlitchFocusWidth', params.get('glitchFocusWidth', 0.0)))
    e_glitchFocusWidth = float(params.get('endGlitchFocusWidth', s_glitchFocusWidth))
    avg_glitchFocusWidth = (s_glitchFocusWidth + e_glitchFocusWidth) / 2.0
    
    s_glitchFocusExp = float(params.get('startGlitchFocusExp', params.get('glitchFocusExp', 0.0)))
    e_glitchFocusExp = float(params.get('endGlitchFocusExp', s_glitchFocusExp))
    avg_glitchFocusExp = (s_glitchFocusExp + e_glitchFocusExp) / 2.0

    N = int(duration * sample_rate)
    
    # --- Precompute glitch bursts using average parameters ---
    positions = []
    bursts = []
    # Use the average base frequency for glitch shaping if it's also transitional
    # For simplicity, using startBaseF here, or could use (startBaseF + endBaseF) / 2
    glitch_shaping_base_freq = (startBaseF + endBaseF) / 2.0

    if avg_glitchInterval > 0 and avg_glitchDur > 0 and avg_glitchNoiseLevel > 0 and N > 0:
        full_n = int(avg_glitchDur * sample_rate)
        if full_n > 0:
            repeats = int(duration / avg_glitchInterval) if avg_glitchInterval > 0 else 0
            for k in range(1, repeats + 1):
                t_end   = k * avg_glitchInterval
                t_start = max(0.0, t_end - avg_glitchDur)
                i0 = int(t_start * sample_rate)
                i1 = i0 + full_n
                if i1 > N:
                    continue

                white = np.random.standard_normal(full_n)
                S = np.fft.rfft(white)
                freqs_denominator = full_n if full_n != 0 else 1
                freqs = np.arange(S.size) * (sample_rate / freqs_denominator)

                if avg_glitchFocusWidth == 0:
                    gauss = np.ones_like(freqs)
                else:
                    gauss = np.exp(-0.5 * ((freqs - glitch_shaping_base_freq) / avg_glitchFocusWidth) ** avg_glitchFocusExp)
                
                shaped = np.fft.irfft(S * gauss, n=full_n)
                max_abs_shaped = np.max(np.abs(shaped))
                if max_abs_shaped < 1e-16:
                    shaped_normalized = np.zeros_like(shaped)
                else:
                    shaped_normalized = shaped / max_abs_shaped
                ramp = np.linspace(0.0, 1.0, full_n, endpoint=True)
                burst_samples = (shaped_normalized * ramp * avg_glitchNoiseLevel).astype(np.float32)
                
                positions.append(i0)
                bursts.append(burst_samples)
    
    if bursts:
        pos_arr   = np.array(positions, dtype=np.int32)
        if any(b.ndim == 0 or b.size == 0 for b in bursts):
            burst_arr = np.empty(0, dtype=np.float32)
            pos_arr = np.empty(0, dtype=np.int32)
        else:
            try:
                burst_arr = np.concatenate(bursts)
            except ValueError:
                burst_arr = np.empty(0, dtype=np.float32)
                pos_arr = np.empty(0, dtype=np.int32)
    else:
        pos_arr   = np.empty(0, dtype=np.int32)
        burst_arr = np.empty(0, dtype=np.float32)

    curve = params.get('transition_curve', 'linear')
    alpha_arr = calculate_transition_alpha(duration, sample_rate, initial_offset, post_offset, curve)

    audio = _binaural_beat_transition_core(
        N, float(duration), float(sample_rate),
        startAmpL, endAmpL, startAmpR, endAmpR,
        startBaseF, endBaseF, startBeatF, endBeatF, leftHigh,
        startStartPhaseL, endStartPhaseL, startStartPhaseR, endStartPhaseR,
        startPOF, endPOF, startPOR, endPOR,
        startAODL, endAODL, startAOFL, endAOFL,
        startAODR, endAODR, startAOFR, endAOFR,
        startAmpOscPhaseOffsetL, endAmpOscPhaseOffsetL,
        startAmpOscPhaseOffsetR, endAmpOscPhaseOffsetR,
        startAmpOscSkewL, endAmpOscSkewL,
        startAmpOscSkewR, endAmpOscSkewR,
        startFORL, endFORL, startFOFL, endFOFL,
        startFORR, endFORR, startFOFR, endFOFR,
        startFreqOscSkewL, endFreqOscSkewL,
        startFreqOscSkewR, endFreqOscSkewR,
        startFreqOscPhaseOffsetL, endFreqOscPhaseOffsetL,
        startFreqOscPhaseOffsetR, endFreqOscPhaseOffsetR,
        shape_int,
        pos_arr, burst_arr, # Pass pre-calculated glitches
        alpha_arr
    )

    if bool(params.get("spatialEnable", False)):
        theta_deg, distance_m = generate_azimuth_trajectory(
            duration, sample_rate,
            segments=params.get(
                "spatialTrajectory",
                [{
                    "mode": "oscillate",
                    "center_deg": 0,
                    "extent_deg": 75,
                    "period_s": 20.0,
                    "distance_m": [1.0, 1.4],
                    "seconds": duration,
                }],
            ),
        )
        audio = spatialize_binaural_mid_only(
            audio.astype(np.float32), float(sample_rate),
            theta_deg, distance_m,
            ild_enable=int(params.get("spatialUseIld", 1)),
            ear_angle_deg=float(params.get("spatialEarAngleDeg", 30.0)),
            head_radius_m=float(params.get("spatialHeadRadiusM", 0.0875)),
            itd_scale=float(params.get("spatialItdScale", 1.0)),
            ild_max_db=float(params.get("spatialIldMaxDb", 1.5)),
            ild_xover_hz=float(params.get("spatialIldXoverHz", 700.0)),
            ref_distance_m=float(params.get("spatialRefDistanceM", 1.0)),
            rolloff=float(params.get("spatialRolloff", 1.0)),
            hf_roll_db_per_m=float(params.get("spatialHfRollDbPerM", 0.0)),
            dz_theta_ms=float(params.get("spatialDezipperThetaMs", 60.0)),
            dz_dist_ms=float(params.get("spatialDezipperDistMs", 80.0)),
            decoder=0 if str(params.get("spatialDecoder", "itd_head")).lower() != "foa_cardioid" else 1,
            min_distance_m=float(params.get("spatialMinDistanceM", 0.1)),
            max_deg_per_s=float(params.get("spatialMaxDegPerS", 90.0)),
            max_delay_step_samples=float(params.get("spatialMaxDelayStepSamples", 0.02)),
            interp_mode=int(params.get("spatialInterp", 1)),
        )

    return audio


@numba.njit(parallel=True, fastmath=True)
def _binaural_beat_transition_core(
    N, duration, sample_rate,
    startAmpL, endAmpL, startAmpR, endAmpR,
    startBaseF, endBaseF, startBeatF, endBeatF, leftHigh,
    startStartPhaseL, endStartPhaseL, startStartPhaseR, endStartPhaseR,
    startPOF, endPOF, startPOR, endPOR,
    startAODL, endAODL, startAOFL, endAOFL,
    startAODR, endAODR, startAOFR, endAOFR,
    startAmpOscPhaseOffsetL, endAmpOscPhaseOffsetL,
    startAmpOscPhaseOffsetR, endAmpOscPhaseOffsetR,
    startAmpOscSkewL, endAmpOscSkewL,
    startAmpOscSkewR, endAmpOscSkewR,
    startFORL, endFORL, startFOFL, endFOFL,
    startFORR, endFORR, startFOFR, endFOFR,
    startFreqOscSkewL, endFreqOscSkewL,
    startFreqOscSkewR, endFreqOscSkewR,
    startFreqOscPhaseOffsetL, endFreqOscPhaseOffsetL,
    startFreqOscPhaseOffsetR, endFreqOscPhaseOffsetR,
    freqOscShape,
    pos, burst, # Glitch arrays (static for this core run)
    alpha_arr
):
    if N <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    dt = duration / N
    
    t_arr = np.empty(N, np.float64)
    ampL_arr = np.empty(N, np.float64)
    ampR_arr = np.empty(N, np.float64)
    baseF_arr = np.empty(N, np.float64)
    beatF_arr = np.empty(N, np.float64)

    pOF_arr = np.empty(N, np.float64)
    pOR_arr = np.empty(N, np.float64)
    aODL_arr = np.empty(N, np.float64)
    aOFL_arr = np.empty(N, np.float64)
    aODR_arr = np.empty(N, np.float64)
    aOFR_arr = np.empty(N, np.float64)
    ampOscPhaseOffsetL_arr = np.empty(N, np.float64)
    ampOscPhaseOffsetR_arr = np.empty(N, np.float64)
    ampOscSkewL_arr = np.empty(N, np.float64)
    ampOscSkewR_arr = np.empty(N, np.float64)
    fORL_arr = np.empty(N, np.float64)
    fOFL_arr = np.empty(N, np.float64)
    fORR_arr = np.empty(N, np.float64)
    fOFR_arr = np.empty(N, np.float64)
    freqOscSkewL_arr = np.empty(N, np.float64)
    freqOscSkewR_arr = np.empty(N, np.float64)
    freqOscPhaseOffsetL_arr = np.empty(N, np.float64)
    freqOscPhaseOffsetR_arr = np.empty(N, np.float64)
    
    instL = np.empty(N, np.float64)
    instR = np.empty(N, np.float64)

    # Linear interpolation of parameters
    for i in numba.prange(N):
        alpha = alpha_arr[i] if alpha_arr.size == N else (i / (N - 1) if N > 1 else 0.0)
        t_arr[i] = i * dt
        
        ampL_arr[i] = startAmpL + (endAmpL - startAmpL) * alpha
        ampR_arr[i] = startAmpR + (endAmpR - startAmpR) * alpha
        baseF_arr[i] = startBaseF + (endBaseF - startBaseF) * alpha
        beatF_arr[i] = startBeatF + (endBeatF - startBeatF) * alpha

        pOF_arr[i] = startPOF + (endPOF - startPOF) * alpha
        pOR_arr[i] = startPOR + (endPOR - startPOR) * alpha
        aODL_arr[i] = startAODL + (endAODL - startAODL) * alpha
        aOFL_arr[i] = startAOFL + (endAOFL - startAOFL) * alpha
        aODR_arr[i] = startAODR + (endAODR - startAODR) * alpha
        aOFR_arr[i] = startAOFR + (endAOFR - startAOFR) * alpha
        ampOscPhaseOffsetL_arr[i] = startAmpOscPhaseOffsetL + (endAmpOscPhaseOffsetL - startAmpOscPhaseOffsetL) * alpha
        ampOscPhaseOffsetR_arr[i] = startAmpOscPhaseOffsetR + (endAmpOscPhaseOffsetR - startAmpOscPhaseOffsetR) * alpha
        ampOscSkewL_arr[i] = startAmpOscSkewL + (endAmpOscSkewL - startAmpOscSkewL) * alpha
        ampOscSkewR_arr[i] = startAmpOscSkewR + (endAmpOscSkewR - startAmpOscSkewR) * alpha
        fORL_arr[i] = startFORL + (endFORL - startFORL) * alpha
        fOFL_arr[i] = startFOFL + (endFOFL - startFOFL) * alpha
        fORR_arr[i] = startFORR + (endFORR - startFORR) * alpha
        fOFR_arr[i] = startFOFR + (endFOFR - startFOFR) * alpha
        freqOscSkewL_arr[i] = startFreqOscSkewL + (endFreqOscSkewL - startFreqOscSkewL) * alpha
        freqOscSkewR_arr[i] = startFreqOscSkewR + (endFreqOscSkewR - startFreqOscSkewR) * alpha
        freqOscPhaseOffsetL_arr[i] = startFreqOscPhaseOffsetL + (endFreqOscPhaseOffsetL - startFreqOscPhaseOffsetL) * alpha
        freqOscPhaseOffsetR_arr[i] = startFreqOscPhaseOffsetR + (endFreqOscPhaseOffsetR - startFreqOscPhaseOffsetR) * alpha

    # Instantaneous frequencies with proper integration of LFO phase
    phaseL_fo = 0.0
    phaseR_fo = 0.0
    for i in range(N):
        halfB_i = beatF_arr[i] * 0.5
        if leftHigh:
            fL_base_i = baseF_arr[i] + halfB_i
            fR_base_i = baseF_arr[i] - halfB_i
        else:
            fL_base_i = baseF_arr[i] - halfB_i
            fR_base_i = baseF_arr[i] + halfB_i

        phaseL_fo += fOFL_arr[i] * dt
        phaseR_fo += fOFR_arr[i] * dt
        phL_frac = _frac(phaseL_fo + freqOscPhaseOffsetL_arr[i] / (2 * np.pi))
        phR_frac = _frac(phaseR_fo + freqOscPhaseOffsetR_arr[i] / (2 * np.pi))
        if freqOscShape == 1:
            vibL_i = (fORL_arr[i] / 2.0) * skewed_triangle_phase(phL_frac, freqOscSkewL_arr[i])
            vibR_i = (fORR_arr[i] / 2.0) * skewed_triangle_phase(phR_frac, freqOscSkewR_arr[i])
        else:
            vibL_i = (fORL_arr[i] / 2.0) * skewed_sine_phase(phL_frac, freqOscSkewL_arr[i])
            vibR_i = (fORR_arr[i] / 2.0) * skewed_sine_phase(phR_frac, freqOscSkewR_arr[i])

        instL_candidate = fL_base_i + vibL_i
        instR_candidate = fR_base_i + vibR_i

        instL[i] = instL_candidate if instL_candidate > 0.0 else 0.0
        instR[i] = instR_candidate if instR_candidate > 0.0 else 0.0

    # Previously the instantaneous frequencies were forced to the base
    # value whenever `beatFreq` was zero.  This prevented any frequency
    # oscillation from being applied when using a 0 Hz beat to create a
    # monaural tone.  Removing that check allows the `freqOsc*` parameters
    # to modulate the base frequency in both channels equally.

    # Phase accumulation (sequential)
    phL = np.empty(N, np.float64)
    phR = np.empty(N, np.float64)
    # Interpolate start phases (initial phase for the accumulation)
    # For phase, the start/end is for the *initial* phase value, not a rate of change of start phase.
    curL = startStartPhaseL # Use the start of the transition for the initial phase value
    curR = startStartPhaseR # Use the start of the transition for the initial phase value

    current_start_phase_L = startStartPhaseL
    current_start_phase_R = startStartPhaseR
    curL = startStartPhaseL
    curR = startStartPhaseR

    for i in range(N): # Sequential loop
        curL += 2.0 * np.pi * instL[i] * dt
        curR += 2.0 * np.pi * instR[i] * dt
        phL[i] = curL
        phR[i] = curR

    # Phase modulation
    for i in numba.prange(N): # Parallel (as it's per sample based on already calculated t_arr and pOF/pOR_arr)
        if pOF_arr[i] != 0.0 or pOR_arr[i] != 0.0:
            dphi = (pOR_arr[i]/2.0) * np.sin(2*np.pi*pOF_arr[i]*t_arr[i])
            phL[i] -= dphi
            phR[i] += dphi
            
    # Amplitude envelopes
    envL = np.empty(N, np.float64)
    envR = np.empty(N, np.float64)
    for i in numba.prange(N): # Parallel
        phL_env = aOFL_arr[i]*t_arr[i] + ampOscPhaseOffsetL_arr[i]/(2*np.pi)
        phR_env = aOFR_arr[i]*t_arr[i] + ampOscPhaseOffsetR_arr[i]/(2*np.pi)
        envL[i] = 1.0 - aODL_arr[i] * (0.5*(1.0 + skewed_sine_phase(_frac(phL_env), ampOscSkewL_arr[i])))
        envR[i] = 1.0 - aODR_arr[i] * (0.5*(1.0 + skewed_sine_phase(_frac(phR_env), ampOscSkewR_arr[i])))

    # Generate output
    out = np.empty((N, 2), dtype=np.float32)
    for i in numba.prange(N): # Parallel
        out[i, 0] = np.float32(np.sin(phL[i]) * envL[i] * ampL_arr[i])
        out[i, 1] = np.float32(np.sin(phR[i]) * envR[i] * ampR_arr[i])

    # Add glitch bursts (using the static pos and burst arrays)
    num_bursts = pos.shape[0]
    if num_bursts > 0 and burst.size > 0:
        if burst.size % num_bursts == 0:
            L_glitch = burst.size // num_bursts
            if L_glitch > 0:
                idx = 0
                for b in range(num_bursts): # Sequential for safety
                    start_idx = pos[b]
                    current_burst_segment = burst[idx : idx + L_glitch]
                    for j in range(L_glitch):
                        p = start_idx + j
                        if p < N:
                            out[p,0] += current_burst_segment[j]
                            out[p,1] += current_burst_segment[j]
                    idx += L_glitch
    return out
