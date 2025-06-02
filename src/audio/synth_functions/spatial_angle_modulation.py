"""Spatial angle modulation synthesis functions."""

import numpy as np
import math
import numba
import traceback
from .monaural_beat_stereo_amps import monaural_beat_stereo_amps, monaural_beat_stereo_amps_transition

# Placeholder for the missing audio_engine module
# If you have the 'audio_engine.py' file, place it in the same directory.
# Otherwise, the SAM functions will not work.
try:
    # Attempt to import the real audio_engine if available
    from .audio_engine import Node, SAMVoice, VALID_SAM_PATHS
    AUDIO_ENGINE_AVAILABLE = True
    print("INFO: audio_engine module loaded successfully in spatial_angle_modulation.")
except ImportError:
    AUDIO_ENGINE_AVAILABLE = False
    print("WARNING: audio_engine module not found in spatial_angle_modulation. SAM functions will not be available.")
    # Define dummy classes/variables if audio_engine is missing
    class Node:
        def __init__(self, *args, **kwargs):
            # Store args needed for generate_samples duration calculation
            # Simplified: Just store duration if provided
            self.duration = args[0] if args else kwargs.get('duration', 0)
            pass
    class SAMVoice:
        def __init__(self, *args, **kwargs):
            # Store args needed for generate_samples duration calculation
            self._nodes = kwargs.get('nodes', [])
            self._sample_rate = kwargs.get('sample_rate', 44100)
            pass
        def generate_samples(self):
            print("WARNING: SAM generate_samples called on dummy class. Returning silence.")
            # Calculate duration from stored nodes
            duration = 0
            if hasattr(self, '_nodes'):
                # Access duration attribute correctly from dummy Node
                duration = sum(node.duration for node in self._nodes if hasattr(node, 'duration'))
            sample_rate = getattr(self, '_sample_rate', 44100)
            N = int(duration * sample_rate) if duration > 0 else int(1.0 * sample_rate) # Default 1 sec if no duration found
            return np.zeros((N, 2))

    VALID_SAM_PATHS = ['circle', 'line', 'lissajous', 'figure_eight', 'arc'] # Example paths


# -----------------------------------------------------------------------------
# Spatial Angle Modulation - Helper Functions
# -----------------------------------------------------------------------------

@numba.njit(parallel=True, fastmath=True)
def _prepare_beats_and_angles(
    mono: np.ndarray,
    sample_rate: float,
    aOD: float, aOF: float, aOP: float,      # AM for this stage
    spatial_freq: float,
    path_radius: float,
    spatial_phase_off: float                 # Initial phase offset for spatial rotation
):
    N = mono.shape[0]
    if N == 0:
        return (np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32))

    mod_beat  = np.empty(N, dtype=np.float32)
    azimuth   = np.empty(N, dtype=np.float32)
    elevation = np.zeros(N, dtype=np.float32) # Elevation is kept at zero
    
    dt = 1.0 / sample_rate

    # 1) Apply custom AM envelope + build mod_beat
    for i in numba.prange(N):
        t = i * dt
        if aOD > 0.0 and aOF > 0.0:
            clamped_aOD = min(max(aOD, 0.0), 2.0)
            env = (1.0 - clamped_aOD/2.0) + (clamped_aOD/2.0) * math.sin(2*math.pi*aOF*t + aOP)
        else:
            env = 1.0
        mod_beat[i] = mono[i] * env # mono is already float32

    # 2) Compute circular path (radius from mod_beat) → azimuth
    # Corrected phase calculation for parallel loop (spatial_freq is constant here)
    for i in numba.prange(N):
        t_i = i * dt
        # Calculate phase at time t_i directly
        current_spatial_phase_at_t = spatial_phase_off + (2 * math.pi * spatial_freq * t_i)
        
        r = path_radius * (0.5 * (mod_beat[i] + 1.0)) # mod_beat is [-1, 1], so (mod_beat+1)/2 is [0,1]
        
        # Cartesian coordinates for HRTF lookup (y is often 'forward' in HRTF)
        # X = R * sin(angle) (side)
        # Y = R * cos(angle) (front/back)
        # atan2(x,y) means angle relative to positive Y axis, clockwise if X is positive.
        x_coord = r * math.sin(current_spatial_phase_at_t)
        y_coord = r * math.cos(current_spatial_phase_at_t)
        
        azimuth[i] = math.degrees(math.atan2(x_coord, y_coord))

    return mod_beat, azimuth, elevation


@numba.njit(parallel=True, fastmath=True)
def _prepare_beats_and_angles_transition_core(
    mono_input: np.ndarray, # Already transitional mono beat
    sample_rate: float,
    sAOD: float, eAOD: float,       # Start/End SAM Amplitude Osc Depth
    sAOF: float, eAOF: float,       # Start/End SAM Amplitude Osc Freq
    sAOP: float, eAOP: float,       # Start/End SAM Amplitude Osc Phase Offset
    sSpatialFreq: float, eSpatialFreq: float,
    sPathRadius: float, ePathRadius: float,
    sSpatialPhaseOff: float, eSpatialPhaseOff: float # Start/End for initial spatial phase
):
    N = mono_input.shape[0]
    if N == 0:
        return (np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32))

    mod_beat    = np.empty(N, dtype=np.float32)
    azimuth_deg = np.empty(N, dtype=np.float32)
    elevation_deg = np.zeros(N, dtype=np.float32) # Elevation is kept at zero
    
    actual_spatial_phase = np.empty(N, dtype=np.float64) # For storing accumulated phase

    dt = 1.0 / sample_rate

    # Loop 1 (parallel): Interpolate AM params and calculate mod_beat
    for i in numba.prange(N):
        alpha = i / (N - 1) if N > 1 else 0.0
        t_i = i * dt

        current_aOD = sAOD + (eAOD - sAOD) * alpha
        current_aOF = sAOF + (eAOF - sAOF) * alpha
        current_aOP = sAOP + (eAOP - sAOP) * alpha
        
        env_factor = 1.0
        if current_aOD > 0.0 and current_aOF > 0.0: # Assuming depth 0-2
            clamped_aOD_i = min(max(current_aOD, 0.0), 2.0)
            env_factor = (1.0 - clamped_aOD_i/2.0) + \
                         (clamped_aOD_i/2.0) * math.sin(2*math.pi*current_aOF*t_i + current_aOP)
        mod_beat[i] = mono_input[i] * env_factor

    # Loop 2 (sequential): Interpolate spatial freq and accumulate spatial phase
    # The 'spatial_phase_off' transition is for the initial phase offset value.
    initial_phase_offset_val = sSpatialPhaseOff + (eSpatialPhaseOff - sSpatialPhaseOff) * 0.0 # Value at alpha=0
    
    current_phase_val = initial_phase_offset_val 
    if N > 0:
      actual_spatial_phase[0] = current_phase_val

    for i in range(N): # Must be sequential due to phase accumulation
        alpha = i / (N - 1) if N > 1 else 0.0
        current_sf_i = sSpatialFreq + (eSpatialFreq - sSpatialFreq) * alpha
        
        if i > 0: # Accumulate phase
            current_phase_val += (2 * math.pi * current_sf_i * dt)
            actual_spatial_phase[i] = current_phase_val
        elif i == 0: # Already set for i=0 if N>0
             actual_spatial_phase[i] = current_phase_val

    # Loop 3 (parallel): Interpolate path radius and calculate azimuth
    for i in numba.prange(N):
        alpha = i / (N - 1) if N > 1 else 0.0
        current_path_r = sPathRadius + (ePathRadius - sPathRadius) * alpha
        
        r_factor = current_path_r * (0.5 * (mod_beat[i] + 1.0))
        
        x_coord = r_factor * math.sin(actual_spatial_phase[i])
        y_coord = r_factor * math.cos(actual_spatial_phase[i])
        azimuth_deg[i] = math.degrees(math.atan2(x_coord, y_coord))
        
    return mod_beat, azimuth_deg, elevation_deg


# -----------------------------------------------------------------------------
# Spatial Angle Modulation Functions
# -----------------------------------------------------------------------------

def spatial_angle_modulation(duration, sample_rate=44100, **params):
    """Spatial Angle Modulation using external audio_engine module."""
    if not AUDIO_ENGINE_AVAILABLE:
        print("Error: SAM function called, but audio_engine module is missing.")
        N = int(sample_rate * duration)
        return np.zeros((N, 2))

    amp = float(params.get('amp', 0.7))
    carrierFreq = float(params.get('carrierFreq', 440.0))
    beatFreq = float(params.get('beatFreq', 4.0))
    pathShape = str(params.get('pathShape', 'circle'))
    pathRadius = float(params.get('pathRadius', 1.0))
    arcStartDeg = float(params.get('arcStartDeg', 0.0))
    arcEndDeg = float(params.get('arcEndDeg', 360.0))
    frame_dur_ms = float(params.get('frame_dur_ms', 46.4))
    overlap_factor = int(params.get('overlap_factor', 8))

    if pathShape not in VALID_SAM_PATHS:
        print(f"Warning: Invalid pathShape '{pathShape}'. Defaulting to 'circle'. Valid: {VALID_SAM_PATHS}")
        pathShape = 'circle'

    try:
        node = Node(duration, carrierFreq, beatFreq, 1.0, 1.0) # Using real Node now
    except Exception as e:
        print(f"Error creating Node for SAM: {e}")
        N = int(sample_rate * duration)
        return np.zeros((N, 2))

    sam_params_dict = {
        'path_shape': pathShape,
        'path_radius': pathRadius,
        'arc_start_deg': arcStartDeg,
        'arc_end_deg': arcEndDeg
    }

    try:
        voice = SAMVoice( # Using real SAMVoice now
            nodes=[node],
            sample_rate=sample_rate,
            frame_dur_ms=frame_dur_ms,
            overlap_factor=overlap_factor,
            source_amp=amp, # Apply amp within SAMVoice
            sam_node_params=[sam_params_dict]
        )
        # Note: Envelope is applied *within* generate_voice_audio if specified there.
        return voice.generate_samples()
    except Exception as e:
        print(f"Error during SAMVoice generation: {e}")
        traceback.print_exc()
        N = int(sample_rate * duration)
        return np.zeros((N, 2))


def spatial_angle_modulation_transition(duration, sample_rate=44100, **params):
    """Spatial Angle Modulation with parameter transitions."""
    if not AUDIO_ENGINE_AVAILABLE:
        print("Error: SAM transition function called, but audio_engine module is missing.")
        N = int(sample_rate * duration)
        return np.zeros((N, 2))

    amp = float(params.get('amp', 0.7))
    startCarrierFreq = float(params.get('startCarrierFreq', 440.0))
    endCarrierFreq = float(params.get('endCarrierFreq', 440.0))
    startBeatFreq = float(params.get('startBeatFreq', 4.0))
    endBeatFreq = float(params.get('endBeatFreq', 4.0))
    startPathShape = str(params.get('startPathShape', 'circle'))
    endPathShape = str(params.get('endPathShape', 'circle'))
    startPathRadius = float(params.get('startPathRadius', 1.0))
    endPathRadius = float(params.get('endPathRadius', 1.0))
    startArcStartDeg = float(params.get('startArcStartDeg', 0.0))
    endArcStartDeg = float(params.get('endArcStartDeg', 0.0))
    startArcEndDeg = float(params.get('startArcEndDeg', 360.0))
    endArcEndDeg = float(params.get('endArcEndDeg', 360.0))
    frame_dur_ms = float(params.get('frame_dur_ms', 46.4))
    overlap_factor = int(params.get('overlap_factor', 8))

    if startPathShape not in VALID_SAM_PATHS:
        print(f"Warning: Invalid startPathShape '{startPathShape}'. Defaulting to 'circle'.")
        startPathShape = 'circle'
    if endPathShape not in VALID_SAM_PATHS:
        print(f"Warning: Invalid endPathShape '{endPathShape}'. Defaulting to 'circle'.")
        endPathShape = 'circle'

    try:
        # Define start and end nodes for transition
        node_start = Node(duration, startCarrierFreq, startBeatFreq, 1.0, 1.0)
        node_end = Node(0.0, endCarrierFreq, endBeatFreq, 1.0, 1.0) # End node has 0 duration
    except Exception as e:
        print(f"Error creating Nodes for SAM transition: {e}")
        N = int(sample_rate * duration)
        return np.zeros((N, 2))

    sam_params_list = [
        { # Parameters for the start node
            'path_shape': startPathShape,
            'path_radius': startPathRadius,
            'arc_start_deg': startArcStartDeg,
            'arc_end_deg': startArcEndDeg
        },
        { # Parameters for the (conceptual) end node
            'path_shape': endPathShape,
            'path_radius': endPathRadius,
            'arc_start_deg': endArcStartDeg,
            'arc_end_deg': endArcEndDeg
        }
    ]

    try:
        voice = SAMVoice( # Using real SAMVoice
            nodes=[node_start, node_end], # Pass both nodes for transition
            sample_rate=sample_rate,
            frame_dur_ms=frame_dur_ms,
            overlap_factor=overlap_factor,
            source_amp=amp, # Apply amp within SAMVoice
            sam_node_params=sam_params_list # Pass list of params
        )
        # Note: Envelope is applied *within* generate_voice_audio if specified there.
        return voice.generate_samples()
    except Exception as e:
        print(f"Error during SAMVoice transition generation: {e}")
        traceback.print_exc()
        N = int(sample_rate * duration)
        return np.zeros((N, 2))


def spatial_angle_modulation_monaural_beat(duration, sample_rate=44100, **params):
    """Spatial Angle Modulation combined with monaural beats."""
    # --- unpack AM params for this specific stage ---
    # These are applied *after* the monaural beat generation's own AM
    sam_aOD = float(params.get('sam_ampOscDepth', params.get('ampOscDepth', 0.0))) # prefix with sam_ to avoid conflict
    sam_aOF = float(params.get('sam_ampOscFreq', params.get('ampOscFreq', 0.0)))
    sam_aOP = float(params.get('sam_ampOscPhaseOffset', params.get('ampOscPhaseOffset', 0.0)))

    # --- prepare core beat args (can have its own AM) ---
    beat_params = {
        'amp_lower_L':   float(params.get('amp_lower_L',   0.5)),
        'amp_upper_L':   float(params.get('amp_upper_L',   0.5)),
        'amp_lower_R':   float(params.get('amp_lower_R',   0.5)),
        'amp_upper_R':   float(params.get('amp_upper_R',   0.5)),
        'baseFreq':      float(params.get('baseFreq',      200.0)),
        'beatFreq':      float(params.get('beatFreq',        4.0)),
        'startPhaseL':   float(params.get('startPhaseL',    0.0)),
        'startPhaseR':   float(params.get('startPhaseR',    0.0)), # for upper component
        'phaseOscFreq':  float(params.get('phaseOscFreq',  0.0)),
        'phaseOscRange': float(params.get('phaseOscRange', 0.0)),
        'ampOscDepth':   float(params.get('monaural_ampOscDepth', 0.0)), # AM for monaural beat
        'ampOscFreq':    float(params.get('monaural_ampOscFreq', 0.0)),
        'ampOscPhaseOffset': float(params.get('monaural_ampOscPhaseOffset', 0.0)) # AM Phase for monaural
    }
    beat_freq         = beat_params['beatFreq']
    spatial_freq      = float(params.get('spatialBeatFreq', beat_freq)) # Default to beatFreq
    spatial_phase_off = float(params.get('spatialPhaseOffset', 0.0)) # Radians

    # --- SAM controls ---
    amp               = float(params.get('amp', 0.7)) # Overall amplitude for HRTF input
    path_radius       = float(params.get('pathRadius', 1.0)) # Normalized radius factor
    frame_dur_ms      = float(params.get('frame_dur_ms', 46.4))
    overlap_fac       = int(params.get('overlap_factor',   8))

    N = int(duration * sample_rate)
    if N <= 0:
        return np.zeros((0,2), dtype=np.float32)

    # Generate core stereo beat & collapse to mono
    beat_stereo = monaural_beat_stereo_amps(duration, sample_rate, **beat_params)
    mono_beat   = np.mean(beat_stereo, axis=1).astype(np.float32)

    # Call Numba helper to get mod_beat + az/el arrays
    mod_beat, azimuth_deg, elevation_deg = _prepare_beats_and_angles(
        mono_beat, float(sample_rate),
        sam_aOD, sam_aOF, sam_aOP, # SAM-specific AM params
        spatial_freq, path_radius,
        spatial_phase_off
    )

    # --- OLA + HRTF (assuming slab and HRTF are available and configured) ---
    stereo_out = np.zeros((N, 2), dtype=np.float32)
    stereo_out[:, 0] = mod_beat * amp
    stereo_out[:, 1] = mod_beat * amp # Simple mono duplicate
    max_val = np.max(np.abs(stereo_out))
    if max_val > 1.0:
       stereo_out /= (max_val / 0.98)
    return stereo_out


def spatial_angle_modulation_monaural_beat_transition(duration, sample_rate=44100, **params):
    """Spatial Angle Modulation monaural beat with transitions."""
    N = int(duration * sample_rate)
    if N <= 0:
        return np.zeros((0,2), dtype=np.float32)

    # --- Parameters for the underlying monaural_beat_stereo_amps_transition ---
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
    sSPL_mono = float(params.get('startStartPhaseL_monaural', params.get('startPhaseL', 0.0)))
    eSPL_mono = float(params.get('endStartPhaseL_monaural', sSPL_mono))
    sSPU_mono = float(params.get('startStartPhaseU_monaural', params.get('startPhaseR', 0.0)))
    eSPU_mono = float(params.get('endStartPhaseU_monaural', sSPU_mono))
    sPhiF_mono = float(params.get('startPhaseOscFreq_monaural', params.get('phaseOscFreq', 0.0)))
    ePhiF_mono = float(params.get('endPhaseOscFreq_monaural', sPhiF_mono))
    sPhiR_mono = float(params.get('startPhaseOscRange_monaural', params.get('phaseOscRange', 0.0)))
    ePhiR_mono = float(params.get('endPhaseOscRange_monaural', sPhiR_mono))
    sAOD_mono = float(params.get('startAmpOscDepth_monaural', params.get('monaural_ampOscDepth', 0.0)))
    eAOD_mono = float(params.get('endAmpOscDepth_monaural', sAOD_mono))
    sAOF_mono = float(params.get('startAmpOscFreq_monaural', params.get('monaural_ampOscFreq', 0.0)))
    eAOF_mono = float(params.get('endAmpOscFreq_monaural', sAOF_mono))
    sAOP_mono = float(params.get('startAmpOscPhaseOffset_monaural', params.get('monaural_ampOscPhaseOffset', 0.0)))
    eAOP_mono = float(params.get('endAmpOscPhaseOffset_monaural', sAOP_mono))

    monaural_trans_params = {
        'start_amp_lower_L': s_ll, 'end_amp_lower_L': e_ll,
        'start_amp_upper_L': s_ul, 'end_amp_upper_L': e_ul,
        'start_amp_lower_R': s_lr, 'end_amp_lower_R': e_lr,
        'start_amp_upper_R': s_ur, 'end_amp_upper_R': e_ur,
        'startBaseFreq': sBF, 'endBaseFreq': eBF,
        'startBeatFreq': sBt, 'endBeatFreq': eBt,
        'startStartPhaseL': sSPL_mono, 'endStartPhaseL': eSPL_mono,
        'startStartPhaseU': sSPU_mono, 'endStartPhaseU': eSPU_mono,
        'startPhaseOscFreq': sPhiF_mono, 'endPhaseOscFreq': ePhiF_mono,
        'startPhaseOscRange': sPhiR_mono, 'endPhaseOscRange': ePhiR_mono,
        'startAmpOscDepth': sAOD_mono, 'endAmpOscDepth': eAOD_mono,
        'startAmpOscFreq': sAOF_mono, 'endAmpOscFreq': eAOF_mono,
        'startAmpOscPhaseOffset': sAOP_mono, 'endAmpOscPhaseOffset': eAOP_mono,
    }

    # --- Parameters for the SAM stage AM and spatialization (transitional) ---
    sSamAOD = float(params.get('start_sam_ampOscDepth', params.get('sam_ampOscDepth', 0.0)))
    eSamAOD = float(params.get('end_sam_ampOscDepth', sSamAOD))
    sSamAOF = float(params.get('start_sam_ampOscFreq', params.get('sam_ampOscFreq', 0.0)))
    eSamAOF = float(params.get('end_sam_ampOscFreq', sSamAOF))
    sSamAOP = float(params.get('start_sam_ampOscPhaseOffset', params.get('sam_ampOscPhaseOffset', 0.0)))
    eSamAOP = float(params.get('end_sam_ampOscPhaseOffset', sSamAOP))

    default_spatial_freq = (sBt + eBt) / 2.0 # Default to average beatFreq
    sSpatialFreq = float(params.get('startSpatialBeatFreq', params.get('spatialBeatFreq', default_spatial_freq)))
    eSpatialFreq = float(params.get('endSpatialBeatFreq', sSpatialFreq))
    
    sSpatialPhaseOff = float(params.get('startSpatialPhaseOffset', params.get('spatialPhaseOffset', 0.0)))
    eSpatialPhaseOff = float(params.get('endSpatialPhaseOffset', sSpatialPhaseOff))

    sPathRadius = float(params.get('startPathRadius', params.get('pathRadius', 1.0)))
    ePathRadius = float(params.get('endPathRadius', sPathRadius))

    # --- SAM controls (non-transitional for OLA) ---
    sAmp = float(params.get('startAmp', params.get('amp', 0.7))) # Overall amplitude for HRTF input
    eAmp = float(params.get('endAmp', sAmp))
    # For OLA, amp is applied per frame. We can interpolate it if needed, or use an average.
    # For now, let's use interpolated amp for mono_src
    
    frame_dur_ms = float(params.get('frame_dur_ms', 46.4))
    overlap_fac  = int(params.get('overlap_factor',   8))

    # 1. Generate transitional monaural beat
    trans_beat_stereo = monaural_beat_stereo_amps_transition(duration, sample_rate, **monaural_trans_params)
    trans_mono_beat   = np.mean(trans_beat_stereo, axis=1).astype(np.float32)

    # 2. Call the new transitional _prepare_beats_and_angles_transition_core
    trans_mod_beat, trans_azimuth_deg, trans_elevation_deg = \
        _prepare_beats_and_angles_transition_core(
            trans_mono_beat, float(sample_rate),
            sSamAOD, eSamAOD, sSamAOF, eSamAOF, sSamAOP, eSamAOP,
            sSpatialFreq, eSpatialFreq,
            sPathRadius, ePathRadius,
            sSpatialPhaseOff, eSpatialPhaseOff
        )
    
    # 3. OLA + HRTF processing (using transitional mod_beat and azimuth)
    # This part remains Python-based.
    # Placeholder for slab integration - replace with actual calls.
    print("spatial_angle_modulation_monaural_beat_transition: HRTF processing part is illustrative.")
    # Fallback: return a simple stereo mix of trans_mod_beat with interpolated amp
    final_amp_coeffs = np.linspace(sAmp, eAmp, N, dtype=np.float32) if N > 0 else np.array([], dtype=np.float32)
    
    temp_out = np.zeros((N, 2), dtype=np.float32)
    if N > 0:
       temp_out[:, 0] = trans_mod_beat * final_amp_coeffs
       temp_out[:, 1] = trans_mod_beat * final_amp_coeffs
    
    max_v = np.max(np.abs(temp_out)) if N > 0 and temp_out.size > 0 else 0.0
    if max_v > 1.0: temp_out /= (max_v / 0.98)
    return temp_out
