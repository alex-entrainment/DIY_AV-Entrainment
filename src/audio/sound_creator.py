import numpy as np
from scipy.signal import butter, lfilter, sosfiltfilt
from scipy.io.wavfile import write
import json
import inspect # Needed to inspect function parameters for GUI
import os # Needed for path checks in main example
import traceback # For detailed error printing

# Import all synth functions from the synth_functions package
from synth_functions import *

# Placeholder for the missing audio_engine module
# If you have the 'audio_engine.py' file, place it in the same directory.
# Otherwise, the SAM functions will not work.
try:
    # Attempt to import the real audio_engine if available
    from audio_engine import Node, SAMVoice, VALID_SAM_PATHS
    AUDIO_ENGINE_AVAILABLE = True
    print("INFO: audio_engine module loaded successfully.")
except ImportError:
    AUDIO_ENGINE_AVAILABLE = False
    print("WARNING: audio_engine module not found. Spatial Angle Modulation (SAM) functions will not be available.")
    # Define dummy classes/variables if audio_engine is missing
    class Node:
        def __init__(self, *args, **kwargs):
            # print("WARNING: Using dummy Node class. SAM functionality disabled.")
            # Store args needed for generate_samples duration calculation
            # Simplified: Just store duration if provided
            self.duration = args[0] if args else kwargs.get('duration', 0)
            pass
    class SAMVoice:
        def __init__(self, *args, **kwargs):
            # print("WARNING: Using dummy SAMVoice class. SAM functionality disabled.")
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
# Crossfade and Assembly Logic
# -----------------------------------------------------------------------------

def crossfade_signals(signal_a, signal_b, sample_rate, transition_duration):
    """
    Crossfades two stereo signals. Assumes signal_a fades out, signal_b fades in.
    Operates on the initial segments of the signals up to transition_duration.
    Returns the blended segment.
    """
    n_samples = int(transition_duration * sample_rate)
    if n_samples <= 0:
        # No crossfade duration, return silence or handle appropriately
        return np.zeros((0, 2))

    # Determine the actual number of samples available for crossfade
    len_a = signal_a.shape[0]
    len_b = signal_b.shape[0]
    actual_crossfade_samples = min(n_samples, len_a, len_b)

    if actual_crossfade_samples <= 0:
        print(f"Warning: Crossfade not possible or zero length. Samples: {n_samples}, SigA: {len_a}, SigB: {len_b}")
        # Return an empty array matching the expected dimensions if no crossfade happens
        return np.zeros((0, 2))

    # Ensure signals are 2D stereo (N, 2) before slicing
    def ensure_stereo(sig):
        if sig.ndim == 1: sig = np.column_stack((sig, sig)) # Mono to Stereo
        elif sig.shape[1] == 1: sig = np.column_stack((sig[:,0], sig[:,0])) # (N, 1) to (N, 2)
        if sig.shape[1] != 2: raise ValueError("Signal must be stereo (N, 2) for crossfade.")
        return sig

    try:
        signal_a = ensure_stereo(signal_a)
        signal_b = ensure_stereo(signal_b)
    except ValueError as e:
        print(f"Error in crossfade_signals: {e}")
        return np.zeros((0, 2)) # Return empty on error

    # Take only the required number of samples for crossfade
    signal_a_seg = signal_a[:actual_crossfade_samples]
    signal_b_seg = signal_b[:actual_crossfade_samples]

    # Linear crossfade ramp (can be replaced with equal power: np.sqrt(fade))
    fade_out = np.linspace(1, 0, actual_crossfade_samples)[:, None] # Column vector for broadcasting
    fade_in = np.linspace(0, 1, actual_crossfade_samples)[:, None]

    # Apply fades and sum
    blended_segment = signal_a_seg * fade_out + signal_b_seg * fade_in
    return blended_segment


# Dictionary mapping function names (strings) to actual functions
# --- UPDATED SYNTH_FUNCTIONS DICTIONARY ---
# Exclude helper/internal functions explicitly
_EXCLUDED_FUNCTION_NAMES = [
    'validate_float', 'validate_int', 'butter_bandpass', 'bandpass_filter',
    'butter_bandstop', 'bandreject_filter', 'lowpass_filter', 'pink_noise',
    'brown_noise', 'sine_wave', 'sine_wave_varying', 'adsr_envelope',
    'create_linear_fade_envelope', 'linen_envelope', 'pan2', 'safety_limiter',
    'crossfade_signals', 'assemble_track_from_data', 'generate_voice_audio',
    'load_track_from_json', 'save_track_to_json', 'generate_wav', 'get_synth_params',
    'trapezoid_envelope_vectorized', '_flanger_effect_stereo_continuous',
    'butter', 'lfilter', 'write', 'ensure_stereo', 'apply_filters', 'design_filter', 
    # Standard library functions that might be imported
    'json', 'inspect', 'os', 'traceback', 'math', 'copy', 'binaural_beat_transition', 'hybrid_qam_monaural_beat_transition', 'isochronic_tone_transition', 'monaural_beat_stereo_amps_transition',
    'rhythmic_waveshaping_transition', 'stereo_am_independent_transition', 'wave_shape_stereo_am_transition',
    'spatial_angle_modulation_transition', 'spatial_angle_modulation_monaural_beat_transition'
]

SYNTH_FUNCTIONS = {}
try:
    # Import the synth_functions package
    import synth_functions
    for name, obj in inspect.getmembers(synth_functions):
        if inspect.isfunction(obj) and name not in _EXCLUDED_FUNCTION_NAMES and not name.startswith('_'):
             # Check if the function is defined in the synth_functions package or its submodules
            if obj.__module__.startswith('synth_functions'):
                SYNTH_FUNCTIONS[name] = obj
except Exception as e:
    print(f"Error inspecting functions: {e}")

print(f"Detected Synth Functions: {list(SYNTH_FUNCTIONS.keys())}")


def get_synth_params(func_name):
    """Gets parameter names and default values for a synth function by inspecting its signature."""
    if func_name not in SYNTH_FUNCTIONS:
        print(f"Warning: Function '{func_name}' not found in SYNTH_FUNCTIONS.")
        return {}

    func = SYNTH_FUNCTIONS[func_name]
    params = {}
    try:
        sig = inspect.signature(func)
        for name, param in sig.parameters.items():
            # Skip standard args and the catch-all kwargs
            if name in ['duration', 'sample_rate'] or param.kind == inspect.Parameter.VAR_KEYWORD:
                continue

            # Store the default value if it exists, otherwise store inspect._empty
            params[name] = param.default # Keep _empty to distinguish from None default

    except Exception as e:
        print(f"Error inspecting signature for '{func_name}': {e}")
        # Fallback to trying source code parsing if signature fails? Or just return empty?
        # For now, return empty on inspection error. Source parsing is done in GUI.
        return {}

    return params


def generate_voice_audio(voice_data, duration, sample_rate, global_start_time):
    """Generates audio for a single voice based on its definition."""
    func_name = voice_data.get("synth_function_name")
    params = voice_data.get("params", {})
    is_transition = voice_data.get("is_transition", False) # Check if this step IS a transition

    # --- Select the correct function (static or transition) ---
    actual_func_name = func_name
    selected_func_is_transition_type = func_name and func_name.endswith("_transition")

    # Determine the function to actually call based on 'is_transition' flag
    if is_transition:
        if not selected_func_is_transition_type:
            transition_func_name = func_name + "_transition"
            if transition_func_name in SYNTH_FUNCTIONS:
                actual_func_name = transition_func_name
                print(f"Note: Step marked as transition, using '{actual_func_name}' instead of base '{func_name}'.")
            else:
                print(f"Warning: Step marked as transition, but transition function '{transition_func_name}' not found for base '{func_name}'. Using static version '{func_name}'. Parameters might mismatch.")
                # Keep actual_func_name as func_name (the static one)
    else: # Not a transition step
        if selected_func_is_transition_type:
            base_func_name = func_name.replace("_transition", "")
            if base_func_name in SYNTH_FUNCTIONS:
                actual_func_name = base_func_name
                print(f"Note: Step not marked as transition, using base function '{actual_func_name}' instead of selected '{func_name}'.")
            else:
                print(f"Warning: Step not marked as transition, selected '{func_name}', but base function '{base_func_name}' not found. Using selected '{func_name}'. Parameters might mismatch.")
                # Keep actual_func_name as func_name (the transition one user selected)

    if not actual_func_name or actual_func_name not in SYNTH_FUNCTIONS:
        print(f"Error: Synth function '{actual_func_name}' (derived from '{func_name}') not found or invalid.")
        N = int(duration * sample_rate)
        return np.zeros((N, 2))

    synth_func = SYNTH_FUNCTIONS[actual_func_name]

    # Clean params: remove None values before passing to function, as functions use .get() with defaults
    cleaned_params = {k: v for k, v in params.items() if v is not None}

    # --- Generate base audio ---
    try:
        print(f"  Calling: {actual_func_name}(duration={duration}, sample_rate={sample_rate}, **{cleaned_params})")
        audio = synth_func(duration=duration, sample_rate=sample_rate, **cleaned_params)
    except Exception as e:
        print(f"Error calling synth function '{actual_func_name}' with params {cleaned_params}:")
        traceback.print_exc()
        N = int(duration * sample_rate)
        return np.zeros((N, 2))

    if audio is None:
        print(f"Error: Synth function '{actual_func_name}' returned None.")
        N = int(duration * sample_rate)
        return np.zeros((N, 2))

    # --- Apply volume envelope if defined ---
    envelope_data = voice_data.get("volume_envelope")
    N_audio = audio.shape[0]
    # Ensure t_rel matches audio length, especially if N calculation differs slightly
    t_rel = np.linspace(0, duration, N_audio, endpoint=False) if N_audio > 0 else np.array([])
    env = np.ones(N_audio) # Default flat envelope

    if envelope_data and isinstance(envelope_data, dict) and N_audio > 0:
        env_type = envelope_data.get("type")
        env_params = envelope_data.get("params", {})
        cleaned_env_params = {k: v for k, v in env_params.items() if v is not None}

        try:
            # Pass duration and sample_rate if needed by envelope func
            if 'duration' not in cleaned_env_params: cleaned_env_params['duration'] = duration
            if 'sample_rate' not in cleaned_env_params: cleaned_env_params['sample_rate'] = sample_rate

            if env_type == "adsr":
                env = adsr_envelope(t_rel, **cleaned_env_params)
            elif env_type == "linen":
                 env = linen_envelope(t_rel, **cleaned_env_params)
            elif env_type == "linear_fade":
                 # This function uses duration/sr internally, ensure they are passed if needed
                 required = ['fade_duration', 'start_amp', 'end_amp']
                 # Check params specific to linear_fade
                 specific_env_params = {k: v for k, v in cleaned_env_params.items() if k in required}
                 if all(p in specific_env_params for p in required):
                      # Pass the main duration and sample rate, not t_rel
                      env = create_linear_fade_envelope(duration, sample_rate, **specific_env_params)
                      # Resample envelope if its length doesn't match audio
                      if len(env) != N_audio:
                            print(f"Warning: Resampling '{env_type}' envelope from {len(env)} to {N_audio} samples.")
                            if len(env) > 0:
                                 env = np.interp(np.linspace(0, 1, N_audio), np.linspace(0, 1, len(env)), env)
                            else:
                                 env = np.ones(N_audio) # Fallback
                 else:
                      print(f"Warning: Missing parameters for 'linear_fade' envelope. Using flat envelope. Got: {specific_env_params}")
            # Add other envelope types here
            # elif env_type == "other_env":
            #    env = other_env_function(t_rel, **cleaned_env_params)
            else:
                print(f"Warning: Unknown envelope type '{env_type}'. Using flat envelope.")

            # Ensure envelope is broadcastable (N,)
            if env.shape != (N_audio,):
                 print(f"Warning: Envelope shape mismatch ({env.shape} vs {(N_audio,)}). Attempting reshape.")
                 if len(env) == N_audio: env = env.reshape(N_audio)
                 else:
                      print("Error: Cannot reshape envelope. Using flat envelope.")
                      env = np.ones(N_audio) # Fallback

        except Exception as e:
            print(f"Error creating envelope type '{env_type}':")
            traceback.print_exc()
            env = np.ones(N_audio) # Fallback

    # Apply the calculated envelope
    try:
        if audio.ndim == 2 and audio.shape[1] == 2 and len(env) == audio.shape[0]:
             audio = audio * env[:, np.newaxis] # Apply envelope element-wise to stereo
        elif audio.ndim == 1 and len(env) == len(audio): # Handle potential mono output from synth
             audio = audio * env
        elif N_audio == 0:
             pass # No audio to apply envelope to
        else:
             print(f"Error: Envelope length ({len(env)}) or audio shape ({audio.shape}) mismatch. Skipping envelope application.")
    except Exception as e:
        print(f"Error applying envelope to audio:")
        traceback.print_exc()


    # --- Ensure output is stereo ---
    if audio.ndim == 1:
        print(f"Note: Synth function '{actual_func_name}' resulted in mono audio. Panning.")
        pan_val = cleaned_params.get('pan', 0.0) # Assume pan param exists or default to center
        audio = pan2(audio, pan_val)
    elif audio.ndim == 2 and audio.shape[1] == 1:
        print(f"Note: Synth function '{actual_func_name}' resulted in mono audio (N, 1). Panning.")
        pan_val = cleaned_params.get('pan', 0.0)
        audio = pan2(audio[:,0], pan_val) # Extract the single column before panning

    # Final check for shape (N, 2)
    if not (audio.ndim == 2 and audio.shape[1] == 2):
          if N_audio == 0: return np.zeros((0, 2)) # Handle zero duration case gracefully
          else:
                print(f"Error: Final audio shape for voice is incorrect ({audio.shape}). Returning silence.")
                N_expected = int(duration * sample_rate)
                return np.zeros((N_expected, 2))


    # Add a small default fade if no specific envelope was requested and audio exists
    # This helps prevent clicks when steps are concatenated without crossfade
    if not envelope_data and N_audio > 0:
        fade_len = min(N_audio, int(0.01 * sample_rate)) # 10ms fade or audio length
        if fade_len > 1:
             fade_in = np.linspace(0, 1, fade_len)
             fade_out = np.linspace(1, 0, fade_len)
             # Apply fade using broadcasting
             audio[:fade_len] *= fade_in[:, np.newaxis]
             audio[-fade_len:] *= fade_out[:, np.newaxis]

    return audio.astype(np.float32) # Ensure float32 output


def assemble_track_from_data(track_data, sample_rate, crossfade_duration):
    """
    Assembles a track from a track_data dictionary.
    Uses crossfading between steps by overlapping their placement.
    Includes per-step normalization to prevent excessive peaks before final mix.
    """
    steps_data = track_data.get("steps", [])
    if not steps_data:
        print("Warning: No steps found in track data.")
        return np.zeros((sample_rate, 2)) # Return 1 second silence

    # --- Calculate Track Length Estimation ---
    estimated_total_duration = sum(float(step.get("duration", 0)) for step in steps_data)
    if estimated_total_duration <= 0:
        print("Warning: Track has zero or negative estimated total duration.")
        return np.zeros((sample_rate, 2))

    # Add buffer for potential rounding errors and final sample
    estimated_total_samples = int(estimated_total_duration * sample_rate) + sample_rate
    track = np.zeros((estimated_total_samples, 2), dtype=np.float32) # Use float32

    # --- Time and Sample Tracking ---
    current_time = 0.0 # Start time for the *next* step to be placed
    last_step_end_sample_in_track = 0 # Tracks the actual last sample index used

    crossfade_samples = int(crossfade_duration * sample_rate)
    if crossfade_samples < 0: crossfade_samples = 0

    print(f"Assembling track: {len(steps_data)} steps, Est. Max Duration: {estimated_total_duration:.2f}s, Crossfade: {crossfade_duration:.2f}s ({crossfade_samples} samples)")

    for i, step_data in enumerate(steps_data):
        step_duration = float(step_data.get("duration", 0))
        if step_duration <= 0:
            print(f"Skipping step {i+1} with zero or negative duration.")
            continue

        # --- Calculate Placement Indices ---
        step_start_sample_abs = int(current_time * sample_rate)
        N_step = int(step_duration * sample_rate)
        step_end_sample_abs = step_start_sample_abs + N_step

        print(f"  Processing Step {i+1}: Place Start: {current_time:.2f}s ({step_start_sample_abs}), Duration: {step_duration:.2f}s, Samples: {N_step}")

        # Generate audio for all voices in this step and mix them
        step_audio_mix = np.zeros((N_step, 2), dtype=np.float32) # Use float32
        voices_data = step_data.get("voices", [])

        if not voices_data:
            print(f"        Warning: Step {i+1} has no voices.")
        else:
            num_voices_in_step = len(voices_data)
            print(f"        Mixing {num_voices_in_step} voice(s) for Step {i+1}...")
            for j, voice_data in enumerate(voices_data):
                func_name_short = voice_data.get('synth_function_name', 'UnknownFunc')
                print(f"          Generating Voice {j+1}/{num_voices_in_step}: {func_name_short}")
                voice_audio = generate_voice_audio(voice_data, step_duration, sample_rate, current_time)

                # Add generated audio if valid
                if voice_audio is not None and voice_audio.shape[0] == N_step and voice_audio.ndim == 2 and voice_audio.shape[1] == 2:
                    step_audio_mix += voice_audio # Sum voices
                elif voice_audio is not None:
                    print(f"          Error: Voice {j+1} ({func_name_short}) generated audio shape mismatch ({voice_audio.shape} vs {(N_step, 2)}). Skipping voice.")

            # --- *** NEW: Per-Step Normalization/Limiting *** ---
            # Check the peak of the mixed step audio
            step_peak = np.max(np.abs(step_audio_mix))
            # Define a threshold slightly above 1.0 to allow headroom but prevent extreme peaks
            step_normalization_threshold = 1.0
            if step_peak > step_normalization_threshold:
                print(f"        Normalizing Step {i+1} mix (peak={step_peak:.3f}) down to {step_normalization_threshold:.2f}")
                step_audio_mix *= (step_normalization_threshold / step_peak)
            # --- *** End Per-Step Normalization *** ---


        # --- Placement and Crossfading ---
        # Clip placement indices to the allocated track buffer boundaries
        safe_place_start = max(0, step_start_sample_abs)
        safe_place_end = min(estimated_total_samples, step_end_sample_abs)
        segment_len_in_track = safe_place_end - safe_place_start

        if segment_len_in_track <= 0:
            print(f"        Skipping Step {i+1} placement (no valid range in track buffer).")
            continue

        # Determine the portion of step_audio_mix to use
        audio_to_use = step_audio_mix[:segment_len_in_track]

        # Double check length (should normally match)
        if audio_to_use.shape[0] != segment_len_in_track:
            print(f"        Warning: Step {i+1} audio length adjustment needed ({audio_to_use.shape[0]} vs {segment_len_in_track}). Padding/Truncating.")
            if audio_to_use.shape[0] < segment_len_in_track:
                audio_to_use = np.pad(audio_to_use, ((0, segment_len_in_track - audio_to_use.shape[0]), (0,0)), 'constant')
            else:
                audio_to_use = audio_to_use[:segment_len_in_track]

        # --- Actual Crossfade Logic ---
        overlap_start_sample_in_track = safe_place_start
        overlap_end_sample_in_track = min(safe_place_end, last_step_end_sample_in_track)
        overlap_samples = overlap_end_sample_in_track - overlap_start_sample_in_track

        can_crossfade = i > 0 and overlap_samples > 0 and crossfade_samples > 0

        if can_crossfade:
            actual_crossfade_samples = min(overlap_samples, crossfade_samples)
            print(f"        Crossfading Step {i+1} with previous. Overlap: {overlap_samples / sample_rate:.3f}s, Actual CF: {actual_crossfade_samples / sample_rate:.3f}s")

            if actual_crossfade_samples > 0:
                # Get segments for crossfading
                prev_segment = track[overlap_start_sample_in_track : overlap_start_sample_in_track + actual_crossfade_samples]
                new_segment = audio_to_use[:actual_crossfade_samples]

                # Perform crossfade
                blended_segment = crossfade_signals(prev_segment, new_segment, sample_rate, actual_crossfade_samples / sample_rate)

                # Place blended segment (overwrite previous tail)
                track[overlap_start_sample_in_track : overlap_start_sample_in_track + actual_crossfade_samples] = blended_segment

                # Add the remainder of the new step (after the crossfaded part)
                remaining_start_index_in_step_audio = actual_crossfade_samples
                remaining_start_index_in_track = overlap_start_sample_in_track + actual_crossfade_samples
                remaining_end_index_in_track = safe_place_end

                if remaining_start_index_in_track < remaining_end_index_in_track:
                    num_remaining_samples_to_add = remaining_end_index_in_track - remaining_start_index_in_track
                    if remaining_start_index_in_step_audio < audio_to_use.shape[0]:
                        remaining_audio_from_step = audio_to_use[remaining_start_index_in_step_audio : remaining_start_index_in_step_audio + num_remaining_samples_to_add]
                        # Add the remaining part (use += as it might overlap with the next step's fade-in region)
                        track[remaining_start_index_in_track : remaining_start_index_in_track + remaining_audio_from_step.shape[0]] += remaining_audio_from_step

            else: # Overlap existed but calculated crossfade samples was zero
                 print(f"        Placing Step {i+1} without crossfade (actual_crossfade_samples=0). Adding.")
                 track[safe_place_start:safe_place_end] += audio_to_use # Add instead of overwrite

        else: # No crossfade (first step or no overlap)
            print(f"        Placing Step {i+1} without crossfade. Adding.")
            # Add the audio (use += because the space might be overlapped by the *next* step's fade)
            track[safe_place_start:safe_place_end] += audio_to_use

        # --- Update Markers for Next Loop ---
        last_step_end_sample_in_track = max(last_step_end_sample_in_track, safe_place_end)
        # Advance current_time for the START of the next step, pulling back by crossfade duration
        effective_advance_duration = max(0.0, step_duration - crossfade_duration) if crossfade_samples > 0 else step_duration
        current_time += effective_advance_duration


    # --- Final Trimming ---
    final_track_samples = last_step_end_sample_in_track
    if final_track_samples <= 0:
        print("Warning: Final track assembly resulted in zero length.")
        return np.zeros((sample_rate, 2))

    if final_track_samples < track.shape[0]:
        track = track[:final_track_samples]
    elif final_track_samples > estimated_total_samples:
         print(f"Warning: Final track samples ({final_track_samples}) exceeded initial estimate ({estimated_total_samples}).")

    print(f"Track assembly finished. Final Duration: {track.shape[0] / sample_rate:.2f}s")
    return track


# -----------------------------------------------------------------------------
# JSON Loading/Saving
# -----------------------------------------------------------------------------

# Custom JSON encoder to handle numpy types (if needed)
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def load_track_from_json(filepath):
    """Loads track definition from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            track_data = json.load(f)
        print(f"Track data loaded successfully from {filepath}")
        # Basic validation
        if not isinstance(track_data, dict) or \
           "global_settings" not in track_data or \
           "steps" not in track_data or \
           not isinstance(track_data["steps"], list) or \
           not isinstance(track_data["global_settings"], dict):
            print("Error: Invalid JSON structure. Missing 'global_settings' dict or 'steps' list.")
            return None
        # Further validation could check step/voice structure if needed
        return track_data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {filepath}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred loading {filepath}:")
        traceback.print_exc()
        return None

def save_track_to_json(track_data, filepath):
    """Saves track definition to a JSON file."""
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(track_data, f, indent=4, cls=NumpyEncoder)
        print(f"Track data saved successfully to {filepath}")
        return True
    except IOError as e:
        print(f"Error writing file to {filepath}: {e}")
        return False
    except TypeError as e:
        print(f"Error serializing track data to JSON: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"An unexpected error occurred saving to {filepath}:")
        traceback.print_exc()
        return False

# -----------------------------------------------------------------------------
# Main Generation Function
# -----------------------------------------------------------------------------

def generate_wav(track_data, output_filename=None):
    """Generates and saves the WAV file based on the track_data."""
    if not track_data:
        print("Error: Cannot generate WAV, track data is missing.")
        return False

    global_settings = track_data.get("global_settings", {})
    try:
        sample_rate = int(global_settings.get("sample_rate", 44100))
        crossfade_duration = float(global_settings.get("crossfade_duration", 1.0))
    except (ValueError, TypeError) as e:
         print(f"Error: Invalid global settings (sample_rate or crossfade_duration): {e}")
         return False

    output_filename = output_filename or global_settings.get("output_filename", "generated_track.wav")
    if not output_filename or not isinstance(output_filename, str):
         print(f"Error: Invalid output filename: {output_filename}")
         return False

    # Ensure output directory exists before assembly
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory for WAV: {output_dir}")
        except OSError as e:
            print(f"Error creating output directory '{output_dir}': {e}")
            return False


    print(f"\n--- Starting WAV Generation ---")
    print(f"Sample Rate: {sample_rate} Hz")
    print(f"Crossfade Duration: {crossfade_duration} s")
    print(f"Output File: {output_filename}")

    # Assemble the track (includes per-step normalization now)
    track_audio = assemble_track_from_data(track_data, sample_rate, crossfade_duration)

    if track_audio is None or track_audio.size == 0:
        print("Error: Track assembly failed or resulted in empty audio.")
        return False

    # --- Final Normalization ---
    max_abs_val = np.max(np.abs(track_audio))

    if max_abs_val > 1e-9: # Avoid division by zero for silent tracks
        # --- *** CHANGED: Increase target level *** ---
        target_level = 0.2 # Normalize closer to full scale (e.g., -0.4 dBFS)
        # --- *** End Change *** ---
        scaling_factor = target_level / max_abs_val
        print(f"Normalizing final track (peak value: {max_abs_val:.4f}) to target level: {target_level}")
        normalized_track = track_audio * scaling_factor
        # Optional: Apply a limiter after normalization as a final safety net
        # normalized_track = np.clip(normalized_track, -target_level, target_level)
    else:
        print("Track is silent or near-silent. Skipping final normalization.")
        normalized_track = track_audio # Already silent or zero

    # Convert normalized float audio to 16-bit PCM
    if not np.issubdtype(normalized_track.dtype, np.floating):
         print(f"Warning: Normalized track data type is not float ({normalized_track.dtype}). Attempting conversion.")
         try: normalized_track = normalized_track.astype(np.float64) # Use float64 for precision before scaling
         except Exception as e:
              print(f"Error converting normalized track to float: {e}")
              return False

    # Scale to 16-bit integer range and clip just in case
    track_int16 = np.int16(np.clip(normalized_track * 32767, -32768, 32767))

    # Write WAV file
    try:
        write(output_filename, sample_rate, track_int16)
        print(f"--- WAV Generation Complete ---")
        print(f"Track successfully written to {output_filename}")
        return True
    except Exception as e:
        print(f"Error writing WAV file {output_filename}:")
        traceback.print_exc()
        return False

def generate_single_step_audio_segment(step_data, global_settings, target_duration_seconds, duration_override=None):
    """
    Generates a raw audio segment for a single step, looping or truncating 
    it to fill a target duration.
    
    Args:
        step_data: Dictionary containing step configuration
        global_settings: Dictionary containing global audio settings
        target_duration_seconds: Target duration for the output segment
        duration_override: Optional override for the step's natural duration when generating audio
    """
    if not step_data or not global_settings:
        print("Error: Invalid step_data or global_settings provided.")
        return np.zeros((0, 2), dtype=np.float32)

    try:
        sample_rate = int(global_settings.get("sample_rate", 44100))
        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
    except (ValueError, TypeError):
        print("Error: Invalid sample rate in global_settings.")
        return np.zeros((0, 2), dtype=np.float32)

    target_total_samples = int(target_duration_seconds * sample_rate)
    output_audio_segment = np.zeros((target_total_samples, 2), dtype=np.float32)

    voices_data = step_data.get("voices", [])
    if not voices_data:
        print("Warning: Step has no voices. Returning silence.")
        return output_audio_segment

    # Use duration override if provided, otherwise use step's natural duration
    if duration_override is not None:
        step_generation_duration = float(duration_override)
        print(f"  Using duration override: {step_generation_duration:.2f}s (natural: {step_data.get('duration', 0):.2f}s)")
    else:
        step_generation_duration = float(step_data.get("duration", 0))
    
    if step_generation_duration <= 0:
        print("Warning: Step generation duration is zero or negative. Returning silence.")
        return output_audio_segment    
    step_generation_samples = int(step_generation_duration * sample_rate)
    if step_generation_samples <= 0:
        print("Warning: Step has zero samples. Returning silence.")
        return output_audio_segment

    # Generate one iteration of the step's audio
    single_iteration_audio_mix = np.zeros((step_generation_samples, 2), dtype=np.float32)
    
    print(f"  Generating single iteration for step (Duration: {step_generation_duration:.2f}s, Samples: {step_generation_samples})")
    for i, voice_data in enumerate(voices_data):
        func_name_short = voice_data.get('synth_function_name', 'UnknownFunc')
        print(f"    Generating Voice {i+1}/{len(voices_data)}: {func_name_short}")
        
        voice_audio = generate_voice_audio(voice_data, step_generation_duration, sample_rate, 0.0)
        
        # Add generated audio if valid
        if voice_audio is not None and voice_audio.shape[0] == step_generation_samples and voice_audio.ndim == 2 and voice_audio.shape[1] == 2:
            single_iteration_audio_mix += voice_audio  # Sum voices
        elif voice_audio is not None:
            print(f"    Warning: Voice {i+1} ({func_name_short}) generated audio shape mismatch ({voice_audio.shape} vs {(step_generation_samples, 2)}). Skipping voice.")

    # Normalize the single iteration audio
    step_peak = np.max(np.abs(single_iteration_audio_mix))
    step_normalization_threshold = 0.95  # Normalize to -0.44 dBFS to leave some headroom
    if step_peak > step_normalization_threshold and step_peak > 1e-9:
        print(f"    Normalizing step mix (peak={step_peak:.3f}) down to {step_normalization_threshold:.2f}")
        single_iteration_audio_mix *= (step_normalization_threshold / step_peak)
    elif step_peak <= 1e-9:
        print("    Warning: Step audio is essentially silent.")

    # Fill the output_audio_segment by looping/truncating the single_iteration_audio_mix
    if step_generation_samples == 0:
        print("    Error: Step has zero generation samples, cannot loop.")
        return output_audio_segment

    current_pos_samples = 0
    while current_pos_samples < target_total_samples:
        remaining_samples = target_total_samples - current_pos_samples
        samples_to_copy = min(remaining_samples, step_generation_samples)
        
        output_audio_segment[current_pos_samples:current_pos_samples + samples_to_copy] = single_iteration_audio_mix[:samples_to_copy]
        current_pos_samples += samples_to_copy

    print(f"  Generated single step audio segment: {target_duration_seconds:.2f}s ({output_audio_segment.shape[0]} samples)")
    return output_audio_segment.astype(np.float32)

