

import numpy as np
import numba
import soundfile as sf
from scipy import signal
from joblib import Parallel, delayed
import time

# --- Parameters ---
DEFAULT_SAMPLE_RATE = 44100  # Hz
DEFAULT_LFO_FREQ = 1.0 / 12.0  # Hz, for 12-second period

@numba.jit(nopython=True)
def generate_pink_noise_samples(n_samples):
    """
    Generates pink noise samples with reduced high frequency content.
    This function is optimized with Numba for high performance.
    """
    white = np.random.randn(n_samples).astype(np.float32)
    pink = np.empty_like(white)

    # State variables for the pinking filter (Paul Kellett's method)
    b0, b1, b2, b3, b4, b5 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    for i in range(n_samples):
        w = white[i]
        
        b0 = 0.99886 * b0 + w * 0.0555179
        b1 = 0.99332 * b1 + w * 0.0750759
        b2 = 0.96900 * b2 + w * 0.1538520
        b3 = 0.86650 * b3 + w * 0.3104856
        b4 = 0.55000 * b4 + w * 0.5329522
        b5 = -0.7616 * b5 - w * 0.0168980
        
        pink_val = b0 + b1 + b2 + b3 + b4 + b5
        pink[i] = pink_val * 0.11
        
    return pink


@numba.jit(nopython=True)
def generate_brown_noise_samples(n_samples):
    """
    Generate brown (red) noise samples.
    This function is optimized with Numba for high performance.
    """
    white = np.random.randn(n_samples).astype(np.float32)
    brown = np.cumsum(white)
    max_abs = np.max(np.abs(brown)) + 1e-8
    return (brown / max_abs).astype(np.float32)


def _apply_deep_swept_notches_single_phase(input_signal, sample_rate, lfo_freq,
                                           min_freq=1000, max_freq=10000,
                                           num_notches=1, notch_spacing_ratio=1.1,
                                           notch_q=30, cascade_count=10,
                                           phase_offset=90, lfo_waveform='sine'):
    """
    Apply very deep swept notch filters without harmonics for a single LFO
    phase. This function is the core processing unit and is designed to be
    run in parallel for different channels or phase offsets.
    """
    n_samples = len(input_signal)
    output = input_signal.copy()
    
    t = np.arange(n_samples) / sample_rate
    
    # --- LFO Generation ---
    # Generate the LFO signal based on the chosen waveform
    if lfo_waveform.lower() == 'triangle':
        # Triangle wave oscillates linearly between -1 and 1
        lfo = signal.sawtooth(2 * np.pi * lfo_freq * t + phase_offset, width=0.5)
    elif lfo_waveform.lower() == 'sine':
        # Cosine wave (a phase-shifted sine) oscillates smoothly between -1 and 1
        lfo = np.cos(2 * np.pi * lfo_freq * t + phase_offset)
    else:
        raise ValueError(f"Unsupported LFO waveform: {lfo_waveform}. Choose 'sine' or 'triangle'.")

    # Linear frequency sweep for smoother motion
    center_freq = (min_freq + max_freq) / 2
    freq_range = (max_freq - min_freq) / 2
    base_freq_sweep = center_freq + freq_range * lfo
    
    # Process in overlapping blocks for smooth transitions (Overlap-Add method)
    block_size = 4096
    hop_size = block_size // 2  # 50% overlap
    window = np.hanning(block_size)
    
    output_accumulator = np.zeros(n_samples + block_size)
    window_accumulator = np.zeros(n_samples + block_size)
    
    # Process each block
    num_blocks = (n_samples + hop_size - 1) // hop_size
    
    for block_idx in range(num_blocks):
        start_idx = block_idx * hop_size
        end_idx = min(start_idx + block_size, n_samples)
        actual_block_size = end_idx - start_idx
        
        if actual_block_size < 100:
            continue
        
        block = np.zeros(block_size)
        block[:actual_block_size] = output[start_idx:end_idx]
        
        windowed_block = block * window

        # Get the representative center frequency for this block
        block_center_idx = start_idx + actual_block_size // 2
        center_freq_for_block = base_freq_sweep[block_center_idx]
        
        filtered_block = windowed_block.copy()
        
        # Apply multiple notches at different frequencies
        for notch_idx in range(num_notches):
            notch_freq = center_freq_for_block * (notch_spacing_ratio ** notch_idx)
            
            if notch_freq >= sample_rate * 0.49: # Nyquist limit safety
                continue
            
            # Applying the same notch multiple times (cascading) makes it deeper.
            # Using filtfilt ensures zero phase distortion.
            for _ in range(cascade_count):
                try:
                    b, a = signal.iirnotch(notch_freq, notch_q, sample_rate)
                    filtered_block = signal.filtfilt(b, a, filtered_block)
                except ValueError: # Filter design can fail if freq is too high/low
                    continue
        
        # Accumulate output with overlap-add
        output_accumulator[start_idx:start_idx + block_size] += filtered_block
        window_accumulator[start_idx:start_idx + block_size] += window
    
    # Normalize by window accumulation to avoid amplitude bumps
    valid_idx = window_accumulator > 1e-8
    output_accumulator[valid_idx] /= window_accumulator[valid_idx]
    
    return output_accumulator[:n_samples]


def apply_deep_swept_notches(input_signal, sample_rate, lfo_freq,
                            min_freq=1000, max_freq=10000,
                            num_notches=1, notch_spacing_ratio=1.1,
                            notch_q=30, cascade_count=10,
                            phase_offset=0, extra_phase_offset=0.0,
                            lfo_waveform='sine'):
    """
    Wrapper function to apply deep swept notch filters. It can apply a second
    set of notches with an additional phase offset if specified.
    """
    output = _apply_deep_swept_notches_single_phase(
        input_signal, sample_rate, lfo_freq, min_freq, max_freq,
        num_notches, notch_spacing_ratio, notch_q, cascade_count, phase_offset,
        lfo_waveform=lfo_waveform
    )

    if extra_phase_offset:
        output = _apply_deep_swept_notches_single_phase(
            output, sample_rate, lfo_freq, min_freq, max_freq,
            num_notches, notch_spacing_ratio, notch_q, cascade_count,
            phase_offset + extra_phase_offset, lfo_waveform=lfo_waveform
        )

    return output


def generate_swept_notch_pink_sound(
    filename="swept_notch_pink_sound.wav",
    duration_seconds=60,
    sample_rate=DEFAULT_SAMPLE_RATE,
    lfo_freq=DEFAULT_LFO_FREQ,
    min_freq=1000,
    max_freq=10000,
    num_notches=1,
    notch_spacing_ratio=1.25,
    notch_q=25,
    cascade_count=10,
    lfo_phase_offset_deg=90,
    intra_phase_offset_deg=0,
    input_audio_path=None,
    noise_type="pink",
    lfo_waveform="sine"
):
    """
    Generates stereo noise with deep swept notches, optimized for speed by
    processing left and right channels in parallel.

    Parameters
    ----------
    lfo_waveform : str, optional
        The shape of the LFO wave. Supported values are 'sine' (default)
        and 'triangle'.
        
    NOTE: This script now requires the 'joblib' library.
    Install it with: pip install joblib
    """
    print(f"Starting Deep Swept Notch generation for '{filename}'...")
    print(f"LFO Waveform: {lfo_waveform}")
    # ... (print other parameters)

    # --- Step 1: Generate or load input audio ---
    start_time = time.time()
    if input_audio_path is None:
        num_samples = int(duration_seconds * sample_rate)
        print(f"Step 1/4: Generating base {noise_type} noise...")
        if noise_type.lower() == "brown":
            noise_mono = generate_brown_noise_samples(num_samples)
        else:
            noise_mono = generate_pink_noise_samples(num_samples)
        
        # Pre-conditioning filters for a warmer sound
        b_warmth, a_warmth = signal.butter(1, 10000, btype='low', fs=sample_rate)
        noise_mono = signal.filtfilt(b_warmth, a_warmth, noise_mono)
        b_hpf, a_hpf = signal.butter(2, 50, btype='high', fs=sample_rate)
        noise_mono = signal.filtfilt(b_hpf, a_hpf, noise_mono)
        
        noise_mono = noise_mono / (np.max(np.abs(noise_mono)) + 1e-8) * 0.8
        input_signal = noise_mono
    else:
        # ... (loading logic remains the same)
        print(f"Step 1/4: Loading input audio from '{input_audio_path}'...")
        data, in_sr = sf.read(input_audio_path)
        if data.ndim > 1:
            input_signal = data[:, 0]
        else:
            input_signal = data
        input_signal = input_signal / (np.max(np.abs(input_signal)) + 1e-8) * 0.8

    # Calculate input RMS for later volume correction
    print("Step 2/4: Calculating input signal RMS for volume compensation...")
    rms_in = np.sqrt(np.mean(input_signal**2))
    if rms_in < 1e-8:
        rms_in = 1e-8 # Avoid division by zero for silent inputs

    # --- Step 3: Apply deep swept notches in parallel ---
    print("Step 3/4: Applying deep swept notches (in parallel for stereo)...")
    
    intra_phase_rad = np.deg2rad(intra_phase_offset_deg)
    right_channel_phase_offset_rad = np.deg2rad(lfo_phase_offset_deg)

    tasks = [
        delayed(apply_deep_swept_notches)(
            input_signal, sample_rate, lfo_freq, min_freq, max_freq,
            num_notches, notch_spacing_ratio, notch_q, cascade_count,
            phase_offset=0, extra_phase_offset=intra_phase_rad,
            lfo_waveform=lfo_waveform
        ),
        delayed(apply_deep_swept_notches)(
            input_signal, sample_rate, lfo_freq, min_freq, max_freq,
            num_notches, notch_spacing_ratio, notch_q, cascade_count,
            phase_offset=right_channel_phase_offset_rad, extra_phase_offset=intra_phase_rad,
            lfo_waveform=lfo_waveform
        )
    ]

    with Parallel(n_jobs=2, backend='loky') as parallel:
        results = parallel(tasks)
    
    left_output, right_output = results

    # --- Step 4: Final processing and save ---
    print("Step 4/4: Applying volume correction and saving...")
    
    # --- VOLUME CORRECTION ---
    rms_left = np.sqrt(np.mean(left_output**2))
    rms_right = np.sqrt(np.mean(right_output**2))

    if rms_left > 1e-8:
        left_output *= (rms_in / rms_left)
    
    if rms_right > 1e-8:
        right_output *= (rms_in / rms_right)

    stereo_output = np.stack((left_output, right_output), axis=-1)
    
    max_val = np.max(np.abs(stereo_output))
    if max_val > 0:
        stereo_output = stereo_output / max_val * 0.95
    
    try:
        sf.write(filename, stereo_output, sample_rate, subtype='PCM_16')
        total_time = time.time() - start_time
        print(f"\nSuccessfully generated and saved to '{filename}' in {total_time:.2f} seconds.")
    except Exception as e:
        print(f"Error saving audio file: {e}")


# --- Main execution ---
if __name__ == '__main__':
    
    # Example 1: Generate with the new Triangle LFO
    generate_swept_notch_pink_sound(
        filename="swept_notch_triangle_lfo.wav",
        duration_seconds=60,
        sample_rate=44100,
        lfo_freq=1.0/12.0,
        min_freq=1000,
        max_freq=10000,
        num_notches=1,
        notch_q=25,
        cascade_count=20,
        lfo_phase_offset_deg=90,
        lfo_waveform='triangle'  # Specify the new waveform here
    )

    # Example 2: Generate with the classic Sine LFO
    generate_swept_notch_pink_sound(
        filename="swept_notch_sine_lfo.wav",
        duration_seconds=60,
        sample_rate=44100,
        lfo_freq=1.0/12.0,
        min_freq=1000,
        max_freq=10000,
        num_notches=1,
        notch_q=25,
        cascade_count=20,
        lfo_phase_offset_deg=90,
        lfo_waveform='sine' # Explicitly using the default
    )

