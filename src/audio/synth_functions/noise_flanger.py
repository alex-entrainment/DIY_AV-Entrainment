
import numpy as np
import numba
import soundfile as sf
from scipy import signal
from joblib import Parallel, delayed
import time
import tempfile

# --- Parameters ---
DEFAULT_SAMPLE_RATE = 44100  # Hz
DEFAULT_LFO_FREQ = 1.0 / 12.0  # Hz, for 12-second period


def _compute_rms_memmap(arr, chunk_size=1_000_000):
    """Compute RMS of a potentially memory-mapped array in chunks."""
    n = len(arr)
    if n == 0:
        return 0.0
    sq_sum = 0.0
    for i in range(0, n, chunk_size):
        chunk = arr[i : i + chunk_size]
        sq_sum += np.sum(chunk.astype(np.float64) ** 2)
    return np.sqrt(sq_sum / n)

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


def _apply_deep_swept_notches_single_phase(
    input_signal,
    sample_rate,
    lfo_freq,
    filter_sweeps,
    notch_q=30,
    cascade_count=10,
    phase_offset=90,
    lfo_waveform='sine',
    use_memmap=False,
):
    """
    Apply one or more deep swept notch filters for a single LFO phase.
    This function is the core processing unit.
    """
    n_samples = len(input_signal)

    if use_memmap:
        tmp_output = tempfile.NamedTemporaryFile(delete=False)
        output = np.memmap(tmp_output.name, dtype=np.float32, mode='w+', shape=n_samples)
        output[:] = input_signal[:]
    else:
        output = input_signal.copy()
    
    t = np.arange(n_samples) / sample_rate
    
    # --- LFO Generation ---
    if lfo_waveform.lower() == 'triangle':
        lfo = signal.sawtooth(2 * np.pi * lfo_freq * t + phase_offset, width=0.5)
    elif lfo_waveform.lower() == 'sine':
        lfo = np.cos(2 * np.pi * lfo_freq * t + phase_offset)
    else:
        raise ValueError(f"Unsupported LFO waveform: {lfo_waveform}. Choose 'sine' or 'triangle'.")

    # --- Generate Frequency Sweeps for each filter ---
    base_freq_sweeps = []
    for min_freq, max_freq in filter_sweeps:
        center_freq = (min_freq + max_freq) / 2
        freq_range = (max_freq - min_freq) / 2
        base_freq_sweeps.append(center_freq + freq_range * lfo)

    # Normalize parameter shapes
    if isinstance(notch_q, (int, float)):
        notch_qs = [float(notch_q)] * len(filter_sweeps)
    else:
        notch_qs = list(notch_q)
    if isinstance(cascade_count, int):
        cascade_counts = [int(cascade_count)] * len(filter_sweeps)
    else:
        cascade_counts = list(cascade_count)
    if len(notch_qs) != len(filter_sweeps) or len(cascade_counts) != len(filter_sweeps):
        raise ValueError("Length of notch_q and cascade_count must match number of filter_sweeps")

    # Process in overlapping blocks for smooth transitions (Overlap-Add method)
    block_size = 4096
    hop_size = block_size // 2
    window = np.hanning(block_size)
    
    if use_memmap:
        tmp_out_acc = tempfile.NamedTemporaryFile(delete=False)
        tmp_win_acc = tempfile.NamedTemporaryFile(delete=False)
        output_accumulator = np.memmap(tmp_out_acc.name, dtype=np.float32, mode='w+', shape=n_samples + block_size)
        window_accumulator = np.memmap(tmp_win_acc.name, dtype=np.float32, mode='w+', shape=n_samples + block_size)
    else:
        output_accumulator = np.zeros(n_samples + block_size, dtype=np.float32)
        window_accumulator = np.zeros(n_samples + block_size, dtype=np.float32)
    
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
        filtered_block = windowed_block.copy()

        block_center_idx = start_idx + actual_block_size // 2

        # --- Apply each independent notch filter sweep ---
        for sweep_idx, sweep in enumerate(base_freq_sweeps):
            notch_freq = sweep[block_center_idx]

            if notch_freq >= sample_rate * 0.49:
                continue

            q_val = notch_qs[sweep_idx]
            cascades = cascade_counts[sweep_idx]
            for _ in range(cascades):
                try:
                    b, a = signal.iirnotch(notch_freq, q_val, sample_rate)
                    # filtfilt can produce extremely loud transients with high Q
                    # settings. Using lfilter prevents those spikes at the cost
                    # of phase shifts which are not perceptible in this use case.
                    filtered_block = signal.lfilter(b, a, filtered_block)
                except ValueError:
                    continue
        
        output_accumulator[start_idx:start_idx + block_size] += filtered_block
        window_accumulator[start_idx:start_idx + block_size] += window
    
    valid_idx = window_accumulator > 1e-8
    output_accumulator[valid_idx] /= window_accumulator[valid_idx]
    
    return output_accumulator[:n_samples]


def apply_deep_swept_notches(
    input_signal,
    sample_rate,
    lfo_freq,
    filter_sweeps,
    notch_q=30,
    cascade_count=10,
    phase_offset=0,
    extra_phase_offset=0.0,
    lfo_waveform='sine',
    use_memmap=False,
):
    """
    Wrapper to apply deep swept notch filters.
    """
    output = _apply_deep_swept_notches_single_phase(
        input_signal,
        sample_rate,
        lfo_freq,
        filter_sweeps,
        notch_q,
        cascade_count,
        phase_offset,
        lfo_waveform,
        use_memmap=use_memmap,
    )

    if extra_phase_offset:
        output = _apply_deep_swept_notches_single_phase(
            output,
            sample_rate,
            lfo_freq,
            filter_sweeps,
            notch_q,
            cascade_count,
            phase_offset + extra_phase_offset,
            lfo_waveform,
            use_memmap=use_memmap,
        )

    return output


def generate_swept_notch_pink_sound(
    filename="swept_notch_sound.wav",
    duration_seconds=60,
    sample_rate=DEFAULT_SAMPLE_RATE,
    lfo_freq=DEFAULT_LFO_FREQ,
    filter_sweeps=None,
    notch_q=25,
    cascade_count=10,
    lfo_phase_offset_deg=90,
    intra_phase_offset_deg=0,
    input_audio_path=None,
    noise_type="pink",
    lfo_waveform="sine",
    memory_efficient=False,
    n_jobs=2,
):
    """
    Generates stereo noise with one or more independent deep swept notches.

    Parameters
    ----------
    filter_sweeps : list of tuples, optional
        A list defining the frequency sweeps. Each tuple should be
        (min_freq, max_freq) for one notch.
        Example: [(500, 1000), (1850, 3350)]
    notch_q : int, float, or sequence, optional
        Q factor(s) for the notch filters. Provide either a single value or
        one value per sweep.
    cascade_count : int or sequence, optional
        Number of cascaded filters for each sweep. Provide a single value or
        one value per sweep.
    memory_efficient : bool, optional
        If True, uses disk-backed arrays and processes channels sequentially to
        reduce RAM usage. Defaults to False.
    n_jobs : int, optional
        Number of parallel jobs. Setting this to 1 can further lower memory
        load when generating very long files.
    """
    if filter_sweeps is None:
        filter_sweeps = [(1000, 10000)]  # Default to old behavior

    # Normalize parameter shapes
    if isinstance(notch_q, (int, float)):
        notch_q = [float(notch_q)] * len(filter_sweeps)
    else:
        notch_q = list(notch_q)
    if isinstance(cascade_count, int):
        cascade_count = [int(cascade_count)] * len(filter_sweeps)
    else:
        cascade_count = list(cascade_count)
    if len(notch_q) != len(filter_sweeps) or len(cascade_count) != len(filter_sweeps):
        raise ValueError(
            "Length of notch_q and cascade_count must match number of filter_sweeps"
        )

    print(f"Starting Deep Swept Notch generation for '{filename}'...")
    print(f"LFO Waveform: {lfo_waveform}, Freq: {lfo_freq:.3f} Hz")
    print("Filter Sweeps:")
    for i, (min_f, max_f) in enumerate(filter_sweeps):
        print(f"  - Sweep {i+1}: {min_f} Hz -> {max_f} Hz")

    # --- Step 1: Generate or load input audio ---
    start_time = time.time()
    if input_audio_path is None:
        num_samples = int(duration_seconds * sample_rate)
        print(f"Step 1/4: Generating base {noise_type} noise...")
        if memory_efficient:
            tmp_input = tempfile.NamedTemporaryFile(delete=False)
            input_signal = np.memmap(tmp_input.name, dtype=np.float32, mode='w+', shape=num_samples)
            if noise_type.lower() == "brown":
                input_signal[:] = generate_brown_noise_samples(num_samples)
            else:
                input_signal[:] = generate_pink_noise_samples(num_samples)
        else:
            if noise_type.lower() == "brown":
                input_signal = generate_brown_noise_samples(num_samples)
            else:
                input_signal = generate_pink_noise_samples(num_samples)
    else:
        print(f"Step 1/4: Loading input audio from '{input_audio_path}'...")
        data, _ = sf.read(input_audio_path)
        input_signal = data[:, 0] if data.ndim > 1 else data

    # Pre-conditioning and normalization
    b_warmth, a_warmth = signal.butter(1, 10000, btype='low', fs=sample_rate)
    input_signal = signal.filtfilt(b_warmth, a_warmth, input_signal)
    b_hpf, a_hpf = signal.butter(2, 50, btype='high', fs=sample_rate)
    input_signal = signal.filtfilt(b_hpf, a_hpf, input_signal)
    input_signal = input_signal / (np.max(np.abs(input_signal)) + 1e-8) * 0.8
    
    # Calculate input RMS for later volume correction
    print("Step 2/4: Calculating input signal RMS for volume compensation...")
    rms_in = np.sqrt(np.mean(input_signal**2))
    if rms_in < 1e-8:
        rms_in = 1e-8

    # --- Step 3: Apply deep swept notches in parallel ---
    print("Step 3/4: Applying deep swept notches (in parallel for stereo)...")
    
    intra_phase_rad = np.deg2rad(intra_phase_offset_deg)
    right_channel_phase_offset_rad = np.deg2rad(lfo_phase_offset_deg)

    tasks = [
        delayed(apply_deep_swept_notches)(
            input_signal,
            sample_rate,
            lfo_freq,
            filter_sweeps,
            notch_q,
            cascade_count,
            0,
            intra_phase_rad,
            lfo_waveform,
            use_memmap=memory_efficient,
        ),
        delayed(apply_deep_swept_notches)(
            input_signal,
            sample_rate,
            lfo_freq,
            filter_sweeps,
            notch_q,
            cascade_count,
            right_channel_phase_offset_rad,
            intra_phase_rad,
            lfo_waveform,
            use_memmap=memory_efficient,
        ),
    ]

    if memory_efficient and n_jobs > 1:
        n_jobs = 1

    with Parallel(n_jobs=n_jobs, backend='loky') as parallel:
        results = parallel(tasks)

    left_output, right_output = results

    # --- Step 4: Final processing and save ---
    print("Step 4/4: Applying volume correction and saving...")
    
    if memory_efficient:
        rms_left = _compute_rms_memmap(left_output)
    else:
        rms_left = np.sqrt(np.mean(left_output**2))
    if rms_left > 1e-8:
        left_output *= (rms_in / rms_left)

    if memory_efficient:
        rms_right = _compute_rms_memmap(right_output)
    else:
        rms_right = np.sqrt(np.mean(right_output**2))
    if rms_right > 1e-8:
        right_output *= (rms_in / rms_right)

    if memory_efficient:
        with sf.SoundFile(filename, mode='w', samplerate=sample_rate, channels=2, subtype='PCM_16') as f:
            block = 1_000_000
            total = len(left_output)
            for i in range(0, total, block):
                chunk_left = left_output[i : i + block]
                chunk_right = right_output[i : i + block]
                stereo_chunk = np.stack((chunk_left, chunk_right), axis=-1)
                max_val = np.max(np.abs(stereo_chunk))
                if max_val > 0.95:
                    stereo_chunk = np.clip(stereo_chunk, -0.95, 0.95)
                elif max_val > 0:
                    stereo_chunk = stereo_chunk / max_val * 0.95
                f.write(stereo_chunk)
        total_time = time.time() - start_time
        print(f"\nSuccessfully generated and saved to '{filename}' in {total_time:.2f} seconds.")
    else:
        stereo_output = np.stack((left_output, right_output), axis=-1)

        max_val = np.max(np.abs(stereo_output))
        if max_val > 0.95:
            stereo_output = np.clip(stereo_output, -0.95, 0.95)
        elif max_val > 0:
            stereo_output = stereo_output / max_val * 0.95

        try:
            sf.write(filename, stereo_output, sample_rate, subtype='PCM_16')
            total_time = time.time() - start_time
            print(f"\nSuccessfully generated and saved to '{filename}' in {total_time:.2f} seconds.")
        except Exception as e:
            print(f"Error saving audio file: {e}")


# --- Main execution ---
if __name__ == '__main__':
    
    # --- Configuration to match the spectrogram image ---
    # Define the two separate sweeps as a list of tuples
    dual_sweeps_config = [
        (500, 1000),      # Lower sweep: 500 Hz to 1000 Hz
        (1850, 3350)      # Upper sweep: 1850 Hz to 3350 Hz
    ]

    generate_swept_notch_pink_sound(
        filename="dual_sweep_triangle_lfo.wav",
        duration_seconds=60,
        sample_rate=44100,
        lfo_freq=1.0/12.0,
        filter_sweeps=dual_sweeps_config, # Use the new multi-sweep config
        notch_q=40,                       # A slightly higher Q might look closer to the image
        cascade_count=15,
        lfo_phase_offset_deg=90,
        lfo_waveform='triangle'           # Use triangle wave for linear sweeps
    )

