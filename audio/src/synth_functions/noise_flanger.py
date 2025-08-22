
import numpy as np
import numba
import soundfile as sf
from scipy import signal
from joblib import Parallel, delayed
import time
import tempfile
from .common import (
    calculate_transition_alpha,
    blue_noise,
    purple_noise,
    green_noise,
    deep_brown_noise,
)

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


def generate_noise_samples(n_samples, noise_type, sample_rate=DEFAULT_SAMPLE_RATE):
    nt = noise_type.lower()
    if nt == "pink":
        return generate_pink_noise_samples(n_samples)
    if nt in ("brown", "red"):
        return generate_brown_noise_samples(n_samples)
    if nt == "deep brown":
        return deep_brown_noise(n_samples).astype(np.float32)
    if nt == "blue":
        return blue_noise(n_samples).astype(np.float32)
    if nt == "purple":
        return purple_noise(n_samples).astype(np.float32)
    if nt == "green":
        return green_noise(n_samples, fs=sample_rate).astype(np.float32)
    return np.random.randn(n_samples).astype(np.float32)


def triangle_wave_varying(freq_array, t, sample_rate=44100):
    """Generate a triangle wave with a time-varying frequency."""
    freq_array = np.maximum(np.asarray(freq_array, dtype=float), 1e-9)
    t = np.asarray(t, dtype=float)
    if len(t) <= 1:
        return np.zeros_like(t)
    dt = np.diff(t, prepend=t[0])
    phase = 2 * np.pi * np.cumsum(freq_array * dt)
    return signal.sawtooth(phase, width=0.5)


def _apply_deep_swept_notches_varying(
    input_signal,
    sample_rate,
    lfo_array,
    min_freq_arrays,
    max_freq_arrays,
    notch_q_arrays,
    cascade_count_arrays,
    use_memmap=True,
):
    """Apply swept notch filters where parameters vary over time."""
    n_samples = len(input_signal)

    if use_memmap:
        tmp_output = tempfile.NamedTemporaryFile(delete=False)
        tmp_output_name = tmp_output.name
        tmp_output.close()
        output = np.memmap(tmp_output_name, dtype=np.float32, mode="w+", shape=n_samples)
        output[:] = input_signal[:]
    else:
        output = input_signal.copy()

    block_size = 4096
    hop_size = block_size // 2
    window = np.hanning(block_size)

    if use_memmap:
        tmp_out_acc = tempfile.NamedTemporaryFile(delete=False)
        tmp_win_acc = tempfile.NamedTemporaryFile(delete=False)
        out_acc_name = tmp_out_acc.name
        win_acc_name = tmp_win_acc.name
        tmp_out_acc.close()
        tmp_win_acc.close()
        output_accumulator = np.memmap(out_acc_name, dtype=np.float32, mode="w+", shape=n_samples + block_size)
        window_accumulator = np.memmap(win_acc_name, dtype=np.float32, mode="w+", shape=n_samples + block_size)
    else:
        output_accumulator = np.zeros(n_samples + block_size, dtype=np.float32)
        window_accumulator = np.zeros(n_samples + block_size, dtype=np.float32)

    num_blocks = (n_samples + hop_size - 1) // hop_size
    num_sweeps = len(min_freq_arrays)

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

        center_idx = start_idx + actual_block_size // 2

        for i in range(num_sweeps):
            min_f = min_freq_arrays[i][center_idx]
            max_f = max_freq_arrays[i][center_idx]
            center_freq = (min_f + max_f) / 2.0
            freq_range = (max_f - min_f) / 2.0
            notch_freq = center_freq + freq_range * lfo_array[center_idx]

            if notch_freq >= sample_rate * 0.49:
                continue

            q_val = float(notch_q_arrays[i][center_idx])
            casc = int(round(cascade_count_arrays[i][center_idx]))
            casc = max(1, casc)
            for _ in range(casc):
                try:
                    b, a = signal.iirnotch(notch_freq, q_val, sample_rate)
                    filtered_block = signal.lfilter(b, a, filtered_block)
                except ValueError:
                    continue

        output_accumulator[start_idx:start_idx + block_size] += filtered_block
        window_accumulator[start_idx:start_idx + block_size] += window

    valid_idx = window_accumulator > 1e-8
    output_accumulator[valid_idx] /= window_accumulator[valid_idx]
    return output_accumulator[:n_samples]


def _apply_deep_swept_notches_single_phase(
    input_signal,
    sample_rate,
    lfo_freq,
    filter_sweeps,
    notch_q=30,
    cascade_count=10,
    phase_offset=90,
    lfo_waveform='sine',
    use_memmap=True,
):
    """
    Apply one or more deep swept notch filters for a single LFO phase.
    This function is the core processing unit.
    """
    n_samples = len(input_signal)

    if use_memmap:
        tmp_output = tempfile.NamedTemporaryFile(delete=False)
        tmp_output_name = tmp_output.name
        tmp_output.close()
        output = np.memmap(tmp_output_name, dtype=np.float32, mode='w+', shape=n_samples)
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
        out_acc_name = tmp_out_acc.name
        win_acc_name = tmp_win_acc.name
        tmp_out_acc.close()
        tmp_win_acc.close()
        output_accumulator = np.memmap(out_acc_name, dtype=np.float32, mode='w+', shape=n_samples + block_size)
        window_accumulator = np.memmap(win_acc_name, dtype=np.float32, mode='w+', shape=n_samples + block_size)
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


def _generate_swept_notch_arrays(
    duration_seconds,
    sample_rate,
    lfo_freq,
    filter_sweeps,
    notch_q,
    cascade_count,
    lfo_phase_offset_deg,
    intra_phase_offset_deg,
    input_audio_path,
    noise_type,
    lfo_waveform,
    memory_efficient,
    n_jobs,
):
    """Internal helper to generate swept notch noise and return stereo array."""

    if filter_sweeps is None:
        filter_sweeps = [(1000, 10000)]

    if isinstance(notch_q, (int, float)):
        notch_q = [float(notch_q)] * len(filter_sweeps)
    else:
        notch_q = list(notch_q)
    if isinstance(cascade_count, int):
        cascade_count = [int(cascade_count)] * len(filter_sweeps)
    else:
        cascade_count = list(cascade_count)
    if len(notch_q) != len(filter_sweeps) or len(cascade_count) != len(filter_sweeps):
        raise ValueError("Length of notch_q and cascade_count must match number of filter_sweeps")

    start_time = time.time()
    if input_audio_path is None:
        num_samples = int(duration_seconds * sample_rate)
        if memory_efficient:
            tmp_input = tempfile.NamedTemporaryFile(delete=False)
            tmp_input_name = tmp_input.name
            tmp_input.close()
            input_signal = np.memmap(tmp_input_name, dtype=np.float32, mode='w+', shape=num_samples)
            input_signal[:] = generate_noise_samples(num_samples, noise_type, sample_rate)
        else:
            input_signal = generate_noise_samples(num_samples, noise_type, sample_rate)
    else:
        data, _ = sf.read(input_audio_path)
        input_signal = data[:, 0] if data.ndim > 1 else data

    b_warmth, a_warmth = signal.butter(1, 10000, btype='low', fs=sample_rate)
    input_signal = signal.filtfilt(b_warmth, a_warmth, input_signal)
    b_hpf, a_hpf = signal.butter(2, 50, btype='high', fs=sample_rate)
    input_signal = signal.filtfilt(b_hpf, a_hpf, input_signal)
    input_signal = input_signal / (np.max(np.abs(input_signal)) + 1e-8) * 0.8

    rms_in = np.sqrt(np.mean(input_signal**2))
    if rms_in < 1e-8:
        rms_in = 1e-8

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

    stereo_output = np.stack((left_output, right_output), axis=-1)

    max_val = np.max(np.abs(stereo_output))
    if max_val > 0.95:
        stereo_output = np.clip(stereo_output, -0.95, 0.95)
    elif max_val > 0:
        stereo_output = stereo_output / max_val * 0.95

    return stereo_output, time.time() - start_time



def _generate_swept_notch_arrays_transition(
    duration_seconds,
    sample_rate,
    start_lfo_freq,
    end_lfo_freq,
    start_filter_sweeps,
    end_filter_sweeps,
    start_notch_q,
    end_notch_q,
    start_cascade_count,
    end_cascade_count,
    start_lfo_phase_offset_deg,
    end_lfo_phase_offset_deg,
    start_intra_phase_offset_deg,
    end_intra_phase_offset_deg,
    input_audio_path,
    noise_type,
    lfo_waveform,
    initial_offset,
    post_offset,
    transition_curve,
    memory_efficient,
    n_jobs,
):
    """Internal helper generating swept notch noise with parameter transitions."""

    if start_filter_sweeps is None:
        start_filter_sweeps = [(1000, 10000)]
    if end_filter_sweeps is None:
        end_filter_sweeps = start_filter_sweeps

    num_sweeps = len(start_filter_sweeps)

    if isinstance(start_notch_q, (int, float)):
        start_notch_q = [float(start_notch_q)] * num_sweeps
    else:
        start_notch_q = list(start_notch_q)
    if isinstance(end_notch_q, (int, float)):
        end_notch_q = [float(end_notch_q)] * num_sweeps
    else:
        end_notch_q = list(end_notch_q)

    if isinstance(start_cascade_count, int):
        start_cascade_count = [int(start_cascade_count)] * num_sweeps
    else:
        start_cascade_count = list(start_cascade_count)
    if isinstance(end_cascade_count, int):
        end_cascade_count = [int(end_cascade_count)] * num_sweeps
    else:
        end_cascade_count = list(end_cascade_count)

    if (
        len(start_notch_q) != num_sweeps
        or len(end_notch_q) != num_sweeps
        or len(start_cascade_count) != num_sweeps
        or len(end_cascade_count) != num_sweeps
    ):
        raise ValueError("Length mismatch between sweep parameters")

    start_time = time.time()
    if input_audio_path is None:
        num_samples = int(duration_seconds * sample_rate)
        if memory_efficient:
            tmp_input = tempfile.NamedTemporaryFile(delete=False)
            tmp_input_name = tmp_input.name
            tmp_input.close()
            input_signal = np.memmap(tmp_input_name, dtype=np.float32, mode="w+", shape=num_samples)
            input_signal[:] = generate_noise_samples(num_samples, noise_type, sample_rate)
        else:
            input_signal = generate_noise_samples(num_samples, noise_type, sample_rate)
    else:
        data, _ = sf.read(input_audio_path)
        input_signal = data[:, 0] if data.ndim > 1 else data

    b_warmth, a_warmth = signal.butter(1, 10000, btype="low", fs=sample_rate)
    input_signal = signal.filtfilt(b_warmth, a_warmth, input_signal)
    b_hpf, a_hpf = signal.butter(2, 50, btype="high", fs=sample_rate)
    input_signal = signal.filtfilt(b_hpf, a_hpf, input_signal)
    input_signal = input_signal / (np.max(np.abs(input_signal)) + 1e-8) * 0.8

    rms_in = np.sqrt(np.mean(input_signal ** 2))
    if rms_in < 1e-8:
        rms_in = 1e-8

    num_samples = len(input_signal)
    t = np.arange(num_samples) / sample_rate
    alpha = calculate_transition_alpha(
        duration_seconds, sample_rate, initial_offset, post_offset, transition_curve
    )
    if len(alpha) != num_samples:
        alpha = np.interp(
            np.linspace(0, 1, num_samples),
            np.linspace(0, 1, len(alpha)),
            alpha,
        )

    lfo_freq_array = start_lfo_freq + (end_lfo_freq - start_lfo_freq) * alpha

    phase_base = np.cumsum(2 * np.pi * lfo_freq_array / sample_rate)

    if lfo_waveform.lower() == "triangle":
        base_wave_fn = lambda ph: signal.sawtooth(ph, width=0.5)
    elif lfo_waveform.lower() == "sine":
        base_wave_fn = np.cos
    else:
        raise ValueError(
            f"Unsupported LFO waveform: {lfo_waveform}. Choose 'sine' or 'triangle'."
        )

    right_phase_rad = np.deg2rad(
        start_lfo_phase_offset_deg
        + (end_lfo_phase_offset_deg - start_lfo_phase_offset_deg) * alpha
    )
    intra_phase_rad = np.deg2rad(
        start_intra_phase_offset_deg
        + (end_intra_phase_offset_deg - start_intra_phase_offset_deg) * alpha
    )

    lfo_left = base_wave_fn(phase_base)
    lfo_left_2 = base_wave_fn(phase_base + intra_phase_rad)
    lfo_right = base_wave_fn(phase_base + right_phase_rad)
    lfo_right_2 = base_wave_fn(phase_base + right_phase_rad + intra_phase_rad)

    min_arrays = []
    max_arrays = []
    q_arrays = []
    casc_arrays = []
    for idx in range(num_sweeps):
        s_min, s_max = start_filter_sweeps[idx]
        e_min, e_max = end_filter_sweeps[idx]
        min_arrays.append(s_min + (e_min - s_min) * alpha)
        max_arrays.append(s_max + (e_max - s_max) * alpha)
        q_arrays.append(start_notch_q[idx] + (end_notch_q[idx] - start_notch_q[idx]) * alpha)
        casc_arrays.append(start_cascade_count[idx] + (end_cascade_count[idx] - start_cascade_count[idx]) * alpha)

    left_output = _apply_deep_swept_notches_varying(
        input_signal,
        sample_rate,
        lfo_left,
        min_arrays,
        max_arrays,
        q_arrays,
        casc_arrays,
        use_memmap=memory_efficient,
    )
    left_output = _apply_deep_swept_notches_varying(
        left_output,
        sample_rate,
        lfo_left_2,
        min_arrays,
        max_arrays,
        q_arrays,
        casc_arrays,
        use_memmap=memory_efficient,
    )

    right_output = _apply_deep_swept_notches_varying(
        input_signal,
        sample_rate,
        lfo_right,
        min_arrays,
        max_arrays,
        q_arrays,
        casc_arrays,
        use_memmap=memory_efficient,
    )
    right_output = _apply_deep_swept_notches_varying(
        right_output,
        sample_rate,
        lfo_right_2,
        min_arrays,
        max_arrays,
        q_arrays,
        casc_arrays,
        use_memmap=memory_efficient,
    )

    if memory_efficient:
        rms_left = _compute_rms_memmap(left_output)
    else:
        rms_left = np.sqrt(np.mean(left_output ** 2))
    if rms_left > 1e-8:
        left_output *= rms_in / rms_left

    if memory_efficient:
        rms_right = _compute_rms_memmap(right_output)
    else:
        rms_right = np.sqrt(np.mean(right_output ** 2))
    if rms_right > 1e-8:
        right_output *= rms_in / rms_right

    stereo_output = np.stack((left_output, right_output), axis=-1)

    max_val = np.max(np.abs(stereo_output))
    if max_val > 0.95:
        stereo_output = np.clip(stereo_output, -0.95, 0.95)
    elif max_val > 0:
        stereo_output = stereo_output / max_val * 0.95

    return stereo_output, time.time() - start_time



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
    """Generate swept notch noise and save to ``filename``."""

    stereo_output, total_time = _generate_swept_notch_arrays(
        duration_seconds,
        sample_rate,
        lfo_freq,
        filter_sweeps,
        notch_q,
        cascade_count,
        lfo_phase_offset_deg,
        intra_phase_offset_deg,
        input_audio_path,
        noise_type,
        lfo_waveform,
        memory_efficient,
        n_jobs,
    )

    try:
        sf.write(filename, stereo_output, sample_rate, subtype='PCM_16')
        print(
            f"\nSuccessfully generated and saved to '{filename}' in {total_time:.2f} seconds."
        )
    except Exception as e:
        print(f"Error saving audio file: {e}")


def generate_swept_notch_pink_sound_transition(
    filename="swept_notch_sound.wav",
    duration_seconds=60,
    sample_rate=DEFAULT_SAMPLE_RATE,
    start_lfo_freq=DEFAULT_LFO_FREQ,
    end_lfo_freq=DEFAULT_LFO_FREQ,
    start_filter_sweeps=None,
    end_filter_sweeps=None,
    start_notch_q=25,
    end_notch_q=25,
    start_cascade_count=10,
    end_cascade_count=10,
    start_lfo_phase_offset_deg=90,
    end_lfo_phase_offset_deg=90,
    start_intra_phase_offset_deg=0,
    end_intra_phase_offset_deg=0,
    input_audio_path=None,
    noise_type="pink",
    lfo_waveform="sine",
    initial_offset=0.0,
    post_offset=0.0,
    transition_curve="linear",
    memory_efficient=False,
    n_jobs=2,
):
    """Generate swept notch noise with parameters transitioning from start to end.

    The output is created by progressively modifying the notch filter sweep
    parameters over ``duration_seconds``; no crossfading between two pre-rendered
    sounds occurs.  ``initial_offset`` and ``post_offset`` specify regions at the
    beginning and end of the file where no transition is applied, matching the
    behaviour of other transition helpers in this package.
    """

    stereo_output, total_time = _generate_swept_notch_arrays_transition(
        duration_seconds,
        sample_rate,
        start_lfo_freq,
        end_lfo_freq,
        start_filter_sweeps,
        end_filter_sweeps if end_filter_sweeps is not None else start_filter_sweeps,
        start_notch_q,
        end_notch_q,
        start_cascade_count,
        end_cascade_count,
        start_lfo_phase_offset_deg,
        end_lfo_phase_offset_deg,
        start_intra_phase_offset_deg,
        end_intra_phase_offset_deg,
        input_audio_path,
        noise_type,
        lfo_waveform,
        initial_offset,
        post_offset,
        transition_curve,
        memory_efficient,
        n_jobs,
    )

    try:
        sf.write(filename, stereo_output, sample_rate, subtype="PCM_16")
        print(
            f"\nSuccessfully generated and saved to '{filename}' in {total_time:.2f} seconds."
        )
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

