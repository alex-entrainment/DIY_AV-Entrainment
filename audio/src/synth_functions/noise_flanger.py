
import numpy as np
import numba
import soundfile as sf
from scipy import signal
from joblib import Parallel, delayed
import time
import tempfile
import argparse

# If these exist in your project; harmless if unused in this file.
from common import (
    calculate_transition_alpha,
    blue_noise,
    purple_noise,
    green_noise,
    deep_brown_noise,
)

# --- Parameters ---
DEFAULT_SAMPLE_RATE = 44100  # Hz
DEFAULT_LFO_FREQ = 1.0 / 12.0  # Hz, for 12-second period


# =========================
# Loudness / limiting helpers
# =========================
def _db_to_lin(db):
    return 10.0 ** (db / 20.0)


def _rms(x):
    return float(np.sqrt(np.mean(np.float64(x) * np.float64(x))) + 1e-20)


def _normalize_rms(x, target_dbfs=-18.0):
    """Scale to a target RMS (dBFS)."""
    cur = _rms(x)
    tgt = _db_to_lin(target_dbfs)
    if cur <= 0.0:
        return x
    return (x * (tgt / cur)).astype(np.float32, copy=False)


def _soft_clip_tanh(x, drive_db=0.0, ceiling_db=-1.0):
    """Gentle tanh soft clip. drive_db > 0 adds a bit of saturation; ceiling sets final peak."""
    if abs(drive_db) <= 1e-9 and ceiling_db >= 0.0:
        return x
    drive = _db_to_lin(drive_db)
    ceil = _db_to_lin(ceiling_db)
    y = np.tanh(x * drive) / np.tanh(1.0)  # unity at 1.0 in
    peak = float(np.max(np.abs(y))) + 1e-12
    y = y * min(ceil / peak, 1.0)
    return y.astype(np.float32, copy=False)


def true_peak_limiter(
    x,
    fs,
    ceiling_db=-1.0,
    lookahead_ms=2.0,
    release_ms=60.0,
    oversample=4,
):
    """
    Look-ahead true-peak limiter:
      - oversamples to catch inter-sample peaks,
      - schedules minimum gain over lookahead,
      - single-pole release toward unity,
      - decimates back and trims to ceiling.
    """
    from collections import deque

    ceil = _db_to_lin(ceiling_db)
    oversample = max(1, int(oversample))

    # Oversample
    if oversample > 1:
        x_os = signal.resample_poly(x, oversample, 1)
        fs_os = fs * oversample
    else:
        x_os = x.astype(np.float32, copy=False)
        fs_os = fs

    N = x_os.size
    eps = 1e-12
    look = max(1, int(lookahead_ms * 1e-3 * fs_os))
    rel_a = np.exp(-1.0 / max(1.0, (release_ms * 1e-3) * fs_os))

    desired = np.minimum(1.0, ceil / (np.abs(x_os) + eps)).astype(np.float32)

    # Sliding min over next 'look' samples -> gmin[i] = min(desired[i:i+look+1])
    gmin = np.ones_like(desired)
    q = deque()
    for i in range(N + look):
        if i < N:
            val = desired[i]
            while q and q[-1][1] >= val:
                q.pop()
            q.append((i, val))
        while q and q[0][0] < i - look:
            q.popleft()
        if i - look >= 0:
            gmin[i - look] = q[0][1] if q else 1.0

    # Release smoothing (toward 1.0, never above gmin)
    g = np.empty_like(gmin)
    gp = 1.0
    for i in range(N):
        gp = min(gmin[i], 1.0 - (1.0 - gp) * rel_a)
        g[i] = gp

    y_os = (x_os * g).astype(np.float32, copy=False)

    # Decimate back
    if oversample > 1:
        y = signal.resample_poly(y_os, 1, oversample)
        peak = float(np.max(np.abs(y))) + eps
        if peak > ceil:
            y = y * (ceil / peak)
        return y.astype(np.float32, copy=False)
    else:
        return y_os


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


def apply_flanger(
    input_signal,
    sample_rate,
    lfo_freq=0.1,
    max_delay_ms=5.0,
    mix=0.5,
    direction="up",
    lfo_waveform="sine",
    feedback=0.0,
):
    """Apply a flanging effect to ``input_signal``.

    Fractional delay interpolation and a light smoothing filter are used to
    reduce zipper noise as the modulation approaches its extremes.
    """

    n = len(input_signal)
    t = np.arange(n) / sample_rate
    if lfo_waveform.lower() == "triangle":
        lfo = signal.sawtooth(2 * np.pi * lfo_freq * t, width=0.5)
    else:
        lfo = np.sin(2 * np.pi * lfo_freq * t)
    if direction == "down":
        lfo = -lfo

    if n > 5:
        lfo = signal.savgol_filter(lfo, 5, 2, mode="interp")

    max_delay = max_delay_ms / 1000.0 * sample_rate
    delay = (lfo + 1.0) * 0.5 * max_delay

    pad = int(np.ceil(max_delay)) + 2
    buffer = np.concatenate((np.zeros(pad, dtype=np.float32), input_signal.astype(np.float32)))
    output = np.zeros(n, dtype=np.float32)

    for i in range(n):
        read_pos = pad + i - delay[i]
        i0 = int(np.floor(read_pos))
        frac = read_pos - i0
        y0 = buffer[i0]
        y1 = buffer[i0 + 1]
        delayed = y0 + frac * (y1 - y0)
        buffer[pad + i] += delayed * feedback
        output[i] = input_signal[i] * (1 - mix) + delayed * mix

    return output


def generate_flanged_noise(
    duration_seconds,
    sample_rate=DEFAULT_SAMPLE_RATE,
    noise_type="pink",
    lfo_freq=0.1,
    max_delay_ms=5.0,
    mix=0.5,
    direction="up",
    lfo_waveform="sine",
):
    """Generate noise of a given color and apply a flanging effect."""

    num_samples = int(duration_seconds * sample_rate)
    noise = generate_noise_samples(num_samples, noise_type, sample_rate)
    return apply_flanger(
        noise,
        sample_rate,
        lfo_freq=lfo_freq,
        max_delay_ms=max_delay_ms,
        mix=mix,
        direction=direction,
        lfo_waveform=lfo_waveform,
    )


# ===========================
# Swept-notch machinery (kept)
# ===========================
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


# =======================================================
# NEW: Flanged noise progression (freeze → ramp → hold)
# =======================================================
def generate_flanged_noise_progression(
    filename="flanged_noise_progression.wav",
    duration_seconds=120.0,
    sample_rate=DEFAULT_SAMPLE_RATE,
    noise_type="pink",
    # macro envelopes (start -> end)
    start_rate_hz=0.32,
    end_rate_hz=0.06,
    start_center_ms=2.0,
    end_center_ms=0.9,
    start_depth_ms=1.5,
    end_depth_ms=0.4,
    start_mix=0.35,
    end_mix=0.22,
    start_feedback=0.40,
    end_feedback=0.18,
    # one-way behavior and stop conditions (direction is in the FREQUENCY domain)
    direction="up",  # "up" => f0 increases; "down" => f0 decreases
    ramp_seconds=None,  # reach the end by this wall-clock time (includes pauses)
    stop_when_spacing_hz=None,  # stop when Δf hits this (Hz)
    stop_when_first_notch_hz=None,  # stop when f0 hits this (Hz)
    stop_at_peak=False,  # stop where triangle would turn around (first extreme)
    freeze_seconds=None,  # total file: initial_freeze + ramp + freeze
    # initial/final freeze
    initial_freeze_seconds=0.0,  # pre-hold before any transition
    final_freeze_at_last_sec=0.0,  # optional tail hold (parity)
    # pause sequencing
    run_range=(2.5, 7.0),
    hold_range=(1.0, 3.0),
    smooth_ms=25.0,
    rng_seed=42,
    pauses_during_ramp=True,
    pause_free_tail_frac=0.25,  # last fraction of ramp with pauses disabled
    # smoothing + mitigation
    delay_slew_ms=12.0,
    ease_to_hold_ms=200.0,  # minimum-jerk blend into the plateau
    mix_slew_ms=15.0,
    feedback_slew_ms=25.0,
    interp="linear",  # "linear" or "cubic"
    # linearization / shaping
    sweep_domain="delay",  # "delay" | "f0" | "spacing" | "logf0"
    progress_gamma=1.0,  # >1 straightens the late portion of the ramp
    # --- Post-processing ---
    normalize_rms_db=None,  # e.g. -18.0; None disables RMS normalization
    tp_ceiling_db=None,  # e.g. -1.0; None disables limiter
    limiter_lookahead_ms=2.0,
    limiter_release_ms=60.0,
    limiter_oversample=4,
    softclip_drive_db=0.0,  # >0 to add gentle tanh saturation after limiting
    output_subtype="PCM_24",  # "PCM_16" | "PCM_24" | "FLOAT"
):
    """
    Flanged-noise progression with:
      - explicit piecewise τ_ms = [initial freeze | ramp | final hold],
      - pauses in early ramp region, *pause-free tail* that guarantees s==1 at end,
      - stop-at-peak and stop-when-f0/Δf,
      - endpoint easing, control slews, and cubic interpolation option,
      - sweep_domain to linearize in 'f0'/'spacing'/'logf0', and progress_gamma shaping,
      - optional RMS normalization + true-peak limiting (+ soft-clip).
    """
    fs = int(sample_rate)

    # ------------------ helpers ------------------
    def _alpha_from_ms(ms):  # one-pole coeff
        return np.exp(-1.0 / max(1.0, (ms * 1e-3) * fs))

    def _one_pole_smoother(x, smooth_ms):
        if smooth_ms <= 0:
            return x.astype(np.float32, copy=False)
        a = _alpha_from_ms(smooth_ms)
        y = np.empty_like(x, dtype=np.float32)
        s = float(x[0])
        for i in range(x.shape[0]):
            s = a * s + (1.0 - a) * float(x[i])
            y[i] = s
        return y

    def _make_gate(n, run_rng, hold_rng, edge_ms, seed):
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        rng = np.random.default_rng(seed)
        out = np.zeros(n, dtype=np.float32)
        i, state = 0, 1
        while i < n:
            dur = rng.uniform(*(run_rng if state == 1 else hold_rng))
            L = max(1, int(dur * fs))
            j = min(n, i + L)
            out[i:j] = 1.0 if state == 1 else 0.0
            i = j
            state ^= 1
        Ls = int(max(3, round(edge_ms * 1e-3 * fs)))
        if Ls % 2 == 0:
            Ls += 1
        if Ls > 3:
            win = np.hanning(Ls).astype(np.float32)
            win /= max(win.sum(), 1e-12)
            out = np.convolve(out, win, mode="same").astype(np.float32)
        return out

    def _smoothstep5(u):
        return (u * u * u) * (10.0 + u * (-15.0 + 6.0 * u))  # C2 smooth

    def _read_linear(buf, read_pos, max_delay):
        i0 = int(np.floor(read_pos)) % max_delay
        i1 = (i0 + 1) % max_delay
        frac = read_pos - np.floor(read_pos)
        return (1.0 - frac) * buf[i0] + frac * buf[i1]

    def _read_cubic(buf, read_pos, max_delay):
        i1 = int(np.floor(read_pos))
        t_ = read_pos - i1
        i0 = (i1 - 1) % max_delay
        i1 = i1 % max_delay
        i2 = (i1 + 1) % max_delay
        i3 = (i1 + 2) % max_delay
        x0, x1, x2, x3 = buf[i0], buf[i1], buf[i2], buf[i3]
        a0 = -0.5 * x0 + 1.5 * x1 - 1.5 * x2 + 0.5 * x3
        a1 = x0 - 2.5 * x1 + 2.0 * x2 - 0.5 * x3
        a2 = -0.5 * x0 + 0.5 * x2
        a3 = x1
        return ((a0 * t_ + a1) * t_ + a2) * t_ + a3

    # ---- compute total duration if ramp-based ----
    has_ramp = ramp_seconds is not None
    n_init = int(max(0.0, float(initial_freeze_seconds)) * fs)
    n_ramp = int((ramp_seconds if has_ramp else 0.0) * fs)
    n_tail = int((freeze_seconds if freeze_seconds is not None else 0.0) * fs)

    if has_ramp:
        duration_seconds = (initial_freeze_seconds or 0.0) + float(ramp_seconds) + (freeze_seconds or 0.0)

    n_total = int(duration_seconds * fs)
    n_init = min(n_init, n_total)
    n_ramp = min(n_ramp, max(0, n_total - n_init))
    n_tail = min(n_tail, max(0, n_total - n_init - n_ramp))
    n_rest = max(0, n_total - (n_init + n_ramp + n_tail))
    n_tail += n_rest  # absorb rounding error

    print(f"[progression] segments: init={n_init/fs:.3f}s, ramp={n_ramp/fs:.3f}s, hold={n_tail/fs:.3f}s")

    # ---- source & macro envelopes (for mix/feedback; delay is piecewise below) ----
    t = np.arange(n_total, dtype=np.float32) / fs
    x = generate_noise_samples(n_total, noise_type, fs).astype(np.float32)

    def _lerp_time(start, end):
        return np.interp(t, [0.0, duration_seconds], [start, end]).astype(np.float32)

    rate_env = _lerp_time(start_rate_hz, end_rate_hz)
    mix_env = _lerp_time(start_mix, end_mix)
    fb_env = _lerp_time(start_feedback, end_feedback)

    # ---- compute start/end extremes τ (ms) from knobs + direction ----
    if direction.lower() == "up":
        tau_start_ms = float(start_center_ms + start_depth_ms)  # max delay → lowest f0
        tau_end_ms = float(end_center_ms - end_depth_ms)  # min delay → highest f0
    else:
        tau_start_ms = float(start_center_ms - start_depth_ms)  # min delay → highest f0
        tau_end_ms = float(end_center_ms + end_depth_ms)  # max delay → lowest f0

    if tau_start_ms <= 0.0 or tau_end_ms <= 0.0:
        raise ValueError(
            f"Invalid extremes: τ_start={tau_start_ms:.6f} ms, τ_end={tau_end_ms:.6f} ms. "
            "Choose center/depth so both are positive."
        )

    # ================== PIECEWISE τ_ms TARGET ==================
    tau_ms_target = np.empty(n_total, dtype=np.float32)

    # -- 1) INITIAL FREEZE: constant τ at start extreme --
    if n_init > 0:
        tau_ms_target[:n_init] = tau_start_ms

    # -- 2) RAMP: create progress s∈[0,1] over n_ramp samples (pauses in first part, pause-free tail) --
    idx_start_ramp = n_init
    idx_end_ramp = n_init + n_ramp

    if n_ramp > 0:
        pft = float(np.clip(pause_free_tail_frac, 0.0, 1.0))
        cutoff = int(np.floor(n_ramp * (1.0 - pft)))
        cutoff = max(0, min(cutoff, n_ramp))

        # Part A: pauses region → rescaled to alpha progress at cutoff
        if cutoff > 0 and pauses_during_ramp:
            g = _make_gate(cutoff, run_range, hold_range, smooth_ms, rng_seed)
            run_mask = (g >= 0.5).astype(np.float32)
            cum_run = np.cumsum(run_mask)
            total_active = float(cum_run[-1]) if cutoff > 0 else 1.0
            if total_active < 1.0:
                total_active = 1.0
            alpha = cutoff / float(n_ramp)
            s_first = alpha * (cum_run / total_active).astype(np.float32)
        elif cutoff > 0:
            alpha = cutoff / float(n_ramp)
            idx_first = np.arange(cutoff, dtype=np.float32)
            denom = max(1.0, cutoff - 1.0)
            s_first = alpha * (idx_first / denom)
        else:
            s_first = np.zeros(0, dtype=np.float32)
            alpha = 0.0

        # Part B: pause-free tail → inclusive normalization ensures last ramp sample hits s==1.0
        tail = n_ramp - cutoff
        if tail > 0:
            idx_tail = np.arange(tail, dtype=np.float32)
            s_second = alpha + (1.0 - alpha) * ((idx_tail + 1.0) / float(tail))
        else:
            s_second = np.zeros(0, dtype=np.float32)

        s_ramp = np.concatenate([s_first, s_second], axis=0)
        if progress_gamma != 1.0:
            s_ramp = np.power(np.clip(s_ramp, 0.0, 1.0), float(progress_gamma)).astype(np.float32)

        # Map s_ramp to τ in desired sweep domain
        if sweep_domain.lower() == "delay":
            tau_ramp_ms = (1.0 - s_ramp) * tau_start_ms + s_ramp * tau_end_ms
        elif sweep_domain.lower() == "spacing":
            df_start = 1000.0 / tau_start_ms
            df_end = 1000.0 / tau_end_ms
            df = (1.0 - s_ramp) * df_start + s_ramp * df_end
            tau_ramp_ms = 1000.0 / np.maximum(df, 1e-9)
        elif sweep_domain.lower() == "f0":
            f0_start = 500.0 / tau_start_ms
            f0_end = 500.0 / tau_end_ms
            f0 = (1.0 - s_ramp) * f0_start + s_ramp * f0_end
            tau_ramp_ms = 500.0 / np.maximum(f0, 1e-9)
        elif sweep_domain.lower() == "logf0":
            f0_start = 500.0 / tau_start_ms
            f0_end = 500.0 / tau_end_ms
            log_f0 = (1.0 - s_ramp) * np.log10(f0_start) + s_ramp * np.log10(f0_end)
            f0 = np.power(10.0, log_f0).astype(np.float32)
            tau_ramp_ms = 500.0 / np.maximum(f0, 1e-9)
        else:
            raise ValueError(f"Unsupported sweep_domain: {sweep_domain}")

        # Early stop conditions — search ONLY within the ramp window
        idx_stop = None
        if (stop_when_first_notch_hz is not None) or (stop_when_spacing_hz is not None):
            tau_s = tau_ramp_ms * 1e-3
            series = (
                0.5 / np.maximum(tau_s, 1e-9)
                if (stop_when_first_notch_hz is not None)
                else 1.0 / np.maximum(tau_s, 1e-9)
            )
            target = float(
                stop_when_first_notch_hz if stop_when_first_notch_hz is not None else stop_when_spacing_hz
            )
            s0 = float(series[0])
            if np.isclose(s0, target, atol=1e-6, rtol=0.0):
                idx_stop = 0
            elif s0 < target:
                mask = series >= target
                idx_stop = int(np.argmax(mask)) if np.any(mask) else None
            else:
                mask = series <= target
                idx_stop = int(np.argmax(mask)) if np.any(mask) else None

        # Peak = last ramp sample
        idx_peak_ramp = (n_ramp - 1) if (n_ramp > 0) else None
        # Decide freeze point within ramp (local index)
        if idx_stop is not None:
            freeze_local = idx_stop
            hold_target_ms = float(tau_ramp_ms[idx_stop])
        else:
            # stop_at_peak simply means "freeze at the end of the ramp"
            freeze_local = idx_peak_ramp
            hold_target_ms = float(tau_end_ms)

        # Apply endpoint easing inside the ramp (blend last E ms toward hold_target_ms)
        tau_ramp_ms_eased = tau_ramp_ms.copy()
        if ease_to_hold_ms > 0 and freeze_local is not None and freeze_local > 0:
            L = int((ease_to_hold_ms * 1e-3) * fs)
            if L > 1:
                k0 = max(0, freeze_local - L)
                seg = tau_ramp_ms_eased[k0:freeze_local].copy()
                m = seg.size
                if m > 0:
                    u = np.linspace(0.0, 1.0, m, endpoint=False, dtype=np.float32)
                    w = _smoothstep5(u)
                    tau_ramp_ms_eased[k0:freeze_local] = seg * (1.0 - w) + hold_target_ms * w
                tau_ramp_ms_eased[freeze_local:] = hold_target_ms
        else:
            if freeze_local is not None:
                tau_ramp_ms_eased[freeze_local:] = hold_target_ms

        # Write ramp block into global τ
        tau_ms_target[idx_start_ramp:idx_end_ramp] = tau_ramp_ms_eased

        # -- 3) FINAL HOLD: constant τ == hold_target_ms --
        if n_tail > 0:
            tau_ms_target[idx_end_ramp:] = hold_target_ms
    else:
        # no ramp: fully frozen at start extreme
        tau_ms_target[:] = tau_start_ms

    # Optional extra tail freeze (parity with older interface)
    if final_freeze_at_last_sec and final_freeze_at_last_sec > 0:
        k = n_total - int(final_freeze_at_last_sec * fs)
        if 0 < k < n_total:
            tau_ms_target[k:] = tau_ms_target[k]

    # ---- Smooth delay trajectory ----
    d_ms_target = np.maximum(tau_ms_target, 1e-6)
    d_samp_target = (d_ms_target * 1e-3 * fs).astype(np.float32)
    d_samp = _one_pole_smoother(d_samp_target, delay_slew_ms)

    # ---- Render flanger (cubic optional) ----
    use_cubic = (str(interp).lower() == "cubic")
    a_mix = np.exp(-1.0 / max(1.0, (mix_slew_ms * 1e-3) * fs))
    a_fb = np.exp(-1.0 / max(1.0, (feedback_slew_ms * 1e-3) * fs))

    max_delay = int(np.ceil(float(np.max(d_samp)))) + 4
    buf = np.zeros(max_delay, dtype=np.float32)
    y = np.zeros(n_total, dtype=np.float32)
    w = 0
    mix_state = float(mix_env[0])
    fb_state = float(fb_env[0])

    for i in range(n_total):
        mix_state = a_mix * mix_state + (1.0 - a_mix) * float(mix_env[i])
        fb_state = a_fb * fb_state + (1.0 - a_fb) * float(fb_env[i])

        fb = fb_state * (y[i - 1] if i > 0 else 0.0)
        buf[w] = x[i] + fb

        read_pos = w - d_samp[i]
        while read_pos < 0:
            read_pos += max_delay

        if use_cubic:
            delayed = _read_cubic(buf, read_pos, max_delay)
        else:
            delayed = _read_linear(buf, read_pos, max_delay)

        y[i] = (1.0 - mix_state) * x[i] + mix_state * delayed
        w = (w + 1) % max_delay

    # ---------------- POST: loudness + peak control ----------------
    y_post = y
    if normalize_rms_db is not None:
        y_post = _normalize_rms(y_post, normalize_rms_db)
    if tp_ceiling_db is not None:
        y_post = true_peak_limiter(
            y_post,
            fs,
            ceiling_db=tp_ceiling_db,
            lookahead_ms=limiter_lookahead_ms,
            release_ms=limiter_release_ms,
            oversample=limiter_oversample,
        )
    if softclip_drive_db and abs(softclip_drive_db) > 1e-9:
        ceiling_for_clip = tp_ceiling_db if tp_ceiling_db is not None else -0.1
        y_post = _soft_clip_tanh(y_post, drive_db=softclip_drive_db, ceiling_db=ceiling_for_clip)

    # Write in a headroom-friendly format by default
    sf.write(filename, y_post, fs, subtype=output_subtype)
    return filename


def main():
    parser = argparse.ArgumentParser(
        description="Generate noise with swept notch filters or flanging (including one-way progression)."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # Simple flange (legacy)
    flange = sub.add_parser("flange", help="Apply a basic flanger to noise")
    flange.add_argument("--output", type=str, default="flanged_noise.wav")
    flange.add_argument("--duration", type=float, default=60.0)
    flange.add_argument("--noise-type", type=str, default="pink")
    flange.add_argument("--lfo-freq", type=float, default=0.1)
    flange.add_argument("--max-delay-ms", type=float, default=5.0)
    flange.add_argument("--mix", type=float, default=0.5)
    flange.add_argument("--direction", choices=["up", "down"], default="up")
    flange.add_argument("--lfo-waveform", choices=["sine", "triangle"], default="sine")

    # Notch path (kept)
    notch = sub.add_parser("notch", help="Generate swept-notch filtered noise")
    notch.add_argument("--output", type=str, default="dual_sweep_triangle_lfo.wav")
    notch.add_argument("--duration", type=float, default=60.0)
    notch.add_argument("--lfo-freq", type=float, default=DEFAULT_LFO_FREQ)
    notch.add_argument("--lfo-waveform", choices=["sine", "triangle"], default="triangle")

    # NEW: one-way flange progression
    prog = sub.add_parser("flange-progression", help="One-way flanging progression with freezes, pauses, and hold.")
    prog.add_argument("--output", type=str, default="flanged_progression.wav")
    prog.add_argument("--duration", type=float, default=120.0)
    prog.add_argument("--noise-type", type=str, default="pink")

    # Macro envelopes
    prog.add_argument("--start-rate-hz", type=float, default=0.32)
    prog.add_argument("--end-rate-hz", type=float, default=0.06)
    prog.add_argument("--start-center-ms", type=float, default=2.0)
    prog.add_argument("--end-center-ms", type=float, default=0.9)
    prog.add_argument("--start-depth-ms", type=float, default=1.5)
    prog.add_argument("--end-depth-ms", type=float, default=0.4)
    prog.add_argument("--start-mix", type=float, default=0.35)
    prog.add_argument("--end-mix", type=float, default=0.22)
    prog.add_argument("--start-feedback", type=float, default=0.40)
    prog.add_argument("--end-feedback", type=float, default=0.18)

    # Direction & timing
    prog.add_argument("--direction", choices=["up", "down"], default="up")
    prog.add_argument("--ramp-seconds", type=float, default=60.0)
    prog.add_argument("--freeze-seconds", type=float, default=30.0)  # final hold
    prog.add_argument("--initial-freeze-seconds", type=float, default=0.0)

    # Pause sequencing during ramp
    prog.add_argument("--pauses-during-ramp", action="store_true", default=False)
    prog.add_argument("--run-min", type=float, default=2.5)
    prog.add_argument("--run-max", type=float, default=7.0)
    prog.add_argument("--hold-min", type=float, default=1.0)
    prog.add_argument("--hold-max", type=float, default=3.0)
    prog.add_argument("--smooth-ms", type=float, default=25.0)
    prog.add_argument("--seed", type=int, default=42)
    prog.add_argument("--pause-free-tail-frac", type=float, default=0.25,
                      help="Fraction (0..1) of the ramp at the end where pauses are disabled (continuous approach).")

    # Stops + easing + slews + interpolation
    prog.add_argument("--stop-when-spacing-hz", type=float, default=None)
    prog.add_argument("--stop-when-first-notch-hz", type=float, default=None)
    prog.add_argument("--stop-at-peak", action="store_true", default=False)
    prog.add_argument("--ease-to-hold-ms", type=float, default=200.0)
    prog.add_argument("--delay-slew-ms", type=float, default=12.0)
    prog.add_argument("--mix-slew-ms", type=float, default=15.0)
    prog.add_argument("--feedback-slew-ms", type=float, default=25.0)
    prog.add_argument("--interp", type=str, choices=["linear", "cubic"], default="linear")

    # Linearization / shaping
    prog.add_argument("--sweep-domain", type=str, default="delay",
                      choices=["delay", "f0", "spacing", "logf0"])
    prog.add_argument("--progress-gamma", type=float, default=1.0,
                      help="Exponent for ramp progress shaping (>1 straightens the late portion).")

    # Optional extra final freeze (parity)
    prog.add_argument("--final-freeze-seconds", type=float, default=0.0)

    # Post / loudness options
    prog.add_argument("--normalize-rms-db", type=float, default=None,
                      help="If set (e.g., -18), normalize RMS to this dBFS before limiting.")
    prog.add_argument("--tp-ceiling-db", type=float, default=None,
                      help="Enable true-peak limiter with this ceiling (dBTP), e.g., -1.0. None disables.")
    prog.add_argument("--limiter-lookahead-ms", type=float, default=2.0)
    prog.add_argument("--limiter-release-ms", type=float, default=60.0)
    prog.add_argument("--limiter-oversample", type=int, default=4, choices=[1, 2, 3, 4, 6, 8],
                      help="Limiter oversampling factor.")
    prog.add_argument("--softclip-drive-db", type=float, default=0.0,
                      help="Optional tanh soft-clip drive in dB (0 disables).")
    prog.add_argument("--output-subtype", type=str, default="PCM_24",
                      choices=["PCM_16", "PCM_24", "FLOAT"],
                      help="Output file subtype/bit-depth.")

    args = parser.parse_args()

    if args.mode == "flange":
        audio = generate_flanged_noise(
            duration_seconds=args.duration,
            sample_rate=DEFAULT_SAMPLE_RATE,
            noise_type=args.noise_type,
            lfo_freq=args.lfo_freq,
            max_delay_ms=args.max_delay_ms,
            mix=args.mix,
            direction=args.direction,
            lfo_waveform=args.lfo_waveform,
        )
        sf.write(args.output, audio, DEFAULT_SAMPLE_RATE, subtype="PCM_16")

    elif args.mode == "notch":
        dual_sweeps_config = [(500, 1000), (1850, 3350)]
        generate_swept_notch_pink_sound(
            filename=args.output,
            duration_seconds=args.duration,
            sample_rate=DEFAULT_SAMPLE_RATE,
            lfo_freq=args.lfo_freq,
            filter_sweeps=dual_sweeps_config,
            notch_q=40,
            cascade_count=15,
            lfo_phase_offset_deg=90,
            lfo_waveform=args.lfo_waveform,
        )

    elif args.mode == "flange-progression":
        generate_flanged_noise_progression(
            filename=args.output,
            duration_seconds=args.duration,
            sample_rate=DEFAULT_SAMPLE_RATE,
            noise_type=args.noise_type,
            # macro
            start_rate_hz=args.start_rate_hz,
            end_rate_hz=args.end_rate_hz,
            start_center_ms=args.start_center_ms,
            end_center_ms=args.end_center_ms,
            start_depth_ms=args.start_depth_ms,
            end_depth_ms=args.end_depth_ms,
            start_mix=args.start_mix,
            end_mix=args.end_mix,
            start_feedback=args.start_feedback,
            end_feedback=args.end_feedback,
            # direction/timing
            direction=args.direction,
            ramp_seconds=args.ramp_seconds,
            freeze_seconds=args.freeze_seconds,
            initial_freeze_seconds=args.initial_freeze_seconds,
            final_freeze_at_last_sec=args.final_freeze_seconds,
            # pauses
            run_range=(args.run_min, args.run_max),
            hold_range=(args.hold_min, args.hold_max),
            smooth_ms=args.smooth_ms,
            rng_seed=args.seed,
            pauses_during_ramp=args.pauses_during_ramp,
            pause_free_tail_frac=args.pause_free_tail_frac,
            # stops + shaping
            stop_when_spacing_hz=args.stop_when_spacing_hz,
            stop_when_first_notch_hz=args.stop_when_first_notch_hz,
            stop_at_peak=args.stop_at_peak,
            ease_to_hold_ms=args.ease_to_hold_ms,
            delay_slew_ms=args.delay_slew_ms,
            mix_slew_ms=args.mix_slew_ms,
            feedback_slew_ms=args.feedback_slew_ms,
            interp=args.interp,
            sweep_domain=args.sweep_domain,
            progress_gamma=args.progress_gamma,
            # post
            normalize_rms_db=args.normalize_rms_db,
            tp_ceiling_db=args.tp_ceiling_db,
            limiter_lookahead_ms=args.limiter_lookahead_ms,
            limiter_release_ms=args.limiter_release_ms,
            limiter_oversample=args.limiter_oversample,
            softclip_drive_db=args.softclip_drive_db,
            output_subtype=args.output_subtype,
        )


if __name__ == "__main__":
    main()

