import numpy as np
import numba
import soundfile as sf
from scipy import signal

# --- Parameters ---
DEFAULT_SAMPLE_RATE = 44100  # Hz
DEFAULT_LFO_FREQ = 1.0 / 12.0  # Hz, for 12-second period

@numba.jit(nopython=True)
def generate_pink_noise_samples(n_samples):
    """
    Generates pink noise samples with reduced high frequency content.
    """
    white = np.random.randn(n_samples).astype(np.float32)
    pink = np.empty_like(white)

    # State variables for the pinking filter
    b0, b1, b2, b3, b4, b5 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    for i in range(n_samples):
        w = white[i]
        
        # Paul Kellett's method
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
    """Generate brown (red) noise samples."""
    white = np.random.randn(n_samples).astype(np.float32)
    brown = np.cumsum(white)
    max_abs = np.max(np.abs(brown)) + 1e-8
    return (brown / max_abs).astype(np.float32)


def _apply_deep_swept_notches_single_phase(input_signal, sample_rate, lfo_freq,
                                           min_freq=1000, max_freq=10000,
                                           num_notches=6, notch_spacing_ratio=1.1,
                                           notch_q=100, cascade_count=3,
                                           phase_offset=0):
    """
    Apply very deep swept notch filters without harmonics for a single LFO
    phase.
    
    Args:
        input_signal: Input signal
        sample_rate: Sample rate
        lfo_freq: LFO frequency for sweep
        min_freq: Minimum sweep frequency
        max_freq: Maximum sweep frequency
        num_notches: Number of parallel notches (not harmonics)
        notch_spacing_ratio: Frequency ratio between adjacent notches
        notch_q: Q factor (higher = narrower notch)
        cascade_count: Number of times to apply each notch for depth
        phase_offset: Phase offset for LFO
    
    Returns:
        Filtered signal
    """
    n_samples = len(input_signal)
    output = input_signal.copy()
    
    # Generate smooth LFO sweep using cosine for better continuity
    t = np.arange(n_samples) / sample_rate
    # Use cosine to ensure smooth periodic motion without flat spots
    lfo = np.cos(2 * np.pi * lfo_freq * t + phase_offset)
    
    # Linear frequency sweep (not logarithmic) for smoother motion
    center_freq = (min_freq + max_freq) / 2
    freq_range = (max_freq - min_freq) / 2
    base_freq_sweep = center_freq + freq_range * lfo
    
    # Process in overlapping blocks for smooth transitions
    block_size = 4096
    hop_size = block_size // 2  # 50% overlap
    window = np.hanning(block_size)
    
    # Output accumulator for overlap-add
    output_accumulator = np.zeros(n_samples + block_size)
    window_accumulator = np.zeros(n_samples + block_size)
    
    # Process each block
    num_blocks = (n_samples + hop_size - 1) // hop_size
    
    for block_idx in range(num_blocks):
        start_idx = block_idx * hop_size
        end_idx = min(start_idx + block_size, n_samples)
        actual_block_size = end_idx - start_idx
        
        if actual_block_size < 100:  # Skip very small blocks
            continue
        
        # Get input block
        if actual_block_size < block_size:
            # Pad the last block
            block = np.zeros(block_size)
            block[:actual_block_size] = output[start_idx:end_idx]
        else:
            block = output[start_idx:end_idx].copy()
        
        # Apply window
        windowed_block = block * window[:len(block)]
        
        # Get center frequency for this block
        block_center_idx = start_idx + actual_block_size // 2
        if block_center_idx < len(base_freq_sweep):
            center_freq_for_block = base_freq_sweep[block_center_idx]
        else:
            center_freq_for_block = base_freq_sweep[-1]
        
        # Apply multiple notches at different frequencies
        filtered_block = windowed_block.copy()
        
        for notch_idx in range(num_notches):
            # Calculate notch frequency (not harmonic, just spaced)
            notch_freq = center_freq_for_block * (notch_spacing_ratio ** notch_idx)
            
            # Skip if too high
            if notch_freq > sample_rate * 0.48:
                continue
            
            # Apply the same notch multiple times for deeper effect
            for cascade in range(cascade_count):
                # Design narrow notch filter
                try:
                    b, a = signal.iirnotch(notch_freq, notch_q, sample_rate)
                    # Apply zero-phase filtering for no phase distortion
                    filtered_block = signal.filtfilt(b, a, filtered_block)
                except:
                    # Skip if filter design fails
                    continue
        
        # Accumulate output with overlap-add
        output_accumulator[start_idx:start_idx + len(filtered_block)] += filtered_block
        window_accumulator[start_idx:start_idx + len(filtered_block)] += window[:len(filtered_block)]
    
    # Normalize by window accumulation
    valid_idx = window_accumulator > 0.1
    output_accumulator[valid_idx] /= window_accumulator[valid_idx]
    
    # Extract the output
    output = output_accumulator[:n_samples]

    return output


def apply_deep_swept_notches(input_signal, sample_rate, lfo_freq,
                            min_freq=1000, max_freq=10000,
                            num_notches=6, notch_spacing_ratio=1.1,
                            notch_q=100, cascade_count=3,
                            phase_offset=0, extra_phase_offset=0.0):
    """Apply deep swept notch filters optionally using a second phase offset.

    Parameters match :func:`_apply_deep_swept_notches_single_phase` with the
    addition of ``extra_phase_offset`` which, when non-zero, applies a second
    set of notches using ``phase_offset + extra_phase_offset``.
    """

    output = _apply_deep_swept_notches_single_phase(
        input_signal,
        sample_rate,
        lfo_freq,
        min_freq=min_freq,
        max_freq=max_freq,
        num_notches=num_notches,
        notch_spacing_ratio=notch_spacing_ratio,
        notch_q=notch_q,
        cascade_count=cascade_count,
        phase_offset=phase_offset,
    )

    if extra_phase_offset:
        output = _apply_deep_swept_notches_single_phase(
            output,
            sample_rate,
            lfo_freq,
            min_freq=min_freq,
            max_freq=max_freq,
            num_notches=num_notches,
            notch_spacing_ratio=notch_spacing_ratio,
            notch_q=notch_q,
            cascade_count=cascade_count,
            phase_offset=phase_offset + extra_phase_offset,
        )

    return output


def generate_swept_notch_pink_sound(
    filename="swept_notch_pink_sound.wav",
    duration_seconds=60,
    sample_rate=DEFAULT_SAMPLE_RATE,
    lfo_freq=DEFAULT_LFO_FREQ,
    min_freq=1000,          # Minimum sweep frequency
    max_freq=10000,         # Maximum sweep frequency
    num_notches=1,          # Number of parallel notches
    notch_spacing_ratio=1.25,  # Spacing between notches (1.1 = 10% apart)
    notch_q=25,            # Q factor for notches
    cascade_count=10,        # Apply each notch this many times
    lfo_phase_offset_deg=90,  # Phase offset for right channel
    intra_phase_offset_deg=0,  # Optional second filter phase offset per channel
    input_audio_path=None,
    noise_type="brown",
):
    """
    Generates noise with deep swept notches (no harmonics).
    
    Parameters
    ----------
    noise_type : str, optional
        Base noise type to generate when ``input_audio_path`` is ``None``.
        Supported values are ``"pink"`` and ``"brown"``.
    """
    print(f"Starting Deep Swept Notch generation for '{filename}'...")
    print(f"Parameters: Duration={duration_seconds}s, Sample Rate={sample_rate}Hz")
    print(f"LFO Freq={lfo_freq}Hz (Period={1/lfo_freq:.1f}s)")
    print(f"Frequency sweep: {min_freq}Hz to {max_freq}Hz")
    print(f"Notches: {num_notches} parallel notches, Q={notch_q}")
    print(f"Cascade count: {cascade_count} (for deeper notches)")
    if intra_phase_offset_deg:
        print(f"Intra-channel phase offset: {intra_phase_offset_deg} deg")
    
    # Step 1: Generate or load input audio
    if input_audio_path is None:
        num_samples = int(duration_seconds * sample_rate)
        print(f"Step 1/3: Generating base {noise_type} noise...")
        if noise_type.lower() == "brown":
            noise_mono = generate_brown_noise_samples(num_samples)
        else:
            noise_mono = generate_pink_noise_samples(num_samples)
        
        # Additional filtering for warmer sound
        # Gentle high-frequency rolloff
        b_warmth, a_warmth = signal.butter(1, 10000, btype='low', fs=sample_rate)
        noise_mono = signal.filtfilt(b_warmth, a_warmth, noise_mono)
        
        # Gentle low-cut to remove rumble
        b_hpf, a_hpf = signal.butter(2, 50, btype='high', fs=sample_rate)
        noise_mono = signal.filtfilt(b_hpf, a_hpf, noise_mono)
        
        # Normalize
        noise_mono = noise_mono / (np.max(np.abs(noise_mono)) + 1e-8) * 0.8
        input_left = noise_mono.copy()
        input_right = noise_mono.copy()
    else:
        print(f"Step 1/3: Loading input audio from '{input_audio_path}'...")
        data, in_sr = sf.read(input_audio_path, always_2d=True)
        if in_sr != sample_rate:
            # Resample if needed
            num_samples = int(len(data) * sample_rate / in_sr)
            data_resampled = np.zeros((num_samples, data.shape[1]))
            for ch in range(data.shape[1]):
                data_resampled[:, ch] = signal.resample(data[:, ch], num_samples)
            data = data_resampled
        
        input_left = data[:, 0]
        if data.shape[1] > 1:
            input_right = data[:, 1]
        else:
            input_right = input_left.copy()
        
        # Normalize
        max_val = max(np.max(np.abs(input_left)), np.max(np.abs(input_right)))
        if max_val > 0:
            input_left = input_left / max_val * 0.8
            input_right = input_right / max_val * 0.8
    
    # Step 2: Apply deep swept notches
    print("Step 2/3: Applying deep swept notches...")
    
    # Left channel
    print("  Processing left channel...")
    intra_phase_rad = np.deg2rad(intra_phase_offset_deg)

    left_output = apply_deep_swept_notches(
        input_left,
        sample_rate,
        lfo_freq,
        min_freq=min_freq,
        max_freq=max_freq,
        num_notches=num_notches,
        notch_spacing_ratio=notch_spacing_ratio,
        notch_q=notch_q,
        cascade_count=cascade_count,
        phase_offset=0,
        extra_phase_offset=intra_phase_rad
    )
    
    # Right channel with phase offset
    print("  Processing right channel...")
    phase_offset_rad = np.deg2rad(lfo_phase_offset_deg)
    right_output = apply_deep_swept_notches(
        input_right,
        sample_rate,
        lfo_freq,
        min_freq=min_freq,
        max_freq=max_freq,
        num_notches=num_notches,
        notch_spacing_ratio=notch_spacing_ratio,
        notch_q=notch_q,
        cascade_count=cascade_count,
        phase_offset=phase_offset_rad,
        extra_phase_offset=intra_phase_rad
    )
    
    # Step 3: Final processing and save
    print("Step 3/3: Final processing and saving...")
    
    # Combine channels
    stereo_output = np.stack((left_output, right_output), axis=-1)
    
    # Final gentle compression to even out levels
    # Soft clipping
    stereo_output = np.tanh(stereo_output * 0.7) / 0.7
    
    # Final normalization
    max_val = np.max(np.abs(stereo_output))
    if max_val > 0:
        stereo_output = stereo_output / max_val * 0.95
    
    # Save
    try:
        sf.write(filename, stereo_output, sample_rate, subtype='PCM_16')
        print(f"Successfully generated and saved to '{filename}'")
        print("\nThe notches should now:")
        print("- Be very deep due to cascading")
        print("- Sweep smoothly without flat spots")
        print("- Have NO harmonic relationship (no 2x, 3x, etc.)")
        print("\nTo adjust:")
        print("  - cascade_count: More cascades = deeper notches (try 4-5)")
        print("  - notch_q: Higher Q = narrower notches (try 150-200)")
        print("  - notch_spacing_ratio: 1.05-1.2 for different spacing")
    except Exception as e:
        print(f"Error saving audio file: {e}")


# --- Main execution ---
if __name__ == '__main__':
    
    output_filename = "monroe_phased_pink_sound.wav"
    duration = 4000  # 2 minutes
    
    # Generate with smooth swept notches (no harmonics)
    generate_swept_notch_pink_sound(
        filename=output_filename,
        duration_seconds=duration,
        sample_rate=44100,
        lfo_freq=1.0/12.0,         # 12-second period
        min_freq=1000,             # 1 kHz minimum
        max_freq=10000,            # 10 kHz maximum  
        num_notches=1,             # 6 parallel notches
        notch_spacing_ratio=1.25,   # 10% spacing between notches
        notch_q=25,               # Very narrow notches
        cascade_count=20,           # Apply 3 times for depth
        lfo_phase_offset_deg=90,   # 90-degree phase offset
        intra_phase_offset_deg=0,
        input_audio_path=None,
    )
