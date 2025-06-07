import numpy as np
import numba
import soundfile as sf

# --- Parameters ---
DEFAULT_SAMPLE_RATE = 44100  # Hz
DEFAULT_LFO_FREQ = 1.0 / 8.0  # Hz, as per "approximate sweep rate is about 1/8 Hz"

@numba.jit(nopython=True)
def generate_pink_noise_samples(n_samples):
    """
    Generates pink noise samples using Paul Kellett's "refined" method (a common interpretation).
    This involves filtering white noise to achieve a 1/f spectral density.

    Args:
        n_samples (int): The number of pink noise samples to generate.

    Returns:
        np.ndarray: Array of pink noise samples, loosely in the range -1.0 to 1.0.
    """
    white = np.random.randn(n_samples).astype(np.float32)
    pink = np.empty_like(white)

    # State variables for the pinking filter (poles)
    b0, b1, b2, b3, b4, b5 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    for i in range(n_samples):
        w = white[i]
        
        # Paul Kellett's "refined" method filter coefficients
        b0 = 0.99886 * b0 + w * 0.0555179
        b1 = 0.99332 * b1 + w * 0.0750759
        b2 = 0.96900 * b2 + w * 0.1538520
        b3 = 0.86650 * b3 + w * 0.3104856
        b4 = 0.55000 * b4 + w * 0.5329522
        b5 = -0.7616 * b5 - w * 0.0168980 
        
        pink_val = b0 + b1 + b2 + b3 + b4 + b5
        # The sum of these poles gives the pink noise.
        # An additional term (b6 = w * 0.115926) and direct white noise (w * 0.5362)
        # are sometimes included in other variations. This version uses 6 poles.
        
        pink[i] = pink_val * 0.11 # Scaler to keep amplitude reasonable (empirical)
                                   # The sum of many random processes can lead to large values.
                                   # This helps to roughly normalize the output.
    return pink

@numba.jit(nopython=True)
def _stereo_flanger_core(
    input_left,
    input_right,
    sample_rate,
    lfo_freq,
    min_delay_samples,
    max_delay_samples,
    comb_mix_factor,
    lfo_phase_offset_right_rad,
):
    """Numba-accelerated stereo flanger.

    Args:
        input_left (np.ndarray): Left channel input signal.
        input_right (np.ndarray): Right channel input signal.
        sample_rate (float): Audio sample rate in Hz.
        lfo_freq (float): LFO frequency in Hz.
        min_delay_samples (int): Minimum delay in samples.
        max_delay_samples (int): Maximum delay in samples.
        comb_mix_factor (float): Mix between dry and delayed signals (0-1).
        lfo_phase_offset_right_rad (float): Phase offset for the right channel LFO.

    Returns:
        tuple: Left and right processed channels.
    """
    num_samples = len(input_left)
    left_channel_out = np.empty(num_samples, dtype=np.float32)
    right_channel_out = np.empty(num_samples, dtype=np.float32)

    # Ensure the delay line is large enough for the maximum delay plus some margin
    delay_line_size = int(max_delay_samples + 10)
    if delay_line_size <= 0:
        delay_line_size = int(sample_rate * 0.05)

    # Initialize delay lines (circular buffers) for left and right channels.
    # The same pink noise source is fed into two different comb filter paths.
    delay_line_l = np.zeros(delay_line_size, dtype=np.float32)
    delay_line_r = np.zeros(delay_line_size, dtype=np.float32)

    # Current write position in the circular delay lines
    write_ptr = 0 # Can use a single write pointer if pink_noise_input is mono
                  # and written to both delay lines identically.
    
    # Time vector for LFO calculation
    t_indices = np.arange(num_samples) # Using indices for direct use in loop

    # Calculate LFO modulation range
    delay_span = max_delay_samples - min_delay_samples
    delay_center = (max_delay_samples + min_delay_samples) / 2.0

    for i in range(num_samples):
        current_sample_l = input_left[i]
        current_sample_r = input_right[i]
        time_sec = t_indices[i] / sample_rate # current time in seconds

        # --- Left Channel LFO and Delay ---
        # "The low-frequency sine wave is used directly to sweep the delay on one channel."
        lfo_val_l = np.sin(2 * np.pi * lfo_freq * time_sec) # Ranges from -1 to 1
        # Modulate delay: LFO maps its -1 to 1 range to [min_delay_samples, max_delay_samples]
        current_delay_l_exact = delay_center + (lfo_val_l * delay_span / 2.0)
        # Ensure delay is an integer and within buffer bounds
        current_delay_l_samps = int(round(current_delay_l_exact))
        if current_delay_l_samps < 1: current_delay_l_samps = 1
        if current_delay_l_samps >= delay_line_size: current_delay_l_samps = delay_line_size - 1
        
        # Read from left delay line (using integer indexing for simplicity with Numba)
        # Interpolated reads would be smoother but add complexity.
        read_ptr_l = (write_ptr - current_delay_l_samps + delay_line_size) % delay_line_size
        delayed_sample_l = delay_line_l[read_ptr_l]
        
        # Comb filter: "averaged with an earlier sample from the delay line"
        # output = (1-mix_factor)*current_pink_sample + mix_factor*delayed_sample
        left_channel_out[i] = (1.0 - comb_mix_factor) * current_sample_l + comb_mix_factor * delayed_sample_l

        # Write current sample to delay line
        delay_line_l[write_ptr] = current_sample_l

        # --- Right Channel LFO and Delay ---
        # "The delay on the other channel is controlled by some mix of the sine and cosine waves."
        # This is achieved by a phase-shifted sine wave.
        lfo_val_r_arg = 2 * np.pi * lfo_freq * time_sec + lfo_phase_offset_right_rad
        lfo_val_r = np.sin(lfo_val_r_arg) # Ranges from -1 to 1
        current_delay_r_exact = delay_center + (lfo_val_r * delay_span / 2.0)
        current_delay_r_samps = int(round(current_delay_r_exact))
        if current_delay_r_samps < 1: current_delay_r_samps = 1
        if current_delay_r_samps >= delay_line_size: current_delay_r_samps = delay_line_size - 1

        read_ptr_r = (write_ptr - current_delay_r_samps + delay_line_size) % delay_line_size
        delayed_sample_r = delay_line_r[read_ptr_r]
        
        right_channel_out[i] = (1.0 - comb_mix_factor) * current_sample_r + comb_mix_factor * delayed_sample_r

        delay_line_r[write_ptr] = current_sample_r

        # Advance write pointer for the next iteration
        write_ptr = (write_ptr + 1) % delay_line_size
        
    return left_channel_out, right_channel_out


def generate_phased_pink_sound_file(
    filename="phased_pink_sound.wav",
    duration_seconds=100,
    sample_rate=DEFAULT_SAMPLE_RATE,
    lfo_freq=DEFAULT_LFO_FREQ,
    min_delay_ms=60,      # Minimum delay for comb filter sweep in milliseconds
    max_delay_ms=80.0,     # Maximum delay for comb filter sweep in milliseconds
    comb_mix_factor=0.5,   # Mix of direct vs. delayed sound in comb filter (0.0 to 1.0)
                           # 0.5 implies an equal average.
    lfo_right_phase_deg=90, # Phase offset for the right channel's LFO in degrees.
    input_audio_path=None,
):
    """
    Generates the flanging effect on either generated pink noise or an input
    audio file.

    Args:
        filename (str): Name of the output WAV file.
        duration_seconds (float): Duration of the generated noise when
            ``input_audio_path`` is ``None``.
        sample_rate (int): Audio sample rate in Hz.
        lfo_freq (float): LFO frequency for sweeping the comb filter delays.
        min_delay_ms (float): Minimum delay for the comb filter in milliseconds.
        max_delay_ms (float): Maximum delay for the comb filter in milliseconds.
        comb_mix_factor (float): The balance between the original pink noise and the
                                 delayed version in the comb filter. 0.5 for equal mix.
        lfo_right_phase_deg (float): Phase offset for the right channel's LFO.
        input_audio_path (str or None): Optional path to an audio file to
            process instead of generating pink noise.
    """
    print(f"Starting Phased Pink Sound generation for '{filename}'...")
    print(f"Parameters: Duration={duration_seconds}s, Sample Rate={sample_rate}Hz, LFO Freq={lfo_freq}Hz")
    
    if input_audio_path is None:
        num_samples = int(duration_seconds * sample_rate)
        print("Step 1/4: Generating base pink noise...")
        input_left = generate_pink_noise_samples(num_samples)
        input_right = input_left
    else:
        print(f"Step 1/4: Loading input audio from '{input_audio_path}'...")
        data, in_sr = sf.read(input_audio_path, always_2d=True)
        if in_sr != sample_rate:
            import librosa
            data = librosa.resample(data.T, orig_sr=in_sr, target_sr=sample_rate, axis=1).T
        input_left = data[:, 0]
        if data.shape[1] > 1:
            input_right = data[:, 1]
        else:
            input_right = input_left
        num_samples = len(input_left)
    

    # Step 2: Prepare parameters for the Numba core function
    print("Step 2/4: Preparing parameters for core processing...")
    min_delay_samps = int((min_delay_ms / 1000.0) * sample_rate)
    max_delay_samps = int((max_delay_ms / 1000.0) * sample_rate)
    
    # Ensure valid delay ranges
    if min_delay_samps < 1: 
        print(f"Warning: min_delay_ms ({min_delay_ms}ms) is too short for current sample rate. Setting to 1 sample.")
        min_delay_samps = 1
    if max_delay_samps <= min_delay_samps:
        max_delay_samps = min_delay_samps + int(0.01 * sample_rate) # Ensure a sweep range of at least 10ms
        print(f"Warning: max_delay_ms was less than or equal to min_delay_ms. Adjusted max_delay_samps to {max_delay_samps}.")
    
    lfo_right_phase_rad = np.deg2rad(lfo_right_phase_deg)
    
    print(f"  Comb filter delay range: {min_delay_samps} samples to {max_delay_samps} samples.")
    print(f"  LFO for right channel phase offset: {lfo_right_phase_deg} degrees ({lfo_right_phase_rad:.2f} radians).")

    # Step 3: Call the core Numba-JIT compiled generator
    print("Step 3/4: Applying phased comb filtering (Numba accelerated)...")
    left_channel_output, right_channel_output = _stereo_flanger_core(
        input_left,
        input_right,
        float(sample_rate),
        lfo_freq,
        min_delay_samps,
        max_delay_samps,
        comb_mix_factor,
        lfo_right_phase_rad,
    )

    # Step 4: Combine into stereo and normalize for output
    print("Step 4/4: Normalizing and saving audio...")
    stereo_signal = np.stack((left_channel_output, right_channel_output), axis=-1)
    
    # Normalize the final stereo output to prevent clipping (max absolute value to +/- 1.0)
    max_abs_final = np.max(np.abs(stereo_signal))
    if max_abs_final == 0:
        print("Warning: Generated signal is completely silent.")
    elif max_abs_final > 1.0 : # Only normalize if values exceed the standard -1 to 1 range
         stereo_signal /= max_abs_final
    # Optionally, apply a final gain reduction if desired, e.g., stereo_signal *= 0.8

    # Save to WAV file
    try:
        sf.write(filename, stereo_signal, sample_rate, subtype='PCM_16') # 16-bit PCM
        print(f"Successfully generated and saved Phased Pink Sound to '{filename}'")
    except Exception as e:
        print(f"Error saving audio file: {e}")
        print("Please ensure you have 'soundfile' and its dependencies (like libsndfile) installed.")
        print("You can install it via pip: pip install soundfile")

# --- Main execution ---
if __name__ == '__main__':
    
    output_filename = "monroe_phased_pink_sound.wav"
    duration = 3600# seconds, for a good sample
    
    generate_phased_pink_sound_file(
        filename=output_filename,
        duration_seconds=duration,
        sample_rate=44100,         # Standard CD quality sample rate
        lfo_freq=1.0/8.0,          # As per patent: "approximate sweep rate is about 1/8 Hz"
        min_delay_ms=0.5,          # Typical short delay for flanging/phasing (e.g., 1-5 ms)
        max_delay_ms=10,         # Typical longer delay for phasing (e.g., 10-25 ms)
        comb_mix_factor=0.75,       # Slightly more delayed signal, adjust to taste (0.5 is equal)
        lfo_right_phase_deg=90   # 90 degrees creates a quadrature LFO pair (sine/cosine like)
                                   # for strong stereo imaging.
    )

