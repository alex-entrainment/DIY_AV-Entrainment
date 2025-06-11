import numpy as np
import numba
import soundfile as sf
from scipy import signal
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# --- Parameters ---
DEFAULT_SAMPLE_RATE = 44100  # Hz
DEFAULT_LFO_FREQ = 1.0 / 12.0  # Hz, for 12-second period

@numba.jit(nopython=True, cache=True, parallel=True)
def generate_pink_noise_samples(n_samples):
    """
    Generates pink noise samples with reduced high frequency content.
    Optimized with parallel execution for large arrays.
    """
    white = np.random.randn(n_samples).astype(np.float32)
    pink = np.empty_like(white)

    # Process in chunks for better cache utilization
    chunk_size = 65536
    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    
    for chunk_idx in numba.prange(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, n_samples)
        
        # State variables for the pinking filter
        b0, b1, b2, b3, b4, b5 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        for i in range(start, end):
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


@numba.jit(nopython=True, cache=True)
def generate_brown_noise_samples(n_samples):
    """Generate brown (red) noise samples."""
    white = np.random.randn(n_samples).astype(np.float32)
    brown = np.cumsum(white)
    max_abs = np.max(np.abs(brown)) + 1e-8
    return (brown / max_abs).astype(np.float32)


class FilterBank:
    """Pre-computed filter bank for efficient processing."""
    def __init__(self, sample_rate, min_freq, max_freq, num_freqs=100):
        self.sample_rate = sample_rate
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.num_freqs = num_freqs
        self.filters = {}
        self._precompute_filters()
    
    def _precompute_filters(self):
        """Pre-compute filter coefficients for a range of frequencies."""
        freqs = np.linspace(self.min_freq, self.max_freq, self.num_freqs)
        for freq in freqs:
            for q in [25, 50, 75, 100, 125, 150, 200]:
                try:
                    b, a = signal.iirnotch(freq, q, self.sample_rate)
                    self.filters[(freq, q)] = (b, a)
                except:
                    pass
    
    def get_filter(self, freq, q):
        """Get filter coefficients, using nearest pre-computed if available."""
        # Find nearest pre-computed filter
        best_freq = min(self.filters.keys(), 
                       key=lambda x: abs(x[0] - freq) if x[1] == q else float('inf'))
        if abs(best_freq[0] - freq) < 50 and best_freq[1] == q:
            return self.filters[best_freq]
        else:
            # Compute on the fly if not close enough
            try:
                return signal.iirnotch(freq, q, self.sample_rate)
            except:
                return None, None


def apply_swept_notches_fft(input_signal, sample_rate, lfo_freq,
                           min_freq=1000, max_freq=10000,
                           num_notches=6, notch_spacing_ratio=1.1,
                           notch_q=100, cascade_count=3,
                           phase_offset=0):
    """
    FFT-based swept notch implementation for better performance on long segments.
    """
    n_samples = len(input_signal)
    
    # Generate LFO
    t = np.arange(n_samples) / sample_rate
    lfo = np.cos(2 * np.pi * lfo_freq * t + phase_offset)
    center_freq = (min_freq + max_freq) / 2
    freq_range = (max_freq - min_freq) / 2
    base_freq_sweep = center_freq + freq_range * lfo
    
    # Process in large FFT blocks for efficiency
    fft_size = 2**18  # 262144 samples (~6 seconds at 44.1kHz)
    hop_size = fft_size // 2
    
    output = np.zeros(n_samples)
    window = np.hanning(fft_size)
    
    for start_idx in range(0, n_samples, hop_size):
        end_idx = min(start_idx + fft_size, n_samples)
        block_size = end_idx - start_idx
        
        if block_size < fft_size // 4:
            continue
        
        # Pad block if necessary
        block = np.zeros(fft_size)
        block[:block_size] = input_signal[start_idx:end_idx]
        
        # Window the block
        windowed_block = block * window
        
        # FFT
        spectrum = np.fft.rfft(windowed_block)
        freqs = np.fft.rfftfreq(fft_size, 1/sample_rate)
        
        # Get center frequency for this block
        block_center_idx = start_idx + block_size // 2
        if block_center_idx < len(base_freq_sweep):
            center_freq_for_block = base_freq_sweep[block_center_idx]
        else:
            center_freq_for_block = base_freq_sweep[-1]
        
        # Apply notches in frequency domain
        for notch_idx in range(num_notches):
            notch_freq = center_freq_for_block * (notch_spacing_ratio ** notch_idx)
            
            if notch_freq > sample_rate * 0.48:
                continue
            
            # Create notch in frequency domain
            notch_width = notch_freq / notch_q
            for _ in range(cascade_count):
                mask = np.exp(-((freqs - notch_freq) / notch_width) ** 2)
                spectrum *= (1 - 0.99 * (1 - mask))
        
        # IFFT
        filtered_block = np.fft.irfft(spectrum, n=fft_size)
        
        # Overlap-add
        output[start_idx:end_idx] += filtered_block[:block_size] * window[:block_size]
    
    # Normalize overlap regions
    for start_idx in range(hop_size, n_samples, hop_size):
        overlap_start = start_idx - hop_size // 2
        overlap_end = start_idx + hop_size // 2
        if overlap_end <= n_samples:
            output[overlap_start:overlap_end] *= 0.5
    
    return output


def process_channel(args):
    """Process a single channel - used for parallel processing."""
    (input_signal, sample_rate, lfo_freq, min_freq, max_freq, 
     num_notches, notch_spacing_ratio, notch_q, cascade_count, 
     phase_offset, extra_phase_offset, use_fft) = args
    
    if use_fft and len(input_signal) > 441000:  # Use FFT for > 10 seconds
        output = apply_swept_notches_fft(
            input_signal, sample_rate, lfo_freq,
            min_freq, max_freq, num_notches,
            notch_spacing_ratio, notch_q, cascade_count,
            phase_offset
        )
        if extra_phase_offset:
            output = apply_swept_notches_fft(
                output, sample_rate, lfo_freq,
                min_freq, max_freq, num_notches,
                notch_spacing_ratio, notch_q, cascade_count,
                phase_offset + extra_phase_offset
            )
    else:
        # Use original method for shorter segments
        from paste import apply_deep_swept_notches
        output = apply_deep_swept_notches(
            input_signal, sample_rate, lfo_freq,
            min_freq, max_freq, num_notches,
            notch_spacing_ratio, notch_q, cascade_count,
            phase_offset, extra_phase_offset
        )
    
    return output


def generate_swept_notch_pink_sound_optimized(
    filename="swept_notch_pink_sound.wav",
    duration_seconds=60,
    sample_rate=DEFAULT_SAMPLE_RATE,
    lfo_freq=DEFAULT_LFO_FREQ,
    min_freq=1000,
    max_freq=10000,
    num_notches=6,
    notch_spacing_ratio=1.1,
    notch_q=100,
    cascade_count=3,
    lfo_phase_offset_deg=90,
    intra_phase_offset_deg=0,
    input_audio_path=None,
    noise_type="pink",
    use_parallel=True,
    use_fft=True,
    chunk_minutes=10
):
    """
    Optimized version with parallel processing and FFT option.
    
    Parameters
    ----------
    use_parallel : bool
        Use parallel processing for stereo channels
    use_fft : bool
        Use FFT-based filtering for long segments
    chunk_minutes : int
        Process audio in chunks of this many minutes for very long files
    """
    print(f"Starting Optimized Deep Swept Notch generation for '{filename}'...")
    print(f"Parameters: Duration={duration_seconds}s, Sample Rate={sample_rate}Hz")
    print(f"Optimizations: Parallel={use_parallel}, FFT={use_fft}")
    
    # Step 1: Generate or load input audio
    if input_audio_path is None:
        num_samples = int(duration_seconds * sample_rate)
        samples_per_chunk = int(chunk_minutes * 60 * sample_rate)
        
        if num_samples > samples_per_chunk:
            print(f"Processing in {chunk_minutes}-minute chunks...")
            
            # Process in chunks for memory efficiency
            with sf.SoundFile(filename, 'w', sample_rate, 2, 'PCM_16') as outfile:
                for chunk_start in range(0, num_samples, samples_per_chunk):
                    chunk_end = min(chunk_start + samples_per_chunk, num_samples)
                    chunk_samples = chunk_end - chunk_start
                    chunk_duration = chunk_samples / sample_rate
                    
                    print(f"\nProcessing chunk {chunk_start//samples_per_chunk + 1}/"
                          f"{(num_samples + samples_per_chunk - 1)//samples_per_chunk}")
                    
                    # Generate noise for this chunk
                    if noise_type.lower() == "brown":
                        noise_mono = generate_brown_noise_samples(chunk_samples)
                    else:
                        noise_mono = generate_pink_noise_samples(chunk_samples)
                    
                    # Apply filtering
                    b_warmth, a_warmth = signal.butter(1, 10000, btype='low', fs=sample_rate)
                    noise_mono = signal.filtfilt(b_warmth, a_warmth, noise_mono)
                    
                    b_hpf, a_hpf = signal.butter(2, 50, btype='high', fs=sample_rate)
                    noise_mono = signal.filtfilt(b_hpf, a_hpf, noise_mono)
                    
                    noise_mono = noise_mono / (np.max(np.abs(noise_mono)) + 1e-8) * 0.8
                    
                    # Process channels
                    phase_offset_rad = np.deg2rad(lfo_phase_offset_deg)
                    intra_phase_rad = np.deg2rad(intra_phase_offset_deg)
                    
                    if use_parallel:
                        with ThreadPoolExecutor(max_workers=2) as executor:
                            left_args = (noise_mono.copy(), sample_rate, lfo_freq, min_freq, max_freq,
                                       num_notches, notch_spacing_ratio, notch_q, cascade_count,
                                       0, intra_phase_rad, use_fft)
                            right_args = (noise_mono.copy(), sample_rate, lfo_freq, min_freq, max_freq,
                                        num_notches, notch_spacing_ratio, notch_q, cascade_count,
                                        phase_offset_rad, intra_phase_rad, use_fft)
                            
                            futures = [executor.submit(process_channel, left_args),
                                     executor.submit(process_channel, right_args)]
                            
                            left_output = futures[0].result()
                            right_output = futures[1].result()
                    else:
                        left_output = process_channel(
                            (noise_mono.copy(), sample_rate, lfo_freq, min_freq, max_freq,
                             num_notches, notch_spacing_ratio, notch_q, cascade_count,
                             0, intra_phase_rad, use_fft)
                        )
                        right_output = process_channel(
                            (noise_mono.copy(), sample_rate, lfo_freq, min_freq, max_freq,
                             num_notches, notch_spacing_ratio, notch_q, cascade_count,
                             phase_offset_rad, intra_phase_rad, use_fft)
                        )
                    
                    # Combine and normalize
                    stereo_output = np.stack((left_output, right_output), axis=-1)
                    stereo_output = np.tanh(stereo_output * 0.7) / 0.7
                    max_val = np.max(np.abs(stereo_output))
                    if max_val > 0:
                        stereo_output = stereo_output / max_val * 0.95
                    
                    # Write chunk
                    outfile.write(stereo_output)
            
            print(f"\nSuccessfully generated and saved to '{filename}'")
            return
        
        # For shorter files, process normally
        print(f"Step 1/3: Generating base {noise_type} noise...")
        if noise_type.lower() == "brown":
            noise_mono = generate_brown_noise_samples(num_samples)
        else:
            noise_mono = generate_pink_noise_samples(num_samples)
        
        # Additional filtering for warmer sound
        b_warmth, a_warmth = signal.butter(1, 10000, btype='low', fs=sample_rate)
        noise_mono = signal.filtfilt(b_warmth, a_warmth, noise_mono)
        
        b_hpf, a_hpf = signal.butter(2, 50, btype='high', fs=sample_rate)
        noise_mono = signal.filtfilt(b_hpf, a_hpf, noise_mono)
        
        noise_mono = noise_mono / (np.max(np.abs(noise_mono)) + 1e-8) * 0.8
        input_left = noise_mono.copy()
        input_right = noise_mono.copy()
    else:
        print(f"Step 1/3: Loading input audio from '{input_audio_path}'...")
        data, in_sr = sf.read(input_audio_path, always_2d=True)
        if in_sr != sample_rate:
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
        
        max_val = max(np.max(np.abs(input_left)), np.max(np.abs(input_right)))
        if max_val > 0:
            input_left = input_left / max_val * 0.8
            input_right = input_right / max_val * 0.8
    
    # Step 2: Apply deep swept notches
    print("Step 2/3: Applying deep swept notches...")
    
    phase_offset_rad = np.deg2rad(lfo_phase_offset_deg)
    intra_phase_rad = np.deg2rad(intra_phase_offset_deg)
    
    if use_parallel and mp.cpu_count() > 1:
        print(f"  Using parallel processing with {mp.cpu_count()} cores...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            left_args = (input_left, sample_rate, lfo_freq, min_freq, max_freq,
                        num_notches, notch_spacing_ratio, notch_q, cascade_count,
                        0, intra_phase_rad, use_fft)
            right_args = (input_right, sample_rate, lfo_freq, min_freq, max_freq,
                         num_notches, notch_spacing_ratio, notch_q, cascade_count,
                         phase_offset_rad, intra_phase_rad, use_fft)
            
            futures = [executor.submit(process_channel, left_args),
                      executor.submit(process_channel, right_args)]
            
            left_output = futures[0].result()
            right_output = futures[1].result()
    else:
        print("  Processing left channel...")
        left_output = process_channel(
            (input_left, sample_rate, lfo_freq, min_freq, max_freq,
             num_notches, notch_spacing_ratio, notch_q, cascade_count,
             0, intra_phase_rad, use_fft)
        )
        
        print("  Processing right channel...")
        right_output = process_channel(
            (input_right, sample_rate, lfo_freq, min_freq, max_freq,
             num_notches, notch_spacing_ratio, notch_q, cascade_count,
             phase_offset_rad, intra_phase_rad, use_fft)
        )
    
    # Step 3: Final processing and save
    print("Step 3/3: Final processing and saving...")
    
    stereo_output = np.stack((left_output, right_output), axis=-1)
    stereo_output = np.tanh(stereo_output * 0.7) / 0.7
    
    max_val = np.max(np.abs(stereo_output))
    if max_val > 0:
        stereo_output = stereo_output / max_val * 0.95
    
    try:
        sf.write(filename, stereo_output, sample_rate, subtype='PCM_16')
        print(f"Successfully generated and saved to '{filename}'")
    except Exception as e:
        print(f"Error saving audio file: {e}")


# --- Main execution ---
if __name__ == '__main__':
    
    output_filename = "monroe_phased_pink_sound_optimized.wav"
    duration = 4000  # ~67 minutes
    
    # Generate with optimizations
    generate_swept_notch_pink_sound_optimized(
        filename=output_filename,
        duration_seconds=duration,
        sample_rate=44100,
        lfo_freq=1.0/12.0,
        min_freq=1000,
        max_freq=10000,
        num_notches=1,
        notch_spacing_ratio=1.25,
        notch_q=25,
        cascade_count=20,
        lfo_phase_offset_deg=90,
        intra_phase_offset_deg=0,
        input_audio_path=None,
        noise_type="pink",
        use_parallel=True,      # Enable parallel processing
        use_fft=True,          # Use FFT for long segments
        chunk_minutes=10       # Process in 10-minute chunks
    )
