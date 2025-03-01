import numpy as np
import wave
import struct
import math
import random

def generate_rfm_offset(total_frames, sr, rfm_range, rfm_speed):
    """
    Generate a random frequency modulation (RFM) offset as a smooth random walk.
    Instead of random flips per sample, this uses a Gaussian random walk
    and clips the offset to ±rfm_range.
    """
    step_std = rfm_speed / sr
    steps = np.random.normal(loc=0.0, scale=step_std, size=total_frames)
    rfm = np.cumsum(steps)
    rfm = np.clip(rfm, -rfm_range, rfm_range)
    return rfm

def generate_pink_noise(total_frames, noise_level=0.1):
    """
    Generate pink noise using the Voss-McCartney algorithm.
    Pink noise has power that is inversely proportional to frequency (1/f).
    
    Parameters:
    total_frames (int): Number of samples to generate
    noise_level (float): Amplitude of the noise (0.0 to 1.0)
    
    Returns:
    np.array: Pink noise samples scaled to specified level
    """
    # Number of random generators to use
    num_generators = 16
    
    # Create array of white noise generators
    white_values = np.random.randn(num_generators, total_frames//num_generators + 1)
    
    # Generate pink noise by summing
    pink_array = np.zeros(total_frames)
    for i in range(num_generators):
        pink_array += np.repeat(white_values[i, :], 2**i)[:total_frames]
    
    # Normalize and scale
    pink_array = pink_array / np.sqrt(num_generators)
    pink_array = pink_array / np.max(np.abs(pink_array)) * noise_level
    
    return pink_array

def add_background_noise(signal, sr, noise_type='pink', noise_level=0.1):
    """
    Add background noise to the signal.
    
    Parameters:
    signal (np.array): Input signal to add noise to
    sr (int): Sample rate
    noise_type (str): Type of noise ('pink', 'white')
    noise_level (float): Level of noise (0.0 to 1.0)
    
    Returns:
    np.array: Signal with added noise
    """
    if noise_type is None or noise_level <= 0.001:
        return signal
    
    if noise_type == 'pink':
        noise = generate_pink_noise(signal.shape[0], noise_level)
        # Duplicate for stereo if needed
        if len(signal.shape) > 1 and signal.shape[1] == 2:
            noise = np.column_stack((noise, noise))
    else:  # Default to white noise
        noise = np.random.normal(0.0, noise_level, signal.shape)
    
    return signal + noise

def generate_waveform(duration, sr, settings):
    """
    Generate a stereo waveform array based on the settings.
    Supports binaural, isochronic, and monaural modes with multiple carriers.
    Returns a NumPy array of shape (total_frames, 2) with values in [-1, 1].
    (This function is used for continuous audio generation outside the step-based mode.)
    """
    total_frames = int(duration * sr)
    t = np.linspace(0, duration, total_frames, endpoint=False)
    
    # Check if RFM is enabled globally
    global_rfm_enabled = settings.get('enable_rfm', False)
    global_rfm_range = settings.get('rfm_range', 0.0)
    global_rfm_speed = settings.get('rfm_speed', 0.0)
    
    # Initialize stereo output
    left_channel = np.zeros(total_frames)
    right_channel = np.zeros(total_frames)
    
    # Get carriers from settings
    carriers = settings.get('carriers', [])
    if not carriers:
        # Legacy support: use single carrier from old settings
        carriers = [{
            'enabled': True,
            'start_freq': settings.get('carrier_freq', 200.0),
            'end_freq': settings.get('carrier_freq', 200.0),
            'volume': 1.0,
            'enable_rfm': global_rfm_enabled,
            'rfm_range': global_rfm_range,
            'rfm_speed': global_rfm_speed
        }]
    
    beat_freq = settings.get('beat_freq', 10.0)
    is_binaural = settings.get('is_binaural', False)
    is_isochronic = settings.get('is_isochronic', False)
    
    # Process each carrier
    for carrier in carriers:
        if not carrier.get('enabled', True):
            continue
            
        # Get carrier settings
        volume = carrier.get('volume', 1.0)
        
        # Get channel-specific frequency settings if available, otherwise use defaults
        start_freq_left = carrier.get('start_freq_left', carrier.get('start_freq', 200.0))
        end_freq_left = carrier.get('end_freq_left', carrier.get('end_freq', 200.0))
        start_freq_right = carrier.get('start_freq_right', carrier.get('start_freq', 200.0))
        end_freq_right = carrier.get('end_freq_right', carrier.get('end_freq', 200.0))
        
        # Apply carrier-specific RFM if enabled, otherwise use global RFM
        if carrier.get('enable_rfm', False):
            rfm_range = carrier.get('rfm_range', 0.5)
            rfm_speed = carrier.get('rfm_speed', 0.2)
            rfm_offset_left = generate_rfm_offset(total_frames, sr, rfm_range, rfm_speed)
            rfm_offset_right = generate_rfm_offset(total_frames, sr, rfm_range, rfm_speed)
        elif global_rfm_enabled:
            rfm_offset_left = generate_rfm_offset(total_frames, sr, global_rfm_range, global_rfm_speed)
            rfm_offset_right = generate_rfm_offset(total_frames, sr, global_rfm_range, global_rfm_speed)
        else:
            rfm_offset_left = np.zeros(total_frames)
            rfm_offset_right = np.zeros(total_frames)
        
        # Create frequency ramps for left and right channels
        freq_left = np.linspace(start_freq_left, end_freq_left, total_frames)
        freq_right = np.linspace(start_freq_right, end_freq_right, total_frames)
        
        # Apply RFM offsets to carrier frequencies
        effective_freq_left = freq_left + rfm_offset_left
        effective_freq_right = freq_right + rfm_offset_right
        
        # Generate waveforms based on mode
        if is_binaural:
            # In binaural mode, use directly specified left/right frequencies
            phase_left = 2 * np.pi * np.cumsum(effective_freq_left) / sr
            phase_right = 2 * np.pi * np.cumsum(effective_freq_right) / sr
            
            carrier_left = np.sin(phase_left) * volume
            carrier_right = np.sin(phase_right) * volume
            
        elif is_isochronic:
            # In isochronic mode, use the average of left and right frequencies
            avg_freq = (effective_freq_left + effective_freq_right) / 2
            phase = 2 * np.pi * np.cumsum(avg_freq) / sr
            carrier_signal = np.sin(phase) * volume
            
            # Calculate the isochronic modulation frequency from the difference
            # between left and right channels (or use the beat_freq if difference is 0)
            mod_freq = np.abs(effective_freq_left - effective_freq_right)
            if np.mean(mod_freq) < 0.1:  # If channels are nearly identical
                mod_freq = np.ones(total_frames) * beat_freq
                
            # Generate modulation phase by integrating the mod_freq over time
            mod_phase = 2 * np.pi * np.cumsum(mod_freq) / sr
            mod_signal = np.where(np.sin(mod_phase) >= 0, 1.0, 0.0)
            
            carrier_left = carrier_signal * mod_signal
            carrier_right = carrier_left
            
        else:  # Monaural or default
            # In monaural mode, use the average of left and right frequencies
            avg_freq = (effective_freq_left + effective_freq_right) / 2
            phase = 2 * np.pi * np.cumsum(avg_freq) / sr
            carrier_signal = np.sin(phase) * volume
            carrier_left = carrier_signal
            carrier_right = carrier_signal
        
        # Add this carrier to the output channels
        left_channel += carrier_left
        right_channel += carrier_right
    
    # Normalize if sum of carriers exceeds [-1, 1]
    max_amplitude = max(np.max(np.abs(left_channel)), np.max(np.abs(right_channel)))
    if max_amplitude > 1.0:
        left_channel = left_channel / max_amplitude
        right_channel = right_channel / max_amplitude
    
    # Combine channels into stereo
    stereo_wave = np.vstack((left_channel, right_channel)).T
    
    # Add background noise if enabled
    if settings.get('enable_pink_noise', False):
        noise_level = settings.get('pink_noise_volume', 0.1)
        stereo_wave = add_background_noise(stereo_wave, sr, 'pink', noise_level)
    
    return stereo_wave

def generate_audio_file(audio_filename, duration, audio_settings):
    """
    Generate a .wav file of 'duration' seconds using parameters from 'audio_settings'.
    """
    if not audio_settings.get('enabled', True):
        print("Audio is disabled. Not generating audio file.")
        return

    print(f"Generating audio file: {audio_filename} (Duration: {duration:.1f} s)")
    
    sr = audio_settings.get('sample_rate', 44100)
    amplitude = 32767
    
    waveform = generate_waveform(duration, sr, audio_settings)
    
    waveform_int16 = (amplitude * waveform).astype(np.int16)
    
    with wave.open(audio_filename, 'wb') as wavef:
        num_channels = 2
        wavef.setnchannels(num_channels)
        wavef.setsampwidth(2)
        wavef.setframerate(sr)
        wavef.writeframes(waveform_int16.tobytes())
    
    print(f"Done. Generated {audio_filename}")

def generate_audio_file_for_steps_offline_rfm(steps, audio_filename, audio_settings):
    """
    Generate a single .wav file for a sequence of steps with enhanced features:
    - Multiple carrier frequencies (up to 3)
    - Independent left/right channel frequencies for each carrier
    - Carrier frequency transitions for left and right channels
    - Optional background pink noise
    
    For each step the primary oscillator (steps[0]) is used to determine the LED pulse rate.
    Each carrier now has separate frequency settings for left and right channels.
    
    In binaural mode:
      - Left/right frequencies are directly used from carrier settings
      - Binaural beat is the difference between left and right frequencies
    
    In isochronic mode:
      - Carrier is the average of left and right frequencies
      - Modulation frequency is derived from the difference between channels
    
    RFM functionality is preserved for both global and carrier-specific settings.
    """
    sr = audio_settings.get('sample_rate', 44100)
    amplitude = 32767

    # Global audio settings
    global_enable_rfm = audio_settings.get('enable_rfm', False)
    global_rfm_range = audio_settings.get('rfm_range', 0.5)
    global_rfm_speed = audio_settings.get('rfm_speed', 0.2)
    binaural = audio_settings.get('is_binaural', True)
    isochronic = audio_settings.get('is_isochronic', False)
    
    # Get carriers from settings
    carriers = audio_settings.get('carriers', [])
    if not carriers:
        # Legacy support: use single carrier from old settings
        carriers = [{
            'enabled': True,
            'start_freq': audio_settings.get('carrier_freq', 200.0),
            'end_freq': audio_settings.get('carrier_freq', 200.0),
            'volume': 1.0,
            'enable_rfm': global_enable_rfm,
            'rfm_range': global_rfm_range,
            'rfm_speed': global_rfm_speed
        }]
    
    # Pink noise settings
    enable_pink_noise = audio_settings.get('enable_pink_noise', False)
    pink_noise_volume = audio_settings.get('pink_noise_volume', 0.1)
    
    samples_list = []
    
    # For continuous phase accumulation across steps
    phase_accumulator = {
        'carrier': {},  # Dictionary to store phase for each carrier
        'mod': 0.0      # Phase for modulation (used in isochronic mode)
    }
    
    for step in steps:
        # Use the primary oscillator of the step to determine the LED pulse (beat) frequency.
        osc = step.oscillators[0]
        duration = step.duration
        total_frames = int(duration * sr)
        t = np.linspace(0, duration, total_frames, endpoint=False)
        
        # Compute the LED pulse frequency as a linear ramp from osc.start_freq to osc.end_freq.
        # This is kept for backward compatibility and for carriers that don't specify left/right frequencies
        led_freq = osc.start_freq + (osc.end_freq - osc.start_freq) * (t / duration)
        
        # Apply offline RFM to the LED frequency if enabled.
        if global_enable_rfm:
            dt = 1.0 / sr
            steps_rand = np.random.normal(loc=0.0, scale=global_rfm_speed * dt, size=total_frames)
            rfm_offset = np.cumsum(steps_rand)
            rfm_offset = np.clip(rfm_offset, -global_rfm_range, global_rfm_range)
        else:
            rfm_offset = np.zeros(total_frames)
        
        # The effective LED frequency (for backward compatibility)
        effective_led_freq = led_freq + rfm_offset
        
        # Initialize stereo arrays for this step
        left_channel = np.zeros(total_frames)
        right_channel = np.zeros(total_frames)
        
        # Process each carrier
        for carrier_idx, carrier in enumerate(carriers):
            if not carrier.get('enabled', True):
                continue
                
            # Get carrier settings
            volume = carrier.get('volume', 1.0)
            
            # Check for channel-specific frequencies
            start_freq = carrier.get('start_freq', 200.0)
            end_freq = carrier.get('end_freq', 200.0)
            
            # Get or derive left channel frequencies
            start_freq_left = carrier.get('start_freq_left', start_freq + 0.5 * osc.start_freq)
            end_freq_left = carrier.get('end_freq_left', end_freq + 0.5 * osc.end_freq)
            
            # Get or derive right channel frequencies
            start_freq_right = carrier.get('start_freq_right', start_freq - 0.5 * osc.start_freq)
            end_freq_right = carrier.get('end_freq_right', end_freq - 0.5 * osc.end_freq)
            
            # Create a unique key for this carrier in the phase accumulator
            carrier_key = f'carrier_{carrier_idx}'
            if carrier_key not in phase_accumulator['carrier']:
                phase_accumulator['carrier'][carrier_key] = {'left': 0.0, 'right': 0.0}
            
            # Apply carrier-specific RFM if enabled
            if carrier.get('enable_rfm', False):
                rfm_range = carrier.get('rfm_range', 0.5)
                rfm_speed = carrier.get('rfm_speed', 0.2)
                rfm_left = generate_rfm_offset(total_frames, sr, rfm_range, rfm_speed)
                rfm_right = generate_rfm_offset(total_frames, sr, rfm_range, rfm_speed)
            else:
                rfm_left = np.zeros(total_frames)
                rfm_right = np.zeros(total_frames)
            
            # Create frequency ramps for left and right channels
            freq_left = np.linspace(start_freq_left, end_freq_left, total_frames)
            freq_right = np.linspace(start_freq_right, end_freq_right, total_frames)
            
            # Apply RFM offsets
            freq_left += rfm_left
            freq_right += rfm_right
            
            # Generate audio based on the mode
            dt = 1.0 / sr
            if binaural:
                # In binaural mode, use the exact left/right frequencies
                carrier_left = np.zeros(total_frames)
                carrier_right = np.zeros(total_frames)
                
                for i in range(total_frames):
                    # Update phase accumulators using the channel-specific frequencies
                    phase_accumulator['carrier'][carrier_key]['left'] += 2 * math.pi * freq_left[i] * dt
                    phase_accumulator['carrier'][carrier_key]['right'] += 2 * math.pi * freq_right[i] * dt
                    
                    # Generate sine waves with the accumulated phases
                    carrier_left[i] = math.sin(phase_accumulator['carrier'][carrier_key]['left']) * volume
                    carrier_right[i] = math.sin(phase_accumulator['carrier'][carrier_key]['right']) * volume
                
            elif isochronic:
                # In isochronic mode, carrier is avg frequency, modulation is the difference
                carrier_signal = np.zeros(total_frames)
                mod_signal = np.zeros(total_frames)
                
                for i in range(total_frames):
                    # Use average frequency for carrier
                    avg_freq = (freq_left[i] + freq_right[i]) / 2
                    # Modulation freq is the difference (or fallback to LED freq)
                    mod_freq = abs(freq_left[i] - freq_right[i])
                    if mod_freq < 0.1:  # If frequencies are nearly identical
                        mod_freq = effective_led_freq[i]
                    
                    # Update carrier phase using average frequency
                    phase_accumulator['carrier'][carrier_key]['left'] += 2 * math.pi * avg_freq * dt
                    
                    # Update modulation phase based on the frequency difference
                    phase_accumulator['mod'] += 2 * math.pi * mod_freq * dt
                    
                    # Generate the carrier and modulation signals
                    carrier_signal[i] = math.sin(phase_accumulator['carrier'][carrier_key]['left'])
                    mod_signal[i] = 1.0 if math.sin(phase_accumulator['mod']) >= 0 else 0.0
                
                # Apply modulation and volume
                carrier_left = carrier_signal * mod_signal * volume
                carrier_right = carrier_left.copy()
                
            else:  # Monaural mode
                # In monaural mode, use average of left/right frequencies
                carrier_signal = np.zeros(total_frames)
                
                for i in range(total_frames):
                    # Use average frequency
                    avg_freq = (freq_left[i] + freq_right[i]) / 2
                    
                    # Update carrier phase using average frequency
                    phase_accumulator['carrier'][carrier_key]['left'] += 2 * math.pi * avg_freq * dt
                    
                    # Generate the carrier signal
                    carrier_signal[i] = math.sin(phase_accumulator['carrier'][carrier_key]['left']) * volume
                
                carrier_left = carrier_signal
                carrier_right = carrier_signal.copy()
            
            # Add this carrier's output to the channels
            left_channel += carrier_left
            right_channel += carrier_right
        
        # Normalize if the sum of carriers exceeds [-1, 1]
        max_amplitude = max(np.max(np.abs(left_channel)), np.max(np.abs(right_channel)))
        if max_amplitude > 1.0:
            left_channel = left_channel / max_amplitude
            right_channel = right_channel / max_amplitude
        
        # Combine into stereo
        stereo = np.vstack((left_channel, right_channel)).T
        
        # Add pink noise if enabled
        if enable_pink_noise:
            stereo = add_background_noise(stereo, sr, 'pink', pink_noise_volume)
        
        samples_list.append(stereo)
    
    # Stitch all step waveforms together.
    waveform = np.concatenate(samples_list, axis=0)
    waveform_int16 = (amplitude * waveform).astype(np.int16)

    with wave.open(audio_filename, 'wb') as wavef:
        wavef.setnchannels(2)
        wavef.setsampwidth(2)
        wavef.setframerate(sr)
        wavef.writeframes(waveform_int16.tobytes())
    
    print(f"Done. Generated stepwise audio file: {audio_filename}")

# Example usage:
if __name__ == '__main__':
    # This example generates a 30-second binaural tone with multiple carriers
    # using independent left/right channel frequencies
    settings = {
        'enabled': True,
        'carriers': [
            {
                'enabled': True,
                'start_freq': 200.0,  # Base frequency (for backward compatibility)
                'end_freq': 200.0,    # Base frequency (for backward compatibility)
                'start_freq_left': 205.0,  # Left channel specific start frequency
                'end_freq_left': 205.0,    # Left channel specific end frequency
                'start_freq_right': 195.0, # Right channel specific start frequency
                'end_freq_right': 195.0,   # Right channel specific end frequency
                'volume': 0.7,
                'enable_rfm': True,
                'rfm_range': 5.0,
                'rfm_speed': 0.2
            },
            {
                'enabled': True,
                'start_freq_left': 305.0,  # Left channel specific start frequency
                'end_freq_left': 295.0,    # Left channel specific end frequency
                'start_freq_right': 295.0, # Right channel specific start frequency
                'end_freq_right': 285.0,   # Right channel specific end frequency
                'volume': 0.3,
                'enable_rfm': False
            }
        ],
        'beat_freq': 10.0,  # Fallback if left/right frequencies are not specified
        'is_binaural': True,
        'is_isochronic': False,
        'enable_rfm': True,
        'rfm_range': 5.0,
        'rfm_speed': 0.2,
        'enable_pink_noise': True,
        'pink_noise_volume': 0.05,
        'sample_rate': 44100
    }
    generate_audio_file("enhanced_audio.wav", duration=30.0, audio_settings=settings)
