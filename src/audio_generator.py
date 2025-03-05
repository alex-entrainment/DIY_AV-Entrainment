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
    Generate pink noise using an improved Voss-McCartney algorithm with better perceptual qualities.
    Pink noise has power that is inversely proportional to frequency (1/f).

    Parameters:
      total_frames (int): Number of samples to generate.
      noise_level (float): Amplitude of the noise (0.0 to 1.0).

    Returns:
      np.array: Pink noise samples scaled to the specified level.
    """
    # Use fewer generators but with better spectral distribution
    num_octaves = 8
    pink_array = np.zeros(total_frames)
    
    # Use octave-spaced generators for smoother spectrum
    octave_values = np.zeros(num_octaves)
    max_random_index = int(np.ceil(total_frames / 2**num_octaves)) + 1
    
    # Pre-generate all random values for efficiency
    random_arrays = [np.random.randn(max_random_index) for _ in range(num_octaves)]
    
    # Generate samples with improved spectral balance
    for i in range(total_frames):
        # Update octave values at their respective intervals
        for j in range(num_octaves):
            if i % (2**j) == 0:
                idx = i // (2**j)
                if idx < len(random_arrays[j]):
                    octave_values[j] = random_arrays[j][idx]
        
        # Sum all octave generators with proper weighting to ensure 1/f spectrum
        value = 0
        for j in range(num_octaves):
            # Weight lower frequencies higher (1/f spectrum)
            value += octave_values[j] / (2**((j+1)/2))
        
        pink_array[i] = value
    
    # Apply a gentle low-pass filter to smooth high frequencies
    filter_size = 5
    window = np.hanning(filter_size)
    window = window / np.sum(window)
    pink_array = np.convolve(pink_array, window, mode='same')
    
    # Normalize using RMS (root mean square) for more natural amplitude perception
    rms = np.sqrt(np.mean(np.square(pink_array)))
    if rms > 0:
        pink_array = pink_array * noise_level / rms
    else:
        pink_array = np.zeros(total_frames)
    
    return pink_array

def smooth_window(size, fade_in=0.1, fade_out=0.1):
    """
    Create a smoothing window with specified fade in/out.
    Used to create smooth transitions for isochronic tones.
    """
    window = np.ones(size)
    fade_in_size = int(size * fade_in)
    fade_out_size = int(size * fade_out)
    
    if fade_in_size > 0:
        window[:fade_in_size] = np.sin(np.linspace(0, np.pi/2, fade_in_size))**2
    
    if fade_out_size > 0:
        window[-fade_out_size:] = np.sin(np.linspace(np.pi/2, 0, fade_out_size))**2
    
    return window

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


def generate_isochronic_modulation(entrainment_freq, sr, total_frames, pulse_shape='square'):
    """
    Generate an isochronic modulation envelope with distinct pulses.
    
    Parameters:
        entrainment_freq: Array of entrainment frequencies over time
        sr: Sample rate
        total_frames: Total number of frames to generate
        pulse_shape: Shape of pulse ('square', 'soft', 'sine')
        
    Returns:
        Modulation signal with values between 0.0 and 1.0
    """
    # Create time array
    t = np.arange(total_frames) / sr
    
    # Create modulation signal based on pulse shape
    if pulse_shape == 'square':
        # Generate phase progression for the entrainment frequency
        phase = np.zeros(total_frames)
        for i in range(1, total_frames):
            phase[i] = phase[i-1] + 2 * np.pi * entrainment_freq[i-1] / sr
            
        # Create square wave with 50% duty cycle
        # Critical: using > 0 creates a true square wave with 50% duty cycle
        mod_signal = np.where(np.sin(phase) > 0, 1.0, 0.0)
        
    elif pulse_shape == 'sine':
        # Generate phase progression
        phase = np.zeros(total_frames)
        for i in range(1, total_frames):
            phase[i] = phase[i-1] + 2 * np.pi * entrainment_freq[i-1] / sr
            
        # Create sine-based modulation (0.0 to 1.0 range)
        mod_signal = 0.5 + 0.5 * np.sin(phase)
        
    else:  # 'soft' (default)
        mod_signal = np.zeros(total_frames)
        
        # We'll track the current period to create soft pulses
        current_period = 0
        period_samples = sr / np.mean(entrainment_freq)
        pulse_position = 0
        pulse_active = False
        rise_time = 0.1  # 10% rise time
        fall_time = 0.3  # 30% fall time
        
        for i in range(total_frames):
            # Update period based on current frequency
            period_samples = sr / entrainment_freq[i] if entrainment_freq[i] > 0 else sr
            
            # Increment position in the current pulse period
            pulse_position += 1
            
            # If we've completed a period, reset and start a new pulse
            if pulse_position >= period_samples:
                pulse_position = 0
                pulse_active = True
                current_period += 1
                
            # Calculate the shape of the pulse
            if pulse_active:
                # Attack phase
                attack_samples = int(period_samples * rise_time)
                if pulse_position < attack_samples:
                    mod_signal[i] = pulse_position / attack_samples
                    
                # Sustain phase (full amplitude)
                elif pulse_position < period_samples / 2:
                    mod_signal[i] = 1.0
                    
                # Decay phase
                elif pulse_position < period_samples / 2 + int(period_samples * fall_time):
                    decay_position = pulse_position - period_samples / 2
                    decay_samples = int(period_samples * fall_time)
                    mod_signal[i] = 1.0 - (decay_position / decay_samples)
                    
                # Off phase
                else:
                    mod_signal[i] = 0.0
                    
    return mod_signal

def generate_waveform(duration, sr, settings):
    """
    Generate a stereo waveform array based on the settings.
    Supports binaural, isochronic, and monaural modes with multiple carriers.
    Returns a NumPy array of shape (total_frames, 2) with values in [-1, 1].
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
    
    # Process each carrier
    for carrier in carriers:
        if not carrier.get('enabled', True):
            continue
            
        # Get carrier settings
        volume = carrier.get('volume', 1.0)
        tone_mode = carrier.get('tone_mode', '').lower()
        
        # Use global settings if carrier doesn't specify a mode
        is_binaural = (tone_mode == 'binaural' or 
                      (tone_mode == '' and settings.get('is_binaural', False)))
        is_isochronic = (tone_mode == 'isochronic' or 
                        (tone_mode == '' and settings.get('is_isochronic', False)))
        
        # Apply carrier-specific RFM if enabled, otherwise use global RFM
        if carrier.get('enable_rfm', False):
            rfm_range = carrier.get('rfm_range', 0.5)
            rfm_speed = carrier.get('rfm_speed', 0.2)
            rfm_offset = generate_rfm_offset(total_frames, sr, rfm_range, rfm_speed)
        elif global_rfm_enabled:
            rfm_offset = generate_rfm_offset(total_frames, sr, global_rfm_range, global_rfm_speed)
        else:
            rfm_offset = np.zeros(total_frames)
        
        if is_binaural:
            # Get channel-specific frequency settings if available, otherwise use defaults
            start_freq_left = carrier.get('start_freq_left', carrier.get('start_freq', 200.0) + 5.0)
            end_freq_left = carrier.get('end_freq_left', carrier.get('end_freq', 200.0) + 5.0)
            start_freq_right = carrier.get('start_freq_right', carrier.get('start_freq', 200.0) - 5.0)
            end_freq_right = carrier.get('end_freq_right', carrier.get('end_freq', 200.0) - 5.0)
            
            # Create frequency ramps
            freq_left = np.linspace(start_freq_left, end_freq_left, total_frames)
            freq_right = np.linspace(start_freq_right, end_freq_right, total_frames)
            
            # Apply RFM
            freq_left += rfm_offset
            freq_right += rfm_offset
            
            # Generate phase and carrier signals
            phase_left = 2 * np.pi * np.cumsum(freq_left / sr)
            phase_right = 2 * np.pi * np.cumsum(freq_right / sr)
            
            carrier_left = np.sin(phase_left) * volume
            carrier_right = np.sin(phase_right) * volume
            
        elif is_isochronic:
            # For isochronic, handle carrier and entrainment frequencies separately
            start_carrier_freq = carrier.get('start_carrier_freq', carrier.get('start_freq', 200.0))
            end_carrier_freq = carrier.get('end_carrier_freq', carrier.get('end_freq', 200.0))
            
            # Get or calculate entrainment frequency 
            start_entrainment = carrier.get('start_entrainment_freq', 10.0)
            end_entrainment = carrier.get('end_entrainment_freq', 10.0)
            pulse_shape = carrier.get('pulse_shape', 'square').lower()
            
            # Create ramps for carrier and entrainment frequencies
            carrier_freq = np.linspace(start_carrier_freq, end_carrier_freq, total_frames)
            entrainment_freq = np.linspace(start_entrainment, end_entrainment, total_frames)
            
            # Apply RFM to carrier frequency only
            carrier_freq += rfm_offset
            
            # Generate carrier signal
            phase = 2 * np.pi * np.cumsum(carrier_freq / sr)
            carrier_signal = np.sin(phase) * volume
            
            # Generate isochronic modulation envelope with distinct pulses
            mod_signal = generate_isochronic_modulation(entrainment_freq, sr, total_frames, pulse_shape)
            
            # Apply modulation to carrier - same for both channels
            carrier_left = carrier_signal * mod_signal
            carrier_right = carrier_left.copy()
            
        else:  # Monaural or default
            # Get frequency settings
            start_freq = carrier.get('start_freq', 200.0)
            end_freq = carrier.get('end_freq', 200.0)
            
            # Create frequency ramp
            freq = np.linspace(start_freq, end_freq, total_frames)
            
            # Apply RFM
            freq += rfm_offset
            
            # Generate carrier
            phase = 2 * np.pi * np.cumsum(freq / sr)
            carrier_signal = np.sin(phase) * volume
            
            # Same for both channels
            carrier_left = carrier_signal
            carrier_right = carrier_signal.copy()
        
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
    Generate a single .wav file for a sequence of steps with enhanced features.
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
        'mod': {}       # Dictionary to store modulation phase for isochronic
    }
    
    for step in steps:
        # Use the primary oscillator of the step to determine the LED pulse (beat) frequency
        osc = step.oscillators[0]
        duration = step.duration
        total_frames = int(duration * sr)
        t = np.linspace(0, duration, total_frames, endpoint=False)
        
        # Compute the LED pulse frequency as a linear ramp
        led_freq = osc.start_freq + (osc.end_freq - osc.start_freq) * (t / duration)
        
        # Apply offline RFM to the LED frequency if enabled
        if global_enable_rfm:
            dt = 1.0 / sr
            steps_rand = np.random.normal(loc=0.0, scale=global_rfm_speed * dt, size=total_frames)
            rfm_offset = np.cumsum(steps_rand)
            rfm_offset = np.clip(rfm_offset, -global_rfm_range, global_rfm_range)
        else:
            rfm_offset = np.zeros(total_frames)
        
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
            tone_mode = carrier.get('tone_mode', '').lower()
            
            # Use global settings if carrier doesn't specify a mode
            is_binaural_carrier = (tone_mode == 'binaural' or 
                                  (tone_mode == '' and binaural))
            is_isochronic_carrier = (tone_mode == 'isochronic' or 
                                    (tone_mode == '' and isochronic))
            
            # Create a unique key for this carrier in the phase accumulator
            carrier_key = f'carrier_{carrier_idx}'
            if carrier_key not in phase_accumulator['carrier']:
                phase_accumulator['carrier'][carrier_key] = {'left': 0.0, 'right': 0.0}
            if carrier_key not in phase_accumulator['mod']:
                phase_accumulator['mod'][carrier_key] = 0.0
            
            # Apply carrier-specific RFM if enabled
            if carrier.get('enable_rfm', False):
                rfm_range = carrier.get('rfm_range', 0.5)
                rfm_speed = carrier.get('rfm_speed', 0.2)
                rfm_carrier = generate_rfm_offset(total_frames, sr, rfm_range, rfm_speed)
            else:
                rfm_carrier = np.zeros(total_frames)
            
            if is_binaural_carrier:
                # For binaural mode
                start_freq_left = carrier.get('start_freq_left', carrier.get('start_freq', 200.0) + 0.5 * osc.start_freq)
                end_freq_left = carrier.get('end_freq_left', carrier.get('end_freq', 200.0) + 0.5 * osc.end_freq)
                start_freq_right = carrier.get('start_freq_right', carrier.get('start_freq', 200.0) - 0.5 * osc.start_freq)
                end_freq_right = carrier.get('end_freq_right', carrier.get('end_freq', 200.0) - 0.5 * osc.end_freq)
                
                # Create frequency ramps
                freq_left = np.linspace(start_freq_left, end_freq_left, total_frames)
                freq_right = np.linspace(start_freq_right, end_freq_right, total_frames)
                
                # Apply RFM
                freq_left += rfm_carrier
                freq_right += rfm_carrier
                
                # Generate binaural carrier
                carrier_left = np.zeros(total_frames)
                carrier_right = np.zeros(total_frames)
                
                dt = 1.0 / sr
                for i in range(total_frames):
                    # Update phase accumulators
                    phase_accumulator['carrier'][carrier_key]['left'] += 2 * np.pi * freq_left[i] * dt
                    phase_accumulator['carrier'][carrier_key]['right'] += 2 * np.pi * freq_right[i] * dt
                    
                    # Generate sine waves
                    carrier_left[i] = np.sin(phase_accumulator['carrier'][carrier_key]['left']) * volume
                    carrier_right[i] = np.sin(phase_accumulator['carrier'][carrier_key]['right']) * volume
                
            elif is_isochronic_carrier:
                # For isochronic mode, separate carrier and entrainment
                start_carrier_freq = carrier.get('start_carrier_freq', carrier.get('start_freq', 200.0))
                end_carrier_freq = carrier.get('end_carrier_freq', carrier.get('end_freq', 200.0))
                
                # Get or calculate entrainment frequency
                start_entrainment = carrier.get('start_entrainment_freq', osc.start_freq)
                end_entrainment = carrier.get('end_entrainment_freq', osc.end_freq)
                pulse_shape = carrier.get('pulse_shape', 'square').lower()
                
                # Create frequency ramps
                carrier_freq = np.linspace(start_carrier_freq, end_carrier_freq, total_frames)
                entrainment_freq = np.linspace(start_entrainment, end_entrainment, total_frames)
                
                # Apply RFM to carrier only
                carrier_freq += rfm_carrier
                
                # Generate isochronic signals
                carrier_signal = np.zeros(total_frames)
                mod_signal = np.zeros(total_frames)
                
                dt = 1.0 / sr
                
                # Generate carrier
                for i in range(total_frames):
                    # Update carrier phase
                    phase_accumulator['carrier'][carrier_key]['left'] += 2 * np.pi * carrier_freq[i] * dt
                    carrier_signal[i] = np.sin(phase_accumulator['carrier'][carrier_key]['left'])
                
                # For true isochronic tones, we need distinct on/off pulses
                if pulse_shape == 'square':
                    # Generate square wave modulation with 50% duty cycle
                    for i in range(total_frames):
                        # Update modulation phase
                        phase_accumulator['mod'][carrier_key] += 2 * np.pi * entrainment_freq[i] * dt
                        # Create true square wave (critical!)
                        mod_signal[i] = 1.0 if np.sin(phase_accumulator['mod'][carrier_key]) > 0 else 0.0
                        
                elif pulse_shape == 'sine':
                    # Generate sine wave modulation (softer)
                    for i in range(total_frames):
                        phase_accumulator['mod'][carrier_key] += 2 * np.pi * entrainment_freq[i] * dt
                        mod_signal[i] = 0.5 + 0.5 * np.sin(phase_accumulator['mod'][carrier_key])
                        
                else:  # 'soft' pulse
                    # Track pulse state for this carrier
                    if 'pulse_state' not in phase_accumulator:
                        phase_accumulator['pulse_state'] = {}
                    if carrier_key not in phase_accumulator['pulse_state']:
                        phase_accumulator['pulse_state'][carrier_key] = {
                            'is_on': False,
                            'position': 0,
                            'period': sr / entrainment_freq[0]
                        }
                    
                    # Create soft pulses with attack/decay
                    for i in range(total_frames):
                        current_period = sr / entrainment_freq[i] if entrainment_freq[i] > 0 else sr
                        state = phase_accumulator['pulse_state'][carrier_key]
                        
                        # Update position in current period
                        state['position'] += 1
                        
                        # Check if we need to start a new period
                        if state['position'] >= current_period:
                            state['position'] = 0
                            state['period'] = current_period
                            state['is_on'] = True
                        
                        # Calculate pulse shape based on position in period
                        if state['is_on']:
                            attack_time = 0.05  # 5% of period
                            attack_samples = int(current_period * attack_time)
                            
                            if state['position'] < attack_samples:
                                # Attack phase (ramp up)
                                mod_signal[i] = state['position'] / attack_samples
                            elif state['position'] < current_period * 0.5:
                                # Sustain phase at full volume
                                mod_signal[i] = 1.0
                            else:
                                # Turn off after 50% duty cycle
                                mod_signal[i] = 0.0
                                state['is_on'] = False
                        else:
                            # Off phase
                            mod_signal[i] = 0.0
                
                # Apply modulation with volume
                carrier_left = carrier_signal * mod_signal * volume
                carrier_right = carrier_left.copy()
                
            else:  # Monaural mode
                # Get frequency settings
                start_freq = carrier.get('start_freq', 200.0)
                end_freq = carrier.get('end_freq', 200.0)
                
                # Create frequency ramp
                freq = np.linspace(start_freq, end_freq, total_frames)
                
                # Apply RFM
                freq += rfm_carrier
                
                # Generate carrier
                carrier_signal = np.zeros(total_frames)
                
                dt = 1.0 / sr
                for i in range(total_frames):
                    phase_accumulator['carrier'][carrier_key]['left'] += 2 * np.pi * freq[i] * dt
                    carrier_signal[i] = np.sin(phase_accumulator['carrier'][carrier_key]['left']) * volume
                
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
    
    # Stitch all step waveforms together
    waveform = np.concatenate(samples_list, axis=0)
    waveform_int16 = (amplitude * waveform).astype(np.int16)

    with wave.open(audio_filename, 'wb') as wavef:
        wavef.setnchannels(2)
        wavef.setsampwidth(2)
        wavef.setframerate(sr)
        wavef.writeframes(waveform_int16.tobytes())
    
    print(f"Done. Generated stepwise audio file: {audio_filename}")
