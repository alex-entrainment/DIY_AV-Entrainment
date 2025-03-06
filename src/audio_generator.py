
# audio_generator.py
import numpy as np
import wave
import math
import random

def generate_rfm_offset(total_frames, sr, rfm_range, rfm_speed):
    step_std = rfm_speed / sr
    steps = np.random.normal(loc=0.0, scale=step_std, size=total_frames)
    rfm = np.cumsum(steps)
    rfm = np.clip(rfm, -rfm_range, rfm_range)
    return rfm

def generate_pink_noise(total_frames, noise_level=0.1):
    num_generators = 16
    pink_array = np.zeros(total_frames)
    for i in range(num_generators):
        update_interval = 2 ** i
        num_values = int(np.ceil(total_frames / update_interval))
        random_values = np.random.randn(num_values)
        repeated = np.repeat(random_values, update_interval)[:total_frames]
        pink_array += repeated
    pink_array = pink_array / np.sqrt(num_generators)
    pink_array = pink_array / np.max(np.abs(pink_array)) * noise_level
    return pink_array

def add_background_noise(signal, sr, noise_type='pink', noise_level=0.1):
    if noise_type is None or noise_level <= 0.001:
        return signal
    if noise_type == 'pink':
        noise = generate_pink_noise(signal.shape[0], noise_level)
        if len(signal.shape) > 1 and signal.shape[1] == 2:
            noise = np.column_stack((noise, noise))
    else:
        noise = np.random.normal(0.0, noise_level, signal.shape)
    return signal + noise

def generate_audio_file(audio_filename, duration, audio_settings):
    if not audio_settings.get('enabled', True):
        print("Audio is disabled. Not generating audio file.")
        return
    print(f"Generating audio file: {audio_filename} (Duration: {duration:.1f} s)")
    sr = audio_settings.get('sample_rate', 44100)
    amplitude = 32767
    waveform = generate_waveform(duration, sr, audio_settings)
    waveform_int16 = (amplitude * waveform).astype(np.int16)
    with wave.open(audio_filename, 'wb') as wavef:
        wavef.setnchannels(2)
        wavef.setsampwidth(2)
        wavef.setframerate(sr)
        wavef.writeframes(waveform_int16.tobytes())
    print(f"Done. Generated {audio_filename}")

def generate_waveform(duration, sr, settings):
    # For continuous audio generation (not step-wise), not modified.
    total_frames = int(duration * sr)
    t = np.linspace(0, duration, total_frames, endpoint=False)
    left_channel = np.zeros(total_frames)
    right_channel = np.zeros(total_frames)
    carriers = settings.get('carriers', [])
    if not carriers:
        carriers = [{
            'enabled': True,
            'start_freq_left': settings.get('carrier_freq', 200.0),
            'end_freq_left': settings.get('carrier_freq', 200.0),
            'start_freq_right': settings.get('carrier_freq', 200.0),
            'end_freq_right': settings.get('carrier_freq', 200.0),
            'volume': 1.0,
            'enable_rfm': settings.get('enable_rfm', False),
            'rfm_range': settings.get('rfm_range', 0.0),
            'rfm_speed': settings.get('rfm_speed', 0.0),
            'tone_mode': 'Binaural'
        }]
    dt = 1.0/sr
    for carrier in carriers:
        if not carrier.get('enabled', True):
            continue
        volume = carrier.get('volume', 1.0)
        sfl = carrier.get('start_freq_left', 200.0)
        efl = carrier.get('end_freq_left', 200.0)
        sfr = carrier.get('start_freq_right', 200.0)
        efr = carrier.get('end_freq_right', 200.0)
        tone_mode = carrier.get('tone_mode', 'Binaural')
        enable_rfm = carrier.get('enable_rfm', False)
        rfm_range = carrier.get('rfm_range', 0.5)
        rfm_speed = carrier.get('rfm_speed', 0.2)
        total_frames = int(duration * sr)
        freq_left = np.linspace(sfl, efl, total_frames)
        freq_right = np.linspace(sfr, efr, total_frames)
        if enable_rfm:
            rfm_left = generate_rfm_offset(total_frames, sr, rfm_range, rfm_speed)
            rfm_right = generate_rfm_offset(total_frames, sr, rfm_range, rfm_speed)
        else:
            rfm_left = np.zeros(total_frames)
            rfm_right = np.zeros(total_frames)
        effective_left = freq_left + rfm_left
        effective_right = freq_right + rfm_right
        if tone_mode == "Binaural":
            phase_left = 2 * np.pi * np.cumsum(effective_left) * dt
            phase_right = 2 * np.pi * np.cumsum(effective_right) * dt
            carrier_left = np.sin(phase_left) * volume
            carrier_right = np.sin(phase_right) * volume
        elif tone_mode == "Isochronic":
            avg_freq = (effective_left + effective_right) / 2
            phase = 2 * np.pi * np.cumsum(avg_freq) * dt
            carrier_signal = np.sin(phase) * volume
            mod_freq = np.abs(effective_left - effective_right)
            mod_freq[mod_freq < 0.1] = np.mean(mod_freq) if np.mean(mod_freq)>0 else 10.0
            mod_phase = 2 * np.pi * np.cumsum(mod_freq) * dt
            mod_signal = np.where(np.sin(mod_phase) >= 0, 1.0, 0.0)
            carrier_left = carrier_signal * mod_signal
            carrier_right = carrier_left.copy()
        else:  # Monaural
            avg_freq = (effective_left + effective_right) / 2
            phase = 2 * np.pi * np.cumsum(avg_freq) * dt
            carrier_signal = np.sin(phase) * volume
            carrier_left = carrier_signal
            carrier_right = carrier_signal.copy()
        left_channel += carrier_left
        right_channel += carrier_right
    max_amplitude = max(np.max(np.abs(left_channel)), np.max(np.abs(right_channel)))
    if max_amplitude > 1.0:
        left_channel /= max_amplitude
        right_channel /= max_amplitude
    stereo_wave = np.vstack((left_channel, right_channel)).T
    if settings.get('enable_pink_noise', False):
        noise_level = settings.get('pink_noise_volume', 0.1)
        stereo_wave = add_background_noise(stereo_wave, sr, 'pink', noise_level)
    return stereo_wave

def generate_audio_file_for_steps_offline_rfm(steps, audio_filename, global_audio_settings):
    """
    Generate a single .wav file for a sequence of steps using per-step audio settings.
    This implementation uses vectorized NumPy operations for efficiency.
    """
    sr = global_audio_settings.get('sample_rate', 44100)
    amplitude = 32767
    segments = []
    dt = 1.0 / sr
    for step in steps:
        duration = step.duration
        total_frames = int(duration * sr)
        # Use per-step audio settings; if missing, fall back to defaults.
        step_audio = step.audio_settings if step.audio_settings else {}
        default_audio = {
            "enabled": True,
            "enable_pink_noise": global_audio_settings.get("enable_pink_noise", False),
            "pink_noise_volume": global_audio_settings.get("pink_noise_volume", 0.1),
            "sample_rate": sr,
            "carriers": []
        }
        carriers = step_audio.get("carriers", [])
        if not carriers:
            carriers = [{
                "enabled": True,
                "start_freq_left": 200.0,
                "end_freq_left": 200.0,
                "start_freq_right": 200.0,
                "end_freq_right": 200.0,
                "tone_mode": "Binaural",
                "volume": 1.0,
                "enable_rfm": False,
                "rfm_range": 0.5,
                "rfm_speed": 0.2
            }]
        step_audio["carriers"] = carriers
        left_channel = np.zeros(total_frames)
        right_channel = np.zeros(total_frames)
        # Process each carrier in this step.
        for carrier in carriers:
            if not carrier.get("enabled", True):
                continue
            volume = carrier.get("volume", 1.0)
            sfl = carrier.get("start_freq_left", 200.0)
            efl = carrier.get("end_freq_left", 200.0)
            sfr = carrier.get("start_freq_right", 200.0)
            efr = carrier.get("end_freq_right", 200.0)
            tone_mode = carrier.get("tone_mode", "Binaural")
            enable_rfm = carrier.get("enable_rfm", False)
            rfm_range = carrier.get("rfm_range", 0.5)
            rfm_speed = carrier.get("rfm_speed", 0.2)
            freq_left = np.linspace(sfl, efl, total_frames)
            freq_right = np.linspace(sfr, efr, total_frames)
            if enable_rfm:
                rfm_left = generate_rfm_offset(total_frames, sr, rfm_range, rfm_speed)
                rfm_right = generate_rfm_offset(total_frames, sr, rfm_range, rfm_speed)
            else:
                rfm_left = np.zeros(total_frames)
                rfm_right = np.zeros(total_frames)
            effective_left = freq_left + rfm_left
            effective_right = freq_right + rfm_right
            if tone_mode == "Binaural":
                phase_left = 2 * np.pi * np.cumsum(effective_left) * dt
                phase_right = 2 * np.pi * np.cumsum(effective_right) * dt
                carrier_left = np.sin(phase_left) * volume
                carrier_right = np.sin(phase_right) * volume
            elif tone_mode == "Isochronic":
                avg_freq = (effective_left + effective_right) / 2
                phase = 2 * np.pi * np.cumsum(avg_freq) * dt
                carrier_signal = np.sin(phase) * volume
                mod_freq = np.abs(effective_left - effective_right)
                mod_freq[mod_freq < 0.1] = np.mean(mod_freq) if np.mean(mod_freq)>0 else 10.0
                mod_phase = 2 * np.pi * np.cumsum(mod_freq) * dt
                mod_signal = np.where(np.sin(mod_phase) >= 0, 1.0, 0.0)
                carrier_left = carrier_signal * mod_signal
                carrier_right = carrier_left.copy()
            else:  # Monaural
                avg_freq = (effective_left + effective_right) / 2
                phase = 2 * np.pi * np.cumsum(avg_freq) * dt
                carrier_signal = np.sin(phase) * volume
                carrier_left = carrier_signal
                carrier_right = carrier_signal.copy()
            left_channel += carrier_left
            right_channel += carrier_right
        max_amp = max(np.max(np.abs(left_channel)), np.max(np.abs(right_channel)))
        if max_amp > 1.0:
            left_channel /= max_amp
            right_channel /= max_amp
        segment = np.vstack((left_channel, right_channel)).T
        if step_audio.get("enable_pink_noise", default_audio["enable_pink_noise"]):
            noise_level = step_audio.get("pink_noise_volume", default_audio["pink_noise_volume"])
            segment = add_background_noise(segment, sr, 'pink', noise_level)
        segments.append(segment)
    waveform = np.concatenate(segments, axis=0)
    waveform_int16 = (amplitude * waveform).astype(np.int16)
    with wave.open(audio_filename, 'wb') as wavef:
        wavef.setnchannels(2)
        wavef.setsampwidth(2)
        wavef.setframerate(sr)
        wavef.writeframes(waveform_int16.tobytes())
    print(f"Done. Generated stepwise audio file: {audio_filename}")

