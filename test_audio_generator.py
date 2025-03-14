import numpy as np
from gnaural.audio_generator import AudioGenerator
from gnaural.voice import VoiceParams

def test_audio_generator():
    # Initialize generator with CD quality settings
    generator = AudioGenerator(sample_rate=44100, bit_depth=16)
    
    # Test Binaural Beat Voice
    bb_voice = generator.add_voice('binaural', 'bb_test', 'Binaural Beat Demo')
    bb_voice.add_node(VoiceParams(
        duration=2.0,
        beat_freq=10.0,
        base_freq=200.0,
        vol_left=0.8,
        vol_right=0.8
    ))
    bb_voice.add_node(VoiceParams(
        duration=3.0,
        beat_freq=15.0,
        base_freq=220.0,
        vol_left=0.5,
        vol_right=0.9
    ))
    
    # Test Isochronic Voice
    iso_voice = generator.add_voice('isochronic', 'iso_test', 'Isochronic Demo')
    iso_voice.add_node(VoiceParams(
        duration=2.0,
        beat_freq=10.0,
        base_freq=300.0,
        vol_left=1.0,
        vol_right=1.0
    ))
    
    # Test Alternating Isochronic Voice
    alt_voice = generator.add_voice('alternating', 'alt_test', 'Alternating Demo')
    alt_voice.add_node(VoiceParams(
        duration=4.0,
        beat_freq=5.0,
        base_freq=400.0,
        vol_left=0.9,
        vol_right=0.9
    ))
    
    # Test Pink Noise Voice
    noise_voice = generator.add_voice('noise', 'noise_test', 'Noise Demo')
    noise_voice.add_node(VoiceParams(
        duration=5.0,
        beat_freq=0.0,  # Unused for noise
        base_freq=0.0,   # Unused for noise
        vol_left=0.7,
        vol_right=0.3
    ))
    
    # Generate and export test files
    total_duration = 5.0  # seconds
    audio_data = generator.generate_audio(total_duration)
    
    # Export combined mix
    generator.export_audio('combined_test.wav', audio_data)
    
    # Export individual voices for verification
    for voice_id in generator.voices:
        voice = generator.voices[voice_id]
        solo_data = voice.generate(total_duration)
        generator.export_audio(f'{voice_id}_solo.wav', solo_data)

if __name__ == '__main__':
    test_audio_generator()
