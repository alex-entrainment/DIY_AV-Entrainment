import numpy as np
import soundfile as sf

from gnaural.voice import BinauralBeatVoice, IsochronicVoice, AlternatingIsoVoice

def constant_interpolate(time, param):
    if param == 'base_freq':
        return 4
    elif param in ['vol_left', 'vol_right']:
        return 1.0
    elif param == 'carrier_freq':
        return 200
    else:
        return 0

sample_rate = 44100
duration = 5.0

# Create voice instances with test settings
binaural = BinauralBeatVoice(voice_id="binaural", sample_rate=sample_rate, description="Binaural test")
isochronic = IsochronicVoice(voice_id="isochronic", sample_rate=sample_rate, description="Isochronic test")
alternating = AlternatingIsoVoice(voice_id="alternating", sample_rate=sample_rate, description="Alternating isochronic test")

# Override _interpolate_parameter to return constants for test
binaural._interpolate_parameter = constant_interpolate
isochronic._interpolate_parameter = constant_interpolate
alternating._interpolate_parameter = constant_interpolate

# Generate audio segments (each 5 seconds long)
audio_binaural = binaural.generate(duration)
audio_isochronic = isochronic.generate(duration)
audio_alternating = alternating.generate(duration)

# Concatenate segments into a single stereo track (15 seconds total)
audio_track = np.concatenate((audio_binaural, audio_isochronic, audio_alternating), axis=0)

# Write the output as a .wav file
sf.write("test_output.wav", audio_track, sample_rate)
print("test_output.wav generated successfully.")
