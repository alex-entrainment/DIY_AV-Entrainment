import numpy as np

def apply_envelope(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply smooth envelope to prevent clicks at audio boundaries"""
    num_samples = audio.shape[0]
    
    # Create fade-in/out windows
    fade_time = min(0.01, num_samples / sample_rate)  # Max 10ms fade
    fade_samples = int(fade_time * sample_rate)
    
    if fade_samples > 0:
        # Hanning window for smooth fades
        fade_in = np.hanning(fade_samples * 2)[:fade_samples]
        fade_out = np.hanning(fade_samples * 2)[fade_samples:]
        
        # Apply to both channels
        audio[:fade_samples] *= fade_in[:, np.newaxis]
        audio[-fade_samples:] *= fade_out[:, np.newaxis]
        
    # DC offset removal
    audio -= np.mean(audio, axis=0)
    
    # Soft clipping to prevent harsh distortion
    audio = np.tanh(audio * 0.8) * 1.2
    
    return audio
