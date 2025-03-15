import numpy as np
import soundfile as sf
from pydub import AudioSegment
from scipy.signal import butter, lfilter
from typing import List, Dict
from dataclasses import dataclass
from .voice import BaseVoice, BinauralBeatVoice, IsochronicVoice, PinkNoiseVoice, AlternatingIsoVoice, ExternalAudioVoice
from .node import VoiceNode
from .envelope import apply_envelope

class AudioGenerator:
    """Core audio generation engine that mixes multiple voices"""
    
    def __init__(self, sample_rate: int = 44100, bit_depth: int = 16):
        self.sample_rate = sample_rate
        self.bit_depth = bit_depth
        self.voices: Dict[str, BaseVoice] = {}
        self.master_output = None
        
    def add_voice(self, voice_type: str, voice_id: str, description: str = "") -> BaseVoice:
        """Create and register a new voice of specified type"""
        voice_classes = {
            'binaural': BinauralBeatVoice,
            'isochronic': IsochronicVoice,
            'alternating': AlternatingIsoVoice,
            'noise': PinkNoiseVoice,
            'external': ExternalAudioVoice
        }
        
        if voice_type not in voice_classes:
            raise ValueError(f"Invalid voice type: {voice_type}")
            
        voice = voice_classes[voice_type](voice_id, self.sample_rate, description)
        self.voices[voice_id] = voice
        return voice
        
    def generate_audio(self, duration: float) -> np.ndarray:
        """Generate mixed audio from all voices"""
        total_samples = int(duration * self.sample_rate)
        mix = np.zeros((total_samples, 2), dtype=np.float32)
        
        for voice in self.voices.values():
            voice_data = voice.generate(duration)
            mix += voice_data
            
        # Normalize and prevent clipping
        peak = np.max(np.abs(mix))
        if peak > 1.0:
            mix /= peak
            
        return mix
    
    def export_audio(self, filename: str, audio_data: np.ndarray, format: str = 'wav'):
        """Export audio to file in specified format"""
        if format == 'wav':
            sf.write(filename, audio_data, self.sample_rate, subtype=f'PCM_{self.bit_depth}')
        elif format == 'flac':
            sf.write(filename, audio_data, self.sample_rate, format='FLAC')
        elif format == 'mp3':
            audio = AudioSegment(
                audio_data.tobytes(), 
                frame_rate=self.sample_rate,
                sample_width=audio_data.dtype.itemsize,
                channels=2
            )
            audio.export(filename, format='mp3', bitrate=f'{self.bit_depth}k')
