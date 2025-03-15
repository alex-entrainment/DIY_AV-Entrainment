from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass
from typing import List
from .node import VoiceNode
from .envelope import apply_envelope
import librosa

@dataclass
class VoiceParams:
    duration: float
    beat_freq: float
    base_freq: float
    vol_left: float
    vol_right: float

class BaseVoice(ABC):
    """Abstract base class for all voice types"""
    
    def __init__(self, voice_id: str, sample_rate: int, description: str = ""):
        self.id = voice_id
        self.description = description
        self.sample_rate = sample_rate
        self.nodes: List[VoiceNode] = []
        self.current_phase = 0.0
        
    def add_node(self, params: VoiceParams):
        """Add a new configuration node"""
        self.nodes.append(VoiceNode(params))
        
    @abstractmethod
    def generate(self, duration: float) -> np.ndarray:
        """Generate stereo audio output for this voice"""
        pass
        
    def _interpolate_parameter(self, time: np.ndarray, param_name: str) -> np.ndarray:
        """Calculate interpolated parameter values across time array"""
        if not self.nodes:
            return np.zeros_like(time)
            
        values = np.zeros_like(time)
        current_time = 0.0
        current_value = getattr(self.nodes[0].params, param_name)
        
        for i, node in enumerate(self.nodes):
            if i == 0:
                # First node - constant value until next node
                next_time = node.params.duration
                mask = (time >= current_time) & (time < next_time)
                values[mask] = current_value
                current_time = next_time
                continue
                
            # Get interpolation configuration
            interp = node.interpolations.get(param_name)
            if interp is None:
                # No interpolation - jump to new value
                mask = (time >= current_time) & (time < current_time + node.params.duration)
                values[mask] = getattr(node.params, param_name)
                current_time += node.params.duration
                continue
                
            # Calculate interpolated values
            interp_end = current_time + interp.duration
            mask = (time >= current_time) & (time < interp_end)
            t_interp = time[mask] - current_time
            values[mask] = interp.start_value + (getattr(node.params, param_name) - interp.start_value) * (t_interp / interp.duration)
            
            # Remaining node duration after interpolation
            remaining_duration = node.params.duration - interp.duration
            if remaining_duration > 0:
                mask = (time >= interp_end) & (time < interp_end + remaining_duration)
                values[mask] = getattr(node.params, param_name)
                current_time = interp_end + remaining_duration
            else:
                current_time = interp_end
                
        return values
    
class BinauralBeatVoice(BaseVoice):
    """Generates binaural beats with configurable carrier and beat frequencies"""
    
    def generate(self, duration: float) -> np.ndarray:
        if not self.nodes:
            return np.zeros((int(duration * self.sample_rate), 2), dtype=np.float32)
            
        # Get interpolated parameters across duration
        time = np.arange(int(duration * self.sample_rate)) / self.sample_rate
        left_freq = self._interpolate_parameter(time, 'base_freq') 
        right_freq = left_freq + self._interpolate_parameter(time, 'beat_freq')
        left_vol = self._interpolate_parameter(time, 'vol_left')
        right_vol = self._interpolate_parameter(time, 'vol_right')
        
        # Generate oscillator signals
        left = np.sin(2 * np.pi * np.cumsum(left_freq) / self.sample_rate + self.current_phase)
        right = np.sin(2 * np.pi * np.cumsum(right_freq) / self.sample_rate + self.current_phase)
        
        # Store phase for seamless continuation
        self.current_phase = (self.current_phase + 2 * np.pi * left_freq[-1] * 
                            (1 / self.sample_rate)) % (2 * np.pi)
                            
        # Apply volumes and envelope
        stereo = np.column_stack((left * left_vol, right * right_vol))
        return apply_envelope(stereo, self.sample_rate)
        
class IsochronicVoice(BaseVoice):
    """Generates isochronic pulse trains with configurable duty cycle"""
    
    def generate(self, duration: float) -> np.ndarray:
        total_samples = int(duration * self.sample_rate)
        time = np.linspace(0, duration, total_samples, endpoint=False)
        freq = self._interpolate_parameter(time, 'base_freq')
        left_vol = self._interpolate_parameter(time, 'vol_left')
        right_vol = self._interpolate_parameter(time, 'vol_right')
        pulse = (np.sin(2 * np.pi * freq * time) > 0).astype(np.float32)
        stereo = np.column_stack((pulse * left_vol, pulse * right_vol))
        return apply_envelope(stereo, self.sample_rate)
        
class AlternatingIsoVoice(BaseVoice):
    """Generates alternating isochronic pulses between channels"""
    
    def generate(self, duration: float) -> np.ndarray:
        total_samples = int(duration * self.sample_rate)
        time = np.linspace(0, duration, total_samples, endpoint=False)
        freq = self._interpolate_parameter(time, 'base_freq')
        left_vol = self._interpolate_parameter(time, 'vol_left')
        right_vol = self._interpolate_parameter(time, 'vol_right')
        sine_wave = np.sin(2 * np.pi * freq * time)
        pulses = (sine_wave > 0).astype(np.int32)
        alt = np.mod(np.floor(time * freq), 2)
        left = pulses * ((alt % 2) == 0) * left_vol
        right = pulses * ((alt % 2) == 1) * right_vol
        stereo = np.column_stack((left, right)).astype(np.float32)
        return apply_envelope(stereo, self.sample_rate)
        
class PinkNoiseVoice(BaseVoice):
    """Generates colored noise with configurable spectral profile"""
    
    def generate(self, duration: float) -> np.ndarray:
        total_samples = int(duration * self.sample_rate)
        white = np.random.normal(0, 1, (total_samples, 2)).astype(np.float32)
        pink = np.cumsum(white, axis=0)
        pink /= np.max(np.abs(pink))
        t = np.linspace(0, duration, total_samples, endpoint=False)
        left_vol = self._interpolate_parameter(t, 'vol_left')
        right_vol = self._interpolate_parameter(t, 'vol_right')
        pink[:, 0] *= left_vol
        pink[:, 1] *= right_vol
        return apply_envelope(pink, self.sample_rate)
class ExternalAudioVoice(BaseVoice):
    """Loads an external audio file and uses it as voice source for generation."""
    
    def __init__(self, voice_id: str, sample_rate: int, file_path: str, description: str = ""):
        super().__init__(voice_id, sample_rate, description)
        self.file_path = file_path
        audio_data, sr = librosa.load(file_path, sr=sample_rate, mono=False)
        if audio_data.ndim == 1:
            audio_data = np.vstack([audio_data, audio_data])
        self.audio_data = audio_data
        self.duration_in_samples = audio_data.shape[1]
        
    def generate(self, duration: float) -> np.ndarray:
        total_samples = int(duration * self.sample_rate)
        repeats = -(-total_samples // self.duration_in_samples)
        audio = np.tile(self.audio_data, repeats)[:, :total_samples]
        stereo = audio.T.astype(np.float32)
        return apply_envelope(stereo, self.sample_rate)
