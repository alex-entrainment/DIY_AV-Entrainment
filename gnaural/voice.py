from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass
from typing import List
from .node import VoiceNode
from .envelope import apply_envelope

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
        
    def _calculate_parameter_ramp(self, duration: float, start_val: float, end_val: float) -> np.ndarray:
        """Create linear ramp between two values over duration"""
        samples = int(duration * self.sample_rate)
        return np.linspace(start_val, end_val, samples)
    
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
        # Similar structure to binaural but with pulsed amplitude
        pass
        
class AlternatingIsoVoice(BaseVoice):
    """Generates alternating isochronic pulses between channels"""
    
    def generate(self, duration: float) -> np.ndarray:
        # Implementation with alternating pulses
        pass
        
class PinkNoiseVoice(BaseVoice):
    """Generates colored noise with configurable spectral profile"""
    
    def generate(self, duration: float) -> np.ndarray:
        # Pink/brown noise implementation
        pass
