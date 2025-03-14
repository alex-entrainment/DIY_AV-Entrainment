import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Callable
from .voice import VoiceParams

@dataclass
class NodeInterpolation:
    duration: float
    start_value: float
    end_value: float

class VoiceNode:
    """Represents a configuration node with interpolation capabilities"""
    
    def __init__(self, params: 'VoiceParams'):
        self.params = params
        self.interpolations: Dict[str, NodeInterpolation] = {}
        
    def get_interpolated_value(self, param_name: str, t: float) -> float:
        """Calculate interpolated parameter value at time t"""
        if param_name not in self.interpolations:
            return getattr(self.params, param_name)
            
        interp = self.interpolations[param_name]
        if interp.duration == 0:
            return interp.end_value
            
        progress = min(t / interp.duration, 1.0)
        return interp.start_value + (interp.end_value - interp.start_value) * progress

    def set_interpolation(self, param_name: str, start_value: float, duration: float):
        """Configure interpolation for a parameter"""
        self.interpolations[param_name] = NodeInterpolation(
            duration=duration,
            start_value=start_value,
            end_value=getattr(self.params, param_name)
        )
