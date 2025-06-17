"""Synth functions package for sound generation."""

# This module contains all the synthesizer voice functions
# extracted from sound_creator.py for better organization.

from .rhythmic_waveshaping import rhythmic_waveshaping, rhythmic_waveshaping_transition
from .stereo_am_independent import stereo_am_independent, stereo_am_independent_transition
from .wave_shape_stereo_am import wave_shape_stereo_am, wave_shape_stereo_am_transition
from .binaural_beat import binaural_beat, binaural_beat_transition
from .isochronic_tone import isochronic_tone, isochronic_tone_transition
from .monaural_beat_stereo_amps import monaural_beat_stereo_amps, monaural_beat_stereo_amps_transition
from .qam_beat import qam_beat, qam_beat_transition
from .hybrid_qam_monaural_beat import hybrid_qam_monaural_beat, hybrid_qam_monaural_beat_transition
from .spatial_angle_modulation import (
    spatial_angle_modulation,
    spatial_angle_modulation_transition,
    spatial_angle_modulation_monaural_beat,
    spatial_angle_modulation_monaural_beat_transition
)
from .noise_flanger import (
    generate_swept_notch_pink_sound,
    generate_swept_notch_pink_sound_transition,
)
from .subliminals import subliminal_encode

__all__ = [
    'rhythmic_waveshaping',
    'rhythmic_waveshaping_transition',
    'stereo_am_independent', 
    'stereo_am_independent_transition',
    'wave_shape_stereo_am',
    'wave_shape_stereo_am_transition',
    'binaural_beat',
    'binaural_beat_transition',
    'isochronic_tone',
    'isochronic_tone_transition',
    'monaural_beat_stereo_amps',
    'monaural_beat_stereo_amps_transition',
    'qam_beat',
    'qam_beat_transition',
    'hybrid_qam_monaural_beat',
    'hybrid_qam_monaural_beat_transition',
    'spatial_angle_modulation',
    'spatial_angle_modulation_transition', 
    'spatial_angle_modulation_monaural_beat',
    'spatial_angle_modulation_monaural_beat_transition',
    'generate_swept_notch_pink_sound',
    'generate_swept_notch_pink_sound_transition',
    'subliminal_encode',
]
