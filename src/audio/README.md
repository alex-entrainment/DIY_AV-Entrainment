# Audio Synth Functions

This folder contains a small library of Python synthesis routines used by `sound_creator.py` and the accompanying GUI to build entrainment audio tracks.  Each function creates a block of stereo samples from a dictionary of parameters.  Transition versions (ending in `_transition`) sweep parameters from a starting value to an ending value over the step.

---

## Quick Usage

```
from synth_functions.sound_creator import generate_audio, load_track_from_json

data = load_track_from_json("track.json")
generate_audio(data, "output.wav")
```

Noise generator settings can be stored separately using `.noise` files:

```
from audio import NoiseParams, save_noise_params, load_noise_params

params = NoiseParams()
save_noise_params(params, "my_settings.noise")
loaded = load_noise_params("my_settings.noise")
```

Audio can also be generated step by step by calling any synth function directly:

```
from synth_functions import rhythmic_waveshaping
samples = rhythmic_waveshaping(duration=10.0, sample_rate=44100, carrierFreq=200)
```

---

## Available Voices

The table below lists the currently implemented synth functions and the meaning of their most important parameters.  Frequencies are given in hertz and amplitudes generally range from 0.0 to 1.0.

### rhythmic_waveshaping
A carrier is modulated by an LFO, then passed through a tanh waveshaper.

| Parameter | Default | Effect |
|-----------|---------|-------|
| `amp` | 0.25 | Overall level of the output |
| `carrierFreq` | 200 | Frequency of the carrier tone |
| `modFreq` | 4 | LFO rate used for amplitude modulation |
| `modDepth` | 1.0 | Depth of that modulation |
| `shapeAmount` | 5.0 | Amount of distortion applied |
| `pan` | 0 | Stereo position (-1 left to +1 right) |

### stereo_am_independent
Independent amplitude modulation for left and right channels with slight detuning between them.

| Parameter | Default | Effect |
|-----------|---------|-------|
| `amp` | 0.25 | Base loudness |
| `carrierFreq` | 200 | Central carrier frequency |
| `modFreqL` / `modFreqR` | 4.0 / 4.1 | LFO rates for each channel |
| `modDepthL` / `modDepthR` | 0.8 | AM depth per channel |
| `modPhaseL` / `modPhaseR` | 0 | Phase offsets for the LFOs |
| `stereo_width_hz` | 0.2 | Amount of frequency detune between channels |

### wave_shape_stereo_am
Combines rhythmic waveshaping with stereo AM.

| Parameter | Default | Effect |
|-----------|---------|-------|
| `amp` | 0.15 | Output level |
| `carrierFreq` | 200 | Carrier tone frequency |
| `shapeModFreq` | 4 | Rate of the waveshaping LFO |
| `shapeModDepth` | 0.8 | Depth of waveshaping modulation |
| `shapeAmount` | 0.5 | Distortion amount |
| `stereoModFreqL` / `stereoModFreqR` | 4.1 / 4.0 | AM rate for each channel |
| `stereoModDepthL` / `stereoModDepthR` | 0.8 | Depth of the channel AM |
| `stereoModPhaseL` / `stereoModPhaseR` | 0 / π/2 | Phase of each AM LFO |

### binaural_beat
Creates the classic binaural beat illusion by playing two close frequencies.

| Parameter | Default | Effect |
|-----------|---------|-------|
| `ampL` / `ampR` | 0.5 | Amplitude of each ear |
| `baseFreq` | 200 | Center frequency of the beat |
| `beatFreq` | 4 | Difference between left and right tones |
| `forceMono` | False | If true, forces both ears to the same tone |
| `startPhaseL` / `startPhaseR` | 0 | Initial phase of each tone |
| `ampOscDepthL` / `ampOscDepthR` | 0 | Depth of amplitude modulation |
| `ampOscFreqL` / `ampOscFreqR` | 0 | Frequency of amplitude modulation |
| `freqOscRangeL` / `freqOscRangeR` | 0 | Range of vibrato modulation |
| `freqOscFreqL` / `freqOscFreqR` | 0 | Rate of vibrato modulation |
| `phaseOscFreq` | 0 | Rate of interaural phase modulation |
| `phaseOscRange` | 0 | Amount of phase modulation |

### isochronic_tone
A tone that pulses on and off at a fixed rate using a trapezoid envelope.

| Parameter | Default | Effect |
|-----------|---------|-------|
| `amp` | 0.5 | Output level |
| `baseFreq` | 200 | Frequency of the tone |
| `beatFreq` | 4 | Pulse repetition rate |
| `rampPercent` | 0.2 | Portion of each pulse used for fade in/out |
| `gapPercent` | 0.15 | Fraction of the cycle that is silent |
| `pan` | 0 | Stereo pan location |

### monaural_beat_stereo_amps
Produces a monaural beat while allowing different amplitudes for the lower and upper components in each ear.

| Parameter | Default | Effect |
|-----------|---------|-------|
| `amp_lower_L`/`amp_upper_L` | 0.5 | Levels of the two tones in the left ear |
| `amp_lower_R`/`amp_upper_R` | 0.5 | Levels of the two tones in the right ear |
| `baseFreq` | 200 | Center frequency of the pair |
| `beatFreq` | 4 | Difference between the two tones |
| `startPhaseL`/`startPhaseR` | 0 | Initial phases |
| `phaseOscFreq` | 0 | Phase modulation rate |
| `phaseOscRange` | 0 | Phase modulation depth |
| `ampOscDepth` | 0 | Depth of amplitude modulation |
| `ampOscFreq` | 0 | Rate of amplitude modulation |
| `ampOscPhaseOffset` | 0 | Phase of the amplitude modulation |

### qam_beat
An advanced quadrature amplitude modulation beat generator with several optional enhancements.

Key parameters include channel amplitudes (`ampL`, `ampR`), base frequencies (`baseFreqL`, `baseFreqR`), one or two AM oscillators per channel (`qamAmFreqL`, `qamAmDepthL`, ...), cross-channel coupling (`crossModDepth`, `crossModDelay`), harmonic and sub‑harmonic enhancers, phase oscillation controls, and optional attack/release times.

### hybrid_qam_monaural_beat
Left channel uses QAM style modulation while the right channel runs a monaural beat.

Important parameters: `ampL`, `ampR`, QAM carrier and AM settings for the left side (`qamCarrierFreqL`, `qamAmFreqL`, `qamAmDepthL`, ...), plus monaural beat controls for the right side (`monoCarrierFreqR`, `monoBeatFreqInChannelR`, AM/FM/phase settings, etc.).

### spatial_angle_modulation
Moves a tone along a geometric path using the optional `audio_engine` module.

| Parameter | Default | Effect |
|-----------|---------|-------|
| `amp` | 0.7 | Output level |
| `carrierFreq` | 440 | Base frequency of the tone |
| `beatFreq` | 4 | Rate of position modulation |
| `pathShape` | "circle" | Shape of the path (circle, line, arc...) |
| `pathRadius` | 1.0 | Size of the movement path |
| `arcStartDeg`/`arcEndDeg` | 0 / 360 | Limits for the `arc` path |
| `frame_dur_ms` | 46.4 | Frame size used for processing |
| `overlap_factor` | 8 | Amount of overlap between frames |

### spatial_angle_modulation_monaural_beat
Combines the monaural beat voice with spatial angle modulation.  Parameters mirror those of `monaural_beat_stereo_amps` along with spatial movement controls (`spatialBeatFreq`, `spatialPhaseOffset`, `pathRadius`, etc.).

### subliminal_encode
Encodes audio files as ultrasonic subliminals. Multiple files may be supplied via `audio_paths` and can either be mixed together (`mode="stack"`) or played one after another (`mode="sequence"`). When sequencing, a one second silence is inserted between each subliminal. The chosen arrangement loops for the duration of the step.

| Parameter | Default | Effect |
|-----------|---------|-------|
| `audio_path` / `audio_paths` | – | Path(s) to the audio file(s) to encode |
| `mode` | `"sequence"` | Sequential or stacked playback |
| `carrierFreq` | 17500 | Modulation frequency (15000‑20000 Hz) |
| `amp` | 0.5 | Output level |

---

## Volume Envelopes
Each voice may optionally include a volume envelope definition:

- `linear_fade` – specify `fade_in` and `fade_out` durations in seconds.
- `adsr` – attack, decay, sustain, and release times.
- Leaving the envelope undefined applies a small 10 ms fade to avoid clicks.

## Transition Curves
Transition voices accept a `transition_curve` parameter controlling how values
move from their start settings to the final ones. Supported options are:

- `linear` – constant rate from start to finish (default).
- `logarithmic` – fast start that slows near the end.
- `exponential` – slow start that accelerates toward the end.

Any other string will be evaluated as a Python expression using `alpha` as the
linear ramp from 0 to 1. `numpy` is available via `np` and `math` functions may
be used as well. For example:

```python
transition_curve="np.sin(alpha * np.pi / 2)"
```

If omitted, `linear` is used.

---

## Tips
- Start simple and audition short sections first.
- Use subtle panning and moderate modulation rates.
- Crossfade between steps to avoid abrupt changes.


