# Binaural Audio Track Generator

A Python-based tool for building customized brainwave entrainment audio files. Choose from a variety of **synth voices** (AM, noise-based, binaural beats, isochronic pulses, spatial-angle modulation, and more) and assemble them into a sequence of **steps**. Configure everything via a simple JSON file or with the included PyQt5 GUI.

---

## Key Features

- **Multiple Voices**: AM (Amplitude Modulation), FSAM (Filter‑Selective AM), rhythmic waveshaping, independent stereo AM, pure binaural beats, isochronic tones, continuous flanger, and spatial-angle modulation (SAM).
- **Transitions**: Gradually change voice parameters over time (e.g., sweep from one frequency to another).
- **Volume Envelopes**: Prevent clicks with fade‑ins/outs, or shape volume with ADSR or linear fades.
- **JSON-Driven**: Define your track in a human‑readable JSON schema.
- **GUI and Code Modes**: Use the PyQt5 interface (`main.py`) or call functions directly from Python.
- **Extensible**: Add your own voices by following the `basic_am` template in `sound_creator.py`.

---

## Installation

1. **Clone** this repository:
   ```bash
   git clone https://github.com/alex-entrainment/DIY_AV-Entrainment.git
   cd DIY_AV-Entrainment/
   ```
2. **Install dependencies**:
   ```bash
   pip install numpy scipy pyqt5 colorednoise
   ```

---

## Quick Start


1. **GUI mode**  
   ```bash
   python main.py
   ```
   Use **Load JSON**, **Add Step**, **Add Voice**, etc., then **Generate WAV**.

2. **Programmatic mode**  
   ```python
   from sound_creator import load_track_from_json, assemble_track_from_data, generate_wav
   data = load_track_from_json("my_track.json")
   generate_wav(data, "output.wav")
   ```

---

## JSON Schema Overview

```json
{
  "global_settings": {
    "sample_rate": 44100,
    "crossfade_duration": 1.0,
    "output_filename": "my_track"
  },
  "steps": [
    {
      "duration": 120.0,
      "description": "Intro: slow AM",
      "voices": [
        {
          "synth_function_name": "basic_am",
          "is_transition": false,
          "params": { /* see Voices below */ },
          "volume_envelope": {
            "type": "linear_fade",
            "params": { "fade_in": 0.5, "fade_out": 0.5 }
          }
        }
      ]
    }
  ]
}
```

- **`duration`** (seconds): How long this step plays.
- **`description`**: A label to remind you what’s happening.
- **`voices`**: One or more synth voices with parameters.
- **`is_transition`**: If `true`, parameters in `params` will sweep from `start…` to `end…` values.

---

## Voice Reference

Below are the built‑in synth voices and their key parameters. All **frequency** values are in Hz (cycles per second), and **depth**, **pan**, and **amp** range between 0 and 1.

NOTE - NOT ALL DEFAULTS ARE CURRENTLY UP TO DATE*

### 1. `basic_am`
A carrier tone whose amplitude is modulated by a low-frequency oscillator (LFO).

| Parameter       | Default | What it does                                     |
|-----------------|---------|--------------------------------------------------|
| `amp`           | 0.25    | Overall loudness.                                |
| `carrierFreq`   | 200     | Pitch of the main tone. Higher = higher pitch.   |
| `modFreq`       | 4       | Speed of volume wobble (Hz).                     |
| `modDepth`      | 0.8     | How deep the wobble is (0 = none, 1 = full).     |
| `pan`           | 0       | Stereo position (-1 left … 0 center … +1 right). |

**Transition version**: Use `startCarrierFreq`, `endCarrierFreq`, etc., to sweep values over the step.

---

### 2. `fsam_filter_bank`
Filters noise around a center frequency, then applies AM.

| Parameter         | Default | What it does                                                  |
|-------------------|---------|---------------------------------------------------------------|
| `amp`             | 0.15    | Overall loudness.                                             |
| `noiseType`       | 1       | 1=white, 2=pink, 3=brown noise.                               |
| `filterCenterFreq`| 1000    | Center for the bandpass filter.                               |
| `filterRQ`        | 0.5     | Inverse of Q factor: lower = narrower band.                   |
| `modFreq`         | 4       | Speed of volume wobble on filtered band.                      |
| `modDepth`        | 0.8     | Depth of that wobble.                                         |
| `pan`             | 0       | Stereo position.                                              |

---

### 3. `rhythmic_waveshaping`
Applies a distortion (`tanh`) to an AM signal for richer texture.

| Parameter     | Default | What it does                               |
|---------------|---------|--------------------------------------------|
| `amp`         | 0.25    | Overall loudness.                          |
| `carrierFreq` | 200     | Pitch of the main tone.                    |
| `modFreq`     | 4       | Speed of wobble before distortion.         |
| `modDepth`    | 1.0     | How strongly the wobble affects distortion.|
| `shapeAmount` | 5.0     | Degree of distortion (higher = harsher).   |
| `pan`         | 0       | Stereo position.                           |

---

### 4. `wave_shape_stereo_am`
Combines waveshaping + independent left/right AM for stereo movement.

| Parameter         | Default | What it does                                           |
|-------------------|---------|--------------------------------------------------------|
| `amp`             | 0.15    | Overall loudness.                                      |
| `carrierFreq`     | 200     | Base tone pitch.                                       |
| `shapeModFreq`    | 4       | LFO for distortion.                                    |
| `shapeModDepth`   | 0.8     | Distortion modulation depth.                           |
| `shapeAmount`     | 0.5     | Amount of waveshaping.                                 |
| `stereoModFreqL`  | 4.1     | AM rate on left channel.                               |
| `stereoModDepthL` | 0.8     | AM depth on left channel.                              |
| `stereoModPhaseL` | 0       | Phase offset for left LFO.                             |
| `stereoModFreqR`  | 4.0     | AM rate on right channel.                              |
| `stereoModDepthR` | 0.8     | AM depth on right channel.                             |
| `stereoModPhaseR` | π/2     | Phase offset for right LFO (default offset).           |

---

### 5. `stereo_am_independent`
Simple stereo AM: each channel has its own LFO and small pitch detune.

| Parameter          | Default | What it does                                     |
|--------------------|---------|--------------------------------------------------|
| `amp`              | 0.25    | Overall loudness.                                |
| `carrierFreq`      | 200     | Base tone pitch.                                 |
| `modFreqL`         | 4.0     | Left-channel wobble speed.                      |
| `modDepthL`        | 0.8     | Left-channel wobble depth.                      |
| `modFreqR`         | 4.1     | Right-channel wobble speed.                     |
| `modDepthR`        | 0.8     | Right-channel wobble depth.                     |
| `modPhaseL`        | 0       | Phase offset for left LFO.                      |
| `modPhaseR`        | 0       | Phase offset for right LFO.                     |
| `stereoWidthHz`    | 0.2     | Tiny frequency offset between channels.         |

---

### 6. `binaural_beat`
Two pure tones with a slight frequency difference—your brain perceives the beat.

| Parameter     | Default | What it does                             |
|---------------|---------|------------------------------------------|
| `amp`         | 0.5     | Overall loudness.                        |
| `baseFreq`    | 200     | Central tone frequency.                  |
| `beatFreq`    | 4       | Difference between left/right tones.    |

---

### 7. `isochronic_tone`
A pulsed tone that turns on/off rapidly for clear, distinct beats.

| Parameter    | Default | What it does                                      |
|--------------|---------|---------------------------------------------------|
| `amp`        | 0.5     | Overall loudness.                                 |
| `baseFreq`   | 200     | Tone frequency.                                   |
| `beatFreq`   | 4       | Pulse repetition rate.                            |
| `rampPercent`| 0.2     | Portion of each pulse spent fading up/down (0–1). |
| `gapPercent` | 0.15    | Silent portion of each cycle (0–1).               |
| `pan`        | 0       | Stereo position.                                  |

---

### 8. `flanged_voice`
Applies a slow, sweeping delay to create a whoosh or jet‑engine effect.

| Parameter            | Default | What it does                              |
|----------------------|---------|-------------------------------------------|
| `amp`                | 0.7     | Overall loudness.                         |
| `max_delay_ms`       | 15      | Maximum delay time (ms).                  |
| `min_delay_ms`       | 1       | Minimum delay time (ms).                  |
| `rate_hz`            | 0.2     | Speed of sweep (Hz).                      |
| `lfo_start_phase_rad`| 0       | Starting phase of the flanger LFO.        |
| `dry_mix` / `wet_mix`| 0.5     | Balance between original and delayed.     |

---

### 9. `spatial_angle_modulation`
Move a tone around in 2D space following a path (circle, line, Lissajous, etc.).

| Parameter         | Default | What it does                               |
|-------------------|---------|--------------------------------------------|
| `amp`             | 0.7     | Overall loudness.                          |
| `carrierFreq`     | 440     | Tone pitch.                                |
| `beatFreq`        | 4       | Speed of movement along path (Hz).         |
| `pathShape`       | circle  | Trajectory: circle, line, figure_eight…    |
| `pathRadius`      | 1.0     | Size of the path.                          |
| `arcStartDeg`     | 0       | Start angle for `arc` shape.               |
| `arcEndDeg`       | 360     | End angle for `arc` shape.                 |
| `frame_dur_ms`    | 46.4    | Processing frame size for smooth motion.   |
| `overlap_factor`  | 8       | How much frames overlap (higher = smoother).|

---

## Volume Envelopes

- **None (default)**: Tiny 10 ms fade to avoid clicks.
- **`linear_fade`**: Specify `fade_in` / `fade_out` times in seconds.
- **ADSR**: Attack / Decay / Sustain / Release stages for more control.

Use the `volume_envelope` field under each voice to choose and configure envelopes.

---

## Tips for Effective Tracks

- **Start simple**: Use one voice and short steps to test settings.
- **Pan subtly**: Small pan moves (<0.5) create a sense of space without disorientation.
- **Keep modulation rates low** (<10 Hz) for meditative effects; higher rates feel more energetic.
- **Use crossfades** (1 s default) to smooth between steps.
- **Experiment**: Combine voices and transitions to craft unique experiences.

---

Happy entraining!

