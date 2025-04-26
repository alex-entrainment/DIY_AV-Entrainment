# Binaural Track Editor & Audio Generator README

## Overview

This project provides a Python-based **Brainwave Entrainment Audio File Generator**, enabling you to design complex audio sequences—combining amplitude modulation, filter-based modulation, waveshaping, isochronic pulses, binaural beats, spatial-angle modulation (SAM), flanging, and more—via a JSON-driven step/voice architecture.

- **`main.py`** implements a Qt5 GUI for creating/editing a JSON “track definition.”
- **`sound_creator.py`** implements a suite of **synth functions** (called “voices”) that generate stereo NumPy arrays from step durations and parameter dictionaries.
- A **JSON schema** describes global settings (sample rate, crossfade, output file) and a list of **steps**, each containing one or more **voices**.

---

## Installation

1. Clone this repository and `cd` into it.
2. Install dependencies:
   ```bash
   pip install numpy scipy colorednoise pyqt5
   ```
3. (Optional) Place `audio_engine.py` alongside `sound_creator.py` to enable Spatial Angle Modulation (SAM) voices .

---

## Usage

1. **GUI mode**  
   ```bash
   python main.py
   ```
   Use **Load JSON**, **Add Step**, **Add Voice**, etc., then **Generate WAV**.

2. **Programmatic mode**  
   ```python
   from sound_creator import load_track_from_json, assemble_track_from_data, write_track_to_wav
   data = load_track_from_json("my_track.json")
   audio = assemble_track_from_data(data, sample_rate=44100, crossfade_duration=1.0)
   write_track_to_wav(audio, "output.wav", sample_rate=44100)
   ```

---

## JSON Track Schema

```jsonc
{
  "global_settings": {
    "sample_rate": 44100,
    "crossfade_duration": 1.0,
    "output_filename": "my_track.wav"
  },
  "steps": [
    {
      "duration": 120.0,
      "voices": [
        {
          "synth_function_name": "basic_am",
          "is_transition": false,
          "params": { /* see below */ },
          "volume_envelope": { /* optional ADSR/linear */ }
        },
        // … more voices …
      ]
    },
    // … more steps …
  ]
}
```

---

## Voices Reference

Below is each **synth function** (“voice”) you can select, along with its **parameters** and how they shape the sound.

---

### 1. `basic_am` / `basic_am_transition`
A classic amplitude modulation (AM) carrier + LFO.

| Parameter         | Range & Default      | Effect                                                                                                                                          |
|-------------------|----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| `amp`             | 0.0–1.0 (0.25)       | Overall output volume .                                                                                                          |
| `carrierFreq`     | >0 Hz (200)          | Base oscillator frequency. Higher values → higher pitched carrier.                                                                              |
| `modFreq`         | >0 Hz (4)            | LFO frequency (modulation rate). E.g. 4 Hz gives a slow tremolo.                                                                                 |
| `modDepth`        | 0.0–1.0 (0.8)        | Modulation depth: 0 = no AM, 1 = full swing. Increasing deepens the amplitude swings.                                                            |
| `pan`             | –1.0…+1.0 (0)        | Stereo position: –1 = full left, +1 = full right.                                                                                                |
| **Transition-only** parameters: `startCarrierFreq`, `endCarrierFreq`, `startModFreq`, `endModFreq`, `startModDepth`, `endModDepth` interpolate these over the step’s duration .

---

### 2. `fsam_filter_bank` / `fsam_filter_bank_transition`
**Frequency-Selective AM**: noise source filtered, then AM’d.

| Parameter             | Default       | Effect                                                                                                 |
|-----------------------|---------------|--------------------------------------------------------------------------------------------------------|
| `amp`                 | 0.15          | Output gain .                                                                             |
| `noiseType`           | 1 = white     | 1→white, 2→pink, 3→brown noise.                                                                            |
| `filterCenterFreq`    | 1000 Hz       | Center frequency for bandpass/reject filters. Higher → modulations emphasize higher-frequency noise.   |
| `filterRQ`            | 0.5           | Q factor inverse (lower RQ → narrower band).                                                            |
| `modFreq`             | 4 Hz          | LFO rate for band signal.                                                                               |
| `modDepth`            | 0.8           | Depth of AM on the filtered band.                                                                       |
| `pan`                 | 0             | Stereo pan.                                                                                              |
| **Transition-only**: `startFilterCenterFreq`→`endFilterCenterFreq`, `startFilterRQ`→`endFilterRQ`, `startModFreq`→`endModFreq`, `startModDepth`→`endModDepth` .

---

### 3. `rhythmic_waveshaping` / `rhythmic_waveshaping_transition`
AM pre-shaped by a **tanh** waveshaper.

| Parameter     | Default | Effect                                                                                 |
|---------------|---------|----------------------------------------------------------------------------------------|
| `amp`         | 0.25    | Overall volume .                                                         |
| `carrierFreq` | 200 Hz  | Carrier pitch.                                                                          |
| `modFreq`     | 4 Hz    | LFO rate modulating amplitude pre-shaper.                                              |
| `modDepth`    | 1.0     | Depth of waveshaping LFO.                                                               |
| `shapeAmount` | 5.0     | Sharpness of tanh shaping: ↑makes more distortion.                                      |
| `pan`         | 0       | Stereo pan.                                                                             |
| **Transition-only**: `start…`/`end…` versions of the above .

---

### 4. `wave_shape_stereo_am` / `wave_shape_stereo_am_transition`
Combines rhythmic waveshaping **and** stereo AM with independent LFOs.

| Parameter         | Default      | Effect                                                                                         |
|-------------------|--------------|------------------------------------------------------------------------------------------------|
| `amp`             | 0.15         | Overall level .                                                                   |
| `carrierFreq`     | 200 Hz       | Base pitch.                                                                                     |
| **Waveshaper**    |              |                                                                                                |
| `shapeModFreq`    | 4 Hz         | LFO rate feeding waveshaper.                                                                    |
| `shapeModDepth`   | 0.8          | Depth of pre-shaper modulation.                                                                 |
| `shapeAmount`     | 0.5          | Tanh shaping intensity.                                                                         |
| **Stereo AM L**   |              |                                                                                                |
| `stereoModFreqL`  | 4.1 Hz       | Left-channel AM rate.                                                                           |
| `stereoModDepthL` | 0.8          | Left-channel AM depth.                                                                          |
| `stereoModPhaseL` | 0 rad        | LFO phase offset for left.                                                                      |
| **Stereo AM R**   |              |                                                                                                |
| `stereoModFreqR`  | 4.0 Hz       | Right-channel AM rate.                                                                          |
| `stereoModDepthR` | 0.8          | Right-channel AM depth.                                                                         |
| `stereoModPhaseR` | π/2 rad      | Phase offset (default quadrature) for right.                                                   |
| **Transition-only**: all of the above have `start…`/`end…` counterparts for linear interpolation .

I recommend keeping the waveshape modulation depth and amount fairly low. Otherwise, the distortion will be too extreme. 

---

### 5. `stereo_am_independent` / `stereo_am_independent_transition`
Independent AM on left/right carriers with **detune** stereo width.

| Parameter            | Default   | Effect                                                                                       |
|----------------------|-----------|----------------------------------------------------------------------------------------------|
| `amp`                | 0.25      | Volume.                                                                                      |
| `carrierFreq`        | 200 Hz    | Center pitch.                                                                                |
| `modFreqL/R`         | 4 Hz /4.1 Hz | Separate AM rates per channel.                                                        |
| `modDepthL/R`        | 0.8       | Depth of each channel’s AM.                                                                  |
| `modPhaseL/R`        | 0 rad     | Phase offsets.                                                                               |
| `stereoWidthHz`      | 0.2 Hz    | Frequency detune between left/right carriers, creating a slow binaural-like width.          |
| **Transition-only**: all above as `start…`/`end…` .

---

### 6. `binaural_beat` / `binaural_beat_transition`
Pure binaural beats by offsetting left/right frequencies.

| Parameter         | Default        | Effect                                                                                             |
|-------------------|----------------|----------------------------------------------------------------------------------------------------|
| `amp`             | 0.5            | Volume .                                                                               |
| `baseFreq`        | 200 Hz         | Center frequency.                                                                                   |
| `beatFreq`        | 4 Hz           | Perceived binaural beat (difference).                                                             |
| **Transition-only**: `startBaseFreq`→`endBaseFreq`, `startBeatFreq`→`endBeatFreq` .

---

### 7. `isochronic_tone` / `isochronic_tone_transition`
Pulsed (isochronic) tone with a trapezoidal envelope.

| Parameter      | Default   | Effect                                                                                              |
|----------------|-----------|-----------------------------------------------------------------------------------------------------|
| `amp`          | 0.5       | Volume .                                                                               |
| `baseFreq`     | 200 Hz    | Carrier tone frequency.                                                                             |
| `beatFreq`     | 4 Hz      | Pulse repetition rate.                                                                              |
| `rampPercent`  | 0.2       | Fraction of each pulse spent ramping up/down.                                                      |
| `gapPercent`   | 0.15      | Silent portion between pulses.                                                                      |
| `pan`          | 0.0       | Stereo position.                                                                                    |
| **Transition-only**: `start…`/`end…` for base/beat frequencies .

---

### 8. `flanged_voice`
Noise-based continuous flanging.

| Parameter    | Default | Effect                                                                                  |
|--------------|---------|-----------------------------------------------------------------------------------------|
| `amp`        | 0.7     | Output gain .                                                             |
| `max_delay_ms` | 15    | Maximum flange delay.                                                                    |
| `min_delay_ms` | 1     | Minimum flange delay.                                                                    |
| `rate_hz`    | 0.2     | Flanger sweep rate.                                                                     |
| `lfo_start_phase_rad` | 0 | Starting phase of the flanger LFO.                                                     |
| `dry_mix`/`wet_mix` | 0.5 | Blend of original vs. delayed signal.                                                  |

*WORK IN PROGRESS, EXPERIMENTAL*

---

### 9. **Spatial Angle Modulation** (`spatial_angle_modulation` / `spatial_angle_modulation_transition`)
Advanced 2D path-based SAM via external `audio_engine.py`.

| Parameter       | Default     | Effect                                                                                         |
|-----------------|-------------|------------------------------------------------------------------------------------------------|
| `amp`           | 0.7         | Volume .                                                                         |
| `carrierFreq`   | 440 Hz      | Tone frequency traveling along a path.                                                          |
| `beatFreq`      | 4 Hz        | Speed of traversal along the path (in beats per second).                                        |
| `pathShape`     | `circle`    | Trajectory: `circle`, `line`, `lissajous`, `figure_eight`, or `arc`.                            |
| `pathRadius`    | 1.0         | Size scale of the path.                                                                         |
| `arcStartDeg`/`arcEndDeg` | 0/360 | For `arc` only: segment of circle to trace.                                                  |
| `frame_dur_ms`  | 46.4 ms     | SAM processing frame length.                                                                    |
| `overlap_factor`| 8           | How frames overlap for smooth motion.                                                           |
| **Transition-only**: each of the above has `start…`/`end…` variants .

_NOTE:_ _This voice requires heavy calculation to produce and will take a very long time and potentially a lot of memory for long-duration steps. It is highly experimental_

---

## Envelopes

Voices support optional **volume envelopes**:

- **None** (default): a 10 ms fade-in/out to prevent clicks.
- **`linear_fade`**: ramp in/out over a fixed duration.
- *(Future: ADSR, Linen, etc.)*

---

## Notes & Tips

- **Parameter ranges** outside sensible bounds are automatically clipped/validated.
- **Transitions** interpolate parameters **linearly** over the step.
- **Crossfade** (global) overlaps adjacent steps to smooth transitions.
- **Panning** is equal-power sine/cosine for a natural stereo image.
- For **SAM voices**, include `audio_engine.py` to enable real spatial-angle behavior; otherwise, placeholders produce silence .
- **KEEP AMPLITUDES LOW WHEN USING MORE THAN 2 VOICES. < 0.1 IS RECOMMENDED. OTHERWISE THERE WILL BE CLIPPING YOU WILL HAVE TO POST-PROCESS OUT**
   - This will be adjusted to a more reasonable state. Normalization process needs adjustment. 

---

Enjoy crafting deeply immersive, customized brainwave entrainment tracks with **unprecedented flexibility**!
