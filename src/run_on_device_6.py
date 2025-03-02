import argparse
import json
import math
import time
import random
import os
import numpy as np
from collections import deque

import board
import busio
from adafruit_pca9685 import PCA9685

import simpleaudio

from sequence_model import (
    Step,
    Oscillator,
    StrobeSet,
    Waveform,
    PatternMode,
    AudioSettings
)


# ------------------------------------------------------------------
# 0) Smooth RFM Implementation
# ------------------------------------------------------------------

class SmoothRFM:
    """
    Implements a smoother random frequency modulation with:
    - Constrained Gaussian random walk
    - Frequency-proportional modulation
    - Temporal smoothing
    - Protection against extreme frequency dips
    """
    def __init__(self, rfm_range=0.5, rfm_speed=0.2, history_length=5):
        self.target_offset = 0.0
        self.current_offset = 0.0
        self.rfm_range = rfm_range
        self.rfm_speed = rfm_speed
        self.smoothing_factor = 0.1  # How quickly current approaches target
        self.history = deque(maxlen=history_length)
        self.prev_base_freq = None
        
    def update(self, dt, base_freq):
        """
        Update the RFM offset based on time step and current base frequency.
        Returns the new offset to apply.
        """
        # Adjust range proportionally to frequency to prevent extreme dips
        # during low-frequency transitions
        effective_range = min(self.rfm_range, 0.25 * base_freq)
        
        # Only allow tiny offsets at very low frequencies
        if base_freq < 1.0:
            effective_range = effective_range * base_freq
            
        # Generate a new target with Gaussian random step
        step = np.random.normal(0, self.rfm_speed * dt)
        self.target_offset += step
        
        # Apply range constraint with soft boundary
        if abs(self.target_offset) > effective_range:
            # Push back toward range, more strongly the further outside
            overshoot = abs(self.target_offset) - effective_range
            pushback = 0.2 * overshoot * dt  # Strength of boundary
            self.target_offset -= np.sign(self.target_offset) * pushback
        
        # Apply frequency transition protection
        # If base frequency is changing, limit RFM impact
        if self.prev_base_freq is not None:
            freq_change_rate = abs(base_freq - self.prev_base_freq) / dt
            if freq_change_rate > 0.05:  # If frequency is changing significantly
                # Reduce RFM impact during fast frequency changes
                reduction_factor = 1.0 / (1.0 + 5.0 * freq_change_rate)
                self.target_offset *= reduction_factor
        
        # Smoothly approach the target offset
        self.current_offset += (self.target_offset - self.current_offset) * self.smoothing_factor
        
        # Update history and calculate average for additional smoothing
        self.history.append(self.current_offset)
        smoothed_offset = sum(self.history) / len(self.history)
        
        # Update previous frequency for next iteration
        self.prev_base_freq = base_freq
        
        return smoothed_offset


# ------------------------------------------------------------------
# 1) PHASE & BRIGHTNESS PATTERN FUNCTIONS
# ------------------------------------------------------------------

def compute_phase_offset(osc: Oscillator, i: int, t: float) -> float:
    """
    Returns an additional phase offset in RADIANS.
    i: Index of the oscillator (useful for per-oscillator offset).
    t: Current time in seconds.

    Implemented modes:
      NONE
      SACRED_GEOMETRY
      FRACTAL_ARC
      PHI_SPIRAL
    """
    mode = osc.phase_pattern
    strength = osc.pattern_strength
    freq = osc.pattern_freq

    if mode == PatternMode.NONE:
        return 0.0

    elif mode == PatternMode.SACRED_GEOMETRY:
        # Example: offset using the "golden angle" ~2.39996, modulated by time & oscillator index
        golden_angle = 2.39996323
        swirl = math.sin(2 * math.pi * freq * t + i * golden_angle)
        return strength * swirl

    elif mode == PatternMode.FRACTAL_ARC:
        # Repeatedly apply sin() to get a fractal-like wave
        depth = 3
        val = freq * t + i * 0.25
        for _ in range(depth):
            val = math.sin(val)
        return strength * val

    elif mode == PatternMode.PHI_SPIRAL:
        # revolve around a circle at a rate tied to the golden ratio φ ~1.618
        phi = 1.61803399
        base = phi * freq * t + i * 0.666  # offset among oscillators
        spiral = math.sin(base) + 0.5 * math.cos(2 * base)
        return strength * spiral

    # Fallback if unknown mode
    return 0.0


def compute_brightness_mod(osc: Oscillator, i: int, t: float) -> float:
    """
    Returns a brightness multiplier in [0..∞).
    Typically 0..2 or so; 1.0 means no change to brightness.

    Implemented modes:
      NONE
      SACRED_GEOMETRY
      FRACTAL_ARC
      PHI_SPIRAL
    """
    mode = osc.brightness_pattern
    strength = osc.pattern_strength
    freq = osc.pattern_freq

    if mode == PatternMode.NONE:
        return 1.0

    elif mode == PatternMode.SACRED_GEOMETRY:
        golden_angle = 2.39996323
        swirl = math.sin(2 * math.pi * freq * t + i * golden_angle)
        # swirl in [-1..1], shift to [0..1], then scale
        factor = 1.0 + strength * ((swirl + 1.0) / 2.0)
        return factor

    elif mode == PatternMode.FRACTAL_ARC:
        depth = 3
        val = freq * t + i * 0.35
        for _ in range(depth):
            val = math.sin(val)
        # map [-1..1] => [0..1], multiply by strength, offset by 1.0
        return 1.0 + strength * ((val + 1.0) / 2.0)

    elif mode == PatternMode.PHI_SPIRAL:
        phi = 1.61803399
        base = phi * freq * t + i * 0.918
        swirl = math.sin(base)
        factor = 1.0 + strength * swirl
        # clamp to avoid negative brightness
        return factor if factor > 0.0 else 0.0

    # Fallback
    return 1.0


# ------------------------------------------------------------------
# 2) BASE OSCILLATOR & STROBE HELPER FUNCTIONS
# ------------------------------------------------------------------

def compute_frequency(osc: Oscillator, t: float, step_duration: float, offset: float) -> float:
    """
    Return the current frequency (Hz), factoring in:
      - a linear ramp from start_freq -> end_freq
      - random offset (RFM) if applicable
    """
    base_freq = osc.start_freq + (osc.end_freq - osc.start_freq) * (t / step_duration)
    return max(0.1, base_freq + offset)


def compute_duty_cycle(osc: Oscillator, t: float, step_duration: float) -> float:
    """
    Return the current duty cycle (0..100), factoring in a linear ramp
    from start_duty -> end_duty.
    """
    return osc.start_duty + (osc.end_duty - osc.start_duty) * (t / step_duration)


def compute_oscillator_value_with_phase_accumulation(osc: Oscillator, freq: float, duty: float, 
                                                      current_phase: float, dt: float, 
                                                      phase_offset: float=0.0) -> (float, float):
    """
    Evaluate the oscillator's output using proper phase accumulation.
    Returns a tuple of (output_value, new_phase) where:
    - output_value is in [0..1] for SINE or SQUARE. OFF => 0.0
    - new_phase is the updated phase in radians for the next iteration
    """
    if osc.waveform == Waveform.OFF:
        return 0.0, current_phase
    
    # Accumulate phase based on the current frequency and time step
    # This is the key improvement - we accumulate phase instead of calculating from scratch
    new_phase = current_phase + (2 * math.pi * freq * dt)
    
    # Apply phase offset and keep phase in [0, 2π]
    effective_phase = (new_phase + phase_offset) % (2 * math.pi)
    
    if osc.waveform == Waveform.SINE:
        output_value = (math.sin(effective_phase) + 1.0) / 2.0
    else:
        # SQUARE
        duty_phase = 2 * math.pi * (duty / 100.0)
        output_value = 1.0 if (effective_phase < duty_phase) else 0.0
    
    return output_value, new_phase


def compute_strobe_intensity(sset: StrobeSet, t: float, step_duration: float, osc_values: list[float]) -> float:
    """
    Combine oscillator outputs (osc_values) with sset.oscillator_weights,
    then scale by an intensity ramp from start_intensity -> end_intensity (0..100).
    Returns a base intensity in [0..1].
    """
    base = sset.start_intensity + (sset.end_intensity - sset.start_intensity) * (t / step_duration)
    mixed = sum(w * v for w, v in zip(sset.oscillator_weights, osc_values))
    return (base / 100.0) * mixed


# ------------------------------------------------------------------
# 3) MAIN run_sequence FUNCTION
# ------------------------------------------------------------------

def run_sequence(steps: list[Step], audio_settings: AudioSettings, sequence_filename: str):
    """
    Main routine to run the LED sequence with optional audio playback,
    applying advanced phase & brightness patterns on top of the base strobing.
    """
    audio_play_obj = None
    if audio_settings.enabled:
        base, _ = os.path.splitext(sequence_filename)
        audio_filename = base + ".wav"
        if os.path.exists(audio_filename):
            try:
                wave_obj = simpleaudio.WaveObject.from_wave_file(audio_filename)
                audio_play_obj = wave_obj.play()
                print(f"Audio started: {audio_filename}")
            except Exception as e:
                print(f"ERROR: Could not play audio file {audio_filename}: {e}")
        else:
            print(f"No matching audio file found: {audio_filename}")

    # PCA9685 setup
    i2c = busio.I2C(board.SCL, board.SDA)
    pca = PCA9685(i2c)
    pca.frequency = 1000

    try:
        for step_index, step in enumerate(steps):
            print(f"\n=== Starting step #{step_index+1}: '{step.description}' ===")
            print(f" Duration = {step.duration}s, {len(step.oscillators)} oscillator(s).")

            # Create SmoothRFM instances for each oscillator
            rfm_modules = []
            for osc in step.oscillators:
                if osc.enable_rfm and osc.rfm_range > 0.0 and osc.rfm_speed > 0.0:
                    rfm_modules.append(SmoothRFM(osc.rfm_range, osc.rfm_speed))
                else:
                    rfm_modules.append(None)
            
            # Initialize phase accumulators for each oscillator
            phase_accumulators = [0.0] * len(step.oscillators)

            start_time = time.monotonic()
            prev_time = start_time
            while True:
                current_time = time.monotonic()
                elapsed = current_time - start_time
                dt = current_time - prev_time  # Actual time step
                prev_time = current_time
                
                if elapsed >= step.duration:
                    break

                # Sleep a bit, but not as long as before since we need more precise timing
                time.sleep(0.005)  # Shorter sleep for better timing precision

                # 1) Update RFM offsets
                rfm_offsets = [0.0] * len(step.oscillators)
                for i, osc in enumerate(step.oscillators):
                    if rfm_modules[i] is not None:
                        # Calculate base frequency (without RFM)
                        base_freq = osc.start_freq + (osc.end_freq - osc.start_freq) * (elapsed / step.duration)
                        # Get RFM offset from the smooth module
                        rfm_offsets[i] = rfm_modules[i].update(dt, base_freq)

                # 2) Compute each oscillator's value with proper phase accumulation
                osc_values = []
                for i, osc in enumerate(step.oscillators):
                    freq = compute_frequency(osc, elapsed, step.duration, rfm_offsets[i])
                    duty = compute_duty_cycle(osc, elapsed, step.duration)
                    phase_off = compute_phase_offset(osc, i, elapsed)
                    
                    # Use the new function that properly accumulates phase
                    val, phase_accumulators[i] = compute_oscillator_value_with_phase_accumulation(
                        osc, freq, duty, phase_accumulators[i], dt, phase_off)
                    
                    osc_values.append(val)

                # 3) For each strobe set, combine oscillator outputs & apply brightness patterns
                for sset in step.strobe_sets:
                    # base intensity in [0..1]
                    raw_intensity = compute_strobe_intensity(sset, elapsed, step.duration, osc_values)

                    # Weighted sum of brightness mods from oscillators
                    brightness_factor = 0.0
                    for (w, v, osc) in zip(sset.oscillator_weights, osc_values, step.oscillators):
                        if w > 0.001:
                            # We can pass the same index i used in the phase offset,
                            # or just pass 0 if we don't care which oscillator is which.
                            bmod = compute_brightness_mod(osc, 0, elapsed)
                            brightness_factor += w * bmod

                    if brightness_factor <= 0.0:
                        brightness_factor = 1.0

                    final_intensity = raw_intensity * brightness_factor
                    # clamp if > 1.0
                    if final_intensity > 1.0:
                        final_intensity = 1.0

                    duty_cycle = int(0xFFFF * final_intensity)
                    for channel in sset.channels:
                        pca.channels[channel].duty_cycle = duty_cycle

            print(f"Finished step: {step.description}")

        print("\n=== Sequence complete. ===")

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        # Turn off all LED channels
        for ch in range(6):
            pca.channels[ch].duty_cycle = 0

        pca.deinit()
        print("All LED channels off, PCA9685 deinitialized.")

        # Stop audio if still playing
        if audio_play_obj and audio_play_obj.is_playing():
            audio_play_obj.stop()
            print("Audio stopped.")


# ------------------------------------------------------------------
# 4) MAIN ENTRY POINT
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run a 6-LED strobe sequence with advanced patterns.")
    parser.add_argument("--file", default="my_sequence.json", help="Path to the sequence JSON file")
    args = parser.parse_args()

    filename = args.file
    with open(filename, "r") as f:
        data = json.load(f)

    # Build steps
    steps = []
    for s_dict in data["steps"]:
        step_obj = Step.from_dict(s_dict)
        steps.append(step_obj)

    # Build audio settings
    audio_settings_dict = data.get("audio_settings", {})
    audio_settings = AudioSettings.from_dict(audio_settings_dict)

    # Run
    run_sequence(steps, audio_settings, filename)


if __name__ == "__main__":
    main()
