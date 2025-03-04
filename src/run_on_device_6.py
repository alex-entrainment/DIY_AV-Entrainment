import argparse
import json
import math
import time
import os
import keyboard
import numpy as np
from collections import deque

import board
import busio
from adafruit_pca9685 import PCA9685

import pygame 
from pydub import AudioSegment 

from sequence_model import (
    Step,
    Oscillator,
    StrobeSet,
    Waveform,
    PatternMode,
    AudioSettings
)

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# Audio Player class to handle precise audio positioning and control
class AudioPlayer:
    """
    A class to handle audio playback with precise positioning and control.
    Uses pygame for playback control and pydub for audio processing.
    """
    def __init__(self, audio_file):
        """
        Initialize the audio player with a specific audio file.
        
        Parameters:
            audio_file (str): Path to the audio file (.wav format)
        """
        self.audio_file = audio_file
        self.playing = False
        self.current_position = 0  # in milliseconds
        
        # Load the audio file
        try:
            pygame.mixer.music.load(audio_file)
            # Also load with pydub for duration info
            self.audio = AudioSegment.from_wav(audio_file)
            self.duration = len(self.audio)  # Duration in ms
            self.loaded = True
        except Exception as e:
            print(f"Error loading audio file: {e}")
            self.loaded = False
    
    def play(self, start_position_ms=0):
        """
        Start playing the audio from the specified position.
        
        Parameters:
            start_position_ms (int): Position to start playing from in milliseconds
        """
        if not self.loaded:
            return False
        
        try:
            # Ensure the position is within bounds
            if start_position_ms >= self.duration:
                print(f"Start position {start_position_ms}ms exceeds audio duration")
                return False
            
            # Set starting position and play
            self.current_position = max(0, start_position_ms)
            pygame.mixer.music.play(start=self.current_position / 1000.0)  # Convert to seconds
            self.playing = True
            return True
        except Exception as e:
            print(f"Error playing audio: {e}")
            return False
    
    def pause(self):
        """Pause audio playback and store current position"""
        if self.playing:
            pygame.mixer.music.pause()
            self.current_position = pygame.mixer.music.get_pos() + self.current_position
            self.playing = False
    
    def resume(self):
        """Resume audio playback from stored position"""
        if not self.playing and self.loaded:
            pygame.mixer.music.play(start=self.current_position / 1000.0)
            self.playing = True
    
    def stop(self):
        """Stop audio playback"""
        pygame.mixer.music.stop()
        self.playing = False
        self.current_position = 0
    
    def is_playing(self):
        """Check if audio is currently playing"""
        return self.playing and pygame.mixer.music.get_busy()

# ------------------------------------------------------------------
# 1) Smooth RFM Implementation
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
# 2) PHASE & BRIGHTNESS PATTERN FUNCTIONS
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
# 3) BASE OSCILLATOR & STROBE HELPER FUNCTIONS
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
# 4) MAIN run_sequence FUNCTION
# ------------------------------------------------------------------


def run_sequence(steps: list[Step], audio_settings: AudioSettings, sequence_filename: str, start_from: float = 0.0):
    """
    Main routine to run the LED sequence with optional audio playback,
    applying advanced phase & brightness patterns on top of the base strobing.
    Now supports precise audio positioning and pause functionality.
    
    Parameters:
        steps (list[Step]): The sequence steps to run
        audio_settings (AudioSettings): Audio configuration
        sequence_filename (str): The filename of the sequence JSON
        start_from (float): Time offset (in seconds) to start the sequence from
    """
    # Shared variables for pause functionality
    pause_event = threading.Event()
    is_paused = False
    sequence_running = True
    total_pause_time = 0.0  # Track cumulative pause duration
    
    # Set up keyboard listener for pause functionality
    def on_key_press(event):
        nonlocal is_paused, sequence_running
        if event.name == 'p' or event.name == 'space':
            is_paused = not is_paused
            if is_paused:
                pause_event.clear()
                # Pause audio if playing
                if audio_player and audio_player.is_playing():
                    audio_player.pause()
                print("\n*** SEQUENCE PAUSED - Press 'p' or space to resume ***")
            else:
                # Resume audio if it was playing
                if audio_player and audio_settings.enabled:
                    audio_player.resume()
                pause_event.set()
                print("\n*** SEQUENCE RESUMED ***")
        elif event.name == 'q':
            sequence_running = False
            pause_event.set()  # In case we're paused, to allow clean exit
            print("\n*** SEQUENCE TERMINATED ***")
    
    # Register keyboard listener
    keyboard.on_press(on_key_press)
    pause_event.set()  # Initially not paused
    
    # Audio playback setup with enhanced player
    audio_player = None
    
    if audio_settings.enabled:
        base, _ = os.path.splitext(sequence_filename)
        audio_filename = base + ".wav"
        if os.path.exists(audio_filename):
            try:
                # Initialize our custom audio player
                audio_player = AudioPlayer(audio_filename)
                if not audio_player.loaded:
                    print(f"Failed to load audio file: {audio_filename}")
                    audio_player = None
                else:
                    print(f"Audio loaded: {audio_filename}")
            except Exception as e:
                print(f"ERROR: Could not initialize audio player for {audio_filename}: {e}")
                audio_player = None
        else:
            print(f"No matching audio file found: {audio_filename}")

    # PCA9685 setup
    i2c = busio.I2C(board.SCL, board.SDA)
    pca = PCA9685(i2c)
    pca.frequency = 1000

    try:
        # Calculate the total sequence duration to determine where to start
        sequence_durations = [step.duration for step in steps]
        cumulative_durations = [sum(sequence_durations[:i+1]) for i in range(len(sequence_durations))]
        
        # Find which step to start from based on start_from parameter
        start_step_index = 0
        time_offset_within_step = start_from
        
        for i, duration in enumerate(cumulative_durations):
            if start_from < duration:
                start_step_index = i
                # If not the first step, calculate offset within this step
                if i > 0:
                    time_offset_within_step = start_from - cumulative_durations[i-1]
                break
        
        # If start_from is beyond sequence end, start from the beginning
        if start_from >= sum(sequence_durations):
            print(f"Start time {start_from}s exceeds sequence duration. Starting from beginning.")
            start_step_index = 0
            time_offset_within_step = 0
        
        # If we're starting from a non-zero point, provide feedback
        if start_from > 0:
            print(f"Starting sequence from step #{start_step_index+1} at {time_offset_within_step:.2f}s into that step")
            
        # Start audio playback if enabled, with proper positioning
        if audio_settings.enabled and audio_player and audio_player.loaded:
            # Convert sequence start position from seconds to milliseconds
            start_position_ms = int(start_from * 1000)
            if audio_player.play(start_position_ms):
                print(f"Audio started at position {start_from:.2f}s")
            else:
                print("Failed to start audio playback")
        
        # Process each step
        for step_index, step in enumerate(steps[start_step_index:], start_step_index):
            if not sequence_running:
                break
                
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

            # Start time for this step
            start_time = time.monotonic()
            prev_time = start_time
            step_is_complete = False
            
            # Apply offset for the first step if starting mid-sequence
            initial_elapsed = 0
            if step_index == start_step_index and time_offset_within_step > 0:
                initial_elapsed = time_offset_within_step
            
            while not step_is_complete and sequence_running:
                # Handle pause if needed
                if is_paused:
                    pause_start = time.monotonic()
                    # Wait until unpaused
                    pause_event.wait()
                    if not sequence_running:
                        break
                    # Calculate how long we were paused
                    pause_duration = time.monotonic() - pause_start
                    total_pause_time += pause_duration
                    # Adjust timing references to account for the pause
                    start_time += pause_duration
                    prev_time = time.monotonic()
                
                current_time = time.monotonic()
                elapsed = current_time - start_time + initial_elapsed  # Add offset if needed
                dt = current_time - prev_time  # Actual time step
                prev_time = current_time
                
                # Check if this step is complete
                if elapsed >= step.duration:
                    step_is_complete = True
                    continue

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
            
            # Clear initial elapsed for subsequent steps
            initial_elapsed = 0

        if sequence_running:
            print("\n=== Sequence complete. ===")
        else:
            print("\n=== Sequence terminated by user. ===")

    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"Error running sequence: {e}")
    finally:
        # Cleanup keyboard listener
        keyboard.unhook_all()
        
        # Turn off all LED channels
        for ch in range(6):
            pca.channels[ch].duty_cycle = 0

        pca.deinit()
        print("All LED channels off, PCA9685 deinitialized.")

        # Stop audio if still playing
        if audio_player:
            audio_player.stop()
            print("Audio stopped.")


# ------------------------------------------------------------------
# 4) MAIN ENTRY POINT
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run a 6-LED strobe sequence with advanced patterns.")
    parser.add_argument("--file", default="my_sequence.json", help="Path to the sequence JSON file")
    parser.add_argument("--start-from", type=float, default=0.0, 
                       help="Start the sequence from this time offset (in seconds)")
    args = parser.parse_args()

    filename = args.file
    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: File '{filename}' is not a valid JSON file.")
        return

    # Build steps
    steps = []
    for s_dict in data["steps"]:
        step_obj = Step.from_dict(s_dict)
        steps.append(step_obj)

    # Build audio settings
    audio_settings_dict = data.get("audio_settings", {})
    audio_settings = AudioSettings.from_dict(audio_settings_dict)

    # Run with the specified start time
    run_sequence(steps, audio_settings, filename, args.start_from)


if __name__ == "__main__":
    main()
