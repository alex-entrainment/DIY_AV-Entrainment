
import argparse
import threading
import json
import math
import time
import os
import numpy as np
from collections import deque
import subprocess  # For calling ffplay

import board
import busio
from adafruit_pca9685 import PCA9685

import wave
import pyaudio
import soundfile as sf
import io

from sequence_model import (
    Step,
    Oscillator,
    StrobeSet,
    Waveform,
    PatternMode,
    AudioSettings
)

# ------------------------------------------------------------------
# Streaming AudioPlayer class supporting WAV, FLAC, and MP3 formats
# ------------------------------------------------------------------

class AudioPlayer:
    """
    An optimized streaming audio player for Raspberry Pi using PyAudio's callback mechanism.
    Supports WAV, FLAC, and MP3 formats.
    
    For FLAC files, instead of loading the file into memory, a subprocess call is made
    to ffplay (with -nodisp and -autoexit) to play the file, which you can synchronize
    with your LED sequence.
    """
    def __init__(self, audio_file, volume=1.0, chunk_size=8192):
        """
        Parameters:
            audio_file (str): Path to the audio file (.wav, .flac, or .mp3)
            volume (float): Volume multiplier (1.0 is normal volume)
            chunk_size (int): Number of frames per buffer (increased for smoother playback)
        """
        self.audio_file = audio_file
        self.volume = volume
        self.chunk_size = chunk_size
        self.playing = False
        self.paused = False
        self.stop_flag = False
        self.lock = threading.Lock()
        self.temp_wav_file = None  # Not used in this version
        self.proc = None  # For ffplay subprocess (FLAC playback)

        # Determine file format from extension
        _, file_ext = os.path.splitext(audio_file.lower())
        self.file_format = file_ext[1:]  # Remove the dot

        try:
            if self.file_format == 'wav':
                self._load_wav()
            elif self.file_format == 'flac':
                self._prepare_flac()  # Use ffprobe/info, not loading full data
            elif self.file_format == 'mp3':
                self._load_mp3()
            else:
                raise ValueError(f"Unsupported audio format: {self.file_format}")

            # Log audio file details for debugging
            print(f"Audio file details: {self.channels} channels, {self.sample_width} bytes/sample, {self.frame_rate} Hz")
            if self.frame_rate > 44100 and self.channels > 1:
                print(f"Warning: High sample rate ({self.frame_rate}Hz) with {self.channels} channels may cause issues on Raspberry Pi")
                
            self.loaded = True
        except Exception as e:
            print(f"Error loading audio file: {e}")
            self.loaded = False
            
        # Initialize PyAudio (only used for wav/mp3 playback)
        self.pyaudio_instance = pyaudio.PyAudio()
        self.stream = None

    def _load_wav(self):
        """Load a WAV file using the wave module"""
        self.wf = wave.open(self.audio_file, 'rb')
        self.channels = self.wf.getnchannels()
        self.sample_width = self.wf.getsampwidth()
        self.frame_rate = self.wf.getframerate()
        self.total_frames = self.wf.getnframes()
        self.duration = (self.total_frames / self.frame_rate) * 1000  # in milliseconds

    def _prepare_flac(self):
        """
        Prepare a FLAC file by reading its info without loading full data.
        We use sf.info() to retrieve parameters and then rely on ffplay for playback.
        """
        info = sf.info(self.audio_file)
        self.channels = info.channels
        self.frame_rate = info.samplerate
        self.total_frames = info.frames
        self.duration = (self.total_frames / self.frame_rate) * 1000  # in milliseconds
        self.sample_width = 2  # Assuming 16-bit audio

    def _load_mp3(self):
        """Load an MP3 file using ffmpeg to convert to an in-memory WAV stream"""
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", self.audio_file,
            "-f", "wav",
            "-"
        ]
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.returncode != 0:
            raise Exception("ffmpeg error: " + process.stderr.decode())
        wav_data = process.stdout
        wav_io = io.BytesIO(wav_data)
        self.wf = wave.open(wav_io, 'rb')
        self.channels = self.wf.getnchannels()
        self.frame_rate = self.wf.getframerate()
        self.sample_width = self.wf.getsampwidth()
        self.total_frames = self.wf.getnframes()
        self.duration = (self.total_frames / self.frame_rate) * 1000  # in milliseconds

    def _callback(self, in_data, frame_count, time_info, status):
        """
        PyAudio callback function that streams audio data from the file.
        This is used for WAV and MP3 playback.
        """
        data = self.wf.readframes(frame_count)
        if len(data) == 0:
            return (data, pyaudio.paComplete)
        # Apply volume adjustment if needed (only for 16-bit audio)
        if self.volume != 1.0 and self.sample_width == 2:
            try:
                audio_data = np.frombuffer(data, dtype=np.int16)
                adjusted = np.clip(audio_data * self.volume, -32768, 32767).astype(np.int16)
                data = adjusted.tobytes()
            except Exception as e:
                print(f"Error adjusting volume in callback: {e}")
        return (data, pyaudio.paContinue)

    def play(self, start_position_ms=0):
        if not self.loaded:
            return False
        try:
            if self.file_format == 'flac':
                # For FLAC files, spawn ffplay to handle playback.
                start_seconds = start_position_ms / 1000.0
                cmd = [
                    "ffplay",
                    "-nodisp",
                    "-autoexit",
                    "-loglevel", "error",
                    "-ss", str(start_seconds),
                    self.audio_file
                ]
                self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                # For WAV and MP3, use PyAudio's callback mechanism.
                if self.stream is not None:
                    self.stop()
                # Set file pointer
                start_frame = int((start_position_ms / 1000.0) * self.frame_rate)
                self.wf.setpos(start_frame)
                self.stop_flag = False
                self.playing = True
                self.paused = False
                self.stream = self.pyaudio_instance.open(
                    format=self.pyaudio_instance.get_format_from_width(self.sample_width),
                    channels=self.channels,
                    rate=self.frame_rate,
                    output=True,
                    frames_per_buffer=self.chunk_size,
                    output_device_index=0,
                    stream_callback=self._callback
                )
                self.stream.start_stream()
            return True
        except Exception as e:
            print(f"Error playing audio: {e}")
            return False

    def stop(self):
        with self.lock:
            self.stop_flag = True
            self.playing = False
        if self.file_format == 'flac':
            if self.proc is not None:
                try:
                    self.proc.terminate()
                    self.proc = None
                except Exception as e:
                    print(f"Error stopping ffplay process: {e}")
        else:
            if self.stream is not None:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except Exception as e:
                    print(f"Error closing stream: {e}")
                self.stream = None
            if self.loaded:
                self.wf.rewind()

    def is_playing(self):
        return self.playing and not self.paused

    def __del__(self):
        """Ensure cleanup when object is deleted"""
        self.stop()
        if hasattr(self, 'pyaudio_instance') and self.pyaudio_instance:
            self.pyaudio_instance.terminate()
        if hasattr(self, 'wf') and self.file_format in ['wav', 'mp3'] and self.wf:
            self.wf.close()
        if self.proc is not None:
            try:
                self.proc.terminate()
            except:
                pass

# ------------------------------------------------------------------
# Smooth RFM Implementation (unchanged)
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
        self.smoothing_factor = 0.1
        self.history = deque(maxlen=history_length)
        self.prev_base_freq = None
        
    def update(self, dt, base_freq):
        effective_range = min(self.rfm_range, 0.25 * base_freq)
        if base_freq < 1.0:
            effective_range = effective_range * base_freq
        step = np.random.normal(0, self.rfm_speed * dt)
        self.target_offset += step
        if abs(self.target_offset) > effective_range:
            overshoot = abs(self.target_offset) - effective_range
            pushback = 0.2 * overshoot * dt
            self.target_offset -= np.sign(self.target_offset) * pushback
        if self.prev_base_freq is not None:
            freq_change_rate = abs(base_freq - self.prev_base_freq) / dt
            if freq_change_rate > 0.05:
                reduction_factor = 1.0 / (1.0 + 5.0 * freq_change_rate)
                self.target_offset *= reduction_factor
        self.current_offset += (self.target_offset - self.current_offset) * self.smoothing_factor
        self.history.append(self.current_offset)
        smoothed_offset = sum(self.history) / len(self.history)
        self.prev_base_freq = base_freq
        return smoothed_offset

# ------------------------------------------------------------------
# Phase & Brightness Pattern Functions (unchanged)
# ------------------------------------------------------------------

def compute_phase_offset(osc: Oscillator, i: int, t: float) -> float:
    mode = osc.phase_pattern
    strength = osc.pattern_strength
    freq = osc.pattern_freq

    if mode == PatternMode.NONE:
        return 0.0
    elif mode == PatternMode.SACRED_GEOMETRY:
        golden_angle = 2.39996323
        swirl = math.sin(2 * math.pi * freq * t + i * golden_angle)
        return strength * swirl
    elif mode == PatternMode.FRACTAL_ARC:
        depth = 3
        val = freq * t + i * 0.25
        for _ in range(depth):
            val = math.sin(val)
        return strength * val
    elif mode == PatternMode.PHI_SPIRAL:
        phi = 1.61803399
        base = phi * freq * t + i * 0.666
        spiral = math.sin(base) + 0.5 * math.cos(2 * base)
        return strength * spiral
    return 0.0

def compute_brightness_mod(osc: Oscillator, i: int, t: float) -> float:
    mode = osc.brightness_pattern
    strength = osc.pattern_strength
    freq = osc.pattern_freq

    if mode == PatternMode.NONE:
        return 1.0
    elif mode == PatternMode.SACRED_GEOMETRY:
        golden_angle = 2.39996323
        swirl = math.sin(2 * math.pi * freq * t + i * golden_angle)
        factor = 1.0 + strength * ((swirl + 1.0) / 2.0)
        return factor
    elif mode == PatternMode.FRACTAL_ARC:
        depth = 3
        val = freq * t + i * 0.35
        for _ in range(depth):
            val = math.sin(val)
        return 1.0 + strength * ((val + 1.0) / 2.0)
    elif mode == PatternMode.PHI_SPIRAL:
        phi = 1.61803399
        base = phi * freq * t + i * 0.918
        swirl = math.sin(base)
        factor = 1.0 + strength * swirl
        return factor if factor > 0.0 else 0.0
    return 1.0

# ------------------------------------------------------------------
# Base Oscillator & Strobe Helper Functions (unchanged)
# ------------------------------------------------------------------

def compute_frequency(osc: Oscillator, t: float, step_duration: float, offset: float) -> float:
    base_freq = osc.start_freq + (osc.end_freq - osc.start_freq) * (t / step_duration)
    return max(0.1, base_freq + offset)

def compute_duty_cycle(osc: Oscillator, t: float, step_duration: float) -> float:
    return osc.start_duty + (osc.end_duty - osc.start_duty) * (t / step_duration)

def compute_oscillator_value_with_phase_accumulation(osc: Oscillator, freq: float, duty: float, 
                                                      current_phase: float, dt: float, 
                                                      phase_offset: float=0.0) -> (float, float):
    if osc.waveform == Waveform.OFF:
        return 0.0, current_phase
    new_phase = current_phase + (2 * math.pi * freq * dt)
    effective_phase = (new_phase + phase_offset) % (2 * math.pi)
    if osc.waveform == Waveform.SINE:
        output_value = (math.sin(effective_phase) + 1.0) / 2.0
    else:
        duty_phase = 2 * math.pi * (duty / 100.0)
        output_value = 1.0 if (effective_phase < duty_phase) else 0.0
    return output_value, new_phase

def compute_strobe_intensity(sset: StrobeSet, t: float, step_duration: float, osc_values: list[float]) -> float:
    base = sset.start_intensity + (sset.end_intensity - sset.start_intensity) * (t / step_duration)
    mixed = sum(w * v for w, v in zip(sset.oscillator_weights, osc_values))
    return (base / 100.0) * mixed

# ------------------------------------------------------------------
# Main run_sequence FUNCTION
# ------------------------------------------------------------------

def run_sequence(steps: list[Step], audio_settings: AudioSettings, sequence_filename: str,
                 start_from: float = 0.0, volume: float = 1.0):
    """
    Main routine to run the LED sequence with audio playback.
    Audio detection is always enabled by default.
    For FLAC files, playback is handled by ffplay via a subprocess.
    """
    audio_player = None
    base, _ = os.path.splitext(sequence_filename)
    audio_formats = [".wav", ".flac", ".mp3"]
    audio_filename = None

    for ext in audio_formats:
        temp_filename = base + ext
        if os.path.exists(temp_filename):
            audio_filename = temp_filename
            print(f"Found audio file: {audio_filename}")
            break

    if audio_filename:
        try:
            # For WAV files, do a preliminary check with wave module
            _, file_ext = os.path.splitext(audio_filename.lower())
            if file_ext == ".wav":
                with wave.open(audio_filename, 'rb') as wf:
                    print(f"Audio file check: {wf.getnchannels()} channels, {wf.getsampwidth()} bytes/sample, {wf.getframerate()} Hz")
            audio_player = AudioPlayer(audio_filename, volume, chunk_size=8192)
            if not audio_player.loaded:
                print(f"Failed to load audio file: {audio_filename}")
                audio_player = None
            else:
                print(f"Audio loaded: {audio_filename}")
        except Exception as e:
            print(f"ERROR: Could not initialize audio player for {audio_filename}: {e}")
            audio_player = None
    else:
        print(f"No matching audio file found for {base} with extensions {audio_formats}")
        print("Continuing sequence without audio playback")

    i2c = busio.I2C(board.SCL, board.SDA)
    pca = PCA9685(i2c)
    pca.frequency = 1000

    try:
        sequence_durations = [step.duration for step in steps]
        cumulative_durations = [sum(sequence_durations[:i+1]) for i in range(len(sequence_durations))]
        start_step_index = 0
        time_offset_within_step = start_from
        for i, duration in enumerate(cumulative_durations):
            if start_from < duration:
                start_step_index = i
                if i > 0:
                    time_offset_within_step = start_from - cumulative_durations[i-1]
                break
        if start_from >= sum(sequence_durations):
            print(f"Start time {start_from}s exceeds sequence duration. Starting from beginning.")
            start_step_index = 0
            time_offset_within_step = 0
        if start_from > 0:
            print(f"Starting sequence from step #{start_step_index+1} at {time_offset_within_step:.2f}s into that step")
        if audio_player and audio_player.loaded:
            start_position_ms = int(start_from * 1000)
            if audio_player.play(start_position_ms):
                print(f"Audio started at position {start_from:.2f}s")
            else:
                print("Failed to start audio playback")

        for step_index, step in enumerate(steps[start_step_index:], start_step_index):
            print(f"\n=== Starting step #{step_index+1}: '{step.description}' ===")
            print(f" Duration = {step.duration}s, {len(step.oscillators)} oscillator(s).")
            rfm_modules = []
            for osc in step.oscillators:
                if osc.enable_rfm and osc.rfm_range > 0.0 and osc.rfm_speed > 0.0:
                    rfm_modules.append(SmoothRFM(osc.rfm_range, osc.rfm_speed))
                else:
                    rfm_modules.append(None)
            phase_accumulators = [0.0] * len(step.oscillators)
            start_time = time.monotonic()
            prev_time = start_time
            step_is_complete = False
            initial_elapsed = time_offset_within_step if (step_index == start_step_index and time_offset_within_step > 0) else 0
            while not step_is_complete:
                current_time = time.monotonic()
                elapsed = current_time - start_time + initial_elapsed
                dt = current_time - prev_time
                prev_time = current_time
                if elapsed >= step.duration:
                    step_is_complete = True
                    continue
                rfm_offsets = [0.0] * len(step.oscillators)
                for i, osc in enumerate(step.oscillators):
                    if rfm_modules[i] is not None:
                        base_freq = osc.start_freq + (osc.end_freq - osc.start_freq) * (elapsed / step.duration)
                        rfm_offsets[i] = rfm_modules[i].update(dt, base_freq)
                osc_values = []
                for i, osc in enumerate(step.oscillators):
                    freq = compute_frequency(osc, elapsed, step.duration, rfm_offsets[i])
                    duty = compute_duty_cycle(osc, elapsed, step.duration)
                    phase_off = compute_phase_offset(osc, i, elapsed)
                    val, phase_accumulators[i] = compute_oscillator_value_with_phase_accumulation(
                        osc, freq, duty, phase_accumulators[i], dt, phase_off)
                    osc_values.append(val)
                for sset in step.strobe_sets:
                    raw_intensity = compute_strobe_intensity(sset, elapsed, step.duration, osc_values)
                    brightness_factor = 0.0
                    for (w, v, osc) in zip(sset.oscillator_weights, osc_values, step.oscillators):
                        if w > 0.001:
                            bmod = compute_brightness_mod(osc, 0, elapsed)
                            brightness_factor += w * bmod
                    if brightness_factor <= 0.0:
                        brightness_factor = 1.0
                    final_intensity = raw_intensity * brightness_factor
                    if final_intensity > 1.0:
                        final_intensity = 1.0
                    duty_cycle = int(0xFFFF * final_intensity)
                    for channel in sset.channels:
                        pca.channels[channel].duty_cycle = duty_cycle
            print(f"Finished step: {step.description}")
            initial_elapsed = 0
        print("\n=== Sequence complete. ===")
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"Error running sequence: {e}")
    finally:
        for ch in range(6):
            pca.channels[ch].duty_cycle = 0
        pca.deinit()
        print("All LED channels off, PCA9685 deinitialized.")
        if audio_player:
            audio_player.stop()
            print("Audio stopped.")

# ------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run a 6-LED strobe sequence with advanced patterns.")
    parser.add_argument("--file", default="my_sequence.json", help="Path to the sequence JSON file")
    parser.add_argument("--start-from", type=float, default=0.0,
                        help="Start the sequence from this time offset (in seconds)")
    parser.add_argument("--volume", type=float, default=1.0,
                        help="Volume multiplier for audio playback (default 1.0, where 1.0 is normal volume)")
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
    steps = [Step.from_dict(s_dict) for s_dict in data["steps"]]
    
    audio_settings_dict = data.get("audio_settings", {})
    audio_settings_dict["enabled"] = True  # Ensure audio is always enabled
    audio_settings = AudioSettings.from_dict(audio_settings_dict)
    run_sequence(steps, audio_settings, filename, args.start_from, args.volume)

if __name__ == "__main__":
    main()

