# controller.py
import serial
import time
import threading
import os
import io
import wave
import subprocess
import numpy as np
import pyaudio      # Requires PortAudio
import soundfile as sf # Requires libsndfile
from pathlib import Path
import configparser # For reading config
import sys          # For exiting on error
import platform     # For OS-specific checks
import re           # For parsing manual input

# --- Configuration Loading ---
config = configparser.ConfigParser()
# Assume config.ini is in the same directory as this script
script_dir = Path(__file__).parent
config_file = script_dir / 'config.ini'

DEFAULT_AUDIO_VOLUME = 0.7 # Fallback default volume
SERIAL_PORT = None # Will be read from config

if not config_file.is_file():
    print(f"ERROR: Configuration file '{config_file}' not found.")
    print(f"Please run setup.py in the script directory ({script_dir}) first.")
    sys.exit(1) # Exit if config is missing

try:
    config.read(config_file)
    SERIAL_PORT = config.get('Controller', 'serial_port', fallback=None) # Use get with fallback
    try:
         volume_str = config.get('Controller', 'default_volume', fallback=str(DEFAULT_AUDIO_VOLUME))
         DEFAULT_AUDIO_VOLUME = max(0.0, min(1.0, float(volume_str))) # Read and clamp volume 0.0-1.0
    except ValueError:
         print(f"Warning: Invalid volume value in config.ini. Using default: {DEFAULT_AUDIO_VOLUME}")
         # Keep the fallback value if conversion fails

    if not SERIAL_PORT:
         raise ValueError("serial_port key missing or empty in [Controller] section.")

except Exception as e:
    print(f"ERROR reading configuration '{config_file}': {e}")
    print("Please run setup.py again.")
    sys.exit(1)

# --- Other Constants ---
BAUD_RATE = 115200
READ_TIMEOUT = 1 # Seconds for reading responses

# --- Audio Player Class Definition ---
class AudioPlayer:
    """
    An optimized streaming audio player using PyAudio's callback mechanism or ffplay.
    Supports WAV, FLAC, and MP3 formats. Includes volume control.
    """
    # Class variable to hold the single PyAudio instance (initialized on first use)
    _pyaudio_instance = None

    @classmethod
    def _get_pyaudio_instance(cls):
        """Initializes PyAudio instance if not already done."""
        if cls._pyaudio_instance is None:
            print("Initializing PyAudio...")
            try:
                cls._pyaudio_instance = pyaudio.PyAudio()
                print("PyAudio initialized.")
            except Exception as pa_error:
                print(f"ERROR initializing PyAudio: {pa_error}")
                print("Hint: Ensure PortAudio library/development headers are installed.")
                return None # Return None on failure
        return cls._pyaudio_instance

    @classmethod
    def terminate_pyaudio(cls):
        """Terminates the shared PyAudio instance if it exists."""
        if cls._pyaudio_instance:
            print("Terminating shared PyAudio instance.")
            try:
                 cls._pyaudio_instance.terminate()
            except Exception as e:
                 print(f"Warning: Error during PyAudio termination (may already be closed): {e}")
            finally:
                 cls._pyaudio_instance = None

    def __init__(self, audio_file, volume=1.0, chunk_size=8192):
        """
        Parameters:
            audio_file (str or Path): Path to the audio file (.wav, .flac, or .mp3)
            volume (float): Volume multiplier (0.0 to 1.0)
            chunk_size (int): Number of frames per buffer
        """
        self.audio_file_path = Path(audio_file) # Store as Path object internally
        self.volume = max(0.0, min(1.0, volume)) # Clamp volume 0.0 - 1.0
        self.chunk_size = chunk_size
        self.playing = False
        self.paused = False # Note: Pause/Resume not implemented
        self.stop_flag = False
        self.lock = threading.Lock() # For thread safety accessing shared attributes
        self.proc = None  # For ffplay subprocess
        self.wf = None    # For wave file object (WAV or decoded MP3)
        self.pyaudio_instance = None # Instance reference, obtained from class method
        self.stream = None # PyAudio stream
        self.loaded = False
        self.duration = 0 # Duration in milliseconds
        self.channels = 0
        self.sample_width = 0
        self.frame_rate = 0
        self.file_format = ''
        self._mp3_wav_data_io = None # Store in-memory BytesIO for MP3 data

        # --- Load Audio File ---
        try:
            if not self.audio_file_path.is_file():
                raise FileNotFoundError(f"Audio file not found: {self.audio_file_path}")

            self.file_format = self.audio_file_path.suffix[1:].lower()
            print(f"Attempting to load audio format: {self.file_format}")

            if self.file_format == 'wav':
                self._load_wav()
            elif self.file_format == 'flac':
                self._prepare_flac()
            elif self.file_format == 'mp3':
                self._load_mp3()
            else:
                raise ValueError(f"Unsupported audio format: {self.file_format}")

            # Get PyAudio instance only if needed (WAV/MP3) AFTER loading file info
            if self.file_format in ['wav', 'mp3']:
                 self.pyaudio_instance = self._get_pyaudio_instance()
                 if not self.pyaudio_instance:
                      # If PyAudio failed, we can't play WAV/MP3
                      raise Exception("PyAudio initialization failed. Cannot play WAV/MP3.")

            print(f"Audio loaded: {self.audio_file_path.name}")
            print(f"  Format: {self.file_format}, Ch: {self.channels}, Rate: {self.frame_rate} Hz, Width: {self.sample_width} bytes")
            print(f"  Duration: {self.duration / 1000.0:.2f} s, Volume: {self.volume:.2f}")

            if self.frame_rate > 48000: # Example warning threshold
                print(f"  Warning: High sample rate ({self.frame_rate}Hz) may require more processing power.")

            self.loaded = True

        except Exception as e:
            print(f"Error loading audio file '{self.audio_file_path}': {e}")
            # Add hints for common external dependency issues
            if 'ffmpeg' in str(e) or 'ffplay' in str(e):
                 print("  Hint: Ensure ffmpeg/ffplay is installed and in your system's PATH.")
            if 'soundfile' in str(e) or 'libsndfile' in str(e):
                 print("  Hint: Ensure libsndfile library is installed for soundfile operation.")
            self.loaded = False # Ensure loaded is false on any error

    def _load_wav(self):
        """Load WAV file info using wave module."""
        self.wf = wave.open(str(self.audio_file_path), 'rb')
        self.channels = self.wf.getnchannels()
        self.sample_width = self.wf.getsampwidth()
        self.frame_rate = self.wf.getframerate()
        self.total_frames = self.wf.getnframes()
        self.duration = (self.total_frames / self.frame_rate) * 1000 if self.frame_rate > 0 else 0

    def _prepare_flac(self):
        """Get FLAC file info using soundfile."""
        try:
            info = sf.info(str(self.audio_file_path))
            self.channels = info.channels
            self.frame_rate = info.samplerate
            self.total_frames = info.frames
            self.duration = info.duration * 1000 # soundfile duration is in seconds
            # Estimate sample width based on subtype (primarily for info, ffplay handles decoding)
            subtype_major = info.subtype.split('_')[0] if info.subtype else ''
            if subtype_major in ['PCM_16', '']: self.sample_width = 2
            elif subtype_major == 'PCM_24': self.sample_width = 3
            elif subtype_major in ['PCM_32', 'FLOAT', 'DOUBLE']: self.sample_width = 4
            else: self.sample_width = 2; print(f"Warning: Unusual FLAC subtype '{info.subtype}'. Assuming 16-bit for info.")
        except Exception as e:
            # Raise exception if soundfile fails (e.g., libsndfile not found)
            raise Exception(f"Error getting FLAC info with soundfile: {e}. Hint: libsndfile needed.")

    def _load_mp3(self):
        """Convert MP3 to WAV in memory using ffmpeg."""
        print("Converting MP3 to WAV in memory using ffmpeg...")
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
               "-i", str(self.audio_file_path), # Input file
               "-f", "wav",  # Output format WAV
               "-"           # Output to stdout pipe
              ]
        try:
            # check=True raises CalledProcessError on non-zero exit code from ffmpeg
            process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except FileNotFoundError:
             # Provide specific error if ffmpeg command itself isn't found
             raise FileNotFoundError("ffmpeg command not found. Ensure ffmpeg is installed and in your system PATH.")
        except subprocess.CalledProcessError as e:
             # If ffmpeg runs but fails, show its error output
             raise Exception(f"ffmpeg error during MP3 conversion: {e.stderr.decode(errors='ignore')}")
        except Exception as e:
             # Catch other potential errors like permissions
             raise Exception(f"Unexpected error running ffmpeg: {e}")

        wav_data = process.stdout
        if not wav_data or not wav_data.startswith(b'RIFF'): # Basic check for WAV header
             raise ValueError(f"ffmpeg output did not produce valid WAV data. stderr: {process.stderr.decode(errors='ignore')}")

        self._mp3_wav_data_io = io.BytesIO(wav_data) # Store in-memory data
        self.wf = wave.open(self._mp3_wav_data_io, 'rb') # Open wave reader on BytesIO
        # Extract WAV properties from the decoded data
        self.channels = self.wf.getnchannels()
        self.frame_rate = self.wf.getframerate()
        self.sample_width = self.wf.getsampwidth()
        self.total_frames = self.wf.getnframes()
        self.duration = (self.total_frames / self.frame_rate) * 1000 if self.frame_rate > 0 else 0
        print("MP3 conversion complete.")

    def _callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback function for WAV/MP3 streaming."""
        with self.lock: # Ensure thread safety for accessing self.wf etc.
            if self.stop_flag or self.wf is None:
                return (None, pyaudio.paComplete) # Signal stream completion

            try:
                # Read audio frames
                data = self.wf.readframes(frame_count)
                if not data: # End of file reached
                    return (None, pyaudio.paComplete)
            except wave.Error as wave_err:
                 print(f"Wave Error reading frames in callback: {wave_err}")
                 return (None, pyaudio.paAbort) # Signal error
            except Exception as e:
                 # Catch other potential errors (e.g., reading from closed BytesIO)
                 print(f"General Error reading frames in callback: {e}")
                 return (None, pyaudio.paAbort) # Signal error

            # Apply volume adjustment (only for 16-bit PCM currently)
            if self.volume < 0.999 and self.sample_width == 2 and data:
                try:
                    num_samples = len(data) // (self.sample_width * self.channels) * self.channels
                    if num_samples > 0:
                        # Process valid audio data
                        audio_data = np.frombuffer(data, dtype=np.int16, count=num_samples)
                        # Use float32 for calculation to prevent intermediate overflow
                        adjusted_float = audio_data.astype(np.float32, copy=False) * self.volume
                        # Clip back to int16 range and convert type
                        adjusted = np.clip(adjusted_float, -32768, 32767).astype(np.int16)
                        data = adjusted.tobytes()
                    # else: data was too short, skip volume adjustment
                except Exception as e:
                    # Don't abort stream on volume error, just print warning
                    print(f"Warning: Error adjusting volume in callback: {e}")

            # Return audio data and signal to continue stream
            return (data, pyaudio.paContinue)

    def play(self, start_position_ms=0):
        """Starts audio playback from the specified position in milliseconds."""
        if not self.loaded: print("Cannot play: Audio not loaded."); return False
        with self.lock: # Ensure play/stop are atomic
            if self.playing: print("Already playing."); return False
            self.stop_flag = False; self.paused = False

            try:
                # --- FLAC Playback using ffplay ---
                if self.file_format == 'flac':
                    print("Starting FLAC playback with ffplay...")
                    start_seconds = max(0, start_position_ms / 1000.0)
                    ffplay_path = "ffplay" # Assume ffplay is in system PATH
                    try: # Attempt to find full path for robustness
                         where_cmd = ['where', 'ffplay'] if platform.system() == 'Windows' else ['which', 'ffplay']
                         ffplay_path_out = subprocess.check_output(where_cmd, stderr=subprocess.PIPE).decode().strip().splitlines()
                         if ffplay_path_out: ffplay_path = ffplay_path_out[0]
                    except Exception: pass # Ignore if check fails, just use 'ffplay'

                    # Map volume [0.0 - 1.0] to ffplay -volume [0 - ~128].
                    ffplay_volume = int(max(0.0, min(1.0, self.volume)) * 128 + 0.5)

                    cmd = [ ffplay_path,
                            "-nodisp",       # No graphical window
                            "-autoexit",     # Exit when playback finishes
                            "-loglevel", "error", # Only show errors
                            "-ss", str(start_seconds), # Start position
                            "-volume", str(ffplay_volume), # Volume argument
                            str(self.audio_file_path) ] # Input file path

                    print(f"  (ffplay command args: -ss {start_seconds:.2f} -volume {ffplay_volume})")
                    # Run ffplay. Use DEVNULL only if output is unwanted, otherwise useful for debugging.
                    self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    self.playing = True
                    print(f"ffplay process started (PID: {self.proc.pid}).")
                    return True # Success

                # --- WAV/MP3 Playback using PyAudio ---
                else:
                    if not self.pyaudio_instance: print("Error: PyAudio not available."); return False
                    # Close previous stream if somehow still open
                    if self.stream: self._safe_close_stream()

                    print("Starting WAV/MP3 playback with PyAudio callback...")
                    start_frame = max(0, int((start_position_ms / 1000.0) * self.frame_rate))
                    if start_frame >= self.total_frames: print("Start position beyond end."); return False

                    try: # Seek to start frame (handle MP3 BytesIO reset)
                         if self.file_format == 'mp3' and hasattr(self, '_mp3_wav_data_io'):
                             # Re-open wave reader on the BytesIO object
                             if not getattr(self.wf, 'closed', True): self.wf.close() # Close previous reader
                             self._mp3_wav_data_io.seek(0) # Reset BytesIO position
                             self.wf = wave.open(self._mp3_wav_data_io, 'rb') # Re-open
                         # For both WAV and MP3 (after potential reopen)
                         self.wf.setpos(start_frame)
                    except Exception as e: print(f"Error seeking to start frame {start_frame}: {e}"); return False

                    # Open PyAudio stream using callback
                    self.stream = self.pyaudio_instance.open(
                        format=self.pyaudio_instance.get_format_from_width(self.sample_width),
                        channels=self.channels, rate=self.frame_rate, output=True,
                        frames_per_buffer=self.chunk_size, output_device_index=None, # Use default output
                        stream_callback=self._callback, start=False ) # Start manually

                    self.stream.start_stream() # Begin callback execution
                    self.playing = True
                    print("PyAudio stream started.")
                    return True # Success

            except FileNotFoundError as e:
                 # Catch if ffplay/ffmpeg executable is not found
                 print(f"ERROR starting playback: Command not found ({e}). Check system PATH.")
                 return False
            except Exception as e:
                 # Catch other errors during stream/process start
                 print(f"Error initiating audio playback: {e}")
                 self.playing = False # Ensure state is correct on error
                 self._safe_close_stream(); self._safe_terminate_proc() # Attempt cleanup
                 return False

    def _safe_close_stream(self):
        """Safely stop and close the PyAudio stream if it exists."""
        if self.stream:
            try:
                 # Check if stream object is still valid and active before operations
                 # Note: is_active() might raise exception if stream is already invalid
                 try:
                      is_active = self.stream.is_active()
                 except:
                      is_active = False # Assume inactive/invalid on error

                 if is_active:
                      self.stream.stop_stream()
                 self.stream.close()
            except Exception as e:
                 print(f"Warning: Error closing PyAudio stream (may be already closed): {e}")
            finally:
                 self.stream = None # Ensure handle is cleared

    def _safe_terminate_proc(self):
        """Safely terminate the ffplay process if it exists."""
        if self.proc:
            try:
                 if self.proc.poll() is None: # Check if running
                      print(f"Terminating ffplay process (PID: {self.proc.pid})...")
                      self.proc.terminate()
                      self.proc.wait(timeout=0.2) # Brief wait for graceful exit
            except subprocess.TimeoutExpired:
                 print(f"ffplay (PID: {self.proc.pid}) did not terminate quickly, sending kill.")
                 self.proc.kill() # Force kill if necessary
            except Exception as e:
                 print(f"Warning: Error stopping ffplay process: {e}")
            finally:
                 self.proc = None # Ensure handle is cleared

    def stop(self):
        """Stops audio playback."""
        with self.lock: # Prevent race conditions with play/is_playing
            if not self.playing and not self.paused: return # Already stopped

            print("Stopping audio playback...")
            self.stop_flag = True # Signal callback for PyAudio stream
            self.playing = False; self.paused = False # Update state immediately

            self._safe_terminate_proc() # Stop ffplay process if running
            self._safe_close_stream() # Stop PyAudio stream if running

            # Rewind source file/stream for potential next playback
            if self.wf is not None and not getattr(self.wf, 'closed', True):
                try:
                    if hasattr(self, '_mp3_wav_data_io') and self._mp3_wav_data_io: # MP3
                         self._mp3_wav_data_io.seek(0) # Rewind BytesIO
                         # Re-open wave reader on BytesIO as stream closure might affect it
                         self.wf.close()
                         self.wf = wave.open(self._mp3_wav_data_io, 'rb')
                    elif hasattr(self.wf, 'rewind'): # WAV
                         self.wf.rewind()
                except Exception as e: print(f"Warning: Error rewinding audio source: {e}")
            print("Audio stop process completed.")

    def is_playing(self):
        """Checks if audio is actively playing."""
        with self.lock:
            if not self.playing: return False

            current_status = False
            if self.stream: # Check PyAudio stream status
                try: current_status = self.stream.is_active()
                except Exception: current_status = False; self.stream = None # Clear potentially invalid stream
            elif self.proc: # Check ffplay process status
                if self.proc.poll() is None: current_status = True # Still running
                else: self.proc = None # Process ended

            # Update internal state ONLY if check shows it stopped
            if not current_status: self.playing = False
            return self.playing

    def close(self):
         """Explicitly stop playback and release all associated resources."""
         print(f"Closing AudioPlayer resources for {self.audio_file_path.name}...")
         self.stop() # Ensure playback is fully stopped first
         # Close wave file object handle
         if self.wf is not None and not getattr(self.wf, 'closed', True):
              try: self.wf.close()
              except Exception: pass
              self.wf = None
         # Close BytesIO object if it exists (from MP3 conversion)
         if hasattr(self, '_mp3_wav_data_io'):
              try: self._mp3_wav_data_io.close()
              except Exception: pass
              del self._mp3_wav_data_io # Remove reference
         # Shared PyAudio instance is terminated globally at script exit
         print("AudioPlayer resources released.")

    def __del__(self):
        """Ensure cleanup when object is garbage collected."""
        self.close()

# --- ESP32 Controller Class ---
class ESP32Controller:
    """Handles serial communication with the ESP32 device."""
    def __init__(self, port, baudrate, timeout=1):
        self.port = port; self.baudrate = baudrate; self.timeout = timeout; self.ser = None
        print(f"Attempting connection: {self.port} @ {self.baudrate} baud...")
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout, rtscts=False, dsrdtr=False)
            time.sleep(2.0); self.ser.flushInput(); print("Serial connection established.")
            self.read_initial_messages()
        except serial.SerialException as e: print(f"ERROR connecting: {e}"); self.ser = None
        except Exception as e: print(f"ERROR during serial connection: {e}"); self.ser = None

    def is_connected(self): return self.ser is not None and self.ser.is_open
    def send_command(self, command):
        if not self.is_connected(): print("Error: Not connected."); return False
        try:
            print(f"Sending to ESP32: {command}"); self.ser.write((command + '\n').encode('utf-8')); self.ser.flush(); return True
        except Exception as e: print(f"Error writing: {e}"); self.close(); return False
    def read_line(self, timeout_override=None):
        if not self.is_connected(): return None
        original_timeout = self.ser.timeout
        try:
            if timeout_override is not None: self.ser.timeout = timeout_override
            line = self.ser.readline()
            if timeout_override is not None: self.ser.timeout = original_timeout
            return line.decode('utf-8', errors='ignore').strip() if line else None
        except Exception as e: print(f"Error reading: {e}"); self.close(); return None
    def read_initial_messages(self, num_lines=10, timeout_per_line=0.3):
        if not self.is_connected(): return; print("--- Reading initial device output ---")
        ready_found = False; ready_signals = ["Ready for commands", "Setup complete"]
        for _ in range(num_lines):
            line = self.read_line(timeout_override=timeout_per_line)
            if line: print(f"<<< ESP32: {line}");
            if line and any(signal in line for signal in ready_signals): ready_found = True; break
            if not line: break
        if not ready_found: print("Warning: Did not receive initial 'Ready' message.")
        print("------------------------------------")
    def read_until_ready(self, timeout=10):
        if not self.is_connected(): return False; print("Waiting for ESP32 'Ready'...")
        start_time = time.time(); ready_signals = ["Ready for new commands", "Sequence finished or stopped"]
        while time.time() - start_time < timeout:
            line = self.read_line(timeout_override=0.5)
            if line: print(f"<<< ESP32: {line}");
            if line and any(signal in line for signal in ready_signals): print("ESP32 is ready."); return True
            if not self.is_connected(): print("Connection lost."); return False
        print("Timeout waiting for ESP32 'Ready'."); return False
    def run_sequence(self, sequence_name, start_offset_ms=0): # Accepts offset
        cmd = f"RUN:{sequence_name}"
        if start_offset_ms > 0: cmd += f":{start_offset_ms}" # Append offset if non-zero
        return self.send_command(cmd)
    def stop_sequence(self): return self.send_command("STOP")
    def close(self):
        if self.is_connected(): port = self.ser.port; print(f"Closing serial {port}...");
        try: self.ser.cancel_read(); self.ser.cancel_write(); self.ser.flush(); self.ser.close(); print(f"Port {port} closed.")
        except Exception as e: print(f"Error closing {port}: {e}")
        finally: self.ser = None

# --- Helper Function to Find Audio File ---
def find_matching_audio_file(sequence_name):
    """Searches script's directory for audio file: .wav > .flac > .mp3"""
    base_path = Path(__file__).parent; extensions = ['.wav', '.flac', '.mp3']
    print(f"Searching for '{sequence_name}.*' in '{base_path}'...")
    for ext in extensions:
        f = base_path / (sequence_name + ext)
        if f.is_file(): print(f"Found: {f}"); return f
    return None

# --- Synchronization Function ---
def synchronized_sequence(controller : ESP32Controller, sequence_name : str, start_offset_ms : int = 0):
    """Finds audio, runs sequence on ESP32 from offset, plays audio from offset."""
    if not controller or not controller.is_connected(): print("Cannot start: Not connected."); return

    print(f"\n--- Starting Sequence '{sequence_name}' at offset {start_offset_ms} ms ---")
    audio_path = find_matching_audio_file(sequence_name)
    audio_player = None; audio_duration_sec = 0; has_audio = False

    if audio_path:
        print(f"  Audio File: {audio_path.name}");
        try:
            # Pass configured volume when creating player
            audio_player = AudioPlayer(audio_path, volume=DEFAULT_AUDIO_VOLUME)
            if audio_player.loaded: audio_duration_sec = audio_player.duration / 1000.0; has_audio = True
            else: print("ERROR: Load failed. No audio.")
        except Exception as e: print(f"ERROR initializing audio: {e}. No audio.")
    else: print("  Audio File: None found.")

    start_time = time.time(); led_running = False; audio_playing = False
    try:
        print(f"Sending RUN command to ESP32 (Offset: {start_offset_ms} ms)...")
        # Pass offset in command via run_sequence method
        if not controller.run_sequence(sequence_name, start_offset_ms): raise Exception("Failed to send RUN.")
        led_running = True; time.sleep(0.05)

        if has_audio and audio_player:
            print(f"Starting audio playback from {start_offset_ms} ms...")
            # Pass offset to audio player's play method
            if audio_player.play(start_position_ms=start_offset_ms):
                audio_playing = True; print(f"Audio started ({audio_player.file_format}, {audio_duration_sec:.2f}s total).")
            else: print("ERROR starting audio. LEDs continue."); has_audio = False

        # --- Wait Logic ---
        if audio_playing:
            # Calculate remaining duration based on offset
            effective_duration_sec = max(0.1, audio_duration_sec - (start_offset_ms / 1000.0))
            print(f"Waiting for audio ({effective_duration_sec:.2f}s remaining) or interrupt...")
            wait_start = time.time()
            # Wait slightly longer than calculated remaining duration
            while time.time() - wait_start < effective_duration_sec + 0.2:
                 if not audio_player.is_playing(): print("Audio stopped."); break
                 msg = controller.read_line(0.01); # Non-blocking read ESP32
                 if msg: print(f"<<< ESP32: {msg}")
                 time.sleep(0.05) # Prevent busy-waiting
            print("Audio wait finished.")
        else: # No audio, wait for ESP32 or interrupt
            print("LEDs running. Waiting for ESP32 'Ready' or interrupt (Ctrl+C)...")
            while controller.is_connected():
                 line = controller.read_line(0.5); # Poll ESP32 status
                 if line: print(f"<<< ESP32: {line}");
                 if line and any(s in line for s in ["Ready", "finished", "stopped"]): print("ESP32 ready."); break
            if not controller.is_connected(): print("Lost connection.")

    except KeyboardInterrupt: print("\nKeyboard Interrupt!"); led_running = False
    except Exception as e: print(f"\nERROR during sequence: {e}"); led_running = False
    finally:
        print("Cleaning up sequence...");
        if audio_player: audio_player.close() # Close player if used
        # Only send STOP if sequence was interrupted or errored, not on normal completion
        if not led_running and controller and controller.is_connected():
             print("Sending STOP due to interrupt/error...")
             controller.stop_sequence()
        if controller and controller.is_connected(): controller.read_until_ready(timeout=5) # Wait for confirmation
        print(f"--- Sequence '{sequence_name}' Finished (Total time: {time.time() - start_time:.2f}s) ---")

# --- Main Execution ---
if __name__ == "__main__":
    script_dir = Path(__file__).parent
    try: os.chdir(script_dir); print(f"Running in directory: {os.getcwd()}")
    except Exception as e: print(f"Warning: Could not change directory: {e}")

    esp_controller = ESP32Controller(SERIAL_PORT, BAUD_RATE)

    if esp_controller.is_connected():
        try:
            print("\n--- Manual Control Interface ---")
            print(f"ESP32 on {SERIAL_PORT}. Default Volume: {DEFAULT_AUDIO_VOLUME:.2f}")
            print("Known available sequences (via RUN:<name>):")
            # >>> KNOWN_SEQUENCES_MARKER <<< (Converter script adds lines below)
            print("  - rampTestSequence") # Added by converter
            print("  - h_gamma_3") # Added by converter
            print("  - three_step_frac") # Added by converter
            print("  - _40hzsplit") # Added by converter
            # (Add more manually or via converter script)
            print("\nEnter commands:")
            print("  RUN:<name> [START=<ms>] (e.g., RUN:h_gamma_3 START=30000)")
            print("  STOP")
            print("  EXIT")
            print("----------------------------------------------------")

            while True:
                while esp_controller.is_connected(): # Read async msgs
                    msg = esp_controller.read_line(0.01);
                    if msg: print(f"<<< ESP32: {msg}")
                    else: break
                if not esp_controller.is_connected(): print("Connection lost."); break

                try: cmd_input = input("> ").strip()
                except EOFError: cmd_input = "EXIT"; print("EXIT")
                except KeyboardInterrupt: cmd_input = "EXIT"; print("\nEXIT")

                if not cmd_input: continue

                cmd_upper = cmd_input.upper()
                if cmd_upper == "EXIT":
                    if esp_controller.is_connected(): esp_controller.stop_sequence()
                    break

                elif cmd_upper == "STOP":
                    print("Manual STOP sent.");
                    if esp_controller.is_connected(): esp_controller.stop_sequence()

                elif cmd_upper.startswith("RUN:"):
                    parts = cmd_input.split(maxsplit=1)
                    run_cmd_part = parts[0]
                    start_arg_part = parts[1] if len(parts) > 1 else ""

                    sequence_name = run_cmd_part[4:].strip()
                    start_ms = 0 # Default start time

                    # Parse optional START=<ms> argument using regex for robustness
                    if start_arg_part:
                        match = re.match(r"START\s*=\s*(\d+)", start_arg_part.upper())
                        if match:
                            try: start_ms = int(match.group(1))
                            except ValueError: print("Warning: Invalid START time value, using 0ms.")
                        else: print(f"Warning: Ignoring unrecognized argument '{start_arg_part}'. Use START=<ms>.")

                    if sequence_name:
                         print("Requesting previous sequence STOP (if any)...")
                         if esp_controller.is_connected():
                              esp_controller.stop_sequence(); time.sleep(0.1)
                              # esp_controller.read_until_ready(timeout=3) # Optional wait
                         else: print("Cannot stop, disconnected."); break

                         if esp_controller.is_connected():
                              # Call synchronized_sequence with name and parsed offset
                              synchronized_sequence(esp_controller, sequence_name, start_ms)
                         else: print("Cannot run sequence, disconnected.")
                    else:
                         print("Please specify a sequence name after RUN:")
                else:
                    print(f"Unknown command: '{cmd_input}'. Use RUN:<name> [START=<ms>], STOP, or EXIT.")

        except KeyboardInterrupt: print("\nExiting.")
        except Exception as e: print(f"\nMain loop error: {e}")
        finally:
             print("Cleaning up before exit...")
             AudioPlayer.terminate_pyaudio() # Terminate shared PyAudio
             if esp_controller: esp_controller.close() # Close serial
    else:
        print(f"Initial connection failed to {SERIAL_PORT}. Check config.ini and device.")
        AudioPlayer.terminate_pyaudio() # Terminate PyAudio if it was init'd

    print("\nPython script finished.")
