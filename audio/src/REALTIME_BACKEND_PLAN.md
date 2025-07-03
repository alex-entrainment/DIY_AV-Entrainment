Project: Real-time Audio Backend
This refined plan outlines the porting of the audio generation logic from the Python files (sound_creator.py, noise_flanger.py, common.py, audio_engine.py) to a high-performance, real-time Rust backend.

Phase 1: Project Setup and Dependencies
Goal: Establish the Rust project structure and declare all necessary dependencies for DSP, Python interoperability, and audio output.

TODO 1.1: Initialize Rust Crate

Task: Create the Rust library crate.

Location: audio/src/realtime_backend/

Command: cargo new --lib .

Acceptance Criteria: Cargo.toml and src/lib.rs files are created.

TODO 1.2: Configure Cargo.toml with Dependencies

Task: Add dependencies for Python bindings, audio I/O, DSP, and serialization.

Configuration:

Ini, TOML

[package]
name = "realtime_backend"
version = "0.1.0"
edition = "2021"

[lib]
name = "realtime_backend"
crate-type = ["cdylib"] # Compile as a dynamic library for Python

[dependencies]
# Python FFI
pyo3 = { version = "0.21.0", features = ["extension-module"] }

# Audio Output
cpal = "0.15.3"

# Serialization (for track data)
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# DSP & Numerics
rustfft = "6.1"  # For any FFT-based processing
biquad = "0.4"   # For IIR filters (replacement for scipy.signal)
rand = "0.8"     # For noise generation

# Concurrency
crossbeam = "0.8" # For message passing between threads
parking_lot = "0.12" # For efficient Mutexes
Acceptance Criteria: The Cargo.toml is updated. The project builds successfully with cargo build.

Phase 2: Porting Core DSP and Synthesis Logic
Goal: Re-implement the fundamental audio building blocks from Python to Rust, ensuring functional parity and high performance.

TODO 2.1: Define Core Data Structures in Rust

Task: Create Rust structs that mirror the JSON track format and voice presets. Use serde for deserialization. Reference sound_creator.py's track_data structure and voice_file.py's VoicePreset.

File: audio/src/realtime_backend/src/models.rs

Implementation:

Rust

// In src/models.rs
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize, Debug, Clone)]
pub struct VolumeEnvelope {
    #[serde(rename = "type")]
    pub envelope_type: String,
    pub params: HashMap<String, f64>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct VoiceData {
    pub synth_function_name: String,
    pub params: HashMap<String, serde_json::Value>,
    pub volume_envelope: Option<VolumeEnvelope>,
    #[serde(default)]
    pub is_transition: bool,
}

#[derive(Deserialize, Debug, Clone)]
pub struct StepData {
    pub duration: f64,
    pub voices: Vec<VoiceData>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct GlobalSettings {
    pub sample_rate: u32,
    pub crossfade_duration: f64,
    pub crossfade_curve: String,
}

#[derive(Deserialize, Debug, Clone)]
pub struct TrackData {
    pub global_settings: GlobalSettings,
    pub steps: Vec<StepData>,
    // Add background_noise and clips later
}
Acceptance Criteria: Rust structs accurately represent the JSON structure. They can be deserialized from a sample JSON file exported by the GUI.

TODO 2.2: Port Fundamental Synthesis Functions

Task: Re-implement the core DSP utilities from common.py and the optimized noise generators from noise_flanger.py.

File: audio/src/realtime_backend/src/dsp/ (create a module for these).

Functions to Port:

generate_pink_noise_samples (from noise_flanger.py, as it's Numba-optimized).

generate_brown_noise_samples (from noise_flanger.py).

sine_wave and sine_wave_varying (from common.py).

adsr_envelope, linen_envelope, trapezoid_envelope_vectorized (from common.py).

pan2 (from common.py).

calculate_transition_alpha (from common.py).

Acceptance Criteria: Each Rust function passes unit tests that compare its output against pre-computed results from the original Python function to ensure parity.

TODO 2.3: Implement the Real-time Scheduler and Mixer

Task: Create the main engine that manages the track's lifecycle, replicating the logic of assemble_track_from_data from sound_creator.py in a real-time, block-based manner.

File: audio/src/realtime_backend/src/scheduler.rs

Implementation Steps:

Define a Voice trait:

Rust

pub trait Voice: Send + Sync {
    fn process(&mut self, output_buffer: &mut [f32]);
    fn is_finished(&self) -> bool;
}
Create concrete voice synthesizers: For each function in SYNTH_FUNCTIONS that you want to support (e.g., binaural_beat_synth, isochronic_tone_synth), create a struct that implements the Voice trait. These structs will hold the state for one voice (phase, current envelope value, etc.).

Build the TrackScheduler:

This struct will hold the TrackData and the state of the audio stream (e.g., current_sample, current_step_index).

It will maintain a Vec<Box<dyn Voice>> of currently active voices.

Implement the main processing loop (process_block):

This method will be called by the audio callback to fill a buffer (e.g., 256 samples).

Inside the loop, it checks if it needs to transition to the next step based on current_sample.

When a new step starts, it instantiates the required Voice structs for that step.

It calls process on each active voice to get its output.

It mixes the outputs of all voices into a single block.

It implements the crossfading logic from crossfade_signals when transitioning between steps. This means for the duration of a crossfade, voices from both the old and new steps will be active.

It removes voices that are is_finished().

Acceptance Criteria: The TrackScheduler can load a track definition, process audio in blocks, and correctly manage the lifecycle of voices according to the steps in the track data.

Phase 3: Audio Output and Python Interface
Goal: Stream the generated audio to the system's output device and create a safe, clean FFI layer for Python to control the engine.

TODO 3.1: Implement CPAL Audio Callback

Task: Set up the CPAL audio stream and connect it to the TrackScheduler. The state (scheduler, stream) must be managed in a thread-safe manner.

File: audio/src/realtime_backend/src/audio_io.rs

Implementation:

Rust

// In lib.rs or a dedicated module
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use parking_lot::Mutex;
use std::sync::Arc;

// Global state, to be managed by PyO3 module
struct AudioEngineState {
    scheduler: TrackScheduler,
    // Any other state...
}

// The CPAL stream object. Keep it alive to keep playing.
static STREAM: Mutex<Option<cpal::Stream>> = Mutex::new(None);

// This function will set up and run the stream.
pub fn run_audio_stream(scheduler: Arc<Mutex<TrackScheduler>>) {
    // ... CPAL host/device setup ...
    let audio_callback = move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
        // This closure is the real-time audio thread
        let mut scheduler_lock = scheduler.lock();
        scheduler_lock.process_block(data);
    };
    let stream = device.build_output_stream(&config, audio_callback, err_fn, None).unwrap();
    stream.play().unwrap();
    *STREAM.lock() = Some(stream); // Store the stream to keep it alive
}

pub fn stop_audio_stream() {
    *STREAM.lock() = None; // Drop the stream, stopping audio
}
Acceptance Criteria: The Rust application can open an audio stream and continuously feed it data generated by the TrackScheduler.

TODO 3.2: Expose FFI Bindings with PyO3

Task: Create the Python module interface, managing the TrackScheduler and audio stream in a separate thread.

File: audio/src/realtime_backend/src/lib.rs

Implementation:

Rust

use pyo3::prelude::*;
use std::sync::{Arc, Mutex};
use std::thread;

// Use a global, thread-safe handle to the engine state
static ENGINE_STATE: Mutex<Option<Arc<Mutex<TrackScheduler>>>> = Mutex::new(None);

#[pyfunction]
fn start_stream(track_json_str: String) -> PyResult<()> {
    let track_data: TrackData = serde_json::from_str(&track_json_str)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Create a new scheduler for this track
    let scheduler = Arc::new(Mutex::new(TrackScheduler::new(track_data)));

    // Store the handle so we can control it later
    *ENGINE_STATE.lock() = Some(scheduler.clone());

    // Spawn a dedicated audio thread
    thread::spawn(move || {
        // This function contains the CPAL setup and stream loop
        audio_io::run_audio_stream(scheduler); 
    });

    Ok(())
}

#[pyfunction]
fn stop_stream() -> PyResult<()> {
    *ENGINE_STATE.lock() = None; // Drop our reference to the scheduler
    audio_io::stop_audio_stream(); // Stop the audio stream itself
    Ok(())
}

#[pymodule]
fn realtime_backend(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(start_stream, m)?)?;
    m.add_function(wrap_pyfunction!(stop_stream, m)?)?;
    Ok(())
}
Acceptance Criteria: The Rust code compiles into a realtime_backend.pyd (or .so) file that can be imported and used in Python.

Phase 4: Python Wrapper and Final Integration
Goal: Create a clean Python interface and integrate the real-time engine into the main application.

TODO 4.1: Build the Python Wrapper Module

Task: Create a high-level Python module that abstracts the realtime_backend.

File: src/audio/realtime.py

Implementation: This module will be almost identical to the plan from our first conversation, providing a friendly play_track(track_definition: dict) function that handles JSON serialization.

Acceptance Criteria: realtime.py can successfully call the Rust backend functions.

TODO 4.2: Build and Install the Rust Extension

Task: Use maturin to manage the build and installation process.

Command:

Bash

pip install maturin
maturin develop # This compiles and installs the module in the current venv
Acceptance Criteria: After running maturin develop, import realtime_backend works in a Python shell, and the functions from the wrapper module can be called.

Phase 5 & 6: Web Service and Testing
These phases remain conceptually the same as the previous plan, but the testing becomes more concrete.

TODO 5.1: Implement Web Service (FastAPI/Flask)

Task: Build the web server using the new realtime.py wrapper.

Acceptance Criteria: REST endpoints like /play and /stop correctly control the audio playback via the Rust engine.

TODO 6.1: Write Specific Unit Tests

Task: For each ported DSP function in Rust, create a test that compares its output to a pre-computed "golden file" generated by the original Python function.

Example (pytest test in Python to generate data):

Python

from src.common import adsr_envelope
import numpy as np

def test_generate_adsr_golden_file():
    t = np.linspace(0, 1, 44100)
    env = adsr_envelope(t, attack=0.1, decay=0.2, sustain_level=0.5, release=0.3)
    np.savetxt("tests/golden_files/adsr_golden.txt", env)
Example (Rust test):

Rust

#[test]
fn test_adsr_against_golden_file() {
    // let golden_data = load_golden_file("tests/golden_files/adsr_golden.txt");
    // let t = ...;
    // let rust_env = dsp::adsr_envelope(t, ...);
    // assert_close(golden_data, rust_env, tolerance);
}
Acceptance Criteria: All Rust DSP functions are verified against their Python counterparts.

TODO 6.2: Create Full Track Integration Tests

Task: Write a Python test that:

Loads a complex track JSON file.

Renders it to a WAV file using the original sound_creator.generate_audio.

Plays the same track JSON using the new realtime.play_track with a mock audio output that captures the sample data to a buffer.

Compares the two outputs. They won't be identical due to timing differences, but their overall structure and sound should be very similar (e.g., compare via RMS or cross-correlation).

Acceptance Criteria: The real-time engine produces audio that is functionally equivalent to the offline renderer for a given track.
