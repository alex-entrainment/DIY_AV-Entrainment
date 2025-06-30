Project: Real-time Audio Backend
This project is broken down into several phases, from setting up the environment to deploying the web service. Each step includes a description of the task, the goal, and the acceptance criteria.

Phase 1: Project Setup and Initial Rust Crate
Goal: Create the foundational structure for the Rust backend and ensure the build environment is correctly configured.

TODO 1.1: Create the Rust Crate

Task: Initialize a new Rust library crate within the project structure.

Location: src/audio/realtime_backend/

Command:

Bash

cd src/audio/realtime_backend
cargo new --lib .
Acceptance Criteria:

A Cargo.toml file is created in the specified directory.

A src/lib.rs file is created.

TODO 1.2: Configure Cargo.toml for a Python Module

Task: Add the necessary dependencies to Cargo.toml to build a Python extension module.

Dependencies:

pyo3: For Python bindings.

cpal: For cross-platform audio output.

serde, serde_json: For parsing JSON data from Python.

Configuration:

Ini, TOML

[package]
name = "realtime_backend"
version = "0.1.0"
edition = "2021"

[lib]
name = "realtime_backend"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.21.0", features = ["extension-module"] }
cpal = "0.15.3"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
Acceptance Criteria:

The Cargo.toml file is updated with the correct dependencies and library configuration.

The project can be built successfully with cargo build.

Phase 2: Core DSP Engine in Rust
Goal: Re-implement all audio synthesis and processing logic from Python to Rust, ensuring functional parity.

TODO 2.1: Define Data Structures for Track Definitions

Task: Create Rust structs that mirror the JSON structure of the track definitions. Use serde to enable deserialization.

File: src/audio/realtime_backend/src/data_structures.rs

Implementation:

Rust

use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub struct Envelope {
    pub attack: f32,
    pub decay: f32,
    pub sustain: f32,
    pub release: f32,
}

#[derive(Deserialize, Debug)]
pub struct Voice {
    pub voice_type: String,
    pub frequency: f32,
    pub intensity: f32,
    pub envelope: Envelope,
    // ... other voice parameters
}

#[derive(Deserialize, Debug)]
pub struct Step {
    pub duration: f32,
    pub voices: Vec<Voice>,
    // ... other step parameters
}

#[derive(Deserialize, Debug)]
pub struct Track {
    pub steps: Vec<Step>,
    // ... other track parameters
}
Acceptance Criteria:

Structs for Track, Step, Voice, and Envelope are defined.

These structs can be deserialized from a sample JSON string.

TODO 2.2: Implement Synthesis Functions

Task: Port the synthesis algorithms from the Python synth_functions module to a new Rust module.

File: src/audio/realtime_backend/src/synthesis.rs

Functions to Port:

generate_binaural_beats(...)

generate_isochronic_tones(...)

apply_qam(...)

Any other custom synthesis functions.

Implementation: Focus on numerical accuracy and efficiency. For example, a sine wave generator could be implemented as follows:

Rust

pub fn sine_wave(phase: &mut f32, frequency: f32, sample_rate: u32) -> f32 {
    let sample = (2.0 * std::f32::consts::PI * *phase).sin();
    *phase += frequency / sample_rate as f32;
    if *phase > 1.0 {
        *phase -= 1.0;
    }
    sample
}
Acceptance Criteria:

All synthesis algorithms are implemented in Rust.

Unit tests are created for each function to verify the output matches the Python implementation for a given input.

TODO 2.3: Implement the Audio Streamer

Task: Create a struct that manages the state of the audio stream, generates samples in blocks, and applies envelopes.

File: src/audio/realtime_backend/src/streamer.rs

Implementation:

Rust

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
// ... other imports

pub struct AudioStreamer {
    // ... fields for track data, sample rate, running state, etc.
}

impl AudioStreamer {
    pub fn new(track: Track, sample_rate: u32) -> Self {
        // ... constructor logic
    }

    pub fn run(&self) {
        // ... CPAL setup and stream generation loop
    }

    fn generate_sample_block(&mut self, output: &mut [f32]) {
        // ... logic to fill the output buffer with samples from the synth functions
    }
}
Acceptance Criteria:

The AudioStreamer can be initialized with track data.

It can generate a continuous stream of audio samples based on the track definition.

Phase 3: Python FFI Bindings
Goal: Expose the Rust DSP engine as a Python module so that it can be controlled from the existing Python codebase.

TODO 3.1: Create the PyO3 Module

Task: Define the Python module and the functions that will be exposed.

File: src/audio/realtime_backend/src/lib.rs

Implementation:

Rust

use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
fn start_stream(track_json_str: &str, sample_rate: u32) -> PyResult<()> {
    // 1. Deserialize the JSON string into Rust structs
    // 2. Initialize and run the AudioStreamer
    // This will likely need to run in a separate thread
    Ok(())
}

#[pyfunction]
fn stop_stream() -> PyResult<()> {
    // Logic to signal the audio thread to stop
    Ok(())
}

#[pymodule]
fn realtime_backend(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(start_stream, m)?)?;
    m.add_function(wrap_pyfunction!(stop_stream, m)?)?;
    Ok(())
}
Acceptance Criteria:

The realtime_backend module can be compiled.

The start_stream and stop_stream functions are defined.

TODO 3.2: Build and Test the Python Extension

Task: Use a build tool like maturin to compile the Rust crate into a Python wheel and install it.

Command:

Bash

pip install maturin
maturin develop
Acceptance Criteria:

The Rust crate compiles successfully into a .so or .pyd file.

You can run python and execute import realtime_backend.

The start_stream and stop_stream functions can be called from Python.

Phase 4: Python Wrapper and Integration
Goal: Create a user-friendly Python module that abstracts the low-level Rust backend.

TODO 4.1: Create the Python Wrapper Module

Task: Write a thin Python wrapper that handles data conversion and provides a clean API.

File: src/audio/realtime.py

Implementation:

Python

import json
import realtime_backend # This is the Rust extension

def play_track(track_definition: dict, sample_rate: int = 44100):
    """
    Starts streaming audio for the given track definition.
    """
    try:
        track_json_str = json.dumps(track_definition)
        realtime_backend.start_stream(track_json_str, sample_rate)
    except Exception as e:
        print(f"Error starting stream: {e}")

def stop_playback():
    """
    Stops the currently playing audio stream.
    """
    realtime_backend.stop_stream()
Acceptance Criteria:

The play_track function correctly serializes the Python dictionary to a JSON string.

The module successfully delegates calls to the Rust backend.

TODO 4.2: Integrate with Existing Codebase

Task: Modify the existing Python scripts that currently render audio to disk to use the new real-time backend instead.

Acceptance Criteria:

The application can now play tracks in real-time without writing to a file first.

The existing track definition .json files work without modification.

Phase 5: Web Service Layer
Goal: Expose the real-time audio generation functionality through a web API.

TODO 5.1: Set up a FastAPI/Flask Server

Task: Create a simple web server application.

File: src/web/main.py

Dependencies: fastapi, uvicorn, websockets

Implementation (FastAPI example):

Python

from fastapi import FastAPI, WebSocket
from src.audio.realtime import play_track, stop_playback

app = FastAPI()

@app.post("/play")
async def play_track_endpoint(track_definition: dict):
    play_track(track_definition)
    return {"status": "playing"}

@app.post("/stop")
async def stop_track_endpoint():
    stop_playback()
    return {"status": "stopped"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        # Handle WebSocket communication for real-time updates
        data = await websocket.receive_text()
        await websocket.send_text(f"Message text was: {data}")
Acceptance Criteria:

The server can be started.

The /play and /stop endpoints are functional and can be tested with a tool like curl or Postman.

TODO 5.2: Implement WebSocket for Real-time Control

Task: Enhance the WebSocket endpoint to allow for more granular control (e.g., pause, skip to next step) and to send status updates to the client.

Acceptance Criteria:

A web client can connect to the /ws endpoint.

Sending a command like {"command": "pause"} through the WebSocket pauses the audio.

The server sends status updates (e.g., {"status": "paused"}) back to the client.

Phase 6: Testing and Performance Validation
Goal: Ensure the backend is robust, bug-free, and meets the performance objectives.

TODO 6.1: Write Unit Tests for Rust Code

Task: Create unit tests for all synthesis functions and DSP logic.

Command: cargo test

Acceptance Criteria:

All Rust modules have comprehensive test coverage.

All tests pass.

TODO 6.2: Write Integration Tests

Task: Create Python tests that load a JSON track, start playback, and verify that the audio is being generated.

Framework: pytest

Acceptance Criteria:

Tests confirm that calling play_track results in audio output.

Tests verify that stop_playback correctly terminates the audio.

TODO 6.3: Measure Performance

Task: Profile the application to measure CPU usage and latency.

Tools: System monitoring tools (like htop or Task Manager), and logging within the Rust application to measure block generation time.

Acceptance Criteria:

CPU usage remains below a predefined threshold (e.g., 20% on a target machine) when playing a complex track.

Audio latency is low enough to be unnoticeable (typically under 100ms).
