# Real-time Audio Backend Plan

This document outlines a proposed approach for a high-speed audio generation backend that works with the existing Python generator in this repository. The goal is to stream audio in real time from the track definition `.json` files created by the GUI editor while achieving better performance than pure Python can offer.

## 1. Objectives

- **Real-time generation** of audio defined by the GUI editor without pre-rendering to disk.
- **High performance** using a compiled language so that multiple voices and transitions can be mixed on the fly.
- **Seamless integration** with the current Python code base so existing track definitions remain valid.
- **Web UI compatibility** to enable a meditation audio service.

## 2. Technology Choice

Rust is suggested for the backend because it offers:

- Native speeds comparable to C/C++ with modern safety guarantees.
- Excellent cross-platform support and tooling.
- Straightforward interoperability with Python via [PyO3](https://pyo3.rs/).

Other languages (such as C++) could also work, but Rust's package ecosystem (crates.io) and safety benefits make it a solid option.

## 3. Architecture Overview

1. **Track Parsing**
   - Reuse the existing Python functions that load JSON track files. The parsed data (steps, voices, envelopes) is passed to Rust through FFI bindings.
2. **Core DSP Engine in Rust**
   - Implement the synthesis algorithms found in `synth_functions` (binaural beats, isochronic tones, QAM, etc.) using efficient numeric routines.
   - Generate samples in blocks (e.g., 64 or 128 frames) to keep latency low.
   - Apply envelopes and crossfades just like the Python version.
3. **Python Bindings**
   - Expose the Rust engine as a Python module using PyO3.
   - Provide functions such as `start_stream(track_json: dict, sample_rate: int)` and `stop_stream()` so that existing scripts can invoke real-time playback with minimal changes.
4. **Audio Output**
   - Use a Rust audio library (for example, [CPAL](https://github.com/RustAudio/cpal)) to stream the generated samples to the system audio device.
   - Python remains responsible for sequence control and GUI interactions, while Rust handles the heavy DSP work.
5. **Web Integration**
   - Wrap the Python/Rust combo in a simple web server (Flask/FastAPI).
   - Provide endpoints to start, stop, or update a playing track. Web clients connect via WebSocket to receive status updates and control playback.
   - Audio can be sent to the client as an HTTP stream or played server-side and relayed over a low-latency audio transport if desired.

## 4. Implementation Steps

1. **Create Rust Crate**
   - Set up a new `realtime_backend` crate under `src/audio/realtime_backend/`.
   - Reimplement the core synth algorithms in Rust, closely matching the Python versions to ensure the same track definitions work.
2. **Expose FFI Bindings**
   - Using PyO3, export a minimal API:
     ```rust
     #[pyfunction]
     fn start_stream(track_json: &PyAny, sample_rate: u32) -> PyResult<()> { /* ... */ }

     #[pyfunction]
     fn stop_stream() { /* ... */ }
     ```
   - Build the crate as a Python extension so it can be imported as `import realtime_backend`.
3. **Python Wrapper**
   - Provide a thin Python module (e.g., `audio/realtime.py`) that forwards calls to the compiled Rust extension and performs any required data conversion.
4. **Web Service Layer**
   - Implement an optional Flask/FastAPI server in Python that exposes REST/WebSocket endpoints for the meditation service. The server loads track definitions, passes them to the Rust backend, and manages user sessions.
5. **Testing**
   - Verify that tracks exported by the GUI play correctly in real time with the Rust engine.
   - Measure latency and CPU usage to ensure performance goals are met.

## 5. Future Enhancements

- Add SIMD optimizations in Rust for further speed improvements.
- Support additional file formats or streaming encoded audio directly to the browser.
- Provide headless generation so the service can run on minimal hardware.

---

This plan focuses on integrating a Rust-based DSP engine with the existing Python code so that real-time generation is possible without rewriting the GUI or track definition logic. It keeps the current JSON format intact, allowing a smooth transition from offline rendering to low-latency playback inside a future web UI.

