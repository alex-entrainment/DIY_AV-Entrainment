# Realtime Backend (Work in progress)

This crate provides the initial structure for a Rust-based audio generation engine.
It mirrors the JSON track format used by the Python implementation and exposes a
minimal PyO3 interface for starting and stopping playback.

Implemented components:

- **Project setup** with Cargo, PyO3, CPAL and DSP-related dependencies.
- **Track models** mirroring the Python data structures.
- **Basic DSP utilities** (noise generators, sine wave, ADSR, pan).
- **Skeleton scheduler** capable of processing blocks and advancing steps.
- **Audio thread bootstrap** using CPAL with a stoppable loop.
- **Python bindings** with `start_stream` and `stop_stream` functions.

Remaining tasks (see `REALTIME_BACKEND_PLAN.md` for full roadmap):

- Proper voice implementations for each synth function.
- Crossfade and transition handling in `TrackScheduler`.
- Unit tests comparing DSP routines with the Python version.
- Integration with the rest of the application via a high-level Python wrapper.

## Building the Python Extension

Install [maturin](https://github.com/PyO3/maturin) and compile the library as a Python module:

```bash
pip install maturin
cd src/audio/realtime_backend
maturin develop --release
```

This produces a `realtime_backend` extension that can be imported from Python:

```python
import realtime_backend

# `track_json` should be a JSON string exported by the GUI
# start playback 30 seconds into the track
realtime_backend.start_stream(track_json, start_time=30.0)

# Render a 60 second sample to a wav file
realtime_backend.render_sample_wav(track_json, "sample.wav")

# Render the entire track to a wav file
realtime_backend.render_full_wav(track_json, "full_output.wav")
```

Call `realtime_backend.stop_stream()` to halt playback.

## WebAssembly Build

The backend can also be compiled to WebAssembly for use in the browser.
Install `wasm-pack` and build the crate:

```bash
wasm-pack build --target web --release
```

The generated `pkg/` folder contains `realtime_backend.js` and `realtime_backend_bg.wasm`.  See `WASM_GUIDE.md` for integration details.

## Command Line Usage

A command line interface is provided for quickly auditioning or rendering tracks
without Python. Build the project normally and run the `realtime_backend`
binary. Pass the path to a track JSON file and optionally enable full
rendering:

```bash
cargo run --bin realtime_backend -- --path path/to/track.json --generate false
```

Use the `--gpu` flag to enable GPU acceleration (build with `--features gpu`):

```bash
cargo run --bin realtime_backend --features gpu -- --path path/to/track.json --gpu true
```

If `--generate true` is supplied, the entire track is written to the
`outputFilename` specified in the JSON. Otherwise it streams the audio directly
to the default output device. Press `Ctrl+C` to stop streaming.

## Optional GPU Acceleration

The backend includes an experimental GPU mixing path behind the `gpu` Cargo
feature. When enabled, a compute shader is used to sum the active voice buffers
for each processed block.

Build the library with GPU support using:

```bash
cargo build --features gpu
```

If the feature is disabled (the default), mixing falls back to a CPU
implementation.
