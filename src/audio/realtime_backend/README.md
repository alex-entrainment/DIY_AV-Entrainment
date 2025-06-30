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
