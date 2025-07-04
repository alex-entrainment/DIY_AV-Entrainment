# WebAssembly Guide

This document explains how to compile the Rust based realtime DSP backend into a WebAssembly component and use it for high speed Web Audio.

## Prerequisites

- Rust toolchain (stable or nightly)
- [`wasm-pack`](https://rustwasm.github.io/wasm-pack/installer/)
- `wasm32-unknown-unknown` target installed via `rustup target add wasm32-unknown-unknown`
- A modern JavaScript bundler or build tool (e.g. Vite, Webpack)

## Building the WASM module

1. Navigate to the realtime backend crate:

   ```bash
   cd src/audio/realtime_backend
   ```

2. Build with `wasm-pack` (using the `web` feature):

   ```bash
   wasm-pack build --target web --release --no-default-features --features web
   ```

   This generates a `pkg/` directory containing `realtime_backend.js` and `realtime_backend_bg.wasm`.

3. Copy the contents of `pkg/` into your web application's source directory or serve them directly.

## Using in the Browser

Import the generated module and initialize it before starting audio playback:

```javascript
import init, {
  start_stream,
  stop_stream,
  pause_stream,
  resume_stream,
  current_step,
  elapsed_samples
} from './realtime_backend.js';

async function initAudio(trackJson, sampleRate) {
  await init(); // loads realtime_backend_bg.wasm

  await start_stream(JSON.stringify(trackJson), trackJson.global.sample_rate);

}
```

The exported functions mirror the Python bindings. `start_stream` begins playback using the Web Audio API under the hood, while `stop_stream` halts it.
`pause_stream` temporarily silences output without losing playback position and `resume_stream` continues from where it left off. You can poll progress using `current_step()` and `elapsed_samples()`.

### Performance Notes

WebAssembly allows the DSP routines to run at near-native speed in the browser. For best results, ensure the audio worklet thread is not blocked by heavy JavaScript processing. The recommended setup uses an `AudioWorklet` that reads from a shared ring buffer filled by the WASM engine. This avoids repeated `Float32Array` allocations and lowers latency compared to a `ScriptProcessor` based approach.

### Limitations

- Only the implemented voices in `realtime_backend` are available.
- Browser security policies may require user interaction before audio can start.

Refer to `REALTIME_BACKEND_PLAN.md` for the remaining tasks and planned features.

## Example Vite Integration (web_ui)

For a working browser demo see the `web_ui` folder inside this repository
(`audio/src/web_ui`). It contains a minimal Vite project that consumes the WASM
module generated above. The setup copies the `pkg` output into the project's
`src/pkg` directory via `npm run sync-wasm` so Vite can bundle the files.

The example creates a `SharedRingBuffer` (defined in
`web_ui/src/ringbuffer.js`) backed by `SharedArrayBuffer` objects. The main
thread fills this buffer using the `process_block` export from the WASM module
while an `AudioWorklet` (`web_ui/src/wasm-worklet.js`) reads from it each audio
callback. This pattern avoids reallocating `Float32Array` objects and keeps the
audio thread free of heavy JavaScript work.

You can reuse these components in your own project by importing the ring buffer
class and worklet script and wiring them up to your build system. The
`web_ui/src/main.js` file demonstrates the complete flow: initializing the
module, setting up the `AudioContext`, populating the ring buffer, and controlling
playback. Adapting this code provides a reliable blueprint for any application
that needs low-latency audio streaming from WebAssembly.

