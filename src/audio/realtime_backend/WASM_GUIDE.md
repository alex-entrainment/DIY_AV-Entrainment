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
import init, { start_stream, stop_stream } from './realtime_backend.js';

async function initAudio(trackJson) {
  await init(); // loads realtime_backend_bg.wasm
  await start_stream(JSON.stringify(trackJson));
}
```

The exported functions mirror the Python bindings. `start_stream` begins playback using the Web Audio API under the hood, while `stop_stream` halts it.

### Performance Notes

WebAssembly allows the DSP routines to run at near-native speed in the browser. For best results, ensure the audio worklet thread is not blocked by heavy JavaScript processing. The module currently outputs stereo samples at the browser's sample rate and writes them into a `ScriptProcessor` or `AudioWorklet` node.

### Limitations

- Only the implemented voices in `realtime_backend` are available.
- Browser security policies may require user interaction before audio can start.

Refer to `REALTIME_BACKEND_PLAN.md` for the remaining tasks and planned features.

