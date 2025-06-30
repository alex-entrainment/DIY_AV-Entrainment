# Web UI for the Realtime Backend

This directory provides a minimal browser interface for the Rust audio engine
when compiled to WebAssembly. It is meant for quick experimentation directly
from a web page.

## Building the WebAssembly Package

1. Install [`wasm-pack`](https://rustwasm.github.io/wasm-pack/installer/).
2. Build the backend with the `web` feature enabled:
   ```bash
   cd ../realtime_backend
   wasm-pack build --target web --release --no-default-features --features web
   ```
3. Copy the generated `pkg` folder into this directory so `index.html` can load
   `realtime_backend.js` and `realtime_backend_bg.wasm`.

## Running the Demo

Serve the `web_ui` directory with any static web server. One simple option is:

```bash
python -m http.server
```

Then open `http://localhost:8000` in your browser. Paste a track JSON object into
the text box and click **Start** to begin playback. Press **Stop** to halt the
engine.
