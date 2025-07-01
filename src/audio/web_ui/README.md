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
3. Copy the generated `pkg` folder into `public/` so Vite can serve
    `realtime_backend.js` and `realtime_backend_bg.wasm`.

## Running the Demo

Install the npm dependencies and start the development server. **Vite 5 requires Node.js 20 or newer**, so ensure you have an up-to-date Node.js installation:

```bash
npm install
npm run dev
```

Vite will serve the application at the printed URL. You can either paste a track
JSON object into the text box or use the **Upload** field to load a `.json`
file. Click **Start** to begin playback and **Stop** to halt the engine.
