import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    open: true,
    // Add these headers to enable SharedArrayBuffer
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  build: {
    outDir: 'dist',
  },
  // Optimize dependencies to prevent vite from reloading the page on wasm changes
  optimizeDeps: {
    exclude: ['/src/pkg/realtime_backend.js?import']
  }
});
