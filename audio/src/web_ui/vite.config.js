import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
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
    rollupOptions: {
      input: {
        main: 'index.html',
        logo: 'logo.html'
      }
    }
  },
  // Optimize dependencies to prevent vite from reloading the page on wasm changes
  optimizeDeps: {
    exclude: ['/src/pkg/realtime_backend.js?import']
  }
});
