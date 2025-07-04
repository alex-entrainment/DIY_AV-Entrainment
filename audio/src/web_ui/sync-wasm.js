import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { cpSync, rmSync, existsSync } from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const srcDir = resolve(__dirname, '../realtime_backend/pkg');
const destDir = resolve(__dirname, 'src/pkg');

if (!existsSync(srcDir)) {
  console.error(`WASM package not found at ${srcDir}. Build it with wasm-pack first.`);
  process.exit(1);
}

rmSync(destDir, { recursive: true, force: true });
cpSync(srcDir, destDir, { recursive: true });
console.log(`Copied WASM package from ${srcDir} to ${destDir}`);
