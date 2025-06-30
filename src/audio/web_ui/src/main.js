import init, { start_stream, process_block, stop_stream } from '/pkg/realtime_backend.js';

let audioCtx = null;
let scriptNode = null;
let wasmLoaded = false;

async function ensureWasmLoaded() {
  if (!wasmLoaded) {
    await init();
    wasmLoaded = true;
  }
}

function setupAudio() {
  const bufferSize = 1024;
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  scriptNode = audioCtx.createScriptProcessor(bufferSize, 0, 2);
  scriptNode.onaudioprocess = (e) => {
    const frames = e.outputBuffer.length;
    const data = process_block(frames * 2);
    const left = e.outputBuffer.getChannelData(0);
    const right = e.outputBuffer.getChannelData(1);
    for (let i = 0; i < frames; i++) {
      left[i] = data[i * 2];
      right[i] = data[i * 2 + 1];
    }
  };
  scriptNode.connect(audioCtx.destination);
}

export async function start() {
  await ensureWasmLoaded();
  const trackJson = document.getElementById('track-json').value;
  start_stream(trackJson);
  setupAudio();
}

export function stop() {
  stop_stream();
  if (scriptNode) {
    scriptNode.disconnect();
    scriptNode = null;
  }
  if (audioCtx) {
    audioCtx.close();
    audioCtx = null;
  }
}

document.getElementById('start').addEventListener('click', start);
document.getElementById('stop').addEventListener('click', stop);
