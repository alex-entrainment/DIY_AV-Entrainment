import init, { start_stream, process_block, stop_stream } from '/pkg/realtime_backend.js';
import SharedRingBuffer from './ringbuffer.js';

let audioCtx = null;
let workletNode = null;
let ringBuffer = null;
let fillTimer = null;
let wasmLoaded = false;

async function ensureWasmLoaded() {
  if (!wasmLoaded) {
    await init();
    wasmLoaded = true;
  }
}

function setupAudio(sampleRate) {
  const bufferFrames = 16384;
  audioCtx = new (window.AudioContext || window.webkitAudioContext)({
    sampleRate,
  });
  const sabBuf = new SharedArrayBuffer(bufferFrames * Float32Array.BYTES_PER_ELEMENT);
  const sabIdx = new SharedArrayBuffer(8);
  ringBuffer = new SharedRingBuffer(sabIdx, sabBuf);

  return audioCtx.audioWorklet.addModule('/src/wasm-worklet.js').then(() => {
    workletNode = new AudioWorkletNode(audioCtx, 'wasm-worklet', {
      processorOptions: { indices: sabIdx, buffer: sabBuf },
    });
    workletNode.connect(audioCtx.destination);

    const fillBlock = 512;
    const fill = () => {
      if (!workletNode) return;
      while (ringBuffer.availableWrite() >= fillBlock * 2) {
        const data = process_block(fillBlock * 2);
        ringBuffer.push(data);
      }
      fillTimer = setTimeout(fill, 10);
    };
    fill();
  });
}

export async function start() {
  await ensureWasmLoaded();
  const trackJson = document.getElementById('track-json').value;
  let sampleRate = 44100;
  try {
    const trackObj = JSON.parse(trackJson);
    if (trackObj.global && trackObj.global.sample_rate) {
      sampleRate = trackObj.global.sample_rate;
    } else if (trackObj.global_settings && trackObj.global_settings.sample_rate) {
      sampleRate = trackObj.global_settings.sample_rate;
    } else if (trackObj.sample_rate) {
      sampleRate = trackObj.sample_rate;
    }
  } catch (e) {
    console.warn('Unable to parse track JSON for sample rate:', e);
  }
  start_stream(trackJson, sampleRate);
  await setupAudio(sampleRate);
}

export function stop() {
  stop_stream();
  if (workletNode) {
    workletNode.disconnect();
    workletNode = null;
  }
  if (audioCtx) {
    audioCtx.close();
    audioCtx = null;
  }
  if (fillTimer) {
    clearTimeout(fillTimer);
    fillTimer = null;
  }
  ringBuffer = null;
}

document.getElementById('start').addEventListener('click', start);
document.getElementById('stop').addEventListener('click', stop);

document.getElementById('json-upload').addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById('track-json').value = e.target.result;
  };
  reader.readAsText(file);
});
