import init, {
  start_stream,
  process_block,
  stop_stream,
  pause_stream,
  resume_stream,
  start_from,
  update_track,
  enable_gpu,
  current_step,
  elapsed_samples,
} from '/src/pkg/realtime_backend.js?import';
import SharedRingBuffer from './ringbuffer.js';

let audioCtx = null;
let workletNode = null;
let ringBuffer = null;
let fillTimer = null;
let wasmLoaded = false;
console.debug('Web UI script loaded');
let statusTimer = null;

async function ensureWasmLoaded() {
  if (!wasmLoaded) {
    console.debug('Loading WASM module');
    await init();
    wasmLoaded = true;
    console.debug('WASM module initialized');
  }
}

function setupAudio(sampleRate) {
  const bufferFrames = 16384;
  audioCtx = new (window.AudioContext || window.webkitAudioContext)({
    sampleRate,
  });
  console.debug('AudioContext created with sampleRate', sampleRate);
  const sabBuf = new SharedArrayBuffer(bufferFrames * Float32Array.BYTES_PER_ELEMENT);
  const sabIdx = new SharedArrayBuffer(8);
  ringBuffer = new SharedRingBuffer(sabIdx, sabBuf);
  console.debug('SharedRingBuffer initialized with', bufferFrames, 'frames');
  return audioCtx.audioWorklet.addModule('/src/wasm-worklet.js').then(() => {
    workletNode = new AudioWorkletNode(audioCtx, 'wasm-worklet', {
      processorOptions: { indices: sabIdx, buffer: sabBuf },
    });
    workletNode.connect(audioCtx.destination);
    console.debug('AudioWorkletNode added and connected');

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
    console.debug('Started ring buffer fill loop');
  });
}

export async function start() {
  await ensureWasmLoaded();
  const trackJson = document.getElementById('track-json').value;
  const startTime = parseFloat(document.getElementById('start-time').value) || 0;
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
  console.debug('Starting stream with sampleRate', sampleRate, 'startTime', startTime);
  start_stream(trackJson, sampleRate, startTime);
  console.debug('Stream started');
  await setupAudio(sampleRate);

  if (audioCtx.state === 'suspended') {
    await audioCtx.resume();
  }

  console.debug('Audio setup complete');
  startStatusUpdates();
}

export function stop() {
  console.debug('Stopping stream');
  stop_stream();
  stopStatusUpdates();
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

export function pause() {
  console.debug('Pausing stream');
  pause_stream();
}

export function resume() {
  console.debug('Resuming stream');
  resume_stream();
}

export function seek() {
  const pos = parseFloat(document.getElementById('seek-time').value);
  if (!isNaN(pos)) {
    console.debug('Seeking to', pos, 'seconds');
    start_from(pos);
  }
}

export function sendUpdate() {
  const trackJson = document.getElementById('track-json').value;
  console.debug('Sending track update');
  update_track(trackJson);
}

export function toggleGpu(event) {
  console.debug('GPU toggle', event.target.checked);
  enable_gpu(event.target.checked);
}

function startStatusUpdates() {
  console.debug('Starting status updates');
  statusTimer = setInterval(() => {
    if (workletNode) {
      document.getElementById('current-step').textContent = current_step();
      document.getElementById('elapsed-samples').textContent = elapsed_samples();
    }
  }, 500);
}

function stopStatusUpdates() {
  if (statusTimer) {
    clearInterval(statusTimer);
    statusTimer = null;
    console.debug('Stopped status updates');
  }
}

document.getElementById('start').addEventListener('click', start);
document.getElementById('stop').addEventListener('click', stop);
document.getElementById('pause').addEventListener('click', pause);
document.getElementById('resume').addEventListener('click', resume);
document.getElementById('seek-button').addEventListener('click', seek);
document.getElementById('update-track').addEventListener('click', sendUpdate);
document.getElementById('gpu-enable').addEventListener('change', toggleGpu);

document.getElementById('json-upload').addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById('track-json').value = e.target.result;
  };
  reader.readAsText(file);
});
