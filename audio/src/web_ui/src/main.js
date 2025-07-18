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
  push_clip_samples,
} from '/src/pkg/realtime_backend.js?import';
import SharedRingBuffer from './ringbuffer.js';

let audioCtx = null;
let workletNode = null;
let ringBuffer = null;
let fillTimer = null;
let wasmLoaded = false;
console.debug('Web UI script loaded');
let statusTimer = null;

// Keep track of files uploaded via the UI so they can be re-selected
const uploadedTracks = {};
const uploadedNoises = {};
const uploadedClips = {};

async function populateSelect(id, url) {
  try {
    const list = await fetch(url).then(r => r.json());
    const select = document.getElementById(id);
    if (select && Array.isArray(list)) {
      for (const name of list) {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        select.appendChild(opt);
      }
    }
  } catch (err) {
    console.warn('Failed to populate', id, err);
  }
}

function initSelects() {
  populateSelect('track-select', '/tracks/index.json');
  populateSelect('noise-select', '/noise/index.json');
  populateSelect('clip-select', '/clips/index.json');
}

async function streamOverlayClip(index, path) {
  try {
    let arrayBuf;
    if (path.startsWith('data:')) {
      const comma = path.indexOf(',');
      arrayBuf = Uint8Array.from(atob(path.slice(comma + 1)), c => c.charCodeAt(0)).buffer;
    } else {
      const resp = await fetch(path);
      arrayBuf = await resp.arrayBuffer();
    }
    const audioBuf = await audioCtx.decodeAudioData(arrayBuf);
    const l = audioBuf.getChannelData(0);
    const r = audioBuf.numberOfChannels > 1 ? audioBuf.getChannelData(1) : l;
    const total = audioBuf.length;
    const chunkFrames = 2048;
    for (let pos = 0; pos < total; pos += chunkFrames) {
      const frames = Math.min(chunkFrames, total - pos);
      const chunk = new Float32Array(frames * 2);
      for (let i = 0; i < frames; i++) {
        chunk[i * 2] = l[pos + i];
        chunk[i * 2 + 1] = r[pos + i];
      }
      push_clip_samples(index, chunk, pos + frames >= total);
      await new Promise(res => setTimeout(res, 0));
    }
  } catch (err) {
    console.error('Failed to stream clip', path, err);
  }
}

async function ensureWasmLoaded() {
  if (!wasmLoaded) {
    console.debug('Loading WASM module');
    await init();
    wasmLoaded = true;
    console.debug('WASM module initialized');
  }
}

async function setupAudio(sampleRate) {
  await init();

  // 1️⃣ Number of channels in the worklet:
  const channels = 2;

  // 2️⃣ How many frames of look-ahead buffering you want:
  const bufferFrames = 16384;  // e.g. ~0.37s at 44.1 kHz

  // 3️⃣ Allocate the SharedArrayBuffer for ALL floats (frames × channels):
  const sabBuf = new SharedArrayBuffer(
    bufferFrames * channels * Float32Array.BYTES_PER_ELEMENT
  );
  const sabIdx = new SharedArrayBuffer(2 * Int32Array.BYTES_PER_ELEMENT);
  ringBuffer = new SharedRingBuffer(sabIdx, sabBuf);

  // 4️⃣ Create the AudioContext & worklet:
  audioCtx = new AudioContext({ sampleRate });
  await audioCtx.audioWorklet.addModule(
    new URL('./wasm-worklet.js', import.meta.url)
  );
  workletNode = new AudioWorkletNode(audioCtx, 'wasm-worklet', {
    outputChannelCount: [2],       // force stereo
    channelCount: 2,
    channelCountMode: 'explicit',
    processorOptions: { indices: sabIdx, buffer: sabBuf },
  });
  workletNode.connect(audioCtx.destination);

  // 5️⃣ Decide on your “chunk” size in **frames**:
  const fillFrames = 512;                // how many frames per fill
  const samplesPerFill = fillFrames * channels; // floats per fill

  // 6️⃣ INITIAL FILL — fill the buffer completely once before starting:
  while (ringBuffer.availableWrite() >= samplesPerFill) {
    const data = process_block(fillFrames);  // returns exactly fillFrames×channels floats
    ringBuffer.push(data);
  }

  // 7️⃣ REFILL LOOP — keeps you topped up in small slices:
  // Use setInterval at ~5 ms so you stay ahead of the 128-frame (~2.9 ms) worklet callbacks.
  fillTimer = setInterval(() => {
    // ringBuffer may be null if playback was stopped but an interval
    // callback is still queued. Guard against accessing properties of
    // a cleared buffer to avoid console errors.
    if (!ringBuffer) {
      return;
    }

    let free = ringBuffer.availableWrite();
    // Write as many small, correctly‐sized chunks as will fit right now:
    while (free >= samplesPerFill) {
      const data = process_block(fillFrames);
      ringBuffer.push(data);
      free -= samplesPerFill;
    }
  }, 5);

  console.log('🎧 Audio setup complete — buffer primed and refill running.');
}
export async function start() {
  await ensureWasmLoaded();
  const trackJson = document.getElementById('track-json').value;
  const startTime = parseFloat(document.getElementById('start-time').value) || 0;
  let sampleRate = 44100;
  let trackObj = null;
  try {
    trackObj = JSON.parse(trackJson);
    if (trackObj.global && trackObj.global.sample_rate) {
      sampleRate = trackObj.global.sample_rate;
    } else if (
      trackObj.global_settings &&
      trackObj.global_settings.sample_rate
    ) {
      sampleRate = trackObj.global_settings.sample_rate;
    } else if (trackObj.sample_rate) {
      sampleRate = trackObj.sample_rate;
    }
  } catch (e) {
    console.warn('Unable to parse track JSON for sample rate:', e);
  }

  await setupAudio(sampleRate);
  console.debug('Starting stream with sampleRate', sampleRate, 'startTime', startTime);
  start_stream(trackJson, sampleRate, startTime);
  console.debug('Stream started');

  if (trackObj && Array.isArray(trackObj.overlay_clips)) {
    trackObj.overlay_clips.forEach((clip, idx) => {
      if (clip.file_path) {
        streamOverlayClip(idx, clip.file_path);
      }
    });
  }

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
    clearInterval(fillTimer);
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

export function handleJsonUpload(event) {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById('track-json').value = e.target.result;
    uploadedTracks[file.name] = e.target.result;
    const select = document.getElementById('track-select');
    if (select && !select.querySelector(`option[value="${file.name}"]`)) {
      const opt = new Option(file.name, file.name);
      opt.dataset.uploaded = 'true';
      select.appendChild(opt);
      select.value = file.name;
    }
  };
  reader.readAsText(file);
}

export function handleNoiseUpload(event) {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const params = JSON.parse(e.target.result);
      const textarea = document.getElementById('track-json');
      let track;
      try {
        track = JSON.parse(textarea.value);
      } catch (err) {
        console.warn('Invalid track JSON, resetting');
        track = { global: { sample_rate: 44100 }, progression: [], background_noise: {}, overlay_clips: [] };
      }
      if (!track.background_noise) {
        track.background_noise = {};
      }
      track.background_noise.params = params;
      if (track.background_noise.amp === undefined) {
        track.background_noise.amp = 1.0;
      }
      textarea.value = JSON.stringify(track, null, 2);
      uploadedNoises[file.name] = e.target.result;
      const select = document.getElementById('noise-select');
      if (select && !select.querySelector(`option[value="${file.name}"]`)) {
        const opt = new Option(file.name, file.name);
        opt.dataset.uploaded = 'true';
        select.appendChild(opt);
        select.value = file.name;
      }
    } catch (err) {
      console.error('Failed to parse .noise file', err);
    }
  };
  reader.readAsText(file);
}

export function handleClipUpload(event) {
  const files = Array.from(event.target.files || []);
  if (!files.length) return;
  const textarea = document.getElementById('track-json');
  let track;
  try {
    track = JSON.parse(textarea.value);
  } catch (err) {
    console.warn('Invalid track JSON, resetting');
    track = { global: { sample_rate: 44100 }, progression: [], background_noise: {}, overlay_clips: [] };
  }
  if (!Array.isArray(track.overlay_clips)) {
    track.overlay_clips = [];
  }
  const readers = files.map(
    (f) =>
      new Promise((resolve) => {
        const r = new FileReader();
        r.onload = () => {
          track.overlay_clips.push({ file_path: r.result, start: 0, amp: 1.0 });
          uploadedClips[f.name] = r.result;
          const select = document.getElementById('clip-select');
          if (select && !select.querySelector(`option[value="${f.name}"]`)) {
            const opt = new Option(f.name, f.name);
            opt.dataset.uploaded = 'true';
            select.appendChild(opt);
            select.selectedIndex = select.options.length - 1;
          }
          resolve();
        };
        r.readAsDataURL(f);
      })
  );
  Promise.all(readers).then(() => {
    textarea.value = JSON.stringify(track, null, 2);
  });
}


async function loadTrackFromServer() {
  const select = document.getElementById('track-select');
  const file = select && select.value;
  if (!file) return;
  const opt = select.selectedOptions[0];
  if (opt && opt.dataset.uploaded === 'true') {
    const text = uploadedTracks[file];
    if (text) {
      document.getElementById('track-json').value = text;
    }
    return;
  }
  try {
    const text = await fetch(`/tracks/${file}`).then(r => r.text());
    document.getElementById('track-json').value = text;
  } catch (err) {
    console.error('Failed to load track', err);
  }
}

async function loadNoiseFromServer() {
  const select = document.getElementById('noise-select');
  const file = select && select.value;
  if (!file) return;
  const opt = select.selectedOptions[0];
  if (opt && opt.dataset.uploaded === 'true') {
    try {
      const params = JSON.parse(uploadedNoises[file]);
      const textarea = document.getElementById('track-json');
      let track;
      try {
        track = JSON.parse(textarea.value);
      } catch (_) {
        track = { global: { sample_rate: 44100 }, progression: [], background_noise: {}, overlay_clips: [] };
      }
      if (!track.background_noise) track.background_noise = {};
      track.background_noise.params = params;
      if (track.background_noise.amp === undefined) {
        track.background_noise.amp = 1.0;
      }
      textarea.value = JSON.stringify(track, null, 2);
    } catch (err) {
      console.error('Failed to parse uploaded noise data', err);
    }
    return;
  }
  try {
    const params = await fetch(`/noise/${file}`).then(r => r.json());
    const textarea = document.getElementById('track-json');
    let track;
    try {
      track = JSON.parse(textarea.value);
    } catch (_) {
      track = { global: { sample_rate: 44100 }, progression: [], background_noise: {}, overlay_clips: [] };
    }
    if (!track.background_noise) track.background_noise = {};
    track.background_noise.params = params;
    if (track.background_noise.amp === undefined) {
      track.background_noise.amp = 1.0;
    }
    textarea.value = JSON.stringify(track, null, 2);
  } catch (err) {
    console.error('Failed to load noise file', err);
  }
}

async function addClipFromServer() {
  const select = document.getElementById('clip-select');
  const files = Array.from(select ? select.selectedOptions : []).map(o => o.value);
  if (!files.length) return;
  const textarea = document.getElementById('track-json');
  let track;
  try {
    track = JSON.parse(textarea.value);
  } catch (_) {
    track = { global: { sample_rate: 44100 }, progression: [], background_noise: {}, overlay_clips: [] };
  }
  if (!Array.isArray(track.overlay_clips)) track.overlay_clips = [];
  const promises = files.map(f => {
    const opt = select.querySelector(`option[value="${f}"]`);
    if (opt && opt.dataset.uploaded === 'true') {
      return Promise.resolve(uploadedClips[f]);
    }
    return fetch(`/clips/${f}`)
      .then(r => r.blob())
      .then(
        b =>
          new Promise(res => {
            const r = new FileReader();
            r.onload = () => res(r.result);
            r.readAsDataURL(b);
          })
      );
  });
  const urls = await Promise.all(promises);
  for (let i = 0; i < urls.length; i++) {
    const u = urls[i];
    track.overlay_clips.push({ file_path: u, start: 0, amp: 1.0 });
  }
  textarea.value = JSON.stringify(track, null, 2);
}

export {
  initSelects,
  loadTrackFromServer,
  loadNoiseFromServer,
  addClipFromServer,
};
