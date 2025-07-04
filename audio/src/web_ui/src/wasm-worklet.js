import SharedRingBuffer from './ringbuffer.js';

class WasmWorklet extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const { indices, buffer } = options.processorOptions;
    this.ring = new SharedRingBuffer(indices, buffer);
    this.temp = new Float32Array(128 * 2);
    console.debug('WasmWorklet initialized');
  }

  process(inputs, outputs) {
    const output = outputs[0];
    const left = output[0];
    const right = output[1];
    const frames = left.length;
    const needed = frames * 2;
    if (this.temp.length < needed) {
      this.temp = new Float32Array(needed);
    }
    const available = this.ring.pop(this.temp.subarray(0, needed));
    const readFrames = available / 2;
    if (readFrames < frames) {
      console.debug('AudioWorklet underflow: expected', frames, 'got', readFrames);
    }
    for (let i = 0; i < frames; i++) {
      if (i < readFrames) {
        left[i] = this.temp[i * 2];
        right[i] = this.temp[i * 2 + 1];
      } else {
        left[i] = 0;
        right[i] = 0;
      }
    }
    return true;
  }
}

registerProcessor('wasm-worklet', WasmWorklet);
