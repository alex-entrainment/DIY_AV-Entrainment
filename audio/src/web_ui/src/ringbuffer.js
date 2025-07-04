// Copyright (c) DIY Audio-Visual Entrainment
// SPDX-License-Identifier: MIT

/**
 * Shared ring buffer for passing Float32 samples between threads.
 * The buffer uses a SharedArrayBuffer so the main thread and
 * AudioWorklet thread can access the same memory concurrently.
 */
export default class SharedRingBuffer {
  /**
   * @param {SharedArrayBuffer} indexSab  Two Int32 values for read/write indices
   * @param {SharedArrayBuffer} dataSab   Float32 sample storage
   */
  constructor(indexSab, dataSab) {
    this._index = new Int32Array(indexSab);
    this._data = new Float32Array(dataSab);
    this._capacity = this._data.length;
  }

  /** Reset the buffer to an empty state. */
  reset() {
    Atomics.store(this._index, 0, 0);
    Atomics.store(this._index, 1, 0);
  }

  /** Number of samples ready to be read. */
  availableRead() {
    const r = Atomics.load(this._index, 0);
    const w = Atomics.load(this._index, 1);
    return w >= r ? w - r : this._capacity - (r - w);
  }

  /** Remaining space for writing additional samples. */
  availableWrite() {
    const r = Atomics.load(this._index, 0);
    const w = Atomics.load(this._index, 1);
    return r > w ? r - w - 1 : this._capacity - (w - r) - 1;
  }

  /** Returns true if no samples are available to read. */
  isEmpty() {
    return Atomics.load(this._index, 0) === Atomics.load(this._index, 1);
  }

  /** Returns true if the buffer has no free slots for writing. */
  isFull() {
    return this.availableWrite() === 0;
  }

  /** Alias for reset(). */
  clear() {
    this.reset();
  }

  /**
   * Push an array of samples into the buffer.
   * @param {Float32Array|number[]} samples
   * @returns {number} Count of samples actually written
   */
  push(samples) {
    let r = Atomics.load(this._index, 0);
    let w = Atomics.load(this._index, 1);
    let written = 0;
    const cap = this._capacity;

    for (let i = 0; i < samples.length; i++) {
      const next = (w + 1) % cap;
      if (next === r) break; // buffer full
      this._data[w] = samples[i];
      w = next;
      written++;
    }

    Atomics.store(this._index, 1, w);
    return written;
  }

  /**
   * Pop up to dest.length samples from the buffer.
   * @param {Float32Array} dest
   * @returns {number} Number of samples read
   */
  pop(dest) {
    let r = Atomics.load(this._index, 0);
    let w = Atomics.load(this._index, 1);
    let read = 0;
    const cap = this._capacity;

    while (read < dest.length && r !== w) {
      dest[read] = this._data[r];
      r = (r + 1) % cap;
      read++;
    }

    Atomics.store(this._index, 0, r);
    return read;
  }
}
