
// ringbuffer.js

export default class SharedRingBuffer {
  /**
   * @param {SharedArrayBuffer} indicesSab  Two-element Int32Array SAB storing [readIndex, writeIndex]
   * @param {SharedArrayBuffer} dataSab     Float32Array SAB for sample storage
   */
  constructor(indicesSab, dataSab) {
    // indicesSab: Int32Array([readPtr, writePtr])
    this.indices = new Int32Array(indicesSab);
    // buffer: Float32Array of size N
    this.buffer = new Float32Array(dataSab);
    this.size = this.buffer.length;
  }

  /**
   * How many floats can we write without overwriting unread data?
   * Always leaves one slot empty to distinguish full vs empty.
   * @returns {number}
   */
  availableWrite() {
    const r = Atomics.load(this.indices, 0);
    const w = Atomics.load(this.indices, 1);
    if (w >= r) {
      return this.size - (w - r) - 1;
    }
    return r - w - 1;
  }

  /**
   * How many floats are available to read?
   * @returns {number}
   */
  availableRead() {
    const r = Atomics.load(this.indices, 0);
    const w = Atomics.load(this.indices, 1);
    if (w >= r) {
      return w - r;
    }
    return this.size - (r - w);
  }

  /**
   * Push new data into the ring buffer.
   * Any excess beyond capacity is dropped.
   * @param {Float32Array} data
   */
  push(data) {
    let r = Atomics.load(this.indices, 0);
    let w = Atomics.load(this.indices, 1);
    for (let i = 0; i < data.length; i++) {
      const next = (w + 1) % this.size;
      if (next === r) {
        console.debug('RingBuffer full, dropping samples');
        break;
      }
      this.buffer[w] = data[i];
      w = next;
    }
    Atomics.store(this.indices, 1, w);
  }

  /**
   * Pop up to target.length floats into the provided array.
   * @param {Float32Array} target
   * @returns {number}  The actual number of floats written into target.
   */
  pop(target) {
    let r = Atomics.load(this.indices, 0);
    const w = Atomics.load(this.indices, 1);
    let count = 0;
    while (r !== w && count < target.length) {
      target[count++] = this.buffer[r];
      r = (r + 1) % this.size;
    }
    Atomics.store(this.indices, 0, r);
    if (count < target.length) {
      console.debug(
        'RingBuffer underflow: requested', target.length,
        'got', count
      );
    }
    return count;
  }
}

