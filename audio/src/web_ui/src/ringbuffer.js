class SharedRingBuffer {
  constructor(indicesSab, dataSab) {
    this.indices = new Int32Array(indicesSab);
    this.buffer = new Float32Array(dataSab);
    this.size = this.buffer.length;
  }

  availableWrite() {
    const r = Atomics.load(this.indices, 0);
    const w = Atomics.load(this.indices, 1);
    if (w >= r) {
      return this.size - (w - r) - 1;
    }
    return r - w - 1;
  }

  availableRead() {
    const r = Atomics.load(this.indices, 0);
    const w = Atomics.load(this.indices, 1);
    if (w >= r) {
      return w - r;
    }
    return this.size - (r - w);
  }

  push(data) {
    let r = Atomics.load(this.indices, 0);
    let w = Atomics.load(this.indices, 1);
    for (let i = 0; i < data.length; i++) {
      const next = (w + 1) % this.size;
      if (next === r) {
        console.debug('RingBuffer full, dropping samples');
        break; // full
      }
      this.buffer[w] = data[i];
      w = next;
    }
    Atomics.store(this.indices, 1, w);
  }

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
      console.debug('RingBuffer underflow: requested', target.length, 'got', count);
    }
    return count;
  }
}

export default SharedRingBuffer;
