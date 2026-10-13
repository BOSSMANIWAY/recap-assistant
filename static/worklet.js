/* Захват микрофона: кадры int16 PCM (100 мс @ 16 кГц) + RMS для индикатора.
 * Кадры уходят бинарными ArrayBuffer'ами — в ~10 раз компактнее JSON,
 * ничего не теряется при сериализации (было: float→JSON→float). */
class CapturerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.frame = new Int16Array(1600); // 100 мс @ 16 кГц
    this.pos = 0;
    this.sum = 0;
  }
  process(inputs) {
    const ch = inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      const s = ch[i] < -1 ? -1 : ch[i] > 1 ? 1 : ch[i];
      this.sum += s * s;
      this.frame[this.pos++] = s < 0 ? s * 32768 : s * 32767;
      if (this.pos === this.frame.length) {
        const rms = Math.sqrt(this.sum / this.pos);
        const copy = this.frame.slice();
        this.port.postMessage({ b: copy.buffer, r: rms }, [copy.buffer]);
        this.pos = 0;
        this.sum = 0;
      }
    }
    return true;
  }
}
registerProcessor('capturer', CapturerProcessor);
