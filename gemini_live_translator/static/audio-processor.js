/**
 * AudioWorkletProcessor that captures microphone audio, batches it into
 * ~100 ms frames, converts Float32 [-1, 1] samples to little-endian PCM 16-bit,
 * and ships each chunk to the main thread via a transferable ArrayBuffer.
 *
 * Running this on the audio thread (instead of the deprecated
 * ScriptProcessorNode on the main thread) keeps capture glitch-free even when
 * the UI is busy.
 *
 * `sampleRate` here is a global injected by the AudioWorklet scope and equals
 * the rate of the owning AudioContext (we create that context at 16 kHz).
 */
class PCMCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = [];
    // 100 ms of audio at the context's sample rate.
    this._frameSize = Math.round(sampleRate * 0.1);
  }

  process(inputs) {
    const input = inputs[0];
    if (input && input[0]) {
      const channel = input[0];
      for (let i = 0; i < channel.length; i++) {
        this._buffer.push(channel[i]);
      }

      while (this._buffer.length >= this._frameSize) {
        const frame = this._buffer.splice(0, this._frameSize);
        const pcm16 = new Int16Array(frame.length);
        for (let i = 0; i < frame.length; i++) {
          let s = Math.max(-1, Math.min(1, frame[i]));
          pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        // Transfer ownership of the buffer to avoid a copy.
        this.port.postMessage(pcm16.buffer, [pcm16.buffer]);
      }
    }
    // Keep the processor alive.
    return true;
  }
}

registerProcessor("pcm-capture-processor", PCMCaptureProcessor);
