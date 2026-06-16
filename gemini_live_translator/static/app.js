/**
 * Gemini Live Translator — browser client.
 *
 * Pipeline
 * --------
 *   Capture:  getUserMedia -> AudioContext(16 kHz) -> AudioWorklet
 *             -> PCM16 100 ms chunks -> WebSocket (binary, upstream)
 *
 *   Playback: WebSocket (binary, downstream, PCM16 @ 24 kHz)
 *             -> scheduled AudioBufferSourceNodes -> speakers
 *
 *   Text:     WebSocket (JSON) -> live transcript panel
 *
 * The capture and playback graphs use two separate AudioContexts because the
 * input rate (16 kHz, what Gemini expects) differs from the output rate
 * (24 kHz, what Gemini returns).
 */

const INPUT_SAMPLE_RATE = 16000;
const OUTPUT_SAMPLE_RATE = 24000;

class TranslatorClient {
  constructor() {
    this.ws = null;
    this.captureContext = null;
    this.playbackContext = null;
    this.workletNode = null;
    this.micSource = null;
    this.mediaStream = null;
    this.running = false;

    // Playback scheduling cursor (in playbackContext time).
    this.nextPlayTime = 0;

    this._bindUI();
  }

  _bindUI() {
    this.els = {
      toggleBtn: document.getElementById("toggleBtn"),
      toggleLabel: document.getElementById("toggleLabel"),
      toggleIcon: document.getElementById("toggleIcon"),
      statusDot: document.getElementById("statusDot"),
      statusText: document.getElementById("statusText"),
      transcript: document.getElementById("transcript"),
      placeholder: document.getElementById("placeholder"),
      clearBtn: document.getElementById("clearBtn"),
      voiceSelect: document.getElementById("voiceSelect"),
      modelInfo: document.getElementById("modelInfo"),
    };

    this.els.toggleBtn.addEventListener("click", () => {
      if (this.running) this.stop();
      else this.start();
    });

    this.els.clearBtn.addEventListener("click", () => {
      this.els.transcript.innerHTML = "";
      this._sourceLine = null;
      this._translationLine = null;
    });

    // Pre-fill model info from the health endpoint.
    fetch("/api/health")
      .then((r) => r.json())
      .then((d) => {
        this.els.modelInfo.textContent = d.model || "—";
        if (!d.api_key_configured) {
          this._setStatus("error", "API key not configured on server");
        }
      })
      .catch(() => {});
  }

  // --- Status helpers ------------------------------------------------------

  _setStatus(state, text) {
    const colors = {
      idle: "bg-slate-500",
      connecting: "bg-amber-400 pulse-dot",
      live: "bg-emerald-400 pulse-dot",
      error: "bg-rose-500",
    };
    this.els.statusDot.className = `h-2.5 w-2.5 rounded-full ${
      colors[state] || colors.idle
    }`;
    this.els.statusText.textContent = text;
  }

  // --- Session lifecycle ---------------------------------------------------

  async start() {
    try {
      this._setStatus("connecting", "Connecting…");
      this.els.toggleBtn.disabled = true;

      await this._openSocket();
      await this._startCapture();
      this._initPlayback();

      this.running = true;
      this.els.toggleLabel.textContent = "Stop";
      this.els.toggleIcon.textContent = "⏹️";
      this.els.toggleBtn.disabled = false;
      this._setStatus("live", "Live — speak now");
    } catch (err) {
      console.error(err);
      this._setStatus("error", err.message || "Failed to start");
      this.els.toggleBtn.disabled = false;
      await this._teardown();
    }
  }

  async stop() {
    this.running = false;
    // Tell the server this turn is finished, then close down audio.
    this._sendControl({ action: "end" });
    await this._teardown();

    this.els.toggleLabel.textContent = "Start Translating";
    this.els.toggleIcon.textContent = "🎙️";
    this._setStatus("idle", "Idle");
  }

  async _teardown() {
    if (this.workletNode) {
      this.workletNode.port.onmessage = null;
      this.workletNode.disconnect();
      this.workletNode = null;
    }
    if (this.micSource) {
      this.micSource.disconnect();
      this.micSource = null;
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((t) => t.stop());
      this.mediaStream = null;
    }
    if (this.captureContext) {
      await this.captureContext.close().catch(() => {});
      this.captureContext = null;
    }
    if (this.playbackContext) {
      await this.playbackContext.close().catch(() => {});
      this.playbackContext = null;
    }
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.close();
    }
    this.ws = null;
  }

  // --- WebSocket -----------------------------------------------------------

  _openSocket() {
    return new Promise((resolve, reject) => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${proto}://${location.host}/api/stream`);
      ws.binaryType = "arraybuffer";

      ws.onopen = () => {
        // Send chosen voice as an optional config message.
        ws.send(
          JSON.stringify({
            action: "config",
            voice: this.els.voiceSelect.value,
          })
        );
        resolve();
      };
      ws.onerror = () => reject(new Error("WebSocket connection failed"));
      ws.onclose = () => {
        if (this.running) {
          this.running = false;
          this._setStatus("idle", "Disconnected");
          this.els.toggleLabel.textContent = "Start Translating";
          this.els.toggleIcon.textContent = "🎙️";
          this._teardown();
        }
      };
      ws.onmessage = (ev) => this._onMessage(ev);

      this.ws = ws;
    });
  }

  _sendControl(obj) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(obj));
    }
  }

  _onMessage(ev) {
    if (ev.data instanceof ArrayBuffer) {
      this._enqueueAudio(ev.data);
      return;
    }
    let msg;
    try {
      msg = JSON.parse(ev.data);
    } catch {
      return;
    }

    switch (msg.type) {
      case "status":
        if (msg.model) this.els.modelInfo.textContent = msg.model;
        break;
      case "transcript":
        this._appendTranscript(msg.role, msg.text);
        break;
      case "turn_complete":
        this._sourceLine = null;
        this._translationLine = null;
        break;
      case "interrupted":
        // User barged in — drop any queued audio so we don't talk over them.
        this._flushPlayback();
        break;
      case "error":
        this._setStatus("error", msg.message || "Server error");
        break;
    }
  }

  // --- Capture (mic -> server) ---------------------------------------------

  async _startCapture() {
    this.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    this.captureContext = new (window.AudioContext || window.webkitAudioContext)(
      { sampleRate: INPUT_SAMPLE_RATE }
    );
    await this.captureContext.audioWorklet.addModule("/static/audio-processor.js");

    this.micSource = this.captureContext.createMediaStreamSource(this.mediaStream);
    this.workletNode = new AudioWorkletNode(
      this.captureContext,
      "pcm-capture-processor"
    );

    // Each message is a transferable ArrayBuffer of PCM16 samples.
    this.workletNode.port.onmessage = (e) => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(e.data);
      }
    };

    this.micSource.connect(this.workletNode);
    // Worklet doesn't need to reach the speakers; connecting to a muted gain
    // keeps the graph "pulled" across browsers without echoing the mic.
    const sink = this.captureContext.createGain();
    sink.gain.value = 0;
    this.workletNode.connect(sink);
    sink.connect(this.captureContext.destination);
  }

  // --- Playback (server -> speaker) ----------------------------------------

  _initPlayback() {
    this.playbackContext = new (window.AudioContext || window.webkitAudioContext)(
      { sampleRate: OUTPUT_SAMPLE_RATE }
    );
    this.nextPlayTime = 0;
  }

  _enqueueAudio(arrayBuffer) {
    if (!this.playbackContext) return;

    // Incoming bytes are little-endian PCM16; convert to Float32 [-1, 1].
    const pcm16 = new Int16Array(arrayBuffer);
    const float32 = new Float32Array(pcm16.length);
    for (let i = 0; i < pcm16.length; i++) {
      float32[i] = pcm16[i] / 0x8000;
    }

    const buffer = this.playbackContext.createBuffer(
      1,
      float32.length,
      OUTPUT_SAMPLE_RATE
    );
    buffer.copyToChannel(float32, 0);

    const source = this.playbackContext.createBufferSource();
    source.buffer = buffer;
    source.connect(this.playbackContext.destination);

    // Schedule sequentially so chunks play gaplessly. If we've fallen behind,
    // resync to "now" plus a small safety margin.
    const now = this.playbackContext.currentTime;
    if (this.nextPlayTime < now) {
      this.nextPlayTime = now + 0.05;
    }
    source.start(this.nextPlayTime);
    this.nextPlayTime += buffer.duration;
  }

  _flushPlayback() {
    // Rebuild the playback context to instantly cut queued audio.
    if (this.playbackContext) {
      this.playbackContext.close().catch(() => {});
    }
    this._initPlayback();
  }

  // --- Transcript UI -------------------------------------------------------

  _appendTranscript(role, text) {
    if (!text) return;
    if (this.els.placeholder) {
      this.els.placeholder.remove();
      this.els.placeholder = null;
    }

    const isTranslation = role === "translation";
    const lineKey = isTranslation ? "_translationLine" : "_sourceLine";

    // Stream incremental tokens into the same bubble until the turn completes.
    if (!this[lineKey]) {
      const wrap = document.createElement("div");
      wrap.className = isTranslation ? "flex justify-end" : "flex justify-start";

      const bubble = document.createElement("div");
      bubble.className = isTranslation
        ? "max-w-[80%] rounded-2xl rounded-tr-sm bg-indigo-600/90 px-4 py-2.5 text-sm text-white"
        : "max-w-[80%] rounded-2xl rounded-tl-sm bg-slate-800 px-4 py-2.5 text-sm text-slate-200";

      const label = document.createElement("div");
      label.className = "mb-0.5 text-[10px] font-semibold uppercase tracking-wide opacity-60";
      label.textContent = isTranslation ? "Translation" : "Source";

      const body = document.createElement("span");
      bubble.appendChild(label);
      bubble.appendChild(body);
      wrap.appendChild(bubble);
      this.els.transcript.appendChild(wrap);
      this[lineKey] = body;
    }

    this[lineKey].textContent += text;
    this.els.transcript.scrollTop = this.els.transcript.scrollHeight;
  }
}

window.addEventListener("DOMContentLoaded", () => {
  if (!navigator.mediaDevices || !window.AudioContext) {
    alert("This browser does not support the Web Audio API required for the translator.");
    return;
  }
  window.translator = new TranslatorClient();
});
