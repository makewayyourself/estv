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

// Supported interpreting languages (code -> display label). Keep in sync with
// SUPPORTED_LANGUAGES in services/gemini_live.py.
const LANGUAGES = {
  ko: "한국어 (Korean)",
  en: "영어 (English)",
  ja: "일본어 (Japanese)",
  zh: "중국어 (Chinese)",
  fr: "프랑스어 (French)",
  es: "스페인어 (Spanish)",
  ar: "아랍어 (Arabic)",
  ru: "러시아어 (Russian)",
};
const DEFAULT_LANG_A = "ko";
const DEFAULT_LANG_B = "en";

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

    // Meeting notes: persisted log of finalized turns + in-progress accumulators.
    this.meetingLog = [];
    this._curSource = "";
    this._curTranslation = "";

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
      serverUrl: document.getElementById("serverUrl"),
      accessToken: document.getElementById("accessToken"),
      saveServerBtn: document.getElementById("saveServerBtn"),
      langA: document.getElementById("langA"),
      langB: document.getElementById("langB"),
      // Meeting notes
      notes: document.getElementById("notes"),
      notesPlaceholder: document.getElementById("notesPlaceholder"),
      noteCount: document.getElementById("noteCount"),
      summaryLang: document.getElementById("summaryLang"),
      summarizeBtn: document.getElementById("summarizeBtn"),
      exportBtn: document.getElementById("exportBtn"),
      newMeetingBtn: document.getElementById("newMeetingBtn"),
      summaryBox: document.getElementById("summaryBox"),
      summaryContent: document.getElementById("summaryContent"),
    };

    this._setupLanguages();
    this._setupMeetingNotes();

    this.els.toggleBtn.addEventListener("click", () => {
      if (this.running) this.stop();
      else this.start();
    });

    this.els.clearBtn.addEventListener("click", () => {
      this.els.transcript.innerHTML = "";
      this._sourceLine = null;
      this._translationLine = null;
    });

    // Server URL + access token: remembered on the device, pre-filled from the
    // saved value or the build-time defaults in config.js.
    this.els.serverUrl.value = this._serverBase();
    this.els.accessToken.value = this._accessToken();
    this.els.saveServerBtn.addEventListener("click", () => {
      const v = this.els.serverUrl.value.trim().replace(/\/+$/, "");
      if (v) localStorage.setItem("serverUrl", v);
      else localStorage.removeItem("serverUrl");

      const tok = this.els.accessToken.value.trim();
      if (tok) localStorage.setItem("accessToken", tok);
      else localStorage.removeItem("accessToken");

      this._setStatus("idle", "Settings saved");
      this._refreshHealth();
    });

    this._refreshHealth();
  }

  /** Populate the two language dropdowns and restore the saved pair. */
  _setupLanguages() {
    const fill = (sel, selected) => {
      sel.innerHTML = "";
      for (const [code, label] of Object.entries(LANGUAGES)) {
        const opt = document.createElement("option");
        opt.value = code;
        opt.textContent = label;
        if (code === selected) opt.selected = true;
        sel.appendChild(opt);
      }
    };
    fill(this.els.langA, localStorage.getItem("langA") || DEFAULT_LANG_A);
    fill(this.els.langB, localStorage.getItem("langB") || DEFAULT_LANG_B);

    const remember = () => {
      localStorage.setItem("langA", this.els.langA.value);
      localStorage.setItem("langB", this.els.langB.value);
    };
    this.els.langA.addEventListener("change", remember);
    this.els.langB.addEventListener("change", remember);
  }

  /** Set up the meeting-notes panel: restore saved log, wire the buttons. */
  _setupMeetingNotes() {
    // Summary language dropdown mirrors the supported languages.
    const sel = this.els.summaryLang;
    sel.innerHTML = "";
    for (const [code, label] of Object.entries(LANGUAGES)) {
      const opt = document.createElement("option");
      opt.value = code;
      opt.textContent = label;
      sel.appendChild(opt);
    }
    sel.value = localStorage.getItem("summaryLang") || localStorage.getItem("langA") || DEFAULT_LANG_A;
    sel.addEventListener("change", () =>
      localStorage.setItem("summaryLang", sel.value)
    );

    // Restore persisted meeting log.
    try {
      this.meetingLog = JSON.parse(localStorage.getItem("meetingLog") || "[]");
    } catch {
      this.meetingLog = [];
    }
    this._renderNotes();

    this.els.summarizeBtn.addEventListener("click", () => this._summarize());
    this.els.exportBtn.addEventListener("click", () => this._exportNotes());
    this.els.newMeetingBtn.addEventListener("click", () => this._newMeeting());
  }

  /** Append a finalized turn to the meeting log and persist it. */
  _commitTurn() {
    const source = this._curSource.trim();
    const translation = this._curTranslation.trim();
    this._curSource = "";
    this._curTranslation = "";
    if (!source && !translation) return;

    this.meetingLog.push({ t: Date.now(), source, translation });
    try {
      localStorage.setItem("meetingLog", JSON.stringify(this.meetingLog));
    } catch {
      /* storage full — keep going in memory */
    }
    this._renderNotes();
  }

  _renderNotes() {
    const n = this.meetingLog.length;
    this.els.noteCount.textContent = `(${n})`;
    if (this.els.notesPlaceholder) {
      this.els.notesPlaceholder.style.display = n ? "none" : "";
    }
    // Re-render the log list (skip the placeholder node).
    this.els.notes
      .querySelectorAll("[data-note]")
      .forEach((el) => el.remove());

    for (const entry of this.meetingLog) {
      const time = new Date(entry.t).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      });
      const row = document.createElement("div");
      row.dataset.note = "1";
      row.className = "rounded-lg bg-slate-950/50 px-3 py-2 text-sm";
      row.innerHTML = `
        <div class="mb-0.5 text-[10px] font-mono text-slate-600">${time}</div>
        <div class="text-slate-400">${this._escape(entry.source)}</div>
        <div class="text-slate-100">${this._escape(entry.translation)}</div>`;
      this.els.notes.appendChild(row);
    }
    this.els.notes.scrollTop = this.els.notes.scrollHeight;
  }

  _escape(s) {
    const d = document.createElement("div");
    d.textContent = s || "";
    return d.innerHTML;
  }

  /** Build a plain-text transcript for the summarizer / export. */
  _transcriptText() {
    return this.meetingLog
      .map((e) => {
        const time = new Date(e.t).toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
        });
        const parts = [];
        if (e.source) parts.push(e.source);
        if (e.translation) parts.push(`→ ${e.translation}`);
        return `[${time}] ${parts.join("  ")}`;
      })
      .join("\n");
  }

  async _summarize() {
    if (!this.meetingLog.length) {
      this._setStatus("idle", "No notes to summarize yet");
      return;
    }
    const base = this._serverBase();
    if (!base) {
      this._setStatus("error", "Set the server URL first");
      return;
    }

    this.els.summarizeBtn.disabled = true;
    const prevLabel = this.els.summarizeBtn.textContent;
    this.els.summarizeBtn.textContent = "Summarizing…";
    try {
      const res = await fetch(`${base}/api/summarize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          transcript: this._transcriptText(),
          language: this.els.summaryLang.value,
          token: this._accessToken() || undefined,
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      this._lastSummary = data.summary || "";
      this.els.summaryContent.textContent = this._lastSummary;
      this.els.summaryBox.classList.remove("hidden");
      this._setStatus(this.running ? "live" : "idle", "Summary updated");
    } catch (e) {
      this._setStatus("error", `Summary failed: ${e.message}`);
    } finally {
      this.els.summarizeBtn.disabled = false;
      this.els.summarizeBtn.textContent = prevLabel;
    }
  }

  _exportNotes() {
    if (!this.meetingLog.length) {
      this._setStatus("idle", "Nothing to export");
      return;
    }
    const date = new Date().toLocaleString();
    let md = `# 회의 노트 / Meeting Notes\n\n_${date}_\n\n`;
    if (this._lastSummary) {
      md += `## 요약 / Summary\n\n${this._lastSummary}\n\n`;
    }
    md += `## 전체 기록 / Full transcript\n\n`;
    for (const e of this.meetingLog) {
      const time = new Date(e.t).toLocaleTimeString();
      md += `- **[${time}]** ${e.source}`;
      if (e.translation) md += `\n  - → ${e.translation}`;
      md += `\n`;
    }

    const blob = new Blob([md], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `meeting-notes-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, "")}.md`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  _newMeeting() {
    if (this.meetingLog.length && !confirm("현재 회의 노트를 지우고 새로 시작할까요?")) {
      return;
    }
    this.meetingLog = [];
    this._curSource = "";
    this._curTranslation = "";
    this._lastSummary = "";
    localStorage.removeItem("meetingLog");
    this.els.summaryContent.textContent = "";
    this.els.summaryBox.classList.add("hidden");
    this._renderNotes();
    this._setStatus(this.running ? "live" : "idle", "New meeting started");
  }

  /** Resolve the backend base URL (no trailing slash) for this device. */
  _serverBase() {
    const saved = (localStorage.getItem("serverUrl") || "").trim();
    if (saved) return saved.replace(/\/+$/, "");
    const fromConfig = (window.DEFAULT_SERVER_URL || "").trim();
    if (fromConfig) return fromConfig.replace(/\/+$/, "");
    // On the web (served over http/https) default to the current origin.
    if (location.protocol === "http:" || location.protocol === "https:") {
      return location.origin;
    }
    return "";
  }

  /** Resolve the server access token for this device (may be empty). */
  _accessToken() {
    const saved = (localStorage.getItem("accessToken") || "").trim();
    if (saved) return saved;
    return (window.DEFAULT_ACCESS_TOKEN || "").trim();
  }

  /** Pull model / key status from the backend's health endpoint. */
  _refreshHealth() {
    const base = this._serverBase();
    if (!base) {
      this._setStatus("idle", "Set the server URL to begin");
      return;
    }
    fetch(`${base}/api/health`)
      .then((r) => r.json())
      .then((d) => {
        this.els.modelInfo.textContent = d.model || "—";
        if (!d.api_key_configured) {
          this._setStatus("error", "API key not configured on server");
        }
      })
      .catch(() => this._setStatus("error", "Cannot reach server"));
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

  _wsUrl() {
    const base = this._serverBase();
    if (!base) {
      throw new Error("No server URL set — enter your cloud backend address");
    }
    // http(s)://host -> ws(s)://host/api/stream
    const wsBase = base.replace(/^http/i, "ws");
    const params = new URLSearchParams({
      a: this.els.langA.value,
      b: this.els.langB.value,
      voice: this.els.voiceSelect.value,
    });
    const token = this._accessToken();
    if (token) params.set("token", token);
    return `${wsBase}/api/stream?${params.toString()}`;
  }

  _openSocket() {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(this._wsUrl());
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
        this._commitTurn(); // save this turn into the meeting notes
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
    await this.captureContext.audioWorklet.addModule("./audio-processor.js");

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

    // Accumulate the raw text so the turn can be saved to the meeting notes.
    if (isTranslation) this._curTranslation += text;
    else this._curSource += text;

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
