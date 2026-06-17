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

// ---- UI internationalization (Korean / English) ----------------------------
const UI_I18N = {
  ko: {
    subtitle: "실시간 음성-대-음성 동시통역 · 다국어",
    session: "세션 제어",
    "ctrl.start": "통역 시작",
    "ctrl.stop": "정지",
    "ctrl.pause": "⏸️ 일시정지",
    "ctrl.resume": "▶️ 재개",
    "ctrl.replay": "🔁 다시 듣기",
    "ctrl.pronounce": "🗣️ 발음",
    "mode.audio": "🔊 음성+자막",
    "mode.text": "📝 자막만",
    "pron.title": "발음",
    "pron.roman": "로마자",
    "pron.hangul": "한글",
    "field.serverUrl": "서버 주소",
    "btn.save": "저장",
    "help.serverUrl": "클라우드 백엔드 주소. 안드로이드 앱에선 필수, 웹에선 비워두면 현재 서버를 사용합니다.",
    "field.accessToken": "접속 토큰",
    "ph.accessToken": "서버 접속 토큰",
    "help.accessToken": "서버의 ACCESS_TOKEN과 일치해야 합니다. Save 버튼으로 주소와 함께 저장됩니다.",
    "field.languages": "언어",
    "help.languages": "Language B = 들리는 언어(출력). 기본 통역 모델은 입력 언어를 자동 감지(70+)하므로 A는 자동으로 두어도 됩니다.",
    "field.displayLangs": "표시 언어 (각자 읽을 자막 · 최대 3)",
    "help.displayLangs": "설정하면 각 발화를 이 언어들로도 자막에 함께 표시합니다(참가자 각자 읽기). 음성은 위 출력 언어 1개로 나갑니다.",
    "opt.none": "— 없음 —",
    "field.voice": "목소리",
    "help.voice": "Persona 모델에서만 적용됩니다. 기본 통역 모델은 화자 본인의 목소리를 그대로 살립니다.",
    "field.speed": "빠르기",
    "help.speed": "통역 음성의 재생 속도를 조절합니다(실시간).",
    "risk.title": "🛡️ 리스크 감지",
    "help.risk": "발화가 끝난 뒤 별도로 분석합니다. 통역 속도에는 영향이 없습니다.",
    "ph.riskContext": "분야 (예: oil trading, 알루미늄 수출)",
    "clarify.title": "🔎 의미 확인",
    "help.clarify": '발음이 어눌해 잘못 통역된 것 같으면 문맥으로 "혹시 이런 뜻?"을 제안합니다.',
    "info.model": "모델",
    "info.input": "입력",
    "info.output": "출력",
    "info.latency": "지연",
    "clarify.heading": "🔎 혹시 이런 뜻이었나요?",
    "clarify.corrected": "교정 번역",
    "risk.suggestedQ": "추천 질문",
    "btn.copy": "복사",
    "transcript.title": "실시간 자막",
    "btn.clear": "지우기",
    "transcript.ph": "두 언어를 고르고 통역 시작을 누른 뒤, 마이크를 허용하고 말하세요…",
    "notes.title": "회의 노트",
    "notes.summaryIn": "요약 언어",
    "notes.summarize": "요약",
    "notes.saveAs": "저장:",
    "notes.new": "새 회의",
    "qa.title": "💬 대화 내용 질문",
    "ph.ask": "예: 결제 조건이 어떻게 정해졌지?",
    "qa.ask": "질문",
    "summary.title": "요약",
    "notes.ph": "회의가 시작되면 발화가 시간순으로 기록됩니다. 기록은 이 기기에 저장되어 앱을 닫아도 유지됩니다.",
    footer: "Google Gemini Multimodal Live API 기반 · 제로-추론 오디오 파이프라인",
    // dynamic
    "status.idle": "대기",
    "status.connecting": "연결 중…",
    "status.live": "연결됨 — 말하세요",
    "status.paused": "일시정지됨",
    "status.saved": "설정 저장됨",
    "status.setServer": "서버 주소를 입력하세요",
    "status.cannotReach": "서버에 연결할 수 없음",
    "status.keyMissing": "서버에 API 키가 설정되지 않음",
    "status.tokenMismatch": "Access Token이 서버와 불일치",
    "status.noServerFirst": "먼저 서버 주소를 설정하세요",
    "status.failStart": "시작 실패",
    "msg.nothingExport": "내보낼 내용이 없습니다",
    "msg.noConv": "질문할 대화가 아직 없습니다",
    "msg.newMeeting": "새 회의를 시작했습니다",
    "msg.confirmNew": "현재 회의 노트를 지우고 새로 시작할까요?",
    "msg.askFail": "질문 실패",
    "msg.exportFail": "내보내기 실패",
    "msg.summaryFail": "요약 실패",
    "msg.summaryUpdated": "요약이 갱신되었습니다",
    "msg.pronFail": "발음 변환 실패",
  },
  en: {
    subtitle: "Real-time audio-to-audio simultaneous interpreter · multilingual",
    session: "Session Control",
    "ctrl.start": "Start Translating",
    "ctrl.stop": "Stop",
    "ctrl.pause": "⏸️ Pause",
    "ctrl.resume": "▶️ Resume",
    "ctrl.replay": "🔁 Replay",
    "ctrl.pronounce": "🗣️ Pronounce",
    "mode.audio": "🔊 Audio + captions",
    "mode.text": "📝 Captions only",
    "pron.title": "Pronunciation",
    "pron.roman": "Roman",
    "pron.hangul": "Hangul",
    "field.serverUrl": "Server URL",
    "btn.save": "Save",
    "help.serverUrl": "Cloud backend address. Required in the Android app; leave blank on the web to use this same server.",
    "field.accessToken": "Access Token",
    "ph.accessToken": "server access token",
    "help.accessToken": "Must match ACCESS_TOKEN on the server. Saved with the URL via the Save button.",
    "field.languages": "Languages",
    "help.languages": "Language B = the language you hear (output). The default translate model auto-detects the source (70+), so A can stay on Auto.",
    "field.displayLangs": "Caption languages (each reads their own · up to 3)",
    "help.displayLangs": "If set, each utterance is also shown in these languages (everyone reads their own). Audio plays in the single output language above.",
    "opt.none": "— none —",
    "field.voice": "Voice",
    "help.voice": "Applies to the persona model only. The default translate model preserves the speaker's own voice.",
    "field.speed": "Speed",
    "help.speed": "Adjusts playback speed of the translated voice (live).",
    "risk.title": "🛡️ Risk Guard",
    "help.risk": "Analyzed after each turn finishes — no effect on translation speed.",
    "ph.riskContext": "Industry (e.g. oil trading, aluminum export)",
    "clarify.title": "🔎 Clarify",
    "help.clarify": 'If unclear pronunciation seems mistranslated, suggests "Did you mean…?" from context.',
    "info.model": "Model",
    "info.input": "Input",
    "info.output": "Output",
    "info.latency": "Latency",
    "clarify.heading": "🔎 Did you mean…?",
    "clarify.corrected": "Corrected translation",
    "risk.suggestedQ": "Suggested question",
    "btn.copy": "Copy",
    "transcript.title": "Live Transcript",
    "btn.clear": "Clear",
    "transcript.ph": "Pick your two languages, press Start, allow the mic, and start speaking…",
    "notes.title": "Meeting Notes",
    "notes.summaryIn": "Summary in",
    "notes.summarize": "Summarize",
    "notes.saveAs": "Save:",
    "notes.new": "New",
    "qa.title": "💬 Ask about this conversation",
    "ph.ask": "e.g. What payment terms were agreed?",
    "qa.ask": "Ask",
    "summary.title": "Summary",
    "notes.ph": "Once a meeting starts, utterances are logged here in order. The log is saved on this device and survives app restarts.",
    footer: "Powered by the Google Gemini Multimodal Live API · zero-inference audio pipeline",
    // dynamic
    "status.idle": "Idle",
    "status.connecting": "Connecting…",
    "status.live": "Live — speak now",
    "status.paused": "Paused",
    "status.saved": "Settings saved",
    "status.setServer": "Set the server URL to begin",
    "status.cannotReach": "Cannot reach server",
    "status.keyMissing": "API key not configured on server",
    "status.tokenMismatch": "Access Token does not match the server",
    "status.noServerFirst": "Set the server URL first",
    "status.failStart": "Failed to start",
    "msg.nothingExport": "Nothing to export",
    "msg.noConv": "No conversation to ask about yet",
    "msg.newMeeting": "New meeting started",
    "msg.confirmNew": "Clear the current meeting notes and start over?",
    "msg.askFail": "Question failed",
    "msg.exportFail": "Export failed",
    "msg.summaryFail": "Summary failed",
    "msg.summaryUpdated": "Summary updated",
    "msg.pronFail": "Pronunciation failed",
  },
};

let UI_LANG = localStorage.getItem("uiLang") || "ko";

function t(key) {
  const table = UI_I18N[UI_LANG] || UI_I18N.ko;
  return table[key] != null ? table[key] : UI_I18N.ko[key] || key;
}

function applyI18n(lang) {
  UI_LANG = UI_I18N[lang] ? lang : "ko";
  localStorage.setItem("uiLang", UI_LANG);
  document.documentElement.lang = UI_LANG;
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const v = t(el.getAttribute("data-i18n"));
    if (v) el.textContent = v;
  });
  document.querySelectorAll("[data-i18n-ph]").forEach((el) => {
    const v = t(el.getAttribute("data-i18n-ph"));
    if (v) el.setAttribute("placeholder", v);
  });
}

function applyTheme(theme) {
  const light = theme === "light";
  document.body.classList.toggle("light", light);
  localStorage.setItem("theme", light ? "light" : "dark");
  const btn = document.getElementById("themeToggle");
  if (btn) btn.textContent = light ? "☀️" : "🌙";
}

class TranslatorClient {
  constructor() {
    this.ws = null;
    this.captureContext = null;
    this.playbackContext = null;
    this.workletNode = null;
    this.micSource = null;
    this.mediaStream = null;
    this.running = false;
    this._lastError = "";

    // Playback scheduling cursor (in playbackContext time).
    this.nextPlayTime = 0;

    // Meeting notes: persisted log of finalized turns + in-progress accumulators.
    this.meetingLog = [];
    this._curSource = "";
    this._curTranslation = "";

    // Quick-action state.
    this.paused = false; // when true, mic audio is not sent upstream
    this.playbackRate = parseFloat(localStorage.getItem("playbackRate") || "1.0");
    this._curAudioChunks = []; // Float32 pieces of the current turn's output audio
    this._lastAudio = null; // concatenated Float32Array of the last turn
    this._lastTranslationText = ""; // text of the last finalized translation
    this._turnTimer = null; // debounce for committing a turn to the notes
    // Output mode: true = hear translated audio + captions; false = captions only.
    this.audioOutput = localStorage.getItem("audioOutput") !== "0";

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
      displayLang1: document.getElementById("displayLang1"),
      displayLang2: document.getElementById("displayLang2"),
      displayLang3: document.getElementById("displayLang3"),
      // Meeting notes
      notes: document.getElementById("notes"),
      notesPlaceholder: document.getElementById("notesPlaceholder"),
      noteCount: document.getElementById("noteCount"),
      summaryLang: document.getElementById("summaryLang"),
      summarizeBtn: document.getElementById("summarizeBtn"),
      exportMdBtn: document.getElementById("exportMdBtn"),
      exportDocxBtn: document.getElementById("exportDocxBtn"),
      exportPdfBtn: document.getElementById("exportPdfBtn"),
      newMeetingBtn: document.getElementById("newMeetingBtn"),
      summaryBox: document.getElementById("summaryBox"),
      summaryContent: document.getElementById("summaryContent"),
      askInput: document.getElementById("askInput"),
      askBtn: document.getElementById("askBtn"),
      askAnswer: document.getElementById("askAnswer"),
      // Clarify
      clarifyToggle: document.getElementById("clarifyToggle"),
      clarifyAlert: document.getElementById("clarifyAlert"),
      clarifyText: document.getElementById("clarifyText"),
      clarifyCorrected: document.getElementById("clarifyCorrected"),
      clarifyDismiss: document.getElementById("clarifyDismiss"),
      // Quick actions
      pauseBtn: document.getElementById("pauseBtn"),
      replayBtn: document.getElementById("replayBtn"),
      pronounceBtn: document.getElementById("pronounceBtn"),
      pronounceBox: document.getElementById("pronounceBox"),
      pronounceContent: document.getElementById("pronounceContent"),
      scriptSelect: document.getElementById("scriptSelect"),
      speedRange: document.getElementById("speedRange"),
      speedValue: document.getElementById("speedValue"),
      // Risk guard
      riskToggle: document.getElementById("riskToggle"),
      riskContext: document.getElementById("riskContext"),
      riskAlert: document.getElementById("riskAlert"),
      riskIcon: document.getElementById("riskIcon"),
      riskLevelLabel: document.getElementById("riskLevelLabel"),
      riskTypes: document.getElementById("riskTypes"),
      riskDismiss: document.getElementById("riskDismiss"),
      riskAlertText: document.getElementById("riskAlertText"),
      riskReason: document.getElementById("riskReason"),
      riskQuestionWrap: document.getElementById("riskQuestionWrap"),
      riskQuestion: document.getElementById("riskQuestion"),
      riskCopyBtn: document.getElementById("riskCopyBtn"),
      uiLang: document.getElementById("uiLang"),
      themeToggle: document.getElementById("themeToggle"),
      modeAudioBtn: document.getElementById("modeAudioBtn"),
      modeTextBtn: document.getElementById("modeTextBtn"),
    };

    this._setupAppearance();
    this._setupOutputMode();
    this._setupLanguages();
    this._setupMeetingNotes();
    this._setupControls();
    this._setupRiskGuard();

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

      this._setStatus("idle", t("status.saved"));
      this._refreshHealth();
    });

    this._refreshHealth();
  }

  /** Output-mode toggle: hear audio + captions, or captions only. */
  _setupOutputMode() {
    const render = () => {
      const on = "border-indigo-500 bg-indigo-600 text-white";
      const off = "border-slate-700 bg-slate-800 text-slate-300 hover:bg-slate-700";
      this.els.modeAudioBtn.className = `rounded-lg border px-2 py-2 font-medium transition ${this.audioOutput ? on : off}`;
      this.els.modeTextBtn.className = `rounded-lg border px-2 py-2 font-medium transition ${this.audioOutput ? off : on}`;
    };
    const set = (audio) => {
      this.audioOutput = audio;
      localStorage.setItem("audioOutput", audio ? "1" : "0");
      render();
      if (!audio) this._flushPlayback(); // stop any audio currently playing
    };
    this.els.modeAudioBtn.addEventListener("click", () => set(true));
    this.els.modeTextBtn.addEventListener("click", () => set(false));
    render();
  }

  /** UI language (한/영) + light/dark theme. */
  _setupAppearance() {
    this.els.uiLang.value = UI_LANG;
    applyI18n(UI_LANG);
    applyTheme(localStorage.getItem("theme") || "dark");
    this._refreshDynamicLabels();
    this._setStatus("idle", t("status.idle"));

    this.els.uiLang.addEventListener("change", () => {
      applyI18n(this.els.uiLang.value);
      this._refreshDynamicLabels();
      if (!this.running) this._setStatus("idle", t("status.idle"));
    });
    this.els.themeToggle.addEventListener("click", () => {
      const isLight = document.body.classList.contains("light");
      applyTheme(isLight ? "dark" : "light");
    });
  }

  /** Re-apply labels whose text depends on runtime state, in the current lang. */
  _refreshDynamicLabels() {
    this.els.toggleLabel.textContent = this.running ? t("ctrl.stop") : t("ctrl.start");
    this.els.pauseBtn.textContent = this.paused ? t("ctrl.resume") : t("ctrl.pause");
  }

  /** Populate the two language dropdowns and restore the saved pair. */
  _setupLanguages() {
    const fill = (sel, selected, withAuto) => {
      sel.innerHTML = "";
      const entries = withAuto
        ? [["auto", "🌐 자동 감지 (Auto-detect)"], ...Object.entries(LANGUAGES)]
        : Object.entries(LANGUAGES);
      for (const [code, label] of entries) {
        const opt = document.createElement("option");
        opt.value = code;
        opt.textContent = label;
        if (code === selected) opt.selected = true;
        sel.appendChild(opt);
      }
    };
    // Language A may be "auto" (detect any language -> Language B).
    fill(this.els.langA, localStorage.getItem("langA") || DEFAULT_LANG_A, true);
    fill(this.els.langB, localStorage.getItem("langB") || DEFAULT_LANG_B, false);

    const remember = () => {
      localStorage.setItem("langA", this.els.langA.value);
      localStorage.setItem("langB", this.els.langB.value);
    };
    this.els.langA.addEventListener("change", remember);
    this.els.langB.addEventListener("change", remember);

    // Display (caption) languages — each optional, "none" first.
    const displaySels = [this.els.displayLang1, this.els.displayLang2, this.els.displayLang3];
    const saved = JSON.parse(localStorage.getItem("displayLangs") || "[]");
    displaySels.forEach((sel, i) => {
      sel.innerHTML = "";
      for (const [code, label] of [["", t("opt.none")], ...Object.entries(LANGUAGES)]) {
        const opt = document.createElement("option");
        opt.value = code;
        opt.textContent = label;
        if (code === (saved[i] || "")) opt.selected = true;
        sel.appendChild(opt);
      }
      sel.addEventListener("change", () => {
        localStorage.setItem("displayLangs", JSON.stringify(displaySels.map((s) => s.value)));
      });
    });
  }

  /** Selected caption languages (deduped, non-empty). */
  _displayLangs() {
    const v = [this.els.displayLang1.value, this.els.displayLang2.value, this.els.displayLang3.value];
    return [...new Set(v.filter(Boolean))];
  }

  /** Wire the quick-action controls: speed, pause, replay, pronounce. */
  _setupControls() {
    // Playback speed.
    this.els.speedRange.value = String(this.playbackRate);
    this.els.speedValue.textContent = `${this.playbackRate.toFixed(2)}×`;
    this.els.speedRange.addEventListener("input", () => {
      this.playbackRate = parseFloat(this.els.speedRange.value);
      this.els.speedValue.textContent = `${this.playbackRate.toFixed(2)}×`;
      localStorage.setItem("playbackRate", String(this.playbackRate));
    });

    this.els.pauseBtn.addEventListener("click", () => this._togglePause());
    this.els.replayBtn.addEventListener("click", () => this._replayLast());
    this.els.pronounceBtn.addEventListener("click", () => this._pronounce());
    this.els.scriptSelect.addEventListener("change", () => {
      if (this._lastTranslationText) this._pronounce();
    });
  }

  _togglePause() {
    if (!this.running) return;
    this.paused = !this.paused;
    this.els.pauseBtn.textContent = this.paused ? t("ctrl.resume") : t("ctrl.pause");
    this._setStatus(this.paused ? "connecting" : "live", this.paused ? t("status.paused") : t("status.live"));
    if (this.paused) this._flushPlayback(); // stop any audio currently playing
  }

  /** Play a Float32 PCM buffer (24 kHz) honoring the current speed. */
  _playBuffer(float32) {
    if (!float32 || !float32.length) return;
    let ctx = this.playbackContext;
    if (!ctx || ctx.state === "closed") {
      ctx = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: OUTPUT_SAMPLE_RATE,
      });
    }
    const buffer = ctx.createBuffer(1, float32.length, OUTPUT_SAMPLE_RATE);
    buffer.copyToChannel(float32, 0);
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.playbackRate.value = this.playbackRate;
    source.connect(ctx.destination);
    source.start();
  }

  _replayLast() {
    if (!this._lastAudio) {
      this._setStatus(this.running ? "live" : "idle", "Nothing to replay yet");
      return;
    }
    this._playBuffer(this._lastAudio);
  }

  async _pronounce() {
    const text = this._lastTranslationText.trim();
    if (!text) {
      this._setStatus(this.running ? "live" : "idle", "No sentence to pronounce yet");
      return;
    }
    const base = this._serverBase();
    if (!base) {
      this._setStatus("error", t("status.noServerFirst"));
      return;
    }
    this.els.pronounceBox.classList.remove("hidden");
    this.els.pronounceContent.textContent = "…";
    try {
      const res = await fetch(`${base}/api/pronounce`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          script: this.els.scriptSelect.value,
          token: this._accessToken() || undefined,
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      this.els.pronounceContent.textContent = data.pronunciation || "(none)";
    } catch (e) {
      this.els.pronounceContent.textContent = `${t("msg.pronFail")}: ${e.message}`;
    }
  }

  /** Risk-detection copilot: restore settings, wire toggle/dismiss/copy. */
  _setupRiskGuard() {
    this.els.riskToggle.checked = localStorage.getItem("riskGuard") === "1";
    this.els.riskContext.value = localStorage.getItem("riskContext") || "";
    this.els.riskToggle.addEventListener("change", () => {
      localStorage.setItem("riskGuard", this.els.riskToggle.checked ? "1" : "0");
      if (!this.els.riskToggle.checked) this.els.riskAlert.classList.add("hidden");
    });
    this.els.riskContext.addEventListener("change", () =>
      localStorage.setItem("riskContext", this.els.riskContext.value.trim())
    );
    this.els.riskDismiss.addEventListener("click", () =>
      this.els.riskAlert.classList.add("hidden")
    );
    this.els.riskCopyBtn.addEventListener("click", () => {
      const q = this.els.riskQuestion.textContent || "";
      if (q && navigator.clipboard) navigator.clipboard.writeText(q).catch(() => {});
    });

    // Meaning-clarification toggle (shares the per-turn analysis call).
    this.els.clarifyToggle.checked = localStorage.getItem("clarify") === "1";
    this.els.clarifyToggle.addEventListener("change", () => {
      localStorage.setItem("clarify", this.els.clarifyToggle.checked ? "1" : "0");
      if (!this.els.clarifyToggle.checked) this.els.clarifyAlert.classList.add("hidden");
    });
    this.els.clarifyDismiss.addEventListener("click", () =>
      this.els.clarifyAlert.classList.add("hidden")
    );
  }

  /** Recent transcript text (last few turns) for context-aware analysis. */
  _recentHistory(n = 6) {
    return this.meetingLog
      .slice(-n)
      .map((e) => `${e.source}${e.translation ? "  → " + e.translation : ""}`)
      .join("\n");
  }

  /**
   * Per-turn analysis (risk + meaning clarification), off the real-time path.
   * One model call returns both; we render only what the user enabled.
   */
  async _analyzeTurn(entry) {
    const base = this._serverBase();
    if (!base || !entry.source) return;
    try {
      const res = await fetch(`${base}/api/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          original: entry.source,
          translation: entry.translation,
          alert_language: this.els.summaryLang.value,
          context: this.els.riskContext.value.trim(),
          history: this._recentHistory(),
          want_risk: this.els.riskToggle.checked,
          want_clarify: this.els.clarifyToggle.checked,
          token: this._accessToken() || undefined,
        }),
      });
      if (!res.ok) return;
      const data = await res.json();
      if (!data) return;

      let changed = false;

      // Risk (only if Risk Guard is on).
      if (this.els.riskToggle.checked && data.risk_level && data.risk_level !== "none") {
        entry.risk = {
          risk_level: data.risk_level,
          risk_types: data.risk_types,
          subtitle_alert: data.subtitle_alert,
          reason: data.reason,
          suggested_question: data.suggested_question,
        };
        this._renderRiskAlert(entry.risk);
        changed = true;
      }

      // Meaning clarification (only if Clarify is on).
      if (this.els.clarifyToggle.checked && data.clarify_suspected) {
        entry.clarify = {
          did_you_mean: data.clarify_did_you_mean,
          corrected: data.clarify_corrected_translation,
        };
        this._renderClarify(entry.clarify);
        changed = true;
      }

      if (changed) {
        try {
          localStorage.setItem("meetingLog", JSON.stringify(this.meetingLog));
        } catch {
          /* ignore */
        }
      }
    } catch {
      /* analysis is best-effort; never disturb the conversation */
    }
  }

  _renderClarify(c) {
    this.els.clarifyText.textContent = c.did_you_mean || "";
    this.els.clarifyCorrected.textContent = c.corrected || "";
    this.els.clarifyAlert.classList.remove("hidden");
  }

  _renderRiskAlert(risk) {
    const labels =
      UI_LANG === "ko"
        ? { low: "참고", medium: "확인 필요", high: "주의" }
        : { low: "Note", medium: "Check", high: "Caution" };
    const styles = {
      low: { box: "border-slate-600 bg-slate-800 text-slate-200", icon: "💡", label: labels.low },
      medium: { box: "border-amber-500/60 bg-amber-950/40 text-amber-100", icon: "⚠️", label: labels.medium },
      high: { box: "border-rose-500/70 bg-rose-950/40 text-rose-100", icon: "🚨", label: labels.high },
    };
    const s = styles[risk.risk_level] || styles.low;
    this.els.riskAlert.className = `mb-4 rounded-2xl border p-4 ${s.box}`;
    this.els.riskIcon.textContent = s.icon;
    this.els.riskLevelLabel.textContent = `${s.label} (${risk.risk_level})`;
    this.els.riskTypes.textContent = (risk.risk_types || []).join(" · ");
    this.els.riskAlertText.textContent = risk.subtitle_alert || "";
    this.els.riskReason.textContent = risk.reason || "";
    if (risk.suggested_question) {
      this.els.riskQuestion.textContent = risk.suggested_question;
      this.els.riskQuestionWrap.classList.remove("hidden");
    } else {
      this.els.riskQuestionWrap.classList.add("hidden");
    }
    this.els.riskAlert.classList.remove("hidden");
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
    this.els.exportMdBtn.addEventListener("click", () => this._export("md"));
    this.els.exportDocxBtn.addEventListener("click", () => this._export("docx"));
    this.els.exportPdfBtn.addEventListener("click", () => this._export("pdf"));
    this.els.newMeetingBtn.addEventListener("click", () => this._newMeeting());
    this.els.askBtn.addEventListener("click", () => this._ask());
    this.els.askInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") this._ask();
    });
  }

  /** Append a finalized turn to the meeting log and persist it. */
  /** Finalize the current turn (commit to notes + reset the live bubbles). */
  _finalizeTurn() {
    clearTimeout(this._turnTimer);
    this._turnTimer = null;
    this._commitTurn();
    this._sourceLine = null;
    this._translationLine = null;
  }

  _commitTurn() {
    const source = this._curSource.trim();
    const translation = this._curTranslation.trim();
    this._curSource = "";
    this._curTranslation = "";

    // Finalize the turn's audio for "다시 듣기".
    this._finalizeTurnAudio();
    if (translation) {
      this._lastTranslationText = translation;
      this.els.pronounceBtn.disabled = false;
    }

    if (!source && !translation) return;

    const entry = { t: Date.now(), source, translation };
    this.meetingLog.push(entry);
    try {
      localStorage.setItem("meetingLog", JSON.stringify(this.meetingLog));
    } catch {
      /* storage full — keep going in memory */
    }
    this._renderNotes();

    // Fire-and-forget per-turn analysis (does not block translation).
    if (this.els.riskToggle.checked || this.els.clarifyToggle.checked) {
      this._analyzeTurn(entry);
    }
    // Multilingual captions: translate this utterance into the display languages.
    if (this._displayLangs().length && entry.source) {
      this._multiTranslate(entry);
    }
  }

  /** Translate a finalized turn into the caption languages and show them. */
  async _multiTranslate(entry) {
    const base = this._serverBase();
    const targets = this._displayLangs();
    if (!base || !targets.length) return;
    try {
      const res = await fetch(`${base}/api/translate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: entry.source,
          targets,
          token: this._accessToken() || undefined,
        }),
      });
      if (!res.ok) return;
      const data = await res.json();
      const tr = data.translations || {};
      if (!Object.keys(tr).length) return;
      entry.multi = tr;
      try {
        localStorage.setItem("meetingLog", JSON.stringify(this.meetingLog));
      } catch {
        /* ignore */
      }
      this._appendMultiLines(tr);
      this._renderNotes();
    } catch {
      /* best effort */
    }
  }

  /** Append labeled per-language caption lines to the live transcript. */
  _appendMultiLines(translations) {
    if (this.els.placeholder) {
      this.els.placeholder.remove();
      this.els.placeholder = null;
    }
    const wrap = document.createElement("div");
    wrap.className = "rounded-lg bg-slate-800/60 px-3 py-2 text-sm";
    wrap.innerHTML = Object.entries(translations)
      .map(
        ([code, text]) =>
          `<div class="flex gap-2"><span class="shrink-0 font-mono text-[10px] uppercase text-sky-400">${code}</span><span class="text-slate-200">${this._escape(text)}</span></div>`
      )
      .join("");
    this.els.transcript.appendChild(wrap);
    this.els.transcript.scrollTop = this.els.transcript.scrollHeight;
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
      let riskHtml = "";
      if (entry.risk && entry.risk.risk_level && entry.risk.risk_level !== "none") {
        const color =
          entry.risk.risk_level === "high"
            ? "text-rose-300"
            : entry.risk.risk_level === "medium"
            ? "text-amber-300"
            : "text-slate-400";
        riskHtml = `<div class="mt-1 ${color} text-xs">⚠️ ${this._escape(
          entry.risk.subtitle_alert || ""
        )}</div>`;
      }
      let multiHtml = "";
      if (entry.multi && Object.keys(entry.multi).length) {
        multiHtml = Object.entries(entry.multi)
          .map(
            ([code, text]) =>
              `<div class="mt-0.5 flex gap-2 text-xs"><span class="shrink-0 font-mono text-[10px] uppercase text-sky-400">${code}</span><span class="text-slate-300">${this._escape(text)}</span></div>`
          )
          .join("");
      }
      row.innerHTML = `
        <div class="mb-0.5 text-[10px] font-mono text-slate-600">${time}</div>
        <div class="text-slate-400">${this._escape(entry.source)}</div>
        <div class="text-slate-100">${this._escape(entry.translation)}</div>${multiHtml}${riskHtml}`;
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
      this._setStatus("idle", t("msg.noConv"));
      return;
    }
    const base = this._serverBase();
    if (!base) {
      this._setStatus("error", t("status.noServerFirst"));
      return;
    }

    this.els.summarizeBtn.disabled = true;
    const prevLabel = this.els.summarizeBtn.textContent;
    this.els.summarizeBtn.textContent = "…";
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
      this._setStatus(this.running ? "live" : "idle", t("msg.summaryUpdated"));
    } catch (e) {
      this._setStatus("error", `${t("msg.summaryFail")}: ${e.message}`);
    } finally {
      this.els.summarizeBtn.disabled = false;
      this.els.summarizeBtn.textContent = prevLabel;
    }
  }

  /** Build the structured entry list the export endpoint expects. */
  _exportEntries() {
    return this.meetingLog.map((e) => {
      let translation = e.translation || "";
      if (e.multi && Object.keys(e.multi).length) {
        const extra = Object.entries(e.multi)
          .map(([code, text]) => `[${code}] ${text}`)
          .join("  ");
        translation = translation ? `${translation}  ${extra}` : extra;
      }
      return {
        time: new Date(e.t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        source: e.source || "",
        translation,
        risk:
          e.risk && e.risk.risk_level && e.risk.risk_level !== "none"
            ? `[${e.risk.risk_level}] ${e.risk.subtitle_alert || ""}`
            : "",
      };
    });
  }

  /** Export the notes as md / docx / pdf via the server, then download. */
  async _export(format) {
    if (!this.meetingLog.length) {
      this._setStatus("idle", t("msg.nothingExport"));
      return;
    }
    const base = this._serverBase();
    if (!base) {
      this._setStatus("error", t("status.noServerFirst"));
      return;
    }
    this._setStatus(this.running ? "live" : "idle", `Exporting ${format.toUpperCase()}…`);
    try {
      const res = await fetch(`${base}/api/export`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title: "회의 노트 / Meeting Notes",
          entries: this._exportEntries(),
          summary: this._lastSummary || "",
          format,
          token: this._accessToken() || undefined,
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `meeting-notes-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, "")}.${format}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      this._setStatus(this.running ? "live" : "idle", `${format.toUpperCase()} saved`);
    } catch (e) {
      this._setStatus("error", `${t("msg.exportFail")}: ${e.message}`);
    }
  }

  /** Ask a question grounded in the saved conversation. */
  async _ask() {
    const question = this.els.askInput.value.trim();
    if (!question) return;
    if (!this.meetingLog.length) {
      this._setStatus("idle", t("msg.noConv"));
      return;
    }
    const base = this._serverBase();
    if (!base) {
      this._setStatus("error", t("status.noServerFirst"));
      return;
    }
    this.els.askBtn.disabled = true;
    this.els.askAnswer.classList.remove("hidden");
    this.els.askAnswer.textContent = "…";
    try {
      const res = await fetch(`${base}/api/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          transcript: this._transcriptText(),
          question,
          language: this.els.summaryLang.value,
          token: this._accessToken() || undefined,
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      this.els.askAnswer.textContent = data.answer || "(no answer)";
    } catch (e) {
      this.els.askAnswer.textContent = `${t("msg.askFail")}: ${e.message}`;
    } finally {
      this.els.askBtn.disabled = false;
    }
  }

  _newMeeting() {
    if (this.meetingLog.length && !confirm(t("msg.confirmNew"))) {
      return;
    }
    this.meetingLog = [];
    this._curSource = "";
    this._curTranslation = "";
    this._lastSummary = "";
    localStorage.removeItem("meetingLog");
    this.els.summaryContent.textContent = "";
    this.els.summaryBox.classList.add("hidden");
    this.els.clarifyAlert.classList.add("hidden");
    this.els.riskAlert.classList.add("hidden");
    this.els.askAnswer.classList.add("hidden");
    this._renderNotes();
    this._setStatus(this.running ? "live" : "idle", t("msg.newMeeting"));
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
      this._setStatus("idle", t("status.setServer"));
      return;
    }
    fetch(`${base}/api/health`)
      .then((r) => r.json())
      .then((d) => {
        this.els.modelInfo.textContent = d.model || "—";
        if (!d.api_key_configured) {
          this._setStatus("error", t("status.keyMissing"));
        }
      })
      .catch(() => this._setStatus("error", t("status.cannotReach")));
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
      this._lastError = "";
      this._setStatus("connecting", t("status.connecting"));
      this.els.toggleBtn.disabled = true;

      await this._openSocket();
      await this._startCapture();
      this._initPlayback();

      this.running = true;
      this.paused = false;
      this.els.toggleLabel.textContent = t("ctrl.stop");
      this.els.toggleIcon.textContent = "⏹️";
      this.els.toggleBtn.disabled = false;
      this.els.pauseBtn.disabled = false;
      this.els.pauseBtn.textContent = t("ctrl.pause");
      this._setStatus("live", t("status.live"));
    } catch (err) {
      console.error(err);
      this._setStatus("error", err.message || t("status.failStart"));
      this.els.toggleBtn.disabled = false;
      await this._teardown();
    }
  }

  async stop() {
    this.running = false;
    this.paused = false;
    // Tell the server this turn is finished, then close down audio.
    this._sendControl({ action: "end" });
    await this._teardown();

    this.els.toggleLabel.textContent = t("ctrl.start");
    this.els.toggleIcon.textContent = "🎙️";
    this.els.pauseBtn.disabled = true;
    this.els.pauseBtn.textContent = t("ctrl.pause");
    // Replay / pronounce remain available for the last sentence.
    this._setStatus("idle", t("status.idle"));
  }

  async _teardown() {
    // Commit any in-progress turn before tearing down.
    if (this._turnTimer) this._finalizeTurn();
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
      ws.onerror = () => reject(new Error("WebSocket connection failed (서버 주소/네트워크 확인)"));
      ws.onclose = (ev) => {
        if (this.running) {
          this.running = false;
          // Surface the real reason: a server error message if we got one,
          // otherwise the close code (1008 = wrong access token).
          let why = this._lastError;
          if (!why) {
            why =
              ev.code === 1008
                ? t("status.tokenMismatch")
                : `${UI_LANG === "ko" ? "연결 끊김" : "Disconnected"} (code ${ev.code})`;
          }
          this._setStatus("error", why);
          this.els.toggleLabel.textContent = t("ctrl.start");
          this.els.toggleIcon.textContent = "🎙️";
          this.els.pauseBtn.disabled = true;
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
        this._finalizeTurn(); // save this turn into the meeting notes
        break;
      case "interrupted":
        // User barged in — drop any queued audio so we don't talk over them.
        this._flushPlayback();
        break;
      case "error":
        this._lastError = msg.message || "Server error";
        this._setStatus("error", this._lastError);
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
    // While paused we simply drop the audio (the session stays open).
    this.workletNode.port.onmessage = (e) => {
      if (this.paused) return;
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

    // Keep a copy so the turn can be replayed ("다시 듣기").
    this._curAudioChunks.push(float32);

    // Text-only mode: capture audio (for replay) but don't play it back.
    if (!this.audioOutput) return;

    const buffer = this.playbackContext.createBuffer(
      1,
      float32.length,
      OUTPUT_SAMPLE_RATE
    );
    buffer.copyToChannel(float32, 0);

    const source = this.playbackContext.createBufferSource();
    source.buffer = buffer;
    source.playbackRate.value = this.playbackRate;
    source.connect(this.playbackContext.destination);

    // Schedule sequentially so chunks play gaplessly. If we've fallen behind,
    // resync to "now" plus a small safety margin. Duration scales with speed.
    const now = this.playbackContext.currentTime;
    if (this.nextPlayTime < now) {
      this.nextPlayTime = now + 0.05;
    }
    source.start(this.nextPlayTime);
    this.nextPlayTime += buffer.duration / this.playbackRate;
  }

  /** Merge the current turn's audio chunks into one replayable buffer. */
  _finalizeTurnAudio() {
    if (!this._curAudioChunks.length) return;
    const total = this._curAudioChunks.reduce((n, c) => n + c.length, 0);
    const merged = new Float32Array(total);
    let off = 0;
    for (const c of this._curAudioChunks) {
      merged.set(c, off);
      off += c.length;
    }
    this._lastAudio = merged;
    this._curAudioChunks = [];
    this.els.replayBtn.disabled = false;
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

    // Fallback turn boundary: the streaming translate model may not send
    // turn_complete, so commit the turn after a short pause in transcripts.
    clearTimeout(this._turnTimer);
    this._turnTimer = setTimeout(() => this._finalizeTurn(), 1800);

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
