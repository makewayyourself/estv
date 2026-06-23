/**
 * Hyun Live Translator — menu-driven multi-view client.
 *
 * Views: home · translate (quick, ephemeral) · notes (list+search) ·
 *        note (record/log/summary) · settings.
 * Shared translation engine (WebSocket + AudioWorklet capture + 24kHz playback)
 * writes transcripts into the active view's log; meeting notes are persisted.
 */

const INPUT_SAMPLE_RATE = 16000;
const OUTPUT_SAMPLE_RATE = 24000;

const LANGUAGES = {
  ko: "한국어 (Korean)", en: "영어 (English)", ja: "일본어 (Japanese)",
  zh: "중국어 (Chinese)", fr: "프랑스어 (French)", es: "스페인어 (Spanish)",
  ar: "아랍어 (Arabic)", ru: "러시아어 (Russian)",
};
const DEFAULT_LANG_B = "ko";

// ---- i18n -----------------------------------------------------------------
const I18N = {
  ko: {
    "home.translate": "빠른 통역", "home.translateDesc": "즉석 실시간 통역·자막 (저장 안 함)",
    "home.notes": "회의 노트", "home.notesDesc": "녹음·기록·요약 · 주제/날짜로 검색",
    "home.settings": "설정", "home.settingsDesc": "언어·테마·서버·표시언어·보기모드",
    "home.tagline": "Hyun Live Translator · 자동 언어 감지 · 다국어 자막",
    "tr.ph": "아래 시작을 누르고 말하면 통역됩니다.<br />표시 언어는 설정에서 최대 3개까지 고를 수 있어요.",
    "notes.search": "주제·내용·날짜 검색", "notes.empty": "아직 노트가 없습니다. 아래 \"새 노트\"로 시작하세요.", "notes.new": "＋ 새 노트",
    "note.tabLog": "기록", "note.tabSummary": "요약", "vm.both": "원문+번역", "vm.source": "원문만", "vm.trans": "번역만",
    "note.summarize": "요약", "note.delete": "삭제", "note.ph": "아래 <b>시작</b>을 누르면 녹음·통역이 시작되고 여기에 기록됩니다.",
    "qa.ph": "예: 결제 조건이 어떻게 정해졌지?", "qa.ask": "질문",
    "set.uiLang": "앱 언어", "set.theme": "테마", "set.dark": "🌙 밤(어둡게)", "set.light": "☀️ 낮(밝게)",
    "set.fontSize": "글자 크기", "set.output": "출력(들리는) 언어", "set.outputHelp": "음성으로 나갈 1개 언어. 입력 언어는 자동 감지됩니다.",
    "set.display": "표시(자막) 언어 · 최대 3", "set.displayHelp": "각 발화를 이 언어들로 함께 자막 표시(참가자 각자 읽기).",
    "set.voice": "음성 (목소리)", "set.speed": "빠르기", "set.risk": "🛡️ 리스크 감지", "set.riskCtx": "분야 (예: oil trading)",
    "set.clarify": "🔎 의미 확인", "set.server": "서버 주소", "set.token": "접속 토큰", "set.tokenPh": "서버 접속 토큰",
    "set.save": "저장", "set.model": "모델", "set.admin": "🔒 관리자 · 서버 설정",
    "admin.note": "서버 운영자용 설정입니다. 일반 사용자는 건드릴 필요가 없습니다.", "title.admin": "관리자 설정",
    "vm.label": "보기 모드", "note.save": "저장(내보내기)", "note.summarizeBtn": "요약 생성 / 갱신",
    "mode.audio": "🔊 음성+자막", "mode.text": "📝 자막만", "ctrl.start": "시작", "ctrl.stop": "정지",
    "pron.title": "발음", "pron.roman": "로마자", "pron.hangul": "한글",
    "title.home": "Hyun Live Translator", "title.translate": "빠른 통역", "title.notes": "회의 노트", "title.settings": "설정",
    "st.idle": "대기", "st.connecting": "연결 중…", "st.live": "통역 중 — 말하세요", "st.paused": "일시정지",
    "st.saved": "저장됨", "st.tokenBad": "Access Token이 서버와 불일치", "st.cantReach": "서버 연결 불가",
    "st.noKey": "서버에 API 키 미설정", "st.setServer": "설정에서 서버 주소를 입력하세요",
    "msg.confirmDelete": "이 노트를 삭제할까요?", "msg.newNote": "새 노트",
  },
  en: {
    "home.translate": "Quick Translate", "home.translateDesc": "Instant live interpreting · captions (not saved)",
    "home.notes": "Meeting Notes", "home.notesDesc": "Record · log · summary · search by topic/date",
    "home.settings": "Settings", "home.settingsDesc": "Language · theme · server · caption langs",
    "home.tagline": "Hyun Live Translator · auto language detection · multilingual captions",
    "tr.ph": "Tap Start below and speak to translate.<br />Pick up to 3 caption languages in Settings.",
    "notes.search": "Search topic · content · date", "notes.empty": "No notes yet. Tap \"New note\" below to start.", "notes.new": "＋ New note",
    "note.tabLog": "Log", "note.tabSummary": "Summary", "vm.both": "Source+Translation", "vm.source": "Source only", "vm.trans": "Translation only",
    "note.summarize": "Summarize", "note.delete": "Delete", "note.ph": "Tap <b>Start</b> below to begin recording — it appears here.",
    "qa.ph": "e.g. What payment terms were agreed?", "qa.ask": "Ask",
    "set.uiLang": "App language", "set.theme": "Theme", "set.dark": "🌙 Dark", "set.light": "☀️ Light",
    "set.fontSize": "Font size", "set.output": "Output (spoken) language", "set.outputHelp": "One spoken language. Source is auto-detected.",
    "set.display": "Caption languages · up to 3", "set.displayHelp": "Show each utterance in these languages (everyone reads their own).",
    "set.voice": "Voice", "set.speed": "Speed", "set.risk": "🛡️ Risk Guard", "set.riskCtx": "Industry (e.g. oil trading)",
    "set.clarify": "🔎 Clarify", "set.server": "Server URL", "set.token": "Access token", "set.tokenPh": "server access token",
    "set.save": "Save", "set.model": "Model", "set.admin": "🔒 Admin · server settings",
    "admin.note": "Operator settings. Regular users don't need these.", "title.admin": "Admin",
    "vm.label": "View mode", "note.save": "Export", "note.summarizeBtn": "Generate / refresh summary",
    "mode.audio": "🔊 Audio + captions", "mode.text": "📝 Captions only", "ctrl.start": "Start", "ctrl.stop": "Stop",
    "pron.title": "Pronunciation", "pron.roman": "Roman", "pron.hangul": "Hangul",
    "title.home": "Hyun Live Translator", "title.translate": "Quick Translate", "title.notes": "Meeting Notes", "title.settings": "Settings",
    "st.idle": "Idle", "st.connecting": "Connecting…", "st.live": "Live — speak now", "st.paused": "Paused",
    "st.saved": "Saved", "st.tokenBad": "Access token does not match the server", "st.cantReach": "Cannot reach server",
    "st.noKey": "API key not configured on server", "st.setServer": "Set the server URL in Settings",
    "msg.confirmDelete": "Delete this note?", "msg.newNote": "New note",
  },
};
let UI_LANG = localStorage.getItem("uiLang") || "ko";
const t = (k) => (I18N[UI_LANG] && I18N[UI_LANG][k] != null ? I18N[UI_LANG][k] : (I18N.ko[k] || k));

function applyI18n() {
  document.documentElement.lang = UI_LANG;
  document.querySelectorAll("[data-i18n]").forEach((el) => { el.innerHTML = t(el.getAttribute("data-i18n")); });
  document.querySelectorAll("[data-i18n-ph]").forEach((el) => { el.setAttribute("placeholder", t(el.getAttribute("data-i18n-ph"))); });
}

function esc(s) { const d = document.createElement("div"); d.textContent = s || ""; return d.innerHTML; }
const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------
class App {
  constructor() {
    // engine state
    this.ws = null; this.captureContext = null; this.playbackContext = null;
    this.workletNode = null; this.micSource = null; this.mediaStream = null;
    this.running = false; this.paused = false; this._lastError = "";
    this.nextPlayTime = 0; this.playbackRate = parseFloat(localStorage.getItem("playbackRate") || "1.0");
    this.audioOutput = localStorage.getItem("audioOutput") !== "0";
    this._curAudioChunks = []; this._lastAudio = null; this._lastTranslationText = "";
    this._turnTimer = null; this._curSource = ""; this._curTranslation = "";
    this._srcLine = null; this._trLine = null;

    // data
    this.notes = this._loadNotes();
    this.quickLog = [];
    this.context = "quick";       // "quick" | "note"
    this.activeNoteId = null;
    this.viewMode = localStorage.getItem("viewMode") || "both";

    // active view targets (set on navigation)
    this.view = "home";
    this.transcriptEl = null; this.riskEl = null; this.clarifyEl = null;

    this._cache();
    this._bind();
    this._applyAppearance();
    this._populateLangs();
    this._refreshHealth();
    this.show("home");
  }

  _cache() {
    this.el = {};
    ["backBtn","viewTitle","statusDot","statusText","controlBar","newNoteBtn",
     "toggleBtn","toggleIcon","toggleLabel","pauseBtn","replayBtn","pronounceBtn",
     "pronounceBox","pronounceContent","scriptSelect","modeAudioBtn","modeTextBtn",
     "qkTranscript","qkRisk","qkClarify","noteTranscript","noteRisk","noteClarify",
     "noteTitle","noteDate","noteMenuBtn","noteMenu","noteSummarizeBtn","noteExportMd","noteExportDocx","noteExportPdf","noteDelete",
     "homeVer","verInfo",
     "askInput","askBtn","askAnswer","summaryContent","viewMode",
     "notesList","notesEmpty","noteSearch",
     "uiLang","themeSel","fontSel","langB","displayLang1","displayLang2","displayLang3",
     "voiceSelect","speedRange","speedValue","riskToggle","riskContext","clarifyToggle",
     "serverUrl","accessToken","saveServerBtn","modelInfo"].forEach((id) => (this.el[id] = $(id)));
  }

  // ---- navigation ---------------------------------------------------------
  show(view, opts = {}) {
    // leaving a recording-capable view while live → stop
    if (this.running && view !== this.view) this.stop();

    this.view = view;
    document.querySelectorAll("[data-view]").forEach((s) => (s.hidden = s.getAttribute("data-view") !== view));
    this.el.backBtn.classList.toggle("hidden", view === "home");
    this.el.controlBar.hidden = !(view === "translate" || view === "note");
    this.el.newNoteBtn.hidden = view !== "notes";

    // Meeting notes are silent (listen + transcribe quietly): hide audio
    // controls and force captions-only. Quick Translate keeps audio.
    const audioUI = view === "translate";
    this.el.modeAudioBtn.parentElement.style.display = audioUI ? "" : "none";
    this.el.replayBtn.style.display = audioUI ? "" : "none";
    this.el.pronounceBtn.style.display = audioUI ? "" : "none";

    if (view === "translate") {
      this.context = "quick"; this.quickLog = []; this._clearTranscript(this.el.qkTranscript, "qk-ph");
      this.transcriptEl = this.el.qkTranscript; this.riskEl = this.el.qkRisk; this.clarifyEl = this.el.qkClarify;
      this._setMode(localStorage.getItem("audioOutput") !== "0", true);
      this.el.viewTitle.innerHTML = t("title.translate");
    } else if (view === "note") {
      this.context = "note"; this.activeNoteId = opts.id;
      this.audioOutput = false; // notes never play audio
      this.transcriptEl = this.el.noteTranscript; this.riskEl = this.el.noteRisk; this.clarifyEl = this.el.noteClarify;
      this._openNote(opts.id);
    } else {
      this.el.viewTitle.textContent = t("title." + view);
      if (view === "notes") this._renderNotesList();
    }
    if (view === "home") this.el.viewTitle.innerHTML = `<span class="bg-gradient-to-r from-sky-400 to-indigo-400 bg-clip-text text-transparent">${t("title.home")}</span>`;
    this._refreshToggle();
  }

  // ---- binding ------------------------------------------------------------
  _bind() {
    document.querySelectorAll("[data-go]").forEach((b) => b.addEventListener("click", () => this.show(b.getAttribute("data-go"))));
    this.el.backBtn.addEventListener("click", () => this.show(this.view === "note" ? "notes" : this.view === "admin" ? "settings" : "home"));
    this.el.noteMenuBtn.addEventListener("click", () => (this.el.noteMenu.hidden = !this.el.noteMenu.hidden));

    this.el.toggleBtn.addEventListener("click", () => (this.running ? this.stop() : this.start()));
    this.el.pauseBtn.addEventListener("click", () => this._togglePause());
    this.el.replayBtn.addEventListener("click", () => this._replayLast());
    this.el.pronounceBtn.addEventListener("click", () => this._pronounce());
    this.el.scriptSelect.addEventListener("change", () => this._lastTranslationText && this._pronounce());
    this.el.modeAudioBtn.addEventListener("click", () => this._setMode(true));
    this.el.modeTextBtn.addEventListener("click", () => this._setMode(false));

    // notes
    this.el.newNoteBtn.addEventListener("click", () => this._newNote());
    this.el.noteSearch.addEventListener("input", () => this._renderNotesList());
    this.el.noteTitle.addEventListener("change", () => this._renameActive());
    this.el.noteDelete.addEventListener("click", () => this._deleteActive());
    this.el.noteSummarizeBtn.addEventListener("click", () => this._summarize());
    this.el.askBtn.addEventListener("click", () => this._ask());
    this.el.askInput.addEventListener("keydown", (e) => e.key === "Enter" && this._ask());
    this.el.noteExportMd.addEventListener("click", () => this._export("md"));
    this.el.noteExportDocx.addEventListener("click", () => this._export("docx"));
    this.el.noteExportPdf.addEventListener("click", () => this._export("pdf"));
    document.querySelectorAll(".note-tab").forEach((b) => b.addEventListener("click", () => this._tab(b.getAttribute("data-tab"))));
    this.el.viewMode.addEventListener("change", () => {
      this.viewMode = this.el.viewMode.value; localStorage.setItem("viewMode", this.viewMode);
      if (this.context === "note") this._renderLog(this.el.noteTranscript, this._note(this.activeNoteId).log, "note-ph");
    });

    // settings
    this.el.uiLang.value = UI_LANG;
    this.el.uiLang.addEventListener("change", () => { UI_LANG = this.el.uiLang.value; localStorage.setItem("uiLang", UI_LANG); applyI18n(); this._refreshToggle(); });
    this.el.themeSel.addEventListener("change", () => this._applyTheme(this.el.themeSel.value));
    this.el.fontSel.addEventListener("change", () => this._applyFont(this.el.fontSel.value));
    this.el.speedRange.addEventListener("input", () => { this.playbackRate = parseFloat(this.el.speedRange.value); this.el.speedValue.textContent = `${this.playbackRate.toFixed(2)}×`; localStorage.setItem("playbackRate", String(this.playbackRate)); });
    this.el.riskToggle.checked = localStorage.getItem("riskGuard") === "1";
    this.el.clarifyToggle.checked = localStorage.getItem("clarify") === "1";
    this.el.riskContext.value = localStorage.getItem("riskContext") || "";
    this.el.riskToggle.addEventListener("change", () => localStorage.setItem("riskGuard", this.el.riskToggle.checked ? "1" : "0"));
    this.el.clarifyToggle.addEventListener("change", () => localStorage.setItem("clarify", this.el.clarifyToggle.checked ? "1" : "0"));
    this.el.riskContext.addEventListener("change", () => localStorage.setItem("riskContext", this.el.riskContext.value.trim()));
    this.el.serverUrl.value = this._serverBase();
    this.el.accessToken.value = this._token();
    this.el.saveServerBtn.addEventListener("click", () => {
      const v = this.el.serverUrl.value.trim().replace(/\/+$/, "");
      v ? localStorage.setItem("serverUrl", v) : localStorage.removeItem("serverUrl");
      const tok = this.el.accessToken.value.trim();
      tok ? localStorage.setItem("accessToken", tok) : localStorage.removeItem("accessToken");
      this._setStatus("idle", t("st.saved")); this._refreshHealth();
    });
  }

  _applyAppearance() {
    applyI18n();
    this._applyTheme(localStorage.getItem("theme") || "dark");
    this._applyFont(localStorage.getItem("fontSize") || "md");
    this.el.speedRange.value = String(this.playbackRate);
    this.el.speedValue.textContent = `${this.playbackRate.toFixed(2)}×`;
    this.el.viewMode.value = this.viewMode;
    this._setMode(this.audioOutput, true);
  }
  _applyTheme(theme) { document.body.classList.toggle("light", theme === "light"); localStorage.setItem("theme", theme); this.el.themeSel.value = theme; }
  _applyFont(size) { ["sm","md","lg","xl"].forEach((s) => { this.el.qkTranscript.classList.toggle("fs-" + s, s === size); this.el.noteTranscript.classList.toggle("fs-" + s, s === size); }); localStorage.setItem("fontSize", size); this.el.fontSel.value = size; }
  _setMode(audio, silent) {
    this.audioOutput = audio; if (!silent) localStorage.setItem("audioOutput", audio ? "1" : "0");
    const on = "border-indigo-500 bg-indigo-600 text-white", off = "border-slate-700 bg-slate-800 text-slate-300";
    this.el.modeAudioBtn.className = `rounded-lg border px-3 py-1.5 text-xs font-medium transition ${audio ? on : off}`;
    this.el.modeTextBtn.className = `rounded-lg border px-3 py-1.5 text-xs font-medium transition ${audio ? off : on}`;
    if (!audio) this._flushPlayback();
  }

  // ---- languages ----------------------------------------------------------
  _populateLangs() {
    const fillB = (sel, sv) => { sel.innerHTML = ""; for (const [c, l] of Object.entries(LANGUAGES)) { const o = document.createElement("option"); o.value = c; o.textContent = l; if (c === sv) o.selected = true; sel.appendChild(o); } };
    fillB(this.el.langB, localStorage.getItem("langB") || DEFAULT_LANG_B);
    this.el.langB.addEventListener("change", () => localStorage.setItem("langB", this.el.langB.value));
    const ds = [this.el.displayLang1, this.el.displayLang2, this.el.displayLang3];
    const saved = JSON.parse(localStorage.getItem("displayLangs") || "[]");
    ds.forEach((sel, i) => {
      sel.innerHTML = "";
      for (const [c, l] of [["", "—"], ...Object.entries(LANGUAGES)]) { const o = document.createElement("option"); o.value = c; o.textContent = l; if (c === (saved[i] || "")) o.selected = true; sel.appendChild(o); }
      sel.addEventListener("change", () => localStorage.setItem("displayLangs", JSON.stringify(ds.map((s) => s.value))));
    });
  }
  _displayLangs() { return [...new Set([this.el.displayLang1.value, this.el.displayLang2.value, this.el.displayLang3.value].filter(Boolean))]; }

  // ---- status / toggle ----------------------------------------------------
  _setStatus(state, text) {
    const c = { idle: "bg-slate-500", connecting: "bg-amber-400 pulse-dot", live: "bg-emerald-400 pulse-dot", error: "bg-rose-500" };
    this.el.statusDot.className = `h-2 w-2 rounded-full ${c[state] || c.idle}`;
    this.el.statusText.textContent = text;
  }
  _refreshToggle() {
    this.el.toggleLabel.textContent = this.running ? t("ctrl.stop") : t("ctrl.start");
    this.el.toggleIcon.textContent = this.running ? "⏹️" : "▶️";
    this.el.pauseBtn.textContent = this.paused ? "▶️" : "⏸️";
    if (!this.running) this._setStatus("idle", t("st.idle"));
  }
  _refreshHealth() {
    const base = this._serverBase();
    if (!base) { this._setStatus("idle", t("st.setServer")); return; }
    fetch(`${base}/api/health`).then((r) => r.json()).then((d) => {
      this.el.modelInfo.textContent = d.model || "—";
      if (this.el.homeVer) this.el.homeVer.textContent = d.version || "?";
      if (this.el.verInfo) this.el.verInfo.textContent = d.version || "?";
      if (!d.api_key_configured) this._setStatus("error", t("st.noKey"));
    }).catch(() => this._setStatus("error", t("st.cantReach")));
  }

  // ---- notes data ---------------------------------------------------------
  _loadNotes() { try { return JSON.parse(localStorage.getItem("notesV2") || "[]"); } catch { return []; } }
  _saveNotes() { try { localStorage.setItem("notesV2", JSON.stringify(this.notes)); } catch {} }
  _note(id) { return this.notes.find((n) => n.id === id); }
  _currentLog() { return this.context === "note" ? (this._note(this.activeNoteId)?.log || []) : this.quickLog; }
  _persist() { if (this.context === "note") { const n = this._note(this.activeNoteId); if (n) n.updatedAt = Date.now(); this._saveNotes(); } }

  _newNote() {
    const id = "n" + Date.now();
    this.notes.unshift({ id, title: t("msg.newNote"), createdAt: Date.now(), updatedAt: Date.now(), log: [], summary: "", topic: "" });
    this._saveNotes();
    this.show("note", { id });
  }
  _deleteActive() {
    if (!confirm(t("msg.confirmDelete"))) return;
    this.notes = this.notes.filter((n) => n.id !== this.activeNoteId);
    this._saveNotes(); this.show("notes");
  }
  _renameActive() { const n = this._note(this.activeNoteId); if (n) { n.title = this.el.noteTitle.value.trim() || t("msg.newNote"); this._saveNotes(); } }

  _renderNotesList() {
    const q = this.el.noteSearch.value.trim().toLowerCase();
    const list = this.notes
      .slice()
      .sort((a, b) => b.updatedAt - a.updatedAt)
      .filter((n) => {
        if (!q) return true;
        const hay = [n.title, n.topic, n.summary, ...n.log.map((e) => `${e.source} ${e.translation}`),
          new Date(n.createdAt).toLocaleString()].join(" ").toLowerCase();
        return hay.includes(q);
      });
    this.el.notesEmpty.style.display = list.length ? "none" : "";
    this.el.notesList.innerHTML = "";
    for (const n of list) {
      const date = new Date(n.createdAt).toLocaleString();
      const preview = n.topic || (n.log[0] ? n.log[0].source : "");
      const div = document.createElement("button");
      div.className = "block w-full rounded-xl border border-slate-800 bg-slate-900/60 p-4 text-left transition hover:border-indigo-500/60";
      div.innerHTML = `<div class="flex items-center justify-between gap-2"><span class="font-semibold">${esc(n.title)}</span><span class="text-xs text-slate-500">${n.log.length}</span></div>
        <div class="mt-0.5 text-xs text-slate-500">${esc(date)}</div>
        ${preview ? `<div class="mt-1 line-clamp-1 text-sm text-slate-400">${esc(preview)}</div>` : ""}`;
      div.addEventListener("click", () => this.show("note", { id: n.id }));
      this.el.notesList.appendChild(div);
    }
  }

  _openNote(id) {
    const n = this._note(id);
    if (!n) return this.show("notes");
    this.el.noteMenu.hidden = true;
    this.el.noteTitle.value = n.title;
    this.el.noteDate.textContent = new Date(n.createdAt).toLocaleString();
    this.el.viewTitle.textContent = n.title;
    this.el.summaryContent.textContent = n.summary || "";
    this.el.askAnswer.classList.add("hidden");
    this._tab("log");
    this._renderLog(this.el.noteTranscript, n.log, "note-ph");
  }
  _tab(which) {
    document.querySelectorAll(".note-tab").forEach((b) => {
      const on = b.getAttribute("data-tab") === which;
      b.classList.toggle("border-indigo-500", on); b.classList.toggle("text-slate-100", on);
      b.classList.toggle("border-transparent", !on); b.classList.toggle("text-slate-500", !on);
    });
    document.querySelectorAll("[data-pane]").forEach((p) => (p.hidden = p.getAttribute("data-pane") !== which));
  }

  // ---- transcript rendering ----------------------------------------------
  _clearTranscript(el, phClass) { el.querySelectorAll("[data-row]").forEach((n) => n.remove()); const ph = el.querySelector("." + phClass); if (ph) ph.style.display = ""; }
  _hidePh(el) { const ph = el.querySelector(".qk-ph,.note-ph"); if (ph) ph.style.display = "none"; }

  _bubble(role, text) {
    const isTr = role === "translation";
    const wrap = document.createElement("div"); wrap.dataset.row = "1";
    wrap.className = isTr ? "flex justify-end" : "flex justify-start";
    const b = document.createElement("div");
    b.className = isTr ? "max-w-[85%] rounded-2xl rounded-tr-sm bg-indigo-600/90 px-4 py-2.5 text-white"
                       : "max-w-[85%] rounded-2xl rounded-tl-sm bg-slate-800 px-4 py-2.5 text-slate-200";
    b.textContent = text; wrap.appendChild(b); return wrap;
  }
  _showRole(role) {
    if (this.viewMode === "source") return role === "source";
    if (this.viewMode === "trans") return role === "translation";
    return true;
  }

  /** Render a full saved log into a container (note open / view-mode change). */
  _renderLog(el, log, phClass) {
    el.querySelectorAll("[data-row]").forEach((n) => n.remove());
    const ph = el.querySelector("." + phClass); if (ph) ph.style.display = log.length ? "none" : "";
    for (const e of log) {
      if (e.source && this._showRole("source")) el.appendChild(this._bubble("source", e.source));
      if (e.translation && this._showRole("translation")) el.appendChild(this._bubble("translation", e.translation));
      if (e.multi) el.appendChild(this._multiBlock(e.multi));
      const wh = this._warnHtml(e);
      if (wh) { const w = document.createElement("div"); w.dataset.row = "1"; w.className = "rounded-lg border border-slate-800 bg-slate-950/40 px-3 py-1.5 text-xs"; w.innerHTML = wh; el.appendChild(w); }
    }
    el.scrollTop = el.scrollHeight;
  }
  _multiBlock(tr) {
    const w = document.createElement("div"); w.dataset.row = "1";
    w.className = "rounded-lg bg-slate-800/60 px-3 py-2";
    w.innerHTML = Object.entries(tr).map(([c, x]) => `<div class="flex gap-2"><span class="shrink-0 font-mono text-[10px] uppercase text-sky-400">${c}</span><span class="text-slate-300">${esc(x)}</span></div>`).join("");
    return w;
  }

  _appendTranscript(role, text) {
    if (!text) return;
    this._hidePh(this.transcriptEl);
    if (role === "translation") this._curTranslation += text; else this._curSource += text;
    clearTimeout(this._turnTimer);
    this._turnTimer = setTimeout(() => this._finalizeTurn(), 1800);
    if (!this._showRole(role)) return;
    const key = role === "translation" ? "_trLine" : "_srcLine";
    if (!this[key]) { const w = this._bubble(role, ""); this.transcriptEl.appendChild(w); this[key] = w.firstChild; }
    this[key].textContent += text;
    this.transcriptEl.scrollTop = this.transcriptEl.scrollHeight;
  }

  _finalizeTurn() { clearTimeout(this._turnTimer); this._turnTimer = null; this._commitTurn(); this._srcLine = null; this._trLine = null; }

  _commitTurn() {
    const source = this._curSource.trim(), translation = this._curTranslation.trim();
    this._curSource = ""; this._curTranslation = "";
    this._finalizeAudio();
    if (translation) { this._lastTranslationText = translation; this.el.pronounceBtn.disabled = false; }
    if (!source && !translation) return;
    const entry = { t: Date.now(), source, translation };
    this._currentLog().push(entry);
    if (this.context === "note") {
      const n = this._note(this.activeNoteId);
      if (n && (n.title === t("msg.newNote") || !n.title) && source) n.title = source.slice(0, 24);
      this.el.noteTitle.value = n ? n.title : this.el.noteTitle.value;
      this.el.viewTitle.textContent = n ? n.title : this.el.viewTitle.textContent;
    }
    this._persist();
    // Notes always get the context/meaning analysis (warnings in 참고);
    // Quick Translate only when the toggles are on.
    if (this.context === "note" || this.el.riskToggle.checked || this.el.clarifyToggle.checked) this._analyzeTurn(entry);
    if (this._displayLangs().length && source) this._multiTranslate(entry);
  }

  async _multiTranslate(entry) {
    const base = this._serverBase(), targets = this._displayLangs();
    if (!base || !targets.length) return;
    try {
      const r = await fetch(`${base}/api/translate`, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: entry.source, targets, token: this._token() || undefined }) });
      if (!r.ok) return;
      const tr = (await r.json()).translations || {};
      if (!Object.keys(tr).length) return;
      entry.multi = tr; this._persist();
      this.transcriptEl.appendChild(this._multiBlock(tr));
      this.transcriptEl.scrollTop = this.transcriptEl.scrollHeight;
    } catch {}
  }

  async _analyzeTurn(entry) {
    const base = this._serverBase(); if (!base || !entry.source) return;
    const isNote = this.context === "note";
    const wantRisk = isNote || this.el.riskToggle.checked;
    const wantClarify = isNote || this.el.clarifyToggle.checked;
    try {
      const r = await fetch(`${base}/api/analyze`, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ original: entry.source, translation: entry.translation, alert_language: this.el.langB.value,
          context: this.el.riskContext.value.trim(), history: this._currentLog().slice(-6).map((e) => `${e.source} → ${e.translation}`).join("\n"),
          want_risk: wantRisk, want_clarify: wantClarify, token: this._token() || undefined }) });
      if (!r.ok) return;
      const d = await r.json(); if (!d) return;
      let warned = false;
      if (wantRisk && d.risk_level && d.risk_level !== "none") { entry.risk = d; warned = true; this._renderRisk(d); }
      if (wantClarify && d.clarify_suspected) { entry.clarify = d; warned = true; this._renderClarify(d); }
      if (warned) { this._persist(); if (isNote) this._appendWarn(entry); }
    } catch {}
  }
  /** Inline "참고/Note" warning under a turn (notes). */
  _warnHtml(entry) {
    let h = "";
    if (entry.risk && entry.risk.risk_level && entry.risk.risk_level !== "none") {
      const col = entry.risk.risk_level === "high" ? "text-rose-300" : entry.risk.risk_level === "medium" ? "text-amber-300" : "text-slate-400";
      h += `<div class="${col}">⚠️ ${UI_LANG === "ko" ? "참고" : "Note"}: ${esc(entry.risk.subtitle_alert || "")}</div>`;
    }
    if (entry.clarify && entry.clarify.clarify_suspected) {
      h += `<div class="text-sky-300">🔎 ${esc(entry.clarify.clarify_did_you_mean || "")}</div>`;
    }
    return h;
  }
  _appendWarn(entry) {
    const h = this._warnHtml(entry); if (!h) return;
    const w = document.createElement("div"); w.dataset.row = "1";
    w.className = "rounded-lg border border-slate-800 bg-slate-950/40 px-3 py-1.5 text-xs";
    w.innerHTML = h; this.transcriptEl.appendChild(w);
    this.transcriptEl.scrollTop = this.transcriptEl.scrollHeight;
  }
  _renderClarify(d) {
    this.clarifyEl.innerHTML = `<div class="text-xs font-bold uppercase">🔎 ${UI_LANG === "ko" ? "혹시 이런 뜻?" : "Did you mean…?"}</div>
      <div class="mt-1 text-sm font-medium">${esc(d.clarify_did_you_mean)}</div>
      <div class="mt-1 rounded bg-black/20 p-2 text-sm">${esc(d.clarify_corrected_translation)}</div>`;
    this.clarifyEl.classList.remove("hidden");
  }
  _renderRisk(d) {
    const lv = { low: ["border-slate-600 bg-slate-800 text-slate-200","💡"], medium: ["border-amber-500/60 bg-amber-950/40 text-amber-100","⚠️"], high: ["border-rose-500/70 bg-rose-950/40 text-rose-100","🚨"] }[d.risk_level] || ["",""];
    this.riskEl.className = `mb-3 rounded-xl border p-3 ${lv[0]}`;
    this.riskEl.innerHTML = `<div class="text-sm font-medium">${lv[1]} ${esc(d.subtitle_alert)}</div>
      ${d.suggested_question ? `<div class="mt-2 rounded bg-black/20 p-2 text-sm">❓ ${esc(d.suggested_question)}</div>` : ""}`;
    this.riskEl.classList.remove("hidden");
  }

  // ---- session lifecycle --------------------------------------------------
  async start() {
    try {
      this._lastError = ""; this._setStatus("connecting", t("st.connecting")); this.el.toggleBtn.disabled = true;
      await this._openSocket(); await this._startCapture(); this._initPlayback();
      this.running = true; this.paused = false;
      this.el.toggleBtn.disabled = false; this.el.pauseBtn.disabled = false;
      this._refreshToggle(); this._setStatus("live", t("st.live"));
    } catch (e) { console.error(e); this._setStatus("error", e.message || "start failed"); this.el.toggleBtn.disabled = false; await this._teardown(); }
  }
  async stop() {
    this.running = false; this.paused = false; this._sendCtrl({ action: "end" }); await this._teardown();
    this.el.pauseBtn.disabled = true; this._refreshToggle();
  }
  async _teardown() {
    if (this._turnTimer) this._finalizeTurn();
    if (this.workletNode) { this.workletNode.port.onmessage = null; this.workletNode.disconnect(); this.workletNode = null; }
    if (this.micSource) { this.micSource.disconnect(); this.micSource = null; }
    if (this.mediaStream) { this.mediaStream.getTracks().forEach((x) => x.stop()); this.mediaStream = null; }
    if (this.captureContext) { await this.captureContext.close().catch(() => {}); this.captureContext = null; }
    if (this.playbackContext) { await this.playbackContext.close().catch(() => {}); this.playbackContext = null; }
    if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.close(); this.ws = null;
  }
  _togglePause() {
    if (!this.running) return;
    this.paused = !this.paused; this.el.pauseBtn.textContent = this.paused ? "▶️" : "⏸️";
    this._setStatus(this.paused ? "connecting" : "live", this.paused ? t("st.paused") : t("st.live"));
    if (this.paused) this._flushPlayback();
  }

  // ---- WebSocket ----------------------------------------------------------
  _serverBase() {
    const s = (localStorage.getItem("serverUrl") || "").trim(); if (s) return s.replace(/\/+$/, "");
    const c = (window.DEFAULT_SERVER_URL || "").trim(); if (c) return c.replace(/\/+$/, "");
    return (location.protocol === "http:" || location.protocol === "https:") ? location.origin : "";
  }
  _token() { const s = (localStorage.getItem("accessToken") || "").trim(); return s || (window.DEFAULT_ACCESS_TOKEN || "").trim(); }
  _wsUrl() {
    const base = this._serverBase(); if (!base) throw new Error(t("st.setServer"));
    const p = new URLSearchParams({ a: "auto", b: this.el.langB.value, voice: this.el.voiceSelect.value });
    const tok = this._token(); if (tok) p.set("token", tok);
    return `${base.replace(/^http/i, "ws")}/api/stream?${p.toString()}`;
  }
  _openSocket() {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(this._wsUrl()); ws.binaryType = "arraybuffer";
      ws.onopen = () => resolve();
      ws.onerror = () => reject(new Error(t("st.cantReach")));
      ws.onclose = (ev) => {
        if (this.running) {
          this.running = false;
          const why = this._lastError || (ev.code === 1008 ? t("st.tokenBad") : `${UI_LANG === "ko" ? "연결 끊김" : "Disconnected"} (${ev.code})`);
          this._setStatus("error", why); this.el.pauseBtn.disabled = true; this._refreshToggle(); this._teardown();
        }
      };
      ws.onmessage = (e) => this._onMessage(e);
      this.ws = ws;
    });
  }
  _sendCtrl(o) { if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(o)); }
  _onMessage(e) {
    if (e.data instanceof ArrayBuffer) return this._enqueueAudio(e.data);
    let m; try { m = JSON.parse(e.data); } catch { return; }
    if (m.type === "status") { if (m.model) this.el.modelInfo.textContent = m.model; }
    else if (m.type === "transcript") this._appendTranscript(m.role, m.text);
    else if (m.type === "turn_complete") this._finalizeTurn();
    else if (m.type === "interrupted") this._flushPlayback();
    else if (m.type === "error") { this._lastError = m.message || "error"; this._setStatus("error", this._lastError); }
  }

  // ---- capture ------------------------------------------------------------
  async _startCapture() {
    this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    this.captureContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: INPUT_SAMPLE_RATE });
    await this.captureContext.audioWorklet.addModule("./audio-processor.js");
    this.micSource = this.captureContext.createMediaStreamSource(this.mediaStream);
    this.workletNode = new AudioWorkletNode(this.captureContext, "pcm-capture-processor");
    this.workletNode.port.onmessage = (e) => { if (!this.paused && this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.send(e.data); };
    this.micSource.connect(this.workletNode);
    const sink = this.captureContext.createGain(); sink.gain.value = 0;
    this.workletNode.connect(sink); sink.connect(this.captureContext.destination);
  }

  // ---- playback -----------------------------------------------------------
  _initPlayback() { this.playbackContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: OUTPUT_SAMPLE_RATE }); this.nextPlayTime = 0; }
  _enqueueAudio(buf) {
    if (!this.playbackContext) return;
    const pcm = new Int16Array(buf), f32 = new Float32Array(pcm.length);
    for (let i = 0; i < pcm.length; i++) f32[i] = pcm[i] / 0x8000;
    this._curAudioChunks.push(f32);
    if (!this.audioOutput) return;
    const b = this.playbackContext.createBuffer(1, f32.length, OUTPUT_SAMPLE_RATE); b.copyToChannel(f32, 0);
    const s = this.playbackContext.createBufferSource(); s.buffer = b; s.playbackRate.value = this.playbackRate; s.connect(this.playbackContext.destination);
    const now = this.playbackContext.currentTime; if (this.nextPlayTime < now) this.nextPlayTime = now + 0.05;
    s.start(this.nextPlayTime); this.nextPlayTime += b.duration / this.playbackRate;
  }
  _finalizeAudio() {
    if (!this._curAudioChunks.length) return;
    const total = this._curAudioChunks.reduce((a, c) => a + c.length, 0), m = new Float32Array(total);
    let o = 0; for (const c of this._curAudioChunks) { m.set(c, o); o += c.length; }
    this._lastAudio = m; this._curAudioChunks = []; this.el.replayBtn.disabled = false;
  }
  _playBuffer(f32) {
    if (!f32 || !f32.length) return;
    let ctx = this.playbackContext; if (!ctx || ctx.state === "closed") ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: OUTPUT_SAMPLE_RATE });
    const b = ctx.createBuffer(1, f32.length, OUTPUT_SAMPLE_RATE); b.copyToChannel(f32, 0);
    const s = ctx.createBufferSource(); s.buffer = b; s.playbackRate.value = this.playbackRate; s.connect(ctx.destination); s.start();
  }
  _flushPlayback() { if (this.playbackContext) this.playbackContext.close().catch(() => {}); this._initPlayback(); }
  _replayLast() { if (this._lastAudio) this._playBuffer(this._lastAudio); }

  // ---- pronounce / summarize / ask / export -------------------------------
  async _pronounce() {
    const text = this._lastTranslationText.trim(); if (!text) return;
    const base = this._serverBase(); if (!base) return this._setStatus("error", t("st.setServer"));
    this.el.pronounceBox.classList.remove("hidden"); this.el.pronounceContent.textContent = "…";
    try {
      const r = await fetch(`${base}/api/pronounce`, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, script: this.el.scriptSelect.value, token: this._token() || undefined }) });
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.status);
      this.el.pronounceContent.textContent = (await r.json()).pronunciation || "—";
    } catch (e) { this.el.pronounceContent.textContent = "발음 변환 실패: " + e.message; }
  }
  async _summarize() {
    const log = this._currentLog(); if (!log.length) return;
    const base = this._serverBase(); if (!base) return this._setStatus("error", t("st.setServer"));
    const prev = this.el.noteSummarizeBtn.textContent; this.el.noteSummarizeBtn.disabled = true; this.el.noteSummarizeBtn.textContent = "…";
    try {
      const r = await fetch(`${base}/api/summarize`, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transcript: this._transcriptText(), language: this.el.langB.value, token: this._token() || undefined }) });
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.status);
      const summary = (await r.json()).summary || "";
      this.el.summaryContent.textContent = summary;
      if (this.context === "note") { const n = this._note(this.activeNoteId); if (n) { n.summary = summary; n.topic = summary.split("\n").map((l) => l.replace(/[#*\-•]/g, "").trim()).filter(Boolean)[0] || n.topic; this._saveNotes(); } }
      this._tab("summary");
    } catch (e) { this._setStatus("error", "요약 실패: " + e.message); }
    finally { this.el.noteSummarizeBtn.disabled = false; this.el.noteSummarizeBtn.textContent = prev; }
  }
  async _ask() {
    const q = this.el.askInput.value.trim(); if (!q) return;
    const log = this._currentLog(); if (!log.length) return;
    const base = this._serverBase(); if (!base) return this._setStatus("error", t("st.setServer"));
    this.el.askBtn.disabled = true; this.el.askAnswer.classList.remove("hidden"); this.el.askAnswer.textContent = "…";
    try {
      const r = await fetch(`${base}/api/ask`, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transcript: this._transcriptText(), question: q, language: this.el.langB.value, token: this._token() || undefined }) });
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.status);
      this.el.askAnswer.textContent = (await r.json()).answer || "—";
    } catch (e) { this.el.askAnswer.textContent = "질문 실패: " + e.message; }
    finally { this.el.askBtn.disabled = false; }
  }
  _transcriptText() {
    return this._currentLog().map((e) => {
      const tm = new Date(e.t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      const parts = []; if (e.source) parts.push(e.source); if (e.translation) parts.push("→ " + e.translation);
      return `[${tm}] ${parts.join("  ")}`;
    }).join("\n");
  }
  async _export(format) {
    if (this.el.noteMenu) this.el.noteMenu.hidden = true;
    const log = this._currentLog(); if (!log.length) return;
    const base = this._serverBase(); if (!base) return this._setStatus("error", t("st.setServer"));
    const entries = log.map((e) => {
      let tr = e.translation || "";
      if (e.multi) tr += "  " + Object.entries(e.multi).map(([c, x]) => `[${c}] ${x}`).join("  ");
      const warn = [];
      if (e.risk && e.risk.risk_level && e.risk.risk_level !== "none") warn.push(`[${e.risk.risk_level}] ${e.risk.subtitle_alert || ""}`);
      if (e.clarify && e.clarify.clarify_suspected) warn.push(`(?) ${e.clarify.clarify_did_you_mean || ""}`);
      return { time: new Date(e.t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }), source: e.source || "", translation: tr.trim(),
        risk: warn.join(" · ") };
    });
    const title = this.context === "note" ? (this._note(this.activeNoteId)?.title || "회의 노트") : "빠른 통역";
    try {
      const r = await fetch(`${base}/api/export`, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, entries, summary: this.el.summaryContent.textContent || "", format, token: this._token() || undefined }) });
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.status);
      const blob = await r.blob(), url = URL.createObjectURL(blob), a = document.createElement("a");
      a.href = url; a.download = `notes-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, "")}.${format}`;
      document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
    } catch (e) { this._setStatus("error", "내보내기 실패: " + e.message); }
  }
}

window.addEventListener("DOMContentLoaded", () => {
  if (!navigator.mediaDevices || !window.AudioContext) { alert("This browser does not support the Web Audio API."); return; }
  window.app = new App();
});
