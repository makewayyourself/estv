/**
 * Gemini Live Translator — menu-driven multi-view client.
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
    "home.translate": "실시간 통역", "home.translateDesc": "즉석 실시간 통역·자막 (저장 안 함)",
    "home.notes": "회의 노트", "home.notesDesc": "녹음·기록·요약 · 주제/날짜로 검색",
    "home.settings": "관리자 설정", "home.settingsDesc": "모든 기능 설정 · 서버·토큰 (운영자)",
    "home.room": "방 만들기 (QR)", "home.roomDesc": "QR로 참가자 초대 · 각자 자기 언어로 듣기",
    "title.room": "여러 기기 통역", "title.join": "통역 방 참여",
    "room.intro": "이 기기가 말하는 내용을 송출합니다. 참가자는 QR/링크로 들어와 각자 원하는 언어로 통역 음성을 듣습니다.",
    "room.start": "📡 방 시작 (마이크 송출)", "room.scan": "참가자가 이 QR을 스캔하거나 아래 링크로 접속하면 됩니다.",
    "room.copy": "복사", "room.participants": "참여자", "room.stop": "방 종료",
    "join.intro": "통역 방에 참여합니다. 들을 언어를 선택하세요.", "join.lang": "듣기 언어", "join.go": "🎧 참여하고 듣기",
    "join.listening": "듣는 중", "join.leave": "나가기",
    "home.meet": "다화자 회의 (QR 마이크)", "home.meetDesc": "참가자 폰이 각자 마이크 · 화자별 자막 자동 구분",
    "title.meet": "다화자 회의", "title.mjoin": "회의 참여 (마이크)",
    "meet.intro": "참가자들이 각자 휴대폰을 마이크로 사용합니다. 누가 말했는지 자동으로 구분되어 화자별 자막으로 표시됩니다.",
    "meet.lang": "공용 표시(번역) 언어", "meet.start": "👥 회의 시작 (QR 띄우기)",
    "meet.scan": "참가자가 이 QR을 스캔하면 자기 폰이 마이크가 됩니다.",
    "meet.save": "💾 노트로 저장", "meet.stop": "회의 종료", "meet.noteTitle": "다화자 회의",
    "mjoin.intro": "회의에 참여합니다. 당신의 휴대폰이 마이크가 되어, 말하면 자동으로 통역·기록됩니다.",
    "mjoin.name": "표시 이름 (선택)", "mjoin.namePh": "예: 홍길동 / 영업팀", "mjoin.go": "🎙️ 참여하고 말하기",
    "mjoin.you": "당신", "mjoin.speak": "말하면 자동으로 전송됩니다",
    "set.earphone": "🎧 이어폰(블루투스) 모드", "set.earphoneHelp": "블루투스 이어폰으로 소리가 안 나오면 켜세요. 에코 제거를 꺼서 이어폰으로 음성이 정상 출력됩니다(이어폰 착용 시에만 권장).", "set.earphoneApply": "이어폰 모드 변경 — 다시 시작하면 적용됩니다",
    "set.noiseGate": "🔇 주변 소음 필터", "set.noiseGateHelp": "작은 배경 소리(주변 대화·소음)를 걸러 보내지 않습니다. 엉뚱한 언어 인식·오번역을 줄여줍니다. 내 목소리가 잘리면 끄세요.",
    "set.glossary": "📖 용어집 — 이름·회사·전문용어 (오인식 자동 교정에 사용)", "set.glossaryPh": "예: 안현모(내 이름), ABC무역(회사), FOB, 선하증권",
    "note.audio": "🔊 회의 녹음", "note.audioDl": "다운로드", "note.audioDel": "삭제",
    "note.audioDelConfirm": "이 노트의 녹음 파일을 삭제할까요? (기록·요약은 유지됩니다)",
    "set.assistLang": "🌐 AI 보조 표시 언어 (내가 읽을 수 있는 언어)",
    "set.install": "앱 설치 (Android)", "set.installHelp": "최신 APK를 받아 폰에 설치하세요. 참가자에게 이 버튼/링크를 공유해도 됩니다.",
    "set.apkBtn": "📥 APK 다운로드", "set.apkCopy": "🔗 다운로드 링크 복사", "set.apkCopied": "✓ 복사됨",
    "st.hosting": "방송 중 — 말하세요", "st.joined": "연결됨 — 듣는 중", "st.roomEnded": "방이 종료/없음",
    "st.waking": "서버 연결 중… (무료 서버는 처음 30~60초 걸릴 수 있어요)", "st.timeout": "연결 시간 초과 — 서버가 깨어나는 중일 수 있어요. 잠시 후 다시 시도하세요.",
    "st.langSwitch": "출력 언어 변경 중…", "tr.outLang": "🔊 출력 언어",
    "mode.prec": "🎯 고정밀", "tr.inLang": "🎙️ 입력",
    "st.precLive": "🎯 고정밀 자막 — 문장을 마치면 자막이 표시됩니다",
    "tr.duo": "🪟 대면 자막", "duo.intro": "투명 스크린(빔프로젝터)용 마주보기 자막: 위·아래에 서로 다른 언어가 뜨고, 아래쪽은 유리 반대편에서 읽히도록 반전됩니다.",
    "duo.top": "위쪽(내 앞) 언어", "duo.bottom": "아래쪽(맞은편) 언어", "duo.flip": "아래쪽 표시 방향 (프로젝터 설치에 맞게)",
    "duo.flipMirror": "좌우 반전 (거울)", "duo.flipRotate": "180° 회전", "duo.flipNone": "그대로",
    "duo.start": "🪟 대면 자막 시작", "duo.exitHint": "화면을 탭하면 종료",
    "tr.room": "🔗 방 만들기 (QR)", "tr.save": "💾 노트로 저장", "st.savedNote": "노트로 저장됨", "msg.nothingSave": "저장할 내용이 없습니다",
    "set.aiAssist": "AI 보조 (실시간)", "set.answer": "💬 답변 제안", "set.upgrade": "✨ 표현 업그레이드",
    "note.tabFeedback": "피드백", "note.feedbackBtn": "📊 피드백 생성 / 갱신",
    "assist.answer": "답변 제안", "assist.upgrade": "표현 업그레이드", "assist.speak": "🔊 들려주기", "assist.copy": "복사", "msg.feedbackFail": "피드백 실패",
    "home.tagline": "Gemini Live Translator · 자동 언어 감지 · 다국어 자막",
    "tr.ph": "아래 시작을 누르고 말하면 통역됩니다.<br />표시 언어는 설정에서 최대 3개까지 고를 수 있어요.",
    "notes.search": "주제·내용·날짜 검색", "notes.empty": "아직 노트가 없습니다. 아래 \"새 노트\"로 시작하세요.", "notes.new": "＋ 새 노트",
    "note.tabLog": "기록", "note.tabSummary": "요약", "vm.both": "원문+번역", "vm.source": "원문만", "vm.trans": "번역만",
    "note.summarize": "요약", "note.delete": "삭제", "note.ph": "아래 <b>시작</b>을 누르면 녹음·통역이 시작되고 여기에 기록됩니다.",
    "qa.ph": "예: 결제 조건이 어떻게 정해졌지?", "qa.ask": "질문",
    "set.uiLang": "앱 언어", "set.theme": "테마", "set.dark": "🌙 밤(어둡게)", "set.light": "☀️ 낮(밝게)",
    "set.fontSize": "글자 크기", "set.output": "출력(들리는) 언어", "set.outputHelp": "음성으로 나갈 1개 언어.",
    "set.auto": "자동 감지", "set.source": "말하는(입력) 언어", "set.sourceHelp": "보통 '자동 감지'면 됩니다. 원문 자막이 엉뚱한 언어로 뜨면(예: 아랍어) 여기서 말하는 언어를 직접 지정하세요.",
    "set.display": "표시(자막) 언어 · 최대 3", "set.displayHelp": "각 발화를 이 언어들로 함께 자막 표시(참가자 각자 읽기).",
    "set.voice": "음성 (목소리)", "set.speed": "빠르기", "set.risk": "🛡️ 리스크 감지", "set.riskCtx": "분야 (예: oil trading)",
    "set.clarify": "🔎 의미 확인", "set.server": "서버 주소", "set.token": "접속 토큰", "set.tokenPh": "서버 접속 토큰",
    "set.save": "저장", "set.health": "🔄 진단", "set.model": "모델", "set.admin": "🔒 관리자 · 서버 설정",
    "admin.note": "서버에 연결하려면 주소와 토큰을 입력 후 저장하세요. (호스트/운영자용 · 참가자는 QR로 들어오므로 불필요)", "title.admin": "관리자 설정",
    "vm.label": "보기 모드", "note.save": "저장(내보내기)", "note.summarizeBtn": "요약 생성 / 갱신",
    "mode.audio": "🔊 음성+자막", "mode.text": "📝 자막만", "mode.focus": "🌙 집중",
    "focus.live": "통역 중", "focus.hint": "화면을 탭하면 해제됩니다",
    "ctrl.start": "시작", "ctrl.stop": "정지",
    "pron.title": "발음", "pron.roman": "로마자", "pron.hangul": "한글",
    "title.home": "Gemini Live Translator", "title.translate": "빠른 통역", "title.notes": "회의 노트", "title.settings": "관리자 설정",
    "st.idle": "대기", "st.connecting": "연결 중…", "st.live": "통역 중 — 말하세요", "st.paused": "일시정지",
    "st.saved": "저장됨", "st.tokenBad": "Access Token이 서버와 불일치", "st.cantReach": "서버 연결 불가",
    "st.noKey": "서버에 API 키 미설정", "st.setServer": "설정에서 서버 주소를 입력하세요",
    "msg.confirmDelete": "이 노트를 삭제할까요?", "msg.newNote": "새 노트",
  },
  en: {
    "home.translate": "Live Translate", "home.translateDesc": "Instant live interpreting · captions (not saved)",
    "home.notes": "Meeting Notes", "home.notesDesc": "Record · log · summary · search by topic/date",
    "home.settings": "Admin · Settings", "home.settingsDesc": "Language · theme · server · caption langs",
    "home.room": "Create room (QR)", "home.roomDesc": "Invite via QR · everyone hears their own language",
    "title.room": "Multi-device", "title.join": "Join a room",
    "room.intro": "This device broadcasts what you speak. Participants join via QR/link and hear it in their own language.",
    "room.start": "📡 Start room (broadcast mic)", "room.scan": "Participants scan this QR or open the link below.",
    "room.copy": "Copy", "room.participants": "Participants", "room.stop": "End room",
    "join.intro": "Join an interpreting room. Choose the language to hear.", "join.lang": "Listen in", "join.go": "🎧 Join & listen",
    "join.listening": "Listening", "join.leave": "Leave",
    "home.meet": "Multi-speaker meeting (QR mics)", "home.meetDesc": "Each phone is a mic · captions auto-labelled by speaker",
    "title.meet": "Multi-speaker meeting", "title.mjoin": "Join meeting (mic)",
    "meet.intro": "Each participant uses their own phone as a microphone. Who said what is detected automatically and shown as per-speaker captions.",
    "meet.lang": "Shared display (translation) language", "meet.start": "👥 Start meeting (show QR)",
    "meet.scan": "When a participant scans this QR, their phone becomes a microphone.",
    "meet.save": "💾 Save as note", "meet.stop": "End meeting", "meet.noteTitle": "Multi-speaker meeting",
    "mjoin.intro": "Join the meeting. Your phone becomes a mic — speak and it's translated and logged automatically.",
    "mjoin.name": "Display name (optional)", "mjoin.namePh": "e.g. Alex / Sales", "mjoin.go": "🎙️ Join & speak",
    "mjoin.you": "You", "mjoin.speak": "Speak — audio is sent automatically",
    "set.earphone": "🎧 Earphone (Bluetooth) mode", "set.earphoneHelp": "Turn on if sound doesn't reach your Bluetooth earphones. Disables echo cancellation so audio routes to the earphones (recommended only while wearing earphones).", "set.earphoneApply": "Earphone mode changed — restart to apply",
    "set.noiseGate": "🔇 Ambient noise filter", "set.noiseGateHelp": "Skips quiet background sound (nearby chatter/noise) so it never reaches the model — reduces wrong-language detection and mistranslation. Turn off if your own voice gets clipped.",
    "set.glossary": "📖 Glossary — names, companies, terms (used to auto-correct mishearings)", "set.glossaryPh": "e.g. Hyunmo Ahn (my name), ABC Trading (company), FOB, B/L",
    "note.audio": "🔊 Meeting audio", "note.audioDl": "Download", "note.audioDel": "Delete",
    "note.audioDelConfirm": "Delete this note's recording? (The transcript and summary are kept.)",
    "set.assistLang": "🌐 AI assist language (the language YOU read)",
    "set.install": "Install app (Android)", "set.installHelp": "Download the latest APK and install it on your phone. Share this button/link with participants too.",
    "set.apkBtn": "📥 Download APK", "set.apkCopy": "🔗 Copy download link", "set.apkCopied": "✓ Copied",
    "st.hosting": "Broadcasting — speak now", "st.joined": "Connected — listening", "st.roomEnded": "Room ended / not found",
    "st.waking": "Connecting… (a free server can take 30–60s the first time)", "st.timeout": "Connection timed out — the server may be waking up. Try again shortly.",
    "st.langSwitch": "Switching output language…", "tr.outLang": "🔊 Output",
    "mode.prec": "🎯 Precision", "tr.inLang": "🎙️ Input",
    "st.precLive": "🎯 Precision captions — text appears when you finish a sentence",
    "tr.duo": "🪟 Glass captions", "duo.intro": "Face-to-face captions for a beam projector on transparent glass: top and bottom show different languages; the bottom is flipped so it reads correctly through the glass.",
    "duo.top": "Top (my side) language", "duo.bottom": "Bottom (far side) language", "duo.flip": "Bottom orientation (match your projector)",
    "duo.flipMirror": "Mirrored", "duo.flipRotate": "Rotated 180°", "duo.flipNone": "As-is",
    "duo.start": "🪟 Start glass captions", "duo.exitHint": "Tap to exit",
    "tr.room": "🔗 Create room (QR)", "tr.save": "💾 Save as note", "st.savedNote": "Saved as a note", "msg.nothingSave": "Nothing to save",
    "set.aiAssist": "AI assist (live)", "set.answer": "💬 Answer suggestion", "set.upgrade": "✨ Expression upgrade",
    "note.tabFeedback": "Feedback", "note.feedbackBtn": "📊 Generate / refresh feedback",
    "assist.answer": "Answer suggestion", "assist.upgrade": "Expression upgrade", "assist.speak": "🔊 Speak", "assist.copy": "Copy", "msg.feedbackFail": "Feedback failed",
    "home.tagline": "Gemini Live Translator · auto language detection · multilingual captions",
    "tr.ph": "Tap Start below and speak to translate.<br />Pick up to 3 caption languages in Settings.",
    "notes.search": "Search topic · content · date", "notes.empty": "No notes yet. Tap \"New note\" below to start.", "notes.new": "＋ New note",
    "note.tabLog": "Log", "note.tabSummary": "Summary", "vm.both": "Source+Translation", "vm.source": "Source only", "vm.trans": "Translation only",
    "note.summarize": "Summarize", "note.delete": "Delete", "note.ph": "Tap <b>Start</b> below to begin recording — it appears here.",
    "qa.ph": "e.g. What payment terms were agreed?", "qa.ask": "Ask",
    "set.uiLang": "App language", "set.theme": "Theme", "set.dark": "🌙 Dark", "set.light": "☀️ Light",
    "set.fontSize": "Font size", "set.output": "Output (spoken) language", "set.outputHelp": "One spoken language.",
    "set.auto": "Auto-detect", "set.source": "Spoken (input) language", "set.sourceHelp": "Usually 'Auto-detect' is fine. If the original-text caption shows the wrong language (e.g. Arabic), set the spoken language here.",
    "set.display": "Caption languages · up to 3", "set.displayHelp": "Show each utterance in these languages (everyone reads their own).",
    "set.voice": "Voice", "set.speed": "Speed", "set.risk": "🛡️ Risk Guard", "set.riskCtx": "Industry (e.g. oil trading)",
    "set.clarify": "🔎 Clarify", "set.server": "Server URL", "set.token": "Access token", "set.tokenPh": "server access token",
    "set.save": "Save", "set.health": "🔄 Check", "set.model": "Model", "set.admin": "🔒 Admin · server settings",
    "admin.note": "Enter the server URL and token, then Save, to connect. (Host/operator only — participants join via QR.)", "title.admin": "Admin",
    "vm.label": "View mode", "note.save": "Export", "note.summarizeBtn": "Generate / refresh summary",
    "mode.audio": "🔊 Audio + captions", "mode.text": "📝 Captions only", "mode.focus": "🌙 Focus",
    "focus.live": "Translating", "focus.hint": "Tap the screen to exit",
    "ctrl.start": "Start", "ctrl.stop": "Stop",
    "pron.title": "Pronunciation", "pron.roman": "Roman", "pron.hangul": "Hangul",
    "title.home": "Gemini Live Translator", "title.translate": "Quick Translate", "title.notes": "Meeting Notes", "title.settings": "Admin · Settings",
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

// --- Meeting-audio store -----------------------------------------------------
// Recordings stay ON DEVICE (IndexedDB), never uploaded: business conversations
// are sensitive, and the free-tier server has an ephemeral disk anyway (files
// would vanish on every deploy). A recording lives and dies with its note —
// deleting the note deletes the audio. No auto-expiry, no delete-on-download
// (users re-listen to dubious passages repeatedly); manual delete is provided.
const AUDIO_DB = { name: "estvAudio", store: "rec" };
function idbOpen() { return new Promise((res, rej) => { const r = indexedDB.open(AUDIO_DB.name, 1); r.onupgradeneeded = () => r.result.createObjectStore(AUDIO_DB.store); r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error); }); }
async function idbPut(key, blob) { const db = await idbOpen(); return new Promise((res, rej) => { const tx = db.transaction(AUDIO_DB.store, "readwrite"); tx.objectStore(AUDIO_DB.store).put(blob, key); tx.oncomplete = () => res(); tx.onerror = () => rej(tx.error); }); }
async function idbGet(key) { const db = await idbOpen(); return new Promise((res, rej) => { const rq = db.transaction(AUDIO_DB.store).objectStore(AUDIO_DB.store).get(key); rq.onsuccess = () => res(rq.result || null); rq.onerror = () => rej(rq.error); }); }
async function idbDel(key) { const db = await idbOpen(); return new Promise((res, rej) => { const tx = db.transaction(AUDIO_DB.store, "readwrite"); tx.objectStore(AUDIO_DB.store).delete(key); tx.oncomplete = () => res(); tx.onerror = () => rej(tx.error); }); }
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
    this.transcriptEl = null;

    this._cache();
    this._bind();
    this._applyAppearance();
    this._populateLangs();
    this._refreshHealth();
    const qs = new URLSearchParams(location.search);
    const room = qs.get("room"), meet = qs.get("meet");
    if (meet) this.show("mjoin", { room: meet });
    else if (room) this.show("join", { room });
    else this.show("home");
  }

  _cache() {
    this.el = {};
    ["backBtn","viewTitle","statusDot","statusText","controlBar","newNoteBtn",
     "toggleBtn","toggleIcon","toggleLabel","pauseBtn","replayBtn","pronounceBtn",
     "pronounceBox","pronounceContent","scriptSelect","modeAudioBtn","modeTextBtn","modePrecBtn","precSrcSel","precSrcLabel",
     "qkTranscript","noteTranscript","assistCards",
     "focusBtn","focusOverlay","focusCaption",
     "noteTitle","noteDate","noteMenuBtn","noteMenu","noteSummarizeBtn","noteExportMd","noteExportDocx","noteExportPdf","noteDelete",
     "noteAudioRow","noteAudioInfo","noteAudioDl","noteAudioDel",
     "homeVer","verInfo",
     "roomStartBtn","roomIdle","roomLive","roomQr","roomLink","roomCopyBtn","roomCount","roomStopBtn",
     "joinForm","joinLang","joinBtn","joinLive","joinTranscript","joinLangLabel","joinLeaveBtn",
     "meetIdle","meetLang","meetStartBtn","meetLive","meetQr","meetLink","meetCopyBtn","meetRoster","meetTranscript","meetSaveBtn","meetStopBtn",
     "mjoinForm","mjoinName","mjoinBtn","mjoinLive","mjoinLabel","mjoinSpeak","mjoinTranscript","mjoinLeaveBtn",
     "apkDownloadBtn","apkCopyBtn",
     "qkRoomBtn","qkSaveBtn","qkLangSel",
     "qkDuoBtn","duoConfig","duoTopLang","duoBottomLang","duoFlipSel","duoStartBtn","duoOverlay","duoTop","duoBottom",
     "askInput","askBtn","askAnswer","summaryContent","viewMode","noteFeedbackBtn","feedbackContent",
     "answerToggle","upgradeToggle",
     "notesList","notesEmpty","noteSearch",
     "uiLang","themeSel","fontSel","langB","displayLang1","displayLang2","displayLang3",
     "voiceSelect","speedRange","speedValue","riskToggle","riskContext","clarifyToggle","earphoneToggle","noiseGateToggle","glossaryInput","assistLang",
     "serverUrl","accessToken","saveServerBtn","modelInfo",
     "operatorBox","healthBtn","healthStatus","buildTap","setVer"].forEach((id) => (this.el[id] = $(id)));
  }

  // ---- navigation ---------------------------------------------------------
  show(view, opts = {}) {
    // leaving a recording-capable view while live → stop / close connections
    if (this.view === "room" && view !== "room") this._roomStop(true);
    if (this.view === "join" && view !== "join") this._joinLeave(true);
    if (this.view === "meet" && view !== "meet") this._meetStop(true);
    if (this.view === "mjoin" && view !== "mjoin") this._mjoinLeave(true);
    if (this.running && view !== this.view) this.stop();
    if (this._focusOn) this._exitFocus();

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
      this.transcriptEl = this.el.qkTranscript; this._clearAssist();
      this._setMode(localStorage.getItem("audioOutput") !== "0", true);
      this.el.viewTitle.innerHTML = t("title.translate");
    } else if (view === "note") {
      this.context = "note"; this.activeNoteId = opts.id;
      this.audioOutput = false; // notes never play audio
      this.transcriptEl = this.el.noteTranscript; this._clearAssist();
      this._openNote(opts.id);
    } else {
      this.el.viewTitle.textContent = t("title." + view);
      if (view === "notes") this._renderNotesList();
      if (view === "room") { this.el.roomIdle.hidden = false; this.el.roomLive.hidden = true; }
      if (view === "join") this._joinSetup(opts.room || this.joinRoom);
      if (view === "meet") this._meetSetup();
      if (view === "mjoin") this._mjoinSetup(opts.room || this.meetRoom);
    }
    if (view === "home") this.el.viewTitle.innerHTML = `<span class="bg-gradient-to-r from-sky-400 to-indigo-400 bg-clip-text text-transparent">${t("title.home")}</span>`;
    this._refreshToggle();
  }

  // ---- binding ------------------------------------------------------------
  _bind() {
    document.querySelectorAll("[data-go]").forEach((b) => b.addEventListener("click", () => this.show(b.getAttribute("data-go"))));
    this.el.backBtn.addEventListener("click", () => this.show(this.view === "note" ? "notes" : this.view === "room" ? "translate" : "home"));
    this.el.noteMenuBtn.addEventListener("click", () => (this.el.noteMenu.hidden = !this.el.noteMenu.hidden));
    // quick translate: create a room / save as a note
    this.el.qkRoomBtn.addEventListener("click", () => this.show("room"));
    this.el.qkSaveBtn.addEventListener("click", () => this._saveQuickAsNote());
    // rooms
    this.el.roomStartBtn.addEventListener("click", () => this._roomStart());
    this.el.roomStopBtn.addEventListener("click", () => this._roomStop(false));
    this.el.roomCopyBtn.addEventListener("click", () => { navigator.clipboard && navigator.clipboard.writeText(this.el.roomLink.value).catch(() => {}); });
    this.el.joinBtn.addEventListener("click", () => this._joinStart());
    this.el.joinLeaveBtn.addEventListener("click", () => this._joinLeave(false));
    // multi-mic meeting (diarized)
    this.el.meetStartBtn.addEventListener("click", () => this._meetStart());
    this.el.meetStopBtn.addEventListener("click", () => this._meetStop(false));
    this.el.meetCopyBtn.addEventListener("click", () => { navigator.clipboard && navigator.clipboard.writeText(this.el.meetLink.value).catch(() => {}); });
    this.el.meetSaveBtn.addEventListener("click", () => this._meetSaveNote());
    this.el.mjoinBtn.addEventListener("click", () => this._mjoinStart());
    this.el.mjoinLeaveBtn.addEventListener("click", () => this._mjoinLeave(false));
    // transparent-screen counter display (duo captions)
    this.el.qkDuoBtn.addEventListener("click", () => { this.el.duoConfig.hidden = !this.el.duoConfig.hidden; });
    this.el.duoStartBtn.addEventListener("click", () => this._duoStart());
    this.el.duoOverlay.addEventListener("click", () => this._duoExit());
    this.el.apkCopyBtn.addEventListener("click", () => {
      const url = this.el.apkDownloadBtn.getAttribute("href");
      if (navigator.clipboard) navigator.clipboard.writeText(url).then(() => { const b = this.el.apkCopyBtn, p = b.textContent; b.textContent = t("set.apkCopied"); setTimeout(() => { b.textContent = p; }, 1500); }).catch(() => {});
    });

    this.el.toggleBtn.addEventListener("click", () => (this.running ? this.stop() : this.start()));
    this.el.pauseBtn.addEventListener("click", () => this._togglePause());
    this.el.replayBtn.addEventListener("click", () => this._replayLast());
    this.el.pronounceBtn.addEventListener("click", () => this._pronounce());
    this.el.scriptSelect.addEventListener("change", () => this._lastTranslationText && this._pronounce());
    this.el.modeAudioBtn.addEventListener("click", () => this._setMode(true));
    this.el.modeTextBtn.addEventListener("click", () => this._setMode(false));
    this.el.modePrecBtn.addEventListener("click", () => this._setPrec(!this.precision));
    this.el.focusBtn.addEventListener("click", () => this._enterFocus());
    this.el.focusOverlay.addEventListener("click", () => this._exitFocus());

    // notes
    this.el.newNoteBtn.addEventListener("click", () => this._newNote());
    this.el.noteSearch.addEventListener("input", () => this._renderNotesList());
    this.el.noteTitle.addEventListener("change", () => this._renameActive());
    this.el.noteDelete.addEventListener("click", () => this._deleteActive());
    this.el.noteAudioDl.addEventListener("click", async () => {
      const blob = await idbGet(this.activeNoteId).catch(() => null); if (!blob) return;
      const url = URL.createObjectURL(blob), a = document.createElement("a");
      a.href = url; a.download = `${(this._note(this.activeNoteId)?.title || "meeting").replace(/[\\/:*?"<>|]/g, "_")}.webm`;
      document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 10000);
    });
    this.el.noteAudioDel.addEventListener("click", async () => {
      if (!confirm(t("note.audioDelConfirm"))) return;
      await idbDel(this.activeNoteId).catch(() => {}); this._refreshAudioRow();
    });
    this.el.noteSummarizeBtn.addEventListener("click", () => this._summarize());
    this.el.noteFeedbackBtn.addEventListener("click", () => this._feedback());
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
    this.el.clarifyToggle.checked = localStorage.getItem("clarify") !== "0"; // default ON — mishearing names is fatal in business
    this.el.answerToggle.checked = localStorage.getItem("answer") === "1";
    this.el.upgradeToggle.checked = localStorage.getItem("upgrade") === "1";
    this.el.earphoneToggle.checked = localStorage.getItem("earphoneMode") === "1";
    this.el.noiseGateToggle.checked = localStorage.getItem("noiseGate") !== "0";
    this.el.noiseGateToggle.addEventListener("change", () => localStorage.setItem("noiseGate", this.el.noiseGateToggle.checked ? "1" : "0"));
    this.el.glossaryInput.value = localStorage.getItem("glossary") || "";
    this.el.glossaryInput.addEventListener("change", () => localStorage.setItem("glossary", this.el.glossaryInput.value.trim()));
    this.el.riskContext.value = localStorage.getItem("riskContext") || "";
    this.el.riskToggle.addEventListener("change", () => localStorage.setItem("riskGuard", this.el.riskToggle.checked ? "1" : "0"));
    this.el.clarifyToggle.addEventListener("change", () => localStorage.setItem("clarify", this.el.clarifyToggle.checked ? "1" : "0"));
    this.el.answerToggle.addEventListener("change", () => localStorage.setItem("answer", this.el.answerToggle.checked ? "1" : "0"));
    this.el.upgradeToggle.addEventListener("change", () => localStorage.setItem("upgrade", this.el.upgradeToggle.checked ? "1" : "0"));
    this.el.earphoneToggle.addEventListener("change", () => {
      localStorage.setItem("earphoneMode", this.el.earphoneToggle.checked ? "1" : "0");
      if (this.running || this.meetWs) this._setStatus(this.running ? "live" : "idle", t("set.earphoneApply"));
    });
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
    // Operator section unlock: tap the build version 5 times.
    let taps = 0, last = 0;
    this.el.buildTap.addEventListener("click", () => {
      const now = performance.now();
      taps = now - last < 1200 ? taps + 1 : 1; last = now;
      if (taps >= 5) { this.el.operatorBox.hidden = false; taps = 0; if (navigator.vibrate) navigator.vibrate(30); }
    });
    this.el.healthBtn.addEventListener("click", () => this._healthCheck());
  }

  async _healthCheck() {
    const base = this._serverBase();
    if (!base) { this.el.healthStatus.textContent = "🔴 " + t("st.setServer"); return; }
    const ko = UI_LANG === "ko";
    // Free-tier servers sleep; the first request must spin the container up
    // (30-60s). Retry a few times — each attempt keeps the wake-up going —
    // instead of giving up after one and falsely reporting "unreachable".
    const tries = 3;
    let lastErr = "";
    for (let i = 1; i <= tries; i++) {
      this.el.healthStatus.textContent = i === 1 ? "⏳ …" : `⏳ ${ko ? "서버 깨우는 중" : "waking server"}… (${i}/${tries})`;
      try {
        const ctrl = new AbortController(); const to = setTimeout(() => ctrl.abort(), 25000);
        const r = await fetch(`${base}/api/health`, { signal: ctrl.signal }); clearTimeout(to);
        if (!r.ok) throw new Error("HTTP " + r.status);
        const d = await r.json();
        if (this.el.homeVer) this.el.homeVer.textContent = d.version || "?";
        if (this.el.verInfo) this.el.verInfo.textContent = d.version || "?";
        if (this.el.setVer) this.el.setVer.textContent = d.version || "?";
        if (d.model) this.el.modelInfo.textContent = d.model;
        this.el.healthStatus.textContent = `🟢 ${ko ? "정상" : "OK"} · ${d.version || ""}${d.api_key_configured ? "" : (ko ? " · ⚠️ 키 미설정" : " · ⚠️ no key")}`;
        return;
      } catch (e) {
        lastErr = e.name === "AbortError" ? (ko ? "응답 없음(시간 초과)" : "timeout") : e.message;
      }
    }
    this.el.healthStatus.textContent = "🔴 " + (ko
      ? `연결 실패 (${lastErr}) — Render 서버가 꺼져 있거나 배포가 실패했을 수 있어요. Render 대시보드에서 서비스 상태를 확인하세요.`
      : `Unreachable (${lastErr}) — the Render server may be down or the deploy failed. Check the Render dashboard.`);
  }

  _applyAppearance() {
    applyI18n();
    this._applyTheme(localStorage.getItem("theme") || "dark");
    this._applyFont(localStorage.getItem("fontSize") || "md");
    this.el.speedRange.value = String(this.playbackRate);
    this.el.speedValue.textContent = `${this.playbackRate.toFixed(2)}×`;
    this.el.viewMode.value = this.viewMode;
    this._setMode(this.audioOutput, true);
    this._setPrec(localStorage.getItem("precision") === "1", true);
  }
  _applyTheme(theme) { document.body.classList.toggle("light", theme === "light"); localStorage.setItem("theme", theme); this.el.themeSel.value = theme; }
  _applyFont(size) { ["sm","md","lg","xl"].forEach((s) => { this.el.qkTranscript.classList.toggle("fs-" + s, s === size); this.el.noteTranscript.classList.toggle("fs-" + s, s === size); }); localStorage.setItem("fontSize", size); this.el.fontSel.value = size; }
  _setMode(audio, silent) {
    this.audioOutput = audio; if (!silent) localStorage.setItem("audioOutput", audio ? "1" : "0");
    if (!silent && this.precision) this._setPrec(false, true); // audio/text picks leave precision
    this._paintModes();
    if (!audio) this._flushPlayback();
  }
  _setPrec(on, silent) {
    // Precision-caption cascade (accuracy-first, Chattr-style): per-utterance
    // transcription+translation on the stable model, language pinned, glossary
    // injected into recognition, no audio playback (no echo by construction).
    this.precision = !!on;
    if (!silent) localStorage.setItem("precision", on ? "1" : "0");
    if (on) this.audioOutput = false;
    this.el.precSrcSel.hidden = this.el.precSrcLabel.hidden = !on;
    this._paintModes();
    if (this.running && !silent) { // switch engines live
      this._setStatus("connecting", t("st.langSwitch"));
      this.stop().then(() => this.start());
    }
  }
  _paintModes() {
    const on = "border-indigo-500 bg-indigo-600 text-white", off = "border-slate-700 bg-slate-800 text-slate-300";
    const prec = this.precision, audio = this.audioOutput;
    this.el.modeAudioBtn.className = `rounded-lg border px-3 py-1.5 text-xs font-medium transition ${!prec && audio ? on : off}`;
    this.el.modeTextBtn.className = `rounded-lg border px-3 py-1.5 text-xs font-medium transition ${!prec && !audio ? on : off}`;
    this.el.modePrecBtn.className = `rounded-lg border px-3 py-1.5 text-xs font-medium transition ${prec ? "border-emerald-500 bg-emerald-600 text-white" : off}`;
  }

  // ---- languages ----------------------------------------------------------
  _populateLangs() {
    const fillB = (sel, sv) => { sel.innerHTML = ""; for (const [c, l] of Object.entries(LANGUAGES)) { const o = document.createElement("option"); o.value = c; o.textContent = l; if (c === sv) o.selected = true; sel.appendChild(o); } };
    fillB(this.el.langB, localStorage.getItem("langB") || DEFAULT_LANG_B);
    this.el.langB.addEventListener("change", () => { localStorage.setItem("langB", this.el.langB.value); this.el.qkLangSel.value = this.el.langB.value; });
    // In-place output-language switch on the Quick Translate screen: going to
    // Settings drops the session to idle, so switch here and — if a session is
    // live — transparently reconnect with the new target language (the Gemini
    // config is fixed per connection).
    fillB(this.el.qkLangSel, localStorage.getItem("langB") || DEFAULT_LANG_B);
    // Precision-mode input language: pinning it is a main accuracy lever.
    const savedSrc = localStorage.getItem("precSrc") || "auto";
    this.el.precSrcSel.innerHTML = "";
    for (const [c, l] of [["auto", t("set.auto")], ...Object.entries(LANGUAGES)]) {
      const o = document.createElement("option"); o.value = c; o.textContent = l; if (c === savedSrc) o.selected = true; this.el.precSrcSel.appendChild(o);
    }
    this.el.precSrcSel.addEventListener("change", () => localStorage.setItem("precSrc", this.el.precSrcSel.value));
    fillB(this.el.duoTopLang, localStorage.getItem("duoTop") || "en");
    fillB(this.el.duoBottomLang, localStorage.getItem("duoBottom") || (UI_LANG === "ko" ? "ko" : DEFAULT_LANG_B));
    this.el.duoFlipSel.value = localStorage.getItem("duoFlip") || "mirror";
    this.el.duoTopLang.addEventListener("change", () => localStorage.setItem("duoTop", this.el.duoTopLang.value));
    this.el.duoBottomLang.addEventListener("change", () => localStorage.setItem("duoBottom", this.el.duoBottomLang.value));
    this.el.duoFlipSel.addEventListener("change", () => localStorage.setItem("duoFlip", this.el.duoFlipSel.value));
    this.el.qkLangSel.addEventListener("change", async () => {
      const v = this.el.qkLangSel.value;
      this.el.langB.value = v; localStorage.setItem("langB", v);
      if (this.running) {
        this._setStatus("connecting", t("st.langSwitch"));
        await this.stop();
        this.start();
      }
    });
    // Language the AI-assist cards (answer/clarify/risk) are written in —
    // the language the USER reads, independent of the app UI language.
    fillB(this.el.assistLang, localStorage.getItem("assistLang") || UI_LANG);
    this.el.assistLang.addEventListener("change", () => localStorage.setItem("assistLang", this.el.assistLang.value));
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
      if (this.el.setVer) this.el.setVer.textContent = d.version || "?";
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
  /** Snapshot the current Quick Translate transcript into a saved meeting note. */
  _saveQuickAsNote() {
    if (!this.quickLog.length) { this._setStatus(this.running ? "live" : "idle", t("msg.nothingSave")); return; }
    const log = this.quickLog.map((e) => ({ ...e }));
    const title = log[0] && log[0].source ? log[0].source.slice(0, 24) : t("msg.newNote");
    const id = "n" + Date.now();
    this.notes.unshift({ id, title, createdAt: Date.now(), updatedAt: Date.now(), log, summary: "", topic: "" });
    this._saveNotes();
    if (this._lastRecBlob) { idbPut(id, this._lastRecBlob).catch(() => {}); this._lastRecBlob = null; }
    this._setStatus(this.running ? "live" : "idle", t("st.savedNote"));
  }
  _deleteActive() {
    if (!confirm(t("msg.confirmDelete"))) return;
    idbDel(this.activeNoteId).catch(() => {}); // recording lives and dies with its note
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
    this.el.feedbackContent.textContent = n.feedback || "";
    this.el.askAnswer.classList.add("hidden");
    this._tab("log");
    this._renderLog(this.el.noteTranscript, n.log, "note-ph");
    this._refreshAudioRow();
  }
  _refreshAudioRow() {
    if (!this.el.noteAudioRow) return;
    idbGet(this.activeNoteId).then((b) => {
      this.el.noteAudioRow.hidden = !b;
      this.el.noteAudioInfo.textContent = b ? `(${(b.size / 1048576).toFixed(1)} MB)` : "";
    }).catch(() => { this.el.noteAudioRow.hidden = true; });
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
    wrap.className = isTr ? "flex flex-col items-end" : "flex flex-col items-start";
    const tag = document.createElement("span");
    tag.className = "mb-0.5 px-1 text-[10px] font-semibold uppercase tracking-wide " + (isTr ? "text-indigo-300" : "text-slate-500");
    tag.textContent = isTr ? (UI_LANG === "ko" ? "번역" : "Translation") : (UI_LANG === "ko" ? "원문" : "Source");
    const b = document.createElement("div");
    b.className = isTr ? "max-w-[85%] rounded-2xl rounded-tr-sm bg-indigo-600/90 px-4 py-2.5 text-white shadow-sm"
                       : "max-w-[85%] rounded-2xl rounded-tl-sm bg-slate-800 px-4 py-2.5 text-slate-200 shadow-sm";
    b.textContent = text; wrap.appendChild(tag); wrap.appendChild(b); return wrap;
  }
  _showRole(role) {
    // View-mode only filters meeting notes; Quick Translate always shows all.
    if (this.context !== "note") return true;
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
    this._autoScroll(el);
  }
  _multiBlock(tr) {
    const w = document.createElement("div"); w.dataset.row = "1";
    w.className = "rounded-lg bg-slate-800/60 px-3 py-2";
    w.innerHTML = Object.entries(tr).map(([c, x]) => `<div class="flex gap-2"><span class="shrink-0 font-mono text-[10px] uppercase text-sky-400">${c}</span><span class="text-slate-300">${esc(x)}</span></div>`).join("");
    return w;
  }

  _autoScroll(el) {
    if (!el) return;
    // The transcript isn't a fixed-height scroll box — it grows the page — so
    // element.scrollTop is a no-op; the window is the actual scroller. Nudge the
    // window to the bottom so new captions stay visible. (If a layout ever makes
    // the element itself scrollable, honor that too.)
    if (el.scrollHeight > el.clientHeight + 4) el.scrollTop = el.scrollHeight;
    requestAnimationFrame(() => window.scrollTo(0, document.documentElement.scrollHeight));
  }

  _appendTranscript(role, text) {
    if (!text) return;
    this._hidePh(this.transcriptEl);
    if (role === "translation") this._curTranslation += text; else this._curSource += text;
    clearTimeout(this._turnTimer);
    this._turnTimer = setTimeout(() => this._finalizeTurn(), 1800);
    if (!this._showRole(role)) return;
    const key = role === "translation" ? "_trLine" : "_srcLine";
    if (!this[key]) { const w = this._bubble(role, ""); this.transcriptEl.appendChild(w); this[key] = w.lastElementChild; }
    this[key].textContent += text;
    this._autoScroll(this.transcriptEl);
    if (this._focusOn && key === "_trLine") this.el.focusCaption.textContent = this[key].textContent;
  }

  // ---- focus / screen-off mode -------------------------------------------
  // Dims the screen to save battery during long meetings; audio keeps running.
  // Wake Lock prevents the OS sleeping and cutting the mic/session. Tap exits.
  async _enterFocus() {
    if (this.view !== "translate") return;
    this._focusOn = true;
    this.el.focusCaption.textContent = this._lastTranslationText || "";
    this.el.focusOverlay.hidden = false;
    try {
      if ("wakeLock" in navigator) this._wakeLock = await navigator.wakeLock.request("screen");
    } catch (_) { /* wake lock optional */ }
  }

  _exitFocus() {
    this._focusOn = false;
    this.el.focusOverlay.hidden = true;
    if (this._wakeLock) { try { this._wakeLock.release(); } catch (_) {} this._wakeLock = null; }
  }

  // ---- duo / transparent-screen counter display ---------------------------
  // Beam-projected glass between two people: top half in one language, bottom
  // half in the other, flipped so the far side reads it through the glass.
  async _duoStart() {
    this.el.duoConfig.hidden = true;
    this._duoOn = true;
    const flip = this.el.duoFlipSel.value;
    this.el.duoBottom.className = (flip === "mirror" ? "duo-mirror " : flip === "rotate" ? "duo-rotate " : "") + "text-3xl font-bold leading-relaxed text-emerald-100";
    this.el.duoTop.textContent = ""; this.el.duoBottom.textContent = "";
    this.el.duoOverlay.hidden = false;
    try { if ("wakeLock" in navigator && !this._wakeLock) this._wakeLock = await navigator.wakeLock.request("screen"); } catch (_) {}
    if (!this.running) this.start(); // reuse the quick-translate session
  }
  _duoExit() {
    this._duoOn = false;
    this.el.duoOverlay.hidden = true;
    if (this._wakeLock && !this._focusOn) { try { this._wakeLock.release(); } catch (_) {} this._wakeLock = null; }
  }
  async _duoRender(entry) {
    const base = this._serverBase(); if (!base || !entry.source) return;
    const top = this.el.duoTopLang.value, bottom = this.el.duoBottomLang.value;
    try {
      const r = await fetch(`${base}/api/translate`, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: entry.source, targets: [...new Set([top, bottom])], token: this._token() || undefined }) });
      if (!r.ok) return;
      const tr = (await r.json()).translations || {};
      if (!this._duoOn) return; // exited while translating
      this.el.duoTop.textContent = tr[top] || entry.source;
      this.el.duoBottom.textContent = tr[bottom] || "";
    } catch {}
  }

  _finalizeTurn() { clearTimeout(this._turnTimer); this._turnTimer = null; this._commitTurn(); this._srcLine = null; this._trLine = null; }

  _commitTurn() {
    const source = this._curSource.trim(), translation = this._curTranslation.trim();
    this._curSource = ""; this._curTranslation = "";
    this._finalizeAudio();
    // Self-echo guard: we know exactly what we just played — if the "heard"
    // source is (nearly) identical to a recently played translation, the mic
    // re-captured our own speaker output. Discard the turn (remove its
    // bubbles, keep it out of the log) instead of translating our own voice.
    if (source && this._isSelfEcho(source)) {
      for (const el of [this._srcLine, this._trLine]) {
        const w = el && el.closest && el.closest("[data-row]"); if (w) w.remove();
      }
      return;
    }
    if (translation) { this._lastTranslationText = translation; this.el.pronounceBtn.disabled = false; this._rememberPlayed(translation); }
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
    // Pass the live caption elements so a context correction can rewrite them
    // in place (they are nulled right after commit returns).
    if (this.context === "note" || this.el.riskToggle.checked || this.el.clarifyToggle.checked || this.el.answerToggle.checked || this.el.upgradeToggle.checked) this._analyzeTurn(entry, this._trLine, this._srcLine);
    if (this._displayLangs().length && source) this._multiTranslate(entry);
    if (this._duoOn && source) this._duoRender(entry);
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
      this._autoScroll(this.transcriptEl);
    } catch {}
  }

  async _analyzeTurn(entry, trEl, srcEl) {
    const base = this._serverBase(); if (!base || !entry.source) return;
    const isNote = this.context === "note";
    const wantRisk = isNote || this.el.riskToggle.checked;
    const wantClarify = isNote || this.el.clarifyToggle.checked;
    const wantAnswer = this.el.answerToggle.checked;
    const wantUpgrade = this.el.upgradeToggle.checked;
    try {
      const r = await fetch(`${base}/api/analyze`, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ original: entry.source, translation: entry.translation,
          alert_language: (localStorage.getItem("assistLang") || UI_LANG), target_language: this.el.langB.value,
          context: [this.el.riskContext.value.trim(), (localStorage.getItem("glossary") || "").trim() && ("Glossary (correct names/terms): " + localStorage.getItem("glossary").trim())].filter(Boolean).join("\n"),
          history: this._currentLog().slice(-6).map((e) => `${e.source} → ${e.translation}`).join("\n"),
          want_risk: wantRisk, want_clarify: wantClarify, want_answer: wantAnswer, want_upgrade: wantUpgrade, token: this._token() || undefined }) });
      if (!r.ok) return;
      const d = await r.json(); if (!d) return;
      let warned = false;
      if (wantRisk && d.risk_level && d.risk_level !== "none") { entry.risk = d; warned = true; this._renderRisk(d); }
      if (wantClarify && d.clarify_suspected) {
        entry.clarify = d; warned = true;
        // Self-heal: don't just suggest — rewrite the on-screen caption and the
        // saved log with the contextually corrected translation. The spoken
        // audio has already played (can't be unsaid), but everything the user
        // reads/exports from here on is the corrected text.
        const mark = ` <span class="ml-1 text-[10px] font-semibold text-emerald-400">✓${UI_LANG === "ko" ? "교정" : "fixed"}</span>`;
        const fixed = (d.clarify_corrected_translation || "").trim();
        if (fixed && fixed !== entry.translation) {
          entry.translation_raw = entry.translation;
          entry.translation = fixed;
          this._lastTranslationText = fixed;
          if (trEl && trEl.isConnected) trEl.innerHTML = esc(fixed) + mark;
          if (this._focusOn) this.el.focusCaption.textContent = fixed;
        }
        // Heal the ORIGINAL caption too (e.g. '7시간' → '실시간'): the source
        // line is what bilingual users actually read back.
        const fixedSrc = (d.clarify_corrected_source || "").trim();
        if (fixedSrc && fixedSrc !== entry.source) {
          entry.source_raw = entry.source;
          entry.source = fixedSrc;
          if (srcEl && srcEl.isConnected) srcEl.innerHTML = esc(fixedSrc) + mark;
        }
        // Card rendered AFTER healing so its 맞아요/아니에요 buttons can either
        // confirm the applied correction or revert to the raw text.
        this._renderClarify(d, entry, trEl, srcEl);
      }
      if (wantAnswer && d.should_answer && (d.answer_native || d.answer_local)) this._renderAnswer(d);
      if (wantUpgrade && d.upgrade && d.upgrade.trim()) this._renderUpgrade(d.upgrade.trim());
      if (warned) { this._persist(); if (isNote) this._appendWarn(entry); }
    } catch {}
  }
  _renderAnswer(d) {
    const body = `<div class="font-medium">${esc(d.answer_native || "")}</div>` +
      (d.answer_local ? `<div class="mt-1 rounded bg-black/20 p-2 opacity-90">${esc(d.answer_local)}</div>` : "");
    const actions = [
      { label: t("assist.speak"), onClick: () => this._speak(d.answer_native, this.el.langB.value) },
      { label: t("assist.copy"), onClick: () => navigator.clipboard && navigator.clipboard.writeText(d.answer_native || "").catch(() => {}) },
    ];
    this._pushAssist("answer", t("assist.answer"), body, actions);
  }
  _renderUpgrade(text) {
    this._pushAssist("upgrade", t("assist.upgrade"), `<div class="font-medium">${esc(text)}</div>`,
      [{ label: t("assist.copy"), onClick: () => navigator.clipboard && navigator.clipboard.writeText(text).catch(() => {}) }]);
  }
  /** Speak text aloud with the browser TTS in the given language. */
  _speak(text, langCode) {
    if (!text || !window.speechSynthesis) return;
    const u = new SpeechSynthesisUtterance(text);
    const map = { ko: "ko-KR", en: "en-US", ja: "ja-JP", zh: "zh-CN", fr: "fr-FR", es: "es-ES", ar: "ar-SA", ru: "ru-RU" };
    u.lang = map[langCode] || "en-US";
    window.speechSynthesis.cancel(); window.speechSynthesis.speak(u);
  }
  /** Inline "참고/Note" warning under a turn (notes). */
  _warnHtml(entry) {
    let h = "";
    if (entry.risk && entry.risk.risk_level && entry.risk.risk_level !== "none") {
      const col = entry.risk.risk_level === "high" ? "text-rose-300" : entry.risk.risk_level === "medium" ? "text-amber-300" : "text-slate-400";
      h += `<div class="${col}">⚠️ ${UI_LANG === "ko" ? "참고" : "Note"}: ${esc(entry.risk.subtitle_alert || "")}</div>`;
    }
    if (entry.clarify && entry.clarify.clarify_suspected && !entry.clarify.rejected) {
      h += `<div class="text-sky-300">🔎 ${esc(entry.clarify.clarify_did_you_mean || "")}</div>`;
    }
    return h;
  }
  _appendWarn(entry) {
    const h = this._warnHtml(entry); if (!h) return;
    const w = document.createElement("div"); w.dataset.row = "1";
    w.className = "rounded-lg border border-slate-800 bg-slate-950/40 px-3 py-1.5 text-xs";
    w.innerHTML = h; this.transcriptEl.appendChild(w);
    this._autoScroll(this.transcriptEl);
  }
  /**
   * Push a live assist card into the shared stack (above the control bar).
   * type → accent color. Keeps at most 3 cards (newest on top).
   */
  _pushAssist(type, headline, bodyHtml, actions) {
    const theme = {
      answer: ["border-indigo-500/60 bg-indigo-950/40 text-indigo-100", "💬"],
      upgrade: ["border-emerald-500/60 bg-emerald-950/40 text-emerald-100", "✨"],
      risk_low: ["border-slate-600 bg-slate-800 text-slate-200", "💡"],
      risk_medium: ["border-amber-500/60 bg-amber-950/40 text-amber-100", "⚠️"],
      risk_high: ["border-rose-500/70 bg-rose-950/40 text-rose-100", "🚨"],
      clarify: ["border-sky-500/60 bg-sky-950/40 text-sky-100", "🔎"],
    }[type] || ["border-slate-600 bg-slate-800 text-slate-200", "•"];
    const card = document.createElement("div");
    card.className = `slide-up rounded-xl border p-3 text-sm ${theme[0]}`;
    card.innerHTML =
      `<div class="mb-1 flex items-start justify-between gap-2">
         <span class="text-xs font-bold">${theme[1]} ${esc(headline)}</span>
         <button class="shrink-0 text-xs opacity-60 hover:opacity-100" data-x>✕</button>
       </div>${bodyHtml}`;
    card.querySelector("[data-x]").addEventListener("click", () => card.remove());
    if (actions) {
      const row = document.createElement("div");
      row.className = "mt-2 flex flex-wrap gap-2";
      for (const a of actions) {
        const b = document.createElement("button");
        b.className = "rounded-md border border-white/20 px-2 py-1 text-xs hover:bg-white/10";
        b.textContent = a.label;
        b.addEventListener("click", () => { a.onClick(); if (a.dismiss) card.remove(); });
        row.appendChild(b);
      }
      card.appendChild(row);
    }
    this.el.assistCards.prepend(card);
    while (this.el.assistCards.children.length > 3) this.el.assistCards.lastChild.remove();
  }
  _clearAssist() { if (this.el.assistCards) this.el.assistCards.innerHTML = ""; }

  _renderClarify(d, entry, trEl, srcEl) {
    const head = UI_LANG === "ko" ? "혹시 이런 뜻? (자막에 적용됨)" : "Did you mean…? (applied to captions)";
    const actions = entry ? [
      { label: UI_LANG === "ko" ? "✔ 맞아요" : "✔ Correct", dismiss: true,
        onClick: () => { if (entry.clarify) entry.clarify.confirmed = true; this._persist(); } },
      { label: UI_LANG === "ko" ? "✖ 아니에요" : "✖ No", dismiss: true,
        onClick: () => { this._revertClarify(entry, trEl, srcEl); this._pushClarifyEdit(entry, trEl, srcEl); } },
    ] : null;
    this._pushAssist("clarify", head,
      `<div class="font-medium">${esc(d.clarify_did_you_mean)}</div>
       <div class="mt-1 rounded bg-black/20 p-2">${esc(d.clarify_corrected_translation)}</div>`, actions);
  }
  _pushClarifyEdit(entry, trEl, srcEl) {
    // "Neither the correction nor the raw text is right" — let the user type
    // the exact intended sentence; the translation is redone automatically so
    // captions, log, notes and summaries all carry the user's authoritative fix.
    const ko = UI_LANG === "ko";
    const card = document.createElement("div");
    card.className = "slide-up rounded-xl border border-amber-500/60 bg-amber-950/40 p-3 text-sm text-amber-100";
    card.innerHTML =
      `<div class="mb-1 flex items-start justify-between gap-2">
         <span class="text-xs font-bold">✍️ ${ko ? "정확한 뜻을 직접 입력 (원문도 틀렸다면 고쳐 쓰세요)" : "Type what was actually meant"}</span>
         <button class="shrink-0 text-xs opacity-60 hover:opacity-100" data-x>✕</button>
       </div>
       <input data-fix class="w-full rounded-lg border border-white/20 bg-black/20 px-2 py-1.5 text-sm text-amber-50 focus:outline-none" />
       <div class="mt-2 flex gap-2">
         <button data-save class="rounded-md border border-white/20 px-2 py-1 text-xs hover:bg-white/10">💾 ${ko ? "저장 (자동 재번역)" : "Save (retranslate)"}</button>
       </div>`;
    card.querySelector("[data-fix]").value = entry.source || "";
    card.querySelector("[data-x]").addEventListener("click", () => card.remove());
    card.querySelector("[data-save]").addEventListener("click", () => {
      const v = card.querySelector("[data-fix]").value.trim();
      if (!v) return;
      card.remove();
      this._applyUserFix(entry, v, trEl, srcEl);
    });
    this.el.assistCards.prepend(card);
    while (this.el.assistCards.children.length > 3) this.el.assistCards.lastChild.remove();
  }
  async _applyUserFix(entry, text, trEl, srcEl) {
    const mark = ` <span class="ml-1 text-[10px] font-semibold text-amber-300">✎${UI_LANG === "ko" ? "수정" : "edited"}</span>`;
    entry.source = text; entry.user_fixed = true;
    if (entry.clarify) entry.clarify.rejected = true;
    if (srcEl && srcEl.isConnected) srcEl.innerHTML = esc(text) + mark;
    this._persist();
    // Redo the translation from the user's authoritative sentence.
    const base = this._serverBase(); if (!base) return;
    try {
      const r = await fetch(`${base}/api/translate`, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, targets: [this.el.langB.value], token: this._token() || undefined }) });
      if (!r.ok) return;
      const tr = ((await r.json()).translations || {})[this.el.langB.value];
      if (tr) {
        entry.translation = tr; this._lastTranslationText = tr;
        if (trEl && trEl.isConnected) trEl.innerHTML = esc(tr) + mark;
        if (this._focusOn) this.el.focusCaption.textContent = tr;
        this._persist();
      }
    } catch {}
  }
  _revertClarify(entry, trEl, srcEl) {
    // User says the correction was wrong: restore the raw recognition in the
    // captions and the log, and mark it rejected so notes/summaries/exports
    // use the original and drop the (?) warning.
    if (entry.translation_raw != null) {
      entry.translation = entry.translation_raw; delete entry.translation_raw;
      this._lastTranslationText = entry.translation;
      if (trEl && trEl.isConnected) trEl.textContent = entry.translation;
      if (this._focusOn) this.el.focusCaption.textContent = entry.translation;
    }
    if (entry.source_raw != null) {
      entry.source = entry.source_raw; delete entry.source_raw;
      if (srcEl && srcEl.isConnected) srcEl.textContent = entry.source;
    }
    if (entry.clarify) entry.clarify.rejected = true;
    this._persist();
  }
  _renderRisk(d) {
    const head = UI_LANG === "ko" ? "참고" : "Note";
    this._pushAssist("risk_" + d.risk_level, head,
      `<div class="font-medium">${esc(d.subtitle_alert)}</div>` +
      (d.suggested_question ? `<div class="mt-2 rounded bg-black/20 p-2">❓ ${esc(d.suggested_question)}</div>` : ""));
  }

  // ---- session lifecycle --------------------------------------------------
  async start() {
    try {
      this._lastError = ""; this._setStatus("connecting", t("st.connecting")); this.el.toggleBtn.disabled = true;
      if (this.precision) {
        // Precision-caption cascade: no live socket, no playback — utterances
        // are VAD-segmented locally and sent to /api/transcribe one by one.
        this._audioSink = this._precSink();
        await this._startCapture();
        this.running = true; this.paused = false;
        this.el.toggleBtn.disabled = false; this.el.pauseBtn.disabled = false;
        this._refreshToggle(); this._setStatus("live", t("st.precLive"));
        return;
      }
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
  _keepAlive(on) {
    try {
      const p = window.Capacitor && window.Capacitor.Plugins && window.Capacitor.Plugins.KeepAlive;
      if (p) (on ? p.start() : p.stop());
    } catch (_) { /* web / plugin missing — fine */ }
  }
  async _teardown() {
    if (this._turnTimer) this._finalizeTurn();
    this._audioSink = null;
    this._keepAlive(false);
    if (this.recorder) {
      const rec = this.recorder; this.recorder = null;
      await new Promise((res) => { rec.onstop = res; try { rec.stop(); } catch (_) { res(); } });
      const blob = new Blob(this._recChunks || [], { type: "audio/webm" }); this._recChunks = [];
      if (blob.size > 4096) {
        if (this.context === "note" && this.activeNoteId) { try { await idbPut(this.activeNoteId, blob); this._refreshAudioRow(); } catch (_) {} }
        else this._lastRecBlob = blob; // attached when the quick session is saved as a note
      }
    }
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
    else if (m.type === "interrupted") { /* streaming model: keep playing (no flush) */ }
    else if (m.type === "error") { this._lastError = m.message || "error"; this._setStatus("error", this._lastError); }
  }

  // ---- capture ------------------------------------------------------------
  async _startCapture() {
    // Earphone(Bluetooth) mode: turning OFF the voice-processing constraints
    // keeps Android in MEDIA audio mode so BT routes via A2DP and the
    // translated audio actually plays in the earphones. With echoCancellation
    // on, Android forces communication mode (SCO/HFP) which a WebView can't
    // drive, so sound never reaches the BT earphones.
    const proc = localStorage.getItem("earphoneMode") !== "1";
    this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: proc, noiseSuppression: proc, autoGainControl: proc } });
    this.captureContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: INPUT_SAMPLE_RATE });
    await this.captureContext.audioWorklet.addModule("./audio-processor.js");
    this.micSource = this.captureContext.createMediaStreamSource(this.mediaStream);
    this.workletNode = new AudioWorkletNode(this.captureContext, "pcm-capture-processor");
    this.workletNode.port.onmessage = (e) => {
      if (this.paused) return;
      if (this._audioSink) return this._audioSink(e.data);
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
      // Gates replace frames with SILENCE rather than dropping them: Gemini
      // detects end-of-turn from silence in the stream, so dropped frames
      // starve turn detection (very late, garbled translations).
      // While our own translated audio is playing we do NOT mute the mic (the
      // other party may keep talking); instead _vadGate raises its threshold —
      // speaker echo arrives quieter than direct speech, so echo is gated
      // while a real nearby voice still passes.
      const gated =
        (this._noiseGateOn() || this._playbackBusy()) && !this._vadGate(e.data);
      this.ws.send(gated ? new ArrayBuffer(e.data.byteLength) : e.data);
    };
    this.micSource.connect(this.workletNode);
    const sink = this.captureContext.createGain(); sink.gain.value = 0;
    this.workletNode.connect(sink); sink.connect(this.captureContext.destination);
    // Keep capturing when the screen locks or another app takes the front:
    // start the native microphone foreground service (no-op on plain web).
    this._keepAlive(true);
    // Record the raw meeting audio in parallel (opus ≈ 15MB/h) so dubious
    // passages can be re-listened and the transcript corrected afterwards.
    this._recChunks = [];
    if ((this.context === "note" || this.view === "translate") && window.MediaRecorder) {
      try {
        this.recorder = new MediaRecorder(this.mediaStream, { mimeType: "audio/webm;codecs=opus", audioBitsPerSecond: 32000 });
        this.recorder.ondataavailable = (ev) => { if (ev.data && ev.data.size) this._recChunks.push(ev.data); };
        this.recorder.start(1000);
      } catch (_) { this.recorder = null; }
    }
  }

  // ---- playback -----------------------------------------------------------
  _initPlayback() {
    this.playbackContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: OUTPUT_SAMPLE_RATE });
    this.nextPlayTime = 0;
    // Root cause of the speaker→mic repeat loop: Android WebView's echo
    // canceller only references media-element/WebRTC output — raw WebAudio
    // playback is invisible to it, so AEC never cancelled our own translation
    // audio from the mic. Routing playback through an <audio srcObject> sink
    // puts it on the media path where AEC can subtract it.
    try {
      this._playbackDest = this.playbackContext.createMediaStreamDestination();
      if (!this._playbackEl) { this._playbackEl = new Audio(); this._playbackEl.autoplay = true; }
      this._playbackEl.srcObject = this._playbackDest.stream;
      this._playbackEl.play().catch(() => {});
    } catch (_) { this._playbackDest = null; }
  }
  _playbackSink(ctx) { return (ctx === this.playbackContext && this._playbackDest) ? this._playbackDest : ctx.destination; }
  _enqueueAudio(buf) {
    if (!this.playbackContext) return;
    const pcm = new Int16Array(buf), f32 = new Float32Array(pcm.length);
    for (let i = 0; i < pcm.length; i++) f32[i] = pcm[i] / 0x8000;
    this._curAudioChunks.push(f32);
    if (!this.audioOutput) return;
    const b = this.playbackContext.createBuffer(1, f32.length, OUTPUT_SAMPLE_RATE); b.copyToChannel(f32, 0);
    const s = this.playbackContext.createBufferSource(); s.buffer = b; s.playbackRate.value = this.playbackRate; s.connect(this._playbackSink(this.playbackContext));
    const now = this.playbackContext.currentTime; if (this.nextPlayTime < now) this.nextPlayTime = now + 0.05;
    s.start(this.nextPlayTime); this.nextPlayTime += b.duration / this.playbackRate;
    // Remember when playback (incl. queue) ends + a 300ms reverb tail, so the
    // half-duplex mic gate knows how long to hold.
    const perfNow = (window.performance && performance.now) ? performance.now() : Date.now();
    this._lastPlaybackEnd = perfNow + (this.nextPlayTime - now) * 1000 + 300;
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
    const s = ctx.createBufferSource(); s.buffer = b; s.playbackRate.value = this.playbackRate; s.connect(this._playbackSink(ctx)); s.start();
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
  async _feedback() {
    const log = this._currentLog(); if (!log.length) return;
    const base = this._serverBase(); if (!base) return this._setStatus("error", t("st.setServer"));
    const prev = this.el.noteFeedbackBtn.textContent; this.el.noteFeedbackBtn.disabled = true; this.el.noteFeedbackBtn.textContent = "…";
    try {
      const r = await fetch(`${base}/api/feedback`, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transcript: this._transcriptText(), language: UI_LANG, token: this._token() || undefined }) });
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.status);
      const fb = (await r.json()).feedback || "";
      this.el.feedbackContent.textContent = fb;
      if (this.context === "note") { const n = this._note(this.activeNoteId); if (n) { n.feedback = fb; this._saveNotes(); } }
    } catch (e) { this._setStatus("error", t("msg.feedbackFail") + ": " + e.message); }
    finally { this.el.noteFeedbackBtn.disabled = false; this.el.noteFeedbackBtn.textContent = prev; }
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
      if (e.clarify && e.clarify.clarify_suspected && !e.clarify.rejected) warn.push(`(?) ${e.clarify.clarify_did_you_mean || ""}`);
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

  // ---- multi-device rooms -------------------------------------------------
  _wsBase() { return this._serverBase().replace(/^http/i, "ws"); }

  async _roomStart() {
    const base = this._serverBase(); if (!base) return this._setStatus("error", t("st.setServer"));
    const id = Math.random().toString(36).slice(2, 8);
    this.roomId = id;
    const tok = this._token();
    const url = `${this._wsBase()}/api/room/host?room=${id}${tok ? `&token=${encodeURIComponent(tok)}` : ""}`;
    this.el.roomStartBtn.disabled = true;
    this._setStatus("connecting", t("st.waking"));
    try {
      await new Promise((res, rej) => {
        const ws = new WebSocket(url); ws.binaryType = "arraybuffer";
        const to = setTimeout(() => { try { ws.close(); } catch (_) {} rej(new Error(t("st.timeout"))); }, 30000);
        ws.onopen = () => { clearTimeout(to); res(); };
        ws.onerror = () => { clearTimeout(to); rej(new Error(t("st.cantReach"))); };
        ws.onmessage = (e) => { try { const m = JSON.parse(e.data); if (m.type === "participants") this.el.roomCount.textContent = m.count; else if (m.type === "error") this._setStatus("error", m.message); } catch {} };
        ws.onclose = () => { clearTimeout(to); if (this.hostWs) this._roomStop(true); };
        this.hostWs = ws;
      });
      this.ws = this.hostWs; this.running = true; this.paused = false;
      await this._startCapture();
      const link = `${base}/?room=${id}`;
      this.el.roomLink.value = link;
      if (window.QRCode) window.QRCode.toCanvas(this.el.roomQr, link, { width: 200, margin: 1 }, () => {});
      this.el.roomCount.textContent = "0";
      this.el.roomIdle.hidden = true; this.el.roomLive.hidden = false;
      this._setStatus("live", t("st.hosting"));
    } catch (e) { this._setStatus("error", e.message); this._roomStop(true); }
    finally { this.el.roomStartBtn.disabled = false; }
  }
  async _roomStop(silent) {
    this.running = false;
    const ws = this.hostWs; this.hostWs = null;
    await this._teardown();
    if (ws && ws.readyState === WebSocket.OPEN) ws.close();
    this.el.roomLive.hidden = true; this.el.roomIdle.hidden = false;
    if (!silent) this._setStatus("idle", t("st.idle"));
  }

  _joinSetup(roomId) {
    this.joinRoom = roomId;
    if (!this.el.joinLang.options.length) {
      for (const [c, l] of Object.entries(LANGUAGES)) { const o = document.createElement("option"); o.value = c; o.textContent = l; this.el.joinLang.appendChild(o); }
      this.el.joinLang.value = localStorage.getItem("langB") || DEFAULT_LANG_B;
    }
    this.el.joinForm.hidden = false; this.el.joinLive.hidden = true;
  }
  async _joinStart() {
    const base = this._serverBase(); if (!base) return this._setStatus("error", t("st.setServer"));
    const lang = this.el.joinLang.value;
    const url = `${this._wsBase()}/api/room/join?room=${this.joinRoom}&lang=${lang}`;
    this.el.joinBtn.disabled = true;
    try {
      this._initPlayback(); this.audioOutput = true;
      await new Promise((res, rej) => {
        const ws = new WebSocket(url); ws.binaryType = "arraybuffer";
        ws.onopen = () => res();
        ws.onerror = () => rej(new Error(t("st.roomEnded")));
        ws.onmessage = (e) => {
          if (e.data instanceof ArrayBuffer) return this._enqueueAudio(e.data);
          try { const m = JSON.parse(e.data);
            if (m.type === "transcript") this._joinAppend(m.role, m.text);
            else if (m.type === "turn_complete") { this._joinSrc = null; this._joinTr = null; }
            else if (m.type === "error") this._setStatus("error", m.message);
          } catch {}
        };
        ws.onclose = () => { if (this.joinWs) { this.joinWs = null; this._setStatus("error", t("st.roomEnded")); this.el.joinLive.hidden = true; this.el.joinForm.hidden = false; } };
        this.joinWs = ws;
      });
      this.el.joinLangLabel.textContent = LANGUAGES[lang] || lang;
      this.el.joinTranscript.innerHTML = ""; this._joinSrc = null; this._joinTr = null;
      this.el.joinForm.hidden = true; this.el.joinLive.hidden = false;
      this._setStatus("live", t("st.joined"));
    } catch (e) { this._setStatus("error", e.message); }
    finally { this.el.joinBtn.disabled = false; }
  }
  _joinLeave(silent) {
    const ws = this.joinWs; this.joinWs = null;
    if (ws && ws.readyState === WebSocket.OPEN) ws.close();
    if (this.playbackContext) { this.playbackContext.close().catch(() => {}); this.playbackContext = null; }
    this.el.joinLive.hidden = true; this.el.joinForm.hidden = false;
    if (!silent) this._setStatus("idle", t("st.idle"));
  }
  _joinAppend(role, text) {
    if (!text) return;
    const key = role === "translation" ? "_joinTr" : "_joinSrc";
    if (!this[key]) { const w = this._bubble(role, ""); this.el.joinTranscript.appendChild(w); this[key] = w.lastElementChild; }
    this[key].textContent += text;
    this._autoScroll(this.el.joinTranscript);
  }

  // ---- multi-mic meeting (diarized; channel = speaker) --------------------
  _meetSetup() {
    if (!this.el.meetLang.options.length) {
      for (const [c, l] of Object.entries(LANGUAGES)) { const o = document.createElement("option"); o.value = c; o.textContent = l; this.el.meetLang.appendChild(o); }
      this.el.meetLang.value = localStorage.getItem("langB") || DEFAULT_LANG_B;
    }
    this.el.meetIdle.hidden = false; this.el.meetLive.hidden = true;
  }
  async _meetStart() {
    const base = this._serverBase(); if (!base) return this._setStatus("error", t("st.setServer"));
    const id = Math.random().toString(36).slice(2, 8);
    this.meetId = id; this._meetLog = []; this._diarTurn = {};
    const lang = this.el.meetLang.value, tok = this._token();
    const url = `${this._wsBase()}/api/meeting/host?room=${id}&lang=${lang}${tok ? `&token=${encodeURIComponent(tok)}` : ""}`;
    this.el.meetStartBtn.disabled = true;
    this._setStatus("connecting", t("st.waking"));
    try {
      await new Promise((res, rej) => {
        const ws = new WebSocket(url);
        const to = setTimeout(() => { try { ws.close(); } catch (_) {} rej(new Error(t("st.timeout"))); }, 30000);
        ws.onopen = () => { clearTimeout(to); res(); };
        ws.onerror = () => { clearTimeout(to); rej(new Error(t("st.cantReach"))); };
        ws.onmessage = (e) => this._onMeetMsg(e);
        ws.onclose = (ev) => { clearTimeout(to); if (this.meetHostWs) { this.meetHostWs = null; if (this.view === "meet") this._setStatus("error", ev.code === 1008 ? t("st.tokenBad") : t("st.roomEnded")); } };
        this.meetHostWs = ws;
      });
      this._diarTranscript = this.el.meetTranscript; this.el.meetTranscript.innerHTML = "";
      const link = `${base}/?meet=${id}`;
      this.el.meetLink.value = link;
      if (window.QRCode) window.QRCode.toCanvas(this.el.meetQr, link, { width: 200, margin: 1 }, () => {});
      this.el.meetRoster.innerHTML = "";
      this.el.meetIdle.hidden = true; this.el.meetLive.hidden = false;
      // The host socket is display-only, but the host is usually a speaker too
      // — open a participant channel for this phone's mic so the convener's
      // own speech is transcribed. Non-fatal: the meeting still works as a
      // display board if the mic/channel fails.
      try {
        const hostName = (localStorage.getItem("mjoinName") || "").trim() || (UI_LANG === "ko" ? "호스트" : "Host");
        await new Promise((res, rej) => {
          const ws2 = new WebSocket(`${this._wsBase()}/api/meeting/join?room=${id}&name=${encodeURIComponent(hostName)}`);
          ws2.onopen = () => res();
          ws2.onerror = () => rej(new Error("host mic channel failed"));
          ws2.onmessage = () => {}; // transcripts already arrive on the host socket
          ws2.onclose = () => { this.meetSpeakWs = null; };
          this.meetSpeakWs = ws2;
        });
        this._audioSink = this._meetSink(() => this.meetSpeakWs);
        await this._startCapture();
      } catch (_) { /* display-only fallback */ }
      this._setStatus("live", t("st.hosting"));
    } catch (e) { this._setStatus("error", e.message); }
    finally { this.el.meetStartBtn.disabled = false; }
  }
  _meetStop(silent) {
    const ws = this.meetHostWs; this.meetHostWs = null;
    const ws2 = this.meetSpeakWs; this.meetSpeakWs = null;
    this._teardown(); // stops the host mic capture (and clears _audioSink)
    if (ws2 && ws2.readyState === WebSocket.OPEN) ws2.close();
    if (ws && ws.readyState === WebSocket.OPEN) ws.close();
    this.el.meetLive.hidden = true; this.el.meetIdle.hidden = false;
    if (!silent) this._setStatus("idle", t("st.idle"));
  }
  _onMeetMsg(e) {
    let m; try { m = JSON.parse(e.data); } catch { return; }
    if (m.type === "diar") this._onDiar(m);
    else if (m.type === "roster") this._renderRoster(m.speakers);
    else if (m.type === "error") this._setStatus("error", m.message);
  }
  _meetSaveNote() {
    if (!this._meetLog || !this._meetLog.length) { this._setStatus("live", t("msg.nothingSave")); return; }
    const log = this._meetLog.map((e) => ({ t: e.t, source: `${e.name}: ${e.source}`, translation: e.translation }));
    const title = (this._meetLog[0] && this._meetLog[0].source ? this._meetLog[0].source.slice(0, 24) : t("meet.noteTitle"));
    this.notes.unshift({ id: "n" + Date.now(), title, createdAt: Date.now(), updatedAt: Date.now(), log, summary: "", topic: "" });
    this._saveNotes();
    this._setStatus("live", t("st.savedNote"));
  }

  // participant side (a microphone)
  _mjoinSetup(roomId) {
    this.meetRoom = roomId;
    this.el.mjoinName.value = localStorage.getItem("mjoinName") || "";
    this.el.mjoinForm.hidden = false; this.el.mjoinLive.hidden = true;
  }
  async _mjoinStart() {
    const base = this._serverBase(); if (!base) return this._setStatus("error", t("st.setServer"));
    const name = this.el.mjoinName.value.trim(); localStorage.setItem("mjoinName", name);
    const url = `${this._wsBase()}/api/meeting/join?room=${this.meetRoom}${name ? `&name=${encodeURIComponent(name)}` : ""}`;
    this.el.mjoinBtn.disabled = true;
    this._meetLog = []; this._diarTurn = {};
    try {
      await new Promise((res, rej) => {
        const ws = new WebSocket(url);
        ws.onopen = () => res();
        ws.onerror = () => rej(new Error(t("st.roomEnded")));
        ws.onmessage = (e) => this._onMjoinMsg(e);
        ws.onclose = () => { if (this.meetWs) { this.meetWs = null; if (this.view === "mjoin") { this._setStatus("error", t("st.roomEnded")); this.el.mjoinLive.hidden = true; this.el.mjoinForm.hidden = false; this._teardown(); } } };
        this.meetWs = ws;
      });
      this._diarTranscript = this.el.mjoinTranscript; this.el.mjoinTranscript.innerHTML = "";
      this._audioSink = this._meetSink(() => this.meetWs);
      await this._startCapture();
      this.el.mjoinForm.hidden = true; this.el.mjoinLive.hidden = false;
      this._setStatus("live", t("st.joined"));
    } catch (e) { this._setStatus("error", e.message); this.meetWs = null; }
    finally { this.el.mjoinBtn.disabled = false; }
  }
  _onMjoinMsg(e) {
    let m; try { m = JSON.parse(e.data); } catch { return; }
    if (m.type === "joined_meeting") {
      this.el.mjoinLabel.textContent = m.name || m.sid;
      if (Array.isArray(m.history)) m.history.forEach((u) => this._onDiar({ ...u, phase: "final" }));
    } else if (m.type === "diar") this._onDiar(m);
    else if (m.type === "meeting_ended") { this._setStatus("idle", t("st.roomEnded")); this._mjoinLeave(true); }
    else if (m.type === "error") this._setStatus("error", m.message);
  }
  _mjoinLeave(silent) {
    const ws = this.meetWs; this.meetWs = null;
    this._teardown();
    if (ws && ws.readyState === WebSocket.OPEN) ws.close();
    this.el.mjoinLive.hidden = true; this.el.mjoinForm.hidden = false;
    if (!silent) this._setStatus("idle", t("st.idle"));
  }
  // ---- precision-caption cascade -------------------------------------------
  _precSink() {
    // Segment utterances locally: buffer while the VAD is open, finalize after
    // 700ms of closed gate, force-flush at 30s so nothing grows unbounded.
    this._vadUntil = 0;
    let buf = [], bytes = 0, sawSpeech = false, quietSince = 0;
    const flush = () => {
      if (sawSpeech && bytes >= 12800) { // ≥0.4s of speech
        const all = new Uint8Array(bytes); let o = 0;
        for (const c of buf) { all.set(c, o); o += c.length; }
        this._precSend(all);
      }
      buf = []; bytes = 0; sawSpeech = false; quietSince = 0;
    };
    return (chunk) => {
      const speaking = this._vadGate(chunk);
      const now = (window.performance && performance.now) ? performance.now() : Date.now();
      if (speaking) {
        sawSpeech = true; quietSince = 0;
        buf.push(new Uint8Array(chunk.slice(0))); bytes += chunk.byteLength;
        if (bytes >= 960000) flush(); // 30s cap
      } else if (sawSpeech) {
        buf.push(new Uint8Array(chunk.slice(0))); bytes += chunk.byteLength; // short tail
        if (!quietSince) quietSince = now;
        else if (now - quietSince > 700) flush();
      }
    };
  }
  _b64(bytes) {
    let s = ""; const CH = 0x8000;
    for (let i = 0; i < bytes.length; i += CH) s += String.fromCharCode.apply(null, bytes.subarray(i, i + CH));
    return btoa(s);
  }
  _precSend(pcmBytes) {
    const base = this._serverBase(); if (!base) return;
    const body = JSON.stringify({
      audio_b64: this._b64(pcmBytes),
      source_lang: this.el.precSrcSel.value || "auto",
      target_lang: this.el.langB.value,
      glossary: (localStorage.getItem("glossary") || "").trim(),
      history: this._currentLog().slice(-4).map((e) => `${e.source} → ${e.translation}`).join("\n"),
      token: this._token() || undefined,
    });
    // Sequential queue: keep utterances in spoken order even if the network
    // returns them out of order.
    this._precQueue = (this._precQueue || Promise.resolve()).then(async () => {
      try {
        const r = await fetch(`${base}/api/transcribe`, { method: "POST", headers: { "Content-Type": "application/json" }, body });
        if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.status);
        const d = await r.json();
        if (!this.running || !this.precision) return;
        if (d.source || d.translation) {
          // Reuse the normal turn pipeline: bubbles, log, assist, multi-lang, duo.
          if (d.source) this._appendTranscript("source", d.source);
          if (d.translation) this._appendTranscript("translation", d.translation);
          this._finalizeTurn();
        }
      } catch (e) { this._setStatus("live", (UI_LANG === "ko" ? "전사 실패: " : "Transcribe failed: ") + e.message); }
    });
  }
  _normEcho(s) { return (s || "").toLowerCase().replace(/[^\p{L}\p{N}]+/gu, ""); }
  _rememberPlayed(text) {
    this._playedTr = (this._playedTr || []).filter((x) => Date.now() - x.t < 20000);
    const n = this._normEcho(text);
    if (n.length >= 8) this._playedTr.push({ t: Date.now(), n });
  }
  _isSelfEcho(source) {
    const ns = this._normEcho(source);
    if (ns.length < 8) return false; // too short to judge — let it through
    const grams = (s) => { const g = new Set(); for (let i = 0; i + 3 <= s.length; i++) g.add(s.slice(i, i + 3)); return g; };
    const a = grams(ns);
    return (this._playedTr || []).some((x) => {
      if (Date.now() - x.t > 20000) return false;
      if (x.n.includes(ns) || ns.includes(x.n)) return true;
      // Echo transcripts come back partially garbled, so exact containment
      // misses them — fall back to trigram overlap (≥55% of the heard text's
      // trigrams appearing in what we played ⇒ it's our own audio).
      const b = grams(x.n);
      if (!a.size || !b.size) return false;
      let hit = 0; for (const g of a) if (b.has(g)) hit++;
      return hit / a.size >= 0.55;
    });
  }
  _meetSink(getWs) {
    // VAD-gated meeting mic: stream only while speaking (idle costs nothing),
    // but after speech ends send a short silence tail so Gemini's turn
    // detection can finalize the utterance — dropping to nothing mid-stream
    // leaves turns hanging (late/garbled results).
    this._vadUntil = 0; this._meetTailUntil = 0;
    return (buf) => {
      const ws = getWs();
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      const now = (window.performance && performance.now) ? performance.now() : Date.now();
      if (this._vadGate(buf)) { this._meetTailUntil = 0; ws.send(buf); }
      else if (!this._meetTailUntil) { this._meetTailUntil = now + 1200; ws.send(new ArrayBuffer(buf.byteLength)); }
      else if (now < this._meetTailUntil) ws.send(new ArrayBuffer(buf.byteLength));
    };
  }
  _playbackBusy() {
    // True while queued translated audio is still playing (+300ms tail for
    // room reverb), and only in a mode that actually emits audio.
    if (!this.audioOutput || !this.playbackContext || this.playbackContext.state === "closed") return false;
    return this.nextPlayTime > this.playbackContext.currentTime + 0.001 ||
           (this._lastPlaybackEnd || 0) > ((window.performance && performance.now) ? performance.now() : Date.now());
  }
  _noiseGateOn() { return localStorage.getItem("noiseGate") !== "0"; } // default ON
  _vadGate(buf) {
    const pcm = new Int16Array(buf); let sum = 0;
    for (let i = 0; i < pcm.length; i++) { const v = pcm[i] / 0x8000; sum += v * v; }
    const rms = Math.sqrt(sum / pcm.length);
    const now = (window.performance && performance.now) ? performance.now() : Date.now();
    // 0.010: low enough for normal-volume speech (0.018 missed quiet voices);
    // 1s hangover so soft syllables mid-sentence don't chop the utterance.
    // While our own playback is audible the threshold rises to 0.04: speaker
    // echo reaching the mic is much quieter than direct speech, so this drops
    // the echo without deafening the app to a real speaker (full duplex).
    const thr = this._playbackBusy() ? 0.04 : 0.010;
    if (rms > thr) this._vadUntil = now + 1000;
    const active = now < (this._vadUntil || 0);
    if (this.el.mjoinSpeak) this.el.mjoinSpeak.style.opacity = active ? "1" : "0.2";
    return active;
  }

  // ---- diarized transcript (shared by host board + participant) ----------
  _speakerColor(sid) {
    const pal = ["border-sky-500/40 bg-sky-500/5","border-emerald-500/40 bg-emerald-500/5","border-amber-500/40 bg-amber-500/5","border-fuchsia-500/40 bg-fuchsia-500/5","border-rose-500/40 bg-rose-500/5","border-indigo-500/40 bg-indigo-500/5"];
    const i = (sid.charCodeAt(0) - 65 + (sid.length - 1) * 7) % pal.length;
    return pal[(i + pal.length) % pal.length];
  }
  _diarCard(m) {
    const card = document.createElement("div");
    card.className = "rounded-xl border p-2.5 " + this._speakerColor(m.sid);
    const head = document.createElement("div"); head.className = "mb-1 text-xs font-semibold text-slate-300"; head.textContent = m.name || ("화자 " + m.sid);
    const src = document.createElement("div"); src.className = "diar-src text-sm text-slate-200";
    const tr = document.createElement("div"); tr.className = "diar-tr mt-0.5 text-sm font-medium text-sky-300";
    card.append(head, src, tr);
    return card;
  }
  _onDiar(m) {
    const cont = this._diarTranscript; if (!cont) return;
    let turn = this._diarTurn[m.sid];
    if (!turn) { const c = this._diarCard(m); cont.appendChild(c); turn = this._diarTurn[m.sid] = { card: c }; }
    const srcEl = turn.card.querySelector(".diar-src"), trEl = turn.card.querySelector(".diar-tr");
    if (m.phase === "final") {
      srcEl.textContent = m.source || ""; trEl.textContent = m.translation || "";
      delete this._diarTurn[m.sid];
      (this._meetLog || (this._meetLog = [])).push({ name: m.name, source: m.source || "", translation: m.translation || "", t: Date.now() });
    } else {
      (m.role === "translation" ? trEl : srcEl).textContent += m.text;
    }
    this._autoScroll(cont);
  }
  _renderRoster(speakers) {
    if (!this.el.meetRoster) return;
    this.el.meetRoster.innerHTML = "";
    (speakers || []).forEach((s) => {
      const chip = document.createElement("span");
      chip.className = "rounded-full border px-2 py-0.5 text-xs " + this._speakerColor(s.sid);
      chip.textContent = s.name || ("화자 " + s.sid);
      this.el.meetRoster.appendChild(chip);
    });
  }
}

window.addEventListener("DOMContentLoaded", () => {
  if (!navigator.mediaDevices || !window.AudioContext) { alert("This browser does not support the Web Audio API."); return; }
  window.app = new App();
});
