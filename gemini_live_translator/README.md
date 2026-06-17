# Gemini Live Translator

A real-time, low-latency, **audio-to-audio** bidirectional interpreter built on
the Google Gemini **Multimodal Live API**. Pick any two languages; speak in
either one and immediately hear the other — with live text transcripts on screen.

**Supported languages:** the default translate model handles **70+ languages**
with the source auto-detected — you just pick **Language B = the language you
want to hear**. The UI lists 8 common languages (Korean, English, Japanese,
Mandarin, French, Spanish, Arabic, Russian); add more by extending `LANGUAGES`
in `static/app.js` and `SUPPORTED_LANGUAGES` in `services/gemini_live.py`. On the
persona model (`gemini-2.0-flash-exp`) the pair is two-way A ⇄ B, and Language A
can be **🌐 Auto-detect**.

**Controls:** male/female voice selection, playback **speed** (0.5×–1.5×, live),
**Pause/Resume** (mutes the mic without dropping the session), **Replay** the
last translated sentence, and **Pronounce** — a phonetic transliteration of the
last translation in romaji or Hangul (`POST /api/pronounce`).

**🛡️ Risk Guard (negotiation copilot):** an opt-in toggle that, *after* each
turn finishes, analyzes it on a separate text model (`POST /api/analyze`,
structured JSON) for business/contract/payment/number/etiquette risks and shows
an alert card (low/medium/high) with a short warning and a suggested follow-up
question. Because it runs off the real-time path — fired only on `turn_complete`
— **it never delays the live translation.** An optional industry-context box
(e.g. "oil trading") biases detection toward domain terms (LOI, SCO, POP, SGS,
Platts, CIF/FOB, …). Risks are saved into the meeting notes and the export.

There is no STT → Translate → TTS chain. Microphone audio is streamed directly
into Gemini's bidirectional live channel, and translated audio streams straight
back out, which is what keeps latency low.

**Meeting notes & summary:** every finalized turn is logged with a timestamp
(bilingual original → translation) and persisted on the device, so the record
survives an app restart. **Summarize** returns structured notes (one-line
summary, key points, decisions, action items). **Export** saves the notes +
summary as **Markdown, Word (.docx) or PDF** (`POST /api/export`; PDF uses a
bundled Korean font). **Ask** answers questions grounded in the saved
conversation (`POST /api/ask`).

**🔎 Clarify (mis-hearing guard):** because speech is transcribed live, unclear
or slurred pronunciation can produce a wrong translation. With Clarify on, each
turn is checked against the recent context and — if a mis-recognition is
suspected — a card proposes *"Did you mean …?"* with a corrected translation.
It shares the single per-turn analysis call with Risk Guard, so it adds no
extra latency to the live translation.

> **📱 안드로이드 앱으로 쓰려면 → [`ANDROID.md`](./ANDROID.md) (한글 가이드)**
> Capacitor로 APK를 빌드하고, 백엔드는 클라우드(Render)에 배포하는 전체 과정.

```
[ Browser ]  ⇄ (PCM16 audio + JSON over WebSocket) ⇄  [ FastAPI ]  ⇄ (Live API) ⇄  [ Gemini ]
```

## Model stack

| Job | Model (default) | Why |
| --- | --- | --- |
| Speech-to-speech translation | `gemini-3.5-live-translate-preview` | Dedicated low-latency S2S, 70+ langs |
| Live captions | Live API input/output transcription | Free with the audio session |
| Risk detection (risk-only turn) | `gemini-3.1-flash-lite` | Cheap/fast; runs often, off the realtime path |
| Meaning clarification / any turn with Clarify on | `gemini-3.5-flash` + thinking | Needs real reasoning to infer intent from garbled speech |
| Meeting summary / Q&A | `gemini-3.5-flash` + thinking | Stronger reasoning over the whole transcript |

Each is overridable via env (`GEMINI_LIVE_MODEL`, `RISK_MODEL`, `CLARIFY_MODEL`,
`SUMMARY_MODEL`). `ANALYSIS_THINKING_BUDGET` (default 4096) controls the extra
reasoning budget for the clarification and Q&A calls; set 0 to disable. Turning
on **Clarify** automatically routes that turn to the stronger model, so the
deeper reasoning lands exactly where it matters without making every turn
expensive. (`gemini-3.5-pro` was not yet a public API id as of June 2026; raise
`CLARIFY_MODEL`/`SUMMARY_MODEL` to it once available for even stronger reasoning.)

## Project layout

```
gemini_live_translator/
├── main.py                     # FastAPI app + /api/stream WebSocket bridge
├── services/
│   └── gemini_live.py          # google-genai client + LiveConnectConfig
├── static/                     # web bundle (also packaged into the APK)
│   ├── index.html              # Tailwind dashboard (status, transcript, voice)
│   ├── app.js                  # capture + playback + WebSocket controller
│   ├── audio-processor.js      # AudioWorklet: Float32 → PCM16 100 ms chunks
│   └── config.js               # build-time DEFAULT_SERVER_URL for the app
├── Dockerfile                  # backend container (Render/Railway/Fly/Cloud Run)
│   (Render blueprint lives at the repo root: ../render.yaml)
├── capacitor.config.json       # Android wrapper config
├── package.json                # Capacitor tooling
├── scripts/patch-android.mjs   # injects mic permissions into the APK
├── ANDROID.md                  # 📱 한글 앱 빌드/설치 가이드
├── requirements.txt
└── .env.example
```

> The frontend is served from the web **root** (`/`, `/app.js`, …) rather than
> `/static`, so the exact same asset layout works both on the web and inside the
> Capacitor app (which serves the bundle from the app root).

## Setup

```bash
cd gemini_live_translator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env and set GEMINI_API_KEY=...   (https://aistudio.google.com/apikey)
```

## Run

```bash
uvicorn main:app --reload
# or: python main.py
```

Open <http://localhost:8000>, click **Start Translating**, allow microphone
access, and start speaking. The translated audio plays back through your
speakers and both the source and translation transcripts appear live.

> Use headphones. The mic has echo cancellation on, but playing the translation
> out loud near an open mic invites feedback loops.

## How it works

| Stage    | Detail                                                                   |
| -------- | ------------------------------------------------------------------------ |
| Capture  | `getUserMedia` → `AudioContext(16 kHz)` → AudioWorklet → PCM16 100 ms chunks |
| Upstream | Binary WebSocket frames → `session.send_realtime_input(audio=Blob(...))` |
| Model    | `client.aio.live.connect()` with `response_modalities=["AUDIO"]`         |
| Downstream | `session.receive()` → audio bytes (24 kHz) + input/output transcripts  |
| Playback | PCM16 @ 24 kHz → scheduled `AudioBufferSourceNode`s for gapless audio    |

The interpreter persona, voice, and transcription config live in
`services/gemini_live.py`.

## Notes / deviations from the original spec

These are deliberate corrections so the app actually runs against the real API:

- **Model.** The default is `gemini-3.5-live-translate-preview` — Google's
  dedicated low-latency speech-to-speech **translation** model (public preview,
  70+ languages, ~$0.023/min, needs billing). It is configured with
  `translation_config(target_language_code=…, echo_target_language=True)`: the
  source is auto-detected and **Language B is the language you hear**; it also
  preserves the speaker's own voice (so the voice picker doesn't apply). For a
  free-tier, persona-based two-way interpreter with a selectable voice, set
  `GEMINI_LIVE_MODEL=gemini-2.0-flash-exp`. The backend auto-switches its config
  shape based on the model id.
- **Response modalities.** The Live API accepts a *single* modality. To get both
  spoken output and on-screen text we request `AUDIO` and additionally enable
  `input_audio_transcription` + `output_audio_transcription`. (`["AUDIO","TEXT"]`
  together is rejected by the API.)
- **Output rate is 24 kHz, not 16 kHz.** Input to Gemini is 16 kHz; Gemini's
  synthesized audio comes back at 24 kHz, so capture and playback use separate
  `AudioContext`s at their respective rates.
- **AudioWorklet over `ScriptProcessorNode`.** The spec referenced both;
  `ScriptProcessorNode` is deprecated and runs on the main thread, so capture
  uses an AudioWorklet (`audio-processor.js`) for glitch-free sampling.

## Configuration (`.env`)

| Variable            | Default               | Purpose                              |
| ------------------- | --------------------- | ------------------------------------ |
| `GEMINI_API_KEY`    | —                     | **Required.** Your Gemini API key.   |
| `GEMINI_LIVE_MODEL` | `gemini-3.5-live-translate-preview` | Live model id (translate or persona). |
| `RISK_MODEL`        | `gemini-3.1-flash-lite` | Per-turn risk analysis (fast/cheap). |
| `SUMMARY_MODEL`     | `gemini-3.5-flash`    | Meeting summaries (stronger Flash).  |
| `GEMINI_VOICE`      | `Aoede`               | Prebuilt TTS voice.                  |
| `ACCESS_TOKEN`      | _(empty)_             | Gate the WebSocket; client must send `?token=`. **Set this for any public deploy.** |
| `HOST` / `PORT`     | `0.0.0.0` / `8000`    | Server bind address.                 |

## Security / billing

The API key lives **only** in the server's environment (`GEMINI_API_KEY`), never
in the code or the APK — so a public GitHub repo does not leak it.

The real exposure is the deployed endpoint: without protection, anyone who
learns the public URL can stream audio and spend your Gemini quota. Set
**`ACCESS_TOKEN`** on the server (Render's blueprint auto-generates a strong
one) and enter the same value in the app's *Access Token* field. Connections
without a matching `?token=` are rejected before any Gemini session is opened,
so they cost nothing. The token is never committed to the repo. As extra safety
nets, consider making the repo private and setting a billing cap in Google AI
Studio.
