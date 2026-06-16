# Gemini Live Translator

A real-time, low-latency, **audio-to-audio** bidirectional interpreter built on
the Google Gemini **Multimodal Live API**. Speak Korean → hear natural English;
speak English → hear polite Korean — with live text transcripts on screen.

There is no STT → Translate → TTS chain. Microphone audio is streamed directly
into Gemini's bidirectional live channel, and translated audio streams straight
back out, which is what keeps latency low.

> **📱 안드로이드 앱으로 쓰려면 → [`ANDROID.md`](./ANDROID.md) (한글 가이드)**
> Capacitor로 APK를 빌드하고, 백엔드는 클라우드(Render)에 배포하는 전체 과정.

```
[ Browser ]  ⇄ (PCM16 audio + JSON over WebSocket) ⇄  [ FastAPI ]  ⇄ (Live API) ⇄  [ Gemini ]
```

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
├── render.yaml                 # one-click cloud deploy blueprint
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

- **Model name.** "Gemini 3.5" is not a public model. The Live API is served by
  models like `gemini-2.0-flash-exp` (default) and `gemini-live-2.5-flash-preview`.
  Override with `GEMINI_LIVE_MODEL` in `.env`.
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
| `GEMINI_LIVE_MODEL` | `gemini-2.0-flash-exp`| Live model id.                       |
| `GEMINI_VOICE`      | `Aoede`               | Prebuilt TTS voice.                  |
| `HOST` / `PORT`     | `0.0.0.0` / `8000`    | Server bind address.                 |
