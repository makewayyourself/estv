"""
FastAPI entrypoint for the Gemini Live audio-to-audio translator.

Responsibilities:
  * Serve the static frontend (``static/index.html`` + assets).
  * Expose a WebSocket endpoint at ``/api/stream`` that bridges the browser to
    a Gemini Live session.

Bridge topology, per connection::

    browser ──(PCM16 16kHz binary + JSON control)──▶  FastAPI  ──▶  Gemini
    browser ◀──(PCM16 24kHz binary + JSON transcripts)── FastAPI  ◀── Gemini

Two asyncio tasks run concurrently for the lifetime of each connection:
  * ``pump_client_to_gemini`` — forwards mic audio upstream.
  * ``pump_gemini_to_client`` — forwards translated audio + transcripts down.
"""

from __future__ import annotations

import asyncio
import base64
import enum
import json
import logging
import os
import secrets
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from google.genai import types
from pydantic import BaseModel

from services.gemini_live import (
    DEFAULT_LANG_A,
    DEFAULT_LANG_B,
    DEFAULT_MODEL,
    INPUT_MIME_TYPE,
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    RISK_MODEL,
    SUMMARY_MODEL,
    SUPPORTED_LANGUAGES,
    build_config,
    build_pronounce_prompt,
    build_qa_prompt,
    build_risk_prompt,
    build_summary_prompt,
    get_client,
    normalize_lang,
)
from services.exporters import EXPORTERS

# Cap how much transcript we send to the summarizer (keep the most recent part).
MAX_SUMMARY_CHARS = 40_000

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gemini_live_translator")

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Gemini Live Translator")

# Allow the packaged Android app (capacitor://localhost / https://localhost) to
# call the HTTP API. WebSocket handshakes are not subject to CORS, but the
# health probe is, so we keep this permissive.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict:
    """Lightweight readiness probe; also reports whether a key is configured."""
    return {
        "status": "ok",
        "model": DEFAULT_MODEL,
        "api_key_configured": bool(os.getenv("GEMINI_API_KEY")),
        "auth_required": bool(os.getenv("ACCESS_TOKEN")),
        "risk_model": RISK_MODEL,
        "summary_model": SUMMARY_MODEL,
        "languages": SUPPORTED_LANGUAGES,
        "input_sample_rate": INPUT_SAMPLE_RATE,
        "output_sample_rate": OUTPUT_SAMPLE_RATE,
    }


def _check_token(provided: str | None) -> None:
    """Raise 401 if an ACCESS_TOKEN is configured and the value doesn't match."""
    expected = os.getenv("ACCESS_TOKEN")
    if expected and not secrets.compare_digest(provided or "", expected):
        raise HTTPException(status_code=401, detail="Unauthorized: invalid access token")


class SummarizeRequest(BaseModel):
    transcript: str
    language: str = DEFAULT_LANG_A
    token: str | None = None


@app.post("/api/summarize")
async def summarize(req: SummarizeRequest) -> dict:
    """Summarize the accumulated meeting transcript into structured notes.

    Runs on the server (where the Gemini key lives) and is gated by the same
    ACCESS_TOKEN as the audio stream.
    """
    _check_token(req.token)

    transcript = (req.transcript or "").strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript is empty")
    # Keep the most recent portion if the meeting is very long.
    if len(transcript) > MAX_SUMMARY_CHARS:
        transcript = transcript[-MAX_SUMMARY_CHARS:]

    try:
        client = get_client()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    language = normalize_lang(req.language, DEFAULT_LANG_A)
    prompt = build_summary_prompt(transcript, language)

    try:
        response = await client.aio.models.generate_content(
            model=SUMMARY_MODEL, contents=prompt
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Summarization failed")
        raise HTTPException(status_code=502, detail=f"Summarization failed: {exc}")

    return {"summary": response.text or "", "model": SUMMARY_MODEL, "language": language}


class RiskLevel(str, enum.Enum):
    none = "none"
    low = "low"
    medium = "medium"
    high = "high"


class RiskAnalysis(BaseModel):
    """Structured per-turn analysis (risk + meaning clarification)."""

    risk_level: RiskLevel
    risk_types: List[str]
    subtitle_alert: str
    reason: str
    suggested_question: str
    # Meaning-clarification (mis-recognition due to unclear pronunciation).
    clarify_suspected: bool
    clarify_did_you_mean: str
    clarify_corrected_translation: str


class AnalyzeRequest(BaseModel):
    original: str
    translation: str = ""
    alert_language: str = DEFAULT_LANG_A
    context: str = ""
    history: str = ""
    token: str | None = None


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest) -> dict:
    """Risk-detect one finalized utterance (runs off the real-time path).

    Returns structured guidance: risk level, types, a short subtitle alert, and
    a suggested follow-up question — written in ``alert_language``.
    """
    _check_token(req.token)

    original = (req.original or "").strip()
    if not original:
        raise HTTPException(status_code=400, detail="Original text is empty")

    try:
        client = get_client()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    alert_language = normalize_lang(req.alert_language, DEFAULT_LANG_A)
    prompt = build_risk_prompt(
        original[:2000],
        (req.translation or "")[:2000],
        alert_language,
        req.context[:200],
        (req.history or "")[:4000],
    )
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=RiskAnalysis,
        temperature=0.2,
    )
    try:
        response = await client.aio.models.generate_content(
            model=RISK_MODEL, contents=prompt, config=config
        )
        data = json.loads(response.text or "{}")
    except Exception as exc:  # noqa: BLE001 — never break the UI on analysis failure
        logger.warning("Per-turn analysis failed: %s", exc)
        return {
            "risk_level": "none", "risk_types": [], "subtitle_alert": "",
            "reason": "", "suggested_question": "", "clarify_suspected": False,
            "clarify_did_you_mean": "", "clarify_corrected_translation": "",
        }

    return data


class AskRequest(BaseModel):
    transcript: str
    question: str
    language: str = DEFAULT_LANG_A
    token: str | None = None


@app.post("/api/ask")
async def ask(req: AskRequest) -> dict:
    """Answer a question grounded in the saved conversation transcript."""
    _check_token(req.token)

    transcript = (req.transcript or "").strip()
    question = (req.question or "").strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript is empty")
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty")
    if len(transcript) > MAX_SUMMARY_CHARS:
        transcript = transcript[-MAX_SUMMARY_CHARS:]

    try:
        client = get_client()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    language = normalize_lang(req.language, DEFAULT_LANG_A)
    prompt = build_qa_prompt(transcript, question[:1000], language)
    try:
        response = await client.aio.models.generate_content(
            model=SUMMARY_MODEL, contents=prompt
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Q&A failed")
        raise HTTPException(status_code=502, detail=f"Q&A failed: {exc}")

    return {"answer": (response.text or "").strip()}


class ExportEntry(BaseModel):
    time: str = ""
    source: str = ""
    translation: str = ""
    risk: str = ""


class ExportRequest(BaseModel):
    title: str = "회의 노트 / Meeting Notes"
    entries: List[ExportEntry] = []
    summary: str = ""
    format: str = "md"  # md | docx | pdf
    token: str | None = None


@app.post("/api/export")
async def export(req: ExportRequest) -> Response:
    """Render the meeting notes as a downloadable md / docx / pdf file."""
    _check_token(req.token)

    fmt = (req.format or "md").lower()
    if fmt not in EXPORTERS:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt}")
    if not req.entries:
        raise HTTPException(status_code=400, detail="No notes to export")

    builder, media_type, ext = EXPORTERS[fmt]
    entries = [e.model_dump() for e in req.entries]
    try:
        content = builder(req.title, entries, req.summary)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Export failed")
        raise HTTPException(status_code=500, detail=f"Export failed: {exc}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    filename = f"meeting-notes-{stamp}.{ext}"
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


class PronounceRequest(BaseModel):
    text: str
    script: str = "roman"  # "roman" or "hangul"
    token: str | None = None


@app.post("/api/pronounce")
async def pronounce(req: PronounceRequest) -> dict:
    """Return a phonetic transliteration (pronunciation) of a sentence."""
    _check_token(req.token)

    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty")
    text = text[:2000]
    script = "hangul" if req.script == "hangul" else "roman"

    try:
        client = get_client()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    prompt = build_pronounce_prompt(text, script)
    try:
        response = await client.aio.models.generate_content(
            model=RISK_MODEL, contents=prompt
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pronunciation failed")
        raise HTTPException(status_code=502, detail=f"Pronunciation failed: {exc}")

    return {"pronunciation": (response.text or "").strip(), "script": script}


async def _send_json(ws: WebSocket, payload: dict) -> None:
    """Best-effort JSON send that never raises on a closed socket."""
    with suppress(Exception):
        await ws.send_text(json.dumps(payload))


@app.websocket("/api/stream")
async def stream(ws: WebSocket) -> None:
    await ws.accept()
    logger.info("Client connected")

    # --- Access control -----------------------------------------------------
    # If ACCESS_TOKEN is set on the server, reject any connection that does not
    # present the matching ?token=... — this stops strangers who merely learn
    # the public URL from spending your Gemini quota. We reject *before* opening
    # a Gemini session, so an unauthorized connection costs nothing.
    expected_token = os.getenv("ACCESS_TOKEN")
    if expected_token:
        provided = ws.query_params.get("token", "")
        if not secrets.compare_digest(provided, expected_token):
            logger.warning("Rejected connection: invalid or missing token")
            await _send_json(
                ws, {"type": "error", "message": "Unauthorized: invalid access token"}
            )
            await ws.close(code=1008)  # policy violation
            return

    # Session settings come from the connection query string (the model config
    # must be fixed before the Gemini session opens, so they can't be sent as a
    # later message). All fall back to server defaults.
    voice = ws.query_params.get("voice") or os.getenv("GEMINI_VOICE", "Aoede")
    lang_a = normalize_lang(
        ws.query_params.get("a"), os.getenv("LANG_A", DEFAULT_LANG_A), allow_auto=True
    )
    lang_b = normalize_lang(
        ws.query_params.get("b"), os.getenv("LANG_B", DEFAULT_LANG_B)
    )

    try:
        client = get_client()
    except RuntimeError as exc:
        await _send_json(ws, {"type": "error", "message": str(exc)})
        await ws.close()
        return

    config = build_config(voice_name=voice, lang_a=lang_a, lang_b=lang_b)

    try:
        async with client.aio.live.connect(model=DEFAULT_MODEL, config=config) as session:
            await _send_json(
                ws,
                {
                    "type": "status",
                    "state": "connected",
                    "model": DEFAULT_MODEL,
                    "voice": voice,
                    "lang_a": lang_a,
                    "lang_b": lang_b,
                    "input_sample_rate": INPUT_SAMPLE_RATE,
                    "output_sample_rate": OUTPUT_SAMPLE_RATE,
                },
            )

            async def pump_client_to_gemini() -> None:
                """Forward microphone audio (and control messages) to Gemini."""
                while True:
                    message = await ws.receive()

                    if message["type"] == "websocket.disconnect":
                        raise WebSocketDisconnect()

                    # Binary frame == raw PCM16 16kHz audio chunk.
                    data = message.get("bytes")
                    if data:
                        await session.send_realtime_input(
                            audio=types.Blob(data=data, mime_type=INPUT_MIME_TYPE)
                        )
                        continue

                    # Text frame == JSON control message.
                    text = message.get("text")
                    if not text:
                        continue
                    try:
                        control = json.loads(text)
                    except json.JSONDecodeError:
                        continue

                    action = control.get("action")
                    if action == "audio":
                        # Audio delivered base64-encoded inside JSON (fallback path).
                        raw = base64.b64decode(control.get("data", ""))
                        if raw:
                            await session.send_realtime_input(
                                audio=types.Blob(data=raw, mime_type=INPUT_MIME_TYPE)
                            )
                    elif action == "end":
                        # Signal end of the user's turn / mic stopped.
                        await session.send_realtime_input(audio_stream_end=True)
                    # Unknown actions are ignored on purpose.

            async def pump_gemini_to_client() -> None:
                """Forward translated audio + transcripts back to the browser."""
                while True:
                    async for response in session.receive():
                        # 1) Synthesized translated audio (PCM16 @ 24kHz).
                        if response.data:
                            with suppress(Exception):
                                await ws.send_bytes(response.data)

                        server_content = response.server_content
                        if not server_content:
                            continue

                        # 2) Transcript of what the *user* said (source language).
                        if server_content.input_transcription and (
                            server_content.input_transcription.text
                        ):
                            await _send_json(
                                ws,
                                {
                                    "type": "transcript",
                                    "role": "source",
                                    "text": server_content.input_transcription.text,
                                },
                            )

                        # 3) Transcript of the *translation* (target language).
                        if server_content.output_transcription and (
                            server_content.output_transcription.text
                        ):
                            await _send_json(
                                ws,
                                {
                                    "type": "transcript",
                                    "role": "translation",
                                    "text": server_content.output_transcription.text,
                                },
                            )

                        # 4) The model was interrupted (user barged in).
                        if server_content.interrupted:
                            await _send_json(ws, {"type": "interrupted"})

                        # 5) A full turn finished — flush UI line buffers.
                        if server_content.turn_complete:
                            await _send_json(ws, {"type": "turn_complete"})

            # Run both directions until either side closes.
            upstream = asyncio.create_task(pump_client_to_gemini())
            downstream = asyncio.create_task(pump_gemini_to_client())
            done, pending = await asyncio.wait(
                {upstream, downstream}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            # Surface any non-disconnect exception from the finished task.
            for task in done:
                exc = task.exception()
                if exc and not isinstance(exc, WebSocketDisconnect):
                    raise exc

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as exc:  # noqa: BLE001 - report any failure to the client
        logger.exception("Session error")
        await _send_json(ws, {"type": "error", "message": str(exc)})
    finally:
        with suppress(Exception):
            await ws.close()
        logger.info("Connection closed")


# Mounted last (at the web root) so the explicit /api routes above take
# precedence. Serving at "/" — rather than "/static" — makes the same asset
# layout work both here and inside the packaged Capacitor app, where the web
# bundle is served from the app root.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
