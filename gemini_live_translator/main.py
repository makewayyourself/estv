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
import json
import logging
import os
import secrets
from contextlib import suppress
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from google.genai import types

from services.gemini_live import (
    DEFAULT_LANG_A,
    DEFAULT_LANG_B,
    DEFAULT_MODEL,
    INPUT_MIME_TYPE,
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    SUPPORTED_LANGUAGES,
    build_config,
    get_client,
    normalize_lang,
)

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
        "languages": SUPPORTED_LANGUAGES,
        "input_sample_rate": INPUT_SAMPLE_RATE,
        "output_sample_rate": OUTPUT_SAMPLE_RATE,
    }


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
        ws.query_params.get("a"), os.getenv("LANG_A", DEFAULT_LANG_A)
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
