"""
Multi-device "rooms" for QR-based group interpreting.

A host publishes microphone audio into a room; participants join via a shared
link/QR and pick the language they want to HEAR. For each distinct language a
dedicated Gemini Live translate session is opened (so participants who share a
language share one session — cost scales with the number of languages, not
participants). Each session's translated audio + captions are broadcast to the
participants listening in that language.

In-memory only (single server instance — fine for Render's free tier).
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress

from google.genai import types

from .gemini_live import (
    DEFAULT_MODEL,
    INPUT_MIME_TYPE,
    build_config,
)

logger = logging.getLogger("hyun_rooms")


class LangChannel:
    """One Gemini Live translate session for a single target language."""

    def __init__(self, room: "Room", lang: str) -> None:
        self.room = room
        self.lang = lang
        self.subscribers: set = set()  # participant WebSockets
        self.session = None
        self._cm = None
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        config = build_config(lang_b=self.lang)  # translate model, target=lang
        self._cm = self.room.client.aio.live.connect(model=DEFAULT_MODEL, config=config)
        self.session = await self._cm.__aenter__()
        self._task = asyncio.create_task(self._pump_out())

    async def _pump_out(self) -> None:
        try:
            async for resp in self.session.receive():
                if resp.data:
                    await self._broadcast_bytes(resp.data)
                sc = resp.server_content
                if not sc:
                    continue
                if sc.input_transcription and sc.input_transcription.text:
                    await self._broadcast_json(
                        {"type": "transcript", "role": "source", "text": sc.input_transcription.text}
                    )
                if sc.output_transcription and sc.output_transcription.text:
                    await self._broadcast_json(
                        {"type": "transcript", "role": "translation", "text": sc.output_transcription.text}
                    )
                if sc.turn_complete or sc.generation_complete:
                    await self._broadcast_json({"type": "turn_complete"})
        except Exception as exc:  # noqa: BLE001
            logger.warning("channel %s/%s ended: %s", self.room.id, self.lang, exc)

    async def feed(self, pcm: bytes) -> None:
        if self.session:
            with suppress(Exception):
                await self.session.send_realtime_input(
                    audio=types.Blob(data=pcm, mime_type=INPUT_MIME_TYPE)
                )

    async def _broadcast_bytes(self, data: bytes) -> None:
        for ws in list(self.subscribers):
            with suppress(Exception):
                await ws.send_bytes(data)

    async def _broadcast_json(self, payload: dict) -> None:
        text = json.dumps(payload)
        for ws in list(self.subscribers):
            with suppress(Exception):
                await ws.send_text(text)

    async def close(self) -> None:
        if self._task:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
        if self._cm:
            with suppress(Exception):
                await self._cm.__aexit__(None, None, None)


class Room:
    def __init__(self, room_id: str, client) -> None:
        self.id = room_id
        self.client = client
        self.host_ws = None
        self.channels: dict[str, LangChannel] = {}
        self._lock = asyncio.Lock()

    async def notify_host(self) -> None:
        if self.host_ws:
            with suppress(Exception):
                await self.host_ws.send_text(
                    json.dumps({"type": "participants", "count": self.participant_count()})
                )

    async def ensure_channel(self, lang: str) -> LangChannel:
        async with self._lock:
            ch = self.channels.get(lang)
            if ch is None:
                ch = LangChannel(self, lang)
                await ch.start()
                self.channels[lang] = ch
            return ch

    async def feed_audio(self, pcm: bytes) -> None:
        chans = list(self.channels.values())
        if chans:
            await asyncio.gather(*(c.feed(pcm) for c in chans), return_exceptions=True)

    def participant_count(self) -> int:
        return sum(len(c.subscribers) for c in self.channels.values())

    async def close(self) -> None:
        for ch in list(self.channels.values()):
            await ch.close()
        self.channels.clear()


class RoomManager:
    def __init__(self) -> None:
        self.rooms: dict[str, Room] = {}

    def create(self, room_id: str, client) -> Room:
        room = Room(room_id, client)
        self.rooms[room_id] = room
        return room

    def get(self, room_id: str) -> Room | None:
        return self.rooms.get(room_id)

    async def close(self, room_id: str) -> None:
        room = self.rooms.pop(room_id, None)
        if room:
            await room.close()


manager = RoomManager()
