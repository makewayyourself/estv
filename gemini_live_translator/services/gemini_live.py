"""
Core integration with the Google Gemini Multimodal **Live API**.

This module is intentionally thin: it owns nothing about the network transport
that talks to the browser (that lives in ``main.py``). Its only job is to:

  1. Construct an authenticated async ``google-genai`` client.
  2. Build a ``LiveConnectConfig`` that turns Gemini into a real-time
     simultaneous interpreter (audio-in -> translated audio-out + transcripts).

A few important real-world notes (the original spec was slightly aspirational):

* "Gemini 3.5" does not exist as a public model. The Live API is served by
  models such as ``gemini-2.0-flash-exp`` and ``gemini-live-2.5-flash-preview``.
  The model id is configurable via the ``GEMINI_LIVE_MODEL`` env var.

* The Live API accepts **exactly one** value in ``response_modalities`` —
  either ``AUDIO`` or ``TEXT``, not both. To get *audio output AND live text*
  we request ``AUDIO`` and additionally enable input/output transcription,
  which streams text alongside the synthesized speech.

* Input audio must be raw little-endian PCM, 16-bit, 16 kHz, mono.
  Output audio from Gemini is raw PCM 16-bit at **24 kHz**, mono — the client
  must play it back at 24 kHz, not 16 kHz.
"""

from __future__ import annotations

import os

from google import genai
from google.genai import types

# --- Configuration ---------------------------------------------------------

DEFAULT_MODEL = os.getenv("GEMINI_LIVE_MODEL", "gemini-2.0-flash-exp")
DEFAULT_VOICE = os.getenv("GEMINI_VOICE", "Aoede")

# Text model used to summarize the accumulated meeting transcript. A regular
# (non-live) fast model is ideal here.
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gemini-2.0-flash")

# Audio format constants shared with the client. Input is what we feed Gemini;
# output is what Gemini hands back to us.
INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 24_000
INPUT_MIME_TYPE = f"audio/pcm;rate={INPUT_SAMPLE_RATE}"

# Languages the interpreter can work in. Keys are the codes the client sends;
# values are the English names used inside the prompt (the model understands
# these reliably). Add more here to expand coverage.
SUPPORTED_LANGUAGES: dict[str, str] = {
    "ko": "Korean",
    "en": "English",
    "ja": "Japanese",
    "zh": "Mandarin Chinese",
    "fr": "French",
    "es": "Spanish",
    "ar": "Arabic",
    "ru": "Russian",
}

DEFAULT_LANG_A = "ko"
DEFAULT_LANG_B = "en"


AUTO_DETECT = "auto"  # special lang_a value: detect any language -> lang_b


def normalize_lang(code: str | None, fallback: str, allow_auto: bool = False) -> str:
    """Return a supported language code, falling back if unknown/empty.

    When ``allow_auto`` is set, the special ``"auto"`` value is accepted (used
    for the source side to mean "detect any language").
    """
    code = (code or "").strip().lower()
    if allow_auto and code == AUTO_DETECT:
        return AUTO_DETECT
    return code if code in SUPPORTED_LANGUAGES else fallback


def build_system_instruction(lang_a: str, lang_b: str) -> str:
    """Build the interpreter persona.

    * If ``lang_a`` is ``"auto"``: detect any incoming language and translate it
      into ``lang_b`` (one-directional comprehension mode).
    * Otherwise: two-way between ``lang_a`` and ``lang_b`` (auto-detect which of
      the two is spoken and translate into the other).
    """
    b = SUPPORTED_LANGUAGES.get(lang_b, "English")

    if lang_a == AUTO_DETECT:
        return (
            "You are a highly professional, real-time simultaneous interpreter. "
            "Detect the language of the incoming speech and immediately translate "
            f"it into natural, fluent {b}. If the speech is already in {b}, simply "
            f"restate it clearly in {b}. Preserve tone, idioms and cultural nuance "
            "rather than translating word for word. Output ONLY the spoken "
            "translation itself. Never add conversational filler such as 'Sure' or "
            "'Here is the translation', and never explain what you are doing."
        )

    a = SUPPORTED_LANGUAGES.get(lang_a, "Korean")

    if a == b:
        # Degenerate pair: nothing to translate between. Keep it harmless.
        return (
            f"You are a professional interpreter. Repeat back what is said in "
            f"clear, natural {a}. Output only the speech itself, no filler."
        )

    return (
        "You are a highly professional, real-time simultaneous interpreter "
        f"working between {a} and {b}. "
        f"Detect whether the speaker is speaking {a} or {b}. "
        f"If they speak {a}, immediately translate it into natural, fluent {b}. "
        f"If they speak {b}, immediately translate it into natural, fluent {a}. "
        "Preserve tone, idioms and cultural nuance rather than translating word "
        "for word. Output ONLY the spoken translation itself. Never add "
        "conversational filler such as 'Sure', 'Okay', or 'Here is the "
        "translation', and never explain what you are doing."
    )


def build_summary_prompt(transcript: str, language: str) -> str:
    """Build a meeting-notes summarization prompt in the requested language."""
    lang_name = SUPPORTED_LANGUAGES.get(language, "English")
    return (
        "You are a precise meeting-notes assistant. The transcript below is from "
        "a live interpreted meeting; each turn may contain the original utterance "
        "and its translation — treat them as the same content, do not duplicate.\n\n"
        f"Write ALL of your output, including the section headings, in {lang_name}.\n"
        "Use this structure with Markdown headings:\n"
        "1. A one-line summary of the meeting.\n"
        "2. Key discussion points (concise bullets).\n"
        "3. Decisions made.\n"
        "4. Action items — for each: owner, task, and due date if mentioned.\n\n"
        "Be faithful to the transcript; do not invent facts. If a section has "
        "nothing, state that briefly.\n\n"
        "=== TRANSCRIPT ===\n"
        f"{transcript}"
    )


def build_pronounce_prompt(text: str, script: str) -> str:
    """Build a transliteration prompt: how to *pronounce* the text (no meaning).

    ``script`` is "hangul" (Korean phonetic) or "roman" (Latin romanization).
    """
    if script == "hangul":
        target = (
            "Korean Hangul characters that approximate how it sounds to a Korean "
            "speaker"
        )
    else:
        target = "Latin-alphabet (romanized) letters showing how it sounds"
    return (
        "You are a pronunciation guide. Transliterate the following text using "
        f"{target}. Do NOT translate the meaning — show ONLY how it is "
        "pronounced. Output only the pronunciation on a single line, nothing "
        "else.\n\nTEXT:\n" + text
    )


def get_client() -> genai.Client:
    """Return an async-capable Gemini client, or raise if no key is configured.

    The returned client exposes the bidirectional Live API at
    ``client.aio.live.connect(...)``.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return genai.Client(api_key=api_key)


def build_config(
    voice_name: str | None = None,
    lang_a: str = DEFAULT_LANG_A,
    lang_b: str = DEFAULT_LANG_B,
    system_instruction: str | None = None,
) -> types.LiveConnectConfig:
    """Build the Live session configuration.

    Args:
        voice_name: Prebuilt voice for the synthesized translation.
        lang_a: First language code of the interpreting pair.
        lang_b: Second language code of the interpreting pair.
        system_instruction: Explicit persona override (skips lang_a/lang_b).
    """
    voice = voice_name or DEFAULT_VOICE
    instruction = system_instruction or build_system_instruction(lang_a, lang_b)

    return types.LiveConnectConfig(
        # Single modality only — see the module docstring.
        response_modalities=["AUDIO"],
        system_instruction=types.Content(
            parts=[types.Part.from_text(text=instruction)]
        ),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
            )
        ),
        # These two give us live text transcripts for the UI even though the
        # primary response modality is AUDIO.
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )
