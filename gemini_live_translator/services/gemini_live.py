"""
Core integration with the Google Gemini Multimodal **Live API**.

This module is intentionally thin: it owns nothing about the network transport
that talks to the browser (that lives in ``main.py``). Its only job is to:

  1. Construct an authenticated async ``google-genai`` client.
  2. Build a ``LiveConnectConfig`` that turns Gemini into a real-time
     simultaneous interpreter (audio-in -> translated audio-out + transcripts).

A few important real-world notes:

* The default model is ``gemini-3.5-live-translate-preview`` — Google's dedicated
  low-latency speech-to-speech translation model (public preview, 70+ languages).
  It is configured via ``translation_config`` (a target language) rather than a
  persona prompt, and preserves the speaker's own voice. The general persona
  path (e.g. ``gemini-2.0-flash-exp``) is still supported as a fallback. Switch
  with the ``GEMINI_LIVE_MODEL`` env var.

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

# Default to the dedicated low-latency speech-to-speech translation model
# (public preview, 70+ languages). Override with GEMINI_LIVE_MODEL. For the
# general persona-based path, set e.g. gemini-2.0-flash-exp.
DEFAULT_MODEL = os.getenv("GEMINI_LIVE_MODEL", "gemini-3.5-live-translate-preview")
DEFAULT_VOICE = os.getenv("GEMINI_VOICE", "Aoede")

# Text models for the non-realtime helpers.
#  * RISK_MODEL    : lightweight per-turn risk check → fast/cheap "Lite".
#  * CLARIFY_MODEL : meaning-clarification reasoning (infer intent from garbled
#                    speech) → stronger Flash; also used for the per-turn call
#                    whenever Clarify is on, so reasoning is deeper where it
#                    matters most.
#  * SUMMARY_MODEL : whole-transcript summary / Q&A → stronger Flash.
# (gemini-2.0-flash / 2.0-flash-lite were shut down on 2026-06-01;
#  gemini-3.5-pro is not yet a public API model id as of June 2026.)
RISK_MODEL = os.getenv("RISK_MODEL", "gemini-3.1-flash-lite")
CLARIFY_MODEL = os.getenv("CLARIFY_MODEL", "gemini-3.5-flash")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gemini-3.5-flash")
# Precision-caption cascade: per-utterance audio → verbatim transcription +
# text translation on a stable GA model (accuracy-first, Chattr-style),
# bypassing the S2S preview model whose transcripts are a byproduct.
CAPTION_MODEL = os.getenv("CAPTION_MODEL", "gemini-3.5-flash")

# Extra "thinking" budget (tokens) for the deeper analysis path. 0 disables it.
# Boosts reasoning quality for clarification at some extra cost/latency (off the
# real-time path, so it never slows the live translation).
ANALYSIS_THINKING_BUDGET = int(os.getenv("ANALYSIS_THINKING_BUDGET", "4096"))

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


def build_caption_prompt(
    source_lang: str,
    target_lang: str,
    glossary: str = "",
    history: str = "",
    candidates: list[str] | None = None,
) -> str:
    """Precision transcription + translation of ONE utterance (audio attached).

    The language pin, glossary and conversation context are injected directly
    into the recognition step — the main accuracy levers the live S2S path
    cannot offer.
    """
    tgt = SUPPORTED_LANGUAGES.get(target_lang, "English")
    if candidates:
        # The meeting's language set (the user's 3 display languages): constrain
        # recognition to these instead of a 70-language free-for-all.
        names = ", ".join(SUPPORTED_LANGUAGES.get(c, c) for c in candidates)
        src_line = (
            f"The speaker is speaking ONE of these languages: {names}. Identify "
            "which one and transcribe in that language only — never any other "
            "language."
        )
    elif source_lang and source_lang != AUTO_DETECT:
        src_line = (
            f"The speaker is speaking {SUPPORTED_LANGUAGES.get(source_lang, source_lang)}. "
            "Transcribe in that language only — do not switch languages."
        )
    else:
        src_line = "Detect the spoken language and transcribe in it."
    glossary_line = (
        f"\nGlossary — these are the CORRECT spellings of names/terms that may "
        f"occur; prefer them over phonetically similar words: {glossary.strip()}\n"
        if glossary.strip() else ""
    )
    history_line = (
        f"\nRecent conversation (context for ambiguous words):\n{history.strip()}\n"
        if history.strip() else ""
    )
    return (
        "You are a professional transcriber and translator for a live business "
        "conversation. For the attached audio of ONE utterance:\n"
        f"1. Transcribe it VERBATIM. {src_line} Fix nothing except obvious "
        "disfluencies (um/uh). Use the glossary and context to resolve unclear "
        "words — a proper noun misheard as a common word should be written as "
        "the glossary/context indicates.\n"
        f"2. Translate the transcription into natural, fluent {tgt}, preserving "
        "tone and business register.\n"
        f"{glossary_line}{history_line}"
        "If the audio contains no intelligible speech (noise/music/silence), "
        "return empty strings for both fields."
    )


def build_digest_prompt(transcript: str, language: str) -> str:
    """Rolling live digest: a short running summary of the conversation so far
    plus a few smart questions the user could ask next to move a business
    discussion forward. Output is JSON, written in ``language``."""
    lang_name = SUPPORTED_LANGUAGES.get(language, "English")
    return (
        "You are a real-time meeting copilot for a live interpreted business "
        "conversation. From the running transcript below, produce:\n"
        "1. summary: 2-4 very short bullet lines capturing where the "
        "conversation stands right now (decisions, numbers, open points) — not "
        "a transcript, just the gist. Keep it current; weight recent turns.\n"
        "2. questions: up to 3 concise, high-value questions the listener could "
        "ask NEXT — to clarify ambiguity, pin down terms/prices/dates, or move "
        "the deal forward. Make them specific to what was actually said, not "
        "generic. Empty list if nothing useful.\n"
        f"Write everything in {lang_name}. Be brief.\n\n"
        "=== TRANSCRIPT (most recent last) ===\n"
        f"{transcript}"
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


def build_risk_prompt(
    original: str,
    translation: str,
    alert_language: str,
    context: str = "",
    history: str = "",
    target_language: str = "en",
) -> str:
    """Build the per-turn copilot prompt: risk + clarify + answer + upgrade.

    Returns structured JSON (enforced via response_schema). ``alert_language`` is
    the user's language (for explanations); ``target_language`` is what they
    actually speak to the other party. ``history`` is recent context.
    """
    lang_name = SUPPORTED_LANGUAGES.get(alert_language, "English")
    tgt_name = SUPPORTED_LANGUAGES.get(target_language, "English")
    context_line = ""
    if context.strip():
        context_line = (
            f"\nBusiness/industry context: {context.strip()}. Consider domain "
            "terms (e.g. for oil/commodity trading: LOI, SCO, FCO, POP, SGS/BV, "
            "Platts, CIF/FOB, upfront fee, allocation, mandate) and flag them.\n"
        )
    history_line = ""
    if history.strip():
        history_line = f"\nRecent conversation (for context):\n{history.strip()}\n"

    return (
        "You are a real-time business-interpreting copilot. For the LATEST "
        "utterance, do FOUR things.\n\n"
        "PART A — Risk: flag anything the listener should double-check before "
        "acting (money/payment, contracts & liability, guarantees/penalties/"
        "advance-payment, numbers·dates·quantities·units, rude/blunt phrasing, "
        "sensitive data, low translation confidence). Be conservative — 'none' "
        "for ordinary small talk; medium/high only for a real business risk.\n\n"
        "PART B — Clarify: live transcription can mishear unclear speech. Read "
        "the LATEST utterance IN THE FLOW of the recent conversation and judge "
        "whether every content word actually makes sense there — not just "
        "whether the sentence is grammatical. Two checks:\n"
        "(1) Proper-noun plausibility: if a person/company/product name was "
        "transcribed as a common word that cannot plausibly be a name (e.g. a "
        "Korean self-introduction rendered as '내 이름은 안녕' — '안녕' is a "
        "greeting, never a name), that IS a mis-recognition.\n"
        "(2) Context consistency: if a word is impossible or incoherent given "
        "the conversation history, the business context, or the glossary (e.g. "
        "a price, product, or term that contradicts what was just discussed), "
        "treat it as misheard and infer the intended word FROM that context.\n"
        "(3) Phonetic confusion + language convention: consider PHONETICALLY "
        "SIMILAR words the speaker more plausibly said, and reject readings a "
        "native speaker would never produce. Korean examples: ASR often turns "
        "'실시간' (real-time) into '7시간/칠시간' (ㅅ↔ㅊ confusion) — but Korean "
        "speakers read durations with NATIVE numerals ('일곱 시간'), never "
        "'칠시간', so '칠시간/7시간' outside a duration context almost certainly "
        "means '실시간'. Likewise '뭘 했어도' gets misheard as '어렸어도'. Apply "
        "the same reasoning to other sino-vs-native numeral readings and common "
        "consonant/onset confusions. In general: whenever the utterance sounds "
        "unnatural, ungrammatical, or out of place for a fluent speaker in this "
        "conversation, ACTIVELY search for a phonetically similar sentence a "
        "fluent speaker WOULD say there, and prefer that reconstruction.\n"
        "When the context/glossary contains the correct term or name, CORRECT "
        "the sentence with it. Otherwise reconstruct the most plausible intended "
        "meaning; only if genuinely irrecoverable, suggest politely asking the "
        "speaker to repeat that specific word. If mis-recognition is likely, set "
        "clarify_suspected=true, put the corrected/intended meaning in "
        "clarify_did_you_mean as a confirmation, put the FULL corrected "
        "translation (the whole sentence as it should read, in the same language "
        "as LATEST TRANSLATION) in clarify_corrected_translation, and put the "
        "corrected ORIGINAL sentence (in the same language as LATEST ORIGINAL) "
        "in clarify_corrected_source — both REPLACE the displayed captions, so "
        "each must be a complete, natural sentence, not a note. Else false and "
        "empty.\n\n"
        "PART C — Answer suggestion: decide if the LATEST utterance is a QUESTION "
        "or request that the listener should answer. If yes, set should_answer="
        f"true and propose a concise, natural reply in {tgt_name} "
        "(answer_native) plus the same reply written in "
        f"{lang_name} so the user understands it (answer_local). If it is not a "
        "question to answer, set should_answer=false and leave those empty.\n\n"
        "PART D — Expression upgrade: rewrite the LATEST TRANSLATION into a more "
        f"natural, native-level {tgt_name} version (upgrade). If it is already "
        "natural, return an empty string.\n\n"
        "PART E — Term glossary: pick up to 3 genuinely SPECIALIZED terms in the "
        "latest utterance (technical, legal, trade/finance, industry jargon, or "
        "acronyms) that a non-expert might not know — NOT ordinary words. For "
        f"each, give term and a one-line plain-language meaning in {lang_name}. "
        "Empty list if there are none.\n\n"
        f"Write subtitle_alert, clarify_*, answer_local and term meanings in "
        f"{lang_name}. Write answer_native and upgrade in {tgt_name}."
        f"{context_line}{history_line}\n"
        f"LATEST ORIGINAL (source language): {original}\n"
        f"LATEST TRANSLATION: {translation}\n"
    )


def build_feedback_prompt(transcript: str, language: str) -> str:
    """Post-meeting feedback/coaching report (in the user's language)."""
    lang_name = SUPPORTED_LANGUAGES.get(language, "English")
    return (
        "You are a language coach. From the meeting transcript below, produce a "
        f"concise feedback report in {lang_name} using Markdown headings:\n"
        "1. 핵심 표현 / Key expressions — useful phrases that appeared, polished.\n"
        "2. 더 자연스럽게 / More natural — for clumsy lines, a native alternative.\n"
        "3. 학습 포인트 / Learning points — 2-4 short tips.\n"
        "Be specific and tied to the transcript; do not invent facts.\n\n"
        "=== TRANSCRIPT ===\n" + transcript
    )


def build_multitranslate_prompt(text: str, target_codes: list[str]) -> str:
    """Translate one utterance into several languages at once (JSON output).

    Returns an object keyed by the language code, e.g. {"ko": "...", "ja": "..."}.
    """
    names = {c: SUPPORTED_LANGUAGES.get(c, c) for c in target_codes}
    listing = ", ".join(f'"{c}" ({n})' for c, n in names.items())
    return (
        "You are a translator. Translate the TEXT into each of these languages and "
        "return ONLY a JSON object keyed by the language code, with the translation "
        "as the value. Keys to include: "
        f"{listing}.\n"
        "If the text is already in one of those languages, still provide that "
        "language's natural form. No extra commentary.\n\n"
        f"TEXT:\n{text}"
    )


def build_qa_prompt(transcript: str, question: str, language: str) -> str:
    """Build a grounded Q&A prompt over the saved conversation transcript."""
    lang_name = SUPPORTED_LANGUAGES.get(language, "English")
    return (
        "You are a meeting assistant. Answer the user's QUESTION using ONLY the "
        "TRANSCRIPT below. If the answer is not present, say you cannot find it "
        f"in the conversation. Be concise and specific. Answer in {lang_name}.\n\n"
        f"=== TRANSCRIPT ===\n{transcript}\n\n"
        f"=== QUESTION ===\n{question}\n"
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


# Map our language codes to the BCP-47 target codes the translate model wants.
# Anything not listed is passed through unchanged (our codes are ISO 639-1).
TRANSLATE_TARGET_CODES: dict[str, str] = {
    "zh": "cmn-CN",  # Mandarin Chinese
}


def is_translate_model(model: str) -> bool:
    """True for the dedicated speech-to-speech translate model family."""
    return "translate" in (model or "").lower()


def _target_code(lang: str) -> str:
    return TRANSLATE_TARGET_CODES.get(lang, lang)


def build_config(
    voice_name: str | None = None,
    lang_a: str = DEFAULT_LANG_A,
    lang_b: str = DEFAULT_LANG_B,
    system_instruction: str | None = None,
    model: str | None = None,
) -> types.LiveConnectConfig:
    """Build the Live session configuration for the active model.

    Two shapes depending on the model:

    * Dedicated translate model (``gemini-3.5-live-translate-preview``): uses
      ``translation_config`` with a target language. The source language is
      auto-detected (70+ languages) and the model preserves the speaker's own
      voice, so no persona prompt or prebuilt voice is sent. ``lang_b`` is the
      language you want to hear.
    * General live model (e.g. ``gemini-2.0-flash-exp``): persona-based two-way
      interpreting between ``lang_a`` and ``lang_b`` with a selectable voice.
    """
    model = model or DEFAULT_MODEL

    if is_translate_model(model):
        # NOTE: AudioTranscriptionConfig.language_codes (a source-language hint)
        # is only supported in Vertex / Enterprise Agent Platform mode, NOT the
        # Gemini Developer API (API-key) mode we run on — passing it crashes the
        # session. So we leave the input transcription on auto-detect.
        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            translation_config=types.TranslationConfig(
                target_language_code=_target_code(lang_b),
                # echo_target_language=True re-speaks anything already in the
                # target language. Combined with the phone speaker feeding the
                # mic, that created an infinite repeat loop (translation plays →
                # mic hears it → it's target-language → model re-speaks it → …).
                # Off: target-language speech simply passes without re-speaking.
                echo_target_language=False,
            ),
        )

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
