"""Self-hosted recitation correction engine for Qurani Jannati.

The engine mirrors the data model used by Tarteel's AI correction SDK
(`ai.tarteel.shared.data.Mistake`) so that the app can store and display
corrections identically, but everything here runs on YOUR server:

  1. The audio is transcribed with an open-source model (faster-whisper)
     if it is installed (see requirements-whisper.txt).
  2. The recognized transcript is aligned to the expected ayah text with
     a word-level edit-distance alignment (this file).
  3. Mistakes are emitted with the same types as the Tarteel SDK:
     INCORRECT_TASHKEEL, INCORRECT_WORDS, MISSED_WORDS, EXTRA_WORDS.

If faster-whisper is not installed the endpoint still works and returns
engine="unavailable" so the app can record sessions without crashing.
"""

from __future__ import annotations

import os
import re
import time
import unicodedata
import uuid
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Mistake types (match ai/tarteel/shared/wird/MistakeType)
# ---------------------------------------------------------------------------

MISTAKE_INCORRECT_TASHKEEL = "INCORRECT_TASHKEEL"
MISTAKE_INCORRECT_WORDS = "INCORRECT_WORDS"
MISTAKE_MISSED_WORDS = "MISSED_WORDS"
MISTAKE_EXTRA_WORDS = "EXTRA_WORDS"
MISTAKE_PEEKED_WORDS = "PEEKED_WORDS"

# Arabic diacritics / marks that do not change the letters themselves.
_TASHKEEL_RE = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")


def strip_tashkeel(word: str) -> str:
    """Remove vowel marks so 'كِتَاب' and 'كِتٰب' compare equal on letters."""
    return _TASHKEEL_RE.sub("", word)


def normalize_word(word: str) -> str:
    """Normalize common Arabic letter variants for fair comparison."""
    text = unicodedata.normalize("NFKC", word)
    text = (
        text.replace("\u0623", "\u0627")  # أ -> ا
        .replace("\u0625", "\u0627")  # إ -> ا
        .replace("\u0622", "\u0627")  # آ -> ا
        .replace("\u0624", "\u0648")  # ؤ -> و
        .replace("\u0626", "\u064A")  # ئ -> ي
        .replace("\u0649", "\u064A")  # ى -> ي
        .replace("\u06CC", "\u064A")  # ی (farsi ye) -> ي
    )
    return strip_tashkeel(text)


def tokenize(text: str) -> List[str]:
    """Split a transcript into words, keeping tashkeel on each word."""
    return [w for w in re.split(r"[\s\u06D6-\u06ED]+", text.strip()) if w]


# ---------------------------------------------------------------------------
# Word alignment (Needleman-Wunsch with tuned costs)
# ---------------------------------------------------------------------------


def _words_match(a: str, b: str) -> bool:
    return normalize_word(a) == normalize_word(b)


def align_words(
    expected: List[str], received: List[str]
) -> List[Tuple[Optional[int], Optional[int]]]:
    """Align expected words to received words.

    Returns a list of (expected_index, received_index) pairs. A pair with
    expected_index None is an extra spoken word (insertion). A pair with
    received_index None is a missed expected word (deletion).
    """
    n, m = len(expected), len(received)
    inf = float("inf")
    dp = [[inf] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = 0 if _words_match(expected[i - 1], received[j - 1]) else 1
            dp[i][j] = min(
                dp[i - 1][j - 1] + sub,
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
            )

    pairs: List[Tuple[Optional[int], Optional[int]]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sub = 0 if _words_match(expected[i - 1], received[j - 1]) else 1
            if dp[i][j] == dp[i - 1][j - 1] + sub:
                pairs.append((i - 1, j - 1))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            pairs.append((i - 1, None))
            i -= 1
            continue
        pairs.append((None, j - 1))
        j -= 1
    pairs.reverse()
    return pairs


# ---------------------------------------------------------------------------
# Mistake classification + timing
# ---------------------------------------------------------------------------

_WordTiming = Tuple[Optional[float], Optional[float]]  # (start_ms, end_ms)


def _infer_timings(
    expected: List[str],
    received: List[str],
    received_timings: Optional[List[_WordTiming]],
    audio_duration_ms: Optional[float],
) -> Tuple[List[_WordTiming], List[_WordTiming]]:
    """Return (expected_word_timing, received_word_timing) in milliseconds.

    Received words use real timestamps when available, otherwise the audio
    duration is distributed evenly across words. Expected words inherit the
    timing of the received word they align to; unaligned words interpolate.
    """
    total_received = len(received)
    if received_timings and len(received_timings) == total_received:
        rec_ms = list(received_timings)
    elif total_received and audio_duration_ms:
        step = audio_duration_ms / total_received
        rec_ms = [(i * step, (i + 1) * step) for i in range(total_received)]
    else:
        rec_ms = [(None, None)] * total_received

    pairs = align_words(expected, received)
    exp_ms: List[_WordTiming] = []
    for ei, ri in pairs:
        if ei is not None:
            exp_ms.append(rec_ms[ri] if ri is not None else (None, None))
    # Fill holes in expected timings by linear interpolation.
    known = [(i, t) for i, t in enumerate(exp_ms) if t[0] is not None]
    if known and len(known) < len(exp_ms):
        for i in range(len(exp_ms)):
            if exp_ms[i][0] is not None:
                continue
            left = next((t for k, t in reversed(known) if k < i), None)
            right = next((t for k, t in known if k > i), None)
            if left and right:
                span = right[0] - left[1]
                mid = (left[1] + right[0]) / 2.0
                exp_ms[i] = (max(left[0], mid - span / 4), mid + span / 4)
            elif left:
                exp_ms[i] = left
            elif right:
                exp_ms[i] = right
            else:
                exp_ms[i] = (0.0, 0.0)
    return exp_ms, rec_ms


def compute_mistakes(
    expected_text: str,
    received_text: str,
    *,
    session_id: str = "",
    ayah: int = 0,
    surah: int = 0,
    received_timings: Optional[List[_WordTiming]] = None,
    audio_duration_ms: Optional[float] = None,
    created_at_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """Align the recognized transcript to the expected ayah text and emit
    mistakes using the Tarteel-compatible schema."""
    expected = tokenize(expected_text)
    received = tokenize(received_text)
    created_at = created_at_ms or int(time.time() * 1000)

    exp_ms, rec_ms = _infer_timings(
        expected, received, received_timings, audio_duration_ms
    )

    mistakes: List[Dict[str, Any]] = []
    correct = 0
    total = max(len(expected), 1)

    for ei, ri in align_words(expected, received):
        if ei is None:
            # Extra spoken word.
            start, end = rec_ms[ri]
            mistakes.append(
                _mistake(
                    session_id, MISTAKE_EXTRA_WORDS,
                    "", received[ri], [{"index": -1}],
                    ayah, surah, start, end, created_at,
                )
            )
            continue
        start, end = exp_ms[ei]
        if ri is None:
            # Expected word that was never spoken.
            mistakes.append(
                _mistake(
                    session_id, MISTAKE_MISSED_WORDS,
                    expected[ei], "", [{"index": ei}],
                    ayah, surah, start, end, created_at,
                )
            )
            continue
        if expected[ei] == received[ri]:
            correct += 1
            continue
        if strip_tashkeel(expected[ei]) == strip_tashkeel(received[ri]):
            mistakes.append(
                _mistake(
                    session_id, MISTAKE_INCORRECT_TASHKEEL,
                    expected[ei], received[ri], [{"index": ei}],
                    ayah, surah, start, end, created_at,
                )
            )
        else:
            mistakes.append(
                _mistake(
                    session_id, MISTAKE_INCORRECT_WORDS,
                    expected[ei], received[ri], [{"index": ei}],
                    ayah, surah, start, end, created_at,
                )
            )

    by_type: Dict[str, int] = {}
    for m in mistakes:
        by_type[m["mistakeType"]] = by_type.get(m["mistakeType"], 0) + 1

    return {
        "sessionId": session_id,
        "ayah": ayah,
        "surah": surah,
        "expectedTranscript": expected_text,
        "receivedTranscript": received_text,
        "wordCount": len(expected),
        "correctWords": correct,
        "accuracy": round(correct / total, 4),
        "mistakeCount": len(mistakes),
        "mistakesByType": by_type,
        "mistakes": mistakes,
        "createdAt": created_at,
    }


def _mistake(
    session_id: str,
    mistake_type: str,
    expected: str,
    received: str,
    positions: List[Dict[str, Any]],
    ayah: int,
    surah: int,
    start_ms: Optional[float],
    end_ms: Optional[float],
    created_at: int,
) -> Dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "sessionId": session_id,
        "mistakeType": mistake_type,
        "expectedTranscript": expected,
        "receivedTranscript": received,
        "positions": positions,
        "ayah": ayah,
        "surah": surah,
        "stateIndexStart": 0,
        "stateIndexEnd": 0,
        "startTimeMs": int(start_ms) if start_ms is not None else 0,
        "endTimeMs": int(end_ms) if end_ms is not None else 0,
        "createdAt": created_at,
        "updatedAt": created_at,
    }


# ---------------------------------------------------------------------------
# Whisper transcription (lazy, open-source)
# ---------------------------------------------------------------------------

_MODEL: Any = None
_MODEL_SIZE: Optional[str] = None


def _ensure_model():
    global _MODEL, _MODEL_SIZE
    if _MODEL is not None:
        return _MODEL
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError:
        return None
    size = os.environ.get("QURANI_JANNATI_WHISPER_MODEL", "small")
    device = os.environ.get("QURANI_JANNATI_WHISPER_DEVICE", "auto")
    compute = os.environ.get("QURANI_JANNATI_WHISPER_COMPUTE", "auto")
    _MODEL = WhisperModel(size, device=device, compute_type=compute)
    _MODEL_SIZE = size
    return _MODEL


def engine_status() -> Dict[str, Any]:
    model = _ensure_model()
    return {
        "available": model is not None,
        "engine": "whisper" if model is not None else "unavailable",
        "model": _MODEL_SIZE or None,
    }


def transcribe(
    audio_path: str, language: str
) -> Tuple[str, List[Tuple[str, float, float]]]:
    """Transcribe audio with faster-whisper.

    Returns (full_text, [(word, start_ms, end_ms), ...]).
    Raises RuntimeError when the engine is not installed.
    """
    model = _ensure_model()
    if model is None:
        raise RuntimeError(
            "faster-whisper is not installed. "
            "Run: pip install -r requirements-whisper.txt"
        )
    segments, _info = model.transcribe(
        audio_path, language=language or None, word_timestamps=True
    )
    words: List[Tuple[str, float, float]] = []
    for segment in segments:
        for word in segment.words:
            text = word.word.strip()
            if text:
                words.append((text, word.start * 1000.0, word.end * 1000.0))
    return " ".join(w[0] for w in words), words


def correct_audio(
    audio_path: str,
    expected_text: str,
    *,
    session_id: str,
    ayah: int,
    surah: int,
    language: str = "ar",
    audio_duration_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """End-to-end correction for one recorded ayah."""
    status = engine_status()
    if not status["available"]:
        return {
            "engine": "unavailable",
            "notice": "Speech engine not configured. Install faster-whisper "
            "(pip install -r requirements-whisper.txt) to enable AI correction.",
            "sessionId": session_id,
            "ayah": ayah,
            "surah": surah,
            "expectedTranscript": expected_text,
            "receivedTranscript": "",
            "wordCount": len(tokenize(expected_text)),
            "correctWords": 0,
            "accuracy": 0.0,
            "mistakeCount": 0,
            "mistakesByType": {},
            "mistakes": [],
            "createdAt": int(time.time() * 1000),
        }

    try:
        received_text, word_timings = transcribe(audio_path, language)
    except Exception as exc:  # transcription errors must not kill the request
        return {
            "engine": "whisper",
            "error": str(exc),
            "sessionId": session_id,
            "ayah": ayah,
            "surah": surah,
            "expectedTranscript": expected_text,
            "receivedTranscript": "",
            "wordCount": len(tokenize(expected_text)),
            "correctWords": 0,
            "accuracy": 0.0,
            "mistakeCount": 0,
            "mistakesByType": {},
            "mistakes": [],
            "createdAt": int(time.time() * 1000),
        }

    timings = [(s, e) for (_w, s, e) in word_timings]
    result = compute_mistakes(
        expected_text,
        received_text,
        session_id=session_id,
        ayah=ayah,
        surah=surah,
        received_timings=timings,
        audio_duration_ms=audio_duration_ms,
    )
    result["engine"] = "whisper"
    return result
