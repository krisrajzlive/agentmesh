"""Deterministic text-analysis tools exposed to the ADK agent."""

from __future__ import annotations

import re
from collections import Counter

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]+")
_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have in is it its of on or that the this to "
    "was were will with not they their we you your our can all any more most other some such "
    "than then there these those which who would about into over also".split()
)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\s().-]?){9,14}\d(?!\w)")


def text_statistics(text: str) -> dict[str, float | int]:
    """Count words, sentences and characters and estimate reading time of ``text``."""
    words = _WORD.findall(text)
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    return {
        "characters": len(text),
        "words": len(words),
        "sentences": len(sentences),
        "avg_words_per_sentence": round(len(words) / len(sentences), 2) if sentences else 0.0,
        "reading_time_minutes": round(len(words) / 220, 2),
    }


def extract_keywords(text: str, top_n: int = 5) -> list[dict[str, str | int]]:
    """Return the ``top_n`` most frequent non-trivial words in ``text``."""
    counts = Counter(w.lower() for w in _WORD.findall(text) if w.lower() not in _STOPWORDS)
    return [{"keyword": w, "count": c} for w, c in counts.most_common(max(1, min(top_n, 25)))]


def redact_pii(text: str) -> dict[str, str | int]:
    """Mask e-mail addresses and phone numbers in ``text``."""
    redacted, emails = _EMAIL.subn("[EMAIL]", text)
    redacted, phones = _PHONE.subn("[PHONE]", redacted)
    return {"redacted_text": redacted, "emails_masked": emails, "phones_masked": phones}
