"""Pure document-processing functions (no I/O, easy to test)."""

from __future__ import annotations

import csv
import io
import re
from collections import Counter
from collections.abc import Iterator
from typing import Any

MAX_DOCUMENT_BYTES = 1_048_576
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]+")
_STOP = frozenset(
    "a an and are as at be but by for from has have in is it its of on or that the this to "
    "was were will with not they their we you your our can".split()
)


class DocumentError(ValueError):
    """The document cannot be processed."""


def decode_text(raw: bytes) -> str:
    if len(raw) > MAX_DOCUMENT_BYTES:
        raise DocumentError(f"document exceeds {MAX_DOCUMENT_BYTES // 1024} KiB limit")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentError("document is not valid UTF-8 text") from exc


def extractive_summary(text: str, max_sentences: int = 3) -> str:
    """Pick the highest-scoring sentences (by word frequency), preserving original order."""
    sentences = [s.strip() for s in _SENTENCE.split(text.strip()) if s.strip()]
    if len(sentences) <= max_sentences:
        return " ".join(sentences)
    freq = Counter(w.lower() for w in _WORD.findall(text) if w.lower() not in _STOP)
    scored = [
        (sum(freq[w.lower()] for w in _WORD.findall(s)) / (len(_WORD.findall(s)) or 1), i)
        for i, s in enumerate(sentences)
    ]
    best = sorted(sorted(scored, reverse=True)[:max_sentences], key=lambda pair: pair[1])
    return " ".join(sentences[i] for _, i in best)


def text_profile(text: str) -> dict[str, Any]:
    words = _WORD.findall(text)
    return {
        "characters": len(text),
        "words": len(words),
        "sentences": len([s for s in _SENTENCE.split(text.strip()) if s.strip()]),
        "top_terms": [
            w for w, _ in Counter(x.lower() for x in words if x.lower() not in _STOP).most_common(5)
        ],
    }


def _infer(values: list[str]) -> str:
    def all_match(cast: type) -> bool:
        try:
            for v in values:
                cast(v)
        except ValueError:
            return False
        return True

    if not values:
        return "empty"
    if all_match(int):
        return "integer"
    if all_match(float):
        return "float"
    return "string"


def profile_csv(text: str) -> Iterator[dict[str, Any]]:
    """Yield one profile dict per column, so callers can stream progress."""
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise DocumentError("CSV has no header row")
    rows = list(reader)
    for name in reader.fieldnames:
        raw = [(row.get(name) or "").strip() for row in rows]
        present = [v for v in raw if v]
        kind = _infer(present)
        profile: dict[str, Any] = {
            "column": name,
            "type": kind,
            "rows": len(raw),
            "nulls": len(raw) - len(present),
            "distinct": len(set(present)),
        }
        if kind in {"integer", "float"}:
            numbers = [float(v) for v in present]
            profile.update(
                min=min(numbers), max=max(numbers), mean=round(sum(numbers) / len(numbers), 6)
            )
        yield profile
