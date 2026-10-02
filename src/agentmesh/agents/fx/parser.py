"""Turn free-form conversation text into a (possibly partial) conversion request."""

from __future__ import annotations

import contextlib
import json
import re
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from typing import Protocol

from agentmesh.llm import ChatMessage, LLMError, LLMProvider
from agentmesh.observability.logging import get_logger

log = get_logger(__name__)

SUPPORTED = frozenset(
    [
        "AUD",
        "BGN",
        "BRL",
        "CAD",
        "CHF",
        "CNY",
        "CZK",
        "DKK",
        "EUR",
        "GBP",
        "HKD",
        "HUF",
        "IDR",
        "ILS",
        "INR",
        "ISK",
        "JPY",
        "KRW",
        "MXN",
        "MYR",
        "NOK",
        "NZD",
        "PHP",
        "PLN",
        "RON",
        "SEK",
        "SGD",
        "THB",
        "TRY",
        "USD",
        "ZAR",
    ]
)
NAMES = {
    "dollar": "USD", "dollars": "USD", "buck": "USD", "bucks": "USD",
    "euro": "EUR", "euros": "EUR",
    "pound": "GBP", "pounds": "GBP", "sterling": "GBP",
    "yen": "JPY", "rupee": "INR", "rupees": "INR",
    "yuan": "CNY", "franc": "CHF", "francs": "CHF",
}  # fmt: skip
_TOKEN = re.compile(r"[A-Za-z]+|\d[\d,]*(?:\.\d+)?|->|→")
_AMOUNT = re.compile(r"(?<![A-Za-z])(\d[\d,]*(?:\.\d+)?)(?![A-Za-z])")


@dataclass(frozen=True, slots=True)
class ConversionRequest:
    amount: Decimal | None = None
    source: str | None = None
    target: str | None = None

    @property
    def missing(self) -> list[str]:
        return [
            name
            for name, value in (
                ("amount", self.amount),
                ("source", self.source),
                ("target", self.target),
            )
            if value is None
        ]

    @property
    def complete(self) -> bool:
        return not self.missing


class IntentParser(Protocol):
    async def parse(self, user_texts: list[str]) -> ConversionRequest: ...


# Codes that double as English words must be written in capitals to count as currencies.
_CASE_SENSITIVE = frozenset({"TRY", "RON", "ISK"})


def _currency(word: str) -> str | None:
    upper = word.upper()
    if upper in SUPPORTED and len(word) == 3:
        if upper in _CASE_SENSITIVE and not word.isupper():
            return None
        return upper
    return NAMES.get(word.lower())


def parse_turn(text: str, state: ConversionRequest) -> ConversionRequest:
    """Merge one user utterance into ``state``."""
    tokens = _TOKEN.findall(text)
    amount = state.amount
    if match := _AMOUNT.search(text):
        with contextlib.suppress(InvalidOperation):
            amount = Decimal(match.group(1).replace(",", ""))

    source, target = state.source, state.target
    unresolved: list[str] = []
    for index, token in enumerate(tokens):
        code = _currency(token)
        if code is None:
            continue
        previous = tokens[index - 1].lower() if index else ""
        if previous == "from":
            source = code
        elif previous in {"to", "into", "in", "->", "→", "for"}:
            target = code
        else:
            unresolved.append(code)
    for code in unresolved:
        if source is None:
            source = code
        elif target is None:
            target = code
    return replace(state, amount=amount, source=source, target=target)


class RuleBasedParser:
    async def parse(self, user_texts: list[str]) -> ConversionRequest:
        state = ConversionRequest()
        for text in user_texts:
            state = parse_turn(text, state)
        return state


_SYSTEM = (
    "Extract a currency conversion request from the conversation. Reply with JSON only: "
    '{"amount": number|null, "source": ISO-4217 code|null, "target": ISO-4217 code|null}. '
    "Use null for anything the user has not stated. Never guess."
)


class LLMParser:
    """LLM extraction validated against the supported-currency set, with rule-based fallback."""

    def __init__(self, llm: LLMProvider, fallback: IntentParser | None = None) -> None:
        self._llm = llm
        self._fallback = fallback or RuleBasedParser()

    async def parse(self, user_texts: list[str]) -> ConversionRequest:
        transcript = "\n".join(f"user: {t}" for t in user_texts)
        try:
            reply = await self._llm.complete(
                [ChatMessage("system", _SYSTEM), ChatMessage("user", transcript)],
                json_mode=True,
                max_tokens=120,
            )
            data = json.loads(reply.content)
            amount = Decimal(str(data["amount"])) if data.get("amount") is not None else None
            source = str(data["source"]).upper() if data.get("source") else None
            target = str(data["target"]).upper() if data.get("target") else None
            if (source and source not in SUPPORTED) or (target and target not in SUPPORTED):
                raise ValueError("unsupported currency in model output")
            if amount is not None and amount <= 0:
                raise ValueError("non-positive amount")
            return ConversionRequest(amount, source, target)
        except (LLMError, ValueError, KeyError, InvalidOperation, TypeError) as exc:
            log.warning("fx_llm_parse_fallback", error=str(exc))
            return await self._fallback.parse(user_texts)
