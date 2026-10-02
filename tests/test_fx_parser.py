from __future__ import annotations

from decimal import Decimal

import pytest

from agentmesh.agents.fx.parser import (
    ConversionRequest,
    LLMParser,
    RuleBasedParser,
    parse_turn,
)
from agentmesh.llm import LLMResponse, LLMUnavailableError


@pytest.mark.parametrize(
    ("text", "amount", "source", "target"),
    [
        ("Convert 100 USD to EUR", "100", "USD", "EUR"),
        ("how much is 1,000.50 GBP in INR?", "1000.50", "GBP", "INR"),
        ("convert from EUR to JPY 25 yen", "25", "EUR", "JPY"),
        ("50 dollars to euros", "50", "USD", "EUR"),
        ("USD -> CAD 7", "7", "USD", "CAD"),
    ],
)
def test_parse_complete_requests(text, amount, source, target):
    parsed = parse_turn(text, ConversionRequest())
    assert (parsed.amount, parsed.source, parsed.target) == (Decimal(amount), source, target)
    assert parsed.complete


def test_partial_request_reports_missing_fields():
    parsed = parse_turn("convert 100 to EUR", ConversionRequest())
    assert parsed.missing == ["source"]
    assert parse_turn("hello there", ConversionRequest()).missing == ["amount", "source", "target"]


def test_ambiguous_english_words_are_not_currencies():
    # "try" and "ron" are valid ISO codes but also ordinary words.
    assert parse_turn("please try 10 USD to EUR", ConversionRequest()).target == "EUR"


async def test_bare_reply_fills_first_missing_slot():
    parsed = await RuleBasedParser().parse(["convert 250 to EUR", "USD"])
    assert (parsed.amount, parsed.source, parsed.target) == (Decimal(250), "USD", "EUR")


class FakeLLM:
    name = "fake"
    model = "fake"

    def __init__(self, reply: str | Exception) -> None:
        self.reply = reply

    async def complete(self, messages, **_):
        if isinstance(self.reply, Exception):
            raise self.reply
        return LLMResponse(self.reply, "fake", "fake")


async def test_llm_parser_uses_model_output():
    llm = FakeLLM('{"amount": 12.5, "source": "gbp", "target": "EUR"}')
    parsed = await LLMParser(llm).parse(["ignored"])
    assert parsed == ConversionRequest(Decimal("12.5"), "GBP", "EUR")


@pytest.mark.parametrize(
    "reply",
    [
        "not json",
        '{"amount": 1, "source": "XXX", "target": "EUR"}',  # unsupported currency
        '{"amount": -5, "source": "USD", "target": "EUR"}',
        LLMUnavailableError("down"),
    ],
)
async def test_llm_parser_falls_back_to_rules_on_bad_output(reply):
    parsed = await LLMParser(FakeLLM(reply)).parse(["Convert 100 USD to EUR"])
    assert parsed == ConversionRequest(Decimal(100), "USD", "EUR")
