"""Currency conversion agent: multi-turn clarification, structured artifacts."""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Decimal

from a2a.helpers import get_message_text, new_data_part, new_text_part
from a2a.server.agent_execution import RequestContext
from a2a.server.tasks import TaskUpdater
from a2a.types import AgentSkill, Role

from agentmesh.agents.fx.parser import ConversionRequest, IntentParser
from agentmesh.agents.fx.rates import RateProvider, RateUnavailableError
from agentmesh.runtime import AgentSpec, MeshAgentExecutor, TaskInputError

FX_SPEC = AgentSpec(
    slug="fx",
    name="FX Conversion Agent",
    description=(
        "Converts amounts between major currencies using ECB reference rates. "
        "Asks follow-up questions when the request is incomplete."
    ),
    skills=[
        AgentSkill(
            id="currency-conversion",
            name="Currency conversion",
            description="Convert an amount between currencies at the latest reference rate.",
            tags=["finance", "currency", "fx"],
            examples=["Convert 250 USD to EUR", "How much is 1,000 INR in GBP?"],
        )
    ],
)

_QUESTIONS = {
    "amount": "What amount would you like to convert?",
    "source": "Which currency are you converting from?",
    "target": "Which currency are you converting to?",
}


def _user_texts(context: RequestContext) -> list[str]:
    """All user utterances for this task: history plus the current message."""
    texts: list[str] = []
    if context.current_task is not None:
        texts.extend(
            get_message_text(m) for m in context.current_task.history if m.role == Role.ROLE_USER
        )
    current = context.message
    if current is not None:
        text = get_message_text(current)
        if not texts or texts[-1] != text:
            texts.append(text)
    return [t for t in texts if t.strip()]


class FxExecutor(MeshAgentExecutor):
    agent_name = "fx"

    def __init__(self, parser: IntentParser, rates: RateProvider) -> None:
        super().__init__()
        self._parser = parser
        self._rates = rates

    async def run(self, context: RequestContext, updater: TaskUpdater) -> None:
        texts = _user_texts(context)
        if not texts:
            raise TaskInputError("Send a message such as 'Convert 100 USD to EUR'.")
        await updater.start_work()

        request: ConversionRequest = await self._parser.parse(texts)
        if not request.complete:
            question = _QUESTIONS[request.missing[0]]
            await updater.requires_input(self.text(updater, question))
            return
        assert request.amount is not None
        assert request.source
        assert request.target

        try:
            rate, as_of = await self._rates.rate(request.source, request.target)
        except RateUnavailableError:
            await updater.failed(
                self.text(updater, f"No rate available for {request.source} to {request.target}.")
            )
            return

        converted = (request.amount * rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)
        summary = (
            f"{request.amount:,} {request.source} = {converted:,} {request.target} "
            f"(rate {rate.normalize():f}, as of {as_of})"
        )
        await updater.add_artifact(
            [
                new_text_part(summary),
                new_data_part(
                    {
                        "amount": str(request.amount),
                        "source": request.source,
                        "target": request.target,
                        "rate": str(rate),
                        "converted": str(converted),
                        "as_of": as_of,
                    }
                ),
            ],
            name="conversion",
        )
        await updater.complete(self.text(updater, summary))
