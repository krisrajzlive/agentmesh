"""Document agent: text/file/data parts in, structured artifacts out."""

from __future__ import annotations

from a2a.helpers import get_data_parts, get_text_parts, new_data_part, new_text_part
from a2a.server.agent_execution import RequestContext
from a2a.server.tasks import TaskUpdater
from a2a.types import AgentSkill, Part

from agentmesh.agents.documents.processing import (
    DocumentError,
    decode_text,
    extractive_summary,
    profile_csv,
    text_profile,
)
from agentmesh.llm import ChatMessage, LLMError, LLMProvider
from agentmesh.observability.logging import get_logger
from agentmesh.runtime import AgentSpec, MeshAgentExecutor, TaskInputError

log = get_logger(__name__)

DOCUMENTS_SPEC = AgentSpec(
    slug="documents",
    name="Document Processing Agent",
    description=(
        "Summarises documents and profiles CSV files. Accepts plain text, uploaded files "
        "(text, markdown, JSON, CSV) and structured data parts."
    ),
    input_modes=["text/plain", "text/markdown", "text/csv", "application/json"],
    output_modes=["text/plain", "application/json"],
    skills=[
        AgentSkill(
            id="document-summary",
            name="Document summary",
            description="Summarise a text document and report basic statistics.",
            tags=["documents", "summarization"],
            examples=["Summarise the attached report"],
            input_modes=["text/plain", "text/markdown"],
        ),
        AgentSkill(
            id="csv-profile",
            name="CSV profiling",
            description="Per-column type, null count, distinct values and numeric ranges.",
            tags=["documents", "data-quality"],
            examples=["Profile this CSV"],
            input_modes=["text/csv"],
        ),
    ],
)


def _is_csv(part: Part) -> bool:
    filename: str = part.filename
    return bool(part.media_type == "text/csv" or filename.lower().endswith(".csv"))


class DocumentsExecutor(MeshAgentExecutor):
    agent_name = "documents"

    def __init__(self, llm: LLMProvider | None = None) -> None:
        super().__init__()
        self._llm = llm

    async def run(self, context: RequestContext, updater: TaskUpdater) -> None:
        message = context.message
        assert message is not None
        files = [p for p in message.parts if p.HasField("raw")]
        inline = [t for t in get_text_parts(message.parts) if t.strip()]
        data = get_data_parts(message.parts)

        document: str | None = None
        csv_mode = False
        try:
            if files:
                document = decode_text(files[0].raw)
                csv_mode = _is_csv(files[0])
            elif data and isinstance(data[0], dict) and isinstance(data[0].get("text"), str):
                document = data[0]["text"]
            elif inline and len(" ".join(inline)) > 200:
                document = " ".join(inline)
        except DocumentError as exc:
            raise TaskInputError(str(exc)) from exc

        if document is None or not document.strip():
            await updater.requires_input(
                self.text(updater, "Attach a document (text or CSV) or paste the text to analyse.")
            )
            return

        await updater.start_work(self.text(updater, "Processing document"))
        if csv_mode:
            await self._profile_csv(document, updater)
        else:
            await self._summarise(document, updater)

    async def _profile_csv(self, document: str, updater: TaskUpdater) -> None:
        try:
            columns = []
            for index, column in enumerate(profile_csv(document), start=1):
                columns.append(column)
                await updater.start_work(
                    self.text(updater, f"Profiled column '{column['column']}' ({index})")
                )
        except DocumentError as exc:
            raise TaskInputError(str(exc)) from exc
        summary = f"Profiled {len(columns)} column(s)."
        await updater.add_artifact(
            [new_text_part(summary), new_data_part({"columns": columns})], name="csv-profile"
        )
        await updater.complete(self.text(updater, summary))

    async def _summarise(self, document: str, updater: TaskUpdater) -> None:
        summary = await self._llm_summary(document) or extractive_summary(document)
        await updater.add_artifact(
            [new_text_part(summary), new_data_part(text_profile(document))], name="summary"
        )
        await updater.complete(self.text(updater, summary))

    async def _llm_summary(self, document: str) -> str | None:
        if self._llm is None:
            return None
        try:
            reply = await self._llm.complete(
                [
                    ChatMessage("system", "Summarise the document in at most three sentences."),
                    ChatMessage("user", document[:12_000]),
                ],
                max_tokens=200,
            )
        except LLMError as exc:
            log.warning("documents_llm_fallback", error=str(exc))
            return None
        return reply.content.strip() or None
