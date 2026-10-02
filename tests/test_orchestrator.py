from __future__ import annotations

import json

import pytest

from agentmesh.client import AgentClient
from agentmesh.llm import LLMResponse, LLMUnavailableError
from agentmesh.orchestrator import LLMPlanner, PlanningError, RuleBasedPlanner
from agentmesh.orchestrator.planner import split_clauses
from agentmesh.registry.models import AgentRecord, SkillInfo
from tests.support import build_mesh


def agent(agent_id: str, name: str, tags: list[str], description: str = "") -> AgentRecord:
    return AgentRecord(
        id=agent_id,
        name=name,
        url=f"http://{agent_id}.test",
        description=description,
        skills=[SkillInfo(id=f"{agent_id}-skill", name=name, description=description, tags=tags)],
    )


CATALOG = [
    agent(
        "fx", "Currency conversion", ["finance", "currency"], "Convert amounts between currencies"
    ),
    agent("text", "Keyword extraction", ["text", "nlp"], "Find the most frequent terms"),
]


# ---------------------------------------------------------------------- planning


def test_split_clauses_distinguishes_parallel_and_sequential():
    assert split_clauses("a; b") == [("a", False), ("b", False)]
    assert split_clauses("a then b") == [("a", False), ("b", True)]
    assert split_clauses("a and then b\nc") == [("a", False), ("b", True), ("c", False)]
    assert split_clauses("   ") == []


async def test_rule_based_planner_routes_each_clause():
    steps = await RuleBasedPlanner().plan(
        "Convert 100 USD to EUR currency; extract keywords from the text", CATALOG
    )
    assert [(s.agent_id, s.depends_on) for s in steps] == [("fx", ()), ("text", ())]


async def test_rule_based_planner_rejects_unroutable_and_empty_requests():
    with pytest.raises(PlanningError, match="no available agent"):
        await RuleBasedPlanner().plan("sing me a lullaby", CATALOG)
    with pytest.raises(PlanningError, match="no agents"):
        await RuleBasedPlanner().plan("anything", [])


class ScriptedLLM:
    name = "scripted"
    model = "scripted"

    def __init__(self, reply: str | Exception) -> None:
        self.reply = reply

    async def complete(self, messages, **_):
        if isinstance(self.reply, Exception):
            raise self.reply
        return LLMResponse(self.reply, "scripted", "scripted")


async def test_llm_planner_uses_valid_model_plan():
    plan = {
        "steps": [
            {"id": "s1", "agent": "text", "instruction": "extract keywords"},
            {
                "id": "s2",
                "agent": "fx",
                "instruction": "convert 5 USD to EUR",
                "depends_on": ["s1"],
            },
        ]
    }
    steps = await LLMPlanner(ScriptedLLM(json.dumps(plan))).plan("whatever", CATALOG)
    assert [(s.id, s.agent_id, s.depends_on) for s in steps] == [
        ("s1", "text", ()),
        ("s2", "fx", ("s1",)),
    ]


@pytest.mark.parametrize(
    "reply",
    [
        "not json",
        json.dumps({"steps": [{"id": "s1", "agent": "ghost", "instruction": "x"}]}),
        json.dumps(
            {"steps": [{"id": "s1", "agent": "fx", "instruction": "x", "depends_on": ["s9"]}]}
        ),
        json.dumps({"steps": []}),
        LLMUnavailableError("down"),
    ],
)
async def test_llm_planner_falls_back_to_rules_on_invalid_plans(reply):
    steps = await LLMPlanner(ScriptedLLM(reply)).plan("convert currency amounts", CATALOG)
    assert [s.agent_id for s in steps] == ["fx"]


# ------------------------------------------------------------------ integration


async def run(front, text: str, **kwargs):
    client = await AgentClient.connect("http://orch.test", http=front, streaming=False)
    return client, await client.send(text, **kwargs)


async def test_orchestrator_spans_native_adk_and_agent_framework_agents():
    request = (
        "Convert 100 USD to EUR currency; "
        "Monthly payment on 250000 loan at 6.5% over 30 years; "
        "Compute text statistics and reading time of this passage"
    )
    async with build_mesh() as mesh, mesh.frontdoor() as front:
        _, result = await run(front, request)

    assert result.succeeded
    assert "Completed 3 of 3" in result.text
    assert "90.00 EUR" in result.text  # native agent
    assert "1580.17" in result.text  # Microsoft Agent Framework agent
    assert "4 words" in result.text  # Google ADK agent
    names = {a.name for a in result.artifacts}
    assert {"s1:fx-conversion-agent", "s2:quantitative-analysis-agent", "summary"} <= names


async def test_dependent_steps_receive_upstream_results():
    async with build_mesh() as mesh, mesh.frontdoor() as front:
        _, result = await run(
            front, "Convert 100 USD to EUR currency then compute text statistics of the passage"
        )
        summary = result.artifact("summary")
    assert result.succeeded
    assert summary is not None
    assert [s["ok"] for s in summary.data[0]["steps"]] == [True, True]


async def test_unreachable_agent_yields_partial_result():
    request = "Convert 100 USD to EUR currency; Monthly payment on 250000 loan over 30 years"
    async with build_mesh(unreachable=("quant",)) as mesh, mesh.frontdoor() as front:
        _, result = await run(front, request)
    assert result.state_name == "completed"  # fx still delivered
    assert "Completed 1 of 2" in result.text
    assert "90.00 EUR" in result.text
    assert "unreachable" in result.text


async def test_all_agents_unreachable_fails_the_task():
    async with build_mesh(unreachable=("fx",)) as mesh, mesh.frontdoor() as front:
        _, result = await run(front, "Convert 100 USD to EUR currency")
    assert result.state_name == "failed"
    assert "unreachable" in result.text


async def test_input_required_is_surfaced_and_resumed():
    async with build_mesh() as mesh, mesh.frontdoor() as front:
        client, first = await run(front, "convert 250 to EUR currency")
        assert first.needs_input
        assert "fx" in first.text and "converting from" in first.text.lower()

        second = await client.send("USD", task_id=first.task_id, context_id=first.context_id)
    assert second.succeeded
    assert "225.00 EUR" in second.text


async def test_unroutable_request_is_rejected_with_reason():
    async with build_mesh() as mesh, mesh.frontdoor() as front:
        _, result = await run(front, "compose a symphony in d minor")
    assert result.state_name == "rejected"
    assert "no available agent" in result.text


async def test_files_are_forwarded_to_the_selected_agent():
    from a2a.helpers import new_raw_part, new_text_part

    parts = [
        new_text_part("profile this csv document"),
        new_raw_part(b"id,price\n1,2.5\n2,3.5\n", media_type="text/csv", filename="p.csv"),
    ]
    async with build_mesh() as mesh, mesh.frontdoor() as front:
        client = await AgentClient.connect("http://orch.test", http=front, streaming=False)
        result = await client.send(parts)
    assert result.succeeded
    assert "Profiled 2 column" in result.text
