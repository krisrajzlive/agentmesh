from __future__ import annotations

import asyncio
import json
from decimal import Decimal

import respx
from a2a.client import ClientConfig, create_client
from a2a.types import (
    AgentSkill,
    GetExtendedAgentCardRequest,
    SendMessageConfiguration,
    TaskPushNotificationConfig,
)

from agentmesh.agents.fx import FX_SPEC, StaticRates, build_fx_executor
from agentmesh.config import Settings
from agentmesh.runtime import create_agent_app
from agentmesh.runtime.push import (
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    DeadLetterQueue,
    sign_payload,
    verify_push_signature,
)
from tests.conftest import BASE_URL, serve
from tests.helpers import user_request

SECRET = "whsec-" + "a" * 32
HOOK = "http://hooks.test/callback"


def test_signature_roundtrip_and_tamper_detection():
    body = b'{"a":1}'
    sig = sign_payload(SECRET, "1000", body)
    assert verify_push_signature(SECRET, body=body, timestamp="1000", signature=sig, now=1100)
    assert not verify_push_signature(
        SECRET, body=b'{"a":2}', timestamp="1000", signature=sig, now=1100
    )
    assert not verify_push_signature("other", body=body, timestamp="1000", signature=sig, now=1100)
    assert not verify_push_signature(SECRET, body=body, timestamp="1000", signature=sig, now=9999)
    assert not verify_push_signature(SECRET, body=body, timestamp="x", signature=sig)


async def _wait_for(predicate, timeout: float = 5.0) -> None:  # noqa: ASYNC109
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise TimeoutError("condition not met")
        await asyncio.sleep(0.02)


@respx.mock
async def test_completed_task_is_pushed_with_valid_signature():
    hook = respx.post(HOOK).respond(200)
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, push_signing_secret=SECRET, allow_private_push_urls=True
    )
    executor = build_fx_executor(settings, rates=StaticRates({"EUR": Decimal("0.9")}))
    app = create_agent_app(FX_SPEC, executor, settings, public_url=BASE_URL)

    async with serve(app) as http:
        client = await create_client(
            BASE_URL,
            client_config=ClientConfig(
                httpx_client=http, streaming=False, supported_protocol_bindings=["JSONRPC"]
            ),
        )
        config = SendMessageConfiguration(
            task_push_notification_config=TaskPushNotificationConfig(url=HOOK, token="tok-1")
        )
        _ = [
            e
            async for e in client.send_message(
                user_request("Convert 10 USD to EUR", configuration=config)
            )
        ]
        await _wait_for(lambda: hook.called)

    request = hook.calls.last.request
    body = request.read()
    assert request.headers["x-a2a-notification-token"] == "tok-1"
    assert verify_push_signature(
        SECRET,
        body=body,
        timestamp=request.headers[TIMESTAMP_HEADER],
        signature=request.headers[SIGNATURE_HEADER],
    )
    assert json.loads(body)  # valid JSON payload describing the task event


@respx.mock
async def test_unreachable_webhook_is_dead_lettered(monkeypatch):
    async def instant(_delay: float) -> None:
        return None

    monkeypatch.setattr("asyncio.sleep", instant)
    respx.post(HOOK).respond(500)
    settings = Settings(_env_file=None, allow_private_push_urls=True)  # type: ignore[call-arg]
    executor = build_fx_executor(settings, rates=StaticRates({"EUR": Decimal("0.9")}))
    app = create_agent_app(FX_SPEC, executor, settings, public_url=BASE_URL)
    dead: DeadLetterQueue = app.state.push_sender.dead_letters

    async with serve(app) as http:
        client = await create_client(
            BASE_URL,
            client_config=ClientConfig(
                httpx_client=http, streaming=False, supported_protocol_bindings=["JSONRPC"]
            ),
        )
        config = SendMessageConfiguration(
            task_push_notification_config=TaskPushNotificationConfig(url=HOOK)
        )
        _ = [
            e
            async for e in client.send_message(
                user_request("Convert 10 USD to EUR", configuration=config)
            )
        ]
        await _wait_for(lambda: len(dead) > 0)
    assert dead.items()[0].url == HOOK


async def test_private_webhook_urls_are_rejected_by_default():
    """SSRF guard: with default settings, loopback/private targets are never contacted."""
    from a2a.utils.push_url_validator import validate_push_notification_url

    assert not await validate_push_notification_url("http://127.0.0.1:8080/hook")
    assert not await validate_push_notification_url("http://169.254.169.254/latest/meta-data")
    assert not await validate_push_notification_url("file:///etc/passwd")


async def test_extended_agent_card_adds_privileged_skills():
    from dataclasses import replace

    spec = replace(
        FX_SPEC,
        extended_skills=[
            AgentSkill(
                id="bulk-conversion",
                name="Bulk conversion",
                description="Batch conversions for authenticated partners.",
                tags=["finance"],
            )
        ],
    )
    settings = Settings(_env_file=None, auth_mode="api_key", api_keys="partner:pk")  # type: ignore[call-arg]
    app = create_agent_app(spec, build_fx_executor(settings), settings, public_url=BASE_URL)
    async with serve(app) as http:
        public = (await http.get("/.well-known/agent-card.json")).json()
        client = await create_client(
            BASE_URL,
            client_config=ClientConfig(
                httpx_client=http, streaming=False, supported_protocol_bindings=["JSONRPC"]
            ),
        )
        http.headers["x-api-key"] = "pk"
        extended = await client.get_extended_agent_card(GetExtendedAgentCardRequest())
    assert {s["id"] for s in public["skills"]} == {"currency-conversion"}
    assert {s.id for s in extended.skills} == {"currency-conversion", "bulk-conversion"}
    assert public["capabilities"]["extendedAgentCard"] is True
