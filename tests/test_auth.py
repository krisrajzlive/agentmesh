from __future__ import annotations

from decimal import Decimal

import pytest
from a2a.client import ClientConfig, create_client
from a2a.types import GetTaskRequest

from agentmesh.agents.fx import FX_SPEC, StaticRates, build_fx_executor
from agentmesh.config import Settings
from agentmesh.runtime import create_agent_app
from agentmesh.security import (
    ApiKeyAuthenticator,
    ChainAuthenticator,
    JWTAuthenticator,
    issue_token,
)
from tests.conftest import BASE_URL, serve
from tests.helpers import final_state, task_id_of, user_request

SECRET = "x" * 40


def _settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)  # type: ignore[call-arg]


def _app(settings: Settings):
    executor = build_fx_executor(settings, rates=StaticRates({"EUR": Decimal("0.9")}))
    return create_agent_app(FX_SPEC, executor, settings, public_url=BASE_URL)


async def _client(http):
    return await create_client(
        BASE_URL,
        client_config=ClientConfig(
            httpx_client=http, streaming=False, supported_protocol_bindings=["JSONRPC"]
        ),
    )


def test_api_key_authenticator():
    auth = ApiKeyAuthenticator({"k-1": "orchestrator"})
    assert auth.authenticate({"x-api-key": "k-1"}).name == "orchestrator"
    assert auth.authenticate({"authorization": "Bearer k-1"}).name == "orchestrator"
    assert auth.authenticate({"x-api-key": "nope"}) is None
    assert auth.authenticate({}) is None


def test_jwt_authenticator_validates_signature_audience_and_expiry():
    auth = JWTAuthenticator(SECRET, audience="agentmesh")
    good = issue_token(SECRET, "svc-a", audience="agentmesh", scopes=["tasks:write"])
    principal = auth.authenticate({"authorization": f"Bearer {good}"})
    assert principal.name == "svc-a"
    assert principal.scopes == ("tasks:write",)

    wrong_aud = issue_token(SECRET, "svc-a", audience="other")
    wrong_key = issue_token("y" * 40, "svc-a", audience="agentmesh")
    expired = issue_token(SECRET, "svc-a", audience="agentmesh", ttl_seconds=-10)
    for token in (wrong_aud, wrong_key, expired, "garbage"):
        assert auth.authenticate({"authorization": f"Bearer {token}"}) is None


def test_chain_tries_each_authenticator():
    chain = ChainAuthenticator(
        [ApiKeyAuthenticator({"k": "a"}), JWTAuthenticator(SECRET, audience="agentmesh")]
    )
    token = issue_token(SECRET, "b", audience="agentmesh")
    assert chain.authenticate({"x-api-key": "k"}).name == "a"
    assert chain.authenticate({"authorization": f"Bearer {token}"}).name == "b"


async def test_protected_agent_rejects_anonymous_but_serves_card():
    app = _app(_settings(auth_mode="api_key", api_keys="svc:secret-key"))
    async with serve(app) as http:
        card = await http.get("/.well-known/agent-card.json")
        rpc = await http.post("/a2a/jsonrpc", json={})
        health = await http.get("/healthz")
    assert card.status_code == 200
    assert "apiKey" in card.json()["securitySchemes"]
    assert rpc.status_code == 401
    assert rpc.headers["www-authenticate"].startswith("Bearer")
    assert health.status_code == 200


async def test_api_key_grants_access_end_to_end():
    app = _app(_settings(auth_mode="api_key", api_keys="svc:secret-key"))
    async with serve(app) as http:
        client = await _client(http)
        http.headers["x-api-key"] = "secret-key"
        events = [e async for e in client.send_message(user_request("Convert 10 USD to EUR"))]
    assert final_state(events) == "TASK_STATE_COMPLETED"


async def test_jwt_grants_access_end_to_end():
    app = _app(_settings(auth_mode="jwt", jwt_secret=SECRET))
    async with serve(app) as http:
        client = await _client(http)
        http.headers["authorization"] = "Bearer " + issue_token(SECRET, "svc", audience="agentmesh")
        events = [e async for e in client.send_message(user_request("Convert 10 USD to EUR"))]
    assert final_state(events) == "TASK_STATE_COMPLETED"


async def test_tasks_are_isolated_per_principal():
    app = _app(_settings(auth_mode="api_key", api_keys="alice:ka,bob:kb"))
    async with serve(app) as http:
        client = await _client(http)

        http.headers["x-api-key"] = "ka"
        events = [e async for e in client.send_message(user_request("Convert 10 USD to EUR"))]
        task_id = task_id_of(events)
        assert (await client.get_task(GetTaskRequest(id=task_id))).id == task_id

        http.headers["x-api-key"] = "kb"
        with pytest.raises(Exception, match=r"(?i)not found|TaskNotFound"):
            await client.get_task(GetTaskRequest(id=task_id))


def test_production_requires_authentication():
    with pytest.raises(ValueError, match="production"):
        _settings(environment="production", auth_mode="none")
