# AgentMesh

**An A2A-compliant multi-agent orchestration platform.**

AgentMesh hosts, secures, discovers and coordinates AI agents using the
[Agent2Agent (A2A) protocol](https://a2a-protocol.org). It is built on the official
[`a2a-sdk`](https://github.com/a2aproject/a2a-python) (A2A spec v1.0) and adds the
production concerns the SDK leaves to you: authentication, persistence, resilient LLM
access, signed webhooks, observability and deployment.

> Status: under active development. See the [roadmap](#roadmap) for what is shipped.

## Highlights

- **Agent runtime** – host any A2A `AgentExecutor` behind JSON-RPC *and* HTTP+JSON bindings with
  one call. Native agents, Google ADK agents and Microsoft Agent Framework agents run on the same
  hardened runtime.
- **Security** – API-key and JWT authentication, per-principal task isolation, SSRF-safe webhook
  validation, secrets never logged.
- **Resilient LLM layer** – OpenAI, Hugging Face and Ollama behind one interface, with retries,
  circuit breakers and ordered failover.
- **Reliable push notifications** – HMAC-signed webhooks, background delivery with retries and a
  dead-letter queue.
- **Persistence** – in-memory, SQLite or PostgreSQL task stores.
- **Observability** – structured logs with request correlation, Prometheus metrics, health and
  readiness probes.

## Quick start

```bash
uv sync
cp .env.example .env            # add provider keys (optional)
uv run agentmesh agents         # list hostable agents
uv run agentmesh serve fx --port 9001
curl http://localhost:9001/.well-known/agent-card.json
```

The FX agent works with no API key (rule-based parsing). Set `AGENTMESH_LLM_PROVIDERS` and a
provider key to let an LLM interpret free-form requests, with automatic fallback to rules.

## Configuration

All settings are environment variables prefixed `AGENTMESH_` (see [`.env.example`](.env.example)).
Provider credentials use each vendor's standard names: `OPENAI_API_KEY`, `HF_TOKEN`,
`OLLAMA_API_KEY`.

| Variable | Purpose |
|---|---|
| `AGENTMESH_ENVIRONMENT` | `production` refuses to start without authentication |
| `AGENTMESH_AUTH_MODE` | `none`, `api_key`, `jwt`, `api_key_or_jwt` |
| `AGENTMESH_TASK_STORE_URL` | `memory` or a SQLAlchemy async URL |
| `AGENTMESH_LLM_PROVIDERS` | Failover order, e.g. `openai,huggingface,ollama` |
| `AGENTMESH_PUSH_SIGNING_SECRET` | Enables HMAC signing of webhook deliveries |

## Development

```bash
uv sync --all-extras --prerelease=allow
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

Tests run entirely in-process and need no network or API keys. Tests that call real LLM
providers are marked `live` and are skipped by default.

## Roadmap

- [x] Agent runtime, auth, persistence, signed push, observability
- [x] LLM provider layer with failover
- [x] FX agent (multi-turn, structured artifacts)
- [ ] Google ADK and Microsoft Agent Framework agents
- [ ] Registry, client SDK and CLI
- [ ] Orchestrator and gateway
- [ ] Container images, Kubernetes manifests, architecture docs

## License

Apache-2.0
