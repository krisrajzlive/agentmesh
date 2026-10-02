"""Prometheus metrics shared across components."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS = Counter(
    "agentmesh_http_requests_total",
    "HTTP requests handled",
    ["component", "method", "path", "status"],
)
HTTP_LATENCY = Histogram(
    "agentmesh_http_request_duration_seconds", "HTTP request latency", ["component", "path"]
)
LLM_REQUESTS = Counter(
    "agentmesh_llm_requests_total", "LLM provider calls", ["provider", "outcome"]
)
LLM_TOKENS = Counter("agentmesh_llm_tokens_total", "LLM tokens consumed", ["provider", "kind"])
LLM_LATENCY = Histogram(
    "agentmesh_llm_request_duration_seconds", "LLM provider latency", ["provider"]
)
CIRCUIT_STATE = Gauge(
    "agentmesh_circuit_open", "1 when the circuit for a dependency is open", ["dependency"]
)
TASKS_TOTAL = Counter("agentmesh_agent_tasks_total", "Agent task executions", ["agent", "outcome"])
