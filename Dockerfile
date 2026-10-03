# syntax=docker/dockerfile:1.7

# ---- build: resolve and install locked dependencies into a virtualenv ----------------------
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Optional extras baked into the image, e.g. --build-arg EXTRAS="--all-extras"
#   adk   -> Google ADK agent        maf -> Microsoft Agent Framework agent
#   mcp   -> MCP tool bridge         redis -> shared gateway rate limiting
ARG EXTRAS="--extra redis"

COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project ${EXTRAS}

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable ${EXTRAS}

# ---- runtime: only the virtualenv, no build tooling, non-root ------------------------------
FROM python:3.12-slim AS runtime

RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin agentmesh
COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AGENTMESH_LOG_JSON=true \
    PORT=9000

USER 10001
EXPOSE 9000

HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ['PORT'], timeout=2)"

ENTRYPOINT ["agentmesh"]
CMD ["serve", "fx", "--port", "9000"]
