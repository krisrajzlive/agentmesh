"""Run ASGI apps on real sockets (uvicorn in a thread) for tests that need true streaming."""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import httpx
import uvicorn
from starlette.applications import Starlette


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def serve_live(build_app: Callable[[str], Starlette]) -> Iterator[str]:
    """Start ``build_app(base_url)`` on a free port and yield its base URL.

    The app is built after the port is chosen because an Agent Card advertises its own URL.
    """
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    server = uvicorn.Server(
        uvicorn.Config(build_app(base), host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{base}/readyz", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    else:
        server.should_exit = True
        raise RuntimeError("server did not become ready")
    try:
        yield base
    finally:
        server.should_exit = True
        thread.join(timeout=10)
