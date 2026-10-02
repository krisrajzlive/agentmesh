"""Resilience primitives: circuit breaker and token-bucket rate limiter."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from enum import StrEnum

from agentmesh.observability.metrics import CIRCUIT_STATE


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised when a call is rejected because the circuit is open."""


class CircuitBreaker:
    """Opens after ``failure_threshold`` consecutive failures; probes after ``reset_timeout``."""

    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 3,
        reset_timeout: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self._threshold = failure_threshold
        self._reset_timeout = reset_timeout
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None
        self._probe_in_flight = False

    @property
    def state(self) -> CircuitState:
        if self._opened_at is None:
            return CircuitState.CLOSED
        if self._clock() - self._opened_at >= self._reset_timeout:
            return CircuitState.HALF_OPEN
        return CircuitState.OPEN

    def allow(self) -> bool:
        """Return True if a call may proceed (and claim the probe slot when half-open)."""
        state = self.state
        if state is CircuitState.CLOSED:
            return True
        if state is CircuitState.HALF_OPEN and not self._probe_in_flight:
            self._probe_in_flight = True
            return True
        return False

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None
        self._probe_in_flight = False
        CIRCUIT_STATE.labels(self.name).set(0)

    def record_failure(self) -> None:
        self._probe_in_flight = False
        self._failures += 1
        if self._failures >= self._threshold or self._opened_at is not None:
            self._opened_at = self._clock()
            CIRCUIT_STATE.labels(self.name).set(1)


class TokenBucket:
    """Async token bucket: ``rate`` tokens/second with burst capacity ``capacity``."""

    def __init__(
        self,
        rate: float,
        capacity: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if rate <= 0 or capacity <= 0:
            raise ValueError("rate and capacity must be positive")
        self._rate = rate
        self._capacity = capacity
        self._tokens = capacity
        self._clock = clock
        self._updated = clock()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = self._clock()
        self._tokens = min(self._capacity, self._tokens + (now - self._updated) * self._rate)
        self._updated = now

    @property
    def available(self) -> float:
        """Tokens currently in the bucket (as of the last refill)."""
        return self._tokens

    async def try_acquire(self, cost: float = 1.0) -> bool:
        async with self._lock:
            self._refill()
            if self._tokens >= cost:
                self._tokens -= cost
                return True
            return False

    async def retry_after(self, cost: float = 1.0) -> float:
        async with self._lock:
            self._refill()
            return max(0.0, (cost - self._tokens) / self._rate)
