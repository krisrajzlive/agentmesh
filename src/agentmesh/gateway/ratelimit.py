"""Per-principal rate limiting (in-memory token bucket, or Redis for multi-replica gateways)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from agentmesh.resilience import TokenBucket


@dataclass(frozen=True, slots=True)
class RateDecision:
    allowed: bool
    limit: int
    remaining: int
    retry_after_seconds: float = 0.0


class RateLimiter(Protocol):
    async def check(self, key: str) -> RateDecision: ...


class InMemoryRateLimiter:
    """Token bucket per key: ``limit_per_minute`` sustained, with a burst of the same size."""

    def __init__(self, limit_per_minute: int) -> None:
        self._limit = limit_per_minute
        self._buckets: dict[str, TokenBucket] = {}

    async def check(self, key: str) -> RateDecision:
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = self._buckets[key] = TokenBucket(self._limit / 60.0, float(self._limit))
        if await bucket.try_acquire():
            return RateDecision(True, self._limit, remaining=int(bucket.available))
        return RateDecision(False, self._limit, 0, await bucket.retry_after())


class RedisFixedWindowLimiter:
    """Fixed-window counter shared by every gateway replica through Redis."""

    def __init__(
        self,
        client: Any,  # redis.asyncio.Redis
        limit_per_minute: int,
        *,
        window_seconds: int = 60,
        prefix: str = "agentmesh:ratelimit:",
        clock: Any = time.time,
    ) -> None:
        self._client = client
        self._limit = limit_per_minute
        self._window = window_seconds
        self._prefix = prefix
        self._clock = clock

    async def check(self, key: str) -> RateDecision:
        now = self._clock()
        window_id = int(now // self._window)
        redis_key = f"{self._prefix}{key}:{window_id}"
        count = int(await self._client.incr(redis_key))
        if count == 1:
            await self._client.expire(redis_key, self._window + 1)
        if count <= self._limit:
            return RateDecision(True, self._limit, self._limit - count)
        return RateDecision(False, self._limit, 0, self._window - (now % self._window))


def build_rate_limiter(limit_per_minute: int, redis_url: str | None) -> RateLimiter:
    if redis_url:
        try:
            from redis.asyncio import Redis
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "install the 'redis' extra to use AGENTMESH_GATEWAY_REDIS_URL"
            ) from exc
        return RedisFixedWindowLimiter(Redis.from_url(redis_url), limit_per_minute)
    return InMemoryRateLimiter(limit_per_minute)
