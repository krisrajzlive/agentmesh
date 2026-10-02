"""Exchange-rate sources."""

from __future__ import annotations

import time
from decimal import Decimal
from typing import Protocol

import httpx


class RateUnavailableError(Exception):
    """The rate source could not provide a quote."""


class RateProvider(Protocol):
    async def rate(self, source: str, target: str) -> tuple[Decimal, str]:
        """Return ``(rate, as_of_date)`` for converting one unit of ``source`` into ``target``."""
        ...

    async def aclose(self) -> None: ...


class FrankfurterRates:
    """ECB reference rates via the free Frankfurter API, with a short TTL cache."""

    def __init__(
        self,
        base_url: str = "https://api.frankfurter.dev/v1",
        *,
        ttl_seconds: float = 300.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._ttl = ttl_seconds
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=10.0)
        self._cache: dict[tuple[str, str], tuple[float, Decimal, str]] = {}

    async def rate(self, source: str, target: str) -> tuple[Decimal, str]:
        if source == target:
            return Decimal(1), "n/a"
        key = (source, target)
        cached = self._cache.get(key)
        if cached and time.monotonic() - cached[0] < self._ttl:
            return cached[1], cached[2]
        try:
            response = await self._client.get(
                f"{self._base_url}/latest", params={"base": source, "symbols": target}
            )
            response.raise_for_status()
            data = response.json()
            value = Decimal(str(data["rates"][target]))
            as_of = str(data["date"])
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise RateUnavailableError(f"no rate for {source}->{target}") from exc
        self._cache[key] = (time.monotonic(), value, as_of)
        return value, as_of

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class StaticRates:
    """Fixed rate table (USD-based) for tests and offline operation."""

    def __init__(self, usd_rates: dict[str, Decimal], as_of: str = "2026-01-01") -> None:
        self._usd = {"USD": Decimal(1), **usd_rates}
        self._as_of = as_of

    async def rate(self, source: str, target: str) -> tuple[Decimal, str]:
        try:
            return self._usd[target] / self._usd[source], self._as_of
        except KeyError as exc:
            raise RateUnavailableError(f"no rate for {source}->{target}") from exc

    async def aclose(self) -> None:
        return None
