"""Request authentication: API keys and JWT bearer tokens."""

from __future__ import annotations

import hmac
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import jwt

from agentmesh.config import Settings


@dataclass(frozen=True, slots=True)
class Principal:
    """An authenticated caller. ``name`` becomes the task owner (tenant isolation)."""

    name: str
    method: str
    scopes: tuple[str, ...] = ()


class Authenticator(Protocol):
    def authenticate(self, headers: Mapping[str, str]) -> Principal | None:
        """Return the caller, or ``None`` if these headers carry no valid credential."""
        ...


def _bearer(headers: Mapping[str, str]) -> str | None:
    value = headers.get("authorization", "")
    scheme, _, token = value.partition(" ")
    return token.strip() if scheme.lower() == "bearer" and token.strip() else None


class ApiKeyAuthenticator:
    """Static API keys supplied as ``X-API-Key`` (or a bearer token)."""

    HEADER = "x-api-key"

    def __init__(self, keys: Mapping[str, str]) -> None:
        self._keys = dict(keys)

    def authenticate(self, headers: Mapping[str, str]) -> Principal | None:
        candidate = headers.get(self.HEADER) or _bearer(headers)
        if not candidate:
            return None
        match: str | None = None
        # Compare against every key so timing does not reveal which prefix matched.
        for key, principal in self._keys.items():
            if hmac.compare_digest(candidate.encode(), key.encode()):
                match = principal
        return Principal(match, "api_key") if match else None


class JWTAuthenticator:
    """HMAC- or RSA-signed JWT bearer tokens. ``sub`` is the principal name."""

    def __init__(
        self,
        key: str,
        *,
        algorithms: Sequence[str] = ("HS256",),
        audience: str | None = None,
        issuer: str | None = None,
    ) -> None:
        self._key = key
        self._algorithms = list(algorithms)
        self._audience = audience
        self._issuer = issuer

    def authenticate(self, headers: Mapping[str, str]) -> Principal | None:
        token = _bearer(headers)
        if not token:
            return None
        try:
            claims = jwt.decode(
                token,
                self._key,
                algorithms=self._algorithms,
                audience=self._audience,
                issuer=self._issuer,
                options={"require": ["exp", "sub"]},
            )
        except jwt.PyJWTError:
            return None
        scopes = claims.get("scope", "")
        scope_list = scopes.split() if isinstance(scopes, str) else list(scopes)
        return Principal(str(claims["sub"]), "jwt", tuple(scope_list))


class ChainAuthenticator:
    def __init__(self, authenticators: Sequence[Authenticator]) -> None:
        self._authenticators = list(authenticators)

    def authenticate(self, headers: Mapping[str, str]) -> Principal | None:
        for authenticator in self._authenticators:
            if principal := authenticator.authenticate(headers):
                return principal
        return None


def build_authenticator(settings: Settings) -> Authenticator | None:
    """Return the authenticator for ``settings.auth_mode`` (``None`` means open access)."""
    chain: list[Authenticator] = []
    if settings.auth_mode in {"api_key", "api_key_or_jwt"} and settings.api_keys:
        chain.append(ApiKeyAuthenticator(settings.api_key_map()))
    if settings.auth_mode in {"jwt", "api_key_or_jwt"} and settings.jwt_secret:
        chain.append(
            JWTAuthenticator(
                settings.jwt_secret.get_secret_value(),
                algorithms=[settings.jwt_algorithm],
                audience=settings.jwt_audience,
                issuer=settings.jwt_issuer,
            )
        )
    if not chain:
        return None
    return chain[0] if len(chain) == 1 else ChainAuthenticator(chain)


def issue_token(
    secret: str,
    subject: str,
    *,
    audience: str,
    ttl_seconds: int = 3600,
    scopes: Sequence[str] = (),
    algorithm: str = "HS256",
    issuer: str | None = None,
) -> str:
    """Mint a signed JWT (used by the CLI and by service-to-service callers)."""
    import time

    now = int(time.time())
    claims: dict[str, object] = {
        "sub": subject,
        "aud": audience,
        "iat": now,
        "exp": now + ttl_seconds,
        "scope": " ".join(scopes),
    }
    if issuer:
        claims["iss"] = issuer
    return jwt.encode(claims, secret, algorithm=algorithm)
