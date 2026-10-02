"""Outbound-URL validation to stop server-side request forgery."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urlparse


class UnsafeURLError(ValueError):
    """The URL points somewhere this service must not connect to."""


def _blocked(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address, *, allow_private: bool
) -> bool:
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped  # ::ffff:169.254.169.254 is still the metadata service
    # Cloud metadata endpoints, multicast and unspecified addresses are never acceptable.
    if address.is_link_local or address.is_multicast or address.is_unspecified:
        return True
    if address.is_loopback or address.is_private:
        return not allow_private
    # Python flags the whole ::/8 block (including ::1) as reserved, hence the order above.
    return address.is_reserved


async def validate_outbound_url(url: str, *, allow_private: bool) -> None:
    """Raise :class:`UnsafeURLError` unless ``url`` is an http(s) URL to an acceptable host.

    ``allow_private`` permits RFC1918/loopback targets (needed for in-cluster agents) but
    link-local ranges such as ``169.254.169.254`` stay blocked either way.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise UnsafeURLError("only absolute http(s) URLs are allowed")
    if parsed.username or parsed.password:
        raise UnsafeURLError("credentials in URLs are not allowed")

    host = parsed.hostname
    try:
        literals = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            infos = await asyncio.get_running_loop().getaddrinfo(
                host,
                parsed.port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            raise UnsafeURLError(f"host {host!r} cannot be resolved") from exc
        literals = [ipaddress.ip_address(info[4][0].split("%", 1)[0]) for info in infos]

    for address in literals:
        if _blocked(address, allow_private=allow_private):
            raise UnsafeURLError(f"host {host!r} resolves to a disallowed address")
