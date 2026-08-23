"""Dashboard SSRF destination gate for `POST /api/scans` (DASH-07, Phase 11).

This gate runs at SUBMISSION time, before `build_dashboard_scan_config`, id
generation, or `dispatch_scan` -- it deliberately leaves a DNS-rebinding
TOCTOU gap DOCUMENTED rather than closed (D-11): a low-TTL DNS record can
resolve differently between this check and the moment `adapters/http_app.py`
actually opens its connection. Closing that gap would mean per-connection IP
pinning inside `HttpAppAdapter`'s request path, which is out of scope here --
`adapters/http_app.py` is NOT D-73-pinned (unlike `orchestrator.py` and
`adapters/llm_api.py`), so a future phase may close it. Disclosed plainly in
the README (plan 11-03), never implied closed (honest degradation over
silent success, CLAUDE.md).

`raw_llm` targets are structurally exempt from this module entirely --
`litellm` resolves a provider-qualified model string (e.g.
`openai/gpt-4o-mini`) through its own routing layer, never an operator-
supplied host, so there is no hostname here to validate (D-08). The
caller (`routes.py`) never invokes `validate_target_not_internal` for a
`raw_llm` body at all -- the exemption is a structurally absent call, not a
no-op branch inside this module.
"""

from __future__ import annotations

import asyncio
from ipaddress import IPv4Address, IPv6Address, ip_address
from urllib.parse import urlsplit


class TargetRejected(Exception):
    """Raised when a submitted `http_app` target URL is not scannable."""


#: Bound on DNS resolution inside the SSRF gate (11-REVIEW.md WR-03). An
#: unbound `getaddrinfo()` call leaves `POST /api/scans` pending for as
#: long as the resolver's own (often very long or absent) internal
#: timeout allows, tying up a request for a hostname whose authoritative
#: nameserver never responds. This is a request-time bound, not a
#: fresh-config-dial like `consumption_latency_threshold_ms` -- there is
#: no honest "configurable default" to document here, only a fixed guard.
_DNS_RESOLVE_TIMEOUT_SECONDS = 5.0


async def _resolve_ips(hostname: str) -> list[IPv4Address | IPv6Address]:
    """Resolve `hostname` to its IP address literals via the running loop.

    The ONE seam tests monkeypatch -- mirrors `api.py::_build_adapter()`'s
    "one seam, monkeypatched directly, never mock the stdlib underneath it"
    discipline. Uses `asyncio.get_running_loop().getaddrinfo(...)`, NEVER
    the blocking `socket.getaddrinfo()` directly: calling the blocking
    stdlib resolver from inside an async request handler would stall the
    entire event loop -- and every OTHER concurrent request -- for the
    duration of the DNS lookup. Bounded by `_DNS_RESOLVE_TIMEOUT_SECONDS`;
    a `TimeoutError` here is caught by the caller's existing
    `except Exception` and folded into the same "could not be resolved"
    `TargetRejected`, consistent with the NXDOMAIN-is-a-400 discipline.
    """
    loop = asyncio.get_running_loop()
    addrinfo = await asyncio.wait_for(
        loop.getaddrinfo(hostname, None), timeout=_DNS_RESOLVE_TIMEOUT_SECONDS
    )
    return [ip_address(entry[4][0]) for entry in addrinfo]


def _is_disallowed(addr: IPv4Address | IPv6Address) -> bool:
    """Return whether `addr` is loopback, private, link-local, reserved,
    multicast, unspecified, non-globally-routable, or an IPv4-mapped-IPv6
    equivalent of any of those.

    FIRST unwraps an IPv4-mapped IPv6 address via `addr.ipv4_mapped or addr`,
    THEN tests the unwrapped address's properties -- this ordering is
    load-bearing: `is_loopback`/`is_private` on a raw IPv4-mapped IPv6
    address (`::ffff:127.0.0.1`) are only correct on sufficiently recent
    CPython patch releases (cpython issue 77614), and this project's
    `requires-python` carries no patch-level floor (11-RESEARCH.md
    Pitfall 4). Unwrapping first sidesteps that version dependency
    entirely rather than relying on it.

    The six named properties alone miss RFC 6598 Carrier-Grade-NAT / Shared
    Address Space (`100.64.0.0/10`) -- which satisfies none of
    is_private/is_loopback/is_link_local/is_reserved/is_multicast/
    is_unspecified, yet is not globally routable, and covers at least one
    real-world cloud metadata endpoint (Alibaba Cloud ECS's
    `100.100.100.200`, the functional equivalent of `169.254.169.254`).
    `or not unwrapped.is_global` closes that gap. It cannot REPLACE the six
    properties, though: `is_global` alone treats a multicast address
    (`224.0.0.1`) as globally routable (verified against the stdlib), so
    `is_multicast` must stay explicit in the OR chain, not folded into
    `is_global` alone. Together this tracks the IANA special-purpose
    registry the way the module intends, without hand-maintaining a CIDR
    list.
    """
    ipv4_mapped = getattr(addr, "ipv4_mapped", None)
    unwrapped = ipv4_mapped if ipv4_mapped is not None else addr
    return (
        unwrapped.is_private
        or unwrapped.is_loopback
        or unwrapped.is_link_local
        or unwrapped.is_reserved
        or unwrapped.is_multicast
        or unwrapped.is_unspecified
        or not unwrapped.is_global
    )


async def validate_target_not_internal(url: str) -> None:
    """Raise `TargetRejected` if `url` is not a scannable `http`/`https` target.

    Rejects (FIXED, non-interpolating message -- never echoes the
    submitted URL, 11-PATTERNS.md's error-detail discipline):
    - a scheme other than `http`/`https`
    - a URL with no hostname
    - a hostname that fails to resolve (NXDOMAIN, timeout) -- an
      unresolvable host is not a server error, so this is a 400-shaped
      rejection, never left to surface as an unhandled 500

    Rejects (message interpolates the parsed `hostname` ONLY -- bracket-
    stripped, no userinfo, no query string, no header value, no
    `body_template`; 11-UI-SPEC.md renders this string verbatim):
    - any resolved address that `_is_disallowed()` flags. ANY disallowed
      record among multiple resolved addresses rejects the whole target.
    """
    try:
        parsed = urlsplit(url)
    except ValueError as exc:
        # urlsplit() raises a bare ValueError for some malformed inputs (e.g.
        # an unterminated IPv6 host literal like "http://[::1/path") -- the
        # module's own "never surface as an unhandled 500" promise applies
        # to parse failures too, not only to NXDOMAIN/resolver errors below.
        raise TargetRejected(
            "Target URL must be an http:// or https:// URL with a hostname."
        ) from exc
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise TargetRejected("Target URL must be an http:// or https:// URL with a hostname.")

    hostname = parsed.hostname
    try:
        resolved = await _resolve_ips(hostname)
    except Exception as exc:
        raise TargetRejected(
            f"Target '{hostname}' could not be resolved."
        ) from exc

    if any(_is_disallowed(addr) for addr in resolved):
        raise TargetRejected(f"Target '{hostname}' resolves to a disallowed address.")
