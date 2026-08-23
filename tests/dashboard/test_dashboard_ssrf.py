"""SSRF destination gate tests (Task 3, 11-01-PLAN.md, DASH-07).

Unit tests call `validate_target_not_internal` directly with `_resolve_ips`
monkeypatched to controlled literals -- no `TestClient` needed except for
the `raw_llm`-is-structurally-exempt case.
"""

from __future__ import annotations

from ipaddress import ip_address

import pytest

pytest.importorskip("fastapi", reason='dashboard suite requires pip install -e ".[dashboard,dev]"')

from llmsec.dashboard.ssrf_guard import (
    TargetRejected,
    _is_disallowed,
    validate_target_not_internal,
)

_DISALLOWED_LITERALS = [
    "127.0.0.1",
    "::1",
    "169.254.169.254",
    "fe80::1",
    "10.0.0.1",
    "172.16.0.1",
    "192.168.1.1",
    "fc00::1",
    "0.0.0.0",
    "224.0.0.1",
    "::ffff:127.0.0.1",
    "::ffff:169.254.169.254",
    # 11-REVIEW.md CR-01: RFC 6598 Carrier-Grade-NAT / Shared Address Space
    # -- satisfies none of is_private/is_loopback/is_link_local/
    # is_reserved/is_multicast/is_unspecified, yet is not globally
    # routable. 100.100.100.200 is Alibaba Cloud's ECS metadata endpoint,
    # the functional equivalent of 169.254.169.254.
    "100.64.0.1",
    "100.100.100.200",
]


@pytest.mark.parametrize("literal", _DISALLOWED_LITERALS)
def test_is_disallowed_rejects_every_matrix_entry(literal: str) -> None:
    assert _is_disallowed(ip_address(literal)) is True


def test_is_disallowed_accepts_a_public_address() -> None:
    assert _is_disallowed(ip_address("93.184.216.34")) is False


async def _validate_with_resolved(monkeypatch: pytest.MonkeyPatch, url: str, literals: list[str]):
    from llmsec.dashboard import ssrf_guard

    async def _fake_resolve(hostname: str):
        return [ip_address(literal) for literal in literals]

    monkeypatch.setattr(ssrf_guard, "_resolve_ips", _fake_resolve)
    await validate_target_not_internal(url)


@pytest.mark.parametrize("literal", _DISALLOWED_LITERALS)
async def test_validate_target_not_internal_rejects_each_disallowed_literal(
    monkeypatch: pytest.MonkeyPatch, literal: str
) -> None:
    with pytest.raises(TargetRejected) as exc_info:
        await _validate_with_resolved(monkeypatch, "http://internal.example/chat", [literal])
    assert "internal.example" in str(exc_info.value)
    assert "http://internal.example/chat" not in str(exc_info.value)


async def test_validate_target_not_internal_accepts_a_public_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Must not raise.
    await _validate_with_resolved(monkeypatch, "http://public.example/chat", ["93.184.216.34"])


async def test_validate_target_not_internal_rejects_if_any_resolved_record_is_disallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(TargetRejected):
        await _validate_with_resolved(
            monkeypatch, "http://mixed.example/chat", ["93.184.216.34", "127.0.0.1"]
        )


async def test_validate_target_not_internal_rejects_non_http_scheme() -> None:
    with pytest.raises(TargetRejected) as exc_info:
        await validate_target_not_internal("ftp://example.test/chat")
    assert "example.test" not in str(exc_info.value)
    assert "ftp://example.test/chat" not in str(exc_info.value)


async def test_validate_target_not_internal_rejects_hostname_less_url() -> None:
    with pytest.raises(TargetRejected) as exc_info:
        await validate_target_not_internal("http:///no-host-path")
    assert "http:///no-host-path" not in str(exc_info.value)


async def test_validate_target_not_internal_rejects_malformed_url_as_400_not_500() -> None:
    """11-REVIEW.md CR-02: `urlsplit()` raises a bare `ValueError` for some
    malformed inputs (an unterminated IPv6 host literal) -- this must
    surface as the documented `TargetRejected` (400-shaped), never an
    unhandled exception that would propagate to the client as a 500.
    """
    with pytest.raises(TargetRejected) as exc_info:
        await validate_target_not_internal("http://[::1/path")
    assert "http://[::1/path" not in str(exc_info.value)


async def test_validate_target_not_internal_rejects_unresolvable_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llmsec.dashboard import ssrf_guard

    async def _raising_resolve(hostname: str):
        raise OSError("simulated NXDOMAIN")

    monkeypatch.setattr(ssrf_guard, "_resolve_ips", _raising_resolve)

    with pytest.raises(TargetRejected):
        await validate_target_not_internal("http://does-not-resolve.example/chat")


def test_raw_llm_target_skips_ssrf_check(
    scans_client, auth_headers, controllable_run_scan, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `raw_llm` target must never reach `_resolve_ips` -- the seam is
    structurally unreachable for it (D-08), proven here by making the seam
    raise and confirming the request still succeeds.
    """
    from llmsec.dashboard import ssrf_guard

    async def _explode(hostname: str):
        raise AssertionError("_resolve_ips must never be called for a raw_llm target")

    monkeypatch.setattr(ssrf_guard, "_resolve_ips", _explode)

    body = {
        "target": {
            "type": "raw_llm",
            "model": "openai/gpt-4o-mini",
            "api_key_env": "OPENAI_API_KEY",
        },
        "authorization_attested": True,
    }
    res = scans_client.post("/api/scans", json=body, headers=auth_headers)
    assert res.status_code == 202
