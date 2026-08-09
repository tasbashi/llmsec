"""Authorization gate for llmsec scans (D-01/D-02/D-03).

The single chokepoint gating whether live requests to a target are permitted.
Called from both `api.py`'s `run_scan()` (library path, plan 08) and
`cli.py`'s `scan_cmd` (CLI path, plan 09) — never CLI-only.
"""

from __future__ import annotations

import os
import sys

import typer

AUTH_ENV_VAR = "LLMSEC_AUTHORIZED"

DISCLAIMER = (
    "\n⚠️  llmsec is about to send live requests to the configured target.\n"
    "This tool is intended for AUTHORIZED security testing only. Scanning systems\n"
    "without explicit written permission from the owner may violate computer fraud\n"
    "laws (e.g. CFAA) and provider Terms of Service. The authors assume no liability\n"
    "for unauthorized use.\n"
)


class AuthorizationDeclined(Exception):
    """Raised when the operator has not confirmed (or has declined) authorization."""


def confirm_authorization(bypass_flag: bool) -> None:
    """Confirm the operator is authorized to test the configured target.

    D-01: shows a disclaimer and requires interactive y/N confirmation before
    the first live request is sent.
    D-02: a non-interactive terminal is NEVER treated as implicit consent —
    the `sys.stdin.isatty()` check below runs unconditionally BEFORE any
    `typer.confirm()` call, so piped stdin can never reach (or satisfy) the
    prompt. An explicit bypass (`bypass_flag` or `LLMSEC_AUTHORIZED=1`) is
    required for non-interactive/CI use.
    D-03: the disclaimer is a short warning + y/N confirmation, not a
    typed-confirmation ("type the hostname") flow.
    """
    if bypass_flag or os.environ.get(AUTH_ENV_VAR) == "1":
        return  # explicit bypass — logged by caller, not silently accepted

    if not sys.stdin.isatty():
        # D-02: non-interactive terminal is NEVER treated as implicit consent.
        raise AuthorizationDeclined(
            "Refusing to run in a non-interactive session without an explicit bypass. "
            f"Pass --yes-i-am-authorized or set {AUTH_ENV_VAR}=1."
        )

    typer.echo(DISCLAIMER, err=True)
    if not typer.confirm("Do you have explicit authorization to test this target?"):
        raise AuthorizationDeclined("Authorization not confirmed.")
