"""Bearer-token guard for the dashboard's `/api/*` surface (DASH-03).

`DASHBOARD_TOKEN_ENV_VAR` mirrors `llmsec.auth_gate.AUTH_ENV_VAR`'s naming
convention -- the shared secret is read from the environment by NAME only,
never stored in `llmsec.config.yaml` or any other config surface (project
invariant D-08).

`verify_bearer_token()` is applied once, at the `/api` router level (see
`routes.py`), never per-endpoint -- so an endpoint added later cannot be
born ungated.
"""

from __future__ import annotations

import os
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

#: Naming mirrors `llmsec.auth_gate.AUTH_ENV_VAR = "LLMSEC_AUTHORIZED"`.
DASHBOARD_TOKEN_ENV_VAR = "LLMSEC_DASHBOARD_TOKEN"

# auto_error=False: this function, not FastAPI's default HTTPBearer
# behavior, decides the status code and copy for a missing/malformed
# Authorization header.
_bearer_scheme = HTTPBearer(auto_error=False)


def verify_bearer_token(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> None:
    """FastAPI dependency gating every `/api/*` route.

    Order matters and is deliberate:

    1. Read `LLMSEC_DASHBOARD_TOKEN` from the environment fresh on every
       call (never cached at import time, so an operator can rotate the
       secret with a process restart and no code change) with NO default
       argument to `os.environ.get`.
    2. If that value is falsy, raise 503. An unset (or empty-string) env
       var is a misconfiguration, never an "auth disabled" mode -- a
       default of `""` would let an empty submitted credential compare
       equal to it, silently granting access to everyone (RESEARCH.md
       Pitfall #2). This check MUST run before any comparison.
    3. Only once a real expected token exists do we check whether the
       caller supplied one at all.
    4. Compare with `secrets.compare_digest` over UTF-8-encoded bytes on
       both sides -- never `==` (timing-attack surface) and never bare
       `str` operands (`compare_digest` raises `TypeError` on `str`
       arguments containing non-ASCII characters, and a token pasted from
       a password manager can contain any character).

    Neither exception's `detail` may contain the submitted or the expected
    credential (prohibition P-02) -- both are fixed strings naming only the
    env var NAME.
    """
    expected = os.environ.get(DASHBOARD_TOKEN_ENV_VAR)
    if not expected:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Dashboard is not configured: {DASHBOARD_TOKEN_ENV_VAR} is not set.",
        )

    if creds is None or not creds.credentials:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token.",
        )

    if not secrets.compare_digest(
        creds.credentials.encode("utf-8"), expected.encode("utf-8")
    ):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token.",
        )
