"""FastAPI app factory for `llmsec dashboard` (D-01/D-02).

Wires the token-gated `/api/*` router and the publicly-served SPA shell.

As of Phase 11 (11-01), the dashboard DOES dispatch scans -- `POST
/api/scans` calls `llmsec.api.run_scan()` via `dashboard/scans.py`. This
does NOT violate D-73: that invariant forbids EDITING
`llmsec.orchestrator` and `llmsec.adapters.llm_api` byte-for-byte, never
importing them from a new caller. Neither module is imported anywhere in
this package -- `dashboard/scans.py` imports only `llmsec.api`, itself a
new caller of the pinned target path, not an edit to it -- and both
`orchestrator.py` and `adapters/llm_api.py` remain digest-pinned and
untouched (verified every phase by
`tests/attacker/test_target_path_unchanged.py`).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from llmsec.dashboard.routes import build_api_router

#: The Vite build output directory -- git-tracked, deliberately not named
#: `dist` (the repo's `.gitignore` blanket-ignores any directory literally
#: named `dist` at any depth; see RESEARCH.md Pitfall #1).
DEFAULT_STATIC_DIR = Path(__file__).parent / "static"


def create_app(output_dir: Path, static_dir: Path | None = None) -> FastAPI:
    """Build the dashboard's `FastAPI` app.

    The API router is registered FIRST, then the SPA frontend mount --
    though per FastAPI's own `app.frontend()` contract, registered path
    operations are always checked before the frontend fallback regardless
    of declaration order, so the two surfaces cannot collide by
    construction (RESEARCH.md Pattern 1).

    `check_dir=False` is deliberate: it lets `create_app()` construct even
    when the frontend has not been built yet, so the backend test suite
    never depends on a Node toolchain.

    No middleware is installed at all -- in particular no CORS middleware.
    The SPA and the API are served from one origin by one process, so
    cross-origin support is unnecessary; `demo-app/backend/main.py` is a
    deliberately-vulnerable scan target, never a security reference
    (RESEARCH.md Anti-Patterns).

    `output_dir` is never mounted, served, or otherwise exposed as static
    files anywhere -- the only code that may open a file under it is a
    handler behind the `/api` guard (ROADMAP success criterion 4).
    """
    app = FastAPI(title="llmsec dashboard")
    app.include_router(build_api_router(output_dir))
    app.frontend("/", directory=static_dir or DEFAULT_STATIC_DIR, check_dir=False)
    return app
