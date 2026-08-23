"""Dashboard scan dispatch (DASH-01/DASH-02, Phase 11).

This module is the FIRST in the `dashboard` package to import `llmsec.api`
-- it is a new *caller* of the pinned target path (`api.run_scan()`), and
this is never an edit to `llmsec.orchestrator` or `llmsec.adapters.llm_api`,
which remain byte-for-byte unchanged (D-73). Only `api.py` -- explicitly not
on the D-73-pinned list -- is the correct touch point for a new caller.

`_DashboardScanConfig` builds a `ScanConfig` from ONLY the HTTP request
body's fields plus two explicit init-supplied values (`output_dir`,
`attacker=None`) -- it never lets `ScanConfig`'s inherited `BaseSettings`
sources read an ambient `llmsec.config.yaml` or process environment. This
stops a config sourced from the dashboard's own filesystem cwd (whatever it
happens to be) from silently re-enabling deep mode (D-02) or writing the
report somewhere the dashboard's own reports endpoint can never find it
(RESEARCH.md Pitfall 1, empirically reproduced).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from llmsec.api import run_scan
from llmsec.config import ScanConfig
from llmsec.models import ScanReport

logger = logging.getLogger(__name__)

#: The only phase values a `ScanState.phase` may ever hold. Declared once,
#: here, so the route's response model and every test read from this one
#: source rather than a restated literal list.
SCAN_PHASES: tuple[str, ...] = ("queued", "running", "scoring", "complete", "failed")


@dataclass
class ScanState:
    """The dashboard's own in-memory record of one dispatched scan's progress."""

    scan_id: str
    phase: str
    error: str | None = None


class ScanStateStore:
    """An in-memory `scan_id -> ScanState` store.

    Deliberately carries NO `asyncio.Lock`: every mutation
    (`set_phase`/`set_failed`) is a single, non-`await`-ing dict write, and
    every read (`get`) is a single dict lookup -- no `await` ever occurs
    between reading and writing any one entry, so a lock would add
    complexity with no correctness benefit (RESEARCH.md Alternatives
    Considered). Python's GIL (or, on free-threaded builds, the absence of
    any `await` point inside these methods) already makes each of these
    calls atomic with respect to the event loop.
    """

    def __init__(self) -> None:
        self._states: dict[str, ScanState] = {}

    def set_phase(self, scan_id: str, phase: str) -> None:
        self._states[scan_id] = ScanState(scan_id=scan_id, phase=phase)

    def set_failed(self, scan_id: str, error: str) -> None:
        self._states[scan_id] = ScanState(scan_id=scan_id, phase="failed", error=error)

    def get(self, scan_id: str) -> ScanState | None:
        return self._states.get(scan_id)


class _DashboardScanConfig(ScanConfig):
    """A `ScanConfig` subclass whose ONLY settings source is init-supplied values.

    `settings_customise_sources` normally layers CLI overrides > YAML file >
    process env (`ScanConfig`'s own docstring). This subclass drops the YAML
    and environment sources entirely, keeping only `init_settings` -- the
    HTTP request body (via `build_dashboard_scan_config` below) is the ONLY
    source of truth for a dashboard-originated config. This is stricter than
    `load_config()`'s three-source tuple on purpose: an ambient
    `llmsec.config.yaml` sitting in the server process's cwd would otherwise
    silently supply `attacker` (re-enabling deep mode against D-02, since
    the browser form never collected a budget for it) and `output_dir`
    (writing the report somewhere `GET /api/reports/{scan_id}` can never
    find it) -- verified empirically in 11-RESEARCH.md Pitfall 1.
    """

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (init_settings,)


def build_dashboard_scan_config(body: Any, output_dir: str) -> _DashboardScanConfig:
    """Build a `_DashboardScanConfig` from a `ScanRequest`-shaped `body`.

    `output_dir` and `attacker=None` are passed as explicit init-supplied
    values -- both are ALWAYS present, never absent-and-therefore-inherited
    -- so `_DashboardScanConfig.settings_customise_sources`'s init-only
    source has something concrete to return regardless of what the request
    body did or didn't set. `output_dir` must be the router's own closed-
    over report directory; a request body is never permitted to supply its
    own `output_dir` or `attacker` value.
    """
    return _DashboardScanConfig(
        target=body.target,
        enabled_modules=body.enabled_modules,
        judge_model=body.judge_model,
        judge_api_key_env=body.judge_api_key_env,
        known_system_prompt=body.known_system_prompt,
        max_concurrency=body.max_concurrency,
        output_dir=output_dir,
        attacker=None,
    )


#: Strong-reference holder for scheduled background scan tasks. `asyncio`
#: keeps only a WEAK reference to a task created via `asyncio.create_task`
#: (per the stdlib's own documented caveat) -- without holding a strong
#: reference somewhere, the task object can be garbage-collected mid-scan,
#: silently aborting it. Each task removes itself via `add_done_callback`
#: once it finishes (success or failure), so this set never grows unbounded.
_background_tasks: set[asyncio.Task] = set()


def dispatch_scan(scan_id: str, config: ScanConfig, store: ScanStateStore) -> None:
    """Schedule `scan_id`'s scan as a background `asyncio.Task` and return immediately.

    Deliberately uses `asyncio.create_task`, never FastAPI's `BackgroundTasks`:
    Starlette awaits an async background task as part of the SAME ASGI call
    that sends the response, which would block the POST (and `TestClient`)
    for the scan's full duration and make intermediate phase transitions
    unobservable through polling (11-RESEARCH.md Pitfall 3).
    """
    store.set_phase(scan_id, "queued")

    def _on_phase_change(phase: str) -> None:
        store.set_phase(scan_id, phase)

    async def _run() -> ScanReport | None:
        # `report` is assigned rather than returned directly from inside the
        # `try` -- a `return` statement inside `try` exits the function
        # immediately and SKIPS the `else` clause entirely, which would
        # silently prevent `store.set_phase(scan_id, "complete")` from ever
        # running on the success path.
        try:
            report = await run_scan(
                config,
                bypass_flag=True,
                scan_id=scan_id,
                on_phase_change=_on_phase_change,
            )
        except Exception as exc:
            # T-01-18's containment discipline, one level up: an un-awaited
            # exception escaping into asyncio's default "exception was
            # never retrieved" handler is exactly the silent-failure mode
            # this except clause exists to prevent. Never re-raise here.
            logger.exception("Dashboard-dispatched scan %s failed", scan_id)
            store.set_failed(scan_id, str(exc))
            return None
        else:
            store.set_phase(scan_id, "complete")
            return report

    task = asyncio.create_task(_run())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
