"""Dashboard package root and the single `llmsec dashboard` availability gate.

The `[dashboard]` optional extra (`fastapi` + `uvicorn`, see `pyproject.toml`)
is never a core dependency. `llmsec scan`/`report`/`list-modules` must keep
working with it entirely absent.

`require_dashboard_extra()` is the single chokepoint the `dashboard` CLI
command calls before doing anything else -- mirrors
`llmsec.attacker.require_deep_extra()`'s structure exactly (D-74 precedent):
a module-name tuple, an install-hint constant, a `RuntimeError` subclass,
a `find_spec`-inside-try/except probe (never a bare `import`, so the probe
stays cheap and side-effect-free), and a never-raising `*_available()`
companion.

This module itself performs ZERO module-scope imports of the dashboard
stack, so `llmsec.dashboard` stays importable with the extra entirely
absent. A future edit must not add one -- any real `import fastapi` /
`import uvicorn` belongs inside `dashboard/app.py`, `dashboard/auth.py`,
`dashboard/routes.py`, or a deferred import inside `cli.py`'s command body,
never here.
"""

from __future__ import annotations

import importlib.util

#: The two importable module names probed by `require_dashboard_extra()`,
#: one per distribution pinned in the `[dashboard]` extra of `pyproject.toml`.
DASHBOARD_EXTRA_MODULES: tuple[str, ...] = ("fastapi", "uvicorn")

#: The exact operator-facing install instruction surfaced in every failure
#: message this gate raises.
DASHBOARD_EXTRA_INSTALL_HINT: str = 'pip install ".[dashboard]"'


class DashboardExtraNotInstalled(RuntimeError):
    """Raised by `require_dashboard_extra()` when `llmsec dashboard` cannot run.

    `cli.py` catches this type and renders it as a clean operator-facing
    message (never a traceback), the same style as `cli.py`'s existing
    `AttackerExtraNotInstalled` handling for `--deep`.
    """


def require_dashboard_extra() -> None:
    """Raise `DashboardExtraNotInstalled` if `llmsec dashboard` cannot run here.

    Probes each of `DASHBOARD_EXTRA_MODULES` in order via
    `importlib.util.find_spec` (never a bare `import`), raising on the first
    module not found. Returns `None` on success -- a caller reaching past
    this function without an exception may safely assume the dashboard stack
    is importable.
    """
    for module_name in DASHBOARD_EXTRA_MODULES:
        try:
            spec = importlib.util.find_spec(module_name)
        except (ModuleNotFoundError, ValueError):
            spec = None
        if spec is None:
            raise DashboardExtraNotInstalled(
                f"The dashboard requires the '{module_name}' module, which is "
                f"not installed. Install it with `{DASHBOARD_EXTRA_INSTALL_HINT}`, "
                f"then rerun `llmsec dashboard`."
            )


def dashboard_extra_available() -> bool:
    """Return whether `llmsec dashboard` can run here. Never raises.

    Equivalent to calling `require_dashboard_extra()` and checking whether it
    raised, but safe to call from any context that must not itself crash.
    """
    try:
        require_dashboard_extra()
    except DashboardExtraNotInstalled:
        return False
    except Exception:  # pragma: no cover - defensive: never raise, ever
        return False
    return True
