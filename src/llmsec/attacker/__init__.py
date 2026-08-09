"""Attacker-team package root and the single `--deep` availability gate (D-74).

The `[deep]` optional extra (LangChain + DeepAgents on LangGraph, five exact
`==` pins -- see `pyproject.toml`) is never a core dependency. Everything in
Phase 5's attacker team assumes it may be entirely absent, and it must fail
loudly rather than let a scan silently degrade to the static-only path.

`require_deep_extra()` is the single chokepoint every `--deep` entry path
must call before spending anything -- `cli.py` and `api.run_scan()` both
route through it, never through a bare `import langgraph`. It checks the
Python interpreter floor FIRST (before any import probe), then probes each
required module name via `importlib.util.find_spec` -- cheap and
side-effect-free, unlike a real import.

This module itself performs ZERO module-scope imports of the attacker
stack, so `llmsec.attacker` is importable on Python 3.10 (below the
`deepagents` floor) with the extra entirely absent. Do not add one.
"""

from __future__ import annotations

import importlib.util
import sys

#: The minimum Python version the `[deep]` extra supports, driven by
#: `deepagents==0.7.3`'s `Requires-Python: >=3.11,<4.0` -- one minor above
#: this project's own `requires-python = ">=3.10"` (05-RESEARCH.md Pitfall 1).
MIN_DEEP_PYTHON: tuple[int, int] = (3, 11)

#: The five importable module names probed by `require_deep_extra()`, one per
#: distribution pinned in the `[deep]` extra of `pyproject.toml`.
#: `langgraph-checkpoint` and `langgraph-checkpoint-sqlite` are separate
#: distributions but both import under the `langgraph.checkpoint*` namespace.
DEEP_EXTRA_MODULES: tuple[str, ...] = (
    "langchain",
    "langgraph",
    "langgraph.checkpoint",
    "langgraph.checkpoint.sqlite",
    "deepagents",
)

#: The exact operator-facing install instruction surfaced in every failure
#: message this gate raises.
DEEP_EXTRA_INSTALL_HINT: str = 'pip install ".[deep]"'


class AttackerExtraNotInstalled(RuntimeError):
    """Raised by `require_deep_extra()` when `--deep` mode cannot run.

    Covers both failure branches -- an interpreter below `MIN_DEEP_PYTHON`,
    or the `[deep]` extra not installed. `cli.py` catches this type and
    renders it as a clean operator-facing message (never a traceback), per
    the same style as `cli.py`'s existing `typer.echo(..., err=True)` +
    `typer.Exit(code=1)` handling.
    """


def require_deep_extra() -> None:
    """Raise `AttackerExtraNotInstalled` if `--deep` mode cannot run here.

    Checks the Python version floor FIRST, before touching any import --
    an old interpreter must never trigger an import probe of a stack that
    may not even be resolvable there. Only once the version floor is
    satisfied does this probe each of `DEEP_EXTRA_MODULES` in order via
    `importlib.util.find_spec` (never a bare `import`, so the probe stays
    cheap and side-effect-free), raising on the first module not found.

    Returns `None` on success. Never returns anything else and never
    silently degrades -- a caller reaching past this function without an
    exception may safely assume the full attacker stack is importable.
    """
    if sys.version_info[:2] < MIN_DEEP_PYTHON:
        running = "{}.{}".format(sys.version_info[0], sys.version_info[1])
        required = "{}.{}".format(*MIN_DEEP_PYTHON)
        raise AttackerExtraNotInstalled(
            f"--deep mode requires Python >= {required}, but this "
            f"interpreter is {running}. Deep mode is unavailable and the "
            f"scan did NOT fall back to static-only -- rerun the scan "
            f"without --deep, or use a Python {required}+ interpreter with "
            f"the [deep] extra installed ({DEEP_EXTRA_INSTALL_HINT})."
        )

    for module_name in DEEP_EXTRA_MODULES:
        try:
            spec = importlib.util.find_spec(module_name)
        except (ModuleNotFoundError, ValueError):
            spec = None
        if spec is None:
            raise AttackerExtraNotInstalled(
                f"--deep mode requires the '{module_name}' module, which is "
                f"not installed. Deep mode is unavailable and the scan did "
                f"NOT fall back to static-only -- install the attacker "
                f"stack with `{DEEP_EXTRA_INSTALL_HINT}`, then rerun with "
                f"--deep."
            )


def deep_extra_available() -> bool:
    """Return whether `--deep` mode can run here. Never raises.

    Equivalent to calling `require_deep_extra()` and checking whether it
    raised, but safe to call from any context (e.g. CLI help text, a
    `--deep-profile` preflight display) that must not itself crash.
    """
    try:
        require_deep_extra()
    except AttackerExtraNotInstalled:
        return False
    except Exception:  # pragma: no cover - defensive: never raise, ever
        return False
    return True
