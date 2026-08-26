"""Extras-absent behavior (DASH-03/D-03/Pitfall #7) -- must NOT skip itself
when the extra is missing. This file's entire job is covering the
[dashboard]-extra-absent path and it must pass in every environment,
including one where the extra IS installed.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

from typer.testing import CliRunner

from llmsec.dashboard import (
    DASHBOARD_EXTRA_INSTALL_HINT,
    DASHBOARD_EXTRA_MODULES,
    DashboardExtraNotInstalled,
    dashboard_extra_available,
    require_dashboard_extra,
)

CLI_PATH = Path(__file__).resolve().parent.parent.parent / "src" / "llmsec" / "cli.py"


def _fake_find_spec_missing(target: str):
    real_find_spec = importlib.util.find_spec

    def _fake(name, *args, **kwargs):
        if name == target:
            return None
        return real_find_spec(name, *args, **kwargs)

    return _fake


def test_require_dashboard_extra_raises_when_fastapi_missing(monkeypatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec_missing("fastapi"))
    with pytest_raises_with_message("fastapi", DASHBOARD_EXTRA_INSTALL_HINT):
        require_dashboard_extra()


def test_dashboard_extra_available_false_when_fastapi_missing(monkeypatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec_missing("fastapi"))
    assert dashboard_extra_available() is False


def test_require_dashboard_extra_raises_when_uvicorn_missing(monkeypatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec_missing("uvicorn"))
    with pytest_raises_with_message("uvicorn", DASHBOARD_EXTRA_INSTALL_HINT):
        require_dashboard_extra()


def test_dashboard_extra_available_false_when_uvicorn_missing(monkeypatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec_missing("uvicorn"))
    assert dashboard_extra_available() is False


def test_dashboard_extra_available_never_raises_even_if_find_spec_raises(monkeypatch) -> None:
    def _boom(name, *args, **kwargs):
        raise RuntimeError("simulated find_spec failure")

    monkeypatch.setattr(importlib.util, "find_spec", _boom)
    # Must not propagate -- dashboard_extra_available() is documented to
    # never raise, ever (defensive except Exception branch).
    assert dashboard_extra_available() is False


def test_module_names_are_exactly_fastapi_and_uvicorn() -> None:
    assert DASHBOARD_EXTRA_MODULES == ("fastapi", "uvicorn")


class _RaisesContext:
    def __init__(self, *required_substrings: str) -> None:
        self._required = required_substrings

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        assert exc_type is DashboardExtraNotInstalled, f"expected DashboardExtraNotInstalled, got {exc_type}"
        message = str(exc)
        for substring in self._required:
            assert substring in message, f"{substring!r} missing from {message!r}"
        return True


def pytest_raises_with_message(*required_substrings: str) -> _RaisesContext:
    return _RaisesContext(*required_substrings)


def test_dashboard_cmd_exits_1_with_install_hint_and_no_traceback(monkeypatch) -> None:
    import llmsec.cli as cli_module

    def _raise(*args, **kwargs):
        raise DashboardExtraNotInstalled(
            f"The dashboard requires the 'fastapi' module, which is not installed. "
            f"Install it with `{DASHBOARD_EXTRA_INSTALL_HINT}`, then rerun `llmsec dashboard`."
        )

    # Patch at the usage site inside the deferred import -- cli.py imports
    # `require_dashboard_extra` from `llmsec.dashboard` inside the command
    # body, so patching the module it re-imports from at call time is what
    # actually takes effect.
    monkeypatch.setattr("llmsec.dashboard.require_dashboard_extra", _raise)

    runner = CliRunner()
    result = runner.invoke(cli_module.app, ["dashboard"])

    assert result.exit_code == 1
    assert DASHBOARD_EXTRA_INSTALL_HINT in result.output
    assert "Traceback" not in result.output


def test_cli_source_has_no_module_level_import_of_dashboard_stack() -> None:
    """AST check (not a text grep) -- a mention inside a docstring/comment
    must not trip this, and a real top-level import must not slip past it.
    """
    source = CLI_PATH.read_text()
    tree = ast.parse(source)

    forbidden_names = {"fastapi", "uvicorn"}

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_level = alias.name.split(".")[0]
                assert top_level not in forbidden_names, f"module-level `import {alias.name}`"
                assert top_level != "llmsec" or not alias.name.startswith(
                    "llmsec.dashboard"
                ), f"module-level `import {alias.name}`"
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            top_level = module.split(".")[0]
            assert top_level not in forbidden_names, f"module-level `from {module} import ...`"
            assert not module.startswith("llmsec.dashboard"), f"module-level `from {module} import ...`"
