"""Tests for the single `--deep` availability gate (D-74).

Covers every bullet in 05-01-PLAN.md's `<behavior>` block. Monkeypatches
`sys.version_info` for the interpreter-floor branch and
`importlib.util.find_spec` for the missing-module branch, so this test suite
passes identically whether or not the `[deep]` extra is actually installed
in the environment running it.
"""

from __future__ import annotations

import importlib
import sys

import pytest

import llmsec.attacker as attacker


def _fake_find_spec_all_present(name: str, *args, **kwargs):
    """Stand-in for importlib.util.find_spec that reports every module found."""
    return object()  # any truthy, non-None sentinel is sufficient


def _fake_find_spec_missing(missing_name: str):
    """Build a find_spec stand-in that reports exactly one module missing."""

    def _inner(name: str, *args, **kwargs):
        if name == missing_name:
            return None
        return object()

    return _inner


class TestDeepExtraAvailableHappyPath:
    def test_require_deep_extra_returns_none_when_all_present(self, monkeypatch):
        monkeypatch.setattr(sys, "version_info", (3, 11, 0, "final", 0))
        monkeypatch.setattr(
            importlib.util, "find_spec", _fake_find_spec_all_present
        )
        assert attacker.require_deep_extra() is None

    def test_deep_extra_available_true_when_all_present(self, monkeypatch):
        monkeypatch.setattr(sys, "version_info", (3, 12, 1, "final", 0))
        monkeypatch.setattr(
            importlib.util, "find_spec", _fake_find_spec_all_present
        )
        assert attacker.deep_extra_available() is True


class TestPythonVersionFloor:
    def test_require_deep_extra_raises_below_min_python(self, monkeypatch):
        monkeypatch.setattr(sys, "version_info", (3, 10, 4, "final", 0))
        # Sentinel: if the gate incorrectly proceeds to probe imports before
        # checking the version floor, this raises AssertionError instead of
        # a plain "module missing" failure -- proves import-avoidance too.
        def _should_not_be_called(name: str, *args, **kwargs):
            raise AssertionError(
                "find_spec must not be called when Python is below MIN_DEEP_PYTHON"
            )

        monkeypatch.setattr(importlib.util, "find_spec", _should_not_be_called)

        with pytest.raises(attacker.AttackerExtraNotInstalled) as excinfo:
            attacker.require_deep_extra()

        message = str(excinfo.value)
        assert "3.11" in message
        assert "3.10" in message

    def test_deep_extra_available_false_below_min_python(self, monkeypatch):
        monkeypatch.setattr(sys, "version_info", (3, 9, 0, "final", 0))
        assert attacker.deep_extra_available() is False


class TestMissingModule:
    def test_require_deep_extra_raises_on_missing_module(self, monkeypatch):
        monkeypatch.setattr(sys, "version_info", (3, 11, 5, "final", 0))
        missing = "deepagents"
        monkeypatch.setattr(
            importlib.util, "find_spec", _fake_find_spec_missing(missing)
        )

        with pytest.raises(attacker.AttackerExtraNotInstalled) as excinfo:
            attacker.require_deep_extra()

        message = str(excinfo.value)
        assert attacker.DEEP_EXTRA_INSTALL_HINT in message
        assert missing in message

    def test_deep_extra_available_false_on_missing_module(self, monkeypatch):
        monkeypatch.setattr(sys, "version_info", (3, 12, 0, "final", 0))
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            _fake_find_spec_missing("langgraph.checkpoint.sqlite"),
        )
        assert attacker.deep_extra_available() is False


class TestNeverRaisesContract:
    def test_deep_extra_available_never_raises_on_weird_find_spec_error(
        self, monkeypatch
    ):
        monkeypatch.setattr(sys, "version_info", (3, 11, 0, "final", 0))

        def _raises_value_error(name: str, *args, **kwargs):
            raise ValueError("malformed module name")

        monkeypatch.setattr(importlib.util, "find_spec", _raises_value_error)

        # Must not raise -- deep_extra_available() degrades to False.
        assert attacker.deep_extra_available() is False


class TestNoModuleScopeImport:
    def test_llmsec_attacker_import_does_not_import_attacker_stack(self):
        # This test's own successful collection/import already proves
        # llmsec.attacker imports cleanly without the extra installed (in
        # environments where it truly isn't). Additionally assert none of
        # the five attacker-stack top-level packages were pulled into
        # sys.modules purely as a side effect of `import llmsec.attacker`.
        for module_name in ("langchain", "langgraph", "deepagents"):
            # Absence is not asserted (another test module may have imported
            # them already in-process); instead assert llmsec.attacker itself
            # carries no reference obtained via a module-scope import.
            assert module_name not in dir(attacker)

    def test_module_constants_present(self):
        assert attacker.MIN_DEEP_PYTHON == (3, 11)
        assert attacker.DEEP_EXTRA_INSTALL_HINT == 'pip install ".[deep]"'
        assert isinstance(attacker.DEEP_EXTRA_MODULES, tuple)
        assert set(attacker.DEEP_EXTRA_MODULES) == {
            "langchain",
            "langgraph",
            "langgraph.checkpoint",
            "langgraph.checkpoint.sqlite",
            "deepagents",
        }


class TestMessagesStateNoSilentFallback:
    def test_version_floor_message_states_no_fallback(self, monkeypatch):
        monkeypatch.setattr(sys, "version_info", (3, 10, 0, "final", 0))
        monkeypatch.setattr(
            importlib.util, "find_spec", _fake_find_spec_all_present
        )
        with pytest.raises(attacker.AttackerExtraNotInstalled) as excinfo:
            attacker.require_deep_extra()
        message = str(excinfo.value).lower()
        assert "static" in message or "fall back" in message or "fallback" in message

    def test_missing_module_message_states_no_fallback(self, monkeypatch):
        monkeypatch.setattr(sys, "version_info", (3, 11, 0, "final", 0))
        monkeypatch.setattr(
            importlib.util, "find_spec", _fake_find_spec_missing("langchain")
        )
        with pytest.raises(attacker.AttackerExtraNotInstalled) as excinfo:
            attacker.require_deep_extra()
        message = str(excinfo.value).lower()
        assert "static" in message or "fall back" in message or "fallback" in message
