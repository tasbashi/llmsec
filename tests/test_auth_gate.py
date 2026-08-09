"""Tests for llmsec.auth_gate — the authorization gate (D-01/D-02/D-03)."""

from unittest.mock import MagicMock

import pytest

from llmsec.auth_gate import AUTH_ENV_VAR, AuthorizationDeclined, confirm_authorization


def test_bypass_flag_returns_immediately(monkeypatch):
    # Should not even consult isatty or typer.confirm.
    monkeypatch.delenv(AUTH_ENV_VAR, raising=False)
    isatty_mock = MagicMock()
    monkeypatch.setattr("sys.stdin.isatty", isatty_mock)
    confirm_mock = MagicMock()
    monkeypatch.setattr("typer.confirm", confirm_mock)

    confirm_authorization(bypass_flag=True)

    isatty_mock.assert_not_called()
    confirm_mock.assert_not_called()


def test_env_var_bypass_returns_immediately(monkeypatch):
    monkeypatch.setenv(AUTH_ENV_VAR, "1")
    confirm_mock = MagicMock()
    monkeypatch.setattr("typer.confirm", confirm_mock)

    confirm_authorization(bypass_flag=False)

    confirm_mock.assert_not_called()


def test_non_interactive_no_bypass_raises_without_calling_confirm(monkeypatch):
    monkeypatch.delenv(AUTH_ENV_VAR, raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    confirm_mock = MagicMock()
    monkeypatch.setattr("typer.confirm", confirm_mock)

    with pytest.raises(AuthorizationDeclined):
        confirm_authorization(bypass_flag=False)

    confirm_mock.assert_not_called()


def test_interactive_confirm_false_raises(monkeypatch):
    monkeypatch.delenv(AUTH_ENV_VAR, raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("typer.confirm", lambda *a, **kw: False)

    with pytest.raises(AuthorizationDeclined):
        confirm_authorization(bypass_flag=False)


def test_interactive_confirm_true_returns(monkeypatch):
    monkeypatch.delenv(AUTH_ENV_VAR, raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("typer.confirm", lambda *a, **kw: True)

    confirm_authorization(bypass_flag=False)  # should not raise
