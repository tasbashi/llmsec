"""Shared root-level pytest fixtures for `tests/`.

Reused across this plan's adapter tests (`test_adapters.py`) and plan 06's
judge tests:

- `mock_target_response`: factory fixture building `TargetResponse`
  instances with sane, overridable defaults.
- `respx_mock`: thin wrapper around `respx`'s mock router, for mocking
  `httpx` calls made by `HttpAppAdapter`.
- `mock_litellm_acompletion`: monkeypatches `litellm.acompletion` with an
  `AsyncMock` returning a configurable canned response object, so tests
  never make live LLM API calls.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Callable
from unittest.mock import AsyncMock

import pytest
import respx

from llmsec.models import TargetResponse


@pytest.fixture
def mock_target_response() -> Callable[..., TargetResponse]:
    """Factory fixture: build a `TargetResponse` with sane defaults.

    Usage: `mock_target_response(raw_text="leaked system prompt")`.
    """

    def _make(
        case_id: str = "case-1",
        raw_text: str = "mock response",
        status_code: int | None = 200,
        latency_ms: float = 1.0,
        tokens_used: int | None = None,
    ) -> TargetResponse:
        return TargetResponse(
            case_id=case_id,
            raw_text=raw_text,
            status_code=status_code,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
        )

    return _make


@pytest.fixture
def respx_mock():
    """Re-export of `respx`'s mock router fixture for `HttpAppAdapter` tests."""
    with respx.mock() as mock_router:
        yield mock_router


def _make_completion_response(content: str = "mock content", total_tokens: int = 42) -> SimpleNamespace:
    """Build a canned object shaped like litellm's `ModelResponse`.

    Exposes `.choices[0].message.content` and `.usage.total_tokens`, the two
    attributes `LLMApiAdapter.send()` reads.
    """
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(total_tokens=total_tokens)
    return SimpleNamespace(choices=[choice], usage=usage)


@pytest.fixture
def mock_litellm_acompletion(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Monkeypatch `litellm.acompletion` with a configurable `AsyncMock`.

    Defaults to returning a canned response shaped like litellm's
    `ModelResponse`. Tests can reconfigure per-case via the returned mock's
    `.return_value` or `.side_effect`.
    """
    mock = AsyncMock(return_value=_make_completion_response())
    monkeypatch.setattr("litellm.acompletion", mock)
    return mock
