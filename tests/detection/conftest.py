"""Local fixtures for `tests/detection/` judge tests.

`judge_client` in `llmsec.detection.judge` is a module-level Instructor-wrapped
singleton (`instructor.from_litellm(litellm.acompletion)`) captured at import
time. Patching the underlying `litellm.acompletion` function *after* that
point would not affect the already-bound client, so `mock_judge_client`
monkeypatches `judge_client.chat.completions.create` directly — the correct
boundary for this singleton shape.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from llmsec.detection import judge as judge_module


@pytest.fixture
def mock_judge_client(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Monkeypatch `judge_client.chat.completions.create` with an `AsyncMock`.

    Tests configure the mock's `.return_value` (a `JudgeVerdict`) or
    `.side_effect` (e.g. `InstructorRetryException`) per case, then assert on
    `.call_args.kwargs` to verify message content and explicit parameters.
    """
    mock = AsyncMock()
    monkeypatch.setattr(judge_module.judge_client.chat.completions, "create", mock)
    return mock
