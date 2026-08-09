"""Tests for the D-27/D-28 optional NER detection layer (`pii_ner.py`).

Presidio is NOT a project dependency -- it lives behind the `[pii-ner]`
optional extra (D-27) -- so every test here mocks `presidio_analyzer`'s
`AnalyzerEngine` directly on the `pii_ner` module. This keeps CI green
without installing the extra or downloading the ~746MB `en_core_web_lg`
spaCy model, per the plan's explicit "tests must NOT require the live
extra" acceptance criterion.

Covers every `<behavior>` bullet from 03-05-PLAN.md Task 3: the
forced-unavailable path (SKIPPED_NOT_INSTALLED, distinguishable from an
empty match list per D-28), mocked-engine RAN_MATCH/RAN_NO_MATCH, lazy
engine construction (never at import), MAX_RESPONSE_CHARS truncation, and
the never-raises contract on empty/None/adversarial input.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from llmsec.detection import pii_ner
from llmsec.detection.judge import MAX_RESPONSE_CHARS
from llmsec.detection.pii_ner import NerStatus, find_ner_pii, find_ner_pii_async, ner_available


class _FakeRecognizerResult:
    """Minimal stand-in for `presidio_analyzer`'s `RecognizerResult` --
    only the three attributes `find_ner_pii()` reads."""

    def __init__(self, entity_type: str, start: int, end: int) -> None:
        self.entity_type = entity_type
        self.start = start
        self.end = end


def test_engine_not_constructed_at_import() -> None:
    """The lazily-constructed singleton must be `None` right after import --
    constructing `AnalyzerEngine()` loads the spaCy model, which must not
    happen just because the module was imported (D-27 invariant 2)."""
    assert pii_ner._engine is None


def test_ner_available_reflects_module_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """`ner_available()` is a cheap, always-safe reflection of the
    module-level import-success flag -- never raises."""
    monkeypatch.setattr(pii_ner, "_NER_AVAILABLE", True)
    assert ner_available() is True

    monkeypatch.setattr(pii_ner, "_NER_AVAILABLE", False)
    assert ner_available() is False


def test_find_ner_pii_skipped_when_extra_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SKIPPED_NOT_INSTALLED must be distinguishable from an
    installed-but-empty match list (D-28) -- never collapsed together, and
    returned regardless of what `text` contains since the layer never ran."""
    monkeypatch.setattr(pii_ner, "_NER_AVAILABLE", False)

    status, matches = find_ner_pii("My name is John Smith and I live in Paris.")

    assert status is NerStatus.SKIPPED_NOT_INSTALLED
    assert matches == []


def test_find_ner_pii_ran_match_with_mocked_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a mocked `AnalyzerEngine`, a detected entity yields RAN_MATCH
    carrying `(entity_type, matched_text)` tuples."""
    text = "My name is John Smith."
    mock_engine = MagicMock()
    mock_engine.analyze.return_value = [_FakeRecognizerResult("PERSON", 11, 21)]
    monkeypatch.setattr(pii_ner, "_NER_AVAILABLE", True)
    monkeypatch.setattr(pii_ner, "_engine", None)
    monkeypatch.setattr(pii_ner, "AnalyzerEngine", MagicMock(return_value=mock_engine))

    status, matches = find_ner_pii(text)

    assert status is NerStatus.RAN_MATCH
    assert matches == [("PERSON", "John Smith")]
    mock_engine.analyze.assert_called_once_with(text=text, language="en")


def test_find_ner_pii_ran_no_match_with_mocked_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a mocked `AnalyzerEngine` that finds nothing, RAN_NO_MATCH is
    returned -- distinct from SKIPPED_NOT_INSTALLED even though both carry
    an empty match list (the `status` field is what disambiguates them)."""
    mock_engine = MagicMock()
    mock_engine.analyze.return_value = []
    monkeypatch.setattr(pii_ner, "_NER_AVAILABLE", True)
    monkeypatch.setattr(pii_ner, "_engine", None)
    monkeypatch.setattr(pii_ner, "AnalyzerEngine", MagicMock(return_value=mock_engine))

    status, matches = find_ner_pii("The weather is nice today.")

    assert status is NerStatus.RAN_NO_MATCH
    assert matches == []


def test_engine_constructed_lazily_on_first_use_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """The engine constructor must not be called until the first real
    `find_ner_pii()` call, and must be reused (not rebuilt) thereafter."""
    mock_engine = MagicMock()
    mock_engine.analyze.return_value = []
    mock_constructor = MagicMock(return_value=mock_engine)
    monkeypatch.setattr(pii_ner, "_NER_AVAILABLE", True)
    monkeypatch.setattr(pii_ner, "_engine", None)
    monkeypatch.setattr(pii_ner, "AnalyzerEngine", mock_constructor)

    mock_constructor.assert_not_called()

    find_ner_pii("some text")
    mock_constructor.assert_called_once()

    find_ner_pii("more text")
    mock_constructor.assert_called_once()  # singleton reused, not rebuilt


def test_find_ner_pii_truncates_text_over_max_response_chars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Text longer than MAX_RESPONSE_CHARS is truncated before analysis
    (T-03-NER-DOS bound), reusing the same bound the judge tier uses."""
    long_text = "a" * (MAX_RESPONSE_CHARS + 500)
    mock_engine = MagicMock()
    mock_engine.analyze.return_value = []
    monkeypatch.setattr(pii_ner, "_NER_AVAILABLE", True)
    monkeypatch.setattr(pii_ner, "_engine", None)
    monkeypatch.setattr(pii_ner, "AnalyzerEngine", MagicMock(return_value=mock_engine))

    find_ner_pii(long_text)

    called_text = mock_engine.analyze.call_args.kwargs["text"]
    assert len(called_text) == MAX_RESPONSE_CHARS


@pytest.mark.parametrize("bad_input", [None, "", "   "])
def test_find_ner_pii_empty_or_none_input_never_raises(
    monkeypatch: pytest.MonkeyPatch, bad_input: str | None
) -> None:
    """[EDGE:empty] Empty/whitespace-only/None input returns cleanly (never
    SKIPPED, since the extra IS available here) and never raises."""
    monkeypatch.setattr(pii_ner, "_NER_AVAILABLE", True)

    status, matches = find_ner_pii(bad_input)

    assert status is NerStatus.RAN_NO_MATCH
    assert matches == []


def test_find_ner_pii_never_raises_when_engine_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unexpected engine-internal failure degrades to RAN_NO_MATCH rather
    than propagating -- mirrors `canary.find_canary()`'s never-raises
    exception-swallowing discipline."""
    mock_engine = MagicMock()
    mock_engine.analyze.side_effect = RuntimeError("boom")
    monkeypatch.setattr(pii_ner, "_NER_AVAILABLE", True)
    monkeypatch.setattr(pii_ner, "_engine", None)
    monkeypatch.setattr(pii_ner, "AnalyzerEngine", MagicMock(return_value=mock_engine))

    status, matches = find_ner_pii("some text that triggers a crash")

    assert status is NerStatus.RAN_NO_MATCH
    assert matches == []


def test_find_ner_pii_skipped_when_extra_not_installed_reflects_real_environment() -> None:
    """Sanity check against the REAL module state (no monkeypatching): in
    this test environment the `[pii-ner]` extra is not installed, so the
    module-level flag must already be False and the public API must reflect
    that honestly without any patching."""
    assert ner_available() is False
    status, matches = find_ner_pii("My name is John Smith.")
    assert status is NerStatus.SKIPPED_NOT_INSTALLED
    assert matches == []


# --- WR-03: find_ner_pii_async() offloads to a worker thread -------------


async def test_find_ner_pii_async_returns_same_result_as_sync_when_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`find_ner_pii_async()` must return the identical `(status, matches)`
    contract as `find_ner_pii()` -- it is a pure thread-offloading wrapper,
    never a behavior change."""
    monkeypatch.setattr(pii_ner, "_NER_AVAILABLE", False)

    status, matches = await find_ner_pii_async("My name is John Smith and I live in Paris.")

    assert status is NerStatus.SKIPPED_NOT_INSTALLED
    assert matches == []


async def test_find_ner_pii_async_returns_same_result_as_sync_with_mocked_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "My name is John Smith."
    mock_engine = MagicMock()
    mock_engine.analyze.return_value = [_FakeRecognizerResult("PERSON", 11, 21)]
    monkeypatch.setattr(pii_ner, "_NER_AVAILABLE", True)
    monkeypatch.setattr(pii_ner, "_engine", None)
    monkeypatch.setattr(pii_ner, "AnalyzerEngine", MagicMock(return_value=mock_engine))

    status, matches = await find_ner_pii_async(text)

    assert status is NerStatus.RAN_MATCH
    assert matches == [("PERSON", "John Smith")]


async def test_find_ner_pii_async_runs_the_blocking_call_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WR-03 regression: the synchronous `engine.analyze()` call must run in
    a worker thread, not directly on the event-loop thread that
    `find_ner_pii_async()` itself was awaited from -- otherwise a
    CPU-bound/model-loading NER call would still block every other
    concurrently in-flight case's I/O, exactly the defect this fix closes.
    """
    import threading

    caller_thread = threading.current_thread()
    observed_thread: list[threading.Thread] = []

    def _fake_analyze(*, text: str, language: str):
        observed_thread.append(threading.current_thread())
        return []

    mock_engine = MagicMock()
    mock_engine.analyze.side_effect = _fake_analyze
    monkeypatch.setattr(pii_ner, "_NER_AVAILABLE", True)
    monkeypatch.setattr(pii_ner, "_engine", None)
    monkeypatch.setattr(pii_ner, "AnalyzerEngine", MagicMock(return_value=mock_engine))

    status, matches = await find_ner_pii_async("some text")

    assert status is NerStatus.RAN_NO_MATCH
    assert matches == []
    assert len(observed_thread) == 1
    assert observed_thread[0] is not caller_thread


async def test_get_engine_is_thread_safe_under_concurrent_first_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CR-01 regression: several `find_ner_pii_async()` calls racing on
    first use (as happens under `ScanOrchestrator`'s bounded concurrency,
    see 03-REVIEW.md CR-01) must construct `AnalyzerEngine()` exactly
    ONCE, not once per concurrent caller. Mocks a slow constructor (a
    stand-in for the real, slow `en_core_web_lg` model load) so the
    concurrency window from the review's own reproduction is reliably
    exercised.
    """
    construct_count = {"n": 0}

    class SlowEngine:
        def __init__(self) -> None:
            time.sleep(0.05)  # simulate slow spaCy model load
            construct_count["n"] += 1

        def analyze(self, *, text: str, language: str):
            return []

    monkeypatch.setattr(pii_ner, "_NER_AVAILABLE", True)
    monkeypatch.setattr(pii_ner, "_engine", None)
    monkeypatch.setattr(pii_ner, "AnalyzerEngine", SlowEngine)

    results = await asyncio.gather(*[find_ner_pii_async("some text") for _ in range(8)])

    assert construct_count["n"] == 1, (
        f"AnalyzerEngine() constructed {construct_count['n']} time(s) for 8 "
        "concurrent first-use calls -- expected exactly 1 (CR-01)"
    )
    assert all(status is NerStatus.RAN_NO_MATCH for status, _ in results)
