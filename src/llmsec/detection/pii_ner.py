"""Optional NER (Named Entity Recognition) detection layer (D-27/D-28).

The FEATURES.md §5.3.4 tier-3 detector for unstructured PII (names,
addresses, organizations) that the D-29 regex/Luhn taxonomy cannot
structurally match. Ships behind the `[pii-ner]` optional install extra
(Presidio + spaCy), NEVER as a core dependency (D-27) -- the regex/canary/
judge tiers all work without this module's dependencies installed.

This is the first lazy-import optional-dependency pattern in the codebase
(RESEARCH.md/PATTERNS.md: "no precedent yet"). Two invariants this file
must uphold:

1. **Honest degradation (D-28).** When the extra is not installed,
   `find_ner_pii()` returns the distinguishable `NerStatus.SKIPPED_NOT_INSTALLED`
   state -- NEVER an empty "ran and found nothing" result. A caller (and,
   downstream, a report reader) must always be able to tell "this layer did
   not run" apart from "this layer ran and found no unstructured PII."
   Conflating the two would let a scan without the extra installed silently
   present itself as a clean, fully-covered result (the exact failure mode
   D-28 exists to prevent).
2. **Lazy engine construction.** `AnalyzerEngine()` loads spaCy's
   `en_core_web_lg` model into memory -- a real cost -- so it must be
   constructed on first real use only, never at import time, even when the
   extra IS installed. A scan that never calls `find_ner_pii()` must never
   pay that cost.

Never-raises contract (mirrors `canary.py`'s `find_canary()` discipline):
`find_ner_pii()` never raises on falsy input, a missing extra, or an
adversarial/malformed response -- any unexpected engine failure degrades to
`NerStatus.RAN_NO_MATCH` rather than crashing the scan.

Text is truncated to `judge.MAX_RESPONSE_CHARS` (the same bound the judge
tier uses) before analysis, so a very large response cannot drive unbounded
NER latency/memory (threat T5-NER / T-03-NER-DOS).
"""

from __future__ import annotations

import asyncio
import logging
import threading
from enum import Enum

from llmsec.detection.judge import MAX_RESPONSE_CHARS

logger = logging.getLogger(__name__)

try:
    # Heavy import: presidio_analyzer pulls in spaCy (and, once the model is
    # downloaded separately, en_core_web_lg). Attempting this import is what
    # `ner_available()` reports on -- never re-attempted per call.
    from presidio_analyzer import AnalyzerEngine

    _NER_AVAILABLE = True
except ImportError:
    AnalyzerEngine = None  # type: ignore[assignment,misc]
    _NER_AVAILABLE = False

# Lazily-constructed singleton. Constructing AnalyzerEngine() loads the
# spaCy model into memory, so this must stay None until find_ner_pii()'s
# first real (non-empty) call -- never at import time, even when the extra
# IS installed (module docstring invariant 2).
_engine = None

# CR-01: guards `_engine`'s construction. `find_ner_pii_async()` offloads
# `find_ner_pii()` (and therefore `_get_engine()`) to a real OS thread via
# `asyncio.to_thread()` (WR-03), so multiple `ScanOrchestrator`-bounded
# concurrent tasks can enter `_get_engine()` on genuinely different
# threads at once -- a plain `threading.Lock()` is required here (not
# `asyncio.Lock`, which only guards concurrent coroutines on a single
# thread and provides no cross-thread mutual exclusion for this
# double-checked-locking pattern).
_engine_lock = threading.Lock()


class NerStatus(str, Enum):
    """Three-state result distinguishing "layer not installed" from "layer
    ran, found nothing" (D-28) -- the two must never be conflated.
    """

    RAN_MATCH = "ran_match"
    RAN_NO_MATCH = "ran_no_match"
    SKIPPED_NOT_INSTALLED = "skipped_not_installed"


def ner_available() -> bool:
    """Cheap, always-safe check every call site can use before attempting
    NER. Never raises; reflects only whether `presidio_analyzer` imported
    successfully at module load."""
    return _NER_AVAILABLE


def _get_engine() -> "AnalyzerEngine":  # type: ignore[valid-type]
    """Return the lazily-constructed engine singleton, building it on first
    call only (module docstring invariant 2).

    CR-01: double-checked locking guarded by `_engine_lock`. Before
    `find_ner_pii_async()` (WR-03) offloaded calls to real OS threads via
    `asyncio.to_thread()`, this check-then-set was safe by construction --
    only ever entered from the single-threaded event loop. Now several
    worker threads can call this concurrently (bounded by
    `config.max_concurrency`), so an unsynchronized check-then-set let
    each one pass `if _engine is None:` before any finished constructing
    `AnalyzerEngine()`, building up to `max_concurrency` full spaCy models
    simultaneously. The outer unlocked check keeps the common
    already-constructed case lock-free; the inner re-check after
    acquiring the lock ensures only the first thread to acquire it
    actually constructs the engine.
    """
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:  # re-check inside the lock
                _engine = AnalyzerEngine()
    return _engine


def find_ner_pii(text: str | None) -> tuple[NerStatus, list[tuple[str, str]]]:
    """Run the optional NER layer against `text`, honestly reporting when it
    could not run at all.

    Returns `(status, matches)` where `matches` is a list of
    `(entity_type, matched_text)` tuples. `status` is:

    - `SKIPPED_NOT_INSTALLED` if the `[pii-ner]` extra is not installed --
      returned regardless of `text`, since the layer did not run at all
      (D-28: never conflate with an installed-but-empty result).
    - `RAN_NO_MATCH` if the extra is installed but no entities were found,
      including for empty/whitespace-only/`None` input (cheap early return,
      no engine construction needed) and for any unexpected engine failure
      (never-raises contract, mirrors `canary.find_canary()`).
    - `RAN_MATCH` if the extra is installed and at least one entity was
      detected.

    Text longer than `judge.MAX_RESPONSE_CHARS` is truncated before
    analysis (T-03-NER-DOS bound). Never raises.
    """
    if not _NER_AVAILABLE:
        return (NerStatus.SKIPPED_NOT_INSTALLED, [])

    if not text or not text.strip():
        return (NerStatus.RAN_NO_MATCH, [])

    truncated = text[:MAX_RESPONSE_CHARS]
    if len(text) > MAX_RESPONSE_CHARS:
        logger.info(
            "NER input truncated from %d to %d chars", len(text), MAX_RESPONSE_CHARS
        )

    try:
        engine = _get_engine()
        results = engine.analyze(text=truncated, language="en")
    except Exception:
        # Never-raises contract: an adversarial/malformed response or an
        # engine-internal failure must degrade honestly, not crash the scan.
        logger.warning("NER analysis failed; degrading to RAN_NO_MATCH", exc_info=True)
        return (NerStatus.RAN_NO_MATCH, [])

    matches = [(result.entity_type, truncated[result.start : result.end]) for result in results]
    if not matches:
        return (NerStatus.RAN_NO_MATCH, [])
    return (NerStatus.RAN_MATCH, matches)


async def find_ner_pii_async(text: str | None) -> tuple[NerStatus, list[tuple[str, str]]]:
    """WR-03: async wrapper offloading `find_ner_pii()` — including the
    lazy `AnalyzerEngine()` construction and the synchronous, CPU-bound
    `engine.analyze()` call — to a worker thread via `asyncio.to_thread()`.

    `find_ner_pii()` itself stays a plain synchronous function (unchanged)
    so non-async callers/tests are unaffected. Every `async def` call site
    (e.g. `pii_exfiltration.py`'s `_classify_pii_tier()`, itself called
    from `PiiExfiltrationModule.evaluate()`) must use this wrapper instead
    of calling `find_ner_pii()` directly: `evaluate()` runs as one of
    `ScanOrchestrator`'s `asyncio.Semaphore`-bounded concurrent tasks, and
    a first-call NER invocation (which pays the full spaCy model-load
    cost) run directly on the event loop would block every other
    in-flight case's I/O for its duration, silently violating the
    orchestrator's bounded-concurrency contract. Same return contract and
    never-raises guarantee as `find_ner_pii()` (this wrapper does not add
    or change any behavior beyond thread offloading).
    """
    return await asyncio.to_thread(find_ner_pii, text)
