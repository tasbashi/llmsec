"""Shared, schema-validated YAML payload-corpus loader (D-18/D-19).

`load_corpus` is a framework capability — importable by any test module,
including third-party plugins — not private to `prompt_injection`. It never
raises into the caller: a missing file, a malformed top level, or a
malformed entry all degrade to a logged warning/error plus an empty list or
a shorter-than-expected list, never an exception (T-02-03, T-02-08).
"""

from __future__ import annotations

import importlib.resources
import logging

import yaml
from pydantic import ValidationError

from llmsec.payloads.schema import CORPUS_SCHEMA_VERSION, PayloadEntry

logger = logging.getLogger(__name__)

CORPUS_DIR_PACKAGE = "llmsec.modules.payloads"

__all__ = ["load_corpus", "PayloadEntry", "CORPUS_DIR_PACKAGE"]


def _corpus_path(name: str):
    """Resolve the `importlib.resources` traversable for `{name}.yaml`.

    Factored out as its own seam so tests can monkeypatch corpus resolution
    without reaching into `importlib.resources` internals.
    """
    return importlib.resources.files(CORPUS_DIR_PACKAGE).joinpath(f"{name}.yaml")


def load_corpus(name: str) -> list[PayloadEntry]:
    """Load and schema-validate the `{name}` YAML payload corpus.

    Returns an empty list (never raises) if the file is missing, the top
    level is not a mapping, `entries` is absent, or parsing fails for any
    reason — a malformed corpus must never crash a scan (T-02-08). Entries
    that fail `PayloadEntry` validation are individually skipped and logged
    at WARNING; sibling valid entries are still returned (T-02-09).
    """
    try:
        resource = _corpus_path(name)
        if not resource.is_file():
            logger.error("Corpus %r not found (no file at %s)", name, resource)
            return []
        raw_text = resource.read_text(encoding="utf-8")
        # yaml.safe_load ONLY — never yaml.load/unsafe_load/a custom Loader.
        # Corpus files may arrive from a third-party contribution, a forked
        # repo, or a tampered package install; safe_load never constructs
        # arbitrary Python objects, so a `!!python/object` tag simply raises
        # a constructor error caught below (T-02-03).
        document = yaml.safe_load(raw_text)
    except Exception:
        logger.exception("Failed to read/parse corpus %r", name)
        return []

    if not isinstance(document, dict) or "entries" not in document:
        logger.error(
            "Corpus %r has an invalid top level (expected a mapping with an "
            "`entries` key)",
            name,
        )
        return []

    version = document.get("version")
    if version is not None and version != CORPUS_SCHEMA_VERSION:
        logger.warning(
            "Corpus %r declares version %r, expected %r; continuing anyway",
            name,
            version,
            CORPUS_SCHEMA_VERSION,
        )

    raw_entries = document["entries"]
    if not isinstance(raw_entries, list):
        logger.error("Corpus %r `entries` is not a list; returning []", name)
        return []

    entries: list[PayloadEntry] = []
    for index, raw_entry in enumerate(raw_entries):
        entry_label = None
        if isinstance(raw_entry, dict):
            entry_label = raw_entry.get("id")
        entry_label = entry_label if entry_label is not None else f"index {index}"
        try:
            entries.append(PayloadEntry(**raw_entry))
        except (ValidationError, TypeError) as exc:
            logger.warning(
                "Skipping malformed corpus entry %s in %r: %s", entry_label, name, exc
            )
            continue

    return entries
