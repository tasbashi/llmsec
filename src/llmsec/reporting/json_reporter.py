"""JSON persistence reporter (REPORT-01) — the source of truth `llmsec
report` reads back (CLI-02).

The JSON write IS the persistence layer for Phase 1 — no separate
database. `load_report()` deliberately does NOT wrap
`ScanReport.model_validate_json()` in a try/except: a malformed/corrupted
persisted scan must fail loudly with `pydantic.ValidationError` rather than
silently substituting default values (T-01-11, threat_model prohibition).
"""

from __future__ import annotations

import os
from pathlib import Path

from llmsec.models import ScanReport
from llmsec.reporting.base import BaseReporter

# ASVS V6 / PITFALLS P10-D: scan output may contain leaked system-prompt
# content, so the output directory is created with restrictive
# (owner-only) permissions.
_OUTPUT_DIR_MODE = 0o700


class JsonReporter(BaseReporter):
    """Writes a `ScanReport` as indented JSON to `output_dir/scan_<id>.json`."""

    async def write(self, report: ScanReport, output_dir: Path) -> Path:
        # WR-05 (CR-02): `Path.mkdir(mode=...)` only applies `mode` when the
        # directory is actually created — if `output_dir` already exists
        # (pre-created by an operator, or left looser by a prior run's
        # umask), `exist_ok=True` silently skips tightening its
        # permissions. Explicitly `chmod` afterward so the restrictive
        # mode invariant holds unconditionally, mirroring
        # `MarkdownReporter.write()`'s fix for the same gap.
        output_dir.mkdir(parents=True, exist_ok=True, mode=_OUTPUT_DIR_MODE)
        os.chmod(output_dir, _OUTPUT_DIR_MODE)
        path = output_dir / f"scan_{report.scan_id}.json"
        path.write_text(report.model_dump_json(indent=2))
        return path


def load_report(path: Path) -> ScanReport:
    """Load a previously-written `scan_<id>.json` back into a `ScanReport`.

    Deliberately lets `ScanReport.model_validate_json()`'s `ValidationError`
    propagate uncaught on malformed/corrupted/partial JSON — a broken
    persisted scan must fail loudly, never produce a report with
    fabricated-looking-valid substituted defaults.
    """
    return ScanReport.model_validate_json(path.read_text())
