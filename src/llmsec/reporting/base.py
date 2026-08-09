"""Shared reporter contract for `llmsec.reporting`.

Every reporter (JSON, Markdown, and any future community-contributed
reporter) implements the same `write(report, output_dir) -> Path` shape so
`api.py`'s `run_scan()` (plan 08) and `cli.py`'s `report_cmd` (plan 09) can
treat reporters interchangeably.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from llmsec.models import ScanReport


class BaseReporter(ABC):
    """Abstract base class every concrete reporter implements."""

    @abstractmethod
    async def write(self, report: ScanReport, output_dir: Path) -> Path:
        """Write `report` under `output_dir` and return the written file's path."""
        raise NotImplementedError
