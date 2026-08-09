"""llmsec: automated vulnerability scanner for LLM-integrated applications.

Covers the OWASP Top 10 for LLMs (prompt injection, system prompt leakage,
PII exfiltration, insecure output handling, and more) via a plugin-based
module system, with dual raw-LLM / HTTP-app target adapters.

Usable both as a CLI (`llmsec scan`) and as a library (`import llmsec`).
"""

from importlib import metadata as _metadata

from llmsec.api import run_scan

try:
    __version__ = _metadata.version("llm-security-tester")
except _metadata.PackageNotFoundError:
    # Running from source without an installed distribution (e.g. a bare
    # checkout never `pip install -e .`'d) — pyproject.toml remains the
    # single source of truth for the real version in every installed case.
    __version__ = "0.0.0.dev0"

__all__ = ["__version__", "run_scan"]
