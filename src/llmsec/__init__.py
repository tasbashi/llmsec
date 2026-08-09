"""llmsec: automated vulnerability scanner for LLM-integrated applications.

Covers the OWASP Top 10 for LLMs (prompt injection, system prompt leakage,
PII exfiltration, insecure output handling, and more) via a plugin-based
module system, with dual raw-LLM / HTTP-app target adapters.

Usable both as a CLI (`llmsec scan`) and as a library (`import llmsec`).
"""

from llmsec.api import run_scan

__version__ = "0.1.0"

__all__ = ["__version__", "run_scan"]
