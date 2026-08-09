"""Package data directory for `llmsec.payloads.load_corpus`.

Holds the YAML payload corpora consumed by test modules (e.g.
`prompt_injection.yaml`). This `__init__.py` exists solely so the directory
is an importable package, letting `importlib.resources.files()` address it.
"""
