"""Shared payload-corpus schema (D-18/D-19).

`PayloadEntry` is the validated shape every YAML corpus entry must satisfy.
This module has no I/O — `llmsec.payloads.load_corpus` owns file access and
constructs `PayloadEntry` instances from parsed YAML mappings.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

CORPUS_SCHEMA_VERSION = 1


class TechniqueFamily(str, Enum):
    """The five payload technique families (D-26) — no others.

    Each family maps to exactly one remediation theme, keeping the corpus's
    per-entry `remediation` field grounded in a fixed, reportable taxonomy.
    """

    INSTRUCTION_OVERRIDE = "instruction_override"
    PERSONA_JAILBREAK = "persona_jailbreak"
    ENCODING_OBFUSCATION = "encoding_obfuscation"
    MULTI_TURN_ESCALATION = "multi_turn_escalation"
    INDIRECT_DATA_AS_INSTRUCTION = "indirect_data_as_instruction"


class PiiAttackVector(str, Enum):
    """FEATURES.md §5.3.2 PII attack-vector taxonomy (8 members, D-37).

    Disjoint from `TechniqueFamily` — never merged into it. Kept as its own
    closed enum so each module's family set stays independently
    reportable/remediation-mappable per D-26's per-family precedent. Both
    `TechniqueFamily` and `PiiAttackVector` are `str, Enum` subclasses, so
    `PayloadEntry.technique_family`'s widened `Union` type below preserves
    value-based `==` comparisons for existing `TechniqueFamily` consumers
    (e.g. `prompt_injection.py`'s `entry.technique_family ==
    TechniqueFamily.INDIRECT_DATA_AS_INSTRUCTION` checks) unchanged.
    """

    TRAINING_DATA_EXTRACTION = "training_data_extraction"
    CONTEXT_REPLAY = "context_replay"
    CREDENTIAL_PROBING = "credential_probing"
    CANARY_TRIGGERING = "canary_triggering"
    MEMBERSHIP_INFERENCE = "membership_inference"
    RAG_PII_EXTRACTION = "rag_pii_extraction"
    URL_EXFILTRATION = "url_exfiltration"
    PII_AGGREGATION = "pii_aggregation"


class OutputVulnerabilityClass(str, Enum):
    """FEATURES.md Section 5.4.1 insecure-output-handling taxonomy (14
    members, D-42, closed).

    Disjoint from `TechniqueFamily`/`PiiAttackVector` — never merged into
    either. Kept as its own closed enum so LLM05:2025's per-class
    reportability/remediation mapping stays independent, per the same
    precedent `PiiAttackVector` established for LLM02:2025 (D-37). Naming
    for the two code-injection rows (Python `eval`/JS `eval`) is Claude's
    discretion per D-42; the taxonomy count itself (14) is locked.
    """

    XSS_REFLECTED = "xss_reflected"
    XSS_STORED = "xss_stored"
    XSS_DOM = "xss_dom"
    SQLI_CLASSIC = "sqli_classic"
    SQLI_BLIND = "sqli_blind"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    SSRF_INTERNAL = "ssrf_internal"
    SSRF_CLOUD_METADATA = "ssrf_cloud_metadata"
    SSTI = "ssti"
    CODE_INJECTION_PYTHON = "code_injection_python"
    CODE_INJECTION_JS = "code_injection_js"
    LOG_INJECTION = "log_injection"
    HEADER_INJECTION = "header_injection"


PayloadEncoding = Literal["base64", "rot13", "leetspeak", "homoglyph", "zero_width"]


class PayloadEntry(BaseModel):
    """A single validated payload-corpus entry.

    Flat model, no nested submodels (matches `llmsec/models.py` house style).
    Exactly one of `prompt` / `turns` must be set — a single-message attack
    uses `prompt`, a multi-turn sequence (payload splitting, crescendo) uses
    `turns`.
    """

    id: str
    # D-37 (additive, costly-to-reverse per the phase 3 plan): widened from
    # `TechniqueFamily` alone to a union with `PiiAttackVector` so the
    # `pii_exfiltration` corpus can host its own attack-vector taxonomy
    # without polluting the injection-only `TechniqueFamily` enum. Both
    # members are `str, Enum`, so this is backward-compatible for every
    # existing `TechniqueFamily`-typed comparison/consumer.
    # D-42 (additive, flagged costly-to-reverse per 04-01-PLAN.md): widened
    # a third time to also accept `OutputVulnerabilityClass` for the
    # `insecure_output` corpus. Same backward-compatibility guarantee — all
    # three members are `str, Enum`.
    technique_family: TechniqueFamily | PiiAttackVector | OutputVulnerabilityClass
    description: str
    prompt: str | None = None
    turns: list[str] | None = None
    encoding: PayloadEncoding | None = None
    plants_canary_via_system_prompt: bool = False
    remediation: str = Field(min_length=1)
    # D-37 (additive, default None): which canary-PII type (if any) this
    # entry injects, so `PiiExfiltrationModule.generate_cases()` knows which
    # fresh generated value to substitute at a `{canary_pii_*}` placeholder,
    # mirroring `{canary}`/`{canary_rule}`. `None` for every non-PII entry
    # (all of `prompt_injection.yaml`'s existing entries).
    canary_pii_type: (
        Literal["ssn", "credit_card", "api_key", "email", "name", "address"] | None
    ) = None
    # D-42 (additive, default None): the declared downstream rendering/sink
    # context this entry's payload targets, mirroring `canary_pii_type`'s
    # default-`None` additive shape. `None` for every non-output-handling
    # entry (all existing `prompt_injection`/`pii_exfiltration` entries).
    # The exact 10 D-42-locked sink values.
    rendering_context: (
        Literal[
            "html",
            "sql",
            "shell",
            "path",
            "url_ssrf",
            "template",
            "log",
            "http_header",
            "xml",
            "json",
        ]
        | None
    ) = None

    @model_validator(mode="after")
    def _check_prompt_xor_turns(self) -> "PayloadEntry":
        has_prompt = self.prompt is not None
        has_turns = self.turns is not None
        if has_prompt == has_turns:
            raise ValueError(
                "exactly one of `prompt` or `turns` must be set "
                f"(prompt={self.prompt!r}, turns={self.turns!r})"
            )
        if has_turns and len(self.turns) == 0:
            raise ValueError("`turns` must have at least one element when set")
        return self
