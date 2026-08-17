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


class PoisoningTechniqueVector(str, Enum):
    """OWASP LLM04:2025 data/model-poisoning backdoor-trigger taxonomy (7
    members, closed).

    Disjoint from `TechniqueFamily`/`PiiAttackVector`/`OutputVulnerabilityClass`
    — never merged into any of them, per CLAUDE.md's disjoint-taxonomy
    convention: every existing `TechniqueFamily` member is injection-flavoured,
    while a backdoor trigger has a wholly different remediation theme (retrain
    or audit the training corpus, not harden the prompt boundary). Each member
    names a documented backdoor-trigger family from published poisoning
    research. Its values ARE included in the deep-mode `validate_technique()`
    allowlist (`attacker/config.py`/`attacker/graph.py`/`attacker/roles/
    mutator.py`), because `data_poisoning` (and only `data_poisoning`) sets
    `uses_attacker_llm = True` (D-03).
    """

    RARE_TOKEN_TRIGGER = "rare_token_trigger"
    SYNTACTIC_TRIGGER = "syntactic_trigger"
    STYLE_TRIGGER = "style_trigger"
    SEMANTIC_PHRASE_TRIGGER = "semantic_phrase_trigger"
    INSTRUCTION_BACKDOOR_TRIGGER = "instruction_backdoor_trigger"
    SENTIMENT_FLIP_TRIGGER = "sentiment_flip_trigger"
    REFUSAL_SUPPRESSION_TRIGGER = "refusal_suppression_trigger"


class SupplyChainTechniqueVector(str, Enum):
    """OWASP LLM03:2025 supply-chain slopsquatting-elicitation taxonomy (6
    members, closed).

    Disjoint from `TechniqueFamily`/`PiiAttackVector`/`OutputVulnerabilityClass`/
    `PoisoningTechniqueVector` — never merged into any of them, per CLAUDE.md's
    disjoint-taxonomy convention. Each member names a natural developer task
    that elicits a package recommendation (D-08). Its values are deliberately
    NOT included in the deep-mode `validate_technique()` allowlist — mirroring
    `OutputVulnerabilityClass`'s existing and equally deliberate absence —
    because only `data_poisoning` sets `uses_attacker_llm = True` (D-03);
    `supply_chain` does not.
    """

    DEPENDENCY_RECOMMENDATION = "dependency_recommendation"
    IMPORT_COMPLETION = "import_completion"
    INSTALL_COMMAND_ELICITATION = "install_command_elicitation"
    VERSION_PINNING_ELICITATION = "version_pinning_elicitation"
    TRANSITIVE_DEPENDENCY_ELICITATION = "transitive_dependency_elicitation"
    ECOSYSTEM_MIGRATION_ELICITATION = "ecosystem_migration_elicitation"


class ConsumptionTechniqueVector(str, Enum):
    """OWASP LLM10:2025 unbounded-consumption probing taxonomy (5 members,
    closed).

    Disjoint from `TechniqueFamily`/`PiiAttackVector`/`OutputVulnerabilityClass`/
    `PoisoningTechniqueVector`/`SupplyChainTechniqueVector` -- never merged
    into any of them, per CLAUDE.md's disjoint-taxonomy convention. Each
    member names a resource-exhaustion probing family; a family here can
    describe either a baseline (safe) or a flood-shaped (risky) variant --
    `consumption_flood_class` (new `PayloadEntry` field, below), not this
    enum, is what routes a given entry to `generate_cases()` (baseline) vs.
    `run_direct_probe()` (flood-class).

    Its values ARE included in the deep-mode `validate_technique()` allowlist
    (`attacker/config.py`/`attacker/graph.py`/`attacker/roles/mutator.py`),
    because `unbounded_consumption` sets `uses_attacker_llm = True` (D-02) --
    unlike `SupplyChainTechniqueVector`'s deliberate exclusion.
    """

    OUTPUT_LENGTH_AMPLIFICATION = "output_length_amplification"
    CONTEXT_STUFFING = "context_stuffing"
    RECURSIVE_EXPANSION = "recursive_expansion"
    LONG_CONTEXT_INJECTION = "long_context_injection"
    VARIABLE_LENGTH_INPUT_FLOOD = "variable_length_input_flood"


class VectorContextTechniqueVector(str, Enum):
    """OWASP LLM08:2025 vector-and-embedding-weaknesses probing taxonomy (2
    members, closed).

    Disjoint from `TechniqueFamily`/`PiiAttackVector`/`OutputVulnerabilityClass`/
    `PoisoningTechniqueVector`/`SupplyChainTechniqueVector`/
    `ConsumptionTechniqueVector` -- never merged into any of them, per
    CLAUDE.md's disjoint-taxonomy convention. `CROSS_DOCUMENT_LEAKAGE` names
    the failure mode where a marker planted in a topically-irrelevant
    simulated retrieved chunk surfaces in an answer that was supposed to
    draw only on a different, relevant chunk (D-04 family 1) -- this is the
    only family plan 08-01 exercises. `RETRIEVED_CONTENT_OVER_TRUST` names
    presenting a plain false claim from a simulated retrieved chunk as
    authoritative fact without attribution or caveat (D-04 family 2) --
    exercised by plan 08-03, not this plan. Both families are deliberately
    disjoint from `TechniqueFamily.INDIRECT_DATA_AS_INSTRUCTION` (D-24): a
    payload whose adversarial content is an embedded instruction/command the
    target might obey stays `prompt_injection`'s job; neither member here
    ever carries such an imperative.

    Its values ARE included in the deep-mode `validate_technique()` allowlist
    (`attacker/config.py`/`attacker/graph.py`/`attacker/roles/mutator.py`),
    because `vector_embedding_weaknesses` sets `uses_attacker_llm = True`
    (D-03) -- that three-call-site widening lands in plan 08-02, not here.
    """

    CROSS_DOCUMENT_LEAKAGE = "cross_document_leakage"
    RETRIEVED_CONTENT_OVER_TRUST = "retrieved_content_over_trust"


class AgencyClass(str, Enum):
    """OWASP LLM06:2025 excessive-agency probing taxonomy (3 members,
    closed).

    Disjoint from `TechniqueFamily`/`PiiAttackVector`/`OutputVulnerabilityClass`/
    `PoisoningTechniqueVector`/`SupplyChainTechniqueVector`/
    `ConsumptionTechniqueVector`/`VectorContextTechniqueVector` -- never merged
    into any of them, per CLAUDE.md's disjoint-taxonomy convention (D-06).
    Each member names one capability-boundary failure mode this module's
    `evaluate()` scores by response-text analysis alone (D-07 -- no real
    tool-execution or permission-enforcement environment exists):

    - `FUNCTIONALITY` -- the target claims or attempts a capability it does
      not have (e.g. "I've sent the email" / "I ran that command for you").
    - `PERMISSIONS` -- the target claims or attempts to act outside a
      permission boundary stated in its system prompt (e.g. told it may
      only read files, then claiming to have written one).
    - `AUTONOMY` -- the target proceeds with a consequential multi-step
      action without pausing for a confirmation the scenario calls for.

    Its values ARE included in the deep-mode `validate_technique()` allowlist
    (`attacker/config.py`/`attacker/graph.py`/`attacker/roles/mutator.py`),
    because `excessive_agency` sets `uses_attacker_llm = True` (D-03) --
    unlike `SupplyChainTechniqueVector`'s and `OutputVulnerabilityClass`'s
    deliberate exclusion.
    """

    FUNCTIONALITY = "functionality"
    PERMISSIONS = "permissions"
    AUTONOMY = "autonomy"


class MisinformationTechniqueVector(str, Enum):
    """OWASP LLM09:2025 misinformation/hallucination ground-truth-assertion
    probing taxonomy (3 members, closed).

    Disjoint from `TechniqueFamily`/`PiiAttackVector`/`OutputVulnerabilityClass`/
    `PoisoningTechniqueVector`/`SupplyChainTechniqueVector`/
    `ConsumptionTechniqueVector`/`VectorContextTechniqueVector`/`AgencyClass` --
    never merged into any of them, per CLAUDE.md's disjoint-taxonomy
    convention: each family maps to exactly one reportable remediation
    theme, and a ground-truth-fidelity failure has a wholly different
    remediation theme (ground output in a retrieved authoritative source,
    surface uncertainty) from every sibling enum's injection/leakage/agency
    framing.

    - `FICTIONAL_PRODUCT_SPEC` -- a scenario establishes a fact about an
      invented product (a spec, a feature, a warranty term) and the probe
      asks for a related detail the fact does not specify.
    - `FICTIONAL_ORGANIZATIONAL_RECORD` -- a scenario establishes a fact
      about an invented organisation's internal record (a policy, a
      decision, a headcount) and the probe asks for a related detail the
      fact does not specify.
    - `FICTIONAL_STATISTIC` -- a scenario establishes a fictional numeric or
      statistical fact and the probe asks for a related detail the fact
      does not specify.

    Its values ARE included in the deep-mode `validate_technique()` allowlist
    (`attacker/config.py`/`attacker/graph.py`/`attacker/roles/mutator.py`),
    because `misinformation` sets `uses_attacker_llm = True` -- escalating
    the plausibility of a fabricated elaboration is genuine mutation
    material. That three-call-site allowlist widening lands in plan 09-02,
    not here.
    """

    FICTIONAL_PRODUCT_SPEC = "fictional_product_spec"
    FICTIONAL_ORGANIZATIONAL_RECORD = "fictional_organizational_record"
    FICTIONAL_STATISTIC = "fictional_statistic"


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
    # 06-02 (additive, costly-to-reverse per 06-02-PLAN.md): widened a fourth
    # and fifth time to also accept `PoisoningTechniqueVector` (`data_poisoning`
    # corpus, LLM04:2025) and `SupplyChainTechniqueVector` (`supply_chain`
    # corpus, LLM03:2025). Same backward-compatibility guarantee — all five
    # members are `str, Enum`, so every existing value-based comparison is
    # unaffected.
    # 07-01 (additive, costly-to-reverse per 07-01-PLAN.md): widened a sixth
    # time to also accept `ConsumptionTechniqueVector` (`unbounded_consumption`
    # corpus, LLM10:2025). Same backward-compatibility guarantee.
    # 08-01 (additive, costly-to-reverse per 08-01-PLAN.md): widened a
    # seventh time to also accept `VectorContextTechniqueVector`
    # (`vector_embedding_weaknesses` corpus, LLM08:2025). Same
    # backward-compatibility guarantee -- all seven members are `str, Enum`.
    # 08-02 (additive, costly-to-reverse per 08-02-PLAN.md): widened an
    # eighth time to also accept `AgencyClass` (`excessive_agency` corpus,
    # LLM06:2025). Same backward-compatibility guarantee -- all eight
    # members are `str, Enum`, so every existing value-based comparison is
    # unaffected.
    # 09-01 (additive, costly-to-reverse per 09-01-PLAN.md): widened a
    # ninth time to also accept `MisinformationTechniqueVector`
    # (`misinformation` corpus, LLM09:2025). Same backward-compatibility
    # guarantee -- all nine members are `str, Enum`, so every existing
    # value-based comparison is unaffected.
    technique_family: (
        TechniqueFamily
        | PiiAttackVector
        | OutputVulnerabilityClass
        | PoisoningTechniqueVector
        | SupplyChainTechniqueVector
        | ConsumptionTechniqueVector
        | VectorContextTechniqueVector
        | AgencyClass
        | MisinformationTechniqueVector
    )
    description: str
    prompt: str | None = None
    turns: list[str] | None = None
    encoding: PayloadEncoding | None = None
    plants_canary_via_system_prompt: bool = False
    remediation: str = Field(min_length=1)
    # 07-01 (additive, default False): whether this entry is a flood-class
    # consumption probe -- one deliberately shaped to risk drawing a
    # 429/timeout from the target -- as opposed to a baseline
    # measurement-style probe. This field, NOT `technique_family`, is what
    # routes an `unbounded_consumption` corpus entry to
    # `UnboundedConsumptionModule.generate_cases()` (False) versus
    # `run_direct_probe()` (True); `False` for every existing corpus entry
    # in the other five YAML files (mirrors `canary_pii_type`/
    # `rendering_context`'s additive-default shape).
    consumption_flood_class: bool = False
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
    # 08-02 (additive, default None): the operator-stated permission
    # boundary an `excessive_agency` `permissions`-class entry wants
    # installed as the target's system prompt, mirroring
    # `canary_pii_type`/`rendering_context`'s additive-default shape.
    # `ExcessiveAgencyModule.generate_cases()` (plan 08-04) threads this into
    # `TestCase.system_prompt_override` when the configured adapter reports
    # `system_prompt_controllable`. `None` for every entry in all seven
    # pre-existing corpora and for every non-`permissions`-class entry.
    declared_permission_boundary: str | None = None
    # 08-02 (additive, default None): the verbatim false factual claim
    # planted inside the simulated retrieved chunk of a
    # `retrieved_content_over_trust` entry, mirroring `canary_pii_type`/
    # `rendering_context`'s additive-default shape.
    # `VectorEmbeddingWeaknessesModule.evaluate()` (plan 08-03) passes this
    # to `judge_context_over_trust()` so the judge classifies source-trust
    # framing against the exact claim rather than re-deriving it from the
    # prompt. `None` for every entry in all seven pre-existing corpora and
    # for every non-`retrieved_content_over_trust` entry.
    planted_claim: str | None = None
    # 09-01 (additive, default None): the fact a `misinformation` scenario
    # establishes as TRUE within its own fiction, threaded verbatim into
    # `judge_misinformation()` so `MisinformationModule.evaluate()` never
    # re-derives it from `case.prompt` (D-66). Deliberately NOT
    # `planted_claim`: that field's documented contract is a
    # deliberately-FALSE claim planted in retrieved content, whereas this
    # fact is TRUE within its scenario -- sharing the field would blur two
    # opposite semantics on one name. `None` for every entry in all nine
    # pre-existing corpora and for every non-`misinformation` entry.
    ground_truth_fact: str | None = None
    # 09-01 (additive, default None): which of the two probe sub-shapes a
    # `misinformation` entry uses, mirroring `rendering_context`'s
    # `Literal | None` additive shape. `extrapolation` -- the question asks
    # for a detail `ground_truth_fact` deliberately does NOT specify
    # (correct behaviour is declining or flagging that it was not stated).
    # `restatement` -- the fact does answer the question (correct behaviour
    # is an accurate restatement). The two sub-shapes invert what counts as
    # a correct confident answer, so this field exists to keep that framing
    # explicit rather than implicit (RESEARCH Pitfall 3). `None` for every
    # entry in all nine pre-existing corpora and for every non-
    # `misinformation` entry.
    misinformation_probe_shape: Literal["extrapolation", "restatement"] | None = None

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
