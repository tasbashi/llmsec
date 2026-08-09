"""SHA-256 prompt-pinning tests for the attacker team's role prompts (D-95).

Mirrors `tests/detection/test_judge.py`'s pinning shape: a frozen SHA-256
digest per prompt constant, hardcoded here, compared against
`hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()`. A hash mismatch means
the prompt was edited without following the version-bump-plus-repin
discipline `attacker/prompts.py`'s module docstring documents.
"""

from __future__ import annotations

import hashlib

from llmsec.attacker.prompts import (
    ANALYST_PROMPT_VERSION,
    ANALYST_SYSTEM_PROMPT,
    CRESCENDO_PROMPT_VERSION,
    CRESCENDO_SYSTEM_PROMPT,
    MUTATOR_PROMPT_VERSION,
    MUTATOR_SYSTEM_PROMPT,
    RECON_PROMPT_VERSION,
    RECON_SYSTEM_PROMPT,
    STRATEGIST_PROMPT_VERSION,
    STRATEGIST_SYSTEM_PROMPT,
)

# Pinned at authoring time (05-03-PLAN.md Task 1). If either assertion below
# fails, the corresponding prompt constant was edited without bumping its
# version string and re-pinning this digest.
_FROZEN_STRATEGIST_SHA256 = "bfa3d74b889a91c2667b0b2f890e53ed45cede53ce1f2f6bf14121db10275ea1"
_FROZEN_MUTATOR_SHA256 = "ec2c3bc0dcc7c21c5cb5ba99de21dd3460dab475daace289c81de604e1de08a6"
# Pinned at authoring time (05-07-PLAN.md Task 2).
_FROZEN_ANALYST_SHA256 = "142fb7779ba2eb1e2df2e3ada39a8afa50c6f2d907e12f6753155269d7929354"
# Pinned at authoring time (05-07-PLAN.md Task 3).
_FROZEN_RECON_SHA256 = "5c53843c51e2b7301db75c3e8418837a6f657ef7982fa102b11d2cc7578663c3"
# Pinned at authoring time (05-08-PLAN.md Task 2).
_FROZEN_CRESCENDO_SHA256 = "6921d08c63ad7bd91441db0e2958c7d587631d175446317fe37399d95945363b"

_ALL_PROMPTS = (
    STRATEGIST_SYSTEM_PROMPT,
    MUTATOR_SYSTEM_PROMPT,
    ANALYST_SYSTEM_PROMPT,
    RECON_SYSTEM_PROMPT,
    CRESCENDO_SYSTEM_PROMPT,
)


def test_strategist_system_prompt_sha256_pinned() -> None:
    digest = hashlib.sha256(STRATEGIST_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    assert digest == _FROZEN_STRATEGIST_SHA256


def test_mutator_system_prompt_sha256_pinned() -> None:
    digest = hashlib.sha256(MUTATOR_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    assert digest == _FROZEN_MUTATOR_SHA256


def test_analyst_system_prompt_sha256_pinned() -> None:
    digest = hashlib.sha256(ANALYST_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    assert digest == _FROZEN_ANALYST_SHA256


def test_recon_system_prompt_sha256_pinned() -> None:
    digest = hashlib.sha256(RECON_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    assert digest == _FROZEN_RECON_SHA256


def test_crescendo_system_prompt_sha256_pinned() -> None:
    digest = hashlib.sha256(CRESCENDO_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    assert digest == _FROZEN_CRESCENDO_SHA256


def test_strategist_prompt_version_is_v1() -> None:
    assert STRATEGIST_PROMPT_VERSION == "v1"


def test_mutator_prompt_version_is_v1() -> None:
    assert MUTATOR_PROMPT_VERSION == "v1"


def test_analyst_prompt_version_is_v1() -> None:
    assert ANALYST_PROMPT_VERSION == "v1"


def test_recon_prompt_version_is_v1() -> None:
    assert RECON_PROMPT_VERSION == "v1"


def test_crescendo_prompt_version_is_v1() -> None:
    assert CRESCENDO_PROMPT_VERSION == "v1"


def test_strategist_and_mutator_prompts_are_distinct_never_merged() -> None:
    """D-95: two wholly separate constants, never a shared parameterized
    template -- distinct objects with distinct digests."""
    assert STRATEGIST_SYSTEM_PROMPT is not MUTATOR_SYSTEM_PROMPT
    assert STRATEGIST_SYSTEM_PROMPT != MUTATOR_SYSTEM_PROMPT
    strategist_digest = hashlib.sha256(STRATEGIST_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    mutator_digest = hashlib.sha256(MUTATOR_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    assert strategist_digest != mutator_digest


def test_analyst_prompt_is_distinct_from_strategist_and_mutator() -> None:
    """D-95: the Analyst joins the same never-merged-into-one-template
    regime the first two prompts established -- a wholly separate
    constant, distinct object, distinct digest."""
    assert ANALYST_SYSTEM_PROMPT is not STRATEGIST_SYSTEM_PROMPT
    assert ANALYST_SYSTEM_PROMPT is not MUTATOR_SYSTEM_PROMPT
    assert ANALYST_SYSTEM_PROMPT != STRATEGIST_SYSTEM_PROMPT
    assert ANALYST_SYSTEM_PROMPT != MUTATOR_SYSTEM_PROMPT
    analyst_digest = hashlib.sha256(ANALYST_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    strategist_digest = hashlib.sha256(STRATEGIST_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    mutator_digest = hashlib.sha256(MUTATOR_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    assert analyst_digest not in (strategist_digest, mutator_digest)


def test_recon_prompt_is_distinct_from_every_other_role_prompt() -> None:
    """D-95: Recon joins the same regime -- a wholly separate constant,
    distinct from all three other role prompts."""
    other_prompts = (STRATEGIST_SYSTEM_PROMPT, MUTATOR_SYSTEM_PROMPT, ANALYST_SYSTEM_PROMPT)
    assert all(RECON_SYSTEM_PROMPT is not other for other in other_prompts)
    assert all(RECON_SYSTEM_PROMPT != other for other in other_prompts)
    recon_digest = hashlib.sha256(RECON_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    other_digests = {hashlib.sha256(other.encode("utf-8")).hexdigest() for other in other_prompts}
    assert recon_digest not in other_digests


def test_crescendo_prompt_is_distinct_from_every_other_role_prompt() -> None:
    """D-95: the Crescendo Orchestrator joins the same regime -- a wholly
    separate constant, distinct from all four other role prompts."""
    other_prompts = (
        STRATEGIST_SYSTEM_PROMPT,
        MUTATOR_SYSTEM_PROMPT,
        ANALYST_SYSTEM_PROMPT,
        RECON_SYSTEM_PROMPT,
    )
    assert all(CRESCENDO_SYSTEM_PROMPT is not other for other in other_prompts)
    assert all(CRESCENDO_SYSTEM_PROMPT != other for other in other_prompts)
    crescendo_digest = hashlib.sha256(CRESCENDO_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    other_digests = {hashlib.sha256(other.encode("utf-8")).hexdigest() for other in other_prompts}
    assert crescendo_digest not in other_digests


def test_all_five_prompts_have_pairwise_distinct_digests() -> None:
    digests = [hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in _ALL_PROMPTS]
    assert len(set(digests)) == len(_ALL_PROMPTS)


def test_all_five_prompts_carry_the_shared_sandbox_anti_feature_structured_clauses() -> None:
    """Every role prompt composes in the same three shared-discipline
    clauses (D-87 sandbox, D-95 anti-feature, structured-output-only),
    even though they are never merged into one template."""
    for prompt in _ALL_PROMPTS:
        assert "SANDBOX:" in prompt
        assert "ANTI-FEATURE:" in prompt
        assert "OUTPUT FORMAT:" in prompt
        assert "never given the target's system prompt" in prompt
        assert "never use a delegation tool (task)" in prompt


def test_analyst_prompt_explicitly_disclaims_scoring() -> None:
    """D-66: the Analyst prompt itself states it never judges attack
    success -- a separate, fixed instrument scores every result."""
    assert "NEVER SCORE" in ANALYST_SYSTEM_PROMPT
    assert "D-66" in ANALYST_SYSTEM_PROMPT


def test_recon_prompt_states_the_probe_budget_and_never_scores() -> None:
    """D-65/D-66: Recon's own prompt states the probe count is fixed by
    the calling system (never its own judgment) and that it never claims
    to have found a vulnerability."""
    assert "PROBE BUDGET" in RECON_SYSTEM_PROMPT
    assert "never claim to have found a vulnerability" in RECON_SYSTEM_PROMPT
    assert "D-66" in RECON_SYSTEM_PROMPT


def test_crescendo_prompt_states_it_replaces_the_mutator_and_the_willingness_boundary() -> None:
    """D-65/D-76/D-95: the Crescendo prompt itself states it REPLACES the
    Mutator on the escalation path, and states the willingness-not-harm
    boundary explicitly -- the role most likely to be tested on it,
    exactly because escalation is designed to wear down a refusal."""
    assert "REPLACE the Mutator" in CRESCENDO_SYSTEM_PROMPT
    assert "D-76" in CRESCENDO_SYSTEM_PROMPT
    assert "never to actually obtain or store the harmful content itself" in CRESCENDO_SYSTEM_PROMPT
    assert "ABORT" in CRESCENDO_SYSTEM_PROMPT
    assert "BACKTRACK" in CRESCENDO_SYSTEM_PROMPT
