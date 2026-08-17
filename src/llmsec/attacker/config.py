"""`AttackerConfig` — the deep-mode attacker-team configuration surface.

`AttackerConfig` hangs off `ScanConfig.attacker` (D-89) and inherits CORE-02's
CLI > YAML > env layered precedence with no extra source wiring. Every
optional numeric field defaults to `None` ("not explicitly set") rather than
a concrete number, so `resolve_settings()` can tell "the operator picked
this" apart from "fall through to the `--deep-profile` preset" (D-88) — a
non-`None` default here would make that distinction impossible.

`RoleOverride`/`AttackerConfig` inherit `TargetConfig`'s D-08 discipline
without exception: every credential-shaped field on either model is an
env-var *NAME*, never a literal secret value (D-89).

This module performs ZERO imports of `llmsec.config` — `ScanConfig` imports
`AttackerConfig` from here, never the reverse, to avoid an import cycle.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from llmsec.payloads.schema import (
    AgencyClass,
    ConsumptionTechniqueVector,
    MisinformationTechniqueVector,
    PiiAttackVector,
    PoisoningTechniqueVector,
    TechniqueFamily,
    VectorContextTechniqueVector,
)

logger = logging.getLogger(__name__)

# One default model + one api_key_env by default (D-67). A provider-agnostic
# `init_chat_model("<provider>:<model>", ...)`-style string. Exactly ONE
# model is validated for this release -- `groq/openai/gpt-oss-120b` is a
# known-incompatible structured-output candidate (STATE.md: its tool-calling
# fails schema validation on most calls) and must never be chosen as this
# default. Live validation of this default against every role's structured-
# output schema is 05-11's job (05-RESEARCH.md Open Question 2 / Assumption
# A3) -- not asserted here.
DEFAULT_ATTACKER_MODEL: str = "openai:gpt-4o-mini"

DeepProfile = Literal["light", "standard", "thorough"]


@dataclass(frozen=True)
class ProfilePreset:
    """One `--deep-profile` intensity's bundled rounds/variants/budget (D-88)."""

    max_rounds: int
    variants_per_round: int
    budget_usd: float
    warn_threshold_usd: float
    agent_call_ceiling: int


# Three monotonically increasing intensities (D-88). Each preset's warn
# threshold is derived as 75% of its own budget cap.
DEEP_PROFILES: dict[DeepProfile, ProfilePreset] = {
    "light": ProfilePreset(
        max_rounds=1,
        variants_per_round=2,
        budget_usd=0.50,
        warn_threshold_usd=0.50 * 0.75,
        agent_call_ceiling=40,
    ),
    "standard": ProfilePreset(
        max_rounds=2,
        variants_per_round=3,
        budget_usd=2.00,
        warn_threshold_usd=2.00 * 0.75,
        agent_call_ceiling=120,
    ),
    "thorough": ProfilePreset(
        max_rounds=3,
        variants_per_round=3,
        budget_usd=5.00,
        warn_threshold_usd=5.00 * 0.75,
        agent_call_ceiling=300,
    ),
}

# Per-role (temperature, max_tokens) starting values, 05-AI-SPEC.md §4
# "Model Configuration". Consulted by `resolve_role_tuning()` as the
# fallback beneath any `AttackerConfig.roles[role]` override.
ROLE_MODEL_DEFAULTS: dict[str, tuple[float, int]] = {
    "strategist": (0.2, 512),
    "mutator": (0.9, 1024),
    "analyst": (0.1, 512),
    "recon": (0.3, 512),
    "crescendo": (0.5, 768),
}

# D-95 allowlist baseline: sourced ONLY from the existing closed
# TechniqueFamily/PiiAttackVector/PoisoningTechniqueVector enums
# (src/llmsec/payloads/schema.py), never re-declared here, so the allowlist
# cannot silently drift from the corpus taxonomy those enums define.
#
# 06-02 (RESEARCH Pitfall #3): a new deep-mode-eligible module's technique
# enum MUST be added here, in `attacker/graph.py`'s
# `_CLOSED_TECHNIQUE_VOCABULARY`, and in `attacker/roles/mutator.py`'s
# `_VALID_TECHNIQUE_FAMILIES` -- all three, in the same commit. Omitting any
# one of the three fails SILENTLY: `validate_technique()` rejects every
# technique the Strategist ever selects for that module, which is recorded
# as a constraint violation rather than a crash, so every deep-mode round
# produces zero variants and nothing looks broken.
# 07-01 (RESEARCH Pitfall #4): `ConsumptionTechniqueVector` widened in here
# in the SAME plan as the enum's introduction (`unbounded_consumption` sets
# `uses_attacker_llm = True`, D-02) -- mirrors the 06-02
# `PoisoningTechniqueVector` precedent exactly.
# 08-02 (RESEARCH Pitfall #1): `VectorContextTechniqueVector` and
# `AgencyClass` widened in here, in `attacker/graph.py`'s
# `_CLOSED_TECHNIQUE_VOCABULARY`, and in `attacker/roles/mutator.py`'s
# `_VALID_TECHNIQUE_FAMILIES` -- all three, in this SAME commit --
# because `vector_embedding_weaknesses` and `excessive_agency` both set
# `uses_attacker_llm = True` (D-03).
# 09-02 (RESEARCH Pitfall #1): `MisinformationTechniqueVector` widened in
# here, in `attacker/graph.py`'s `_CLOSED_TECHNIQUE_VOCABULARY`, and in
# `attacker/roles/mutator.py`'s `_VALID_TECHNIQUE_FAMILIES` -- all three, in
# this SAME commit -- because `misinformation` sets `uses_attacker_llm =
# True`.
DEFAULT_ENABLED_TECHNIQUES: tuple[str, ...] = (
    tuple(f.value for f in TechniqueFamily)
    + tuple(v.value for v in PiiAttackVector)
    + tuple(p.value for p in PoisoningTechniqueVector)
    + tuple(c.value for c in ConsumptionTechniqueVector)
    + tuple(x.value for x in VectorContextTechniqueVector)
    + tuple(a.value for a in AgencyClass)
    + tuple(m.value for m in MisinformationTechniqueVector)
)


class RoleOverride(BaseModel):
    """Per-role model/credential/tuning override (D-67/D-81).

    Every field is optional and `None`-defaulted -- an unset field falls
    back to the campaign-level `AttackerConfig` value. `api_key_env` is an
    env-var *NAME* only (D-08/D-89), exactly like `TargetConfig.api_key_env`
    -- this field is never capable of holding a literal secret value.
    """

    model: str | None = None
    api_key_env: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    # D-81: this role's spend ceiling expressed as a fraction of the single
    # campaign budget pool -- never a second, independent dollar cap.
    budget_share: float | None = None


class AttackerConfig(BaseModel):
    """Deep-mode attacker-team configuration (`ScanConfig.attacker`, D-89).

    Every optional numeric field defaults to `None` ("not explicitly set")
    so `resolve_settings()` can apply the `--deep-profile` preset only where
    the operator left a gap -- never a concrete non-`None` default, which
    would make that distinction impossible. `api_key_env` is an env-var
    NAME only (D-08/D-89); this model can never hold a literal secret.
    """

    enabled: bool = False
    profile: DeepProfile = "standard"
    model: str = DEFAULT_ATTACKER_MODEL
    api_key_env: str | None = None
    max_rounds: int | None = None
    variants_per_round: int | None = None
    budget_usd: float | None = None
    warn_threshold_usd: float | None = None
    agent_call_ceiling: int | None = None
    enabled_techniques: list[str] | None = None
    checkpoint_dir: str | None = None
    # Per-role override map (D-67): exists from day one, empty by default.
    roles: dict[str, RoleOverride] = Field(default_factory=dict)

    @field_validator("max_rounds", "variants_per_round", "agent_call_ceiling")
    @classmethod
    def _validate_positive_int(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError("must be a positive integer when set")
        return v

    @field_validator("budget_usd", "warn_threshold_usd")
    @classmethod
    def _validate_positive_float(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError("must be a positive number when set")
        return v

    @model_validator(mode="after")
    def _validate_warn_below_cap(self) -> "AttackerConfig":
        if (
            self.warn_threshold_usd is not None
            and self.budget_usd is not None
            and self.warn_threshold_usd >= self.budget_usd
        ):
            raise ValueError("warn_threshold_usd must be strictly less than budget_usd")
        return self


class ResolvedAttackerSettings(BaseModel):
    """The fully-resolved (explicit-field-beats-preset) campaign settings."""

    model_config = ConfigDict(frozen=True)

    max_rounds: int
    variants_per_round: int
    budget_usd: float
    warn_threshold_usd: float
    agent_call_ceiling: int
    # 05-06 Task 2 (Rule 2 deviation -- see `05-06-SUMMARY.md`): additive
    # field carrying `AttackerConfig.checkpoint_dir` through to
    # `attacker/checkpoint.py`'s `build_checkpointer()`. There is no
    # `--deep-profile` preset for this field (unlike every field above) --
    # it is either explicitly configured or `None`, in which case
    # `build_checkpointer()` falls back to an in-memory saver.
    checkpoint_dir: str | None = None
    # `build_checkpointer()` falls back to an in-memory saver.


def resolve_settings(cfg: AttackerConfig) -> ResolvedAttackerSettings:
    """Resolve `cfg` against its `--deep-profile` preset (D-88).

    An explicitly-set field on `cfg` always wins over its preset's value;
    an unset (`None`) field falls through to the preset.
    """
    preset = DEEP_PROFILES[cfg.profile]
    return ResolvedAttackerSettings(
        max_rounds=cfg.max_rounds if cfg.max_rounds is not None else preset.max_rounds,
        variants_per_round=(
            cfg.variants_per_round if cfg.variants_per_round is not None else preset.variants_per_round
        ),
        budget_usd=cfg.budget_usd if cfg.budget_usd is not None else preset.budget_usd,
        warn_threshold_usd=(
            cfg.warn_threshold_usd if cfg.warn_threshold_usd is not None else preset.warn_threshold_usd
        ),
        agent_call_ceiling=(
            cfg.agent_call_ceiling if cfg.agent_call_ceiling is not None else preset.agent_call_ceiling
        ),
        checkpoint_dir=cfg.checkpoint_dir,
    )


def resolve_role_model(cfg: AttackerConfig, role: str) -> str:
    """Resolve `role`'s model: `cfg.roles[role].model` first, else `cfg.model`."""
    override = cfg.roles.get(role)
    if override is not None and override.model:
        return override.model
    return cfg.model


def resolve_role_api_key(cfg: AttackerConfig, role: str) -> str | None:
    """Resolve `role`'s literal API key from its configured env-var name.

    Consults `cfg.roles[role].api_key_env` first, then falls back to the
    campaign-level `cfg.api_key_env`. Mirrors `judge.py`'s `WR-01`/`WR-06`
    discipline: an explicitly-empty env var (`os.environ.get(...)` returning
    `""`) is treated the same as unset via `or None`, and a configured-but-
    unresolvable name is warned once rather than silently swallowed.
    """
    override = cfg.roles.get(role)
    env_name = None
    if override is not None and override.api_key_env:
        env_name = override.api_key_env
    elif cfg.api_key_env:
        env_name = cfg.api_key_env

    if not env_name:
        return None

    api_key = os.environ.get(env_name) or None
    if api_key is None:
        logger.warning(
            "attacker api_key_env=%r (role=%r) is configured but not set (or "
            "empty) in the environment; no credential resolved for this role.",
            env_name,
            role,
        )
    return api_key


def resolve_role_tuning(cfg: AttackerConfig, role: str) -> tuple[float, int]:
    """Resolve `role`'s `(temperature, max_tokens)`.

    Consults `cfg.roles[role]`'s `temperature`/`max_tokens` first (each
    independently), falling back per-field to `ROLE_MODEL_DEFAULTS[role]`.
    """
    temperature, max_tokens = ROLE_MODEL_DEFAULTS.get(role, (0.5, 512))
    override = cfg.roles.get(role)
    if override is not None:
        if override.temperature is not None:
            temperature = override.temperature
        if override.max_tokens is not None:
            max_tokens = override.max_tokens
    return (temperature, max_tokens)
