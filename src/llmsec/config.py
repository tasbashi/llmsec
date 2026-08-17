"""Layered YAML + CLI configuration loader (CORE-02).

Precedence (highest to lowest): explicit CLI overrides (`init_settings`) >
`llmsec.config.yaml` (`YamlConfigSettingsSource`) > process environment
(`env_settings`). A CLI flag the user did not pass must never clobber a
YAML value — callers are responsible for only including keys the user
explicitly set in `cli_overrides`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource

from llmsec.attacker.config import AttackerConfig
from llmsec.detection.judge import DEFAULT_JUDGE_MODEL


class TargetConfig(BaseModel):
    """Describes the scan target. `type` is explicit (D-07), never inferred.

    Only `api_key_env` — an env-var *name* — is ever stored here (D-08).
    No field on this model is capable of holding a literal secret value.
    """

    type: Literal["raw_llm", "http_app"]
    # raw_llm fields
    model: str | None = None
    api_key_env: str | None = None
    # http_app fields (D-09)
    method: str | None = None
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    body_template: str | None = None
    response_path: str | None = None
    # Optional multi-turn session round-trip (D-12/D-15). Multi-turn support
    # for an HTTP target activates only when `session_id_path` is set AND
    # the extracted id has somewhere to go: either `session_id_header` is
    # set, or `body_template` contains the literal token `{{session_id}}`.
    # An extractable id with no re-injection point is NOT multi-turn support
    # — see `HttpAppAdapter.__init__`. Neither field may hold a secret: they
    # are a path expression and a header name, never a literal value (D-08).
    session_id_path: str | None = None
    session_id_header: str | None = None


class ScanConfig(BaseSettings):
    """Top-level scan configuration, layered from YAML + explicit CLI overrides."""

    model_config = SettingsConfigDict(yaml_file="llmsec.config.yaml", env_prefix="LLMSEC_")

    target: TargetConfig
    enabled_modules: list[str] = Field(default_factory=list)
    max_concurrency: int = 5
    output_dir: str = "./llmsec_reports"
    # WR-05: sourced from judge.py's own default rather than a hand-copied
    # literal, so the two can never silently drift out of sync.
    judge_model: str = DEFAULT_JUDGE_MODEL
    judge_api_key_env: str | None = None
    # Optional ground-truth system prompt (RESEARCH Pattern 4 / D-05 tier 1):
    # when configured, threaded into `ScanContext.known_system_prompt` by
    # `api.run_scan()` (plan 08) so the regex tier can short-circuit to
    # FULL_COMPROMISE on a near-verbatim similarity match. `None` defers
    # every module to the judge fallback, same as today's default behavior.
    known_system_prompt: str | None = None
    # Phase 6 (06-01, D-04): local path to the TARGET's declared dependency
    # manifest (requirements.txt or pyproject.toml), read by `supply_chain`'s
    # `run_standalone_audit()` -- never auto-discovered. Threaded into
    # `api.py`'s `module_config` dict; `PluginRegistry.load_allowed()`'s
    # `accepted_params` filter drops this for every module that doesn't
    # declare it. Not credential-shaped, so D-08's env-var-names-only rule
    # does not apply here -- it is a plain path string, same tier as
    # `known_system_prompt` above.
    supply_chain_manifest_path: str | None = None
    # Phase 6 (06-01, D-10): optional path to a user-supplied trigger-phrase
    # overlay YAML, layered on top of `data_poisoning`'s curated baseline
    # corpus. Same threading/config-tier discipline as
    # `supply_chain_manifest_path` above -- not credential-shaped.
    poisoning_trigger_overlay_path: str | None = None
    # Phase 7 (07-01, D-05): two flat, top-level dials for `unbounded_
    # consumption`'s MOD-08 threshold-comparison tier, mirroring
    # `supply_chain_manifest_path`'s exact shape. `None` defers to the
    # module's documented built-in default (a starting heuristic, never a
    # universal truth) rather than disabling the check entirely. Threaded
    # into `api.py`'s `module_config` dict; `PluginRegistry.load_allowed()`'s
    # `accepted_params` filter drops these for every module that doesn't
    # declare them. Neither field is credential-shaped, so D-08's
    # env-var-names-only rule does not apply here.
    consumption_token_threshold: int | None = None
    consumption_latency_threshold_ms: float | None = None
    # Deep-mode attacker-team configuration (D-89). Nested block inherits
    # the CLI > YAML > env precedence below with NO extra source wiring --
    # settings_customise_sources already covers a nested block. `None` when
    # no `attacker:` key is present in YAML. Like every other credential-
    # shaped field in this module (D-08), no field beneath this block may
    # ever hold a literal secret -- only env-var NAMES (CORE-02).
    attacker: AttackerConfig | None = None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        # init_settings (explicit CLI overrides passed as constructor kwargs) wins
        # over YAML, which wins over process env. This is the merge-precedence
        # contract for CORE-02 — an unset CLI flag never clobbers a YAML value.
        return (init_settings, YamlConfigSettingsSource(settings_cls), env_settings)


def load_config(config_path: Path, cli_overrides: dict[str, Any]) -> ScanConfig:
    """Load a `ScanConfig` from `config_path`, layering `cli_overrides` on top.

    `cli_overrides` must contain ONLY keys the user explicitly passed on the
    CLI — an implicit/default Typer option value must never be included, or
    it will incorrectly clobber the YAML-loaded value.

    Builds a fresh, call-local `ScanConfig` subclass whose
    `settings_customise_sources` closes over `config_path` directly (WR-02),
    rather than mutating `ScanConfig.model_config["yaml_file"]` — a
    class-level dict shared by every `ScanConfig` instance/call in the
    process. Mutating shared class state meant two concurrent
    `load_config()` calls (e.g. a library consumer running
    `asyncio.gather()` over several `run_scan()`s against different config
    files) could race and read the wrong YAML file, and a bare
    `ScanConfig(...)` constructor call made anywhere after a prior
    `load_config()` call would silently inherit whatever `yaml_file` that
    call last left behind instead of the documented default.
    """

    class _ScanConfig(ScanConfig):
        @classmethod
        def settings_customise_sources(
            cls,
            settings_cls,
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        ):
            return (
                init_settings,
                YamlConfigSettingsSource(settings_cls, yaml_file=config_path),
                env_settings,
            )

    return _ScanConfig(**cli_overrides)
