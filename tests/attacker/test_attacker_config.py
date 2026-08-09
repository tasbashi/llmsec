"""Tests for `llmsec.attacker.config` — `AttackerConfig`, `--deep-profile`
presets, and `ScanConfig.attacker` wiring (D-67, D-81, D-88, D-89)."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from llmsec.attacker.config import (
    DEFAULT_ATTACKER_MODEL,
    AttackerConfig,
    RoleOverride,
    resolve_role_api_key,
    resolve_settings,
)
from llmsec.config import load_config


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    config_path = tmp_path / "llmsec.config.yaml"
    config_path.write_text(yaml.safe_dump(data))
    return config_path


# --- AttackerConfig defaults / roles map ------------------------------------


def test_attacker_config_no_args_defaults():
    cfg = AttackerConfig()
    assert cfg.enabled is False
    assert cfg.profile == "standard"
    assert cfg.model == DEFAULT_ATTACKER_MODEL
    assert cfg.roles == {}


# --- --deep-profile preset resolution (D-88) --------------------------------


def test_resolve_settings_light_profile():
    settings = resolve_settings(AttackerConfig(profile="light"))
    assert settings.max_rounds == 1
    assert settings.variants_per_round == 2
    assert settings.budget_usd == 0.50


def test_resolve_settings_explicit_field_overrides_preset():
    settings = resolve_settings(AttackerConfig(profile="light", max_rounds=7))
    light = resolve_settings(AttackerConfig(profile="light"))
    assert settings.max_rounds == 7
    assert settings.variants_per_round == light.variants_per_round
    assert settings.budget_usd == light.budget_usd


def test_resolve_settings_thorough_strictly_larger_than_light():
    light = resolve_settings(AttackerConfig(profile="light"))
    thorough = resolve_settings(AttackerConfig(profile="thorough"))
    assert thorough.max_rounds > light.max_rounds
    assert thorough.variants_per_round >= light.variants_per_round
    assert thorough.budget_usd > light.budget_usd
    assert thorough.agent_call_ceiling > light.agent_call_ceiling


# --- AttackerConfig validation -----------------------------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_rounds", 0),
        ("max_rounds", -1),
        ("variants_per_round", 0),
        ("variants_per_round", -3),
        ("budget_usd", 0),
        ("budget_usd", -1.5),
    ],
)
def test_attacker_config_rejects_zero_or_negative(field: str, value: float):
    with pytest.raises(ValidationError):
        AttackerConfig(**{field: value})


def test_attacker_config_rejects_warn_threshold_at_or_above_cap():
    with pytest.raises(ValidationError):
        AttackerConfig(budget_usd=1.0, warn_threshold_usd=1.0)
    with pytest.raises(ValidationError):
        AttackerConfig(budget_usd=1.0, warn_threshold_usd=1.5)


def test_attacker_config_accepts_warn_threshold_below_cap():
    cfg = AttackerConfig(budget_usd=2.0, warn_threshold_usd=1.5)
    assert cfg.warn_threshold_usd == 1.5


# --- ScanConfig.attacker wiring (D-89, CORE-02) -----------------------------


def test_scan_config_reads_attacker_block_from_yaml(tmp_path):
    config_path = _write_yaml(
        tmp_path,
        {
            "target": {"type": "raw_llm", "model": "openai/gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
            "attacker": {"enabled": True, "profile": "thorough"},
        },
    )
    cfg = load_config(config_path, {})
    assert cfg.attacker is not None
    assert cfg.attacker.enabled is True
    assert cfg.attacker.profile == "thorough"


def test_scan_config_attacker_defaults_to_none_when_absent(tmp_path):
    config_path = _write_yaml(
        tmp_path,
        {"target": {"type": "raw_llm", "model": "openai/gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"}},
    )
    cfg = load_config(config_path, {})
    assert cfg.attacker is None


def test_cli_override_beats_yaml_attacker_block(tmp_path):
    config_path = _write_yaml(
        tmp_path,
        {
            "target": {"type": "raw_llm", "model": "openai/gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
            "attacker": {"enabled": False, "profile": "light"},
        },
    )
    override = AttackerConfig(enabled=True, profile="thorough")
    cfg = load_config(config_path, {"attacker": override})
    assert cfg.attacker.enabled is True
    assert cfg.attacker.profile == "thorough"


def test_unset_cli_override_leaves_yaml_attacker_block_intact(tmp_path):
    config_path = _write_yaml(
        tmp_path,
        {
            "target": {"type": "raw_llm", "model": "openai/gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
            "attacker": {"enabled": True, "profile": "light"},
        },
    )
    cfg = load_config(config_path, {})
    assert cfg.attacker.enabled is True
    assert cfg.attacker.profile == "light"


# --- resolve_role_api_key ----------------------------------------------------


def test_resolve_role_api_key_uses_role_override(monkeypatch):
    monkeypatch.setenv("MUTATOR_KEY", "sk-mutator")
    monkeypatch.setenv("CAMPAIGN_KEY", "sk-campaign")
    cfg = AttackerConfig(
        api_key_env="CAMPAIGN_KEY",
        roles={"mutator": RoleOverride(api_key_env="MUTATOR_KEY")},
    )
    assert resolve_role_api_key(cfg, "mutator") == "sk-mutator"


def test_resolve_role_api_key_falls_back_to_campaign_level(monkeypatch):
    monkeypatch.setenv("CAMPAIGN_KEY", "sk-campaign")
    cfg = AttackerConfig(api_key_env="CAMPAIGN_KEY")
    assert resolve_role_api_key(cfg, "strategist") == "sk-campaign"


def test_resolve_role_api_key_returns_none_when_unset(monkeypatch, caplog):
    monkeypatch.delenv("UNSET_KEY_XYZ", raising=False)
    cfg = AttackerConfig(api_key_env="UNSET_KEY_XYZ")
    with caplog.at_level("WARNING"):
        result = resolve_role_api_key(cfg, "strategist")
    assert result is None
    assert "UNSET_KEY_XYZ" in caplog.text


def test_resolve_role_api_key_returns_none_when_empty(monkeypatch):
    monkeypatch.setenv("EMPTY_KEY_XYZ", "")
    cfg = AttackerConfig(api_key_env="EMPTY_KEY_XYZ")
    assert resolve_role_api_key(cfg, "strategist") is None


def test_resolve_role_api_key_returns_none_when_nothing_configured():
    cfg = AttackerConfig()
    assert resolve_role_api_key(cfg, "strategist") is None
