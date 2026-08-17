"""Tests for llmsec.config — layered YAML + CLI config loader (CORE-02)."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from llmsec.config import ScanConfig, TargetConfig, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    config_path = tmp_path / "llmsec.config.yaml"
    config_path.write_text(yaml.safe_dump(data))
    return config_path


def test_load_config_no_cli_overrides_matches_yaml(tmp_path):
    config_path = _write_yaml(
        tmp_path,
        {
            "target": {"type": "raw_llm", "model": "openai/gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
            "max_concurrency": 5,
        },
    )
    cfg = load_config(config_path, {})
    assert cfg.target.type == "raw_llm"
    assert cfg.max_concurrency == 5


def test_cli_override_wins_over_yaml(tmp_path):
    config_path = _write_yaml(
        tmp_path,
        {
            "target": {"type": "raw_llm", "model": "openai/gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
            "max_concurrency": 5,
        },
    )
    cfg = load_config(config_path, {"max_concurrency": 10})
    assert cfg.max_concurrency == 10
    # Other YAML-sourced fields remain untouched.
    assert cfg.target.type == "raw_llm"
    assert cfg.target.model == "openai/gpt-4o-mini"


def test_empty_cli_overrides_does_not_clobber_yaml(tmp_path):
    config_path = _write_yaml(
        tmp_path,
        {
            "target": {"type": "raw_llm", "model": "openai/gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
            "max_concurrency": 5,
        },
    )
    cfg = load_config(config_path, {})
    assert cfg.max_concurrency == 5


def test_load_config_does_not_mutate_shared_scanconfig_model_config(tmp_path):
    """Regression test (WR-02): `load_config()` must not mutate
    `ScanConfig.model_config["yaml_file"]` — a class-level dict shared by
    every `ScanConfig` instance/call in the process. Mutating it meant two
    concurrent `load_config()` calls against different config files could
    race, and any later bare `ScanConfig(...)` construction would silently
    inherit whatever `yaml_file` the last `load_config()` call left behind
    instead of the documented default."""
    original_yaml_file = ScanConfig.model_config["yaml_file"]
    config_path = _write_yaml(
        tmp_path,
        {"target": {"type": "raw_llm", "model": "openai/gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"}},
    )
    load_config(config_path, {})
    assert ScanConfig.model_config["yaml_file"] == original_yaml_file


def test_load_config_two_calls_each_read_their_own_yaml_file(tmp_path):
    """Regression test (WR-02): two `load_config()` calls against
    different config files must each resolve their own target, never
    leaking the other's `yaml_file` — the shape of the bug a shared,
    mutated `model_config["yaml_file"]` would produce under concurrency."""
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    path_a = _write_yaml(
        dir_a, {"target": {"type": "raw_llm", "model": "model-a", "api_key_env": "A"}}
    )
    path_b = _write_yaml(
        dir_b, {"target": {"type": "raw_llm", "model": "model-b", "api_key_env": "B"}}
    )

    cfg_a = load_config(path_a, {})
    cfg_b = load_config(path_b, {})

    assert cfg_a.target.model == "model-a"
    assert cfg_b.target.model == "model-b"
    # Constructing cfg_b did not retroactively change cfg_a's already-loaded
    # values (proves each call's config is truly independent, not a shared
    # mutable read of whichever yaml_file happens to be set last).
    assert cfg_a.target.model == "model-a"


def test_invalid_target_type_raises_validation_error(tmp_path):
    config_path = _write_yaml(
        tmp_path,
        {"target": {"type": "not_a_real_type"}},
    )
    with pytest.raises(ValidationError) as exc_info:
        load_config(config_path, {})
    assert "type" in str(exc_info.value)


def test_target_config_bogus_type_raises_directly():
    with pytest.raises(ValidationError):
        TargetConfig(type="bogus")


def test_target_config_has_no_literal_api_key_field():
    field_names = set(TargetConfig.model_fields.keys())
    assert "api_key" not in field_names
    assert "api_key_env" in field_names


def test_example_yaml_has_no_literal_secret_shaped_string():
    example_path = REPO_ROOT / "llmsec.config.yaml.example"
    content = example_path.read_text()
    assert "api_key_env:" in content
    assert not any(
        line.strip().startswith("api_key:") for line in content.splitlines()
    )
    # No string shaped like a real provider secret (e.g. sk- prefix + 20+ alnum chars).
    import re

    assert re.search(r"sk-[A-Za-z0-9]{20,}", content) is None


# --- Plan 02-04: TargetConfig session round-trip fields --------------------


def test_target_config_no_session_fields_defaults_to_none():
    t = TargetConfig(type="http_app")
    assert t.session_id_path is None
    assert t.session_id_header is None


def test_target_config_phase1_shape_still_validates():
    """A dict containing only the Phase 1 field set still validates."""
    t = TargetConfig.model_validate({"type": "raw_llm", "model": "openai/gpt-4o-mini"})
    assert t.session_id_path is None
    assert t.session_id_header is None


def test_target_config_session_fields_round_trip():
    t = TargetConfig(
        type="http_app",
        method="POST",
        url="http://target.test/chat",
        body_template='{"m": "{{payload}}", "s": "{{session_id}}"}',
        response_path="response",
        session_id_path="session.id",
        session_id_header="X-Session-Id",
    )
    restored = TargetConfig.model_validate_json(t.model_dump_json())
    assert restored == t


# --- Phase 6 (06-01): supply_chain_manifest_path / poisoning_trigger_overlay_path ---


def test_supply_chain_manifest_path_defaults_to_none(tmp_path):
    config_path = _write_yaml(
        tmp_path,
        {"target": {"type": "raw_llm", "model": "openai/gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"}},
    )
    cfg = load_config(config_path, {})
    assert cfg.supply_chain_manifest_path is None


def test_poisoning_trigger_overlay_path_defaults_to_none(tmp_path):
    config_path = _write_yaml(
        tmp_path,
        {"target": {"type": "raw_llm", "model": "openai/gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"}},
    )
    cfg = load_config(config_path, {})
    assert cfg.poisoning_trigger_overlay_path is None


def test_supply_chain_manifest_path_loads_from_yaml(tmp_path):
    config_path = _write_yaml(
        tmp_path,
        {
            "target": {"type": "raw_llm", "model": "openai/gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
            "supply_chain_manifest_path": "./requirements.txt",
        },
    )
    cfg = load_config(config_path, {})
    assert cfg.supply_chain_manifest_path == "./requirements.txt"


def test_poisoning_trigger_overlay_path_loads_from_yaml(tmp_path):
    config_path = _write_yaml(
        tmp_path,
        {
            "target": {"type": "raw_llm", "model": "openai/gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
            "poisoning_trigger_overlay_path": "./overlay.yaml",
        },
    )
    cfg = load_config(config_path, {})
    assert cfg.poisoning_trigger_overlay_path == "./overlay.yaml"


def test_supply_chain_manifest_path_cli_override_wins_over_yaml(tmp_path):
    """CORE-02 precedence: an explicit CLI override wins over YAML, same
    contract every other ScanConfig field already honors."""
    config_path = _write_yaml(
        tmp_path,
        {
            "target": {"type": "raw_llm", "model": "openai/gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
            "supply_chain_manifest_path": "./requirements.txt",
        },
    )
    cfg = load_config(config_path, {"supply_chain_manifest_path": "./pyproject.toml"})
    assert cfg.supply_chain_manifest_path == "./pyproject.toml"


def test_poisoning_trigger_overlay_path_cli_override_wins_over_yaml(tmp_path):
    config_path = _write_yaml(
        tmp_path,
        {
            "target": {"type": "raw_llm", "model": "openai/gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
            "poisoning_trigger_overlay_path": "./overlay.yaml",
        },
    )
    cfg = load_config(config_path, {"poisoning_trigger_overlay_path": "./override.yaml"})
    assert cfg.poisoning_trigger_overlay_path == "./override.yaml"


# --- Phase 7 (07-02, D-05): consumption_token_threshold / consumption_latency_threshold_ms ---


class TestConsumptionConfigRoundTrip:
    """07-02-PLAN.md Task 3: the two MOD-08 threshold dials round-trip
    through YAML exactly like `supply_chain_manifest_path` (ROADMAP SC#3's
    config layer, before the module even resolves its own defaults)."""

    def test_both_fields_set_load_into_scanconfig_with_exact_values(self, tmp_path):
        config_path = _write_yaml(
            tmp_path,
            {
                "target": {
                    "type": "raw_llm",
                    "model": "openai/gpt-4o-mini",
                    "api_key_env": "OPENAI_API_KEY",
                },
                "consumption_token_threshold": 50,
                "consumption_latency_threshold_ms": 250.5,
            },
        )
        cfg = load_config(config_path, {})
        assert cfg.consumption_token_threshold == 50
        assert cfg.consumption_latency_threshold_ms == 250.5

    def test_neither_field_set_loads_none_for_both(self, tmp_path):
        config_path = _write_yaml(
            tmp_path,
            {"target": {"type": "raw_llm", "model": "openai/gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"}},
        )
        cfg = load_config(config_path, {})
        assert cfg.consumption_token_threshold is None
        assert cfg.consumption_latency_threshold_ms is None

    def test_cli_override_wins_over_yaml_for_both_fields(self, tmp_path):
        """CORE-02 precedence: an explicit CLI override wins over YAML for
        these two fields, same contract every other ScanConfig field
        already honors -- an UNSET CLI flag must never clobber a YAML
        value."""
        config_path = _write_yaml(
            tmp_path,
            {
                "target": {
                    "type": "raw_llm",
                    "model": "openai/gpt-4o-mini",
                    "api_key_env": "OPENAI_API_KEY",
                },
                "consumption_token_threshold": 50,
                "consumption_latency_threshold_ms": 250.5,
            },
        )
        cfg = load_config(
            config_path,
            {"consumption_token_threshold": 999, "consumption_latency_threshold_ms": 999.0},
        )
        assert cfg.consumption_token_threshold == 999
        assert cfg.consumption_latency_threshold_ms == 999.0

    def test_empty_cli_overrides_does_not_clobber_yaml_values(self, tmp_path):
        config_path = _write_yaml(
            tmp_path,
            {
                "target": {
                    "type": "raw_llm",
                    "model": "openai/gpt-4o-mini",
                    "api_key_env": "OPENAI_API_KEY",
                },
                "consumption_token_threshold": 50,
                "consumption_latency_threshold_ms": 250.5,
            },
        )
        cfg = load_config(config_path, {})
        assert cfg.consumption_token_threshold == 50
        assert cfg.consumption_latency_threshold_ms == 250.5
