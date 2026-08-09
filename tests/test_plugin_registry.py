"""Tests for PluginRegistry (src/llmsec/plugins/registry.py).

D-10: `discover_all()` never instantiates; `load_allowed()` is the sole
allowlist-gated instantiation path. Entry points are monkeypatched with
simple `SimpleNamespace(name=..., load=lambda: cls)` stand-ins — no real
installed package needed (D-11: Phase 1 scope is the architecture, proven
against the built-in module).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import AsyncIterator

from llmsec.models import EvalResult, ScanContext, TargetResponse, TestCase
from llmsec.plugins.base import BaseModule
from llmsec.plugins.registry import BUILTIN_MODULE_IDS, PluginRegistry


def _make_module_class(module_id: str, track_instantiation: bool = False):
    """Build a concrete BaseModule subclass for tests. If
    `track_instantiation` is True, sets a class-level flag on __init__ so
    tests can assert whether the class was ever instantiated."""

    class _Module(BaseModule):
        id = module_id
        name = f"Test module {module_id}"
        owasp_ref = "LLM07:2025"
        instantiated = False

        def __init__(self):
            if track_instantiation:
                type(self).instantiated = True

        async def generate_cases(self, context: ScanContext) -> AsyncIterator[TestCase]:
            yield TestCase(case_id="c1", prompt="p", technique_id="t1")

        async def evaluate(self, case: TestCase, response: TargetResponse) -> EvalResult:
            return EvalResult(
                case_id=case.case_id,
                verdict="blocked",
                confidence=1.0,
                evidence="",
                detection_layer="regex",
            )

    _Module.__name__ = f"Module_{module_id}"
    return _Module


def _make_configurable_module_class(module_id: str, track_instantiation: bool = False):
    """Build a BaseModule subclass whose `__init__` accepts
    `known_system_prompt`/`judge_model` kwargs (mirroring
    `SystemPromptLeakageModule`), for Gap 2 config-threading tests."""

    class _ConfigurableModule(BaseModule):
        id = module_id
        name = f"Configurable module {module_id}"
        owasp_ref = "LLM07:2025"
        instantiated = False

        def __init__(self, known_system_prompt=None, judge_model="openai/gpt-4o-mini"):
            self.known_system_prompt = known_system_prompt
            self.judge_model = judge_model
            if track_instantiation:
                type(self).instantiated = True

        async def generate_cases(self, context: ScanContext) -> AsyncIterator[TestCase]:
            yield TestCase(case_id="c1", prompt="p", technique_id="t1")

        async def evaluate(self, case: TestCase, response: TargetResponse) -> EvalResult:
            return EvalResult(
                case_id=case.case_id,
                verdict="blocked",
                confidence=1.0,
                evidence="",
                detection_layer="regex",
            )

    _ConfigurableModule.__name__ = f"ConfigurableModule_{module_id}"
    return _ConfigurableModule


class _NotAModule:
    """A class that does NOT subclass BaseModule — should be rejected."""


def _fake_entry_point(name: str, loader):
    return SimpleNamespace(name=name, load=loader)


def test_discover_all_returns_classes_without_instantiating(monkeypatch):
    mod_a = _make_module_class("module_a", track_instantiation=True)
    mod_b = _make_module_class("module_b", track_instantiation=True)
    eps = [
        _fake_entry_point("module_a", lambda: mod_a),
        _fake_entry_point("module_b", lambda: mod_b),
    ]
    monkeypatch.setattr("llmsec.plugins.registry.entry_points", lambda group=None: eps)

    registry = PluginRegistry()
    discovered = registry.discover_all()

    assert discovered == {"module_a": mod_a, "module_b": mod_b}
    assert mod_a.instantiated is False
    assert mod_b.instantiated is False


def test_discover_all_skips_broken_entry_point_and_logs_warning(monkeypatch, caplog):
    def _broken_loader():
        raise RuntimeError("boom")

    good = _make_module_class("good_module")
    eps = [
        _fake_entry_point("broken_module", _broken_loader),
        _fake_entry_point("good_module", lambda: good),
    ]
    monkeypatch.setattr("llmsec.plugins.registry.entry_points", lambda group=None: eps)

    registry = PluginRegistry()
    with caplog.at_level(logging.WARNING):
        discovered = registry.discover_all()

    assert discovered == {"good_module": good}
    assert "broken_module" in caplog.text


def test_discover_all_skips_non_basemodule_subclass_and_logs_warning(monkeypatch, caplog):
    good = _make_module_class("good_module")
    eps = [
        _fake_entry_point("not_a_module", lambda: _NotAModule),
        _fake_entry_point("good_module", lambda: good),
    ]
    monkeypatch.setattr("llmsec.plugins.registry.entry_points", lambda group=None: eps)

    registry = PluginRegistry()
    with caplog.at_level(logging.WARNING):
        discovered = registry.discover_all()

    assert discovered == {"good_module": good}
    assert "not_a_module" in caplog.text


def test_discover_all_logs_warning_on_duplicate_name_collision(monkeypatch, caplog):
    first = _make_module_class("dup_module")
    second = _make_module_class("dup_module")
    eps = [
        _fake_entry_point("dup_module", lambda: first),
        _fake_entry_point("dup_module", lambda: second),
    ]
    monkeypatch.setattr("llmsec.plugins.registry.entry_points", lambda group=None: eps)

    registry = PluginRegistry()
    with caplog.at_level(logging.WARNING):
        discovered = registry.discover_all()

    assert len(discovered) == 1
    assert discovered["dup_module"] is second
    assert "dup_module" in caplog.text
    assert "duplicate" in caplog.text.lower() or "collision" in caplog.text.lower()


def test_load_allowed_instantiates_only_the_allowlisted_module(monkeypatch):
    mod_a = _make_module_class("module_a", track_instantiation=True)
    mod_b = _make_module_class("module_b", track_instantiation=True)
    eps = [
        _fake_entry_point("module_a", lambda: mod_a),
        _fake_entry_point("module_b", lambda: mod_b),
    ]
    monkeypatch.setattr("llmsec.plugins.registry.entry_points", lambda group=None: eps)

    registry = PluginRegistry()
    loaded = registry.load_allowed(["module_a"])

    assert set(loaded.keys()) == {"module_a"}
    assert isinstance(loaded["module_a"], mod_a)
    assert mod_a.instantiated is True
    assert mod_b.instantiated is False


def test_load_allowed_falls_back_to_builtin_module_ids_when_none(monkeypatch):
    builtin_id = next(iter(BUILTIN_MODULE_IDS))
    builtin_cls = _make_module_class(builtin_id, track_instantiation=True)
    other_cls = _make_module_class("some_other_module", track_instantiation=True)
    eps = [
        _fake_entry_point(builtin_id, lambda: builtin_cls),
        _fake_entry_point("some_other_module", lambda: other_cls),
    ]
    monkeypatch.setattr("llmsec.plugins.registry.entry_points", lambda group=None: eps)

    registry = PluginRegistry()
    loaded = registry.load_allowed(None)

    assert set(loaded.keys()) == {builtin_id}
    assert builtin_cls.instantiated is True
    assert other_cls.instantiated is False


def test_load_allowed_with_unknown_id_logs_error_and_does_not_raise(monkeypatch, caplog):
    eps = []
    monkeypatch.setattr("llmsec.plugins.registry.entry_points", lambda group=None: eps)

    registry = PluginRegistry()
    with caplog.at_level(logging.ERROR):
        loaded = registry.load_allowed(["not_installed_id"])

    assert loaded == {}
    assert "not_installed_id" in caplog.text


def test_load_allowed_threads_module_config_kwargs_to_instance(monkeypatch):
    cfg_module = _make_configurable_module_class("system_prompt_leakage")
    eps = [_fake_entry_point("system_prompt_leakage", lambda: cfg_module)]
    monkeypatch.setattr("llmsec.plugins.registry.entry_points", lambda group=None: eps)

    registry = PluginRegistry()
    loaded = registry.load_allowed(
        ["system_prompt_leakage"],
        module_config={
            "system_prompt_leakage": {
                "known_system_prompt": "GROUND TRUTH",
                "judge_model": "custom/model",
            }
        },
    )

    instance = loaded["system_prompt_leakage"]
    assert instance.known_system_prompt == "GROUND TRUTH"
    assert instance.judge_model == "custom/model"


def test_load_allowed_ignores_config_for_non_allowlisted_module(monkeypatch):
    allowed_cls = _make_configurable_module_class("module_a", track_instantiation=True)
    other_cls = _make_configurable_module_class("module_b", track_instantiation=True)
    eps = [
        _fake_entry_point("module_a", lambda: allowed_cls),
        _fake_entry_point("module_b", lambda: other_cls),
    ]
    monkeypatch.setattr("llmsec.plugins.registry.entry_points", lambda group=None: eps)

    registry = PluginRegistry()
    loaded = registry.load_allowed(
        ["module_a"],
        module_config={
            "module_a": {"known_system_prompt": "A"},
            "module_b": {"known_system_prompt": "SHOULD NEVER BE APPLIED"},
        },
    )

    assert set(loaded.keys()) == {"module_a"}
    assert other_cls.instantiated is False


def test_load_allowed_drops_unknown_kwargs_without_raising(monkeypatch):
    mod_a = _make_module_class("module_a", track_instantiation=True)
    eps = [_fake_entry_point("module_a", lambda: mod_a)]
    monkeypatch.setattr("llmsec.plugins.registry.entry_points", lambda group=None: eps)

    registry = PluginRegistry()
    loaded = registry.load_allowed(
        ["module_a"],
        module_config={"module_a": {"not_a_real_param": "whatever"}},
    )

    assert set(loaded.keys()) == {"module_a"}
    assert mod_a.instantiated is True


def test_load_allowed_drops_literal_self_kwarg_without_raising(monkeypatch):
    """Regression test (IN-04): `inspect.signature(cls.__init__).parameters`
    includes "self" (the unbound `__init__`'s first parameter). A
    `module_config` entry with a literal "self" key must be dropped like
    any other unaccepted kwarg — never forwarded to `cls(self=..., ...)`,
    which would raise `TypeError: multiple values for argument 'self'`."""
    mod_a = _make_module_class("module_a", track_instantiation=True)
    eps = [_fake_entry_point("module_a", lambda: mod_a)]
    monkeypatch.setattr("llmsec.plugins.registry.entry_points", lambda group=None: eps)

    registry = PluginRegistry()
    loaded = registry.load_allowed(
        ["module_a"],
        module_config={"module_a": {"self": "whatever"}},
    )

    assert set(loaded.keys()) == {"module_a"}
    assert mod_a.instantiated is True


# --- Plan 02-08: prompt_injection discoverability + independent selectability ---
# These tests deliberately do NOT monkeypatch `entry_points` — they exercise
# the REAL installed distribution's `[project.entry-points."llmsec.modules"]`
# table (pyproject.toml), proving the actual packaging/registration works,
# not just the registry's internal dict-handling logic covered above.


def test_prompt_injection_discoverable():
    """`prompt_injection` is discoverable via the real installed entry
    points, as a class (never instantiated), with the right identity."""
    from llmsec.modules.prompt_injection import PromptInjectionModule

    registry = PluginRegistry()
    discovered = registry.discover_all()

    assert "prompt_injection" in discovered
    cls = discovered["prompt_injection"]
    assert isinstance(cls, type)
    assert cls is PromptInjectionModule
    assert cls.id == "prompt_injection"
    assert cls.owasp_ref == "LLM01:2025"


def test_discover_all_real_builtins_returns_classes_not_instances():
    """Both real built-ins come back as classes from discover_all(),
    preserving the D-10 discovery/instantiation split against the actual
    installed package, not just fakes."""
    registry = PluginRegistry()
    discovered = registry.discover_all()

    for module_id in ("system_prompt_leakage", "prompt_injection"):
        assert module_id in discovered
        assert isinstance(discovered[module_id], type)


def test_load_allowed_prompt_injection_alone_excludes_leakage(monkeypatch):
    """Allowlisting only `prompt_injection` loads exactly that module —
    `system_prompt_leakage` is never instantiated."""
    import llmsec.modules.prompt_injection as pi_mod
    import llmsec.modules.system_prompt_leakage as spl_mod

    leakage_instantiated = False
    original_leakage_init = spl_mod.SystemPromptLeakageModule.__init__

    def tracking_leakage_init(self, *args, **kwargs):
        nonlocal leakage_instantiated
        leakage_instantiated = True
        original_leakage_init(self, *args, **kwargs)

    monkeypatch.setattr(
        spl_mod.SystemPromptLeakageModule, "__init__", tracking_leakage_init
    )

    registry = PluginRegistry()
    loaded = registry.load_allowed(["prompt_injection"])

    assert set(loaded.keys()) == {"prompt_injection"}
    assert isinstance(loaded["prompt_injection"], pi_mod.PromptInjectionModule)
    assert leakage_instantiated is False


def test_load_allowed_system_prompt_leakage_alone_excludes_prompt_injection(
    monkeypatch,
):
    """Allowlisting only `system_prompt_leakage` loads exactly that
    module — `prompt_injection` is never instantiated."""
    import llmsec.modules.prompt_injection as pi_mod
    import llmsec.modules.system_prompt_leakage as spl_mod

    injection_instantiated = False
    original_injection_init = pi_mod.PromptInjectionModule.__init__

    def tracking_injection_init(self, *args, **kwargs):
        nonlocal injection_instantiated
        injection_instantiated = True
        original_injection_init(self, *args, **kwargs)

    monkeypatch.setattr(
        pi_mod.PromptInjectionModule, "__init__", tracking_injection_init
    )

    registry = PluginRegistry()
    loaded = registry.load_allowed(["system_prompt_leakage"])

    assert set(loaded.keys()) == {"system_prompt_leakage"}
    assert isinstance(loaded["system_prompt_leakage"], spl_mod.SystemPromptLeakageModule)
    assert injection_instantiated is False


def test_load_allowed_none_loads_both_real_builtins():
    """With no allowlist configured, every built-in module loads — none of
    their presence depends on another.

    Updated in 03-01 (Rule 1 auto-fix) from a two-member set to three:
    registering `pii_exfiltration` (D-39) grows `BUILTIN_MODULE_IDS`.
    Updated again in 04-01 (Rule 1 auto-fix) from three to four:
    registering `insecure_output` (D-42) grows it further."""
    registry = PluginRegistry()
    loaded = registry.load_allowed(None)

    assert set(loaded.keys()) == {
        "system_prompt_leakage",
        "prompt_injection",
        "pii_exfiltration",
        "insecure_output",
    }
