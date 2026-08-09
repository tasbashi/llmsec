"""`AgentRole` protocol + registry (D-65).

Mirrors `plugins/registry.py`'s allowlist-as-boundary shape: `get_role()`
refuses an unregistered name rather than constructing anything. Adding a
role in a later plan (Analyst/Recon/Crescendo Orchestrator) is a registry
entry plus a pinned prompt in a new `roles/<name>.py` file -- never a
`graph.py` control-flow edit (D-65).

This package (`llmsec.attacker.roles`) DOES import the `langchain`/
`deepagents` stack (unlike `llmsec.attacker.config`/`llmsec.attacker.state`,
which are deliberately deep-extra-free so the package root stays
importable without the `[deep]` extra). Every caller that reaches into
`llmsec.attacker.roles.*` must therefore do so only AFTER
`require_deep_extra()` has already cleared -- `runner.py` enforces this by
deferring its own import of this subpackage until after that gate.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from llmsec.attacker.config import (
    AttackerConfig,
    ResolvedAttackerSettings,
    resolve_role_api_key,
    resolve_role_model,
    resolve_role_tuning,
)
from llmsec.attacker.state import CampaignState

logger = logging.getLogger(__name__)


@runtime_checkable
class AgentRole(Protocol):
    """One attacker-team role: a name, a compiled-agent factory, its
    structured-output schema, and a brief-builder reading only
    `CampaignState` (never privileged context, D-87)."""

    name: str
    output_schema: type[BaseModel]
    #: 05-08/D-95: the role's own pinned, versioned system-prompt constant
    #: (`attacker/prompts.py`) -- exposed here (additive, every existing
    #: `_XRole` class already composes one into its `build()` call) so a
    #: registry-driven test can assert every registered role's anti-feature/
    #: sandbox clause presence by ITERATING `ROLE_REGISTRY`, never by
    #: hand-listing prompt constants -- a sixth role added later cannot
    #: skip the assertion by omission.
    system_prompt: str

    def build(
        self,
        settings: ResolvedAttackerSettings,
        cfg: AttackerConfig,
        *,
        model: str | BaseChatModel | None = None,
    ) -> Any:
        """Return a compiled `CompiledStateGraph` for this role.

        `model`, when given, overrides the resolved production model --
        the sole test-injection hook every role factory supports, so no
        attacker test ever needs real credentials or a live network call.
        """
        ...

    def brief(self, state: CampaignState) -> str:
        """Build this role's invocation message from `CampaignState` alone."""
        ...


#: Every registered role, keyed by its own `.name`. Never iterated in
#: dict/set order for anything user-visible -- callers that need a fixed
#: order use `llmsec.attacker.state.ROLE_NAMES` instead.
ROLE_REGISTRY: dict[str, AgentRole] = {}


def register_role(role: AgentRole) -> None:
    """Register `role` under its own `.name`.

    Re-registering an already-registered name overwrites, with a warning
    (mirrors `plugins/registry.py`'s duplicate-entry_points-name
    discipline) -- never silent.
    """
    if role.name in ROLE_REGISTRY:
        logger.warning(
            "Re-registering attacker role %r, overwriting the prior registration", role.name
        )
    ROLE_REGISTRY[role.name] = role


def get_role(name: str) -> AgentRole:
    """Return the registered role named `name`.

    Refuses an unregistered name rather than constructing anything (D-65's
    registry-is-the-only-dispatch-surface discipline, mirroring
    `plugins/registry.py`'s allowlist-as-boundary shape).
    """
    try:
        return ROLE_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown attacker role {name!r}; registered roles: {sorted(ROLE_REGISTRY)}"
        ) from None


def build_role_chat_model(
    settings: ResolvedAttackerSettings,
    cfg: AttackerConfig,
    role: str,
    model: str | BaseChatModel | None = None,
) -> str | BaseChatModel:
    """Resolve the `model=` argument for `role`'s `create_deep_agent()` call.

    `model`, when given, is returned verbatim -- the sole test-injection
    hook every role factory (`build_strategist_agent`/`build_mutator_agent`)
    supports. Otherwise resolves `cfg`'s per-role model/api-key/tuning
    (D-67/D-89) via `attacker/config.py`'s existing `resolve_role_*`
    helpers and constructs a real `BaseChatModel` via
    `langchain.chat_models.init_chat_model` -- shared here so no role file
    duplicates this resolution logic.
    """
    if model is not None:
        return model

    from langchain.chat_models import init_chat_model

    model_name = resolve_role_model(cfg, role)
    api_key = resolve_role_api_key(cfg, role)
    temperature, max_tokens = resolve_role_tuning(cfg, role)
    kwargs: dict[str, Any] = {"temperature": temperature, "max_tokens": max_tokens}
    if api_key is not None:
        kwargs["api_key"] = api_key
    return init_chat_model(model_name, **kwargs)
