"""PluginRegistry — the single trusted chokepoint for module loading (D-10).

`discover_all()` reads every `llmsec.modules` entry_points advertisement and
returns the classes WITHOUT instantiating any of them. `load_allowed()` is
the ONLY method that instantiates modules, and only those explicitly
allowlisted (or `BUILTIN_MODULE_IDS` when no allowlist is configured).

This structural split must never be collapsed into a single "load
everything found" method — that is exactly the arbitrary-code-execution
trap D-10 exists to prevent (any pip-installed package advertising the
`llmsec.modules` entry_points group must not auto-run).
"""

from __future__ import annotations

import inspect
import logging
from importlib.metadata import entry_points
from typing import Any

from llmsec.plugins.base import BaseModule

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "llmsec.modules"

# Built-in modules always exist even if the package is somehow installed
# without its own entry_points registering (defense in depth for D-11).
BUILTIN_MODULE_IDS = {
    "system_prompt_leakage",
    "prompt_injection",
    "pii_exfiltration",
    "insecure_output",
}


class PluginRegistry:
    def discover_all(self) -> dict[str, type[BaseModule]]:
        """Return every entry_points-advertised module class, WITHOUT
        instantiating or trusting any of them. Used by `list-modules` to
        show what's installed vs what's actually allowlisted."""
        discovered: dict[str, type[BaseModule]] = {}
        eps = entry_points(group=ENTRY_POINT_GROUP)
        for ep in eps:
            try:
                cls = ep.load()
            except Exception as exc:  # a broken plugin must never crash discovery
                logger.warning("Failed to load module entry point %r: %s", ep.name, exc)
                continue
            if not (isinstance(cls, type) and issubclass(cls, BaseModule)):
                logger.warning("Entry point %r does not subclass BaseModule; skipping", ep.name)
                continue
            if ep.name in discovered:
                logger.warning(
                    "Duplicate entry_points name %r — keeping last-discovered class",
                    ep.name,
                )
            discovered[ep.name] = cls
        return discovered

    def load_allowed(
        self,
        allowlist: list[str] | None,
        module_config: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, BaseModule]:
        """Instantiate only modules explicitly allowlisted
        (config.enabled_modules), or the built-in defaults if no allowlist
        is configured (D-10). This is the ONLY method that calls `cls()`.

        `module_config` optionally threads per-module operator config
        (e.g. `known_system_prompt`, `judge_model`) into instantiation —
        but ONLY after a module id has already cleared the allowlist gate
        above. A `module_config` entry for a non-allowlisted/undiscovered
        id is never read and never triggers instantiation; config
        threading is strictly downstream of the allowlist gate, never a
        second load path (D-10).
        """
        discovered = self.discover_all()
        # WR-03: preserve the operator's configured order (de-duplicated,
        # first-occurrence-wins) instead of routing through a `set`, whose
        # str-key iteration order depends on per-process hash
        # randomization. Falls back to a deterministic sorted order (rather
        # than `BUILTIN_MODULE_IDS` itself, a `set`) when no allowlist is
        # configured. This keeps `loaded`'s key order — and therefore
        # `ScanReport.module_ids` — reproducible across process invocations.
        effective_allowlist = (
            list(dict.fromkeys(allowlist)) if allowlist else sorted(BUILTIN_MODULE_IDS)
        )
        loaded: dict[str, BaseModule] = {}
        for module_id in effective_allowlist:
            cls = discovered.get(module_id)
            if cls is None:
                logger.error(
                    "Module %r is allowlisted but not discoverable via entry_points",
                    module_id,
                )
                continue
            requested = (module_config or {}).get(module_id, {})
            try:
                # IN-04: exclude "self" — `inspect.signature(cls.__init__)`
                # includes the unbound `__init__`'s first parameter, so a
                # module_config entry with a literal "self" key would
                # otherwise be silently accepted and forwarded as
                # `cls(self=..., ...)`, producing a confusing
                # "multiple values for argument 'self'" TypeError instead
                # of being dropped like every other unrecognized kwarg.
                accepted_params = set(inspect.signature(cls.__init__).parameters) - {"self"}
            except (TypeError, ValueError):  # pragma: no cover — defensive
                accepted_params = set()
            filtered_kwargs = {}
            for key, value in requested.items():
                if key in accepted_params:
                    filtered_kwargs[key] = value
                else:
                    logger.debug(
                        "Dropping unaccepted config kwarg %r for module %r", key, module_id
                    )
            try:
                loaded[module_id] = cls(**filtered_kwargs)
            except Exception as exc:
                logger.error("Failed to instantiate module %r: %s", module_id, exc)
        return loaded
