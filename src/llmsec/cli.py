"""The `llmsec` Typer CLI — `scan`, `report`, `list-modules` (CLI-01/02/03).

Thin, user-facing wrapper around `config.py`, `auth_gate.py`, `api.run_scan()`,
`reporting/`, and `plugins/registry.py`. This module is the console-script
target declared in `pyproject.toml`'s `[project.scripts] llmsec = "llmsec.cli:app"`
(CORE-01's CLI half).

The asyncio event loop is started only at this CLI boundary — never inside
`api.py`, `orchestrator.py`, or any adapter/module code, per RESEARCH.md's
"Anti-Patterns to Avoid". Library callers (e.g.
`import llmsec; await llmsec.run_scan(...)`) manage their own event loop.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Annotated

import click
import typer
from pydantic import ValidationError

from llmsec import api
from llmsec.attacker import AttackerExtraNotInstalled, require_deep_extra
from llmsec.auth_gate import AuthorizationDeclined
from llmsec.config import load_config
from llmsec.plugins.registry import BUILTIN_MODULE_IDS, PluginRegistry
from llmsec.reporting.json_reporter import JsonReporter, load_report
from llmsec.reporting.markdown_reporter import MarkdownReporter

logger = logging.getLogger(__name__)

app = typer.Typer(name="llmsec")


@app.command("scan")
def scan_cmd(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("llmsec.config.yaml"),
    max_concurrency: Annotated[int | None, typer.Option("--max-concurrency")] = None,  # None = not passed
    output_dir: Annotated[str | None, typer.Option("--output-dir")] = None,
    yes_i_am_authorized: Annotated[bool, typer.Option("--yes-i-am-authorized")] = False,
    deep: Annotated[
        bool,
        typer.Option(
            "--deep",
            help="Enable deep mode: an Attacker LLM team dynamically mutates failed "
            "static payloads (requires the [deep] extra). Never the default.",
        ),
    ] = False,
    quick: Annotated[
        bool,
        typer.Option(
            "--quick", help="Static-payloads-only scan -- the default behavior either way."
        ),
    ] = False,
    deep_profile: Annotated[
        str | None,
        typer.Option(
            "--deep-profile",
            click_type=click.Choice(["light", "standard", "thorough"]),
            help="Deep-mode intensity preset. Only valid together with --deep.",
        ),
    ] = None,
    resume: Annotated[
        str | None,
        typer.Option(
            "--resume",
            help="Resume a previously checkpointed --deep campaign by scan_id "
            "(requires --deep and a configured attacker.checkpoint_dir). Continues "
            "under the campaign's original budget cap unless --budget-top-up-usd "
            "is also given; a changed configuration since checkpointing refuses "
            "rather than running with a silently different setup.",
        ),
    ] = None,
    budget_top_up_usd: Annotated[
        float | None,
        typer.Option(
            "--budget-top-up-usd",
            help="Explicitly raise a --resume'd campaign's budget cap to this value. "
            "Prior spend is printed before the new ceiling takes effect. Only valid "
            "together with --resume -- never silently applied.",
        ),
    ] = None,
) -> None:
    """Load config, run the authorization gate, and scan the configured target."""
    # CLI-04/D-93: --deep and --quick are mutually exclusive, and a profile
    # preset only makes sense in deep mode -- both rejected with a clean
    # message BEFORE load_config is ever called (CORE-02 discipline: no
    # config/adapter/scan cost incurred for an invalid flag combination).
    if deep and quick:
        typer.echo("--deep and --quick cannot both be passed -- choose one.", err=True)
        raise typer.Exit(code=1)
    if deep_profile is not None and not deep:
        typer.echo(
            "--deep-profile only applies in deep mode -- pass --deep as well.", err=True
        )
        raise typer.Exit(code=1)
    # 05-06 Task 3/D-75: --resume is rejected up front, before load_config,
    # exactly like the two checks above -- a resumed campaign is always a
    # deep-mode concept, and a top-up value with no --resume to apply it to
    # is a silently-ignored flag waiting to confuse an operator.
    if resume is not None and not deep:
        typer.echo("--resume requires --deep.", err=True)
        raise typer.Exit(code=1)
    if budget_top_up_usd is not None and resume is None:
        typer.echo("--budget-top-up-usd only applies together with --resume.", err=True)
        raise typer.Exit(code=1)

    # D-88: only the flags the operator actually passed enter the override
    # dict, mirroring the existing max_concurrency/output_dir comprehension's
    # discipline -- an unset flag must never clobber a YAML-configured
    # `attacker:` block back to the schema default (CORE-02).
    attacker_override: dict[str, object] = {}
    if deep:
        attacker_override["enabled"] = True
    elif quick:
        # CORE-02/D-93: an explicitly-passed --quick is an explicit CLI
        # override and must win over a YAML-configured `attacker.enabled:
        # true`, exactly like --deep does in the other direction -- CLI
        # overrides > YAML > env. Only an *unset* flag must not clobber
        # YAML (the case the comment below still covers); a *passed*
        # --quick is not unset.
        attacker_override["enabled"] = False
    if deep_profile is not None:
        attacker_override["profile"] = deep_profile

    cli_overrides: dict[str, object] = {
        k: v
        for k, v in {"max_concurrency": max_concurrency, "output_dir": output_dir}.items()
        if v is not None
    }
    if attacker_override:
        cli_overrides["attacker"] = attacker_override

    try:
        cfg = load_config(config, cli_overrides)
    except ValidationError as exc:
        typer.echo(f"Invalid config at {config}: {exc}", err=True)
        raise typer.Exit(code=1)

    if resume is not None:
        # 05-06 Task 3/D-75: --resume bypasses the normal static-batch +
        # fresh-campaign flow entirely -- the restored checkpoint ALREADY
        # holds the case queue and campaign progress, so there is nothing
        # for `api.run_scan()`'s own orchestrator pass to redo. Deferred
        # imports: this branch is the only cli.py code path that touches
        # `resume_attacker_campaign()`/`_build_adapter()` directly, reached
        # only after the --deep/--resume validation above has cleared.
        require_deep_extra()
        from llmsec.api import _build_adapter
        from llmsec.attacker.runner import (
            CampaignResult,
            ConfigFingerprintMismatchError,
            UnknownScanIdError,
            resume_attacker_campaign,
        )

        if cfg.attacker is None or not cfg.attacker.enabled:
            typer.echo(
                "--resume requires an `attacker:` block with enabled: true in config "
                f"(scan_id={resume!r}).",
                err=True,
            )
            raise typer.Exit(code=1)

        def _announce_prior_spend(prior_spent: float, original_cap: float, effective_cap: float) -> None:
            # D-75.1: printed BEFORE any further spend decision, using the
            # SAME labelled wording `render_cost_notice()`'s own budget
            # line uses, so the two surfaces never drift apart.
            typer.echo(
                f"Resuming scan {resume}: prior spend ${prior_spent:.2f} of "
                f"${original_cap:.2f} original cap."
            )
            if effective_cap != original_cap:
                typer.echo(
                    f"Budget top-up applied: cap raised to ${effective_cap:.2f} "
                    f"(${max(effective_cap - prior_spent, 0.0):.2f} remaining)."
                )
            else:
                typer.echo(f"${max(effective_cap - prior_spent, 0.0):.2f} remaining under the original cap.")

        effective_module_ids = cfg.enabled_modules or list(BUILTIN_MODULE_IDS)
        module_config = {
            module_id: {
                "known_system_prompt": cfg.known_system_prompt,
                "judge_model": cfg.judge_model,
                "judge_api_key_env": cfg.judge_api_key_env,
            }
            for module_id in effective_module_ids
        }
        modules = PluginRegistry().load_allowed(cfg.enabled_modules, module_config=module_config)

        try:
            adapter = _build_adapter(cfg)
        except ValueError as exc:
            typer.echo(f"Invalid target configuration: {exc}", err=True)
            raise typer.Exit(code=1)

        async def _do_resume() -> CampaignResult:
            try:
                return await resume_attacker_campaign(
                    config=cfg,
                    adapter=adapter,
                    modules=modules,
                    scan_id=resume,
                    budget_top_up_usd=budget_top_up_usd,
                    on_prior_spend=_announce_prior_spend,
                )
            finally:
                await adapter.close()

        try:
            campaign_result = asyncio.run(_do_resume())
        except (UnknownScanIdError, ConfigFingerprintMismatchError) as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1)

        final_state = campaign_result.final_state
        typer.echo(
            f"Resumed scan {resume} complete: {len(campaign_result.eval_results)} "
            f"case(s) evaluated this resume, termination="
            f"{final_state.get('termination_reason')}."
        )
        for note in campaign_result.limitations:
            typer.echo(f"Note: {note}", err=True)
        return

    try:
        # D-74: checked before asyncio.run()/any adapter construction, so a
        # missing [deep] extra fails clean with NO target request ever made
        # -- never a silent fall-back to static-only.
        if deep:
            require_deep_extra()
            # D-82: the operator sees a labelled typical/worst-case range
            # beside the hard cap BEFORE any spend -- `render_cost_notice()`
            # is echoed here, strictly before `asyncio.run(api.run_scan(...))`
            # is ever entered, so no target request or attacker call has
            # occurred when the operator reads it. Deferred import: this is
            # the only cli.py code path that touches the `[deep]`-extra-only
            # `attacker.budget` module, reached only after the gate above
            # has already cleared.
            from llmsec.attacker.budget import estimate_campaign_cost, render_cost_notice
            from llmsec.attacker.config import resolve_settings

            if cfg.attacker is not None:
                discovered = PluginRegistry().discover_all()
                effective_ids = set(cfg.enabled_modules) if cfg.enabled_modules else BUILTIN_MODULE_IDS
                # Display-only discovery never instantiates a module
                # (T-01-19/D-10, same discipline `list_modules_cmd` already
                # follows) -- `uses_attacker_llm` is read off the CLASS.
                queue_size = sum(
                    1
                    for module_id, module_cls in discovered.items()
                    if module_id in effective_ids and getattr(module_cls, "uses_attacker_llm", False)
                )
                estimate = estimate_campaign_cost(
                    resolve_settings(cfg.attacker), queue_size, cfg.attacker.model
                )
                typer.echo(render_cost_notice(estimate))
        # This is the ONLY place `asyncio`'s event loop is started to drive
        # the scan pipeline — `api.run_scan()` itself never starts its own.
        report = asyncio.run(api.run_scan(cfg, bypass_flag=yes_i_am_authorized))
    except AttackerExtraNotInstalled as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)
    except AuthorizationDeclined as exc:
        # A declined/non-interactive authorization must surface as a clean
        # CLI message, never an unhandled traceback (T-01-20).
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)
    except ValueError as exc:
        # `_build_adapter()` raises `ValueError` on missing/invalid required
        # target fields (e.g. `raw_llm` without `model`/`api_key_env`) — must
        # surface as a clean CLI message, never an unhandled traceback,
        # same as the AuthorizationDeclined case above (WR-01).
        typer.echo(f"Invalid target configuration: {exc}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Scan {report.scan_id} complete: {len(report.findings)} finding(s).")


@app.command("report")
def report_cmd(
    scan_id: Annotated[str, typer.Argument()],
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("./llmsec_reports"),
    fmt: Annotated[
        str, typer.Option("--format", click_type=click.Choice(["json", "markdown"]))
    ] = "markdown",
) -> None:
    """Regenerate a report from a previously persisted `scan_<id>.json` — no re-scan."""
    report_path = output_dir / f"scan_{scan_id}.json"
    try:
        report = load_report(report_path)
    except FileNotFoundError:
        typer.echo(f"No persisted scan found for scan_id={scan_id!r} at {report_path}.", err=True)
        raise typer.Exit(code=1)
    except ValidationError as exc:
        typer.echo(f"Persisted scan at {report_path} is invalid: {exc}", err=True)
        raise typer.Exit(code=1)

    reporter = {"json": JsonReporter(), "markdown": MarkdownReporter()}[fmt]
    # Report regeneration is a separate, short-lived event-loop invocation
    # from `scan_cmd`'s — neither command's loop persists across the other.
    path = asyncio.run(reporter.write(report, output_dir))
    typer.echo(f"Report written to {path}")


@app.command("list-modules")
def list_modules_cmd(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("llmsec.config.yaml"),
) -> None:
    """List every module `discover_all()` finds, annotated with its effective
    allowlist status. Never instantiates a module for display purposes
    (T-01-19, D-10) — only `discover_all()` is called here, never
    `load_allowed()`.
    """
    discovered = PluginRegistry().discover_all()

    try:
        cfg = load_config(config, {})
        effective_allowlist = set(cfg.enabled_modules) if cfg.enabled_modules else BUILTIN_MODULE_IDS
    except Exception as exc:
        # A missing/invalid config must never crash a display-only command —
        # fall back to the same built-in default `load_allowed()` would use.
        # Log at WARNING so an operator can distinguish "no config file
        # present" from "config file present but broken" (WR-03), matching
        # every other degrade path in this codebase.
        logger.warning(
            "Failed to load %s for list-modules allowlist display: %s", config, exc
        )
        effective_allowlist = BUILTIN_MODULE_IDS

    if not discovered:
        typer.echo("No modules discovered.")
        return

    # Deterministic, diffable output across repeated runs (CLI-03).
    for module_id in sorted(discovered.keys()):
        status = "[loaded]" if module_id in effective_allowlist else "[not allowlisted]"
        typer.echo(f"{module_id} {status}")
