# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`llmsec` (`llm-security-tester`) — an automated vulnerability scanner for applications that
integrate LLMs, covering all ten OWASP LLM Top 10 (2025) categories with static payload corpora
and an opt-in Attacker LLM team for dynamic red-teaming (`--deep`). Ships as a CLI (`llmsec`),
an importable library (`await llmsec.run_scan(config)`), and an optional local web dashboard
(`llmsec dashboard`) for browsing reports and triggering scans from a browser.

Package layout is src-layout: `src/llmsec/`. Python >= 3.10.

## Commands

```bash
# Setup (editable install with dev tooling)
uv pip install -e ".[dev]"
# Optional NER tier for pii_exfiltration — TWO steps, the extra alone is not enough:
uv pip install -e ".[pii-ner]" && python -m spacy download en_core_web_lg   # ~746MB
# Optional extras for other opt-in surfaces:
uv pip install -e ".[deep]"           # attacker-LLM team for `llmsec scan --deep`
uv pip install -e ".[supply-chain]"   # pip-audit/OSV.dev CVE tier for the supply_chain module
uv pip install -e ".[dashboard]"      # FastAPI/uvicorn backend for `llmsec dashboard`

# Fast test suite (golden/live tests auto-excluded via pyproject addopts)
pytest

# Single file / single test
pytest tests/test_orchestrator.py
pytest tests/test_api.py::test_run_scan_redacts_credentials -v
pytest tests/dashboard/ -q   # skipped (not failed) without the [dashboard] extra installed

# Live judge-eval gates — require a real credential (e.g. OPENAI_API_KEY), never run in CI
pytest tests/evaluators/ -m golden -v --tb=short

# Lint + typecheck
ruff check src tests
mypy src

# Run against a target
cp llmsec.config.yaml.example llmsec.config.yaml   # edit target block
llmsec scan --config llmsec.config.yaml             # add --deep for the attacker-LLM team
llmsec list-modules
llmsec report <scan_id> --format markdown
llmsec dashboard --output-dir ./llmsec_reports       # requires [dashboard] + LLMSEC_DASHBOARD_TOKEN
```

`demo-app/` (FastAPI backend + Vite frontend) is a deliberately vulnerable target to scan
against locally; the example config points at its `/chat` endpoint.

## Git workflow

- **Never push `master` to origin** (including tags) without the user explicitly asking for
  that push, in that moment. This holds even right after a local commit/merge/`release:` bump
  that the user did ask for -- committing, merging, and publishing to PyPI are not standing
  authorization to also push `master`. `scripts/publish-pypi.sh` and
  `scripts/sync-public-branch.sh` are exempt for the parts of their own job that inherently
  require a push (`sync-public-branch.sh` pushing `main`; `publish-pypi.sh` uploading to
  PyPI) -- but their own end-of-script reminder to also `git push origin master --tags` is a
  suggestion to relay to the user, never something to act on unprompted.

## Architecture

### Scan pipeline

`cli.py` → `api.run_scan()` → `PluginRegistry.load_allowed()` → `ScanOrchestrator.run()` →
per-module `generate_cases()`/`evaluate()` → `scoring.score()` + redaction → `JsonReporter` /
`MarkdownReporter`.

`api.run_scan()` is the real engine; `cli.py` is a thin wrapper that owns the event loop.
Everything below `cli.py` is `async def` and **never** starts a loop.

### The five contracts in `models.py`

`Verdict` / `TestCase` / `TargetResponse` / `EvalResult` / `Finding` / `ScanReport` /
`ScanContext` are the one shared vocabulary. Nothing downstream may define a second verdict
scheme or an ad-hoc result type. Verdicts are four-tier (`blocked` / `partial_leak` /
`full_compromise` / `uncertain`), never binary.

### Modules (the plugin layer)

A test module subclasses `BaseModule` (`plugins/base.py`) and implements `generate_cases()` +
`evaluate()`. Registration is via the `llmsec.modules` entry_points group in `pyproject.toml`.
All ten OWASP LLM Top 10 (2025) categories are built in, each independently selectable via
`enabled_modules` in config: `system_prompt_leakage` (LLM07), `prompt_injection` (LLM01),
`pii_exfiltration` (LLM02), `insecure_output` (LLM05), `supply_chain` (LLM03),
`data_poisoning` (LLM04), `unbounded_consumption` (LLM10),
`vector_embedding_weaknesses` (LLM08), `excessive_agency` (LLM06), `misinformation` (LLM09).

Payloads live in versioned YAML corpora (`src/llmsec/modules/payloads/*.yaml`), validated by
`payloads/schema.py`. Each corpus entry's `technique_family` comes from a **closed** enum
per module family (`TechniqueFamily`, `PiiAttackVector`, `OutputVulnerabilityClass`,
`PoisoningTechniqueVector`, `SupplyChainTechniqueVector`, `ConsumptionTechniqueVector`,
`AgencyClass`, `MisinformationTechniqueVector`) — these are deliberately kept disjoint and
never merged into one parameterized taxonomy, because each family maps to exactly one
reportable remediation theme.

Two modules (`supply_chain`, `data_poisoning`) don't fit the request/response probing contract
and instead implement `BaseModule.run_standalone_audit()` (an ABC-default async-generator hook,
Phase 6) or `run_direct_probe()` (a retry-exempt single-shot dispatch hook, Phase 7) — both
additive, both leaving `orchestrator.py` untouched. `supply_chain`'s CVE/SBOM tier
(pip-audit + OSV.dev) is gated behind the optional `[supply-chain]` extra.

### Adapters

`TargetAdapter` ABC (`adapters/base.py`) with two implementations: `LLMApiAdapter` (litellm,
raw LLM APIs) and `HttpAppAdapter` (httpx, custom HTTP apps). One adapter instance is
constructed by `run_scan()` and shared across all concurrent cases — adapters hold no
per-call mutable state. Both are byte-for-byte pinned since Phase 2 (D-73) and never touched
by the attacker-team or dashboard layers.

### Layered detection

Modules resolve verdicts through cheap deterministic tiers before paying for an LLM judge:

- `prompt_injection`: canary decode-then-match → judge
- `pii_exfiltration`: canary echo → regex/Luhn → optional NER → judge (four tiers)
- `insecure_output`: refusal fast-path → 14-class regex library → judge
- `supply_chain`: deterministic package-name extraction (negation-aware) → index lookup → judge
- `unbounded_consumption`: token/latency threshold scoring (no judge call)

`EvalResult.detection_layer` records which tier actually fired and is threaded all the way
into the report.

### Attacker LLM team (`--deep` mode, `attacker/`)

Opt-in per module (`BaseModule.uses_attacker_llm`), gated behind the `[deep]` extra
(LangChain/LangGraph/DeepAgents). Five roles in `attacker/roles/` (Strategist, Mutator,
Analyst, Recon, Crescendo Orchestrator) run as LangGraph nodes on a fixed round topology
(`attacker/graph.py`) — round control, the technique allowlist gate, and campaign termination
all live in graph edges, never in a role's own reasoning. `attacker/budget.py` tracks a
checkpointed dollar+call-ceiling ledger; `attacker/checkpoint.py` persists a redacting SQLite
checkpointer for `--resume`; `attacker/audit.py` writes a fully-redacted per-scan JSONL trail.
Mutated variants are still scored by the owning module's own unchanged `evaluate()` — no
attacker role ever assigns a `Verdict`.

### Dashboard (`dashboard/`, optional `[dashboard]` extra)

FastAPI backend (`app.py`, `routes.py`) serving a bearer-token-gated `/api/*` router plus a
built React/Vite/Tailwind SPA (source in `dashboard/frontend/`, built output in
`dashboard/static/` — the source tree is excluded from the wheel). `reports.py` handles
read-only report listing/filtering/detail/download; `scans.py` dispatches an authorized
`api.run_scan()` call as a background task and tracks coarse progress
(`queued`/`running`/`scoring`/`complete`/`failed`) in an in-memory per-process store;
`ssrf_guard.py` validates a submitted `http_app` target's resolved address before any
scan/adapter/task is constructed. Every dashboard-triggered scan requires a fresh per-request
`authorization_attested: true` — the process-wide `LLMSEC_AUTHORIZED=1` CLI bypass has no
effect on this path.

## Invariants that must not be broken

These are load-bearing and each was paid for with a review/gap-closure cycle. Read the
surrounding docstring before touching any of them.

- **Plugin loading split (D-10).** `discover_all()` returns classes without instantiating;
  `load_allowed()` is the only method that calls `cls()`, and only for allowlisted ids. Never
  collapse into one "load everything found" method — any pip-installed package advertising
  the entry_points group would then auto-execute.
- **Authorization gate.** `confirm_authorization()` runs inside `api.run_scan()`, not only in
  `cli.py`, so library callers get the same guarantee. A non-interactive terminal is never
  implicit consent; explicit bypass requires `--yes-i-am-authorized` or `LLMSEC_AUTHORIZED=1`.
- **Failure containment (T-01-18).** A per-case transport or `evaluate()` failure degrades to
  a recorded `UNCERTAIN` result — it must never raise and cancel sibling `asyncio.gather`
  tasks. `run()` returns exactly one result per generated case, never fewer.
- **Score before redact.** `score()` runs against raw evidence; redaction happens afterward.
  Never invert.
- **Redaction order.** `redact_pii_match()` must run before `redact_credential_match()`. The
  generic 32+-char catch-all would otherwise mangle a dot-structured secret (JWT) so the
  precise pattern can no longer match the remaining segments, leaving real payload unredacted.
- **Canary exemption is per-literal.** Canary PII values are shown verbatim to prove an echo,
  but only those literals — `redact_all_protecting_literals()` still redacts a real secret
  co-located in the same excerpt, including one whose match span overlaps the literal.
- **Judge prompts are frozen and SHA-256 pinned.** `JUDGE_SYSTEM_PROMPT` (v1.6),
  `INJECTION_JUDGE_SYSTEM_PROMPT`, `PII_JUDGE_SYSTEM_PROMPT`, `OUTPUT_JUDGE_SYSTEM_PROMPT` are
  four separate constants, deliberately never merged into one template, each verified
  byte-for-byte by tests in `tests/detection/`. Editing one requires a version bump and
  re-running the live golden gate. History: v1.2 and v1.7 each fixed one thing and regressed
  another.
- **Config never holds a literal secret (D-08).** Only env-var *names* (`api_key_env`,
  `judge_api_key_env`). Precedence is explicit CLI overrides > `llmsec.config.yaml` > process
  env; `load_config()` must only receive CLI keys the user actually passed, or an unset flag's
  default will clobber YAML.
- **Additive plugin evolution.** `PLUGIN_API_VERSION` has stayed `"1.0"` across four phases.
  New capability arrives as an optional model field with a default (`detection_layer`), an ABC
  *default* method (`send_conversation()`), or a duck-typed hook resolved by `getattr`
  (`should_abort_sequence`) — never as a new abstract method.
- **Honest degradation over silent success.** A tier that could not run says so in
  `ScanReport.limitations` (NER extra missing, concatenated multi-turn transport, simulated
  indirect injection). A clean report must never be indistinguishable from an untested one.
- **The deterministic package-extraction tier defers on negation, it never overrules it
  (CR-01).** `_extract_package_names()` treats a structural install/pin/import match as
  definitive only when no negation cue appears in the fixed lookbehind window before
  `match.start()`; a suppressed span that normalises to a real third-party name makes the
  whole extraction defer to `judge_extract_packages()`, whose
  `PACKAGE_EXTRACTION_JUDGE_SYSTEM_PROMPT` already carries the EXCLUSIONS clause for exactly
  this case; the cue boundaries are hyphen-aware rather than plain `\b` because PyPI names are
  hyphen-separated and a plain word boundary would read the `not` inside
  `totally-fake-pkg-that-does-not-exist-xyz` as a refusal; and the guard is one-directional —
  it may suppress or defer, never add a candidate, and never turn a suppressed real name into a
  definitively-clean result. Pinned by `tests/test_supply_chain.py::TestNegationWindowGuard`.
  Cites D-09 and CR-01.
- **Target adapter stays on its pinned transport (D-73 mitigation 3).** `adapters/llm_api.py`
  and `orchestrator.py` are byte-for-byte unchanged since Phase 2 and never migrate to the
  attacker-side LangChain/LangGraph/DeepAgents stack — `--quick` is provably unaffected by
  the attacker team's existence, extra-installed or not. Verified every phase by
  `tests/attacker/test_target_path_unchanged.py`'s import-statement scan and content-digest
  pin. Any fix belongs in the attacker layer, never in either target-path module.
- **Round control lives in graph topology, never in an agent's reasoning (D-70/D-72).** The
  deep-mode campaign's round cap, early-exhaust reason codes, and termination are all decided
  by `attacker/graph.py`'s conditional edges reading checkpointed state — never by a role
  deciding to keep going. This is what makes the pre-run cost estimate and the hard budget cap
  trustworthy numbers rather than best-effort hopes.
- **The budget ledger lives in checkpointed graph state (D-75).** Never a runner-local
  attribute — retrofitting it later would touch every agent node and every checkpoint write.
  Both attacker-side and target-side spend are counted in the one ledger.
- **Every audit line and every checkpoint byte passes the redaction chokepoint, in the fixed
  order, with no exemptions (D-86).** `attacker/audit.py`'s `redact_audit_text()` and
  `attacker/checkpoint.py`'s `RedactingJsonPlusSerializer` both call
  `redact_credential_match(redact_pii_match(text))` — never the report path's canary-literal
  exemption, since the Analyst quotes the target's raw response to its peers over the same
  channel this redaction protects.
- **No attacker agent assigns a verdict (D-66).** Mutated variants are scored by the owning
  module's own unchanged `evaluate()` — the identical instrument static payloads use. An
  Analyst's `ObservedDefence` is a structured observation only; grep for `Verdict(`/`Finding(`
  in `attacker/graph.py` and find neither.
- **The technique allowlist is enforced at the delegation boundary, never trusted from the
  caller (D-95).** `attacker/graph.py`'s `validate_technique()` runs inside `strategist_node`,
  before either mutation role is ever constructed or invoked — mirrors the plugin registry's
  own allowlist-as-boundary discipline (D-10).
- **Attacker role prompts are versioned and digest-pinned like the judge prompts (D-95).**
  Every `AgentRole.system_prompt` constant in `attacker/prompts.py` carries an explicit version
  string and is verified byte-for-byte by `tests/attacker/test_prompt_pinning.py` — the same
  discipline `JUDGE_SYSTEM_PROMPT` and its three siblings already require, non-negotiable given
  how easily a "cleanup" edit regressed a passing gate before (see `JUDGE_SYSTEM_PROMPT`
  history above).

## Testing conventions

- `pyproject.toml` sets `asyncio_mode = "auto"` and `addopts = "-m 'not golden'"`. Tests
  marked `@pytest.mark.golden` hit real LLM providers and are a pre-release gate, not a
  per-commit check.
- `judge_client` is a module-level Instructor singleton captured at import time. Patch
  `judge_client.chat.completions.create` directly (see `tests/detection/conftest.py`) —
  patching `litellm.acompletion` afterward has no effect on the already-bound client.
- Root `tests/conftest.py` supplies `mock_target_response`, `respx_mock`, and
  `mock_litellm_acompletion`. No test in the fast suite may make a live network call.
- Golden datasets are JSONL under `tests/evaluators/golden/`, loaded at module scope so
  `--collect-only` never needs a credential.
- `tests/dashboard/` guards each module with `pytest.importorskip("fastapi", ...)` — without
  the `[dashboard]` extra installed those tests are skipped, not failed, so a green fast-suite
  run on a bare `[dev]` environment says nothing about dashboard coverage.
- `tests/attacker/` similarly needs the `[deep]` extra; without it those tests error at
  collection (missing `langgraph`/`langchain_core`), a known environment artifact, not a
  regression — see `.planning/milestones/v1.1-phases/11-*/deferred-items.md`.

## Planning artifacts

`.planning/` holds the GSD workflow state for this project: `ROADMAP.md` (milestone summary,
links into archives), `STATE.md` (current position, blockers, deferred items), `PROJECT.md`
(requirements history, key decisions), and `RETROSPECTIVE.md` (per-milestone lessons). A
milestone in progress also has a live `REQUIREMENTS.md` (the `CORE-`/`MOD-`/`ATK-`/`CLI-`/
`DASH-` ids referenced throughout the code) and per-phase `.planning/phases/*/` directories
(`CONTEXT.md`/`PLAN.md`/`VERIFICATION.md`/`REVIEW.md`); both are archived to
`.planning/milestones/v[X.Y]-*` and removed from the working set when a milestone ships, so
their absence at the root means no milestone is currently in progress, not that they were
never written.

Code comments cite decision ids (`D-08`, `T-01-18`, `WR-05`, `CR-01`) that resolve to those
phase documents, current or archived. When a comment's reasoning is unclear, grep both
`.planning/` and `.planning/milestones/` for the id rather than guessing — the rationale is
written down.

**v1.0 MVP** (Phases 1-5, shipped 2026-08-08): the core scan pipeline, the first four modules
(LLM01/02/05/07), both adapters, and the opt-in attacker-LLM team described above. **v1.1 Full
OWASP Coverage + Web UI** (Phases 6-11, shipped 2026-08-23): the remaining six modules
(LLM03/04/06/08/09/10) plus the web dashboard (report browsing, then authorized scan
triggering with SSRF gating and live progress). Full detail for both:
`.planning/milestones/v1.0-ROADMAP.md` / `v1.1-ROADMAP.md`, `v1.1-REQUIREMENTS.md`,
`v1.1-MILESTONE-AUDIT.md`, and `.planning/RETROSPECTIVE.md`. No milestone is in progress as of
the v1.1 close — the next one starts with `/gsd-new-milestone`, which creates a fresh
`REQUIREMENTS.md`.

## Dashboard UI design direction

- **Sketch findings for testai** (validated dashboard visual direction, CSS patterns,
  type/color system) → `Skill("sketch-findings-testai")`
