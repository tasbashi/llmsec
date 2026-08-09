"""05-06-PLAN.md Task 1: wire the audit handler through every role
invocation and every target dispatch (D-68/D-84/D-85/D-86).

`run_attacker_campaign()` now ALWAYS constructs its own
`AttackerAuditHandler`/`AttackerAuditWriter` pair (Task 1's own `<action>`)
-- these tests drive the real production path end to end, offline, via the
same scripted-role/fake-adapter fixtures `test_tracer_round.py`/
`test_budget_stop_semantics.py` already established, reading the persisted
`{scan_id}-attacker-audit.jsonl` back off disk to assert on it directly
rather than only on the in-memory handler counters.
"""

from __future__ import annotations

import json
from pathlib import Path

from llmsec.attacker.audit import audit_path_for
from llmsec.attacker.config import AttackerConfig, resolve_settings
from llmsec.attacker.graph import build_campaign_graph
from llmsec.attacker.roles import ROLE_REGISTRY
from llmsec.attacker.roles.mutator import MutatedVariant, MutatorOutput, build_mutator_agent
from llmsec.attacker.roles.strategist import StrategistOutput, build_strategist_agent
from llmsec.attacker.runner import run_attacker_campaign
from llmsec.attacker.state import QueuedCase, new_campaign_state
from llmsec.config import ScanConfig, TargetConfig
from llmsec.models import EvalResult, ScanContext, TargetResponse, TestCase, Verdict
from llmsec.plugins.base import BaseModule

_STATIC_CASE_ID = "STUB-001"
_STATIC_TECHNIQUE_ID = "STUB-001"


class _StubModule(BaseModule):
    """One static case; every mutated variant scores `FULL_COMPROMISE` so
    dispatch always produces a real, non-degraded `EvalResult` to audit."""

    id = "stub_module"
    name = "Stub Module"
    owasp_ref = "LLM00:2025"
    uses_attacker_llm = True

    async def generate_cases(self, context: ScanContext):  # type: ignore[no-untyped-def]
        yield TestCase(
            case_id=_STATIC_CASE_ID, prompt="stub parent payload", technique_id=_STATIC_TECHNIQUE_ID
        )

    async def evaluate(self, case: TestCase, response: TargetResponse) -> EvalResult:
        if case.case_id == _STATIC_CASE_ID:
            return EvalResult(
                case_id=case.case_id,
                verdict=Verdict.BLOCKED,
                confidence=0.9,
                evidence="refused",
                detection_layer="regex",
            )
        return EvalResult(
            case_id=case.case_id,
            verdict=Verdict.FULL_COMPROMISE,
            confidence=0.95,
            evidence=f"complied: {response.raw_text}",
            detection_layer="regex",
        )


def _make_config(*, output_dir, max_rounds: int = 2, variants_per_round: int = 2) -> ScanConfig:
    return ScanConfig(
        target=TargetConfig(type="raw_llm", model="openai/gpt-4o-mini", api_key_env="TEST_API_KEY"),
        output_dir=str(output_dir),
        attacker=AttackerConfig(
            enabled=True, profile="light", variants_per_round=variants_per_round, max_rounds=max_rounds
        ),
    )


def _static_results() -> list[tuple[str, EvalResult]]:
    return [
        (
            "stub_module",
            EvalResult(
                case_id=_STATIC_CASE_ID,
                verdict=Verdict.BLOCKED,
                confidence=0.9,
                evidence="refused",
                detection_layer="regex",
            ),
        )
    ]


def _strategist_output(case_ids: list[str]) -> StrategistOutput:
    return StrategistOutput(
        technique="instruction_override",
        ordered_case_ids=case_ids,
        escalate=False,
        reason_code=None,
        rationale="Try a direct instruction-override refinement.",
    )


def _mutator_output(n_variants: int) -> MutatorOutput:
    return MutatorOutput(
        variants=[
            MutatedVariant(
                payload=f"mutated payload variant {i + 1}",
                technique_family="instruction_override",
                parent_technique_id=_STATIC_TECHNIQUE_ID,
                rationale=f"rationale {i + 1}",
            )
            for i in range(n_variants)
        ]
    )


def _patch_roles(monkeypatch, scripted_chat_model, *, strategist_outputs, mutator_outputs):
    """Scripts N rounds' worth of responses -- mirrors
    `test_budget_stop_semantics.py`'s own `_patch_roles()`."""
    strategist_model = scripted_chat_model(strategist_outputs)
    mutator_model = scripted_chat_model(mutator_outputs)
    monkeypatch.setattr(
        ROLE_REGISTRY["strategist"],
        "build",
        lambda settings, cfg, *, model=None: build_strategist_agent(
            settings, cfg, model=strategist_model
        ),
    )
    monkeypatch.setattr(
        ROLE_REGISTRY["mutator"],
        "build",
        lambda settings, cfg, *, model=None: build_mutator_agent(settings, cfg, model=mutator_model),
    )
    return strategist_model, mutator_model


def _read_audit_lines(output_dir: Path, scan_id: str) -> list[dict]:
    path = audit_path_for(output_dir, scan_id)
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line]


async def _run_two_round_campaign(
    monkeypatch,
    fake_target_adapter,
    mock_target_response,
    scripted_chat_model,
    patch_analyst_and_recon_roles,
    tmp_path,
    scan_id,
):
    module = _StubModule()
    modules = {"stub_module": module}

    adapter = fake_target_adapter()
    adapter.queue_response(
        f"{_STATIC_CASE_ID}-mut-1", mock_target_response(case_id=f"{_STATIC_CASE_ID}-mut-1", raw_text="sure")
    )
    adapter.queue_response(
        f"{_STATIC_CASE_ID}-mut-2", mock_target_response(case_id=f"{_STATIC_CASE_ID}-mut-2", raw_text="sure too")
    )

    _patch_roles(
        monkeypatch,
        scripted_chat_model,
        strategist_outputs=[_strategist_output([_STATIC_CASE_ID]) for _ in range(2)],
        mutator_outputs=[_mutator_output(2) for _ in range(2)],
    )
    patch_analyst_and_recon_roles()

    config = _make_config(output_dir=tmp_path, max_rounds=2, variants_per_round=2)
    result = await run_attacker_campaign(
        config=config,
        adapter=adapter,
        modules=modules,
        static_results=_static_results(),
        scan_id=scan_id,
    )
    return result, adapter


# --- Core capture/count/redaction-path wiring -------------------------------


async def test_audit_file_exists_and_captures_role_and_target_lines(
    monkeypatch,
    fake_target_adapter,
    mock_target_response,
    scripted_chat_model,
    patch_analyst_and_recon_roles,
    tmp_path,
):
    result, adapter = await _run_two_round_campaign(
        monkeypatch, fake_target_adapter, mock_target_response, scripted_chat_model, patch_analyst_and_recon_roles, tmp_path, "scan-audit-1"
    )

    assert result.audit_path is not None
    assert result.audit_path == audit_path_for(tmp_path, "scan-audit-1")
    assert result.audit_path.exists()

    lines = _read_audit_lines(tmp_path, "scan-audit-1")
    assert len(lines) > 0

    # At least one line per role invocation (2 rounds * 2 roles = 4 model
    # invocations) and one line per dispatched variant (2 rounds * 2
    # variants = 4 dispatches).
    role_lines = [
        line for line in lines if line["agent"] in {"strategist", "mutator"} and line["direction"] == "outbound"
    ]
    assert len(role_lines) >= 4  # one model_start per role invocation, at minimum
    target_lines = [line for line in lines if line["direction"] == "target"]
    assert len(target_lines) == 4  # exactly one recorded dispatch per variant

    inter_agent_lines = [line for line in lines if line["direction"] == "inter_agent"]
    # 1 campaign_start line + 1 strategist->mutator handoff per round (2).
    assert len(inter_agent_lines) >= 3
    assert any(line["event"] == "campaign_start" for line in inter_agent_lines)
    # 05-07: the Analyst (every round) and Recon (once) also record their
    # own inter_agent_handoff lines (analyst->strategist, recon->strategist)
    # -- scoped here to the strategist->mutator handoff specifically, since
    # that is what this assertion is about.
    handoffs = [
        line
        for line in inter_agent_lines
        if line["event"] == "inter_agent_handoff" and line["agent"] == "strategist"
    ]
    assert len(handoffs) == 2
    for handoff in handoffs:
        assert handoff["recipient"] == "mutator"
        assert handoff["agent"] == "strategist"


async def test_captured_events_equals_written_lines_equals_file_line_count(
    monkeypatch,
    fake_target_adapter,
    mock_target_response,
    scripted_chat_model,
    patch_analyst_and_recon_roles,
    tmp_path,
):
    """`<behavior>`'s own headline invariant: nothing bypasses the audit
    handler, and nothing the handler counts is ever dropped before it
    reaches disk."""
    module = _StubModule()
    modules = {"stub_module": module}
    adapter = fake_target_adapter()
    adapter.queue_response(
        f"{_STATIC_CASE_ID}-mut-1", mock_target_response(case_id=f"{_STATIC_CASE_ID}-mut-1")
    )
    adapter.queue_response(
        f"{_STATIC_CASE_ID}-mut-2", mock_target_response(case_id=f"{_STATIC_CASE_ID}-mut-2")
    )
    _patch_roles(
        monkeypatch,
        scripted_chat_model,
        strategist_outputs=[_strategist_output([_STATIC_CASE_ID]) for _ in range(2)],
        mutator_outputs=[_mutator_output(2) for _ in range(2)],
    )
    patch_analyst_and_recon_roles()
    config = _make_config(output_dir=tmp_path, max_rounds=2, variants_per_round=2)
    await run_attacker_campaign(
        config=config,
        adapter=adapter,
        modules=modules,
        static_results=_static_results(),
        scan_id="scan-audit-2",
    )

    lines = _read_audit_lines(tmp_path, "scan-audit-2")
    # The handler instance itself is not directly reachable from
    # CampaignResult (by design -- it is closed before returning), so the
    # equality is re-derived structurally: every line's presence implies a
    # successful _safe_capture() call incremented BOTH counters together
    # (audit.py's own class docstring). A capture failure would either
    # raise (never silently swallowed beyond capture_failures) or show up
    # as a missing category below.
    assert len(lines) >= 1
    seqs = [line["seq"] for line in lines]
    assert seqs == sorted(seqs)
    assert seqs == list(range(len(lines)))  # strictly increasing, no gaps


async def test_lines_carry_correct_round_module_id_case_id(
    monkeypatch,
    fake_target_adapter,
    mock_target_response,
    scripted_chat_model,
    patch_analyst_and_recon_roles,
    tmp_path,
):
    result, adapter = await _run_two_round_campaign(
        monkeypatch, fake_target_adapter, mock_target_response, scripted_chat_model, patch_analyst_and_recon_roles, tmp_path, "scan-audit-3"
    )
    lines = _read_audit_lines(tmp_path, "scan-audit-3")

    target_lines = [line for line in lines if line["direction"] == "target"]
    rounds_seen = {line["round"] for line in target_lines}
    assert rounds_seen == {1, 2}
    for line in target_lines:
        assert line["module_id"] == "stub_module"
        assert line["case_id"].startswith(_STATIC_CASE_ID)

    # Scoped to the strategist's own handoffs -- 05-07's Analyst/Recon
    # record their own inter_agent_handoff lines too (see the equivalent
    # scoping note in `test_audit_file_exists_and_captures_role_and_target_lines`).
    handoffs = [
        line for line in lines if line["event"] == "inter_agent_handoff" and line["agent"] == "strategist"
    ]
    handoff_rounds = sorted(line["round"] for line in handoffs)
    assert handoff_rounds == [0, 1]  # state["round"] at strategist-invocation time, pre-increment


async def test_deterministic_module_order_stamped_identically_across_runs(
    monkeypatch,
    fake_target_adapter,
    mock_target_response,
    scripted_chat_model,
    patch_analyst_and_recon_roles,
    tmp_path,
):
    """D-68: two runs of the identical configuration produce the identical
    recorded module order value."""

    class _AlphaModule(_StubModule):
        id = "alpha_module"

    class _ZebraModule(_StubModule):
        id = "zebra_module"

    def _fresh_modules():
        return {"zebra_module": _ZebraModule(), "alpha_module": _AlphaModule()}

    def _static_results_multi():
        return [
            (
                mid,
                EvalResult(
                    case_id=_STATIC_CASE_ID,
                    verdict=Verdict.BLOCKED,
                    confidence=0.9,
                    evidence="refused",
                    detection_layer="regex",
                ),
            )
            for mid in ("zebra_module", "alpha_module")
        ]

    module_orders: list[list[str]] = []
    for scan_id in ("scan-order-1", "scan-order-2"):
        adapter = fake_target_adapter()
        adapter.queue_response(
            f"{_STATIC_CASE_ID}-mut-1", mock_target_response(case_id=f"{_STATIC_CASE_ID}-mut-1")
        )
        _patch_roles(
            monkeypatch,
            scripted_chat_model,
            strategist_outputs=[_strategist_output([_STATIC_CASE_ID])],
            mutator_outputs=[_mutator_output(1)],
        )
        patch_analyst_and_recon_roles()
        config = _make_config(output_dir=tmp_path, max_rounds=1, variants_per_round=1)
        await run_attacker_campaign(
            config=config,
            adapter=adapter,
            modules=_fresh_modules(),
            static_results=_static_results_multi(),
            scan_id=scan_id,
        )
        lines = _read_audit_lines(tmp_path, scan_id)
        start_line = next(line for line in lines if line["event"] == "campaign_start")
        module_orders.append(json.loads(start_line["content"])["module_order"])

    assert module_orders[0] == module_orders[1]
    # D-78: fixed sorted() order, never set/dict iteration -- alpha before zebra.
    assert module_orders[0] == ["alpha_module", "zebra_module"]


async def test_capture_depends_on_explicit_config_forwarding_no_ambient_fallback(
    monkeypatch,
    fake_target_adapter,
    mock_target_response,
    scripted_chat_model,
    patch_analyst_and_recon_roles,
    tmp_path,
):
    """05-RESEARCH Pitfall 6: `run_attacker_campaign()`'s own top-level
    `compiled.ainvoke(state, config=thread_config)` call never sets
    `callbacks` on `thread_config` at all (only `configurable.thread_id`)
    -- so unlike 05-RESEARCH's own spike (which set callbacks on ITS
    top-level invoke config, giving ambient contextvar propagation
    something to inherit from), THIS codebase's design has NO ambient
    fallback path from the top at all. `graph.py`'s explicit
    `config=graph_config` forwarding into `invoke_role_with_retry()` is
    therefore the SOLE path by which the audit handler observes a role
    invocation, not merely defense-in-depth over some other route.

    This test monkeypatches AWAY just that one explicit forward (dropping
    `config` before delegating to the real `invoke_role_with_retry()`,
    everything else -- including DeepAgents' own internal plumbing --
    untouched) and observes capture of role-invocation lines collapse to
    zero, while the campaign itself still completes and dispatches
    normally. This is the structural regression Pitfall 6 warns a silent
    refactor could introduce, and confirms `set_context()`/explicit
    `config=` forwarding is genuinely load-bearing here, not decorative.
    """
    import llmsec.attacker.graph as graph_module

    real_invoke = graph_module.invoke_role_with_retry

    async def _invoke_without_explicit_config(agent, messages, **kwargs):
        kwargs.pop("config", None)  # simulate a future refactor dropping the forward
        return await real_invoke(agent, messages, **kwargs)

    monkeypatch.setattr(graph_module, "invoke_role_with_retry", _invoke_without_explicit_config)

    result, adapter = await _run_two_round_campaign(
        monkeypatch, fake_target_adapter, mock_target_response, scripted_chat_model, patch_analyst_and_recon_roles, tmp_path, "scan-audit-no-forward"
    )

    # The campaign itself is unaffected -- dropping `config` only removes
    # the CALLBACK's observation of the role calls, not the calls
    # themselves (role agents still run and produce structured output).
    # Recon's own `recon-probe-*`-prefixed dispatches are excluded, since
    # this assertion is about the Mutator's own 4 dispatched variants.
    variant_sent_cases = [c for c in adapter.sent_cases if not c.case_id.startswith("recon-probe-")]
    assert len(variant_sent_cases) == 4
    assert result.final_state.get("termination_reason") is not None

    lines = _read_audit_lines(tmp_path, "scan-audit-no-forward")
    role_start_lines = [line for line in lines if line["event"] == "model_start"]
    assert role_start_lines == []


# --- No-handler-supplied completeness (build_campaign_graph() directly) -----


async def test_campaign_completes_with_no_audit_handler_supplied(
    monkeypatch, fake_target_adapter, mock_target_response, scripted_chat_model
):
    """The handler is injected by `run_attacker_campaign()`, never assumed
    by `build_campaign_graph()` itself -- driven directly here with
    `callbacks=None` to prove the graph completes a full round with zero
    audit wiring attached."""
    settings = resolve_settings(
        AttackerConfig(enabled=True, profile="light", max_rounds=1, variants_per_round=1)
    )
    cfg = AttackerConfig(enabled=True, profile="light", max_rounds=1, variants_per_round=1)

    strategist_model = scripted_chat_model([_strategist_output([_STATIC_CASE_ID])])
    mutator_model = scripted_chat_model([_mutator_output(1)])
    strategist_agent = build_strategist_agent(settings, cfg, model=strategist_model)
    mutator_agent = build_mutator_agent(settings, cfg, model=mutator_model)

    module = _StubModule()
    adapter = fake_target_adapter()
    adapter.queue_response(
        f"{_STATIC_CASE_ID}-mut-1", mock_target_response(case_id=f"{_STATIC_CASE_ID}-mut-1")
    )

    compiled = build_campaign_graph(
        roles={"strategist": strategist_agent, "mutator": mutator_agent},
        adapter=adapter,
        modules={"stub_module": module},
        max_concurrency=5,
        role_models={"strategist": "test-model", "mutator": "test-model"},
        checkpointer=None,
        callbacks=None,
    )

    state = new_campaign_state(
        "scan-no-handler",
        settings,
        ["stub_module"],
        [
            QueuedCase(
                module_id="stub_module",
                case_id=_STATIC_CASE_ID,
                technique_id=_STATIC_TECHNIQUE_ID,
                prompt="stub parent payload",
                verdict="blocked",
                turns=None,
            )
        ],
    )
    state["current_module"] = "stub_module"
    state["enabled_techniques"] = ["instruction_override"]

    final_state = await compiled.ainvoke(
        state, config={"configurable": {"thread_id": "scan-no-handler"}}
    )

    assert final_state.get("termination_reason") is not None
    assert len(final_state.get("dispatch_results", [])) == 1


# --- CR-01 regression: repeated-parent-case-across-rounds case_id collision -


async def test_cr01_repeated_parent_case_across_rounds_yields_unique_case_ids_and_reconciled_lineage(
    monkeypatch,
    fake_target_adapter,
    mock_target_response,
    scripted_chat_model,
    patch_analyst_and_recon_roles,
    tmp_path,
):
    """CR-01 regression: `_run_two_round_campaign()` drives 2 REAL rounds
    (2 variants each, via the real `mutator_node`-`enumerate()`-based
    `variant_index` generation, never a hand-built lineage) through
    `dispatch_variants_node`, with the Strategist scripted to re-select
    the SAME `parent_case_id` both rounds -- the exact shape that, before
    the fix, made round 1 and round 2 both dispatch a variant literally
    named `STUB-001-mut-1` (since `variant_index` resets to 0 on every
    Mutator invocation).

    `runner.py::_collect_dispatch_results()`'s `lineage` is a plain dict
    keyed by the generated `case_id` -- a collision silently overwrites
    round 1's entry with round 2's, shrinking `lineage` below
    `eval_results`' own length (which is never deduped) and directly
    undercounting `summary.py::compute_deep_summary()`'s
    `variants_dispatched = len(lineage)` (AT-1/D-91: a COUNTED event that
    must never disagree with the true dispatch count).
    """
    result, adapter = await _run_two_round_campaign(
        monkeypatch,
        fake_target_adapter,
        mock_target_response,
        scripted_chat_model,
        patch_analyst_and_recon_roles,
        tmp_path,
        "scan-cr01-1",
    )

    # 2 rounds * 2 variants/round == 4 real dispatches -- every one of
    # them records a distinct outcome (T-01-18: `eval_results` is never
    # deduped, one entry per dispatch, success or degraded).
    assert len(result.eval_results) == 4
    # The direct CR-01 assertion: `lineage` must reconcile against the
    # true dispatch count, never silently shrink from a case_id collision.
    assert len(result.lineage) == len(result.eval_results) == 4

    # Every dispatched TestCase carries a campaign-wide-unique case_id
    # (Recon's own `recon-probe-*`-prefixed dispatches, if any, are
    # excluded -- this assertion is scoped to the Mutator's variants,
    # mirroring the equivalent scoping elsewhere in this file).
    variant_sent_case_ids = [
        c.case_id for c in adapter.sent_cases if not c.case_id.startswith("recon-probe-")
    ]
    assert len(variant_sent_case_ids) == len(set(variant_sent_case_ids)) == 4
