"""Tests for `llmsec.reporting` — JSON (REPORT-01) and Markdown (REPORT-02)
reporters.

JSON round-trip/malformed-input/permissions cases exercise the
`load_report()` fail-loudly prohibition (T-01-11). Markdown structure
cases (added alongside `markdown_reporter.py`) exercise the Jinja2-templated
human-readable report (REPORT-02).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmsec.models import DeepModeSummary, EvalResult, Finding, RoleActivity, ScanReport, Verdict
from llmsec.reporting.json_reporter import JsonReporter, load_report
from llmsec.reporting.markdown_reporter import MarkdownReporter


def _make_finding(case_id: str = "case-1", severity: str = "high") -> Finding:
    return Finding(
        case_id=case_id,
        technique_id="direct_ask",
        verdict=Verdict.FULL_COMPROMISE,
        severity=severity,
        owasp_ref="LLM07",
        evidence="the assistant repeated its full system prompt verbatim",
        remediation="Add output filtering to strip system-prompt-shaped content.",
    )


def _make_report(findings: list[Finding] | None = None) -> ScanReport:
    return ScanReport(
        scan_id="abc123",
        target_summary="raw_llm target, model=gpt-4o-mini",
        module_ids=["system_prompt_leakage"],
        findings=findings if findings is not None else [_make_finding()],
        case_log=[
            EvalResult(
                case_id="case-1",
                verdict=Verdict.FULL_COMPROMISE,
                confidence=0.95,
                evidence="the assistant repeated its full system prompt verbatim",
                detection_layer="regex",
            )
        ],
        started_at="2026-07-21T00:00:00Z",
        completed_at="2026-07-21T00:00:05Z",
    )


@pytest.mark.asyncio
async def test_json_roundtrip(tmp_path: Path) -> None:
    """Written JSON reloads to a field-by-field equal ScanReport."""
    report = _make_report()
    reporter = JsonReporter()
    output_dir = tmp_path / "reports"

    written_path = await reporter.write(report, output_dir)

    assert written_path.exists()
    reloaded = load_report(written_path)
    assert reloaded == report


@pytest.mark.asyncio
async def test_json_malformed_raises(tmp_path: Path) -> None:
    """A corrupted scan_<id>.json raises instead of silently substituting defaults."""
    output_dir = tmp_path / "reports"
    output_dir.mkdir()
    bad_path = output_dir / "scan_broken.json"
    bad_path.write_text("{not valid json,,,")

    with pytest.raises((ValueError, json.JSONDecodeError)):
        load_report(bad_path)


@pytest.mark.asyncio
async def test_json_output_dir_restrictive_permissions(tmp_path: Path) -> None:
    """output_dir is created with 0o700 (owner-only) permissions."""
    report = _make_report()
    reporter = JsonReporter()
    output_dir = tmp_path / "reports"

    await reporter.write(report, output_dir)

    assert oct(output_dir.stat().st_mode)[-3:] == "700"


@pytest.mark.asyncio
async def test_json_write_is_deterministic(tmp_path: Path) -> None:
    """Two writes of an unchanged in-memory report produce byte-identical JSON."""
    report = _make_report()
    reporter = JsonReporter()
    output_dir = tmp_path / "reports"

    path1 = await reporter.write(report, output_dir)
    contents1 = path1.read_bytes()
    path2 = await reporter.write(report, output_dir)
    contents2 = path2.read_bytes()

    assert contents1 == contents2


@pytest.mark.asyncio
async def test_json_output_dir_permissions_tightened_when_dir_preexists_looser(
    tmp_path: Path,
) -> None:
    """Regression test (CR-02): if `output_dir` already exists with looser
    permissions (e.g. operator pre-created it, or a prior process left it
    with a looser umask), `write()` must still tighten it to 0o700 rather
    than silently leaving the looser mode in place via `mkdir(exist_ok=True)`.
    Mirrors `test_markdown_output_dir_permissions_tightened_when_dir_preexists_looser`
    so the two reporters' permission guarantees can't silently drift apart."""
    report = _make_report()
    reporter = JsonReporter()
    output_dir = tmp_path / "reports"
    output_dir.mkdir(mode=0o755)
    output_dir.chmod(0o755)  # mkdir(mode=...) is umask-affected; force it explicitly
    assert oct(output_dir.stat().st_mode)[-3:] == "755"

    await reporter.write(report, output_dir)

    assert oct(output_dir.stat().st_mode)[-3:] == "700"


@pytest.mark.asyncio
async def test_markdown_structure(tmp_path: Path) -> None:
    """Written Markdown contains severity summary and per-finding sections."""
    report = _make_report()
    reporter = MarkdownReporter()
    output_dir = tmp_path / "reports"

    written_path = await reporter.write(report, output_dir)

    assert written_path.exists()
    content = written_path.read_text()
    assert "## Findings" in content
    assert "LLM07" in content
    assert "high" in content.lower()
    assert "the assistant repeated its full system prompt verbatim" in content


@pytest.mark.asyncio
async def test_markdown_empty_findings(tmp_path: Path) -> None:
    """An empty findings list still produces a valid, non-crashing Markdown file."""
    report = _make_report(findings=[])
    reporter = MarkdownReporter()
    output_dir = tmp_path / "reports"

    written_path = await reporter.write(report, output_dir)

    content = written_path.read_text()
    assert "No findings." in content


@pytest.mark.asyncio
async def test_markdown_output_dir_restrictive_permissions(tmp_path: Path) -> None:
    """`MarkdownReporter.write()` creates output_dir with 0o700 (owner-only)
    permissions, mirroring `JsonReporter`'s guarantee (ASVS V6 / PITFALLS
    P10-D: Markdown output may also contain leaked system-prompt content)."""
    report = _make_report()
    reporter = MarkdownReporter()
    output_dir = tmp_path / "reports"

    await reporter.write(report, output_dir)

    assert oct(output_dir.stat().st_mode)[-3:] == "700"


@pytest.mark.asyncio
async def test_markdown_output_dir_permissions_tightened_when_dir_preexists_looser(
    tmp_path: Path,
) -> None:
    """Regression test (WR-05): if `output_dir` already exists with looser
    permissions (e.g. operator pre-created it, or a prior process left it
    with a looser umask), `write()` must still tighten it to 0o700 rather
    than silently leaving the looser mode in place via `mkdir(exist_ok=True)`."""
    report = _make_report()
    reporter = MarkdownReporter()
    output_dir = tmp_path / "reports"
    output_dir.mkdir(mode=0o755)
    output_dir.chmod(0o755)  # mkdir(mode=...) is umask-affected; force it explicitly
    assert oct(output_dir.stat().st_mode)[-3:] == "755"

    await reporter.write(report, output_dir)

    assert oct(output_dir.stat().st_mode)[-3:] == "700"


# --- Task 2 (plan 09): limitations section + transport-mode rendering ---


@pytest.mark.asyncio
async def test_markdown_renders_scan_limitations_section_when_non_empty(tmp_path: Path) -> None:
    limitations = ["Canary caveat text.", "Indirect-injection caveat text."]
    report = _make_report()
    report = report.model_copy(update={"limitations": limitations})
    reporter = MarkdownReporter()

    written_path = await reporter.write(report, tmp_path / "reports")

    content = written_path.read_text()
    assert "## Scan Limitations" in content
    for item in limitations:
        assert item in content


@pytest.mark.asyncio
async def test_markdown_omits_scan_limitations_heading_when_empty(tmp_path: Path) -> None:
    report = _make_report()  # default limitations=[]
    reporter = MarkdownReporter()

    written_path = await reporter.write(report, tmp_path / "reports")

    content = written_path.read_text()
    assert "## Scan Limitations" not in content


@pytest.mark.asyncio
async def test_markdown_renders_transport_line_when_set_and_omits_when_none(tmp_path: Path) -> None:
    finding_with_transport = _make_finding(case_id="case-1").model_copy(
        update={"transport_mode": "multi_turn_concatenated"}
    )
    finding_without_transport = Finding(
        case_id="case-2",
        technique_id="second_technique",
        verdict=Verdict.PARTIAL_LEAK,
        severity="medium",
        owasp_ref="LLM01:2025",
        evidence="clean evidence, no transport mode",
        remediation="Some remediation.",
    )
    report = _make_report(findings=[finding_with_transport, finding_without_transport])
    reporter = MarkdownReporter()

    written_path = await reporter.write(report, tmp_path / "reports")

    content = written_path.read_text()
    assert "**Transport:** multi_turn_concatenated" in content
    # Split on the second finding's own heading to isolate its block, and
    # confirm no Transport line appears there.
    case2_section = content.split("### second_technique — medium")[-1]
    assert "**Transport:**" not in case2_section


@pytest.mark.asyncio
async def test_json_report_round_trips_limitations_and_transport_mode(tmp_path: Path) -> None:
    finding = _make_finding().model_copy(update={"transport_mode": "multi_turn_concatenated"})
    report = _make_report(findings=[finding]).model_copy(update={"limitations": ["some caveat"]})
    reporter = JsonReporter()

    written_path = await reporter.write(report, tmp_path / "reports")
    reloaded = load_report(written_path)

    assert reloaded.limitations == ["some caveat"]
    assert reloaded.findings[0].transport_mode == "multi_turn_concatenated"


@pytest.mark.asyncio
async def test_markdown_survives_evidence_with_fence_and_heading_control_chars(tmp_path: Path) -> None:
    """Evidence containing a fenced-code delimiter and a heading marker must
    not break the surrounding document: the finding's own section and every
    following section still render (T-02-31)."""
    hostile_evidence = "```\nfake closing fence above\n```\n# Fake Heading\nmore text"
    finding_hostile = _make_finding(case_id="case-1").model_copy(update={"evidence": hostile_evidence})
    finding_after = Finding(
        case_id="case-2",
        technique_id="second_technique",
        verdict=Verdict.PARTIAL_LEAK,
        severity="medium",
        owasp_ref="LLM01:2025",
        evidence="clean evidence text",
        remediation="Some remediation.",
    )
    report = _make_report(findings=[finding_hostile, finding_after])
    reporter = MarkdownReporter()

    written_path = await reporter.write(report, tmp_path / "reports")

    content = written_path.read_text()
    # The hostile evidence's own fence markers are present, but the later
    # finding's heading and content must still appear intact afterward.
    assert "### second_technique — medium" in content
    assert "clean evidence text" in content
    assert "**Remediation:**" in content
    assert "Some remediation." in content


def test_markdown_reporter_evidence_block_fence_exceeds_backtick_runs() -> None:
    from llmsec.reporting.markdown_reporter import _safe_evidence_block

    evidence = "some text with `````` six backticks"
    block = _safe_evidence_block(evidence)
    lines = block.splitlines()
    fence = lines[0]
    assert fence == "`" * 7
    assert lines[-1] == fence


@pytest.mark.asyncio
async def test_markdown_survives_remediation_with_fence_and_heading_control_chars(
    tmp_path: Path,
) -> None:
    """Regression test (IN-03, 02-REVIEW.md): `finding.remediation` gets the
    same heading/fence-neutralizing treatment as `finding.evidence` -- a
    remediation string containing a fenced-code delimiter and a heading
    marker must not break the surrounding document, and every following
    finding's section must still render intact."""
    hostile_remediation = "```\nfake closing fence above\n```\n# Fake Heading\nmore text"
    finding_hostile = _make_finding(case_id="case-1").model_copy(
        update={"remediation": hostile_remediation}
    )
    finding_after = Finding(
        case_id="case-2",
        technique_id="second_technique",
        verdict=Verdict.PARTIAL_LEAK,
        severity="medium",
        owasp_ref="LLM01:2025",
        evidence="clean evidence text",
        remediation="Some clean remediation.",
    )
    report = _make_report(findings=[finding_hostile, finding_after])
    reporter = MarkdownReporter()

    written_path = await reporter.write(report, tmp_path / "reports")

    content = written_path.read_text()
    # The hostile remediation's own fence markers are present, but the later
    # finding's heading and content must still appear intact afterward.
    assert "### second_technique — medium" in content
    assert "Some clean remediation." in content


# --- Phase 3 (03-01): finding.detection_layer rendering (SC#3) -------------


@pytest.mark.asyncio
async def test_markdown_renders_detection_layer_line_when_set_and_omits_when_none(
    tmp_path: Path,
) -> None:
    finding_with_layer = _make_finding(case_id="case-1").model_copy(
        update={"detection_layer": "regex"}
    )
    finding_without_layer = Finding(
        case_id="case-2",
        technique_id="second_technique",
        verdict=Verdict.PARTIAL_LEAK,
        severity="medium",
        owasp_ref="LLM01:2025",
        evidence="clean evidence, no detection layer",
        remediation="Some remediation.",
    )
    report = _make_report(findings=[finding_with_layer, finding_without_layer])
    reporter = MarkdownReporter()

    written_path = await reporter.write(report, tmp_path / "reports")

    content = written_path.read_text()
    assert "**Detection Layer:** regex" in content
    case2_section = content.split("### second_technique — medium")[-1]
    assert "**Detection Layer:**" not in case2_section


@pytest.mark.asyncio
async def test_json_report_round_trips_detection_layer(tmp_path: Path) -> None:
    finding = _make_finding().model_copy(update={"detection_layer": "regex"})
    report = _make_report(findings=[finding])
    reporter = JsonReporter()

    written_path = await reporter.write(report, tmp_path / "reports")
    reloaded = load_report(written_path)

    assert reloaded.findings[0].detection_layer == "regex"
    persisted = json.loads(written_path.read_text())
    assert persisted["findings"][0]["detection_layer"] == "regex"


# --- Phase 5 (05-09): Deep Mode summary block rendering (D-91) -------------

#: The exact pre-phase-5 rendering of `_make_report()` (captured before the
#: Deep Mode section existed) -- the byte-identity oracle for a report
#: whose `deep_summary is None` (T-05-09-05/D-93).
_PRE_PHASE_5_RENDERING = (
    "\n# Scan Report: abc123\n\n"
    "**Target:** raw_llm target, model=gpt-4o-mini\n"
    "**Modules:** system_prompt_leakage\n"
    "**Started:** 2026-07-21T00:00:00Z\n"
    "**Completed:** 2026-07-21T00:00:05Z\n\n"
    "## Severity Summary\n\n"
    "- **none**: 0\n"
    "- **low**: 0\n"
    "- **medium**: 0\n"
    "- **high**: 1\n"
    "- **critical**: 0\n\n"
    "## Findings\n\n"
    "### direct_ask — high\n\n"
    "- **Case ID:** case-1\n"
    "- **Verdict:** full_compromise\n"
    "- **OWASP Ref:** LLM07\n"
    "- **Evidence:**\n\n"
    "```\nthe assistant repeated its full system prompt verbatim\n```\n\n"
    "- **Remediation:**\n\n"
    "```\nAdd output filtering to strip system-prompt-shaped content.\n```\n\n"
)


def _make_deep_summary(
    *,
    bypasses_found: int = 1,
    bypass_case_ids: list[str] | None = None,
    cost_per_bypass_usd: float | None = 0.05,
    truncated: bool = False,
    audit_log_path: str | None = "/tmp/scan_abc123-attacker-audit.jsonl",
    role_structural_failures: int = 0,
) -> DeepModeSummary:
    return DeepModeSummary(
        cases_attacked=1,
        rounds_run=2,
        variants_dispatched=2,
        bypasses_found=bypasses_found,
        bypass_case_ids=bypass_case_ids if bypass_case_ids is not None else ["case-1-mut-1"],
        spend_usd=0.10,
        cost_per_bypass_usd=cost_per_bypass_usd,
        agent_calls=7,
        per_role_activity={
            "recon": RoleActivity(calls=1, spend_usd=0.01, bypasses=0),
            "strategist": RoleActivity(calls=2, spend_usd=0.02, bypasses=0),
            "mutator": RoleActivity(calls=1, spend_usd=0.02, bypasses=1),
            "analyst": RoleActivity(calls=2, spend_usd=0.02, bypasses=0),
            "crescendo": RoleActivity(calls=1, spend_usd=0.03, bypasses=0),
        },
        termination_reason="TECHNIQUES_EXHAUSTED",
        constraint_violations=0,
        abandoned_arcs=0,
        role_structural_failures=role_structural_failures,
        truncated=truncated,
        audit_log_path=audit_log_path,
    )


@pytest.mark.asyncio
async def test_markdown_no_deep_summary_byte_identical_to_pre_phase5_rendering(tmp_path: Path) -> None:
    """A report with `deep_summary is None` renders no Deep Mode section and
    is byte-identical to the pre-phase-5 rendering for the same report."""
    report = _make_report()  # deep_summary defaults to None
    reporter = MarkdownReporter()

    written_path = await reporter.write(report, tmp_path / "reports")

    content = written_path.read_text()
    assert "Deep Mode" not in content
    assert content == _PRE_PHASE_5_RENDERING


@pytest.mark.asyncio
async def test_markdown_renders_deep_mode_section_with_labelled_figures(tmp_path: Path) -> None:
    deep_summary = _make_deep_summary()
    finding = _make_finding(case_id="case-1-mut-1").model_copy(
        update={"parent_case_id": "case-1", "contributing_agent": "mutator"}
    )
    report = _make_report(findings=[finding]).model_copy(update={"deep_summary": deep_summary})
    reporter = MarkdownReporter()

    written_path = await reporter.write(report, tmp_path / "reports")

    content = written_path.read_text()
    assert "## Deep Mode Summary" in content
    assert "**Cases attacked:** 1" in content
    assert "**Rounds run:** 2" in content
    assert "**Bypasses found (static payloads alone missed):** 1" in content
    assert "counted actual spend" in content
    assert "$0.0500" in content  # the cost-per-bypass figure
    # 05-11 Rule 1/2 fix: rendered as its own labelled line, distinct from
    # "Constraint violations" so a reader can tell the two figures apart --
    # zero-safe (renders "0", never omitted) when no failures occurred.
    assert "**Role structural failures:** 0" in content


@pytest.mark.asyncio
async def test_markdown_renders_nonzero_role_structural_failures(tmp_path: Path) -> None:
    """A nonzero `role_structural_failures` count renders distinctly from
    `constraint_violations`, proving the two figures are never conflated."""
    deep_summary = _make_deep_summary(role_structural_failures=3)
    report = _make_report(findings=[]).model_copy(update={"deep_summary": deep_summary})
    reporter = MarkdownReporter()

    written_path = await reporter.write(report, tmp_path / "reports")

    content = written_path.read_text()
    deep_section = content.split("## Deep Mode Summary")[-1]
    assert "**Role structural failures:** 3" in deep_section
    assert "**Constraint violations:** 0" in deep_section


@pytest.mark.asyncio
async def test_markdown_cost_per_bypass_states_absence_explicitly_when_none(tmp_path: Path) -> None:
    deep_summary = _make_deep_summary(bypasses_found=0, bypass_case_ids=[], cost_per_bypass_usd=None)
    report = _make_report(findings=[]).model_copy(update={"deep_summary": deep_summary})
    reporter = MarkdownReporter()

    written_path = await reporter.write(report, tmp_path / "reports")

    content = written_path.read_text()
    deep_section = content.split("## Deep Mode Summary")[-1]
    assert "N/A" in deep_section
    assert "$0.00" not in deep_section
    assert "No bypasses found this run." in deep_section


@pytest.mark.asyncio
async def test_markdown_bypass_rows_show_case_id_and_parent_case_id(tmp_path: Path) -> None:
    deep_summary = _make_deep_summary(bypass_case_ids=["case-1-mut-1"])
    finding = _make_finding(case_id="case-1-mut-1").model_copy(
        update={"parent_case_id": "case-1", "contributing_agent": "mutator"}
    )
    report = _make_report(findings=[finding]).model_copy(update={"deep_summary": deep_summary})
    reporter = MarkdownReporter()

    written_path = await reporter.write(report, tmp_path / "reports")

    content = written_path.read_text()
    deep_section = content.split("## Deep Mode Summary")[-1]
    assert "`case-1-mut-1`" in deep_section
    assert "`case-1`" in deep_section


@pytest.mark.asyncio
async def test_markdown_truncated_deep_run_still_renders_deep_mode_section(tmp_path: Path) -> None:
    """The Deep Mode section renders independently of the Scan Limitations
    truncation disclosure — the two are separate, complementary surfaces."""
    deep_summary = _make_deep_summary(truncated=True)
    limitations = ["The --deep attacker campaign hit its $5.00 hard budget cap ..."]
    report = _make_report(findings=[]).model_copy(
        update={"deep_summary": deep_summary, "limitations": limitations}
    )
    reporter = MarkdownReporter()

    written_path = await reporter.write(report, tmp_path / "reports")

    content = written_path.read_text()
    assert "## Deep Mode Summary" in content
    assert "## Scan Limitations" in content


@pytest.mark.asyncio
async def test_json_report_round_trips_deep_summary(tmp_path: Path) -> None:
    deep_summary = _make_deep_summary()
    report = _make_report(findings=[]).model_copy(update={"deep_summary": deep_summary})
    reporter = JsonReporter()

    written_path = await reporter.write(report, tmp_path / "reports")
    reloaded = load_report(written_path)

    assert reloaded.deep_summary == deep_summary
    persisted = json.loads(written_path.read_text())
    assert persisted["deep_summary"]["bypasses_found"] == 1
    assert persisted["deep_summary"]["per_role_activity"]["mutator"]["bypasses"] == 1
