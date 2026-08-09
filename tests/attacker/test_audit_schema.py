"""Tests for `llmsec.attacker.audit`'s artifact half (05-05-PLAN.md Task 1):
`AuditLine`, `audit_path_for()`, `AttackerAuditWriter` -- the ordered,
append-only, greppable D-85 JSONL artifact, proven against hostile content
before any capture path is wired to it (05-05 Task 2)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from llmsec.attacker.audit import (
    AttackerAuditWriter,
    AuditLine,
    audit_path_for,
)

_REQUIRED_FIELDS = {
    "seq",
    "timestamp",
    "scan_id",
    "agent",
    "role",
    "direction",
    "round",
    "module_id",
    "case_id",
    "event",
    "content",
    "cost_usd",
    "tokens",
}


def _line(**overrides) -> AuditLine:
    base = dict(
        seq=0,
        timestamp="2026-08-05T00:00:00+00:00",
        scan_id="scan-1",
        agent="strategist",
        role="strategist",
        direction="outbound",
        round=1,
        module_id="prompt_injection",
        case_id="DIRECT-001",
        event="model_end",
        content="hello",
        cost_usd=0.001,
        tokens=42,
    )
    base.update(overrides)
    return AuditLine(**base)


# --- AuditLine schema shape ----------------------------------------------


def test_audit_line_field_set_is_a_superset_of_the_d85_required_fields():
    assert _REQUIRED_FIELDS <= set(AuditLine.model_fields)


def test_audit_line_direction_admits_exactly_the_closed_literal_values():
    for direction in ("inbound", "outbound", "inter_agent", "target"):
        _line(direction=direction)
    with pytest.raises(ValidationError):
        _line(direction="raw_bypass")  # not one of the four -- must reject


def test_audit_line_requires_every_named_field_present():
    payload = _line().model_dump()
    del payload["content"]
    with pytest.raises(ValidationError):
        AuditLine(**payload)


# --- audit_path_for --------------------------------------------------------


def test_audit_path_for_builds_the_scan_id_prefixed_jsonl_name(tmp_path):
    path = audit_path_for(tmp_path, "abc123")
    assert path == tmp_path / "abc123-attacker-audit.jsonl"


# --- AttackerAuditWriter: ordering, seq, directory mode ---------------------


def test_writer_seq_is_exactly_0_to_n_minus_1_in_file_order(tmp_path):
    writer = AttackerAuditWriter(tmp_path, "scan-seq")
    for i in range(5):
        # Deliberately pass a WRONG caller-supplied seq -- the writer must
        # overwrite it, never trust it.
        writer.write(_line(seq=999, content=f"line-{i}"))
    writer.close()

    lines = audit_path_for(tmp_path, "scan-seq").read_text().splitlines()
    seqs = [json.loads(line)["seq"] for line in lines]
    assert seqs == list(range(5))


def test_line_count_equals_number_of_write_calls_for_content_with_embedded_newlines(
    tmp_path,
):
    writer = AttackerAuditWriter(tmp_path, "scan-newlines")
    hostile_contents = [
        "line one\nline two\nline three",
        "control chars: \x00\x01\x1f embedded",
        "trailing newline\n",
        "\n\n\nleading newlines",
        "mixed\r\ncrlf\ncontent",
    ]
    for content in hostile_contents:
        writer.write(_line(content=content))
    writer.close()

    physical_lines = audit_path_for(tmp_path, "scan-newlines").read_text().splitlines()
    assert len(physical_lines) == len(hostile_contents)
    # Every physical line must be independently valid JSON.
    parsed = [json.loads(line) for line in physical_lines]
    assert [p["content"] for p in parsed] == hostile_contents


def test_writer_never_raises_on_unpaired_surrogate_content(tmp_path):
    writer = AttackerAuditWriter(tmp_path, "scan-surrogate")
    hostile = "before \ud800 after \ude00 end"
    writer.write(_line(content=hostile))
    writer.close()

    lines = audit_path_for(tmp_path, "scan-surrogate").read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["content"] == hostile


def test_read_back_line_by_line_yields_same_order_as_written(tmp_path):
    writer = AttackerAuditWriter(tmp_path, "scan-order")
    expected = [f"event-{i}" for i in range(10)]
    for content in expected:
        writer.write(_line(content=content))
    writer.close()

    lines = audit_path_for(tmp_path, "scan-order").read_text().splitlines()
    actual = [json.loads(line)["content"] for line in lines]
    assert actual == expected


def test_writer_creates_missing_output_directory_with_restrictive_mode(tmp_path):
    output_dir = tmp_path / "fresh-output"
    assert not output_dir.exists()
    writer = AttackerAuditWriter(output_dir, "scan-fresh")
    writer.close()
    assert output_dir.exists()
    assert (output_dir.stat().st_mode & 0o777) == 0o700


def test_writer_tightens_mode_of_a_pre_existing_looser_directory(tmp_path):
    output_dir = tmp_path / "loose-output"
    output_dir.mkdir(mode=0o777)
    output_dir.chmod(0o777)  # mkdir's mode arg is umask-affected; force it
    writer = AttackerAuditWriter(output_dir, "scan-loose")
    writer.close()
    assert (output_dir.stat().st_mode & 0o777) == 0o700


# --- close() idempotency and post-close behavior ----------------------------


def test_close_is_idempotent(tmp_path):
    writer = AttackerAuditWriter(tmp_path, "scan-close")
    writer.write(_line())
    writer.close()
    writer.close()  # must not raise
    writer.close()  # must not raise


def test_write_after_close_raises_rather_than_silently_dropping(tmp_path):
    writer = AttackerAuditWriter(tmp_path, "scan-postclose")
    writer.write(_line())
    writer.close()
    with pytest.raises(RuntimeError):
        writer.write(_line(content="dropped?"))
    # The already-written line must still be intact -- one line, not zero.
    lines = audit_path_for(tmp_path, "scan-postclose").read_text().splitlines()
    assert len(lines) == 1
