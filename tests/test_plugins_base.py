"""Tests for BaseModule ABC (src/llmsec/plugins/base.py)."""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from llmsec.models import EvalResult, ScanContext, TargetResponse, TestCase
from llmsec.plugins.base import BaseModule


class _FullyImplementedModule(BaseModule):
    id = "fully_implemented"
    name = "Fully Implemented Test Module"
    owasp_ref = "LLM07:2025"

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


class _PartiallyImplementedModule(BaseModule):
    id = "partially_implemented"
    name = "Partially Implemented Test Module"
    owasp_ref = "LLM07:2025"

    async def generate_cases(self, context: ScanContext) -> AsyncIterator[TestCase]:
        yield TestCase(case_id="c1", prompt="p", technique_id="t1")

    # `evaluate` intentionally NOT implemented


def test_base_module_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseModule()


def test_fully_implemented_subclass_instantiates():
    module = _FullyImplementedModule()
    assert module.id == "fully_implemented"


def test_partially_implemented_subclass_raises_type_error():
    with pytest.raises(TypeError):
        _PartiallyImplementedModule()
