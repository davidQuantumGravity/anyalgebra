"""Execution-lifecycle vocabulary remains distinct from terminal receipts."""

from __future__ import annotations

from typing import get_args, get_type_hints

import pytest

from anyalgebra.core.parents import SemanticHash
from anyalgebra.evidence.contracts import CalculationContract
from anyalgebra.evidence.run import (
    CalculationResult,
    ExecutionStatus,
    ReceiptError,
    ResultReceipt,
    TerminalExecutionStatus,
    result_receipt_record,
    run_calculation,
)
import anyalgebra.evidence.run as run_module


def _hash(character: str) -> SemanticHash:
    return SemanticHash("sha256", character * 64)


def _contract() -> CalculationContract:
    return CalculationContract.create(
        contract_name="execution-lifecycle",
        project_id="test.project",
        question="Exercise the terminal receipt boundary.",
        acceptable_outcomes=("constructed",),
        input_hashes=(("input", _hash("a")),),
        source_anchors=(),
        convention_manifests=(),
        algorithms=(("reference", "1"),),
        backend_request="exact",
        backend_name="reference",
        backend_version="1",
        assumptions=(),
        theorem_hypotheses=(),
        bounds=(),
        required_cross_checks=(),
        expected_artifacts=(),
        acceptance_predicates=("stored",),
    )


def test_lifecycle_enum_and_terminal_receipt_type_are_exact() -> None:
    assert tuple(status.value for status in ExecutionStatus) == (
        "not_run",
        "running",
        "completed",
        "failed",
        "blocked",
    )
    assert set(get_args(TerminalExecutionStatus)) == {"completed", "failed"}
    assert get_type_hints(ResultReceipt)["execution_status"] == TerminalExecutionStatus


@pytest.mark.parametrize("status", ("not_run", "running", "blocked"))
def test_nonterminal_lifecycle_values_cannot_construct_receipts(status: str) -> None:
    with pytest.raises(ReceiptError, match="classification"):
        run_module._build_receipt(
            contract=_contract(),
            environment=run_module.ExecutionEnvironment.create(),
            started_at="2026-07-25T00:00:00Z",
            finished_at="2026-07-25T00:00:00Z",
            execution_status=status,
            outcome="constructed",
            summary="constructed fixture",
        )


def test_terminal_receipts_are_string_serialized_and_canonical() -> None:
    receipt = run_calculation(
        _contract(),
        lambda _: CalculationResult.create(
            outcome="constructed", summary="constructed fixture"
        ),
    )
    assert receipt.execution_status == ExecutionStatus.COMPLETED.value
    record = result_receipt_record(receipt)
    assert type(record["executionStatus"]) is str
    assert record["executionStatus"] == "completed"

    def fails(_: CalculationContract) -> object:
        raise RuntimeError("controlled failure")

    failed = run_calculation(_contract(), fails)
    assert failed.execution_status == ExecutionStatus.FAILED.value
    assert failed.mathematical_outcome == "implementation_error"
    failed_record = result_receipt_record(failed)
    assert type(failed_record["executionStatus"]) is str
    assert failed_record["executionStatus"] == "failed"
