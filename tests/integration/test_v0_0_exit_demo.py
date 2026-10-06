"""Release exit-demo contract for semantic records, not evidence wrappers."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from anyalgebra.algebra.multilinear import BasisProductTable, FiniteMultilinearStructure
from anyalgebra.core.domains import QQ
from anyalgebra.core.modules import Basis, FreeModule
from anyalgebra.core.parents import FiniteCarrier, Sort
from anyalgebra.persistence.registry import SchemaError
from anyalgebra.persistence.semantic import semantic_record_registry
from anyalgebra.structures.deductive import DeductiveSystem, InferenceRule
from anyalgebra.structures.operations import PartialOperation
from anyalgebra.structures.outcomes import Defined
from anyalgebra.structures.signatures import OperationSymbol
from anyalgebra.structures.terms import Term
from anyalgebra.validation.factories import commutative_law
from anyalgebra.validation.validate import Disproved, validate_law
from examples import v0_0_exit_demo


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_exit_demo_round_trips_five_semantic_objects_and_reproduces_results() -> None:
    """The release demo reloads actual structures, then recomputes their results."""
    result = v0_0_exit_demo.run_demo()

    assert result["record_types"] == [
        "PartialOperation",
        "DeductiveSystem",
        "FiniteMultilinearStructure",
        "AlgebraPresentationBundle",
        "Disproved",
    ]
    assert result["round_trips"] == {
        "algebra": True,
        "deduction": True,
        "partial_operation": True,
        "presentations": True,
        "validation_report": True,
    }
    assert result["partial_results"] == {"defined": "c", "undefined": True}
    assert result["deductive_closure"] == ["seed", "middle", "goal"]
    assert result["algebra_nonassociative"] is True
    assert result["presentations_agree"] is True
    assert result["validation_status"] == "disproved"
    assert result["validation_witness"] == ["e1", "e0", "e1"]


def test_exit_demo_records_reject_unknown_and_malformed_input_without_execution() -> (
    None
):
    """Only the five fixed safe schema tags can reconstruct semantic objects."""
    registry = semantic_record_registry()

    with pytest.raises(SchemaError, match="unknown_schema_type"):
        registry.from_record({"schemaType": "python.object", "schemaVersion": 1})
    malformed: tuple[tuple[str, dict[str, object]], ...] = (
        ("finite_partial_operation", {"carrier": ["a", "a"], "table": []}),
        ("rule_generated_deductive", {"premises": ["seed"], "rules": []}),
        ("table_algebra", {"basis": ["e0"], "cells": []}),
        ("algebra_presentations", {"algebra": {"basis": [], "cells": []}}),
        (
            "law_validation_report",
            {"carrier": ["e0"], "table": ["e0"], "expectedWitness": ["e0"]},
        ),
    )
    for schema_type, body in malformed:
        with pytest.raises(SchemaError, match="parser_failed"):
            registry.from_record(
                {"schemaType": schema_type, "schemaVersion": 1, **body}
            )

    with pytest.raises(SchemaError, match="unknown_schema_version"):
        registry.from_record({"schemaType": "table_algebra", "schemaVersion": 2})
    with pytest.raises(SchemaError, match="invalid_record_value"):
        registry.from_record(
            {"schemaType": "table_algebra", "schemaVersion": 1, "cells": len}
        )


def test_exit_demo_codecs_fail_closed_for_extra_or_tampered_semantic_data() -> None:
    """Every parser consumes exactly its declared body and no ignored witness data."""
    registry = semantic_record_registry()
    for value in v0_0_exit_demo._fixture():
        record = registry.to_record(value)
        record["unexpected"] = 1
        with pytest.raises(SchemaError, match="parser_failed"):
            registry.from_record(record)

    report = registry.to_record(v0_0_exit_demo._fixture()[-1])
    report["expectedWitness"] = ["e0", "e0", "e0"]
    with pytest.raises(SchemaError, match="parser_failed"):
        registry.from_record(report)


def test_exit_demo_codecs_reject_unsupported_semantic_forms() -> None:
    """Reject coercive, callable, heterogeneous, and multi-sort data."""
    registry = semantic_record_registry()
    left, right = Sort("left"), Sort("right")
    left_carrier = FiniteCarrier(("a",), sort=left)
    right_carrier = FiniteCarrier(("b",), sort=right)
    heterogeneous = PartialOperation.from_table(
        OperationSymbol("heterogeneous", (left, right), left),
        ((left, left_carrier), (right, right_carrier)),
        ((("a", "b"), "a"),),
        undefined_marker=object(),
    )
    same_sort = Sort("same")
    same_carrier = FiniteCarrier(("a",), sort=same_sort)
    callable_partial = PartialOperation.from_callable(
        OperationSymbol("callable", (same_sort, same_sort), same_sort),
        ((same_sort, same_carrier),),
        lambda _left, _right: Defined("a"),
    )
    other_sort = Sort("other")
    multi_sorted_system = DeductiveSystem(
        (
            InferenceRule(
                (Term.apply(OperationSymbol("seed", (), same_sort)),),
                Term.apply(OperationSymbol("goal", (), other_sort)),
            ),
        )
    )
    domain = QQ()
    module = FreeModule(domain, Basis(("e",), coefficient_domain=domain))
    rational_algebra = FiniteMultilinearStructure.from_tables(
        module,
        BasisProductTable.from_cells(module, 2, (module.element({0: 1}),)),
    )

    for value in (heterogeneous, multi_sorted_system, rational_algebra):
        with pytest.raises(SchemaError, match="encoder_failed"):
            registry.to_record(value)
    with pytest.raises(SchemaError, match="unregistered_value_type"):
        registry.to_record(callable_partial)

    associative_report = v0_0_exit_demo._fixture()[-1]
    unrelated_report = validate_law(
        associative_report.structure,
        commutative_law(associative_report.structure.operations[0].symbol),
    )
    assert type(unrelated_report) is Disproved
    with pytest.raises(SchemaError, match="encoder_failed"):
        registry.to_record(unrelated_report)


def test_exit_demo_runs_as_clean_compact_cli_from_another_directory(
    tmp_path: Path,
) -> None:
    """The release proof is executable without depending on the current directory."""
    completed = subprocess.run(
        [sys.executable, str(REPOSITORY_ROOT / "examples" / "v0_0_exit_demo.py")],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == v0_0_exit_demo.run_demo()
