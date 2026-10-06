"""Executable contracts for the required arbitrary-structure examples."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from anyalgebra.structures.deductive import CompleteClosure
from anyalgebra.structures.outcomes import Defined, Undefined
from anyalgebra.structures.relations import RelationResult
from anyalgebra.structures.structure import Structure
from anyalgebra.structures.operations import PartialOperation

from examples import deductive_system, partial_magma, two_sorted_structure


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_partial_magma_example_keeps_undefinedness_explicit() -> None:
    example = partial_magma.build_example()
    operation = example["operation"]

    assert isinstance(operation, PartialOperation)
    assert operation.totality == "partial"
    assert operation.apply("a", "b") == Defined("c")
    missing = operation.apply("b", "c")
    assert isinstance(missing, Undefined)
    assert missing.reason == "table cell explicitly marked undefined"
    assert partial_magma.run_example() == {
        "defined": "c",
        "explicit_undefined": "table cell explicitly marked undefined",
        "omitted_undefined": "table cell is not declared",
        "totality": "partial",
    }


def test_two_sorted_example_assembles_and_exercises_operation_and_relation() -> None:
    example = two_sorted_structure.build_example()
    structure = example["structure"]
    operation = example["operation"]
    relation = example["relation"]

    assert isinstance(structure, Structure)
    assert operation.symbol.arity == 3
    assert len(set(operation.symbol.inputs)) == 2
    evaluation = example["evaluation"]
    assert evaluation.outcome == Defined("right")
    assert evaluation.path == ()
    holds = relation.apply("right", "blue")
    does_not_hold = relation.apply("left", "blue")
    assert holds == RelationResult(True)
    assert does_not_hold == RelationResult(False)
    assert two_sorted_structure.run_example() == {
        "evaluation": "right",
        "relation_false": False,
        "relation_true": True,
        "ternary_arity": 3,
    }


def test_deductive_example_returns_complete_modus_ponens_trace() -> None:
    example = deductive_system.build_example()
    result = example["closure"]

    assert isinstance(result, CompleteClosure)
    assert [
        term.symbol.name for term in result.conclusions if term.symbol is not None
    ] == [
        "rain",
        "rain_implies_wet",
        "wet",
    ]
    trace = result.derivations[0]
    assert trace.rule_index == 0
    assert trace.premise_indices == (0, 1)
    assert tuple(result.conclusions[index] for index in trace.premise_indices) == (
        trace.premises
    )
    assert deductive_system.run_example() == {
        "conclusions": ["rain", "rain_implies_wet", "wet"],
        "premise_indices": [0, 1],
        "status": "complete",
    }


def test_required_examples_run_as_compact_stable_cli_from_another_cwd(
    tmp_path: Path,
) -> None:
    scripts = (
        ("partial_magma.py", partial_magma.run_example()),
        ("two_sorted_structure.py", two_sorted_structure.run_example()),
        ("deductive_system.py", deductive_system.run_example()),
    )
    for filename, expected in scripts:
        completed = subprocess.run(
            [sys.executable, str(REPOSITORY_ROOT / "examples" / filename)],
            cwd=tmp_path,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stderr == ""
        assert json.loads(completed.stdout) == expected
