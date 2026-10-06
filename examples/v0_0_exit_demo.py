"""The v0.0 exit demonstration: five semantic objects survive safe reload.

This is a finite infrastructure control.  It does not identify the displayed
nonassociative table with a named algebra or make a scientific claim.
"""

from __future__ import annotations

import json
from typing import cast

from anyalgebra.algebra.multilinear import FiniteMultilinearStructure
from anyalgebra.core.domains import DomainElement
from anyalgebra.linear.matrix import Matrix
from anyalgebra.persistence.semantic import (
    AlgebraPresentationBundle,
    FinitePartialOperationRecord,
    LawValidationReportRecord,
    RuleGeneratedDeductiveRecord,
    TableAlgebraRecord,
    semantic_record_registry,
)
from anyalgebra.structures.deductive import DeductiveSystem, closure
from anyalgebra.structures.operations import PartialOperation
from anyalgebra.structures.outcomes import Defined, Indeterminate, Undefined
from anyalgebra.validation.validate import Disproved


def _fixture() -> tuple[
    PartialOperation,
    DeductiveSystem,
    FiniteMultilinearStructure,
    AlgebraPresentationBundle,
    Disproved,
]:
    """Build exactly the five generic semantic values required for v0.0 exit."""
    partial = FinitePartialOperationRecord.create(
        ("a", "b", "c"), (("a", "a", "a"), ("a", "b", "c"), ("b", "c", None))
    ).rebuild()
    deduction = RuleGeneratedDeductiveRecord.create(
        ((("seed",), "middle"), (("middle",), "goal"))
    ).rebuild()
    algebra = TableAlgebraRecord.create(
        ("e0", "e1"), ((1, 0), (0, 1), (1, 0), (1, 0))
    ).rebuild()
    presentations = AlgebraPresentationBundle.from_algebra(algebra)
    report = LawValidationReportRecord.create(
        ("e0", "e1"), ("e0", "e1", "e0", "e0"), ("e1", "e0", "e1")
    ).rebuild_and_validate()
    assert type(report) is Disproved
    return partial, deduction, algebra, presentations, report


def _closure_names(value: DeductiveSystem) -> list[str]:
    """Reproduce the declared closure using the seed in the reloaded rules."""
    seed = value.rules[0].premises[0]
    result = closure((seed,), value, max_rounds=2)
    return [term.symbol.name for term in result.conclusions if term.symbol is not None]


def _table(value: FiniteMultilinearStructure) -> tuple[tuple[int, ...], ...]:
    """Compare independently reparented algebras by exact table data only."""
    cells: list[tuple[int, ...]] = []
    for left in range(value.module.rank):
        for right in range(value.module.rank):
            coordinates = value.evaluate_basis(left, right).coordinates()
            cell: list[int] = []
            for index in range(value.module.rank):
                if index not in coordinates:
                    cell.append(0)
                    continue
                coefficient = coordinates[index].value
                if type(coefficient) is not int:
                    raise RuntimeError("exit demo requires exact integer coefficients")
                cell.append(coefficient)
            cells.append(tuple(cell))
    return tuple(cells)


def _partial_grid(value: PartialOperation) -> tuple[tuple[str, object], ...]:
    """Project every finite partial-operation cell into a parent-independent result."""
    cells: list[tuple[str, object]] = []
    for left in value.output_carrier:
        for right in value.output_carrier:
            outcome = value.apply(left, right)
            if type(outcome) is Defined:
                cells.append(("Defined", outcome.value))
            elif type(outcome) is Undefined or type(outcome) is Indeterminate:
                cells.append((type(outcome).__name__, outcome.reason))
            else:
                cells.append((type(outcome).__name__, None))
    return tuple(cells)


def _deduction_rules(value: DeductiveSystem) -> tuple[tuple[tuple[str, ...], str], ...]:
    """Project a reloadable ground rule system without retaining source identities."""
    return tuple(
        (
            tuple(
                term.symbol.name for term in rule.premises if term.symbol is not None
            ),
            cast(str, rule.conclusion.symbol.name if rule.conclusion.symbol else None),
        )
        for rule in value.rules
    )


def _matrix_projection(
    matrices: tuple[Matrix, ...],
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Compare matrix presentations by entries, not distinct matrix-parent identity."""
    result: list[tuple[tuple[int, ...], ...]] = []
    for matrix in matrices:
        rows: list[tuple[int, ...]] = []
        for row in matrix.entries:
            entries: list[int] = []
            for entry in row:
                value = cast(DomainElement[object], entry).value
                if type(value) is not int:
                    raise RuntimeError(
                        "exit demo requires exact integer matrix entries"
                    )
                entries.append(value)
            rows.append(tuple(entries))
        result.append(tuple(rows))
    return tuple(result)


def _associator_witness(
    value: FiniteMultilinearStructure,
) -> tuple[tuple[int, int, int], tuple[int, ...], tuple[int, ...]] | None:
    """Find a basis-triple associator using the algebra's actual products."""
    table = _table(value)
    rank = value.module.rank
    for left in range(rank):
        for middle in range(rank):
            for right in range(rank):
                left_product = table[left * rank + middle]
                right_product = table[middle * rank + right]
                left_associated = tuple(
                    sum(
                        coefficient * table[index * rank + right][output]
                        for index, coefficient in enumerate(left_product)
                    )
                    for output in range(rank)
                )
                right_associated = tuple(
                    sum(
                        coefficient * table[left * rank + index][output]
                        for index, coefficient in enumerate(right_product)
                    )
                    for output in range(rank)
                )
                if left_associated != right_associated:
                    return (left, middle, right), left_associated, right_associated
    return None


def run_demo() -> dict[str, object]:
    """Serialize each semantic object through the fixed allow-list, then recompute."""
    original = _fixture()
    original_report = original[4]
    assert original_report.witness is not None
    registry = semantic_record_registry()
    records = tuple(registry.to_record(value) for value in original)
    assert [(record["schemaType"], record["schemaVersion"]) for record in records] == [
        ("finite_partial_operation", 1),
        ("rule_generated_deductive", 1),
        ("table_algebra", 1),
        ("algebra_presentations", 1),
        ("law_validation_report", 1),
    ]
    reloaded = tuple(registry.from_record(record) for record in records)
    partial, deduction, algebra, presentations, report = reloaded
    assert type(partial) is PartialOperation
    assert type(deduction) is DeductiveSystem
    assert type(algebra) is FiniteMultilinearStructure
    assert type(presentations) is AlgebraPresentationBundle
    assert type(report) is Disproved
    assert report.witness is not None

    partial_defined = partial.apply("a", "b")
    partial_undefined = partial.apply("b", "c")
    symbolic, coordinates, constants, matrices = presentations.presentations()
    original_symbolic, original_coordinates, original_constants, original_matrices = (
        original[3].presentations()
    )
    expected_constants = _table(algebra)
    witness = [
        report.structure.carriers[0][index]
        for index in report.witness.substitution_indices
    ]
    associator = _associator_witness(algebra)
    return {
        "record_types": [type(value).__name__ for value in reloaded],
        "round_trips": {
            "algebra": _table(algebra) == _table(original[2]),
            "deduction": _deduction_rules(deduction) == _deduction_rules(original[1]),
            "partial_operation": _partial_grid(partial) == _partial_grid(original[0]),
            "presentations": (
                symbolic == original_symbolic
                and coordinates == original_coordinates
                and constants == original_constants
                and _matrix_projection(matrices)
                == _matrix_projection(original_matrices)
            ),
            "validation_report": (
                report.witness.substitution_indices
                == original_report.witness.substitution_indices
                and report.evaluated_assignments
                == original_report.evaluated_assignments
                and report.expected_assignments == original_report.expected_assignments
                and tuple(output for _, output in report.structure.operations[0].table)
                == tuple(
                    output
                    for _, output in original_report.structure.operations[0].table
                )
            ),
        },
        "partial_results": {
            "defined": cast(Defined[object], partial_defined).value,
            "undefined": type(partial_undefined).__name__ == "Undefined",
        },
        "deductive_closure": _closure_names(deduction),
        "algebra_nonassociative": associator is not None,
        "presentations_agree": (
            coordinates == ((1, 0), (0, 1))
            and constants == expected_constants
            and len(matrices) == len(symbolic)
        ),
        "validation_status": "disproved",
        "validation_witness": witness,
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), sort_keys=True, separators=(",", ":")))
