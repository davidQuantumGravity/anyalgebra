"""Three exact operator-recovery outcomes; this example makes no research claim."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any, TypedDict, cast

from anyalgebra.core.domains import QQ
from anyalgebra.linear.matrix import Matrix, MatrixSpace
from anyalgebra.linear.recovery import RecoveryFailure, RecoveryReport, RecoverySuccess
from anyalgebra.linear.recovery import recover_structure_constants
from anyalgebra.linear.span import NonClosed, OutsideSpan, DeclaredBilinearOperation


class RecoveryScenario(TypedDict):
    """One family, report, and observable number of user-bracket invocations."""

    calls: int
    declaration: DeclaredBilinearOperation
    family: tuple[Matrix, ...]
    report: RecoveryReport


class OperatorRecoveryExample(TypedDict):
    """The three non-interchangeable recovery outcomes retained for replay."""

    closed: RecoveryScenario
    dependent: RecoveryScenario
    nonclosed: RecoveryScenario


def _matrix_entries(value: Matrix) -> list[list[list[int]]]:
    """Serialize QQ matrices without implementation addresses or callable reprs."""
    return [
        [
            [
                cast(Any, entry).value.numerator,
                cast(Any, entry).value.denominator,
            ]
            for entry in row
        ]
        for row in value.entries
    ]


def dependency_replays(family: tuple[Matrix, ...], relation: Matrix) -> bool:
    """Check that a reported coordinate relation combines the family to zero."""
    if relation.parent.rows != len(family) or relation.parent.columns != 1:
        return False
    space = family[0].parent
    if any(operator.parent is not space for operator in family):
        return False
    entries = tuple(
        tuple(
            _sum(
                cast(Any, operator.entry(row, column)).multiply(
                    relation.entry(index, 0)
                )
                for index, operator in enumerate(family)
            )
            for column in range(space.columns)
        )
        for row in range(space.rows)
    )
    return space.element(entries) == space.zero()


def _sum(values: Iterable[object]) -> object:
    """Add a finite QQ scalar iterable without depending on a matrix backend."""
    total = cast(Any, QQ().element(0))
    for value in values:
        total = total.add(value)
    return total


def _pair(left: Matrix, right: Matrix) -> object:
    """Evaluate the public entrywise QQ pairing used by span witnesses."""
    return _sum(
        cast(Any, left_entry).multiply(right_entry)
        for left_row, right_row in zip(left.entries, right.entries, strict=True)
        for left_entry, right_entry in zip(left_row, right_row, strict=True)
    )


def nonclosure_witness_replays(report: RecoveryReport) -> bool:
    """Check the public separating functional against its product and basis."""
    if (
        type(report) is not RecoveryFailure
        or type(report.closure_evidence) is not NonClosed
    ):
        return False
    witness = report.closure_evidence.witness
    if type(witness) is not OutsideSpan:
        return False
    functional = witness.functional
    product = report.closure_evidence.product
    pairing = cast(Any, _pair(functional, product))
    if pairing != witness.target_pairing or pairing.value.numerator == 0:
        return False
    return all(
        cast(Any, _pair(functional, generator)).value.numerator == 0
        for generator in report.closure_evidence.basis
    )


def build_example() -> OperatorRecoveryExample:
    """Build closed, dependent, and first-pair nonclosed recovery reports."""
    space = MatrixSpace(1, 2, QQ())
    e0 = space.element(((1, 0),))
    e1 = space.element(((0, 1),))

    closed_calls = 0

    def closed_bracket(left: Matrix, right: Matrix) -> Matrix:
        nonlocal closed_calls
        closed_calls += 1
        if left is e0 and right is e1:
            return e1
        if left is e1 and right is e0:
            return space.element(((0, -1),))
        return space.zero()

    closed_declaration = DeclaredBilinearOperation.create(
        space, closed_bracket, bilinear=True
    )
    closed_report = recover_structure_constants((e0, e1), closed_declaration)
    dependent_calls = 0

    def dependent_bracket(left: Matrix, right: Matrix) -> Matrix:
        nonlocal dependent_calls
        dependent_calls += 1
        return space.zero()

    dependent_declaration = DeclaredBilinearOperation.create(
        space, dependent_bracket, bilinear=True
    )
    dependent_report = recover_structure_constants((e0, e0), dependent_declaration)
    nonclosed_calls = 0

    def nonclosed_bracket(left: Matrix, right: Matrix) -> Matrix:
        nonlocal nonclosed_calls
        nonclosed_calls += 1
        return e1

    nonclosed_declaration = DeclaredBilinearOperation.create(
        space, nonclosed_bracket, bilinear=True
    )
    nonclosed_report = recover_structure_constants((e0,), nonclosed_declaration)
    return {
        "closed": {
            "calls": closed_calls,
            "declaration": closed_declaration,
            "family": (e0, e1),
            "report": closed_report,
        },
        "dependent": {
            "calls": dependent_calls,
            "declaration": dependent_declaration,
            "family": (e0, e0),
            "report": dependent_report,
        },
        "nonclosed": {
            "calls": nonclosed_calls,
            "declaration": nonclosed_declaration,
            "family": (e0,),
            "report": nonclosed_report,
        },
    }


def run_example() -> dict[str, object]:
    """Return compact report metadata and replayable exact witnesses."""
    example = build_example()
    closed = example["closed"]["report"]
    dependent = example["dependent"]["report"]
    nonclosed = example["nonclosed"]["report"]
    if type(closed) is not RecoverySuccess:
        raise RuntimeError("closed operator family did not recover constants")
    if type(dependent) is not RecoveryFailure or not dependent.dependencies:
        raise RuntimeError("dependent operator family did not return a relation")
    if (
        type(nonclosed) is not RecoveryFailure
        or type(nonclosed.closure_evidence) is not NonClosed
    ):
        raise RuntimeError("nonclosed operator family did not return a witness")
    evidence = nonclosed.closure_evidence
    witness = evidence.witness
    if type(witness) is not OutsideSpan:
        raise RuntimeError("nonclosed report did not retain an outside-span witness")
    return {
        "closed": {
            "bracket_calls": example["closed"]["calls"],
            "constants": [
                [
                    *key,
                    cast(Any, coefficient).value.numerator,
                    cast(Any, coefficient).value.denominator,
                ]
                for key, coefficient in closed.constants.entries
            ],
            "reconstruction_exact": [
                check.exact for check in closed.reconstruction_checks
            ],
            "status": closed.status,
        },
        "dependent": {
            "bracket_calls": example["dependent"]["calls"],
            "relation": _matrix_entries(dependent.dependencies[0]),
            "relation_replays": dependency_replays(
                example["dependent"]["family"], dependent.dependencies[0]
            ),
            "status": dependent.status,
        },
        "nonclosed": {
            "bracket_calls": example["nonclosed"]["calls"],
            "first_pair": [evidence.left_index, evidence.right_index],
            "functional": _matrix_entries(witness.functional),
            "product": _matrix_entries(evidence.product),
            "separating_pairing": [
                cast(Any, witness.target_pairing).value.numerator,
                cast(Any, witness.target_pairing).value.denominator,
            ],
            "status": nonclosed.status,
            "witness_replays": nonclosure_witness_replays(nonclosed),
        },
        "scope": (
            "finite exact recovery infrastructure example only; "
            "no research claim verified"
        ),
    }


if __name__ == "__main__":
    print(json.dumps(run_example(), sort_keys=True, separators=(",", ":")))
