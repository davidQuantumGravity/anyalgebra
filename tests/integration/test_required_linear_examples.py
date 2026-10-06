"""Executable public-API contracts for presentation and recovery examples."""

from __future__ import annotations

import json
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Any, cast

from anyalgebra.algebra.multilinear import BasisProductTable, FiniteMultilinearStructure
from anyalgebra.core.domains import QQ
from anyalgebra.linear.matrix import Matrix
from anyalgebra.linear.recovery import RecoveryFailure, RecoverySuccess
from anyalgebra.linear.span import NonClosed
from examples import operator_recovery, presentations, rational_algebra


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _pair(left: Matrix, right: Matrix) -> object:
    """Recompute the public QQ entrywise pairing behind an outside-span witness."""
    total = cast(Any, QQ().element(0))
    for left_row, right_row in zip(left.entries, right.entries, strict=True):
        for left_entry, right_entry in zip(left_row, right_row, strict=True):
            total = total.add(cast(Any, left_entry).multiply(right_entry))
    return total


def test_presentation_example_replays_explicit_round_trips_and_boundary() -> None:
    example = presentations.build_example()
    assert example["symbolic_round_trip"] == example["symbolic"]
    assert example["table_round_trip"] == example["table"]
    assert len(example["ambiguity_candidates"]) == 2
    payload = presentations.run_example()
    assert payload["symbolic_coordinate"] == {
        "round_trip_equal": True,
        "round_trip_id": "example.presentations.symbolic-coordinate",
        "route": [
            ["example.presentations.symbolic-to-coordinate", "forward"],
            ["example.presentations.symbolic-to-coordinate", "inverse"],
        ],
        "validity_domain": "literal-module:example.presentations",
    }
    assert payload["table_structure"] == {
        "round_trip_equal": True,
        "round_trip_id": "example.presentations.table-structure",
        "route": [
            ["example.presentations.table-to-structure", "forward"],
            ["example.presentations.table-to-structure", "inverse"],
        ],
        "validity_domain": "literal-module:example.presentations;arity:2",
    }
    assert payload["ambiguity"] == {
        "automatic_route_selected": False,
        "candidate_routes": [
            [["example.presentations.symbolic-to-coordinate", "forward"]],
            [["example.presentations.alternate-symbolic-coordinate", "forward"]],
        ],
        "container_shape_inference": "not attempted; source kind is explicit",
    }
    assert presentations.run_example() == payload


def test_rational_algebra_example_replays_products_and_transport() -> None:
    example = rational_algebra.build_example()
    expected_basis_products = {
        "p_p": example["module"].zero(),
        "p_q": example["module"].element({0: (3, 2)}),
        "q_p": example["module"].element({1: (-1, 2)}),
        "q_q": example["module"].element({1: 1}),
    }
    assert example["basis_products"] == expected_basis_products
    assert {
        f"{left}_{right}": example["structure"].evaluate_basis(left_index, right_index)
        for left_index, left in enumerate(("p", "q"))
        for right_index, right in enumerate(("p", "q"))
    } == expected_basis_products
    assert example["basis_products"]["p_q"] != example["basis_products"]["q_p"]
    assert (
        example["generic_product"]
        == example["module"].element({0: (-3, 2), 1: -4})
        == example["expected_generic_product"]
    )
    assert example["table_round_trip"] == example["table"]
    rebuilt = FiniteMultilinearStructure.from_tables(
        example["module"], example["table_round_trip"]
    )
    assert rebuilt.operation.constants == example["structure"].operation.constants
    assert (
        BasisProductTable.from_cells(
            example["module"],
            2,
            tuple(
                rebuilt.evaluate_basis(*indices)
                for indices in product(range(2), repeat=2)
            ),
        )
        == example["table"]
    )
    assert example["transported_constants"].entries == (
        ((0, 0, 0), example["domain"].element(1)),
        ((0, 1, 0), example["domain"].element((-1, 2))),
        ((1, 0, 1), example["domain"].element((3, 2))),
    )
    certificate = example["isomorphism"].certificate
    assert (
        certificate.algorithm,
        certificate.source_rank,
        certificate.target_rank,
        certificate.expected_forward_after_inverse_count,
        certificate.evaluated_forward_after_inverse_count,
        certificate.expected_inverse_after_forward_count,
        certificate.evaluated_inverse_after_forward_count,
        certificate.exact_certified,
    ) == ("explicit-basis-composites-v1", 2, 2, 2, 2, 2, 2, True)
    payload = rational_algebra.run_example()
    assert payload["basis_products"] == {
        "p_p": [],
        "p_q": [[0, 3, 2]],
        "q_p": [[1, -1, 2]],
        "q_q": [[1, 1, 1]],
    }
    assert payload["generic_product"] == [[0, -3, 2], [1, -4, 1]]
    assert payload["basis_change"] == {
        "certificate": "explicit-basis-composites-v1",
        "certificate_counts": [2, 2, 2, 2, 2, 2],
        "exact_certified": True,
        "transported_constants": [
            [0, 0, 0, 1, 1],
            [0, 1, 0, -1, 2],
            [1, 0, 1, 3, 2],
        ],
    }
    assert payload["table_constants_round_trip"] is True
    assert rational_algebra.run_example() == payload


def test_operator_recovery_example_replays_all_three_report_types() -> None:
    example = operator_recovery.build_example()
    closed = example["closed"]
    dependent = example["dependent"]
    nonclosed = example["nonclosed"]

    assert type(closed["report"]) is RecoverySuccess
    closed_report = closed["report"]
    assert closed_report.status == "success"
    assert closed["calls"] == 4
    assert closed_report.declaration is closed["declaration"]
    assert closed_report.closure.declaration is closed["declaration"]
    assert closed_report.constants.entries == (
        ((0, 1, 1), QQ().element(1)),
        ((1, 0, 1), QQ().element(-1)),
    )
    assert len(closed_report.reconstruction_checks) == 4
    assert all(
        check.exact and check.captured_product == check.reconstructed_product
        for check in closed_report.reconstruction_checks
    )
    assert type(dependent["report"]) is RecoveryFailure
    dependent_report = dependent["report"]
    assert dependent_report.status == "failure"
    assert dependent["calls"] == 0
    assert dependent_report.declaration is dependent["declaration"]
    assert len(dependent_report.dependencies) == 1
    assert dependent_report.dependencies[0].entries == (
        (QQ().element(-1),),
        (QQ().element(1),),
    )
    assert operator_recovery.dependency_replays(
        dependent["family"], dependent_report.dependencies[0]
    )
    assert type(nonclosed["report"]) is RecoveryFailure
    nonclosed_report = nonclosed["report"]
    assert nonclosed_report.status == "failure"
    assert nonclosed["calls"] == 1
    assert nonclosed_report.declaration is nonclosed["declaration"]
    assert type(nonclosed_report.closure_evidence) is NonClosed
    evidence = nonclosed_report.closure_evidence
    assert evidence.declaration is nonclosed["declaration"]
    assert (evidence.left_index, evidence.right_index) == (0, 0)
    assert evidence.product == evidence.product.parent.element(((0, 1),))
    witness = evidence.witness
    assert _pair(witness.functional, evidence.product) == witness.target_pairing
    assert witness.target_pairing.value.numerator != 0
    assert all(
        _pair(witness.functional, operator) == QQ().element(0)
        for operator in evidence.basis
    )
    assert operator_recovery.nonclosure_witness_replays(nonclosed_report)
    payload = operator_recovery.run_example()
    assert payload["closed"] == {
        "bracket_calls": 4,
        "constants": [[0, 1, 1, 1, 1], [1, 0, 1, -1, 1]],
        "reconstruction_exact": [True, True, True, True],
        "status": "success",
    }
    assert payload["dependent"] == {
        "bracket_calls": 0,
        "relation": [[[-1, 1]], [[1, 1]]],
        "relation_replays": True,
        "status": "failure",
    }
    assert payload["nonclosed"] == {
        "bracket_calls": 1,
        "first_pair": [0, 0],
        "functional": [[[0, 1], [1, 1]]],
        "product": [[[0, 1], [1, 1]]],
        "separating_pairing": [1, 1],
        "status": "failure",
        "witness_replays": True,
    }
    assert operator_recovery.run_example() == payload


def test_linear_example_clis_are_compact_json_and_cwd_independent(
    tmp_path: Path,
) -> None:
    scripts = (
        ("presentations.py", presentations.run_example()),
        ("rational_algebra.py", rational_algebra.run_example()),
        ("operator_recovery.py", operator_recovery.run_example()),
    )
    for filename, expected in scripts:
        command = [sys.executable, str(REPOSITORY_ROOT / "examples" / filename)]
        first = subprocess.run(
            command, cwd=tmp_path, check=False, capture_output=True, text=True
        )
        second = subprocess.run(
            command, cwd=tmp_path, check=False, capture_output=True, text=True
        )
        assert first.returncode == second.returncode == 0
        assert first.stderr == second.stderr == ""
        assert first.stdout == second.stdout
        assert json.loads(first.stdout) == expected
        assert (
            first.stdout
            == json.dumps(expected, sort_keys=True, separators=(",", ":")) + "\n"
        )
        assert "0x" not in first.stdout and "<callable" not in first.stdout
        scope = expected["scope"]
        assert type(scope) is str and "no research claim verified" in scope
