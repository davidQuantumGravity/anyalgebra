"""V00-093 release example for generic composition-algebra table fixtures."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from anyalgebra.algebra.multilinear import FiniteMultilinearStructure
from anyalgebra.core.domains import QQ
from anyalgebra.fixtures.composition import CompositionFixtureError
from examples import generic_composition_algebras


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    """Compute the bounded exact release report once for this test module."""
    return generic_composition_algebras.run_example()


@pytest.mark.parametrize(
    "identifier",
    ("algmul.H.v1", "algmul.Hs.v1", "algmul.O.v1", "algmul.Os.v1"),
)
def test_each_named_table_is_loaded_as_one_generic_qq_structure(
    identifier: str,
) -> None:
    structure = generic_composition_algebras.load_analysis_fixture(identifier)

    assert type(structure) is FiniteMultilinearStructure
    assert structure.name == identifier
    assert structure.module.domain is QQ()
    assert structure.arity == 2
    assert structure.module.rank == (
        4 if identifier in {"algmul.H.v1", "algmul.Hs.v1"} else 8
    )
    assert not hasattr(generic_composition_algebras, "quaternion_fixture")
    assert not hasattr(generic_composition_algebras, "octonion_fixture")


def test_associative_fixtures_have_complete_theorem_backed_reports(
    report: dict[str, Any],
) -> None:
    by_id = {item["fixture_id"]: item for item in report["fixtures"]}

    for identifier in ("algmul.H.v1", "algmul.Hs.v1"):
        item = by_id[identifier]
        law = item["associativity"]
        assert law["status"] == "Proved"
        assert law["cases_expected"] == law["cases_evaluated"] == 64
        assert law["case_coverage_complete"] is True
        assert law["termination"] == "complete_basis_grid"
        assert law["witness"] is None
        assert law["reduction_complete"] is True
        assert law["reduction_hypothesis"] == (
            "operation is the exact bilinear FiniteMultilinearStructure table over QQ"
        )
        assert law["claim_scope"] == (
            "associativity on every vector in the free QQ module for this "
            "exact table, by trilinearity of the associator"
        )


def test_nonassociative_fixtures_have_deterministic_minimal_witnesses(
    report: dict[str, Any],
) -> None:
    by_id = {item["fixture_id"]: item for item in report["fixtures"]}
    expected_coefficients = {"algmul.O.v1": 2, "algmul.Os.v1": -2}

    for identifier, coefficient in expected_coefficients.items():
        item = by_id[identifier]
        law = item["associativity"]
        assert law["status"] == "Disproved"
        assert law["cases_expected"] == 512
        assert law["cases_evaluated"] == 85
        assert law["case_coverage_complete"] is False
        assert law["termination"] == "first_counterexample"
        assert law["claim_scope"] == (
            "nonassociativity of this exact QQ table, established by "
            "an explicit declared-basis-vector counterexample"
        )
        assert law["witness"] == {
            "indices": [1, 2, 4],
            "labels": ["e1", "e2", "e4"],
            "nonzero_difference_coordinates": [
                {
                    "basis": "e7",
                    "coefficient": {"numerator": coefficient, "denominator": 1},
                }
            ],
        }


def test_report_states_finite_grid_infinite_coefficient_scope_and_bounds(
    report: dict[str, Any],
) -> None:
    for item in report["fixtures"]:
        assert item["table_coefficient_domain"] == "ZZ"
        assert item["analysis_coefficient_domain"] == "QQ"
        assert item["analysis_coefficient_domain_is_finite"] is False
        assert item["substitution_domain"] == "declared ordered basis triples"
        assert item["substitution_domain_is_finite"] is True
        assert item["enumeration_order"] == "lexicographic"
        assert item["sampling"] == "none"
        assert item["fingerprint"]["bounds"] == {
            "max_dimension": 32,
            "max_constraints": 25_000,
            "max_elimination_work": 3_000_000,
        }
        assert item["fingerprint"]["isomorphism_rejection_only"] is True
        assert item["fingerprint"]["ideal_classification_status"] == (
            "incomplete_canonical_discovery"
        )
        assert (
            "canonical ideal discovery is incomplete over QQ"
            in item["fingerprint"]["conventions"]
        )

    assert report["scope"] == (
        "infrastructure evidence for four exact named table conventions; "
        "no specialized algebra-family API and no scientific claim"
    )


def test_sampled_pass_remains_inconclusive_and_cannot_replace_the_exact_report(
    report: dict[str, Any],
) -> None:
    sample = report["sampled_pass_nonproof"]

    assert sample == {
        "fixture_id": "algmul.O.v1",
        "law": "associativity",
        "status": "Inconclusive",
        "population": "512 ordered basis triples",
        "method": "deterministic_lexicographic_prefix",
        "sample_size": 8,
        "cases_evaluated": 8,
        "passing_cases": 8,
        "seed": None,
        "distribution": "lexicographic_prefix_not_random",
        "shrinker": "none",
        "reason": "a passing proper subset is not an exhaustive proof",
    }
    exact = next(
        item for item in report["fixtures"] if item["fixture_id"] == "algmul.O.v1"
    )
    assert exact["associativity"]["status"] == "Disproved"


def test_exact_basis_change_preserves_only_the_declared_fingerprint_data(
    report: dict[str, Any],
) -> None:
    invariance = report["basis_change_invariance"]

    assert invariance == {
        "fixture_id": "algmul.H.v1",
        "coefficient_domain": "QQ",
        "change": "last basis vector maps to first plus last",
        "certificate_algorithm": "explicit-basis-composites-v1",
        "expected_composites": 8,
        "evaluated_composites": 8,
        "exact_certified": True,
        "source_associativity_status": "Proved",
        "target_associativity_status": "Proved",
        "fingerprint_equal": True,
        "fingerprint_role": "isomorphism_rejection_data_not_a_complete_invariant",
    }


def test_invalid_fixture_id_is_a_typed_boundary() -> None:
    with pytest.raises(CompositionFixtureError, match="not registered"):
        generic_composition_algebras.load_analysis_fixture("H")


def test_executable_report_retains_the_invalid_identifier_boundary(
    report: dict[str, Any],
) -> None:
    assert report["invalid_fixture_boundary"] == {
        "input": "H",
        "error_type": "CompositionFixtureError",
        "message": "fixture identifier is not registered",
    }


def test_cli_is_deterministic_json_and_claim_bounded(
    report: dict[str, Any], tmp_path: Path
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "examples" / "generic_composition_algebras.py"),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=1800,  # a hang guard, not a performance budget
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == report
    assert (
        completed.stdout
        == json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
    )
    assert all(
        phrase not in completed.stdout.lower()
        for phrase in ("scientifically verified", "research claim confirmed")
    )
