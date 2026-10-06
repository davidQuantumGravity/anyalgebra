"""Run exact, convention-scoped reports on four generic table fixtures.

The familiar algebra names select immutable fixtures only.  Every calculation
uses ``FiniteMultilinearStructure`` and the generic table, structure-constant,
fingerprint, and basis-change APIs.  The finite substitution grid and the
infinite coefficient domain are reported separately.
"""

from __future__ import annotations

import json
from typing import Final, cast

from anyalgebra.algebra.multilinear import (
    FiniteMultilinearStructure,
    StructureConstants,
)
from anyalgebra.analysis.elementary import AnalysisBounds, associator
from anyalgebra.analysis.fingerprint import (
    AlgebraFingerprint,
    LawReport,
    algebra_fingerprint,
)
from anyalgebra.core.domains import QQ
from anyalgebra.core.modules import Basis, FreeModule
from anyalgebra.fixtures.composition import (
    CompositionFixtureError,
    composition_fixture,
)
from anyalgebra.presentations.basis_change import (
    ExactBasisIsomorphism,
    change_basis,
)


FIXTURE_IDS: Final = (
    "algmul.H.v1",
    "algmul.Hs.v1",
    "algmul.O.v1",
    "algmul.Os.v1",
)
FINGERPRINT_BOUNDS: Final = AnalysisBounds(
    max_dimension=32,
    max_constraints=25_000,
    max_elimination_work=3_000_000,
)
SCOPE: Final = (
    "infrastructure evidence for four exact named table conventions; "
    "no specialized algebra-family API and no scientific claim"
)


def load_analysis_fixture(identifier: str) -> FiniteMultilinearStructure:
    """Load a named generic ZZ table and rebuild it over exact QQ.

    The lift changes only the coefficient parent.  It preserves the declared
    basis order and every integral structure constant.  ``QQ`` is required by
    the bounded fingerprint API and is infinite; the later finite enumeration
    is over ordered basis substitutions, not over all coefficients.
    """
    source = composition_fixture(identifier)
    domain = QQ()
    module = FreeModule(
        domain,
        Basis(source.module.basis.labels, coefficient_domain=domain),
    )
    constants = StructureConstants.from_sparse(
        (module.basis, module.basis),
        module.basis,
        tuple(
            (key, domain.element(coefficient.value))
            for key, coefficient in source.operation.constants.entries
        ),
    )
    return FiniteMultilinearStructure.from_structure_constants(
        module, constants, name=source.name
    )


def _witness(report: LawReport, labels: tuple[str, ...]) -> dict[str, object] | None:
    """Serialize one exact sparse rational witness without lossy conversion."""
    if report.witness_indices is None:
        return None
    assert report.witness_value is not None
    return {
        "indices": list(report.witness_indices),
        "labels": [labels[index] for index in report.witness_indices],
        "nonzero_difference_coordinates": [
            {
                "basis": labels[index],
                "coefficient": {
                    "numerator": coefficient.numerator,
                    "denominator": coefficient.denominator,
                },
            }
            for index, coefficient in enumerate(report.witness_value)
            if coefficient.numerator != 0
        ],
    }


def _fingerprint_record(value: AlgebraFingerprint) -> dict[str, object]:
    """Keep conventions, resource bounds, and rejection-only role explicit."""
    return {
        "dimension": value.dimension,
        "unit_status": value.unit_status,
        "commutative": value.commutative,
        "associative": value.associative,
        "nucleus_dimension": value.nucleus_dimension,
        "center_dimension": value.center_dimension,
        "derivation_dimension": value.derivation_dimension,
        "ideal_classification_status": value.ideal_classification_status,
        "conventions": list(value.conventions),
        "bounds": {
            "max_dimension": value.bounds.max_dimension,
            "max_constraints": value.bounds.max_constraints,
            "max_elimination_work": value.bounds.max_elimination_work,
        },
        "isomorphism_rejection_only": value.isomorphism_rejection_only,
        "algorithm": value.algorithm,
        "algorithm_version": value.algorithm_version,
    }


def _associativity_record(
    algebra: FiniteMultilinearStructure, fingerprint: AlgebraFingerprint
) -> dict[str, object]:
    """Serialize complete proof or first-counterexample evidence honestly."""
    report = fingerprint.associativity_report
    proved = report.status == "Proved"
    return {
        "law": "associativity",
        "status": report.status,
        "cases_expected": report.expected_assignments,
        "cases_evaluated": report.checked_assignments,
        "case_coverage_complete": proved,
        "termination": "complete_basis_grid" if proved else "first_counterexample",
        "witness": _witness(report, algebra.module.basis.labels),
        "reduction_hypothesis": report.reduction_hypothesis,
        "reduction_complete": report.reduction_complete,
        "claim_scope": (
            "associativity on every vector in the free QQ module for this "
            "exact table, by trilinearity of the associator"
            if proved
            else "nonassociativity of this exact QQ table, established by "
            "an explicit declared-basis-vector counterexample"
        ),
        "algorithm": report.algorithm,
        "algorithm_version": report.algorithm_version,
    }


def _fixture_record(identifier: str) -> dict[str, object]:
    """Compute one exact bounded report through the shared generic APIs."""
    algebra = load_analysis_fixture(identifier)
    fingerprint = algebra_fingerprint(algebra, options=FINGERPRINT_BOUNDS)
    rank = algebra.module.rank
    return {
        "fixture_id": identifier,
        "basis_order": list(algebra.module.basis.labels),
        "table_coefficient_domain": "ZZ",
        "analysis_coefficient_domain": "QQ",
        "analysis_coefficient_domain_is_finite": False,
        "substitution_domain": "declared ordered basis triples",
        "substitution_domain_is_finite": True,
        "substitution_cases": rank**3,
        "enumeration_order": "lexicographic",
        "sampling": "none",
        "associativity": _associativity_record(algebra, fingerprint),
        "fingerprint": _fingerprint_record(fingerprint),
    }


def _sampled_pass_nonproof() -> dict[str, object]:
    """Check a proper prefix and preserve its necessarily inconclusive status."""
    algebra = load_analysis_fixture("algmul.O.v1")
    basis = tuple(
        algebra.module.element({index: 1}) for index in range(algebra.module.rank)
    )
    checked = 0
    passing = 0
    sample_size = 8
    for left in basis:
        for middle in basis:
            for right in basis:
                if checked == sample_size:
                    break
                checked += 1
                if associator(algebra, left, middle, right) == algebra.module.zero():
                    passing += 1
            if checked == sample_size:
                break
        if checked == sample_size:
            break
    assert checked == passing == sample_size
    return {
        "fixture_id": "algmul.O.v1",
        "law": "associativity",
        "status": "Inconclusive",
        "population": "512 ordered basis triples",
        "method": "deterministic_lexicographic_prefix",
        "sample_size": sample_size,
        "cases_evaluated": checked,
        "passing_cases": passing,
        "seed": None,
        "distribution": "lexicographic_prefix_not_random",
        "shrinker": "none",
        "reason": "a passing proper subset is not an exhaustive proof",
    }


def _basis_change_invariance() -> dict[str, object]:
    """Check fingerprint data after one separately certified exact shear."""
    source = load_analysis_fixture("algmul.H.v1")
    domain = QQ()
    target = FreeModule(
        domain,
        Basis(
            tuple(f"f{index}" for index in range(source.module.rank)),
            coefficient_domain=domain,
        ),
    )
    last = source.module.rank - 1
    forward = tuple(
        target.element({index: 1})
        if index != last
        else target.element({0: 1, index: 1})
        for index in range(source.module.rank)
    )
    inverse = tuple(
        source.module.element({index: 1})
        if index != last
        else source.module.element({0: -1, index: 1})
        for index in range(source.module.rank)
    )
    change = ExactBasisIsomorphism.from_images(source.module, target, forward, inverse)
    transported = FiniteMultilinearStructure.from_structure_constants(
        target,
        cast(
            StructureConstants,
            change_basis(source.operation.constants, change),
        ),
        name="algmul.H.v1.exact_shear",
    )
    source_fingerprint = algebra_fingerprint(source, options=FINGERPRINT_BOUNDS)
    target_fingerprint = algebra_fingerprint(transported, options=FINGERPRINT_BOUNDS)
    certificate = change.certificate
    return {
        "fixture_id": "algmul.H.v1",
        "coefficient_domain": "QQ",
        "change": "last basis vector maps to first plus last",
        "certificate_algorithm": certificate.algorithm,
        "expected_composites": (
            certificate.expected_forward_after_inverse_count
            + certificate.expected_inverse_after_forward_count
        ),
        "evaluated_composites": (
            certificate.evaluated_forward_after_inverse_count
            + certificate.evaluated_inverse_after_forward_count
        ),
        "exact_certified": certificate.exact_certified,
        "source_associativity_status": (source_fingerprint.associativity_report.status),
        "target_associativity_status": (target_fingerprint.associativity_report.status),
        "fingerprint_equal": source_fingerprint == target_fingerprint,
        "fingerprint_role": "isomorphism_rejection_data_not_a_complete_invariant",
    }


def _invalid_fixture_boundary() -> dict[str, str]:
    """Exercise the exact-ID boundary without weakening the loader contract."""
    try:
        load_analysis_fixture("H")
    except CompositionFixtureError as error:
        return {
            "input": "H",
            "error_type": type(error).__name__,
            "message": str(error),
        }
    raise AssertionError("an unregistered fixture identifier was accepted")


def run_example() -> dict[str, object]:
    """Return one deterministic JSON-safe release report."""
    return {
        "fixtures": [_fixture_record(identifier) for identifier in FIXTURE_IDS],
        "sampled_pass_nonproof": _sampled_pass_nonproof(),
        "basis_change_invariance": _basis_change_invariance(),
        "invalid_fixture_boundary": _invalid_fixture_boundary(),
        "scope": SCOPE,
    }


def main() -> None:
    """Print stable canonical-style JSON for executable-example tests."""
    print(json.dumps(run_example(), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
