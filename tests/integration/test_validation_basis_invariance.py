"""Integration evidence for exact basis invariance and a shipped witness.

The basis grid below is a theorem reduction, not a sample: the generic finite
structure extends its exact structure constants bilinearly, so its associator
is trilinear.  Thus the 8**3 ordered basis triples determine associativity on
the full free ``QQ`` module for this fixture.
"""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Final, cast

import pytest

from anyalgebra.algebra.multilinear import (
    FiniteMultilinearStructure,
    StructureConstants,
)
from anyalgebra.analysis.elementary import AnalysisBounds, associator
from anyalgebra.analysis.fingerprint import algebra_fingerprint
from anyalgebra.core.domains import QQ
from anyalgebra.core.elements import SparseElement
from anyalgebra.core.modules import Basis, FreeModule
from anyalgebra.core.rational import Rational
from anyalgebra.fixtures.composition import octonion_fixture, quaternion_fixture
from anyalgebra.presentations.basis_change import ExactBasisIsomorphism, change_basis


ASSOCIATOR_BASIS_REDUCTION_HYPOTHESIS: Final[tuple[tuple[str, str], ...]] = (
    ("theorem", "exact bilinearity implies a trilinear associator"),
    ("coefficient_domain", "QQ"),
    ("fixture_convention", "algmul.O.v1"),
    ("basis_grid", "all 8^3 ordered basis triples in lexicographic order"),
    ("sampling", "none"),
)
_FINGERPRINT_BOUNDS: Final = AnalysisBounds(
    max_constraints=25_000,
    max_elimination_work=3_000_000,
)


def _qq_lift(value: FiniteMultilinearStructure) -> FiniteMultilinearStructure:
    """Rebuild one named integral table over the exact ``QQ`` parent."""
    domain = QQ()
    module = FreeModule(
        domain, Basis(value.module.basis.labels, coefficient_domain=domain)
    )
    constants = StructureConstants.from_sparse(
        (module.basis, module.basis),
        module.basis,
        tuple(
            (key, domain.element(coefficient.value))
            for key, coefficient in value.operation.constants.entries
        ),
    )
    return FiniteMultilinearStructure.from_structure_constants(module, constants)


def _shear_change(source: FreeModule) -> ExactBasisIsomorphism:
    """Return a certified non-permutation exact shear in the final coordinate."""
    target = FreeModule(
        QQ(),
        Basis(
            tuple(f"f{index}" for index in range(source.rank)),
            coefficient_domain=QQ(),
        ),
    )
    forward = tuple(
        target.element({index: 1})
        if index != source.rank - 1
        else target.element({0: 1, index: 1})
        for index in range(source.rank)
    )
    inverse = tuple(
        source.element({index: 1})
        if index != source.rank - 1
        else source.element({0: -1, index: 1})
        for index in range(source.rank)
    )
    return ExactBasisIsomorphism.from_images(source, target, forward, inverse)


def _transport(
    source: FiniteMultilinearStructure, change: ExactBasisIsomorphism
) -> FiniteMultilinearStructure:
    """Transport only through the certified generic structure-constant API."""
    constants = change_basis(source.operation.constants, change)
    return FiniteMultilinearStructure.from_structure_constants(
        change.target, cast(StructureConstants, constants)
    )


def _basis(module: FreeModule) -> tuple[SparseElement, ...]:
    """Return all basis vectors in the declared deterministic basis order."""
    return tuple(module.element({index: 1}) for index in range(module.rank))


@pytest.mark.exhaustive
def test_exact_shear_preserves_fingerprint_and_all_octonion_basis_associators() -> None:
    """Exhaustively transport all 512 basis associators; this is not sampling."""
    source = _qq_lift(octonion_fixture())
    change = _shear_change(source.module)
    target = _transport(source, change)

    source_fingerprint = algebra_fingerprint(source, options=_FINGERPRINT_BOUNDS)
    target_fingerprint = algebra_fingerprint(target, options=_FINGERPRINT_BOUNDS)

    assert change.certificate.exact_certified is True
    assert source_fingerprint == target_fingerprint
    assert source_fingerprint.isomorphism_rejection_only is True
    assert (
        source_fingerprint.bounds.max_constraints,
        source_fingerprint.bounds.max_elimination_work,
    ) == (25_000, 3_000_000)
    assert source_fingerprint.associative is False
    assert target_fingerprint.associative is False

    source_report = source_fingerprint.associativity_report
    target_report = target_fingerprint.associativity_report
    assert source_report.reduction_hypothesis == (
        "operation is the exact bilinear FiniteMultilinearStructure table over QQ"
    )
    assert source_report.reduction_complete is True
    assert (
        source_report.expected_assignments,
        source_report.checked_assignments,
        source_report.witness_indices,
    ) == (512, 85, (1, 2, 4))
    assert target_report.witness_indices == (1, 2, 4)
    assert source_report.witness_value == tuple(
        Rational(2 if index == 7 else 0) for index in range(8)
    )
    assert target_report.witness_value == tuple(
        Rational(2 if index in (0, 7) else 0) for index in range(8)
    )

    source_basis = _basis(source.module)
    target_basis = tuple(change.forward(element) for element in source_basis)
    for left_index, left in enumerate(source_basis):
        for middle_index, middle in enumerate(source_basis):
            for right_index, right in enumerate(source_basis):
                source_value = associator(
                    source,
                    left,
                    middle,
                    right,
                )
                target_value = associator(
                    target,
                    target_basis[left_index],
                    target_basis[middle_index],
                    target_basis[right_index],
                )
                assert change.forward(source_value) == target_value


@pytest.mark.exhaustive
def test_exact_shear_preserves_an_associative_fixture_at_its_complete_boundary() -> (
    None
):
    """A complete 4**3 basis grid remains proved after the same exact transport."""
    source = _qq_lift(quaternion_fixture())
    change = _shear_change(source.module)
    target = _transport(source, change)

    source_report = algebra_fingerprint(
        source, options=_FINGERPRINT_BOUNDS
    ).associativity_report
    target_fingerprint = algebra_fingerprint(target, options=_FINGERPRINT_BOUNDS)
    target_report = target_fingerprint.associativity_report

    assert source_report.status == target_report.status == "Proved"
    assert (
        source_report.expected_assignments,
        source_report.checked_assignments,
        source_report.witness_indices,
    ) == (64, 64, None)
    assert (
        target_report.expected_assignments,
        target_report.checked_assignments,
        target_report.witness_indices,
    ) == (64, 64, None)
    assert target_fingerprint != algebra_fingerprint(
        _qq_lift(octonion_fixture()), options=_FINGERPRINT_BOUNDS
    )


def test_octonion_counterexample_example_is_executable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The required documentation example must remain an executable artifact."""
    example = (
        Path(__file__).resolve().parents[2] / "examples" / "octonion_counterexample.py"
    )
    namespace = runpy.run_path(str(example), run_name="__main__")
    assert namespace["WITNESS_INDICES"] == (1, 2, 4)
    assert namespace["CONVENTION_ID"] == "algmul.O.v1"
    assert namespace["MULTILINEARITY_REDUCTION_HYPOTHESIS"] == (
        "The exact ZZ-bilinear table makes the associator ZZ-trilinear; "
        "the complete ordered basis grid is sufficient."
    )
    output = capsys.readouterr().out
    assert "first nonzero associator after 85 checks: ('e1', 'e2', 'e4')" in output
    assert "associator: (e1*e2)*e4 - e1*(e2*e4) = 2*e7" in output
    assert "it does not promote a scientific claim" in output
