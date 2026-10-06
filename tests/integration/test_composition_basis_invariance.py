"""Exact transport controls for all four named composition fixtures.

These tests concern coordinate invariance of already declared algebraic data.
They do not identify two algebras from equal fingerprints and do not promote a
fixture calculation to a classification or scientific claim.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final, cast

import pytest

from anyalgebra.algebra.multilinear import (
    FiniteMultilinearStructure,
    StructureConstants,
    evaluate_multilinear,
)
from anyalgebra.analysis.elementary import AnalysisBounds
from anyalgebra.analysis.fingerprint import algebra_fingerprint
from anyalgebra.core.domains import QQ
from anyalgebra.core.elements import SparseElement
from anyalgebra.core.modules import Basis, FreeModule
from anyalgebra.core.rational import Rational
from anyalgebra.fixtures.composition import (
    octonion_fixture,
    quaternion_fixture,
    split_octonion_fixture,
    split_quaternion_fixture,
)
from anyalgebra.presentations.basis_change import (
    BasisChangeError,
    ExactBasisIsomorphism,
    change_basis,
)


_FIXTURES: Final = (
    quaternion_fixture,
    split_quaternion_fixture,
    octonion_fixture,
    split_octonion_fixture,
)
_BOUNDS: Final = AnalysisBounds(
    max_constraints=25_000,
    max_elimination_work=3_000_000,
)


def _qq_lift(value: FiniteMultilinearStructure) -> FiniteMultilinearStructure:
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


def _basis(module: FreeModule) -> tuple[SparseElement, ...]:
    return tuple(module.element({index: 1}) for index in range(module.rank))


def _target(source: FreeModule, suffix: str) -> FreeModule:
    domain = QQ()
    return FreeModule(
        domain,
        Basis(
            tuple(f"{suffix}{index}" for index in range(source.rank)),
            coefficient_domain=domain,
        ),
    )


def _shear(source: FreeModule) -> ExactBasisIsomorphism:
    target = _target(source, "s")
    last = source.rank - 1
    forward = tuple(
        target.element({index: 1}) if index != last else target.element({0: 1, last: 1})
        for index in range(source.rank)
    )
    inverse = tuple(
        source.element({index: 1})
        if index != last
        else source.element({0: -1, last: 1})
        for index in range(source.rank)
    )
    return ExactBasisIsomorphism.from_images(source, target, forward, inverse)


def _signed_permutation(source: FreeModule) -> ExactBasisIsomorphism:
    target = _target(source, "p")
    rank = source.rank
    permutation = tuple(reversed(range(rank)))
    signs = tuple(-1 if index % 2 else 1 for index in range(rank))
    forward = tuple(
        target.element({permutation[index]: signs[index]}) for index in range(rank)
    )
    inverse_by_target = [source.zero() for _ in range(rank)]
    for source_index, target_index in enumerate(permutation):
        inverse_by_target[target_index] = source.element(
            {source_index: signs[source_index]}
        )
    return ExactBasisIsomorphism.from_images(
        source, target, forward, tuple(inverse_by_target)
    )


def _transport(
    source: FiniteMultilinearStructure, change: ExactBasisIsomorphism
) -> FiniteMultilinearStructure:
    constants = cast(
        StructureConstants, change_basis(source.operation.constants, change)
    )
    return FiniteMultilinearStructure.from_structure_constants(change.target, constants)


def _conjugate(value: SparseElement) -> SparseElement:
    return value.parent.element(
        {
            index: (
                cast(Rational, coefficient.value)
                if index == 0
                else cast(Rational, coefficient.value).negate()
            )
            for index, coefficient in value.coordinates().items()
        }
    )


def _norm(algebra: FiniteMultilinearStructure, value: SparseElement) -> Rational:
    product = evaluate_multilinear(algebra, value, _conjugate(value))
    coordinates = product.coordinates()
    assert all(index == 0 for index in coordinates)
    coefficient = coordinates.get(0)
    return Rational(0) if coefficient is None else cast(Rational, coefficient.value)


def _generic_elements(module: FreeModule) -> tuple[SparseElement, ...]:
    last = module.rank - 1
    return (
        module.zero(),
        module.element({0: Rational(1, 2)}),
        module.element({0: -2, 1: 3, last: Rational(-1, 3)}),
        module.element(
            {
                index: Rational((-1) ** index * (index + 1), index + 2)
                for index in range(module.rank)
            }
        ),
    )


@pytest.mark.exhaustive
@pytest.mark.parametrize("fixture", _FIXTURES)
@pytest.mark.parametrize("make_change", (_shear, _signed_permutation))
def test_composition_operations_and_invariants_commute_with_exact_transport(
    fixture: Callable[[], FiniteMultilinearStructure],
    make_change: Callable[[FreeModule], ExactBasisIsomorphism],
) -> None:
    source = _qq_lift(fixture())
    change = make_change(source.module)
    target = _transport(source, change)

    source_fingerprint = algebra_fingerprint(source, options=_BOUNDS)
    target_fingerprint = algebra_fingerprint(target, options=_BOUNDS)
    assert source_fingerprint == target_fingerprint
    assert source_fingerprint.isomorphism_rejection_only is True
    assert target_fingerprint.isomorphism_rejection_only is True
    assert (
        source_fingerprint.commutativity_report.status,
        source_fingerprint.associativity_report.status,
        source_fingerprint.unit_status,
    ) == (
        target_fingerprint.commutativity_report.status,
        target_fingerprint.associativity_report.status,
        target_fingerprint.unit_status,
    )

    source_values = _basis(source.module) + _generic_elements(source.module)
    target_values = tuple(change.forward(value) for value in source_values)
    for source_value, target_value in zip(source_values, target_values, strict=True):
        transported_conjugate = change.forward(_conjugate(source_value))
        assert change.inverse(transported_conjugate) == _conjugate(source_value)
        assert change.inverse(target_value) == source_value
        # The target norm is the transported scalar-valued map N composed with
        # the certified inverse; it is not a privileged target coordinate.
        assert _norm(source, change.inverse(target_value)) == _norm(
            source, source_value
        )

    source_basis = _basis(source.module)
    for left in source_basis:
        for right in source_basis:
            source_product = evaluate_multilinear(source, left, right)
            target_product = evaluate_multilinear(
                target, change.forward(left), change.forward(right)
            )
            assert change.forward(source_product) == target_product

    # Generic products and conjugation reversal exercise multilinear
    # extension beyond the complete basis-pair transport grid.
    for left, right in zip(source_values, reversed(source_values), strict=True):
        source_product = evaluate_multilinear(source, left, right)
        target_product = evaluate_multilinear(
            target, change.forward(left), change.forward(right)
        )
        assert change.forward(source_product) == target_product
        reversed_conjugate = evaluate_multilinear(
            source, _conjugate(right), _conjugate(left)
        )
        assert _conjugate(source_product) == reversed_conjugate


def test_basis_transport_rejects_an_element_from_an_equal_looking_parent() -> None:
    source = _qq_lift(quaternion_fixture())
    change = _shear(source.module)
    lookalike = FreeModule(
        QQ(), Basis(source.module.basis.labels, coefficient_domain=QQ())
    )

    with pytest.raises(BasisChangeError, match="literal source parent"):
        change.forward(lookalike.element({0: 1}))


def test_zero_is_the_exact_transport_boundary_for_every_fixture() -> None:
    for fixture in _FIXTURES:
        source = _qq_lift(fixture())
        for make_change in (_shear, _signed_permutation):
            change = make_change(source.module)
            target = _transport(source, change)
            assert change.forward(source.module.zero()) == target.module.zero()
            assert change.inverse(target.module.zero()) == source.module.zero()
            assert (
                evaluate_multilinear(target, target.module.zero(), target.module.zero())
                == target.module.zero()
            )
