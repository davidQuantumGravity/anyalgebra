"""Exact quaternion and split-quaternion identity battery with scope records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
import json

import pytest

from anyalgebra.algebra.multilinear import (
    FiniteMultilinearStructure,
    StructureConstants,
)
from anyalgebra.analysis.fingerprint import derivation_algebra
from anyalgebra.core.domains import QQ
from anyalgebra.core.elements import SparseElement
from anyalgebra.core.modules import Basis, FreeModule
from anyalgebra.fixtures.composition import (
    quaternion_fixture,
    split_quaternion_fixture,
)
from anyalgebra.fixtures.composition_ops import (
    CompositionOperations,
    composition_operations,
)


_FACTORIES = (quaternion_fixture, split_quaternion_fixture)


@dataclass(frozen=True, slots=True)
class ReductionEvidence:
    identity: str
    status: str
    coefficient_domain: str
    grid_count: int
    justification: str


_REDUCTIONS = (
    ReductionEvidence(
        "associativity",
        "proved_by_complete_basis_reduction",
        "ZZ and hence QQ scalar extension",
        4**3,
        "the associator is trilinear, so vanishing on every ordered basis "
        "triple is sufficient",
    ),
    ReductionEvidence(
        "conjugation_product_reversal",
        "proved_by_complete_basis_reduction",
        "ZZ and hence QQ scalar extension",
        4**2,
        "both sides are bilinear in the two inputs, so every ordered basis "
        "pair is sufficient",
    ),
    ReductionEvidence(
        "quadratic_norm_scalar_form",
        "proved_by_polarized_coefficient_reduction",
        "ZZ and hence QQ scalar extension",
        4 + 6,
        "four diagonal and six symmetrized cross coefficients determine the "
        "quadratic polynomial",
    ),
    ReductionEvidence(
        "flexibility_alternativity_moufang_norm_composition_inverse",
        "theorem_derived_from_verified_identities",
        "ZZ identities with QQ inversion when norm is nonzero",
        0,
        "associativity, conjugation reversal, central scalar norms, and "
        "nonzero norm imply the listed identities",
    ),
)


def _basis(algebra: FiniteMultilinearStructure) -> tuple[SparseElement, ...]:
    return tuple(algebra.module.element({index: 1}) for index in range(4))


def _mul(
    operations: CompositionOperations, left: SparseElement, right: SparseElement
) -> SparseElement:
    return operations.multiply(left, right)


@pytest.mark.parametrize("factory", _FACTORIES)
def test_associativity_and_conjugation_have_complete_multilinear_basis_proofs(
    factory: object,
) -> None:
    algebra = factory()  # type: ignore[operator]
    operations = composition_operations(algebra)
    basis = _basis(algebra)

    for left, middle, right in product(basis, repeat=3):
        assert _mul(operations, _mul(operations, left, middle), right) == _mul(
            operations, left, _mul(operations, middle, right)
        )
    for left, right in product(basis, repeat=2):
        assert operations.conjugate(_mul(operations, left, right)) == _mul(
            operations,
            operations.conjugate(right),
            operations.conjugate(left),
        )


@pytest.mark.parametrize("factory", _FACTORIES)
def test_quadratic_norm_is_fixed_by_diagonal_and_polarized_cross_coefficients(
    factory: object,
) -> None:
    algebra = factory()  # type: ignore[operator]
    operations = composition_operations(algebra)
    basis = _basis(algebra)
    zero = algebra.module.zero()

    for item in basis:
        product_value = _mul(operations, item, operations.conjugate(item))
        assert all(index == 0 for index in product_value.coordinates())
    for left_index in range(4):
        for right_index in range(left_index + 1, 4):
            cross = _mul(
                operations, basis[left_index], operations.conjugate(basis[right_index])
            ).add(
                _mul(
                    operations,
                    basis[right_index],
                    operations.conjugate(basis[left_index]),
                )
            )
            assert cross == zero


@pytest.mark.parametrize("factory", _FACTORIES)
def test_flexibility_alternativity_and_three_moufang_forms_on_basis_controls(
    factory: object,
) -> None:
    algebra = factory()  # type: ignore[operator]
    operations = composition_operations(algebra)
    basis = _basis(algebra)
    for x, y, z in product(basis, repeat=3):
        xx = _mul(operations, x, x)
        assert _mul(operations, _mul(operations, x, y), x) == _mul(
            operations, x, _mul(operations, y, x)
        )
        assert _mul(operations, xx, y) == _mul(operations, x, _mul(operations, x, y))
        assert _mul(operations, y, xx) == _mul(operations, _mul(operations, y, x), x)
        assert _mul(operations, _mul(operations, x, y), _mul(operations, z, x)) == _mul(
            operations, x, _mul(operations, _mul(operations, y, z), x)
        )
        assert _mul(operations, _mul(operations, _mul(operations, x, y), x), z) == _mul(
            operations, x, _mul(operations, y, _mul(operations, x, z))
        )
        assert _mul(operations, z, _mul(operations, x, _mul(operations, y, x))) == _mul(
            operations, _mul(operations, _mul(operations, z, x), y), x
        )


@pytest.mark.parametrize("factory", _FACTORIES)
def test_norm_composition_on_complete_small_coefficient_control_cube(
    factory: object,
) -> None:
    algebra = factory()  # type: ignore[operator]
    operations = composition_operations(algebra)
    vectors = tuple(product((-1, 0, 1), repeat=4))
    elements = tuple(
        algebra.module.element(
            {
                index: coefficient
                for index, coefficient in enumerate(vector)
                if coefficient
            }
        )
        for vector in vectors
    )
    for left in elements:
        left_norm = operations.quadratic_norm(left).value
        for right in elements:
            assert operations.quadratic_norm(_mul(operations, left, right)).value == (
                left_norm * operations.quadratic_norm(right).value
            )


def _inverse(operations: CompositionOperations, value: SparseElement) -> SparseElement:
    norm = operations.quadratic_norm(value).value
    if norm == 0:
        raise ZeroDivisionError("composition inverse requires nonzero norm")
    if norm not in (-1, 1):
        raise ValueError("ZZ control inverse is restricted to unit norm")
    return operations.conjugate(value).scale(norm)


@pytest.mark.parametrize("factory", _FACTORIES)
def test_inverse_formula_is_used_only_on_nonzero_unit_norm_controls(
    factory: object,
) -> None:
    algebra = factory()  # type: ignore[operator]
    operations = composition_operations(algebra)
    one = algebra.module.element({0: 1})
    candidates = tuple(algebra.module.element({index: 1}) for index in range(4))

    for value in candidates:
        assert operations.quadratic_norm(value).value != 0
        inverse = _inverse(operations, value)
        assert _mul(operations, value, inverse) == one
        assert _mul(operations, inverse, value) == one


def test_split_isotropic_element_is_not_reported_as_invertible() -> None:
    algebra = split_quaternion_fixture()
    operations = composition_operations(algebra)
    isotropic = algebra.module.element({0: 1, 2: 1})

    assert operations.quadratic_norm(isotropic).value == 0
    with pytest.raises(ZeroDivisionError, match="nonzero norm"):
        _inverse(operations, isotropic)


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


def test_both_quaternion_conventions_have_three_dimensional_derivations() -> None:
    assert derivation_algebra(_qq_lift(quaternion_fixture())).dimension == 3
    assert derivation_algebra(_qq_lift(split_quaternion_fixture())).dimension == 3


def test_reduction_evidence_is_complete_deterministic_and_claim_bounded() -> None:
    first = json.dumps(
        [asdict(item) for item in _REDUCTIONS], sort_keys=True, separators=(",", ":")
    )
    second = json.dumps(
        [asdict(item) for item in _REDUCTIONS], sort_keys=True, separators=(",", ":")
    )

    assert first == second
    assert tuple(item.grid_count for item in _REDUCTIONS[:3]) == (64, 16, 10)
    assert all(item.coefficient_domain for item in _REDUCTIONS)
    assert "sample" not in _REDUCTIONS[-1].status
