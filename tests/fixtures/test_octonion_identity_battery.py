"""Exact octonion and split-octonion identity battery with proof scopes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations, product
import json
from typing import cast

import pytest

from anyalgebra.algebra.multilinear import (
    FiniteMultilinearStructure,
    StructureConstants,
)
from anyalgebra.analysis.elementary import AnalysisBounds, center, nucleus
from anyalgebra.analysis.fingerprint import derivation_algebra
from anyalgebra.core.domains import QQ
from anyalgebra.core.elements import SparseElement
from anyalgebra.core.modules import Basis, FreeModule
from anyalgebra.fixtures.composition import octonion_fixture, split_octonion_fixture
from anyalgebra.fixtures.composition_ops import (
    CompositionOperations,
    composition_operations,
)


_FACTORIES = (octonion_fixture, split_octonion_fixture)


@dataclass(frozen=True, slots=True)
class ReductionEvidence:
    identity: str
    status: str
    coefficient_domain: str
    grid_count_per_fixture: int
    justification: str


_REDUCTIONS = (
    ReductionEvidence(
        "conjugation_product_reversal",
        "proved_by_complete_basis_reduction",
        "ZZ and hence QQ scalar extension",
        8**2,
        "both sides are bilinear, so all ordered basis pairs suffice",
    ),
    ReductionEvidence(
        "left_right_alternativity_and_flexibility",
        "proved_by_complete_polarization",
        "ZZ and hence QQ scalar extension, characteristic zero",
        3 * (8 + 28) * 8,
        "each identity is quadratic in its repeated variable and linear in the other",
    ),
    ReductionEvidence(
        "three_moufang_forms",
        "proved_by_complete_polarization",
        "ZZ and hence QQ scalar extension, characteristic zero",
        3 * (8 + 28) * 8**2,
        "each form is quadratic in x and linear in y,z; basis vectors and "
        "pair sums determine all coefficients",
    ),
    ReductionEvidence(
        "quadratic_norm_scalar_form",
        "proved_by_complete_polarization",
        "ZZ and hence QQ scalar extension, characteristic zero",
        8 + 28,
        "diagonal and symmetrized cross coefficients determine x*conjugate(x)",
    ),
    ReductionEvidence(
        "norm_composition",
        "proved_by_complete_biquadratic_polarization",
        "ZZ and hence QQ scalar extension, characteristic zero",
        (8 + 28) ** 2,
        "the identity is separately quadratic in x and y; both polarization "
        "grids determine every coefficient",
    ),
)


def _basis(algebra: FiniteMultilinearStructure) -> tuple[SparseElement, ...]:
    return tuple(algebra.module.element({index: 1}) for index in range(8))


def _polarization_vectors(
    algebra: FiniteMultilinearStructure,
) -> tuple[SparseElement, ...]:
    basis = _basis(algebra)
    return (
        *basis,
        *(basis[left].add(basis[right]) for left, right in combinations(range(8), 2)),
    )


def _mul(
    operations: CompositionOperations, left: SparseElement, right: SparseElement
) -> SparseElement:
    return operations.multiply(left, right)


def _moufang_values(
    operations: CompositionOperations,
    x: SparseElement,
    y: SparseElement,
    z: SparseElement,
) -> tuple[tuple[SparseElement, SparseElement], ...]:
    return (
        (
            _mul(operations, _mul(operations, x, y), _mul(operations, z, x)),
            _mul(operations, x, _mul(operations, _mul(operations, y, z), x)),
        ),
        (
            _mul(operations, _mul(operations, _mul(operations, x, y), x), z),
            _mul(operations, x, _mul(operations, y, _mul(operations, x, z))),
        ),
        (
            _mul(operations, z, _mul(operations, x, _mul(operations, y, x))),
            _mul(operations, _mul(operations, _mul(operations, z, x), y), x),
        ),
    )


@pytest.mark.parametrize(
    ("factory", "expected_left_sign", "expected_right_sign"),
    (
        (octonion_fixture, 1, -1),
        (split_octonion_fixture, -1, 1),
    ),
)
def test_each_convention_retains_its_own_minimal_nonassociativity_witness(
    factory: object, expected_left_sign: int, expected_right_sign: int
) -> None:
    algebra = factory()  # type: ignore[operator]
    operations = composition_operations(algebra)
    basis = _basis(algebra)
    found: tuple[tuple[int, int, int], SparseElement, SparseElement] | None = None
    for raw_indices in product(range(8), repeat=3):
        indices = cast(tuple[int, int, int], raw_indices)
        left, middle, right = (basis[index] for index in indices)
        left_value = _mul(operations, _mul(operations, left, middle), right)
        right_value = _mul(operations, left, _mul(operations, middle, right))
        if left_value != right_value:
            found = indices, left_value, right_value
            break

    assert found is not None
    indices, left_value, right_value = found
    assert indices == (1, 2, 4)
    assert algebra.name in ("algmul.O.v1", "algmul.Os.v1")
    assert left_value == algebra.module.element({7: expected_left_sign})
    assert right_value == algebra.module.element({7: expected_right_sign})


@pytest.mark.parametrize("factory", _FACTORIES)
def test_alternativity_and_flexibility_on_complete_polarization_grids(
    factory: object,
) -> None:
    algebra = factory()  # type: ignore[operator]
    operations = composition_operations(algebra)
    basis = _basis(algebra)
    for x, y in product(_polarization_vectors(algebra), basis):
        xx = _mul(operations, x, x)
        assert _mul(operations, xx, y) == _mul(operations, x, _mul(operations, x, y))
        assert _mul(operations, y, xx) == _mul(operations, _mul(operations, y, x), x)
        assert _mul(operations, _mul(operations, x, y), x) == _mul(
            operations, x, _mul(operations, y, x)
        )


@pytest.mark.parametrize("factory", _FACTORIES)
def test_all_three_moufang_forms_on_complete_polarization_grids(
    factory: object,
) -> None:
    algebra = factory()  # type: ignore[operator]
    operations = composition_operations(algebra)
    for x, y, z in product(
        _polarization_vectors(algebra), _basis(algebra), _basis(algebra)
    ):
        for left, right in _moufang_values(operations, x, y, z):
            assert left == right


@pytest.mark.parametrize("factory", _FACTORIES)
def test_conjugation_and_scalar_norm_have_complete_coefficient_reductions(
    factory: object,
) -> None:
    algebra = factory()  # type: ignore[operator]
    operations = composition_operations(algebra)
    basis = _basis(algebra)
    zero = algebra.module.zero()
    for left, right in product(basis, repeat=2):
        assert operations.conjugate(_mul(operations, left, right)) == _mul(
            operations,
            operations.conjugate(right),
            operations.conjugate(left),
        )
    for item in basis:
        scalar = _mul(operations, item, operations.conjugate(item))
        assert all(index == 0 for index in scalar.coordinates())
    for left_index, right_index in combinations(range(8), 2):
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
def test_norm_composition_on_complete_biquadratic_polarization_grid(
    factory: object,
) -> None:
    algebra = factory()  # type: ignore[operator]
    operations = composition_operations(algebra)
    vectors = _polarization_vectors(algebra)
    for left, right in product(vectors, repeat=2):
        assert operations.quadratic_norm(_mul(operations, left, right)).value == (
            operations.quadratic_norm(left).value
            * operations.quadratic_norm(right).value
        )


def _inverse(operations: CompositionOperations, value: SparseElement) -> SparseElement:
    norm = operations.quadratic_norm(value).value
    if norm == 0:
        raise ZeroDivisionError("composition inverse requires nonzero norm")
    if norm not in (-1, 1):
        raise ValueError("ZZ control inverse is restricted to unit norm")
    return operations.conjugate(value).scale(norm)


@pytest.mark.parametrize("factory", _FACTORIES)
def test_inverse_formula_is_restricted_to_anisotropic_unit_norm_controls(
    factory: object,
) -> None:
    algebra = factory()  # type: ignore[operator]
    operations = composition_operations(algebra)
    one = algebra.module.element({0: 1})
    for value in _basis(algebra):
        assert operations.quadratic_norm(value).value != 0
        inverse = _inverse(operations, value)
        assert _mul(operations, value, inverse) == one
        assert _mul(operations, inverse, value) == one


def test_split_isotropic_element_is_not_promoted_to_a_division_result() -> None:
    algebra = split_octonion_fixture()
    operations = composition_operations(algebra)
    isotropic = algebra.module.element({0: 1, 4: 1})

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


@pytest.mark.parametrize("factory", _FACTORIES)
def test_nucleus_center_and_derivation_dimensions_match_octonion_forms(
    factory: object,
) -> None:
    algebra = _qq_lift(factory())  # type: ignore[operator]
    bounds = AnalysisBounds(max_elimination_work=3_000_000)

    assert nucleus(algebra, options=bounds).dimension == 1
    assert center(algebra, options=bounds).dimension == 1
    assert derivation_algebra(algebra, options=bounds).dimension == 14


def test_reduction_evidence_is_deterministic_and_exactly_scoped() -> None:
    first = json.dumps(
        [asdict(item) for item in _REDUCTIONS], sort_keys=True, separators=(",", ":")
    )
    second = json.dumps(
        [asdict(item) for item in _REDUCTIONS], sort_keys=True, separators=(",", ":")
    )

    assert first == second
    assert tuple(item.grid_count_per_fixture for item in _REDUCTIONS) == (
        64,
        864,
        6912,
        36,
        1296,
    )
    assert all("proved" in item.status for item in _REDUCTIONS)
