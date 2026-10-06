"""Exact convention-scoped conjugation, scalar part, trace, and norm."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import pytest

from anyalgebra.algebra.multilinear import BasisProductTable, FiniteMultilinearStructure
from anyalgebra.fixtures.composition import (
    octonion_fixture,
    quaternion_fixture,
    split_octonion_fixture,
    split_quaternion_fixture,
)
from anyalgebra.fixtures.composition_ops import (
    CompositionOperationError,
    composition_operations,
    composition_operations_canonical_bytes,
)
from anyalgebra.core.domains import ZZ
from anyalgebra.core.elements import SparseElement
from anyalgebra.core.modules import Basis, FreeModule


_FACTORIES = (
    quaternion_fixture,
    split_quaternion_fixture,
    octonion_fixture,
    split_octonion_fixture,
)


@pytest.mark.parametrize("factory", _FACTORIES)
def test_conjugation_is_involutive_on_generic_exact_elements(factory: object) -> None:
    algebra = factory()  # type: ignore[operator]
    operations = composition_operations(algebra)
    element = algebra.module.element(
        {index: (-1) ** index * (index + 1) for index in range(algebra.module.rank)}
    )

    assert operations.conjugate(operations.conjugate(element)) == element
    assert operations.structure is algebra
    assert operations.conjugation_signs == (1,) + (-1,) * (algebra.module.rank - 1)


@pytest.mark.parametrize("factory", _FACTORIES)
def test_conjugation_reverses_every_basis_product(factory: object) -> None:
    algebra = factory()  # type: ignore[operator]
    operations = composition_operations(algebra)
    basis = tuple(
        algebra.module.element({index: 1}) for index in range(algebra.module.rank)
    )

    for left in basis:
        for right in basis:
            assert operations.conjugate(
                operations.multiply(left, right)
            ) == operations.multiply(
                operations.conjugate(right), operations.conjugate(left)
            )


@pytest.mark.parametrize(
    ("factory", "squares"),
    (
        (quaternion_fixture, (-1, -1, -1)),
        (split_quaternion_fixture, (-1, 1, 1)),
        (octonion_fixture, (-1, -1, -1, -1, -1, -1, -1)),
        (split_octonion_fixture, (-1, -1, -1, 1, 1, 1, 1)),
    ),
)
def test_real_part_trace_and_quadratic_norm_follow_declared_squares(
    factory: object, squares: tuple[int, ...]
) -> None:
    algebra = factory()  # type: ignore[operator]
    operations = composition_operations(algebra)
    coefficients = tuple(range(1, algebra.module.rank + 1))
    element = algebra.module.element(dict(enumerate(coefficients)))
    expected_norm = coefficients[0] ** 2 - sum(
        square * coefficient**2
        for square, coefficient in zip(squares, coefficients[1:], strict=True)
    )

    assert operations.real_part(element).value == 1
    assert operations.trace(element).value == 2
    assert operations.quadratic_norm(element).value == expected_norm
    assert operations.quadratic_norm(element).parent is ZZ()


def test_wrong_parent_and_malformed_inputs_fail_closed() -> None:
    first = quaternion_fixture()
    second = quaternion_fixture()
    operations = composition_operations(first)
    with pytest.raises(CompositionOperationError) as caught:
        operations.conjugate(second.module.element({0: 1}))
    assert caught.value.field == "element"
    with pytest.raises(CompositionOperationError) as caught:
        operations.multiply(first.module.element({0: 1}), object())  # type: ignore[arg-type]
    assert caught.value.field == "right"


def test_unknown_or_spoofed_fixture_is_rejected() -> None:
    domain = ZZ()
    module = FreeModule(domain, Basis(("1", "i", "j", "k"), coefficient_domain=domain))
    zero = FiniteMultilinearStructure.from_tables(
        module,
        BasisProductTable.from_cells(module, 2, ({} for _ in range(16))),
        name="algmul.H.v1",
    )
    with pytest.raises(CompositionOperationError) as caught:
        composition_operations(zero)
    assert caught.value.field == "structure"
    with pytest.raises(CompositionOperationError):
        composition_operations(object())  # type: ignore[arg-type]


def test_operation_record_is_sealed_canonical_and_drift_checked() -> None:
    algebra = octonion_fixture()
    first_result = composition_operations(algebra)
    second_result = composition_operations(algebra)
    first = composition_operations_canonical_bytes(first_result)

    assert first_result == second_result
    assert first == composition_operations_canonical_bytes(second_result)
    assert (
        json.loads(first)["contentHash"]["digest"] == first_result.semantic_hash.digest
    )
    with pytest.raises(CompositionOperationError, match="factory-owned"):
        type(first_result)()
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type("Derived", (type(first_result),), {})
    with pytest.raises(FrozenInstanceError):
        first_result.identifier = "other"  # type: ignore[misc]
    object.__setattr__(first_result, "conjugation_signs", ())
    with pytest.raises(CompositionOperationError, match="content drift"):
        composition_operations_canonical_bytes(first_result)


def test_conjugation_output_is_literal_parent_sparse_element() -> None:
    algebra = split_octonion_fixture()
    operations = composition_operations(algebra)
    result = operations.conjugate(algebra.module.element({0: 2, 4: -3}))

    assert type(result) is SparseElement
    assert result.parent is algebra.module
    assert result == algebra.module.element({0: 2, 4: 3})
