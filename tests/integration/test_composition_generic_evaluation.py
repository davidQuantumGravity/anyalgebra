"""Differential arbitrary-element evaluation for all composition fixtures."""

from __future__ import annotations

from itertools import combinations

import pytest

from anyalgebra.algebra.multilinear import (
    MultilinearEvaluationError,
    evaluate_multilinear,
)
from anyalgebra.fixtures.composition import (
    _FIXTURES,
    octonion_fixture,
    quaternion_fixture,
    split_octonion_fixture,
    split_quaternion_fixture,
)


_FACTORIES = (
    quaternion_fixture,
    split_quaternion_fixture,
    octonion_fixture,
    split_octonion_fixture,
)


def _vectors(rank: int) -> tuple[tuple[int, ...], ...]:
    values = [
        (0,) * rank,
        (1,) * rank,
        tuple((-1) ** index * (index + 1) for index in range(rank)),
    ]
    values.extend(
        tuple(1 if coordinate == index else 0 for coordinate in range(rank))
        for index in range(rank)
    )
    values.extend(
        tuple(
            1 if coordinate == left else -1 if coordinate == right else 0
            for coordinate in range(rank)
        )
        for left, right in combinations(range(rank), 2)
    )
    return tuple(values)


def _oracle(
    table: tuple[tuple[tuple[int, int], ...], ...],
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> tuple[int, ...]:
    output = [0 for _ in left]
    for left_index, left_coefficient in enumerate(left):
        for right_index, right_coefficient in enumerate(right):
            sign, output_index = table[left_index][right_index]
            output[output_index] += left_coefficient * right_coefficient * sign
    return tuple(output)


def _coordinates(element: object, rank: int) -> tuple[int, ...]:
    coordinates = element.coordinates()  # type: ignore[attr-defined]
    return tuple(
        0 if index not in coordinates else coordinates[index].value
        for index in range(rank)
    )


@pytest.mark.parametrize("factory", _FACTORIES)
def test_generated_sparse_products_agree_with_independent_integer_oracle(
    factory: object,
) -> None:
    algebra = factory()  # type: ignore[operator]
    metadata = next(item for item in _FIXTURES if item.identifier == algebra.name)
    vectors = _vectors(algebra.module.rank)

    for left in vectors:
        left_element = algebra.module.element(
            {index: value for index, value in enumerate(left) if value}
        )
        for right in vectors:
            right_element = algebra.module.element(
                {index: value for index, value in enumerate(right) if value}
            )
            actual = evaluate_multilinear(algebra, left_element, right_element)
            assert _coordinates(actual, algebra.module.rank) == _oracle(
                metadata.product_table, left, right
            )


def test_zero_and_dense_boundaries_preserve_literal_output_parent() -> None:
    algebra = octonion_fixture()
    zero = algebra.module.zero()
    dense = algebra.module.element({index: index + 1 for index in range(8)})

    assert evaluate_multilinear(algebra, zero, dense) == zero
    assert evaluate_multilinear(algebra, dense, zero) == zero
    assert evaluate_multilinear(algebra, dense, dense).parent is algebra.module


def test_same_labels_and_same_fixture_name_do_not_merge_distinct_parents() -> None:
    first = quaternion_fixture()
    second = quaternion_fixture()
    with pytest.raises(MultilinearEvaluationError) as caught:
        evaluate_multilinear(
            first,
            first.module.element({0: 1}),
            second.module.element({0: 1}),
        )

    assert caught.value.code == "parent"
    assert caught.value.expected_parent is first.module
    assert caught.value.actual_parent is second.module


def test_mixed_named_fixture_parents_cannot_be_implicitly_identified() -> None:
    ordinary = quaternion_fixture()
    split = split_quaternion_fixture()
    with pytest.raises(MultilinearEvaluationError) as caught:
        evaluate_multilinear(
            ordinary,
            ordinary.module.element({1: 1}),
            split.module.element({1: 1}),
        )

    assert caught.value.code == "parent"
    assert caught.value.actual_parent is split.module


@pytest.mark.parametrize("factory", _FACTORIES)
def test_differential_result_sequence_is_deterministic(factory: object) -> None:
    algebra = factory()  # type: ignore[operator]
    metadata = next(item for item in _FIXTURES if item.identifier == algebra.name)
    vectors = _vectors(algebra.module.rank)

    first = tuple(
        _oracle(metadata.product_table, left, right)
        for left in vectors
        for right in vectors
    )
    second = tuple(
        _oracle(metadata.product_table, left, right)
        for left in vectors
        for right in vectors
    )
    assert first == second
