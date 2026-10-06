"""Exact convention tests for generic composition-table fixtures."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from anyalgebra.algebra.multilinear import FiniteMultilinearStructure
from anyalgebra.fixtures.composition import (
    OCTONION_POSITIVE_TRIPLES,
    CompositionFixtureError,
    _FIXTURES,
    composition_fixture,
    octonion_fixture,
    quaternion_fixture,
    split_octonion_fixture,
    split_quaternion_fixture,
)
from anyalgebra.core.domains import ZZ


# Independent oracle transcribed directly from conventions-v0.0.md §4.5.
# This deliberately does not import an implementation product-table constant.
_EXPECTED_SPLIT_OCTONION_TABLE: tuple[tuple[tuple[int, int], ...], ...] = (
    ((1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7)),
    ((1, 1), (-1, 0), (1, 3), (-1, 2), (-1, 5), (1, 4), (-1, 7), (1, 6)),
    ((1, 2), (-1, 3), (-1, 0), (1, 1), (-1, 6), (1, 7), (1, 4), (-1, 5)),
    ((1, 3), (1, 2), (-1, 1), (-1, 0), (-1, 7), (-1, 6), (1, 5), (1, 4)),
    ((1, 4), (1, 5), (1, 6), (1, 7), (1, 0), (1, 1), (1, 2), (1, 3)),
    ((1, 5), (-1, 4), (-1, 7), (1, 6), (-1, 1), (1, 0), (1, 3), (-1, 2)),
    ((1, 6), (1, 7), (-1, 4), (-1, 5), (-1, 2), (-1, 3), (1, 0), (1, 1)),
    ((1, 7), (-1, 6), (1, 5), (-1, 4), (-1, 3), (1, 2), (-1, 1), (1, 0)),
)


@pytest.mark.parametrize(
    ("factory", "name", "labels", "squares"),
    (
        (quaternion_fixture, "algmul.H.v1", ("1", "i", "j", "k"), (1, -1, -1, -1)),
        (
            split_quaternion_fixture,
            "algmul.Hs.v1",
            ("1", "i", "j", "k"),
            (1, -1, 1, 1),
        ),
        (
            octonion_fixture,
            "algmul.O.v1",
            ("1", "e1", "e2", "e3", "e4", "e5", "e6", "e7"),
            (1, -1, -1, -1, -1, -1, -1, -1),
        ),
        (
            split_octonion_fixture,
            "algmul.Os.v1",
            ("1", "e1", "e2", "e3", "e4", "e5", "e6", "e7"),
            (1, -1, -1, -1, 1, 1, 1, 1),
        ),
    ),
)
def test_named_fixtures_have_literal_basis_unity_and_declared_squares(
    factory: object,
    name: str,
    labels: tuple[str, ...],
    squares: tuple[int, ...],
) -> None:
    structure = factory()  # type: ignore[operator]

    assert type(structure) is FiniteMultilinearStructure
    assert structure.name == name
    assert structure.module.basis.labels == labels
    assert structure.module.domain is ZZ()
    for index, square in enumerate(squares):
        assert _signed_basis_product(structure, index, index) == (square, 0)
    for index in range(structure.module.rank):
        assert _signed_basis_product(structure, 0, index) == (1, index)
        assert _signed_basis_product(structure, index, 0) == (1, index)


def test_quaternion_and_split_quaternion_orientations_are_explicit() -> None:
    quaternion = quaternion_fixture()
    split = split_quaternion_fixture()

    assert _signed_basis_product(quaternion, 1, 2) == (1, 3)
    assert _signed_basis_product(quaternion, 2, 3) == (1, 1)
    assert _signed_basis_product(quaternion, 3, 1) == (1, 2)
    assert _signed_basis_product(quaternion, 2, 1) == (-1, 3)
    assert _signed_basis_product(split, 1, 2) == (1, 3)
    assert _signed_basis_product(split, 2, 3) == (-1, 1)
    assert _signed_basis_product(split, 3, 1) == (1, 2)
    assert _signed_basis_product(split, 3, 2) == (1, 1)


def test_octonion_fano_rule_and_split_octonion_literal_table_are_auditable() -> None:
    octonion = octonion_fixture()
    split = split_octonion_fixture()

    assert OCTONION_POSITIVE_TRIPLES == (
        (1, 2, 3),
        (1, 4, 5),
        (1, 7, 6),
        (2, 4, 6),
        (2, 5, 7),
        (3, 4, 7),
        (3, 6, 5),
    )
    for left, middle, right in OCTONION_POSITIVE_TRIPLES:
        assert _signed_basis_product(octonion, left, middle) == (1, right)
        assert _signed_basis_product(octonion, middle, right) == (1, left)
        assert _signed_basis_product(octonion, right, left) == (1, middle)
    for left, row in enumerate(_EXPECTED_SPLIT_OCTONION_TABLE):
        for right, expected in enumerate(row):
            assert _signed_basis_product(split, left, right) == expected


@pytest.mark.parametrize(
    "factory",
    (
        quaternion_fixture,
        split_quaternion_fixture,
        octonion_fixture,
        split_octonion_fixture,
    ),
)
def test_all_fixture_imaginary_generators_anticommute(factory: object) -> None:
    structure = factory()  # type: ignore[operator]
    for left in range(1, structure.module.rank):
        for right in range(left + 1, structure.module.rank):
            sign, output = _signed_basis_product(structure, left, right)
            reverse_sign, reverse_output = _signed_basis_product(structure, right, left)
            assert output == reverse_output
            assert sign == -reverse_sign


def test_quaternion_fixture_is_associative_and_octonion_has_an_explicit_failure() -> (
    None
):
    quaternion = quaternion_fixture()
    for left in range(4):
        for middle in range(4):
            for right in range(4):
                assert _multiply_signed(
                    quaternion, _signed_basis_product(quaternion, left, middle), right
                ) == _multiply_signed(
                    quaternion,
                    (1, left),
                    _signed_basis_product(quaternion, middle, right),
                )

    octonion = octonion_fixture()
    assert _multiply_signed(octonion, _signed_basis_product(octonion, 1, 2), 4) == (
        1,
        7,
    )
    assert _multiply_signed(
        octonion, (1, 1), _signed_basis_product(octonion, 2, 4)
    ) == (-1, 7)


def test_generic_selector_and_invalid_fixture_identifier_boundary() -> None:
    assert composition_fixture("algmul.H.v1").name == "algmul.H.v1"
    assert composition_fixture("algmul.Hs.v1").name == "algmul.Hs.v1"
    assert composition_fixture("algmul.O.v1").name == "algmul.O.v1"
    assert composition_fixture("algmul.Os.v1").name == "algmul.Os.v1"
    with pytest.raises(CompositionFixtureError, match="identifier"):
        composition_fixture("H")
    with pytest.raises(CompositionFixtureError, match="built-in str"):
        composition_fixture(1)  # type: ignore[arg-type]


def test_fixture_metadata_is_frozen_anchored_and_exact_id_selected() -> None:
    metadata_by_id = {metadata.identifier: metadata for metadata in _FIXTURES}

    assert type(_FIXTURES) is tuple
    assert tuple(metadata_by_id) == (
        "algmul.H.v1",
        "algmul.Hs.v1",
        "algmul.O.v1",
        "algmul.Os.v1",
    )
    assert metadata_by_id["algmul.H.v1"].source_anchors == (
        "docs/conventions/conventions-v0.0.md#4.2-quaternions",
    )
    assert metadata_by_id["algmul.Hs.v1"].source_anchors == (
        "docs/conventions/conventions-v0.0.md#4.3-split-quaternions",
    )
    assert metadata_by_id["algmul.O.v1"].source_anchors == (
        "docs/conventions/conventions-v0.0.md#4.4-octonions",
    )
    assert metadata_by_id["algmul.Os.v1"].source_anchors == (
        "docs/conventions/conventions-v0.0.md#4.5-split-octonions",
        "docs/legacy/generated/algmul-evaluated-surface.json",
    )
    h_metadata = metadata_by_id["algmul.H.v1"]
    assert not hasattr(h_metadata, "__dict__")
    with pytest.raises(FrozenInstanceError):
        h_metadata.identifier = "other"  # type: ignore[misc]
    repository_root = Path(__file__).resolve().parents[2]
    for metadata in _FIXTURES:
        for anchor in metadata.source_anchors:
            path, _separator, _fragment = anchor.partition("#")
            assert (repository_root / path).is_file()


def test_fixture_calls_construct_independent_generic_structure_parents() -> None:
    first = composition_fixture("algmul.H.v1")
    second = composition_fixture("algmul.H.v1")

    assert first is not second
    assert first.module is not second.module
    assert first.name == second.name == "algmul.H.v1"
    for left in range(first.module.rank):
        for right in range(first.module.rank):
            assert _signed_basis_product(first, left, right) == _signed_basis_product(
                second, left, right
            )


def _signed_basis_product(
    structure: FiniteMultilinearStructure, left: int, right: int
) -> tuple[int, int]:
    coordinates = tuple(structure.evaluate_basis(left, right).coordinates().items())
    assert len(coordinates) == 1
    output, coefficient = coordinates[0]
    assert coefficient.parent is ZZ()
    assert type(coefficient.value) is int
    return coefficient.value, output


def _multiply_signed(
    structure: FiniteMultilinearStructure,
    left: tuple[int, int],
    right: int | tuple[int, int],
) -> tuple[int, int]:
    if isinstance(right, int):
        right_sign, right_index = 1, right
    else:
        right_sign, right_index = right
    left_sign, left_index = left
    product_sign, product_index = _signed_basis_product(
        structure, left_index, right_index
    )
    return left_sign * right_sign * product_sign, product_index
