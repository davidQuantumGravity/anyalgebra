"""Contracts for the sealed finite-carrier CensusSpec core."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from anyalgebra.census.spec import CensusSpecCore, CensusSpecError


def test_valid_core_is_canonical_immutable_and_safe_to_represent() -> None:
    core = CensusSpecCore.create(
        carrier_size=3,
        arity=2,
        distinguished_elements=(("zero", 0), ("one", 1)),
        corpus_name="three-pointed-binary",
    )

    assert core.carrier_size == 3
    assert core.arity == 2
    assert core.distinguished_elements == (("one", 1), ("zero", 0))
    assert core.corpus_name == "three-pointed-binary"
    assert core.input_tuple_count == 9
    assert core.candidate_count == 3**9
    assert repr(core) == (
        "CensusSpecCore(carrier_size=3, arity=2, "
        "distinguished_count=2, corpus_name='three-pointed-binary', "
        "input_tuple_count=9, candidate_count_bits=15)"
    )
    assert not hasattr(core, "__dict__")
    with pytest.raises(FrozenInstanceError):
        core.arity = 1  # type: ignore[misc]
    with pytest.raises(TypeError, match="unhashable"):
        hash(core)


def test_unordered_point_declarations_have_deterministic_equality() -> None:
    left = CensusSpecCore.create(
        carrier_size=2,
        arity=2,
        distinguished_elements=(("zero", 0), ("one", 1)),
        corpus_name="control",
    )
    right = CensusSpecCore.create(
        carrier_size=2,
        arity=2,
        distinguished_elements={"one": 1, "zero": 0},
        corpus_name="control",
    )

    assert left == right
    assert repr(left) == repr(right)


@pytest.mark.parametrize("field", ("carrier_size", "arity"))
@pytest.mark.parametrize("value", (True, False, -1, 1.0, "1", None))
def test_sizes_reject_booleans_negative_and_noninteger_values(
    field: str, value: object
) -> None:
    carrier_size = cast(int, value) if field == "carrier_size" else 2
    arity = cast(int, value) if field == "arity" else 2

    with pytest.raises(CensusSpecError) as caught:
        CensusSpecCore.create(
            carrier_size=carrier_size,
            arity=arity,
            distinguished_elements=(),
            corpus_name="invalid",
        )

    assert caught.value.field == field
    assert caught.value.index is None


def test_hard_size_and_work_limits_fail_before_iterable_consumption() -> None:
    class Trap:
        def __iter__(self) -> object:
            raise AssertionError("distinguished elements were consumed")

    cases = (
        ({"carrier_size": 257, "arity": 1}, "carrier_size"),
        ({"carrier_size": 2, "arity": 65}, "arity"),
        ({"carrier_size": 3, "arity": 11}, "input_tuple_count"),
    )
    for values, field in cases:
        with pytest.raises(CensusSpecError) as caught:
            CensusSpecCore.create(
                **values,
                distinguished_elements=Trap(),  # type: ignore[arg-type]
                corpus_name="preflight",
            )
        assert caught.value.field == field


def test_empty_and_singleton_carrier_candidate_boundaries() -> None:
    empty_nullary = CensusSpecCore.create(
        carrier_size=0,
        arity=0,
        distinguished_elements=(),
        corpus_name="empty-nullary",
    )
    empty_unary = CensusSpecCore.create(
        carrier_size=0,
        arity=1,
        distinguished_elements=(),
        corpus_name="empty-unary",
    )
    singleton_nullary = CensusSpecCore.create(
        carrier_size=1,
        arity=0,
        distinguished_elements=(),
        corpus_name="singleton-nullary",
    )

    assert (empty_nullary.input_tuple_count, empty_nullary.candidate_count) == (1, 0)
    assert (empty_unary.input_tuple_count, empty_unary.candidate_count) == (0, 1)
    assert (singleton_nullary.input_tuple_count, singleton_nullary.candidate_count) == (
        1,
        1,
    )


def test_distinguished_elements_validate_shape_name_index_and_duplicates() -> None:
    invalid = (
        ((["zero", 0],), "exact pair", 0),
        ((("zero",),), "exact pair", 0),
        ((("", 0),), "non-empty", 0),
        (((" zero", 0),), "outer whitespace", 0),
        ((("x" * 257, 0),), "text limit", 0),
        ((("\ud800", 0),), "invalid Unicode", 0),
        ((("zero", True),), "built-in int", 0),
        ((("zero", -1),), "carrier range", 0),
        ((("zero", 2),), "carrier range", 0),
        ((("zero", 0), ("zero", 1)), "duplicate name", 1),
    )
    for declarations, reason, index in invalid:
        with pytest.raises(CensusSpecError, match=reason) as caught:
            CensusSpecCore.create(
                carrier_size=2,
                arity=2,
                distinguished_elements=declarations,  # type: ignore[arg-type]
                corpus_name="invalid-points",
            )
        assert caught.value.field == "distinguished_elements"
        assert caught.value.index == index


def test_two_names_may_designate_the_same_singleton_element() -> None:
    core = CensusSpecCore.create(
        carrier_size=1,
        arity=2,
        distinguished_elements=(("zero", 0), ("one", 0)),
        corpus_name="trivial-pointed-magma",
    )

    assert core.distinguished_elements == (("one", 0), ("zero", 0))


@pytest.mark.parametrize("name", ("", " ", " x", "x ", 1, True, None))
def test_corpus_name_is_a_bounded_trimmed_exact_string(name: object) -> None:
    with pytest.raises(CensusSpecError) as caught:
        CensusSpecCore.create(
            carrier_size=1,
            arity=1,
            distinguished_elements=(),
            corpus_name=name,  # type: ignore[arg-type]
        )
    assert caught.value.field == "corpus_name"

    with pytest.raises(CensusSpecError, match="text limit"):
        CensusSpecCore.create(
            carrier_size=1,
            arity=1,
            distinguished_elements=(),
            corpus_name="x" * 257,
        )


def test_distinguished_iterable_is_snapshotted_once_and_bounded() -> None:
    declarations = [("zero", 0)]
    core = CensusSpecCore.create(
        carrier_size=1,
        arity=1,
        distinguished_elements=(item for item in declarations),
        corpus_name="snapshot",
    )
    declarations.clear()

    assert core.distinguished_elements == (("zero", 0),)
    with pytest.raises(CensusSpecError, match="declaration limit"):
        CensusSpecCore.create(
            carrier_size=256,
            arity=1,
            distinguished_elements=((f"p{index}", index % 256) for index in range(257)),
            corpus_name="too-many-points",
        )


def test_direct_construction_and_subclass_bypasses_are_rejected() -> None:
    with pytest.raises(CensusSpecError, match="factory-owned"):
        CensusSpecCore()

    with pytest.raises(TypeError, match="cannot be subclassed"):

        class Attempt(CensusSpecCore):
            pass
