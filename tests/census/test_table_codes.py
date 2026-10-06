"""Contracts for lexicographic finite operation-table rank and unrank."""

from __future__ import annotations

from itertools import product
from typing import cast

import pytest

from anyalgebra.census.spec import CensusSpecCore
from anyalgebra.census.table_codes import (
    OperationTableCodeError,
    rank_operation_table,
    unrank_operation_table,
)


def _core(size: int, arity: int) -> CensusSpecCore:
    return CensusSpecCore.create(
        carrier_size=size,
        arity=arity,
        corpus_name=f"table-code-{size}-{arity}",
    )


@pytest.mark.parametrize(
    ("size", "arity"),
    (
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 4),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 0),
        (3, 1),
        (3, 2),
        (4, 0),
        (4, 1),
    ),
)
def test_rank_and_unrank_are_inverse_on_every_small_control(
    size: int, arity: int
) -> None:
    core = _core(size, arity)

    for index in range(core.candidate_count):
        outputs = unrank_operation_table(core, index)
        assert len(outputs) == core.input_tuple_count
        assert rank_operation_table(core, outputs) == index


@pytest.mark.parametrize(("size", "arity"), ((2, 2), (3, 1), (4, 1)))
def test_unrank_sequence_is_exact_lexicographic_product_order(
    size: int, arity: int
) -> None:
    core = _core(size, arity)
    direct = product(range(size), repeat=core.input_tuple_count)

    assert tuple(
        unrank_operation_table(core, index) for index in range(core.candidate_count)
    ) == tuple(direct)


def test_first_cell_is_most_significant_base_digit() -> None:
    core = _core(3, 1)

    assert rank_operation_table(core, (0, 0, 0)) == 0
    assert rank_operation_table(core, (0, 0, 1)) == 1
    assert rank_operation_table(core, (0, 1, 0)) == 3
    assert rank_operation_table(core, (1, 0, 0)) == 9
    assert unrank_operation_table(core, 17) == (1, 2, 2)


def test_empty_carrier_boundaries_are_explicit() -> None:
    empty_unary = _core(0, 1)
    empty_nullary = _core(0, 0)

    assert empty_unary.candidate_count == 1
    assert rank_operation_table(empty_unary, ()) == 0
    assert unrank_operation_table(empty_unary, 0) == ()

    assert empty_nullary.candidate_count == 0
    with pytest.raises(OperationTableCodeError, match="no operation tables"):
        rank_operation_table(empty_nullary, ())
    with pytest.raises(OperationTableCodeError, match="outside"):
        unrank_operation_table(empty_nullary, 0)


def test_rank_accepts_a_one_shot_iterable_and_snapshots_it_once() -> None:
    core = _core(2, 2)
    source = [1, 0, 1, 1]

    assert rank_operation_table(core, (value for value in source)) == 11
    source.clear()


@pytest.mark.parametrize("value", (True, -1, 1.5, "1", None))
def test_unrank_rejects_nonexact_negative_candidate_indices(value: object) -> None:
    with pytest.raises(OperationTableCodeError) as caught:
        unrank_operation_table(_core(2, 2), cast(int, value))
    assert caught.value.field == "candidate_index"


def test_unrank_rejects_upper_boundary_before_allocating_a_table() -> None:
    core = _core(2, 16)

    with pytest.raises(OperationTableCodeError) as caught:
        unrank_operation_table(core, core.candidate_count)
    assert caught.value.field == "candidate_index"
    assert "outside" in caught.value.reason


def test_arbitrary_precision_rank_has_no_machine_integer_overflow() -> None:
    core = _core(2, 16)
    final_index = core.candidate_count - 1

    outputs = unrank_operation_table(core, final_index)

    assert len(outputs) == 65_536
    assert set(outputs) == {1}
    assert rank_operation_table(core, outputs) == final_index
    assert final_index.bit_length() == 65_536


@pytest.mark.parametrize(
    ("outputs", "reason"),
    (
        ((0, 1, 0), "wrong cell count"),
        ((0, 1, 0, 1, 0), "wrong cell count"),
        ((0, 1, True, 0), "exact built-in ints"),
        ((0, 1, -1, 0), "outside the carrier"),
        ((0, 1, 2, 0), "outside the carrier"),
        ("0101", "iterable"),
    ),
)
def test_rank_rejects_wrong_length_type_and_digit_range(
    outputs: object, reason: str
) -> None:
    with pytest.raises(OperationTableCodeError, match=reason):
        rank_operation_table(_core(2, 2), cast(tuple[int, ...], outputs))


def test_rank_caps_an_unbounded_iterable_at_one_cell_past_expected() -> None:
    consumed = 0

    def forever() -> object:
        nonlocal consumed
        while True:
            consumed += 1
            yield 0

    with pytest.raises(OperationTableCodeError, match="wrong cell count"):
        rank_operation_table(_core(2, 2), cast(tuple[int, ...], forever()))
    assert consumed == 5


def test_no_candidate_failure_precedes_user_iterable_consumption() -> None:
    consumed = False

    def exploding() -> object:
        nonlocal consumed
        consumed = True
        raise RuntimeError("must not be consumed")
        yield 0

    with pytest.raises(OperationTableCodeError, match="no operation tables"):
        rank_operation_table(_core(0, 0), cast(tuple[int, ...], exploding()))
    assert consumed is False


def test_exact_core_ownership_and_sanitized_diagnostics() -> None:
    with pytest.raises(OperationTableCodeError) as caught:
        rank_operation_table(cast(CensusSpecCore, None), (0,))
    assert caught.value.field == "core"
    assert repr((0,)) not in str(caught.value)
