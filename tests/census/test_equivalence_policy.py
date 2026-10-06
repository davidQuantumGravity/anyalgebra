"""Contracts for explicit finite-carrier relabeling policies."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from itertools import permutations
from typing import cast

import pytest

from anyalgebra.census.equivalence import (
    EquivalencePolicy,
    EquivalencePolicyError,
    SortBlock,
)
from anyalgebra.census.spec import CensusSpecCore


def _core(
    size: int = 4,
    *,
    points: tuple[tuple[str, int], ...] = (),
) -> CensusSpecCore:
    return CensusSpecCore.create(
        carrier_size=size,
        arity=2,
        distinguished_elements=points,
        corpus_name=f"equivalence-control-{size}",
    )


def test_literal_policy_admits_only_identity() -> None:
    policy = EquivalencePolicy.literal(_core(3))

    assert policy.kind == "literal"
    assert policy.fixed_elements == (0, 1, 2)
    assert policy.sort_blocks == (SortBlock.create(name="carrier", elements=(0, 1, 2)),)
    assert policy.permutation_count == 1
    assert policy.allows((0, 1, 2)) is True
    assert policy.allows((1, 0, 2)) is False


def test_unrestricted_relabeling_admits_the_full_symmetric_group() -> None:
    policy = EquivalencePolicy.relabeling(_core(4))
    candidates = tuple(permutations(range(4)))

    assert policy.kind == "carrier_relabeling"
    assert policy.fixed_elements == ()
    assert policy.permutation_count == 24
    assert sum(policy.allows(candidate) for candidate in candidates) == 24


def test_distinguished_and_explicit_fixed_elements_are_always_fixed() -> None:
    policy = EquivalencePolicy.relabeling(
        _core(4, points=(("one", 1), ("zero", 0))), fixed_elements=(3, 1)
    )
    allowed = tuple(
        candidate for candidate in permutations(range(4)) if policy.allows(candidate)
    )

    assert policy.fixed_elements == (0, 1, 3)
    assert policy.permutation_count == 1
    assert allowed == ((0, 1, 2, 3),)


def test_sort_blocks_canonicalize_and_restrict_images_to_the_same_block() -> None:
    left = SortBlock.create(name="left", elements=(2, 0))
    right = SortBlock.create(name="right", elements=(3, 1))
    forward = EquivalencePolicy.relabeling(_core(), sort_blocks=(right, left))
    reverse = EquivalencePolicy.relabeling(_core(), sort_blocks=(left, right))

    assert forward == reverse
    assert tuple(block.name for block in forward.sort_blocks) == ("left", "right")
    assert forward.permutation_count == 4
    assert forward.allows((2, 3, 0, 1)) is True
    assert forward.allows((1, 0, 3, 2)) is False


@pytest.mark.parametrize(
    ("blocks", "reason"),
    (
        ((SortBlock.create(name="a", elements=(0, 1)),), "incomplete"),
        (
            (
                SortBlock.create(name="a", elements=(0, 1)),
                SortBlock.create(name="b", elements=(1, 2, 3)),
            ),
            "duplicate carrier element",
        ),
        (
            (
                SortBlock.create(name="a", elements=(0, 1)),
                SortBlock.create(name="a", elements=(2, 3)),
            ),
            "duplicate block name",
        ),
        (
            (
                SortBlock.create(name="a", elements=(0, 1)),
                SortBlock.create(name="b", elements=(2, 4)),
            ),
            "carrier range",
        ),
    ),
)
def test_incomplete_duplicate_ambiguous_and_out_of_range_blocks_fail(
    blocks: tuple[SortBlock, ...], reason: str
) -> None:
    with pytest.raises(EquivalencePolicyError, match=reason):
        EquivalencePolicy.relabeling(_core(), sort_blocks=blocks)


@pytest.mark.parametrize(
    "elements",
    ((), (0, 0), (True,), (-1,), (4,)),
)
def test_sort_block_and_fixed_element_boundaries(elements: tuple[object, ...]) -> None:
    if not elements:
        with pytest.raises(EquivalencePolicyError, match="non-empty"):
            SortBlock.create(name="empty", elements=cast(tuple[int, ...], elements))
        return
    if elements == (0, 0):
        with pytest.raises(EquivalencePolicyError, match="duplicate"):
            SortBlock.create(name="duplicate", elements=cast(tuple[int, ...], elements))
        return
    with pytest.raises(EquivalencePolicyError):
        EquivalencePolicy.relabeling(
            _core(), fixed_elements=cast(tuple[int, ...], elements)
        )


def test_empty_carrier_has_one_empty_relabeling_and_no_blocks() -> None:
    policy = EquivalencePolicy.relabeling(_core(0))

    assert policy.sort_blocks == ()
    assert policy.fixed_elements == ()
    assert policy.permutation_count == 1
    assert policy.allows(()) is True


@pytest.mark.parametrize(
    "candidate",
    ((0, 1), (0, 1, 1), (0, 1, 3), (0, True, 2), [0, 1, 2]),
)
def test_malformed_permutations_raise_instead_of_becoming_disallowed(
    candidate: object,
) -> None:
    policy = EquivalencePolicy.relabeling(_core(3))

    with pytest.raises(EquivalencePolicyError, match="permutation"):
        policy.allows(cast(tuple[int, ...], candidate))


def test_records_are_frozen_slotted_unhashable_sealed_and_safe_to_repr() -> None:
    block = SortBlock.create(name="even", elements=(2, 0))
    policy = EquivalencePolicy.relabeling(
        _core(),
        fixed_elements=(0,),
        sort_blocks=(block, SortBlock.create(name="odd", elements=(1, 3))),
    )

    assert repr(block) == "SortBlock(name='even', size=2)"
    assert repr(policy) == (
        "EquivalencePolicy(kind='carrier_relabeling', carrier_size=4, "
        "fixed_count=1, block_count=2, permutation_count=2)"
    )
    for value in (block, policy):
        assert not hasattr(value, "__dict__")
        with pytest.raises(TypeError, match="unhashable"):
            hash(value)
    with pytest.raises(FrozenInstanceError):
        policy.fixed_elements = ()  # type: ignore[misc]
    with pytest.raises(EquivalencePolicyError, match="factory-owned"):
        EquivalencePolicy()
    with pytest.raises(TypeError, match="cannot be subclassed"):

        class Attempt(EquivalencePolicy):
            pass


def test_policy_sources_snapshot_once_and_reject_nonrecords_and_over_limit() -> None:
    blocks = [SortBlock.create(name="carrier", elements=(0, 1, 2, 3))]
    policy = EquivalencePolicy.relabeling(_core(), sort_blocks=iter(blocks))
    blocks.clear()
    assert policy.sort_blocks[0].elements == (0, 1, 2, 3)

    with pytest.raises(EquivalencePolicyError, match="exact SortBlock"):
        EquivalencePolicy.relabeling(_core(), sort_blocks=(lambda value: value,))  # type: ignore[arg-type]
    with pytest.raises(EquivalencePolicyError, match="declaration limit"):
        EquivalencePolicy.relabeling(
            _core(256),
            sort_blocks=(
                SortBlock.create(name=f"b{index}", elements=(index,))
                for index in range(257)
            ),
        )
