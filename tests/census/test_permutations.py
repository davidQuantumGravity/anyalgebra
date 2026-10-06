"""Contracts for exact finite-carrier permutation groups."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from itertools import permutations, product
from typing import cast

import pytest

from anyalgebra.census.equivalence import EquivalencePolicy, SortBlock
from anyalgebra.census.permutations import (
    CarrierPermutation,
    PermutationGenerationError,
    generate_allowed_permutations,
)
from anyalgebra.census.spec import CensusSpecCore


def _core(size: int, *, points: tuple[tuple[str, int], ...] = ()) -> CensusSpecCore:
    return CensusSpecCore.create(
        carrier_size=size,
        arity=2,
        distinguished_elements=points,
        corpus_name=f"permutation-{size}",
    )


@pytest.mark.parametrize(
    "policy",
    (
        EquivalencePolicy.literal(_core(4)),
        EquivalencePolicy.relabeling(_core(4)),
        EquivalencePolicy.relabeling(_core(4), fixed_elements=(0, 3)),
        EquivalencePolicy.relabeling(_core(4, points=(("zero", 0),))),
        EquivalencePolicy.relabeling(
            _core(5),
            sort_blocks=(
                SortBlock.create(name="even", elements=(0, 2, 4)),
                SortBlock.create(name="odd", elements=(1, 3)),
            ),
            fixed_elements=(2,),
        ),
        EquivalencePolicy.relabeling(_core(0)),
    ),
)
def test_generated_images_equal_independent_itertools_oracle(
    policy: EquivalencePolicy,
) -> None:
    observed = tuple(item.images for item in generate_allowed_permutations(policy))
    expected = tuple(
        candidate
        for candidate in permutations(range(policy.carrier_size))
        if policy.allows(candidate)
    )

    assert observed == expected
    assert len(observed) == policy.permutation_count
    assert observed == tuple(sorted(observed))


def test_identity_inverse_composition_and_policy_preservation_are_exhaustive() -> None:
    policy = EquivalencePolicy.relabeling(_core(4), fixed_elements=(0,))
    group = generate_allowed_permutations(policy)
    identity = CarrierPermutation.identity(policy)

    for left in group:
        assert policy.allows(left.images)
        assert left.then(identity) == left
        assert identity.then(left) == left
        assert left.then(left.inverse()) == identity
        assert left.inverse().then(left) == identity
        assert left.inverse().inverse() == left
    for left, right, third in product(group, repeat=3):
        assert left.then(right).then(third) == left.then(right.then(third))
        assert policy.allows(left.then(right).images)


def test_direction_is_old_index_to_new_index_and_then_is_function_composition() -> None:
    policy = EquivalencePolicy.relabeling(_core(3))
    first = CarrierPermutation.create(policy, (1, 2, 0))
    second = CarrierPermutation.create(policy, (2, 0, 1))

    assert tuple(first(index) for index in range(3)) == first.images
    assert first.then(second).images == tuple(
        second(first(index)) for index in range(3)
    )
    assert first.then(second).images == (0, 1, 2)


def test_cross_policy_composition_fails_even_when_images_have_same_size() -> None:
    left_policy = EquivalencePolicy.relabeling(_core(3))
    right_policy = EquivalencePolicy.relabeling(_core(3), fixed_elements=(0,))
    left = CarrierPermutation.identity(left_policy)
    right = CarrierPermutation.identity(right_policy)

    with pytest.raises(PermutationGenerationError) as caught:
        left.then(right)
    assert caught.value.field == "policy"


@pytest.mark.parametrize(
    "images",
    ((0, 1), (0, 1, 1), (0, 1, 3), (0, True, 2), [0, 1, 2]),
)
def test_factory_rejects_malformed_or_policy_forbidden_images(images: object) -> None:
    policy = EquivalencePolicy.relabeling(_core(3), fixed_elements=(0,))
    with pytest.raises(PermutationGenerationError):
        CarrierPermutation.create(policy, cast(tuple[int, ...], images))

    forbidden = (1, 0, 2)
    with pytest.raises(PermutationGenerationError, match="policy"):
        CarrierPermutation.create(policy, forbidden)


def test_generation_limit_is_checked_before_constructing_millions_of_images() -> None:
    policy = EquivalencePolicy.relabeling(_core(10))

    with pytest.raises(PermutationGenerationError) as caught:
        generate_allowed_permutations(policy)
    assert caught.value.field == "permutation_count"
    assert caught.value.requested == 3_628_800


def test_large_literal_policy_constructs_identity_without_factorial_search() -> None:
    policy = EquivalencePolicy.literal(_core(256))
    generated = generate_allowed_permutations(policy)

    assert len(generated) == 1
    assert generated[0].images == tuple(range(256))


def test_records_are_factory_owned_sealed_immutable_and_safely_represented() -> None:
    policy = EquivalencePolicy.relabeling(_core(3))
    value = CarrierPermutation.identity(policy)

    with pytest.raises(PermutationGenerationError, match="factory-owned"):
        CarrierPermutation()
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type("Derived", (CarrierPermutation,), {})
    with pytest.raises(FrozenInstanceError):
        value.images = (0, 1, 2)  # type: ignore[misc]
    assert not hasattr(value, "__dict__")
    assert repr(value) == "CarrierPermutation(size=3, images=(0, 1, 2))"


def test_nonexact_policy_fails_closed() -> None:
    with pytest.raises(PermutationGenerationError) as caught:
        generate_allowed_permutations(cast(EquivalencePolicy, None))
    assert caught.value.field == "policy"
