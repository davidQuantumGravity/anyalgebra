"""Deterministic explicit search for finite-table isomorphism witnesses."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from typing import cast

import pytest

from anyalgebra.census.equivalence import EquivalencePolicy
from anyalgebra.census.isomorphism import (
    IsomorphismOutcomeError,
    isomorphism_search_canonical_bytes,
    search_isomorphism,
)
from anyalgebra.census.partition import compute_invariant_partition
from anyalgebra.census.permutations import generate_allowed_permutations
from anyalgebra.census.spec import CensusSpecCore
from anyalgebra.census.table_codes import unrank_operation_table
from anyalgebra.census.transport import transport_operation_table


def _core(size: int, arity: int = 2) -> CensusSpecCore:
    return CensusSpecCore.create(
        carrier_size=size,
        arity=arity,
        corpus_name=f"isomorphism-search-{size}-{arity}",
    )


def test_every_relabelled_order_two_table_returns_a_cell_verified_map() -> None:
    core = _core(2)
    policy = EquivalencePolicy.relabeling(core)
    for index in range(core.candidate_count):
        source = unrank_operation_table(core, index)
        for permutation in generate_allowed_permutations(policy):
            target = transport_operation_table(core, source, permutation).target_outputs
            result = search_isomorphism(core, source, target, policy)

            assert result.found is True
            assert result.witness is not None
            assert result.outcome is not None
            assert result.witness.verify_cells() is True
            assert result.witness.target_outputs == target
            assert result.outcome.witness == result.witness


def test_search_returns_first_lexicographic_color_compatible_witness() -> None:
    core = _core(3, 1)
    policy = EquivalencePolicy.relabeling(core)
    source = (0, 0, 0)
    target = (1, 1, 1)
    left_partition = compute_invariant_partition(core, source, policy)
    right_partition = compute_invariant_partition(core, target, policy)
    compatible = tuple(
        permutation
        for permutation in generate_allowed_permutations(policy)
        if all(
            left_partition.colors[old] == right_partition.colors[permutation(old)]
            for old in range(core.carrier_size)
        )
    )
    expected = next(
        permutation
        for permutation in compatible
        if transport_operation_table(core, source, permutation).target_outputs == target
    )
    result = search_isomorphism(core, source, target, policy)

    assert result.witness is not None
    assert result.witness.permutation == expected
    assert result.permutations_examined == compatible.index(expected) + 1


def test_complete_no_map_search_has_no_positive_outcome() -> None:
    core = _core(2)
    policy = EquivalencePolicy.relabeling(core)
    result = search_isomorphism(core, (0, 0, 0, 0), (0, 0, 0, 1), policy)

    assert result.found is False
    assert result.complete is True
    assert result.stop_reason == "exhausted_color_compatible_bijections"
    assert result.witness is None
    assert result.outcome is None
    assert result.permutations_examined == result.color_compatible_count


def test_population_equations_and_search_order_are_stable() -> None:
    core = _core(3, 1)
    policy = EquivalencePolicy.relabeling(core)
    first = search_isomorphism(core, (2, 2, 0), (1, 0, 0), policy)
    second = search_isomorphism(core, (2, 2, 0), (1, 0, 0), policy)

    assert first == second
    assert first.total_allowed_count == policy.permutation_count
    assert (
        first.color_compatible_count + first.partition_rejected_count
        == first.total_allowed_count
    )
    assert first.permutations_examined <= first.color_compatible_count
    assert first.strategy == "color_compatible_lexicographic_v1"
    assert isomorphism_search_canonical_bytes(first) == (
        isomorphism_search_canonical_bytes(second)
    )
    assert json.loads(isomorphism_search_canonical_bytes(first))["status"] in {
        "isomorphic",
        "search_exhausted_no_map",
    }


def test_distinguished_elements_are_preserved_by_every_returned_map() -> None:
    core = CensusSpecCore.create(
        carrier_size=3,
        arity=1,
        distinguished_elements=(("zero", 0),),
        corpus_name="isomorphism-search-pointed",
    )
    policy = EquivalencePolicy.relabeling(core)
    source = (0, 2, 1)
    permutation = generate_allowed_permutations(policy)[1]
    target = transport_operation_table(core, source, permutation).target_outputs
    result = search_isomorphism(core, source, target, policy)

    assert result.witness is not None
    assert result.witness.permutation(0) == 0
    assert policy.allows(result.witness.permutation.images)


def test_nullary_and_empty_carrier_boundaries() -> None:
    nullary = _core(2, 0)
    nullary_policy = EquivalencePolicy.relabeling(nullary)
    result = search_isomorphism(nullary, (0,), (1,), nullary_policy)
    assert result.found is True
    assert result.witness is not None
    assert result.witness.target_outputs == (1,)

    empty = _core(0, 1)
    empty_policy = EquivalencePolicy.relabeling(empty)
    empty_result = search_isomorphism(empty, (), (), empty_policy)
    assert empty_result.found is True
    assert empty_result.total_allowed_count == 1


def test_invalid_inputs_policy_mismatch_and_no_table_core_fail_closed() -> None:
    core = _core(2)
    other = _core(2)
    with pytest.raises(IsomorphismOutcomeError) as caught:
        search_isomorphism(
            core,
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            EquivalencePolicy.relabeling(other),
        )
    assert caught.value.field == "equivalence"
    with pytest.raises(IsomorphismOutcomeError) as caught:
        search_isomorphism(core, (0,), (0, 0, 0, 0), EquivalencePolicy.relabeling(core))
    assert caught.value.field == "left_outputs"
    with pytest.raises(IsomorphismOutcomeError) as caught:
        search_isomorphism(
            cast(CensusSpecCore, None), (), (), cast(EquivalencePolicy, None)
        )
    assert caught.value.field == "core"

    impossible = _core(0, 0)
    with pytest.raises(IsomorphismOutcomeError, match="no operation tables"):
        search_isomorphism(
            impossible,
            (),
            (),
            EquivalencePolicy.literal(impossible),
        )


def test_search_result_is_sealed_immutable_and_drift_fails_closed() -> None:
    core = _core(2)
    policy = EquivalencePolicy.relabeling(core)
    result = search_isomorphism(core, (0, 0, 0, 0), (1, 1, 1, 1), policy)
    with pytest.raises(IsomorphismOutcomeError, match="factory-owned"):
        type(result)()
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type("Derived", (type(result),), {})
    with pytest.raises(FrozenInstanceError):
        result.found = False  # type: ignore[misc]

    object.__setattr__(result, "permutations_examined", 99)
    with pytest.raises(IsomorphismOutcomeError):
        isomorphism_search_canonical_bytes(result)
