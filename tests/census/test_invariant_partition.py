"""Soundness contracts for element-invariant canonical-search ordering."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from anyalgebra.census.canonical import canonicalize_operation_table
from anyalgebra.census.equivalence import EquivalencePolicy, SortBlock
from anyalgebra.census.partition import (
    InvariantPartitionError,
    canonicalize_with_invariant_partition,
    compute_invariant_partition,
    order_permutations_by_partition,
)
from anyalgebra.census.permutations import generate_allowed_permutations
from anyalgebra.census.spec import CensusSpecCore
from anyalgebra.census.table_codes import unrank_operation_table
from anyalgebra.census.transport import transport_operation_table


def _core(size: int, arity: int = 2) -> CensusSpecCore:
    return CensusSpecCore.create(
        carrier_size=size,
        arity=arity,
        corpus_name=f"partition-{size}-{arity}",
    )


def test_every_allowed_order_two_isomorphism_preserves_element_colors() -> None:
    core = _core(2)
    policy = EquivalencePolicy.relabeling(core)
    for index in range(core.candidate_count):
        source = unrank_operation_table(core, index)
        source_partition = compute_invariant_partition(core, source, policy)
        for permutation in generate_allowed_permutations(policy):
            target = transport_operation_table(core, source, permutation).target_outputs
            target_partition = compute_invariant_partition(core, target, policy)
            assert all(
                source_partition.colors[old]
                == target_partition.colors[permutation(old)]
                for old in range(core.carrier_size)
            )


def test_sort_blocks_fixed_points_and_distinguished_points_seed_colors() -> None:
    core = CensusSpecCore.create(
        carrier_size=4,
        arity=1,
        distinguished_elements=(("zero", 0),),
        corpus_name="partition-policy",
    )
    policy = EquivalencePolicy.relabeling(
        core,
        fixed_elements=(3,),
        sort_blocks=(
            SortBlock.create(name="even", elements=(0, 2)),
            SortBlock.create(name="odd", elements=(1, 3)),
        ),
    )
    partition = compute_invariant_partition(core, (0, 1, 2, 3), policy)

    assert len(set(partition.colors)) == 4
    assert dict(partition.blocks) == {
        color: (element,) for element, color in enumerate(partition.colors)
    }


def test_partition_order_contains_the_exact_allowed_group_once() -> None:
    core = _core(3, 1)
    policy = EquivalencePolicy.relabeling(core)
    partition = compute_invariant_partition(core, (2, 2, 0), policy)
    ordered = order_permutations_by_partition(partition)
    reference = generate_allowed_permutations(policy)

    assert len(ordered) == len(reference)
    assert {item.images for item in ordered} == {item.images for item in reference}
    assert tuple(item.images for item in ordered) == tuple(
        item.images for item in order_permutations_by_partition(partition)
    )


def test_refinement_changes_accounted_work_but_not_canonical_result() -> None:
    core = _core(3, 1)
    policy = EquivalencePolicy.relabeling(core)
    source = (2, 2, 0)
    reference = canonicalize_operation_table(core, source, policy)
    coarse = canonicalize_with_invariant_partition(
        core, source, policy, max_refinement_rounds=0
    )
    refined = canonicalize_with_invariant_partition(core, source, policy)

    assert coarse.partition.signature_evaluations == 0
    assert refined.partition.signature_evaluations > 0
    for result in (coarse, refined):
        assert result.label.canonical_outputs == reference.canonical_outputs
        assert result.label.canonical_bytes == reference.canonical_bytes
        assert result.label.canonical_id == reference.canonical_id
        assert result.label.orbit_size == reference.orbit_size
        assert result.label.certificate == reference.certificate
        assert result.permutations_checked == policy.permutation_count
        assert result.permutations_pruned == 0


def test_partition_is_explicitly_ordering_only_not_a_complete_invariant() -> None:
    core = _core(2)
    policy = EquivalencePolicy.relabeling(core)
    first = compute_invariant_partition(core, (0, 0, 0, 0), policy)
    second = compute_invariant_partition(core, (0, 0, 0, 1), policy)

    assert first.colors == second.colors
    assert (
        canonicalize_operation_table(core, first.source_outputs, policy).canonical_id
        != canonicalize_operation_table(
            core, second.source_outputs, policy
        ).canonical_id
    )
    assert first.complete_invariant is False
    assert first.search_role == "ordering_only"


def test_nullary_literal_and_empty_carrier_boundaries() -> None:
    nullary_core = _core(2, 0)
    literal = EquivalencePolicy.literal(nullary_core)
    nullary = canonicalize_with_invariant_partition(nullary_core, (1,), literal)
    assert nullary.label.canonical_outputs == (1,)
    assert nullary.partition.colors == (0, 1)

    empty_core = _core(0, 1)
    empty_policy = EquivalencePolicy.relabeling(empty_core)
    empty = canonicalize_with_invariant_partition(empty_core, (), empty_policy)
    assert empty.partition.colors == ()
    assert empty.partition.blocks == ()
    assert empty.permutations_checked == 1


def test_invalid_core_policy_table_and_round_bound_fail_closed() -> None:
    core = _core(2)
    other = _core(2)
    policy = EquivalencePolicy.relabeling(core)
    with pytest.raises(InvariantPartitionError) as caught:
        compute_invariant_partition(
            core, (0, 0, 0, 0), EquivalencePolicy.relabeling(other)
        )
    assert caught.value.field == "equivalence"
    with pytest.raises(InvariantPartitionError) as caught:
        compute_invariant_partition(core, (0, 0, 0), policy)
    assert caught.value.field == "source_outputs"
    with pytest.raises(InvariantPartitionError) as caught:
        compute_invariant_partition(
            core, (0, 0, 0, 0), policy, max_refinement_rounds=True
        )
    assert caught.value.field == "max_refinement_rounds"
    with pytest.raises(InvariantPartitionError) as caught:
        compute_invariant_partition(
            cast(CensusSpecCore, None), (), cast(EquivalencePolicy, None)
        )
    assert caught.value.field == "core"


def test_partition_and_guided_result_are_sealed_immutable_and_deterministic() -> None:
    core = _core(3, 1)
    policy = EquivalencePolicy.relabeling(core)
    first = canonicalize_with_invariant_partition(core, (2, 2, 0), policy)
    second = canonicalize_with_invariant_partition(core, (2, 2, 0), policy)

    assert first == second
    assert repr(first) == repr(second)
    with pytest.raises(InvariantPartitionError, match="factory-owned"):
        type(first.partition)()
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type("Derived", (type(first.partition),), {})
    with pytest.raises(FrozenInstanceError):
        first.partition.colors = ()  # type: ignore[misc]
