"""Contracts for brute-force canonical labels under allowed relabelings."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json
from typing import cast

import pytest

from anyalgebra.census.canonical import (
    CanonicalLabelError,
    canonicalize_operation_table,
)
from anyalgebra.census.equivalence import EquivalencePolicy, SortBlock
from anyalgebra.census.permutations import generate_allowed_permutations
from anyalgebra.census.spec import CensusSpecCore
from anyalgebra.census.table_codes import unrank_operation_table
from anyalgebra.census.transport import transport_operation_table
from anyalgebra.core.parents import SemanticHash


def _core(size: int, arity: int = 2) -> CensusSpecCore:
    return CensusSpecCore.create(
        carrier_size=size,
        arity=arity,
        corpus_name=f"canonical-{size}-{arity}",
    )


def test_every_orbit_member_has_identical_canonical_bytes_and_identifier() -> None:
    core = _core(2)
    policy = EquivalencePolicy.relabeling(core)
    group = generate_allowed_permutations(policy)

    for index in range(core.candidate_count):
        source = unrank_operation_table(core, index)
        orbit = tuple(
            transport_operation_table(core, source, permutation).target_outputs
            for permutation in group
        )
        labels = tuple(
            canonicalize_operation_table(core, item, policy) for item in orbit
        )
        assert len({label.canonical_bytes for label in labels}) == 1
        assert len({label.canonical_id for label in labels}) == 1
        assert all(label.canonical_outputs == min(orbit) for label in labels)


def test_canonical_table_is_independent_minimum_over_all_allowed_images() -> None:
    core = _core(3, 1)
    policy = EquivalencePolicy.relabeling(core, fixed_elements=(0,))
    source = (2, 0, 1)
    direct_images = tuple(
        transport_operation_table(core, source, permutation).target_outputs
        for permutation in generate_allowed_permutations(policy)
    )
    result = canonicalize_operation_table(core, source, policy)

    assert result.canonical_outputs == min(direct_images)
    assert result.permutations_checked == policy.permutation_count
    assert result.orbit_size == len(set(direct_images))
    assert result.certificate.target_outputs == result.canonical_outputs
    assert result.certificate.verify_cells() is True


def test_ties_choose_first_lexicographic_permutation_certificate() -> None:
    core = _core(4)
    policy = EquivalencePolicy.relabeling(core)
    constant_zero = (0,) * core.input_tuple_count
    result = canonicalize_operation_table(core, constant_zero, policy)

    tied = tuple(
        permutation
        for permutation in generate_allowed_permutations(policy)
        if transport_operation_table(core, constant_zero, permutation).target_outputs
        == result.canonical_outputs
    )
    assert result.certificate.permutation == tied[0]
    assert tied[0].images == tuple(range(4))


def test_orbit_stabilizer_count_is_exact() -> None:
    core = _core(3, 1)
    policy = EquivalencePolicy.relabeling(core)
    source = (0, 0, 0)
    result = canonicalize_operation_table(core, source, policy)

    assert result.stabilizer_size * result.orbit_size == policy.permutation_count
    assert result.stabilizer_size == sum(
        transport_operation_table(core, source, permutation).target_outputs == source
        for permutation in generate_allowed_permutations(policy)
    )


def test_canonical_bytes_have_one_exact_content_hash() -> None:
    core = _core(2)
    policy = EquivalencePolicy.relabeling(core)
    result = canonicalize_operation_table(core, (0, 1, 1, 0), policy)
    record = json.loads(result.canonical_bytes)

    assert record["schemaType"] == "anyalgebra.census.canonical_table"
    assert record["schemaVersion"] == 1
    assert record["outputs"] == list(result.canonical_outputs)
    assert result.canonical_id == SemanticHash(
        "sha256", hashlib.sha256(result.canonical_bytes).hexdigest()
    )


def test_equivalence_policy_is_part_of_canonical_identity() -> None:
    core = _core(2)
    source = (0, 1, 1, 0)
    literal = canonicalize_operation_table(
        core, source, EquivalencePolicy.literal(core)
    )
    relabeling = canonicalize_operation_table(
        core, source, EquivalencePolicy.relabeling(core)
    )

    assert literal.canonical_outputs == relabeling.canonical_outputs
    assert literal.canonical_id != relabeling.canonical_id


def test_sort_blocks_and_fixed_points_restrict_the_canonical_search() -> None:
    core = _core(4, 1)
    policy = EquivalencePolicy.relabeling(
        core,
        fixed_elements=(0,),
        sort_blocks=(
            SortBlock.create(name="left", elements=(0, 2)),
            SortBlock.create(name="right", elements=(1, 3)),
        ),
    )
    result = canonicalize_operation_table(core, (3, 2, 1, 0), policy)

    assert policy.allows(result.certificate.permutation.images)
    assert result.permutations_checked == 2


def test_literal_empty_and_nullary_boundaries() -> None:
    nullary = _core(2, 0)
    literal = EquivalencePolicy.literal(nullary)
    result = canonicalize_operation_table(nullary, (1,), literal)
    assert result.canonical_outputs == (1,)
    assert result.orbit_size == result.stabilizer_size == 1

    empty = _core(0, 1)
    empty_result = canonicalize_operation_table(
        empty, (), EquivalencePolicy.relabeling(empty)
    )
    assert empty_result.canonical_outputs == ()

    impossible = _core(0, 0)
    with pytest.raises(CanonicalLabelError, match="no operation tables"):
        canonicalize_operation_table(
            impossible, (), EquivalencePolicy.relabeling(impossible)
        )


def test_core_policy_and_table_mismatches_fail_closed() -> None:
    core = _core(2)
    other = _core(2)
    with pytest.raises(CanonicalLabelError) as caught:
        canonicalize_operation_table(
            core, (0, 0, 0, 0), EquivalencePolicy.relabeling(other)
        )
    assert caught.value.field == "equivalence"

    with pytest.raises(CanonicalLabelError) as caught:
        canonicalize_operation_table(
            core, (0, 0, 0), EquivalencePolicy.relabeling(core)
        )
    assert caught.value.field == "source_outputs"


def test_result_is_factory_owned_sealed_immutable_and_safely_represented() -> None:
    core = _core(2)
    result = canonicalize_operation_table(
        core, (0, 0, 0, 0), EquivalencePolicy.relabeling(core)
    )

    with pytest.raises(CanonicalLabelError, match="factory-owned"):
        type(result)()
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type("Derived", (type(result),), {})
    with pytest.raises(FrozenInstanceError):
        result.canonical_outputs = ()  # type: ignore[misc]
    assert result.core.corpus_name not in repr(result)
    assert result.canonical_id.digest in repr(result)


def test_nonexact_core_fails_before_other_inputs() -> None:
    with pytest.raises(CanonicalLabelError) as caught:
        canonicalize_operation_table(
            cast(CensusSpecCore, None), (), cast(EquivalencePolicy, None)
        )
    assert caught.value.field == "core"
