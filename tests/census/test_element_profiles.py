"""Exact identity, zero, idempotent, nilpotence, and color profiles."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import pytest

from anyalgebra.census.analysis import (
    FiniteLawAnalysisError,
    analyze_finite_elements,
    element_profile_canonical_bytes,
)
from anyalgebra.census.equivalence import EquivalencePolicy
from anyalgebra.census.spec import CensusSpecCore


def _core(size: int, arity: int = 2) -> CensusSpecCore:
    return CensusSpecCore.create(
        carrier_size=size,
        arity=arity,
        corpus_name=f"element-profile-{size}-{arity}",
    )


def test_cyclic_group_two_has_unique_identity_and_no_absorbing_zero() -> None:
    core = _core(2)
    result = analyze_finite_elements(
        core, (0, 1, 1, 0), EquivalencePolicy.relabeling(core)
    )

    assert (
        result.left_identities == result.right_identities == result.identities == (0,)
    )
    assert result.left_zeros == result.right_zeros == result.zeros == ()
    assert result.idempotents == (0,)
    assert result.identity_status == "unique"
    assert result.zero_status == "absent"
    table = result.outputs
    for identity in result.identities:
        assert all(
            table[identity * 2 + x] == table[x * 2 + identity] == x for x in range(2)
        )


def test_boolean_meet_has_identity_zero_and_two_idempotents() -> None:
    core = _core(2)
    result = analyze_finite_elements(
        core, (0, 0, 0, 1), EquivalencePolicy.relabeling(core)
    )

    assert result.identities == (1,)
    assert result.zeros == (0,)
    assert result.idempotents == (0, 1)
    assert result.identity_status == result.zero_status == "unique"


def test_nonunique_one_sided_structures_remain_explicit() -> None:
    core = _core(2)
    left_projection = analyze_finite_elements(
        core, (0, 0, 1, 1), EquivalencePolicy.relabeling(core)
    )

    assert left_projection.left_identities == ()
    assert left_projection.right_identities == (0, 1)
    assert left_projection.identities == ()
    assert left_projection.right_identity_status == "nonunique"
    assert left_projection.identity_status == "absent"


def test_declared_left_associated_nilpotence_is_exact_and_bounded() -> None:
    core = _core(3)
    # Truncated addition: 1*1=2 and 2*1=0; zero is absorbing.
    outputs = (0, 0, 0, 0, 2, 0, 0, 0, 0)
    result = analyze_finite_elements(
        core,
        outputs,
        EquivalencePolicy.relabeling(core),
        nilpotence_zero=0,
        max_nilpotence_index=4,
    )
    profiles = {item.element: item for item in result.elements}

    assert result.nilpotence_status == "computed_left_associated_bounded"
    assert profiles[0].nilpotence_index == 1
    assert profiles[1].nilpotence_index == 3
    assert profiles[2].nilpotence_index == 2
    assert result.nilpotent_elements == (0, 1, 2)


def test_undeclared_or_nonbinary_capabilities_are_typed_unsupported() -> None:
    unary = _core(3, 1)
    result = analyze_finite_elements(
        unary, (0, 1, 2), EquivalencePolicy.relabeling(unary)
    )

    assert result.identity_status == result.zero_status == "unsupported_nonbinary"
    assert result.left_identities == result.zeros == ()
    assert result.idempotents == (0, 1, 2)
    assert result.nilpotence_status == "unsupported_not_declared"
    assert all(item.nilpotence_index is None for item in result.elements)


def test_element_colors_are_equivariant_and_profiles_are_carrier_complete() -> None:
    core = _core(2)
    policy = EquivalencePolicy.relabeling(core)
    result = analyze_finite_elements(core, (0, 0, 0, 0), policy)

    assert tuple(item.element for item in result.elements) == (0, 1)
    assert tuple(item.color for item in result.elements) == result.colors
    assert result.examined_element_count == core.carrier_size


def test_invalid_nilpotence_declarations_fail_closed() -> None:
    core = _core(2)
    policy = EquivalencePolicy.relabeling(core)
    for zero, bound in ((None, 3), (0, None), (2, 3), (0, True), (0, 0)):
        with pytest.raises(FiniteLawAnalysisError) as caught:
            analyze_finite_elements(
                core,
                (0, 0, 0, 1),
                policy,
                nilpotence_zero=zero,
                max_nilpotence_index=bound,
            )
        assert caught.value.field == "nilpotence"


def test_empty_carrier_and_canonical_drift_boundaries() -> None:
    core = _core(0, 1)
    result = analyze_finite_elements(core, (), EquivalencePolicy.relabeling(core))
    assert result.elements == result.idempotents == ()
    assert result.examined_element_count == 0
    first = element_profile_canonical_bytes(result)
    assert first == element_profile_canonical_bytes(
        analyze_finite_elements(core, (), EquivalencePolicy.relabeling(core))
    )
    assert json.loads(first)["contentHash"]["digest"] == result.semantic_hash.digest
    with pytest.raises(FiniteLawAnalysisError, match="factory-owned"):
        type(result)()
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type("Derived", (type(result),), {})
    with pytest.raises(FrozenInstanceError):
        result.complete = False  # type: ignore[misc]
    object.__setattr__(result, "examined_element_count", 1)
    with pytest.raises(FiniteLawAnalysisError):
        element_profile_canonical_bytes(result)
