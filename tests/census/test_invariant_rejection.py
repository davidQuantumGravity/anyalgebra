"""Sound rejection-only invariant comparisons for finite operation tables."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json
from typing import cast

import pytest

from anyalgebra.census.equivalence import EquivalencePolicy
from anyalgebra.census.invariants import (
    InvariantComparisonError,
    compare_table_invariants,
    invariant_comparison_canonical_bytes,
    invariant_comparison_record,
)
from anyalgebra.census.permutations import generate_allowed_permutations
from anyalgebra.census.spec import CensusSpecCore
from anyalgebra.census.table_codes import unrank_operation_table
from anyalgebra.census.transport import transport_operation_table
from anyalgebra.core.parents import SemanticHash


def _core(size: int, arity: int = 2, *, name: str = "invariants") -> CensusSpecCore:
    return CensusSpecCore.create(
        carrier_size=size,
        arity=arity,
        corpus_name=f"{name}-{size}-{arity}",
    )


def test_every_relabelled_order_two_table_survives_all_rejection_filters() -> None:
    core = _core(2)
    policy = EquivalencePolicy.relabeling(core)
    for index in range(core.candidate_count):
        source = unrank_operation_table(core, index)
        for permutation in generate_allowed_permutations(policy):
            target = transport_operation_table(core, source, permutation).target_outputs
            result = compare_table_invariants(
                core,
                source,
                policy,
                core,
                target,
                policy,
            )
            assert result.rejected is False
            assert result.first_mismatch is None
            assert all(check.matches for check in result.checks)


def test_signature_mismatch_is_localized_before_table_specific_fields() -> None:
    left_core = _core(2, 1, name="left")
    right_core = _core(3, 1, name="right")
    result = compare_table_invariants(
        left_core,
        (0, 1),
        EquivalencePolicy.relabeling(left_core),
        right_core,
        (0, 1, 2),
        EquivalencePolicy.relabeling(right_core),
    )

    assert result.rejected is True
    assert result.first_mismatch == "carrier_size"
    mismatch_names = tuple(check.name for check in result.checks if not check.matches)
    assert mismatch_names[0] == "carrier_size"


def test_law_output_and_color_fields_are_recomputed_on_both_inputs() -> None:
    core = _core(2)
    policy = EquivalencePolicy.relabeling(core)
    left = (0, 0, 0, 0)
    right = (0, 1, 1, 0)
    result = compare_table_invariants(core, left, policy, core, right, policy)

    assert result.left.outputs == left
    assert result.right.outputs == right
    assert result.left.law_profile != result.right.law_profile
    assert result.left.output_multiplicity_histogram == (0, 4)
    assert result.right.output_multiplicity_histogram == (2, 2)
    assert "law_profile" in {check.name for check in result.checks if not check.matches}
    assert result.evidence_role == "rejection_only"
    assert result.complete_invariant is False


def test_distinguished_and_sort_policy_fields_are_part_of_rejection_data() -> None:
    left_core = CensusSpecCore.create(
        carrier_size=2,
        arity=1,
        distinguished_elements=(("zero", 0),),
        corpus_name="left-policy",
    )
    right_core = CensusSpecCore.create(
        carrier_size=2,
        arity=1,
        distinguished_elements=(("one", 1),),
        corpus_name="right-policy",
    )
    result = compare_table_invariants(
        left_core,
        (0, 1),
        EquivalencePolicy.relabeling(left_core),
        right_core,
        (0, 1),
        EquivalencePolicy.relabeling(right_core),
    )

    mismatches = {check.name for check in result.checks if not check.matches}
    assert "distinguished_elements" in mismatches
    assert "fixed_elements" in mismatches


def test_equal_invariants_never_construct_or_claim_isomorphism() -> None:
    core = _core(2)
    policy = EquivalencePolicy.relabeling(core)
    result = compare_table_invariants(
        core, (0, 0, 0, 0), policy, core, (0, 0, 0, 0), policy
    )

    assert result.rejected is False
    assert result.status == "not_rejected"
    assert result.complete_invariant is False
    assert not hasattr(result, "witness")


def test_comparison_record_is_canonical_content_addressed_and_deterministic() -> None:
    core = _core(2)
    policy = EquivalencePolicy.relabeling(core)
    result = compare_table_invariants(
        core, (0, 0, 0, 0), policy, core, (0, 1, 1, 0), policy
    )
    first = invariant_comparison_canonical_bytes(result)
    second = invariant_comparison_canonical_bytes(result)
    record = json.loads(first)
    body = {key: value for key, value in record.items() if key != "contentHash"}
    encoded = json.dumps(
        body,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    assert first == second
    assert record["contentHash"]["digest"] == hashlib.sha256(encoded).hexdigest()
    assert result.semantic_hash == SemanticHash(
        "sha256", record["contentHash"]["digest"]
    )


def test_nullary_and_empty_carrier_profiles_are_exact() -> None:
    nullary = _core(2, 0)
    nullary_policy = EquivalencePolicy.relabeling(nullary)
    result = compare_table_invariants(
        nullary, (0,), nullary_policy, nullary, (1,), nullary_policy
    )
    assert result.rejected is False
    assert dict(result.left.law_profile)["idempotent_applicable"] is False

    empty = _core(0, 1)
    empty_policy = EquivalencePolicy.relabeling(empty)
    empty_result = compare_table_invariants(
        empty, (), empty_policy, empty, (), empty_policy
    )
    assert empty_result.left.output_multiplicity_histogram == ()
    assert empty_result.rejected is False


def test_invalid_inputs_and_no_table_core_fail_closed() -> None:
    core = _core(2)
    policy = EquivalencePolicy.relabeling(core)
    with pytest.raises(InvariantComparisonError) as caught:
        compare_table_invariants(core, (0,), policy, core, (0, 0, 0, 0), policy)
    assert caught.value.field == "left_outputs"
    with pytest.raises(InvariantComparisonError) as caught:
        compare_table_invariants(
            cast(CensusSpecCore, None),
            (),
            cast(EquivalencePolicy, None),
            core,
            (0, 0, 0, 0),
            policy,
        )
    assert caught.value.field == "left_core"

    impossible = _core(0, 0)
    with pytest.raises(InvariantComparisonError, match="no operation tables"):
        compare_table_invariants(
            impossible,
            (),
            EquivalencePolicy.literal(impossible),
            impossible,
            (),
            EquivalencePolicy.literal(impossible),
        )


def test_records_are_sealed_immutable_and_drift_fails_closed() -> None:
    core = _core(2)
    policy = EquivalencePolicy.relabeling(core)
    result = compare_table_invariants(
        core, (0, 0, 0, 0), policy, core, (0, 1, 1, 0), policy
    )
    with pytest.raises(InvariantComparisonError, match="factory-owned"):
        type(result)()
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type("Derived", (type(result),), {})
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type("DerivedCheck", (type(result.checks[0]),), {})
    with pytest.raises(FrozenInstanceError):
        result.rejected = False  # type: ignore[misc]

    object.__setattr__(result, "first_mismatch", None)
    with pytest.raises(InvariantComparisonError) as caught:
        invariant_comparison_record(result)
    assert caught.value.field in {"comparison", "semantic_hash"}
