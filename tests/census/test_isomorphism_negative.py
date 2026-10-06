"""Exact invariant, exhaustive, and bounded comparison outcomes."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import pytest

from anyalgebra.census.equivalence import EquivalencePolicy
from anyalgebra.census.isomorphism import (
    ExhaustiveNonIsomorphic,
    Inconclusive,
    InvariantNonIsomorphic,
    Isomorphic,
    IsomorphismOutcomeError,
    compare_isomorphism,
    comparison_outcome_canonical_bytes,
)
from anyalgebra.census.spec import CensusSpecCore


def _core(size: int = 2, arity: int = 2) -> CensusSpecCore:
    return CensusSpecCore.create(
        carrier_size=size,
        arity=arity,
        corpus_name=f"negative-comparison-{size}-{arity}",
    )


def _unrejected_nonisomorphic_pair() -> tuple[tuple[int, ...], tuple[int, ...]]:
    return (1, 0, 1, 0), (1, 1, 0, 0)


def test_first_invariant_mismatch_returns_exact_zero_search_negative() -> None:
    core = _core()
    policy = EquivalencePolicy.relabeling(core)
    result = compare_isomorphism(core, (0, 0, 0, 0), (0, 1, 1, 0), policy)

    assert type(result) is InvariantNonIsomorphic
    assert result.status == "non_isomorphic"
    assert result.proof_kind == "invariant_mismatch"
    assert result.mismatch.name == result.comparison.first_mismatch
    assert result.mismatch.matches is False
    assert result.permutations_examined == 0


def test_exhaustive_negative_accounts_for_every_allowed_bijection() -> None:
    core = _core()
    policy = EquivalencePolicy.relabeling(core)
    left, right = _unrejected_nonisomorphic_pair()
    result = compare_isomorphism(core, left, right, policy)

    assert type(result) is ExhaustiveNonIsomorphic
    assert result.status == "non_isomorphic"
    assert result.proof_kind == "exhaustive_no_map"
    assert result.total_allowed_count == policy.permutation_count
    assert result.checked_count == result.total_allowed_count
    assert (
        result.partition_rejected_count + result.operation_checked_count
        == result.checked_count
    )
    assert result.search.found is False
    assert result.search.complete is True


def test_bound_exhaustion_with_remaining_work_is_never_nonisomorphic() -> None:
    core = _core()
    policy = EquivalencePolicy.relabeling(core)
    left, right = _unrejected_nonisomorphic_pair()
    result = compare_isomorphism(core, left, right, policy, max_permutations=0)

    assert type(result) is Inconclusive
    assert result.status == "bounded_inconclusive"
    assert result.complete is False
    assert result.bound_name == "max_permutations"
    assert result.examined_count == 0
    assert result.remaining_count > 0
    assert not isinstance(result, ExhaustiveNonIsomorphic)


def test_sufficient_bound_can_return_positive_or_exhaustive_negative() -> None:
    core = _core()
    policy = EquivalencePolicy.relabeling(core)
    positive = compare_isomorphism(
        core, (0, 0, 0, 0), (1, 1, 1, 1), policy, max_permutations=2
    )
    assert type(positive) is Isomorphic
    assert positive.witness.verify_cells() is True

    left, right = _unrejected_nonisomorphic_pair()
    negative = compare_isomorphism(
        core, left, right, policy, max_permutations=policy.permutation_count
    )
    assert type(negative) is ExhaustiveNonIsomorphic


@pytest.mark.parametrize("invalid", [-1, True, 1.0, "1"])
def test_invalid_search_bounds_fail_before_comparison(invalid: object) -> None:
    core = _core()
    policy = EquivalencePolicy.relabeling(core)
    with pytest.raises(IsomorphismOutcomeError) as caught:
        compare_isomorphism(
            core,
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            policy,
            max_permutations=invalid,  # type: ignore[arg-type]
        )
    assert caught.value.field == "max_permutations"


@pytest.mark.parametrize(
    "result_factory",
    [
        lambda core, policy: compare_isomorphism(
            core, (0, 0, 0, 0), (0, 1, 1, 0), policy
        ),
        lambda core, policy: compare_isomorphism(
            core, *_unrejected_nonisomorphic_pair(), policy
        ),
        lambda core, policy: compare_isomorphism(
            core, *_unrejected_nonisomorphic_pair(), policy, max_permutations=0
        ),
    ],
)
def test_all_negative_and_bounded_records_are_canonical_and_deterministic(
    result_factory: object,
) -> None:
    core = _core()
    policy = EquivalencePolicy.relabeling(core)
    result = result_factory(core, policy)  # type: ignore[operator]
    first = comparison_outcome_canonical_bytes(result)
    second = comparison_outcome_canonical_bytes(result)
    record = json.loads(first)

    assert first == second
    assert record["status"] in {"non_isomorphic", "bounded_inconclusive"}
    assert record["contentHash"]["digest"] == result.semantic_hash.digest


def test_new_negative_records_are_sealed_immutable_and_drift_fails_closed() -> None:
    core = _core()
    policy = EquivalencePolicy.relabeling(core)
    left, right = _unrejected_nonisomorphic_pair()
    result = compare_isomorphism(core, left, right, policy)
    assert type(result) is ExhaustiveNonIsomorphic

    with pytest.raises(IsomorphismOutcomeError, match="factory-owned"):
        type(result)()
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type("Derived", (type(result),), {})
    with pytest.raises(FrozenInstanceError):
        result.checked_count = 0  # type: ignore[misc]

    object.__setattr__(result, "checked_count", 0)
    with pytest.raises(IsomorphismOutcomeError):
        comparison_outcome_canonical_bytes(result)
