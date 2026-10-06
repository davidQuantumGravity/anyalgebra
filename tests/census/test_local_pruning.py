"""Soundness contracts for incremental local constraint pruning."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from anyalgebra.census import CensusSpec
from anyalgebra.census.bounds import CensusOrdering, EnumerationBounds
from anyalgebra.census.constraints import CensusConstraint, CensusTerm, ConstraintSet
from anyalgebra.census.equivalence import EquivalencePolicy
from anyalgebra.census.filtering import compile_constraint_filter
from anyalgebra.census.pruning import (
    PrefixPruningError,
    check_prefix,
    enumerate_pruned,
)
from anyalgebra.census.reference import enumerate_reference
from anyalgebra.census.spec import CensusSpecCore


def _spec(
    constraints: tuple[CensusConstraint, ...],
    *,
    size: int = 2,
    arity: int = 2,
) -> CensusSpec:
    core = CensusSpecCore.create(
        carrier_size=size,
        arity=arity,
        corpus_name=f"pruning-{size}-{arity}",
    )
    return CensusSpec.create(
        carrier_size=size,
        arity=arity,
        constraints=ConstraintSet.create(core, constraints),
        equivalence=EquivalencePolicy.literal(core),
        ordering=CensusOrdering.reference(),
        bounds=EnumerationBounds.create(
            max_candidates=100_000,
            max_orbits=100_000,
            max_work_units=1_000_000,
            max_memory_bytes=100_000_000,
        ),
    )


@pytest.mark.parametrize(
    ("constraint", "prefix", "reason", "skipped"),
    (
        (CensusConstraint.commutative(), (0, 0, 1), "commutativity_mismatch", 2),
        (CensusConstraint.idempotent(), (1,), "idempotence_mismatch", 8),
        (
            CensusConstraint.identity(element=0, side="left"),
            (1,),
            "identity_mismatch",
            8,
        ),
        (CensusConstraint.quasigroup(), (0, 0), "quasigroup_duplicate", 4),
    ),
)
def test_local_assigned_cell_contradictions_prune_exact_subtrees(
    constraint: CensusConstraint,
    prefix: tuple[int, ...],
    reason: str,
    skipped: int,
) -> None:
    decision = check_prefix(_spec((constraint,)), prefix)

    assert decision.pruned is True
    assert decision.skipped_candidate_count == skipped
    assert decision.violations[0].reason == reason
    assert decision.violations[0].witness


def test_nullary_value_can_prune_its_single_completed_candidate() -> None:
    spec = _spec((CensusConstraint.nullary_value(element=1),), size=2, arity=0)

    rejected = check_prefix(spec, (0,))
    accepted = check_prefix(spec, (1,))
    assert rejected.pruned is True
    assert rejected.skipped_candidate_count == 1
    assert accepted.pruned is False


def test_unresolved_cells_never_cause_speculative_pruning() -> None:
    constraints = (
        CensusConstraint.commutative(),
        CensusConstraint.idempotent(),
        CensusConstraint.identity(element=0, side="two_sided"),
        CensusConstraint.quasigroup(),
    )

    decision = check_prefix(_spec(constraints), ())
    assert decision.pruned is False
    assert decision.violations == ()
    assert decision.skipped_candidate_count == 0


def test_equation_prunes_only_after_both_sides_are_resolved() -> None:
    x, y, z = (CensusTerm.variable(index) for index in range(3))
    associative = CensusConstraint.equation(
        CensusTerm.apply(CensusTerm.apply(x, y), z),
        CensusTerm.apply(x, CensusTerm.apply(y, z)),
        variable_count=3,
    )
    spec = _spec((associative,))
    raw = enumerate_reference(spec)
    compiled = compile_constraint_filter(spec)
    rejected = next(
        item for item in raw.candidates if not compiled.evaluate(item).accepted
    )

    assert check_prefix(spec, rejected.outputs[:-1]).pruned is False
    full = check_prefix(spec, rejected.outputs)
    assert full.pruned is True
    assert full.skipped_candidate_count == 1
    assert full.violations[0].reason == "equation_mismatch"


@pytest.mark.parametrize(
    "constraints",
    (
        (CensusConstraint.commutative(),),
        (CensusConstraint.idempotent(),),
        (CensusConstraint.identity(element=0, side="two_sided"),),
        (CensusConstraint.quasigroup(),),
        (CensusConstraint.commutative(), CensusConstraint.idempotent()),
        (
            CensusConstraint.commutative(),
            CensusConstraint.idempotent(),
            CensusConstraint.identity(element=0, side="two_sided"),
        ),
    ),
)
def test_pruned_and_unpruned_accepted_sequences_are_identical(
    constraints: tuple[CensusConstraint, ...],
) -> None:
    spec = _spec(constraints)
    raw = enumerate_reference(spec)
    compiled = compile_constraint_filter(spec)
    expected = tuple(
        item for item in raw.candidates if compiled.evaluate(item).accepted
    )
    pruned = enumerate_pruned(spec)

    assert pruned.accepted_candidates == expected
    assert pruned.examined_candidate_count + pruned.skipped_candidate_count == 16
    assert pruned.complete is True


def test_every_pruning_event_replays_to_the_same_violation_and_subtree() -> None:
    spec = _spec((CensusConstraint.commutative(), CensusConstraint.idempotent()))
    result = enumerate_pruned(spec)

    assert result.events
    for event in result.events:
        replay = check_prefix(spec, event.prefix)
        assert replay == event


def test_all_detectable_violations_are_retained_in_canonical_order() -> None:
    spec = _spec(
        (
            CensusConstraint.commutative(),
            CensusConstraint.idempotent(),
            CensusConstraint.identity(element=0, side="left"),
        )
    )
    decision = check_prefix(spec, (1, 0, 1, 0))

    assert decision.pruned is True
    assert tuple(item.constraint.kind for item in decision.violations) == (
        "commutative",
        "idempotent",
        "identity",
    )


@pytest.mark.parametrize(
    ("prefix", "reason"),
    (
        ((0, 0, 0, 0, 0), "longer"),
        ((0, True), "exact built-in ints"),
        ((0, -1), "outside"),
        ((0, 2), "outside"),
        ("00", "iterable"),
    ),
)
def test_malformed_prefixes_fail_before_tree_search(
    prefix: object, reason: str
) -> None:
    with pytest.raises(PrefixPruningError, match=reason):
        check_prefix(_spec(()), cast(tuple[int, ...], prefix))


def test_prefix_source_is_consumed_at_most_one_past_table_length() -> None:
    consumed = 0

    def forever() -> object:
        nonlocal consumed
        while True:
            consumed += 1
            yield 0

    with pytest.raises(PrefixPruningError, match="longer"):
        check_prefix(_spec(()), cast(tuple[int, ...], forever()))
    assert consumed == 5


def test_pruning_records_are_factory_owned_sealed_immutable_and_unhashable() -> None:
    decision = check_prefix(_spec((CensusConstraint.idempotent(),)), (1,))
    enumeration = enumerate_pruned(_spec((CensusConstraint.idempotent(),)))

    with pytest.raises(PrefixPruningError, match="factory-owned"):
        type(decision)()
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type("Derived", (type(decision),), {})
    with pytest.raises(FrozenInstanceError):
        decision.pruned = False  # type: ignore[misc]
    with pytest.raises(TypeError, match="unhashable"):
        hash(decision)
    with pytest.raises(PrefixPruningError, match="factory-owned"):
        type(decision.violations[0])()
    with pytest.raises(PrefixPruningError, match="factory-owned"):
        type(enumeration)()


def test_complete_differential_memory_is_preflighted() -> None:
    spec = _spec((CensusConstraint.idempotent(),))
    constrained = CensusSpec.create(
        carrier_size=spec.carrier_size,
        arity=spec.arity,
        constraints=spec.constraints,
        equivalence=spec.equivalence,
        ordering=spec.ordering,
        bounds=EnumerationBounds.create(
            max_candidates=100,
            max_orbits=100,
            max_work_units=100,
            max_memory_bytes=3_071,
        ),
    )

    with pytest.raises(PrefixPruningError) as caught:
        enumerate_pruned(constrained)
    assert caught.value.field == "max_memory_bytes"


def test_nonexact_spec_is_rejected() -> None:
    with pytest.raises(PrefixPruningError) as caught:
        check_prefix(cast(CensusSpec, None), ())
    assert caught.value.field == "spec"
