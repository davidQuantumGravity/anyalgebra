"""Seeded property and metamorphic controls for the composed v0.1 census.

The module deliberately avoids an optional property-testing dependency.  The
fixed seed, finite parameter grid, case count, and replay text are part of the
test contract.  ``_assert_table_property`` greedily reduces failing output
digits toward zero and reports the reduced table as a copyable fixture.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
from random import Random

import pytest

from anyalgebra.atlas.build import build_finite_algebra_atlas
from anyalgebra.atlas.query import (
    AtlasQueryError,
    load_finite_algebra_atlas,
    query_atlas,
)
from anyalgebra.census.bounds import CensusOrdering, EnumerationBounds
from anyalgebra.census.canonical import canonicalize_operation_table
from anyalgebra.census.constraints import CensusConstraint, ConstraintSet
from anyalgebra.census.equivalence import EquivalencePolicy
from anyalgebra.census.filtering import compile_constraint_filter
from anyalgebra.census.permutations import generate_allowed_permutations
from anyalgebra.census.reference import enumerate_reference
from anyalgebra.census.serialization import (
    census_spec_canonical_bytes,
    census_spec_from_canonical_bytes,
)
from anyalgebra.census.spec import CensusSpec, CensusSpecCore
from anyalgebra.census.table_codes import rank_operation_table, unrank_operation_table
from anyalgebra.census.transport import transport_operation_table


SEED = 0xA11CE_010049
CASES_PER_SHAPE = 8
SHAPES = ((1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 1), (3, 2))


@dataclass(frozen=True, slots=True)
class GeneratedTable:
    """One bounded table with enough metadata for exact failure replay."""

    seed: int
    case_index: int
    carrier_size: int
    arity: int
    outputs: tuple[int, ...]

    @property
    def replay(self) -> str:
        return (
            f"seed={self.seed};case={self.case_index};size={self.carrier_size};"
            f"arity={self.arity};outputs={self.outputs!r}"
        )


def _generated_tables() -> tuple[GeneratedTable, ...]:
    random = Random(SEED)
    result: list[GeneratedTable] = []
    case_index = 0
    for size, arity in SHAPES:
        cell_count = 1 if arity == 0 else size**arity
        for _ in range(CASES_PER_SHAPE):
            result.append(
                GeneratedTable(
                    seed=SEED,
                    case_index=case_index,
                    carrier_size=size,
                    arity=arity,
                    outputs=tuple(random.randrange(size) for _ in range(cell_count)),
                )
            )
            case_index += 1
    return tuple(result)


CASES = _generated_tables()


def _core(case: GeneratedTable) -> CensusSpecCore:
    return CensusSpecCore.create(
        carrier_size=case.carrier_size,
        arity=case.arity,
        corpus_name=f"property-{case.case_index}",
    )


def _policy(core: CensusSpecCore, case_index: int) -> EquivalencePolicy:
    selector = case_index % 3
    if selector == 0:
        return EquivalencePolicy.literal(core)
    if selector == 1 or core.carrier_size == 1:
        return EquivalencePolicy.relabeling(core)
    return EquivalencePolicy.relabeling(core, fixed_elements=(0,))


def _spec(
    *,
    size: int,
    arity: int,
    name: str,
    constraints: tuple[CensusConstraint, ...] = (),
    policy: str = "relabeling",
) -> CensusSpec:
    core = CensusSpecCore.create(
        carrier_size=size,
        arity=arity,
        corpus_name=name,
    )
    equivalence = (
        EquivalencePolicy.literal(core)
        if policy == "literal"
        else EquivalencePolicy.relabeling(core)
    )
    return CensusSpec.create(
        carrier_size=size,
        arity=arity,
        constraints=ConstraintSet.create(core, constraints),
        equivalence=equivalence,
        ordering=CensusOrdering.reference(),
        bounds=EnumerationBounds.create(
            max_candidates=100_000,
            max_orbits=100_000,
            max_work_units=100_000,
            max_memory_bytes=16_000_000,
        ),
    )


def _assert_table_property(
    case: GeneratedTable,
    predicate: Callable[[tuple[int, ...]], bool],
    description: str,
) -> None:
    """Assert a property and greedily shrink failures to a replayable table."""
    if predicate(case.outputs):
        return
    reduced = list(case.outputs)
    for index, value in enumerate(reduced):
        for candidate in range(value):
            trial = tuple((*reduced[:index], candidate, *reduced[index + 1 :]))
            if not predicate(trial):
                reduced[index] = candidate
                break
    raise AssertionError(f"{description}; {case.replay}; reduced={tuple(reduced)!r}")


def test_seed_bounds_and_failure_reducer_are_replayable() -> None:
    assert SEED == 11_071_586_893_897
    assert CASES_PER_SHAPE == 8
    assert len(CASES) == len(SHAPES) * CASES_PER_SHAPE == 64
    assert _generated_tables() == CASES
    assert {(case.carrier_size, case.arity) for case in CASES} == set(SHAPES)

    synthetic = GeneratedTable(SEED, 999, 3, 1, (2, 1, 2))
    with pytest.raises(
        AssertionError,
        match=(
            r"seed=11071586893897;case=999;size=3;arity=1;.*"
            r"reduced=\(0, 0, 1\)"
        ),
    ):
        _assert_table_property(
            synthetic,
            lambda outputs: all(value == 0 for value in outputs),
            "synthetic failure",
        )


def test_rank_unrank_round_trip_for_all_seeded_shapes() -> None:
    for case in CASES:
        core = _core(case)
        rank = rank_operation_table(core, case.outputs)

        def round_trip(
            outputs: tuple[int, ...],
            checked_core: CensusSpecCore = core,
            checked_rank: int = rank,
        ) -> bool:
            return (
                rank_operation_table(checked_core, outputs) == checked_rank
                and unrank_operation_table(checked_core, checked_rank) == outputs
            )

        _assert_table_property(
            case,
            round_trip,
            "rank/unrank mismatch",
        )


def test_transport_inverse_and_composition_are_metamorphic() -> None:
    for case in CASES:
        core = _core(case)
        policy = _policy(core, case.case_index)
        group = generate_allowed_permutations(policy)
        for first in group:
            forward = transport_operation_table(core, case.outputs, first)
            backward = transport_operation_table(
                core, forward.target_outputs, first.inverse()
            )
            assert backward.target_outputs == case.outputs, case.replay
            assert forward.verify_cells(), case.replay
        for first in group:
            for second in group:
                step = transport_operation_table(
                    core,
                    transport_operation_table(core, case.outputs, first).target_outputs,
                    second,
                )
                direct = transport_operation_table(
                    core, case.outputs, first.then(second)
                )
                assert step.target_outputs == direct.target_outputs, case.replay


def test_canonical_identity_is_constant_on_every_generated_orbit() -> None:
    for case in CASES:
        core = _core(case)
        policy = _policy(core, case.case_index)
        source_label = canonicalize_operation_table(core, case.outputs, policy)
        for permutation in generate_allowed_permutations(policy):
            transported = transport_operation_table(
                core, case.outputs, permutation
            ).target_outputs
            label = canonicalize_operation_table(core, transported, policy)
            assert label.canonical_outputs == source_label.canonical_outputs, (
                case.replay
            )
            assert label.canonical_id == source_label.canonical_id, case.replay
            assert label.certificate.verify_cells(), case.replay
            assert label.orbit_size * label.stabilizer_size == policy.permutation_count


def test_specs_round_trip_across_constraints_and_equivalence_policies() -> None:
    specs = (
        _spec(size=1, arity=0, name="property-nullary", policy="literal"),
        _spec(size=2, arity=1, name="property-unary", policy="relabeling"),
        _spec(
            size=2,
            arity=2,
            name="property-commutative",
            constraints=(CensusConstraint.commutative(),),
            policy="literal",
        ),
        _spec(
            size=2,
            arity=2,
            name="property-idempotent",
            constraints=(CensusConstraint.idempotent(),),
            policy="relabeling",
        ),
        _spec(
            size=2,
            arity=2,
            name="property-commutative-idempotent",
            constraints=(
                CensusConstraint.commutative(),
                CensusConstraint.idempotent(),
            ),
        ),
    )
    assert {spec.equivalence.kind for spec in specs} == {
        "literal",
        "carrier_relabeling",
    }
    assert {spec.carrier_size for spec in specs} == {1, 2}
    assert {
        constraint.kind for spec in specs for constraint in spec.constraints.constraints
    } == {"commutative", "idempotent"}
    for spec in specs:
        encoded = census_spec_canonical_bytes(spec)
        decoded = census_spec_from_canonical_bytes(encoded)
        assert census_spec_canonical_bytes(decoded) == encoded
        assert decoded.semantic_hash == spec.semantic_hash


@pytest.mark.parametrize(
    ("constraints", "accepted"),
    (
        ((), 16),
        ((CensusConstraint.commutative(),), 8),
        ((CensusConstraint.idempotent(),), 4),
        (
            (CensusConstraint.commutative(), CensusConstraint.idempotent()),
            2,
        ),
    ),
)
def test_constraint_filters_preserve_exact_counts_under_relabeling(
    constraints: tuple[CensusConstraint, ...], accepted: int
) -> None:
    spec = _spec(
        size=2,
        arity=2,
        name=f"property-filter-{accepted}",
        constraints=constraints,
    )
    compiled = compile_constraint_filter(spec)
    enumeration = enumerate_reference(spec)
    accepted_tables = tuple(
        candidate.outputs
        for candidate in enumeration.candidates
        if compiled.evaluate(candidate).accepted
    )

    assert enumeration.complete
    assert len(accepted_tables) == accepted
    for outputs in accepted_tables:
        for permutation in generate_allowed_permutations(spec.equivalence):
            image = transport_operation_table(
                spec.core, outputs, permutation
            ).target_outputs
            image_index = rank_operation_table(spec.core, image)
            assert compiled.evaluate(enumeration.candidates[image_index]).accepted


def test_atlas_round_trip_and_queries_are_deterministic_across_corpora(
    tmp_path: Path,
) -> None:
    specs = (
        _spec(size=1, arity=0, name="property-atlas-nullary", policy="literal"),
        _spec(size=2, arity=1, name="property-atlas-unary"),
        _spec(
            size=2,
            arity=2,
            name="property-atlas-commutative",
            constraints=(CensusConstraint.commutative(),),
        ),
        _spec(
            size=2,
            arity=2,
            name="property-atlas-idempotent",
            constraints=(CensusConstraint.idempotent(),),
            policy="literal",
        ),
    )
    first = build_finite_algebra_atlas(specs, output_directory=tmp_path / "first")
    second = build_finite_algebra_atlas(
        tuple(reversed(specs)), output_directory=tmp_path / "second"
    )
    first_loaded = load_finite_algebra_atlas(first.output_directory)
    second_loaded = load_finite_algebra_atlas(second.output_directory)

    assert first.atlas.semantic_hash == second.atlas.semantic_hash
    assert first.object_count == second.object_count
    assert first_loaded.atlas == second_loaded.atlas
    assert first_loaded.verified_object_count == first.object_count
    assert query_atlas(first_loaded) == query_atlas(second_loaded)

    binary = query_atlas(first_loaded, carrier_size=2, arity=2)
    assert binary.matches
    assert {match.corpus_id for match in binary.matches} == {
        "property-atlas-commutative",
        "property-atlas-idempotent",
    }
    commutative = query_atlas(first_loaded, laws=("commutativity",))
    assert commutative.matches
    assert all(
        match.law_statuses["commutativity"] == "proved_on_complete_grid"
        for match in commutative.matches
    )


def test_noncanonical_specs_and_unknown_queries_fail_closed(tmp_path: Path) -> None:
    spec = _spec(size=2, arity=1, name="property-fail-closed")
    encoded = census_spec_canonical_bytes(spec)
    with pytest.raises(ValueError):
        census_spec_from_canonical_bytes(encoded + b"\n")

    record = json.loads(encoded)
    record["schemaVersion"] = 2
    mutated = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError):
        census_spec_from_canonical_bytes(mutated)

    built = build_finite_algebra_atlas(
        (spec,), output_directory=tmp_path / "fail-closed-atlas"
    )
    loaded = load_finite_algebra_atlas(built.output_directory)
    with pytest.raises(AtlasQueryError, match="unknown law"):
        query_atlas(loaded, laws=("invented_law",))
    with pytest.raises(AtlasQueryError, match="duplicates"):
        query_atlas(loaded, laws=("totality", "totality"))
