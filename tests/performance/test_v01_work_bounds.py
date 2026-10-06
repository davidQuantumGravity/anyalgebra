"""Deterministic work and allocation ceilings for the stable v0.1 surface.

These are combinatorial contracts, not wall-clock benchmarks.  A test remains
meaningful when run on a different processor or under a slower interpreter.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from anyalgebra.census import analysis as analysis_module
from anyalgebra.census import analysis_records as analysis_records_module
from anyalgebra.census import pruning as pruning_module
from anyalgebra.census.analysis import (
    analyze_finite_elements,
    analyze_finite_laws,
    analyze_finite_magma_subobjects,
)
from anyalgebra.census.bounds import CensusOrdering, EnumerationBounds
from anyalgebra.census.canonical import (
    CanonicalLabelError,
    canonicalize_operation_table,
)
from anyalgebra.census.constraints import CensusConstraint, CensusTerm, ConstraintSet
from anyalgebra.census.equivalence import EquivalencePolicy
from anyalgebra.census.filtering import compile_constraint_filter
from anyalgebra.census.isomorphism import search_isomorphism
from anyalgebra.census.permutations import (
    MAX_GENERATED_PERMUTATIONS,
    PermutationGenerationError,
    generate_allowed_permutations,
)
from anyalgebra.census.reference import (
    REFERENCE_BATCH_LIMIT,
    ReferenceCandidate,
    enumerate_reference,
)
from anyalgebra.census.serialization import census_spec_from_canonical_bytes
from anyalgebra.census.spec import CensusSpec, CensusSpecCore
from anyalgebra.census.subobjects import (
    FiniteSubobjectAnalysisError,
    FiniteSubobjectBounds,
    analyze_finite_subobjects,
)


ROOT = Path(__file__).resolve().parents[2]
WORK_BOUNDS = ROOT / "docs" / "testing" / "v0.1-work-bounds.md"
REFERENCE_SPEC = ROOT / "tests" / "fixtures" / "v01_atlas_spec.json"


def _spec(
    size: int,
    arity: int,
    *,
    name: str,
    constraints: tuple[CensusConstraint, ...] = (),
    max_candidates: int = 100_000,
    max_work_units: int = 100_000,
    max_memory_bytes: int = 16_000_000,
) -> CensusSpec:
    core = CensusSpecCore.create(
        carrier_size=size,
        arity=arity,
        corpus_name=name,
    )
    return CensusSpec.create(
        carrier_size=size,
        arity=arity,
        constraints=ConstraintSet.create(core, constraints),
        equivalence=EquivalencePolicy.relabeling(core),
        ordering=CensusOrdering.reference(),
        bounds=EnumerationBounds.create(
            max_candidates=max_candidates,
            max_orbits=100_000,
            max_work_units=max_work_units,
            max_memory_bytes=max_memory_bytes,
        ),
    )


@pytest.mark.parametrize(
    ("size", "arity", "total", "workspace"),
    (
        (0, 1, 1, 128),
        (1, 2, 1, 144),
        (2, 2, 16, 192),
        (3, 1, 27, 176),
        (3, 2, 19_683, 272),
    ),
)
def test_reference_counts_and_workspace_formula_are_exact(
    size: int, arity: int, total: int, workspace: int
) -> None:
    result = enumerate_reference(_spec(size, arity, name=f"work-{size}-{arity}"))

    assert result.total_candidate_count == total
    assert result.workspace_bytes_per_candidate == workspace
    assert workspace == 128 + 16 * result.core.input_tuple_count
    assert result.examined_count == total
    assert result.complete


def test_reference_atlas_bound_has_exact_fixed_work_and_memory() -> None:
    spec = census_spec_from_canonical_bytes(
        REFERENCE_SPEC.read_bytes().removesuffix(b"\n")
    )
    result = enumerate_reference(spec)

    assert result.total_candidate_count == 3**9 == 19_683
    assert result.workspace_bytes_per_candidate == 272
    assert result.total_candidate_count * result.workspace_bytes_per_candidate == (
        5_353_776
    )
    assert result.total_candidate_count <= spec.bounds.max_candidates
    assert result.total_candidate_count <= spec.bounds.max_work_units
    assert spec.bounds.max_memory_bytes >= 5_353_776
    assert result.total_candidate_count <= REFERENCE_BATCH_LIMIT == 1_000_000


@pytest.mark.parametrize(
    ("constraints", "work", "accepted", "events", "examined", "skipped"),
    (
        ((), 0, 16, 0, 16, 0),
        ((CensusConstraint.commutative(),), 4, 8, 4, 8, 8),
        ((CensusConstraint.idempotent(),), 2, 4, 5, 4, 12),
        (
            (CensusConstraint.commutative(), CensusConstraint.idempotent()),
            6,
            2,
            5,
            2,
            14,
        ),
    ),
)
def test_pruned_work_counts_are_exact_and_account_for_the_full_tree(
    constraints: tuple[CensusConstraint, ...],
    work: int,
    accepted: int,
    events: int,
    examined: int,
    skipped: int,
) -> None:
    spec = _spec(2, 2, name=f"pruned-work-{accepted}", constraints=constraints)
    compiled = compile_constraint_filter(spec)
    result = pruning_module.enumerate_pruned(spec)

    assert compiled.estimated_work_per_candidate == work
    assert len(result.accepted_candidates) == accepted
    assert len(result.events) == events
    assert result.examined_candidate_count == examined
    assert result.skipped_candidate_count == skipped
    assert examined + skipped == spec.core.candidate_count == 16
    assert result.complete


@pytest.mark.parametrize(
    ("candidate_limit", "work_limit", "memory_limit", "batch_limit", "reason"),
    (
        (0, 100, 1_000_000, 1_000_000, "max_candidates"),
        (100, 0, 1_000_000, 1_000_000, "max_work_units"),
        (100, 100, 191, 1_000_000, "max_memory_bytes"),
        (100, 100, 1_000_000, 0, "implementation_batch_limit"),
    ),
)
def test_reference_limits_stop_before_candidate_materialization(
    monkeypatch: pytest.MonkeyPatch,
    candidate_limit: int,
    work_limit: int,
    memory_limit: int,
    batch_limit: int,
    reason: str,
) -> None:
    def forbidden_create(
        cls: type[ReferenceCandidate], core: CensusSpecCore, candidate_index: int
    ) -> ReferenceCandidate:
        del cls, core, candidate_index
        raise AssertionError("candidate materialized before the selected ceiling")

    monkeypatch.setattr(
        ReferenceCandidate,
        "_create",
        classmethod(forbidden_create),
    )
    monkeypatch.setattr(
        "anyalgebra.census.reference.REFERENCE_BATCH_LIMIT", batch_limit
    )
    result = enumerate_reference(
        _spec(
            2,
            2,
            name=f"preflight-{reason}",
            max_candidates=candidate_limit,
            max_work_units=work_limit,
            max_memory_bytes=memory_limit,
        )
    )

    assert result.candidates == ()
    assert result.examined_count == 0
    assert result.next_candidate_index == 0
    assert result.stop_reason == reason


def test_pruned_preflight_stops_before_tree_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_decision(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("tree decision ran before preflight")

    monkeypatch.setattr(pruning_module, "_decision", forbidden_decision)
    spec = _spec(
        2,
        2,
        name="pruned-preflight",
        constraints=(CensusConstraint.idempotent(),),
        max_candidates=15,
    )
    with pytest.raises(pruning_module.PrefixPruningError) as caught:
        pruning_module.enumerate_pruned(spec)
    assert caught.value.field == "max_candidates"


def test_permutation_canonical_and_search_growth_share_the_hard_group_cap() -> None:
    core = CensusSpecCore.create(
        carrier_size=10,
        arity=1,
        corpus_name="permutation-preflight",
    )
    policy = EquivalencePolicy.relabeling(core)
    outputs = (0,) * 10
    assert policy.permutation_count == 3_628_800 > MAX_GENERATED_PERMUTATIONS

    with pytest.raises(PermutationGenerationError) as generated:
        generate_allowed_permutations(policy)
    assert generated.value.requested == 3_628_800
    with pytest.raises(CanonicalLabelError) as canonical:
        canonicalize_operation_table(core, outputs, policy)
    assert canonical.value.field == "equivalence"
    with pytest.raises(PermutationGenerationError):
        search_isomorphism(core, outputs, outputs, policy)


def test_analysis_reports_exact_semantic_work_instead_of_elapsed_time() -> None:
    core = CensusSpecCore.create(
        carrier_size=2,
        arity=2,
        corpus_name="analysis-work",
    )
    outputs = (0, 1, 1, 0)
    laws = analyze_finite_laws(core, outputs)
    elements = analyze_finite_elements(
        core, outputs, EquivalencePolicy.relabeling(core)
    )
    magma = analyze_finite_magma_subobjects(core, outputs)
    subobjects = analyze_finite_subobjects(core, outputs)

    assert laws.declared_grid_count == laws.total_evaluations == 34
    assert tuple(result.assignment_count for result in laws.results) == (
        4,
        2,
        4,
        8,
        4,
        4,
        4,
        4,
    )
    assert elements.examined_element_count == 2
    assert magma.associator_assignments_checked == 24
    assert magma.commutator_pairs_checked == 4
    assert (
        subobjects.subset_candidates_examined,
        subobjects.closure_assignments_checked,
        subobjects.partition_candidates_examined,
        subobjects.compatibility_pairs_checked,
    ) == (3, 6, 2, 20)


def test_declared_equation_grid_is_preflighted_before_law_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = CensusSpecCore.create(
        carrier_size=10,
        arity=1,
        corpus_name="law-grid-preflight",
    )
    variable = CensusTerm.variable(0)
    equation = CensusConstraint.equation(
        variable,
        variable,
        variable_count=8,
    )

    def forbidden_standard_laws(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("law traversal began before aggregate work preflight")

    monkeypatch.setattr(analysis_module, "_standard_laws", forbidden_standard_laws)
    with pytest.raises(analysis_module.FiniteLawAnalysisError) as caught:
        analysis_module.analyze_finite_laws(
            core,
            (0,) * 10,
            equations=(equation,),
        )
    assert caught.value.field == "work"
    assert analysis_module.MAX_FINITE_LAW_ASSIGNMENTS == 20_000_000


def test_assembled_analysis_runs_subobject_preflight_before_other_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = CensusSpecCore.create(
        carrier_size=9,
        arity=1,
        corpus_name="assembled-analysis-preflight",
    )

    def forbidden_laws(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("law analysis began before subobject preflight")

    monkeypatch.setattr(analysis_records_module, "analyze_finite_laws", forbidden_laws)
    with pytest.raises(FiniteSubobjectAnalysisError) as caught:
        analysis_records_module.assemble_finite_carrier_analysis(
            core,
            (0,) * 9,
            EquivalencePolicy.literal(core),
        )
    assert caught.value.field == "bounds"


def test_subobject_bound_stops_before_partition_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_partitions(_size: int) -> object:
        raise AssertionError("partitions materialized before preflight")

    monkeypatch.setattr(
        "anyalgebra.census.subobjects._partitions", forbidden_partitions
    )
    core = CensusSpecCore.create(
        carrier_size=3,
        arity=2,
        corpus_name="subobject-preflight",
    )
    with pytest.raises(FiniteSubobjectAnalysisError) as caught:
        analyze_finite_subobjects(
            core,
            (0,) * 9,
            options=FiniteSubobjectBounds(max_subset_candidates=6),
        )
    assert caught.value.field == "bounds"


def test_work_bound_document_maps_every_stable_v01_operation() -> None:
    text = WORK_BOUNDS.read_text(encoding="utf-8")
    for api_id in (
        "api.algebra.evaluate",
        "api.census.spec",
        "api.census.enumerate",
        "api.census.canonicalize",
        "api.census.compare",
        "api.census.analyze",
        "api.atlas.build",
        "api.atlas.query",
    ):
        assert f"`{api_id}`" in text
    for requirement in (
        "1,000,000 candidates",
        "1,000,000 permutations",
        "5,353,776 bytes",
        "No wall-clock threshold",
        "before materialization",
    ):
        assert requirement in text
