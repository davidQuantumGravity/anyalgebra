"""Targeted mutation sample for the v0.1 completeness boundary.

These five hand-selected mutants are not a global mutation score.  Each test
first makes the perturbation observable and then requires a named independent
semantic oracle or fail-closed guard to reject it.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from anyalgebra.atlas.models import (
    AtlasError,
    FiniteAlgebraAtlas,
    atlas_canonical_bytes,
    create_atlas_corpus,
)
from anyalgebra.census import pruning as pruning_module
from anyalgebra.census.bounds import CensusOrdering, EnumerationBounds
from anyalgebra.census.canonical import canonicalize_operation_table
from anyalgebra.census.certificates import (
    CanonicalCertificate,
    CanonicalCertificateError,
    verify_canonical_certificate,
)
from anyalgebra.census.constraints import CensusConstraint, ConstraintSet
from anyalgebra.census.equivalence import EquivalencePolicy
from anyalgebra.census.filtering import compile_constraint_filter
from anyalgebra.census.permutations import generate_allowed_permutations
from anyalgebra.census.reference import enumerate_reference
from anyalgebra.census.spec import CensusSpec, CensusSpecCore
from anyalgebra.census.transport import transport_operation_table
from anyalgebra.core.parents import SemanticHash


@dataclass(frozen=True, slots=True)
class DeclaredMutant:
    """One bounded source-level fault and the test that must kill it."""

    mutant_id: str
    target: str
    perturbation: str
    killing_test: str


MUTANTS = (
    DeclaredMutant(
        "M01-candidate-count-minus-one",
        "CensusSpecCore.candidate_count",
        "decrement the exact raw table count",
        "test_exact_count_oracle_kills_candidate_count_mutant",
    ),
    DeclaredMutant(
        "M02-prune-valid-prefix",
        "pruning._violation",
        "mark one valid prefix as contradictory",
        "test_reference_differential_kills_false_pruning_mutant",
    ),
    DeclaredMutant(
        "M03-canonical-maximum",
        "canonical table comparison",
        "select the greatest orbit image instead of the least",
        "test_independent_minimum_oracle_kills_canonical_comparison_mutant",
    ),
    DeclaredMutant(
        "M04-skip-certificate-transport",
        "canonical certificate verification",
        "check only the target identifier and skip the transport equation",
        "test_cell_replay_kills_weak_certificate_verifier_mutant",
    ),
    DeclaredMutant(
        "M05-promote-incomplete-atlas",
        "atlas completion promotion",
        "change incomplete corpus and header status to complete",
        "test_model_replay_kills_false_atlas_completion_mutant",
    ),
)


def _spec(constraints: tuple[CensusConstraint, ...] = ()) -> CensusSpec:
    core = CensusSpecCore.create(
        carrier_size=2,
        arity=2,
        corpus_name="mutation-order-two",
    )
    return CensusSpec.create(
        carrier_size=2,
        arity=2,
        constraints=ConstraintSet.create(core, constraints),
        equivalence=EquivalencePolicy.relabeling(core),
        ordering=CensusOrdering.reference(),
        bounds=EnumerationBounds.create(
            max_candidates=100,
            max_orbits=100,
            max_work_units=10_000,
            max_memory_bytes=1_000_000,
        ),
    )


def _hash(digit: str) -> SemanticHash:
    return SemanticHash("sha256", digit * 64)


def test_mutation_manifest_is_bounded_unique_and_names_every_killing_test() -> None:
    assert len(MUTANTS) == 5
    assert len({mutant.mutant_id for mutant in MUTANTS}) == len(MUTANTS)
    for mutant in MUTANTS:
        assert mutant.target
        assert mutant.perturbation
        assert callable(globals()[mutant.killing_test])


def test_exact_count_oracle_kills_candidate_count_mutant() -> None:
    spec = _spec()
    object.__setattr__(spec.core, "candidate_count", 15)
    mutant_result = enumerate_reference(spec)

    assert mutant_result.total_candidate_count == 15
    with pytest.raises(AssertionError):
        assert mutant_result.total_candidate_count == 2 ** (2**2) == 16


def test_reference_differential_kills_false_pruning_mutant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec((CensusConstraint.idempotent(),))
    raw = enumerate_reference(spec)
    compiled = compile_constraint_filter(spec)
    expected = tuple(
        candidate
        for candidate in raw.candidates
        if compiled.evaluate(candidate).accepted
    )
    real_violation = pruning_module._violation

    def false_positive(
        constraint: CensusConstraint,
        prefix: tuple[int, ...],
        core: CensusSpecCore,
    ) -> pruning_module.PrefixViolation | None:
        if prefix == (0,):
            return pruning_module.PrefixViolation._create(
                constraint, "mutant_false_positive", ()
            )
        return real_violation(constraint, prefix, core)

    monkeypatch.setattr(pruning_module, "_violation", false_positive)
    mutant_result = pruning_module.enumerate_pruned(spec)

    assert mutant_result.accepted_candidates != expected
    with pytest.raises(AssertionError):
        assert mutant_result.accepted_candidates == expected


def test_independent_minimum_oracle_kills_canonical_comparison_mutant() -> None:
    core = CensusSpecCore.create(
        carrier_size=3,
        arity=1,
        corpus_name="mutation-canonical-comparison",
    )
    policy = EquivalencePolicy.relabeling(core)
    source = (2, 2, 0)
    images = tuple(
        transport_operation_table(core, source, permutation).target_outputs
        for permutation in generate_allowed_permutations(policy)
    )
    mutant_choice = max(images)
    reference = canonicalize_operation_table(core, source, policy)

    assert min(images) != mutant_choice
    assert reference.canonical_outputs == min(images)
    with pytest.raises(AssertionError):
        assert mutant_choice == min(images)


def test_cell_replay_kills_weak_certificate_verifier_mutant() -> None:
    core = CensusSpecCore.create(
        carrier_size=2,
        arity=2,
        corpus_name="mutation-certificate",
    )
    policy = EquivalencePolicy.literal(core)
    source = (0, 0, 0, 0)
    unrelated_target = (1, 1, 1, 1)
    target_label = canonicalize_operation_table(core, unrelated_target, policy)
    claim = CanonicalCertificate.claim(
        core=core,
        equivalence=policy,
        source_outputs=source,
        canonical_outputs=unrelated_target,
        permutation_images=(0, 1),
        canonical_id=target_label.canonical_id,
    )

    weak_identifier_only_mutant_accepts = (
        canonicalize_operation_table(core, claim.canonical_outputs, policy).canonical_id
        == claim.canonical_id
    )
    assert weak_identifier_only_mutant_accepts
    with pytest.raises(CanonicalCertificateError, match="forward transport"):
        verify_canonical_certificate(claim)


def test_model_replay_kills_false_atlas_completion_mutant() -> None:
    corpus = create_atlas_corpus(
        corpus_id="mutation-incomplete",
        spec_id=_hash("1"),
        enumeration_id=_hash("2"),
        classification_id=None,
        receipt_id=_hash("3"),
        total_candidate_count=16,
        examined_candidate_count=3,
        accepted_labeled_count=2,
        rejected_labeled_count=1,
        representatives=(),
        complete=False,
    )
    atlas = FiniteAlgebraAtlas.create((corpus,))
    assert atlas.header.status == "incomplete"

    object.__setattr__(corpus, "complete", True)
    object.__setattr__(atlas.header, "status", "complete")
    assert corpus.examined_candidate_count < corpus.total_candidate_count
    assert atlas.header.status == "complete"
    with pytest.raises(AtlasError, match="representatives do not account"):
        atlas_canonical_bytes(atlas)
