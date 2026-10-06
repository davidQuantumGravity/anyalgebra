"""Full frozen-control differential for partition-guided canonicalization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from anyalgebra.census import CensusSpec
from anyalgebra.census.bounds import CensusOrdering, EnumerationBounds
from anyalgebra.census.canonical import CanonicalLabel, canonicalize_operation_table
from anyalgebra.census.certificates import (
    CanonicalCertificate,
    verify_canonical_certificate,
)
from anyalgebra.census.constraints import CensusConstraint, ConstraintSet
from anyalgebra.census.equivalence import EquivalencePolicy
from anyalgebra.census.filtering import compile_constraint_filter
from anyalgebra.census import partition as partition_module
from anyalgebra.census.partition import canonicalize_with_invariant_partition
from anyalgebra.census.permutations import CarrierPermutation
from anyalgebra.census.reference import enumerate_reference
from anyalgebra.census.spec import CensusSpecCore


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "v01_reference_corpora.json"


def _entries() -> tuple[dict[str, object], ...]:
    record = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return cast(tuple[dict[str, object], ...], tuple(record["corpora"]))


def _constraints(profile: str) -> tuple[CensusConstraint, ...]:
    return {
        "none": (),
        "identity0_two_sided": (
            CensusConstraint.identity(element=0, side="two_sided"),
        ),
        "commutative": (CensusConstraint.commutative(),),
        "idempotent": (CensusConstraint.idempotent(),),
        "commutative_idempotent": (
            CensusConstraint.commutative(),
            CensusConstraint.idempotent(),
        ),
        "quasigroup": (CensusConstraint.quasigroup(),),
    }[profile]


def _spec_and_policy(
    entry: dict[str, object],
) -> tuple[CensusSpec, EquivalencePolicy]:
    size = cast(int, entry["carrierSize"])
    arity = cast(int, entry["arity"])
    profile = cast(str, entry["constraintProfile"])
    core = CensusSpecCore.create(
        carrier_size=size,
        arity=arity,
        distinguished_elements=(("identity", 0),)
        if profile == "identity0_two_sided"
        else (),
        corpus_name=cast(str, entry["id"]),
    )
    policy = EquivalencePolicy.relabeling(core)
    spec = CensusSpec.create(
        carrier_size=size,
        arity=arity,
        constraints=ConstraintSet.create(core, _constraints(profile)),
        equivalence=policy,
        ordering=CensusOrdering.reference(),
        bounds=EnumerationBounds.create(
            max_candidates=100_000,
            max_orbits=100_000,
            max_work_units=1_000_000,
            max_memory_bytes=100_000_000,
        ),
    )
    return spec, policy


def _assert_exact_differential(
    reference: CanonicalLabel,
    guided: CanonicalLabel,
) -> None:
    assert guided.canonical_outputs == reference.canonical_outputs
    assert guided.canonical_bytes == reference.canonical_bytes
    assert guided.canonical_id == reference.canonical_id
    assert guided.permutations_checked == reference.permutations_checked
    assert guided.orbit_size == reference.orbit_size
    assert guided.stabilizer_size == reference.stabilizer_size
    assert guided.certificate == reference.certificate
    assert guided.certificate.verify_cells() is True
    verification = verify_canonical_certificate(CanonicalCertificate.from_label(guided))
    assert verification.reconstructed_source == guided.source_outputs
    assert verification.reconstructed_target == guided.canonical_outputs


@pytest.mark.parametrize("entry", _entries(), ids=lambda entry: str(entry["id"]))
def test_every_accepted_frozen_control_matches_brute_force(
    entry: dict[str, object],
) -> None:
    spec, policy = _spec_and_policy(entry)
    compiled = compile_constraint_filter(spec)
    accepted = 0
    enumeration = enumerate_reference(spec)
    assert enumeration.complete is True
    for candidate in enumeration.candidates:
        if not compiled.evaluate(candidate).accepted:
            continue
        accepted += 1
        source = candidate.outputs
        reference = canonicalize_operation_table(spec.core, source, policy)
        guided = canonicalize_with_invariant_partition(spec.core, source, policy).label
        _assert_exact_differential(reference, guided)

    assert accepted == entry["expectedAcceptedCount"]


def test_differential_oracle_kills_a_truncated_permutation_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = CensusSpecCore.create(
        carrier_size=3,
        arity=1,
        corpus_name="perturbed-partition-search",
    )
    policy = EquivalencePolicy.relabeling(core)
    source = (2, 2, 0)
    reference = canonicalize_operation_table(core, source, policy)
    real_order = partition_module.order_permutations_by_partition

    def incorrectly_pruned(
        partition: partition_module.InvariantPartition,
    ) -> tuple[CarrierPermutation, ...]:
        return tuple(
            permutation
            for permutation in real_order(partition)
            if permutation.images == tuple(range(partition.core.carrier_size))
        )

    monkeypatch.setattr(
        partition_module, "order_permutations_by_partition", incorrectly_pruned
    )
    perturbed = canonicalize_with_invariant_partition(core, source, policy).label

    with pytest.raises(AssertionError):
        _assert_exact_differential(reference, perturbed)
