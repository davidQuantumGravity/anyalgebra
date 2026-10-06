"""Calculation-contract and replay controls for finite-algebra atlases."""

from __future__ import annotations

from pathlib import Path

import pytest

from anyalgebra.atlas.evidence import (
    atlas_calculation_contract,
    replay_atlas_calculation,
    run_atlas_calculation,
)
from anyalgebra.census import CensusSpec
from anyalgebra.census.bounds import CensusOrdering, EnumerationBounds
from anyalgebra.census.constraints import ConstraintSet
from anyalgebra.census.equivalence import EquivalencePolicy
from anyalgebra.census.spec import CensusSpecCore
from anyalgebra.evidence.run import ExecutionEnvironment
import anyalgebra.evidence.run as run_module


@pytest.fixture(autouse=True)
def _clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_module, "_utc_now", lambda: "2026-08-15T12:00:00Z")


def _spec(name: str = "evidence-order-two", *, max_candidates: int = 100) -> CensusSpec:
    core = CensusSpecCore.create(carrier_size=2, arity=2, corpus_name=name)
    return CensusSpec.create(
        carrier_size=2,
        arity=2,
        constraints=ConstraintSet.create(core, ()),
        equivalence=EquivalencePolicy.relabeling(core),
        ordering=CensusOrdering.reference(),
        bounds=EnumerationBounds.create(
            max_candidates=max_candidates,
            max_orbits=100,
            max_work_units=1_000,
            max_memory_bytes=1_000_000,
        ),
    )


def test_matching_runs_reproduce_atlas_semantics_and_replay_exactly(
    tmp_path: Path,
) -> None:
    first = run_atlas_calculation(
        (_spec(),),
        output_directory=tmp_path / "first",
        environment=ExecutionEnvironment.create(platform="first"),
    )
    second = run_atlas_calculation(
        (_spec(),),
        output_directory=tmp_path / "second",
        environment=ExecutionEnvironment.create(platform="second"),
    )
    replay = replay_atlas_calculation(first.receipt, second.receipt)

    assert first.build.atlas.semantic_hash == second.build.atlas.semantic_hash
    assert first.contract == second.contract
    assert first.receipt.evidence_tier == second.receipt.evidence_tier == "E1-executed"
    assert first.receipt.mathematical_outcome == "verified_within_domain"
    assert first.receipt.artifacts == (first.build.atlas.semantic_hash,)
    assert replay.status == "exact_match"
    assert replay.stale_edges == ()


def test_contract_pins_specs_algorithms_bounds_conventions_and_artifacts() -> None:
    spec = _spec()
    contract = atlas_calculation_contract((spec,))

    assert contract.input_hashes == (("census_spec.000", spec.semantic_hash),)
    assert dict(contract.algorithms) == {
        "analysis": "anyalgebra.capability_aware_analysis_assembly.v1",
        "atlas": "anyalgebra.atlas.finite_algebra.v1",
        "canonicalization": "anyalgebra.census.canonical_bruteforce.v1",
        "enumeration": "anyalgebra.census.reference.v1",
    }
    assert dict(contract.bounds)["declared_candidate_count"] == 16
    assert dict(contract.bounds)["max_candidates"] == 100
    assert dict(contract.assumptions)["conventions"] == (
        "content-pinned transitively by CensusSpec hashes"
    )
    assert contract.expected_artifacts == (
        "atlas.json",
        "content-addressed object closure",
    )


def test_changed_spec_and_bounds_localize_staleness_edges(tmp_path: Path) -> None:
    original = run_atlas_calculation(
        (_spec("original"),), output_directory=tmp_path / "original"
    )
    renamed = run_atlas_calculation(
        (_spec("renamed"),), output_directory=tmp_path / "renamed"
    )
    renamed_replay = replay_atlas_calculation(original.receipt, renamed.receipt)

    assert renamed_replay.status == "stale"
    assert renamed_replay.stale_edges == ("contract.inputs",)

    bounded = run_atlas_calculation(
        (_spec("original", max_candidates=3),),
        output_directory=tmp_path / "bounded",
    )
    bounded_replay = replay_atlas_calculation(original.receipt, bounded.receipt)
    assert bounded_replay.status == "stale"
    assert bounded_replay.stale_edges == (
        "contract.bounds",
        "contract.inputs",
    )


def test_interrupted_corpus_remains_inconclusive_e1_evidence(tmp_path: Path) -> None:
    result = run_atlas_calculation(
        (_spec(max_candidates=3),), output_directory=tmp_path / "partial"
    )

    assert result.build.atlas.header.status == "incomplete"
    assert result.receipt.execution_status == "completed"
    assert result.receipt.evidence_tier == "E1-executed"
    assert result.receipt.mathematical_outcome == "inconclusive"
    assert result.receipt.remaining_branches == ("enumeration frontier remains",)
    assert "proved" not in result.receipt.summary.lower()
    assert "complete" not in result.receipt.summary.lower()
