"""General calculation-contract and replay integration for atlas builds."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

from anyalgebra.census.spec import CensusSpec
from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.evidence.contracts import CalculationContract
from anyalgebra.evidence.replay import ReplayReport, ReplayResolver, verify_replay
from anyalgebra.evidence.run import (
    CalculationResult,
    ExecutionEnvironment,
    ResultReceipt,
    run_calculation,
)

from .build import AtlasBuildResult, build_finite_algebra_atlas


_MAX_CORPORA = 256


class AtlasEvidenceError(AnyAlgebraError, ValueError):
    """An atlas evidence contract or execution boundary failed closed."""

    def __init__(self, *, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"invalid atlas evidence {field}: {reason}")


@dataclass(frozen=True, slots=True)
class AtlasEvidenceRun:
    """One atlas build paired with its immutable E1 execution receipt."""

    contract: CalculationContract
    build: AtlasBuildResult
    receipt: ResultReceipt


def _specs(value: object) -> tuple[CensusSpec, ...]:
    if isinstance(value, str | bytes):
        raise AtlasEvidenceError(field="specs", reason="must be an iterable")
    try:
        iterator = iter(cast(Iterable[object], value))
    except Exception as error:
        raise AtlasEvidenceError(field="specs", reason="must be an iterable") from error
    result: list[object] = []
    try:
        for _ in range(_MAX_CORPORA + 1):
            result.append(next(iterator))
    except StopIteration:
        pass
    except Exception as error:
        raise AtlasEvidenceError(
            field="specs", reason="iterable snapshot failed"
        ) from error
    else:
        raise AtlasEvidenceError(field="specs", reason="item limit exceeded")
    if not result:
        raise AtlasEvidenceError(field="specs", reason="must not be empty")
    if any(type(item) is not CensusSpec for item in result):
        raise AtlasEvidenceError(
            field="specs", reason="must contain exact CensusSpec records"
        )
    checked = tuple(
        sorted(
            cast(tuple[CensusSpec, ...], tuple(result)),
            key=lambda item: str(item.semantic_hash),
        )
    )
    if len({item.semantic_hash for item in checked}) != len(checked):
        raise AtlasEvidenceError(field="specs", reason="contains duplicates")
    return checked


def atlas_calculation_contract(specs: object) -> CalculationContract:
    """Freeze atlas inputs and declared bounds before any build executes."""
    checked = _specs(specs)
    return CalculationContract.create(
        contract_name="finite-algebra-atlas-v1",
        project_id="NEW-01",
        question="construct the bounded finite-algebra atlas for the declared corpora",
        acceptable_outcomes=("inconclusive", "verified_within_domain"),
        input_hashes=tuple(
            (f"census_spec.{index:03d}", spec.semantic_hash)
            for index, spec in enumerate(checked)
        ),
        source_anchors=(),
        convention_manifests=(),
        algorithms=(
            ("analysis", "anyalgebra.capability_aware_analysis_assembly.v1"),
            ("atlas", "anyalgebra.atlas.finite_algebra.v1"),
            (
                "canonicalization",
                "anyalgebra.census.canonical_bruteforce.v1",
            ),
            ("enumeration", "anyalgebra.census.reference.v1"),
        ),
        backend_request="exact",
        backend_name="anyalgebra-python-reference",
        backend_version="0.1",
        assumptions=(
            ("carrier_model", "one total operation on each declared finite carrier"),
            (
                "conventions",
                "content-pinned transitively by CensusSpec hashes",
            ),
            ("equivalence", "each CensusSpec declares its allowed relabeling group"),
        ),
        theorem_hypotheses=(
            ("classification_scope", "only the finite declared candidate corpus"),
            ("positive_maps", "each orbit member has a replayed transport certificate"),
        ),
        bounds=(
            (
                "cases",
                sum(spec.core.candidate_count for spec in checked),
            ),
            ("corpus_count", len(checked)),
            (
                "declared_candidate_count",
                sum(spec.core.candidate_count for spec in checked),
            ),
            ("max_candidates", sum(spec.bounds.max_candidates for spec in checked)),
            (
                "max_memory_bytes",
                sum(spec.bounds.max_memory_bytes for spec in checked),
            ),
            ("max_orbits", sum(spec.bounds.max_orbits for spec in checked)),
            (
                "max_work_units",
                sum(spec.bounds.max_work_units for spec in checked),
            ),
        ),
        required_cross_checks=(
            "atlas object role closure",
            "canonical byte replay",
            "classification count equations",
        ),
        expected_artifacts=("atlas.json", "content-addressed object closure"),
        acceptance_predicates=(
            "all serialized links resolve",
            "interrupted enumeration retains a frontier",
            "published atlas hash matches the calculation artifact",
        ),
    )


def _packet(build: AtlasBuildResult) -> CalculationResult:
    atlas = build.atlas
    complete = atlas.header.status == "complete"
    return CalculationResult.create(
        outcome="verified_within_domain" if complete else "inconclusive",
        summary=(
            "bounded atlas artifact constructed within declared corpus"
            if complete
            else "bounded atlas prefix artifact constructed"
        ),
        case_counts=(
            (
                "accepted_tables",
                sum(item.accepted_labeled_count for item in atlas.corpora),
            ),
            (
                "casesExpected",
                sum(item.total_candidate_count for item in atlas.corpora),
            ),
            ("casesRun", sum(item.examined_candidate_count for item in atlas.corpora)),
            ("corpora", atlas.header.corpus_count),
            (
                "examined_candidates",
                sum(item.examined_candidate_count for item in atlas.corpora),
            ),
            ("orbits", atlas.header.representative_count),
            (
                "rejected_tables",
                sum(item.rejected_labeled_count for item in atlas.corpora),
            ),
            (
                "total_candidates",
                sum(item.total_candidate_count for item in atlas.corpora),
            ),
        ),
        warnings=(() if complete else ("one or more corpus frontiers remain",)),
        artifacts=(atlas.semantic_hash,),
        checks_performed=(
            "canonical atlas loader role closure",
            "content-address equality",
            "enumeration and orbit accounting",
        ),
        remaining_branches=(() if complete else ("enumeration frontier remains",)),
    )


def run_atlas_calculation(
    specs: object,
    *,
    output_directory: object,
    environment: ExecutionEnvironment | None = None,
) -> AtlasEvidenceRun:
    """Execute one trusted atlas build under a frozen general contract."""
    checked = _specs(specs)
    contract = atlas_calculation_contract(checked)
    built: list[AtlasBuildResult] = []

    def calculate(_: CalculationContract) -> CalculationResult:
        result = build_finite_algebra_atlas(checked, output_directory=output_directory)
        built.append(result)
        return _packet(result)

    receipt = run_calculation(contract, calculate, environment=environment)
    if not built or receipt.execution_status != "completed":
        raise AtlasEvidenceError(
            field="execution", reason="atlas build did not produce a published artifact"
        )
    return AtlasEvidenceRun(contract=contract, build=built[0], receipt=receipt)


def replay_atlas_calculation(original: object, rerun: object) -> ReplayReport:
    """Compare two sealed atlas receipts through the general replay engine."""
    if type(original) is not ResultReceipt or type(rerun) is not ResultReceipt:
        raise AtlasEvidenceError(
            field="receipts", reason="must be exact ResultReceipt values"
        )
    resolver = ReplayResolver.create(
        receipts=(original, rerun),
        reruns=((original.semantic_hash, rerun.semantic_hash),),
    )
    return verify_replay(original, resolver=resolver)


__all__ = (
    "AtlasEvidenceError",
    "AtlasEvidenceRun",
    "atlas_calculation_contract",
    "replay_atlas_calculation",
    "run_atlas_calculation",
)
