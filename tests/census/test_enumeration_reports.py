"""Contracts for complete and bounded-incomplete enumeration reports."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from typing import cast

import pytest

from anyalgebra.census import CensusSpec
from anyalgebra.census.bounds import CensusOrdering, EnumerationBounds
from anyalgebra.census.constraints import CensusConstraint, ConstraintSet
from anyalgebra.census.equivalence import EquivalencePolicy
from anyalgebra.census.pruning import enumerate_pruned
from anyalgebra.census.reference import ReferenceEnumeration, enumerate_reference
from anyalgebra.census.reports import (
    CompleteEnumerationReport,
    EnumerationReportError,
    IncompleteEnumerationReport,
    enumeration_report_canonical_bytes,
    enumeration_report_record,
    report_pruned_enumeration,
    report_reference_enumeration,
)
from anyalgebra.census.spec import CensusSpecCore
from anyalgebra.core.parents import SemanticHash


def _spec(
    constraints: tuple[CensusConstraint, ...] = (),
    *,
    max_candidates: int = 100,
    work: int = 1_000_000,
) -> CensusSpec:
    core = CensusSpecCore.create(
        carrier_size=2,
        arity=2,
        corpus_name="enumeration-report-control",
    )
    return CensusSpec.create(
        carrier_size=2,
        arity=2,
        constraints=ConstraintSet.create(core, constraints),
        equivalence=EquivalencePolicy.literal(core),
        ordering=CensusOrdering.reference(),
        bounds=EnumerationBounds.create(
            max_candidates=max_candidates,
            max_orbits=100,
            max_work_units=work,
            max_memory_bytes=10_000_000,
        ),
    )


def test_complete_reference_report_checks_all_accounting_fields() -> None:
    spec = _spec((CensusConstraint.commutative(),))
    report = report_reference_enumeration(enumerate_reference(spec))

    assert type(report) is CompleteEnumerationReport
    assert report.status == "complete"
    assert report.spec is spec
    assert report.bounds is spec.bounds
    assert report.total_candidate_count == 16
    assert report.examined_candidate_count == 16
    assert report.rejected_candidate_count == 8
    assert report.pruned_candidate_count == 0
    assert report.emitted_candidate_count == 8
    assert report.rejection_counts == (("commutativity_mismatch", 8),)
    assert report.algorithm == "anyalgebra.census.reference"
    assert report.algorithm_version == 1
    assert type(report.semantic_hash) is SemanticHash
    assert not hasattr(report, "next_candidate_index")


def test_incomplete_reference_report_has_exact_frontier_and_stop_reason() -> None:
    spec = _spec((CensusConstraint.commutative(),), max_candidates=5)
    report = report_reference_enumeration(enumerate_reference(spec))

    assert type(report) is IncompleteEnumerationReport
    assert report.status == "bounded_incomplete"
    assert report.total_candidate_count == 16
    assert report.examined_candidate_count == 5
    assert report.rejected_candidate_count + report.emitted_candidate_count == 5
    assert report.pruned_candidate_count == 0
    assert report.next_candidate_index == 5
    assert report.stop_reason == "max_candidates"


def test_complete_pruned_report_closes_examined_skipped_and_emitted_equation() -> None:
    spec = _spec((CensusConstraint.commutative(),))
    run = enumerate_pruned(spec)
    report = report_pruned_enumeration(run)

    assert type(report) is CompleteEnumerationReport
    assert report.total_candidate_count == 16
    assert report.examined_candidate_count == 8
    assert report.rejected_candidate_count == 0
    assert report.pruned_candidate_count == 8
    assert report.emitted_candidate_count == 8
    assert report.examined_candidate_count + report.pruned_candidate_count == 16
    assert report.algorithm == "anyalgebra.census.local_prefix_pruning"


def test_empty_corpus_complete_report_is_not_confused_with_interruption() -> None:
    core = CensusSpecCore.create(
        carrier_size=0, arity=0, corpus_name="empty-nullary-report"
    )
    spec = CensusSpec.create(
        carrier_size=0,
        arity=0,
        constraints=ConstraintSet.create(core, ()),
        equivalence=EquivalencePolicy.literal(core),
        ordering=CensusOrdering.reference(),
        bounds=EnumerationBounds.create(
            max_candidates=0,
            max_orbits=0,
            max_work_units=0,
            max_memory_bytes=0,
        ),
    )
    report = report_reference_enumeration(enumerate_reference(spec))

    assert type(report) is CompleteEnumerationReport
    assert report.total_candidate_count == 0
    assert report.emitted_candidate_count == 0


def test_complete_and_incomplete_records_have_distinct_tags_and_shapes() -> None:
    complete = report_reference_enumeration(enumerate_reference(_spec()))
    incomplete = report_reference_enumeration(
        enumerate_reference(_spec(max_candidates=1))
    )
    complete_record = enumeration_report_record(complete)
    incomplete_record = enumeration_report_record(incomplete)

    assert complete_record["schemaType"] == "anyalgebra.census.report.complete"
    assert incomplete_record["schemaType"] == "anyalgebra.census.report.incomplete"
    assert "nextCandidateIndex" not in complete_record
    assert "stopReason" not in complete_record
    assert incomplete_record["nextCandidateIndex"] == 1
    assert incomplete_record["stopReason"] == "max_candidates"
    assert set(complete_record) != set(incomplete_record)


def test_report_bytes_are_canonical_deterministic_and_content_addressed() -> None:
    first = report_reference_enumeration(
        enumerate_reference(_spec((CensusConstraint.idempotent(),)))
    )
    second = report_reference_enumeration(
        enumerate_reference(_spec((CensusConstraint.idempotent(),)))
    )
    encoded = enumeration_report_canonical_bytes(first)
    record = enumeration_report_record(first)

    assert first == second
    assert encoded == enumeration_report_canonical_bytes(second)
    assert json.loads(encoded) == record
    assert record["contentHash"] == {
        "algorithm": "sha256",
        "digest": first.semantic_hash.digest,
    }


def test_huge_exact_total_serializes_without_decimal_digit_or_allocation_failure() -> (
    None
):
    core = CensusSpecCore.create(
        carrier_size=2, arity=16, corpus_name="huge-zero-budget-report"
    )
    spec = CensusSpec.create(
        carrier_size=2,
        arity=16,
        constraints=ConstraintSet.create(core, ()),
        equivalence=EquivalencePolicy.literal(core),
        ordering=CensusOrdering.reference(),
        bounds=EnumerationBounds.create(
            max_candidates=0,
            max_orbits=0,
            max_work_units=0,
            max_memory_bytes=0,
        ),
    )
    report = report_reference_enumeration(enumerate_reference(spec))
    record = enumeration_report_record(report)
    encoded = enumeration_report_canonical_bytes(report)

    assert type(report) is IncompleteEnumerationReport
    assert report.total_candidate_count.bit_length() == 65_537
    assert record["totalCandidateCount"] == {
        "digits": "1" + "0" * 16_384,
        "radix": 16,
    }
    assert json.loads(encoded) == record


def test_multiple_rejection_reasons_are_sorted_and_do_not_double_count_candidates() -> (
    None
):
    report = report_reference_enumeration(
        enumerate_reference(
            _spec(
                (
                    CensusConstraint.commutative(),
                    CensusConstraint.idempotent(),
                )
            )
        )
    )

    assert type(report) is CompleteEnumerationReport
    assert tuple(reason for reason, _ in report.rejection_counts) == tuple(
        sorted(reason for reason, _ in report.rejection_counts)
    )
    assert sum(count for _, count in report.rejection_counts) >= (
        report.rejected_candidate_count
    )
    assert report.rejected_candidate_count + report.emitted_candidate_count == 16


def test_reports_are_factory_owned_sealed_immutable_and_safely_represented() -> None:
    report = report_reference_enumeration(enumerate_reference(_spec()))

    with pytest.raises(EnumerationReportError, match="factory-owned"):
        type(report)()
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type("Derived", (type(report),), {})
    with pytest.raises(FrozenInstanceError):
        report.status = "failed"  # type: ignore[misc]
    assert report.spec.core.corpus_name not in repr(report)
    assert report.semantic_hash.digest in repr(report)


def test_nonexact_source_records_fail_closed() -> None:
    with pytest.raises(EnumerationReportError) as caught:
        report_reference_enumeration(cast(ReferenceEnumeration, None))
    assert caught.value.field == "enumeration"
